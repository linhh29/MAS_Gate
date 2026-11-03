#!/usr/bin/env python3
"""Unified MAS quantification with real J-score (vLLM in-process) + NLI R_ij.

Uses canonical quantification library (quantification/unified.py, quantification/utility/utility.py).
All scripts should use this as the single source of truth.

Usage:
    python quantification/scripts/unified_quantify.py --dataset bcp --output-dir experiments/pilot1/results
    python quantification/scripts/unified_quantify.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

SYSTEM_PATH = str(Path(__file__).resolve().parents[2])
if SYSTEM_PATH not in sys.path:
    sys.path.insert(0, SYSTEM_PATH)
if str(Path(__file__).resolve().parents[2] / "building") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "building"))

from quantification.config import DEFAULT_INTER_NODE_MODE, QuantificationConfig
from quantification.metrics import QuantificationEngine
from quantification.adapters.trace_adapter import load_reasoning_graph_from_trace
from quantification.clients.vllm_runtime import vllm_process_lock
from quantification.communication.nli import get_nli_client, reset_nli_client
from quantification.bcp_per_sample import ground_truth_for_index
from quantification.unified import quantify_from_j_state, summarize_results, EPSILON
from building.skill_mas.process_trace_layout import sample_trace_json_dir


def main():
    parser = argparse.ArgumentParser(description="Unified MAS quantification")
    parser.add_argument("--dataset", choices=["bcp", "aime", "drb", "all"], default="all")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Optional output directory override (default: dataset-specific unified_metrics)")
    parser.add_argument("--scoring-model", default="./models/Qwen3.6-35B-A3B")
    parser.add_argument("--embedding-model", default="./models/Qwen3-Embedding-8B")
    parser.add_argument("--nli-model", default="./models/bge-m3-zeroshot-v2.0")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-mem", type=float, default=0.65)
    parser.add_argument("--bcp-threshold", type=float, default=0.5)
    parser.add_argument("--aime-threshold", type=float, default=0.5)
    parser.add_argument("--drb-threshold", type=float, default=0.4)
    parser.add_argument("--agent-model", default="deepseek-v4-flash",
                        help="Agent model name used for inference traces (default: deepseek-v4-flash)")
    parser.add_argument(
        "--inter-node-mode",
        choices=["unified_baseline", "legacy_canonical"],
        default=os.getenv("QUANTIFICATION_INTER_NODE_MODE", DEFAULT_INTER_NODE_MODE),
        help=(
            "Inter-node formula. Default unified_baseline uses "
            "K_i=1+sum_j R_ij(1+G_j); legacy_canonical reproduces the previous behavior."
        ),
    )
    parser.add_argument(
        "--depth-fair-g",
        action="store_true",
        help="Use optional gamma-decayed G instead of normalized descendant count.",
    )
    args = parser.parse_args()

    MAS_ROOT = Path(__file__).resolve().parents[2]

    os.environ["QUANTIFICATION_BACKEND"] = "vllm"
    os.environ["QUANTIFICATION_VLLM_SCORING_GPUS"] = args.gpus
    os.environ["QUANTIFICATION_VLLM_EMBEDDING_GPUS"] = args.gpus
    os.environ["QUANTIFICATION_VLLM_SEQUENTIAL_GPUS"] = "1"
    os.environ["QUANTIFICATION_SCORING_MODEL_PATH"] = args.scoring_model
    os.environ["QUANTIFICATION_EMBEDDING_MODEL_PATH"] = args.embedding_model
    os.environ["QUANTIFICATION_VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["QUANTIFICATION_VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_mem)
    os.environ["QUANTIFICATION_VLLM_SCORING_MAX_NUM_SEQS"] = "1"
    os.environ["QUANTIFICATION_SCORING_CONCURRENCY"] = "4"
    os.environ["QUANTIFICATION_EMBEDDING_CONCURRENCY"] = "4"
    os.environ["QUANTIFICATION_TRACE_CONCURRENCY"] = "1"

    # Build dataset configs
    datasets = []
    global_output_dir = Path(args.output_dir) if args.output_dir else None

    if args.dataset in ("bcp", "all"):
        bcp_root = MAS_ROOT / "datasets/BrowseComp-Plus"
        bcp_trace = bcp_root / "results" / args.agent_model / "skill_mas_process_traces" / "init_skill"
        datasets.append({
            "name": "BCP", "trace_root": bcp_trace,
            "jsonl_path": bcp_root / "data/browsecomp_plus_test.jsonl",
            "output_dir": global_output_dir or (bcp_trace / "unified_metrics"),
            "threshold": args.bcp_threshold,
            "judge_dir": bcp_trace / "scored_samples", "judge_key": "score",
            "race_file": None,
        })
    if args.dataset in ("aime", "all"):
        aime_root = MAS_ROOT / "datasets/aime"
        aime_trace = aime_root / "results" / args.agent_model / "skill_mas_process_traces" / "init_skill"
        datasets.append({
            "name": "AIME", "trace_root": aime_trace,
            "jsonl_path": aime_root / "data/aime_test.jsonl",
            "output_dir": global_output_dir or (aime_trace / "unified_metrics"),
            "threshold": args.aime_threshold,
            "judge_dir": aime_trace / "scored_samples", "judge_key": "score",
            "race_file": None,
        })
    if args.dataset in ("drb", "all"):
        drb_root = MAS_ROOT / "datasets/deep_research_bench"
        drb_trace = drb_root / "results" / args.agent_model / "skill_mas_process_traces" / "init_skill"
        datasets.append({
            "name": "DRB", "trace_root": drb_trace,
            "jsonl_path": drb_root / "data/drb_test.jsonl",
            "output_dir": global_output_dir or (drb_trace / "unified_metrics"),
            "threshold": args.drb_threshold,
            "judge_dir": None, "judge_key": "score",
            "race_file": drb_root / "results/drb_race/raw_results.jsonl",
        })

    config = QuantificationConfig.from_env(
        inter_node_mode=args.inter_node_mode,
        use_depth_fair_g=args.depth_fair_g,
    )
    engine = QuantificationEngine(config)
    print(
        f"Inter-node mode: {config.inter_node_mode} "
        f"(G={'depth_fair' if config.use_depth_fair_g else 'simple_descendants'})"
    )

    all_dataset_results = {}

    for ds_cfg in datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {ds_cfg['name']} (threshold={ds_cfg['threshold']})")
        print(f"{'#'*80}")

        out_dir = ds_cfg["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load judge scores
        judge_scores = {}
        if ds_cfg["judge_dir"] and ds_cfg["judge_dir"].is_dir():
            for f in sorted(ds_cfg["judge_dir"].glob("*.json")):
                try:
                    obj = json.loads(f.read_text())
                    idx = int(f.stem)
                    judge_scores[idx] = float(obj.get(ds_cfg["judge_key"], 0))
                except (ValueError, json.JSONDecodeError):
                    pass
        if ds_cfg["race_file"] and ds_cfg["race_file"].is_file():
            with open(ds_cfg["race_file"]) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    tid = obj.get("id")
                    if tid is not None:
                        judge_scores[int(tid)] = float(obj.get("overall_score", 0.5))
        print(f"  Judge scores: {len(judge_scores)} available")

        nested = sample_trace_json_dir(ds_cfg["trace_root"])
        if not nested.is_dir():
            print(f"  No trace dir: {nested}")
            continue
        trace_files = sorted(nested.glob("*_t0.json"))
        if not trace_files:
            print(f"  No trace files in {nested}")
            continue

        total = len(trace_files)
        results = []

        print(f"  Phase 1: J scoring ({total} traces) ...")
        j_states = {}
        meta = {}

        from quantification.clients.j_prompt_token_limit import reset_max_j_prompt_tokens
        reset_max_j_prompt_tokens()

        with vllm_process_lock:
            engine.begin_j_phase()

            for n, tf in enumerate(trace_files, start=1):
                stem = tf.stem
                idx = int(stem.split("__", 1)[0]) if "__" in stem else n - 1
                task_id = stem.split("__", 1)[1].rsplit("_t", 1)[0] if "__" in stem else str(idx)

                try:
                    gt = ground_truth_for_index(ds_cfg["jsonl_path"], idx, task_id)
                except Exception:
                    gt = ""

                if not gt.strip() and ds_cfg["name"] == "DRB":
                    try:
                        rows = [json.loads(line) for line in ds_cfg["jsonl_path"].read_text(encoding="utf-8").splitlines() if line.strip()]
                        if 0 <= idx < len(rows):
                            gt = str(rows[idx].get("reference_article", "")).strip()
                        elif task_id is not None:
                            for row in rows:
                                if str(row.get("id")) == str(task_id):
                                    gt = str(row.get("reference_article", "")).strip()
                                    break
                    except Exception:
                        gt = ""

                if not gt.strip():
                    print(f"    [{n}/{total}] idx={idx} SKIP (no ground truth)")
                    results.append({"idx": idx, "error": "no_ground_truth"})
                    continue

                meta[idx] = (task_id, tf, gt)
                try:
                    graph = load_reasoning_graph_from_trace(tf, ground_truth=gt)
                except Exception as exc:
                    print(f"    [{n}/{total}] idx={idx} SKIP (trace load failed: {type(exc).__name__}: {exc})")
                    results.append({"idx": idx, "error": f"trace_load: {exc}"})
                    continue

                try:
                    j_state = engine.compute_j_phase(graph)
                    j_states[idx] = j_state
                    print(f"    [{n}/{total}] idx={idx} J0={j_state.j_baseline:.4f} nodes={len(j_state.execution_order)}")
                except Exception as exc:
                    print(f"    [{n}/{total}] idx={idx} J-phase FAILED: {type(exc).__name__}: {exc}")
                    results.append({"idx": idx, "error": f"j_phase: {exc}"})

        n_done = len(j_states)
        print(f"  Phase 2: NLI R_ij + K ({n_done}/{total}) ...")
        engine.release_gpu_engines()
        nli_client = get_nli_client()

        for n, (idx, j_state) in enumerate(j_states.items(), start=1):
            task_id, trace_path, gt = meta[idx]
            tid_int = int(task_id) if str(task_id).isdigit() else 0
            judge_score = judge_scores.get(idx, judge_scores.get(tid_int, 0.5))

            result = quantify_from_j_state(
                j_state=j_state,
                trace_path=trace_path,
                judge_score=judge_score,
                threshold=ds_cfg["threshold"],
                nli_client=nli_client,
                use_depth_fair_g=config.use_depth_fair_g,
                gamma=config.gamma,
                inter_node_mode=config.inter_node_mode,
            )
            if result is None:
                result = {"idx": idx, "error": "quantify_from_j_state returned None"}
            results.append(result)
            if "error" not in result:
                print(f"    [{n}/{n_done}] idx={idx} U={result['mean_U_i']:.4f} K={result['mean_K']:.4f} R={result['mean_R']:.4f} pos={result['pos_U']} neg={result['neg_U']}")
            else:
                print(f"    [{n}/{n_done}] idx={idx} FAILED: {result['error']}")

        reset_nli_client()

        successes = [r for r in results if "error" not in r]
        if successes:
            all_nodes = [nd for r in successes for nd in r["nodes"]]
            all_u = [nd["U_i"] for nd in all_nodes]
            all_k = [nd["K"] for nd in all_nodes]
            all_r = [e["R_ij"] for r in successes for e in r["edges"]]
            all_d = [nd["delta"] for nd in all_nodes]
            all_impact = [nd["impact"] for nd in all_nodes]
            print(f"\n  SUMMARY: {len(successes)}/{total} success")
            print(f"  Nodes: {len(all_nodes)}, Edges: {len(all_r)}")
            print(f"  Mean U_i: {np.mean(all_u):.4f} (range: {min(all_u):.4f} - {max(all_u):.4f})")
            print(f"  Mean delta: {np.mean(all_d):.4f} (range: {min(all_d):.4f} - {max(all_d):.4f})")
            print(f"  Mean K: {np.mean(all_k):.4f}, Mean R_ij: {np.mean(all_r):.4f}")
            print(f"  Mean Impact: {np.mean(all_impact):.4f}" if all_impact else f"  Mean Impact: N/A")
            print(f"  Pos U: {sum(1 for u in all_u if u>0.01)}, Neg U: {sum(1 for u in all_u if u<-0.01)}")
            print(f"  Pos delta: {sum(1 for d in all_d if d>0)}, Neg delta: {sum(1 for d in all_d if d<=0)}")

            summary = {
                "dataset": ds_cfg["name"], "threshold": ds_cfg["threshold"],
                "num_results": len(successes), "total_nodes": len(all_nodes), "total_edges": len(all_r),
                "mean_U_i": round(float(np.mean(all_u)), 4) if all_u else 0.0,
                "mean_delta": round(float(np.mean(all_d)), 4) if all_d else 0.0,
                "mean_K": round(float(np.mean(all_k)), 4) if all_k else 0.0,
                "mean_R_ij": round(float(np.mean(all_r)), 4) if all_r else 0.0,
                "mean_impact": round(float(np.mean(all_impact)), 4) if all_impact else 0.0,
                "inter_node_mode": config.inter_node_mode,
                "g_type": "depth_fair" if config.use_depth_fair_g else "simple_descendants",
                "k_formula": (
                    "1 + sum_j R_ij * (1 + G_j)"
                    if config.inter_node_mode == "unified_baseline"
                    else "sum_j R_ij * G_j (legacy terminal handling)"
                ),
                "pos_U": sum(1 for u in all_u if u > 0.01),
                "neg_U": sum(1 for u in all_u if u < -0.01),
                "pos_delta": sum(1 for d in all_d if d > 0),
                "neg_delta": sum(1 for d in all_d if d <= 0),
                "results": successes,
            }
            out_path = out_dir / f"unified_metrics_{ds_cfg['name'].lower()}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"  Saved to {out_path}")
            all_dataset_results[ds_cfg["name"]] = (ds_cfg, successes)

    if len(all_dataset_results) > 1:
        print(f"\n{'='*80}\nCROSS-DATASET COMPARISON\n{'='*80}")
        print(f"{'Dataset':<10s} {'Threshold':>10s} {'Mean U':>10s} {'Mean K':>10s} {'Mean R':>10s} {'Mean Delta':>10s} {'Pos U':>8s} {'Neg U':>8s}")
        print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        for name, (cfg, results) in all_dataset_results.items():
            all_nodes = [nd for r in results for nd in r["nodes"]]
            all_u = [nd["U_i"] for nd in all_nodes]
            all_k = [nd["K"] for nd in all_nodes]
            all_r = [e["R_ij"] for r in results for e in r["edges"]]
            all_d = [nd["delta"] for nd in all_nodes]
            pos_u = sum(1 for u in all_u if u > 0.01)
            neg_u = sum(1 for u in all_u if u < -0.01)
            print(f"{name:<10s} {cfg['threshold']:>10.3f} {np.mean(all_u):>10.4f} {np.mean(all_k):>10.4f} {np.mean(all_r):>10.4f} {np.mean(all_d):>10.4f} {pos_u:>8d} {neg_u:>8d}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
