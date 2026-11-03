#!/usr/bin/env python3
"""Run an isolated MAS-Gate evaluation campaign over a base model and local checkpoints.

Each candidate is evaluated sequentially because every Orchestra and the shared
Qwen3-30B sub-agent use tensor parallelism across the configured physical GPUs.
The child launcher keeps the candidate's model process alive and uses vLLM
sleep/wake between phases; --resume makes interrupted candidates and individual
sample artifacts restartable without discarding completed work.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mas_gate_rl.bcp_routing_audit import audit_trace_routing

ROOT = Path(__file__).resolve().parents[2]
COMMON_LAUNCHER = ROOT / "verl" / "mas_gate_rl" / "run_orchestra_eval.sh"


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_tag(value: object) -> str:
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("._-")
    return x or "unknown"


def validate_local_model(path: Path) -> None:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Invalid local HuggingFace checkpoint: {path}")
    if not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"No safetensors weights under local checkpoint: {path}")


def discover_local_checkpoints(checkpoint_root: Path, train_dataset: str, round_id: str) -> list[dict[str, Any]]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {checkpoint_root}")
    manifest_path = checkpoint_root / "manifest.json"
    records: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("dataset", "")).lower() != train_dataset or str(manifest.get("round")) != str(round_id):
            raise ValueError(f"Checkpoint manifest dataset/round mismatch: {manifest_path}")
        for item in manifest.get("checkpoints", []):
            model_path = Path(item["model_path"]).expanduser().resolve()
            records.append({**item, "step": int(item["step"]), "round": str(item.get("round", round_id)),
                            "kind": str(item.get("kind", "checkpoint")), "model_path": str(model_path),
                            "manifest_file": str(manifest_path)})
    else:
        # Support both the older exported ``step_N`` layout and verl's native
        # ``global_step_N/actor/huggingface`` layout.  Full MAS-Gate runs write
        # the latter directly, so requiring a separate export/manifest would
        # make a completed training run impossible to sweep.
        for model_path in checkpoint_root.glob("step_*"):
            if not model_path.is_dir():
                continue
            match = re.fullmatch(r"step_(\d+)", model_path.name)
            if not match:
                continue
            metadata_path = model_path / "checkpoint_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            records.append({**metadata, "step": int(match.group(1)), "round": str(metadata.get("round", round_id)),
                            "kind": str(metadata.get("kind", "checkpoint")), "model_path": str(model_path.resolve()),
                            "checkpoint_layout": "exported_step"})

        native_records: list[dict[str, Any]] = []
        for checkpoint_dir in checkpoint_root.glob("global_step_*"):
            if not checkpoint_dir.is_dir():
                continue
            match = re.fullmatch(r"global_step_(\d+)", checkpoint_dir.name)
            if not match:
                continue
            model_path = checkpoint_dir / "actor" / "huggingface"
            metadata_path = model_path / "checkpoint_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            native_records.append({
                **metadata,
                "step": int(match.group(1)),
                "round": str(metadata.get("round", round_id)),
                "kind": str(metadata.get("kind", "intermediate")),
                "model_path": str(model_path.resolve()),
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "checkpoint_layout": "verl_global_step",
            })
        if native_records:
            final_step = max(int(x["step"]) for x in native_records)
            for record in native_records:
                # Native checkpoints do not carry a semantic label.  Once the
                # run is complete, the largest step is the separately requested
                # final checkpoint and all earlier steps are intermediates.
                if int(record["step"]) == final_step and record.get("kind") == "intermediate":
                    record["kind"] = "final"
            records.extend(native_records)
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda x: int(x["step"])):
        step = int(record["step"])
        if step in seen:
            raise RuntimeError(f"Duplicate local model step {step} under {checkpoint_root}")
        seen.add(step)
        validate_local_model(Path(record["model_path"]))
        unique.append(record)
    if not unique:
        raise RuntimeError(f"No local step_NNNN checkpoints under {checkpoint_root}")
    return unique


def select_requested_checkpoints(
    records: list[dict[str, Any]],
    requested_steps: set[int],
    checkpoint_root: Path,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select exact requested steps, failing before a partial campaign can start."""
    discovered_steps = {int(record["step"]) for record in records}
    missing_steps = sorted(requested_steps - discovered_steps)
    if missing_steps:
        raise RuntimeError(
            f"Missing requested checkpoint steps under {checkpoint_root}: {missing_steps}; "
            f"discovered={sorted(discovered_steps)}"
        )
    unexpected_steps = sorted(discovered_steps - requested_steps)
    selected = [record for record in records if int(record["step"]) in requested_steps]
    return selected, unexpected_steps


def candidate_dir(campaign: Path, candidate: dict[str, Any]) -> Path:
    if candidate["source_type"] == "base":
        return campaign / "base" / "qwen3_8b"
    kind = safe_tag(candidate["kind"])
    name = f"step_{int(candidate['step']):04d}_{kind}"
    return campaign / "checkpoints" / f"round_{safe_tag(candidate['round'])}" / name


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if int(obj.get("num_evaluated", -1)) < 0:
            return None
        return obj
    except Exception:
        return None


def validate_completed_run(
    run_dir: Path,
    *,
    expected_cases: int,
    evaluation_dataset: str,
    judge_model: str,
    max_retrieval_rounds: int | None = None,
) -> dict[str, Any]:
    """Prove a candidate contains every selected case and required BCP judgement."""
    summary = load_summary(run_dir)
    if summary is None:
        raise RuntimeError(f"missing/invalid summary: {run_dir}")
    if int(summary.get("num_evaluated", -1)) != expected_cases:
        raise RuntimeError(
            f"candidate evaluated {summary.get('num_evaluated')} cases; expected {expected_cases}: {run_dir}"
        )
    if int(summary.get("num_judged", -1)) != expected_cases:
        raise RuntimeError(
            f"candidate has {summary.get('num_judged')} scored cases; expected {expected_cases}: {run_dir}"
        )
    sample_paths = sorted((run_dir / "samples").glob("*.json"))
    trace_paths = sorted((run_dir / "traces").glob("*.json"))
    if len(sample_paths) != expected_cases or len(trace_paths) != expected_cases:
        raise RuntimeError(
            f"candidate artifact coverage mismatch samples={len(sample_paths)} traces={len(trace_paths)} "
            f"expected={expected_cases}: {run_dir}"
        )
    expected_indices = set(range(expected_cases))
    sample_indices = {int(x.stem) for x in sample_paths}
    trace_indices = {int(x.stem) for x in trace_paths}
    if sample_indices != expected_indices or trace_indices != expected_indices:
        raise RuntimeError(f"candidate index coverage is not exactly 0..{expected_cases - 1}: {run_dir}")
    if evaluation_dataset in {"bcp", "hotpotqa"}:
        if max_retrieval_rounds is None or max_retrieval_rounds <= 0:
            raise ValueError("positive max_retrieval_rounds is required for retrieval-dataset validation")
        mas_success = int(summary.get("num_mas_execution_success", -1))
        num_errors = int(summary.get("num_errors", -1))
        if mas_success != expected_cases or num_errors != 0:
            raise RuntimeError(
                f"MAS execution does not cover all {expected_cases} cases "
                f"(success={mas_success}, errors={num_errors}): {run_dir}"
            )
        judged_success = 0
        for path in sample_paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            if bool(row.get("success")):
                judged_success += 1
                judgement = row.get("llm_judge")
                if row.get("scoring_method") != f"{evaluation_dataset}_llm_judge" or not isinstance(judgement, dict):
                    raise RuntimeError(f"successful semantic-judge case lacks LLM judgement: {path}")
                if str(judgement.get("judge_model", "")) != judge_model:
                    raise RuntimeError(
                        f"wrong semantic judge model at {path}: {judgement.get('judge_model')!r} != {judge_model!r}"
                    )
                if judgement.get("score") not in (0, 1) or judgement.get("correct") not in ("yes", "no"):
                    raise RuntimeError(f"malformed parsed semantic judgement: {path}")
            elif int(row.get("score", 0) or 0) != 0:
                raise RuntimeError(f"failed MAS case must not count as correct: {path}")
        if judged_success != int(summary.get("num_mas_execution_success", -1)):
            raise RuntimeError(
                f"judge coverage does not match MAS success count {judged_success} != "
                f"{summary.get('num_mas_execution_success')}: {run_dir}"
            )
        # The strict tool-routing contract is specific to BrowseComp-Plus.
        # HotpotQA still requires complete semantic-judge coverage, but its
        # retrieval traces must not be rejected by BCP-only routing rules.
        if evaluation_dataset == "bcp":
            for path in trace_paths:
                trace = json.loads(path.read_text(encoding="utf-8"))
                audit_trace_routing(trace, max_rounds=max_retrieval_rounds, label=str(path))
    return summary


def aggregate(campaign: Path, candidates: list[dict[str, Any]], statuses: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        key = cand["candidate_id"]
        run_dir = candidate_dir(campaign, cand)
        summary = load_summary(run_dir)
        status = statuses.get(key, {})
        rows.append({
            "candidate_id": key,
            "source_type": cand["source_type"],
            "train_dataset": cand["train_dataset"],
            "evaluation_dataset": cand["evaluation_dataset"],
            "round": cand.get("round", ""),
            "step": cand.get("step", ""),
            "kind": cand.get("kind", "base"),
            "model_path": cand.get("model_path", ""),
            "status": status.get("status", "complete" if summary else "pending"),
            "num_evaluated": summary.get("num_evaluated", "") if summary else "",
            "num_correct": summary.get("num_correct", "") if summary else "",
            "accuracy": summary.get("accuracy", "") if summary else "",
            "mas_execution_success_rate": summary.get("mas_execution_success_rate", "") if summary else "",
            "num_errors": summary.get("num_errors", "") if summary else "",
            "results_dir": str(run_dir),
        })
    summary_dir = campaign / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(summary_dir / "checkpoint_accuracy.json", rows)
    fields = list(rows[0]) if rows else []
    tmp = summary_dir / f"checkpoint_accuracy.csv.tmp.{os.getpid()}"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, summary_dir / "checkpoint_accuracy.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the local Qwen3-8B base model and every local MAS-Gate checkpoint")
    ap.add_argument("--evaluation-dataset", choices=["bcp", "hotpotqa", "aime", "gpqa"], required=True)
    ap.add_argument("--train-dataset", choices=["bcp", "hotpotqa", "deepscaler"], required=True)
    ap.add_argument("--round", default="1")
    ap.add_argument("--checkpoint-root", type=Path, required=True,
                    help="Local round directory containing manifest.json, step_N models, or global_step_N/actor/huggingface models")
    ap.add_argument("--base-model", type=Path, default=Path("./models/Qwen3-8B"))
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument(
        "--test-file", type=Path, default=None,
        help="Evaluation JSONL. Defaults to the repository's canonical test file for the selected dataset.",
    )
    ap.add_argument("--mode", choices=["all", "base", "checkpoints"], default="all")
    ap.add_argument("--steps", default="", help="Optional comma-separated checkpoint steps")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpu_id_parts = [part.strip() for part in cuda_visible_devices.split(",") if part.strip()]
    if not gpu_id_parts or any(not part.isdigit() for part in gpu_id_parts):
        raise RuntimeError(
            "This MAS-Gate sweep requires CUDA_VISIBLE_DEVICES to be a non-empty "
            f"comma-separated integer list, got {cuda_visible_devices!r}"
        )
    gpu_ids = [int(part) for part in gpu_id_parts]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must contain unique GPU IDs, got {cuda_visible_devices!r}")
    for tp_env in ("ORCHESTRA_TP", "SUBAGENT_TP", "JUDGE_TP"):
        if tp_env in os.environ and int(os.environ[tp_env]) != len(gpu_ids):
            raise RuntimeError(
                f"{tp_env}={os.environ[tp_env]} does not match the {len(gpu_ids)} visible GPUs "
                f"({cuda_visible_devices})"
            )
    default_test_files = {
        "bcp": ROOT / "datasets" / "BrowseComp-Plus" / "data" / "browsecomp_plus_test.jsonl",
        "hotpotqa": ROOT / "datasets" / "BrowseComp-Plus" / "hotpotqa" / "data" / "hotpotqa_test.jsonl",
        "aime": ROOT / "datasets" / "aime" / "data" / "aime_test.jsonl",
        "gpqa": ROOT / "datasets" / "aime" / "gpqa" / "data" / "gpqa_diamond_test.jsonl",
    }
    test_file = (args.test_file or default_test_files[args.evaluation_dataset]).expanduser().resolve()
    if not test_file.is_file():
        raise FileNotFoundError(f"Missing evaluation JSONL: {test_file}")
    with test_file.open(encoding="utf-8") as f:
        expected_cases = sum(1 for line in f if line.strip())
    if expected_cases <= 0:
        raise RuntimeError(f"Evaluation JSONL is empty: {test_file}")
    if not (args.base_model / "config.json").is_file():
        raise FileNotFoundError(f"Missing local base Orchestra: {args.base_model}")
    records = (
        discover_local_checkpoints(args.checkpoint_root.resolve(), args.train_dataset, str(args.round))
        if args.mode != "base" else []
    )
    selected_steps = {int(x) for x in args.steps.split(",") if x.strip()} if args.steps else None
    if selected_steps is not None:
        records, unexpected_steps = select_requested_checkpoints(
            records, selected_steps, args.checkpoint_root
        )
        if unexpected_steps:
            print(f"[campaign] ignoring non-requested checkpoint steps: {unexpected_steps}", flush=True)

    candidates: list[dict[str, Any]] = []
    if args.mode in {"all", "base"}:
        candidates.append({
            "candidate_id": "base_qwen3_8b", "source_type": "base",
            "train_dataset": args.train_dataset, "evaluation_dataset": args.evaluation_dataset,
            "round": str(args.round), "step": "", "kind": "base",
            "model_path": str(args.base_model.resolve()),
        })
    if args.mode in {"all", "checkpoints"}:
        for r in records:
            cid = f"round{r['round']}_step{int(r['step']):04d}_{safe_tag(r['kind'])}"
            candidates.append({**r, "candidate_id": cid, "source_type": "local_checkpoint",
                               "train_dataset": args.train_dataset, "evaluation_dataset": args.evaluation_dataset})

    campaign = args.campaign_root.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    lock = (campaign / ".campaign.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another process owns evaluation campaign {campaign}") from exc

    manifest_path = campaign / "campaign_manifest.json"
    manifest = {
        "schema_version": "mas_gate_checkpoint_eval_campaign/1",
        "project": "mas-gate",
        "train_dataset": args.train_dataset,
        "evaluation_dataset": args.evaluation_dataset,
        "evaluation_test_file": str(test_file),
        "expected_cases_per_candidate": expected_cases,
        "round": str(args.round),
        "gpu_ids": gpu_ids,
        "base_orchestra": str(args.base_model.resolve()),
        "sub_agent": os.environ.get("SUBAGENT_MODEL_PATH", "./models/Qwen3-30B-A3B"),
        "bcp_judge_model": os.environ.get("JUDGE_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507") if args.evaluation_dataset in {"bcp", "hotpotqa"} else "",
        "bcp_judge_model_path": os.environ.get("JUDGE_MODEL_PATH", "./models/Qwen3-30B-A3B") if args.evaluation_dataset in {"bcp", "hotpotqa"} else "",
        "bcp_judge_scoring_method": "BrowseComp-Plus/judge.py LLM judge" if args.evaluation_dataset in {"bcp", "hotpotqa"} else "",
        "subagent_max_model_len": int(os.environ.get("SUB_AGENT_MAX_MODEL_LEN", "32768")),
        "subagent_requested_max_output_tokens": int(os.environ.get("SUB_AGENT_MAX_TOKENS", "8192")),
        "subagent_min_completion_tokens": int(os.environ.get("SUB_AGENT_MIN_COMPLETION_TOKENS", "8192")),
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "mode": args.mode,
        "created_at": datetime.now().astimezone().isoformat(),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    if manifest_path.exists() and args.resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Never silently reuse one campaign for a different candidate set.
        if [x.get("candidate_id") for x in old.get("candidates", [])] != [x["candidate_id"] for x in candidates]:
            raise RuntimeError(f"Existing campaign manifest does not match requested candidates: {manifest_path}")
    else:
        if manifest_path.exists():
            raise FileExistsError(f"Campaign already exists; choose a new EVAL_CAMPAIGN_ID: {campaign}")
        atomic_json(manifest_path, manifest)

    statuses_path = campaign / "campaign_status.json"
    statuses: dict[str, dict[str, Any]] = {}
    if statuses_path.is_file() and args.resume:
        statuses = json.loads(statuses_path.read_text(encoding="utf-8")).get("candidates", {})

    print(f"[campaign] {campaign}", flush=True)
    print(
        f"[campaign] candidates={len(candidates)} GPUs={cuda_visible_devices} "
        f"sub-agent={manifest['sub_agent']}",
        flush=True,
    )
    for i, cand in enumerate(candidates, 1):
        cid = cand["candidate_id"]
        run_dir = candidate_dir(campaign, cand)
        existing = load_summary(run_dir)
        if args.resume and existing is not None and (run_dir / "COMPLETED.json").is_file():
            try:
                existing = validate_completed_run(
                    run_dir, expected_cases=expected_cases,
                    evaluation_dataset=args.evaluation_dataset,
                    judge_model=os.environ.get("JUDGE_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507"),
                    max_retrieval_rounds=(
                        int(os.environ.get("MAS_GATE_BCP_RETRIEVAL_MAX_ROUNDS", "5"))
                        if args.evaluation_dataset in {"bcp", "hotpotqa"} else None
                    ),
                )
            except Exception as exc:
                # A stale/partial completion marker must not make the full sweep
                # abort or silently skip data.  Fall through to the evaluator;
                # its per-case resume path repairs only missing/failed cases.
                print(f"[resume revalidate] {cid} will be repaired: {type(exc).__name__}: {exc}", flush=True)
            else:
                print(f"[skip {i}/{len(candidates)}] {cid} already complete accuracy={existing.get('accuracy')}", flush=True)
                statuses[cid] = {"status": "complete", "results_dir": str(run_dir), "accuracy": existing.get("accuracy")}
                aggregate(campaign, candidates, statuses)
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        # Preserve failed-run evidence before a resumable repair.  The active
        # candidate directory remains append-only from the campaign's point of
        # view: every repair gets its own attempt record while the evaluator
        # reuses successful per-case artifacts.
        previous_failed = run_dir / "FAILED.json"
        previous_running = run_dir / "RUNNING.json"
        repair_attempt = 0
        # A user-initiated stop can leave RUNNING.json without FAILED.json.
        # Preserve that interruption as append-only evidence before resuming;
        # never silently overwrite the prior run record.
        if args.resume and (previous_failed.is_file() or previous_running.is_file()):
            repair_root = run_dir / "repair_attempts"
            repair_root.mkdir(parents=True, exist_ok=True)
            existing_repairs = sorted(repair_root.glob("attempt_*"))
            repair_attempt = len(existing_repairs) + 1
            repair_dir = repair_root / f"attempt_{repair_attempt:03d}"
            repair_dir.mkdir(parents=True, exist_ok=True)
            preserved = []
            if previous_failed.is_file():
                (repair_dir / "FAILED.json").write_bytes(previous_failed.read_bytes())
                preserved.append(str(previous_failed))
            if previous_running.is_file():
                (repair_dir / "INTERRUPTED.json").write_bytes(previous_running.read_bytes())
                preserved.append(str(previous_running))
            atomic_json(repair_dir / "REPAIR.json", {
                "attempt": repair_attempt,
                "started_at": datetime.now().astimezone().isoformat(),
                "source_records": preserved,
                "mode": "resume_existing_case_artifacts",
            })
            print(f"[resume repair] {cid} preserving prior run evidence at {repair_dir}", flush=True)
        run_record = {
            "schema_version": "mas_gate_checkpoint_eval_run/1", **cand,
            "gpu_ids": gpu_ids, "sub_agent": manifest["sub_agent"],
            "results_dir": str(run_dir), "started_at": datetime.now().astimezone().isoformat(),
            "status": "running",
        }
        atomic_json(run_dir / "RUNNING.json", run_record)
        statuses[cid] = {"status": "running", "results_dir": str(run_dir), "started_at": run_record["started_at"]}
        atomic_json(statuses_path, {"updated_at": datetime.now().astimezone().isoformat(), "candidates": statuses})
        print(f"[run {i}/{len(candidates)}] {cid} -> {run_dir}", flush=True)

        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "EVAL_OUTPUT_DIR": str(run_dir),
            "EVAL_RUN_ID": cid,
            "EVAL_RESUME": "1" if args.resume else "0",
            "MAS_GATE_EVAL_DATASET": args.evaluation_dataset,
            "MAS_GATE_EVAL_TEST_FILE": str(test_file),
            "SUBAGENT_MODEL_PATH": env.get("SUBAGENT_MODEL_PATH", "./models/Qwen3-30B-A3B"),
            "JUDGE_MODEL_PATH": env.get("JUDGE_MODEL_PATH", "./models/Qwen3-30B-A3B"),
            "JUDGE_MODEL_NAME": env.get("JUDGE_MODEL_NAME", "Qwen3-30B-A3B-Instruct-2507"),
        })
        env.pop("ORCHESTRA_MODEL_PATH", None)
        env["ORCHESTRA_MODEL_PATH"] = cand["model_path"]

        rc = 0
        if args.dry_run:
            print(f"[dry-run] ORCHESTRA_MODEL_PATH={cand['model_path']}; bash {COMMON_LAUNCHER}", flush=True)
        else:
            candidate_attempts = max(1, int(env.get("EVAL_CANDIDATE_ATTEMPTS", "1")))
            retry_backoff = max(0.0, float(env.get("EVAL_CANDIDATE_RETRY_BACKOFF_SEC", "30")))
            for candidate_attempt in range(1, candidate_attempts + 1):
                print(
                    f"[candidate attempt {candidate_attempt}/{candidate_attempts}] {cid}",
                    flush=True,
                )
                candidate_timeout = max(300, int(env.get("EVAL_CANDIDATE_TIMEOUT_SEC", "21600")))
                proc = subprocess.Popen(
                    ["bash", str(COMMON_LAUNCHER)], cwd=str(ROOT), env=env,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    rc = proc.wait(timeout=candidate_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    print(
                        f"[candidate timeout] {cid} exceeded {candidate_timeout}s; terminating its process group",
                        file=sys.stderr, flush=True,
                    )
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        rc = proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        rc = proc.wait()
                    if rc == 0:
                        rc = 124
                if rc == 0:
                    break
                if candidate_attempt < candidate_attempts:
                    print(
                        f"[candidate retry] {cid} rc={rc}; sleeping {retry_backoff:g}s before resumable retry",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(retry_backoff)

        finished = datetime.now().astimezone().isoformat()
        summary = load_summary(run_dir)
        validation_error = ""
        if rc == 0 and not args.dry_run:
            try:
                summary = validate_completed_run(
                    run_dir, expected_cases=expected_cases,
                    evaluation_dataset=args.evaluation_dataset,
                    judge_model=env["JUDGE_MODEL_NAME"],
                    max_retrieval_rounds=(
                        int(env.get("MAS_GATE_BCP_RETRIEVAL_MAX_ROUNDS", "5"))
                        if args.evaluation_dataset in {"bcp", "hotpotqa"} else None
                    ),
                )
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                print(f"[validation failed] {cid}: {validation_error}", file=sys.stderr, flush=True)
                rc = 97
        if rc == 0 and (args.dry_run or summary is not None):
            state = {**run_record, "status": "dry_run" if args.dry_run else "complete", "finished_at": finished,
                     "return_code": rc, "summary": summary}
            (run_dir / "RUNNING.json").unlink(missing_ok=True)
            (run_dir / "FAILED.json").unlink(missing_ok=True)
            atomic_json(run_dir / ("DRY_RUN.json" if args.dry_run else "COMPLETED.json"), state)
            if repair_attempt:
                atomic_json(run_dir / "repair_attempts" / f"attempt_{repair_attempt:03d}" / "RESULT.json", {
                    "status": state["status"], "finished_at": finished, "return_code": rc,
                })
            statuses[cid] = {"status": state["status"], "results_dir": str(run_dir),
                             "accuracy": summary.get("accuracy") if summary else None, "finished_at": finished}
        else:
            state = {**run_record, "status": "failed", "finished_at": finished, "return_code": rc,
                     "summary_present": summary is not None, "validation_error": validation_error}
            (run_dir / "RUNNING.json").unlink(missing_ok=True)
            atomic_json(run_dir / "FAILED.json", state)
            if repair_attempt:
                atomic_json(run_dir / "repair_attempts" / f"attempt_{repair_attempt:03d}" / "RESULT.json", {
                    "status": state["status"], "finished_at": finished, "return_code": rc,
                    "validation_error": validation_error,
                })
            statuses[cid] = {"status": "failed", "results_dir": str(run_dir), "return_code": rc, "finished_at": finished}
            print(f"[failed] {cid} rc={rc}", file=sys.stderr, flush=True)
        atomic_json(statuses_path, {"updated_at": finished, "candidates": statuses})
        aggregate(campaign, candidates, statuses)
        if statuses[cid]["status"] == "failed" and not args.continue_on_error:
            return rc or 1

    failed = [k for k, v in statuses.items() if v.get("status") == "failed"]
    complete = [k for k, v in statuses.items() if v.get("status") == "complete"]
    final = {"project": "mas-gate", "campaign": str(campaign), "candidate_count": len(candidates),
             "complete_count": len(complete), "failed_count": len(failed), "failed_candidates": failed,
             "finished_at": datetime.now().astimezone().isoformat(), "dry_run": args.dry_run}
    atomic_json(campaign / "CAMPAIGN_COMPLETED.json", final)
    print("[campaign-summary] " + json.dumps(final, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
