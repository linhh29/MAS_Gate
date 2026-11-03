"""Per-sample BrowseComp-Plus quantification hook (run after each Skill-MAS trace)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from building.skill_mas.process_trace_layout import (
    sample_trace_json_dir,
    skill_mas_sample_log_subdir,
)
from quantification.config import QuantificationConfig
from quantification.adapters.trace_adapter import load_reasoning_graph_from_trace
from quantification.metrics import QuantificationEngine
from quantification.run_logger import RunLogger, set_run_logger


def trace_json_path(trace_root: Path, index: int, task_id: str | int) -> Path:
    stem = skill_mas_sample_log_subdir(index, task_id)
    return sample_trace_json_dir(trace_root) / f"{stem}.json"


def metrics_json_path(trace_root: Path, index: int, task_id: str | int) -> Path:
    stem = skill_mas_sample_log_subdir(index, task_id)
    return trace_root / "quantification_metrics" / f"{stem}_metrics.json"


def metrics_log_path(trace_root: Path, index: int, task_id: str | int) -> Path:
    stem = skill_mas_sample_log_subdir(index, task_id)
    return trace_root / "quantification_metrics" / f"{stem}_metrics.log"


def ground_truth_for_index(
    jsonl_path: Path,
    index: int,
    task_id: str | int | None = None,
) -> str:
    rows: list[dict] = []
    for line_no, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"JSONL line {line_no} is not an object")
        rows.append(row)

    if 0 <= index < len(rows):
        answer = rows[index].get("answer")
        if answer is not None:
            return str(answer).strip()

    if task_id is not None:
        keys = {str(task_id), str(int(task_id)) if str(task_id).isdigit() else str(task_id)}
        for row in rows:
            row_id = row.get("id")
            if row_id is not None and str(row_id) in keys:
                answer = row.get("answer")
                if answer is not None:
                    return str(answer).strip()

    raise ValueError(
        f"ground truth not found for index={index} task_id={task_id!r} in {jsonl_path}"
    )


def build_quant_config(
    *,
    concurrency: int = 16,
    mas_concurrency: int | None = None,
    **_: object,
) -> QuantificationConfig:
    """Build quantification config; model endpoints come from model_config.json only."""
    conc = max(1, int(concurrency))
    scoring_max_seqs = int(os.getenv("QUANTIFICATION_VLLM_SCORING_MAX_NUM_SEQS", "1"))
    return QuantificationConfig.from_env(
        scoring_concurrency=conc,
        embedding_concurrency=conc,
        trace_concurrency=1,
        vllm_scoring_max_num_seqs=max(1, scoring_max_seqs),
    )


def find_trace_path(
    trace_root: Path,
    index: int,
    task_id: str | int | None = None,
) -> Path | None:
    """Locate an existing per-sample trace JSON under ``sample_trace_json/``."""
    nested = sample_trace_json_dir(trace_root)
    if not nested.is_dir():
        return None

    if task_id is not None:
        candidate = trace_json_path(trace_root, index, task_id)
        if candidate.is_file():
            return candidate

    prefix = f"{int(index):04d}__"
    matches = sorted(nested.glob(f"{prefix}*_t0.json"))
    if matches:
        return matches[0]

    tid = str(task_id) if task_id is not None else ""
    if tid:
        matches = sorted(nested.glob(f"*__{tid}_t0.json"))
        if matches:
            return matches[0]
    return None


def find_metrics_path(
    trace_root: Path,
    index: int,
    task_id: str | int | None = None,
) -> Path | None:
    if task_id is not None:
        candidate = metrics_json_path(trace_root, index, task_id)
        if candidate.is_file():
            return candidate

    metrics_dir = trace_root / "quantification_metrics"
    if not metrics_dir.is_dir():
        return None

    prefix = f"{int(index):04d}__"
    matches = sorted(metrics_dir.glob(f"{prefix}*_t0_metrics.json"))
    if matches:
        return matches[0]

    tid = str(task_id) if task_id is not None else ""
    if tid:
        matches = sorted(metrics_dir.glob(f"*__{tid}_t0_metrics.json"))
        if matches:
            return matches[0]
    return None


def has_trace(trace_root: Path, index: int, task_id: str | int | None = None) -> bool:
    return find_trace_path(trace_root, index, task_id) is not None


def has_metrics(trace_root: Path, index: int, task_id: str | int | None = None) -> bool:
    return find_metrics_path(trace_root, index, task_id) is not None


def _task_id_from_trace_path(trace_path: Path, index: int) -> str | int:
    stem = trace_path.stem
    if "__" in stem:
        return stem.split("__", 1)[1].rsplit("_t", 1)[0]
    return index


def _log_j_scoring_skip(
    *,
    trace_root: Path,
    index: int,
    task_id: str | int,
    trace_path: Path,
    ground_truth: str,
    engine: QuantificationEngine,
    reason: str,
    learned_limit: int | None = None,
    sample_max_prompt_tokens: int | None = None,
    worst_node: str = "",
) -> Path:
    """Append a skip record to the per-sample metrics log (no metrics JSON)."""
    out_json = metrics_json_path(trace_root, index, task_id)
    out_log = metrics_log_path(trace_root, index, task_id)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(out_log)
    logger.log_config(
        {
            "trace_json": str(trace_path),
            "output_json": str(out_json),
            "ground_truth": ground_truth,
            "backend": engine.config.backend,
            "scoring_model": engine.config.scoring_model,
            "embedding_model": engine.config.embedding_model,
            "scoring_model_path": engine.config.scoring_model_path,
            "embedding_model_path": engine.config.embedding_model_path,
            "scoring_base_url": engine.config.scoring_base_url,
            "embedding_base_url": engine.config.embedding_base_url,
        }
    )
    logger.info("=== skipped: j_scoring ===")
    logger.info(f"  reason = {reason}")
    if sample_max_prompt_tokens is not None:
        logger.info(f"  sample_max_prompt_tokens = {sample_max_prompt_tokens}")
    if worst_node:
        logger.info(f"  worst_node = {worst_node}")
    if learned_limit is not None:
        logger.info(f"  learned_max_prompt_tokens = {learned_limit}")
    logger.info(f"  metrics_json = {out_json} (not written)")
    logger.finish({"status": "skipped", "skip_reason": "j_scoring"})
    return out_log


def _reload_j_scoring_engine(engine: QuantificationEngine) -> None:
    engine.release_gpu_engines()
    engine.begin_j_phase()


def _write_metrics_result(
    *,
    trace_root: Path,
    index: int,
    task_id: str | int,
    trace_path: Path,
    ground_truth: str,
    engine: QuantificationEngine,
    result,
) -> Path:
    out_json = metrics_json_path(trace_root, index, task_id)
    out_log = metrics_log_path(trace_root, index, task_id)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(out_log)
    set_run_logger(logger)
    try:
        logger.log_config(
            {
                "trace_json": str(trace_path),
                "output_json": str(out_json),
                "ground_truth": ground_truth,
                "backend": engine.config.backend,
                "scoring_model": engine.config.scoring_model,
                "embedding_model": engine.config.embedding_model,
                "scoring_model_path": engine.config.scoring_model_path,
                "embedding_model_path": engine.config.embedding_model_path,
                "scoring_base_url": engine.config.scoring_base_url,
                "embedding_base_url": engine.config.embedding_base_url,
            }
        )
        payload = result.to_dict()
        payload["num_nodes"] = len(result.node_metrics)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.finish({"num_nodes": payload["num_nodes"], "metrics_json": str(out_json)})
        return out_json
    except Exception as exc:
        logger.info(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        set_run_logger(None)


def quantify_traces_two_phase(
    *,
    trace_root: Path,
    jsonl_path: Path,
    engine: QuantificationEngine,
    indices: list[int],
    skip_if_exists: bool = True,
) -> list[tuple[int, Path | None, str | None]]:
    """
  Quantify many traces with only two GPU model loads (scoring then embedding).

  Phase 1: load Qwen scoring model once, compute J for all pending samples.
  Phase 2: load embedding model once, compute K and write metrics JSONs.
    """
    pending: list[tuple[int, str | int, Path]] = []
    for idx in indices:
        trace_path = find_trace_path(trace_root, idx)
        if trace_path is None:
            continue
        task_id = _task_id_from_trace_path(trace_path, idx)
        if skip_if_exists and has_metrics(trace_root, idx, task_id):
            continue
        pending.append((idx, task_id, trace_path))

    results: list[tuple[int, Path | None, str | None]] = []
    for idx in indices:
        if find_trace_path(trace_root, idx) is None:
            results.append((idx, None, "trace_missing"))

    if not pending:
        return results

    total = len(pending)
    print(
        f"[browsecomp-quantify] two-phase batch: {total} samples "
        f"(1 scoring load + 1 embedding load)",
        flush=True,
    )

    j_states: dict[int, object] = {}
    meta: dict[int, tuple[str | int, Path, str]] = {}

    from quantification.clients.j_prompt_token_limit import (
        JPromptTokenLimitExceeded,
        finite_max_j_prompt_tokens,
        format_learned_j_prompt_token_limit,
        max_prompt_tokens_for_graph,
        prompt_tokens_allowed,
        reset_max_j_prompt_tokens,
    )
    from quantification.clients.vllm_runtime import vllm_process_lock

    reset_max_j_prompt_tokens()

    try:
        with vllm_process_lock:
            engine.begin_j_phase()
            for n, (idx, task_id, trace_path) in enumerate(pending, start=1):
                ground_truth = ground_truth_for_index(jsonl_path, idx, task_id)
                meta[idx] = (task_id, trace_path, ground_truth)
                print(
                    f"[browsecomp-quantify] J-phase {n}/{total} index={idx}",
                    flush=True,
                )
                graph = load_reasoning_graph_from_trace(trace_path, ground_truth=ground_truth)
                format_context = engine.client.format_context
                sample_max_tokens, worst_node = max_prompt_tokens_for_graph(
                    graph,
                    engine.client._tokenizer,
                    format_context=format_context,
                    ground_truth=ground_truth,
                )
                learned_limit = finite_max_j_prompt_tokens()
                if not prompt_tokens_allowed(sample_max_tokens):
                    reason = (
                        f"max_prompt_tokens={sample_max_tokens} "
                        f"> learned_limit={format_learned_j_prompt_token_limit()}"
                    )
                    _log_j_scoring_skip(
                        trace_root=trace_root,
                        index=idx,
                        task_id=task_id,
                        trace_path=trace_path,
                        ground_truth=ground_truth,
                        engine=engine,
                        reason=reason,
                        learned_limit=learned_limit,
                        sample_max_prompt_tokens=sample_max_tokens,
                        worst_node=worst_node,
                    )
                    print(
                        f"[browsecomp-quantify] SKIP index={idx} ({reason})",
                        flush=True,
                    )
                    results.append((idx, None, f"j_skip: {reason}"))
                    continue
                try:
                    j_states[idx] = engine.compute_j_phase(graph)
                except JPromptTokenLimitExceeded as exc:
                    learned_limit = finite_max_j_prompt_tokens()
                    print(
                        f"[browsecomp-quantify] OOM/limit skip index={idx} "
                        f"learned_limit={format_learned_j_prompt_token_limit()}",
                        flush=True,
                    )
                    _log_j_scoring_skip(
                        trace_root=trace_root,
                        index=idx,
                        task_id=task_id,
                        trace_path=trace_path,
                        ground_truth=ground_truth,
                        engine=engine,
                        reason=str(exc),
                        learned_limit=learned_limit,
                        sample_max_prompt_tokens=sample_max_tokens,
                        worst_node=worst_node,
                    )
                    print(
                        f"[browsecomp-quantify] SKIP index={idx} ({exc})",
                        flush=True,
                    )
                    results.append((idx, None, f"j_skip: {exc}"))
                    _reload_j_scoring_engine(engine)
                    continue
                except Exception as exc:
                    err_text = f"{type(exc).__name__}: {exc}"
                    is_oom = (
                        "out of memory" in err_text.lower()
                        or "enginedeaderror" in err_text.lower()
                    )
                    if is_oom:
                        from quantification.clients.j_prompt_token_limit import (
                            tighten_max_j_prompt_tokens_from_oom,
                        )

                        if finite_max_j_prompt_tokens() is None:
                            tighten_max_j_prompt_tokens_from_oom(sample_max_tokens)
                        learned_limit = finite_max_j_prompt_tokens()
                        _log_j_scoring_skip(
                            trace_root=trace_root,
                            index=idx,
                            task_id=task_id,
                            trace_path=trace_path,
                            ground_truth=ground_truth,
                            engine=engine,
                            reason=err_text,
                            learned_limit=learned_limit,
                            sample_max_prompt_tokens=sample_max_tokens,
                            worst_node=worst_node,
                        )
                        print(
                            f"[browsecomp-quantify] SKIP index={idx} ({err_text})",
                            flush=True,
                        )
                        results.append((idx, None, f"j_skip: {err_text}"))
                        _reload_j_scoring_engine(engine)
                        continue
                    raise

            engine.begin_k_phase()
            k_pending = [(idx, meta[idx]) for idx in j_states]
            for n, (idx, (task_id, trace_path, ground_truth)) in enumerate(k_pending, start=1):
                print(
                    f"[browsecomp-quantify] K-phase {n}/{len(k_pending)} index={idx}",
                    flush=True,
                )
                try:
                    result = engine.finalize_metrics(j_states[idx])
                    out = _write_metrics_result(
                        trace_root=trace_root,
                        index=idx,
                        task_id=task_id,
                        trace_path=trace_path,
                        ground_truth=ground_truth,
                        engine=engine,
                        result=result,
                    )
                    results.append((idx, out, None))
                    print(
                        f"[browsecomp-quantify] done index={idx} -> {out}",
                        flush=True,
                    )
                except Exception as exc:
                    results.append((idx, None, str(exc)))
                    print(
                        f"[browsecomp-quantify] FAILED index={idx} error={exc}",
                        flush=True,
                    )
            engine.release_gpu_engines()
    finally:
        pass

    return results


def quantify_sample(
    *,
    trace_root: Path,
    index: int,
    task_id: str | int,
    jsonl_path: Path,
    engine: QuantificationEngine,
    skip_if_exists: bool = True,
) -> Path | None:
    """
    Quantify one Skill-MAS trace. Returns metrics path on success, None if skipped.
    Raises on failure.
    """
    if skip_if_exists and has_metrics(trace_root, index, task_id):
        return find_metrics_path(trace_root, index, task_id)

    trace_path = find_trace_path(trace_root, index, task_id)
    if trace_path is None:
        raise FileNotFoundError(f"trace not found for index={index} under {trace_root}")

    ground_truth = ground_truth_for_index(jsonl_path, index, task_id)
    out_json = metrics_json_path(trace_root, index, task_id)
    out_log = metrics_log_path(trace_root, index, task_id)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    backend = (
        engine.config.resolved_backend or engine.config.backend or ""
    ).strip().lower()

    def _run_compute() -> object:
        if backend == "vllm":
            from quantification.clients.vllm_runtime import release_all_engines, vllm_process_lock

            with vllm_process_lock:
                release_all_engines()
                return engine.compute_from_trace(trace_path, ground_truth=ground_truth)
        return engine.compute_from_trace(trace_path, ground_truth=ground_truth)

    logger = RunLogger(out_log)
    set_run_logger(logger)
    try:
        logger.log_config(
            {
                "trace_json": str(trace_path),
                "output_json": str(out_json),
                "ground_truth": ground_truth,
                "backend": engine.config.backend,
                "scoring_model": engine.config.scoring_model,
                "embedding_model": engine.config.embedding_model,
                "scoring_model_path": engine.config.scoring_model_path,
                "embedding_model_path": engine.config.embedding_model_path,
                "scoring_base_url": engine.config.scoring_base_url,
                "embedding_base_url": engine.config.embedding_base_url,
            }
        )
        result = _run_compute()
        payload = result.to_dict()
        payload["num_nodes"] = len(result.node_metrics)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.finish({"num_nodes": payload["num_nodes"], "metrics_json": str(out_json)})
        return out_json
    except Exception as exc:
        logger.info(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        set_run_logger(None)


def quantify_existing_traces(
    *,
    trace_root: Path,
    jsonl_path: Path,
    engine: QuantificationEngine,
    indices: list[int] | None = None,
    skip_if_exists: bool = True,
) -> list[tuple[int, Path | None, str | None]]:
    """
    Quantify samples that already have traces.
    Returns (index, metrics_path, error).
    """
    nested = sample_trace_json_dir(trace_root)
    if not nested.is_dir():
        return []

    if indices is None:
        indices = []
        for path in sorted(nested.glob("*_t0.json")):
            stem = path.stem
            if "__" in stem:
                idx_part = stem.split("__", 1)[0]
                if idx_part.isdigit():
                    indices.append(int(idx_part))
        indices = sorted(set(indices))

    backend = (
        engine.config.resolved_backend or engine.config.backend or ""
    ).strip().lower()
    if backend == "vllm":
        return quantify_traces_two_phase(
            trace_root=trace_root,
            jsonl_path=jsonl_path,
            engine=engine,
            indices=indices,
            skip_if_exists=skip_if_exists,
        )

    results: list[tuple[int, Path | None, str | None]] = []
    for idx in indices:
        trace_path = find_trace_path(trace_root, idx)
        if trace_path is None:
            results.append((idx, None, "trace_missing"))
            continue
        stem = trace_path.stem
        task_id = stem.split("__", 1)[1].rsplit("_t", 1)[0] if "__" in stem else idx
        try:
            out = quantify_sample(
                trace_root=trace_root,
                index=idx,
                task_id=task_id,
                jsonl_path=jsonl_path,
                engine=engine,
                skip_if_exists=skip_if_exists,
            )
            results.append((idx, out, None))
        except Exception as exc:
            results.append((idx, None, str(exc)))
    return results
