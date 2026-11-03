#!/usr/bin/env python3
"""Evaluate a trained MAS-Gate Orchestra with entirely local vLLM models.

The inference path deliberately matches training:
  trained Qwen3-8B -> XML Orchestra -> MASGraph -> generated workflow ->
  local Qwen3-30B sub-agents -> final answer -> dataset accuracy.

BCP evaluation uses the real BrowseComp-Plus BM25 corpus/index. Gold/evidence
fields are never injected into inference prompts. Quantification/BGE is not run:
it is a training reward, whereas this command reports answer accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "building", ROOT / "datasets", ROOT / "datasets" / "BrowseComp-Plus"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from building.formalize.compiler import compile_graph_to_code
from building.skill_mas.build import make_planner_call_fn, make_text_call_fn, run_generated_workflow_with_retries
from building.skill_mas.openai_async_client import AsyncOpenAIClient
from openai import AsyncOpenAI
from mas_gate_rl.reward import (
    _collect_usage_totals,
    _extract_sub_agent_logs,
    _is_bcp_dataset,
    _is_hotpotqa_dataset,
    _judge_score,
    _make_retrieval_tool_call_fn,
)
from mas_gate_rl.bcp_routing_audit import audit_graph_routing, audit_trace_routing
from mas_gate_rl.xml_policy import ORCHESTRA_XML_SYSTEM_PROMPT, build_orchestra_user_prompt, graph_from_rollout_xml
from judge import grader_template_for, judge_answer
from aime.gpqa.gpqa_score import GPQAScorer


def _localhost_url(url: str, label: str) -> str:
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme not in {"http", "https"} or u.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"{label} must be a localhost URL; external model APIs are forbidden: {url!r}")
    return url.rstrip("/")


def _server_root(base_url: str) -> str:
    x = base_url.rstrip("/")
    return x[:-3] if x.endswith("/v1") else x


def _post(url: str, timeout: float = 180.0) -> bytes:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def _wait_vllm_idle(root: str, timeout: float) -> None:
    """Drain every in-flight request before sleeping a phase-exclusive server."""
    import re

    deadline = time.time() + timeout
    stable = 0
    last = "metrics unavailable"
    metrics_url = f"{root}/metrics"
    while time.time() < deadline:
        try:
            def fetch_metrics() -> str:
                with urllib.request.urlopen(metrics_url, timeout=10.0) as resp:
                    return resp.read().decode("utf-8", "replace")
            text = await asyncio.to_thread(fetch_metrics)

            def total(name: str) -> float:
                values = re.findall(
                    rf"^{re.escape(name)}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$",
                    text,
                    re.MULTILINE,
                )
                if not values:
                    raise RuntimeError(f"missing metric {name}")
                return sum(float(x) for x in values)

            running = total("vllm:num_requests_running")
            waiting = total("vllm:num_requests_waiting")
            last = f"running={running:g}, waiting={waiting:g}"
            stable = stable + 1 if running == 0 and waiting == 0 else 0
            if stable >= 3:
                print(f"[vllm] {root} drained: {last}", flush=True)
                return
        except Exception as exc:
            stable = 0
            last = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(0.5)
    raise TimeoutError(f"vLLM {root} did not drain before sleep: {last}")


async def _set_sleep(base_url: str, sleeping: bool, *, level: int = 1, timeout: float = 300.0) -> None:
    root = _server_root(base_url)
    try:
        state = await asyncio.to_thread(_get_json, f"{root}/is_sleeping", 10.0)
        if bool(state.get("is_sleeping")) is sleeping:
            print(f"[vllm] {root} already {'sleeping' if sleeping else 'awake'}", flush=True)
            return
    except Exception:
        pass
    if sleeping:
        await _wait_vllm_idle(root, timeout)
    endpoint = f"{root}/sleep?level={level}" if sleeping else f"{root}/wake_up"
    await asyncio.to_thread(_post, endpoint, timeout)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            state = await asyncio.to_thread(_get_json, f"{root}/is_sleeping", 10.0)
            if bool(state.get("is_sleeping")) is sleeping:
                print(f"[vllm] {root} -> {'sleep' if sleeping else 'awake'}", flush=True)
                return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"vLLM {root} did not reach sleeping={sleeping} within {timeout}s")


async def _semantic_health(base_url: str, model: str, role: str, stage: str) -> None:
    """Run the same fail-fast semantic gate used by persistent training."""
    script = ROOT / "verl" / "mas_gate_rl" / "vllm_semantic_health.py"
    command = [
        sys.executable, str(script), "--base-url", base_url.rstrip("/"),
        "--model", model, "--role", role, "--stage", stage,
    ]
    completed = await asyncio.to_thread(
        subprocess.run, command, cwd=str(ROOT), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"semantic health failed for {role} at {stage} rc={completed.returncode}"
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{n}")
            rows.append(obj)
    return rows


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _validate_artifact_identity(
    artifact: dict[str, Any], *, idx: int, task_id: str, question: str, path: Path
) -> None:
    if int(artifact.get("index", -1)) != idx:
        raise RuntimeError(f"artifact index mismatch in {path}: expected {idx}")
    if str(artifact.get("task_id", "")) != task_id:
        raise RuntimeError(f"artifact task_id mismatch in {path}: expected {task_id!r}")
    if str(artifact.get("question", "")).strip() != question.strip():
        raise RuntimeError(f"artifact question mismatch in {path}")
    stored_question_hash = str(artifact.get("question_sha256", "")).strip()
    if stored_question_hash and stored_question_hash != _question_sha256(question):
        raise RuntimeError(f"artifact question hash mismatch in {path}")


def _invalidate_bcp_resume_routing_cases(
    *,
    indices: list[int],
    designs_dir: Path,
    samples_dir: Path,
    traces_dir: Path,
    max_retrieval_rounds: int,
) -> list[dict[str, Any]]:
    """Invalidate successful resume cases whose persisted BCP routing violates the strict contract."""
    if max_retrieval_rounds <= 0:
        raise ValueError("max_retrieval_rounds must be positive")
    invalidated: list[dict[str, Any]] = []
    for idx in indices:
        sample_path = samples_dir / f"{idx}.json"
        trace_path = traces_dir / f"{idx}.json"
        if not sample_path.is_file():
            continue
        try:
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not ("score" in sample and bool(sample.get("success"))):
            continue
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            audit_trace_routing(
                trace,
                max_rounds=max_retrieval_rounds,
                label=str(trace_path),
            )
        except Exception as exc:
            invalidated.append({
                "index": idx,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            # COSEC: idx is range-checked against the loaded dataset before this
            # helper is called, and all deletion targets are fixed child paths.
            for artifact in (designs_dir / f"{idx}.json", sample_path, trace_path):
                artifact.unlink(missing_ok=True)
    return invalidated


def _sample_fields(dataset: str, idx: int, row: dict[str, Any]) -> tuple[str, str, str]:
    if dataset == "bcp":
        return str(row.get("query_id", idx)), str(row.get("query", "")), str(row.get("answer", ""))
    return str(row.get("id", idx)), str(row.get("question", "")), str(row.get("answer", ""))


def _graph_json(graph: Any) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n.id,
                "role": n.role,
                "task": n.task,
                "goal": n.goal,
                "is_final": bool(n.is_final),
                "execution_mode": n.execution_mode,
            }
            for n in graph.nodes
        ],
        "edges": [
            {"from": e.f, "to": e.t, "type": getattr(e.type, "value", str(e.type))}
            for e in graph.edges
        ],
        "topology": getattr(graph.topology, "value", str(graph.topology)),
    }


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.lower()
    if dataset not in {"bcp", "aime", "hotpotqa", "gpqa"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    orchestra_url = _localhost_url(args.orchestra_base_url, "orchestra_base_url")
    subagent_url = _localhost_url(args.subagent_base_url, "subagent_base_url")
    semantic_judge_dataset = dataset in {"bcp", "hotpotqa"}
    judge_url = _localhost_url(args.judge_base_url, "judge_base_url") if semantic_judge_dataset else ""

    test_file = args.test_file.resolve()
    rows = _load_jsonl(test_file)
    test_file_sha256 = _sha256_file(test_file)
    indices = list(range(len(rows)))
    if args.indices:
        indices = sorted({int(x.strip()) for x in args.indices.split(",") if x.strip()})
    elif args.max_samples > 0:
        indices = indices[: args.max_samples]
    for idx in indices:
        if idx < 0 or idx >= len(rows):
            raise IndexError(f"index {idx} out of range [0,{len(rows)})")

    out = args.output_dir.resolve()
    designs_dir, samples_dir, traces_dir = out / "designs", out / "samples", out / "traces"
    design_attempts_dir = designs_dir / "attempts"
    judge_attempts_dir = out / "judge_attempts"
    for d in (designs_dir, design_attempts_dir, samples_dir, traces_dir, judge_attempts_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.max_retrieval_rounds <= 0:
        raise ValueError("max_retrieval_rounds must be positive")
    resume_invalidated_routing = (
        _invalidate_bcp_resume_routing_cases(
            indices=indices,
            designs_dir=designs_dir,
            samples_dir=samples_dir,
            traces_dir=traces_dir,
            max_retrieval_rounds=args.max_retrieval_rounds,
        )
        if args.resume and dataset == "bcp" else []
    )
    for row in resume_invalidated_routing:
        print(f"[resume routing repair] index={row['index']} {row['reason']}", flush=True)
    prior_manifest: dict[str, Any] = {}
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                prior_manifest = loaded_manifest
        except Exception:
            prior_manifest = {}
    prior_test_hash = str(prior_manifest.get("test_file_sha256", "")).strip()
    if prior_test_hash and prior_test_hash != test_file_sha256:
        raise RuntimeError(
            "test dataset changed between evaluation phases: "
            f"manifest sha256={prior_test_hash}, current sha256={test_file_sha256}"
        )
    manifest = {
        "schema_version": "mas_gate_orchestra_eval/1",
        "project": "mas-gate",
        "dataset": dataset,
        "test_file": str(test_file),
        "test_file_sha256": test_file_sha256,
        "orchestra_model": args.orchestra_model,
        "orchestra_model_source": args.orchestra_model_source,
        "subagent_model": args.subagent_model,
        "orchestra_base_url": orchestra_url,
        "subagent_base_url": subagent_url,
        "num_dataset_rows": len(rows),
        "selected_indices": indices,
        "hostname": socket.gethostname(),
        "started_at_unix": prior_manifest.get("started_at_unix", time.time()),
        "scoring": (
            "Qwen3-30B-A3B-Instruct-2507 LLM judge using BrowseComp-Plus/judge.py"
            if dataset == "bcp"
            else "Qwen3-30B semantic judge; HotpotQA official EM/F1 retained as diagnostics"
            if dataset == "hotpotqa"
            else "GPQAScorer strict final-option exact match"
            if dataset == "gpqa"
            else "AIMEScorer boxed/math equivalence"
        ),
        "judge_model": args.judge_model if semantic_judge_dataset else "",
        "judge_base_url": judge_url,
        "judge_prompt_version": grader_template_for(dataset)[1] if semantic_judge_dataset else "",
        "bcp_judge_model": args.judge_model if dataset == "bcp" else "",
        "bcp_judge_base_url": judge_url if dataset == "bcp" else "",
        "bcp_judge_prompt_version": "BrowseComp-Plus/judge.py::GRADER_TEMPLATE" if dataset == "bcp" else "",
        "bcp_gold_docs_used_as_context": False,
        "quantification_used": False,
        "orchestra_attempts": args.orchestra_attempts,
        "execution_attempts": args.execution_attempts,
        "judge_attempts": args.judge_attempts,
        "retry_policy": "retry infrastructure/MAS/Judge failures only; never retry a successfully executed wrong answer",
        "orchestra_thinking_enabled": False,
        "subagent_max_model_len": int(os.environ.get("MAS_GATE_SUB_AGENT_MAX_MODEL_LEN", "32768")),
        "subagent_requested_max_output_tokens": int(os.environ.get("MAS_GATE_SUB_AGENT_MAX_TOKENS", "8192")),
        "subagent_min_completion_tokens": int(os.environ.get("MAS_GATE_SUB_AGENT_MIN_COMPLETION_TOKENS", "8192")),
        "bcp_max_retrieval_rounds": args.max_retrieval_rounds if dataset == "bcp" else None,
        "hotpotqa_max_retrieval_rounds": args.max_retrieval_rounds if dataset == "hotpotqa" else None,
        "resume_invalidated_routing": resume_invalidated_routing,
        "service_lifecycle": args.service_lifecycle,
        "last_completed_phase": prior_manifest.get("last_completed_phase", ""),
    }
    _atomic_json(manifest_path, manifest)

    orchestra = AsyncOpenAIClient(args.orchestra_model, api_key="EMPTY", base_url=orchestra_url)
    subagent = AsyncOpenAIClient(args.subagent_model, api_key="EMPTY", base_url=subagent_url)
    judge = (AsyncOpenAIClient(
                 args.judge_model,
                 api_key="EMPTY",
                 base_url=judge_url,
                 explicit_http_client=True,
                 local_context_config={
                     "max_model_len": int(os.environ.get("MAS_GATE_HOTPOTQA_JUDGE_MAX_MODEL_LEN", os.environ.get("MAS_GATE_BCP_JUDGE_MAX_MODEL_LEN", "12288"))) if dataset == "hotpotqa" else int(os.environ.get("MAS_GATE_BCP_JUDGE_MAX_MODEL_LEN", "12288")),
                     "max_tokens": args.judge_max_tokens,
                     "min_completion_tokens": args.judge_max_tokens,
                     "context_safety_tokens": int(os.environ.get("MAS_GATE_HOTPOTQA_JUDGE_CONTEXT_SAFETY_TOKENS", os.environ.get("MAS_GATE_BCP_JUDGE_CONTEXT_SAFETY_TOKENS", "128"))) if dataset == "hotpotqa" else int(os.environ.get("MAS_GATE_BCP_JUDGE_CONTEXT_SAFETY_TOKENS", "128")),
                     "model_path": (os.environ.get("MAS_GATE_HOTPOTQA_JUDGE_MODEL_PATH", os.environ.get("MAS_GATE_BCP_JUDGE_MODEL_PATH", "")) if dataset == "hotpotqa" else os.environ.get("MAS_GATE_BCP_JUDGE_MODEL_PATH", "")) or None,
                 },
             )
             if semantic_judge_dataset else None)
    # Native OpenAI client is required by BrowseComp's retrieval controller.
    bcp_chat = AsyncOpenAI(api_key="EMPTY", base_url=subagent_url, timeout=args.request_timeout, max_retries=5)

    design_sem = asyncio.Semaphore(max(1, args.orchestra_concurrency))
    exec_sem = asyncio.Semaphore(max(1, args.execution_concurrency))
    judge_sem = asyncio.Semaphore(max(1, args.judge_concurrency))
    print_lock = asyncio.Lock()

    async def generate_design(idx: int) -> dict[str, Any]:
        path = designs_dir / f"{idx}.json"
        if args.resume and path.is_file():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                task_id, question, _gold = _sample_fields(dataset, idx, rows[idx])
                _validate_artifact_identity(
                    old, idx=idx, task_id=task_id, question=question, path=path
                )
                graph = graph_from_rollout_xml(
                    str(old["orchestra_xml"]),
                    question,
                    dataset,
                )
                graph_payload = _graph_json(graph)
                if dataset == "bcp":
                    audit_graph_routing(graph_payload, label=f"resume design index={idx}")
                old["graph"] = graph_payload
                return old
            except Exception:
                pass
        task_id, question, _gold = _sample_fields(dataset, idx, rows[idx])
        last: Exception | None = None
        attempt_events: list[dict[str, Any]] = []
        for attempt in range(1, args.orchestra_attempts + 1):
            t0 = time.perf_counter()
            xml = ""
            usage: dict[str, Any] = {}
            try:
                async with design_sem:
                    xml, usage = await orchestra.generate(
                        user_prompt=build_orchestra_user_prompt(question, dataset),
                        system_prompt=ORCHESTRA_XML_SYSTEM_PROMPT,
                        temperature=args.orchestra_temperature,
                        max_tokens=args.orchestra_max_tokens,
                        # Structured XML generation does not benefit from Qwen3's
                        # long reasoning preamble.  In the old AIME sweep that
                        # preamble frequently consumed the completion budget
                        # before <editor_output> was emitted, making otherwise
                        # valid samples look like MAS execution failures.
                        extra_body={
                            "enable_thinking": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                graph = graph_from_rollout_xml(xml, question, dataset)
                graph_payload = _graph_json(graph)
                if dataset == "bcp":
                    audit_graph_routing(graph_payload, label=f"design index={idx}")
                attempt_record = {
                    "index": idx, "task_id": task_id, "attempt": attempt,
                    "success": True, "orchestra_xml": xml, "usage": usage,
                    "latency_s": time.perf_counter() - t0,
                }
                _atomic_json(design_attempts_dir / str(idx) / f"attempt_{attempt:02d}.json", attempt_record)
                attempt_events.append({k: v for k, v in attempt_record.items() if k != "orchestra_xml"})
                payload = {
                    "index": idx, "task_id": task_id, "question": question,
                    "question_sha256": _question_sha256(question),
                    "orchestra_xml": xml, "graph": graph_payload, "usage": usage,
                    "attempt": attempt, "attempts_used": attempt,
                    "attempt_events": attempt_events,
                    "latency_s": time.perf_counter() - t0,
                }
                _atomic_json(path, payload)
                async with print_lock:
                    print(f"[design] {idx + 1}/{len(rows)} index={idx} agents={len(graph.nodes)} attempt={attempt}", flush=True)
                return payload
            except Exception as exc:
                last = exc
                attempt_record = {
                    "index": idx, "task_id": task_id, "attempt": attempt,
                    "success": False, "orchestra_xml": xml, "usage": usage,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "latency_s": time.perf_counter() - t0,
                }
                _atomic_json(design_attempts_dir / str(idx) / f"attempt_{attempt:02d}.json", attempt_record)
                attempt_events.append({k: v for k, v in attempt_record.items() if k not in {"orchestra_xml", "traceback"}})
                async with print_lock:
                    print(
                        f"[design retry] index={idx} attempt={attempt}/{args.orchestra_attempts} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                await asyncio.sleep(min(8, attempt * 2))
        payload = {
            "index": idx, "task_id": task_id, "question": question,
            "question_sha256": _question_sha256(question), "orchestra_xml": "",
            "error": f"{type(last).__name__}: {last}",
            "attempts_used": args.orchestra_attempts,
            "attempt_events": attempt_events,
        }
        _atomic_json(path, payload)
        return payload

    async def execute_one(idx: int, design: dict[str, Any]) -> dict[str, Any]:
        path = samples_dir / f"{idx}.json"
        if args.resume and path.is_file():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                # A successfully executed but wrong answer is a real model
                # outcome and must not be resampled.  Infrastructure/MAS
                # failures are retried so they are not silently treated as 0.
                if "score" in old and bool(old.get("success")):
                    return old
            except Exception:
                pass
            path.unlink(missing_ok=True)
            (traces_dir / f"{idx}.json").unlink(missing_ok=True)
        task_id, question, gold = _sample_fields(dataset, idx, rows[idx])
        t0 = time.perf_counter()
        retrieval_sessions: list[dict[str, Any]] = []
        state: dict[str, Any] = {}
        final_code = ""
        events: list[dict[str, Any]] = []
        error = ""
        success = False
        attempts = 0
        try:
            _validate_artifact_identity(
                design, idx=idx, task_id=task_id, question=question,
                path=designs_dir / f"{idx}.json",
            )
            xml = str(design.get("orchestra_xml", ""))
            if not xml:
                raise RuntimeError(str(design.get("error", "missing Orchestra XML")))
            graph = graph_from_rollout_xml(xml, question, dataset)
            class_name = f"MasGateEval{dataset.upper()}{idx}_Workflow"
            mas_code = compile_graph_to_code(graph, class_name, dataset_name=dataset)
            tool_call_fn = None
            if dataset in {"bcp", "hotpotqa"}:
                tool_call_fn = _make_retrieval_tool_call_fn(
                    dataset_name=dataset, client=bcp_chat, model=args.subagent_model,
                    question=question, sessions_out=retrieval_sessions,
                )
            async with exec_sem:
                success, state, final_code, attempts, events, exec_err = await asyncio.wait_for(
                    run_generated_workflow_with_retries(
                        mas_code=mas_code,
                        class_name=class_name,
                        task_text=question,
                        text_call_fn=make_text_call_fn(subagent),
                        planner_call_fn=make_planner_call_fn(subagent),
                        tool_call_fn=tool_call_fn,
                        max_attempts=args.execution_attempts,
                        dataset_name=dataset,
                    ),
                    timeout=args.sample_timeout,
                )
            if exec_err:
                error = exec_err
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            events.append({"type": "evaluation_exception", "message": error, "traceback": traceback.format_exc()})
        final_output = str((state or {}).get("final_output", "") or "").strip()
        score, extracted = (
            GPQAScorer().calculate_score(gold, final_output)
            if dataset == "gpqa" else _judge_score(dataset, gold, final_output)
        )
        # Infrastructure/MAS execution failures are always incorrect.  Never
        # allow a parser or an empty/default output to turn them into credit.
        if not success:
            score, extracted = 0, ""
        hotpotqa_metrics: dict[str, Any] | None = None
        if dataset == "hotpotqa":
            from hotpotqa.hotpotqa_score import answer_metrics
            hotpotqa_metrics = answer_metrics(final_output, gold)
        result = {
            "schema_version": "mas_gate_orchestra_eval_sample/1",
            "index": idx, "task_id": task_id, "dataset": dataset,
            "question": question, "question_sha256": _question_sha256(question), "gold": gold, "prediction": final_output,
            "extracted": extracted,
            "score": (None if semantic_judge_dataset else int(score)),
            "correct": (None if semantic_judge_dataset else bool(score)),
            "scoring_method": (
                "bcp_exact_diagnostic_pending_llm_judge" if dataset == "bcp"
                else "hotpotqa_official_diagnostic_pending_llm_judge" if dataset == "hotpotqa"
                else "gpqa_strict_option_scorer" if dataset == "gpqa"
                else "aime_rule_scorer"
            ),
            "exact_score": int(score) if dataset in {"bcp", "hotpotqa"} else None,
            "exact_extracted": extracted if dataset in {"bcp", "hotpotqa"} else "",
            "hotpotqa_metrics": hotpotqa_metrics,
            "answer_f1": float((hotpotqa_metrics or {}).get("f1") or 0.0) if dataset == "hotpotqa" else None,
            "success": bool(success), "error": error, "events": events,
            "design_attempts_used": int(design.get("attempts_used", design.get("attempt", 0)) or 0),
            "execution_attempts_used": int(attempts),
            "usage": _collect_usage_totals(state),
            "sub_agents": _extract_sub_agent_logs(state),
            "retrieval_sessions": retrieval_sessions,
            "orchestra_xml": design.get("orchestra_xml", ""),
            "graph": design.get("graph", {}),
            "latency_s": time.perf_counter() - t0,
        }
        _atomic_json(path, result)
        trace_payload = {**result, "workflow_state": state, "generated_mas_code": final_code}
        _atomic_json(traces_dir / f"{idx}.json", trace_payload)
        async with print_lock:
            done = len(list(samples_dir.glob("*.json")))
            running_scores = []
            for p in samples_dir.glob("*.json"):
                try:
                    running_scores.append(int(json.loads(p.read_text(encoding="utf-8")).get("score", 0)))
                except Exception:
                    pass
            acc = sum(running_scores) / max(1, len(running_scores))
            print(f"[execute] done={done}/{len(indices)} index={idx} score={int(score)} running_acc={acc:.4f} latency={result['latency_s']:.1f}s", flush=True)
        return result

    async def judge_one(result: dict[str, Any]) -> dict[str, Any]:
        if not semantic_judge_dataset:
            return result
        if not bool(result.get("success")):
            result.update({
                "score": 0,
                "correct": False,
                "scoring_method": f"{dataset}_execution_failed_score_zero",
            })
            index = int(result["index"])
            path = samples_dir / f"{index}.json"
            _atomic_json(path, result)
            trace_path = traces_dir / f"{index}.json"
            if trace_path.is_file():
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                trace.update(result)
                _atomic_json(trace_path, trace)
            return result
        index = int(result["index"])
        path = samples_dir / f"{index}.json"
        if (args.resume and result.get("scoring_method") in {"bcp_llm_judge", "hotpotqa_llm_judge"}
                and isinstance(result.get("llm_judge"), dict)):
            # A previous judge attempt may have succeeded after recording a
            # transient parse error.  Do not let that stale marker poison
            # campaign-level validation on resume.
            changed = False
            for stale_key in ("judge_error",):
                if stale_key in result:
                    result.pop(stale_key, None)
                    changed = True
            if changed:
                _atomic_json(path, result)
                trace_path = traces_dir / f"{index}.json"
                if trace_path.is_file():
                    trace = json.loads(trace_path.read_text(encoding="utf-8"))
                    trace.pop("judge_error", None)
                    _atomic_json(trace_path, trace)
            return result
        assert judge is not None
        last: Exception | None = None
        for attempt in range(1, args.judge_attempts + 1):
            t0 = time.perf_counter()
            try:
                async with judge_sem:
                    judgement = await judge_answer(
                        judge_client=judge, question=str(result.get("question", "")),
                        response=str(result.get("prediction", "")), correct_answer=str(result.get("gold", "")),
                        temperature=0.0, max_tokens=args.judge_max_tokens, dataset=dataset,
                    )
                raw = str(judgement.get("raw_judgement", "")).strip()
                if not raw or str(judgement.get("correct", "")).lower() not in {"yes", "no"}:
                    raise RuntimeError(f"malformed semantic judge output for index={index}: {raw[:500]!r}")
                judgement.update({"judge_model": args.judge_model})
                _atomic_json(judge_attempts_dir / str(index) / f"attempt_{attempt:02d}.json", {
                    "index": index, "attempt": attempt, "success": True,
                    "latency_s": time.perf_counter() - t0, "llm_judge": judgement,
                })
                # A resumable repair may be judging a sample whose previous
                # attempt ended with a transient judge_error.  Once this
                # attempt succeeds, remove that stale failure marker so the
                # campaign validator sees the repaired candidate as clean.
                result.pop("judge_error", None)
                result.update({
                    "score": int(judgement["score"]), "correct": bool(judgement["score"]),
                    "extracted": str(judgement.get("extracted_final_answer", "None")),
                    "scoring_method": f"{dataset}_llm_judge", "llm_judge": judgement,
                    "semantic_judge": judgement,
                    "judge_attempts_used": attempt,
                })
                _atomic_json(path, result)
                trace_path = traces_dir / f"{index}.json"
                if trace_path.is_file():
                    trace = json.loads(trace_path.read_text(encoding="utf-8"))
                    trace.update(result)
                    _atomic_json(trace_path, trace)
                async with print_lock:
                    print(f"[judge] index={index} llm={result['score']} exact={result.get('exact_score')} attempt={attempt}", flush=True)
                return result
            except Exception as exc:
                last = exc
                _atomic_json(judge_attempts_dir / str(index) / f"attempt_{attempt:02d}.json", {
                    "index": index, "attempt": attempt, "success": False,
                    "latency_s": time.perf_counter() - t0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                async with print_lock:
                    print(
                        f"[judge retry] index={index} attempt={attempt}/{args.judge_attempts} "
                        f"error={type(exc).__name__}: {exc}", flush=True,
                    )
                if attempt < args.judge_attempts:
                    await asyncio.sleep(min(8, attempt * 2))
        judge_error = (
            f"semantic judge failed after {args.judge_attempts} attempts for index={index}: "
            f"{type(last).__name__}: {last}"
        )
        # 以下为安全注释COSEC：Judge 基础设施失败必须保持未评分，禁止回退到 exact/EM。
        result.update({
            "score": None,
            "correct": None,
            "scoring_method": f"{dataset}_llm_judge_failed",
            "judge_error": judge_error,
            "judge_attempts_used": args.judge_attempts,
        })
        _atomic_json(path, result)
        trace_path = traces_dir / f"{index}.json"
        if trace_path.is_file():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace.update(result)
            _atomic_json(trace_path, trace)
        return result

    results: list[dict[str, Any]] = []

    def load_designs_for_phase() -> list[dict[str, Any]]:
        loaded: list[dict[str, Any]] = []
        for idx in indices:
            path = designs_dir / f"{idx}.json"
            if not path.is_file():
                raise RuntimeError(f"missing design artifact for phase={args.phase}: {path}")
            design = json.loads(path.read_text(encoding="utf-8"))
            task_id, question, _gold = _sample_fields(dataset, idx, rows[idx])
            _validate_artifact_identity(
                design, idx=idx, task_id=task_id, question=question, path=path
            )
            xml = str(design.get("orchestra_xml", ""))
            graph = graph_from_rollout_xml(xml, question, dataset)
            graph_payload = _graph_json(graph)
            if dataset == "bcp":
                audit_graph_routing(graph_payload, label=f"phase input design index={idx}")
            design["graph"] = graph_payload
            loaded.append(design)
        return loaded

    def load_samples_for_phase() -> list[dict[str, Any]]:
        loaded: list[dict[str, Any]] = []
        for idx in indices:
            path = samples_dir / f"{idx}.json"
            if not path.is_file():
                raise RuntimeError(f"missing execution artifact for phase={args.phase}: {path}")
            sample = json.loads(path.read_text(encoding="utf-8"))
            task_id, question, _gold = _sample_fields(dataset, idx, rows[idx])
            _validate_artifact_identity(
                sample, idx=idx, task_id=task_id, question=question, path=path
            )
            loaded.append(sample)
        return loaded

    try:
        if args.phase == "all":
            # Legacy mode.  It is retained for environments where vLLM
            # sleep/wake is known to preserve model semantics.  PPU TP8 Qwen3
            # evaluation uses phase-local fresh processes instead.
            await _set_sleep(subagent_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
            if judge_url:
                await _set_sleep(judge_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
            await _set_sleep(orchestra_url, False, timeout=args.vllm_control_timeout)
            await _semantic_health(orchestra_url, args.orchestra_model, "orchestra", "after_wake_design")
            print(f"[phase] generating Orchestra XML for {len(indices)} samples", flush=True)
            designs = await asyncio.gather(*[generate_design(i) for i in indices])

            await _set_sleep(orchestra_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
            await _set_sleep(subagent_url, False, timeout=args.vllm_control_timeout)
            await _semantic_health(subagent_url, args.subagent_model, "sub_agent", "after_wake_execute")
            print(f"[phase] executing MAS with local sub-agents for {len(indices)} samples", flush=True)
            results = await asyncio.gather(*[execute_one(i, d) for i, d in zip(indices, designs)])
            if semantic_judge_dataset:
                await _set_sleep(subagent_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
                await _set_sleep(judge_url, False, timeout=args.vllm_control_timeout)
                await _semantic_health(judge_url, args.judge_model, "judge", "after_wake_judge")
                print(f"[phase] judging {len(indices)} {dataset} answers with {args.judge_model}", flush=True)
                results = await asyncio.gather(*[judge_one(r) for r in results])
                await _set_sleep(judge_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
        elif args.phase == "design":
            print(f"[phase:fresh] generating Orchestra XML for {len(indices)} samples", flush=True)
            designs = await asyncio.gather(*[generate_design(i) for i in indices])
            failed = [int(d.get("index", -1)) for d in designs if not str(d.get("orchestra_xml", ""))]
            if failed:
                raise RuntimeError(f"design phase produced {len(failed)} failures; first indices={failed[:20]}")
        elif args.phase == "execute":
            designs = load_designs_for_phase()
            print(f"[phase:fresh] executing MAS with local sub-agents for {len(indices)} samples", flush=True)
            results = await asyncio.gather(*[execute_one(i, d) for i, d in zip(indices, designs)])
        elif args.phase == "judge":
            if not semantic_judge_dataset:
                raise ValueError("judge phase is only valid for BCP or HotpotQA")
            results = load_samples_for_phase()
            print(f"[phase:fresh] judging {len(indices)} {dataset} answers with {args.judge_model}", flush=True)
            results = await asyncio.gather(*[judge_one(r) for r in results])
        else:  # argparse enforces this; keep a fail-closed guard for programmatic callers.
            raise ValueError(f"unsupported phase: {args.phase}")
    finally:
        if args.phase == "all":
            try:
                await _set_sleep(orchestra_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
            except Exception:
                pass
            try:
                await _set_sleep(subagent_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
            except Exception:
                pass
            if judge_url:
                try:
                    await _set_sleep(judge_url, True, level=args.sleep_level, timeout=args.vllm_control_timeout)
                except Exception:
                    pass
        for client in (orchestra, subagent, bcp_chat, judge):
            if client is None:
                continue
            try:
                await client.aclose()
            except Exception:
                pass

    phase_complete = {
        "schema_version": "mas_gate_orchestra_eval_phase/1",
        "phase": args.phase,
        "service_lifecycle": args.service_lifecycle,
        "num_selected": len(indices),
        "completed_at_unix": time.time(),
    }
    _atomic_json(out / f"phase_{args.phase}_complete.json", phase_complete)
    manifest["last_completed_phase"] = args.phase
    _atomic_json(manifest_path, manifest)

    # Phase-local design does not yet have answer-level results.  BCP execute
    # also intentionally defers the authoritative score to the judge phase.
    if args.phase == "design" or (args.phase == "execute" and semantic_judge_dataset):
        print("[phase complete] " + json.dumps(phase_complete, ensure_ascii=False), flush=True)
        return phase_complete

    results.sort(key=lambda x: int(x["index"]))
    total = len(results)
    judged_results = [x for x in results if x.get("score") in (0, 1)]
    correct = sum(int(x.get("score", 0)) for x in judged_results)
    successful = sum(bool(x.get("success")) for x in results)
    summary = {
        **manifest,
        "finished_at_unix": time.time(),
        "num_evaluated": total,
        "num_correct": correct,
        "accuracy": correct / max(1, len(judged_results)),
        "accuracy_over_all_selected": correct / max(1, total),
        "num_judged": len(judged_results),
        "answer_f1": (
            sum(float(x.get("answer_f1") or 0.0) for x in results) / max(1, total)
            if dataset == "hotpotqa" else None
        ),
        "num_exact_match_correct": (sum(int(x.get("exact_score", 0)) for x in results) if dataset in {"bcp", "hotpotqa"} else None),
        "judge_coverage": (sum(isinstance(x.get("semantic_judge") or x.get("llm_judge"), dict) for x in results) / max(1, total) if semantic_judge_dataset else None),
        "num_judge_yes_exact_no": (sum(x.get("score") == 1 and x.get("exact_score") == 0 for x in results) if semantic_judge_dataset else None),
        "num_judge_no_exact_yes": (sum(x.get("score") == 0 and x.get("exact_score") == 1 for x in results) if semantic_judge_dataset else None),
        "num_mas_execution_success": successful,
        "mas_execution_success_rate": successful / max(1, total),
        "num_errors": sum(bool(x.get("error") or x.get("judge_error")) for x in results),
        "num_design_retried": sum(int(x.get("design_attempts_used", 0)) > 1 for x in results),
        "num_execution_retried": sum(int(x.get("execution_attempts_used", 0)) > 1 for x in results),
        "num_judge_retried": sum(int(x.get("judge_attempts_used", 0)) > 1 for x in results),
        "num_design_failures": sum(int(x.get("design_attempts_used", 0)) >= args.orchestra_attempts and not x.get("orchestra_xml") for x in results),
        "num_execution_failures": sum(bool(x.get("orchestra_xml")) and not bool(x.get("success")) for x in results),
        "results_dir": str(out),
    }
    _atomic_json(out / "summary.json", summary)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            compact = {k: v for k, v in r.items() if k not in {"sub_agents", "retrieval_sessions", "orchestra_xml"}}
            f.write(json.dumps(compact, ensure_ascii=False, default=str) + "\n")
    summary_keys = ["dataset", "num_evaluated", "num_correct", "accuracy"]
    if dataset == "hotpotqa":
        summary_keys.append("answer_f1")
    summary_keys.extend(["mas_execution_success_rate", "results_dir"])
    print("[summary] " + json.dumps({k: summary[k] for k in summary_keys}, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate trained MAS-Gate Orchestra on BCP, HotpotQA, AIME, or GPQA")
    ap.add_argument("--dataset", choices=["bcp", "hotpotqa", "aime", "gpqa"], required=True)
    ap.add_argument("--test-file", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--orchestra-model", default="qwen3-8b")
    ap.add_argument("--orchestra-model-source", default="")
    ap.add_argument("--orchestra-base-url", default="http://127.0.0.1:8201/v1")
    ap.add_argument("--subagent-model", default="Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--subagent-base-url", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--judge-model", default="Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--judge-base-url", default="http://127.0.0.1:8203/v1")
    ap.add_argument("--judge-concurrency", type=int, default=1024)
    ap.add_argument("--judge-max-tokens", type=int, default=1024)
    ap.add_argument("--judge-attempts", type=int, default=3)
    ap.add_argument("--orchestra-concurrency", type=int, default=1024)
    ap.add_argument("--execution-concurrency", type=int, default=1024)
    ap.add_argument("--orchestra-max-tokens", type=int, default=4096)
    ap.add_argument("--orchestra-temperature", type=float, default=0.6)
    ap.add_argument("--orchestra-attempts", type=int, default=3)
    ap.add_argument("--execution-attempts", type=int, default=2)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--sample-timeout", type=float, default=1800.0)
    ap.add_argument("--vllm-control-timeout", type=float, default=600.0)
    ap.add_argument("--sleep-level", type=int, choices=[1, 2], default=1)
    ap.add_argument("--phase", choices=["all", "design", "execute", "judge"], default="all")
    ap.add_argument(
        "--service-lifecycle",
        choices=["sleep_wake", "fresh_phase_restart"],
        default="sleep_wake",
    )
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--indices", default="")
    ap.add_argument(
        "--max-retrieval-rounds",
        type=int,
        default=int(os.environ.get("MAS_GATE_BCP_RETRIEVAL_MAX_ROUNDS", "5")),
    )
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
