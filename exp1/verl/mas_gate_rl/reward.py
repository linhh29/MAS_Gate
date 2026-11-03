"""MAS-Gate reward entrypoint.

The implementation is loaded from the preserved Python 3.12 bytecode artifact.
The explicit package aliases follow this project's ``datasets`` PYTHONPATH
layout and avoid collision with HuggingFace's top-level ``datasets`` package.
"""
from __future__ import annotations

import importlib.machinery
import hashlib
import json
import re
import string
import importlib.util
import math
import os
import asyncio
import sys
from pathlib import Path

# Alfworld is optional for HotpotQA/BrowseComp reward workers.  The preserved
# implementation only needs these aliases on Alfworld runs; importing them
# eagerly would make every dataset fail when the optional package is absent.
try:
    import alfworld
    import interactive_env
except ModuleNotFoundError:
    alfworld = None
    interactive_env = None
else:
    # Project-local compatibility aliases required by the preserved implementation.
    sys.modules.setdefault("datasets.alfworld", alfworld)
    sys.modules.setdefault("datasets.interactive_env", interactive_env)

_IMPL_PATH = Path(__file__).with_name("_reward_impl.cpython-312.pyc")
_LOADER = importlib.machinery.SourcelessFileLoader("mas_gate_rl._reward_impl", str(_IMPL_PATH))
_SPEC = importlib.util.spec_from_loader(_LOADER.name, _LOADER)
if _SPEC is None:
    raise ImportError(f"cannot create module spec for {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(_IMPL)

for _name, _value in vars(_IMPL).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

del _name, _value


# Preserve the bytecode implementation while bridging ALFWorld's authoritative
# environment score into the existing sub_agent -> qwen_j -> bge_impact sidecar
# protocol.  Other datasets remain byte-for-byte on the original path.
_IMPL_COMPUTE_SCORE_ASYNC = _IMPL._compute_score_async


def _apply_opt_in_ablation(result, info, solution_str):
    tag = os.environ.get("MAS_GATE_ABLATION_TAG", "")
    if tag not in ("MAS_GATE_ABLATION_ADDITIVE_I_U_PLUS_1_PLUS_K", "MAS_GATE_ABLATION_SKYWORK_REWARD_U"):
        return result
    from mas_gate_rl.ablation_reward import apply_ablation
    question = str(info.get("question", info.get("prompt", "")))
    return apply_ablation(result, question=question, answer=solution_str, tag=tag)


async def _compute_score_async(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None,
    **kwargs,
):
    info = dict(extra_info or {})
    dataset_name = str(info.get("dataset", info.get("data_source", data_source)) or "").strip().lower()
    # Opt-in HotpotQA ablation: run normal MAS execution first, then replace
    # fused/quantification reward with binary final-answer accuracy.  Calling
    # _judge_score on Orchestra XML was wrong: XML is policy, not answer.
    acc_only = (
        os.environ.get("MAS_GATE_ABLATION_TAG") == "MAS_GATE_ABLATION_HOTPOTQA_ACC_01"
        and _is_hotpotqa_dataset(dataset_name)
    )
    if acc_only:
        # CPU-only binary accuracy.  Never start or call semantic/quantification
        # models: PPO actor and critic already occupy GPUs 0-3.
        def _norm(text: object) -> str:
            value = str(text or "").lower()
            value = re.sub(r"<[^>]+>", " ", value)
            value = value.translate(str.maketrans("", "", string.punctuation))
            value = re.sub(r"\b(a|an|the)\b", " ", value)
            return " ".join(value.split())

        answer = _norm(ground_truth)
        response = _norm(solution_str)
        correct = bool(answer) and (response == answer or answer in response)
        score = 1.0 if correct else 0.0
        return {
            "score": score,
            "judge_score": score,
            "exact_score": score,
            "metric_key": "hotpotqa_accuracy_01",
            "ablation_tag": "MAS_GATE_ABLATION_HOTPOTQA_ACC_01",
            "acc_only_phase": _reward_phase(),
            "judge_model": "cpu_normalized_exact_match",
            "error": "",
        }
    if not _is_alfworld_dataset(dataset_name) or _reward_phase() != "sub_agent":
        result = await _IMPL_COMPUTE_SCORE_ASYNC(
            data_source, solution_str, ground_truth, info, **kwargs
        )
        return await asyncio.to_thread(_apply_opt_in_ablation, result, info, solution_str)

    result = await _IMPL_COMPUTE_SCORE_ASYNC(
        data_source, solution_str, ground_truth, info, **kwargs
    )
    result = await asyncio.to_thread(_apply_opt_in_ablation, result, info, solution_str)
    if str(result.get("metric_key", "")) != "alfworld_environment":
        return result

    idx = int(info.get("index", info.get("idx", 0)) or 0)
    task_id = str(info.get("task_id", idx))
    trace_root = Path(
        os.environ.get(
            "MAS_GATE_REWARD_TRACE_ROOT",
            str(MAS_GATE_ROOT / "verl" / "outputs" / "mas_gate_reward_traces"),
        )
    )
    trace_path = Path(str(result.get("trace_json", "")))
    if not trace_path.is_file():
        raise RuntimeError(f"ALFWorld execution trace missing: {trace_path}")

    environment_result = dict(result.get("environment_result") or {})
    environment_score = float(environment_result.get("goal_condition_success_rate", result.get("judge_score", 0.0)))
    if not math.isfinite(environment_score) or not 0.0 <= environment_score <= 1.0:
        raise RuntimeError(f"invalid ALFWorld environment score: {environment_score!r}")
    execution_error = str(result.get("execution_error", "") or "").strip()
    execution_ok = not execution_error
    won = bool(environment_result.get("won", environment_score >= 1.0))
    execution_sidecar = _write_execution_sidecar(
        trace_root=trace_root,
        idx=idx,
        task_id=task_id,
        solution_str=solution_str,
        trace_path=trace_path,
        judge_score=environment_score,
        success=execution_ok,
        failure_reason=execution_error or None,
        judge_score_available=True,
        exact_score=environment_score,
        exact_extracted="1" if won else "0",
    )
    return {
        "score": 0.0,
        "judge_score": environment_score,
        "metric_key": "sub_agent_phase",
        "trace_json": str(trace_path),
        "environment_result": environment_result,
        "execution_sidecar": str(execution_sidecar),
        "error": "" if execution_ok else execution_error[:2000],
    }


# Functions loaded from the pyc resolve globals in the implementation module,
# so update that module as well as this public wrapper module.
_IMPL._compute_score_async = _compute_score_async
