#!/usr/bin/env python3
"""Compare teacher-forced J scoring: transformers vs in-process vLLM."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MAS_ROOT = Path(__file__).resolve().parents[2]
if str(MAS_ROOT) not in sys.path:
    sys.path.insert(0, str(MAS_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify vLLM prompt_logprobs match transformers get_j_tilde"
    )
    parser.add_argument(
        "--model-path",
        default="./models/Qwen3.6-35B-A3B",
    )
    parser.add_argument(
        "--context",
        default="Question: What is 2 + 2?\nAnswer:",
    )
    parser.add_argument(
        "--answer",
        default="4",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.25,
        help="Absolute NLL tolerance (MoE/hybrid models need looser defaults than dense LMs)",
    )
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument(
        "--phase",
        choices=("all", "transformers", "vllm"),
        default="all",
        help="Run one backend in isolation (used internally for GPU memory safety)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write per-token NLL vector (.npy) for --phase transformers|vllm",
    )
    return parser


def _run_transformers(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quantification.clients.j_tilde import (  # noqa: E402
        score_token_nlls_transformers,
        tokenize_context_and_answer,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this verification script")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ctx_ids, ans_ids = tokenize_context_and_answer(tokenizer, args.context, args.answer)
    print(f"[transformers] context_tokens={len(ctx_ids)} answer_tokens={len(ans_ids)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    ref_nlls = score_token_nlls_transformers(model, ctx_ids, ans_ids).float().cpu().numpy()
    if args.out is None:
        raise SystemExit("--out is required for --phase transformers")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, ref_nlls)
    print(f"[transformers] wrote NLLs to {args.out}")


def _run_vllm(args: argparse.Namespace) -> None:
    import numpy as np
    from transformers import AutoTokenizer

    from quantification.clients.j_tilde import (  # noqa: E402
        VllmJScorer,
        tokenize_context_and_answer,
    )
    from quantification.config import QuantificationConfig  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    ctx_ids, ans_ids = tokenize_context_and_answer(tokenizer, args.context, args.answer)
    max_model_len = max(512, len(ctx_ids) + len(ans_ids) + 64)

    config = QuantificationConfig.from_env(
        backend="vllm",
        scoring_model_path=args.model_path,
        vllm_tensor_parallel_size=2,
        vllm_max_model_len=max_model_len,
        vllm_enforce_eager=True,
        vllm_gpu_memory_utilization=0.85,
        vllm_kv_cache_memory_bytes=2 * 1024**3,
        scoring_concurrency=1,
    )
    print(f"[vllm] context_tokens={len(ctx_ids)} answer_tokens={len(ans_ids)}")
    print(
        f"[vllm] max_model_len={config.vllm_max_model_len} "
        f"tensor_parallel_size={config.vllm_tensor_parallel_size} "
        f"enforce_eager={config.vllm_enforce_eager}"
    )

    scorer = VllmJScorer(config)
    vllm_nlls = scorer.score_prompt_ids(ctx_ids, ans_ids).cpu().numpy()
    if args.out is None:
        raise SystemExit("--out is required for --phase vllm")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vllm_nlls)
    print(f"[vllm] wrote NLLs to {args.out}")


def _run_compare(args: argparse.Namespace, ref_path: Path, vllm_path: Path) -> None:
    import numpy as np
    import torch

    from quantification.clients.j_tilde import (  # noqa: E402
        compare_transformers_and_vllm_nlls,
        j_metric_from_token_nlls,
        j_tilde_from_token_nlls,
    )

    ref_nlls = torch.from_numpy(np.load(ref_path))
    vllm_nlls = torch.from_numpy(np.load(vllm_path))

    compare_transformers_and_vllm_nlls(
        transformers_nlls=ref_nlls,
        vllm_nlls=vllm_nlls,
        atol=args.atol,
        rtol=args.rtol,
    )

    print("per-token NLL (transformers):", ref_nlls.tolist())
    print("per-token NLL (vLLM):        ", vllm_nlls.tolist())
    print("j_tilde (transformers):", j_tilde_from_token_nlls(ref_nlls))
    print("j_tilde (vLLM):        ", j_tilde_from_token_nlls(vllm_nlls))
    print("j_metric (transformers):", j_metric_from_token_nlls(ref_nlls))
    print("j_metric (vLLM):        ", j_metric_from_token_nlls(vllm_nlls))
    print("OK: transformers and vLLM teacher-forced NLLs are aligned")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    # Avoid vLLM 0.22.x multiproc + prior CUDA init issues in short-lived verify workers.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return env


def _subprocess_cmd(args: argparse.Namespace, phase: str, out_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--model-path",
        args.model_path,
        "--context",
        args.context,
        "--answer",
        args.answer,
        "--out",
        str(out_path),
    ]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.phase == "transformers":
        _run_transformers(args)
        return
    if args.phase == "vllm":
        _run_vllm(args)
        return

    with tempfile.TemporaryDirectory(prefix="j_tilde_verify_") as tmp_dir:
        tmp = Path(tmp_dir)
        ref_path = tmp / "transformers.npy"
        vllm_path = tmp / "vllm.npy"

        print("[verify] phase 1/2: transformers (subprocess)")
        subprocess.run(
            _subprocess_cmd(args, "transformers", ref_path),
            check=True,
            env=_subprocess_env(),
        )

        print("[verify] phase 2/2: vLLM (subprocess)")
        subprocess.run(
            _subprocess_cmd(args, "vllm", vllm_path),
            check=True,
            env=_subprocess_env(),
        )

        _run_compare(args, ref_path, vllm_path)


if __name__ == "__main__":
    main()
