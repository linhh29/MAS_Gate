"""Teacher-forced J scoring aligned with Unified_MAS get_j_tilde."""

from __future__ import annotations

import threading
import time
from typing import Protocol

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantification.config import QuantificationConfig
from quantification.vllm_devices import cuda_visible_devices

_j_tilde_lock = threading.Lock()
_llm_engine: "VllmJScorer | None" = None
_engine_lock = threading.Lock()


def tokenize_context_and_answer(
    tokenizer: AutoTokenizer,
    context: str,
    answer: str,
) -> tuple[list[int], list[int]]:
    """Match Unified_MAS: separate tokenize then concat (not tokenize(context+answer))."""
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids[0].tolist()
    ans_ids = tokenizer(answer, return_tensors="pt").input_ids[0].tolist()
    return ctx_ids, ans_ids


def j_tilde_from_token_nlls(token_nlls: torch.Tensor) -> float:
    """Same as Unified_MAS: j_tilde = -mean(NLL). Higher (less negative) = more confident."""
    return float(-token_nlls.mean().item())


def j_metric_from_token_nlls(token_nlls: torch.Tensor) -> float:
    """
    Metric stored in MAS_Gate pipeline: mean(NLL) = -j_tilde.
    Utility expects J to decrease when a node helps predict the answer.
    """
    return float(token_nlls.mean().item())


def answer_token_positions(context_len: int, num_answer_tokens: int) -> range:
    """
    Positions in prompt_logprobs / prompt_token_ids for each answer token.

    vLLM and transformers both score token t_i with logits conditioned on
    prompt_token_ids[:i]. The first prompt token has no logprob (no context).
    """
    if context_len <= 0:
        raise ValueError("context_len must be positive for teacher-forced J scoring")
    if num_answer_tokens <= 0:
        raise ValueError("num_answer_tokens must be positive")
    return range(context_len, context_len + num_answer_tokens)


def score_token_nlls_transformers(
    model: AutoModelForCausalLM,
    context_ids: list[int],
    answer_ids: list[int],
) -> torch.Tensor:
    """
    Exact Unified_MAS get_j_tilde forward:
    one teacher-forced pass, mean NLL over answer tokens.
    """
    if not answer_ids:
        raise ValueError("answer token sequence is empty")

    device = next(model.parameters()).device
    context_inputs = torch.tensor([context_ids], dtype=torch.long, device=device)
    answer_inputs = torch.tensor([answer_ids], dtype=torch.long, device=device)
    input_ids = torch.cat([context_inputs, answer_inputs], dim=1)

    with torch.inference_mode():
        logits = model(input_ids).logits

    context_len = context_inputs.shape[1]
    # logits[:, t, :] predicts token at index t + 1 in input_ids.
    target_logits = logits[:, context_len - 1 : context_len - 1 + len(answer_ids), :]
    target_logits = target_logits.reshape(-1, target_logits.size(-1))
    target_labels = answer_inputs.reshape(-1)
    token_nlls = F.cross_entropy(
        target_logits,
        target_labels,
        reduction="none",
    )
    return token_nlls


def extract_answer_token_logprob(
    prompt_logprobs: list,
    token_position: int,
    token_id: int,
) -> float:
    """
    Read logprob for one prompt token from vLLM RequestOutput.prompt_logprobs.

    prompt_logprobs[0] is None (no context for the first token). For i >= 1,
    prompt_logprobs[i] holds logprobs for prompt token i.
    """
    if token_position <= 0:
        raise IndexError(
            f"prompt logprob position must be >= 1, got {token_position}"
        )
    if token_position >= len(prompt_logprobs):
        raise IndexError(
            f"prompt_logprobs length {len(prompt_logprobs)} "
            f"< required position {token_position}"
        )

    pos_logprobs = prompt_logprobs[token_position]
    if pos_logprobs is None:
        raise RuntimeError(f"prompt_logprobs[{token_position}] is None")

    if token_id not in pos_logprobs:
        available = sorted(pos_logprobs.keys())
        raise RuntimeError(
            f"missing logprob for token_id={token_id} at position {token_position}; "
            f"available token_ids={available[:8]}"
        )

    entry = pos_logprobs[token_id]
    return float(entry.logprob if hasattr(entry, "logprob") else entry)


def extract_answer_token_nlls_from_prompt_logprobs(
    prompt_logprobs: list,
    context_len: int,
    answer_ids: list[int],
) -> torch.Tensor:
    """Convert vLLM prompt_logprobs into per-answer-token NLLs."""
    nll_values: list[float] = []
    for offset, token_id in enumerate(answer_ids):
        position = context_len + offset
        logprob = extract_answer_token_logprob(prompt_logprobs, position, token_id)
        nll_values.append(-logprob)
    return torch.tensor(nll_values, dtype=torch.float32)


class JScoringBackend(Protocol):
    def score_prompt_ids(
        self,
        context_ids: list[int],
        answer_ids: list[int],
    ) -> torch.Tensor: ...


class VllmJScorer:
    """
    In-process vLLM engine (scheme A): teacher-forced prompt logprobs,
    aligned with score_token_nlls_transformers / Unified_MAS get_j_tilde.
    """

    def __init__(self, config: QuantificationConfig) -> None:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.scoring_model_path,
            trust_remote_code=True,
        )
        tp = max(1, int(config.vllm_tensor_parallel_size))

        def _build_llm() -> object:
            from vllm import LLM

            return LLM(
                model=config.scoring_model_path,
                trust_remote_code=True,
                tensor_parallel_size=tp,
                gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                enforce_eager=config.vllm_enforce_eager,
                disable_custom_all_reduce=config.vllm_disable_custom_all_reduce,
                max_model_len=config.vllm_max_model_len,
                max_num_seqs=max(1, int(config.vllm_scoring_max_num_seqs)),
                limit_mm_per_prompt={"image": 0, "video": 0},
                skip_mm_profiling=True,
            )

        if config.vllm_scoring_device_ids:
            with cuda_visible_devices(config.vllm_scoring_device_ids):
                self._llm = _build_llm()
        else:
            self._llm = _build_llm()

    def score_prompt_ids(
        self,
        context_ids: list[int],
        answer_ids: list[int],
    ) -> torch.Tensor:
        if not answer_ids:
            raise ValueError("answer token sequence is empty")
        if len(context_ids) <= 0:
            raise ValueError("context token sequence is empty")

        prompt_ids = context_ids + answer_ids
        context_len = len(context_ids)
        max_len = int(self.config.vllm_max_model_len)
        # vLLM requires prompt_token_ids + max_tokens <= max_model_len.
        max_prompt_len = max(1, max_len - 1)
        if len(prompt_ids) > max_prompt_len:
            overflow = len(prompt_ids) - max_prompt_len
            if overflow >= len(context_ids):
                raise ValueError(
                    f"J scoring context exceeds vllm_max_model_len={max_len} "
                    f"(prompt_tokens={len(prompt_ids)})"
                )
            context_ids = context_ids[overflow:]
            context_len = len(context_ids)
            prompt_ids = context_ids + answer_ids

        from quantification.clients.j_prompt_token_limit import check_prompt_tokens_or_raise

        check_prompt_tokens_or_raise(len(prompt_ids))

        # prompt_logprobs=0 returns only the actual prompt token logprob at each
        # position (no extra top-k alternatives), matching CrossEntropy targets.
        sampling_params = self._SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=0,
        )
        outputs = self._llm.generate(
            prompts=[self._TokensPrompt(prompt_token_ids=prompt_ids)],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if not outputs:
            raise RuntimeError("vLLM generate returned no outputs")

        prompt_logprobs = outputs[0].prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM returned no prompt_logprobs")

        return extract_answer_token_nlls_from_prompt_logprobs(
            prompt_logprobs,
            context_len,
            answer_ids,
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: object,
    ) -> tuple[str, dict[str, int]]:
        """
        Open-ended chat generation on the same in-process vLLM engine as J scoring.
        Uses a separate code path from teacher-forced prompt_logprobs.
        """
        kwargs.pop("response_format", None)
        kwargs.pop("model", None)
        extra_body = kwargs.pop("extra_body", None)
        chat_template_kwargs = kwargs.pop("chat_template_kwargs", None)
        merged_ctk: dict[str, object] = {}
        if isinstance(chat_template_kwargs, dict):
            merged_ctk.update(chat_template_kwargs)
        if isinstance(extra_body, dict):
            eb_ctk = extra_body.get("chat_template_kwargs")
            if isinstance(eb_ctk, dict):
                merged_ctk.update(eb_ctk)
            if extra_body.get("enable_thinking") is not None:
                merged_ctk["enable_thinking"] = extra_body["enable_thinking"]
        merged_ctk["enable_thinking"] = False

        max_out = int(max_tokens if max_tokens is not None else 4096)
        max_out = max(1, min(max_out, 65536))
        temp = float(temperature if temperature is not None else 0.7)

        sampling_params = self._SamplingParams(
            temperature=temp,
            max_tokens=max_out,
        )
        chat_messages = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
            if isinstance(m, dict)
        ]
        with _j_tilde_lock:
            outputs = self._llm.chat(
                messages=chat_messages,
                sampling_params=sampling_params,
                chat_template_kwargs=merged_ctk,
                use_tqdm=False,
            )
        if not outputs:
            raise RuntimeError("vLLM chat returned no outputs")

        out = outputs[0]
        text = ""
        if out.outputs:
            text = str(out.outputs[0].text or "")
        prompt_tokens = len(out.prompt_token_ids or [])
        completion_tokens = len(out.outputs[0].token_ids) if out.outputs else 0
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        return text, usage


def reset_vllm_j_scorer(skip_runtime_reset: bool = False) -> None:
    """Drop cached in-process scorer so the next call rebuilds a fresh vLLM engine."""
    from quantification.clients.vllm_runtime import reset_runtime_state
    from quantification.clients.vllm_shutdown import shutdown_vllm_llm

    global _llm_engine
    with _engine_lock:
        if _llm_engine is not None:
            shutdown_vllm_llm(_llm_engine._llm)
        _llm_engine = None
    if not skip_runtime_reset:
        reset_runtime_state()


def get_vllm_j_scorer_if_loaded() -> VllmJScorer | None:
    return _llm_engine


def get_vllm_j_scorer(config: QuantificationConfig) -> VllmJScorer:
    global _llm_engine
    with _engine_lock:
        if _llm_engine is None:
            _llm_engine = VllmJScorer(config)
        return _llm_engine


def compare_transformers_and_vllm_nlls(
    *,
    transformers_nlls: torch.Tensor,
    vllm_nlls: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Raise if teacher-forced NLL vectors diverge beyond tolerance."""
    if transformers_nlls.shape != vllm_nlls.shape:
        raise ValueError(
            f"NLL shape mismatch: transformers={tuple(transformers_nlls.shape)} "
            f"vllm={tuple(vllm_nlls.shape)}"
        )
    if not torch.allclose(transformers_nlls, vllm_nlls, atol=atol, rtol=rtol):
        max_delta = (transformers_nlls - vllm_nlls).abs().max().item()
        raise ValueError(
            "transformers/vLLM teacher-forced NLL mismatch; "
            f"max_abs_delta={max_delta:.6g}, atol={atol}, rtol={rtol}"
        )


def compute_j_tilde(
    backend: JScoringBackend,
    tokenizer: AutoTokenizer,
    context: str,
    answer: str,
    *,
    config: QuantificationConfig,
    verify_transformers: AutoModelForCausalLM | None = None,
) -> tuple[float, float, dict]:
    """
    Compute J for (context, answer).

    Returns:
        j_metric: mean(NLL), used by downstream utility (J decreases => node helps)
        j_tilde: -mean(NLL), same sign convention as Unified_MAS get_j_tilde
        log_fields: metadata for api log
    """
    if not answer:
        raise ValueError("ground_truth answer must not be empty")

    question = context.split(config.context_separator, 1)[0]
    contexts = [context]

    last_error: str | None = None
    for ctx in contexts:
        ctx_ids, ans_ids = tokenize_context_and_answer(tokenizer, ctx, answer)
        if not ans_ids:
            raise ValueError(f"answer tokenized to empty sequence: {answer!r}")

        prompt_len = len(ctx_ids) + len(ans_ids)
        max_len = int(config.vllm_max_model_len)
        max_prompt_len = max(1, max_len - 1)
        if prompt_len > max_prompt_len:
            overflow = prompt_len - max_prompt_len
            if overflow >= len(ctx_ids):
                raise ValueError(
                    f"J scoring context exceeds vllm_max_model_len={max_len} "
                    f"(prompt_tokens={prompt_len})"
                )
            ctx_ids = ctx_ids[overflow:]
            prompt_len = len(ctx_ids) + len(ans_ids)

        from quantification.clients.j_prompt_token_limit import (
            JPromptTokenLimitExceeded,
            check_prompt_tokens_or_raise,
            tighten_max_j_prompt_tokens_from_oom,
        )

        check_prompt_tokens_or_raise(prompt_len)

        with _j_tilde_lock:
            try:
                t0 = time.perf_counter()
                token_nlls = backend.score_prompt_ids(ctx_ids, ans_ids)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if verify_transformers is not None:
                    ref_nlls = score_token_nlls_transformers(
                        verify_transformers,
                        ctx_ids,
                        ans_ids,
                    )
                    compare_transformers_and_vllm_nlls(
                        transformers_nlls=ref_nlls,
                        vllm_nlls=token_nlls,
                    )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                err_name = type(exc).__name__
                if isinstance(exc, JPromptTokenLimitExceeded):
                    raise
                if (
                    err_name == "OutOfMemoryError"
                    or err_name == "EngineDeadError"
                    or "out of memory" in str(exc).lower()
                ):
                    # Tighten limit and skip this sample; never retry the same prompt.
                    limit = tighten_max_j_prompt_tokens_from_oom(prompt_len)
                    reset_vllm_j_scorer()
                    raise JPromptTokenLimitExceeded(
                        prompt_len,
                        limit,
                        from_oom=True,
                    ) from exc
                continue

        j_tilde = j_tilde_from_token_nlls(token_nlls)
        j_metric = j_metric_from_token_nlls(token_nlls)
        log_fields = {
            "context_chars": len(ctx),
            "context_tokens": len(ctx_ids),
            "num_answer_tokens": len(ans_ids),
            "answer_token_nlls": token_nlls.tolist(),
            "j_tilde": j_tilde,
            "j_value": j_metric,
            "elapsed_ms": round(elapsed_ms, 2),
        }
        return j_metric, j_tilde, log_fields

    raise RuntimeError(
        "vLLM in-process scoring failed; "
        f"last_error={last_error}"
    )
