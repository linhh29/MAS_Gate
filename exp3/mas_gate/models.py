"""Local model interfaces used by the Experiment 3 reference pipeline.

Heavy dependencies are imported only when a model is constructed. All model
arguments are existing local directories; this module never downloads weights.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ANSWER_SYSTEM_PROMPT = (
    "Answer the original question using the provided information. "
    "Return only the answer."
)
PROFILES = ("qwen30", "qwen35_4b", "qwen36_27b", "mistral24", "llama8")
_PROFILE_ALIASES = {
    "Qwen3-30B-A3B-Instruct-2507": "qwen30",
    "qwen3_30b_a3b_instruct_2507": "qwen30",
    "Qwen3.5-4B": "qwen35_4b",
    "Qwen3.6-27B": "qwen36_27b",
    "Mistral-Small-3.2-24B-Instruct-2506": "mistral24",
    "Llama-3.1-8B-Instruct": "llama8",
    "llama31_8b": "llama8",
}


def engine_kwargs(
    model_path: str | Path,
    profile: str,
    *,
    max_model_len: int = 81920,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_num_batched_tokens: int = 2048,
) -> dict[str, Any]:
    """Translate the five recorded runtime profiles into vLLM arguments."""
    profile = _PROFILE_ALIASES.get(profile, profile)
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    if type(max_model_len) is not int or max_model_len <= 256:
        raise ValueError("max_model_len must be an integer greater than 256")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    for value in (tensor_parallel_size, max_num_batched_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError("parallel size and batch token budget must be positive integers")
    kwargs: dict[str, Any] = {
        "model": str(model_path),
        "runner": "generate",
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max_num_batched_tokens,
    }
    if profile != "llama8":
        kwargs.update(
            language_model_only=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            skip_mm_profiling=True,
        )
    if profile in ("qwen35_4b", "qwen36_27b"):
        kwargs["gdn_prefill_backend"] = "triton"
    if profile == "mistral24":
        kwargs.update(
            tokenizer_mode="mistral",
            config_format="hf",
            load_format="hf",
            # The checkpoint has a distinct lm_head. vLLM 0.19.1 would
            # otherwise tie and skip it when reading this HF configuration.
            hf_overrides={"tie_word_embeddings": False},
        )
    return kwargs


def _local_model(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("model_path must be an existing local checkpoint with config.json")
    return path


def _token_ids(value: Any) -> list[int]:
    ids = value.input_ids[0].tolist()
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("tokenizer must return a nonempty, flat list of token IDs")
    return ids


def prepare_answer_tokens(
    tokenizer: Any, context: str, answer: str, max_model_len: int = 81920
) -> tuple[list[int], list[int]]:
    """Apply the task-answer readout and tokenize its target separately.

    The target is answer text only: no assistant header, EOS, or added special
    tokens. Even the baseline (question-only context) uses this same readout.
    Context is never truncated. One token is reserved for vLLM's scoring call.
    """
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be nonempty text")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be nonempty text")
    if type(max_model_len) is not int or max_model_len < 2:
        raise ValueError("max_model_len must be an integer of at least 2")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or context not in rendered:
        raise ValueError("chat template did not preserve the raw context")
    context_ids = _token_ids(tokenizer(rendered, return_tensors="pt"))
    answer_ids = _token_ids(tokenizer(answer, return_tensors="pt"))
    bare_answer_ids = _token_ids(
        tokenizer(answer, return_tensors="pt", add_special_tokens=False)
    )
    if answer_ids != bare_answer_ids:
        raise ValueError("default answer tokenization changes the target's special tokens")
    for name in ("bos_token_id", "eos_token_id"):
        if getattr(tokenizer, name, None) in answer_ids:
            raise ValueError(f"answer target contains {name}")
    if len(context_ids) + len(answer_ids) + 1 > max_model_len:
        raise ValueError("answer readout exceeds max_model_len; truncation is forbidden")
    return context_ids, answer_ids


def _close_llm(llm: Any) -> None:
    """Release this engine only; never inspect or stop unrelated processes."""
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=60.0)
    else:
        fallback = getattr(engine, "shutdown", None)
        if callable(fallback):
            fallback()


def _clear_cuda_cache() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class AnswerScorer:
    """Qwen3-30B mean teacher-forced answer NLL, the stored J convention."""

    def __init__(
        self,
        model_path: str | Path,
        max_model_len: int = 81920,
        gpu_memory_utilization: float = 0.9,
        *,
        tensor_parallel_size: int = 1,
        max_num_batched_tokens: int = 2048,
    ) -> None:
        path = _local_model(model_path)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True
        )
        kwargs = engine_kwargs(
            path, "qwen30", max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        kwargs.pop("runner")  # Match the original J engine's autodetected runner.
        self._llm = LLM(**kwargs)
        self._sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        self._tokens_prompt = TokensPrompt

    def __call__(self, context: str, answer: str) -> float:
        import torch

        if self._llm is None:
            raise RuntimeError("answer scorer is closed")
        context_ids, answer_ids = prepare_answer_tokens(
            self.tokenizer, context, answer, self.max_model_len
        )
        outputs = self._llm.generate(
            prompts=[self._tokens_prompt(prompt_token_ids=context_ids + answer_ids)],
            sampling_params=self._sampling,
            use_tqdm=False,
        )
        if not outputs or outputs[0].prompt_logprobs is None:
            raise RuntimeError("vLLM returned no teacher-forced prompt log probabilities")
        logprobs = outputs[0].prompt_logprobs
        nlls = []
        for offset, token in enumerate(answer_ids):
            position = len(context_ids) + offset
            if position >= len(logprobs) or not logprobs[position] or token not in logprobs[position]:
                raise RuntimeError("vLLM omitted a teacher-forced answer token log probability")
            entry = logprobs[position][token]
            nll = -float(entry.logprob if hasattr(entry, "logprob") else entry)
            if not math.isfinite(nll) or nll < 0:
                raise RuntimeError("vLLM returned an invalid answer token NLL")
            nlls.append(nll)
        # The saved runs use float32 tensor reduction, not a Python float sum.
        return float(torch.tensor(nlls, dtype=torch.float32).mean().item())

    def close(self) -> None:
        if self._llm is not None:
            llm, self._llm = self._llm, None
            _close_llm(llm)
            del llm
            _clear_cuda_cache()


class NLI:
    """Paired entailment from the local bge-m3-zeroshot-v2.0 classifier."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        max_length: int = 8192,
        batch_size: int = 32,
    ) -> None:
        path = _local_model(model_path)
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True
        )
        self._model.to(device)
        self._model.eval()
        labels = getattr(self._model.config, "id2label", {})
        normalized = {str(label).lower(): int(index) for index, label in labels.items()}
        if set(normalized) != {"entailment", "not_entailment"}:
            raise ValueError("use bge-m3-zeroshot-v2.0 with entailment/not_entailment labels")
        self._entailment_id = normalized["entailment"]

    def probabilities(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch

        if self._model is None:
            raise RuntimeError("NLI model is closed")
        values: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            inputs = self._tokenizer(
                [pair[0] for pair in batch], [pair[1] for pair in batch],
                return_tensors="pt", truncation=True,
                max_length=self.max_length, padding=True,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            # Original output storage is float32, even for other logits dtypes.
            values.extend(
                torch.softmax(logits, dim=-1)[:, self._entailment_id]
                .to(dtype=torch.float32).cpu().tolist()
            )
        return values

    def close(self) -> None:
        if self._model is not None:
            self._model = None
            _clear_cuda_cache()


class Localizer:
    """Full-trajectory, temperature-zero localizer with a 256-token response."""

    def __init__(
        self,
        model_path: str | Path,
        profile: str,
        max_model_len: int = 81920,
        gpu_memory_utilization: float = 0.9,
        *,
        tensor_parallel_size: int = 1,
        max_num_batched_tokens: int = 2048,
    ) -> None:
        path = _local_model(model_path)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.max_model_len = max_model_len
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True
        )
        self._llm = LLM(**engine_kwargs(
            path, profile, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=max_num_batched_tokens,
        ))
        self._sampling = SamplingParams(temperature=0.0, max_tokens=256)

    def generate(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        if self._llm is None:
            raise RuntimeError("localizer is closed")
        messages = [dict(message) for message in messages]
        prompt_ids = self._tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(prompt_ids, list) or not prompt_ids or any(type(t) is not int for t in prompt_ids):
            raise ValueError("chat template must return a nonempty flat token-ID list")
        if len(prompt_ids) + 256 > self.max_model_len:
            raise ValueError("localizer prompt exceeds max_model_len; truncation is forbidden")
        outputs = self._llm.chat(
            messages=messages, sampling_params=self._sampling,
            chat_template_content_format="auto",
            chat_template_kwargs={"enable_thinking": False}, use_tqdm=False,
        )
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no localizer output")
        output = outputs[0]
        prompt_tokens = len(output.prompt_token_ids or [])
        completion_tokens = len(output.outputs[0].token_ids)
        return {
            "text": str(output.outputs[0].text or ""),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def close(self) -> None:
        if self._llm is not None:
            llm, self._llm = self._llm, None
            _close_llm(llm)
            del llm
            _clear_cuda_cache()
