"""OpenAI-compatible remote API client for logprobs scoring and embeddings."""

from __future__ import annotations

import os
import random
import threading
import hashlib
from collections import OrderedDict
from functools import lru_cache
import time
import urllib.request
from urllib.parse import urlsplit
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from quantification.config import QuantificationConfig
from quantification.run_logger import get_run_logger

T = TypeVar("T")

_SCORING_SYSTEM_PROMPT = (
    "Reply with only the exact answer string, no extra words."
)


def _bool_env(name: str, default: str = "0") -> bool:
    return str(__import__("os").environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _vllm_root(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def _is_local_vllm_endpoint(base_url: str) -> bool:
    """Return whether an OpenAI-compatible endpoint is loopback-local.

    MAS-Gate deliberately configures ``backend=openai`` when it talks to the
    already-running local vLLM servers.  The backend name therefore cannot be
    used to decide whether local-vLLM context fitting is required.
    """
    try:
        hostname = (urlsplit(base_url).hostname or "").strip().lower()
    except ValueError:
        return False
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _vllm_control(base_url: str, endpoint: str, query: str = "") -> None:
    # Used only for local OpenAI-compatible vLLM servers with --enable-sleep-mode.
    if not _bool_env("QUANTIFICATION_VLLM_CONTROL_SLEEP", "0"):
        return
    root = _vllm_root(base_url)
    if not root or not root.startswith("http://127.0.0.1"):
        return
    url = f"{root}{endpoint}{query}"
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=float(__import__("os").environ.get("QUANTIFICATION_VLLM_CONTROL_TIMEOUT", "120"))):
            return
    except Exception as exc:
        if _bool_env("QUANTIFICATION_VLLM_CONTROL_STRICT", "0"):
            raise
        print(f"[quantification.openai_client] non-fatal vLLM control failure {url}: {type(exc).__name__}: {exc}", flush=True)


def _join_logprob_tokens(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def _logprobs_match_answer(tokens: Sequence[str], answer: str) -> bool:
    joined = _join_logprob_tokens(tokens)
    if joined == answer:
        return True
    return joined.strip() == answer.strip()


def _select_echoed_answer_logprobs(
    *,
    rendered_prompt: str,
    answer: str,
    echoed_text: str,
    tokens: Sequence[str],
    token_logprobs: Sequence[float | None],
    text_offsets: Sequence[int],
) -> tuple[list[float], list[str]]:
    """Select the fixed answer's prompt logprobs from an echoed completion.

    vLLM's OpenAI ``/v1/completions`` endpoint supports ``echo=True`` and
    ``max_tokens=0``.  Unlike Chat Completions generation logprobs, the echoed
    token logprobs include the *given prompt*.  Rendering the final assistant
    message as a continuation therefore gives the exact conditional NLL of the
    ground-truth answer in one request and does not decode several speculative
    continuations.

    Offsets are used instead of taking the last ``len(tokenize(answer))`` tokens
    so tokenizer boundary merges are handled correctly.  A token is selected
    when its character span overlaps the final answer span.
    """
    if not answer:
        raise ValueError("ground_truth answer must not be empty")
    if not rendered_prompt.endswith(answer):
        raise ValueError("rendered scoring prompt does not end with ground-truth answer")
    if echoed_text != rendered_prompt:
        raise ValueError("vLLM completion echo does not exactly match rendered scoring prompt")
    if not (len(tokens) == len(token_logprobs) == len(text_offsets)):
        raise ValueError(
            "inconsistent echoed completion logprob arrays: "
            f"tokens={len(tokens)} logprobs={len(token_logprobs)} offsets={len(text_offsets)}"
        )

    answer_start = len(rendered_prompt) - len(answer)
    answer_end = len(rendered_prompt)
    selected_logprobs: list[float] = []
    selected_tokens: list[str] = []
    for idx, (token, logprob, start) in enumerate(zip(tokens, token_logprobs, text_offsets, strict=True)):
        end = text_offsets[idx + 1] if idx + 1 < len(text_offsets) else len(echoed_text)
        if end <= answer_start or start >= answer_end:
            continue
        if logprob is None:
            raise ValueError(f"missing prompt logprob for answer token at index={idx}")
        selected_tokens.append(str(token))
        selected_logprobs.append(float(logprob))

    if not selected_logprobs:
        raise ValueError(
            f"no echoed prompt logprobs overlap answer span [{answer_start}, {answer_end})"
        )
    return selected_logprobs, selected_tokens


def _scoring_max_token_candidates(answer: str) -> list[int]:
    """Try a small budget first to avoid hallucinated continuations."""
    estimate = max(4, min(128, len(answer) // 2 + 4))
    raw = [
        min(8, estimate),
        estimate,
        max(4, min(64, len(answer) // 3 + 2)),
        max(4, min(32, len(answer) // 4 + 2)),
        16,
        32,
        64,
        128,
    ]
    seen: set[int] = set()
    candidates: list[int] = []
    for value in raw:
        capped = max(1, min(65536, value))
        if capped not in seen:
            seen.add(capped)
            candidates.append(capped)
    return candidates


def _truncate_scoring_context(
    context: str,
    question: str,
    max_chars: int,
) -> str:
    if len(context) <= max_chars:
        return context
    if context.startswith(question):
        rest = context[len(question) :].lstrip("\n")
        budget = max(max_chars - len(question) - 1, 512)
        if len(rest) > budget:
            rest = rest[-budget:]
        return f"{question}\n{rest}"
    return context[-max_chars:]


@lru_cache(maxsize=4)
def _local_scoring_tokenizer(model_path: str):
    """Load the local Qwen tokenizer only; never resolve a remote URL."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )


def _fit_scoring_messages_to_context(
    messages: Sequence[dict[str, Any]],
    *,
    max_tokens: int,
    config: QuantificationConfig,
) -> tuple[list[dict[str, Any]], int]:
    """Token-accurately fit a J-score request into local vLLM's context.

    Character limits are unsafe for mathematical/LaTeX traces: an 8k-character
    context can exceed 8k Qwen tokens.  Preserve the system prompt, assistant
    answer prefill, the beginning of the task, and the newest dependency output
    at the end of the user message.  A safety margin covers small differences
    between transformers and vLLM chat-template accounting.
    """
    copied = [dict(message) for message in messages]
    is_local_vllm = (
        config.resolved_backend == "vllm"
        or _is_local_vllm_endpoint(config.scoring_base_url)
    )
    if not is_local_vllm:
        return copied, -1

    context_limit = int(
        os.environ.get(
            "QUANTIFICATION_SCORING_MAX_MODEL_LEN",
            os.environ.get(
                "MAS_GATE_SUB_AGENT_MAX_MODEL_LEN", str(config.vllm_max_model_len)
            ),
        )
    )
    safety = max(32, int(os.environ.get("QUANTIFICATION_SCORING_CONTEXT_SAFETY_TOKENS", "192")))
    target_prompt = context_limit - int(max_tokens) - safety
    if target_prompt < 256:
        raise ValueError(
            f"invalid J-score context budget: limit={context_limit}, "
            f"max_tokens={max_tokens}, safety={safety}"
        )

    model_path = (
        config.scoring_model_path
        or os.environ.get("QUANTIFICATION_SCORING_MODEL_PATH", "")
        or os.environ.get("MAS_GATE_SUB_AGENT_MODEL_PATH", "")
        or os.environ.get("SUB_AGENT_MODEL_PATH", "")
    )
    if not model_path or not os.path.isabs(model_path) or not os.path.isdir(model_path):
        raise ValueError(
            "local vLLM J-score context fitting requires an existing local "
            "scoring model directory; set QUANTIFICATION_SCORING_MODEL_PATH"
        )
    tokenizer = _local_scoring_tokenizer(model_path)

    def count_prompt(candidate: Sequence[dict[str, Any]]) -> int:
        kwargs = {"tokenize": True, "add_generation_prompt": False}
        # vLLM continues a final assistant prefill.  Newer transformers exposes
        # the matching flag; older checkpoint code may not, so fall back safely.
        try:
            ids = tokenizer.apply_chat_template(
                candidate, continue_final_message=True, **kwargs
            )
        except (TypeError, ValueError):
            ids = tokenizer.apply_chat_template(candidate, **kwargs)
        return len(ids)

    prompt_tokens = count_prompt(copied)
    if prompt_tokens <= target_prompt:
        return copied, prompt_tokens

    user_idx = next(
        (i for i in range(len(copied) - 1, -1, -1) if copied[i].get("role") == "user"),
        None,
    )
    if user_idx is None:
        raise ValueError(
            f"J-score prompt has no compactable user message: tokens={prompt_tokens}"
        )

    original = str(copied[user_idx].get("content", ""))
    content_ids = tokenizer.encode(original, add_special_tokens=False)
    marker = "\n\n[MAS-GATE: middle of verbose reasoning context omitted]\n\n"
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)

    # Binary search the largest retained user-token budget.  Keep the task
    # prefix and preferentially retain the latest dependency output at the tail.
    low, high = 32, max(32, len(content_ids) - len(marker_ids))
    best: tuple[str, int] | None = None
    while low <= high:
        keep = (low + high) // 2
        head_n = max(16, int(keep * 0.30))
        tail_n = max(16, keep - head_n)
        compact = (
            tokenizer.decode(content_ids[:head_n], skip_special_tokens=True)
            + marker
            + tokenizer.decode(content_ids[-tail_n:], skip_special_tokens=True)
        )
        copied[user_idx]["content"] = compact
        candidate_tokens = count_prompt(copied)
        if candidate_tokens <= target_prompt:
            best = (compact, candidate_tokens)
            low = keep + 1
        else:
            high = keep - 1

    if best is None:
        raise ValueError(
            "J-score prompt cannot be compacted into local context: "
            f"original_tokens={prompt_tokens}, target={target_prompt}"
        )
    copied[user_idx]["content"], prompt_tokens = best
    return copied, prompt_tokens


class OpenAICompatibleClient:
    """
    Unified client for:
    - Answer cross-entropy via Chat Completions API (prefilled assistant + logprobs)
    - Text embeddings via Embeddings API
    """

    def __init__(self, config: QuantificationConfig) -> None:
        config.validate()
        self.config = config
        self._backend = config.resolved_backend or "openai"
        self._scoring_client = OpenAI(
            base_url=config.scoring_base_url,
            api_key=config.scoring_api_key,
            timeout=config.request_timeout,
        )
        self._embedding_client = OpenAI(
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
            timeout=config.request_timeout,
        )
        self._scoring_semaphore = threading.Semaphore(config.scoring_concurrency)
        self._embedding_semaphore = threading.Semaphore(config.embedding_concurrency)
        # Qwen30B is fixed during actor training.  The same J(question, answer)
        # baseline is requested repeatedly for the root node and for all eight
        # GRPO rollouts of one prompt.  Cache exact inputs only; this changes no
        # metric value and avoids expensive duplicate prefill work.
        self._j_cache_max_entries = max(0, int(os.environ.get("QUANTIFICATION_J_CACHE_MAX_ENTRIES", "4096")))
        self._j_cache: OrderedDict[str, float] = OrderedDict()
        self._j_cache_lock = threading.Lock()
        self._j_inflight: dict[str, threading.Event] = {}


    def begin_scoring_phase(self) -> None:
        # MAS-Gate serves Qwen30B and BGE as local vLLM OpenAI servers on the same 8 GPUs.
        # Wake scoring and optionally sleep embedding before J/logprob calls.
        if _bool_env("QUANTIFICATION_SLEEP_EMBEDDING_DURING_SCORING", "1"):
            _vllm_control(self.config.embedding_base_url, "/sleep", f"?level={__import__('os').environ.get('QUANTIFICATION_EMBEDDING_SLEEP_LEVEL', '1')}")
        _vllm_control(self.config.scoring_base_url, "/wake_up")

    def begin_embedding_phase(self) -> None:
        # Sleep Qwen30B and wake BGE before OpenAI /embeddings calls.
        if _bool_env("QUANTIFICATION_SLEEP_SCORING_DURING_EMBEDDING", "1"):
            _vllm_control(self.config.scoring_base_url, "/sleep", f"?level={__import__('os').environ.get('QUANTIFICATION_SCORING_SLEEP_LEVEL', '1')}")
        _vllm_control(self.config.embedding_base_url, "/wake_up")

    def release_all_engines(self) -> None:
        # Best-effort final sleep for both local vLLM servers; no process kill.
        _vllm_control(self.config.scoring_base_url, "/sleep", f"?level={__import__('os').environ.get('QUANTIFICATION_SCORING_SLEEP_LEVEL', '1')}")
        _vllm_control(self.config.embedding_base_url, "/sleep", f"?level={__import__('os').environ.get('QUANTIFICATION_EMBEDDING_SLEEP_LEVEL', '1')}")

    def _retry_delay_s(self, attempt: int) -> float:
        delay = min(
            self.config.retry_max_delay_s,
            self.config.retry_base_delay_s * (2**attempt),
        )
        return delay + random.uniform(0, min(1.0, delay * 0.1))

    def _call_with_retries(
        self,
        *,
        kind: str,
        fn: Callable[[], T],
    ) -> tuple[T, float]:
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                t0 = time.perf_counter()
                result = fn()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return result, elapsed_ms
            except Exception as exc:
                last_exc = exc
                retryable = isinstance(
                    exc, (APIConnectionError, APITimeoutError, RateLimitError)
                )
                if attempt + 1 >= self.config.max_retries or not retryable:
                    raise
                delay_s = self._retry_delay_s(attempt)
                logger = get_run_logger()
                if logger is not None:
                    logger.info(
                        f"retry {kind} attempt={attempt + 1}/{self.config.max_retries} "
                        f"error={type(exc).__name__}: {exc} sleep={delay_s:.1f}s"
                    )
                time.sleep(delay_s)
        raise RuntimeError(f"Failed {kind} API call") from last_exc

    def format_context(self, question: str, prefix_outputs: Sequence[str]) -> str:
        parts = [question]
        for output in prefix_outputs:
            if output:
                parts.append(output)
        return self.config.context_separator.join(parts)

    def compute_answer_cross_entropy(self, context: str, answer: str) -> float:
        """Return exact J(context, answer), coalescing duplicate local requests."""
        if self._j_cache_max_entries <= 0:
            return self._compute_answer_cross_entropy_uncached(context, answer)

        digest = hashlib.sha256()
        digest.update(context.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\x00")
        digest.update(answer.encode("utf-8", errors="surrogatepass"))
        key = digest.hexdigest()

        while True:
            with self._j_cache_lock:
                cached = self._j_cache.get(key)
                if cached is not None:
                    self._j_cache.move_to_end(key)
                    return cached
                event = self._j_inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._j_inflight[key] = event
                    owner = True
                else:
                    owner = False
            if owner:
                break
            # Another reward coroutine/thread is computing this exact fixed-model
            # score.  Wait instead of sending a duplicate long-prefill request.
            event.wait(timeout=max(1.0, self.config.request_timeout * self.config.max_retries))

        try:
            value = self._compute_answer_cross_entropy_uncached(context, answer)
            with self._j_cache_lock:
                self._j_cache[key] = value
                self._j_cache.move_to_end(key)
                while len(self._j_cache) > self._j_cache_max_entries:
                    self._j_cache.popitem(last=False)
            return value
        finally:
            with self._j_cache_lock:
                done = self._j_inflight.pop(key, None)
                if done is not None:
                    done.set()

    def _compute_answer_cross_entropy_uncached(self, context: str, answer: str) -> float:
        mode = os.environ.get("QUANTIFICATION_J_SCORING_MODE", "prompt_logprobs").strip().lower()
        if mode in {"prompt_logprobs", "completion_echo", "echo"}:
            return self._compute_answer_cross_entropy_prompt_logprobs(context, answer)
        if mode in {"legacy_chat", "chat"}:
            return self._compute_answer_cross_entropy_legacy_chat(context, answer)
        raise ValueError(
            "QUANTIFICATION_J_SCORING_MODE must be prompt_logprobs or legacy_chat, "
            f"got {mode!r}"
        )

    def _compute_answer_cross_entropy_prompt_logprobs(self, context: str, answer: str) -> float:
        """Compute exact answer NLL with one local-vLLM echoed completion.

        The old Chat Completions implementation generated a continuation after
        the assistant answer prefill and then compared that continuation with
        the answer.  They normally cannot match, so it retried the same J score
        with several max-token budgets and finally measured generated text
        rather than the fixed ground truth.  Echoed prompt logprobs both restore
        the intended J(context, answer) definition and eliminate those retries.
        """
        if not answer:
            raise ValueError("ground_truth answer must not be empty")

        messages = [
            {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\nAnswer:"},
            {"role": "assistant", "content": answer},
        ]
        fitted_messages, prompt_tokens = _fit_scoring_messages_to_context(
            messages, max_tokens=1, config=self.config
        )

        model_path = (
            self.config.scoring_model_path
            or os.environ.get("QUANTIFICATION_SCORING_MODEL_PATH", "")
            or os.environ.get("MAS_GATE_SUB_AGENT_MODEL_PATH", "")
            or os.environ.get("SUB_AGENT_MODEL_PATH", "")
        )
        if not model_path or not os.path.isabs(model_path) or not os.path.isdir(model_path):
            raise ValueError(
                "prompt-logprob J scoring requires an existing local scoring model directory"
            )
        tokenizer = _local_scoring_tokenizer(model_path)
        render_kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        try:
            rendered_prompt = tokenizer.apply_chat_template(
                fitted_messages,
                continue_final_message=True,
                enable_thinking=False,
                **render_kwargs,
            )
        except (TypeError, ValueError):
            rendered_prompt = tokenizer.apply_chat_template(
                fitted_messages,
                continue_final_message=True,
                **render_kwargs,
            )
        if not rendered_prompt.endswith(answer):
            raise RuntimeError("local chat template did not preserve assistant answer suffix")

        with self._scoring_semaphore:
            response, elapsed_ms = self._call_with_retries(
                kind="j_score_prompt_logprobs",
                fn=lambda: self._scoring_client.completions.create(
                    model=self.config.scoring_model,
                    prompt=rendered_prompt,
                    temperature=0.0,
                    max_tokens=0,
                    echo=True,
                    logprobs=1,
                ),
            )

        choice = response.choices[0]
        logprobs_obj = choice.logprobs
        if (
            logprobs_obj is None
            or logprobs_obj.tokens is None
            or logprobs_obj.token_logprobs is None
            or logprobs_obj.text_offset is None
        ):
            raise RuntimeError("vLLM completion echo returned incomplete prompt logprobs")
        answer_token_logprobs, answer_tokens = _select_echoed_answer_logprobs(
            rendered_prompt=rendered_prompt,
            answer=answer,
            echoed_text=str(choice.text),
            tokens=logprobs_obj.tokens,
            token_logprobs=logprobs_obj.token_logprobs,
            text_offsets=logprobs_obj.text_offset,
        )

        j_value = float(-np.mean(answer_token_logprobs))
        logger = get_run_logger()
        if logger is not None:
            logger.log_api_call(
                "j_score_prompt_logprobs",
                backend=self._backend,
                model=self.config.scoring_model,
                base_url=self.config.scoring_base_url,
                context_chars=len(context),
                answer=answer,
                prompt_tokens=prompt_tokens,
                num_answer_tokens=len(answer_token_logprobs),
                answer_logprobs=answer_token_logprobs,
                returned_tokens=_join_logprob_tokens(answer_tokens),
                tokens_matched_answer=True,
                j_value=j_value,
                elapsed_ms=round(elapsed_ms, 2),
            )
        return j_value

    def _compute_answer_cross_entropy_legacy_chat(self, context: str, answer: str) -> float:
        """J = mean token NLL over answer tokens via Chat Completions logprobs."""
        if not answer:
            raise ValueError("ground_truth answer must not be empty")

        question = context.split(self.config.context_separator, 1)[0]
        context_variants = [context]
        if len(context) > self.config.scoring_context_max_chars:
            truncated = _truncate_scoring_context(
                context,
                question,
                self.config.scoring_context_max_chars,
            )
            if truncated != context:
                context_variants.append(truncated)

        extra_body = dict(self.config.scoring_extra_body or {})
        extra_body.setdefault("chat_template_kwargs", {})
        extra_body["chat_template_kwargs"]["enable_thinking"] = False
        base_request: dict[str, Any] = {
            "model": self.config.scoring_model,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 1,
            "extra_body": extra_body,
        }

        last_mismatch: str | None = None
        elapsed_ms = 0.0
        answer_token_logprobs: list[float] = []
        returned_tokens: list[str] = []
        tokens_matched_answer = False
        used_max_tokens = 0
        used_context_chars = len(context)
        best_fallback: tuple[list[float], list[str], int, int] | None = None

        for ctx in context_variants:
            messages = [
                {"role": "system", "content": _SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": f"{ctx}\nAnswer:"},
                {"role": "assistant", "content": answer},
            ]
            for max_tokens in _scoring_max_token_candidates(answer):
                fitted_messages, _prompt_tokens = _fit_scoring_messages_to_context(
                    messages, max_tokens=max_tokens, config=self.config
                )
                with self._scoring_semaphore:
                    response, elapsed_ms = self._call_with_retries(
                        kind="j_score_chat",
                        fn=lambda mt=max_tokens, msg=fitted_messages: self._scoring_client.chat.completions.create(
                            **{**base_request, "messages": msg, "max_tokens": mt}
                        ),
                    )

                choice = response.choices[0]
                logprobs_obj = choice.logprobs
                if logprobs_obj is None or not logprobs_obj.content:
                    last_mismatch = "missing logprobs"
                    continue

                tokens = [item.token for item in logprobs_obj.content]
                logprob_values = [
                    item.logprob
                    for item in logprobs_obj.content
                    if item.logprob is not None
                ]
                if not logprob_values:
                    last_mismatch = "empty logprobs"
                    continue

                if _logprobs_match_answer(tokens, answer):
                    answer_token_logprobs = logprob_values
                    returned_tokens = list(tokens)
                    tokens_matched_answer = True
                    used_max_tokens = max_tokens
                    used_context_chars = len(ctx)
                    break

                joined = _join_logprob_tokens(tokens)
                last_mismatch = (
                    f"logprob tokens {joined!r} != answer {answer!r} "
                    f"(max_tokens={max_tokens}, context_chars={len(ctx)})"
                )
                best_fallback = (logprob_values, list(tokens), max_tokens, len(ctx))
            if answer_token_logprobs:
                break

        if not answer_token_logprobs and best_fallback is not None:
            answer_token_logprobs, returned_tokens, used_max_tokens, used_context_chars = (
                best_fallback
            )
            logger = get_run_logger()
            if logger is not None:
                logger.info(
                    "j_score_chat using returned logprobs without exact ground-truth token match; "
                    f"last_error={last_mismatch}"
                )

        if not answer_token_logprobs:
            raise RuntimeError(
                "Scoring API returned no usable logprobs; "
                f"last_error={last_mismatch}"
            )

        j_value = float(-np.mean(answer_token_logprobs))
        logger = get_run_logger()
        if logger is not None:
            logger.log_api_call(
                "j_score_chat",
                backend=self._backend,
                model=self.config.scoring_model,
                base_url=self.config.scoring_base_url,
                context_chars=used_context_chars,
                answer=answer,
                max_tokens=used_max_tokens,
                num_answer_tokens=len(answer_token_logprobs),
                answer_logprobs=answer_token_logprobs,
                returned_tokens=_join_logprob_tokens(returned_tokens),
                tokens_matched_answer=tokens_matched_answer,
                j_value=j_value,
                elapsed_ms=round(elapsed_ms, 2),
            )
        return j_value

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        all_vectors: list[list[float]] = []
        batch_size = self.config.embedding_batch_size

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            with self._embedding_semaphore:
                response, elapsed_ms = self._call_with_retries(
                    kind="embedding",
                    fn=lambda: self._embedding_client.embeddings.create(
                        model=self.config.embedding_model,
                        input=batch,
                    ),
                )

            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_vectors.extend([item.embedding for item in sorted_data])
            logger = get_run_logger()
            if logger is not None:
                dim = len(sorted_data[0].embedding) if sorted_data else 0
                logger.log_api_call(
                    "embedding",
                    backend=self._backend,
                    model=self.config.embedding_model,
                    base_url=self.config.embedding_base_url,
                    batch_start=start,
                    num_texts=len(batch),
                    embedding_dim=dim,
                    elapsed_ms=round(elapsed_ms, 2),
                    text_previews=[(t[:120] + "...") if len(t) > 120 else t for t in batch[:3]],
                )

        matrix = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return matrix / norms
