"""In-process vLLM client for Skill-MAS agent / retrieval (shared scoring engine)."""

from __future__ import annotations

import asyncio
import threading
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from quantification.config import QuantificationConfig

from .openai_async_client import enrich_usage_with_cost, load_pricing_table


@dataclass
class _ChatMessage:
    content: str


@dataclass
class _ChatChoice:
    message: _ChatMessage


@dataclass
class _ChatUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _ChatCompletion:
    choices: list[_ChatChoice]
    usage: _ChatUsage


class _VllmChatCompletions:
    """Minimal OpenAI ``chat.completions`` shim for retrieval planner calls."""

    def __init__(self, owner: "AsyncVllmInProcessClient") -> None:
        self._owner = owner

    async def create(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> _ChatCompletion:
        text, usage = await self._owner._generate_messages(messages, **kwargs)
        pt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        ct = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        tt = int(usage.get("total_tokens", pt + ct) or 0)
        return _ChatCompletion(
            choices=[_ChatChoice(message=_ChatMessage(content=text))],
            usage=_ChatUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt),
        )


class _VllmChatNamespace:
    """Mirrors ``AsyncOpenAI.chat`` so ``client.chat.completions.create`` works."""

    def __init__(self, completions: _VllmChatCompletions) -> None:
        self.completions = completions


class _VllmChatAPI:
    """Mirrors ``AsyncOpenAI`` top-level API on ``AsyncOpenAIClient.client``."""

    def __init__(self, completions: _VllmChatCompletions) -> None:
        self.chat = _VllmChatNamespace(completions)


class AsyncVllmInProcessClient:
    """
    Async Skill-MAS client backed by the same in-process vLLM scorer as quantification J.

    - ``generate()``: sub-agent text calls (like OpenAI client)
    - ``client.chat.completions.create()``: BM25 multi-round retrieval planner
    """

    def __init__(
        self,
        model: str,
        pricing_table: Optional[Mapping[str, Any]] = None,
        *,
        config: QuantificationConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or QuantificationConfig.from_env()
        self._scorer = None
        self._scorer_init_lock = threading.Lock()
        max_inflight = max(1, int(self.config.vllm_scoring_max_num_seqs))
        self._generate_semaphore = threading.Semaphore(max_inflight)
        self.reasoning_effort = None
        self.temperature = 0.7
        self.max_tokens = max(1, int(os.environ.get("MAS_GATE_SUB_AGENT_MAX_TOKENS", "8192")))
        self.extra_body: dict[str, Any] | None = {"enable_thinking": False}
        table = dict(pricing_table or load_pricing_table())
        row = table.get(self.model, {})
        if isinstance(row, dict):
            if row.get("temperature") is not None:
                self.temperature = float(row["temperature"])
            if row.get("max_tokens") is not None:
                # max_tokens is output-only.  Keep the MAS-Gate configured pure
                # completion cap independent of prompt length.
                configured_output = max(
                    1, int(os.environ.get("MAS_GATE_SUB_AGENT_MAX_TOKENS", "8192"))
                )
                self.max_tokens = max(1, min(int(row["max_tokens"]), configured_output))
            eb = row.get("extra_body")
            if isinstance(eb, dict):
                self.extra_body = {**eb, "enable_thinking": False}
        self.pricing = {self.model: row} if isinstance(row, dict) else table
        self.client = _VllmChatAPI(_VllmChatCompletions(self))

    def _ensure_scorer(self) -> object:
        if self._scorer is not None:
            return self._scorer
        from quantification.clients.vllm_runtime import prepare_scoring_engine, vllm_process_lock

        with self._scorer_init_lock:
            if self._scorer is None:
                with vllm_process_lock:
                    if self._scorer is None:
                        self._scorer = prepare_scoring_engine(self.config)
        return self._scorer

    def release_scorer(self) -> None:
        from quantification.clients.vllm_runtime import release_all_engines, vllm_process_lock

        with vllm_process_lock:
            self._scorer = None
            release_all_engines()

    async def _generate_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, dict[str, Any]]:
        call_kw = dict(kwargs)
        max_tokens = call_kw.pop("max_tokens", None)
        temperature = call_kw.pop("temperature", None)
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if self.extra_body:
            eb = dict(self.extra_body)
            call_extra = call_kw.pop("extra_body", None)
            if isinstance(call_extra, dict):
                eb.update(call_extra)
            eb["enable_thinking"] = False
            call_kw["extra_body"] = eb
        ctk = call_kw.pop("chat_template_kwargs", None)
        merged_ctk = {"enable_thinking": False}
        if isinstance(ctk, dict):
            merged_ctk.update(ctk)
        merged_ctk["enable_thinking"] = False
        call_kw["chat_template_kwargs"] = merged_ctk

        def _run_generate() -> tuple[str, dict[str, Any]]:
            with self._generate_semaphore:
                scorer = self._ensure_scorer()
                return scorer.generate_chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **call_kw,
                )

        text, usage = await asyncio.to_thread(_run_generate)
        usage = enrich_usage_with_cost(usage, model=self.model, table=self.pricing)
        return text, usage

    async def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
        **create_kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return await self._generate_messages(messages, **create_kwargs)

    async def aclose(self) -> None:
        return None
