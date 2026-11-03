"""Factory for quantification API clients."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from quantification.clients.openai_client import OpenAICompatibleClient
from quantification.clients.vllm_client import VllmQuantificationClient
from quantification.config import QuantificationConfig


class QuantificationClient(Protocol):
    def format_context(self, question: str, prefix_outputs: list[str] | tuple[str, ...]) -> str: ...

    def compute_answer_cross_entropy(self, context: str, answer: str) -> float: ...

    def embed_texts(self, texts: list[str] | tuple[str, ...]) -> np.ndarray: ...


def create_quantification_client(
    config: QuantificationConfig,
) -> OpenAICompatibleClient | VllmQuantificationClient:
    config.validate()
    backend = (config.backend or "vllm").strip().lower()
    if backend == "vllm":
        return VllmQuantificationClient(config)
    return OpenAICompatibleClient(config)
