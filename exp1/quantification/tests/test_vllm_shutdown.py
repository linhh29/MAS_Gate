"""Tests for vLLM engine shutdown helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from quantification.clients.vllm_shutdown import shutdown_vllm_llm


def test_shutdown_vllm_llm_calls_engine_core_shutdown() -> None:
    llm = MagicMock()
    llm.llm_engine.engine_core.shutdown = MagicMock()

    shutdown_vllm_llm(llm, timeout=5.0)

    llm.llm_engine.engine_core.shutdown.assert_called_once_with(timeout=5.0)
