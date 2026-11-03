"""Tests for adaptive J-scoring prompt token limit."""

from __future__ import annotations

import math

import pytest

from quantification.clients.j_prompt_token_limit import (
    JPromptTokenLimitExceeded,
    check_prompt_tokens_or_raise,
    finite_max_j_prompt_tokens,
    format_learned_j_prompt_token_limit,
    get_max_j_prompt_tokens,
    prompt_tokens_allowed,
    reset_max_j_prompt_tokens,
    tighten_max_j_prompt_tokens_from_oom,
)


@pytest.fixture(autouse=True)
def _reset_limit() -> None:
    reset_max_j_prompt_tokens()


def test_starts_unlimited() -> None:
    assert math.isinf(get_max_j_prompt_tokens())
    assert finite_max_j_prompt_tokens() is None
    assert format_learned_j_prompt_token_limit() == "unlimited"
    assert prompt_tokens_allowed(1_000_000)


def test_tighten_from_oom() -> None:
    assert tighten_max_j_prompt_tokens_from_oom(7002) == 7001
    assert finite_max_j_prompt_tokens() == 7001
    assert format_learned_j_prompt_token_limit() == "7001"
    assert not prompt_tokens_allowed(7002)
    assert prompt_tokens_allowed(7001)


def test_monotonic_tightening() -> None:
    tighten_max_j_prompt_tokens_from_oom(7002)
    tighten_max_j_prompt_tokens_from_oom(5000)
    assert get_max_j_prompt_tokens() == 4999


def test_check_prompt_tokens_or_raise() -> None:
    tighten_max_j_prompt_tokens_from_oom(100)
    with pytest.raises(JPromptTokenLimitExceeded):
        check_prompt_tokens_or_raise(100)


def test_compute_j_tilde_engine_dead_error_tightens_limit() -> None:
    import torch

    from quantification.clients.j_tilde import compute_j_tilde
    from quantification.config import QuantificationConfig

    class EngineDeadError(Exception):
        pass

    score_calls = 0

    class _Backend:
        def score_prompt_ids(self, ctx_ids: list[int], ans_ids: list[int]) -> torch.Tensor:
            nonlocal score_calls
            score_calls += 1
            raise EngineDeadError(
                "EngineCore encountered an issue. See stack trace (above) for the root cause."
            )

    class _Tokenizer:
        def __call__(self, text: str, return_tensors: str = "pt") -> object:
            size = 500 if text == "answer" else 5000
            return type("R", (), {"input_ids": torch.tensor([list(range(size))])})()

    config = QuantificationConfig(vllm_max_model_len=65536)
    with pytest.raises(JPromptTokenLimitExceeded) as exc_info:
        compute_j_tilde(
            _Backend(),
            _Tokenizer(),
            "question\ncontext",
            "answer",
            config=config,
        )
    assert exc_info.value.from_oom is True
    assert get_max_j_prompt_tokens() == 5499
    assert score_calls == 1
