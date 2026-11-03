"""Tests for teacher-forced J scoring alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from quantification.clients.j_tilde import (
    extract_answer_token_nlls_from_prompt_logprobs,
    j_metric_from_token_nlls,
    j_tilde_from_token_nlls,
)


@dataclass
class _Logprob:
    logprob: float
    rank: int = 1
    decoded_token: str | None = None


def test_extract_answer_token_nlls_from_prompt_logprobs() -> None:
    # prompt token ids: [ctx0, ctx1, ans0=20, ans1=30]
    prompt_logprobs = [
        None,
        {11: _Logprob(-0.5)},
        {20: _Logprob(-2.0)},
        {30: _Logprob(-3.0)},
    ]

    answer_ids = [20, 30]
    nlls = extract_answer_token_nlls_from_prompt_logprobs(
        prompt_logprobs,
        context_len=2,
        answer_ids=answer_ids,
    )
    assert nlls.tolist() == [2.0, 3.0]
    assert j_tilde_from_token_nlls(nlls) == pytest.approx(-2.5)
    assert j_metric_from_token_nlls(nlls) == pytest.approx(2.5)


def test_j_sign_convention() -> None:
    nlls = torch.tensor([1.0, 3.0])
    assert j_tilde_from_token_nlls(nlls) == pytest.approx(-2.0)
    assert j_metric_from_token_nlls(nlls) == pytest.approx(2.0)
