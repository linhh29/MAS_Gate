from __future__ import annotations

import pytest

from mas_gate_rl import ablation_reward


def _result() -> dict:
    return {"score": 99.0, "metric_key": "legacy", "quant_result": {"mean_U": 2.0, "mean_K": 3.0}}


def test_default_and_unknown_tags_use_multiplication() -> None:
    assert ablation_reward.impact_from_u_k(2, 3) == (8.0, "I=U*(1+K)")
    assert ablation_reward.impact_from_u_k(2, 3, "unknown") == (8.0, "I=U*(1+K)")


def test_additive_tag_uses_addition_without_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("Skywork must not be called for additive ablation")

    monkeypatch.setattr(ablation_reward, "_skywork_request", fail)
    out = ablation_reward.apply_ablation(
        _result(), question="q", answer="a", tag=ablation_reward.ADDITIVE_TAG
    )
    assert out["score"] == 6.0
    assert out["quant_result"]["impact_formula"] == "I=U+(1+K)"
    assert out["quant_result"]["impact_formula_tag"] == ablation_reward.ADDITIVE_TAG


def test_skywork_tag_replaces_u_and_keeps_multiplication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ablation_reward, "_skywork_request", lambda q, a: 5.0)
    out = ablation_reward.apply_ablation(
        _result(), question="q", answer="a", tag=ablation_reward.SKYWORK_TAG
    )
    assert out["skywork_raw_score"] == 5.0
    assert out["skywork_normalized_U"] == pytest.approx(0.4621171573)
    assert out["score"] == pytest.approx(0.4621171573 * 4.0)
    assert out["quant_result"]["impact_formula"] == "I=U*(1+K)"
    assert out["quant_result"]["ablation_K"] == 3.0


def test_non_ablation_returns_same_object_without_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("Skywork must not be called for default path")

    monkeypatch.setattr(ablation_reward, "_skywork_request", fail)
    result = _result()
    assert ablation_reward.apply_ablation(result, question="q", answer="a", tag="") is result
