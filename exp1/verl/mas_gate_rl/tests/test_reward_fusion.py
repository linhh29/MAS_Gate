from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from mas_gate_rl.reward_fusion import fuse_group_zscore_rewards, write_reward_fusion_step


def _token_rewards(values: list[float], width: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.zeros((len(values), width), dtype=torch.float32)
    mask = torch.ones_like(rewards)
    rewards[:, -1] = torch.tensor(values)
    return rewards, mask


def test_group_sample_zscore_lambda_one_without_rank() -> None:
    rewards, mask = _token_rewards([1, 2, 3, 4, 10, 10, 10, 10])
    fused, diag, metrics = fuse_group_zscore_rewards(
        reward_tensor=rewards,
        response_mask=mask,
        judge_scores=[0, 0, 1, 1, 0, 0, 0, 0],
        uids=np.array(["a"] * 4 + ["b"] * 4, dtype=object),
        lambda_correct=1.0,
        expected_group_size=4,
    )
    expected_dense_a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    expected_dense_a = (expected_dense_a - expected_dense_a.mean()) / expected_dense_a.std(unbiased=True)
    expected_correct_a = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    expected_correct_a = (expected_correct_a - expected_correct_a.mean()) / expected_correct_a.std(unbiased=True)
    np.testing.assert_allclose(diag["reward_fusion/z_dense_reward"][:4], expected_dense_a.numpy())
    np.testing.assert_allclose(diag["reward_fusion/z_correctness_reward"][:4], expected_correct_a.numpy())
    np.testing.assert_allclose(
        fused.sum(dim=-1).numpy(), diag["reward_fusion/fused_pre_grpo_reward"], rtol=1e-6, atol=1e-6
    )
    # Homogeneous correctness and dense groups contribute zero after z-score.
    np.testing.assert_array_equal(diag["reward_fusion/z_dense_reward"][4:], 0.0)
    np.testing.assert_array_equal(diag["reward_fusion/z_correctness_reward"][4:], 0.0)
    assert metrics["reward_fusion/mixed_group_ratio"] == pytest.approx(0.5)
    assert metrics["reward_fusion/lambda_correct"] == 1.0


def test_group_size_and_nonfinite_fail_closed() -> None:
    rewards, mask = _token_rewards([1, 2])
    with pytest.raises(ValueError, match="expected 4"):
        fuse_group_zscore_rewards(
            reward_tensor=rewards,
            response_mask=mask,
            judge_scores=[0, 1],
            uids=["a", "a"],
            expected_group_size=4,
        )
    with pytest.raises(ValueError, match="non-finite"):
        fuse_group_zscore_rewards(
            reward_tensor=rewards,
            response_mask=mask,
            judge_scores=[0, float("nan")],
            uids=["a", "a"],
            expected_group_size=2,
        )


def test_write_full_step_trace(tmp_path) -> None:
    rewards, mask = _token_rewards([1, 2])
    _, diag, _ = fuse_group_zscore_rewards(
        reward_tensor=rewards,
        response_mask=mask,
        judge_scores=[0, 1],
        uids=["a", "a"],
        expected_group_size=2,
    )
    advantages = torch.tensor([[2.0] * 4, [-2.0] * 4])
    path = write_reward_fusion_step(
        output_root=tmp_path,
        step=3,
        uids=["a", "a"],
        diagnostics=diag,
        final_advantages=advantages,
        response_mask=mask,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["lambda_correct"] == 0.75
    assert rows[0]["final_grpo_advantage"] == 2.0
    assert rows[1]["final_grpo_advantage"] == -2.0
