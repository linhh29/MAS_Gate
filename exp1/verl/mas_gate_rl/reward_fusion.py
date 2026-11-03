"""Group-wise dense/correctness reward fusion for MAS-Gate GRPO.

The fusion is intentionally performed only after all rollouts for a prompt are
assembled on the trainer driver.  Per-sample reward workers cannot compute a
prompt-group normalization safely.

No rank transform is used.  Both scalar signals use the sample z-score
(``torch.std(..., unbiased=True)``), followed by

    fused = z_dense + lambda_correct * z_correctness

The existing GRPO group normalization remains enabled afterwards.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _as_uid_key(value: Any) -> Any:
    """Return a hashable stable key for numpy/Python uid scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sample_zscore(values: torch.Tensor, eps: float) -> tuple[torch.Tensor, float, float]:
    """Sample z-score with a zero vector for homogeneous groups."""
    mean = values.mean()
    if values.numel() <= 1:
        return torch.zeros_like(values), float(mean.item()), 0.0
    std = values.std(unbiased=True)
    std_value = float(std.item())
    if not math.isfinite(std_value) or std_value <= eps:
        return torch.zeros_like(values), float(mean.item()), 0.0
    return (values - mean) / std, float(mean.item()), std_value


def fuse_group_zscore_rewards(
    *,
    reward_tensor: torch.Tensor,
    response_mask: torch.Tensor,
    judge_scores: Any,
    uids: Any,
    lambda_correct: float = 1.0,
    expected_group_size: int | None = None,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, np.ndarray], dict[str, float]]:
    """Fuse dense and correctness rewards with per-uid sample z-scores.

    Args:
        reward_tensor: Token-level dense reward, shape ``[batch, response_len]``.
        response_mask: Valid response-token mask with the same shape.
        judge_scores: Per-rollout semantic/source correctness or environment success.
        uids: Prompt-group ids aligned with the batch.
        lambda_correct: Weight on standardized correctness; formal training uses 1.0.
        expected_group_size: Optional strict rollout count check (8 for this run).
        eps: Zero-variance threshold.

    Returns:
        A token-level tensor whose per-row sum is the fused scalar reward,
        per-rollout diagnostics, and aggregate metrics.
    """
    if reward_tensor.ndim != 2 or response_mask.shape != reward_tensor.shape:
        raise ValueError(
            f"reward/response mask shape mismatch: reward={tuple(reward_tensor.shape)} "
            f"mask={tuple(response_mask.shape)}"
        )
    batch_size = int(reward_tensor.shape[0])
    uid_values = np.asarray(uids, dtype=object).reshape(-1)
    correct_np = np.asarray(judge_scores, dtype=np.float64).reshape(-1)
    if uid_values.size != batch_size or correct_np.size != batch_size:
        raise ValueError(
            f"fusion inputs must align: batch={batch_size}, uids={uid_values.size}, "
            f"judge_scores={correct_np.size}"
        )
    if not math.isfinite(float(lambda_correct)) or lambda_correct < 0:
        raise ValueError(f"lambda_correct must be finite and non-negative, got {lambda_correct}")

    dense = reward_tensor.detach().sum(dim=-1).to(dtype=torch.float64, device="cpu")
    correct = torch.as_tensor(correct_np, dtype=torch.float64, device="cpu")
    if not torch.isfinite(dense).all():
        raise ValueError("dense reward contains non-finite values; refusing to normalize")
    if not torch.isfinite(correct).all():
        raise ValueError("correctness reward contains non-finite values; refusing to normalize")

    groups: dict[Any, list[int]] = defaultdict(list)
    for i, raw_uid in enumerate(uid_values.tolist()):
        groups[_as_uid_key(raw_uid)].append(i)

    z_dense = torch.zeros_like(dense)
    z_correct = torch.zeros_like(correct)
    dense_mean = torch.zeros_like(dense)
    dense_std = torch.zeros_like(dense)
    correct_mean = torch.zeros_like(correct)
    correct_std = torch.zeros_like(correct)
    mixed = torch.zeros(batch_size, dtype=torch.bool)
    zero_dense_groups = 0
    mixed_groups = 0

    for uid, indices in groups.items():
        if expected_group_size is not None and len(indices) != int(expected_group_size):
            raise ValueError(
                f"uid={uid!r} has {len(indices)} rollouts, expected {expected_group_size}; "
                "group z-score would be invalid"
            )
        idx = torch.as_tensor(indices, dtype=torch.long)
        zd, dmean, dstd = _sample_zscore(dense[idx], eps)
        zc, cmean, cstd = _sample_zscore(correct[idx], eps)
        z_dense[idx] = zd
        z_correct[idx] = zc
        dense_mean[idx] = dmean
        dense_std[idx] = dstd
        correct_mean[idx] = cmean
        correct_std[idx] = cstd
        if dstd == 0.0:
            zero_dense_groups += 1
        if cstd > 0.0:
            mixed[idx] = True
            mixed_groups += 1

    fused = z_dense + float(lambda_correct) * z_correct

    # GRPO consumes only the per-response sum.  Place the fused outcome on the
    # final valid response token, matching the naive reward manager convention.
    mask_cpu = response_mask.detach().to(device="cpu", dtype=torch.bool)
    valid_counts = mask_cpu.sum(dim=-1)
    if torch.any(valid_counts <= 0):
        bad = torch.nonzero(valid_counts <= 0, as_tuple=False).reshape(-1).tolist()
        raise ValueError(f"response_mask has no valid token for rows {bad[:20]}")
    positions = torch.arange(mask_cpu.shape[1], dtype=torch.long).unsqueeze(0).expand_as(mask_cpu)
    last_valid = torch.where(mask_cpu, positions, torch.full_like(positions, -1)).max(dim=-1).values
    fused_tensor = torch.zeros_like(reward_tensor)
    fused_tensor[torch.arange(batch_size, device=reward_tensor.device), last_valid.to(reward_tensor.device)] = fused.to(
        device=reward_tensor.device, dtype=reward_tensor.dtype
    )

    abs_dense = z_dense.abs()
    abs_correct = (float(lambda_correct) * z_correct).abs()
    denom = float((abs_dense + abs_correct).sum().item())
    mixed_mask = mixed
    mixed_denom = float((abs_dense[mixed_mask] + abs_correct[mixed_mask]).sum().item())

    diagnostics = {
        "reward_fusion/raw_dense_reward": dense.numpy(),
        "reward_fusion/raw_correctness_reward": correct.numpy(),
        "reward_fusion/z_dense_reward": z_dense.numpy(),
        "reward_fusion/z_correctness_reward": z_correct.numpy(),
        "reward_fusion/fused_pre_grpo_reward": fused.numpy(),
        "reward_fusion/dense_group_mean": dense_mean.numpy(),
        "reward_fusion/dense_group_std": dense_std.numpy(),
        "reward_fusion/correctness_group_mean": correct_mean.numpy(),
        "reward_fusion/correctness_group_std": correct_std.numpy(),
        "reward_fusion/mixed_correctness_group": mixed.numpy(),
        "reward_fusion/lambda_correct": np.full(batch_size, float(lambda_correct), dtype=np.float64),
    }
    metrics = {
        "reward_fusion/lambda_correct": float(lambda_correct),
        "reward_fusion/group_count": float(len(groups)),
        "reward_fusion/mixed_group_ratio": float(mixed_groups / len(groups)) if groups else float("nan"),
        "reward_fusion/zero_dense_variance_group_ratio": float(zero_dense_groups / len(groups)) if groups else float("nan"),
        "reward_fusion/dense_raw_mean": float(dense.mean().item()),
        "reward_fusion/dense_raw_std": float(dense.std(unbiased=False).item()),
        "reward_fusion/correctness_raw_mean": float(correct.mean().item()),
        "reward_fusion/correctness_raw_std": float(correct.std(unbiased=False).item()),
        "reward_fusion/z_dense_std": float(z_dense.std(unbiased=False).item()),
        "reward_fusion/z_correctness_std": float(z_correct.std(unbiased=False).item()),
        "reward_fusion/fused_pre_grpo_mean": float(fused.mean().item()),
        "reward_fusion/fused_pre_grpo_std": float(fused.std(unbiased=False).item()),
        "reward_fusion/correctness_abs_contribution_share": float(abs_correct.sum().item() / denom) if denom else 0.0,
        "reward_fusion/mixed_correctness_abs_contribution_share": (
            float(abs_correct[mixed_mask].sum().item() / mixed_denom) if mixed_denom else 0.0
        ),
    }
    return fused_tensor, diagnostics, metrics


def write_reward_fusion_step(
    *,
    output_root: str | os.PathLike[str] | None,
    step: int,
    uids: Any,
    diagnostics: dict[str, Any],
    final_advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> Path | None:
    """Persist every rollout's fusion inputs/output/final GRPO advantage.

    The root is the run/checkpoint directory, so these diagnostics remain next
    to all ``global_step_N`` weight snapshots.
    """
    if not output_root:
        return None
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    uid_values = np.asarray(uids, dtype=object).reshape(-1).tolist()
    mask = response_mask.detach().to(dtype=final_advantages.dtype)
    denom = mask.sum(dim=-1).clamp_min(1)
    scalar_adv = ((final_advantages.detach() * mask).sum(dim=-1) / denom).to(device="cpu", dtype=torch.float64).numpy()
    diag_np = {k: np.asarray(v).reshape(-1) for k, v in diagnostics.items()}
    n = len(uid_values)
    if any(len(v) != n for v in diag_np.values()) or len(scalar_adv) != n:
        raise ValueError("reward fusion trace fields are not batch-aligned")

    rows = []
    for i in range(n):
        row = {"step": int(step), "row": i, "uid": str(_as_uid_key(uid_values[i]))}
        for key, values in diag_np.items():
            value = values[i]
            if isinstance(value, np.generic):
                value = value.item()
            row[key.split("/", 1)[-1]] = value
        row["final_grpo_advantage"] = float(scalar_adv[i])
        rows.append(row)

    path = root / f"step_{int(step):04d}.jsonl"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)
    return path
