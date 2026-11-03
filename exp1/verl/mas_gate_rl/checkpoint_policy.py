#!/usr/bin/env python3
"""MAS-Gate full-round checkpoint scheduling helpers.

The requested policy emits ten approximately equally spaced intermediate save
positions strictly before the final trained step, then includes the final step
separately. Normal runs therefore produce eleven unique checkpoint models.
"""
from __future__ import annotations

from pathlib import Path


def intermediate_checkpoint_steps(total_steps: int, count: int = 10, start_step: int = 0) -> list[int]:
    """Return ``count`` schedule positions between ``start_step`` and ``total_steps``.

    Normal full runs have at least ten updates. For unusually short smoke runs,
    positions are clamped to the valid range and may repeat; local checkpoint
    directories remain unique by training step. ``start_step`` lets an HF-only
    continuation keep global checkpoint numbering without re-scheduling saves
    over already completed steps.
    """
    total_steps = int(total_steps)
    count = max(0, int(count))
    start_step = int(start_step)
    if total_steps < 1 or count == 0 or start_step >= total_steps:
        return []
    if start_step < 0:
        raise ValueError(f"start_step must be non-negative, got {start_step}")
    remaining_steps = total_steps - start_step
    # Divide the open interval into count+1 segments so no intermediate aliases
    # the separately labelled final checkpoint. De-duplicate only for very short
    # smoke runs where count unique pre-final integer steps cannot exist.
    positions = [start_step + round(remaining_steps * i / (count + 1)) for i in range(1, count + 1)]
    return sorted({step for step in positions if start_step < step < total_steps})


def checkpoint_steps(total_steps: int, intermediate_count: int = 10) -> list[int]:
    """Return pre-final scheduled positions followed by the separately labelled final."""
    total_steps = int(total_steps)
    if total_steps < 1:
        return []
    return intermediate_checkpoint_steps(total_steps, intermediate_count) + [total_steps]


def checkpoint_hf_dir(checkpoint_root: Path, step: int) -> Path:
    return Path(checkpoint_root) / f"global_step_{int(step)}" / "actor" / "huggingface"


def validate_checkpoint_set(
    checkpoint_root: Path, total_steps: int, intermediate_count: int = 10
) -> tuple[list[tuple[int, Path]], tuple[int, Path]]:
    """Validate scheduled HF directories and return (intermediates, final)."""
    intermediate: list[tuple[int, Path]] = []
    missing: list[str] = []
    for step in intermediate_checkpoint_steps(total_steps, intermediate_count):
        hf_dir = checkpoint_hf_dir(checkpoint_root, step)
        if not hf_dir.is_dir():
            missing.append(str(hf_dir))
        intermediate.append((step, hf_dir))
    final = (int(total_steps), checkpoint_hf_dir(checkpoint_root, total_steps))
    if not final[1].is_dir():
        missing.append(str(final[1]))
    if missing:
        raise FileNotFoundError(
            "Missing scheduled Transformers checkpoint(s):\n  "
            + "\n  ".join(dict.fromkeys(missing))
        )
    return intermediate, final
