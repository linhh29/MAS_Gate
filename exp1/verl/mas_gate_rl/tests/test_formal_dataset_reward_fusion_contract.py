from __future__ import annotations

from pathlib import Path


MAS_GATE_RL = Path(__file__).resolve().parents[1]


def test_formal_training_launchers_fix_reward_fusion_lambda() -> None:
    for dataset in ("hotpotqa", "deepscaler", "alfworld"):
        launcher = (MAS_GATE_RL / f"run_{dataset}_train.sh").read_text(encoding="utf-8")
        assert "export MAS_GATE_REWARD_FUSION_ENABLED=1" in launcher
        assert "export MAS_GATE_REWARD_FUSION_LAMBDA_CORRECT=1.0" in launcher


def test_shared_launcher_accepts_all_formal_fusion_datasets() -> None:
    launcher = (MAS_GATE_RL / "run_mas_gate_grpo.sh").read_text(encoding="utf-8")
    assert "bcp|browsecomp_plus|hotpotqa|hotpot_qa|deepscaler|alfworld" in launcher
    assert "MAS_GATE_REWARD_FUSION_LAMBDA_CORRECT:-1.0" in launcher
