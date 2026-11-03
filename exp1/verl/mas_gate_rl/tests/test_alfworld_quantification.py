from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mas_gate_rl import reward
from mas_gate_rl.data_preprocess import _make_alfworld_row


def test_alfworld_expert_actions_are_reward_only_ground_truth() -> None:
    row = {
        "task_id": "task-1",
        "task_descriptions": ["put the apple away"],
        "expert_actions": ["look", "take apple 1 from table 1", "put apple 1 in fridge 1"],
    }
    converted = _make_alfworld_row(row, 0, "train")
    ground_truth = converted["reward_model"]["ground_truth"]
    prompt_text = json.dumps(converted["prompt"], ensure_ascii=False)
    assert ground_truth == "look\ntake apple 1 from table 1\nput apple 1 in fridge 1"
    assert "take apple 1 from table 1" not in prompt_text
    assert converted["extra_info"]["expert_actions"] == row["expert_actions"]


def test_alfworld_sub_agent_writes_quantification_execution_sidecar(tmp_path: Path) -> None:
    trace_root = tmp_path / "traces"
    trace_path = trace_root / "sample_trace_json" / "trace.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text('{"sub_agents": [], "graph_edges": []}', encoding="utf-8")
    phase_file = tmp_path / "phase.json"
    phase_file.write_text('{"phase":"sub_agent"}', encoding="utf-8")
    original_result = {
        "score": 1.0,
        "judge_score": 1.0,
        "metric_key": "alfworld_environment",
        "trace_json": str(trace_path),
        "environment_result": {"won": 1, "goal_condition_success_rate": 1.0},
        "execution_error": "",
    }
    env = {
        "MAS_GATE_REWARD_TRACE_ROOT": str(trace_root),
        "MAS_GATE_REWARD_PHASE_FILE": str(phase_file),
    }
    info = {"dataset": "alfworld", "index": 3, "task_id": "task-3"}
    with patch.dict(os.environ, env, clear=False), patch.object(
        reward, "_IMPL_COMPUTE_SCORE_ASYNC", AsyncMock(return_value=original_result)
    ):
        result = asyncio.run(reward._compute_score_async("alfworld", "solution", "look", info))
    assert result["score"] == 0.0
    assert result["judge_score"] == 1.0
    sidecar = json.loads(Path(result["execution_sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["status"] == "ok"
    assert sidecar["judge_score"] == 1.0
    assert sidecar["judge_score_available"] is True
    assert sidecar["trace_json"] == str(trace_path)
