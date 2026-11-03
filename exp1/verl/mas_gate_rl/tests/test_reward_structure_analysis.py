import json
from datetime import datetime
from pathlib import Path

import pytest

from mas_gate_rl.analyze_bcp_reward_structure import analyze, ensure_within, load_rows, parse_datetime


def test_multiline_sidecar_cutoff_and_trace_boundary(tmp_path: Path) -> None:
    root = tmp_path / "run"
    sidecars = root / "reward_traces/reward_results"
    traces = root / "reward_traces/traces"
    sidecars.mkdir(parents=True)
    traces.mkdir(parents=True)
    trace = traces / "trace.json"
    trace.write_text(json.dumps({"sub_agents": [], "retrieval_sessions": []}), encoding="utf-8")
    timestamp = datetime.fromisoformat("2026-08-01T12:00:00+08:00").timestamp()
    sidecar = {
        "generated_at_unix": timestamp,
        "index": 7,
        "solution_length": 10,
        "result": {"score": 0.4, "judge_score": 1, "trace_json": str(trace)},
    }
    (sidecars / "sample.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    rows, counters = load_rows(
        root,
        not_before=parse_datetime("2026-08-01T00:00:00+08:00", field="start"),
        not_after=parse_datetime("2026-08-02T00:00:00+08:00", field="end"),
        step_ends=[(1, parse_datetime("2026-08-01T13:00:00+08:00", field="step"))],
    )
    assert counters["rows_loaded"] == 1
    assert rows[0]["index"] == "7"
    assert rows[0]["step"] == 1


def test_trace_path_cannot_escape_checkpoint_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes checkpoint root"):
        ensure_within(outside, root, "trace_json")


def test_analysis_computes_prompt_centered_and_quartile_metrics() -> None:
    rows = []
    for prompt in ("a", "b"):
        for value in range(4):
            rows.append(
                {
                    "step": 1,
                    "index": prompt,
                    "mean_impact": float(value),
                    "correctness": float(value >= 2),
                    "agents": value + 1,
                    "edges": value,
                    "search_sessions": value,
                    "retrieval_rounds": value,
                    "max_retrieval_round": value,
                    "total_tokens": value * 10,
                    "output_chars": value * 20,
                    "new_docs": value,
                    "retrieved_docs": value * 2,
                    "solution_length": value * 3,
                }
            )
    summary = analyze(rows)
    assert summary["rows"] == 8
    assert summary["within_prompt_pearson"]["agents"] == pytest.approx(1.0)
    assert summary["per_step"]["1"]["mixed_groups"] == 2
    assert summary["per_step"]["1"]["high_impact_minus_low_impact"]["agents"] == 3
