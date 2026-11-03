#!/usr/bin/env python3
"""Analyze alignment between BCP mean-impact reward and executed MAS structure."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

FEATURES = (
    "agents",
    "edges",
    "search_sessions",
    "retrieval_rounds",
    "max_retrieval_round",
    "total_tokens",
    "output_chars",
    "new_docs",
    "retrieved_docs",
    "solution_length",
    "correctness",
)
QUARTILE_FEATURES = (
    "agents",
    "edges",
    "search_sessions",
    "retrieval_rounds",
    "total_tokens",
    "output_chars",
    "new_docs",
    "correctness",
)


def parse_datetime(value: str, *, field: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit UTC offset")
    return parsed


def ensure_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    boundary = root.expanduser().resolve(strict=True)
    if not resolved.is_relative_to(boundary):
        raise ValueError(f"{label} escapes checkpoint root: {resolved}")
    return resolved


def pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right, strict=True)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs, strict=True)
    x_mean, y_mean = fmean(xs), fmean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0 or y_var <= 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / math.sqrt(x_var * y_var)


def ranks(values: Iterable[float]) -> list[float]:
    numbers = [float(value) for value in values]
    order = sorted(range(len(numbers)), key=numbers.__getitem__)
    result = [0.0] * len(numbers)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and numbers[order[end]] == numbers[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = average_rank
        start = end
    return result


def spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    xs, ys = list(left), list(right)
    return pearson(ranks(xs), ranks(ys))


def parse_step_end(value: str) -> tuple[int, datetime]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--step-end must use STEP=ISO_DATETIME")
    raw_step, raw_datetime = value.split("=", 1)
    try:
        step = int(raw_step)
        boundary = parse_datetime(raw_datetime, field="--step-end")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if step <= 0:
        raise argparse.ArgumentTypeError("step must be positive")
    return step, boundary


def assign_step(timestamp: float, step_ends: list[tuple[int, datetime]]) -> int | None:
    for step, boundary in step_ends:
        if timestamp <= boundary.timestamp():
            return step
    return None


def extract_row(sidecar: dict[str, Any], trace: dict[str, Any], *, timestamp: float, step: int) -> dict[str, Any]:
    result = sidecar.get("result") or {}
    sessions = trace.get("retrieval_sessions") or []
    trajectories = [session.get("trajectory") or [] for session in sessions]
    sub_agents = trace.get("sub_agents") or []
    return {
        "timestamp": timestamp,
        "step": step,
        "index": str(sidecar.get("index")),
        "mean_impact": float(result["score"]),
        "correctness": float(result.get("judge_score") or 0.0),
        "agents": len(sub_agents),
        "edges": len(trace.get("graph_edges") or []),
        "search_sessions": len(sessions),
        "retrieval_rounds": sum(len(items) for items in trajectories),
        "max_retrieval_round": max((len(items) for items in trajectories), default=0),
        "total_tokens": float((trace.get("usage_totals") or {}).get("total_tokens") or 0.0),
        "output_chars": sum(len(str(agent.get("output") or "")) for agent in sub_agents),
        "new_docs": sum(
            len(item.get("new_docids_first_seen_this_round") or [])
            for trajectory in trajectories
            for item in trajectory
        ),
        "retrieved_docs": sum(
            len(item.get("retrieved_docids") or [])
            for trajectory in trajectories
            for item in trajectory
        ),
        "solution_length": float(sidecar.get("solution_length") or 0.0),
    }


def load_rows(
    checkpoint_root: Path,
    *,
    not_before: datetime,
    not_after: datetime,
    step_ends: list[tuple[int, datetime]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    root = checkpoint_root.expanduser().resolve(strict=True)
    sidecar_root = root / "reward_traces" / "reward_results"
    if not sidecar_root.is_dir():
        raise NotADirectoryError(sidecar_root)
    rows: list[dict[str, Any]] = []
    counters: defaultdict[str, int] = defaultdict(int)
    lower, upper = not_before.timestamp(), not_after.timestamp()
    for path in sorted(sidecar_root.glob("*.json")):
        counters["sidecars_seen"] += 1
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
            timestamp = float(sidecar.get("generated_at_unix") or 0.0)
            if not (lower <= timestamp < upper):
                counters["outside_time_window"] += 1
                continue
            step = assign_step(timestamp, step_ends)
            if step is None:
                counters["after_last_step_end"] += 1
                continue
            raw_trace = Path(str((sidecar.get("result") or {}).get("trace_json") or ""))
            trace_path = ensure_within(raw_trace, root, "trace_json")
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            rows.append(extract_row(sidecar, trace, timestamp=timestamp, step=step))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            counters["invalid_sidecars"] += 1
    counters["rows_loaded"] = len(rows)
    return rows, dict(counters)


def mean_delta(high: list[dict[str, Any]], low: list[dict[str, Any]], feature: str) -> float:
    return fmean(float(row[feature]) for row in high) - fmean(float(row[feature]) for row in low)


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no reward/trace rows matched the requested evidence window")
    global_correlations = {
        feature: {
            "pearson": pearson((row["mean_impact"] for row in rows), (row[feature] for row in rows)),
            "spearman": spearman((row["mean_impact"] for row in rows), (row[feature] for row in rows)),
        }
        for feature in FEATURES
    }

    groups: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["step"]), str(row["index"]))].append(row)

    centered: list[dict[str, float]] = []
    for members in groups.values():
        means = {key: fmean(float(row[key]) for row in members) for key in ("mean_impact", *FEATURES)}
        for row in members:
            centered.append({key: float(row[key]) - means[key] for key in means})
    within_prompt = {
        feature: pearson(
            (row["mean_impact"] for row in centered),
            (row[feature] for row in centered),
        )
        for feature in FEATURES
    }

    per_step: dict[str, Any] = {}
    for step in sorted({int(row["step"]) for row in rows}):
        step_rows = [row for row in rows if int(row["step"]) == step]
        step_centered: list[dict[str, float]] = []
        for (group_step, _), members in groups.items():
            if group_step != step:
                continue
            means = {key: fmean(float(row[key]) for row in members) for key in ("mean_impact", *FEATURES)}
            step_centered.extend(
                {key: float(row[key]) - means[key] for key in means}
                for row in members
            )

        mixed_deltas: defaultdict[str, list[float]] = defaultdict(list)
        quartile_deltas: defaultdict[str, list[float]] = defaultdict(list)
        mixed_count = 0
        quartile_count = 0
        for (group_step, _), members in groups.items():
            if group_step != step:
                continue
            correct = [row for row in members if float(row["correctness"]) >= 0.5]
            wrong = [row for row in members if float(row["correctness"]) < 0.5]
            if correct and wrong:
                mixed_count += 1
                for feature in ("mean_impact", *QUARTILE_FEATURES[:-1]):
                    mixed_deltas[feature].append(mean_delta(correct, wrong, feature))
            if len(members) >= 4:
                ordered = sorted(members, key=lambda row: float(row["mean_impact"]))
                size = max(1, len(ordered) // 4)
                quartile_count += 1
                for feature in QUARTILE_FEATURES:
                    quartile_deltas[feature].append(mean_delta(ordered[-size:], ordered[:size], feature))

        per_step[str(step)] = {
            "rows": len(step_rows),
            "groups": sum(group_step == step for group_step, _ in groups),
            "within_prompt_pearson": {
                feature: pearson(
                    (row["mean_impact"] for row in step_centered),
                    (row[feature] for row in step_centered),
                )
                for feature in FEATURES
            },
            "mixed_groups": mixed_count,
            "correct_minus_wrong": {
                feature: fmean(values) for feature, values in mixed_deltas.items() if values
            },
            "quartile_groups": quartile_count,
            "high_impact_minus_low_impact": {
                feature: fmean(values) for feature, values in quartile_deltas.items() if values
            },
        }

    return {
        "schema_version": "mas_gate_bcp_reward_structure/1",
        "rows": len(rows),
        "unique_prompt_indices": len({str(row["index"]) for row in rows}),
        "prompt_step_groups": len(groups),
        "global_correlations": global_correlations,
        "within_prompt_pearson": within_prompt,
        "per_step": per_step,
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def render_markdown(summary: dict[str, Any], counters: dict[str, int]) -> str:
    lines = [
        "# BCP mean-impact 与真实 MAS 结构分析",
        "",
        f"- 有效 rollout：{summary['rows']}；唯一 prompt index：{summary['unique_prompt_indices']}。",
        f"- 读取计数：`{json.dumps(counters, ensure_ascii=False, sort_keys=True)}`。",
        "- 相关性只描述 reward 与行为的伴随关系，不解释为因果。",
        "",
        "## 全局与题内相关性",
        "",
        "| Feature | Global Pearson | Global Spearman | Within-prompt Pearson |",
        "|---|---:|---:|---:|",
    ]
    for feature in FEATURES:
        global_row = summary["global_correlations"][feature]
        values = (global_row["pearson"], global_row["spearman"], summary["within_prompt_pearson"][feature])
        formatted = ["NA" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {feature} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")
    for step, item in summary["per_step"].items():
        lines.extend([
            "",
            f"## Step {step}",
            "",
            f"- rows={item['rows']}，groups={item['groups']}，mixed_groups={item['mixed_groups']}。",
            "- High-impact top quartile minus low-impact bottom quartile:",
            "",
            "| Feature | Mean difference |",
            "|---|---:|",
        ])
        for feature, value in item["high_impact_minus_low_impact"].items():
            lines.append(f"| {feature} | {value:+.4f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--not-before", required=True)
    parser.add_argument("--not-after", required=True)
    parser.add_argument("--step-end", action="append", type=parse_step_end, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    not_before = parse_datetime(args.not_before, field="--not-before")
    not_after = parse_datetime(args.not_after, field="--not-after")
    if not_before >= not_after:
        raise ValueError("--not-before must be earlier than --not-after")
    step_ends = sorted(args.step_end, key=lambda item: item[0])
    if [step for step, _ in step_ends] != list(range(1, len(step_ends) + 1)):
        raise ValueError("--step-end values must define contiguous steps starting at 1")
    if any(step_ends[i][1] >= step_ends[i + 1][1] for i in range(len(step_ends) - 1)):
        raise ValueError("--step-end timestamps must be strictly increasing")
    if step_ends[-1][1] >= not_after:
        raise ValueError("last --step-end must be earlier than --not-after")

    rows, counters = load_rows(
        args.checkpoint_root,
        not_before=not_before,
        not_after=not_after,
        step_ends=step_ends,
    )
    summary = analyze(rows)
    summary["evidence_window"] = {
        "not_before": not_before.isoformat(),
        "not_after": not_after.isoformat(),
        "step_ends": {str(step): boundary.isoformat() for step, boundary in step_ends},
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / "reward_structure_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output / "reward_structure_rows.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    atomic_write(output / "reward_structure_report.md", render_markdown(summary, counters))
    print(json.dumps({"output_dir": str(output), **counters, "summary_rows": summary["rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
