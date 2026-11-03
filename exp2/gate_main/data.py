"""Small, explicit data contract for frozen workflows and reference traces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .editor import Graph
from .runtime import Trace


def load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("data_file", "reference_dir", "output_dir", "worker", "quantifier", "training"):
        if key not in value:
            raise ValueError(f"missing config field: {key}")
    return value


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for line_number, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(row.get(key) for key in ("id", "task", "answer", "graph")):
            raise ValueError(f"row {line_number} needs id, task, answer, graph")
        task_id = str(row["id"])
        if task_id in ids:
            raise ValueError(f"duplicate task ID: {task_id}")
        ids.add(task_id)
        Graph.from_dict(row["graph"])
        rows.append(row)
    if not rows:
        raise ValueError("training data has no tasks")
    return rows


def task_hash(row: dict[str, Any]) -> str:
    source = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def reference_path(root: str | Path, task_id: str) -> Path:
    stem = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:24]
    return Path(root) / f"{stem}.json"


def save_reference(path: Path, row: dict[str, Any], trace: Trace, c: float, q: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    value = {
        "schema": "gate_reference_v1",
        "id": str(row["id"]),
        "task_hash": task_hash(row),
        "graph": trace.graph.as_dict(),
        "outputs": trace.outputs,
        "prompts": trace.prompts,
        "correctness": c,
        "quality": q,
    }
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_reference(root: str | Path, row: dict[str, Any]) -> tuple[Trace, float, float]:
    value = json.loads(reference_path(root, str(row["id"])).read_text(encoding="utf-8"))
    if value.get("schema") != "gate_reference_v1" or value.get("task_hash") != task_hash(row):
        raise ValueError(f"reference identity/hash mismatch for task {row['id']}")
    graph = Graph.from_dict(value["graph"])
    trace = Trace(str(row["task"]), graph, value["outputs"], value["prompts"])
    if set(trace.outputs) != graph.nodes.keys() or set(trace.prompts) != graph.nodes.keys():
        raise ValueError(f"incomplete reference trace for task {row['id']}")
    return trace, float(value["correctness"]), float(value["quality"])
