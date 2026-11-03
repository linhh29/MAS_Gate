"""Freeze one Orchestra graph per task before reference collection.

Input JSONL has id, task, answer. The answer is copied to training data but
never sent to the planner. Replace the default planner for domain workflows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .backends import load_symbol
from .data import load_config
from .editor import Graph


PLANNER_PROMPT = """Plan a small directed acyclic multi-agent workflow for the task.
Return only JSON with key "nodes" (a list). Each node has: id (unique string),
role (worker instruction), task (local milestone), parents (list of node IDs),
constraints (list of strings), is_final (boolean), replay_safe (boolean).
There must be exactly one final synthesis node. Do not answer the task.

TASK:\n"""


class OpenAIPlanner:
    """Generic frozen planner for text tasks; domain planners can be adapters."""

    def __init__(self, config: dict):
        from openai import OpenAI

        self.model = config["model"]
        self.client = OpenAI(
            base_url=config["base_url"],
            api_key=os.environ.get(config.get("api_key_env", "PLANNER_API_KEY"), "EMPTY"),
            timeout=float(config.get("timeout", 600)),
        )
        self.max_tokens = int(config.get("max_tokens", 2048))

    def plan(self, task: str) -> Graph:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": PLANNER_PROMPT + task}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Orchestra did not return a JSON workflow")
        return Graph.from_dict(json.loads(raw[start : end + 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Raw JSONL: id, task, answer")
    parser.add_argument("--output", required=True, help="Frozen training JSONL")
    args = parser.parse_args()
    config = load_config(args.config)
    options = config.get("planner")
    if not options:
        raise ValueError("config.planner is required for workflow planning")
    planner = (load_symbol(options["adapter"])(config=options)
               if options.get("adapter") else OpenAIPlanner(options))
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    seen: set[str] = set()
    try:
        with Path(args.input).open(encoding="utf-8") as source, temporary.open(
            "x", encoding="utf-8"
        ) as target:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not all(row.get(key) for key in ("id", "task", "answer")):
                    raise ValueError(f"row {number} needs id, task, answer")
                task_id = str(row["id"])
                if task_id in seen:
                    raise ValueError(f"duplicate task ID: {task_id}")
                seen.add(task_id)
                graph = planner.plan(str(row["task"]))
                target.write(json.dumps({
                    "id": task_id, "task": row["task"], "answer": row["answer"],
                    "graph": graph.as_dict(),
                }, ensure_ascii=False) + "\n")
                print(f"Planned {number}: {task_id}", flush=True)
        if not seen:
            raise ValueError("input has no tasks")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
