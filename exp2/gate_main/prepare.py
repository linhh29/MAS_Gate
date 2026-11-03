"""Freeze a no-intervention MAS reference for each training task.

Usage: python -m gate_main.prepare --config config.json
"""

from __future__ import annotations

import argparse

from .data import load_config, load_tasks, reference_path, save_reference
from .editor import Graph
from .factory import correctness_from_config, quantifier_from_config, worker_from_config
from .quality import score_trace
from .runtime import execute_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N tasks")
    args = parser.parse_args()
    config = load_config(args.config)
    tasks = load_tasks(config["data_file"])
    if args.limit > 0:
        tasks = tasks[: args.limit]
    pending = [row for row in tasks if not reference_path(config["reference_dir"], str(row["id"])).exists()]
    if not pending:
        print(f"All {len(tasks)} references already exist; training checks their hashes.")
        return
    worker = worker_from_config(config)
    quantifier = quantifier_from_config(config)
    correctness = correctness_from_config(config)
    for index, row in enumerate(pending, 1):
        graph = Graph.from_dict(row["graph"])
        trace = execute_reference(str(row["task"]), graph, worker)
        c, q = score_trace(str(row["task"]), str(row["answer"]),
                           trace.graph, trace.outputs, quantifier, correctness)
        save_reference(reference_path(config["reference_dir"], str(row["id"])),
                       row, trace, c, q)
        print(f"Reference {index}/{len(pending)}: {row['id']}", flush=True)


if __name__ == "__main__":
    main()
