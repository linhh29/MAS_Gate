#!/usr/bin/env python3
"""Pad a MAS-Gate parquet to a whole number of training batches.

verl's PPO trainer intentionally uses ``drop_last=True``.  For a strict full-data
round we therefore repeat only the minimum number of rows required to reach a
batch multiple.  Every original row remains present; repeated rows are training
padding, not replacements.  The source parquet is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def pad_parquet(source: Path, output: Path, batch_size: int) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the source parquet")
    table = pq.read_table(source)
    original_rows = table.num_rows
    if original_rows <= 0:
        raise ValueError(f"Empty source parquet: {source}")
    additional_rows = (-original_rows) % batch_size
    if additional_rows:
        # additional_rows < batch_size.  A batch can still exceed the source in
        # tiny tests, so cycle indices rather than assuming a single slice.
        indices = pa.array([i % original_rows for i in range(additional_rows)], type=pa.int64())
        padded = pa.concat_tables([table, table.take(indices)])
    else:
        padded = table

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    pq.write_table(padded, tmp)
    os.replace(tmp, output)
    summary = {
        "source": str(source),
        "output": str(output),
        "batch_size": batch_size,
        "original_rows": original_rows,
        "additional_repeated_rows": additional_rows,
        "padded_rows": padded.num_rows,
        "training_steps_per_epoch": padded.num_rows // batch_size,
        "all_original_rows_retained": True,
        "source_overwritten": False,
    }
    summary_path = output.with_suffix(output.suffix + ".padding.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(pad_parquet(args.input, args.output, args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
