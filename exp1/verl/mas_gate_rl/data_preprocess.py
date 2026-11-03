#!/usr/bin/env python3
"""Build MAS-Gate orchestra-RL parquet data from local datasets.

Supports AIME, BrowseComp-Plus, DeepScaleR, HotpotQA, and ALFWorld. This file intentionally lives in
``verl/mas_gate_rl`` so MAS-Gate-specific verl integration is centralized.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import copy
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERL_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _VERL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mas_gate_rl.xml_policy import ORCHESTRA_XML_SYSTEM_PROMPT, build_orchestra_user_prompt


def load_rows(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    """Load either JSONL or a JSON array of objects."""
    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            obj = obj.get("data", obj.get("rows", []))
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON array/list in {path}, got {type(obj).__name__}")
        for row in obj:
            if not isinstance(row, dict):
                continue
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
        return rows

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def _norm_dataset(name: str) -> str:
    s = (name or "").strip().lower()
    if s in {"aime", "hlemath"}:
        return "aime"
    if s in {"bcp", "browsecomp", "browsecomp_plus", "mas_gate_bcp"}:
        return "bcp"
    if s in {"deepscaler", "deep_scale_r", "deepscaler_math", "mas_gate_deepscaler"}:
        return "deepscaler"
    if s in {"hotpotqa", "hotpot_qa", "mas_gate_hotpotqa"}:
        return "hotpotqa"
    if s in {"alfworld", "alf_world", "mas_gate_alfworld"}:
        return "alfworld"
    raise ValueError(f"Unsupported --dataset={name!r}; expected aime, bcp, deepscaler, hotpotqa or alfworld")


def _default_jsonl(dataset: str) -> Path:
    if dataset == "bcp":
        return _REPO_ROOT / "datasets" / "BrowseComp-Plus" / "data" / "browsecomp_plus_train.jsonl"
    if dataset == "deepscaler":
        return _REPO_ROOT / "datasets" / "DeepScaleR" / "deepscaler_train.json"
    if dataset == "hotpotqa":
        return _REPO_ROOT / "datasets" / "BrowseComp-Plus" / "hotpotqa" / "data" / "hotpotqa_train.jsonl"
    if dataset == "alfworld":
        return _REPO_ROOT / "datasets" / "alfworld" / "data" / "alfworld_train.jsonl"
    return _REPO_ROOT / "datasets" / "aime" / "data" / "aime_test.jsonl"


def _default_save_dir(dataset: str) -> Path:
    if dataset == "bcp":
        name = "mas_gate_bcp"
    elif dataset == "deepscaler":
        name = "mas_gate_deepscaler"
    elif dataset == "hotpotqa":
        name = "mas_gate_hotpotqa"
    elif dataset == "alfworld":
        name = "mas_gate_alfworld"
    else:
        name = "mas_gate_aime"
    return _VERL_ROOT / "data" / name


def _make_aime_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    q = str(row.get("question", "")).strip()
    ans = str(row.get("answer", "")).strip()
    tid = str(row.get("id", idx))
    return {
        "data_source": "mas_gate_aime",
        "prompt": [
            {"role": "system", "content": ORCHESTRA_XML_SYSTEM_PROMPT},
            {"role": "user", "content": build_orchestra_user_prompt(q, "aime")},
        ],
        "ability": "mas_orchestra_aime",
        "reward_model": {"style": "rule", "ground_truth": ans},
        "extra_info": {"split": split, "index": idx, "idx": idx, "task_id": tid, "dataset": "aime", "question": q, "answer": ans},
    }


def _make_bcp_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    q = str(row.get("query", row.get("question", ""))).strip()
    ans = str(row.get("answer", "")).strip()
    tid = str(row.get("query_id", row.get("id", idx)))
    gold_docids: list[str] = []
    for d in row.get("gold_docs", []) or []:
        if isinstance(d, dict) and d.get("docid") is not None:
            gold_docids.append(str(d.get("docid")))
    return {
        "data_source": "mas_gate_bcp",
        "prompt": [
            {"role": "system", "content": ORCHESTRA_XML_SYSTEM_PROMPT},
            {"role": "user", "content": build_orchestra_user_prompt(q, "browsecomp_plus")},
        ],
        "ability": "mas_orchestra_bcp",
        "reward_model": {"style": "rule", "ground_truth": ans},
        "extra_info": {
            "split": split,
            "index": idx,
            "idx": idx,
            "task_id": tid,
            "dataset": "browsecomp_plus",
            "question": q,
            "query": q,
            "answer": ans,
            "gold_docids": gold_docids,
        },
    }



def _make_deepscaler_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    q = str(row.get("problem", row.get("question", ""))).strip()
    ans = str(row.get("answer", "")).strip()
    sol = str(row.get("solution", "")).strip()
    tid = str(row.get("id", row.get("problem_id", idx)))
    return {
        "data_source": "mas_gate_deepscaler",
        "prompt": [
            {"role": "system", "content": ORCHESTRA_XML_SYSTEM_PROMPT},
            {"role": "user", "content": build_orchestra_user_prompt(q, "deepscaler")},
        ],
        "ability": "mas_orchestra_deepscaler",
        "reward_model": {"style": "rule", "ground_truth": ans},
        "extra_info": {
            "split": split,
            "index": idx,
            "idx": idx,
            "task_id": tid,
            "dataset": "deepscaler",
            "question": q,
            "problem": q,
            "answer": ans,
            "reference_solution": sol,
        },
    }


def _make_hotpotqa_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    q = str(row.get("query", row.get("question", ""))).strip()
    ans = str(row.get("answer", "")).strip()
    tid = str(row.get("query_id", row.get("id", idx)))
    return {
        "data_source": "mas_gate_hotpotqa",
        "prompt": [
            {"role": "system", "content": ORCHESTRA_XML_SYSTEM_PROMPT},
            {"role": "user", "content": build_orchestra_user_prompt(q, "hotpotqa")},
        ],
        "ability": "mas_orchestra_hotpotqa",
        "reward_model": {"style": "rule", "ground_truth": ans},
        "extra_info": {
            "split": split, "index": idx, "idx": idx, "task_id": tid,
            "dataset": "hotpotqa", "question": q, "query": q, "answer": ans,
            "type": str(row.get("type", "")), "level": str(row.get("level", "")),
            # Trace metadata only; supporting facts are never included in the prompt.
            "supporting_facts": row.get("supporting_facts", []) or [],
        },
    }


def _make_alfworld_row(row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    descriptions = row.get("task_descriptions") or []
    task = str(descriptions[0] if descriptions else row.get("question", row.get("task", ""))).strip()
    task_id = str(row.get("task_id", row.get("id", idx)))
    raw_expert_actions = row.get("expert_actions") or []
    if not isinstance(raw_expert_actions, list):
        raise ValueError(f"ALFWorld expert_actions must be a list for task_id={task_id}")
    expert_actions = [str(action).strip() for action in raw_expert_actions if str(action).strip()]
    if not expert_actions:
        raise ValueError(f"ALFWorld expert_actions are required for quantification task_id={task_id}")
    # The teacher trajectory is reward-only Qwen-J ground truth.  It is never
    # inserted into the policy prompt, preventing expert-action leakage.
    quantification_ground_truth = "\n".join(expert_actions)
    preserved_keys = (
        "id", "task_id", "game_file", "env_step_limit", "task_type",
        "official_split", "floor_plan", "traj_file", "task_descriptions",
    )
    extra = {key: copy.deepcopy(row[key]) for key in preserved_keys if key in row}
    extra.update({
        "split": split,
        "index": idx,
        "idx": idx,
        "task_id": task_id,
        "dataset": "alfworld",
        "question": task,
        "interactive_environment": True,
        "expert_actions": expert_actions,
    })
    return {
        "data_source": "mas_gate_alfworld",
        "prompt": [
            {"role": "system", "content": ORCHESTRA_XML_SYSTEM_PROMPT},
            {"role": "user", "content": build_orchestra_user_prompt(task, "alfworld")},
        ],
        "ability": "mas_orchestra_alfworld",
        "reward_model": {"style": "rule", "ground_truth": quantification_ground_truth},
        "extra_info": extra,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aime", choices=["aime", "bcp", "browsecomp_plus", "deepscaler", "hotpotqa", "hotpot_qa", "alfworld", "alf_world"])
    ap.add_argument("--jsonl", type=Path, default=None)
    ap.add_argument("--local_save_dir", type=Path, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--pad-to-multiple", type=int, default=0)
    args = ap.parse_args()

    dataset = _norm_dataset(args.dataset)
    jsonl = args.jsonl or _default_jsonl(dataset)
    save_dir = args.local_save_dir or _default_save_dir(dataset)

    rows = load_rows(jsonl, max_rows=args.max_samples if args.max_samples and args.max_samples > 0 else 0)
    if not rows:
        raise RuntimeError(f"No rows in {jsonl}")

    if args.val_ratio <= 0:
        # Full-train mode: keep every input row in train.  We still create a tiny
        # validation parquet for verl config compatibility, but test_freq can be -1.
        train_rows = rows
        val_rows = rows[-1:]
    else:
        n_val = max(1, int(len(rows) * args.val_ratio)) if len(rows) > 1 else 1
        train_rows = rows[:-n_val] if len(rows) > 1 else rows
        val_rows = rows[-n_val:]

    makers = {"aime": _make_aime_row, "bcp": _make_bcp_row, "deepscaler": _make_deepscaler_row, "hotpotqa": _make_hotpotqa_row, "alfworld": _make_alfworld_row}
    maker = makers[dataset]
    train = [maker(r, i, "train") for i, r in enumerate(train_rows)]
    original_train_len = len(train)
    if args.pad_to_multiple < 0:
        raise ValueError("--pad-to-multiple must be non-negative")
    if args.pad_to_multiple and train:
        remainder = len(train) % args.pad_to_multiple
        if remainder:
            needed = args.pad_to_multiple - remainder
            for duplicate_seq in range(needed):
                duplicate = copy.deepcopy(train[duplicate_seq % original_train_len])
                extra = duplicate.setdefault("extra_info", {})
                extra["is_padding_duplicate"] = True
                extra["padding_duplicate_of"] = extra.get("index")
                extra["padding_duplicate_seq"] = duplicate_seq
                extra["original_train_len"] = original_train_len
                train.append(duplicate)
    val = [maker(r, len(train_rows) + i, "test") for i, r in enumerate(val_rows)]

    save_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), save_dir / "train.parquet")
    pq.write_table(pa.Table.from_pylist(val), save_dir / "test.parquet")
    print(json.dumps({"dataset": dataset, "jsonl": str(jsonl), "source_train": original_train_len, "train": len(train), "pad_to_multiple": args.pad_to_multiple, "test": len(val), "out": str(save_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
