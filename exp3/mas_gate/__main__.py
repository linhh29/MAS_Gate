"""Small, resumable command-line workflow for the paper attribution settings."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .attribution import (
    build_messages, evaluate, load_aegis, load_who_when, parse_prediction, rank_candidates,
)
from .quantification import finalize_scores, score_utility


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_once(path: Path, value: Any) -> None:
    """Publish a complete file atomically; never replace an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


@contextmanager
def output_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is using this output directory") from exc
        yield


def checkpoint_identity(path: str) -> dict[str, Any]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError("model must be an existing local checkpoint directory")
    metadata = {}
    for pattern in ("*.json", "*.jinja", "*.model", "*.tiktoken", "*.txt"):
        for item in sorted(root.glob(pattern)):
            metadata[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
    weights = {
        item.name: {"bytes": item.stat().st_size, "modified_ns": item.stat().st_mtime_ns}
        for pattern in ("*.safetensors", "*.bin", "*.pt") for item in sorted(root.glob(pattern))
    }
    if not weights:
        raise ValueError("no local model weight files found")
    return {"name": root.name, "metadata_sha256": metadata, "weight_files": weights}


def prepare_manifest(directory: Path, config: dict[str, Any]) -> str:
    config = json.loads(json.dumps(config, allow_nan=False))
    fingerprint = digest(config)
    manifest = {"config": config, "fingerprint": fingerprint}
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("output configuration differs; select a new output directory")
    else:
        write_once(path, manifest)
    return fingerprint


def record_path(directory: Path, identifier: str) -> Path:
    return directory / "records" / (digest(identifier) + ".json")


def load_matching(path: Path, fingerprint: str, example: Any) -> dict[str, Any]:
    record = read_json(path)
    if (record.get("fingerprint") != fingerprint or record.get("id") != example.id
            or record.get("example_sha256") != digest(asdict(example))):
        raise ValueError("saved record does not match the requested data/configuration")
    return record


def source_record(directory: Path, example: Any, kind: str) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    if manifest["config"]["kind"] != kind or digest(manifest["config"]) != manifest["fingerprint"]:
        raise ValueError("invalid source manifest")
    return load_matching(record_path(directory, example.id), manifest["fingerprint"], example)


def run(args: argparse.Namespace) -> None:
    pairs = (load_who_when(args.data) if args.benchmark == "who_when" else load_aegis(args.data))
    if args.limit:
        pairs = pairs[:args.limit]
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    implementation = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(Path(__file__).parent.glob("*.py"))}
    common = {"benchmark": args.benchmark, "count": len(pairs), "schema": 1,
              "implementation_sha256": implementation,
              "dataset_sha256": digest([(asdict(e), asdict(label)) for e, label in pairs])}
    output = Path(args.output)
    if args.command in {"utility", "risk"}:
        config = {**common, "kind": args.command, "checkpoint": checkpoint_identity(args.model)}
        if args.command == "utility":
            config.update(max_model_len=args.max_model_len, memory=args.memory)
        else:
            config.update(nli_max_length=8192, batch_size=32,
                          utility_fingerprint=read_json(Path(args.utility) / "manifest.json")["fingerprint"])
        with output_lock(output), ExitStack() as resources:
            fingerprint = prepare_manifest(output, config)
            model = None
            for number, (example, _) in enumerate(pairs, 1):
                path = record_path(output, example.id)
                if path.exists():
                    load_matching(path, fingerprint, example)
                    continue
                if model is None:
                    from .models import AnswerScorer, NLI
                    model = (AnswerScorer(args.model, max_model_len=args.max_model_len,
                                          gpu_memory_utilization=args.memory)
                             if args.command == "utility" else NLI(args.model, device=args.device))
                    resources.callback(model.close)
                if args.command == "utility":
                    record = score_utility(example, model)
                else:
                    state = source_record(Path(args.utility), example, "utility")
                    edges = [(a.content, b.content) for a, b in zip(example.steps, example.steps[1:])]
                    record = finalize_scores(example, state, model.probabilities(edges))
                write_once(path, {**record, "fingerprint": fingerprint,
                                  "example_sha256": digest(asdict(example))})
                print(f"{args.command}: {number}/{len(pairs)}", flush=True)
        return

    settings = []
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("arms must be distinct")
    if "direct" in args.arms:
        settings.append(("direct", None))
    values = args.k or ([8] if args.benchmark == "who_when" else [4])
    if any(k < 1 for k in values) or len(set(values)) != len(values):
        raise ValueError("k values must be distinct positive integers")
    if args.benchmark == "aegis" and any(a in args.arms for a in ("random", "shuffled")):
        raise ValueError("the reported random/shuffled controls are Who&When only")
    for arm in args.arms:
        if arm != "direct":
            settings.extend((arm, k) for k in values)
    if any(arm != "direct" for arm, _ in settings) and not args.scores:
        raise ValueError("guided arms require --scores")
    config = {**common, "kind": "localization", "checkpoint": checkpoint_identity(args.model),
              "profile": args.profile, "temperature": 0, "max_tokens": 256,
              "max_model_len": args.max_model_len, "memory": args.memory,
              "settings": settings, "engine_seed": 0, "control_seed": 20260923}
    if args.scores:
        config["scores_fingerprint"] = read_json(Path(args.scores) / "manifest.json")["fingerprint"]
    with output_lock(output), ExitStack() as resources:
        fingerprint = prepare_manifest(output, config)
        model = None
        for arm, k in settings:
            name = arm if k is None else f"{arm}_top{k}"
            directory = output / name
            for number, (example, label) in enumerate(pairs, 1):
                path = record_path(directory, example.id)
                if path.exists():
                    load_matching(path, fingerprint, example)
                    continue
                candidates = None
                if arm != "direct":
                    scores = source_record(Path(args.scores), example, "risk")
                    risks = {row["step"]: row["risk_score"] for row in scores["nodes"]}
                    candidates = rank_candidates(example, risks, k, variant=arm)
                messages = build_messages(example, candidates)
                if model is None:
                    from .models import Localizer
                    model = Localizer(args.model, args.profile, max_model_len=args.max_model_len,
                                      gpu_memory_utilization=args.memory)
                    resources.callback(model.close)
                response = model.generate(messages)
                prediction = parse_prediction(response["text"], example)
                record = {"id": example.id, "fingerprint": fingerprint,
                          "example_sha256": digest(asdict(example)), "arm": arm, "k": k,
                          "candidates": candidates, "messages": messages, "response": response,
                          "prediction": prediction, "metrics": evaluate(prediction, label, args.benchmark)}
                write_once(path, record)
                print(f"{name}: {number}/{len(pairs)}", flush=True)
        summarize(output)


def summarize(directory: Path) -> None:
    manifest = read_json(directory / "manifest.json")
    config = manifest["config"]
    if config["kind"] != "localization":
        raise ValueError("summary requires a localization run")
    metric = "exact_step" if config["benchmark"] == "who_when" else "agent_hit"
    for arm, k in config["settings"]:
        name = arm if k is None else f"{arm}_top{k}"
        records = [read_json(p) for p in sorted((directory / name / "records").glob("*.json"))]
        if any(r["fingerprint"] != manifest["fingerprint"] for r in records):
            raise ValueError("mixed configurations in result directory")
        correct = sum(bool(r["metrics"][metric]) for r in records)
        invalid = sum(not r["metrics"]["parse_ok"] for r in records)
        count = config["count"]
        status = "complete" if len(records) == count else "incomplete"
        # Use the expected denominator so missing/failed outputs cannot inflate accuracy.
        print(f"{name}: {len(records)}/{count} records, {correct}/{count} correct "
              f"({100 * correct / count:.2f}%), {invalid} invalid, {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("utility", "risk", "localize"):
        child = subparsers.add_parser(command)
        child.add_argument("--benchmark", choices=("who_when", "aegis"), required=True)
        child.add_argument("--data", required=True)
        child.add_argument("--model", required=True, help="existing local checkpoint directory")
        child.add_argument("--output", required=True)
        child.add_argument("--limit", type=int, help="smoke test: process only the first N selected records")
        if command != "risk":
            child.add_argument("--max-model-len", type=int, default=81920)
            child.add_argument("--memory", type=float, default=0.9)
        if command == "risk":
            child.add_argument("--utility", required=True)
            child.add_argument("--device", default="cuda:0")
        if command == "localize":
            child.add_argument("--profile", choices=("qwen30", "qwen35_4b", "qwen36_27b", "mistral24", "llama8"), required=True)
            child.add_argument("--scores")
            child.add_argument("--arms", nargs="+", choices=("direct", "mas_gate", "random", "shuffled"), default=["direct", "mas_gate"])
            child.add_argument("--k", nargs="+", type=int)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("output", type=Path)
    data_check = subparsers.add_parser("check-data", help="validate and convert raw data in memory; no model needed")
    data_check.add_argument("--benchmark", choices=("who_when", "aegis"), required=True)
    data_check.add_argument("--data", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.output)
    elif args.command == "check-data":
        pairs = load_who_when(args.data) if args.benchmark == "who_when" else load_aegis(args.data)
        print(f"Validated {len(pairs)} {args.benchmark} examples; "
              f"{sum(len(example.steps) for example, _ in pairs)} trajectory steps.")
        print("Canonical input: Example(id, question, answer, steps, observed_final_output, benchmark)")
        print("Evaluation only: Label(agent, step); attribution annotations are not prompt inputs.")
    else:
        run(args)


if __name__ == "__main__":
    main()
