#!/usr/bin/env python3
"""Validate a local HuggingFace Orchestra model for vLLM evaluation."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def validate_hf_model(path: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Orchestra model directory does not exist: {p}")
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json under Orchestra model: {p}")
    if not list(p.glob("*.safetensors")):
        raise FileNotFoundError(f"No Transformers safetensors found under Orchestra model: {p}")
    if not any((p / x).exists() for x in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")):
        raise FileNotFoundError(f"No tokenizer files found under Orchestra model: {p}")
    return str(p)


def resolve_model(*, model_path: str = "") -> dict[str, str]:
    model_path = (model_path or os.environ.get("ORCHESTRA_MODEL_PATH", "")).strip()
    if not model_path:
        raise RuntimeError("No local Orchestra model. Set ORCHESTRA_MODEL_PATH or pass --model-path.")
    resolved = validate_hf_model(model_path)
    return {"source": "local", "source_id": resolved, "model_path": resolved}


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a local MAS-Gate Orchestra model")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--output-file", type=Path, required=True)
    args = ap.parse_args()
    result = resolve_model(model_path=args.model_path)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
