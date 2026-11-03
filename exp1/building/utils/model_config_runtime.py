"""Runtime helpers to read model parameters from building/skill_mas/model_config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MODEL_CONFIG = Path(__file__).resolve().parents[1] / "skill_mas" / "model_config.json"

DEFAULT_SCORING_MODEL = "qwen3.6-35b-a3b"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"


def load_model_config() -> dict[str, Any]:
    payload = json.loads(_MODEL_CONFIG.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def model_runtime_params(model_name: str) -> dict[str, Any]:
    cfg = load_model_config()
    row = cfg.get(str(model_name), {})
    if not isinstance(row, dict):
        return {}
    allowed = ("api_key", "base_url", "temperature", "reasoning_effort", "max_tokens", "extra_body")
    return {k: row[k] for k in allowed if k in row and row[k] is not None}


def api_key_for_model(model_name: str) -> str | None:
    row = model_runtime_params(model_name)
    key = row.get("api_key")
    return str(key) if key else None


def first_available_api_key() -> str | None:
    cfg = load_model_config()
    for name, row in cfg.items():
        if str(name).startswith("_"):
            continue
        if isinstance(row, dict) and row.get("api_key"):
            return str(row["api_key"])
    return None


def endpoint_for_model(model_name: str) -> dict[str, Any]:
    cfg = load_model_config()
    row = cfg.get(str(model_name), {})
    if not isinstance(row, dict):
        row = {}
    extra_body = row.get("extra_body")
    return {
        "model": str(model_name),
        "base_url": str(row.get("base_url") or ""),
        "api_key": str(row.get("api_key") or ""),
        "extra_body": extra_body if isinstance(extra_body, dict) else None,
    }


def bootstrap_pyserini_openai_client() -> None:
    """
    Pyserini creates an OpenAI client at import time (even for BM25-only use).
    Patch openai.OpenAI to supply api_key from model_config.json when missing.
    """
    import openai

    config_key = first_available_api_key()
    if not config_key:
        raise RuntimeError(
            f"No api_key found in {_MODEL_CONFIG}; required to import pyserini"
        )

    _openai_cls = openai.OpenAI

    def _openai_with_model_config_key(*args, **kwargs):
        if not kwargs.get("api_key"):
            kwargs["api_key"] = config_key
        return _openai_cls(*args, **kwargs)

    openai.OpenAI = _openai_with_model_config_key  # type: ignore[misc, assignment]
