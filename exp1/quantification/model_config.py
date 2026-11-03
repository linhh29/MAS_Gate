"""Re-export model_config helpers (single source: building/skill_mas/model_config.json)."""

from __future__ import annotations

from building.utils.model_config_runtime import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SCORING_MODEL,
    endpoint_for_model,
    load_model_config,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_SCORING_MODEL",
    "endpoint_for_model",
    "load_model_config",
]
