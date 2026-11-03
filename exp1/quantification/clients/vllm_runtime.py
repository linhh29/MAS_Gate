"""Sequential vLLM engine orchestration: one GPU-resident model at a time."""

from __future__ import annotations

import gc
import threading
from typing import Literal

from quantification.config import QuantificationConfig

_active: Literal["none", "scoring", "embedding"] = "none"
_runtime_lock = threading.Lock()
# Serializes MAS in-process vLLM and quantification so only one engine lifecycle runs.
vllm_process_lock = threading.RLock()


def _release_embedding_engine() -> None:
    from quantification.clients.vllm_embedding import reset_vllm_embedding_engine
    from quantification.clients.vllm_shutdown import wait_for_gpu_memory

    reset_vllm_embedding_engine(skip_runtime_reset=True)
    gc.collect()
    wait_for_gpu_memory(min_free_gib=10.0)


def _release_scoring_engine() -> None:
    from quantification.clients.j_tilde import reset_vllm_j_scorer
    from quantification.clients.vllm_shutdown import wait_for_gpu_memory

    reset_vllm_j_scorer(skip_runtime_reset=True)
    gc.collect()
    wait_for_gpu_memory(min_free_gib=10.0)


def scoring_engine_is_loaded() -> bool:
    from quantification.clients.j_tilde import get_vllm_j_scorer_if_loaded

    return get_vllm_j_scorer_if_loaded() is not None


def prepare_j_phase_engine(config: QuantificationConfig) -> "VllmJScorer":
    """
    Load scoring for J phase. Reuses a warm engine when already resident
    (e.g. MAS sub-agent execution) to avoid CUDA re-init in the same process.
    """
    with vllm_process_lock:
        if scoring_engine_is_loaded():
            print(
                "[vllm-runtime] reuse warm scoring engine for J phase (no reload)",
                flush=True,
            )
        else:
            release_all_engines()
        return prepare_scoring_engine(config)


def prepare_scoring_engine(config: QuantificationConfig) -> "VllmJScorer":
    """Release embedding if present, then load scoring."""
    from quantification.clients.j_tilde import get_vllm_j_scorer
    from quantification.clients.vllm_embedding import get_vllm_embedding_engine_if_loaded

    global _active
    with _runtime_lock:
        if get_vllm_embedding_engine_if_loaded() is not None:
            print(
                "[vllm-runtime] releasing embedding engine before scoring load",
                flush=True,
            )
            _release_embedding_engine()
        scorer = get_vllm_j_scorer(config)
        _active = "scoring"
        return scorer


def prepare_embedding_engine(config: QuantificationConfig) -> "VllmEmbeddingEngine":
    """Release scoring if present, then load embedding."""
    from quantification.clients.j_tilde import get_vllm_j_scorer_if_loaded
    from quantification.clients.vllm_embedding import get_vllm_embedding_engine

    global _active
    with _runtime_lock:
        if get_vllm_j_scorer_if_loaded() is not None:
            print(
                "[vllm-runtime] releasing scoring engine before embedding load",
                flush=True,
            )
            _release_scoring_engine()
        engine = get_vllm_embedding_engine(config)
        _active = "embedding"
        return engine


def release_all_engines() -> None:
    """Tear down scoring and embedding engines and free GPUs."""
    global _active
    with _runtime_lock:
        _release_scoring_engine()
        from quantification.clients.vllm_embedding import reset_vllm_embedding_engine

        reset_vllm_embedding_engine(skip_runtime_reset=True)
        _active = "none"
    gc.collect()


def reset_runtime_state() -> None:
    global _active
    with _runtime_lock:
        _active = "none"
