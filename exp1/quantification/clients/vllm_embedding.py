"""In-process vLLM embedding for K-metric communication strength."""

from __future__ import annotations

import threading

from quantification.vllm_devices import cuda_visible_devices
from quantification.config import QuantificationConfig

_embedding_lock = threading.Lock()
_embedding_engine: "VllmEmbeddingEngine | None" = None
_engine_lock = threading.Lock()


class VllmEmbeddingEngine:
    """In-process vLLM pooling runner for text embeddings."""

    def __init__(self, config: QuantificationConfig) -> None:
        self.config = config
        tp = max(1, int(config.vllm_embedding_tensor_parallel_size))

        def _build_llm() -> object:
            from vllm import LLM

            return LLM(
                model=config.embedding_model_path,
                trust_remote_code=True,
                runner="pooling",
                tensor_parallel_size=tp,
                gpu_memory_utilization=config.vllm_embedding_gpu_memory_utilization,
                enforce_eager=config.vllm_enforce_eager,
                disable_custom_all_reduce=config.vllm_disable_custom_all_reduce,
                max_model_len=config.vllm_embedding_max_model_len,
                max_num_seqs=max(1, config.embedding_concurrency),
                limit_mm_per_prompt={"image": 0, "video": 0},
                skip_mm_profiling=True,
            )

        if config.vllm_embedding_device_ids:
            with cuda_visible_devices(config.vllm_embedding_device_ids):
                self._llm = _build_llm()
        else:
            self._llm = _build_llm()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with _embedding_lock:
            outputs = self._llm.embed(texts, use_tqdm=False)
        if len(outputs) != len(texts):
            raise RuntimeError(
                f"vLLM embed returned {len(outputs)} vectors for {len(texts)} texts"
            )
        return [output.outputs.embedding for output in outputs]


def reset_vllm_embedding_engine(skip_runtime_reset: bool = False) -> None:
    """Drop cached embedding engine (frees GPU for scoring reload)."""
    from quantification.clients.vllm_runtime import reset_runtime_state
    from quantification.clients.vllm_shutdown import shutdown_vllm_llm

    global _embedding_engine
    with _engine_lock:
        if _embedding_engine is not None:
            shutdown_vllm_llm(_embedding_engine._llm)
        _embedding_engine = None
    if not skip_runtime_reset:
        reset_runtime_state()


def get_vllm_embedding_engine_if_loaded() -> VllmEmbeddingEngine | None:
    return _embedding_engine


def get_vllm_embedding_engine(config: QuantificationConfig) -> VllmEmbeddingEngine:
    global _embedding_engine
    with _engine_lock:
        if _embedding_engine is None:
            _embedding_engine = VllmEmbeddingEngine(config)
        return _embedding_engine
