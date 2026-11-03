"""Local vLLM quantification: in-process J scoring and embedding."""

from __future__ import annotations

import threading
import time
from typing import Sequence

import numpy as np
from transformers import AutoTokenizer

from quantification.clients.j_prompt_token_limit import JPromptTokenLimitExceeded
from quantification.clients.j_tilde import compute_j_tilde
from quantification.clients.vllm_runtime import (
    prepare_embedding_engine,
    prepare_j_phase_engine,
    release_all_engines,
)
from quantification.config import QuantificationConfig
from quantification.run_logger import get_run_logger


class VllmQuantificationClient:
    """
    Scheme A (fully in-process):
    - J: vLLM LLM.generate(prompt_logprobs=0), teacher-forced like get_j_tilde
    - Embedding: vLLM LLM.embed(...) on a pooling runner

    Use begin_scoring_phase / begin_embedding_phase to load each model once per batch.
    """

    def __init__(self, config: QuantificationConfig) -> None:
        config.validate()
        self.config = config
        self._backend = "vllm_inprocess"
        self._j_scorer: object | None = None
        self._embedding_engine: object | None = None
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.scoring_model_path,
            trust_remote_code=True,
        )
        self._embedding_semaphore = threading.Semaphore(config.embedding_concurrency)
        logger = get_run_logger()
        if logger is not None:
            logger.info(
                "vllm_inprocess placement (sequential release/reload): "
                f"scoring_gpus={config.vllm_scoring_device_ids or 'default'} "
                f"embedding_gpus={config.vllm_embedding_device_ids or 'default'} "
                f"scoring_tp={config.vllm_tensor_parallel_size} "
                f"embedding_tp={config.vllm_embedding_tensor_parallel_size} "
                f"sequential={config.vllm_sequential_gpu_sharing}"
            )

    def begin_scoring_phase(self) -> None:
        self._j_scorer = prepare_j_phase_engine(self.config)
        self._tokenizer = self._j_scorer.tokenizer
        self._embedding_engine = None

    def begin_embedding_phase(self) -> None:
        self._embedding_engine = prepare_embedding_engine(self.config)
        self._j_scorer = None

    def release_all_engines(self) -> None:
        self._j_scorer = None
        self._embedding_engine = None
        release_all_engines()

    def format_context(self, question: str, prefix_outputs: Sequence[str]) -> str:
        parts = [question]
        for output in prefix_outputs:
            if output:
                parts.append(output)
        return self.config.context_separator.join(parts)

    def compute_answer_cross_entropy(self, context: str, answer: str) -> float:
        if self._j_scorer is None:
            self.begin_scoring_phase()
        try:
            j_metric, j_tilde, fields = compute_j_tilde(
                self._j_scorer,
                self._tokenizer,
                context,
                answer,
                config=self.config,
            )
        except JPromptTokenLimitExceeded:
            # Engine is dead after OOM; caller reloads before the next sample.
            self._j_scorer = None
            raise
        logger = get_run_logger()
        if logger is not None:
            logger.log_api_call(
                "j_score_vllm_inprocess",
                backend=self._backend,
                model=self.config.scoring_model,
                scoring_model_path=self.config.scoring_model_path,
                base_url="inprocess",
                answer=answer,
                tokens_matched_answer=True,
                **fields,
            )
        return j_metric

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self._embedding_engine is None:
            self.begin_embedding_phase()
        engine = self._embedding_engine
        all_vectors: list[list[float]] = []
        batch_size = self.config.embedding_batch_size

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            with self._embedding_semaphore:
                t0 = time.perf_counter()
                batch_vectors = engine.embed_texts(batch)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            all_vectors.extend(batch_vectors)
            logger = get_run_logger()
            if logger is not None:
                dim = len(batch_vectors[0]) if batch_vectors else 0
                logger.log_api_call(
                    "embedding_vllm_inprocess",
                    backend=self._backend,
                    model=self.config.embedding_model,
                    embedding_model_path=self.config.embedding_model_path,
                    base_url="inprocess",
                    batch_start=start,
                    num_texts=len(batch),
                    embedding_dim=dim,
                    elapsed_ms=round(elapsed_ms, 2),
                    text_previews=[
                        (t[:120] + "...") if len(t) > 120 else t for t in batch[:3]
                    ],
                )

        matrix = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return matrix / norms
