"""Configuration for MAS node quantification metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from quantification.model_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SCORING_MODEL,
    endpoint_for_model,
)
from quantification.vllm_devices import device_ids_overlap, parse_device_ids

DEFAULT_SCORING_MODEL_PATH = os.getenv(
    "QUANTIFICATION_SCORING_MODEL_PATH",
    "/tmp/Qwen3-30B-A3B",
)
DEFAULT_EMBEDDING_MODEL_PATH = os.getenv(
    "QUANTIFICATION_EMBEDDING_MODEL_PATH",
    "/tmp/bge-m3-zeroshot-v2.0",
)
DEFAULT_VLLM_SCORING_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_INPROCESS_BASE_URL = "inprocess"
DEFAULT_VLLM_SCORING_SERVED_MODEL = os.getenv(
    "QUANTIFICATION_SCORING_SERVED_MODEL",
    "Qwen3-30B-A3B-Instruct-2507",
)
DEFAULT_VLLM_EMBEDDING_SERVED_MODEL = os.getenv(
    "QUANTIFICATION_EMBEDDING_SERVED_MODEL",
    "bge-m3-zeroshot-v2.0",
)

DEFAULT_INTER_NODE_MODE = "unified_baseline"
INTER_NODE_MODES = frozenset({
    "unified_baseline",
    "legacy_canonical",
    "legacy_no_sink",
    "legacy_with_sink",
})


@dataclass
class QuantificationConfig:
    """Hyper-parameters and API settings for quantification."""

    backend: str = "vllm"
    scoring_base_url: str = ""
    embedding_base_url: str = ""
    base_url: str = ""
    api_key: str = ""
    scoring_api_key: str = ""
    embedding_api_key: str = ""
    scoring_model: str = ""
    embedding_model: str = ""
    scoring_model_path: str = ""
    embedding_model_path: str = ""
    scoring_extra_body: dict[str, Any] | None = None
    resolved_backend: str = "vllm"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.8
    vllm_max_model_len: int = 8192
    vllm_scoring_max_num_seqs: int = 1
    vllm_enforce_eager: bool = False
    vllm_language_model_only: bool = True
    vllm_disable_custom_all_reduce: bool = True
    vllm_kv_cache_memory_bytes: int | None = None
    vllm_embedding_tensor_parallel_size: int = 1
    vllm_embedding_gpu_memory_utilization: float = 0.35
    vllm_embedding_max_model_len: int = 8192
    vllm_embedding_kv_cache_memory_bytes: int | None = None
    vllm_scoring_device_ids: list[int] | None = None
    vllm_embedding_device_ids: list[int] | None = None
    vllm_sequential_gpu_sharing: bool = True

    epsilon: float = 1e-8
    context_separator: str = "\n"

    gamma: float = 0.5

    use_dag_aware_utility: bool = True
    use_no_sink_k: bool = True
    # Inter-node strength implementation:
    #   unified_baseline (default): K_i = 1 + sum_j R_ij (1 + G_j)
    #   legacy_canonical: previous unified.py behavior (terminal K=0)
    #   legacy_no_sink: previous normalized no-output-sink behavior
    #   legacy_with_sink: previous normalized behavior including output-sink edges
    inter_node_mode: str = DEFAULT_INTER_NODE_MODE
    use_depth_fair_g: bool = False

    request_timeout: float = 180.0
    max_retries: int = 8
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    scoring_context_max_chars: int = 8000
    embedding_batch_size: int = 32
    scoring_concurrency: int = 16
    embedding_concurrency: int = 16
    trace_concurrency: int = 8

    @classmethod
    def from_env(cls, **overrides) -> QuantificationConfig:
        backend = str(
            overrides.get("backend")
            or os.getenv("QUANTIFICATION_BACKEND", "vllm")
        ).strip().lower()

        scoring_device_ids: list[int] | None = None
        embedding_device_ids: list[int] | None = None
        sequential_gpu_sharing = False

        if backend == "vllm":
            scoring_model = str(
                overrides.get("scoring_model")
                or os.getenv("QUANTIFICATION_SCORING_SERVED_MODEL")
                or DEFAULT_VLLM_SCORING_SERVED_MODEL
            )
            embedding_model = str(
                overrides.get("embedding_model")
                or os.getenv("QUANTIFICATION_EMBEDDING_SERVED_MODEL")
                or DEFAULT_VLLM_EMBEDDING_SERVED_MODEL
            )
            scoring_model_path = str(
                overrides.get("scoring_model_path")
                or os.getenv("QUANTIFICATION_SCORING_MODEL_PATH")
                or DEFAULT_SCORING_MODEL_PATH
            )
            embedding_model_path = str(
                overrides.get("embedding_model_path")
                or os.getenv("QUANTIFICATION_EMBEDDING_MODEL_PATH")
                or DEFAULT_EMBEDDING_MODEL_PATH
            )
            scoring_base_url = str(
                overrides.get("scoring_base_url")
                or os.getenv("QUANTIFICATION_SCORING_BASE_URL")
                or DEFAULT_VLLM_SCORING_BASE_URL
            )
            embedding_base_url = str(
                overrides.get("embedding_base_url")
                or os.getenv("QUANTIFICATION_EMBEDDING_BASE_URL")
                or DEFAULT_VLLM_INPROCESS_BASE_URL
            )
            scoring_api_key = str(overrides.get("scoring_api_key") or "EMPTY")
            embedding_api_key = str(overrides.get("embedding_api_key") or "EMPTY")
            scoring_extra_body = overrides.get("scoring_extra_body")
            if scoring_extra_body is None:
                scoring_extra_body = endpoint_for_model(str(scoring_model)).get("extra_body")
            scoring_device_ids = parse_device_ids(
                overrides.get("vllm_scoring_device_ids")
                or os.getenv("QUANTIFICATION_VLLM_SCORING_GPUS")
                or os.getenv("VLLM_SCORING_GPUS")
            )
            embedding_device_ids = parse_device_ids(
                overrides.get("vllm_embedding_device_ids")
                or os.getenv("QUANTIFICATION_VLLM_EMBEDDING_GPUS")
                or os.getenv("VLLM_EMBEDDING_GPU")
            )
            sequential_gpu_sharing = str(
                overrides.get("vllm_sequential_gpu_sharing")
                or os.getenv("QUANTIFICATION_VLLM_SEQUENTIAL_GPUS", "1")
            ).strip().lower() in {"1", "true", "yes", "on"}
            shared_gpu_ids = parse_device_ids(
                overrides.get("vllm_shared_device_ids")
                or os.getenv("QUANTIFICATION_VLLM_GPUS")
                or os.getenv("VLLM_GPUS")
            )
            if sequential_gpu_sharing and shared_gpu_ids:
                scoring_device_ids = shared_gpu_ids
                embedding_device_ids = shared_gpu_ids
            resolved_backend = "vllm"
        else:
            scoring_model = (
                overrides.get("scoring_model")
                or os.getenv("QUANTIFICATION_SCORING_MODEL")
                or os.getenv("QUANTIFICATION_SCORING_SERVED_MODEL")
                or DEFAULT_SCORING_MODEL
            )
            embedding_model = (
                overrides.get("embedding_model")
                or os.getenv("QUANTIFICATION_EMBEDDING_MODEL")
                or os.getenv("QUANTIFICATION_EMBEDDING_SERVED_MODEL")
                or DEFAULT_EMBEDDING_MODEL
            )
            scoring_endpoint = endpoint_for_model(str(scoring_model))
            embedding_endpoint = endpoint_for_model(str(embedding_model))
            # MAS-Gate local training must not fall back to external model-config URLs.
            # Honor explicit env base URLs for the local OpenAI-compatible vLLM servers.
            scoring_base_url = (
                overrides.get("scoring_base_url")
                or os.getenv("QUANTIFICATION_SCORING_BASE_URL")
                or scoring_endpoint["base_url"]
            )
            embedding_base_url = (
                overrides.get("embedding_base_url")
                or os.getenv("QUANTIFICATION_EMBEDDING_BASE_URL")
                or embedding_endpoint["base_url"]
            )
            scoring_api_key = (
                overrides.get("scoring_api_key")
                or os.getenv("QUANTIFICATION_SCORING_API_KEY")
                or os.getenv("QUANTIFICATION_API_KEY")
                or scoring_endpoint["api_key"]
            )
            embedding_api_key = (
                overrides.get("embedding_api_key")
                or os.getenv("QUANTIFICATION_EMBEDDING_API_KEY")
                or os.getenv("QUANTIFICATION_API_KEY")
                or embedding_endpoint["api_key"]
            )
            scoring_extra_body = overrides.get("scoring_extra_body")
            if scoring_extra_body is None:
                scoring_extra_body = scoring_endpoint.get("extra_body")
            # ``backend=openai`` is also used for MAS-Gate's already-running
            # localhost vLLM servers.  Retain their local model paths so the
            # client can perform tokenizer-accurate context fitting without
            # loading another inference engine or resolving any remote model.
            scoring_model_path = str(
                overrides.get("scoring_model_path")
                or os.getenv("QUANTIFICATION_SCORING_MODEL_PATH")
                or ""
            )
            embedding_model_path = str(
                overrides.get("embedding_model_path")
                or os.getenv("QUANTIFICATION_EMBEDDING_MODEL_PATH")
                or ""
            )
            resolved_backend = "openai"

        cfg = cls(
            backend=backend,
            scoring_base_url=str(scoring_base_url or ""),
            embedding_base_url=str(embedding_base_url or ""),
            base_url=str(scoring_base_url or ""),
            api_key=str(scoring_api_key or ""),
            scoring_api_key=str(scoring_api_key or ""),
            embedding_api_key=str(embedding_api_key or ""),
            scoring_model=str(scoring_model),
            embedding_model=str(embedding_model),
            scoring_model_path=scoring_model_path,
            embedding_model_path=embedding_model_path,
            scoring_extra_body=scoring_extra_body,
            scoring_concurrency=int(
                os.getenv("QUANTIFICATION_SCORING_CONCURRENCY", "16")
            ),
            embedding_concurrency=int(
                os.getenv("QUANTIFICATION_EMBEDDING_CONCURRENCY", "16")
            ),
            trace_concurrency=int(os.getenv("QUANTIFICATION_TRACE_CONCURRENCY", "8")),
            request_timeout=float(os.getenv("QUANTIFICATION_REQUEST_TIMEOUT", "180")),
            max_retries=int(os.getenv("QUANTIFICATION_MAX_RETRIES", "8")),
            retry_base_delay_s=float(os.getenv("QUANTIFICATION_RETRY_BASE_DELAY_S", "1.0")),
            retry_max_delay_s=float(os.getenv("QUANTIFICATION_RETRY_MAX_DELAY_S", "30.0")),
            scoring_context_max_chars=int(
                os.getenv("QUANTIFICATION_SCORING_CONTEXT_MAX_CHARS", "8000")
            ),
            vllm_tensor_parallel_size=int(
                os.getenv("QUANTIFICATION_VLLM_TENSOR_PARALLEL_SIZE", "1")
            ),
            vllm_gpu_memory_utilization=float(
                os.getenv("QUANTIFICATION_VLLM_GPU_MEMORY_UTILIZATION", "0.8")
            ),
            vllm_max_model_len=int(
                os.getenv("QUANTIFICATION_VLLM_MAX_MODEL_LEN", "8192")
            ),
            vllm_scoring_max_num_seqs=int(
                os.getenv("QUANTIFICATION_VLLM_SCORING_MAX_NUM_SEQS", "1")
            ),
            vllm_enforce_eager=os.getenv(
                "QUANTIFICATION_VLLM_ENFORCE_EAGER", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            vllm_language_model_only=os.getenv(
                "QUANTIFICATION_VLLM_LANGUAGE_MODEL_ONLY", "1"
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            vllm_disable_custom_all_reduce=os.getenv(
                "QUANTIFICATION_VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1"
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            vllm_embedding_tensor_parallel_size=int(
                os.getenv("QUANTIFICATION_VLLM_EMBEDDING_TENSOR_PARALLEL_SIZE", "1")
            ),
            vllm_embedding_gpu_memory_utilization=float(
                os.getenv(
                    "QUANTIFICATION_VLLM_EMBEDDING_GPU_MEMORY_UTILIZATION",
                    "0.35",
                )
            ),
            vllm_embedding_max_model_len=int(
                os.getenv("QUANTIFICATION_VLLM_EMBEDDING_MAX_MODEL_LEN", "8192")
            ),
            vllm_scoring_device_ids=scoring_device_ids,
            vllm_embedding_device_ids=embedding_device_ids,
            vllm_sequential_gpu_sharing=sequential_gpu_sharing if backend == "vllm" else False,
            resolved_backend=resolved_backend,
            inter_node_mode=str(
                overrides.get("inter_node_mode")
                or os.getenv("QUANTIFICATION_INTER_NODE_MODE", DEFAULT_INTER_NODE_MODE)
            ).strip().lower(),
            use_depth_fair_g=str(
                overrides.get("use_depth_fair_g")
                if overrides.get("use_depth_fair_g") is not None
                else os.getenv("QUANTIFICATION_USE_DEPTH_FAIR_G", "0")
            ).strip().lower() in {"1", "true", "yes", "on"},
        )

        for key, value in overrides.items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)

        if overrides.get("api_key") and "scoring_api_key" not in overrides:
            cfg.scoring_api_key = str(overrides["api_key"])
            cfg.api_key = cfg.scoring_api_key
        if overrides.get("scoring_base_url") and "base_url" not in overrides:
            cfg.base_url = cfg.scoring_base_url

        if cfg.vllm_scoring_device_ids:
            cfg.vllm_tensor_parallel_size = len(cfg.vllm_scoring_device_ids)
        if cfg.vllm_embedding_device_ids:
            cfg.vllm_embedding_tensor_parallel_size = len(cfg.vllm_embedding_device_ids)

        return cfg

    def validate(self) -> None:
        backend = (self.backend or "vllm").strip().lower()
        if backend not in {"openai", "remote", "vllm"}:
            raise ValueError(f"Unsupported quantification backend: {backend!r}")
        self.inter_node_mode = (self.inter_node_mode or DEFAULT_INTER_NODE_MODE).strip().lower()
        if self.inter_node_mode not in INTER_NODE_MODES:
            raise ValueError(
                f"Unsupported inter_node_mode: {self.inter_node_mode!r}; "
                f"choose one of {sorted(INTER_NODE_MODES)!r}"
            )
        if not self.scoring_model:
            raise ValueError("scoring_model is required")
        if not self.embedding_model:
            raise ValueError("embedding_model is required")
        if not self.scoring_base_url:
            raise ValueError("scoring_base_url is required")
        if backend != "vllm" and not self.embedding_base_url:
            raise ValueError("embedding_base_url is required")

        if backend == "vllm":
            if not self.scoring_model_path:
                raise ValueError(
                    "scoring_model_path is required for vllm backend; "
                    "set QUANTIFICATION_SCORING_MODEL_PATH"
                )
            if not self.embedding_model_path:
                raise ValueError(
                    "embedding_model_path is required for vllm backend; "
                    "set QUANTIFICATION_EMBEDDING_MODEL_PATH"
                )
            if self.vllm_scoring_device_ids:
                expected_tp = len(self.vllm_scoring_device_ids)
                if self.vllm_tensor_parallel_size != expected_tp:
                    raise ValueError(
                        "vllm_tensor_parallel_size must match the number of "
                        f"scoring GPUs ({expected_tp}); got "
                        f"{self.vllm_tensor_parallel_size}. "
                        "Set QUANTIFICATION_VLLM_SCORING_GPUS and "
                        "QUANTIFICATION_VLLM_TENSOR_PARALLEL_SIZE consistently."
                    )
            if self.vllm_embedding_device_ids:
                expected_tp = len(self.vllm_embedding_device_ids)
                if self.vllm_embedding_tensor_parallel_size != expected_tp:
                    raise ValueError(
                        "vllm_embedding_tensor_parallel_size must match the number of "
                        f"embedding GPUs ({expected_tp}); got "
                        f"{self.vllm_embedding_tensor_parallel_size}. "
                        "Set QUANTIFICATION_VLLM_EMBEDDING_GPUS and "
                        "QUANTIFICATION_VLLM_EMBEDDING_TENSOR_PARALLEL_SIZE consistently."
                    )
            if (
                not self.vllm_sequential_gpu_sharing
                and self.vllm_scoring_device_ids
                and self.vllm_embedding_device_ids
                and device_ids_overlap(
                    self.vllm_scoring_device_ids, self.vllm_embedding_device_ids
                )
            ):
                raise ValueError(
                    "scoring and embedding GPU sets must not overlap: "
                    f"scoring={self.vllm_scoring_device_ids!r} "
                    f"embedding={self.vllm_embedding_device_ids!r}"
                )
            self.resolved_backend = "vllm"
            self.embedding_base_url = self.embedding_base_url or DEFAULT_VLLM_INPROCESS_BASE_URL
        else:
            if not self.scoring_api_key:
                raise ValueError(
                    f"scoring api_key missing for model {self.scoring_model!r}; "
                    "add it to building/skill_mas/model_config.json"
                )
            if not self.embedding_api_key:
                raise ValueError(
                    f"embedding api_key missing for model {self.embedding_model!r}; "
                    "add it to building/skill_mas/model_config.json"
                )
            self.resolved_backend = "openai"

        self.base_url = self.scoring_base_url
        if not self.api_key:
            self.api_key = self.scoring_api_key
