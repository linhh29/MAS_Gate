"""NLI (Natural Language Inference) client for measuring entailment degree.

Uses bge-m3-zeroshot-v2.0 (MoritzLaurer), a multilingual universal zero-shot
classifier based on XLM-RoBERTa.  The model outputs a continuous softmax
probability P(entailment) in [0, 1] via a 2-way classification head
(entailment vs not_entailment).

Key properties:
  - Multilingual: supports 100+ languages via XLM-RoBERTa backbone
  - Context window: up to 8192 tokens
  - Model size: ~600M parameters
  - License: MIT (foundation model)
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Default local path for the model
DEFAULT_NLI_MODEL_PATH = "/tmp/bge-m3-zeroshot-v2.0"


class NLIClient:
    """Wrapper around bge-m3-zeroshot-v2.0 for entailment scoring.

    The model performs 2-way classification: entailment vs not_entailment.
    P(entailment) is used as the continuous entailment degree.

    For the recommended model (bge-m3-zeroshot-v2.0):
      - Backbone: XLM-RoBERTa (BAAI/bge-m3-retromae)
      - Training: 500+ synthetic tasks + MNLI + FEVER-NLI (commercially clean)
      - Model size: ~600M parameters
      - Context: up to 8192 tokens
      - GPU memory: ~2.4 GB in fp16
    """

    def __init__(
        self,
        model_path: str = DEFAULT_NLI_MODEL_PATH,
        device: str | None = None,
        max_length: int = 8192,
        batch_size: int = 32,
    ) -> None:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._device = device
        local_cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
        configured_max = int(os.environ.get("QUANTIFICATION_NLI_MAX_LENGTH", str(max_length)))
        model_max = int(getattr(local_cfg, "max_position_embeddings", configured_max) or configured_max)
        # The local BGE-M3 checkpoint has max_position_embeddings=8194: 8192
        # content/pair tokens plus XLM-R special-token positions.  Its downloaded
        # tokenizer_config.json still carries the XLM-R legacy value 512, which is
        # metadata inherited from the base tokenizer rather than this checkpoint's
        # actual encoder limit.  Use the model config as the authority and keep the
        # explicitly configured request limit (normally 8192) below it.
        self._max_length = max(1, min(configured_max, model_max))
        self._batch_size = batch_size
        self._lock = threading.Lock()

        self._backend = os.environ.get("QUANTIFICATION_NLI_BACKEND", "transformers").strip().lower()
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # Avoid transformers applying/warning about the stale tokenizer-side 512
        # limit when we explicitly tokenize this long-context BGE-M3 checkpoint.
        self._tokenizer.model_max_length = self._max_length
        self._llm = None
        self._http_base_url = os.environ.get(
            "QUANTIFICATION_NLI_BASE_URL",
            os.environ.get("QUANTIFICATION_EMBEDDING_BASE_URL", "http://127.0.0.1:8002/v1"),
        ).rstrip("/")
        if self._http_base_url.endswith("/v1"):
            self._http_base_url = self._http_base_url[:-3]
        self._http_model = os.environ.get(
            "QUANTIFICATION_NLI_SERVED_MODEL",
            os.environ.get("QUANTIFICATION_EMBEDDING_SERVED_MODEL", "bge-m3-zeroshot-v2.0"),
        )
        self._http_timeout = float(os.environ.get("QUANTIFICATION_NLI_HTTP_TIMEOUT", "180"))
        self._http_retries = max(1, int(os.environ.get("QUANTIFICATION_NLI_HTTP_RETRIES", "3")))

        if self._backend in {"vllm_http", "http", "openai_classify"}:
            # IMPORTANT: RewardLoopWorker is a CPU-only Ray actor.  It must never
            # instantiate CUDA/vLLM.  The classifier lives in a separate local vLLM
            # process and this worker only sends HTTP requests to 127.0.0.1.
            cfg = local_cfg
            id2label = getattr(cfg, "id2label", {})
            self._model = None
        elif self._backend in {"vllm", "vllm_classify", "classify"}:
            # Standalone/offline compatibility only.  MAS-Gate training must use
            # vllm_http because Ray reward actors intentionally have no GPU resource.
            from vllm import LLM

            tp = int(os.environ.get("QUANTIFICATION_VLLM_NLI_TENSOR_PARALLEL_SIZE", os.environ.get("SUB_AGENT_TP", "8")))
            self._llm = LLM(
                model=model_path,
                task="classify",
                trust_remote_code=True,
                tensor_parallel_size=max(1, tp),
                dtype=os.environ.get("QUANTIFICATION_VLLM_NLI_DTYPE", "float16"),
                gpu_memory_utilization=float(os.environ.get("QUANTIFICATION_VLLM_NLI_GPU_MEMORY_UTILIZATION", "0.08")),
                max_seq_len_to_capture=max_length,
                max_model_len=max_length,
                max_num_seqs=max(1, batch_size),
                disable_custom_all_reduce=os.environ.get("QUANTIFICATION_VLLM_DISABLE_CUSTOM_ALL_REDUCE", "1").strip().lower() not in {"0", "false", "no", "off"},
            )
            cfg = getattr(self._llm.llm_engine.model_config, "hf_config", None)
            id2label = getattr(cfg, "id2label", {}) if cfg is not None else {}
            self._model = None
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            self._model.to(device)
            self._model.eval()
            id2label = getattr(self._model.config, "id2label", {})

        # Determine label mapping from model config
        if id2label:
            self._label2id = {v.lower(): k for k, v in id2label.items()}
        else:
            # Fallback: most NLI models use this order
            self._label2id = {"entailment": 0, "not_entailment": 1}

        self._entail_id = self._label2id.get("entailment", 0)
        self._has_neutral = "neutral" in self._label2id

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_length(self) -> int:
        return self._max_length

    def _encode_pair_prompt(self, premise: str, hypothesis: str) -> dict:
        token_ids = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self._max_length,
            padding=False,
        )["input_ids"]
        return {"prompt_token_ids": token_ids}

    def _encode_pair_text(self, premise: str, hypothesis: str) -> str:
        # vLLM /classify accepts strings rather than token-id dictionaries.  For
        # XLM-R/BGE, tokenizing `premise</s></s>hypothesis` with add_special_tokens
        # produces exactly the same ids as tokenizer(premise, hypothesis).
        #
        # Do pair-aware truncation here.  If we sent the original concatenated
        # string and relied only on vLLM's truncate_prompt_tokens, the server would
        # see one sequence and could discard most/all of one side of an oversized
        # NLI pair.  XLM-R uses four special tokens for a pair:
        #   <s> premise </s></s> hypothesis </s>
        # Keep the beginning of both sides and remove tokens longest-first, matching
        # the tokenizer's normal right-truncation semantics.
        sep = str(self._tokenizer.sep_token or "</s>")
        p_ids = self._tokenizer.encode(str(premise or "").strip(), add_special_tokens=False, verbose=False)
        h_ids = self._tokenizer.encode(str(hypothesis or "").strip(), add_special_tokens=False, verbose=False)
        special_count = int(self._tokenizer.num_special_tokens_to_add(pair=True))
        budget = max(0, self._max_length - special_count)
        excess = max(0, len(p_ids) + len(h_ids) - budget)
        while excess:
            if len(p_ids) >= len(h_ids) and p_ids:
                removable = max(1, len(p_ids) - len(h_ids))
                take = min(excess, removable, len(p_ids))
                del p_ids[-take:]
            elif h_ids:
                removable = max(1, len(h_ids) - len(p_ids))
                take = min(excess, removable, len(h_ids))
                del h_ids[-take:]
            else:
                break
            excess -= take
        p_text = self._tokenizer.decode(p_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        h_text = self._tokenizer.decode(h_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return f"{p_text.strip()}{sep}{sep}{h_text.strip()}"

    def _http_classify(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        parsed = urlparse(self._http_base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise RuntimeError(
                "MAS-Gate NLI HTTP backend only permits an exact local vLLM endpoint "
                "on http://127.0.0.1 or http://localhost; "
                f"got {self._http_base_url!r}"
            )
        payload = json.dumps(
            {
                "model": self._http_model,
                "input": [self._encode_pair_text(p, h) for p, h in pairs],
                "truncate_prompt_tokens": self._max_length,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._http_base_url}/classify",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_exc: Exception | None = None
        for attempt in range(self._http_retries):
            try:
                with urllib.request.urlopen(request, timeout=self._http_timeout) as response:
                    obj: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                rows = obj.get("data") or []
                if len(rows) != len(pairs):
                    raise RuntimeError(f"BGE /classify returned {len(rows)} rows for {len(pairs)} pairs: {obj}")
                return np.asarray([float(row["probs"][self._entail_id]) for row in rows], dtype=np.float32)
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self._http_retries:
                    time.sleep(min(8.0, 2.0**attempt))
        raise RuntimeError(
            f"local BGE vLLM classify failed at {self._http_base_url}/classify: "
            f"{type(last_exc).__name__}: {last_exc}"
        ) from last_exc

    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        """Compute P(entailment | premise, hypothesis) as a continuous score."""
        if not premise or not premise.strip():
            return 0.0
        if not hypothesis or not hypothesis.strip():
            return 0.0

        if self._backend in {"vllm_http", "http", "openai_classify"}:
            with self._lock:
                return float(self._http_classify([(premise, hypothesis)])[0])

        if self._llm is not None:
            with self._lock:
                out = self._llm.classify([self._encode_pair_prompt(premise, hypothesis)], use_tqdm=False)[0]
            probs = np.asarray(out.outputs.probs, dtype=np.float32)
            return float(probs[self._entail_id])

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        return float(probs[self._entail_id])

    def entailment_probabilities(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> np.ndarray:
        """Batch compute entailment probabilities for multiple pairs.

        Args:
            pairs: List of (premise, hypothesis) tuples.

        Returns:
            Array of P(entailment) values, shape (len(pairs),).
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        results = np.zeros(len(pairs), dtype=np.float32)

        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            valid_positions = [
                j for j, (premise, hypothesis) in enumerate(batch)
                if str(premise or "").strip() and str(hypothesis or "").strip()
            ]
            if not valid_positions:
                continue
            valid_batch = [batch[j] for j in valid_positions]

            if self._backend in {"vllm_http", "http", "openai_classify"}:
                with self._lock:
                    values = self._http_classify(valid_batch)
                for j, value in zip(valid_positions, values):
                    results[i + j] = value
                continue

            if self._llm is not None:
                prompts = [self._encode_pair_prompt(p, h) for p, h in valid_batch]
                with self._lock:
                    outs = self._llm.classify(prompts, use_tqdm=False)
                values = np.asarray([o.outputs.probs[self._entail_id] for o in outs], dtype=np.float32)
                for j, value in zip(valid_positions, values):
                    results[i + j] = value
                continue

            premises = [p for p, _ in valid_batch]
            hypotheses = [h for _, h in valid_batch]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            values = probs[:, self._entail_id]
            for j, value in zip(valid_positions, values):
                results[i + j] = value

        return results


# Global singleton for reuse across the pipeline
_nli_client: NLIClient | None = None
_nli_client_lock = threading.Lock()


def get_nli_client(
    model_path: str | None = None,
    device: str | None = None,
) -> NLIClient:
    """Get or create the global NLI client singleton.

    MAS-Gate runs fully local; by default use /tmp/bge-m3-zeroshot-v2.0,
    overridable with MAS_GATE_NLI_MODEL_PATH.
    """
    global _nli_client
    resolved_model_path = model_path or os.environ.get("MAS_GATE_NLI_MODEL_PATH", DEFAULT_NLI_MODEL_PATH)
    if _nli_client is None:
        with _nli_client_lock:
            if _nli_client is None:
                _nli_client = NLIClient(model_path=resolved_model_path, device=device)
    return _nli_client


def reset_nli_client() -> None:
    """Release the global NLI client (e.g. for GPU memory management)."""
    global _nli_client
    with _nli_client_lock:
        if _nli_client is not None:
            del _nli_client
            _nli_client = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
