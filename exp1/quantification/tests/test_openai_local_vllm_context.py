"""Regression tests for localhost vLLM context fitting via backend=openai."""

from __future__ import annotations

from pathlib import Path

from quantification.clients import openai_client
from quantification.config import QuantificationConfig


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list(range(sum(len(str(item.get("content", ""))) for item in messages)))

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, token_ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return bytes(token_ids).decode("utf-8", errors="ignore")


def test_openai_backend_retains_explicit_local_model_paths(monkeypatch, tmp_path: Path) -> None:
    scoring = tmp_path / "qwen"
    embedding = tmp_path / "bge"
    scoring.mkdir()
    embedding.mkdir()
    monkeypatch.setenv("QUANTIFICATION_BACKEND", "openai")
    monkeypatch.setenv("QUANTIFICATION_SCORING_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("QUANTIFICATION_EMBEDDING_BASE_URL", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("QUANTIFICATION_SCORING_MODEL", "qwen-local")
    monkeypatch.setenv("QUANTIFICATION_EMBEDDING_MODEL", "bge-local")
    monkeypatch.setenv("QUANTIFICATION_SCORING_API_KEY", "EMPTY")
    monkeypatch.setenv("QUANTIFICATION_EMBEDDING_API_KEY", "EMPTY")
    monkeypatch.setenv("QUANTIFICATION_SCORING_MODEL_PATH", str(scoring))
    monkeypatch.setenv("QUANTIFICATION_EMBEDDING_MODEL_PATH", str(embedding))

    cfg = QuantificationConfig.from_env()

    assert cfg.resolved_backend == "openai"
    assert cfg.scoring_model_path == str(scoring)
    assert cfg.embedding_model_path == str(embedding)


def test_localhost_openai_endpoint_still_fits_context(monkeypatch, tmp_path: Path) -> None:
    scoring = tmp_path / "qwen"
    scoring.mkdir()
    monkeypatch.setenv("QUANTIFICATION_SCORING_MAX_MODEL_LEN", "512")
    monkeypatch.setenv("QUANTIFICATION_SCORING_CONTEXT_SAFETY_TOKENS", "32")
    monkeypatch.setattr(openai_client, "_local_scoring_tokenizer", lambda _: _FakeTokenizer())
    cfg = QuantificationConfig(
        backend="openai",
        resolved_backend="openai",
        scoring_base_url="http://localhost:8000/v1",
        scoring_model_path=str(scoring),
        vllm_max_model_len=512,
    )
    messages = [
        {"role": "system", "content": "score"},
        {"role": "user", "content": "prefix " + ("x" * 900) + " tail"},
        {"role": "assistant", "content": "42"},
    ]

    fitted, prompt_tokens = openai_client._fit_scoring_messages_to_context(
        messages, max_tokens=16, config=cfg
    )

    assert 0 < prompt_tokens <= 512 - 16 - 32
    assert "[MAS-GATE: middle of verbose reasoning context omitted]" in fitted[1]["content"]


def test_remote_openai_endpoint_is_not_locally_tokenized(tmp_path: Path) -> None:
    cfg = QuantificationConfig(
        backend="openai",
        resolved_backend="openai",
        scoring_base_url="https://example.invalid/v1",
        scoring_model_path=str(tmp_path),
    )
    messages = [{"role": "user", "content": "unchanged"}]

    fitted, prompt_tokens = openai_client._fit_scoring_messages_to_context(
        messages, max_tokens=16, config=cfg
    )

    assert fitted == messages
    assert prompt_tokens == -1
