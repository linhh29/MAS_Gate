"""Tests for vLLM GPU placement helpers."""

from __future__ import annotations

import pytest

from quantification.vllm_devices import device_ids_overlap, parse_device_ids
from quantification.config import QuantificationConfig


def test_parse_device_ids() -> None:
    assert parse_device_ids("0,1") == [0, 1]
    assert parse_device_ids("2") == [2]
    assert parse_device_ids("") is None
    assert parse_device_ids([3, 4]) == [3, 4]


def test_device_ids_overlap() -> None:
    assert device_ids_overlap([0, 1], [2]) is False
    assert device_ids_overlap([0, 1], [1, 2]) is True


def test_config_auto_tensor_parallel_from_gpu_sets() -> None:
    cfg = QuantificationConfig.from_env(
        backend="vllm",
        vllm_scoring_device_ids=[0, 1],
        vllm_embedding_device_ids=[2],
    )
    cfg.validate()
    assert cfg.vllm_tensor_parallel_size == 2
    assert cfg.vllm_embedding_tensor_parallel_size == 1


def test_config_allows_overlapping_gpu_sets_when_sequential() -> None:
    cfg = QuantificationConfig.from_env(
        backend="vllm",
        scoring_model_path="/tmp/scoring",
        embedding_model_path="/tmp/embedding",
        scoring_base_url="inprocess",
        embedding_base_url="inprocess",
        scoring_model="scoring",
        embedding_model="embedding",
        vllm_scoring_device_ids=[0, 1, 2, 3],
        vllm_embedding_device_ids=[0, 1, 2, 3],
        vllm_sequential_gpu_sharing=True,
    )
    cfg.validate()
    assert cfg.vllm_tensor_parallel_size == 4
    assert cfg.vllm_embedding_tensor_parallel_size == 4


def test_config_rejects_overlapping_gpu_sets_without_sequential() -> None:
    cfg = QuantificationConfig.from_env(
        backend="vllm",
        scoring_model_path="/tmp/scoring",
        embedding_model_path="/tmp/embedding",
        scoring_base_url="inprocess",
        embedding_base_url="inprocess",
        scoring_model="scoring",
        embedding_model="embedding",
        vllm_scoring_device_ids=[0, 1],
        vllm_embedding_device_ids=[1, 2],
        vllm_sequential_gpu_sharing=False,
    )
    with pytest.raises(ValueError, match="must not overlap"):
        cfg.validate()
