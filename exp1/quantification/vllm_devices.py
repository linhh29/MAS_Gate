"""GPU placement helpers for multiple in-process vLLM engines."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


def parse_device_ids(value: str | list[int] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [int(x) for x in value]
        if not cleaned:
            return None
        return cleaned
    text = str(value).strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def device_ids_overlap(left: list[int], right: list[int]) -> bool:
    return bool(set(left) & set(right))


@contextmanager
def cuda_visible_devices(device_ids: list[int]) -> Iterator[None]:
    """Restrict visible GPUs while constructing a vLLM engine."""
    if not device_ids:
        raise ValueError("device_ids must not be empty")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"duplicate device ids in {device_ids!r}")

    new_value = ",".join(str(device_id) for device_id in device_ids)
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = new_value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
