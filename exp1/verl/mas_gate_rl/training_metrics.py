"""Durable MAS-Gate training-metric snapshots stored beside checkpoints."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


def _json_safe(value: Any) -> Any:
    """Convert metric/config values to portable JSON without NaN/Inf."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(_json_safe(payload), fp, ensure_ascii=False, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _read_history_with_retry(history_path: Path | None, latest: dict[str, Any]) -> bytes:
    """Read the append-only logger history without making checkpointing fragile.

    The checkpoint tree is backed by PCache on the training hosts.  A file can
    be visible and writable to the logger, yet a concurrent reader may receive
    a transient ``EIO``.  Retry that read and, if PCache remains unavailable,
    preserve at least the current step.  A later checkpoint will retry copying
    the complete root history again.
    """
    fallback = (json.dumps(_json_safe(latest), ensure_ascii=False) + "\n").encode("utf-8")
    if history_path is None:
        return fallback
    attempts = max(1, int(os.environ.get("MAS_GATE_METRICS_IO_RETRIES", "5")))
    delay = max(0.0, float(os.environ.get("MAS_GATE_METRICS_IO_RETRY_SECONDS", "1")))
    last_exc: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            if not history_path.is_file():
                return fallback
            return history_path.read_bytes()
        except OSError as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(delay)
    print(
        f"[MAS-Gate metrics] warning: could not read history after {attempts} attempts "
        f"({type(last_exc).__name__}: {last_exc}); checkpoint copy contains current step only",
        flush=True,
    )
    return fallback


def _atomic_copy_or_latest(history_path: Path | None, destination: Path, latest: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_history_with_retry(history_path, latest)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as dst:
            dst.write(payload)
        os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def snapshot_checkpoint_metrics(
    checkpoint_root: str | os.PathLike[str],
    step: int,
    metrics: dict[str, Any],
    config: Any,
    history_path: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Save current metrics, full JSONL history and resolved config beside a checkpoint.

    A copy is written both under ``global_step_N/metrics`` and inside
    ``global_step_N/actor/huggingface``.  The latter deliberately follows the
    exported HF weights so metric history survives MAS-Gate's checkpoint
    migration/export workflow.
    """
    checkpoint_dir = Path(checkpoint_root) / f"global_step_{int(step)}"
    history = Path(history_path) if history_path else None
    latest = {"step": int(step), "data": metrics}
    try:
        resolved_config = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    except Exception:
        resolved_config = config

    targets = [
        checkpoint_dir / "metrics",
        checkpoint_dir / "actor" / "huggingface",
    ]
    written: list[str] = []
    for target in targets:
        try:
            target.mkdir(parents=True, exist_ok=True)
            latest_name = "latest_metrics.json" if target.name == "metrics" else "latest_training_metrics.json"
            config_name = "resolved_config.json" if target.name == "metrics" else "training_config.json"
            latest_path = target / latest_name
            config_path = target / config_name
            history_copy = target / "training_metrics.jsonl"
            _atomic_json(latest_path, latest)
            _atomic_json(config_path, resolved_config)
            _atomic_copy_or_latest(history, history_copy, latest)
            written.extend(str(p) for p in (latest_path, config_path, history_copy))
        except OSError as exc:
            # These files are diagnostic replicas.  The authoritative JSONL is
            # already appended before checkpointing, so a transient PCache read
            # or rename failure must not terminate a valid RL run.
            print(
                f"[MAS-Gate metrics] warning: failed to snapshot metrics into {target}: "
                f"{type(exc).__name__}: {exc}; training will continue",
                flush=True,
            )
    return written
