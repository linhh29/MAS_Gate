"""Run logging for quantification (text log + structured API call records)."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_active: "RunLogger | None" = None


def set_run_logger(logger: "RunLogger | None") -> None:
    global _active
    with _lock:
        _active = logger


def get_run_logger() -> "RunLogger | None":
    return _active


class RunLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path.resolve()
        self.api_path = self.log_path.with_suffix(".api.jsonl")
        self._started = time.perf_counter()
        self._api_seq = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_path.write_text("", encoding="utf-8")

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def info(self, message: str) -> None:
        line = f"[{self._ts()}] {message}\n"
        with _lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        print(message, flush=True)

    def log_config(self, config: dict[str, Any]) -> None:
        self.info("=== quantification run config ===")
        for key, value in config.items():
            self.info(f"  {key} = {value}")

    def log_api_call(self, kind: str, **fields: Any) -> None:
        with _lock:
            self._api_seq += 1
            record = {
                "seq": self._api_seq,
                "ts_utc": self._ts(),
                "kind": kind,
                **fields,
            }
            with self.api_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary_bits = [f"kind={kind}"]
        for key in ("model", "j_value", "num_texts", "elapsed_ms", "num_answer_tokens"):
            if key in fields and fields[key] is not None:
                summary_bits.append(f"{key}={fields[key]}")
        self.info("api_call " + " ".join(summary_bits))

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        elapsed_s = time.perf_counter() - self._started
        self.info(f"=== finished in {elapsed_s:.1f}s ===")
        if summary:
            self.info("=== summary ===")
            for key, value in summary.items():
                self.info(f"  {key} = {value}")
        self.info(f"log_file={self.log_path}")
        self.info(f"api_log_file={self.api_path}")
