"""Helpers to tear down in-process vLLM engines and free GPU memory."""

from __future__ import annotations

import gc
import time


def shutdown_vllm_llm(llm: object, *, timeout: float = 60.0) -> None:
    """Best-effort shutdown so another model can load on the same GPUs."""
    import torch

    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            client = getattr(engine, "engine_core", None)
            if client is not None:
                shutdown = getattr(client, "shutdown", None)
                if callable(shutdown):
                    shutdown(timeout=timeout)
                else:
                    fallback = getattr(engine, "shutdown", None)
                    if callable(fallback):
                        fallback()
            else:
                fallback = getattr(engine, "shutdown", None)
                if callable(fallback):
                    fallback()
    except Exception as exc:
        print(
            f"[vllm-shutdown] shutdown failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    try:
        del llm
    except Exception:
        pass

    gc.collect()

    # Kill vLLM multiprocessing worker subprocesses that hold GPU memory
    _kill_vllm_workers()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        for device_idx in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(device_idx):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        # Worker subprocesses need a moment to release device allocations.
        time.sleep(2.0)
        gc.collect()


def _kill_vllm_workers() -> None:
    """Kill any lingering vLLM worker subprocesses that hold GPU memory."""
    import os
    import signal

    try:
        import psutil

        current_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline)
                if (
                    proc.pid != current_pid
                    and any(
                        kw in cmdline_str
                        for kw in ("VLLM::Worker", "multiproc_executor", "EngineCore")
                    )
                ):
                    print(
                        f"[vllm-shutdown] killing vLLM worker PID={proc.pid}",
                        flush=True,
                    )
                    os.kill(proc.pid, signal.SIGKILL)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
    except ImportError:
        print("[vllm-shutdown] psutil not available, cannot kill worker processes", flush=True)


def wait_for_gpu_memory(
    *,
    min_free_gib: float = 10.0,
    timeout_s: float = 120.0,
    poll_s: float = 2.0,
) -> None:
    """Block until each visible GPU has at least min_free_gib free (best effort)."""
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    min_free_bytes = int(min_free_gib * (1024**3))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        free_per_device = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        if free_per_device and min(free_per_device) >= min_free_bytes:
            return
        time.sleep(poll_s)

    free_per_device = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    if free_per_device:
        print(
            "[vllm-shutdown] warning: GPU memory still low after "
            f"{timeout_s:.0f}s (min free "
            f"{min(b / (1024**3) for b in free_per_device):.2f} GiB, "
            f"wanted {min_free_gib:.2f} GiB)",
            flush=True,
        )
