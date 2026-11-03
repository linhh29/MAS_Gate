#!/usr/bin/env python3
"""Lane-local fresh lifecycle manager for the external Qwen3-MoE TP8 service.

The PPU/vLLM 0.10 stack can corrupt Qwen3-MoE weights after either level-1 or
level-2 sleep/wake.  Training therefore starts one fresh process at the reward
batch boundary, verifies deterministic semantics, keeps it awake for sub-agent,
BCP judge and Qwen-J phases, and terminates it before BGE/actor phases.

Every run owns a distinct runtime directory, port, process group and log.  PID
identity includes /proc start ticks so a stale pid file can never kill another
lane or a recycled process.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import urllib.request


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: str = "0") -> bool:
    return _env(name, default).strip().lower() in {"1", "true", "yes", "on"}


def runtime_dir() -> Path:
    value = _env("MAS_GATE_QWEN_PHASE_RUNTIME_DIR", "")
    if not value:
        run_id = _env("RUN_ID", str(os.getppid()))
        value = f"/tmp/mas_gate_qwen_phase_{run_id}"
    return Path(value)


def metadata_path() -> Path:
    return runtime_dir() / "service.json"


def _proc_start_ticks(pid: int) -> int | None:
    try:
        # comm may contain spaces/parentheses; fields after the last ')' start at state.
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(tail[19])  # field 22 overall: starttime
    except Exception:
        return None


def _alive(meta: dict) -> bool:
    try:
        pid = int(meta["pid"])
        expected_ticks = int(meta["start_ticks"])
    except Exception:
        return False
    return _proc_start_ticks(pid) == expected_ticks


def _read_meta() -> dict:
    try:
        obj = json.loads(metadata_path().read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_meta(meta: dict) -> None:
    path = metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _url_ready(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout):
            return True
    except Exception:
        return False


def build_command() -> list[str]:
    model_path = _env("SUB_AGENT_MODEL_PATH", _env("MAS_GATE_SUB_AGENT_MODEL_PATH", ""))
    served_name = _env("SUB_AGENT_SERVED_NAME", _env("MAS_GATE_SUB_AGENT_MODEL", "Qwen3-30B-A3B-Instruct-2507"))
    if not model_path:
        raise RuntimeError("SUB_AGENT_MODEL_PATH/MAS_GATE_SUB_AGENT_MODEL_PATH is empty")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_name,
        "--host", "127.0.0.1",
        "--port", _env("SUB_AGENT_PORT", "8000"),
        "--tensor-parallel-size", _env("SUB_AGENT_TP", "8"),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", _env("MAS_GATE_QWEN_PHASE_GPU_MEMORY_UTILIZATION", _env("SUB_AGENT_GPU_MEMORY_UTILIZATION", "0.72")),
        "--max-model-len", _env("SUB_AGENT_MAX_MODEL_LEN", _env("MAS_GATE_SUB_AGENT_MAX_MODEL_LEN", "32768")),
        "--max-num-seqs", _env("SUB_AGENT_VLLM_MAX_NUM_SEQS", _env("SUB_AGENT_MAX_NUM_SEQS", "128")),
        "--max-num-batched-tokens", _env("MAS_GATE_QWEN_PHASE_MAX_NUM_BATCHED_TOKENS", _env("SUB_AGENT_MAX_NUM_BATCHED_TOKENS", "32768")),
        "--trust-remote-code",
    ]
    if _bool("SUB_AGENT_DISABLE_CUSTOM_ALL_REDUCE", "1"):
        cmd.append("--disable-custom-all-reduce")
    if _bool("SUB_AGENT_ENFORCE_EAGER", "0"):
        cmd.append("--enforce-eager")
    if _bool("SUB_AGENT_ENABLE_PREFIX_CACHING", "0"):
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")
    if _bool("SUB_AGENT_ENABLE_CHUNKED_PREFILL", "1"):
        cmd.append("--enable-chunked-prefill")
    partial = int(_env("SUB_AGENT_MAX_NUM_PARTIAL_PREFILLS", "0"))
    long_partial = int(_env("SUB_AGENT_MAX_LONG_PARTIAL_PREFILLS", "0"))
    threshold = int(_env("SUB_AGENT_LONG_PREFILL_TOKEN_THRESHOLD", "0"))
    if partial > 0:
        cmd += ["--max-num-partial-prefills", str(partial)]
    if long_partial > 0:
        if partial <= 0 or long_partial > partial:
            raise RuntimeError("SUB_AGENT_MAX_LONG_PARTIAL_PREFILLS must be <= positive SUB_AGENT_MAX_NUM_PARTIAL_PREFILLS")
        cmd += ["--max-long-partial-prefills", str(long_partial)]
    if threshold > 0:
        cmd += ["--long-prefill-token-threshold", str(threshold)]
    if _bool("SUB_AGENT_DISABLE_LOG_REQUESTS", "1"):
        cmd.append("--disable-log-requests")
    # Qwen3-14B has 40 attention heads: TP8 is valid; split two replicas
    # across 16 GPUs. Fresh phase service must preserve same DP contract as
    # direct launcher, otherwise it silently falls back to one replica.
    dp = int(_env("SUB_AGENT_DATA_PARALLEL_SIZE", "1"))
    if dp > 1:
        cmd += ["--data-parallel-size", str(dp)]
        cmd += ["--data-parallel-size-local", _env("SUB_AGENT_DATA_PARALLEL_SIZE_LOCAL", str(dp))]
        if _env("VLLM_USE_V1", "1") == "1":
            cmd += ["--api-server-count", _env("SUB_AGENT_API_SERVER_COUNT", str(dp))]
    # Deliberately never add --enable-sleep-mode.  This process is fresh-loaded
    # and terminated; allowing /sleep would reintroduce the corrupted path.
    return cmd


def _tail(path: Path, lines: int = 120) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except Exception as exc:
        return f"<unable to read {path}: {exc}>"


def _spawn_detached(cmd: list[str], env: dict[str, str], log_path: Path) -> tuple[int, int]:
    """Start a double-forked service and return its PID and process-group ID."""
    read_fd, write_fd = os.pipe()
    launcher_pid = os.fork()
    if launcher_pid == 0:
        os.close(read_fd)
        try:
            os.setsid()
            service_pid = os.fork()
            if service_pid > 0:
                os.write(write_fd, f"{service_pid}:{os.getpid()}".encode("ascii"))
                os.close(write_fd)
                os._exit(0)

            os.close(write_fd)
            log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.dup2(log_fd, 1)
                os.dup2(log_fd, 2)
            finally:
                if log_fd > 2:
                    os.close(log_fd)
            devnull_fd = os.open(os.devnull, os.O_RDONLY)
            try:
                os.dup2(devnull_fd, 0)
            finally:
                if devnull_fd > 2:
                    os.close(devnull_fd)
            os.execvpe(cmd[0], cmd, env)
        except BaseException as exc:
            try:
                os.write(write_fd, f"ERROR:{type(exc).__name__}:{exc}".encode("utf-8", errors="replace"))
            except OSError:
                pass
            os._exit(127)

    os.close(write_fd)
    _, launcher_status = os.waitpid(launcher_pid, 0)
    payload = os.read(read_fd, 4096).decode("utf-8", errors="replace")
    os.close(read_fd)
    if not os.WIFEXITED(launcher_status) or os.WEXITSTATUS(launcher_status) != 0:
        raise RuntimeError(f"detached Qwen launcher failed: {payload or launcher_status}")
    try:
        service_pid_text, process_group_text = payload.split(":", 1)
        return int(service_pid_text), int(process_group_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid detached Qwen launcher response: {payload!r}") from exc


def stop_service(*, quiet: bool = False) -> int:
    meta = _read_meta()
    if not meta:
        return 0
    if not _alive(meta):
        if not quiet:
            print(f"[qwen-phase] stale metadata only; not killing pid={meta.get('pid')}", flush=True)
        metadata_path().unlink(missing_ok=True)
        return 0
    pid = int(meta["pid"])
    timeout = float(_env("MAS_GATE_QWEN_PHASE_STOP_TIMEOUT", "90"))
    if not quiet:
        print(f"[qwen-phase] terminating fresh Qwen process group pid={pid} port={meta.get('port')}", flush=True)
    try:
        os.killpg(int(meta.get("process_group", pid)), signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + timeout
    while time.time() < deadline and _alive(meta):
        time.sleep(0.5)
    if _alive(meta):
        print(f"[qwen-phase] SIGTERM timeout; SIGKILL process group pid={pid}", file=sys.stderr, flush=True)
        try:
            os.killpg(int(meta.get("process_group", pid)), signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.time() + 20
        while time.time() < deadline and _alive(meta):
            time.sleep(0.25)
    if _alive(meta):
        raise RuntimeError(f"Qwen phase process group pid={pid} did not terminate")
    # The active log lives on node-local storage.  Only after all writers are
    # dead do we best-effort snapshot it to the durable checkpoint filesystem,
    # avoiding the PCache/FUSE EIO seen when appending live files there.
    durable_dir_value = _env("MAS_GATE_QWEN_PHASE_DURABLE_LOG_DIR", "").strip()
    if durable_dir_value and meta.get("log"):
        try:
            durable_dir = Path(durable_dir_value)
            durable_dir.mkdir(parents=True, exist_ok=True)
            source = Path(str(meta["log"]))
            if source.is_file():
                shutil.copy2(source, durable_dir / source.name)
            report = durable_dir / f"service_{meta.get('generation', int(time.time() * 1000))}.json"
            report.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            print(f"[qwen-phase] WARNING: durable closed-log snapshot failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    metadata_path().unlink(missing_ok=True)
    port = int(meta.get("port", _env("SUB_AGENT_PORT", "8000")))
    deadline = time.time() + 30
    while time.time() < deadline and _url_ready(port, timeout=1):
        time.sleep(0.5)
    if _url_ready(port, timeout=1):
        raise RuntimeError(f"Qwen phase port {port} still responds after pid={pid} terminated")
    return 0


def start_service() -> int:
    # A fresh start is intentional even if a prior batch leaked a healthy server.
    stop_service(quiet=True)
    rt = runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    port = int(_env("SUB_AGENT_PORT", "8000"))
    if _url_ready(port, timeout=1):
        raise RuntimeError(f"refusing to start: unowned service already responds on lane-local port {port}")
    generation = int(time.time() * 1000)
    # Never write an active vLLM log through PCache/FUSE.  This directory must
    # be node-local; the launcher pins it below the run's unique /tmp runtime.
    live_log_dir = Path(_env("MAS_GATE_QWEN_PHASE_LOG_DIR", str(rt / "logs")))
    live_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = live_log_dir / f"qwen30b_phase_fresh_{generation}.log"
    cmd = build_command()
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = _env("SUB_AGENT_GPUS", _env("ALL_GPUS", "0,1,2,3,4,5,6,7"))
    child_env["VLLM_USE_V1"] = _env("SUB_AGENT_VLLM_USE_V1", _env("VLLM_USE_V1", "1"))
    child_env["VLLM_SERVER_DEV_MODE"] = "1"
    # Qwen-J prompt_logprobs materializes a large fp32 log-softmax workspace.
    # Keep allocator segments expandable to reduce fragmentation at that peak;
    # the dedicated 0.72 KV-cache cap below still provides the real safety margin.
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    service_pid, process_group = _spawn_detached(cmd, child_env, log_path)
    start_ticks = None
    for _ in range(100):
        start_ticks = _proc_start_ticks(service_pid)
        if start_ticks is not None:
            break
        time.sleep(0.01)
    if start_ticks is None:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise RuntimeError(f"cannot identify fresh Qwen pid={service_pid}")
    meta = {
        "pid": service_pid,
        "process_group": process_group,
        "start_ticks": start_ticks,
        "port": port,
        "model": _env("SUB_AGENT_SERVED_NAME", _env("MAS_GATE_SUB_AGENT_MODEL", "Qwen3-30B-A3B-Instruct-2507")),
        "gpu_list": child_env["CUDA_VISIBLE_DEVICES"],
        "log": str(log_path),
        "command": cmd,
        "started_at": time.time(),
        "generation": generation,
    }
    _write_meta(meta)
    timeout = float(_env("MAS_GATE_QWEN_PHASE_START_TIMEOUT", "900"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(meta):
            metadata_path().unlink(missing_ok=True)
            raise RuntimeError(f"fresh Qwen exited before readiness; log={log_path}\n{_tail(log_path)}")
        if _url_ready(port):
            break
        time.sleep(2)
    else:
        stop_service(quiet=True)
        raise RuntimeError(f"fresh Qwen did not become ready within {timeout:g}s; log={log_path}\n{_tail(log_path)}")

    health_script = Path(_env("MAS_GATE_ROOT", ".")) / "verl" / "mas_gate_rl" / "vllm_semantic_health.py"
    health_cmd = [
        sys.executable, str(health_script),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--model", str(meta["model"]),
        "--role", "qwen_shared_subagent_judge_qwenj",
        "--stage", f"fresh_generation_{generation}",
    ]
    health = subprocess.run(health_cmd, text=True, capture_output=True, timeout=600)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n[qwen-phase semantic gate stdout]\n" + health.stdout)
            log.write("\n[qwen-phase semantic gate stderr]\n" + health.stderr)
    except Exception:
        # Never leak a full-memory TP8 process when post-gate bookkeeping fails.
        stop_service(quiet=True)
        raise
    if health.stdout:
        print(health.stdout.strip(), flush=True)
    if health.returncode != 0:
        stop_service(quiet=True)
        raise RuntimeError(
            f"fresh Qwen semantic gate failed rc={health.returncode}; log={log_path}; stderr={health.stderr.strip()}"
        )
    meta["semantic_health_passed_at"] = time.time()
    _write_meta(meta)
    print(
        f"[qwen-phase] fresh semantic-healthy Qwen ready pid={service_pid} port={port} "
        f"gpus={meta['gpu_list']} log={log_path}",
        flush=True,
    )
    return 0


def status_service() -> int:
    meta = _read_meta()
    payload = {"runtime_dir": str(runtime_dir()), "metadata": meta, "alive": _alive(meta) if meta else False}
    if meta:
        payload["ready"] = _url_ready(int(meta.get("port", 0)))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("alive") else 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "stop", "status"))
    args = parser.parse_args()
    rt = runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    lock_path = rt / "service.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if args.action == "start":
            return start_service()
        if args.action == "stop":
            return stop_service()
        return status_service()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[qwen-phase] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
