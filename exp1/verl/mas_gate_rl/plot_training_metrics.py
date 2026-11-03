#!/usr/bin/env python3
"""Create incremental MAS-Gate GRPO training charts from verl console logs.

The script is deliberately self-contained and writes only CSV/JSON/PNG files.  It
can read a normal log, a checkpoint ``training_metrics.jsonl``, a tmux pane
(recommended while training is alive), or any combination of these sources.
Rows are keyed by training/global_step and the most complete/latest row wins.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FLOAT_RE = r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)"
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-")
PAIR_RE = re.compile(rf"^\s*([^:]+):\s*({FLOAT_RE})", re.I)

PLOTS = {
    "losses.png": [
        ("Actor objective", ["actor/loss", "actor/pg_loss"]),
        ("KL loss", ["actor/kl_loss"]),
        ("Optimization", ["actor/grad_norm", "actor/entropy"]),
        ("Clipping", ["actor/pg_clipfrac", "actor/pg_clipfrac_lower"]),
    ],
    "kl.png": [
        ("Policy KL", ["rollout_corr/kl", "rollout_corr/k3_kl"]),
        ("Actor KL terms", ["actor/kl_loss", "actor/ppo_kl"]),
        ("PPL ratio", ["rollout_corr/ppl_ratio"]),
        ("Probability drift", ["training/rollout_probs_diff_mean", "training/rollout_probs_diff_std", "training/rollout_probs_diff_max"]),
    ],
    "rewards.png": [
        ("Raw reward components", ["reward_fusion/dense_raw_mean", "reward_fusion/correctness_raw_mean"]),
        ("Fused reward / contribution", ["reward_fusion/fused_pre_grpo_std", "reward_fusion/correctness_abs_contribution_share", "reward_fusion/mixed_correctness_abs_contribution_share"]),
        ("Informative GRPO groups", ["reward_fusion/mixed_group_ratio", "reward_fusion/zero_dense_variance_group_ratio", "grpo/group_reward_variance/zero_ratio"]),
        ("GRPO reward range", ["critic/rewards/min", "critic/rewards/mean", "critic/rewards/max"]),
    ],
    "performance.png": [
        ("Step phase time (s)", ["timing_s/gen", "timing_s/reward", "timing_s/old_log_prob", "timing_s/ref", "timing_s/update_actor"]),
        ("Total step time (s)", ["timing_s/step", "perf/time_per_step"]),
        ("Tokens / throughput", ["perf/total_num_tokens", "perf/throughput"]),
        ("Actor memory (GiB)", ["actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb"]),
    ],
    "lengths.png": [
        ("Response length", ["response_length/min", "response_length/mean", "response_length/max"]),
        ("Prompt length", ["prompt_length/min", "prompt_length/mean", "prompt_length/max"]),
        ("Response clipping/abort", ["response_length/clip_ratio", "response/aborted_ratio"]),
        ("Sequence tokens", ["global_seqlen/min", "global_seqlen/mean", "global_seqlen/max"]),
    ],
}

DASHBOARD = [
    ("Actor loss", ["actor/loss", "actor/pg_loss"]),
    ("KL", ["rollout_corr/kl", "rollout_corr/k3_kl"]),
    ("Reward components", ["reward_fusion/dense_raw_mean", "reward_fusion/correctness_raw_mean"]),
    ("Reward fusion", ["reward_fusion/fused_pre_grpo_std", "reward_fusion/mixed_group_ratio", "reward_fusion/correctness_abs_contribution_share"]),
    ("Entropy / grad norm", ["actor/entropy", "actor/grad_norm"]),
    ("Step timing", ["timing_s/reward", "timing_s/gen", "timing_s/update_actor", "timing_s/step"]),
    ("Lengths", ["response_length/mean", "prompt_length/mean"]),
    ("Rollout drift", ["training/rollout_probs_diff_mean", "training/rollout_probs_diff_std"]),
    ("Actor memory GiB", ["actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb"]),
]


def _float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def parse_console(texts: Iterable[str]) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for text in texts:
        for line in text.splitlines():
            sm = STEP_RE.search(line)
            if not sm:
                continue
            step = int(sm.group(1))
            row: dict[str, float] = {"step": float(step)}
            # verl separates scalar fields with " - "; parsing only at segment
            # starts avoids accidentally treating timestamps/progress text as metrics.
            for segment in line[sm.end():].split(" - "):
                pm = PAIR_RE.match(segment)
                if pm:
                    key = pm.group(1).strip()
                    row[key] = _float(pm.group(2))
            if "training/global_step" in row:
                step = int(row["training/global_step"])
                row["step"] = float(step)
            old = by_step.get(step)
            if old is None or len(row) >= len(old):
                by_step[step] = row
    return [by_step[k] for k in sorted(by_step)]


def parse_metrics_jsonl(texts: Iterable[str]) -> list[dict[str, float]]:
    """Parse checkpoint metric history rows shaped as ``{"step": N, "data": {...}}``.

    Non-numeric metadata is intentionally excluded from plotting. Malformed JSON
    or an invalid row shape fails loudly instead of silently producing incomplete
    training diagnostics.
    """
    by_step: dict[int, dict[str, float]] = {}
    for source_index, text in enumerate(texts):
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid metrics JSONL source {source_index} line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
                raise ValueError(
                    f"invalid metrics JSONL source {source_index} line {line_number}: expected object with data object"
                )
            try:
                step = int(payload["step"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid metrics JSONL source {source_index} line {line_number}: missing integer step"
                ) from exc
            if step < 0:
                raise ValueError(
                    f"invalid metrics JSONL source {source_index} line {line_number}: negative step"
                )
            row: dict[str, float] = {"step": float(step)}
            for key, value in payload["data"].items():
                if not isinstance(key, str) or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric = float(value)
                    if math.isfinite(numeric):
                        row[key] = numeric
            old = by_step.get(step)
            if old is None or len(row) >= len(old):
                by_step[step] = row
    return [by_step[k] for k in sorted(by_step)]


def merge_metric_rows(*collections: Iterable[dict[str, float]]) -> list[dict[str, float]]:
    """Merge metric sources by step, preferring the row with more scalar fields."""
    by_step: dict[int, dict[str, float]] = {}
    for rows in collections:
        for row in rows:
            step = int(row["step"])
            old = by_step.get(step)
            if old is None or len(row) >= len(old):
                by_step[step] = row
    return [by_step[k] for k in sorted(by_step)]


def capture_tmux(session: str) -> str:
    cp = subprocess.run(
        ["tmux", "capture-pane", "-pJt", session, "-S", "-"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
    )
    if cp.returncode:
        raise RuntimeError(cp.stderr.strip() or f"tmux capture failed for {session}")
    return cp.stdout


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = ["step"] + sorted({k for r in rows for k in r if k != "step"})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: ("" if not math.isfinite(v) else int(v) if k == "step" else v) for k, v in row.items()})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def values(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([r.get(key, math.nan) for r in rows], dtype=float)


def rolling(y: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        a = y[max(0, i-window+1):i+1]
        a = a[np.isfinite(a)]
        if len(a): out[i] = a.mean()
    return out


def draw_panel(ax, rows: list[dict[str, float]], title: str, keys: list[str], ema: bool = True) -> None:
    x = values(rows, "step")
    any_data = False
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, key in enumerate(keys):
        y = values(rows, key)
        good = np.isfinite(y)
        if not good.any():
            continue
        any_data = True
        color = colors[i % len(colors)]
        ax.plot(x[good], y[good], marker="o", markersize=2.8, linewidth=0.8, alpha=.35, color=color, label=key)
        if ema and good.sum() >= 3:
            sm = rolling(y, min(5, max(2, len(y))))
            g2 = np.isfinite(sm)
            ax.plot(x[g2], sm[g2], linewidth=1.8, color=color, label=f"{key} (MA5)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("global step")
    ax.grid(True, alpha=.22)
    if any_data:
        ax.legend(fontsize=6.8, loc="best")
    else:
        ax.text(.5, .5, "metric not emitted yet", ha="center", va="center", transform=ax.transAxes, color="gray")


def atomic_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(tmp, dpi=155, bbox_inches="tight", format="png")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
        plt.close(fig)


def render_grid(rows, panels, path: Path, title: str, shape: tuple[int, int]) -> None:
    fig, axes = plt.subplots(*shape, figsize=(16, 4.1 * shape[0]), squeeze=False)
    for ax, (panel_title, keys) in zip(axes.flat, panels):
        draw_panel(ax, rows, panel_title, keys)
    for ax in list(axes.flat)[len(panels):]: ax.axis("off")
    last = int(rows[-1]["step"])
    fig.suptitle(f"{title} — steps {int(rows[0]['step'])}–{last} (raw + 5-step moving average)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .975))
    atomic_figure(fig, path)


def parse_gpu_log(path: Path) -> list[dict[str, float | str]]:
    if not path.is_file(): return []
    ts = None; samples: list[dict[str, float | str]] = []; gpus = []
    ts_re = re.compile(r"^=====\s+(.+?)\s+=====$")
    gpu_re = re.compile(r"^\[(\d+)\].*?\|.*?(\d+)\s*%\s*\|\s*(\d+)\s*/\s*(\d+)\s*MB")
    def flush():
        nonlocal gpus
        selected = [g for g in gpus if 0 <= g[0] <= 7]
        if ts and selected:
            samples.append({"timestamp": ts, "gpu_avg_util": sum(g[1] for g in selected)/len(selected),
                            "gpu_max_util": max(g[1] for g in selected),
                            "gpu_mem_used_mib": sum(g[2] for g in selected),
                            "gpu_mem_total_mib": sum(g[3] for g in selected)})
        gpus = []
    with path.open(errors="replace") as f:
        for line in f:
            m = ts_re.match(line.strip())
            if m: flush(); ts = m.group(1); continue
            m = gpu_re.match(line.strip())
            if m: gpus.append(tuple(map(int, m.groups())))
    flush()
    return samples


def parse_kv_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open(errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def render_system(gpu_rows, kv_rows, path: Path, title: str) -> None:
    if not gpu_rows and not kv_rows: return
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.2))
    gpu_source = gpu_rows or kv_rows
    if gpu_source:
        x = np.arange(len(gpu_source))
        def safe_num(r, key):
            try: return float(r.get(key, math.nan))
            except (ValueError, TypeError): return math.nan
        axes[0,0].plot(x, [safe_num(r, "gpu_avg_util") for r in gpu_source], label="GPU 0–7 avg util")
        axes[0,0].plot(x, [safe_num(r, "gpu_max_util") for r in gpu_source], alpha=.6, label="max util")
        axes[0,0].set_ylim(0, 105); axes[0,0].legend(fontsize=8)
        mem_pct=[]
        for r in gpu_source:
            used, total = safe_num(r, "gpu_mem_used_mib"), safe_num(r, "gpu_mem_total_mib")
            mem_pct.append(100*used/total if math.isfinite(total) and total > 0 else math.nan)
        axes[0,1].plot(x, mem_pct, label="GPU 0–7 memory %")
        axes[0,1].set_ylim(0, 105); axes[0,1].legend(fontsize=8)
    if kv_rows:
        x = np.arange(len(kv_rows))
        def col(k):
            out=[]
            for r in kv_rows:
                try: out.append(float(r.get(k, "nan")))
                except ValueError: out.append(math.nan)
            return out
        # phase_kv_*.csv uses qwen_* names. Keep legacy fallbacks so old
        # qwen30_kv exports remain renderable as well.
        sample = kv_rows[0] if kv_rows else {}
        running_key = "qwen_running" if "qwen_running" in sample else "running"
        waiting_key = "qwen_waiting" if "qwen_waiting" in sample else "waiting"
        kv_key = "qwen_kv_max" if "qwen_kv_max" in sample else "kv_usage"
        axes[1,0].plot(x, col(running_key), label="Qwen30 running")
        axes[1,0].plot(x, col(waiting_key), label="Qwen30 waiting", alpha=.75)
        axes[1,0].legend(fontsize=8)
        axes[1,1].plot(x, np.asarray(col(kv_key))*100, label="Qwen30 KV cache %")
        axes[1,1].set_ylim(bottom=0); axes[1,1].legend(fontsize=8)
    for ax, name in zip(axes.flat, ["GPU compute utilization", "GPU memory utilization", "Qwen30 concurrency", "Qwen30 KV cache utilization"]):
        ax.set_title(name, fontweight="bold"); ax.set_xlabel("monitor sample"); ax.grid(True, alpha=.22)
    fig.suptitle(f"{title} — system telemetry", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.96)); atomic_figure(fig, path)


def latest_summary(rows, gpu_rows, kv_rows, args) -> dict:
    r = rows[-1]
    keys = ["actor/loss", "actor/pg_loss", "actor/kl_loss", "actor/lr", "actor/pg_clipfrac",
            "rollout_corr/kl", "rollout_corr/k3_kl", "critic/rewards/mean", "critic/rewards/min",
            "critic/rewards/max", "actor/entropy", "actor/grad_norm",
            "reward_fusion/dense_raw_mean", "reward_fusion/correctness_raw_mean",
            "reward_fusion/fused_pre_grpo_std", "reward_fusion/mixed_group_ratio",
            "reward_fusion/correctness_abs_contribution_share",
            "reward_fusion/mixed_correctness_abs_contribution_share",
            "timing_s/gen", "timing_s/reward", "timing_s/update_actor", "timing_s/step",
            "actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb"]
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_tmux_session": args.tmux_session,
        "source_log_files": args.log_file,
        "source_metrics_jsonl": args.metrics_jsonl,
        "first_step": int(rows[0]["step"]), "latest_step": int(r["step"]), "num_steps": len(rows),
        "latest": {k: r[k] for k in keys if k in r and math.isfinite(r[k])},
    }
    if gpu_rows: result["latest_gpu"] = gpu_rows[-1]
    if kv_rows: result["latest_qwen30_kv"] = kv_rows[-1]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-file", action="append", default=[])
    ap.add_argument("--metrics-jsonl", action="append", default=[])
    ap.add_argument("--tmux-session")
    ap.add_argument("--snapshot-file")
    ap.add_argument("--gpu-log")
    ap.add_argument("--kv-csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--title", default="MAS-Gate GRPO")
    args = ap.parse_args()
    texts=[]; metric_texts=[]; source_errors=[]
    for raw in args.log_file:
        p=Path(raw)
        try: texts.append(p.read_text(errors="replace"))
        except Exception as e: source_errors.append(f"{p}: {e}")
    for raw in args.metrics_jsonl:
        p=Path(raw)
        try: metric_texts.append(p.read_text(errors="strict"))
        except Exception as e: source_errors.append(f"{p}: {e}")
    if args.snapshot_file:
        try: texts.append(Path(args.snapshot_file).read_text(errors="replace"))
        except Exception as e: source_errors.append(f"{args.snapshot_file}: {e}")
    if args.tmux_session:
        try: texts.append(capture_tmux(args.tmux_session))
        except Exception as e: source_errors.append(str(e))
    rows=merge_metric_rows(parse_console(texts), parse_metrics_jsonl(metric_texts))
    if not rows:
        raise SystemExit("No complete verl step metric rows found. " + "; ".join(source_errors))
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    atomic_csv(out/"metrics.csv", rows)
    for filename, panels in PLOTS.items(): render_grid(rows, panels, out/filename, args.title, (2,2))
    render_grid(rows, DASHBOARD, out/"training_dashboard.png", args.title, (3,3))
    gpu_rows=parse_gpu_log(Path(args.gpu_log)) if args.gpu_log else []
    kv_rows=parse_kv_csv(Path(args.kv_csv)) if args.kv_csv else []
    render_system(gpu_rows, kv_rows, out/"system_telemetry.png", args.title)
    summary=latest_summary(rows, gpu_rows, kv_rows, args)
    summary["source_errors"] = source_errors
    atomic_text(out/"summary.json", json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    latest=summary["latest"]
    print(f"[training-curves] updated={summary['generated_at']} steps={summary['first_step']}..{summary['latest_step']} rows={summary['num_steps']}")
    print("[training-curves] latest " + " ".join(f"{k}={v:.6g}" for k,v in latest.items()))
    print(f"[training-curves] output={out}")
    if source_errors: print("[training-curves] source warnings: " + "; ".join(source_errors))

if __name__ == "__main__":
    main()
