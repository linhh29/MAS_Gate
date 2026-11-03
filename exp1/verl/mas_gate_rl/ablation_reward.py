"""Opt-in MAS-Gate reward ablations.

The default reward implementation never imports this module.  The Skywork
model is deliberately accessed through one local HTTP process so Ray reward
workers do not each instantiate an 8B model.
"""
from __future__ import annotations

import json
import http.client
import math
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ADDITIVE_TAG = "MAS_GATE_ABLATION_ADDITIVE_I_U_PLUS_1_PLUS_K"
SKYWORK_TAG = "MAS_GATE_ABLATION_SKYWORK_REWARD_U"
SKYWORK_MODEL_PATH = os.environ.get(
    "MAS_GATE_SKYWORK_MODEL_PATH",
    "./models/Skywork-Reward-V2-Qwen3-8B",
)
_SKYWORK_MAX_CONCURRENCY = max(1, int(os.environ.get("MAS_GATE_SKYWORK_MAX_CONCURRENCY", "2")))
_SKYWORK_SEMAPHORE = threading.BoundedSemaphore(_SKYWORK_MAX_CONCURRENCY)


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite: {value!r}")
    return value


def impact_from_u_k(u: float, k: float, tag: str = "") -> tuple[float, str]:
    """Return the explicitly selected impact formula; default is multiplication."""
    u, k = _finite(u, "U"), _finite(k, "K")
    if tag == ADDITIVE_TAG:
        return u + (1.0 + k), "I=U+(1+K)"
    return u * (1.0 + k), "I=U*(1+K)"


def _skywork_request(question: str, answer: str) -> float:
    port = os.environ.get("MAS_GATE_SKYWORK_REWARD_PORT", "9480")
    # vLLM Score API accepts paired inputs and performs batched GPU inference.
    # Keep a legacy JSON fallback for the old CPU shim during controlled rollback.
    api = os.environ.get("MAS_GATE_SKYWORK_REWARD_API", "vllm")
    if api == "legacy":
        payload = json.dumps({"question": question, "answer": answer}).encode()
        path = "/score"
    else:
        payload = json.dumps({"text_1": str(question), "text_2": str(answer)}).encode()
        path = "/score"
    url = f"http://127.0.0.1:{port}{path}"
    timeout = float(os.environ.get("MAS_GATE_SKYWORK_REWARD_TIMEOUT", "300"))
    attempts = max(1, int(os.environ.get("MAS_GATE_SKYWORK_REWARD_RETRIES", "8")))
    delay = max(0.0, float(os.environ.get("MAS_GATE_SKYWORK_RETRY_INITIAL_SECONDS", "2")))
    max_delay = max(delay, float(os.environ.get("MAS_GATE_SKYWORK_RETRY_MAX_SECONDS", "30")))
    retry_codes = {429, 500, 502, 503, 504}
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with _SKYWORK_SEMAPHORE:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    obj = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in retry_codes or attempt == attempts:
                raise
            error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.RemoteDisconnected, OSError) as exc:
            if attempt == attempts:
                raise
            error = f"{type(exc).__name__}: {exc}"
        sleep_for = min(delay, max_delay)
        print(
            f"[mas_gate_rl.ablation] Skywork request failed ({error}); "
            f"retry {attempt}/{attempts} in {sleep_for:.1f}s",
            flush=True,
        )
        if sleep_for:
            time.sleep(sleep_for)
        delay = min(max_delay, max(delay * 2, 0.1))
    if api == "legacy":
        value = obj["score"]
    else:
        data = obj.get("data") or []
        if not data or "score" not in data[0]:
            raise ValueError(f"vLLM score response missing data.score: {obj!r}")
        value = data[0]["score"]
    return _finite(value, "Skywork raw score")


def skywork_u(question: str, answer: str) -> tuple[float, float]:
    """Score an answer and map the unbounded BT score to stable U in [-1, 1]."""
    raw = _skywork_request(question, answer)
    # Skywork scores are Bradley-Terry logits and are not probability-bounded.
    # tanh is a deterministic, monotone normalization suitable for U.
    scale = float(os.environ.get("MAS_GATE_SKYWORK_SCORE_SCALE", "10.0"))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("MAS_GATE_SKYWORK_SCORE_SCALE must be positive")
    return raw, math.tanh(raw / scale)


def apply_ablation(result: dict[str, Any], *, question: str, answer: str, tag: str) -> dict[str, Any]:
    """Rewrite only the final dense score under an explicit ablation tag."""
    if tag not in (ADDITIVE_TAG, SKYWORK_TAG):
        return result
    out = dict(result)
    quant = dict(out.get("quant_result") or {})
    k = quant.get("mean_K", quant.get("mean_k", 0.0))
    # Canonical quantification already exposes mean_U and mean_K.  For the
    # Skywork arm only U changes; K remains the canonical BGE propagation term.
    if tag == SKYWORK_TAG:
        raw, u = skywork_u(question, answer)
        out["skywork_raw_score"] = raw
        out["skywork_normalized_U"] = u
        out["skywork_model_path"] = SKYWORK_MODEL_PATH
    else:
        u = quant.get("mean_U", quant.get("mean_u"))
        if u is None:
            # Preserve the canonical score if a terminal/error result has no U.
            return out
        u = _finite(u, "mean_U")
        raw = None
    impact, formula = impact_from_u_k(u, k, tag)
    out["score"] = impact
    out["metric_key"] = "mean_impact"
    quant["mean_impact"] = impact
    quant["ablation_U"] = u
    quant["ablation_K"] = _finite(k, "mean_K")
    quant["impact_formula"] = formula
    quant["impact_formula_tag"] = tag
    if raw is not None:
        quant["skywork_raw_score"] = raw
        quant["skywork_model_path"] = SKYWORK_MODEL_PATH
    out["quant_result"] = quant
    print(f"[mas_gate_rl.ablation] tag={tag} formula={formula} U={u:.6g} K={float(k):.6g} I={impact:.6g} model={SKYWORK_MODEL_PATH if raw is not None else 'canonical'}", flush=True)
    return out
