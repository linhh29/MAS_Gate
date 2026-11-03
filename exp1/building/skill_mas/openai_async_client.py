"""Async OpenAI client for Skill_MAS code planner."""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

# --- Why ``httpx.ReadError`` / mid-response failures happen (no code fix here; environment/Payload) ---
#
# - **TCP/TLS path**: HPC / NAT / firewall / proxy may RST idle sockets or drop long transfers;
#   ``ReadError`` means the HTTP layer expected more bytes on the body and the connection ended.
# - **Payload shape**: contrastive sends a very large JSON body (trajectories + embedded schema); long
#   requests + long responses increase time on the wire → more exposure to flaky links.
# - **Stack**: ``AsyncOpenAI`` uses **httpx/httpcore**, not ``urllib3``; behavior differs from Vita's
#   ``requests`` path (curl can succeed while Python fails on the same host).
# - **Mitigations** (ops): stable egress, proxy if required, reduce concurrency, smaller reflection
#   batches, or run from a node with reliable WAN — not application retries.
#
# Optional timeouts (seconds); tune if the link is slow but stable.
_DEFAULT_READ_TIMEOUT = float(os.environ.get("SKILL_MAS_OPENAI_READ_TIMEOUT", "600"))
_DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("SKILL_MAS_OPENAI_CONNECT_TIMEOUT", "120"))


class _VLLMRingLeaseClient:
    """MAS_Gate-local client for the process-wide TP8 role coordinator."""

    def __init__(self, controller_url: str, role: str):
        self.controller_url = controller_url.rstrip("/")
        self.role = role
        # Keep acquire and release traffic on separate pools.  At sample
        # concurrency 128, pending /acquire calls can legitimately occupy every
        # connection in a default 100-connection httpx pool.  If /release uses
        # that same pool, completed model calls cannot return their leases and
        # the role handoff deadlocks permanently.
        acquire_connections = max(
            128, int(os.environ.get("MAS_GATE_RING_ACQUIRE_CONNECTIONS", "1024"))
        )
        release_connections = max(
            8, int(os.environ.get("MAS_GATE_RING_RELEASE_CONNECTIONS", "128"))
        )
        acquire_timeout = float(os.environ.get("MAS_GATE_RING_ACQUIRE_TIMEOUT", "600"))
        self.acquire_http = httpx.AsyncClient(
            timeout=httpx.Timeout(acquire_timeout, connect=30.0),
            limits=httpx.Limits(
                max_connections=acquire_connections,
                max_keepalive_connections=acquire_connections,
            ),
            trust_env=False,
        )
        self.release_http = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=release_connections,
                max_keepalive_connections=release_connections,
            ),
            trust_env=False,
        )

    async def acquire(self) -> str:
        # A large sample fan-out can briefly reset localhost HTTP connections.
        # Retry only the coordinator operation; never duplicate an LLM request.
        last: Exception | None = None
        for attempt in range(4):
            try:
                response = await self.acquire_http.post(
                    f"{self.controller_url}/acquire", json={"role": self.role}
                )
                response.raise_for_status()
                lease_id = str(response.json().get("lease_id", ""))
                if not lease_id:
                    raise RuntimeError("MAS_Gate vLLM ring returned an empty lease id")
                return lease_id
            except (httpx.HTTPError, RuntimeError) as exc:
                last = exc
                if attempt < 3:
                    await asyncio.sleep(0.05 * (2 ** attempt))
        raise RuntimeError(f"MAS_Gate ring acquire failed after retries: {last}")

    async def release(self, lease_id: str) -> None:
        # Release is best-effort, but retrying prevents a transient reset from
        # leaking a lease and blocking the next role handoff.
        last: Exception | None = None
        for attempt in range(4):
            try:
                response = await self.release_http.post(
                    f"{self.controller_url}/release", json={"lease_id": lease_id}
                )
                response.raise_for_status()
                return
            except httpx.HTTPError as exc:
                last = exc
                if attempt < 3:
                    await asyncio.sleep(0.05 * (2 ** attempt))
        # Do not turn a successful model response into an evaluation failure.
        print(f"[MAS_Gate ring] release failed (lease={lease_id[:8]}): {last}", flush=True)

    async def close(self) -> None:
        await asyncio.gather(
            self.acquire_http.aclose(),
            self.release_http.aclose(),
        )

MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.json"


@lru_cache(maxsize=4)
def _mas_gate_local_tokenizer(model_path: str):
    """Load only a local tokenizer; never resolve a remote model URL."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )


def _mas_gate_cap_completion_to_context(
    messages: list[dict[str, Any]],
    requested: int | None,
    diagnostics: MutableMapping[str, Any] | None = None,
    *,
    max_model_len: int | None = None,
    max_tokens: int | None = None,
    min_completion_tokens: int | None = None,
    context_safety_tokens: int | None = None,
    model_path: str | None = None,
) -> tuple[int | None, int | None]:
    """Fit a local Qwen30B request into its vLLM context window.

    Later MAS nodes can inherit several verbose upstream answers.  Merely reducing
    ``max_tokens`` can leave only a few output tokens (or make the prompt itself
    illegal).  For MAS-Gate's local Qwen30B only, compact an oversized *user*
    message in tokenizer space while retaining both ends: the beginning contains
    role/global-task context and the end contains dependency conclusions/current
    task.  This guarantees a useful minimum completion budget without changing
    the server's operator-requested 8192-token context limit.

    ``messages`` is intentionally mutated before the OpenAI request is sent.
    """
    diag = diagnostics if diagnostics is not None else {}
    diag.update({
        "requested_max_tokens": requested,
        "context_compacted": False,
        "prompt_tokens_before_compaction": None,
        "prompt_tokens_after_compaction": None,
        "user_tokens_before_compaction": None,
        "user_tokens_after_compaction": None,
        "user_tokens_removed": 0,
    })
    if requested is None:
        return None, None
    context = int(
        max_model_len
        if max_model_len is not None
        else os.environ.get("MAS_GATE_SUB_AGENT_MAX_MODEL_LEN", "32768")
    )
    safety = max(
        16,
        int(
            context_safety_tokens
            if context_safety_tokens is not None
            else os.environ.get("MAS_GATE_SUB_AGENT_CONTEXT_SAFETY_TOKENS", "128")
        ),
    )
    configured_cap = max(
        1,
        int(
            max_tokens
            if max_tokens is not None
            else os.environ.get("MAS_GATE_SUB_AGENT_MAX_TOKENS", "8192")
        ),
    )
    requested_output = min(max(1, int(requested)), configured_cap)
    configured_min_completion = max(
        1,
        int(
            min_completion_tokens
            if min_completion_tokens is not None
            else os.environ.get("MAS_GATE_SUB_AGENT_MIN_COMPLETION_TOKENS", str(requested_output))
        ),
    )
    # Reserve the request-specific completion budget.  Interactive action calls
    # deliberately request only 128 tokens; forcing every one of them to reserve
    # the server-wide 8192-token maximum caused needless prompt compaction and
    # reduced vLLM batching capacity.  Normal 8192-token calls retain the full
    # reservation unchanged.
    min_completion = min(requested_output, configured_min_completion)
    diag.update({
        "context_max_model_len": context,
        "context_safety_tokens": safety,
        "min_completion_tokens": min_completion,
        "configured_max_tokens": configured_cap,
    })
    if context <= safety + min_completion:
        raise ValueError(
            "MAS-Gate Qwen30B context cannot reserve the configured completion budget: "
            f"max_model_len={context}, safety={safety}, min_completion={min_completion}"
        )

    model_path = model_path or os.environ.get(
        "MAS_GATE_SUB_AGENT_MODEL_PATH", "/tmp/Qwen3-30B-A3B"
    )
    tok = None
    try:
        tok = _mas_gate_local_tokenizer(model_path)
        prompt_tokens = len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    except Exception:
        # Conservative text-only fallback; this path cannot safely rewrite in
        # token space, so retain the old estimate and fail clearly if necessary.
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        prompt_tokens = chars // 3 + 256

    diag["prompt_tokens_before_compaction"] = prompt_tokens
    target_prompt_tokens = context - safety - min_completion
    if prompt_tokens > target_prompt_tokens and tok is None:
        raise ValueError(
            "MAS-Gate cannot verify the pure output budget without the local tokenizer: "
            f"prompt_tokens_estimate={prompt_tokens}, target_prompt_tokens={target_prompt_tokens}, "
            f"reserved_output_tokens={min_completion}"
        )
    if prompt_tokens > target_prompt_tokens and tok is not None:
        user_idx = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        if user_idx is None:
            raise ValueError(
                f"MAS-Gate Qwen30B prompt has no compactable user message: prompt_tokens={prompt_tokens}"
            )
        original = str(messages[user_idx].get("content", ""))
        content_ids = tok.encode(original, add_special_tokens=False)
        diag["user_tokens_before_compaction"] = len(content_ids)
        remove_at_least = prompt_tokens - target_prompt_tokens
        keep = max(64, len(content_ids) - remove_at_least - 32)
        marker = "\n\n[MAS-GATE CONTEXT COMPACTION: middle of verbose upstream context omitted]\n\n"

        # Preserve more of the tail because generated workflow prompts place the
        # current sub-task and latest dependency conclusions there.
        while keep >= 64:
            head_n = max(24, int(keep * 0.35))
            tail_n = max(40, keep - head_n)
            compact = tok.decode(content_ids[:head_n], skip_special_tokens=True)
            compact += marker
            compact += tok.decode(content_ids[-tail_n:], skip_special_tokens=True)
            messages[user_idx]["content"] = compact
            prompt_tokens = len(
                tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            )
            if prompt_tokens <= target_prompt_tokens:
                compacted_user_tokens = len(
                    tok.encode(str(messages[user_idx].get("content", "")), add_special_tokens=False)
                )
                diag.update({
                    "context_compacted": True,
                    "user_tokens_after_compaction": compacted_user_tokens,
                    "user_tokens_removed": max(0, len(content_ids) - compacted_user_tokens),
                })
                if os.environ.get("MAS_GATE_VERBOSE_SUBAGENT", "0").strip().lower() in {"1", "true", "yes", "on"}:
                    print(
                        "[MAS-Gate context] compacted Qwen30B prompt "
                        f"from user_tokens={len(content_ids)} to prompt_tokens={prompt_tokens}; "
                        f"reserved_completion>={min_completion}",
                        flush=True,
                    )
                break
            keep -= max(32, prompt_tokens - target_prompt_tokens + 16)
        else:
            messages[user_idx]["content"] = original
            raise ValueError(
                "MAS-Gate Qwen30B prompt cannot be compacted safely: "
                f"prompt_tokens={prompt_tokens}, target={target_prompt_tokens}"
            )

    diag["prompt_tokens_after_compaction"] = prompt_tokens
    available = context - prompt_tokens - safety
    if available < 1:
        raise ValueError(
            f"MAS-Gate Qwen30B prompt is too long for local context: "
            f"prompt_tokens={prompt_tokens}, max_model_len={context}, safety={safety}"
        )
    if available < requested_output:
        # Do not reinterpret max output as a total input+output budget.  If the
        # complete output allowance cannot fit, fail explicitly rather than
        # silently lowering max_tokens.
        raise ValueError(
            "MAS-Gate cannot preserve the pure completion budget: "
            f"prompt_tokens={prompt_tokens}, available_output_tokens={available}, "
            f"requested_output_tokens={requested_output}, max_model_len={context}"
        )
    effective = requested_output
    diag["effective_max_tokens"] = effective
    diag["output_budget_excludes_prompt"] = True
    return effective, prompt_tokens

# ``pipeline.evolve`` merges Vita ``models.get(model_id)`` into optimizer args; that dict often
# contains display-only keys (``name``, ``description``, …) that are **not** OpenAI API parameters.
# Only pass through keys accepted by ``AsyncOpenAI.chat.completions.create``.
_CHAT_COMPLETION_ALLOWED_OPTIONAL_KEYS = frozenset(
    {
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "n",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "logit_bias",
        "user",
        "metadata",
        "modalities",
        "prediction",
        "audio",
        "service_tier",
        "store",
        "web_search_options",
        # DashScope / Qwen-compatible extensions (merged into JSON body by the SDK).
        "extra_body",
    }
)


def chat_completion_extras_from_optimizer_llm_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Filter optimizer/runtime kwargs into arguments accepted by ``chat.completions.create``."""
    if not args:
        return {}
    out: dict[str, Any] = {}
    for k, v in dict(args).items():
        if k not in _CHAT_COMPLETION_ALLOWED_OPTIONAL_KEYS or v is None:
            continue
        out[str(k)] = v
    return out


def structured_json_candidate_text(message: Any) -> str:
    """
    Prefer ``message.content``; for some DashScope/Qwen reasoning responses the structured
    JSON may appear only under ``reasoning_content`` / ``reasoning`` when ``content`` is empty.
    """
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            v = extra.get(key)
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{") or s.startswith("["):
                    return s
    for key in ("reasoning_content", "reasoning"):
        v = getattr(message, key, None)
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("{") or s.startswith("["):
                return s
    return content


def _load_model_config() -> dict[str, Any]:
    if not MODEL_CONFIG_PATH.is_file():
        return {}
    data = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_pricing_table(path: Optional[Path] = None) -> dict[str, Any]:
    """
    Skill_MAS-local pricing loader.
    Uses model_config.json and filters out meta keys like "_defaults".
    """
    p = path or MODEL_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"pricing file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"pricing file must contain a JSON object: {p}")
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def normalize_usage_tokens(usage: Mapping[str, Any] | None) -> tuple[int, int, int]:
    """Return (prompt_tokens, completion_tokens, total_tokens)."""
    if not usage:
        return 0, 0, 0
    u = dict(usage)
    prompt = int(u.get("prompt_token_count", u.get("prompt_tokens", u.get("input_tokens", 0))) or 0)
    output = int(
        u.get("candidates_token_count", u.get("completion_tokens", u.get("output_tokens", 0))) or 0
    )
    total = int(u.get("total_token_count", u.get("total_tokens", prompt + output)) or 0)
    return prompt, output, total


def _row_to_per_1m(row: Any) -> tuple[float, float] | None:
    if not isinstance(row, dict):
        return None
    if "input_per_1m" in row and "output_per_1m" in row:
        return float(row["input_per_1m"]), float(row["output_per_1m"])
    if "input_per_1k" in row and "output_per_1k" in row:
        return float(row["input_per_1k"]) * 1000.0, float(row["output_per_1k"]) * 1000.0
    return None


def lookup_model_rates(table: Mapping[str, Any], model: str) -> tuple[float, float, str]:
    """
    Returns (input_usd_per_1m_tokens, output_usd_per_1m_tokens, resolved_key).
    Strict lookup only: no default row.
    """
    key = (model or "").strip()
    if key and key in table:
        rates = _row_to_per_1m(table[key])
        if rates:
            return rates[0], rates[1], key
    lower_map = {str(k).lower(): k for k in table if not str(k).startswith("_")}
    lk = key.lower()
    if lk and lk in lower_map:
        k = lower_map[lk]
        rates = _row_to_per_1m(table[k])
        if rates:
            return rates[0], rates[1], str(k)
    allowed = sorted(str(k) for k in table if not str(k).startswith("_"))
    raise ValueError(f"Unknown pricing model {model!r}. Available keys: {allowed}")


def estimate_cost_usd(
    *,
    model: str,
    usage: Mapping[str, Any] | None,
    table: Optional[Mapping[str, Any]] = None,
) -> tuple[float, int, int, str]:
    """Returns (cost_usd, prompt_tokens, output_tokens, pricing_key_used)."""
    prompt, output, _ = normalize_usage_tokens(usage)
    t = dict(table) if table is not None else load_pricing_table()
    inp_1m, out_1m, pkey = lookup_model_rates(t, model)
    cost = (inp_1m * prompt + out_1m * output) / 1_000_000.0
    return float(cost), prompt, output, pkey


def _usage_dict_from_chat_completion(response: Any) -> dict[str, Any]:
    """
    Build token counts for pricing from ``chat.completion`` ``usage``.

    Dashscope / DeepSeek / Qwen often include ``completion_tokens_details.reasoning_tokens``.
    That field is a *breakdown* of output: billable ``completion_tokens`` already matches
    ``total_tokens - prompt_tokens`` when the gateway is consistent (see raw API JSON).
    Do **not** add ``reasoning_tokens`` again or output will be double-counted.

    If ``prompt_tokens + completion_tokens != total_tokens``, reconcile using ``total_tokens``
    (some gateways omit overlap correctly only at top level).
    """
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    pt = int(getattr(u, "prompt_tokens", None) or 0)
    ct = int(getattr(u, "completion_tokens", None) or 0)
    tt = int(getattr(u, "total_tokens", None) or 0)
    if tt > 0 and pt + ct != tt:
        ct = max(tt - pt, 0)
    elif tt == 0 and (pt or ct):
        tt = pt + ct
    return {"input_tokens": pt, "output_tokens": ct, "total_tokens": tt}


def enrich_usage_with_cost(
    usage: MutableMapping[str, Any] | None,
    *,
    model: str,
    table: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a new usage dict with estimated_cost_usd and pricing metadata."""
    u = dict(usage or {})
    cost, _p, _o, pkey = estimate_cost_usd(model=model, usage=u, table=table)
    u["estimated_cost_usd"] = round(cost, 10)
    u["pricing_model_key"] = pkey
    return u


class AsyncOpenAIClient:
    """OpenAI-compatible async client for generic Skill_MAS LLM calls."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        explicit_http_client: bool = False,
        local_context_config: Optional[Mapping[str, Any]] = None,
        ring_role: Optional[str] = None,
        ring_controller_url: Optional[str] = None,
    ):
        self.model = model
        self.model_config = _load_model_config()
        row = self.model_config.get(self.model, {}) if isinstance(self.model_config, dict) else {}

        def _env_key(prefix: str, model: str) -> str:
            import re as _re
            return prefix + _re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()

        env_api_key = os.environ.get(_env_key("SKILL_MAS_MODEL_API_KEY_", self.model))
        env_base_url = os.environ.get(_env_key("SKILL_MAS_MODEL_BASE_URL_", self.model))
        # Generic aliases for the two local MAS-Gate models.  These keep this repo
        # isolated from any other verl/MAS project in the same container.
        if self.model.lower() == "qwen3-8b":
            env_base_url = env_base_url or os.environ.get("MAS_GATE_ORCHESTRA_BASE_URL")
            env_api_key = env_api_key or os.environ.get("MAS_GATE_ORCHESTRA_API_KEY")
        if self.model == "Qwen3-30B-A3B-Instruct-2507":
            env_base_url = env_base_url or os.environ.get("MAS_GATE_SUB_AGENT_BASE_URL")
            env_api_key = env_api_key or os.environ.get("MAS_GATE_SUB_AGENT_API_KEY")

        role_base_url = ""
        role_api_key = ""
        if ring_role == "orchestra":
            role_base_url = os.environ.get("MAS_GATE_ORCHESTRA_BASE_URL", "")
            role_api_key = os.environ.get("MAS_GATE_ORCHESTRA_API_KEY", "")
        elif ring_role == "subagent":
            role_base_url = os.environ.get("MAS_GATE_SUB_AGENT_BASE_URL", "")
            role_api_key = os.environ.get("MAS_GATE_SUB_AGENT_API_KEY", "")

        self.api_key = api_key or role_api_key or env_api_key or (row.get("api_key") if isinstance(row, dict) else None)
        self.base_url = base_url or role_base_url or env_base_url or (row.get("base_url") if isinstance(row, dict) else None)
        ring_url = ring_controller_url or os.environ.get("MAS_GATE_VLLM_RING_URL", "")
        if ring_url:
            normalized = str(self.base_url or "").rstrip("/")
            orchestra_url = os.environ.get("MAS_GATE_ORCHESTRA_BASE_URL", "").rstrip("/")
            subagent_url = os.environ.get("MAS_GATE_SUB_AGENT_BASE_URL", "").rstrip("/")
            inferred_role = ring_role
            if inferred_role is None and (
                self.model == "Qwen3-30B-A3B-Instruct-2507"
                or (subagent_url and normalized == subagent_url)
            ):
                inferred_role = "subagent"
            elif inferred_role is None and (
                self.model.lower() == "qwen3-8b"
                or (orchestra_url and normalized == orchestra_url)
            ):
                inferred_role = "orchestra"
            self._ring = (
                _VLLMRingLeaseClient(ring_url, inferred_role) if inferred_role else None
            )
        else:
            self._ring = None
        self.reasoning_effort = row.get("reasoning_effort") if isinstance(row, dict) else None
        self.temperature = row.get("temperature") if isinstance(row, dict) else None
        self.local_context_config = dict(local_context_config or {})
        raw_max_tokens = row.get("max_tokens") if isinstance(row, dict) else None
        if raw_max_tokens is not None:
            # DashScope compatible-mode rejects values outside [1, 65536].
            self.max_tokens = max(1, min(int(raw_max_tokens), 65536))
        else:
            self.max_tokens = None

        # MAS-Gate treats max_tokens as a pure output/completion cap.  The vLLM
        # max_model_len is configured separately and must contain prompt + this
        # output allowance.  An explicit environment override wins.
        if self.model == "Qwen3-30B-A3B-Instruct-2507":
            local_ctx = int(
                self.local_context_config.get(
                    "max_model_len",
                    os.environ.get("MAS_GATE_SUB_AGENT_MAX_MODEL_LEN", "32768"),
                )
            )
            local_completion = int(
                self.local_context_config.get(
                    "max_tokens",
                    os.environ.get("MAS_GATE_SUB_AGENT_MAX_TOKENS", str(max(256, local_ctx // 2))),
                )
            )
            local_completion = max(1, min(local_completion, max(1, local_ctx - 256)))
            self.max_tokens = local_completion if self.max_tokens is None else min(self.max_tokens, local_completion)
        eb = row.get("extra_body") if isinstance(row, dict) else None
        self.extra_body: dict[str, Any] | None = eb if isinstance(eb, dict) else None

        _timeout = httpx.Timeout(
            _DEFAULT_READ_TIMEOUT,
            connect=_DEFAULT_CONNECT_TIMEOUT,
            read=_DEFAULT_READ_TIMEOUT,
            write=_DEFAULT_CONNECT_TIMEOUT,
        )
        # The OpenAI SDK's private default AsyncHttpxClientWrapper launches an
        # untracked ``aclose()`` task from ``__del__``.  MAS-Gate reward workers
        # intentionally create one client per concurrently scored rollout, so
        # those destructor tasks can race ``asyncio.run`` teardown and flood the
        # log with harmless TCPTransport-closed exceptions.  Callers that opt in
        # receive a plain caller-owned httpx client with deterministic teardown.
        ring_enabled = self._ring is not None
        self._explicit_http_client: httpx.AsyncClient | None = None
        if explicit_http_client or ring_enabled:
            max_connections = max(
                128, int(os.environ.get("MAS_GATE_HTTP_MAX_CONNECTIONS", "1024"))
            )
            self._explicit_http_client = httpx.AsyncClient(
                timeout=_timeout,
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections,
                ),
                trust_env=False,
            )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_timeout,
            max_retries=0 if ring_enabled else 5,
            **(
                {"http_client": self._explicit_http_client}
                if self._explicit_http_client is not None
                else {}
            ),
        )
        # A generated MAS may fan out several Qwen30B calls concurrently inside
        # one reward worker.  Keep this distinct from Qwen-J concurrency: with
        # N Ray reward workers the effective upper bound is approximately
        # N * MAS_GATE_SUB_AGENT_CONCURRENCY_PER_WORKER.
        sub_agent_limit = max(1, int(os.environ.get(
            "MAS_GATE_SUB_AGENT_CONCURRENCY_PER_WORKER", "1024"
        )))
        self._generate_semaphore = asyncio.Semaphore(sub_agent_limit)
        has_model_pricing = isinstance(row, dict) and (
            ("input_per_1m" in row and "output_per_1m" in row)
            or ("input_per_1k" in row and "output_per_1k" in row)
        )
        if not has_model_pricing:
            raise ValueError(
                f"Model {self.model!r} must include pricing fields in {MODEL_CONFIG_PATH}. "
                "Require input/output per-1m (or per-1k) rates."
            )
        self.pricing = {self.model: row}
        # Bound active HTTP generations independently of sample fan-out.  A
        # sample concurrency of 512/1024 is useful for environment overlap,
        # but allowing that many simultaneous generations overwhelms vLLM and
        # the localhost coordinator.
        request_env = "MAS_GATE_LLM_CONCURRENCY"
        default_limit = "128"
        # Qwen3-30B-A3B is the memory/throughput bottleneck on the Gate
        # sub-agent endpoint.  Keep evaluator/sample fan-out and vLLM
        # max-num-seqs unchanged, but cap simultaneous HTTP generations for
        # this role so one burst cannot corrupt/abort the local engine.
        if ring_role == "subagent" or self.model == "Qwen3-30B-A3B-Instruct-2507":
            request_env = "MAS_GATE_SUBAGENT_LLM_CONCURRENCY"
            default_limit = "16"
        request_limit = max(1, min(1024, int(os.environ.get(request_env, default_limit))))
        self._request_semaphore = asyncio.Semaphore(request_limit)


    async def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
        **create_kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        call_kw = dict(create_kwargs)
        merged_extra: dict[str, Any] = {}
        if self.extra_body:
            merged_extra.update(self.extra_body)
        call_extra = call_kw.pop("extra_body", None)
        if isinstance(call_extra, dict):
            merged_extra.update(call_extra)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
            **call_kw,
        }
        if self.model == "Qwen3-30B-A3B-Instruct-2507":
            # vLLM applies Qwen3 thinking control through chat_template_kwargs.
            # The historical model config only sent a top-level
            # ``enable_thinking=false`` extension; this vLLM build ignored that
            # shape and generated long <think> blocks for every MAS node.  Keep
            # both spellings for compatibility, but enforce the nested value
            # after per-call kwargs are merged so generated workflow code cannot
            # turn thinking back on.
            merged_extra["enable_thinking"] = False
            chat_template_kwargs = merged_extra.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            merged_extra["chat_template_kwargs"] = {
                **chat_template_kwargs,
                "enable_thinking": False,
            }

        if merged_extra:
            kwargs["extra_body"] = merged_extra

        if self.model == "Qwen3-30B-A3B-Instruct-2507":
            # Cap after create_kwargs are merged so no generated workflow node can
            # override the local 8192-token vLLM context constraint.
            key = "max_completion_tokens" if kwargs.get("max_completion_tokens") is not None else "max_tokens"
            context_diagnostics: dict[str, Any] = {}
            capped, _prompt_tokens = _mas_gate_cap_completion_to_context(
                messages,
                kwargs.get(key),
                context_diagnostics,
                max_model_len=self.local_context_config.get("max_model_len"),
                max_tokens=self.local_context_config.get("max_tokens"),
                min_completion_tokens=self.local_context_config.get("min_completion_tokens"),
                context_safety_tokens=self.local_context_config.get("context_safety_tokens"),
                model_path=self.local_context_config.get("model_path"),
            )
            kwargs[key] = capped
        else:
            requested_max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
            context_diagnostics = {
                "requested_max_tokens": requested_max_tokens,
                "effective_max_tokens": requested_max_tokens,
                "context_compacted": False,
            }
        async with self._generate_semaphore, self._request_semaphore:
            response = None
            max_attempts = max(1, int(os.environ.get("SKILL_MAS_LOCAL_REQUEST_RETRIES", "3")))
            last_error: Exception | None = None
            for request_attempt in range(max_attempts):
                lease_id = await self._ring.acquire() if self._ring is not None else None
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                    break
                except (APIConnectionError, APIStatusError, APITimeoutError, httpx.HTTPError) as exc:
                    last_error = exc
                    if request_attempt + 1 >= max_attempts:
                        raise
                    # The failed HTTP request has no usable completion. Release the
                    # role before reacquiring it so sleep/wake handoff cannot leak.
                    await asyncio.sleep(0.1 * (2 ** request_attempt))
                finally:
                    if self._ring is not None and lease_id is not None:
                        # Do not let task cancellation leak a ring lease.  A leaked
                        # lease appears as a wave of HTTP 500/connection failures
                        # because subsequent requests wait on a role that can no
                        # longer make progress.
                        release_task = asyncio.create_task(self._ring.release(lease_id))
                        try:
                            await asyncio.shield(release_task)
                        except asyncio.CancelledError:
                            try:
                                await release_task
                            finally:
                                raise
            if response is None:
                raise RuntimeError(f"local completion failed after {max_attempts} attempts: {last_error}")
        msg = response.choices[0].message
        text = structured_json_candidate_text(msg)
        usage = _usage_dict_from_chat_completion(response)
        usage = enrich_usage_with_cost(usage, model=self.model, table=self.pricing)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        usage.update(context_diagnostics)
        usage["prompt_tokens_actual"] = int(
            usage.get("input_tokens", context_diagnostics.get("prompt_tokens_after_compaction", 0)) or 0
        )
        usage["finish_reason"] = str(finish_reason or "")
        effective_max_tokens = context_diagnostics.get("effective_max_tokens")
        usage["completion_saturated"] = bool(
            effective_max_tokens
            and int(usage.get("output_tokens", 0) or 0) >= int(effective_max_tokens)
        )
        return text, usage

    async def aclose(self) -> None:
        closer = getattr(self.client, "close", None)
        if closer is None:
            return
        res = closer()
        if hasattr(res, "__await__"):
            await res  # type: ignore[misc]
        # ``AsyncOpenAI.close`` closes the explicit client as well.  Keep this
        # idempotent guard for SDK versions where that ownership changes.
        if self._explicit_http_client is not None and not self._explicit_http_client.is_closed:
            await self._explicit_http_client.aclose()
        if self._ring is not None:
            await self._ring.close()
