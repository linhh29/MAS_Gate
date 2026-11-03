"""External strong-model baseline for Runtime Editor action selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Sequence

from .protocol import (
    Action,
    ActionKind,
    ChatClient,
    EditorObservation,
    Message,
    validate_action,
)


EDITOR_SYSTEM = """You are an external Runtime Editor for a multi-agent system.
Inspect only the public execution prefix. Select the least disruptive legal
action that has positive expected benefit. Do not solve the task yourself and
do not assume hidden correctness signals. Return exactly one JSON object:
{"action":"ACTION","target":"NODE_OR_NULL","payload":{...}}"""


ACTION_GUIDE = """Use only actions and targets listed in legal_actions.
- KEEP: preserve a complete milestone when no specific repair is justified.
- RETRY_CURRENT: resample an incomplete or unreliable current milestone.
- BACKTRACK_PARENT: regenerate a direct parent whose handoff caused the defect.
- ANNOTATE_CURRENT: preserve the output but flag one precise local defect.
  Payload keys: error_type, target_span, diagnosis, downstream_check.
- GUIDE_CHILD: ask an existing child for one precise check or recomputation.
  Payload keys: guidance_type, focus.
- INSERT_SUBTASK: add one bounded verification, bridge, or supplement.
  Payload keys: mode, task.
KEEP, RETRY_CURRENT, and BACKTRACK_PARENT use an empty payload."""


def _parse_action(text: str) -> Action:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response contains no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("action must be a JSON object")
    try:
        kind = ActionKind(str(value["action"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("unknown or missing action") from exc
    target = value.get("target")
    if target is not None and not isinstance(target, str):
        raise ValueError("target must be a node identifier or null")
    payload = value.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return Action(kind=kind, target=target, payload=payload)


@dataclass
class StrongLLMEditor:
    """One implementation shared by every external strong-model comparison."""

    client: ChatClient
    retry_invalid_once: bool = True

    async def choose(self, observation: EditorObservation) -> Action:
        messages: list[Message] = [
            {"role": "system", "content": EDITOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    ACTION_GUIDE
                    + "\n\nPUBLIC OBSERVATION:\n"
                    + json.dumps(
                        observation.as_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            },
        ]
        attempts = 2 if self.retry_invalid_once else 1
        for _ in range(attempts):
            raw = await self.client.complete(messages, temperature=0.0)
            try:
                action = _parse_action(raw)
                validate_action(observation, action)
                return action
            except (ValueError, json.JSONDecodeError) as exc:
                messages.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                f"Invalid action: {exc}. Return one corrected JSON "
                                "object using only the listed legal actions."
                            ),
                        },
                    ]
                )
        fallback = Action(ActionKind.KEEP)
        validate_action(observation, fallback)
        return fallback


class OpenAICompatibleChat:
    """Optional API adapter; model and provider can be replaced at runtime."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        if not model or not api_key:
            raise ValueError("model and api_key are required")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional 'openai' package to use this adapter"
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(cls) -> OpenAICompatibleChat:
        """Read generic runtime configuration without storing any credential."""
        return cls(
            model=os.environ["EDITOR_MODEL"],
            api_key=os.environ["EDITOR_API_KEY"],
            base_url=os.environ.get("EDITOR_BASE_URL"),
            max_tokens=int(os.environ.get("EDITOR_MAX_TOKENS", "2048")),
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("model returned an empty response")
        return content
