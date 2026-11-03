"""Replaceable frozen sub-agent interface for the public training example."""

from __future__ import annotations

import importlib
import json
import os

from .editor import Node


def load_symbol(spec: str):
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError("adapter must use module:function or module:Class")
    return getattr(importlib.import_module(module_name), attribute)


class OpenAIWorker:
    """Run LLM-only MAS nodes through any OpenAI-compatible chat service."""

    def __init__(
        self, *, base_url: str, model: str, api_key_env: str = "WORKER_API_KEY",
        temperature: float = 0.7, max_tokens: int = 2048, timeout: float = 600.0,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env, "EMPTY"),
            timeout=timeout,
        )

    def run(
        self, node: Node, task: str, upstream: dict[str, str],
        guidance: str = "", fixed_prompt: str | None = None,
    ) -> tuple[str, str]:
        prompt = fixed_prompt or (
            "GLOBAL TASK CONTEXT:\n" + task + "\n\nROLE INSTRUCTION:\n" + node.role
            + "\nSUBTASK:\n" + node.task
            + "\nCONSTRAINTS:\n" + "\n".join(node.constraints)
            + "\nUPSTREAM DEPENDENCIES DATA:\n"
            + json.dumps(upstream, ensure_ascii=False)
            + ("\nEDITOR GUIDANCE:\n" + guidance if guidance else "")
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        output = response.choices[0].message.content or ""
        if not output.strip():
            raise RuntimeError(f"frozen worker returned an empty output for {node.id}")
        return output, prompt
