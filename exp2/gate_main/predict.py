"""Run a trained Editor on frozen workflows without answers or score models.

Usage: python -m gate_main.predict --config config.json --checkpoint runs/editor/step-N \
       --data data/predict.jsonl --output runs/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_config
from .editor import Action, ActionKind, Graph, GateState, validate_action
from .factory import worker_from_config
from .prompts import initial_messages, next_gate_message
from .runtime import execute_with_editor
from .train import HFEditorSampler


def _load_model(config: dict, checkpoint: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = config["training"]
    model_name = settings["actor_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, settings.get("dtype", "bfloat16")),
        trust_remote_code=True,
        device_map=settings.get("device_map", "auto"),
    )
    model = PeftModel.from_pretrained(base, str(checkpoint / "adapter"))
    return model.eval(), tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="JSONL with id, task, and graph; no answer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    config = load_config(args.config)
    model, tokenizer = _load_model(config, Path(args.checkpoint))
    settings = config["training"]
    sampler = HFEditorSampler(
        model, tokenizer,
        max_prompt=int(settings.get("max_prompt_tokens", 15000)),
        max_action=int(settings.get("max_action_tokens", 512)),
        temperature=1.0,
        do_sample=False,
    )
    worker = worker_from_config(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.data).open(encoding="utf-8") as source, output_path.open(
        "x", encoding="utf-8"
    ) as destination:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(row.get(key) for key in ("id", "task", "graph")):
                raise ValueError(f"row {number} needs id, task, graph")
            graph = Graph.from_dict(row["graph"])
            messages: list[dict[str, str]] = []
            decisions: list[dict[str, str | dict]] = []

            def decide(state: GateState) -> Action:
                nonlocal messages
                if not messages:
                    messages = initial_messages(state)
                else:
                    messages.append(next_gate_message(state))
                raw, _, _ = sampler.generate(messages)
                try:
                    action = Action.parse(raw)
                    validate_action(state, action, sampler.count_tokens)
                    status = "valid"
                except (ValueError, KeyError, TypeError):
                    # Invalid output cannot mutate the workflow at inference time.
                    action, status = Action(ActionKind.KEEP), "invalid_fallback_keep"
                decisions.append({"node": state.current_node, "status": status,
                                  "action": action.as_dict()})
                messages.append({"role": "assistant", "content": raw})
                return action

            trace = execute_with_editor(str(row["task"]), graph, worker, decide)
            destination.write(json.dumps({
                "id": row["id"], "final_output": trace.final_output(),
                "decisions": decisions, "extra_calls": trace.extra_calls,
            }, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"Prediction {number}: {row['id']}", flush=True)


if __name__ == "__main__":
    main()
