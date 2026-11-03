"""Compact LoRA GRPO trainer for the task-level Runtime Editor episode.

Usage: python -m gate_main.train --config config.json
The actor is the only trainable model. Frozen MAS execution and scoring happen
through the worker and quantifier adapters configured in the JSON file.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import load_config, load_reference, load_tasks
from .episode import Turn, sample_episode
from .factory import correctness_from_config, quantifier_from_config, worker_from_config
from .training import RewardConfig, group_relative_advantages


class HFEditorSampler:
    def __init__(self, model, tokenizer, *, max_prompt: int, max_action: int,
                 temperature: float, do_sample: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.max_prompt = max_prompt
        self.max_action = max_action
        self.temperature = temperature
        self.do_sample = do_sample

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, list[int], list[int]]:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(prompt) > self.max_prompt:
            raise ValueError(f"Editor prompt length {len(prompt)} exceeds {self.max_prompt}")
        device = next(self.model.parameters()).device
        ids = torch.tensor([prompt], device=device)
        self.model.eval()
        with torch.inference_mode():
            generation_options = {
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.do_sample:
                generation_options["temperature"] = self.temperature
            generated = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=self.max_action,
                **generation_options,
            )
        action_ids = generated[0, ids.shape[1]:].tolist()
        return self.tokenizer.decode(action_ids, skip_special_tokens=True), prompt, action_ids


def action_logprobs(model, turn: Turn):
    import torch

    prompt_len = len(turn.prompt_ids)
    if not prompt_len or not turn.action_ids:
        raise ValueError("empty prompt or action span")
    device = next(model.parameters()).device
    ids = torch.tensor([turn.prompt_ids + turn.action_ids], device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    selected = logits[0, prompt_len - 1:-1].float().log_softmax(-1)
    targets = torch.tensor(turn.action_ids, device=device).unsqueeze(1)
    return selected.gather(1, targets).squeeze(1)


def _group_turns(episodes: list[list[Turn]]) -> list[tuple[Turn, float]]:
    grouped: dict[int, list[Turn]] = {}
    for episode in episodes:
        for turn in episode:
            grouped.setdefault(turn.ordinal, []).append(turn)
    result: list[tuple[Turn, float]] = []
    for turns in grouped.values():
        values = group_relative_advantages([turn.reward for turn in turns])
        result.extend(zip(turns, values, strict=True))
    return result


def _save_checkpoint(model, tokenizer, optimizer, root: Path, step: int, config: dict) -> None:
    import torch

    path = root / f"step-{step}"
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    model.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "adapter")
    torch.save({
        "step": step,
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "config": config,
    }, path / "training_state.pt")


def _load_actor(config: dict, resume_from: str | None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = config["training"]
    model_name = settings["actor_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, settings.get("dtype", "bfloat16"))
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
        device_map=settings.get("device_map", "auto"),
    )
    if resume_from:
        model = PeftModel.from_pretrained(base, str(Path(resume_from) / "adapter"),
                                          is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=int(settings.get("lora_rank", 64)),
            lora_alpha=int(settings.get("lora_alpha", 32)),
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ))
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(settings.get("learning_rate", 1e-6)),
    )
    start_step = 0
    if resume_from:
        saved = torch.load(
            Path(resume_from) / "training_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        start_step = int(saved["step"])
    return model, tokenizer, optimizer, start_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["training"]
    rows = load_tasks(config["data_file"])
    # Check all references before allocating actor or scoring models.
    references = [load_reference(config["reference_dir"], row) for row in rows]
    batch_tasks = int(settings.get("batch_tasks", 1))
    rollouts = int(settings.get("rollouts_per_task", 16))
    epochs = int(settings.get("epochs", 1))
    if min(batch_tasks, rollouts, epochs) <= 0:
        raise ValueError("batch_tasks, rollouts_per_task, and epochs must be positive")
    available_steps = (len(rows) // batch_tasks) * epochs
    steps = int(settings.get("steps", available_steps))
    if not 0 < steps <= available_steps:
        raise ValueError(f"steps must lie in [1, {available_steps}]")
    random.seed(int(settings.get("seed", 42)))
    model, tokenizer, optimizer, start_step = _load_actor(config, args.resume_from)
    sampler = HFEditorSampler(
        model, tokenizer,
        max_prompt=int(settings.get("max_prompt_tokens", 15000)),
        max_action=int(settings.get("max_action_tokens", 512)),
        temperature=float(settings.get("temperature", 1.0)),
    )
    worker = worker_from_config(config)
    quantifier = quantifier_from_config(config)
    correctness = correctness_from_config(config)
    rewards = RewardConfig(
        quality_weight=float(settings.get("beta_q", 0.1)),
        cost_weight=float(settings.get("lambda_cost", 1.0)),
        invalid_reward=float(settings.get("invalid_reward", -0.25)),
    )
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_steps = set(int(value) for value in settings.get("checkpoint_steps", []))
    import torch

    for step in range(start_step + 1, steps + 1):
        epoch = (step - 1) // (len(rows) // batch_tasks)
        slot = (step - 1) % (len(rows) // batch_tasks)
        order = list(range(len(rows)))
        random.Random(int(settings.get("seed", 42)) + epoch).shuffle(order)
        selected = order[slot * batch_tasks:(slot + 1) * batch_tasks]
        training_turns: list[tuple[Turn, float]] = []
        for index in selected:
            row = rows[index]
            reference, baseline_c, baseline_q = references[index]
            episodes = [sample_episode(
                reference=reference, answer=str(row["answer"]),
                reference_c=baseline_c, reference_q=baseline_q,
                sampler=sampler, worker=worker, quantifier=quantifier,
                correctness=correctness, reward_config=rewards,
            ) for _ in range(rollouts)]
            training_turns.extend(_group_turns(episodes))

        # Old policy probabilities are frozen before this optimizer step.
        model.eval()
        with torch.no_grad():
            old = [action_logprobs(model, turn).detach() for turn, _ in training_turns]
        optimizer.zero_grad(set_to_none=True)
        model.train()
        token_count = sum(len(turn.action_ids) for turn, _ in training_turns)
        active = any(abs(advantage) > 1e-8 for _, advantage in training_turns)
        clip = float(settings.get("clip_ratio", 0.2))
        kl_weight = float(settings.get("kl_coefficient", 0.001))
        loss_total = 0.0
        if active:
            for (turn, advantage), old_logp in zip(training_turns, old, strict=True):
                new_logp = action_logprobs(model, turn)
                ratio = (new_logp - old_logp).clamp(-20, 20).exp()
                a = torch.as_tensor(advantage, device=new_logp.device)
                surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
                loss = -surrogate.sum() / token_count
                if kl_weight:
                    with torch.no_grad(), model.disable_adapter():
                        base_logp = action_logprobs(model, turn)
                    delta = (base_logp - new_logp).clamp(-20, 20)
                    loss = loss + kl_weight * (delta.exp() - delta - 1).sum() / token_count
                loss.backward()
                loss_total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(settings.get("max_grad_norm", 1.0)),
            )
            optimizer.step()
        print(f"step={step}/{steps} tasks={len(selected)} turns={len(training_turns)} "
              f"loss={loss_total:.5f}", flush=True)
        if step in checkpoint_steps or step == steps:
            _save_checkpoint(model, tokenizer, optimizer, root, step, config)


if __name__ == "__main__":
    main()
