"""Frozen outcome and MAS-Gate trace scores used only by the trainer.

The default outcome scorer is exact answer matching. Interactive environments
and benchmark-specific answer extraction should supply a replacement function.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable

from .editor import Graph
from .training import node_quantities, trace_quality


def exact_answer(prediction: str, answer: str) -> float:
    def normalize(text: str) -> str:
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())
    return float(normalize(prediction) == normalize(answer))


class HFQuantifier:
    """Reference J and entailment scorer with independently frozen HF models.

    Configure devices to match local resources. Large runs can replace this
    local adapter with phase-separated scoring services while retaining the
    same score equations.
    """

    def __init__(
        self, scoring_model: str, nli_model: str, *, scoring_device: str = "cuda:0",
        nli_device: str = "cpu", entailment_label_id: int = 2,
        max_scoring_tokens: int = 4096, max_nli_tokens: int = 4096,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.scoring_device = scoring_device
        self.nli_device = nli_device
        self.entailment_label_id = entailment_label_id
        self.max_scoring_tokens = max_scoring_tokens
        self.max_nli_tokens = max_nli_tokens
        self.j_tokenizer = AutoTokenizer.from_pretrained(scoring_model, trust_remote_code=True)
        self.j_model = AutoModelForCausalLM.from_pretrained(
            scoring_model, torch_dtype="auto", trust_remote_code=True
        ).to(scoring_device).eval()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model, trust_remote_code=True)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_model, torch_dtype="auto", trust_remote_code=True
        ).to(nli_device).eval()
        if not 0 <= entailment_label_id < self.nli_model.config.num_labels:
            raise ValueError("entailment_label_id is outside the NLI classifier labels")

    def j(self, context: str, answer: str) -> float:
        """Mean teacher-forced log probability of the reference answer."""
        torch = self.torch
        messages = [{"role": "user", "content": context}]
        prompt = self.j_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        target = self.j_tokenizer.encode(answer, add_special_tokens=False)
        if not target:
            raise ValueError("reference answer is empty")
        if len(target) >= self.max_scoring_tokens:
            raise ValueError("reference answer exceeds scoring context")
        prompt = prompt[-(self.max_scoring_tokens - len(target)):]
        ids = torch.tensor([prompt + target], device=self.scoring_device)
        with torch.no_grad():
            logits = self.j_model(input_ids=ids).logits[0, len(prompt) - 1:-1]
            logp = logits.float().log_softmax(-1).gather(
                1, torch.tensor(target, device=ids.device).unsqueeze(1)
            ).squeeze(1)
        return float(logp.mean().item())

    def entailment(self, premise: str, hypothesis: str) -> float:
        torch = self.torch
        encoded = self.nli_tokenizer(
            premise, hypothesis, truncation=True, max_length=self.max_nli_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.nli_device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.nli_model(**encoded).logits[0]
            probabilities = logits.float().softmax(-1)
        return float(probabilities[self.entailment_label_id].item())

    def score(self, task: str, answer: str, graph: Graph, outputs: dict[str, str]) -> float:
        """Return Q(tau), mean I_i over the fully executed graph."""
        order = graph.order()
        if set(order) != outputs.keys():
            raise ValueError("Q requires exactly one completed output per node")
        j0 = self.j(task, answer)
        if abs(j0) < 1e-8:
            raise ValueError("baseline J is too close to zero")
        children = {nid: graph.children(nid) for nid in order}
        local_delta: dict[str, float] = {}

        def ancestors(nid: str) -> set[str]:
            seen, pending = set(), list(graph.nodes[nid].parents)
            while pending:
                parent = pending.pop()
                if parent not in seen:
                    seen.add(parent)
                    pending.extend(graph.nodes[parent].parents)
            return seen

        for nid in order:
            prior = [outputs[item] for item in order if item in ancestors(nid)]
            before = task + "\n" + "\n".join(prior)
            after = before + "\n" + outputs[nid]
            local_delta[nid] = (self.j(after, answer) - self.j(before, answer)) / abs(j0)
        edge_entailment = {
            (nid, child): self.entailment(outputs[nid], outputs[child])
            for nid in order for child in children[nid]
        }
        return trace_quality(node_quantities(local_delta, children, edge_entailment))


def score_trace(
    task: str, answer: str, graph: Graph, outputs: dict[str, str],
    quantifier: HFQuantifier, correctness: Callable[[str, str], float] = exact_answer,
) -> tuple[float, float]:
    final = outputs[graph.final_id()]
    outcome = float(correctness(final, answer))
    if not 0 <= outcome <= 1:
        raise ValueError("correctness must lie in [0, 1]")
    return outcome, quantifier.score(task, answer, graph, outputs)
