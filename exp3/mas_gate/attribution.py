"""Paper attribution inputs, prompts, rankings, and evaluation (stdlib only).

Who&When retains zero-based steps. AEGIS retains its one-based source steps,
selects records with one unique injected agent, and predicts an agent directly.
Labels are separate evaluation objects: prompt builders never accept them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Step:
    index: int
    agent: str
    content: str
    phase: str = ""
    selectable: bool = True


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    answer: str
    steps: tuple[Step, ...]
    observed_final_output: str = ""
    benchmark: str = "who_when"


@dataclass(frozen=True)
class Label:
    agent: str
    step: int | None = None


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _validate_example(example: Example) -> None:
    if example.benchmark not in {"who_when", "aegis"}:
        raise ValueError("benchmark must be who_when or aegis")
    start = 0 if example.benchmark == "who_when" else 1
    if not example.steps or [s.index for s in example.steps] != list(
        range(start, start + len(example.steps))
    ):
        raise ValueError("steps must be nonempty and contiguous in benchmark order")
    if any(type(s.index) is not int or not s.agent.strip() for s in example.steps):
        raise ValueError("each step requires an integer index and a nonempty agent")
    if not any(s.selectable for s in example.steps):
        raise ValueError("example has no selectable agent")
    if example.benchmark == "who_when" and not all(s.selectable for s in example.steps):
        raise ValueError("Who&When retains all original steps as selectable")
    if example.benchmark == "aegis" and any(
        s.selectable != (s.agent.casefold() not in {"user", "system"})
        for s in example.steps
    ):
        raise ValueError("AEGIS User/System turns must be context only")


def load_who_when(
    root: str | Path, *, require_complete: bool = True
) -> list[tuple[Example, Label]]:
    """Load the 184 released trajectories, ignoring hidden sidecar files."""
    root = Path(root).resolve()
    paths = sorted(
        (
            p for p in root.rglob("*.json")
            if not any(part.startswith(".") for part in p.relative_to(root).parts)
            and any(part.lower().replace("_", "-") in {
                "algorithm-generated", "hand-crafted"
            } for part in p.relative_to(root).parts)
        ),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    result = []
    for path in paths:
        raw = _object(json.loads(path.read_text(encoding="utf-8")), "record")
        history = raw.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"{path}: history must be a nonempty list")
        steps = []
        for index, entry in enumerate(history):
            entry = _object(entry, "history entry")
            steps.append(Step(
                index=index,
                agent=str(entry.get("name") or entry.get("role") or "Unknown").strip(),
                content=str(entry.get("content") or ""),
            ))
        label = Label(str(raw["mistake_agent"]).strip(), int(raw["mistake_step"]))
        if not label.agent or not 0 <= label.step < len(steps):
            raise ValueError(f"{path}: invalid attribution label")
        example = Example(
            id=path.relative_to(root).with_suffix("").as_posix(),
            question=str(raw.get("question", "")),
            answer=str(raw.get("ground_truth", "")),
            steps=tuple(steps),
            observed_final_output=steps[-1].content,
            benchmark="who_when",
        )
        _validate_example(example)
        result.append((example, label))
    if require_complete and len(result) != 184:
        raise ValueError(f"expected 184 Who&When records, found {len(result)}")
    return result


def load_aegis(
    path: str | Path, *, require_complete: bool = True
) -> list[tuple[Example, Label]]:
    """Allowlist observable fields from the 600-record test split (137 m=1).

    Repeated injections into the same exact agent still count as one agent.
    Injection flags, strategies, faulty outputs, and other annotations are
    discarded. The source ID is retained only for result bookkeeping; it is
    never included in prompts or graph node text.
    """
    result = []
    seen: set[str] = set()
    total = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            total += 1
            raw = _object(json.loads(line), f"record on line {line_number}")
            truth = _object(raw.get("ground_truth"), "ground_truth")
            injected = truth.get("injected_agents")
            if not isinstance(injected, list):
                raise ValueError("ground_truth.injected_agents must be a list")
            names = set()
            for injection in injected:
                name = str(_object(injection, "injected agent").get("agent_name") or "").strip()
                if not name:
                    raise ValueError("injected agent name is empty")
                names.add(name)
            if len(names) != 1:
                continue
            target = next(iter(names))
            data = _object(raw.get("input"), "input")
            history = data.get("conversation_history")
            if not isinstance(history, list) or not history:
                raise ValueError("input.conversation_history must be a nonempty list")
            steps = []
            for entry in history:
                entry = _object(entry, "conversation turn")
                agent = str(entry.get("agent_name") or "").strip()
                steps.append(Step(
                    index=entry.get("step"),
                    agent=agent,
                    content=str(entry.get("content") or ""),
                    phase=str(entry.get("phase") or "").strip(),
                    selectable=agent.casefold() not in {"user", "system"},
                ))
            example = Example(
                id=str(raw.get("id") or "").strip(),
                question=str(data.get("query") or "").strip(),
                answer=str(truth.get("correct_answer") or "").strip(),
                steps=tuple(steps),
                observed_final_output=str(data.get("final_output") or "").strip() or steps[-1].content,
                benchmark="aegis",
            )
            _validate_example(example)
            if not example.id or not example.question or not example.answer:
                raise ValueError("AEGIS id, query, and correct answer must be nonempty")
            if example.id in seen:
                raise ValueError(f"duplicate AEGIS id: {example.id}")
            if not any(s.selectable and s.agent == target for s in steps):
                raise ValueError("injected agent has no exact selectable history turn")
            seen.add(example.id)
            result.append((example, Label(target)))
    if require_complete and (total != 600 or len(result) != 137):
        raise ValueError(f"expected 600 test / 137 m=1 records, found {total} / {len(result)}")
    return result


def _risk(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("risk must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("risk must be finite and nonnegative")
    return value


def rank_candidates(
    example: Example,
    risks: Mapping[int, float],
    k: int,
    variant: str = "mas_gate",
) -> list[dict[str, Any]]:
    """Rank WW steps or AEGIS agents; break risk ties by earliest source step.

    For AEGIS, an agent's score is its maximum selectable step risk. The
    representative is the earliest step attaining that maximum. WW random
    controls preserve the published SHA256 namespace, seed, and fallback
    rotation; these strings are algorithm constants, not run identifiers.
    """
    _validate_example(example)
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if variant not in {"mas_gate", "random", "shuffled"}:
        raise ValueError("variant must be mas_gate, random, or shuffled")
    if example.benchmark == "aegis" and variant != "mas_gate":
        raise ValueError("random and shuffled controls apply only to Who&When")
    rows = []
    for step in example.steps:
        if step.selectable:
            if step.index not in risks:
                raise ValueError(f"missing risk for selectable step {step.index}")
            rows.append({"step": step.index, "agent": step.agent,
                         "risk_score": _risk(risks[step.index])})
    rows.sort(key=lambda row: (-row["risk_score"], row["step"]))
    if example.benchmark == "aegis":
        agents: dict[str, dict[str, Any]] = {}
        for row in rows:
            agents.setdefault(row["agent"], dict(row))
        rows = sorted(agents.values(), key=lambda r: (-r["risk_score"], r["step"], r["agent"]))
        for row in rows:
            row["source_steps"] = [s.index for s in example.steps if s.selectable and s.agent == row["agent"]]
        selected = rows[:k]
    elif variant == "mas_gate":
        selected = rows[:k]
    else:
        control = "random_topk" if variant == "random" else "mas_topk_shuffled"

        def random_key(row: Mapping[str, Any]) -> bytes:
            material = f"who_when_fixed_k_controls_v1|20260923|{control}|{example.id}|{row['step']}"
            return hashlib.sha256(material.encode("utf-8")).digest()

        if variant == "random":
            selected = sorted(sorted(rows, key=lambda r: r["step"]), key=random_key)[:k]
        else:
            selected = sorted(rows[:k], key=random_key)
            if len(selected) > 1 and selected == rows[:k]:
                selected = selected[1:] + selected[:1]
    return [{"rank": rank, **row} for rank, row in enumerate(selected, 1)]


_WW_SYSTEM = (
    "You are a precise failure-attribution evaluator. Follow the requested output "
    "format exactly and do not add headings, markdown fences, or extra lines."
)
_AEGIS_SYSTEM = (
    "You are a precise failure-attribution evaluator. Follow the requested "
    "two-line output format exactly and do not add headings or markdown."
)
_WW_RANKING_INTRO = (
    "\n\nAdditional MAS-Gate attribution information:\n"
    "MAS-Gate is a computation-based failure-attribution method that scores and "
    "ranks trajectory steps using their utility and communication influence. Its "
    "ranking provides useful information for identifying the decisive error. The "
    "following steps received the highest scores and are therefore the most "
    "suspicious candidates.\n\n"
    "MAS-Gate candidate ranking:\n"
)


def _agents(example: Example) -> list[str]:
    return list(dict.fromkeys(s.agent for s in example.steps if s.selectable))


def build_messages(
    example: Example, candidates: Sequence[Mapping[str, Any]] | None = None
) -> list[dict[str, str]]:
    """Build exact paper Direct-with-GT or soft ranked-reference prompts.

    Reference *task answers* are authorized inputs in both arms. Attribution
    labels, record IDs, risk values, and representative AEGIS steps are never
    rendered. Soft guidance leaves every original selectable target available.
    """
    _validate_example(example)
    if candidates is not None:
        if not candidates or [c.get("rank") for c in candidates] != list(range(1, len(candidates) + 1)):
            raise ValueError("candidate ranks must be nonempty and consecutive")
        key = "step" if example.benchmark == "who_when" else "agent"
        values = [c.get(key) for c in candidates]
        valid = {s.index for s in example.steps} if key == "step" else set(_agents(example))
        if len(set(values)) != len(values) or not set(values) <= valid:
            raise ValueError("candidate targets must be unique and selectable")
        if example.benchmark == "aegis":
            canonical = sorted(candidates, key=lambda c: (-_risk(c["risk_score"]), c["step"], c["agent"]))
            if list(candidates) != canonical:
                raise ValueError("AEGIS candidates must be in max-risk order")
    if example.benchmark == "who_when":
        history = "\n\n".join(f"Step {s.index}\nAgent: {s.agent}\nContent:\n{s.content}" for s in example.steps)
        prompt = (
            "Analyze the failed multi-agent trajectory below.\n"
            f"Problem: {example.question}\n"
            f"Reference correct answer: {example.answer}\n"
            f"Valid Step Number range: 0 to {len(example.steps) - 1}, inclusive.\n"
            "Identify the single agent directly responsible for the wrong solution and the first "
            "zero-based step where that agent made the decisive mistake. Agent Name must exactly "
            "match the Agent field shown at the selected step. Give a concise one-line reason.\n"
            "Return exactly these three lines:\n"
            "Agent Name: <name>\n"
            "Step Number: <integer>\n"
            "Reason for Mistake: <reason>\n\n"
            f"Conversation:\n{history}"
        )
        if candidates is not None:
            prompt += _WW_RANKING_INTRO + "\n".join(
                f"Rank {c['rank']}: Step {c['step']}" for c in candidates
            ) + (
                "\n\nUse the MAS-Gate ranking as an important reference when identifying the "
                "single agent and the first decisive-error step.\n"
            )
        system = _WW_SYSTEM
    else:
        blocks = []
        for step in example.steps:
            phase = f"\nPhase: {step.phase}" if step.phase else ""
            status = "selectable agent turn" if step.selectable else "context only; not selectable"
            blocks.append(f"Step {step.index}\nAgent: {step.agent}{phase}\nSelection status: {status}\nContent:\n{step.content}")
        history = "\n\n".join(blocks)
        prompt = (
            "Analyze the failed multi-agent trajectory below.\n"
            f"Problem: {example.question}\n"
            f"Reference correct answer: {example.answer}\n"
            f"Observed final output: {example.observed_final_output}\n"
            f"Valid selectable agents: {', '.join(_agents(example))}.\n"
            "User and System turns are context only and cannot be selected. Choose "
            "exactly one valid agent. Agent Name must exactly match one name in the "
            "valid selectable agents list.\n"
            "Return exactly these two lines:\n"
            "Agent Name: <name>\n"
            "Reason for Mistake: <one-line reason>\n\n"
            f"Conversation:\n{history}"
        )
        if candidates is not None:
            ranks = "\n".join(f"Rank {c['rank']}: {c['agent']}" for c in candidates)
            prompt += (
                f"\n\nMAS-Gate identified the following Top-{len(candidates)} most "
                "suspicious agents, ranked by estimated error risk from highest to "
                "lowest. Rank 1 is the most suspicious:\n"
                f"{ranks}\n"
                "Use this ranking together with the complete trajectory and the "
                "reference correct answer to identify the agent responsible for the "
                "failure."
            )
        system = _AEGIS_SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def _actor_match(left: str, right: str) -> bool:
    left = re.sub(r"[^a-z0-9]", "", left.lower())
    right = re.sub(r"[^a-z0-9]", "", right.lower())
    return bool(left and right) and (left in right or right in left)


def parse_prediction(text: str, example: Example) -> dict[str, Any]:
    """Parse the paper's strict three-line WW or two-line AEGIS format."""
    _validate_example(example)
    pattern = r"[ \t]*Agent[ \t]+Name[ \t]*:[ \t]*(?P<agent>[^\r\n]*?\S)[ \t]*\r?\n"
    if example.benchmark == "who_when":
        pattern += r"[ \t]*Step[ \t]+Number[ \t]*:[ \t]*(?P<step>-?\d+)[ \t]*\r?\n"
    pattern += r"[ \t]*Reason[ \t]+for[ \t]+Mistake[ \t]*:[ \t]*(?P<reason>[^\r\n]*?\S)[ \t]*"
    if example.benchmark == "aegis":
        pattern += r"(?:\r?\n)?[ \t]*"
    match = re.fullmatch(pattern, text, re.IGNORECASE)
    agent = match.group("agent").strip() if match else ""
    reason = match.group("reason").strip() if match else ""
    errors = [] if match else ["format_mismatch"]
    if not agent:
        errors.append("missing_agent")
    elif example.benchmark == "aegis" and agent not in _agents(example):
        errors.append("agent_not_selectable")
    step = int(match.group("step")) if match and example.benchmark == "who_when" else None
    if example.benchmark == "who_when":
        if step is None:
            errors.append("missing_step")
        elif not 0 <= step < len(example.steps):
            errors.append("step_out_of_range")
    if not reason:
        errors.append("missing_reason")
    result = {"agent": agent, "step": step, "reason": reason, "parse_ok": not errors,
              "parse_errors": errors, "raw_response": text}
    if example.benchmark == "who_when":
        actor = example.steps[step].agent if type(step) is int and 0 <= step < len(example.steps) else ""
        result.update(predicted_step_actor=actor, agent_step_consistent=_actor_match(agent, actor))
    return result


def evaluate(prediction: Mapping[str, Any], label: Label, benchmark: str) -> dict[str, bool]:
    """WW primary outcome is exact step; AEGIS uses exact agent identity."""
    if benchmark not in {"who_when", "aegis"}:
        raise ValueError("benchmark must be who_when or aegis")
    parsed = bool(prediction.get("parse_ok"))
    agent = str(prediction.get("agent") or "")
    agent_hit = parsed and (agent == label.agent if benchmark == "aegis" else _actor_match(agent, label.agent))
    result = {"parse_ok": parsed, "agent_hit": agent_hit}
    if benchmark == "who_when":
        if type(label.step) is not int:
            raise ValueError("Who&When evaluation requires a step label")
        step_hit = parsed and type(prediction.get("step")) is int and prediction["step"] == label.step
        result.update(exact_step=step_hit, joint_hit=step_hit and agent_hit)
    return result
