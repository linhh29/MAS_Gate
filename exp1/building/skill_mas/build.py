"""Shared fixed-phase Skill-MAS pipeline builder/executor."""

from __future__ import annotations

import asyncio
import ast
import inspect
import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

_BUILDING_ROOT = Path(__file__).resolve().parents[1]
if str(_BUILDING_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUILDING_ROOT))

from template.mas_code_template import prepend_fixed_imports, stage3_llm_reference_bundle

@dataclass(frozen=True)
class RouteDecision:
    selected_skill_name: str
    reasoning: str
    raw_response: str
    usage: dict[str, Any]


@dataclass(frozen=True)
class PhaseExecutionResult:
    text: str
    usage: dict[str, Any]
    meta: dict[str, Any]


@dataclass(frozen=True)
class PhasePipelineResult:
    prior_outputs: dict[str, str]
    per_phase: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    usage_totals: dict[str, Any]


@dataclass(frozen=True)
class SubTaskSpec:
    role_name: str
    role_instruction: str
    user_prompt: str = ""
    tool_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class BuildStageTrace:
    stage: int
    stage_name: str
    prompt: str
    raw_response: str
    parsed_json: dict[str, Any]
    elapsed_sec: float


@dataclass(frozen=True)
class ThreeStageBuildArtifacts:
    mas_code: str
    stage_traces: list[BuildStageTrace]
    stage1: dict[str, Any]
    stage2: dict[str, Any]
    stage3: dict[str, Any]
    normalized_sub_agents: list[dict[str, Any]]


@dataclass(frozen=True)
class MASRunWithRetryResult:
    success: bool
    final_output: str
    state: dict[str, Any]
    mas_code: str
    artifacts: ThreeStageBuildArtifacts | None
    generation_attempts_used: int
    execution_attempts_used: int
    failure_stage: str | None
    failure_reason: str | None
    retry_events: list[dict[str, Any]]
    graph_edges: list[dict[str, str]] | None = None


AsyncTextCallFn = Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]
SyncTextCallFn = Callable[[str], tuple[str, dict[str, Any]]]
AsyncToolCallFn = Callable[[Any], Awaitable[tuple[str, dict[str, Any]]]]
SyncToolCallFn = Callable[[Any], tuple[str, dict[str, Any]]]

TOOL_EXECUTION_MODES = ("llm_only", "multi_turn_search", "vita_tool", "interactive_env")

# When Stage-2 omits user_prompt, use a short embed-safe default (full task is in GLOBAL TASK CONTEXT at runtime).
_RUNTIME_USER_PROMPT_FALLBACK = (
    "Read GLOBAL TASK CONTEXT for the full task, constraints, and required output format. "
    "Use UPSTREAM DEPENDENCIES DATA when present. Execute only the sub-task described in ROLE INSTRUCTION."
)


def _strip_leading_import_noise(mas_code: str) -> str:
    """
    Remove leading import / __future__ lines from model-emitted ``mas_code`` so that
    ``prepend_fixed_imports`` does not duplicate import blocks (a common cause of SyntaxError).
    """
    lines = (mas_code or "").splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        if s.startswith("from __future__"):
            i += 1
            continue
        if s.startswith("from typing"):
            i += 1
            continue
        if s.startswith("import "):
            i += 1
            continue
        if "template.sub_agent" in s and s.startswith("from "):
            i += 1
            continue
        break
    return "\n".join(lines[i:]).lstrip("\n")


def _normalize_dataset_name(dataset_name: str | None) -> str:
    s = (dataset_name or "").strip().lower()
    if s in {"hle", "hlemath", "aime"}:
        return "hlemath"
    if s in {"gpqa", "gpqa_diamond"}:
        return "gpqa"
    if s in {"drb", "drb_bridge"}:
        return "drb"
    if s in {"bcp", "browsecomp"}:
        return "bcp"
    if s in {"vita", "vitabench"}:
        return "vita"
    if s in {"alfworld", "alf_world"}:
        return "alfworld"
    if s in {"scienceworld", "science_world"}:
        return "scienceworld"
    raise ValueError(
        "Unsupported dataset_name. Expected one of: "
        "hlemath|hle|aime, gpqa|gpqa_diamond, drb|drb_bridge, bcp|browsecomp, vita|vitabench, alfworld, scienceworld; "
        f"got {dataset_name!r}"
    )


def _dataset_final_requirement_text(normalized_dataset: str) -> str:
    """
    What the *benchmark / evaluator* ultimately expects for this dataset.
    Fed into Stage 1 so the JSON \"goal\" and decomposition align with downstream grading.
    Summaries are aligned with dataset-specific runner prompts (e.g. BCP single-agent prompts).
    """
    d = (normalized_dataset or "").strip().lower()

    # hard_constraints = "When you design the whole multi-agent system, you must consider the final required output format of the dataset. For example, when it ask for a comprehensive report, your final node/agent should output a comprehensive report. Even when you use a validator or checker, you should output the final answer in the required format after the validation or checking."
    if d == "bcp":
        return (
            "BrowseComp-Plus (open-domain QA with retrieval): behave as a careful assistant; "
            "answers must be grounded only in the provided retrieval snippets, not in unstated facts. "
            "The required deliverable is a single final answer string only (no chain-of-thought, no bullet list)."
        )
    if d == "gpqa":
        return (
            "GPQA Diamond: answer the multiple-choice science question by selecting exactly one "
            "option letter from A, B, C, or D. Compare all four choices carefully. The final line "
            "must be exactly one answer in the form \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
        )
    if d == "hlemath":
        return (
            "HLE/Math: IMO-level problem solving with emphasis on correctness and concise derivation. "
            "The final line of the solution must be exactly one answer in the form \\boxed{...} "
            "(strict final-answer formatting as in the single-agent user prompt)."
        )
    if d == "drb":
        return (
            "Deep Research Bench: produce a long-form research article that fully addresses the task prompt. "
            "Please provide a comprehensive research report."
        )
    if d == "alfworld":
        return (
            "ALFWorld is an interactive embodied household benchmark. All roles share one live episode. "
            "Use llm_only for planning/critique and interactive_env only for roles that actually execute commands. "
            "Success is determined only by environment won/goal-condition state, not by textual claims."
        )
    if d == "scienceworld":
        return (
            "ScienceWorld is an interactive science laboratory benchmark. All roles share one live episode. "
            "Use llm_only for planning/critique and interactive_env only for roles that execute simulator commands. "
            "Grading uses official simulator score/completion state, not exact-match text. Follow public task clauses "
            "in order. Planning roles may propose procedures but must not infer an experimental result before an "
            "interactive executor obtains live simulator observations. A generic referent such as 'the thing' must be "
            "identified from the requested category and observations before focus; conditional focus/box choices must "
            "follow evidence."
        )
    if d == "vita":
        return (
            "[IMPORTANT] VitaBench grading contract (mandatory for decomposition): scoring is based on **verified tool "
            "calls and simulator state** (orders, reservations, payments), not on natural-language plans or "
            '"audit passed" summaries alone. '
            "When the user asks to buy, book, order, pay, or reserve something, you MUST decompose into "
            "sub-tasks whose **success_criteria** imply completing the transaction via environment tools "
            "(e.g. appropriate create_*/pay_*/book_* style actions as required by the scenario), not stopping "
            "at search/list/price-check/information-only steps unless the task is explicitly lookup-only. "
            "Never assign a sub-agent instructions that forbid booking or purchasing when the task requires a "
            "completed transaction. Names like *_search only are discouraged when the rubric expects a "
            "completed purchase or reservation — prefer sub-task intents that match **done** actions."
        )
    raise ValueError(
        "Dataset not specified or generic: infer the intended deliverable and evaluation style from "
        "[TASK_TEXT] alone; prefer concise outputs that match typical grading for this task type."
    )


# --- HLEMATH-only build quality helpers (must not affect other datasets) ---

_HLEMATH_BANNED_NODE_RE = re.compile(
    r"literature|known_result|web_search|multi_turn|bm25|"
    r"retrieve_evidence|citation_search|open_domain|search_agent",
    re.IGNORECASE,
)
_HLEMATH_VERIFY_NODE_RE = re.compile(
    r"verify|validation|validate|check|audit|reconcile|consistency",
    re.IGNORECASE,
)
_HLEMATH_FINAL_ROLE_RE = re.compile(
    r"final|format|boxed|answer|finalize|reconcile",
    re.IGNORECASE,
)


def _hlemath_topology_templates_block() -> str:
    return (
        "[HLEMATH_TOPOLOGY_TEMPLATE — mandatory]\n"
        "Use solve_audit_repair_finalize: (1) one primary solver derives a complete solution from GLOBAL TASK CONTEXT, "
        "(2) a proof auditor reads that derivation and checks the first potentially fatal step, arithmetic, domains, "
        "boundaries, case completeness, duplicate counting, and extraneous roots, (3) a repair/reconciliation agent "
        "changes the proposed result only when it can exhibit a specific invalid equation, omitted case, or counterexample, "
        "and (4) a final answer agent preserves the audited result and emits one \\boxed{integer}.\n"
        "Use a compact serial chain primary_solver -> proof_auditor -> conservative_repairer -> final_answer. "
        "Do not create multiple independent full solvers: prior trajectories showed high disagreement and error propagation.\n"
        "Forbidden node capabilities for HLEMATH (llm_only): literature recall, web/search retrieval, "
        "citation lookup, or 'known results' databases — they add hallucination risk without tools.\n"
        "Use exactly 4 high-signal nodes.\n"
    )


def _hlemath_stage1_build_constraints() -> list[str]:
    return [
        "Use the mandatory solve_audit_repair_finalize topology in [HLEMATH_TOPOLOGY_TEMPLATE].",
        "Create exactly four nodes in one serial chain: primary_solver -> proof_auditor -> conservative_repairer -> final_answer.",
        "The primary solver must solve the whole problem; downstream nodes audit and repair rather than restart unrelated full solutions.",
        "The proof auditor must identify the first fatal error or explicitly mark the derivation VERIFIED; mere disagreement is not evidence.",
        "The repairer may change the candidate only with a concrete invalid equation, omitted case, boundary counterexample, or arithmetic contradiction.",
        "Do not hardcode unstated assumptions in decomposition; ambiguities must be stated and resolved from the problem text.",
        "Check arithmetic, signs, integer/domain restrictions, boundary cases, completeness, duplicate counting, and extraneous roots as applicable.",
        "Do not design retrieval/literature/search nodes — HLEMATH runs llm_only.",
    ]


def _hlemath_stage2_build_constraints() -> list[str]:
    return (
        "[HLEMATH_AGENT_PROTOCOL]\n"
        "- Use exactly four serial roles: primary solve, proof audit, conservative repair, final formatting.\n"
        "- Primary solver: give a complete derivation and end with `Proposed integer: N` (no boxed answer).\n"
        "- Proof auditor: check the proposed derivation line by line. End with either `Audit: VERIFIED; integer N` or "
        "`Audit: ERROR; corrected integer N; reason <specific fatal step>`. Never change an answer merely because another route feels plausible.\n"
        "- Conservative repairer: preserve a VERIFIED proposal. For ERROR, independently validate the cited fatal step before accepting the correction; "
        "end with `Repaired integer: N` and one-sentence evidence.\n"
        "- FINAL agent: perform only a last consistency and range check; do not restart the problem or override audited evidence without a concrete contradiction. "
        "Final line must be exactly one \\boxed{...}.\n"
        "- All inter-agent deliverables are plain text; no JSON/dict/list payloads or markdown code fences.\n"
        "- Forbid unstated mathematical assumptions; mark unavoidable ones with ASSUMPTION: and justify.\n"
    )

def _stage1_root_names(stage1: dict[str, Any]) -> list[str]:
    raw = stage1.get("sub_tasks")
    items = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
    names = [str(x.get("name") or "") for x in items if str(x.get("name") or "")]
    incoming = {name: 0 for name in names}
    deps = stage1.get("dependencies")
    if isinstance(deps, list):
        for edge in deps:
            if not isinstance(edge, dict):
                continue
            target = str(edge.get("to") or "")
            if target in incoming:
                incoming[target] += 1
    return [name for name in names if incoming[name] == 0]


def _gpqa_stage1_build_constraints() -> list[str]:
    return [
        "Use a compact four-node dual-solver topology: primary_domain_solver and independent_option_eliminator as roots; evidence_gate_reconciler depends on both; final_option_mapper depends on the reconciler.",
        "The two solver nodes have no incoming dependencies and solve from GLOBAL TASK CONTEXT independently.",
        "Require each solver to preserve the original A/B/C/D mapping, compare every option, and state one decisive testable scientific claim.",
        "The evidence gate must preserve root agreement unless it can cite a concrete contradiction; on disagreement it must reject an option only by a specific equation, definition, unit/sign check, limiting case, or factual mechanism.",
        "The final mapper must copy the gate's verified letter after checking only original option mapping; it must not re-solve or overrule the scientific conclusion.",
        "Avoid generic role proliferation and serial chains; use exactly four high-signal nodes.",
    ]


def _gpqa_stage2_build_constraints() -> str:
    return (
        "[GPQA_AGENT_PROTOCOL]\n"
        "- Create exactly two independent root solvers, one conservative evidence-gate reconciler, and one final option mapper.\n"
        "- Root solver 1: solve from scratch, compare all A-D, and end with `Candidate option: X; decisive claim: ...`.\n"
        "- Root solver 2: independently eliminate A-D, seek a counterexample to its leader, and end with `Candidate option: X; decisive claim: ...`.\n"
        "- Evidence gate: if roots agree, preserve that letter unless a concrete calculation/definition/wording contradiction disproves it. "
        "If roots disagree, test the competing decisive claims and name the exact evidence that rejects one; never use confidence, verbosity, or majority vote. "
        "End with exactly `Verified option: X`.\n"
        "- Final mapper: do not solve the science question again. Check only that X maps to the same original option, then output exactly one boxed uppercase letter.\n"
        "- Plain text only; no JSON/YAML/code fences in runtime agent outputs.\n"
    )

def _gpqa_validate_stage1_decomposition(stage1: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    raw = stage1.get("sub_tasks")
    items = [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []
    if len(items) != 4:
        issues.append(f"GPQA decomposition must contain exactly 4 nodes, got {len(items)}.")
    roots = _stage1_root_names(stage1)
    if len(roots) < 2:
        issues.append(f"GPQA requires two independent root solvers, found roots={roots}.")
    blobs = [f"{x.get('name','')} {x.get('description','')}" for x in items]
    if not any(_HLEMATH_VERIFY_NODE_RE.search(blob) for blob in blobs):
        issues.append("GPQA decomposition lacks an explicit verifier/reconciler node.")
    if not any(re.search(r"option|choice|eliminat", blob, re.I) for blob in blobs):
        issues.append("GPQA decomposition lacks option-by-option comparison/elimination capability.")
    return len(issues) == 0, issues


def _canonical_dataset_stage1_decomposition(*, dataset: str, task_text: str) -> dict[str, Any]:
    """Return a deterministic high-signal topology when model-generated repair remains invalid."""
    if dataset == "gpqa":
        return {
            "goal": task_text,
            "sub_tasks": [
                {
                    "name": "primary_domain_solver",
                    "description": "Independently solve the full scientific question from GLOBAL TASK CONTEXT and compare all original options A-D.",
                },
                {
                    "name": "independent_option_eliminator",
                    "description": "Independently test and eliminate every original option A-D, seeking counterexamples to the leading choice.",
                },
                {
                    "name": "evidence_gate_reconciler",
                    "description": "Preserve agreement unless concrete scientific evidence disproves it; on disagreement, test the competing decisive claims and reject one with an explicit reason.",
                },
                {
                    "name": "final_option_mapper",
                    "description": "Map the verified conclusion to the same original option letter without re-solving or overriding it.",
                },
            ],
            "constraints": _gpqa_stage1_build_constraints(),
            "dependencies": [
                {"from": "primary_domain_solver", "to": "evidence_gate_reconciler", "type": "data"},
                {"from": "independent_option_eliminator", "to": "evidence_gate_reconciler", "type": "data"},
                {"from": "evidence_gate_reconciler", "to": "final_option_mapper", "type": "data"},
            ],
            "success_criteria": [
                "Both independent roots compare all A-D options.",
                "The verifier resolves the answer from evidence rather than voting.",
                "The final selector preserves exact original option-letter mapping.",
            ],
            "reasoning": "Deterministic structural fallback enforcing independent solve, adversarial reconciliation, and strict final mapping.",
            "_structural_fallback": {"stage": 1, "reason": "model-generated decomposition and one repair attempt were invalid"},
        }
    if dataset == "hlemath":
        return {
            "goal": task_text,
            "sub_tasks": [
                {
                    "name": "primary_solver",
                    "description": "Independently solve the complete problem from GLOBAL TASK CONTEXT with a full derivation.",
                },
                {
                    "name": "proof_auditor",
                    "description": "Audit the primary derivation line by line and identify the first concrete fatal error, or explicitly mark it verified.",
                },
                {
                    "name": "conservative_repairer",
                    "description": "Preserve a verified result; otherwise validate the cited error and make only the necessary correction.",
                },
                {
                    "name": "final_answer",
                    "description": "Perform a final consistency and range check, preserve the audited result, and emit one boxed AIME integer.",
                },
            ],
            "constraints": _hlemath_stage1_build_constraints(),
            "dependencies": [
                {"from": "primary_solver", "to": "proof_auditor", "type": "data"},
                {"from": "proof_auditor", "to": "conservative_repairer", "type": "data"},
                {"from": "conservative_repairer", "to": "final_answer", "type": "data"},
            ],
            "success_criteria": [
                "One complete solution is audited line by line.",
                "Any answer change is supported by a concrete fatal error or counterexample.",
                "The final answer is a single integer in the AIME range 0..999.",
            ],
            "reasoning": "Deterministic structural fallback enforcing solve, audit, conservative repair, and stable finalization.",
            "_structural_fallback": {"stage": 1, "reason": "model-generated decomposition and one repair attempt were invalid"},
        }
    raise ValueError(f"No canonical Stage-1 decomposition for dataset={dataset!r}")


def _build_dataset_stage1_repair_prompt(*, dataset: str, original_prompt: str, stage1_json: dict[str, Any], issues: list[str]) -> str:
    topology = (
        "exactly four nodes: two independent root solvers -> one verifier/reconciler -> one final selector"
        if dataset == "gpqa" else
        "exactly four serial nodes: primary solver -> proof auditor -> conservative repairer -> final boxed-answer agent"
    )
    return (
        f"You are revising Stage-1 {dataset.upper()} task decomposition only; do not solve the benchmark item.\n"
        f"Required topology: {topology}.\n"
        "Return one valid JSON object matching [OUTPUT_JSON_SCHEMA] in ORIGINAL_STAGE_PROMPT.\n"
        "All node IDs must remain lowercase snake_case; dependencies must explicitly encode the fan-out/fan-in graph.\n\n"
        + "[ISSUES_TO_FIX]\n" + "\n".join(f"- {x}" for x in issues) + "\n\n"
        + f"[PREVIOUS_STAGE1_JSON]\n{json.dumps(stage1_json, ensure_ascii=False, indent=2)}\n\n"
        + f"[ORIGINAL_STAGE_PROMPT]\n{original_prompt}\n\nOutput revised Stage-1 JSON now."
    )


def _hlemath_validate_stage1_decomposition(stage1: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    raw = stage1.get("sub_tasks")
    items: list[dict[str, Any]] = []
    if isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        items = [raw]
    n = len(items)
    if n != 4:
        issues.append(f"HLEMATH requires exactly 4 solve/audit/repair/final nodes, got {n}.")
    roots = _stage1_root_names(stage1)
    if len(roots) != 1:
        issues.append(f"HLEMATH requires one primary root followed by a serial audit chain, found roots={roots}.")
    names = {str(x.get("name") or "") for x in items}
    deps = [x for x in (stage1.get("dependencies") or []) if isinstance(x, dict)]
    valid_edges = [x for x in deps if str(x.get("from") or "") in names and str(x.get("to") or "") in names]
    incoming = {name: 0 for name in names}
    outgoing = {name: 0 for name in names}
    for edge in valid_edges:
        incoming[str(edge.get("to"))] += 1
        outgoing[str(edge.get("from"))] += 1
    if n == 4 and not (
        len(valid_edges) == 3
        and sorted(incoming.values()) == [0, 1, 1, 1]
        and sorted(outgoing.values()) == [0, 1, 1, 1]
    ):
        issues.append("HLEMATH dependencies must form one four-node serial chain without fan-out or fan-in.")
    has_verify = False
    has_repair = False
    has_final = False
    for item in items:
        name = str(item.get("name") or "")
        desc = str(item.get("description") or "")
        blob = f"{name} {desc}"
        if _HLEMATH_BANNED_NODE_RE.search(blob):
            issues.append(
                f"node '{name}' suggests retrieval/literature/search, which is unsuitable for llm_only HLEMATH."
            )
        if _HLEMATH_VERIFY_NODE_RE.search(blob):
            has_verify = True
        if re.search(r"repair|correct|reconcil", blob, re.I):
            has_repair = True
        if _HLEMATH_FINAL_ROLE_RE.search(blob):
            has_final = True
    if n >= 3 and not has_verify:
        issues.append(
            "decomposition lacks an explicit proof-audit node before repair/final formatting."
        )
    if n >= 3 and not has_repair:
        issues.append("decomposition lacks a conservative repair/reconciliation node.")
    if n >= 3 and not has_final:
        issues.append("decomposition lacks an explicit final-answer node.")
    return len(issues) == 0, issues


def _build_hlemath_stage1_repair_prompt(
    *,
    original_prompt: str,
    stage1_json: dict[str, Any],
    issues: list[str],
) -> str:
    return (
        "You are revising Stage-1 HLEMATH task decomposition only (not solving the math problem).\n"
        "Fix the structural issues below while preserving the original task intent.\n"
        "Requirements:\n"
        "- Return ONE valid JSON object matching [OUTPUT_JSON_SCHEMA] from ORIGINAL_STAGE_PROMPT.\n"
        "- Use exactly four serial nodes: primary_solver -> proof_auditor -> conservative_repairer -> final_answer.\n"
        "- Include exactly three dependencies forming that chain; no fan-out or fan-in.\n"
        "- No retrieval/literature/search nodes.\n"
        "- sub_tasks describe capabilities/processes only, not solution content.\n\n"
        f"[ISSUES_TO_FIX]\n" + "\n".join(f"- {x}" for x in issues) + "\n\n"
        f"[PREVIOUS_STAGE1_JSON]\n{json.dumps(stage1_json, ensure_ascii=False, indent=2)}\n\n"
        f"[ORIGINAL_STAGE_PROMPT]\n{original_prompt}\n\n"
        "Output revised Stage-1 JSON now."
    )


def _hlemath_validate_stage2_sub_agents(
    subs: list[SubTaskSpec],
    *,
    final_idx: int,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for idx, spec in enumerate(subs):
        rn = (spec.role_name or "").strip().lower()
        ri = (spec.role_instruction or "").lower()
        is_final = idx == final_idx
        if not is_final:
            if re.search(r"\\boxed\s*\{", ri) or "final line must be" in ri and "boxed" in ri:
                issues.append(
                    f"intermediate agent '{spec.role_name}' instructs a final \\boxed{{...}} output."
                )
            if _HLEMATH_FINAL_ROLE_RE.search(rn) and not _HLEMATH_VERIFY_NODE_RE.search(rn):
                issues.append(
                    f"intermediate agent '{spec.role_name}' looks like a final-role name; "
                    "rename or narrow its scope to non-final derivation."
                )
    return len(issues) == 0, issues


def _hlemath_role_looks_like_final(role_name: str) -> bool:
    rn = (role_name or "").strip().lower()
    if not rn:
        return False
    if _HLEMATH_VERIFY_NODE_RE.search(rn):
        return False
    return bool(_HLEMATH_FINAL_ROLE_RE.search(rn))


def _hlemath_validate_mas_code_behavior(
    mas_code: str,
    *,
    sub_agent_role_names: list[str],
    preferred_final_role_name: str | None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    fo_lines = [
        ln
        for ln in mas_code.splitlines()
        if re.search(r'state\["final_output"\]\s*=', ln)
        and not ln.strip().startswith("#")
    ]
    agent_ref_lines = [
        ln for ln in fo_lines if re.search(r'state\["out_\w+"\]', ln)
    ]
    if len(agent_ref_lines) != 1:
        issues.append(
            f"expected exactly one agent-referenced final_output assignment, found {len(agent_ref_lines)}."
        )
    final_role = (preferred_final_role_name or "").strip()
    if final_role:
        expected = f'state["out_{final_role}"]'
        if agent_ref_lines and expected not in agent_ref_lines[0]:
            issues.append(
                f"final_output should reference {expected} (preferred final agent for HLEMATH)."
            )
        if expected not in mas_code:
            issues.append(f"mas_code never writes {expected}; final agent may be missing from workflow.")
    for role in sub_agent_role_names:
        if f'out_{role}' in mas_code or f'"out_{role}"' in mas_code:
            continue
        # Role keys in generated code may be sanitized; only warn on preferred final
        if role == final_role:
            issues.append(f"preferred final role '{role}' has no out_* state key in mas_code.")
    return len(issues) == 0, issues


def _allowed_tool_modes_for_dataset(dataset_name: str) -> set[str]:
    if dataset_name in {"hlemath", "drb", "aime", "gpqa"}:
        return {"llm_only"}
    if dataset_name == "bcp":
        # BCP needs retrieval for evidence discovery, but reasoning-only nodes
        # should not launch another retrieval session over already collected
        # evidence.  Stage 2 is responsible for routing each role explicitly.
        return {"llm_only", "multi_turn_search"}
    if dataset_name == "vita":
        return {"vita_tool"}
    if dataset_name in {"alfworld", "scienceworld"}:
        # Embodied workflows need reasoning roles as well as environment executors.
        # Forcing every conceptual node through interactive_env previously spent eight
        # actions on pseudo-phases such as initialization, analysis, and reporting.
        return {"llm_only", "interactive_env"}
    raise ValueError(
        "Unsupported dataset_name. Expected one of: "
        "hlemath|hle|aime, gpqa|gpqa_diamond, drb|drb_bridge, bcp|browsecomp, vita|vitabench, alfworld, scienceworld; "
        f"got {dataset_name!r}"
    )


def _default_execution_mode_for_dataset(allowed: set[str]) -> str:
    """When the model omits execution_mode or picks one illegal for this dataset, fall back legally."""
    if len(allowed) == 1:
        return next(iter(allowed))
    if "llm_only" in allowed:
        return "llm_only"
    return sorted(allowed)[0]


def _sanitize_tool_context(tool_context: dict[str, Any] | None, *, dataset_name: str) -> dict[str, Any]:
    tc = dict(tool_context or {})
    raw_mode = str(tc.get("execution_mode", "")).strip().lower()
    allowed = _allowed_tool_modes_for_dataset(dataset_name)
    valid = set(TOOL_EXECUTION_MODES)
    mode = raw_mode if raw_mode in valid else None
    if mode is None:
        mode = _default_execution_mode_for_dataset(allowed)
    elif mode not in allowed:
        mode = _default_execution_mode_for_dataset(allowed)
    tc["execution_mode"] = mode
    return tc


def _bcp_stage1_build_constraints() -> list[str]:
    """Keep BCP decomposition aligned with role-specific tool routing."""
    return [
        "Separate external evidence acquisition from reasoning over already acquired evidence: do not make every node a retrieval node.",
        "Describe each sub-task so Stage 2 can tell whether it truly needs new external documents or only reasons over GLOBAL TASK CONTEXT and UPSTREAM DEPENDENCIES DATA.",
        "Use retrieval-oriented nodes for discovering/identifying facts, entities, sources, or missing evidence; these nodes will use multi_turn_search.",
        "Use reasoning-oriented nodes for constraint reconciliation, evidence comparison, validation without new facts, answer synthesis, and exact-output formatting; these nodes will use llm_only.",
        "The final answer node should normally synthesize the retrieved upstream evidence without starting a fresh broad search.",
    ]


def _bcp_stage2_routing_constraints() -> list[str]:
    """Prompt rules for choosing BCP execution mode by actual node capability."""
    return [
        "For BrowseComp (bcp), route every sub-agent by its actual need for NEW external evidence; do not assign one execution mode to all nodes.",
        "Use tool_context {\"execution_mode\":\"multi_turn_search\"} only for roles that must discover, identify, or retrieve facts/documents not already available in upstream evidence.",
        "Use tool_context {\"execution_mode\":\"llm_only\"} for pure reasoning over GLOBAL TASK CONTEXT and UPSTREAM DEPENDENCIES DATA, including constraint reconciliation, evidence comparison, validation that needs no new facts, final synthesis, and exact-output formatting.",
        "The final synthesis/answer agent should normally be llm_only and consume upstream retrieved evidence. Do not repeat broad retrieval in the final node merely because BCP is a retrieval benchmark.",
        "A verifier may use multi_turn_search only when its explicit responsibility is to fill a concrete evidence gap; otherwise verification over supplied evidence must be llm_only.",
        "Ensure the workflow still contains at least one multi_turn_search evidence-acquisition role for open-domain BCP questions.",
        "In the Stage-2 reasoning field, briefly justify every role's execution_mode choice in terms of whether that role needs new documents.",
    ]


def _enforce_dataset_tool_modes_in_mas_code(mas_code: str, *, dataset_name: str) -> str:
    """
    Stage 3 codegen may ignore Stage-2-normalized ``tool_context`` and copy template placeholders
    (e.g. ``llm_only`` from ``template_text``). Rewrite ``SubAgentRequest`` literals to match the
    dataset's allowed modes (same rules as ``_sanitize_tool_context``).
    """
    try:
        nd = _normalize_dataset_name(dataset_name)
    except ValueError:
        return mas_code
    allowed = _allowed_tool_modes_for_dataset(nd)
    if len(allowed) != 1:
        return mas_code
    only = next(iter(allowed))
    out = mas_code
    if only == "vita_tool":
        for wrong in ("llm_only", "multi_turn_search", "interactive_env"):
            out = out.replace(f'"execution_mode": "{wrong}"', '"execution_mode": "vita_tool"')
            out = out.replace(f"'execution_mode': '{wrong}'", "'execution_mode': 'vita_tool'")
        return out
    if only == "llm_only":
        for wrong in ("vita_tool", "multi_turn_search", "interactive_env"):
            out = out.replace(f'"execution_mode": "{wrong}"', '"execution_mode": "llm_only"')
            out = out.replace(f"'execution_mode': '{wrong}'", "'execution_mode': 'llm_only'")
        return out
    if only == "multi_turn_search":
        for wrong in ("llm_only", "vita_tool", "interactive_env"):
            out = out.replace(f'"execution_mode": "{wrong}"', '"execution_mode": "multi_turn_search"')
            out = out.replace(f"'execution_mode': '{wrong}'", "'execution_mode': 'multi_turn_search'")
        return out
    if only == "interactive_env":
        for wrong in ("llm_only", "vita_tool", "multi_turn_search"):
            out = out.replace(f'"execution_mode": "{wrong}"', '"execution_mode": "interactive_env"')
            out = out.replace(f"'execution_mode': '{wrong}'", "'execution_mode': 'interactive_env'")
        return out
    return out


def _sanitize_final_output_in_mas_code(
    mas_code: str,
    *,
    sub_agent_role_names: list[str],
    preferred_final_role_name: str | None = None,
) -> str:
    """
    Post-process Stage-3 ``mas_code`` to enforce P0 correctness rules:

    1. Remove **all** hardcoded string-literal assignments to ``state["final_output"]``
       (especially ``"UNANSWERABLE: ..."``, but also any other static string).
       The final output MUST come from an executed sub-agent, not a hardcoded fallback.

    2. When multiple assignments to ``state["final_output"]`` exist, keep only the
       **last** one that references a ``state["out_..."]`` key (i.e. a sub-agent output).
       Remove all earlier assignments, especially UNANSWERABLE fallbacks.

    3. If no ``state["final_output"] = state["out_..."]`` assignment exists at all,
       insert one that references the **last** sub-agent defined in ``__init__``.
       This handles the common case where Stage-3 omits or botches the final assignment.

    4. Replace f-string concatenation assignments (``f"...{state.get(...)}..."``)
       with a direct sub-agent output reference, preferring the final/merge agent.

    Parameters
    ----------
    mas_code:
        Generated Python source from Stage 3.
    sub_agent_role_names:
        Ordered list of role names from ``normalized_sub_agents`` — used to determine
        the intended final agent when no valid assignment is found.
    """
    lines = mas_code.splitlines()
    out_lines: list[str] = []
    removed_unanswerable = 0
    removed_hardcoded = 0
    removed_f_string = 0
    kept_agent_ref_lines: list[str] = []

    # Pattern for state["final_output"] = ...
    fo_assign_re = re.compile(r'^(\s*)state\["final_output"\]\s*=\s*(.+)$')
    # Identify value types
    unanswerable_re = re.compile(r'^\s*["\']UNANSWERABLE', re.IGNORECASE)
    hardcoded_str_re = re.compile(r'^\s*["\'][^"\']*["\']\s*$')
    fstring_re = re.compile(r'^\s*f["\']')
    agent_ref_re = re.compile(r'^\s*state\["out_(\w+)"\]')
    # Also match state.get patterns that are used in f-string contexts
    state_get_re = re.compile(r'^\s*state\.get\(')

    for line in lines:
        m = fo_assign_re.match(line)
        if m is None:
            out_lines.append(line)
            continue

        indent = m.group(1)
        value = m.group(2).strip()

        # Case 1: UNANSWERABLE hardcoded string → remove entirely
        if unanswerable_re.match(value):
            removed_unanswerable += 1
            # Drop a comment instead so the code structure stays valid
            out_lines.append(f"{indent}# [P0-SANITIZE] removed UNANSWERABLE hardcoded fallback")
            continue

        # Case 2: Other hardcoded string literal (not UNANSWERABLE, but still not agent output)
        if hardcoded_str_re.match(value) and not agent_ref_re.match(value):
            removed_hardcoded += 1
            out_lines.append(f"{indent}# [P0-SANITIZE] removed hardcoded string final_output")
            continue

        # Case 3: f-string concatenation → remove (will be replaced with agent ref later)
        if fstring_re.match(value):
            removed_f_string += 1
            out_lines.append(f"{indent}# [P0-SANITIZE] removed f-string concatenation final_output")
            continue

        # Case 4: agent reference — keep, and track it
        if agent_ref_re.match(value):
            kept_agent_ref_lines.append(line)
            out_lines.append(line)
            continue

        # Case 5: intermediate variable (e.g. final_summary, final_answer)
        # These may or may not be valid. Keep them for now — if no agent_ref
        # exists, we will insert one later.
        out_lines.append(line)
        # Track as a "kept" line so we can decide later
        kept_agent_ref_lines.append(line)
        continue

    # After first pass: check if any agent_ref assignment survived.
    # Use a relaxed pattern (no ^ anchor) since kept_agent_ref_lines store full lines,
    # not just the RHS value.
    _agent_ref_relaxed = re.compile(r'state\["out_\w+"\]')
    has_agent_ref = any(_agent_ref_relaxed.search(l) for l in kept_agent_ref_lines)

    if not has_agent_ref and sub_agent_role_names:
        # No valid state["final_output"] = state["out_XXX"] found.
        # Insert one referencing the last sub-agent (which _with_hard_instruction_constraints
        # marked as is_final=True, meaning it should produce the complete final answer).
        final_role = (preferred_final_role_name or "").strip() or sub_agent_role_names[-1]
        # Find the indentation of the last forward_async section
        # Look for "return state" line to determine where to insert
        insert_idx = None
        for i, line in enumerate(out_lines):
            if re.match(r'\s*return\s+state\s*', line):
                insert_idx = i
                break

        if insert_idx is not None:
            # Find indentation from surrounding context
            indent_match = re.match(r'^(\s*)', out_lines[insert_idx])
            indent = indent_match.group(1) if indent_match else "        "
            # Insert the final_output assignment just before "return state"
            insert_line = f'{indent}state["final_output"] = state["out_{final_role}"]'
            out_lines.insert(insert_idx, insert_line)
            # Also add usage_final if it doesn't exist
            usage_line = f'{indent}state["usage_final"] = state["usage_{final_role}"]'
            out_lines.insert(insert_idx + 1, usage_line)
            print(
                f"[P0-SANITIZE] inserted final_output assignment: "
                f'state["final_output"] = state["out_{final_role}"]',
                flush=True,
            )
        else:
            # Fallback: append at the very end of forward_async
            # Find the end of forward_async method
            for i in range(len(out_lines) - 1, -1, -1):
                if "return state" in out_lines[i] or "return dict" in out_lines[i]:
                    insert_idx = i
                    break
            if insert_idx is not None:
                indent_match = re.match(r'^(\s*)', out_lines[insert_idx])
                indent = indent_match.group(1) if indent_match else "        "
                out_lines.insert(insert_idx, f'{indent}state["final_output"] = state["out_{final_role}"]')
                out_lines.insert(insert_idx + 1, f'{indent}state["usage_final"] = state["usage_{final_role}"]')
            else:
                print("[P0-SANITIZE] WARNING: could not find return statement to insert final_output", flush=True)

    # Also: if there are multiple agent_ref assignments, keep only the last one
    # (this handles cases where both UNANSWERABLE and agent_ref were present,
    #  and we already removed the UNANSWERABLE ones above)
    _agent_ref_relaxed = re.compile(r'state\["out_\w+"\]')
    agent_ref_indices = [i for i, line in enumerate(out_lines) if fo_assign_re.match(line) and _agent_ref_relaxed.search(line)]
    if len(agent_ref_indices) > 1:
        # Keep only the last one
        for idx in agent_ref_indices[:-1]:
            indent_match = re.match(r'^(\s*)', out_lines[idx])
            indent = indent_match.group(1) if indent_match else "        "
            out_lines[idx] = f"{indent}# [P0-SANITIZE] removed duplicate final_output assignment (kept last one)"

    sanitized = "\n".join(out_lines)
    if removed_unanswerable or removed_hardcoded or removed_f_string:
        print(
            f"[P0-SANITIZE] removed: {removed_unanswerable} UNANSWERABLE, "
            f"{removed_hardcoded} hardcoded strings, {removed_f_string} f-string concatenations",
            flush=True,
        )
    return sanitized


def _enforce_final_output_role_in_mas_code(
    mas_code: str,
    *,
    final_role_name: str | None,
) -> str:
    """
    Force ``state["final_output"]`` / ``state["usage_final"]`` to reference one chosen role.

    Used only for VitaBench, where wrong final-role binding can silently drop completed branches.
    """
    role = (final_role_name or "").strip()
    if not role:
        return mas_code
    role_ref = f'state["out_{role}"]'
    if role_ref not in mas_code:
        # Do not force an impossible role reference when stage-3 omitted this output key.
        return mas_code

    lines = mas_code.splitlines()
    fo_assign_re = re.compile(r'^(\s*)state\["final_output"\]\s*=')
    uf_assign_re = re.compile(r'^(\s*)state\["usage_final"\]\s*=')
    return_idx = None
    base_indent = "        "
    new_lines: list[str] = []
    for i, line in enumerate(lines):
        if return_idx is None and re.match(r"^\s*return\s+state\s*$", line):
            return_idx = i
            m = re.match(r"^(\s*)", line)
            if m:
                base_indent = m.group(1)
        if fo_assign_re.match(line):
            indent = re.match(r"^(\s*)", line).group(1)
            new_lines.append(f"{indent}# [VITA-FINAL-ENFORCE] replaced final_output assignment")
            continue
        if uf_assign_re.match(line):
            indent = re.match(r"^(\s*)", line).group(1)
            new_lines.append(f"{indent}# [VITA-FINAL-ENFORCE] replaced usage_final assignment")
            continue
        new_lines.append(line)

    if return_idx is None:
        return mas_code

    # return_idx is for original lines; relocate in rewritten list by first matching return state.
    insert_idx = None
    for i, line in enumerate(new_lines):
        if re.match(r"^\s*return\s+state\s*$", line):
            insert_idx = i
            break
    if insert_idx is None:
        return mas_code

    forced_lines = [
        f'{base_indent}state["final_output"] = state["out_{role}"]',
        f'{base_indent}state["usage_final"] = state.get("usage_{role}", state.get("usage_final", {{}}))',
    ]
    new_lines[insert_idx:insert_idx] = forced_lines
    return "\n".join(new_lines)


def make_planner_call_fn(
    llm_client: Any,
    *,
    system_prompt: str = "You are a MAS planner. Output JSON only.",
    generation_kwargs: dict[str, Any] | None = None,
) -> Callable[[str], Awaitable[str]]:
    """
    Build a sync ``planner_call_fn`` from a generic async LLM client.

    The client is expected to expose:
      async generate(user_prompt=..., system_prompt=...) -> (text, usage)
    """

    gen_fn = getattr(llm_client, "generate")
    is_async_generate = inspect.iscoroutinefunction(gen_fn)

    planner_kwargs = dict(generation_kwargs or {})

    def _is_unsupported_generation_kw_error(err: Exception) -> bool:
        msg = str(err).lower()
        return any(
            tok in msg
            for tok in (
                "unsupported",
                "unknown",
                "unexpected keyword",
                "invalid parameter",
                "response_format",
            )
        )

    async def _planner_call_async(prompt: str) -> str:
        async def _call_once(with_kwargs: bool) -> tuple[str, dict[str, Any]]:
            call_kwargs = dict(planner_kwargs) if with_kwargs else {}
            if is_async_generate:
                return await gen_fn(user_prompt=prompt, system_prompt=system_prompt, **call_kwargs)
            return await asyncio.to_thread(
                gen_fn,
                user_prompt=prompt,
                system_prompt=system_prompt,
                **call_kwargs,
            )

        try:
            text, _usage = await _call_once(with_kwargs=bool(planner_kwargs))
        except Exception as e:
            if not planner_kwargs or not _is_unsupported_generation_kw_error(e):
                raise
            print("[planner_call_fn] generation_kwargs unsupported by backend; fallback to default kwargs.", flush=True)
            text, _usage = await _call_once(with_kwargs=False)
        return text

    return _planner_call_async


def make_text_call_fn(
    llm_client: Any,
    *,
    system_prompt: str = "",
    generation_kwargs: dict[str, Any] | None = None,
) -> Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]:
    """Build async text_call_fn from a generic async LLM client.

    ``generation_kwargs`` is intentionally supported here as well as in the
    planner wrapper.  Interactive environments require very short action
    responses; allowing their caller to cap completion length prevents a single
    malformed/repetitive Qwen response from occupying a vLLM sequence for
    minutes.
    """

    gen_fn = getattr(llm_client, "generate")
    text_kwargs = dict(generation_kwargs or {})

    async def _text_call_async(prompt: str) -> tuple[str, dict[str, Any]]:
        result = gen_fn(user_prompt=prompt, system_prompt=system_prompt, **text_kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    return _text_call_async


def merge_usage_totals(acc: dict[str, Any], usage: dict[str, Any]) -> None:
    acc["prompt_tokens"] = int(acc.get("prompt_tokens", 0)) + int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    )
    acc["output_tokens"] = int(acc.get("output_tokens", 0)) + int(usage.get("output_tokens", 0) or 0)
    acc["total_tokens"] = int(acc.get("total_tokens", 0)) + int(
        usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("output_tokens", 0)) or 0
    )
    acc["estimated_cost_usd"] = float(acc.get("estimated_cost_usd", 0.0)) + float(
        usage.get("estimated_cost_usd", 0.0) or 0.0
    )


def run_fixed_phase_pipeline(
    *,
    phase_indices: list[int],
    banks: dict[int, list[Any]],
    route_phase_fn: Callable[[int, list[Any], dict[str, str]], RouteDecision],
    build_role_instruction_fn: Callable[[int, str], tuple[str, str]],
    build_user_prompt_fn: Callable[[int, str, dict[str, str]], str],
    execute_phase_fn: Callable[[int, str, str, Any, str], PhaseExecutionResult],
    usage_merge_fn: Callable[[dict[str, Any], dict[str, Any]], None] = merge_usage_totals,
    initial_prior_outputs: dict[str, str] | None = None,
    initial_per_phase: list[dict[str, Any]] | None = None,
    initial_steps: list[dict[str, Any]] | None = None,
    initial_usage_totals: dict[str, Any] | None = None,
) -> PhasePipelineResult:
    prior_outputs: dict[str, str] = dict(initial_prior_outputs or {})
    per_phase: list[dict[str, Any]] = list(initial_per_phase or [])
    steps: list[dict[str, Any]] = list(initial_steps or [])
    usage_totals: dict[str, Any] = dict(
        initial_usage_totals
        or {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    )

    for phase_idx in phase_indices:
        specs = banks.get(phase_idx) or []
        if not specs:
            raise RuntimeError(f"phase_{phase_idx} bank empty")
        route = route_phase_fn(phase_idx, specs, prior_outputs)
        usage_merge_fn(usage_totals, route.usage)
        steps.append(
            {
                "stage": f"skill_router_phase_{phase_idx}",
                "request": {"phase": phase_idx, "candidates": [s.name for s in specs]},
                "response": {
                    "selected_skills": [route.selected_skill_name],
                    "reasoning": route.reasoning,
                    "raw_response": route.raw_response,
                },
                "usage": route.usage,
            }
        )
        selected = next((s for s in specs if s.name == route.selected_skill_name), specs[0])
        role, instruction = build_role_instruction_fn(phase_idx, selected.name)
        user_prompt = build_user_prompt_fn(phase_idx, role, prior_outputs)
        stage = execute_phase_fn(phase_idx, role, instruction, selected, user_prompt)
        usage_merge_fn(usage_totals, stage.usage)
        steps.append(
            {
                "stage": role,
                "active_skill": selected.name,
                "request": {"instruction": instruction, "user_prompt": user_prompt},
                "response": stage.text,
                "usage": stage.usage,
            }
        )
        prior_outputs[role] = stage.text
        per_phase.append(
            {
                "phase": phase_idx,
                "role": role,
                "selected_skill": selected.name,
                "selected_skill_path": str(selected.path.resolve()),
                "router_reasoning": route.reasoning,
                "candidate_skills": [s.name for s in specs],
                "router_raw_response": route.raw_response,
                **(stage.meta or {}),
            }
        )
    return PhasePipelineResult(
        prior_outputs=prior_outputs,
        per_phase=per_phase,
        steps=steps,
        usage_totals=usage_totals,
    )


async def run_fixed_phase_pipeline_async(
    *,
    phase_indices: list[int],
    banks: dict[int, list[Any]],
    route_phase_fn: Callable[[int, list[Any], dict[str, str]], RouteDecision],
    build_role_instruction_fn: Callable[[int, str], tuple[str, str]],
    build_user_prompt_fn: Callable[[int, str, dict[str, str]], str],
    execute_phase_fn: Callable[[int, str, str, Any, str], Awaitable[PhaseExecutionResult]],
    usage_merge_fn: Callable[[dict[str, Any], dict[str, Any]], None] = merge_usage_totals,
    initial_prior_outputs: dict[str, str] | None = None,
    initial_per_phase: list[dict[str, Any]] | None = None,
    initial_steps: list[dict[str, Any]] | None = None,
    initial_usage_totals: dict[str, Any] | None = None,
) -> PhasePipelineResult:
    """Same as ``run_fixed_phase_pipeline`` but awaits async ``execute_phase_fn``."""
    prior_outputs: dict[str, str] = dict(initial_prior_outputs or {})
    per_phase: list[dict[str, Any]] = list(initial_per_phase or [])
    steps: list[dict[str, Any]] = list(initial_steps or [])
    usage_totals: dict[str, Any] = dict(
        initial_usage_totals
        or {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    )

    for phase_idx in phase_indices:
        specs = banks.get(phase_idx) or []
        if not specs:
            raise RuntimeError(f"phase_{phase_idx} bank empty")
        route = route_phase_fn(phase_idx, specs, prior_outputs)
        usage_merge_fn(usage_totals, route.usage)
        steps.append(
            {
                "stage": f"skill_router_phase_{phase_idx}",
                "request": {"phase": phase_idx, "candidates": [s.name for s in specs]},
                "response": {
                    "selected_skills": [route.selected_skill_name],
                    "reasoning": route.reasoning,
                    "raw_response": route.raw_response,
                },
                "usage": route.usage,
            }
        )
        selected = next((s for s in specs if s.name == route.selected_skill_name), specs[0])
        role, instruction = build_role_instruction_fn(phase_idx, selected.name)
        user_prompt = build_user_prompt_fn(phase_idx, role, prior_outputs)
        stage = await execute_phase_fn(phase_idx, role, instruction, selected, user_prompt)
        usage_merge_fn(usage_totals, stage.usage)
        steps.append(
            {
                "stage": role,
                "active_skill": selected.name,
                "request": {"instruction": instruction, "user_prompt": user_prompt},
                "response": stage.text,
                "usage": stage.usage,
            }
        )
        prior_outputs[role] = stage.text
        per_phase.append(
            {
                "phase": phase_idx,
                "role": role,
                "selected_skill": selected.name,
                "selected_skill_path": str(selected.path.resolve()),
                "router_reasoning": route.reasoning,
                "candidate_skills": [s.name for s in specs],
                "router_raw_response": route.raw_response,
                **(stage.meta or {}),
            }
        )
    return PhasePipelineResult(
        prior_outputs=prior_outputs,
        per_phase=per_phase,
        steps=steps,
        usage_totals=usage_totals,
    )


def _load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _enumerate_json_dicts(text: str) -> list[dict[str, Any]]:
    """Scan left-to-right and parse each complete top-level JSON object (non-overlapping)."""
    raw = text or ""
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        try:
            obj, consumed = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i += consumed
    return out


def _stage_hint_from_label(stage_name: str) -> int | None:
    m = re.search(r"Stage\s*(\d+)", stage_name, flags=re.I)
    return int(m.group(1)) if m else None


def _is_tool_context_fragment(obj: dict[str, Any]) -> bool:
    """Small JSON blobs that appear before the real Stage 2 wrapper (examples / tool_context)."""
    ks = set(obj.keys())
    return ks <= {"execution_mode"} or ks == {"execution_mode", "reasoning"}


def _extract_best_json_dict_for_stage(text: str, *, stage_hint: int | None) -> dict[str, Any] | None:
    """Prefer schema-shaped dicts over incidental inner JSON (e.g. {{\"execution_mode\": ...}} only)."""
    cands = _enumerate_json_dicts(text)
    if not cands:
        return None
    if stage_hint is None:
        return cands[0]
    if stage_hint == 1:
        scored: list[tuple[int, dict[str, Any]]] = []
        for obj in cands:
            st = obj.get("sub_tasks")
            score = 0
            if isinstance(st, list) and len(st) > 0:
                score += 100 + len(st)
            elif isinstance(st, list):
                score += 10
            if obj.get("goal"):
                score += 5
            if isinstance(st, dict):
                score += 50
            n = str(obj.get("name", "")).strip()
            d = str(obj.get("description", "")).strip()
            if n and d:
                score += 25
            scored.append((score, obj))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]
    if stage_hint == 2:
        pool = [o for o in cands if not _is_tool_context_fragment(o)] or cands
        best: dict[str, Any] | None = None
        best_score = -1
        for obj in pool:
            sa = obj.get("sub_agents")
            score = 0
            if isinstance(sa, list) and len(sa) > 0:
                score = 100 + len(sa)
            elif isinstance(sa, dict):
                score = 40
            rn = str(obj.get("role_name", "")).strip()
            ri = str(obj.get("role_instruction", "")).strip()
            if rn and ri:
                score = max(score, 80)
            if score > best_score:
                best_score = score
                best = obj
        if best is None and pool:
            return max(pool, key=lambda o: len(o.keys()))
        return best
    if stage_hint == 3:
        best: dict[str, Any] | None = None
        best_len = -1
        for obj in cands:
            mc = obj.get("mas_code")
            if isinstance(mc, str) and mc.strip():
                if len(mc) > best_len:
                    best_len = len(mc)
                    best = obj
        return best
    return cands[0]


def _extract_first_json_dict(text: str) -> dict[str, Any] | None:
    return _extract_best_json_dict_for_stage(text, stage_hint=None)


def _coerce_stage1_subtasks(payload: dict[str, Any]) -> dict[str, Any]:
    """If Stage 1 omits sub_tasks[] but returns a single task at top level, normalize."""
    raw_st = payload.get("sub_tasks")
    if isinstance(raw_st, dict):
        return {**payload, "sub_tasks": [raw_st]}
    if isinstance(raw_st, list) and len(raw_st) > 0:
        return payload
    n = str(payload.get("name", "")).strip()
    d = str(payload.get("description", "")).strip()
    if n and d:
        return {**payload, "sub_tasks": [{"name": n, "description": d}]}
    return payload


def _unwrap_stage2_agent_item(item: dict[str, Any]) -> dict[str, Any]:
    """Turn {\"role_id\": {\"role_name\": \"...\", ...}} into the inner agent dict."""
    if str(item.get("role_name", "")).strip():
        return item
    for v in item.values():
        if isinstance(v, dict) and str(v.get("role_name", "")).strip():
            return v
    if len(item) == 1:
        inner = next(iter(item.values()))
        if isinstance(inner, dict):
            return inner
    return item


def _extract_skill_stages(skill_text: str) -> dict[int, str]:
    """Extract stage module bodies from headings: ## 1., ## 2., ## 3."""
    matches = list(re.finditer(r"^##\s+([123])\.[^\n]*\n", skill_text, flags=re.M))
    out: dict[int, str] = {}
    for i, m in enumerate(matches):
        stage = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(skill_text)
        out[stage] = skill_text[start:end].strip()
    missing = [s for s in (1, 2, 3) if s not in out]
    if missing:
        raise RuntimeError(f"init skill missing stage modules: {missing}")
    return out


def _mas_build_contract_general_constraints() -> str:
    """Parser + cross-stage payload rules—task *how-to* stays in init SKILL.md."""
    return (
        "[GENERAL_CONSTRAINTS]\n"
        "- Single JSON object only (no markdown fences, no surrounding prose).\n"
        "- The runner parses the first top-level `{...}` object from your reply.\n"
        "- Key names must match [OUTPUT_JSON_SCHEMA]; prior-stage outputs are embedded verbatim into "
        "later [STAGE_INPUT_JSON] blobs.\n"
        "- Fields such as ``dataset_final_requirements`` encode **benchmark grading contracts** "
        "(evaluator-facing), not optional narrative hints.\n"
        "- Cross-Stage Execution Safety: Stage 2 ``role_instruction`` / ``user_prompt`` strings and Stage 3 "
        "``mas_code`` are ultimately embedded into **executable Python**. Any content that breaks Python string "
        "literals or syntax (unescaped quotes, unmatched parentheses, duplicate import blocks, invalid dict "
        "keys) will cause **total workflow failure** at runtime. Design all text so it survives embedding as "
        "Python source.\n\n"
    )


def _mas_build_contract_specific_constraints(stage: int) -> str:
    if stage == 1:
        return (
            "[STAGE_SPECIFIC_CONSTRAINTS]\n\n"
            "Stage 1: Task Decomposition — Engineering Constraints. "
            "- Node Count Limits: the number of decomposed sub-tasks must be kept within a reasonable range "
            "(typically 3-6 for most tasks), because over-fragmentation leads to context window explosion and "
            "information dilution, while under-fragmentation defeats the purpose of a MAS. "
            "- Strict Node Naming Convention: all sub-task IDs must be globally unique and formatted strictly "
            "in lowercase snake_case (e.g., data_analyzer). This ID serves as the absolute identifier across "
            "all stages, and modifying the spelling or casing in later stages is strictly forbidden.\n"
            "- Decomposition Principle — Describe WHAT, Not HOW: you are defining the structural blueprint of "
            "the multi-agent system, NOT solving the user's problem. Each sub-task should describe a capability "
            "or processing stage (e.g., 'retrieve_evidence', 'analyze_constraints', 'synthesize_answer'), not "
            "attempt to answer the question or provide solution content. The actual solving happens later when "
            "the generated agents execute at runtime.\n"
            "- Topology Awareness: consider whether the task benefits from a simple sequential chain, or needs "
            "a fan-out/fan-in pattern (multiple independent analyses merged at the end), or an iterative loop "
            "(e.g., retrieve → analyze → refine → retrieve again). Express this via the dependency structure. "
            "For tasks requiring trial-and-error (e.g., open-domain search), design at least one iterative or "
            "fallback-capable node rather than a single one-shot attempt.\n"
            "- Information Preservation: ensure each sub-task's description is self-contained enough that a "
            "downstream agent can understand the context without needing the full original query. Avoid designs "
            "where intermediate outputs become too thin (e.g., a 'filter' node that outputs only null/empty — "
            "design instead for nodes that always produce structured, actionable output even when uncertain.\n\n"
        )
    if stage == 2:
        return (
            "[STAGE_SPECIFIC_CONSTRAINTS]\n\n"
            "Stage 2: Agent Engineering — Engineering Constraints. "
            "- Absolute 1-to-1 Node Mapping: every sub-agent generated in this stage must strictly correspond "
            "to a sub-task defined in Stage 1; the role_name must perfectly match the node ID from Stage 1. "
            "Do not omit any nodes, and do not hallucinate new agents that were not planned. "
            "- Strict Boundary Enforcement: agent instructions and user prompts must NOT contain any routing "
            "or scheduling logic. Agents should never be instructed to \"pass the result to the reviewer "
            'agent.\" Agents must only focus on completing their specific sub-task. The system framework '
            "will handle the data routing.\n"
            "- You Are Engineering Prompts, Not Solving Problems: your output is a set of agent specifications "
            "(role_instruction + user_prompt) that will be used to call an LLM at runtime. Do NOT attempt to "
            "answer the user's question yourself. Your role_instruction defines WHAT the agent should do and "
            "its behavioral boundaries. The user_prompt must be **short** (see Runtime Task Context below); "
            "the full task is always injected at runtime in ``GLOBAL TASK CONTEXT``, so you do not need to "
            "duplicate the entire problem inside ``user_prompt``.\n"
            "- Runtime Task Context (Critical): at execution time, every sub-agent prompt is assembled with a "
            "``GLOBAL TASK CONTEXT`` section that already contains the **full original task_text**. Therefore "
            "your ``user_prompt`` MUST stay **short and Python-embeddable** (plain ASCII where possible): "
            "instruct the agent to read ``GLOBAL TASK CONTEXT`` for the full problem, citations, LaTeX, and "
            "output format — do **NOT** paste the entire task, code fences, or long LaTeX into ``user_prompt`` "
            "because Stage 3 copies these strings into ``SubAgentRequest(...)`` inside generated Python and "
            "unescaped ``\\\"`` or ``\"\"\"`` substrings will cause **SyntaxError** and zero score.\n"
            "- Avoid Empty/Null Output Designs: do NOT design agents whose natural output could be empty, null, "
            "or a simple 'not found' message. For example, instead of a 'filter_candidates' agent that might "
            "output null when no candidates match, design an 'analyze_and_rank' agent that always produces a "
            "structured ranking or analysis. Every agent should produce actionable, substantive output.\n"
            "- Execution Mode Reasoning: choose tool_context.execution_mode based on the actual capabilities "
            "needed. For tasks requiring information retrieval, use 'multi_turn_search' so the agent can "
            "perform multi-round searches. For pure reasoning/analysis, use 'llm_only'. Do not default all "
            "agents to 'llm_only' — the system's effectiveness depends on correctly routing tool-capable "
            "agents to the right execution mode.\n"
            "- Upstream-to-Retrieval Bridge (Critical for multi_turn_search agents): if a sub-agent with "
            "execution_mode='multi_turn_search' depends on an upstream agent, the upstream agent's output "
            "must contain actionable retrieval guidance — not just a re-formatted restatement of the original "
            "query constraints. The multi_turn_search agent uses a BM25 (lexical keyword) retrieval engine "
            "under the hood; its planner receives the upstream output as 'Delegation / planning context'. "
            "If the upstream output only contains vague abstractions like 'amphibian' or 'early 1990s', the "
            "BM25 planner cannot generate precise search queries. Instead, the upstream agent should produce: "
            "(a) concrete candidate entity names, specific search keywords, or named entities that BM25 can "
            "match directly; (b) a 'suggested_search_queries' list of 3-5 distinct BM25-ready queries; "
            "(c) any disambiguation clues (e.g., 'the amphibian reference likely means frog/newt/salamander "
            "in a company name'). If the upstream agent is a pure constraint extractor and cannot supply "
            "such concrete terms, then DO NOT place it before a multi_turn_search agent — instead, make "
            "the multi_turn_search agent the FIRST node in the pipeline so it receives the full original "
            "query directly and can derive its own search strategy.\n\n"
        )
    if stage == 3:
        return (
            "[STAGE_SPECIFIC_CONSTRAINTS]\n\n"
            "Stage 3: Workflow Orchestration & Code Generation — Engineering Constraints. "
            "- Template Immutability: you must strictly build upon the provided MASWorkflowTemplate; do not "
            "modify the function signatures (arguments or return types) of __init__, forward_async, or "
            "build_workflow_subagent_prompt; only inject code between the designated <START> and <END> markers. "
            "- No Third-Party Orchestration Frameworks: try not to import other libraries as much as possible; "
            "if you truly need to import external libraries, do not import or attempt to use external MAS "
            "frameworks like langgraph, autogen, or crewai; use only Python standard libraries. "
            "- No Duplicate Imports: do **NOT** emit ``from __future__ import annotations``, ``from typing ...``, "
            "or ``from template.sub_agent ...`` — the runner prepends a fixed import block. Emit only the "
            "workflow class body. Duplicate imports cause invalid Python and immediate failure.\n"
            "- Syntactically Valid Python: before returning ``mas_code``, mentally verify matching "
            "parentheses/brackets/braces, valid string literals, and valid dict keys in ``upstream_outputs`` "
            "(use simple snake_case string keys only, e.g. ``\"upstream_a\"`` — never use task text fragments "
            "or LaTeX as dict keys).\n"
            "- String Literal Safety for SubAgentRequest fields: when embedding ``role_instruction`` and "
            "``user_prompt`` into Python, prefer **triple-single-quoted** raw strings ``r'''...'''`` or "
            "``'''...'''`` and ensure the chosen delimiter never appears unescaped inside the text. If you "
            "use double-quoted strings, every internal ``\"`` must be escaped as ``\\\"``. Never embed "
            "``\"\"\"`` inside triple-double-quoted strings.\n"
            "- Safe State Dictionary Access: when populating the upstream_outputs dictionary in forward_async, "
            "the values must only reference keys in the state dictionary that have ALREADY been executed and "
            "populated; do not attempt to access a node's output before it has been awaited, and do not "
            "hallucinate variable names. "
            "- Force Sequential Await (No Complex Concurrency): even if the DAG topology contains logically "
            "parallel branches, you must generate the code using sequential await calls; do not use "
            "asyncio.gather() or complex threading logic; the DAG dataflow is perfectly maintained by passing "
            "the correct variables via upstream_outputs without needing concurrent execution logic. "
            "- Mandatory Final State Assignment (Critical — Single-Source Rule): "
            'forward_async must contain **exactly ONE** assignment to state["final_output"], and it must be '
            'of the form ``state["final_output"] = state["out_<FINAL_AGENT_ROLE_NAME>"]`` — a direct reference '
            "to the output of the last/topological-sink sub-agent stored in the state dict. "
            "**This rule has three strict sub-rules:** "
            "(1) **No hardcoded string literals**: do NOT assign a fixed string like ``\"UNANSWERABLE\"`` or "
            '``\"EVIDENCE_INSUFFICIENT\"`` to ``state["final_output"]``. The final answer must always come '
            "from an agent execution, never from a hardcoded fallback string. If retrieval or analysis fails, "
            "the final agent itself must still produce its best-effort answer — a hardcoded ``UNANSWERABLE`` "
            "string completely bypasses the agent and guarantees a zero score. "
            "(2) **No f-string concatenation or intermediate variable**: do NOT use ``f\"Task summary: {...}\"`` "
            "or ``final_summary = ...; state[\"final_output\"] = final_summary`` patterns. The final_output "
            "must be exactly ``state[\"out_<ROLE>\"]`` so the runtime can reliably extract the agent's direct "
            "output without parsing wrapper text, JSON envelopes, or concatenated summaries. "
            "(3) **No duplicate assignments**: do NOT write two ``state[\"final_output\"] = ...`` lines (e.g. "
            "one fallback and one real assignment). Python executes sequentially; the last line wins, making "
            "the intent ambiguous and the code fragile. Write exactly one assignment. "
            'Additionally, assign total token usage to state["usage_final"] (typically '
            'state["usage_<FINAL_AGENT_ROLE_NAME>"]). The system relies on these exact keys to evaluate '
            "the success of the MAS. "
            "- Strict Bounding for Loops and Ensembles (Crucial): if you design advanced topologies like "
            "iterative correction loops (e.g., while not approved:) or ensemble generation (calling the same "
            "agent multiple times), or other advanced topology, you MUST hardcode a strict upper bound limit. "
            "Any loop must be explicitly capped at a maximum of 3 iterations (e.g., using a counter "
            "if attempts >= 3: break or for _ in range(3):). Infinite loops are strictly prohibited and will "
            "cause the system to crash. "
            "- Safe State Dictionary Access and Loop Variable Handling: when populating upstream_outputs or "
            "updating variables within a loop, values must only reference keys in the state dictionary that "
            "have ALREADY been executed; inside loops, ensure you correctly overwrite or append to state keys "
            '(e.g., state["current_code_draft"]) so the next iteration receives the updated data, and this '
            "requirment is the same for ensemble topology or other advanced topologies.\n"
            "- Output Fallback Logic: if a retrieval or tool-based agent produces an empty, null, or otherwise "
            "unusable output, the downstream agent should NOT propagate that null downstream. Instead, design "
            "the code to detect empty upstream outputs and either (a) re-execute the upstream agent with a "
            "modified prompt, or (b) provide a default fallback value with a clear marker. This prevents "
            "cascading null/empty failures through the pipeline.\n"
            "- Final Agent Output Guarantee: the last agent in the workflow MUST produce the complete, "
            "final answer in the exact format required by the benchmark (see dataset_final_requirements in "
            "Stage 1). Do NOT let the final agent produce a summary, meta-commentary, or anything other than "
            "the direct answer. If the upstream data is insufficient, the final agent should still attempt to "
            "produce its best answer based on available information rather than outputting null or empty.\n\n"
        )
    raise ValueError(f"Invalid stage: {stage}")


def _mas_build_contract_prefix(stage: int) -> str:
    """
    MAS-pipeline-only instructions prepended to each stage: general + stage-specific blocks.

    Keep ``SKILL.md`` generic; put JSON/codegen contracts here.
    """
    if stage not in (1, 2, 3):
        raise ValueError(f"Invalid stage: {stage}")
    return (
        f"[MAS_BUILD_CONTRACT — Stage {stage}]\n\n"
        f"{_mas_build_contract_general_constraints()}"
        f"{_mas_build_contract_specific_constraints(stage)}"
    )


def _stage_prompt(
    stage_skill: str,
    stage_input: dict[str, Any],
    output_schema: str,
    *,
    stage_index: int | None = None,
) -> str:
    """
    Minimal prompt contract:
    - optional MAS-build prefix: ``[GENERAL_CONSTRAINTS]`` + ``[STAGE_SPECIFIC_CONSTRAINTS]``
      (see ``_mas_build_contract_prefix``)
    - stage skill text (from SKILL.md sections ## 1. / ## 2. / ## 3.)
    - stage input json
    - required output schema
    """
    prefix = _mas_build_contract_prefix(stage_index) if stage_index is not None else ""
    body = (
        f"{stage_skill}\n\n"
        f"[STAGE_INPUT_JSON]\n{json.dumps(stage_input, ensure_ascii=False, indent=2)}\n\n"
        f"[OUTPUT_JSON_SCHEMA]\n{output_schema}\n"
    )
    return f"{prefix}{body}"


def _normalize_sub_tasks(payload: dict[str, Any]) -> list[SubTaskSpec]:
    raw = payload.get("sub_agents")
    # Some models return one agent dict instead of `"sub_agents": [ {...}, ... ]`, or omit `sub_agents`
    # and flatten `role_*` onto the root object.
    if isinstance(raw, dict):
        raw = [raw]
    elif not isinstance(raw, list):
        rn = str(payload.get("role_name", "")).strip()
        ri = str(payload.get("role_instruction", "")).strip()
        if rn and ri:
            raw = [
                {
                    "role_name": payload.get("role_name"),
                    "role_instruction": payload.get("role_instruction"),
                    "user_prompt": payload.get("user_prompt", ""),
                    "tool_context": payload.get("tool_context"),
                }
            ]
        else:
            return []
    out: list[SubTaskSpec] = []
    for raw_item in raw:
        if not isinstance(raw_item, dict):
            continue
        item = _unwrap_stage2_agent_item(raw_item)
        role_name = str(item.get("role_name", "")).strip()
        role_instruction = str(item.get("role_instruction", "")).strip()
        if not role_name or not role_instruction:
            continue
        user_prompt = str(item.get("user_prompt", "")).strip()
        tool_context = item.get("tool_context")
        if tool_context is not None and not isinstance(tool_context, dict):
            tool_context = None
        out.append(
            SubTaskSpec(
                role_name=role_name,
                role_instruction=role_instruction,
                user_prompt=user_prompt,
                tool_context=tool_context,
            )
        )
    return out


def _fallback_sub_agents_from_stage1(
    stage1_payload: dict[str, Any],
    *,
    dataset_name: str,
) -> list[SubTaskSpec]:
    """
    Deterministic fallback when Stage 2 output is malformed/empty.
    Uses Stage 1 decomposition to build a minimal executable agent list.
    """
    raw = stage1_payload.get("sub_tasks")
    items: list[dict[str, Any]] = []
    if isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        items = [raw]
    if not items:
        return []

    out: list[SubTaskSpec] = []
    for idx, item in enumerate(items, start=1):
        raw_name = str(item.get("name") or "").strip().lower()
        role_name = re.sub(r"[^a-z0-9_]+", "_", raw_name).strip("_") or f"stage1_node_{idx}"
        description = str(item.get("description") or "").strip()
        role_instruction = (
            f"You are the '{role_name}' sub-agent in a generated MAS workflow. "
            "Complete ONLY this sub-task and provide concise, actionable output for downstream nodes.\n\n"
            f"Sub-task description: {description or 'Follow Stage-1 decomposition intent.'}\n\n"
            "Hard constraints:\n"
            "- Do not output null/empty placeholders.\n"
            "- Keep output deterministic and directly usable by later agents.\n"
            "- Follow the required final format only if you are the final node."
        )
        out.append(
            SubTaskSpec(
                role_name=role_name,
                role_instruction=role_instruction,
                user_prompt=_RUNTIME_USER_PROMPT_FALLBACK,
                tool_context=_sanitize_tool_context({"execution_mode": "llm_only"}, dataset_name=dataset_name),
            )
        )
    return out


def _safe_identifier(raw: str, *, prefix: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "")).strip("_")
    if not s:
        s = prefix
    if s[0].isdigit():
        s = f"{prefix}_{s}"
    return s


def _default_interactive_sub_agents(dataset_name: str) -> list[SubTaskSpec]:
    """Safe, compact fallback MAS for stateful embodied benchmarks.

    The base planner is still attempted first.  If its structured output is
    malformed or times out, this fallback keeps the sample evaluable instead of
    reporting a generation failure.  It contains one concise strategy node and
    exactly one persistent live-environment executor, so it does not multiply
    irreversible environment actions.
    """
    dataset = _normalize_dataset_name(dataset_name)
    family_hints = (
        "ALFWorld procedure hints: locate the exact goal object; for clean use "
        "sinkbasin/faucet before placement; for heat use microwave and required "
        "open/close/heat operations; for cool use fridge and required open/close/cool "
        "operations; for two-object tasks finish both objects before stopping; for "
        "look-at tasks arrange illumination and inspect the requested object."
        if dataset == "alfworld" else
        "ScienceWorld procedure hints: follow the public task clauses in order; treat the simulator's valid list "
        "as syntactic possibilities rather than evidence that every action is useful; identify generic requested "
        "categories from observations before focus; preserve the exact referent after a score-gaining focus and "
        "complete any remaining move/place clause; for experiments, locate relevant apparatus/materials, set up and "
        "operate the experiment, read live observations or measurements, and only then make conditional focus/box "
        "choices. Never claim conductivity, friction, genetics, temperature, or another experimental result from "
        "llm_only reasoning without prior live simulator evidence."
    )
    strategist = SubTaskSpec(
        role_name="embodied_task_strategist",
        role_instruction=(
            "Analyze the exact benchmark task, initial observation, inventory, and any "
            "upstream state. Produce a concise executable plan for the live executor. "
            "Do not invent commands, do not claim success, and do not reset the environment. "
            + family_hints
        ),
        user_prompt=_RUNTIME_USER_PROMPT_FALLBACK,
        tool_context={"execution_mode": "llm_only"},
    )
    executor = SubTaskSpec(
        role_name="persistent_interactive_executor",
        role_instruction=(
            "Complete the entire benchmark task in the already-running environment. "
            "Choose only exact candidate valid actions, inspect every new observation, "
            "replan after each step, avoid no-progress cycles and distractors, and continue "
            "until environment success or the action budget is exhausted. " + family_hints
        ),
        user_prompt=_RUNTIME_USER_PROMPT_FALLBACK,
        tool_context={"execution_mode": "interactive_env"},
    )
    return [strategist, executor]


def _is_mas_code_compilable(mas_code: str) -> tuple[bool, str]:
    try:
        ast.parse(mas_code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"{type(e).__name__}: {e}"


def _build_deterministic_sequential_workflow_code(
    *,
    class_name: str,
    subs: list[SubTaskSpec],
) -> str:
    """
    Emit a guaranteed-compilable sequential workflow from normalized sub-agents.
    This avoids hard failures when free-form Stage 3 code generation is unstable.
    """
    if not subs:
        raise RuntimeError("cannot build deterministic workflow without sub-agents")

    lines: list[str] = []
    lines.append(f"class {class_name}:")
    lines.append("    def __init__(")
    lines.append("        self,")
    lines.append("        *,")
    lines.append('        text_call_fn: Callable[[str], tuple[str, dict[str, Any]]],')
    lines.append('        tool_call_fn: Callable[[SubAgentRequest], tuple[str, dict[str, Any]]] | None = None,')
    lines.append("    ):")
    lines.append("        self.text_call_fn = text_call_fn")
    lines.append("        self.tool_call_fn = tool_call_fn")
    lines.append("")

    role_to_attr: list[tuple[str, str]] = []
    for i, spec in enumerate(subs, start=1):
        attr = _safe_identifier(spec.role_name, prefix=f"agent_{i}")
        role_key = _safe_identifier(spec.role_name, prefix=f"node_{i}").lower()
        role_to_attr.append((role_key, attr))
        lines.append(f"        self.{attr} = SubAgentRequest(")
        lines.append(f"            role_name={spec.role_name!r},")
        lines.append(f"            role_instruction={spec.role_instruction!r},")
        lines.append(f"            user_prompt={spec.user_prompt!r},")
        lines.append(f"            tool_context={dict(spec.tool_context or {})!r},")
        lines.append("        )")
        lines.append("")

    lines.append("    async def forward_async(self, task_info: str) -> dict[str, Any]:")
    lines.append('        state: dict[str, Any] = {"task_info": task_info}')
    lines.append("")

    prev_role_key: str | None = None
    for idx, (role_key, attr) in enumerate(role_to_attr, start=1):
        out_key = f"out_{role_key}"
        usage_key = f"usage_{role_key}"
        lines.append(f"        out_{idx} = await self.{attr}.execute_async(")
        lines.append("            text_call_fn=self.text_call_fn,")
        lines.append("            tool_call_fn=self.tool_call_fn,")
        lines.append("            prompt_override=build_workflow_subagent_prompt(")
        lines.append(f"                role_instruction=self.{attr}.role_instruction,")
        lines.append(f"                base_user_prompt=self.{attr}.user_prompt,")
        lines.append('                task_info=state["task_info"],')
        if prev_role_key is None:
            lines.append("                upstream_outputs=None,")
        else:
            lines.append("                upstream_outputs={")
            lines.append(f'                    "{prev_role_key}_output": state["out_{prev_role_key}"],')
            lines.append("                },")
        lines.append("            ),")
        lines.append("        )")
        lines.append(f'        state["{out_key}"] = out_{idx}.text')
        lines.append(f'        state["{usage_key}"] = out_{idx}.usage')
        lines.append("")
        prev_role_key = role_key

    assert prev_role_key is not None
    lines.append(f'        state["final_output"] = state["out_{prev_role_key}"]')
    lines.append(f'        state["usage_final"] = state.get("usage_{prev_role_key}", {{}})')
    lines.append("        return state")
    return "\n".join(lines)


def _extract_user_prompt_from_stage2_raw(raw_text: str, role_name: str) -> str:
    if not raw_text.strip() or not role_name.strip():
        return ""
    role_re = re.escape(role_name)
    m = re.search(
        rf'"role_name"\s*:\s*"{role_re}".*?"user_prompt"\s*:\s*"((?:\\.|[^"\\])*)"',
        raw_text,
        flags=re.S,
    )
    if not m:
        return ""
    encoded = m.group(1)
    try:
        return json.loads(f'"{encoded}"')
    except Exception:
        return encoded.replace('\\"', '"').replace("\\n", "\n")


def _with_hard_instruction_constraints(
    role_instruction: str,
    *,
    is_final: bool,
    dataset_name: str,
    role_name: str = "",
) -> str:
    base = (role_instruction or "").strip()
    if is_final:
        # The final agent is responsible for producing the complete answer.
        # It should NOT be constrained to "keep output concise" or "only do this sub-task"
        # because its sub-task IS the final deliverable.
        rules = [
            "HARD CONSTRAINTS:",
            # "- You are the FINAL agent in this workflow. Your output IS the final answer delivered to the user.",
            # "- Produce a COMPLETE, fully-formed answer — do not output summaries, meta-commentary, or placeholders.",
            # "- Ensure your output strictly matches the format required by the task (e.g., a single answer string, or \\boxed{...} for math).",
            # "- If upstream information is incomplete or uncertain, still produce your best answer based on all available information.",
            # "- Do NOT output null, empty, or 'not found' messages — always provide a substantive answer.",
        ]
        if dataset_name == "bcp":
            rules.extend(
                [
                    "- For retrieval-grounded tasks, avoid deterministic refusal sentinel strings (e.g., NO_VALID_EVIDENCE_FOUND / UNABLE_TO_DETERMINE) unless the task explicitly requires those exact tokens.",
                    "- If evidence is partial, provide the best evidence-supported candidate answer rather than terminating with a failure token.",
                ]
            )
        if dataset_name == "gpqa":
            rules.extend(
                [
                    "- You are the FINAL GPQA option mapper, not another solver. Read the evidence gate's `Verified option: X` and preserve X.",
                    "- Check only that X is one of A/B/C/D and maps to the same original option label; do not recompute the science, introduce a new preference, or override the gate.",
                    r"- Output exactly one line: \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}, using the verified uppercase letter.",
                    "- Do not output reasoning, option text, a numeric value, JSON, a refusal, or any text after the boxed option.",
                ]
            )
        if dataset_name == "hlemath":
            rules.extend(
                [
                    "- You are the FINAL HLEMATH formatter and last consistency gate, not a fresh solver.",
                    "- Read the conservative repairer's `Repaired integer: N`. Preserve N unless a direct range/format inconsistency or an explicit arithmetic contradiction is visible in the supplied audit evidence.",
                    "- Do not restart the problem, invent a new derivation, or change N merely because an alternative feels plausible; answer changes require a concrete cited contradiction.",
                    "- Output plain text only (no JSON/YAML/dict wrappers or markdown code fences).",
                    "- Give at most two short consistency-check sentences, then the boxed result.",
                    "- The last line must be exactly one \\boxed{...} answer.",
                    r"- The final answer must be a single integer in the AIME range 0..999; do not place a non-integer or sentinel in \boxed{...}.",
                ]
            )
    else:
        # Intermediate agents should be concise and focused on their specific sub-task.
        rules = [
            "HARD CONSTRAINTS:",
            # "- You must complete ONLY this sub-task; do not solve downstream sub-tasks.",
            # "- Keep output concise and directly usable by the next sub-agent.",
            # "- Do not add unrelated analysis, checklists, or meta commentary.",
            # "- Always produce substantive, structured output — never output null, empty, or 'not found' messages.",
        ]
        if dataset_name == "gpqa":
            role_blob = f"{role_name} {base}".lower()
            if re.search(r"verif|reconcil|critic|audit|evidence_gate", role_blob):
                rules.extend([
                    "- Act as a conservative GPQA evidence gate. Extract both `Candidate option` letters and their decisive claims.",
                    "- If the roots agree, preserve their letter unless you can state a concrete calculation, definition, unit/sign, limiting-case, wording, or mechanism contradiction.",
                    "- If the roots disagree, test the competing decisive claims. Reject one only with explicit scientific evidence; never use confidence, verbosity, or majority vote.",
                    "- Check original A/B/C/D mapping. End with exactly `Verified option: X` where X is A, B, C, or D.",
                ])
            else:
                rules.extend([
                    "- Solve the complete GPQA question independently from GLOBAL TASK CONTEXT; do not copy or continue an upstream proposed letter.",
                    "- Evaluate all four original options A-D and preserve their exact mapping. State one decisive, testable scientific claim.",
                    "- Actively test your leading option for a counterexample, unit/sign error, definition mismatch, limiting case, or wording trap.",
                    "- End with `Candidate option: X; decisive claim: ...` where X is A, B, C, or D.",
                ])
            rules.append("- Output plain text only; do not use JSON/YAML or markdown code fences.")
        if dataset_name == "hlemath":
            role_blob = f"{role_name} {base}".lower()
            rules.extend([
                "- You are an INTERMEDIATE HLEMATH agent. Do NOT output the final \\boxed{...} answer.",
                "- Use plain text only (no JSON/dict/list schemas or markdown code fences).",
                "- Do not introduce assumptions not grounded in GLOBAL TASK CONTEXT; mark unavoidable ones with ASSUMPTION: and justify them.",
            ])
            if "primary" in role_blob or ("solver" in role_blob and not re.search(r"audit|repair|verif|reconcil", role_blob)):
                rules.extend([
                    "- Solve the complete problem carefully from GLOBAL TASK CONTEXT with explicit equations, case splits, domain/boundary checks, and arithmetic verification.",
                    "- End with exactly `Proposed integer: N`, where N is your AIME-range candidate.",
                ])
            elif re.search(r"audit|check|verif", role_blob):
                rules.extend([
                    "- Audit the upstream primary derivation line by line. Locate the first fatal error, if any, and verify arithmetic, signs, domains, boundaries, case completeness, duplicates, and extraneous roots.",
                    "- Do not replace the proposal based on intuition or a merely different method. A change requires a specific invalid equation, omitted case, counterexample, or arithmetic contradiction.",
                    "- End with either `Audit: VERIFIED; integer N` or `Audit: ERROR; corrected integer N; reason <specific fatal step>`.",
                ])
            else:
                rules.extend([
                    "- Act as a conservative repairer. If the audit says VERIFIED, preserve its integer. If it says ERROR, independently validate the cited fatal step before accepting the correction.",
                    "- Do not restart an unrelated full solution or change the number without a concrete contradiction.",
                    "- End with exactly `Repaired integer: N; evidence: ...`.",
                ])
    return f"{base}\n\n" + "\n".join(rules)


def _choose_final_sub_agent_index_for_dataset(
    subs: list[SubTaskSpec],
    *,
    dataset_name: str,
) -> int:
    """
    Choose which sub-agent should receive FINAL-agent constraints.

    Important:
    - Keep default behavior (last sub-agent) for all non-Vita datasets.
    - For VitaBench only, prefer explicit aggregation/synthesis roles when present.
      This prevents accidentally forcing a branch executor (e.g. book/order node)
      to become the final answer node.
    """
    if not subs:
        return -1
    if dataset_name == "hlemath":
        for i, spec in enumerate(subs):
            rn = (spec.role_name or "").strip().lower()
            if any(
                tok in rn
                for tok in (
                    "final",
                    "format",
                    "boxed",
                    "answer",
                    "finalize",
                )
            ):
                return i
        return len(subs) - 1
    if dataset_name != "vita":
        return len(subs) - 1

    # VitaBench-specific heuristic: prefer explicit "final aggregation" style roles.
    role_name_priority_tokens = (
        "final",
        "report",
        "summary",
        "synth",
        "aggregate",
        "merge",
        "compilation",
        "finalize",
    )
    for i, spec in enumerate(subs):
        rn = (spec.role_name or "").strip().lower()
        if any(tok in rn for tok in role_name_priority_tokens):
            return i

    # Fallback: preserve previous behavior.
    return len(subs) - 1


async def _call_stage_json(
    llm_call_fn: Callable[[str], Awaitable[str]],
    prompt: str,
    *,
    stage_name: str,
) -> dict[str, Any]:
    obj, _raw, _elapsed = await _call_stage_json_with_meta(
        llm_call_fn,
        prompt,
        stage_name=stage_name,
    )
    return obj


async def _call_stage_json_with_meta(
    llm_call_fn: Callable[[str], Awaitable[str]],
    prompt: str,
    *,
    stage_name: str,
) -> tuple[dict[str, Any], str, float]:
    def _is_structurally_valid_for_stage(obj: dict[str, Any], hint: int | None) -> bool:
        if hint == 1:
            st = _coerce_stage1_subtasks(obj).get("sub_tasks")
            return isinstance(st, list) and len(st) > 0
        if hint == 2:
            return len(_normalize_sub_tasks(obj)) > 0
        if hint == 3:
            mc = obj.get("mas_code")
            return isinstance(mc, str) and bool(mc.strip())
        return True

    def _build_json_repair_prompt(
        *,
        stage_label: str,
        original_prompt: str,
        malformed_response: str,
    ) -> str:
        max_chars = 8000
        clipped = (malformed_response or "")[:max_chars]
        if len(malformed_response or "") > max_chars:
            clipped += "\n...[truncated]"
        return (
            "You are a strict JSON repair assistant for a MAS build stage.\n"
            "Rewrite the malformed assistant response into ONE valid JSON object only.\n"
            "Requirements:\n"
            "- Follow [OUTPUT_JSON_SCHEMA] from ORIGINAL_STAGE_PROMPT exactly.\n"
            "- Do not add markdown fences or explanations.\n"
            "- Keep original intent/content; only repair JSON structure/escaping.\n"
            "- Return a single top-level JSON object.\n\n"
            f"[STAGE]\n{stage_label}\n\n"
            f"[ORIGINAL_STAGE_PROMPT]\n{original_prompt}\n\n"
            f"[MALFORMED_ASSISTANT_RESPONSE]\n{clipped}\n\n"
            "Output JSON now."
        )

    print(f"\n[{stage_name}] prompt_chars={len(prompt)}", flush=True)
    t0 = time.perf_counter()
    raw = await llm_call_fn(prompt)
    elapsed = time.perf_counter() - t0
    print(f"[{stage_name}] llm_elapsed_sec={elapsed:.2f}", flush=True)
    preview = (raw or "").strip().replace("\n", " ")
    if len(preview) > 300:
        preview = preview[:300] + "...(truncated)"
    print(f"[{stage_name}] raw_response_preview={preview}", flush=True)
    hint = _stage_hint_from_label(stage_name)
    obj = _extract_best_json_dict_for_stage(raw, stage_hint=hint)
    if isinstance(obj, dict) and _is_structurally_valid_for_stage(obj, hint):
        print(f"[{stage_name}] parsed_keys={sorted(obj.keys())}", flush=True)
        return obj, raw, elapsed

    # One repair attempt for malformed / structurally invalid stage JSON.
    repair_prompt = _build_json_repair_prompt(
        stage_label=stage_name,
        original_prompt=prompt,
        malformed_response=raw,
    )
    print(f"[{stage_name}] attempting json_repair_retry=1", flush=True)
    t1 = time.perf_counter()
    repaired_raw = await llm_call_fn(repair_prompt)
    elapsed += time.perf_counter() - t1
    repaired_obj = _extract_best_json_dict_for_stage(repaired_raw, stage_hint=hint)
    if not isinstance(repaired_obj, dict):
        raise RuntimeError(f"{stage_name} returned no valid JSON object.")
    if not _is_structurally_valid_for_stage(repaired_obj, hint):
        raise RuntimeError(f"{stage_name} returned structurally invalid JSON payload.")
    print(f"[{stage_name}] parsed_keys={sorted(repaired_obj.keys())}", flush=True)
    print(f"[{stage_name}] json_repair_retry=1 succeeded", flush=True)
    return repaired_obj, repaired_raw, elapsed


async def run_three_stage_skill_build(
    *,
    task_text: str,
    init_skill_text: str,
    template_text: str,
    llm_call_fn: Callable[[str], Awaitable[str]],
    class_name: str,
    dataset_name: str = "unknown",
) -> str:
    """
    Execute exactly 3 stages from init_skill and return final MAS code.
    """
    artifacts = await run_three_stage_skill_build_with_trace(
        task_text=task_text,
        init_skill_text=init_skill_text,
        template_text=template_text,
        llm_call_fn=llm_call_fn,
        class_name=class_name,
        dataset_name=dataset_name,
    )
    return artifacts.mas_code


async def run_three_stage_skill_build_with_trace(
    *,
    task_text: str,
    init_skill_text: str,
    template_text: str,
    llm_call_fn: Callable[[str], Awaitable[str]],
    class_name: str,
    dataset_name: str = "unknown",
) -> ThreeStageBuildArtifacts:
    """Execute three-stage build and return full stage traces and mas_code."""
    stages = _extract_skill_stages(init_skill_text)
    stage_traces: list[BuildStageTrace] = []

    normalized_dataset = _normalize_dataset_name(dataset_name)
    dataset_final_requirements = _dataset_final_requirement_text(normalized_dataset)

    s1_schema = (
        '{ "goal": "...", "sub_tasks": [{"name":"...", "description":"..."}], '
        '"constraints": ["..."], "dependencies": [{"from":"...","to":"...","type":"..."}], '
        '"success_criteria": ["..."], '
        '"reasoning": "detailed rationale: how you interpreted the task, why this decomposition/dependencies, and how ambiguity/assumption risks are controlled. '
        'IMPORTANT: sub_tasks descriptions must describe the CAPABILITY/PROCESS each node provides, NOT attempt '
        'to answer the user question or provide solution content." }'
    )
    s1_constraints: list[str] = []
    s1_extra_input: dict[str, Any] = {}
    if normalized_dataset == "hlemath":
        s1_constraints.extend(_hlemath_stage1_build_constraints())
        s1_extra_input["hlemath_topology_templates"] = _hlemath_topology_templates_block()
    elif normalized_dataset == "gpqa":
        s1_constraints.extend(_gpqa_stage1_build_constraints())
    elif normalized_dataset == "bcp":
        s1_constraints.extend(_bcp_stage1_build_constraints())
    s1_prompt = _stage_prompt(
        stages[1],
        {
            "task_text": task_text,
            "dataset": normalized_dataset,
            "dataset_final_requirements": dataset_final_requirements,
            "constraints": s1_constraints,
            **s1_extra_input,
        },
        s1_schema,
        stage_index=1,
    )
    try:
        s1, s1_raw, s1_elapsed = await _call_stage_json_with_meta(
            llm_call_fn, s1_prompt, stage_name="Stage 1"
        )
        s1 = _coerce_stage1_subtasks(s1)
    except Exception as exc:
        if normalized_dataset not in {"alfworld", "scienceworld"}:
            raise
        # Embodied evaluation must not discard an episode because a small local
        # planner emitted malformed JSON. Keep the model's error visible while
        # supplying a minimal stage-local decomposition for Stage 2.
        message = f"{type(exc).__name__}: {exc}"
        s1_raw = message
        s1_elapsed = 0.0
        s1 = {
            "goal": task_text,
            "sub_tasks": [
                {"name": "live_task_strategy", "description": "Interpret the authoritative live task and plan progress-aware admissible actions."},
                {"name": "live_environment_executor", "description": "Execute exact simulator-admissible actions until the benchmark goal is complete."},
                {"name": "progress_verifier", "description": "Check observations and score, detect loops, and direct recovery when needed."},
            ],
            "constraints": ["Never reset the episode", "Use exact admissible actions only"],
            "_structural_fallback": {"stage": 1, "error": message},
        }
        print(f"[Stage 1][interactive] structural fallback used: {message}", flush=True)
    if normalized_dataset == "hlemath":
        s1_ok, s1_issues = _hlemath_validate_stage1_decomposition(s1)
        if not s1_ok:
            print(
                f"[Stage 1][HLEMATH] decomposition issues ({len(s1_issues)}); attempting one structural repair",
                flush=True,
            )
            for issue in s1_issues:
                print(f"[Stage 1][HLEMATH]   - {issue}", flush=True)
            repair_prompt = _build_dataset_stage1_repair_prompt(
                dataset="hlemath", original_prompt=s1_prompt, stage1_json=s1, issues=s1_issues
            )
            t_repair = time.perf_counter()
            repaired_raw = await llm_call_fn(repair_prompt)
            s1_elapsed += time.perf_counter() - t_repair
            repaired_obj = _extract_best_json_dict_for_stage(repaired_raw, stage_hint=1)
            if isinstance(repaired_obj, dict):
                repaired_obj = _coerce_stage1_subtasks(repaired_obj)
                repaired_ok, repaired_issues = _hlemath_validate_stage1_decomposition(repaired_obj)
                if repaired_ok:
                    s1 = repaired_obj
                    s1_raw = repaired_raw
                    print("[Stage 1][HLEMATH] structural repair succeeded", flush=True)
                else:
                    print(
                        "[Stage 1][HLEMATH] structural repair still has issues; using canonical fallback",
                        flush=True,
                    )
                    for issue in repaired_issues:
                        print(f"[Stage 1][HLEMATH]   - {issue}", flush=True)
                    s1 = _canonical_dataset_stage1_decomposition(dataset="hlemath", task_text=task_text)
                    s1_raw = repaired_raw
            else:
                print("[Stage 1][HLEMATH] structural repair returned invalid JSON; using canonical fallback", flush=True)
                s1 = _canonical_dataset_stage1_decomposition(dataset="hlemath", task_text=task_text)
                s1_raw = repaired_raw
    if normalized_dataset == "gpqa":
        s1_ok, s1_issues = _gpqa_validate_stage1_decomposition(s1)
        if not s1_ok:
            print(f"[Stage 1][GPQA] decomposition issues ({len(s1_issues)}); attempting one structural repair", flush=True)
            repair_prompt = _build_dataset_stage1_repair_prompt(
                dataset="gpqa", original_prompt=s1_prompt, stage1_json=s1, issues=s1_issues
            )
            t_repair = time.perf_counter()
            repaired_raw = await llm_call_fn(repair_prompt)
            s1_elapsed += time.perf_counter() - t_repair
            repaired_obj = _extract_best_json_dict_for_stage(repaired_raw, stage_hint=1)
            if isinstance(repaired_obj, dict):
                repaired_obj = _coerce_stage1_subtasks(repaired_obj)
                repaired_ok, repaired_issues = _gpqa_validate_stage1_decomposition(repaired_obj)
                if repaired_ok:
                    s1, s1_raw = repaired_obj, repaired_raw
                    print("[Stage 1][GPQA] structural repair succeeded", flush=True)
                else:
                    print(f"[Stage 1][GPQA] repair still invalid: {repaired_issues}; using canonical fallback", flush=True)
                    s1 = _canonical_dataset_stage1_decomposition(dataset="gpqa", task_text=task_text)
                    s1_raw = repaired_raw
            else:
                print("[Stage 1][GPQA] structural repair returned invalid JSON; using canonical fallback", flush=True)
                s1 = _canonical_dataset_stage1_decomposition(dataset="gpqa", task_text=task_text)
                s1_raw = repaired_raw
    if normalized_dataset in {"hlemath", "gpqa"}:
        # Evaluation evidence showed that free-form Stage-1 role naming/topology
        # reintroduced serial chains and role proliferation despite dataset
        # constraints. Keep the learned model responsible for solving, but make
        # the benchmark-specific orchestration itself deterministic and auditable.
        s1 = _canonical_dataset_stage1_decomposition(
            dataset=normalized_dataset,
            task_text=task_text,
        )
        s1["_structural_fallback"] = {
            "stage": 1,
            "reason": "deterministic dataset-specific orchestration policy",
        }
        print(f"[Stage 1][{normalized_dataset.upper()}] enforcing canonical dataset topology", flush=True)

    stage_traces.append(
        BuildStageTrace(
            stage=1,
            stage_name="Stage 1",
            prompt=s1_prompt,
            raw_response=s1_raw,
            parsed_json=s1,
            elapsed_sec=s1_elapsed,
        )
    )
    s1_subs = s1.get("sub_tasks")
    s1_sub_count = len(s1_subs) if isinstance(s1_subs, list) else 0
    print(f"[Stage 1] sub_tasks_count={s1_sub_count}", flush=True)

    allowed_modes = sorted(_allowed_tool_modes_for_dataset(normalized_dataset))
    s2_constraints = [
        f"Only choose execution_mode from {allowed_modes} for this dataset.",
        "Set tool_context.execution_mode explicitly for every sub-agent.",
    ]
    if normalized_dataset == "vita":
        s2_constraints.append(
            'For VitaBench, every sub-agent must use {"execution_mode":"vita_tool"} only. '
            "Lookup-only or reasoning-only steps still run in the Vita simulator with environment tools; never use llm_only."
        )
        s2_constraints.append(
            "For VitaBench multi-chain plans, explicitly include a final aggregation/synthesis role "
            "(e.g. final_report/final_summary/final_aggregator) that combines all required transaction outcomes. "
            "Do not end the workflow on a single branch action node (like only book/order/verify for one branch)."
        )
    elif normalized_dataset == "bcp":
        s2_constraints.extend(_bcp_stage2_routing_constraints())
    elif normalized_dataset == "hlemath":
        s2_constraints.append(_hlemath_stage2_build_constraints())
    elif normalized_dataset == "gpqa":
        s2_constraints.append(_gpqa_stage2_build_constraints())
    elif normalized_dataset in {"alfworld", "scienceworld"}:
        s2_constraints.extend(
            [
                "For interactive embodied benchmarks, route roles by capability instead of making every node act.",
                'Use {"execution_mode":"llm_only"} for task analysis, planning, state interpretation, critique, and final reporting.',
                'Use {"execution_mode":"interactive_env"} only for action-executor or recovery-executor roles that must advance the shared live episode.',
                "Include at least one interactive_env executor. Prefer a compact plan -> execute -> critique/recover topology rather than one executor for every semantic subgoal.",
                "The environment is already initialized with the exact live task and initial observation. Never create initialization/termination pseudo-actions and never reset it.",
                "Executor instructions must require exact valid commands, observation-based replanning, and no repetition of commands that made no progress.",
            ]
        )
        if normalized_dataset == "scienceworld":
            s2_constraints.extend(
                [
                    "ScienceWorld-specific: valid actions are syntactic possibilities, not a recommendation to manipulate every listed object. Follow task clauses in order and avoid unrelated connect/mix/pour/move actions.",
                    "ScienceWorld-specific: llm_only roles may design an experiment but must not decide its outcome before upstream interactive_env output contains live evidence; the executor must perform setup -> operate/wait -> observe/read -> final focus/place as required.",
                    "ScienceWorld-specific: for generic ordered referents, identify an observation-supported member of the requested category before focus; after score increases on focus, preserve that object's identity through the remaining transport clause.",
                ]
            )
    else:
        s2_constraints.append(
            'For sub-tasks that do not use tools, set tool_context to {"execution_mode":"llm_only"}; do not use null for tool_context.'
        )
    s2_constraints.append(
        "The dataset_final_requirements below defines what the benchmark evaluator expects as the final deliverable. "
        "Ensure the LAST agent (the one that produces final_output) generates output in exactly that format. "
        "Intermediate agents should produce structured outputs that directly support the final agent in meeting this requirement."
    )
    s2_schema = (
        '{ "sub_agents": ['
        '{"role_name":"...", "role_instruction":"...", "user_prompt":"...", '
        '"tool_context": {"execution_mode":"llm_only|multi_turn_search|vita_tool|interactive_env", "...":"..."} }'
        '], "reasoning": "brief rationale: role split, ordering, and execution_mode choices vs Stage 1. '
        'IMPORTANT: role_instruction defines the agent\'s role and behavioral boundaries; user_prompt provides '
        'the concrete task input. Neither should contain solution content or attempt to answer the question." }'
    )
    s2_prompt = _stage_prompt(
        stages[2],
        {
            "task_text": task_text,
            "stage1": s1,
            "dataset": normalized_dataset,
            "dataset_final_requirements": dataset_final_requirements,
            "tool_context_execution_modes": list(TOOL_EXECUTION_MODES),
            "allowed_modes_for_dataset": allowed_modes,
            "constraints": s2_constraints,
        },
        s2_schema,
        stage_index=2,
    )
    try:
        s2, s2_raw, s2_elapsed = await _call_stage_json_with_meta(
            llm_call_fn, s2_prompt, stage_name="Stage 2"
        )
    except Exception as exc:
        if normalized_dataset not in {"alfworld", "scienceworld"}:
            raise
        message = f"{type(exc).__name__}: {exc}"
        fallback_specs = _fallback_sub_agents_from_stage1(s1, dataset_name=normalized_dataset)
        if not fallback_specs:
            fallback_specs = _default_interactive_sub_agents(normalized_dataset)
        s2 = {
            "sub_agents": [spec.__dict__ for spec in fallback_specs],
            "reasoning": "stage-local structural fallback derived from Stage 1",
            "_structural_fallback": {"stage": 2, "error": message},
        }
        s2_raw = message
        s2_elapsed = 0.0
        print(f"[Stage 2][interactive] structural fallback used: {message}", flush=True)
    stage_traces.append(
        BuildStageTrace(
            stage=2,
            stage_name="Stage 2",
            prompt=s2_prompt,
            raw_response=s2_raw,
            parsed_json=s2,
            elapsed_sec=s2_elapsed,
        )
    )
    subs = _normalize_sub_tasks(s2)
    if normalized_dataset in {"hlemath", "gpqa"}:
        # Stage-2 generation frequently expanded a four-node Stage-1 plan to
        # five-to-seven agents. Execute the canonical roles exactly so prompt
        # semantics and graph topology remain stable across samples.
        subs = _fallback_sub_agents_from_stage1(s1, dataset_name=normalized_dataset)
        print(f"[Stage 2][{normalized_dataset.upper()}] enforcing four canonical runtime roles", flush=True)
    elif not subs:
        print("[Stage 2] fallback: deriving sub_agents from Stage 1 decomposition", flush=True)
        subs = _fallback_sub_agents_from_stage1(s1, dataset_name=normalized_dataset)
    if not subs:
        raise RuntimeError("Stage 2 returned invalid/empty sub_agents.")
    final_sub_agent_idx = _choose_final_sub_agent_index_for_dataset(
        subs,
        dataset_name=normalized_dataset,
    )
    if normalized_dataset == "hlemath":
        s2_ok, s2_issues = _hlemath_validate_stage2_sub_agents(
            subs,
            final_idx=final_sub_agent_idx,
        )
        if not s2_ok:
            print(
                f"[Stage 2][HLEMATH] agent-protocol warnings ({len(s2_issues)}); applying hard constraints",
                flush=True,
            )
            for issue in s2_issues:
                print(f"[Stage 2][HLEMATH]   - {issue}", flush=True)
    constrained_subs: list[SubTaskSpec] = []
    for idx, spec in enumerate(subs):
        raw_user_prompt = spec.user_prompt or _extract_user_prompt_from_stage2_raw(s2_raw, spec.role_name)
        # Keep user_prompt short and embed-safe: full task is always in GLOBAL TASK CONTEXT at runtime
        # (see template.sub_agent.build_workflow_subagent_prompt). Never substitute raw task_text here —
        # pasting LaTeX / JSON / quotes into SubAgentRequest literals breaks Stage-3 Python codegen.
        if not raw_user_prompt or len(raw_user_prompt.strip()) < 10:
            raw_user_prompt = _RUNTIME_USER_PROMPT_FALLBACK
        is_final_agent = (idx == final_sub_agent_idx)
        constrained_subs.append(
            SubTaskSpec(
                role_name=spec.role_name,
                role_instruction=_with_hard_instruction_constraints(
                    spec.role_instruction,
                    is_final=is_final_agent,
                    dataset_name=normalized_dataset,
                    role_name=spec.role_name,
                ),
                user_prompt=raw_user_prompt,
                tool_context=_sanitize_tool_context(spec.tool_context, dataset_name=normalized_dataset),
            )
        )
    # Embodied episodes have one shared, sequential simulator.  Allowing every
    # semantic subgoal to call it spent most of the 50-step budget before global
    # recovery could act.  Route exactly one best-suited role to the live
    # environment; all other roles remain reasoning-only.  This transformation is
    # local to each project and leaves every non-embodied dataset untouched.
    if normalized_dataset in {"alfworld", "scienceworld"}:
        def _executor_score(item: tuple[int, SubTaskSpec]) -> tuple[int, int]:
            idx, candidate = item
            role_text = f"{candidate.role_name} {candidate.role_instruction}".lower()
            score = 0
            for word, weight in (
                ("executor", 120), ("recover", 115), ("interact", 110),
                ("manipulate", 95), ("experiment", 90), ("action", 85),
                ("place", 75), ("pick", 72), ("clean", 70), ("heat", 70),
                ("cool", 70), ("navigate", 60), ("move", 55), ("execute", 100),
            ):
                if word in role_text:
                    score += weight
            for word in ("planner", "planning", "analy", "critic", "report", "locate", "identify"):
                if word in role_text:
                    score -= 45
            if (candidate.tool_context or {}).get("execution_mode") == "interactive_env":
                score += 20
            return score, -idx

        executor_idx = max(enumerate(constrained_subs), key=_executor_score)[0]
        rerouted: list[SubTaskSpec] = []
        for idx, spec in enumerate(constrained_subs):
            is_executor = idx == executor_idx
            instruction = spec.role_instruction
            if is_executor:
                instruction += (
                    "\n\nSOLE LIVE-ENVIRONMENT EXECUTOR: Ignore any narrower wording that conflicts "
                    "with this requirement. Use exact valid actions to complete the entire benchmark task "
                    "end-to-end, using the persistent live state and not stopping after only your named subgoal."
                    + (
                        " For ScienceWorld, follow clauses in order, treat valid actions as syntactic only, obtain live "
                        "experimental evidence before conditional focus/placement, and preserve any score-gaining "
                        "focused referent through its remaining move/place clause."
                        if normalized_dataset == "scienceworld" else ""
                    )
                )
            rerouted.append(
                SubTaskSpec(
                    role_name=spec.role_name,
                    role_instruction=instruction,
                    user_prompt=spec.user_prompt,
                    tool_context={
                        **(spec.tool_context or {}),
                        "execution_mode": "interactive_env" if is_executor else "llm_only",
                    },
                )
            )
        # Keep the generated design compact.  More than one reasoning node
        # adds repeated long LLM calls but cannot safely parallelize access to a
        # single stateful simulator.  Preserve one model-designed strategist and
        # the best executor, with the executor last so final_output reflects the
        # actual environment work.
        executor = rerouted[executor_idx]
        planners = [spec for idx, spec in enumerate(rerouted) if idx != executor_idx]
        if planners:
            def _planner_score(spec: SubTaskSpec) -> int:
                text = f"{spec.role_name} {spec.role_instruction}".lower()
                return sum(
                    weight for token, weight in (
                        ("planner", 100), ("strategy", 95), ("analy", 90),
                        ("critic", 70), ("interpret", 65), ("plan", 80),
                    ) if token in text
                )
            strategist = max(planners, key=_planner_score)
        else:
            strategist = _default_interactive_sub_agents(normalized_dataset)[0]
        # The selected executor may have been the original final node and thus
        # carry generic final-format instructions.  Its hard live-executor
        # requirement above takes precedence.
        constrained_subs = [strategist, executor]
    subs = constrained_subs
    print(f"[Stage 2] sub_agents_count={len(subs)} roles={[s.role_name for s in subs]}", flush=True)
    preferred_final_role_name = None
    if 0 <= final_sub_agent_idx < len(subs):
        preferred_final_role_name = subs[final_sub_agent_idx].role_name

    # Stage 3 asks the model to reproduce a large Python template and was the
    # dominant source of context overflow, malformed JSON, and multi-minute
    # requests.  For stateful embodied tasks the simulator is sequential anyway,
    # so compile the model-designed Stage-1/2 roles deterministically.  This keeps
    # the orchestra-built role design while removing unreliable free-form codegen.
    if normalized_dataset in {"alfworld", "scienceworld"}:
        mas_code = prepend_fixed_imports(
            _build_deterministic_sequential_workflow_code(class_name=class_name, subs=subs)
        )
        s3 = {
            "mas_code": mas_code,
            "reasoning": "deterministic embodied workflow compiler; Stage 3 LLM skipped",
        }
        stage_traces.append(
            BuildStageTrace(
                stage=3,
                stage_name="Stage 3 (deterministic embodied compiler)",
                prompt="",
                raw_response="",
                parsed_json=s3,
                elapsed_sec=0.0,
            )
        )
        print(
            f"[Stage 3][interactive] skipped LLM codegen; deterministic mas_code_chars={len(mas_code)}",
            flush=True,
        )
        return ThreeStageBuildArtifacts(
            mas_code=mas_code,
            stage_traces=stage_traces,
            stage1=s1,
            stage2=s2,
            stage3=s3,
            normalized_sub_agents=[spec.__dict__ for spec in subs],
        )

    s3_schema = (
        '{ "mas_code": "full python code string", '
        '"reasoning": "brief rationale: topology, dataflow, and how you kept SubAgentRequest string literals '
        "PYTHON-SAFE (delimiters/escaping, no duplicate imports) before emitting mas_code\" }"
    )
    normalized_sub_agents = [s.__dict__ for s in subs]
    s3_input = {
        "task_text": task_text,
        "stage1": s1,
        "stage2": {"sub_agents": normalized_sub_agents},
        "target_class_name": class_name,
        "template_text": template_text,
        "dataset": normalized_dataset,
        "dataset_final_requirements": dataset_final_requirements,
    }
    s3_prompt = _stage_prompt(stages[3], s3_input, s3_schema, stage_index=3)
    s3, s3_raw, s3_elapsed = await _call_stage_json_with_meta(llm_call_fn, s3_prompt, stage_name="Stage 3")
    stage_traces.append(
        BuildStageTrace(
            stage=3,
            stage_name="Stage 3",
            prompt=s3_prompt,
            raw_response=s3_raw,
            parsed_json=s3,
            elapsed_sec=s3_elapsed,
        )
    )
    mas_code = s3.get("mas_code")
    if not isinstance(mas_code, str) or not mas_code.strip():
        raise RuntimeError("Stage 3 returned empty mas_code.")
    mas_code = mas_code.strip()
    mas_code = _strip_leading_import_noise(mas_code)
    mas_code = _enforce_dataset_tool_modes_in_mas_code(mas_code, dataset_name=normalized_dataset)
    # P0: sanitize final_output assignments — remove UNANSWERABLE/hardcoded fallbacks,
    # f-string concatenations, and ensure a single agent-reference assignment exists.
    sub_agent_role_names = [s.role_name for s in subs]
    mas_code = _sanitize_final_output_in_mas_code(
        mas_code,
        sub_agent_role_names=sub_agent_role_names,
        preferred_final_role_name=preferred_final_role_name,
    )
    if normalized_dataset == "vita":
        mas_code = _enforce_final_output_role_in_mas_code(
            mas_code,
            final_role_name=preferred_final_role_name,
        )
    if normalized_dataset == "hlemath":
        mas_code = _enforce_final_output_role_in_mas_code(
            mas_code,
            final_role_name=preferred_final_role_name,
        )
    mas_code = prepend_fixed_imports(mas_code)
    compilable, compile_err = _is_mas_code_compilable(mas_code)
    use_deterministic_fallback = not compilable
    # Interactive environments must not begin execution with a topology that
    # references agents never instantiated in __init__.  Stage-3 model output
    # can be syntactically valid while still doing exactly that; execution then
    # consumes environment actions before failing and each repair retry repeats
    # the damage.  Keep the model-generated Stage 1/2 agent design, but use the
    # deterministic sequential assembler when the Stage-3 binding is incomplete.
    if compilable and normalized_dataset in {"alfworld", "scienceworld"}:
        defined_agents = set(re.findall(r"self\.([A-Za-z_]\w*)\s*=\s*SubAgentRequest", mas_code))
        executed_agents = set(re.findall(r"self\.([A-Za-z_]\w*)\.execute_async", mas_code))
        missing_agents = sorted(executed_agents - defined_agents)
        if not executed_agents or missing_agents:
            use_deterministic_fallback = True
            print(
                "[Stage 3][interactive] incomplete agent bindings; "
                f"missing={missing_agents or ['no executable agents']}; "
                "switching to deterministic sequential workflow code.",
                flush=True,
            )
    if compilable and normalized_dataset == "hlemath":
        behavior_ok, behavior_issues = _hlemath_validate_mas_code_behavior(
            mas_code,
            sub_agent_role_names=sub_agent_role_names,
            preferred_final_role_name=preferred_final_role_name,
        )
        if not behavior_ok:
            use_deterministic_fallback = True
            print(
                f"[Stage 3][HLEMATH] mas_code behavioral check failed ({len(behavior_issues)}); "
                "switching to deterministic sequential workflow code.",
                flush=True,
            )
            for issue in behavior_issues:
                print(f"[Stage 3][HLEMATH]   - {issue}", flush=True)
    if use_deterministic_fallback:
        if not compilable:
            print(
                f"[Stage 3] fallback: generated mas_code not compilable ({compile_err}); "
                "switching to deterministic sequential workflow code.",
                flush=True,
            )
        mas_code = prepend_fixed_imports(
            _build_deterministic_sequential_workflow_code(class_name=class_name, subs=subs)
        )
    print(f"[Stage 3] mas_code_chars={len(mas_code)}", flush=True)
    return ThreeStageBuildArtifacts(
        mas_code=mas_code,
        stage_traces=stage_traces,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        normalized_sub_agents=normalized_sub_agents,
    )


async def run_generated_workflow(
    *,
    mas_code: str,
    class_name: str,
    task_text: str,
    text_call_fn: AsyncTextCallFn | SyncTextCallFn,
    tool_call_fn: AsyncToolCallFn | SyncToolCallFn | None = None,
) -> dict[str, Any]:
    """
    Exec generated MAS code and run forward().
    Returns workflow state dict.
    """
    # Ensure generated code can import template modules regardless of launch path.
    building_root = str(Path(__file__).resolve().parents[1])
    if building_root not in sys.path:
        sys.path.insert(0, building_root)


    # Generated workflows are model-authored and do not all use the exact
    # constructor signature from the reference template.  Keep the compatibility
    # handling here (rather than changing the evaluator or generated source) so a
    # no-argument workflow, a legacy one-argument workflow, and the current
    # keyword-based workflow all execute against the same live tool callbacks.
    # These imports are intentionally supplied only to the generated namespace;
    # they do not alter benchmark scoring or the fixed template.
    ns: dict[str, Any] = {
        "os": __import__("os"),
        "asyncio": asyncio,
        "json": json,
        "re": re,
    }
    exec(mas_code, ns, ns)
    if class_name not in ns:
        raise RuntimeError(f"Generated code did not define class {class_name}.")
    cls = ns[class_name]
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        signature = None
    kwargs: dict[str, Any] = {}
    if signature is not None:
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "text_call_fn" in parameters:
            kwargs["text_call_fn"] = text_call_fn
        if accepts_kwargs or "tool_call_fn" in parameters:
            kwargs["tool_call_fn"] = tool_call_fn
    try:
        workflow = cls(**kwargs)
    except TypeError as constructor_error:
        # A few older generated workflows expose only a no-arg constructor.
        # Retry only when the failed call supplied callback kwargs; otherwise
        # preserve the original TypeError so genuine constructor bugs are visible.
        if kwargs:
            try:
                workflow = cls()
            except TypeError:
                raise constructor_error
        else:
            raise
    # Legacy workflows may read callbacks as instance attributes even when their
    # constructor does not accept them.  Inject only missing attributes.
    if not hasattr(workflow, "text_call_fn"):
        try:
            workflow.text_call_fn = text_call_fn
        except Exception:
            pass
    if not hasattr(workflow, "tool_call_fn"):
        try:
            workflow.tool_call_fn = tool_call_fn
        except Exception:
            pass
    out = workflow.forward_async(task_text)
    if asyncio.iscoroutine(out):
        out = await out
    if not isinstance(out, dict):
        raise RuntimeError("Workflow forward() must return a dict state.")
    return out


def _extract_predicted_answer(text: str) -> str:
    s = (text or "").strip()
    m = re.search(r"\\boxed\{([^}]+)\}", s)
    if m:
        return m.group(1).strip()
    m2 = re.findall(r"-?\d+(?:\.\d+)?", s)
    if m2:
        return m2[-1].strip()
    return s


async def generate_mas_code_for_task(
    *,
    task_text: str,
    class_name: str = "GeneratedMASWorkflow",
    init_skill_path: str | Path | None = None,
    planner_call_fn: Callable[[str], Awaitable[str]] | None = None,
    dataset_name: str = "unknown",
) -> str:
    """Generate MAS code by running init_skill stage-1/2/3 sequentially."""
    if planner_call_fn is None:
        raise RuntimeError("planner_call_fn is required for three-stage build.")
    template_text = stage3_llm_reference_bundle()
    _skill_text = ""
    if init_skill_path is not None and Path(init_skill_path).is_file():
        _skill_text = _load_text(init_skill_path)
    if not _skill_text:
        raise RuntimeError(f"init skill file not found or empty: {init_skill_path}")
    return await run_three_stage_skill_build(
        task_text=task_text,
        init_skill_text=_skill_text,
        template_text=template_text,
        llm_call_fn=planner_call_fn,
        class_name=class_name,
        dataset_name=dataset_name,
    )


async def generate_mas_artifacts_for_task(
    *,
    task_text: str,
    class_name: str = "GeneratedMASWorkflow",
    init_skill_path: str | Path | None = None,
    planner_call_fn: Callable[[str], Awaitable[str]] | None = None,
    dataset_name: str = "unknown",
) -> ThreeStageBuildArtifacts:
    """Generate MAS code with full stage traces for logging/debugging."""
    if planner_call_fn is None:
        raise RuntimeError("planner_call_fn is required for three-stage build.")
    template_text = stage3_llm_reference_bundle()
    _skill_text = ""
    if init_skill_path is not None and Path(init_skill_path).is_file():
        _skill_text = _load_text(init_skill_path)
    if not _skill_text:
        raise RuntimeError(f"init skill file not found or empty: {init_skill_path}")
    return await run_three_stage_skill_build_with_trace(
        task_text=task_text,
        init_skill_text=_skill_text,
        template_text=template_text,
        llm_call_fn=planner_call_fn,
        class_name=class_name,
        dataset_name=dataset_name,
    )


async def generate_mas_artifacts_with_retries(
    *,
    task_text: str,
    class_name: str = "GeneratedMASWorkflow",
    init_skill_path: str | Path | None = None,
    planner_call_fn: Callable[[str], Awaitable[str]] | None = None,
    max_attempts: int = 5,
    dataset_name: str = "unknown",
) -> tuple[ThreeStageBuildArtifacts | None, int, list[dict[str, Any]]]:
    """Generate MAS artifacts with retry budget for stage/build failures."""
    attempts = max(1, int(max_attempts))
    events: list[dict[str, Any]] = []
    last_error = ""
    for i in range(1, attempts + 1):
        try:
            artifacts = await generate_mas_artifacts_for_task(
                task_text=task_text,
                class_name=class_name,
                init_skill_path=init_skill_path,
                planner_call_fn=planner_call_fn,
                dataset_name=dataset_name,
            )
            if i > 1:
                events.append(
                    {
                        "type": "generation_retry_success",
                        "attempt": i,
                        "message": "generation succeeded after retries",
                    }
                )
            return artifacts, i, events
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            events.append(
                {
                    "type": "generation_retry_error",
                    "attempt": i,
                    "message": last_error,
                }
            )
            print(f"[MAS generation retry {i}/{attempts}] {last_error}", flush=True)
    events.append(
        {
            "type": "generation_retry_exhausted",
            "attempt": attempts,
            "message": last_error or "unknown generation failure",
        }
    )
    return None, attempts, events


async def _repair_mas_code_once(
    *,
    broken_code: str,
    class_name: str,
    task_text: str,
    error_message: str,
    planner_call_fn: Callable[[str], Awaitable[str]],
) -> str:
    repair_prompt = (
        "You are a Python code fixer for a generated MAS workflow. "
        "Return JSON only with key 'mas_code'.\n\n"
        f"[TASK]\n{task_text}\n\n"
        f"[TARGET_CLASS_NAME]\n{class_name}\n\n"
        f"[ERROR]\n{error_message}\n\n"
        "[BROKEN_MAS_CODE]\n"
        f"{broken_code}\n\n"
        "Constraints:\n"
        "- Keep the same class name.\n"
        "- Keep async workflow executable.\n"
        "- forward_async must return a dict and include final_output.\n"
        "- Do NOT include from __future__, from typing, or from template.sub_agent imports "
        "(the runner prepends them).\n"
        "- SubAgentRequest string fields must use Python-safe quoting (prefer triple-single quotes '''...''').\n"
        "- Return valid Python code only in mas_code.\n"
        '- Output JSON format: {"mas_code":"..."}\n'
    )
    raw = await planner_call_fn(repair_prompt)
    obj = _extract_best_json_dict_for_stage(raw, stage_hint=3)
    if not isinstance(obj, dict):
        raise RuntimeError("repair step returned non-JSON content")
    fixed = obj.get("mas_code")
    if not isinstance(fixed, str) or not fixed.strip():
        raise RuntimeError("repair step returned empty mas_code")
    return fixed


async def run_generated_workflow_with_retries(
    *,
    mas_code: str,
    class_name: str,
    task_text: str,
    text_call_fn: AsyncTextCallFn | SyncTextCallFn,
    planner_call_fn: Callable[[str], Awaitable[str]],
    tool_call_fn: AsyncToolCallFn | SyncToolCallFn | None = None,
    max_attempts: int = 3,
    dataset_name: str = "unknown",
    fallback_mas_code: str | None = None,
) -> tuple[bool, dict[str, Any], str, int, list[dict[str, Any]], str | None]:
    """
    Execute generated MAS with retry budget.
    On execution failure, tries to repair mas_code and rerun.
    """
    attempts = max(1, int(max_attempts))
    code = _enforce_dataset_tool_modes_in_mas_code(mas_code.strip(), dataset_name=dataset_name)
    fallback_code: str | None = None
    if isinstance(fallback_mas_code, str) and fallback_mas_code.strip():
        candidate = _enforce_dataset_tool_modes_in_mas_code(
            fallback_mas_code.strip(), dataset_name=dataset_name
        )
        candidate = _strip_leading_import_noise(candidate)
        candidate = prepend_fixed_imports(candidate)
        fallback_ok, fallback_err = _is_mas_code_compilable(candidate)
        if fallback_ok:
            fallback_code = candidate
        else:
            print(
                f"[MAS execution] ignoring invalid deterministic fallback: {fallback_err}",
                flush=True,
            )
    events: list[dict[str, Any]] = []
    last_error: str | None = None
    fallback_used = False
    for i in range(1, attempts + 1):
        try:
            state = await run_generated_workflow(
                mas_code=code,
                class_name=class_name,
                task_text=task_text,
                text_call_fn=text_call_fn,
                tool_call_fn=tool_call_fn,
            )
            if i > 1:
                events.append(
                    {
                        "type": "execution_retry_success",
                        "attempt": i,
                        "message": "execution succeeded after code repair retries",
                    }
                )
            return True, state, code, i, events, None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            events.append(
                {
                    "type": "execution_retry_error",
                    "attempt": i,
                    "message": last_error,
                }
            )
            print(f"[MAS execution retry {i}/{attempts}] {last_error}", flush=True)
            # Skip retries on BadRequestError (context length overflow cannot be fixed by repair)
            if "BadRequestError" in type(e).__name__ or "context length" in str(e).lower():
                print(f"[MAS execution] skipping retries for unrecoverable error: {last_error}", flush=True)
                break
            if i >= attempts:
                break
            # Interactive environments are stateful: repeatedly executing broken,
            # model-repaired orchestration can consume additional irreversible
            # actions.  Prefer the project-local deterministic workflow exactly
            # once before asking the model to repair arbitrary Python.
            if fallback_code is not None and not fallback_used and fallback_code != code:
                code = fallback_code
                fallback_used = True
                events.append(
                    {
                        "type": "execution_deterministic_fallback",
                        "attempt": i,
                        "message": "switched to deterministic sequential MAS code",
                    }
                )
                continue
            # ScienceWorld/ALFWorld episodes are stateful.  Once the local
            # deterministic embodied workflow has been selected, never ask the
            # planner LLM to synthesize/repair arbitrary Python: a repair can
            # have an incompatible constructor or tool signature and can also
            # consume irreversible environment steps.  The next attempt (if any)
            # simply retries the same deterministic workflow.
            if fallback_used and str(dataset_name or "").strip().lower() in {"alfworld", "scienceworld"}:
                events.append(
                    {
                        "type": "execution_repair_skipped_embodied",
                        "attempt": i,
                        "message": "skipped LLM code repair after deterministic embodied fallback",
                    }
                )
                continue
            try:
                repaired_code = await _repair_mas_code_once(
                    broken_code=code,
                    class_name=class_name,
                    task_text=task_text,
                    error_message=last_error,
                    planner_call_fn=planner_call_fn,
                )
                repaired_code = _enforce_dataset_tool_modes_in_mas_code(
                    repaired_code.strip(), dataset_name=dataset_name
                )
                repaired_code = _strip_leading_import_noise(repaired_code)
                repaired_code = prepend_fixed_imports(repaired_code)
                repair_ok, repair_compile_err = _is_mas_code_compilable(repaired_code)
                if not repair_ok:
                    raise RuntimeError(
                        f"repair step returned non-compilable mas_code: {repair_compile_err}"
                    )
                code = repaired_code
                events.append(
                    {
                        "type": "execution_repair_generated",
                        "attempt": i,
                        "message": "generated repaired MAS code",
                    }
                )
            except Exception as repair_err:
                repair_msg = f"{type(repair_err).__name__}: {repair_err}"
                events.append(
                    {
                        "type": "execution_repair_error",
                        "attempt": i,
                        "message": repair_msg,
                    }
                )
                print(f"[MAS repair retry {i}/{attempts}] {repair_msg}", flush=True)
    events.append(
        {
            "type": "execution_retry_exhausted",
            "attempt": attempts,
            "message": last_error or "unknown execution failure",
        }
    )
    return False, {}, code, attempts, events, last_error


async def run_mas_pipeline_with_retries(
    *,
    task_text: str,
    class_name: str,
    init_skill_path: str | Path,
    planner_call_fn: Callable[[str], Awaitable[str]],
    text_call_fn: AsyncTextCallFn | SyncTextCallFn,
    tool_call_fn: AsyncToolCallFn | SyncToolCallFn | None = None,
    max_generation_attempts: int = 5,
    max_execution_attempts: int = 3,
    dataset_name: str = "unknown",
) -> MASRunWithRetryResult:
    """
    End-to-end MAS build + execution with retry policies.
    Never raises for expected build/execution failures.
    """
    retry_events: list[dict[str, Any]] = []
    artifacts, gen_attempts, gen_events = await generate_mas_artifacts_with_retries(
        task_text=task_text,
        class_name=class_name,
        init_skill_path=init_skill_path,
        planner_call_fn=planner_call_fn,
        max_attempts=max_generation_attempts,
        dataset_name=dataset_name,
    )
    retry_events.extend(gen_events)
    if artifacts is None:
        reason = ""
        for ev in reversed(retry_events):
            if ev.get("type") in {"generation_retry_error", "generation_retry_exhausted"}:
                reason = str(ev.get("message") or "")
                break
        normalized_dataset = _normalize_dataset_name(dataset_name)
        if normalized_dataset in {"alfworld", "scienceworld"}:
            fallback_subs = _default_interactive_sub_agents(normalized_dataset)
            fallback_code = prepend_fixed_imports(
                _build_deterministic_sequential_workflow_code(
                    class_name=class_name, subs=fallback_subs
                )
            )
            fallback_stage3 = {
                "mas_code": fallback_code,
                "reasoning": "generation-safe deterministic embodied fallback",
            }
            artifacts = ThreeStageBuildArtifacts(
                mas_code=fallback_code,
                stage_traces=[],
                stage1={"fallback": True, "generation_error": reason},
                stage2={"sub_agents": [spec.__dict__ for spec in fallback_subs]},
                stage3=fallback_stage3,
                normalized_sub_agents=[spec.__dict__ for spec in fallback_subs],
            )
            retry_events.append(
                {
                    "type": "generation_deterministic_embodied_fallback",
                    "attempt": gen_attempts,
                    "message": reason or "planner generation failed; used safe embodied fallback",
                }
            )
            print(
                "[MAS generation] planner failed; continuing with deterministic embodied fallback",
                flush=True,
            )
        else:
            return MASRunWithRetryResult(
                success=False,
                final_output="",
                state={},
                mas_code="",
                artifacts=None,
                generation_attempts_used=gen_attempts,
                execution_attempts_used=0,
                failure_stage="generation",
                failure_reason=reason or "generation failed after retries",
                retry_events=retry_events,
                graph_edges=None,
            )

    fallback_mas_code: str | None = None
    if str(dataset_name or "").strip().lower() in {"alfworld", "scienceworld"}:
        try:
            fallback_subs = [SubTaskSpec(**item) for item in artifacts.normalized_sub_agents]
            fallback_mas_code = prepend_fixed_imports(
                _build_deterministic_sequential_workflow_code(
                    class_name=class_name,
                    subs=fallback_subs,
                )
            )
        except Exception as fallback_err:
            retry_events.append(
                {
                    "type": "execution_deterministic_fallback_error",
                    "attempt": 0,
                    "message": f"{type(fallback_err).__name__}: {fallback_err}",
                }
            )

    # For stateful embodied episodes, always reserve one deterministic fallback
    # attempt when a generated workflow fails. This prevents a malformed Stage-3
    # program from being reported as a generation-only benchmark failure.
    normalized_dataset = _normalize_dataset_name(dataset_name)
    effective_execution_attempts = max(
        2 if normalized_dataset in {"alfworld", "scienceworld"} and fallback_mas_code else 1,
        int(max_execution_attempts),
    )
    ok, state, final_code, exec_attempts, exec_events, exec_err = await run_generated_workflow_with_retries(
        mas_code=artifacts.mas_code,
        class_name=class_name,
        task_text=task_text,
        text_call_fn=text_call_fn,
        planner_call_fn=planner_call_fn,
        tool_call_fn=tool_call_fn,
        max_attempts=effective_execution_attempts,
        dataset_name=dataset_name,
        fallback_mas_code=fallback_mas_code,
    )
    retry_events.extend(exec_events)
    final_output = ""
    if isinstance(state, dict):
        final_output = str(state.get("final_output", "") or "")
    return MASRunWithRetryResult(
        success=ok,
        final_output=final_output,
        state=state if isinstance(state, dict) else {},
        mas_code=final_code,
        artifacts=artifacts,
        generation_attempts_used=gen_attempts,
        execution_attempts_used=exec_attempts,
        failure_stage=None if ok else "execution",
        failure_reason=None if ok else (exec_err or "execution failed after retries"),
        retry_events=retry_events,
        graph_edges=None,
    )




async def generate_mas_code_via_xml_orchestra(
    *,
    task_text: str,
    class_name: str = "GeneratedMASWorkflow",
    orchestra_client: Any = None,
    dataset_name: str = "unknown",
) -> str:
    """Generate MAS code using XML-based orchestra (2-stage XML + assembly).

    The orchestra_client (e.g., qwen3-8b) runs the 2-stage XML decomposition.
    The compiler produces executable Python code from the MASGraph.
    """
    from building.formalize.orchestra import run_formalized_orchestra
    from building.formalize.compiler import compile_graph_to_code

    if orchestra_client is None:
        raise RuntimeError("orchestra_client is required for XML orchestra.")

    graph, trace = await run_formalized_orchestra(
        orchestra_client, task_text, dataset_name
    )
    code = compile_graph_to_code(graph, class_name)
    return code


async def run_mas_pipeline_with_xml_orchestra(
    *,
    task_text: str,
    class_name: str,
    orchestra_client: Any,
    sub_agent_client: Any,
    tool_call_fn: Any = None,
    dataset_name: str = "unknown",
    max_execution_attempts: int = 3,
) -> MASRunWithRetryResult:
    """End-to-end MAS pipeline using XML orchestra.

    - orchestra_client (e.g., qwen3-8b): runs the 2-stage XML decomposition
    - sub_agent_client (e.g., Qwen3-30B): executes the generated workflow
    """
    from building.formalize.orchestra import run_formalized_orchestra
    from building.formalize.compiler import compile_graph_to_code

    retry_events: list[dict[str, Any]] = []

    try:
        graph, trace = await run_formalized_orchestra(
            orchestra_client, task_text, dataset_name
        )
        mas_code = compile_graph_to_code(graph, class_name)
        # Extract edge info from MASGraph for saving in trace JSON
        graph_edges = [
            {"from": e.f, "to": e.t, "type": e.type.value if hasattr(e, "type") else str(e.type)}
            for e in graph.edges
        ]
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        return MASRunWithRetryResult(
            success=False,
            final_output="",
            state={},
            mas_code="",
            artifacts=None,
            generation_attempts_used=1,
            execution_attempts_used=0,
            failure_stage="generation",
            failure_reason=reason,
            retry_events=[{"type": "xml_orchestra_error", "message": reason}],
            graph_edges=None,
        )

    text_call_fn = make_text_call_fn(sub_agent_client)

    ok, state, final_code, exec_attempts, exec_events, exec_err = await run_generated_workflow_with_retries(
        mas_code=mas_code,
        class_name=class_name,
        task_text=task_text,
        text_call_fn=text_call_fn,
        planner_call_fn=make_planner_call_fn(orchestra_client),
        tool_call_fn=tool_call_fn,
        max_attempts=max_execution_attempts,
        dataset_name=dataset_name,
    )
    retry_events.extend(exec_events)
    final_output = ""
    if isinstance(state, dict):
        final_output = str(state.get("final_output", "") or "")
    return MASRunWithRetryResult(
        success=ok,
        final_output=final_output,
        state=state if isinstance(state, dict) else {},
        mas_code=final_code,
        artifacts=None,
        generation_attempts_used=1,
        execution_attempts_used=exec_attempts,
        failure_stage=None if ok else "execution",
        failure_reason=None if ok else (exec_err or "execution failed after retries"),
        retry_events=retry_events,
        graph_edges=graph_edges,
    )


async def _main() -> None:
    # Default quick test: use the first HLEMath sample.
    hlemath_jsonl = Path(__file__).resolve().parents[2] / "datasets" / "hlemath" / "data" / "hlemath_test.jsonl"
    init_skill = Path(__file__).resolve().parents[1] / "init_skill" / "SKILL.md"
    model = "qwen3.5-plus"

    with hlemath_jsonl.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        raise RuntimeError(f"No sample found in {hlemath_jsonl}")
    row = json.loads(first)
    task_text = str(row.get("question", "")).strip()
    if not task_text:
        raise RuntimeError("First sample has empty question")

    from openai_async_client import AsyncOpenAIClient

    client = AsyncOpenAIClient(model=model)
    planner_call_fn = make_planner_call_fn(client)

    class_name = f"HLEMathTask{row.get('id', '0')}_MASWorkflow"
    code = await generate_mas_code_for_task(
        task_text=task_text,
        class_name=class_name,
        init_skill_path=init_skill,
        planner_call_fn=planner_call_fn,
    )
    print(code)

    text_call_fn = make_text_call_fn(client)
    state = await run_generated_workflow(
        mas_code=code,
        class_name=class_name,
        task_text=task_text,
        text_call_fn=text_call_fn,
        tool_call_fn=None,
    )
    final_output = str(state.get("final_output", ""))
    pred = _extract_predicted_answer(final_output)
    gold = str(row.get("answer", "")).strip()
    print("\n===== WORKFLOW RESULT =====")
    print(f"final_output: {final_output}")
    print(f"predicted_answer: {pred}")
    print(f"gold_answer: {gold}")
    print(f"match: {pred == gold}")


if __name__ == "__main__":
    asyncio.run(_main())
