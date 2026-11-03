"""Utilities for using a single verl rollout as a MAS-Gate XML orchestra design."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from building.formalize.orchestra import assemble_graph
from building.formalize.schema import (
    AgentNode,
    AgentSpec,
    AgentSpecs,
    EdgeType,
    GraphEdge,
    MASGraph,
    SubTask,
    TaskDecomposition,
    Topology,
    extract_xml,
)


def _text(v: Any) -> str:
    if isinstance(v, dict):
        return str(v.get("#text", ""))
    if isinstance(v, str):
        return v
    return str(v) if v is not None else ""


def _safe_id(raw: str, used: set[str], fallback_idx: int) -> str:
    s = re.sub(r"\W+", "_", (raw or "").strip()).strip("_")
    if not s or not re.match(r"^[A-Za-z_]", s):
        s = f"agent_{fallback_idx}"
    base = s
    k = 2
    while s in used:
        s = f"{base}_{k}"
        k += 1
    used.add(s)
    return s


_MAS_XML_TAGS = (
    "editor_output|mas_graph|task_decomposition|goal|sub_tasks|sub_task|"
    "agent_specs|agents|agent|role|sub_task_idx|dependencies|nodes|edges|"
    "edge|task|constraints|constraint|execution_mode"
)

_VALID_EXECUTION_MODES = {"llm_only", "multi_turn_search"}
_BCP_DATASETS = {"bcp", "browsecomp", "browsecomp_plus", "mas_gate_bcp"}
_HOTPOTQA_DATASETS = {"hotpotqa", "hotpot_qa"}
_SEARCH_CAPABLE_DATASETS = _BCP_DATASETS | _HOTPOTQA_DATASETS
_SEARCH_TERMS = (
    "search",
    "retrieve",
    "retrieval",
    "evidence",
    "corpus",
    "bm25",
    "document",
    "source",
    "lookup",
    "research",
    "fact finding",
)


def _normalized_execution_mode(value: Any) -> str | None:
    mode = _text(value).strip().lower()
    return mode if mode in _VALID_EXECUTION_MODES else None


def _looks_like_search_node(*, role: str, task: str = "", goal: str = "") -> bool:
    blob = f"{role} {task} {goal}".lower()
    return any(term in blob for term in _SEARCH_TERMS)


def _repair_xml_text_literals(text: str) -> str:
    """Escape common math literals that make otherwise valid policy XML invalid.

    AIME tasks and generated roles frequently contain comparisons such as
    ``0 < x < 2\\pi`` or an unescaped ampersand.  ElementTree correctly rejects
    those raw characters, but the surrounding editor_output is structurally
    valid and should not be counted as a failed MAS design.  Preserve only the
    schema tags accepted by this module and escape every other ``<`` plus bare
    ampersands as text.
    """
    text = re.sub(
        rf"<(?!/?(?:{_MAS_XML_TAGS})(?=[\s>/])|[!?])",
        "&lt;",
        text,
    )
    return re.sub(
        r"&(?!(?:#\d+|#x[0-9A-Fa-f]+|amp|lt|gt|quot|apos);)",
        "&amp;",
        text,
    )


def _parse_task_decomposition(obj: dict[str, Any]) -> TaskDecomposition:
    td = obj.get("task_decomposition", obj)
    goal = _text(td.get("goal", ""))
    sub_tasks: list[SubTask] = []
    sts = td.get("sub_tasks", {})
    st_list = sts.get("sub_task", []) if isinstance(sts, dict) else []
    if isinstance(st_list, dict):
        st_list = [st_list]
    for i, st in enumerate(st_list):
        try:
            idx = int(st.get("@idx", i))
        except Exception:
            idx = i
        sub_tasks.append(SubTask(idx=idx, desc=_text(st)))
    return TaskDecomposition(goal=goal, sub_tasks=sub_tasks)


def _parse_agent_specs(obj: dict[str, Any]) -> AgentSpecs:
    as_data = obj.get("agent_specs", obj)
    agents_data = as_data.get("agents", {}) if isinstance(as_data, dict) else {}
    alist = agents_data.get("agent", []) if isinstance(agents_data, dict) else []
    if isinstance(alist, dict):
        alist = [alist]
    raw_specs: list[AgentSpec] = []
    for i, a in enumerate(alist):
        if not isinstance(a, dict):
            continue
        aid = str(a.get("@id", f"agent_{i}"))
        role = _text(a.get("role", ""))
        is_final = str(a.get("@is_final", "false")).lower() == "true"
        deps_str = _text(a.get("dependencies", ""))
        deps = [d.strip() for d in deps_str.split(",") if d.strip()] if deps_str else []
        try:
            st_idx = int(_text(a.get("sub_task_idx", "0")))
        except Exception:
            st_idx = 0
        execution_mode = _normalized_execution_mode(
            a.get("@execution_mode", a.get("execution_mode"))
        )
        raw_specs.append(
            AgentSpec(
                id=aid,
                role=role,
                sub_task_idx=st_idx,
                dependencies=deps,
                is_final=is_final,
                execution_mode=execution_mode,
            )
        )

    used: set[str] = set()
    id_map: dict[str, str] = {}
    out: list[AgentSpec] = []
    for i, spec in enumerate(raw_specs):
        sid = _safe_id(spec.id, used, i)
        id_map[spec.id] = sid
        out.append(replace(spec, id=sid))
    # Dependencies are graph references, not free-form instructions.  Orchestra
    # rollouts occasionally put an entire sentence (including quotes/LaTeX) in
    # <dependencies>.  Keeping such text creates edges whose endpoint is not a
    # node and, later, invalid generated Python.  Accept only exact earlier agent
    # ids; unknown/self/forward references are ignored.  assemble_graph() adds a
    # safe sequential fallback when no valid edges remain.
    normalized: list[AgentSpec] = []
    seen_ids: set[str] = set()
    for spec in out:
        valid_deps: list[str] = []
        for dep in spec.dependencies:
            mapped = id_map.get(dep)
            if mapped and mapped in seen_ids and mapped != spec.id and mapped not in valid_deps:
                valid_deps.append(mapped)
        normalized.append(replace(spec, dependencies=valid_deps))
        seen_ids.add(spec.id)
    if normalized and not any(s.is_final for s in normalized):
        normalized[-1] = replace(normalized[-1], is_final=True)
    return AgentSpecs(specs=normalized)


def graph_from_rollout_xml(solution_str: str, task_text: str, dataset_name: str = "aime") -> MASGraph:
    """Parse a verl rollout into MASGraph.

    Accepted formats:
      1. ``<mas_graph>...</mas_graph>`` (compiled directly);
      2. ``<editor_output><task_decomposition>...</...><agent_specs>...</...></editor_output>``;
      3. two separate XML blocks containing ``task_decomposition`` and ``agent_specs``.
    """
    text = _repair_xml_text_literals(solution_str or "")
    # Qwen occasionally closes </agents> and </editor_output> but omits the
    # single intervening </agent_specs>.  Repair only that exact container
    # omission.  COSEC: the bounded counts and adjacent closing-tag pattern
    # avoid turning arbitrary untrusted text into a broadly repaired XML tree;
    # extract_xml still rejects all DTD/entity declarations before parsing.
    if (
        len(re.findall(r"<agent_specs(?=[\s>])", text, flags=re.IGNORECASE)) == 1
        and not re.search(r"</agent_specs\s*>", text, flags=re.IGNORECASE)
    ):
        text = re.sub(
            r"(</agents\s*>)(\s*</editor_output\s*>)",
            r"\1</agent_specs>\2",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    # Qwen occasionally emits a recoverable typo such as
    # ``<sub_task_idx>1>`` (missing the explicit closing tag).  xmltodict then
    # returns a partial editor_output and assemble_graph can produce an empty
    # graph.  Repair only this narrowly-defined scalar-tag typo; do not invent
    # agents or silently accept arbitrary malformed policy output.
    text = re.sub(
        r"<sub_task_idx>\s*([^<>\r\n]+?)\s*>",
        lambda m: f"<sub_task_idx>{m.group(1).strip()}</sub_task_idx>",
        text,
    )
    data = extract_xml(text)
    if "mas_graph" in data:
        graph = MASGraph.from_xml(data["mas_graph"])
        if graph.nodes:
            return _normalize_dataset_orchestration(
                _sanitize_graph(graph, dataset_name), task_text, dataset_name
            )

    if "editor_output" in data:
        eo = data["editor_output"]
        td = _parse_task_decomposition(eo.get("task_decomposition", {}))
        specs = _parse_agent_specs(eo.get("agent_specs", {}))
        graph = assemble_graph(td, specs, dataset_name)
        return _normalize_dataset_orchestration(
            _sanitize_graph(graph, dataset_name), task_text, dataset_name
        )

    # extract_xml returns only the first block, so search both blocks manually.
    td_match = re.search(r"<task_decomposition[\s\S]*?</task_decomposition>", text)
    as_match = re.search(r"<agent_specs[\s\S]*?</agent_specs>", text)
    if td_match and as_match:
        td_data = extract_xml(td_match.group(0)).get("task_decomposition", {})
        as_data = extract_xml(as_match.group(0)).get("agent_specs", {})
        graph = assemble_graph(_parse_task_decomposition(td_data), _parse_agent_specs(as_data), dataset_name)
        return _normalize_dataset_orchestration(
            _sanitize_graph(graph, dataset_name), task_text, dataset_name
        )

    raise ValueError("rollout did not contain parseable <mas_graph> or task_decomposition+agent_specs XML")


def _canonical_aime_graph(task_text: str, source_graph: MASGraph) -> MASGraph:
    """Use a conservative solve/audit/repair protocol for AIME.

    Trajectory analysis showed that independent re-solves disagree frequently and
    have a very low candidate oracle.  The useful operation is therefore a
    targeted proof audit followed by a repair only when the audit identifies a
    concrete fatal step.  The original policy XML is still generated and stored
    for auditability, but runtime orchestration is normalized deterministically
    for this dataset.
    """
    goal = source_graph.nodes[0].goal if source_graph.nodes else ""
    goal = goal or "Solve the AIME problem and return its requested integer."
    nodes = [
        AgentNode(
            id="primary_solver",
            role=(
                "PRIMARY AIME SOLVER. Solve the complete problem from GLOBAL TASK CONTEXT using exact "
                "mathematics. Give a coherent derivation, track all domain/range restrictions, and check "
                "the requested quantity. End with exactly `Proposed integer: N`, where N is your candidate "
                "integer in 0..999. Do not discuss orchestration or defer any part of the solution."
            ),
            task="Derive one complete exact solution and a proposed AIME integer.",
            goal=goal,
            is_final=False,
            execution_mode="llm_only",
        ),
        AgentNode(
            id="proof_auditor",
            role=(
                "AIME PROOF AUDITOR. Audit the upstream primary solution line by line against GLOBAL TASK "
                "CONTEXT. Check algebra, case completeness, endpoints, divisibility, extraneous roots, and "
                "whether the computed quantity is the one asked for. Do not replace a valid solution merely "
                "because you know another method. Produce a self-contained report. If no fatal error is found, "
                "end with `Audit: VERIFIED; integer N`. If there is a fatal error, identify the first specific "
                "invalid step, show the minimum exact correction needed, and end with "
                "`Audit: ERROR; corrected integer M`. Never report ERROR based only on uncertainty or style."
            ),
            task="Verify the primary derivation and state a concrete correction only if required.",
            goal=goal,
            is_final=False,
            execution_mode="llm_only",
        ),
        AgentNode(
            id="conservative_repairer",
            role=(
                "CONSERVATIVE AIME REPAIRER. Read the self-contained audit. If it ends VERIFIED, preserve its "
                "integer exactly and do not re-solve. If it ends ERROR, accept a changed integer only when the "
                "report names a concrete fatal step and supplies a mathematically supported correction; verify "
                "that local correction before accepting it. If the alleged error is vague, preserve the original "
                "candidate recorded in the report. End with exactly `Accepted integer: N`."
            ),
            task="Apply only a justified local repair and select the accepted integer.",
            goal=goal,
            is_final=False,
            execution_mode="llm_only",
        ),
        AgentNode(
            id="final_answer",
            role=(
                "FINAL AIME FORMATTER. You are not a fresh solver. Copy the integer after `Accepted integer:` "
                "from the sole upstream dependency. Check only that it is one integer in 0..999. Do not alter it, "
                "do not recompute the problem, and do not add explanation. Output exactly one line: `\\boxed{N}`."
            ),
            task="Map the accepted integer to the required boxed output without new reasoning.",
            goal=goal,
            is_final=True,
            execution_mode="llm_only",
        ),
    ]
    edges = [
        GraphEdge("primary_solver", "proof_auditor", EdgeType.DATA),
        GraphEdge("proof_auditor", "conservative_repairer", EdgeType.DATA),
        GraphEdge("conservative_repairer", "final_answer", EdgeType.DATA),
    ]
    return MASGraph(nodes=nodes, edges=edges, topology=Topology.DAG)


def _canonical_gpqa_graph(task_text: str, source_graph: MASGraph) -> MASGraph:
    """Use two independent roots and a conservative evidence gate for GPQA."""
    goal = source_graph.nodes[0].goal if source_graph.nodes else ""
    goal = goal or "Select the scientifically correct original option A, B, C, or D."
    nodes = [
        AgentNode(
            id="primary_domain_solver",
            role=(
                "Primary domain solver: independently solve the complete question from scratch; evaluate "
                "every original option A-D and preserve exact letter mapping."
            ),
            task=(
                "Produce a complete independent scientific derivation, test all four choices, and end with "
                "Candidate option: X plus the decisive claim."
            ),
            goal=goal,
            is_final=False,
            execution_mode="llm_only",
        ),
        AgentNode(
            id="independent_option_eliminator",
            role=(
                "Independent option eliminator: solve again without relying on another agent; eliminate A-D "
                "one by one and actively seek a counterexample to the leading choice."
            ),
            task=(
                "Independently compare all original choices, audit definitions, units, signs and limiting cases, "
                "and end with Candidate option: X."
            ),
            goal=goal,
            is_final=False,
            execution_mode="llm_only",
        ),
        AgentNode(
            id="evidence_gate_reconciler",
            role=(
                "GPQA EVIDENCE GATE. Compare both independent upstream reports against GLOBAL TASK CONTEXT and "
                "the full original option texts. Confidence, verbosity, and majority are not evidence. If both "
                "roots agree, lock their letter unless you can name a concrete scientific contradiction in that "
                "option. If they disagree, reject one candidate only with explicit evidence such as an equation, "
                "definition, unit/sign check, limiting case, experimentally established mechanism, or exact "
                "qualifier from the question. Resolve the science yourself only as needed to test those competing "
                "claims; never invent a new option mapping. End with exactly `Verified option: X`."
            ),
            task="Conservatively retain agreement or adjudicate disagreement using explicit scientific evidence.",
            goal=goal,
            is_final=False,
            execution_mode="llm_only",
        ),
        AgentNode(
            id="final_option_mapper",
            role=(
                "FINAL GPQA OPTION MAPPER. You are not another solver or verifier. Read only the evidence gate's "
                "upstream output, copy the letter after `Verified option:`, and do not inspect, reinterpret, or "
                "change the scientific conclusion. Output exactly one line and nothing else: `\\boxed{X}`, where "
                "X is A, B, C, or D."
            ),
            task="Copy the evidence gate's verified letter into the strict boxed format.",
            goal=goal,
            is_final=True,
            execution_mode="llm_only",
        ),
    ]
    edges = [
        GraphEdge("primary_domain_solver", "evidence_gate_reconciler", EdgeType.DATA),
        GraphEdge("independent_option_eliminator", "evidence_gate_reconciler", EdgeType.DATA),
        GraphEdge("evidence_gate_reconciler", "final_option_mapper", EdgeType.DATA),
    ]
    return MASGraph(nodes=nodes, edges=edges, topology=Topology.DAG)


def _normalize_dataset_orchestration(
    graph: MASGraph, task_text: str, dataset_name: str
) -> MASGraph:
    ds = (dataset_name or "").strip().lower()
    if ds in {"gpqa", "gpqa_diamond"}:
        return _canonical_gpqa_graph(task_text, graph)
    return graph


def _sanitize_graph(graph: MASGraph, dataset_name: str = "aime") -> MASGraph:
    if not graph.nodes:
        raise ValueError("rollout XML produced an empty MAS graph (no valid agent nodes)")
    used: set[str] = set()
    id_map: dict[str, str] = {}
    for i, n in enumerate(graph.nodes):
        id_map[n.id] = _safe_id(n.id, used, i)
    for n in graph.nodes:
        n.id = id_map.get(n.id, n.id)
    # Drop edges that do not reference real nodes. Rebuild a sequential chain if
    # all emitted edges were invalid, but otherwise preserve the trained policy's
    # decomposition and topology.
    valid_ids = {n.id for n in graph.nodes}
    clean_edges = []
    seen_edges: set[tuple[str, str, str]] = set()
    for e in graph.edges:
        e.f = id_map.get(e.f, e.f)
        e.t = id_map.get(e.t, e.t)
        if e.f not in valid_ids or e.t not in valid_ids or e.f == e.t:
            continue
        key = (e.f, e.t, str(getattr(e.type, "value", e.type)))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        clean_edges.append(e)
    if not clean_edges and len(graph.nodes) > 1:
        clean_edges = [
            GraphEdge(f=graph.nodes[i].id, t=graph.nodes[i + 1].id, type=EdgeType.DATA)
            for i in range(len(graph.nodes) - 1)
        ]
        graph.topology = Topology.DAG
    graph.edges = clean_edges
    if graph.nodes and not any(n.is_final for n in graph.nodes):
        graph.nodes[-1].is_final = True

    ds = (dataset_name or "").strip().lower()
    if ds in _SEARCH_CAPABLE_DATASETS:
        for n in graph.nodes:
            mode = _normalized_execution_mode(n.execution_mode)
            if mode is not None:
                n.execution_mode = mode
            elif not n.is_final and _looks_like_search_node(role=n.role, task=n.task, goal=n.goal):
                n.execution_mode = "multi_turn_search"
            else:
                n.execution_mode = "llm_only"
        if not any((n.execution_mode or "").lower() == "multi_turn_search" for n in graph.nodes):
            target = next(
                (
                    n for n in graph.nodes
                    if not n.is_final and _looks_like_search_node(role=n.role, task=n.task, goal=n.goal)
                ),
                None,
            )
            if target is None:
                target = next((n for n in graph.nodes if not n.is_final), graph.nodes[0])
            if target is not None:
                target.execution_mode = "multi_turn_search"
    else:
        for n in graph.nodes:
            n.execution_mode = "llm_only"
    return graph


ORCHESTRA_XML_SYSTEM_PROMPT = """You are MAS-Gate's orchestra policy. Design a multi-agent system, not the final answer.
Return XML only. Do not solve the problem directly.
"""


def build_orchestra_user_prompt(question: str, dataset_name: str = "aime") -> str:
    ds = (dataset_name or "aime").strip().lower()
    if ds in _BCP_DATASETS:
        dataset_constraints = (
            "BrowseComp-Plus routing constraints: set execution_mode explicitly for every agent. "
            "Use multi_turn_search only when that agent must acquire NEW facts/documents from the local BM25 corpus. "
            "Use llm_only when the agent only reasons over GLOBAL TASK CONTEXT and upstream evidence, including constraint reconciliation, evidence comparison, verification without new facts, final synthesis, and exact-output formatting. "
            "Do not make every agent a retrieval agent. Include at least one multi_turn_search evidence-acquisition agent. "
            "The final agent should normally be llm_only, synthesize only from retrieved upstream snippets, and output the exact short answer string, with no explanation, no bullets, and no XML. "
            "Do not solve from prior knowledge."
        )
    elif ds in _HOTPOTQA_DATASETS:
        dataset_constraints = (
            "HotpotQA routing constraints: set execution_mode explicitly for every agent. "
            "Use at least one multi_turn_search agent to acquire supporting evidence from the local BM25 corpus. "
            "Decompose the multi-hop question so retrieval agents resolve the linked entities or facts needed for the answer; "
            "use llm_only for reasoning over the task and retrieved upstream evidence. Do not make every agent a retrieval agent. "
            "The final agent must be llm_only, reconcile the supporting evidence, and output only the exact short answer "
            "with no explanation, sentence prefix, bullets, citations, or XML. Do not answer from unsupported prior knowledge."
        )
    elif ds in {"gpqa", "gpqa_diamond"}:
        dataset_constraints = (
            r"GPQA protocol: propose exactly four roles: two independent root solvers with no dependencies; "
            r"an evidence gate depending on both roots; and a final option mapper depending only on the gate. "
            r"The gate must lock root agreement unless a concrete scientific contradiction is identified, and "
            r"must resolve disagreement using equations, definitions, units/signs, limiting cases, mechanisms, "
            r"or exact qualifiers rather than confidence or majority. Preserve the exact A/B/C/D mapping. "
            r"The mapper must not re-solve and must output exactly one uppercase letter as \boxed{A}, "
            r"\boxed{B}, \boxed{C}, or \boxed{D}."
        )
    elif ds in {"aime", "hlemath", "hle"}:
        dataset_constraints = (
            r"Math constraints (AIME): preserve exact arithmetic and the requested answer quantity. "
            r"Use problem-specific roles for the actual algebra, geometry, number theory, or combinatorics steps. "
            r"The final non-empty line must be exactly one \boxed{integer} from 0 through 999."
        )
    else:
        dataset_constraints = r"Math constraints: require the final agent to output the final answer in \boxed{...}."
    return f"""Dataset: {dataset_name}
Task:
{question}

Output exactly one XML document with this shape:
<editor_output>
  <task_decomposition>
    <goal>overall objective</goal>
    <sub_tasks>
      <sub_task idx="0">...</sub_task>
      <sub_task idx="1">...</sub_task>
    </sub_tasks>
  </task_decomposition>
  <agent_specs>
    <agents>
      <agent id="agent_0" is_final="false" execution_mode="multi_turn_search">
        <role>specialized role and instructions</role>
        <sub_task_idx>0</sub_task_idx>
        <dependencies></dependencies>
      </agent>
      <agent id="agent_1" is_final="true" execution_mode="llm_only">
        <role>final synthesizer/verifier role and required final-answer format</role>
        <sub_task_idx>1</sub_task_idx>
        <dependencies>agent_0</dependencies>
      </agent>
    </agents>
  </agent_specs>
</editor_output>

Constraints: use 2-6 agents, include exactly one final agent, set execution_mode on every agent, and dependencies must reference earlier agent ids. {dataset_constraints}
"""
