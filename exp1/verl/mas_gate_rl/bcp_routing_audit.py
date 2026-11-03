#!/usr/bin/env python3
"""Strict runtime routing checks for BrowseComp-Plus MAS traces."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

VALID_EXECUTION_MODES = frozenset({"llm_only", "multi_turn_search"})
_ID_KEYS = ("agent_name", "id", "name", "role_name")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _aliases(row: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key in _ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            aliases.add(value.strip())
    return aliases


def _mode_value(sub_agent: dict[str, Any], graph_node: dict[str, Any]) -> str:
    # Preserve explicit invalid values so the auditor fails rather than silently
    # replacing a malformed runtime route with a fallback.
    candidates = (
        sub_agent.get("execution_mode"),
        _mapping(sub_agent.get("tool_context")).get("execution_mode"),
        graph_node.get("execution_mode"),
        _mapping(graph_node.get("tool_context")).get("execution_mode"),
    )
    for value in candidates:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _is_final(sub_agent: dict[str, Any], graph_node: dict[str, Any]) -> bool:
    if "is_final" in sub_agent:
        return bool(sub_agent["is_final"])
    return bool(graph_node.get("is_final"))


def _agent_rows(trace: dict[str, Any], label: str) -> list[dict[str, Any]]:
    graph = _mapping(trace.get("graph"))
    graph_nodes = _rows(graph.get("nodes"))
    sub_agents = _rows(trace.get("sub_agents"))
    if not sub_agents:
        sub_agents = _rows(_mapping(trace.get("orchestra")).get("sub_agents"))

    source_nodes = graph_nodes or sub_agents
    if not source_nodes:
        raise RuntimeError(f"{label}: trace has no agents")

    sub_alias_map: dict[str, dict[str, Any]] = {}
    for sub_agent in sub_agents:
        aliases = _aliases(sub_agent)
        if not aliases:
            raise RuntimeError(f"{label}: runtime sub-agent has no stable identifier")
        for alias in aliases:
            prior = sub_alias_map.get(alias)
            if prior is not None and prior is not sub_agent:
                raise RuntimeError(f"{label}: duplicate runtime agent alias {alias!r}")
            sub_alias_map[alias] = sub_agent

    agents: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()
    matched_sub_ids: set[int] = set()
    for ordinal, graph_node in enumerate(source_nodes):
        aliases = _aliases(graph_node)
        if not aliases:
            raise RuntimeError(f"{label}: graph agent #{ordinal} has no stable identifier")
        matches = {id(sub_alias_map[alias]): sub_alias_map[alias] for alias in aliases if alias in sub_alias_map}
        if len(matches) > 1:
            raise RuntimeError(f"{label}: graph agent aliases map to multiple runtime agents: {sorted(aliases)}")
        sub_agent = next(iter(matches.values()), {})
        if sub_agent:
            matched_sub_ids.add(id(sub_agent))
        canonical = next(
            (str(graph_node[key]).strip() for key in _ID_KEYS if isinstance(graph_node.get(key), str) and str(graph_node[key]).strip()),
            next(iter(sorted(aliases))),
        )
        if canonical in seen_canonical:
            raise RuntimeError(f"{label}: duplicate graph agent id {canonical!r}")
        seen_canonical.add(canonical)
        merged_aliases = aliases | _aliases(sub_agent)
        mode = _mode_value(sub_agent, graph_node)
        if mode not in VALID_EXECUTION_MODES:
            raise RuntimeError(f"{label}: agent {canonical!r} has invalid execution_mode {mode!r}")
        agents.append(
            {
                "id": canonical,
                "aliases": merged_aliases,
                "execution_mode": mode,
                "is_final": _is_final(sub_agent, graph_node),
            }
        )

    if graph_nodes:
        unmatched = [sorted(_aliases(row)) for row in sub_agents if id(row) not in matched_sub_ids]
        if unmatched:
            raise RuntimeError(f"{label}: runtime sub-agents missing from graph: {unmatched}")
    return agents


def audit_graph_routing(graph: dict[str, Any], *, label: str = "graph") -> dict[str, Any]:
    """Validate the routing contract that can be proven before MAS execution."""
    agents = _agent_rows({"graph": graph}, label)
    search_agents = [row for row in agents if row["execution_mode"] == "multi_turn_search"]
    llm_agents = [row for row in agents if row["execution_mode"] == "llm_only"]
    finals = [row for row in agents if row["is_final"]]
    if not search_agents:
        raise RuntimeError(f"{label}: BCP graph has no multi_turn_search agent")
    if not finals:
        raise RuntimeError(f"{label}: BCP graph has no final agent")
    final_search = [row["id"] for row in finals if row["execution_mode"] != "llm_only"]
    if final_search:
        raise RuntimeError(f"{label}: final agents must all be llm_only, got search finals={final_search}")
    if not llm_agents:
        raise RuntimeError(f"{label}: BCP graph has no llm_only agent")
    return {
        "agents": agents,
        "search_agents": search_agents,
        "llm_agents": llm_agents,
        "finals": finals,
    }


def audit_trace_routing(trace: dict[str, Any], *, max_rounds: int, label: str = "trace") -> dict[str, Any]:
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    graph_result = audit_graph_routing(_mapping(trace.get("graph")), label=label)
    agents = graph_result["agents"]
    search_agents = graph_result["search_agents"]
    llm_agents = graph_result["llm_agents"]
    finals = graph_result["finals"]

    alias_to_agent: dict[str, dict[str, Any]] = {}
    for agent in agents:
        for alias in agent["aliases"]:
            prior = alias_to_agent.get(alias)
            if prior is not None and prior is not agent:
                raise RuntimeError(f"{label}: ambiguous agent alias {alias!r}")
            alias_to_agent[alias] = agent

    session_counts: Counter[str] = Counter()
    max_observed_rounds = 0
    sessions = _rows(trace.get("retrieval_sessions"))
    for ordinal, session in enumerate(sessions):
        role_name = session.get("role_name")
        if not isinstance(role_name, str) or not role_name.strip():
            raise RuntimeError(f"{label}: retrieval session #{ordinal} has no role_name")
        role_name = role_name.strip()
        agent = alias_to_agent.get(role_name)
        if agent is None:
            raise RuntimeError(f"{label}: retrieval session role_name {role_name!r} does not map to an agent")
        if agent["execution_mode"] != "multi_turn_search":
            raise RuntimeError(f"{label}: llm_only agent {agent['id']!r} owns a retrieval session")
        trajectory = session.get("trajectory")
        if not isinstance(trajectory, list):
            raise RuntimeError(f"{label}: retrieval session for {role_name!r} has no trajectory list")
        rounds = len(trajectory)
        max_observed_rounds = max(max_observed_rounds, rounds)
        if rounds > max_rounds:
            raise RuntimeError(
                f"{label}: retrieval session for {role_name!r} has {rounds} rounds, cap is {max_rounds}"
            )
        session_counts[agent["id"]] += 1

    bad_search_counts = {
        agent["id"]: session_counts[agent["id"]]
        for agent in search_agents
        if session_counts[agent["id"]] != 1
    }
    if bad_search_counts:
        raise RuntimeError(
            f"{label}: each multi_turn_search agent must own exactly one retrieval session: {bad_search_counts}"
        )

    return {
        "agent_count": len(agents),
        "search_agent_count": len(search_agents),
        "llm_only_agent_count": len(llm_agents),
        "final_agent_count": len(finals),
        "final_llm_only_count": len(finals),
        "retrieval_session_count": len(sessions),
        "max_observed_retrieval_rounds": max_observed_rounds,
        "mixed": True,
    }


def aggregate_routing(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    cases = len(values)
    if cases <= 0:
        raise RuntimeError("cannot aggregate zero routing cases")
    totals = {
        "cases": cases,
        "agents": sum(int(row["agent_count"]) for row in values),
        "search_agents": sum(int(row["search_agent_count"]) for row in values),
        "llm_only_agents": sum(int(row["llm_only_agent_count"]) for row in values),
        "final_agents": sum(int(row["final_agent_count"]) for row in values),
        "final_llm_only_agents": sum(int(row["final_llm_only_count"]) for row in values),
        "retrieval_sessions": sum(int(row["retrieval_session_count"]) for row in values),
        "max_observed_retrieval_rounds": max(int(row["max_observed_retrieval_rounds"]) for row in values),
        "mixed_cases": sum(bool(row["mixed"]) for row in values),
        "all_search_cases": 0,
        "all_llm_cases": 0,
        "final_search_agents": 0,
        "llm_only_retrieval_session_count": 0,
    }
    totals["mixed_case_ratio"] = totals["mixed_cases"] / cases
    totals["final_llm_ratio"] = totals["final_llm_only_agents"] / totals["final_agents"]
    return totals
