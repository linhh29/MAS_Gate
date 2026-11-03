from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import re


class Topology(str, Enum):
    SEQUENTIAL = "sequential"
    DAG = "dag"


class EdgeType(str, Enum):
    DATA = "data"
    CONTROL = "control_flow"
    FEEDBACK = "feedback"
    VERIFY = "verify"


@dataclass
class SubTask:
    idx: int
    desc: str
    dependencies: list[str] = field(default_factory=list)


@dataclass
class TaskDecomposition:
    goal: str
    sub_tasks: list[SubTask] = field(default_factory=list)


@dataclass
class AgentSpec:
    id: str
    role: str
    sub_task_idx: int = 0
    dependencies: list[str] = field(default_factory=list)
    is_final: bool = False
    constraints: list[str] = field(default_factory=list)
    # ``None`` means legacy policy output omitted the field.  Keeping that
    # distinction lets BCP infer a safe mixed route without overriding an
    # explicit llm_only choice made by a newer policy.
    execution_mode: Optional[str] = None


@dataclass
class AgentSpecs:
    specs: list[AgentSpec] = field(default_factory=list)


@dataclass
class AgentNode:
    id: str
    role: str
    task: str = ""
    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    is_final: bool = False
    execution_mode: str = "llm_only"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "role": self.role, "task": self.task, "goal": self.goal,
            "constraints": self.constraints, "is_final": self.is_final,
            "execution_mode": self.execution_mode, "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentNode:
        return cls(
            id=d.get("id", ""), role=d.get("role", ""), task=d.get("task", ""),
            goal=d.get("goal", ""), constraints=d.get("constraints", []),
            is_final=d.get("is_final", False),
            execution_mode=d.get("execution_mode", "llm_only"), meta=d.get("meta", {}),
        )

    def to_xml(self) -> str:
        cs = "".join(f"<constraint>{c}</constraint>" for c in self.constraints)
        return f"<agent id=\"{self.id}\" is_final=\"{str(self.is_final).lower()}\" execution_mode=\"{self.execution_mode}\"><role>{self.role}</role><task>{self.task}</task><goal>{self.goal}</goal><constraints>{cs}</constraints></agent>"

    @classmethod
    def from_xml(cls, elem: dict) -> AgentNode:
        def _text(v):
            if isinstance(v, dict) and "#text" in v:
                return v["#text"]
            return str(v) if v else ""

        constraints_val = elem.get("constraints", {})
        if isinstance(constraints_val, dict):
            raw = constraints_val.get("constraint", [])
            if isinstance(raw, dict):
                raw = [raw]
            constraints = [_text(c) for c in raw]
        elif isinstance(constraints_val, list):
            constraints = constraints_val
        else:
            constraints = []

        return cls(
            id=elem.get("@id", ""),
            role=_text(elem.get("role", "")),
            task=_text(elem.get("task", "")),
            goal=_text(elem.get("goal", "")),
            constraints=constraints,
            is_final=elem.get("@is_final", "false") == "true",
            execution_mode=_text(
                elem.get("@execution_mode", elem.get("execution_mode", "llm_only"))
            ).strip().lower()
            or "llm_only",
        )


@dataclass
class GraphEdge:
    f: str
    t: str
    type: EdgeType = EdgeType.DATA

    def to_dict(self) -> dict[str, Any]:
        return {"f": self.f, "t": self.t, "type": self.type.value}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphEdge:
        return cls(f=d.get("f", ""), t=d.get("t", ""), type=EdgeType(d.get("type", "data")))

    def to_xml(self) -> str:
        return f"<edge f=\"{self.f}\" t=\"{self.t}\" type=\"{self.type.value}\"/>"

    @classmethod
    def from_xml(cls, elem: dict) -> GraphEdge:
        return cls(f=elem.get("@f", ""), t=elem.get("@t", ""), type=EdgeType(elem.get("@type", "data")))


@dataclass
class MASGraph:
    nodes: list[AgentNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    topology: Topology = Topology.SEQUENTIAL

    def node_by_id(self, nid: str) -> Optional[AgentNode]:
        for n in self.nodes:
            if n.id == nid:
                return n
        return None

    def final_node(self) -> Optional[AgentNode]:
        for n in self.nodes:
            if n.is_final:
                return n
        return self.nodes[-1] if self.nodes else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "topology": self.topology.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MASGraph:
        return cls(
            nodes=[AgentNode.from_dict(n) for n in d.get("nodes", [])],
            edges=[GraphEdge.from_dict(e) for e in d.get("edges", [])],
            topology=Topology(d.get("topology", "sequential")),
        )

    def to_xml(self) -> str:
        ns = "".join(n.to_xml() for n in self.nodes)
        es = "".join(e.to_xml() for e in self.edges)
        return f"<mas_graph topology=\"{self.topology.value}\"><nodes>{ns}</nodes><edges>{es}</edges></mas_graph>"

    @classmethod
    def from_xml(cls, elem: dict) -> MASGraph:
        nodes = []
        if isinstance(elem.get("nodes"), dict):
            nlist = elem["nodes"].get("agent", [])
            if isinstance(nlist, dict):
                nlist = [nlist]
            nodes = [AgentNode.from_xml(n) for n in nlist]
        edges = []
        if isinstance(elem.get("edges"), dict):
            elist = elem["edges"].get("edge", [])
            if isinstance(elist, dict):
                elist = [elist]
            edges = [GraphEdge.from_xml(e) for e in elist]
        return cls(
            nodes=nodes,
            edges=edges,
            topology=Topology(elem.get("@topology", "sequential")),
        )


def extract_xml(text: str) -> dict[str, Any]:
    """Extract and parse XML from LLM response using xml.etree.ElementTree."""
    import xml.etree.ElementTree as ET

    # 以下为安全注释COSEC：policy XML is model-generated/untrusted; reject DTD
    # and entity declarations before ElementTree parsing so external/internal
    # entity expansion cannot become part of this server-side parse surface.
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text or "", flags=re.IGNORECASE):
        return {}

    m = re.search(
        r"<(editor_output|mas_graph|task_decomposition|agent_specs)[^>]*>[\s\S]*?</\1>",
        text,
        re.DOTALL,
    )
    if not m:
        return {}
    try:
        root = ET.fromstring(m.group(0))
        return {root.tag: _etree_to_dict(root)}
    except ET.ParseError:
        return {}


def _etree_to_dict(element) -> dict[str, Any]:
    """Convert an ElementTree element to a dict compatible with our from_xml methods."""
    result: dict[str, Any] = {}
    for k, v in element.attrib.items():
        result[f"@{k}"] = v
    text = (element.text or "").strip()
    if text and not len(element):
        result["#text"] = text
    for child in element:
        ctag = child.tag
        child_dict = _etree_to_dict(child)
        if ctag in result:
            if not isinstance(result[ctag], list):
                result[ctag] = [result[ctag]]
            result[ctag].append(child_dict)
        else:
            result[ctag] = child_dict
    return result
