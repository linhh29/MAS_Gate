"""Structural analysis of a reasoning dependency graph."""

from __future__ import annotations

from collections import defaultdict, deque

from quantification.schemas import DependencyEdge, ReasoningGraph


class GraphStructure:
    """Cached graph topology: children, parents, depth, descendants."""

    def __init__(self, graph: ReasoningGraph) -> None:
        self.graph = graph
        self._node_ids = graph.node_ids()
        self._children: dict[str, list[str]] = {nid: [] for nid in self._node_ids}
        self._parents: dict[str, list[str]] = {nid: [] for nid in self._node_ids}

        for edge in graph.edges:
            if edge.from_id in self._children and edge.to_id in self._children:
                self._children[edge.from_id].append(edge.to_id)
                self._parents[edge.to_id].append(edge.from_id)

        for nid in self._node_ids:
            self._children[nid].sort()
            self._parents[nid].sort()

        self._depth = self._compute_depths()
        self._descendants = self._compute_descendants()

    @property
    def node_ids(self) -> list[str]:
        return list(self._node_ids)

    def children(self, node_id: str) -> list[str]:
        return list(self._children.get(node_id, []))

    def parents(self, node_id: str) -> list[str]:
        return list(self._parents.get(node_id, []))

    def depth(self, node_id: str) -> int:
        return self._depth.get(node_id, 1)

    def descendants(self, node_id: str) -> set[str]:
        return set(self._descendants.get(node_id, set()))

    def distance(self, from_id: str, to_id: str) -> int | None:
        """Shortest path distance from from_id to to_id; None if unreachable."""
        if from_id == to_id:
            return 0
        visited = {from_id}
        queue: deque[tuple[str, int]] = deque([(from_id, 0)])
        while queue:
            current, dist = queue.popleft()
            for child in self._children.get(current, []):
                if child == to_id:
                    return dist + 1
                if child not in visited:
                    visited.add(child)
                    queue.append((child, dist + 1))
        return None

    def topological_order(self) -> list[str]:
        """Stable Kahn topological sort; falls back to node list order on cycles."""
        in_degree = {nid: len(self._parents[nid]) for nid in self._node_ids}
        queue = deque(sorted([nid for nid, d in in_degree.items() if d == 0]))
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for child in self._children[nid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._node_ids):
            remaining = [nid for nid in self._node_ids if nid not in order]
            order.extend(sorted(remaining))
        return order

    def execution_order(self) -> list[str]:
        if self.graph.execution_order:
            known = set(self._node_ids)
            order = [nid for nid in self.graph.execution_order if nid in known]
            for nid in self._node_ids:
                if nid not in order:
                    order.append(nid)
            return order
        return self.topological_order()

    def _compute_depths(self) -> dict[str, int]:
        """Layer L(v): longest path from any source to v along execution order."""
        order = self.execution_order()
        order_index = {nid: i for i, nid in enumerate(order)}
        depths: dict[str, int] = {}
        for nid in order:
            parent_depths = [
                depths[p]
                for p in self._parents[nid]
                if p in depths and order_index.get(p, -1) < order_index.get(nid, len(order))
            ]
            depths[nid] = (max(parent_depths) + 1) if parent_depths else 1
        return depths

    def _compute_descendants(self) -> dict[str, set[str]]:
        desc: dict[str, set[str]] = {nid: set() for nid in self._node_ids}
        for nid in reversed(self.topological_order()):
            for child in self._children[nid]:
                desc[nid].add(child)
                desc[nid].update(desc[child])
        return desc

    def _compute_ancestors(self) -> dict[str, set[str]]:
        """Ancestors(v): all nodes that can reach v via directed paths."""
        anc: dict[str, set[str]] = {nid: set() for nid in self._node_ids}
        for nid in self.topological_order():
            for child in self._children[nid]:
                anc[child].add(nid)
                anc[child].update(anc[nid])
        return anc

    def ancestors(self, node_id: str) -> set[str]:
        """Return all ancestor nodes that can reach node_id."""
        if not hasattr(self, "_ancestors"):
            self._ancestors = self._compute_ancestors()
        return set(self._ancestors.get(node_id, set()))

    def reachable_decay_sum(self, node_id: str, gamma: float) -> float:
        """
        T_j = sum_{k in Desc(j)} gamma^{d(j,k)-1}
        Desc(j) is strict descendants only (excludes j itself).
        """
        total = 0.0
        for desc_id in self.descendants(node_id):
            dist = self.distance(node_id, desc_id)
            if dist is not None and dist >= 1:
                total += gamma ** (dist - 1)
        return total

    def max_reachable_decay_at_layer(self, layer: int, gamma: float) -> float:
        layer_nodes = [nid for nid in self._node_ids if self.depth(nid) == layer]
        if not layer_nodes:
            return 0.0
        return max(self.reachable_decay_sum(nid, gamma) for nid in layer_nodes)
