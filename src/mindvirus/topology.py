from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx


@dataclass(frozen=True, slots=True)
class Topology:
    graph: nx.Graph
    origin_id: str
    bridge_id: str | None

    def neighbors(self, agent_id: str) -> list[str]:
        return sorted(str(item) for item in self.graph.neighbors(agent_id))

    def can_message(self, sender: str, recipient: str) -> bool:
        return bool(self.graph.has_edge(sender, recipient))

    def distance_from_origin(self, agent_id: str) -> int | None:
        try:
            return int(nx.shortest_path_length(self.graph, self.origin_id, agent_id))
        except nx.NetworkXNoPath:
            return None

    def direct_origin_recipients(self) -> set[str]:
        return set(self.neighbors(self.origin_id))

    def edge_list(self) -> list[tuple[str, str]]:
        return sorted((str(left), str(right)) for left, right in self.graph.edges())


def build_topology(kind: str, agent_ids: list[str], rng: random.Random) -> Topology:
    if len(agent_ids) < 3:
        raise ValueError("a swarm requires at least three agents")

    ids = list(agent_ids)
    origin = rng.choice(ids)
    clean_ids = [agent_id for agent_id in ids if agent_id != origin]
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    bridge: str | None = None

    if kind == "fully_connected":
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1 :]:
                graph.add_edge(left, right)
    elif kind == "bridge":
        bridge = rng.choice(clean_ids)
        graph.add_edge(origin, bridge)
        for clean_id in clean_ids:
            if clean_id != bridge:
                graph.add_edge(bridge, clean_id)
        for left_index, left in enumerate(clean_ids):
            for right in clean_ids[left_index + 1 :]:
                graph.add_edge(left, right)
    elif kind == "chain":
        ordered = [origin, *rng.sample(clean_ids, len(clean_ids))]
        graph.add_edges_from(pairwise(ordered))
        bridge = ordered[1]
    elif kind == "small_world":
        ordered = rng.sample(ids, len(ids))
        degree = min(4, len(ids) - 1)
        if degree % 2:
            degree -= 1
        base = nx.watts_strogatz_graph(
            len(ids), k=max(2, degree), p=0.15, seed=rng.randrange(2**31)
        )
        mapping = {index: agent_id for index, agent_id in enumerate(ordered)}
        graph = nx.relabel_nodes(base, mapping)
        origin = ordered[0]
        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            for left, right in pairwise(components):
                graph.add_edge(sorted(left)[0], sorted(right)[0])
    else:
        raise ValueError(f"unknown topology {kind!r}")

    return Topology(graph=graph, origin_id=origin, bridge_id=bridge)
