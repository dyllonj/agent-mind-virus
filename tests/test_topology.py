import random

from mindvirus.topology import build_topology


def test_bridge_topology_enforces_multihop_origin_reach() -> None:
    agent_ids = [f"agent_{index}" for index in range(6)]
    topology = build_topology("bridge", agent_ids, random.Random(17))

    assert len(topology.neighbors(topology.origin_id)) == 1
    assert topology.bridge_id in topology.neighbors(topology.origin_id)
    distances = {agent_id: topology.distance_from_origin(agent_id) for agent_id in agent_ids}
    assert sorted(distances.values()) == [0, 1, 2, 2, 2, 2]
    assert all(
        not topology.can_message(topology.origin_id, agent_id)
        for agent_id, distance in distances.items()
        if distance == 2
    )


def test_fully_connected_topology_has_no_primary_eligible_agent() -> None:
    agent_ids = [f"agent_{index}" for index in range(6)]
    topology = build_topology("fully_connected", agent_ids, random.Random(19))
    clean_distances = [
        topology.distance_from_origin(agent_id)
        for agent_id in agent_ids
        if agent_id != topology.origin_id
    ]
    assert clean_distances == [1, 1, 1, 1, 1]
