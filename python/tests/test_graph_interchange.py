"""DOT/JSON/NetworkX graph interchange smoke."""

from causal.graph import (
    dag_from_dot,
    dag_from_json,
    dag_from_networkx_adjacency,
    dag_to_dot,
    dag_to_json,
    dag_to_networkx_adjacency,
)


def test_dot_json_round_trip():
    n, edges = dag_from_dot("digraph { 0 -> 1; 1 -> 2; }")
    assert n == 3
    assert edges == [(0, 1), (1, 2)]
    dot = dag_to_dot(n, edges)
    n2, edges2 = dag_from_dot(dot)
    assert (n2, edges2) == (n, edges)

    js = dag_to_json(2, [(0, 1)], ["x", "y"])
    n3, edges3, names = dag_from_json(js)
    assert n3 == 2
    assert edges3 == [(0, 1)]
    assert names == ["x", "y"]


def test_networkx_adjacency_round_trip():
    js = dag_to_networkx_adjacency(2, [(0, 1)], ["a", "b"])
    n, edges = dag_from_networkx_adjacency(js)
    assert n == 2
    assert edges == [(0, 1)]
