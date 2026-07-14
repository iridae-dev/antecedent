"""DOT/JSON graph interchange smoke."""

from causal.graph import format_dag_dot, format_dag_json, parse_dag_dot, parse_dag_json


def test_dot_json_round_trip():
    n, edges = parse_dag_dot("digraph { 0 -> 1; 1 -> 2; }")
    assert n == 3
    assert edges == [(0, 1), (1, 2)]
    dot = format_dag_dot(n, edges)
    n2, edges2 = parse_dag_dot(dot)
    assert (n2, edges2) == (n, edges)

    js = format_dag_json(2, [(0, 1)], ["x", "y"])
    n3, edges3, names = parse_dag_json(js)
    assert n3 == 2
    assert edges3 == [(0, 1)]
    assert names == ["x", "y"]
