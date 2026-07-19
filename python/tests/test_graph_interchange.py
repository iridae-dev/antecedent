"""DOT/JSON/NetworkX graph interchange smoke."""

from causal.graph import (
    Admg,
    Cpdag,
    Pag,
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


def test_cpdag_pag_admg_oo_codecs():
    cpdag = Cpdag.from_directed_undirected(["a", "b", "c"], [("a", "b")], [("b", "c")])
    assert "->" in cpdag.to_dot()
    assert "--" in cpdag.to_dot()
    back = Cpdag.from_json(cpdag.to_json())
    assert back.node_count() == 3
    assert Cpdag.from_gml(cpdag.to_gml()).node_count() == 3
    assert Cpdag.from_networkx_node_link(cpdag.to_networkx_node_link()).node_count() == 3

    pag = Pag.from_marked_edges(["x", "y"], [("x", "y", "circle", "arrow")])
    assert "mark_a" in pag.to_dot()
    assert Pag.from_dot(pag.to_dot()).node_count() == 2
    assert Pag.from_gml(pag.to_gml()).node_count() == 2
    assert Pag.from_networkx_node_link(pag.to_networkx_node_link()).node_count() == 2

    admg = Admg.from_edges(["z", "t", "y"], [("z", "t"), ("t", "y")], [("z", "y")])
    assert "dir=both" in admg.to_dot()
    assert Admg.from_json(admg.to_json()).node_count() == 3
    assert Admg.from_gml(admg.to_gml()).node_count() == 3
    assert Admg.from_networkx_node_link(admg.to_networkx_node_link()).node_count() == 3
