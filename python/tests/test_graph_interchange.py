"""DOT/JSON/GML/NetworkX graph interchange smoke.

Every ``from_*`` classmethod must recover the variable names a matching
``to_*`` call wrote into the document — losing them silently renames nodes
to dense-index strings (``"0"``, ``"1"``, ...) and breaks any code that
later resolves nodes by name (e.g. restoring an ``AcceptedGraph``).
"""

import antecedent
from antecedent.graph import Admg, Cpdag, Dag, Pag


def test_dot_round_trip():
    dag = Dag.from_dot("digraph { 0 -> 1; 1 -> 2; }")
    assert dag.node_count() == 3
    assert set(dag.edges()) == {("0", "1"), ("1", "2")}
    # Nameless numeric input has no distinct name information in the
    # document, so dense-index strings are the correct (unchanged) fallback.
    assert list(dag.nodes()) == ["0", "1", "2"]

    dot = dag.to_dot()
    dag2 = Dag.from_dot(dot)
    assert dag2.node_count() == dag.node_count()
    assert set(dag2.edges()) == set(dag.edges())
    assert list(dag2.nodes()) == ["0", "1", "2"]


def test_json_round_trip():
    dag = Dag.from_edges(["x", "y"], [("x", "y")])
    js = dag.to_json()
    dag2 = Dag.from_json(js)
    assert dag2.node_count() == 2
    assert set(dag2.edges()) == {("x", "y")}
    assert list(dag2.nodes()) == ["x", "y"]


def test_networkx_adjacency_round_trip():
    dag = Dag.from_edges(["a", "b"], [("a", "b")])
    js = dag.to_networkx_adjacency()
    dag2 = Dag.from_networkx_adjacency(js)
    assert dag2.node_count() == 2
    assert set(dag2.edges()) == {("a", "b")}
    assert list(dag2.nodes()) == ["a", "b"]


def test_dag_all_formats_preserve_names_and_edges():
    """``Dag`` round-trips names + edges through every codec it has."""
    dag = Dag.from_edges(["x", "y"], [("x", "y")])
    expected_nodes = ["x", "y"]
    expected_edges = {("x", "y")}

    back = Dag.from_dot(dag.to_dot())
    assert list(back.nodes()) == expected_nodes
    assert set(back.edges()) == expected_edges

    back = Dag.from_json(dag.to_json())
    assert list(back.nodes()) == expected_nodes
    assert set(back.edges()) == expected_edges

    back = Dag.from_gml(dag.to_gml())
    assert list(back.nodes()) == expected_nodes
    assert set(back.edges()) == expected_edges

    back = Dag.from_networkx_node_link(dag.to_networkx_node_link())
    assert list(back.nodes()) == expected_nodes
    assert set(back.edges()) == expected_edges

    back = Dag.from_networkx_adjacency(dag.to_networkx_adjacency())
    assert list(back.nodes()) == expected_nodes
    assert set(back.edges()) == expected_edges


def test_cpdag_pag_admg_oo_codecs():
    cpdag = Cpdag.from_directed_undirected(["a", "b", "c"], [("a", "b")], [("b", "c")])
    assert "->" in cpdag.to_dot()
    assert "--" in cpdag.to_dot()
    back = Cpdag.from_json(cpdag.to_json())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["a", "b", "c"]
    back = Cpdag.from_gml(cpdag.to_gml())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["a", "b", "c"]
    back = Cpdag.from_networkx_node_link(cpdag.to_networkx_node_link())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["a", "b", "c"]
    back = Cpdag.from_dot(cpdag.to_dot())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["a", "b", "c"]

    pag = Pag.from_marked_edges(["x", "y"], [("x", "y", "circle", "arrow")])
    assert "mark_a" in pag.to_dot()
    back = Pag.from_dot(pag.to_dot())
    assert back.node_count() == 2
    assert list(back.nodes()) == ["x", "y"]
    back = Pag.from_gml(pag.to_gml())
    assert back.node_count() == 2
    assert list(back.nodes()) == ["x", "y"]
    back = Pag.from_networkx_node_link(pag.to_networkx_node_link())
    assert back.node_count() == 2
    assert list(back.nodes()) == ["x", "y"]
    back = Pag.from_json(pag.to_json())
    assert back.node_count() == 2
    assert list(back.nodes()) == ["x", "y"]

    admg = Admg.from_edges(["z", "t", "y"], [("z", "t"), ("t", "y")], [("z", "y")])
    assert "dir=both" in admg.to_dot()
    back = Admg.from_json(admg.to_json())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["z", "t", "y"]
    back = Admg.from_gml(admg.to_gml())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["z", "t", "y"]
    back = Admg.from_networkx_node_link(admg.to_networkx_node_link())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["z", "t", "y"]
    back = Admg.from_dot(admg.to_dot())
    assert back.node_count() == 3
    assert list(back.nodes()) == ["z", "t", "y"]


def test_accepted_graph_cpdag_json_round_trip_preserves_names():
    """Reproduces the production bug: a restored ``AcceptedGraph`` must keep
    the data's column names, not silently rename nodes to dense indices.
    """
    cpdag = Cpdag.from_directed_undirected(["z", "t", "y"], [("z", "t")], [("t", "y")])
    accepted = antecedent.AcceptedGraph.from_graph(cpdag)

    restored = antecedent.AcceptedGraph.from_json(accepted.to_json())

    assert list(restored.graph.nodes()) == list(accepted.graph.nodes())
    assert set(restored.graph.edges()) == set(accepted.graph.edges())
    assert list(restored.graph.nodes()) == ["z", "t", "y"]
