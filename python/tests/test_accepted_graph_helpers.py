"""AcceptedGraph helper branches that the session tests do not hit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from antecedent.accepted_graph import (
    AcceptedGraph,
    _decode_graph,
    _plain_pcmci_temporal_dag,
    _result_to_graph,
    _result_to_temporal_graph,
    accept_discovery,
)
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag


class _Edge:
    def __init__(
        self,
        source: str,
        target: str,
        at_source: str = "tail",
        at_target: str = "arrow",
        source_lag: int = 0,
        target_lag: int = 0,
    ) -> None:
        self.source = source
        self.target = target
        self.at_source = at_source
        self.at_target = at_target
        self.source_lag = source_lag
        self.target_lag = target_lag


class _Link:
    def __init__(self, source: str, target: str, source_lag: int, target_lag: int) -> None:
        self.source = source
        self.target = target
        self.source_lag = source_lag
        self.target_lag = target_lag


def _result(*, edges=(), links=(), nodes=None):
    """A discovery-result double; ``nodes`` are the schema's variable names."""
    if nodes is None:
        nodes = []
        for item in [*edges, *links]:
            for n in (item.source, item.target):
                if n not in nodes:
                    nodes.append(n)
    return SimpleNamespace(graph_edges=list(edges), links=list(links), variable_names=nodes)


def test_result_to_static_and_temporal_graphs():
    dag = _result_to_graph(
        _result(edges=[_Edge("z", "t"), _Edge("t", "y")]),
        "pc",
    )
    assert isinstance(dag, Dag)

    cpdag = _result_to_graph(
        _result(edges=[_Edge("z", "t"), _Edge("t", "y", "tail", "tail")]),
        "pc",
    )
    assert isinstance(cpdag, Cpdag)

    reversed_dag = _result_to_graph(_result(edges=[_Edge("t", "y", "arrow", "tail")]), "pc")
    assert isinstance(reversed_dag, Dag)
    assert ("y", "t") in set(reversed_dag.edges())

    pag = _result_to_graph(_result(edges=[_Edge("x", "y", "circle", "arrow")]), "fci")
    assert isinstance(pag, Pag)
    with pytest.raises(CausalValueError, match="cannot hold edge"):
        _result_to_graph(_result(edges=[_Edge("x", "y", "circle", "arrow")]), "pc")

    empty_pcmci = _plain_pcmci_temporal_dag(_result(nodes=["x", "y"]))
    assert isinstance(empty_pcmci, TemporalDag)
    named = _plain_pcmci_temporal_dag(_result(nodes=["p", "q"]))
    assert isinstance(named, TemporalDag)
    linked = _plain_pcmci_temporal_dag(_result(links=[_Link("p", "q", 1, 0)]))
    assert ("p", 1, "q", 0) in set(linked.edges())

    lpcmci = _result_to_temporal_graph(
        _result(edges=[_Edge("p", "q", "circle", "arrow", 1, 0)]),
        "lpcmci",
    )
    assert isinstance(lpcmci, TemporalPag)

    plus_dag = _result_to_temporal_graph(
        _result(edges=[_Edge("p", "q", "tail", "arrow", 1, 0)]),
        "pcmci+",
    )
    assert isinstance(plus_dag, TemporalDag)
    plus_rev = _result_to_temporal_graph(
        _result(edges=[_Edge("p", "q", "arrow", "tail", 0, 1)]),
        "pcmci+",
    )
    assert isinstance(plus_rev, TemporalDag)
    plus_cpdag = _result_to_temporal_graph(
        _result(edges=[_Edge("p", "q", "tail", "tail", 0, 0)]),
        "pcmci+",
    )
    assert isinstance(plus_cpdag, TemporalCpdag)
    with pytest.raises(CausalValueError, match="cannot hold temporal edge"):
        _result_to_temporal_graph(
            _result(edges=[_Edge("p", "q", "circle", "arrow", 1, 0)]),
            "pcmci+",
        )
    assert isinstance(_result_to_temporal_graph(_result(nodes=["p"]), "pcmci"), TemporalDag)


def test_from_discovery_and_session_guards():
    with pytest.raises(CausalValueError, match="version"):
        AcceptedGraph([("a", "b")], version=0)
    with pytest.raises(CausalValueError, match="algorithm_id"):
        AcceptedGraph.from_discovery(_result(nodes=["a"]), algorithm_id="")

    result = _result(edges=[_Edge("z", "t"), _Edge("t", "y")])
    accepted = AcceptedGraph.from_discovery(result, algorithm_id="PC")
    assert accepted.algorithm_id == "pc"
    temporal = AcceptedGraph.from_discovery(
        _result(links=[_Link("p", "q", 1, 0)]),
        algorithm_id="pcmci_plus",
    )
    assert temporal.algorithm_id == "pcmci+"

    class Cfg:
        algorithm_id = "pc"

        def run(self, data, seed=1, threads=1):
            return result

    held = accept_discovery(Cfg(), {"z": [0.0], "t": [1.0], "y": [2.0]})
    assert isinstance(held.graph, Dag)

    dag = AcceptedGraph.from_graph(
        Dag.from_edges(["z", "t", "y"], [("z", "t"), ("t", "y")]),
        algorithm_id="hand",
    )
    with pytest.raises(CausalUnsupportedError, match="rejects discovery"):
        dag.analyze({"z": [0.0], "t": [1.0], "y": [2.0]}, query=object(), discovery=object())
    cpdag = AcceptedGraph.from_graph(
        Cpdag.from_directed_undirected(["t", "y"], [], [("t", "y")]),
        algorithm_id="pc",
    )
    with pytest.raises(CausalTypeError, match="unsupported query type"):
        cpdag.prepare({"t": [0.0], "y": [1.0]}, query=object())
    with pytest.raises(CausalValueError, match="unsupported AcceptedGraph format"):
        AcceptedGraph.from_json(
            '{"format":"nope","version":1,"kind":"edges","payload":{"edges":[]}}'
        )
    with pytest.raises(CausalValueError, match="unknown AcceptedGraph kind"):
        _decode_graph("mystery", {})


def test_encode_decode_edge_lists_and_admg():
    static = AcceptedGraph.from_graph([("a", "b"), ("b", "c")], algorithm_id="hand")
    restored = AcceptedGraph.from_json(static.to_json())
    assert list(restored) == [("a", "b"), ("b", "c")]
    assert "a" in restored
    assert ("a", "b") in restored
    assert ("b", "a") not in restored
    assert 3 not in restored
    assert len(restored) == 3
    assert restored.pending == ()

    lagged = AcceptedGraph.from_graph([("a", 1, "b", 0)], algorithm_id="pcmci")
    lagged_back = AcceptedGraph.from_json(lagged.to_json())
    assert list(lagged_back) == [("a", 1, "b", 0)]
    assert lagged_back.pending == ()

    admg = AcceptedGraph.from_graph(
        Admg.from_edges(["z", "t", "y"], [("z", "t"), ("t", "y")], [("z", "y")]),
        algorithm_id="hand",
    )
    assert admg.pending == ()
    kinds = {kind for *_, kind in admg}
    assert "directed" in kinds
    assert "bidirected" in kinds
    admg_back = AcceptedGraph.from_json(admg.to_json())
    assert isinstance(admg_back.graph, Admg)


def test_temporal_pending_and_review_edges():
    tpag = TemporalPag.from_marked_lagged_edges(
        ["p", "q"],
        [("p", 1, "q", 0, "circle", "arrow")],
    )
    handle = AcceptedGraph.from_graph(tpag, algorithm_id="lpcmci")
    with pytest.raises(CausalUnsupportedError, match="TemporalPag"):
        _ = handle.pending
    assert "pending=-1" in repr(handle)
    with pytest.raises(CausalTypeError, match="no edge accessor"):
        list(handle)

    tcpdag = TemporalCpdag.from_lagged_edges(["p", "q"], [], [("p", 0, "q", 0)])
    incomplete = AcceptedGraph.from_graph(tcpdag, algorithm_id="pcmci+")
    with pytest.raises(CausalUnsupportedError, match="TemporalCpdag"):
        _ = incomplete.pending

    dag = AcceptedGraph.from_graph(
        Dag.from_edges(["t", "y"], [("t", "y")]),
        algorithm_id="hand",
    )
    bumped = dag.review({})
    assert bumped.version == dag.version + 1

    cpdag = AcceptedGraph.from_graph(
        Cpdag.from_directed_undirected(["t", "y"], [], [("t", "y")]),
        algorithm_id="pc",
    )
    with pytest.raises(CausalValueError, match="not a valid Cpdag orientation"):
        cpdag.review({("t", "y"): ("circle", "arrow")})
    reversed_review = cpdag.review({("t", "y"): ("arrow", "tail")})
    assert isinstance(reversed_review.graph, Dag)

    pag = AcceptedGraph.from_graph(
        Pag.from_marked_edges(
            ["x", "y", "z"],
            [("x", "y", "circle", "arrow"), ("y", "z", "tail", "arrow")],
        ),
        algorithm_id="fci",
    )
    with pytest.raises(CausalValueError, match="not a pending edge"):
        pag.review({("y", "z"): ("tail", "arrow")})
    with pytest.raises(CausalValueError, match="tail/arrow/circle"):
        pag.review({("x", "y"): ("dot", "arrow")})
    still_pending = pag.review({("x", "y"): ("circle", "arrow")})
    assert still_pending.pending


def test_discovery_keeps_isolated_variables_and_never_invents_names():
    # PC on {t, y, z} finds only t-z: y has no adjacency but is still a node.
    result = _result(edges=[_Edge("t", "z")], nodes=["t", "y", "z"])
    dag = _result_to_graph(result, "pc")
    assert isinstance(dag, Dag)
    assert sorted(dag.nodes()) == ["t", "y", "z"]
    cpdag = _result_to_graph(
        _result(edges=[_Edge("t", "z", "tail", "tail")], nodes=["t", "y", "z"]), "pc"
    )
    assert sorted(cpdag.nodes()) == ["t", "y", "z"]
    # A discovery with no edges is a graph over its variables, not over "x", "y".
    empty = _result_to_graph(_result(nodes=["a", "b", "c"]), "pc")
    assert sorted(empty.nodes()) == ["a", "b", "c"]
    temporal = _plain_pcmci_temporal_dag(
        _result(links=[_Link("p", "q", 1, 0)], nodes=["p", "q", "r"])
    )
    assert isinstance(temporal, TemporalDag)

    nameless = SimpleNamespace(graph_edges=[], links=[], variable_names=[])
    with pytest.raises(CausalValueError, match="no variable names"):
        _result_to_graph(nameless, "pc")
    with pytest.raises(CausalValueError, match="no variable names"):
        _plain_pcmci_temporal_dag(nameless)


def test_pag_conflict_marks_are_pending_and_reviewable():
    pag = Pag.from_marked_edges(
        ["a", "b", "c"], [("a", "b", "tail", "conflict"), ("b", "c", "tail", "arrow")]
    )
    handle = AcceptedGraph.from_graph(pag)
    assert [(e.source, e.target) for e in handle.pending] == [("a", "b")]
    reviewed = handle.review({("a", "b"): ("tail", "arrow")})
    assert reviewed.pending == ()


def test_pag_review_refuses_both_directions_of_one_edge():
    pag = Pag.from_marked_edges(["a", "b"], [("a", "b", "circle", "arrow")])
    handle = AcceptedGraph.from_graph(pag)
    with pytest.raises(CausalValueError, match="marked twice"):
        handle.review({("a", "b"): ("tail", "arrow"), ("b", "a"): ("arrow", "tail")})


def test_cpdag_cycle_error_names_the_callers_orientation():
    # a->b and b->c are fixed; the caller orienting c->a closes the cycle.
    cpdag = Cpdag.from_directed_undirected(["a", "b", "c"], [("a", "b"), ("b", "c")], [("a", "c")])
    handle = AcceptedGraph.from_graph(cpdag)
    with pytest.raises(CausalValueError, match=r"orienting 'c'->'a' would create"):
        handle.review({("a", "c"): ("arrow", "tail")})
