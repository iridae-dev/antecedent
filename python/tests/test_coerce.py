"""``antecedent._coerce`` — the only module allowed to accept union input types.

Covers every accepted input shape and every rejection for ``coerce_data``,
``coerce_graph``, ``coerce_query``, ``coerce_refute``, ``coerce_latency``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent import _coerce
from antecedent.graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag
from antecedent.ids import Latency, Refute
from antecedent.query import (
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    InterventionalDistribution,
    MediationEffect,
    PathSpecificEffect,
    PulseEffect,
    SustainedEffect,
    TemporalMediationEffect,
)

# --------------------------------------------------------------------------
# coerce_data
# --------------------------------------------------------------------------


def test_coerce_data_mapping():
    names, cols = _coerce.coerce_data({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    assert names == ["a", "b"]
    assert len(cols) == 2
    assert cols[0].dtype == np.float64
    np.testing.assert_allclose(cols[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cols[1], [4.0, 5.0, 6.0])


def test_coerce_data_names_columns_pair():
    names, cols = _coerce.coerce_data((["a", "b"], [[1.0, 2.0], [3.0, 4.0]]))
    assert names == ["a", "b"]
    np.testing.assert_allclose(cols[0], [1.0, 2.0])
    np.testing.assert_allclose(cols[1], [3.0, 4.0])


def test_coerce_data_names_columns_pair_matches_mapping_form():
    mapping = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    names_map, cols_map = _coerce.coerce_data(mapping)
    names_pair, cols_pair = _coerce.coerce_data((names_map, cols_map))
    assert names_pair == names_map
    for a, b in zip(cols_pair, cols_map, strict=True):
        np.testing.assert_allclose(a, b)


def test_coerce_data_event_frame():
    n = 5
    data = {"x": np.arange(n, dtype=np.float64), "y": np.ones(n)}
    frame = antecedent.data.event(data, np.arange(n, dtype=np.int64), align_interval_ns=1)
    names, cols = _coerce.coerce_data(frame)
    assert names == ["x", "y"]
    np.testing.assert_allclose(cols[0], frame.columns[0])
    np.testing.assert_allclose(cols[1], frame.columns[1])


def test_coerce_data_dataframe_like():
    class _FakeSeries:
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float64)

        def to_numpy(self):
            return self._values

    class _FakeFrame:
        def __init__(self, mapping):
            self._data = {k: _FakeSeries(v) for k, v in mapping.items()}

        @property
        def columns(self):
            return list(self._data.keys())

        def __getitem__(self, key):
            return self._data[key]

        def to_numpy(self):  # only needs to exist for the duck-type check
            raise NotImplementedError

    frame = _FakeFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    names, cols = _coerce.coerce_data(frame)
    assert names == ["a", "b"]
    np.testing.assert_allclose(cols[0], [1.0, 2.0])


def test_coerce_data_rejects_unsupported_type():
    with pytest.raises(TypeError):
        _coerce.coerce_data(5)


# --------------------------------------------------------------------------
# coerce_graph
# --------------------------------------------------------------------------


def test_coerce_graph_dag():
    dag = Dag.from_edges(["a", "b"], [("a", "b")])
    edges = _coerce.coerce_graph(dag)
    assert edges == [("a", "b")]


def test_coerce_graph_cpdag_fully_oriented():
    cpdag = Cpdag.from_directed_undirected(["a", "b"], [("a", "b")], [])
    edges = _coerce.coerce_graph(cpdag)
    assert edges == [("a", "b")]


def test_coerce_graph_cpdag_with_undirected_edge_raises():
    cpdag = Cpdag.from_directed_undirected(["a", "b"], [], [("a", "b")])
    with pytest.raises(ValueError):
        _coerce.coerce_graph(cpdag)


def test_coerce_graph_temporal_dag():
    tdag = TemporalDag.from_lagged_edges(["x", "y"], [("x", 1, "y", 0)])
    edges = _coerce.coerce_graph(tdag)
    assert edges == [("x", 1, "y", 0)]


def test_coerce_graph_temporal_cpdag_fully_oriented():
    tcpdag = TemporalCpdag.from_lagged_edges(["x", "y"], [("x", 1, "y", 0)], None)
    edges = _coerce.coerce_graph(tcpdag)
    assert edges == [("x", 1, "y", 0)]


def test_coerce_graph_pag_passthrough():
    pag = Pag.from_marked_edges(["x", "y"], [("x", "y", "circle", "arrow")])
    assert _coerce.coerce_graph(pag) is pag


def test_coerce_graph_admg_passthrough():
    admg = Admg.from_edges(["z", "t", "y"], [("z", "t"), ("t", "y")], [("z", "y")])
    assert _coerce.coerce_graph(admg) is admg


def test_coerce_graph_temporal_pag_passthrough():
    tpag = TemporalPag.from_marked_lagged_edges(["x", "y"], [("x", 1, "y", 0, "tail", "arrow")])
    assert _coerce.coerce_graph(tpag) is tpag


def test_coerce_graph_static_edge_list():
    edges = _coerce.coerce_graph([("a", "b"), ("b", "c")])
    assert edges == [("a", "b"), ("b", "c")]


def test_coerce_graph_lagged_edge_list():
    edges = _coerce.coerce_graph([("a", 1, "b", 0)])
    assert edges == [("a", 1, "b", 0)]


def test_coerce_graph_rejects_unsupported_type():
    with pytest.raises(TypeError):
        _coerce.coerce_graph(5)


# --------------------------------------------------------------------------
# coerce_query
# --------------------------------------------------------------------------

_QUERY_INSTANCES = [
    AverageEffect("t", "y"),
    PulseEffect("t", "y"),
    SustainedEffect("t", "y"),
    InterventionalDistribution("y"),
    PathSpecificEffect("t", "y"),
    ConditionalEffect("t", "y", "m"),
    MediationEffect("t", "y", mediators=["m"]),
    Counterfactual("t", "y"),
    TemporalMediationEffect("t", "m", "y"),
]


@pytest.mark.parametrize(
    "query", _QUERY_INSTANCES, ids=[type(q).__name__ for q in _QUERY_INSTANCES]
)
def test_coerce_query_accepts_each_query_type(query):
    assert _coerce.coerce_query(query) is query


def test_coerce_query_rejects_unsupported_type():
    with pytest.raises(TypeError):
        _coerce.coerce_query({"kind": "average"})


# --------------------------------------------------------------------------
# coerce_refute
# --------------------------------------------------------------------------


def test_coerce_refute_true_raises_type_error():
    with pytest.raises(TypeError):
        _coerce.coerce_refute(True)


def test_coerce_refute_false_passes_through():
    assert _coerce.coerce_refute(False) is False


def test_coerce_refute_enum_member_to_string():
    assert _coerce.coerce_refute(Refute.FULL) == "full"
    assert _coerce.coerce_refute(Refute.PLACEBO) == "placebo"


def test_coerce_refute_string_passthrough():
    assert _coerce.coerce_refute("placebo") == "placebo"
    assert _coerce.coerce_refute("cheap") == "cheap"


def test_coerce_refute_rejects_unsupported_type():
    with pytest.raises(TypeError):
        _coerce.coerce_refute(5)


# --------------------------------------------------------------------------
# coerce_latency
# --------------------------------------------------------------------------


def test_coerce_latency_none_passthrough():
    assert _coerce.coerce_latency(None) is None


def test_coerce_latency_enum_member():
    assert _coerce.coerce_latency(Latency.STANDARD) == "standard"
    assert _coerce.coerce_latency(Latency.INTERACTIVE) == "interactive"


def test_coerce_latency_string_normalizes_case():
    assert _coerce.coerce_latency("Interactive") == "interactive"
    assert _coerce.coerce_latency(" report ") == "report"


def test_coerce_latency_rejects_unknown_string():
    with pytest.raises(ValueError, match="unknown latency"):
        _coerce.coerce_latency("blazing_fast")


def test_coerce_latency_rejects_unsupported_type():
    with pytest.raises(TypeError):
        _coerce.coerce_latency(5)
