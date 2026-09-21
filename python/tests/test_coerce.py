"""``antecedent._coerce`` — the only module allowed to accept union input types.

Covers every accepted input shape and every rejection for ``coerce_data``,
``coerce_query``, ``coerce_refute``, ``coerce_latency``.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import _coerce
from antecedent.ids import Latency, Refute
from antecedent.query import (
    AnomalyAttribution,
    AverageEffect,
    ChangeAttribution,
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
    with pytest.raises(TypeError, match="data must be a mapping"):
        _coerce.coerce_data(5)


def test_coerce_data_panel_frame_pools_every_unit():
    """``Config.run(panel)`` must see every unit, not silently only unit 0.

    Two units with disjoint value ranges: pooling both units' columns means
    every value from both units is present. Reading only ``unit_columns[0]``
    (the pre-fix behaviour) would leave the second unit's distinctive values
    (100..104) entirely absent from the coerced columns.
    """
    panel = antecedent.data.panel(
        [
            {"x": [0.0, 1.0, 2.0, 3.0, 4.0]},
            {"x": [100.0, 101.0, 102.0, 103.0, 104.0]},
        ]
    )
    names, cols = _coerce.coerce_data(panel)
    assert names == ["x"]
    np.testing.assert_allclose(
        np.sort(cols[0]), [0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    )


def test_coerce_data_refuses_environment_pooling():
    frame = antecedent.data.multi_env(
        [
            {"x": [0.0, 1.0, 2.0]},
            {"x": [50.0, 51.0, 52.0]},
        ]
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="mixture") as info:
        _coerce.coerce_data(frame)
    assert info.value.reason_code == "data_modality_not_licensed"
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="mixture"):
        _coerce.discovery_table([{"x": [0.0, 1.0]}, {"x": [5.0, 6.0]}])


def test_coerce_data_refuses_lagged_panel_pooling():
    panel = antecedent.data.panel(
        [
            {"x": [0.0, 1.0, 2.0, 3.0]},
            {"x": [100.0, 101.0, 102.0, 103.0]},
        ]
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="unit boundary") as info:
        _coerce.coerce_data(panel, temporal=True)
    assert info.value.reason_code == "data_modality_not_licensed"
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="unit boundary"):
        _coerce.discovery_table(panel, temporal=True)
    static = _coerce.discovery_table(panel)
    assert list(static) == ["x"]
    assert len(static["x"]) == 8


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
    AnomalyAttribution(["y"]),
    ChangeAttribution(
        "y", baseline_start=0, baseline_end=40, comparison_start=40, comparison_end=80
    ),
]


@pytest.mark.parametrize(
    "query", _QUERY_INSTANCES, ids=[type(q).__name__ for q in _QUERY_INSTANCES]
)
def test_coerce_query_accepts_each_query_type(query):
    assert _coerce.coerce_query(query) is query


def test_coerce_query_rejects_unsupported_type():
    with pytest.raises(TypeError, match="unsupported query type"):
        _coerce.coerce_query({"kind": "average"})


# --------------------------------------------------------------------------
# coerce_refute
# --------------------------------------------------------------------------


def test_coerce_refute_true_raises_type_error():
    with pytest.raises(TypeError, match="refute=True is ambiguous"):
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
    with pytest.raises(TypeError, match="unsupported refute type"):
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
    with pytest.raises(TypeError, match="unsupported latency type"):
        _coerce.coerce_latency(5)
