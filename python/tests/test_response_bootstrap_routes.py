"""``bootstrap=`` on every response route either takes effect or raises.

Only Frequentist temporal surfaces (TemporalDag, TemporalCpdag/Pag completion
atoms) resample. Every other response route carries analytic, influence-function
or posterior uncertainty and must refuse a requested bootstrap instead of
silently dropping it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("antecedent")

import antecedent
from antecedent.errors import CausalUnsupportedError
from antecedent.estimation import PreparedAnalysis
from antecedent.graph import TemporalCpdag, TieredBackground, WithinTier
from antecedent.intervention import Set

_BAND_WITHHELD = "no pointwise or simultaneous band"


def _static_data(n: int = 300, seed: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = 0.5 * z + rng.normal(size=n)
    y = 1.0 + 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}


def _binary_data(n: int = 600, seed: int = 9) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t1 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(float)
    t2 = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.5 * z))).astype(float)
    y = t1 + t2 + 0.5 * z + 0.3 * rng.normal(size=n)
    return {"z": z, "t1": t1, "t2": t2, "y": y}


def _temporal_data(n: int = 242) -> dict[str, np.ndarray]:
    t = np.array([0.0 if i % 4 in (0, 2) else (1.0 if i % 4 == 1 else -1.0) for i in range(n)])
    y = np.zeros(n)
    for i in range(n):
        y[i] = 1.0 + 2.0 * (t[i - 1] if i >= 1 else 0.0) + 3.0 * (t[i - 2] if i >= 2 else 0.0)
    return {"t": t, "y": y}


_STATIC_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
_TEMPORAL_EDGES = [("t", 1, "y", 0), ("t", 2, "y", 0)]
_CURVE = antecedent.ResponseCurve("t", "y", grid=[-0.5, 0.0, 0.5])
_TEMPORAL_CURVE = antecedent.ResponseCurve(
    "t", "y", grid=[0.0, 1.0], horizons=[1, 2], policy="pulse", treatment_lag=1
)
_TEMPORAL_PATH = antecedent.InterventionResponse(
    "y", intervention=Set("t", 1.0), horizons=[1, 2], policy="pulse", treatment_lag=1
)
_JOINT = antecedent.InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)])
_BAYES = antecedent.Bayesian(backend="conjugate", n_draws=256)


def _refusing_routes() -> list[tuple[str, dict[str, Any], type[Exception], str]]:
    static = _static_data()
    binary = _binary_data()
    temporal = _temporal_data()
    cpdag = antecedent.Cpdag.from_directed_undirected(
        ["z", "t", "y"], [("z", "t"), ("z", "y")], [("t", "y")]
    )
    admg = antecedent.Admg.from_edges(["z", "t", "y"], [("z", "t"), ("t", "y")], [("z", "y")])
    tiered = TieredBackground(
        tiers=[["z"], ["t1", "t2"], ["y"]], within_tier=WithinTier.CODETERMINED
    )
    static_msg = "do not yet expose bootstrap"
    return [
        (
            "dag_response_curve",
            {"data": static, "graph": _STATIC_EDGES, "query": _CURVE},
            ValueError,
            static_msg,
        ),
        (
            "dag_average_derivative",
            {
                "data": static,
                "graph": _STATIC_EDGES,
                "query": antecedent.AverageDerivative("t", "y"),
            },
            ValueError,
            static_msg,
        ),
        (
            "dag_intervention_response",
            {
                "data": static,
                "graph": _STATIC_EDGES,
                "query": antecedent.InterventionResponse("y", intervention=Set("t", 1.0)),
            },
            ValueError,
            static_msg,
        ),
        (
            "cpdag_response_curve",
            {"data": static, "graph": cpdag, "query": _CURVE},
            ValueError,
            static_msg,
        ),
        (
            "admg_response_curve",
            {"data": static, "graph": admg, "query": _CURVE},
            ValueError,
            static_msg,
        ),
        (
            "cell_aipw",
            {
                "data": binary,
                "graph": [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")],
                "query": _JOINT,
                "estimator": "cell.aipw",
            },
            CausalUnsupportedError,
            "cell.aipw uses analytic influence uncertainty",
        ),
        (
            "tiered_background_cell_aipw",
            {"data": binary, "graph": tiered, "query": _JOINT},
            CausalUnsupportedError,
            "cell.aipw uses analytic influence uncertainty",
        ),
        (
            "bayesian_static_curve",
            {"data": static, "graph": _STATIC_EDGES, "query": _CURVE, "inference": _BAYES},
            CausalUnsupportedError,
            "posterior intervals",
        ),
        (
            "bayesian_temporal_curve",
            {
                "data": temporal,
                "graph": _TEMPORAL_EDGES,
                "query": _TEMPORAL_CURVE,
                "inference": _BAYES,
            },
            CausalUnsupportedError,
            "posterior intervals",
        ),
    ]


@pytest.mark.parametrize(
    "route", _refusing_routes(), ids=[route[0] for route in _refusing_routes()]
)
def test_analyze_refuses_bootstrap_where_it_does_not_apply(route) -> None:
    _, kwargs, error, match = route
    data = kwargs.pop("data")
    with pytest.raises(error, match=match):
        antecedent.analyze(data, refute=False, bootstrap=60, **kwargs)


@pytest.mark.parametrize(
    "route", _refusing_routes(), ids=[route[0] for route in _refusing_routes()]
)
def test_prepare_refuses_bootstrap_where_it_does_not_apply(route) -> None:
    _, kwargs, _, _ = route
    data = kwargs.pop("data")
    with pytest.raises(CausalUnsupportedError, match="bootstrap"):
        PreparedAnalysis.prepare(data, refute="none", bootstrap=60, **kwargs)


@pytest.mark.parametrize("query", [_TEMPORAL_CURVE, _TEMPORAL_PATH], ids=["curve", "path"])
def test_temporal_dag_bootstrap_takes_effect(query) -> None:
    data = _temporal_data()
    withheld = antecedent.analyze(
        data, graph=_TEMPORAL_EDGES, query=query, refute=False, bootstrap=0
    )
    banded = antecedent.analyze(
        data, graph=_TEMPORAL_EDGES, query=query, refute=False, bootstrap=60
    )
    assert withheld.uncertainty.kind == "none"
    assert any(_BAND_WITHHELD in warning for warning in withheld.support.warnings)
    assert banded.uncertainty.kind == "pointwise"
    assert banded.simultaneous_band is not None
    assert banded.simultaneous_band.replicates == 60
    for bootstrap, kind in ((0, "none"), (60, "pointwise")):
        prepared = PreparedAnalysis.prepare(
            data, graph=_TEMPORAL_EDGES, query=query, refute="none", bootstrap=bootstrap
        )
        assert prepared.estimate(data).uncertainty.kind == kind


def test_temporal_class_bootstrap_takes_effect() -> None:
    """Replicates ride each completion atom of the TemporalCpdag identified set."""
    data = _temporal_data()
    graph = TemporalCpdag.from_lagged_edges(["t", "y"], _TEMPORAL_EDGES, [])
    withheld = antecedent.analyze(
        data, graph=graph, query=_TEMPORAL_CURVE, refute=False, bootstrap=0
    )
    banded = antecedent.analyze(
        data, graph=graph, query=_TEMPORAL_CURVE, refute=False, bootstrap=60
    )
    assert withheld.uncertainty.kind == banded.uncertainty.kind == "identified_set"
    assert any(_BAND_WITHHELD in warning for warning in withheld.support.warnings)
    assert not any(_BAND_WITHHELD in warning for warning in banded.support.warnings)


def test_temporal_bootstrap_rejects_invalid_counts() -> None:
    data = _temporal_data()
    for bad in (-1, 2.5, True):
        with pytest.raises(antecedent.errors.CausalValueError, match="non-negative integer"):
            PreparedAnalysis.prepare(
                data, graph=_TEMPORAL_EDGES, query=_TEMPORAL_CURVE, refute="none", bootstrap=bad
            )
