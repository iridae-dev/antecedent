"""Pathological-but-valid 1.2 Bayesian families: refuse or stay finite, never silent."""

from __future__ import annotations

import antecedent as ac
import numpy as np
from antecedent.estimation import PreparedAnalysis

BAYES = ac.Bayesian(backend="conjugate", n_draws=256, prior_scale=10.0)


def _run(data, graph, query):
    return PreparedAnalysis.prepare(
        data,
        query=query,
        graph=graph,
        inference=BAYES,
        refute="none",
        latency=None,
        bootstrap=0,
    ).estimate(data, seed=3)


def test_conditional_constant_modifier_stays_finite_or_refuses():
    n = 120
    t = np.tile([0.0, 1.0], n // 2)
    w = np.ones(n)
    y = 1.0 + 2.0 * t + np.linspace(-0.1, 0.1, n)
    data = {"t": t, "w": w, "y": y}
    graph = [("t", "y"), ("w", "y")]
    try:
        result = _run(data, graph, ac.ConditionalEffect("t", "y", "w"))
    except ac.errors.CausalError as err:
        assert (
            "modifier" in str(err).lower()
            or "rank" in str(err).lower()
            or "design" in str(err).lower()
        )
        return
    assert np.isfinite(result.ate)
    assert result.estimate.estimator_id == "conditional.bayesian"


def test_conditional_complete_case_missingness_does_not_crash():
    n = 160
    t = np.tile([0.0, 1.0], n // 2).astype(np.float64)
    w = np.linspace(-1.0, 1.0, n)
    y = 1.0 + 2.0 * t + 0.5 * t * w
    y = y.copy()
    y[::7] = np.nan
    data = {"t": t, "w": w, "y": y}
    try:
        result = _run(data, [("t", "y"), ("w", "y")], ac.ConditionalEffect("t", "y", "w"))
    except ac.errors.CausalError:
        return
    assert np.isfinite(result.ate)


def test_mediation_constant_mediator_refuses_or_stays_finite():
    n = 200
    t = np.tile([0.0, 1.0], n // 2)
    m = np.ones(n)
    y = 0.2 * np.r_[0, t[:-1]] + np.linspace(-0.05, 0.05, n)
    data = {"t": t, "m": m, "y": y}
    graph = [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]
    try:
        result = _run(data, graph, ac.TemporalMediationEffect("t", "m", "y"))
    except ac.errors.CausalError:
        return
    assert np.isfinite(result.ate)


def test_response_tiny_sample_refuses_or_stays_finite():
    t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y = 2.0 * t
    try:
        result = _run({"t": t, "y": y}, [("t", "y")], ac.ResponseCurve("t", "y", grid=[0.0, 1.0]))
    except ac.errors.CausalError:
        return
    assert result.response is not None


def test_sequential_short_series_refuses_or_stays_finite():
    n = 12
    t = np.linspace(0.0, 1.0, n)
    y = np.r_[0.0, 2.0 * t[:-1]]
    graph = [("t", 1, "y", 0), ("t", 2, "y", 0)]
    try:
        result = _run(
            {"t": t, "y": y},
            graph,
            ac.SustainedEffect("t", "y", window=(-2, -1)),
        )
    except ac.errors.CausalError:
        return
    assert np.isfinite(result.ate)
