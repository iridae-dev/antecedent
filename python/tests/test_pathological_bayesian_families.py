"""Pathological-but-valid Bayesian families: each finds its known value or refuses with its code."""

from __future__ import annotations

import antecedent as ac
import numpy as np
import pytest
from antecedent.outcomes import require_success

BAYES = ac.Bayesian(backend="conjugate", n_draws=256, prior_scale=10.0)


def _run(data, graph, query):
    return require_success(
        ac.estimate(
            ac.prepare(
                data,
                query=query,
                graph=graph,
                inference=BAYES,
                refute="none",
                execution=ac.Execution(seed=3),
            )
        )
    )


def test_conditional_constant_modifier_stays_finite_or_refuses():
    n = 120
    t = np.tile([0.0, 1.0], n // 2)
    w = np.ones(n)
    y = 1.0 + 2.0 * t + np.linspace(-0.1, 0.1, n)
    data = {"t": t, "w": w, "y": y}
    graph = [("t", "y"), ("w", "y")]
    # A constant modifier is collinear with the intercept column, but the effect is still the
    # slope 2 of y = 1 + 2 t (the modifier carries no heterogeneity): it must be found, not
    # silently replaced by a refusal or a different number.
    result = _run(data, graph, ac.ConditionalEffect("t", "y", "w"))
    assert result.estimate.estimator_id == "conditional.bayesian"
    assert result.ate == pytest.approx(2.0, abs=0.05)


def test_conditional_complete_case_missingness_does_not_crash():
    n = 160
    t = np.tile([0.0, 1.0], n // 2).astype(np.float64)
    w = np.linspace(-1.0, 1.0, n)
    y = 1.0 + 2.0 * t + 0.5 * t * w
    y = y.copy()
    y[::7] = np.nan
    data = {"t": t, "w": w, "y": y}
    # y = 1 + 2 t + 0.5 t w with w centred: the average effect is 2 on the complete cases.
    result = _run(data, [("t", "y"), ("w", "y")], ac.ConditionalEffect("t", "y", "w"))
    assert result.ate == pytest.approx(2.0, abs=0.05)


def test_mediation_constant_mediator_refuses_or_stays_finite():
    n = 200
    t = np.tile([0.0, 1.0], n // 2)
    m = np.ones(n)
    y = 0.2 * np.r_[0, t[:-1]] + np.linspace(-0.05, 0.05, n)
    data = {"t": t, "m": m, "y": y}
    graph = [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]
    # A constant mediator column is collinear with the intercept: the design is rank deficient
    # (rank 2 of 3 columns) and the route refuses with that code rather than returning a number.
    with pytest.raises(ac.errors.CausalEstimateError) as caught:
        _run(data, graph, ac.TemporalMediationEffect("t", "m", "y"))
    assert caught.value.reason_code == "design_rank_deficient"
    assert "rank=2 ncols=3" in str(caught.value)


def test_response_tiny_sample_refuses_or_stays_finite():
    t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y = 2.0 * t
    # Six complete rows are below the continuous-response floor of twenty.
    with pytest.raises(ac.errors.CausalEstimateError) as caught:
        _run({"t": t, "y": y}, [("t", "y")], ac.ResponseCurve("t", "y", grid=[0.0, 1.0]))
    assert "at least 20 complete rows" in str(caught.value)


def test_sequential_short_series_refuses_or_stays_finite():
    n = 12
    t = np.linspace(0.0, 1.0, n)
    y = np.r_[0.0, 2.0 * t[:-1]]
    graph = [("t", 1, "y", 0), ("t", 2, "y", 0)]
    # Twelve points of a linear ramp make the two lagged treatments collinear: the design is
    # rank deficient and the route says so.
    with pytest.raises(ac.errors.CausalEstimateError) as caught:
        _run({"t": t, "y": y}, graph, ac.SustainedEffect("t", "y", window=(-2, -1)))
    assert caught.value.reason_code == "design_rank_deficient"
