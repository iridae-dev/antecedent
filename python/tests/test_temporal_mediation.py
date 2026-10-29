"""Temporal mediation typed query recovers mediated path product."""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def _ols(columns, response):
    design = np.column_stack([np.ones(len(response)), *columns])
    return np.linalg.lstsq(design, response, rcond=None)[0]


def test_temporal_mediation_decomposition():
    n = 200
    t = np.zeros(n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.3 * t[i - 1] + 0.1 * np.sin(i)
        m[i] = 0.8 * t[i - 1] + 0.05 * np.cos(i)
        # No direct T→Y: mediated-only DGP for MediationContrast::Mediated ID.
        y[i] = 0.5 * m[i] + 0.02 * np.sin(i)
    data = {"t": t, "m": m, "y": y}
    edges = [
        ("t", 1, "m", 0),
        ("m", 0, "y", 0),
    ]
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.mediation is not None
    # Path products of plain least squares on the lag-aligned sample; the decomposition
    # total = direct + mediated holds exactly for nested OLS fits on one sample.
    a = _ols([t[:-1]], m[1:])[1]
    outcome = _ols([t[:-1], m[1:]], y[1:])
    mediation = result.mediation
    assert mediation.mediated == pytest.approx(a * outcome[2], abs=1e-8)
    assert mediation.direct == pytest.approx(outcome[1], abs=1e-8)
    assert mediation.total == pytest.approx(_ols([t[:-1]], y[1:])[1], abs=1e-8)
    assert mediation.total - mediation.direct - mediation.mediated == pytest.approx(0.0, abs=1e-8)


def test_handle_temporal_mediation_bayesian_uses_staged_estimator():
    """analyze is prepare().estimate(); Bayesian TemporalMediationEffect stays licensed."""
    n = 80
    t = np.zeros(n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.3 * t[i - 1] + 0.1 * np.sin(i)
        m[i] = 0.8 * t[i - 1] + 0.05 * np.cos(i)
        y[i] = 0.5 * m[i] + 0.02 * np.sin(i)
    data = {"t": t, "m": m, "y": y}
    graph = antecedent.TemporalDag.from_lagged_edges(
        ["t", "m", "y"],
        [("t", 1, "m", 0), ("m", 0, "y", 0)],
    )
    result = require_success(
        antecedent.analyze(
            data,
            query=antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated"),
            graph=graph,
            inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
            refute=False,
            execution=antecedent.Execution(seed=1, threads=1),
        )
    )
    assert result.estimate.estimator_id == "temporal.mediation.bayesian"
    assert result.posterior is not None
    assert np.isfinite(result.ate)
    # With independent noise the posterior mean recovers the structural path product
    # 0.8 * 0.5 = 0.4 (the sinusoidal toy series above is collinear and does not).
    rng = np.random.default_rng(20260101)
    size = 6000
    t, m, y = np.zeros(size), np.zeros(size), np.zeros(size)
    for i in range(1, size):
        t[i] = 0.3 * t[i - 1] + rng.normal(scale=0.5)
        m[i] = 0.8 * t[i - 1] + rng.normal(scale=0.3)
        y[i] = 0.5 * m[i] + rng.normal(scale=0.3)
    noisy = require_success(
        antecedent.analyze(
            {"t": t, "m": m, "y": y},
            query=antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated"),
            graph=graph,
            inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
            refute=False,
            execution=antecedent.Execution(seed=1, threads=1),
        )
    )
    assert noisy.ate == pytest.approx(0.4, abs=0.03)


def _mediated_series(n: int = 300):
    """``m = 0.8 t[-1]`` and ``y = 0.25 t[-1] + 0.55 m``: mediated 0.44, direct 0.25."""
    index = np.arange(n, dtype=np.float64)
    treatment = np.sin(index * 0.31)
    mediator = 0.8 * np.roll(treatment, 1) + 0.1 * np.sin(index * 0.51)
    outcome = 0.25 * np.roll(treatment, 1) + 0.55 * mediator + 0.01 * np.cos(index * 0.19)
    return {"t": treatment, "m": mediator, "y": outcome}


def _lagged_edges():
    return [("t", 1, "m", 0), ("t", 1, "y", 0), ("m", 0, "y", 0)]


def test_a_dag_mediation_slice_states_its_identification_and_a_frequentist_uncertainty():
    result = require_success(
        antecedent.analyze(
            _mediated_series(),
            graph=antecedent.TemporalDag.from_lagged_edges(["t", "m", "y"], _lagged_edges()),
            query=antecedent.TemporalMediationEffect("t", "m", "y", horizons=[1]),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=3),
        )
    )
    (slice_,) = list(result.mediation_grid)
    assert slice_.horizon == 1
    assert "Identified" in slice_.identification_status
    assert slice_.method
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in slice_.adjustment)
    # The generating coefficients give mediated 0.8 * 0.55 and direct 0.25.
    assert slice_.mediated == pytest.approx(0.44, abs=0.02)
    assert slice_.effect == pytest.approx(slice_.mediated, abs=1e-12)
    assert math.isfinite(slice_.total) and math.isfinite(slice_.direct)
    assert slice_.direct == pytest.approx(0.25, abs=0.02)
    assert slice_.total == pytest.approx(slice_.direct + slice_.mediated, abs=1e-9)
    assert slice_.uncertainty_kind == "frequentist_pointwise"
    assert slice_.q025 is None and slice_.q975 is None
    # A DAG has no completion range.
    assert slice_.identified_lower is None and slice_.identified_upper is None


def test_a_class_mediation_slice_reports_its_completion_range_and_a_bayesian_interval():
    graph = antecedent.graph.TemporalCpdag.from_lagged_edges(["t", "m", "y"], _lagged_edges(), [])
    result = require_success(
        antecedent.analyze(
            _mediated_series(),
            graph=graph,
            query=antecedent.TemporalMediationEffect("t", "m", "y", horizons=[1]),
            inference=antecedent.Bayesian(n_draws=256, backend="conjugate"),
            class_prior=antecedent.ClassPrior.from_ordered([1.0]),
            refute=False,
            execution=antecedent.Execution(seed=5),
        )
    )
    (slice_,) = list(result.mediation_grid)
    assert slice_.uncertainty_kind == "bayesian_pointwise"
    assert slice_.standard_deviation is not None and slice_.standard_deviation > 0.0
    assert slice_.q025 < slice_.effect < slice_.q975
    # One completion: the identified set is the single completion-specific point effect, which
    # is the generating mediated effect 0.8 * 0.55.
    assert slice_.identified_lower is not None
    assert slice_.identified_lower == pytest.approx(slice_.identified_upper, abs=1e-12)
    assert slice_.identified_lower == pytest.approx(0.44, abs=0.02)
    assert slice_.effect == pytest.approx(slice_.identified_lower, abs=0.02)
    assert result.mediation_grid.joint_posterior is False


def test_temporal_mediation_matches_independent_ols_path_products_claims_06():
    """The decomposition equals the path products from a plain numpy OLS, not just its own sum.

    Mediated is ``a * b`` with ``a`` the slope of the mediator on the lag-1 treatment and ``b``
    the mediator slope in the outcome regression on the lag-1 treatment and the mediator; total
    is the slope of the outcome on the lag-1 treatment alone and direct is the treatment slope
    of the second regression. The deterministic sinusoidal noise of the toy series is collinear
    with the lagged treatment, so these sample values differ from the structural 0.4; the
    reference here is the stated estimator, evaluated independently.
    """
    n = 200
    t = np.zeros(n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.3 * t[i - 1] + 0.1 * np.sin(i)
        m[i] = 0.8 * t[i - 1] + 0.05 * np.cos(i)
        y[i] = 0.5 * m[i] + 0.02 * np.sin(i)
    a = _ols([t[:-1]], m[1:])[1]
    outcome = _ols([t[:-1], m[1:]], y[1:])
    expected_total = _ols([t[:-1]], y[1:])[1]
    expected_direct, b = outcome[1], outcome[2]
    result = require_success(
        antecedent.analyze(
            {"t": t, "m": m, "y": y},
            graph=[("t", 1, "m", 0), ("m", 0, "y", 0)],
            query=antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    mediation = result.mediation
    assert mediation.mediated == pytest.approx(a * b, abs=1e-8)
    assert mediation.direct == pytest.approx(expected_direct, abs=1e-8)
    assert mediation.total == pytest.approx(expected_total, abs=1e-8)


def test_temporal_mediation_recovers_the_structural_path_product_claims_06():
    """With independent noise the mediated effect is 0.8 * 0.5 = 0.4 and there is no direct path."""
    rng = np.random.default_rng(20260101)
    n = 6000
    t = np.zeros(n)
    m = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        t[i] = 0.3 * t[i - 1] + rng.normal(scale=0.5)
        m[i] = 0.8 * t[i - 1] + rng.normal(scale=0.3)
        y[i] = 0.5 * m[i] + rng.normal(scale=0.3)
    result = require_success(
        antecedent.analyze(
            {"t": t, "m": m, "y": y},
            graph=[("t", 1, "m", 0), ("m", 0, "y", 0)],
            query=antecedent.TemporalMediationEffect("t", "m", "y", contrast="mediated"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    mediation = result.mediation
    assert mediation.mediated == pytest.approx(0.4, abs=0.03)
    assert mediation.direct == pytest.approx(0.0, abs=0.03)
    assert mediation.total == pytest.approx(0.4, abs=0.03)
