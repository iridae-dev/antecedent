"""Route-native diagnostics reach the result on success, in the exception's shape."""

from __future__ import annotations

import antecedent
import numpy as np

_ORDER = ["convergence", "posterior_ess", "r_hat", "monte_carlo_error"]


def _confounded(n: int = 120, seed: int = 5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (z + rng.normal(size=n) > 0).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}, [("z", "t"), ("z", "y"), ("t", "y")]


def _analyze(inference):
    data, edges = _confounded()
    return antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=inference,
        refute=False,
        seed=1,
    )


def test_bayesian_result_carries_passed_route_records_with_measured_mass():
    result = _analyze(antecedent.Bayesian(n_draws=64, backend="conjugate"))
    records = list(result.route_diagnostics)
    assert records, "a published posterior records the checks it passed"
    routes = [record["route"] for record in records]
    assert routes == sorted(routes, key=_ORDER.index)
    for record in records:
        assert set(record) == {"route", "outcome", "failed", "unevaluable_mass"}
        assert record["outcome"] == "passed"
        assert record["failed"] is False
        assert isinstance(record["unevaluable_mass"], float)


def test_frequentist_result_has_no_route_records():
    result = _analyze(antecedent.Frequentist())
    assert list(result.route_diagnostics) == []


def _modified(n: int = 200, seed: int = 213):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = (rng.random(n) < 0.5).astype(np.float64)
    y = 0.4 + 1.25 * t + 0.6 * x + 0.35 * t * x + rng.normal(size=n)
    return {"t": t, "y": y, "x": x}, [("t", "y"), ("x", "t"), ("x", "y")]


def _bayesian(estimator, query):
    data, edges = _modified()
    return antecedent.analyze(
        data,
        graph=edges,
        query=query,
        estimator=estimator,
        inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
        refute=False,
        seed=19,
    )


def _assert_passed_convergence(result):
    records = list(result.route_diagnostics)
    convergence = [record for record in records if record["route"] == "convergence"]
    assert len(convergence) == 1, records
    assert convergence[0]["outcome"] == "passed"
    assert convergence[0]["failed"] is False
    # A measured mass, not an unreported one: a measured zero is not "unreported".
    assert isinstance(convergence[0]["unevaluable_mass"], float)
    assert all(not record["failed"] for record in records)


def test_basis_result_exposes_its_passed_convergence_record():
    _assert_passed_convergence(
        _bayesian(antecedent.Estimator.BAYESIAN_BASIS_GCOMP, antecedent.AverageEffect("t", "y"))
    )


def test_robust_result_exposes_its_passed_convergence_record():
    _assert_passed_convergence(
        _bayesian(antecedent.Estimator.BAYESIAN_ROBUST_ATE, antecedent.AverageEffect("t", "y"))
    )


def test_conditional_result_exposes_its_passed_convergence_record():
    _assert_passed_convergence(
        _bayesian(
            antecedent.Estimator.BAYESIAN_CONDITIONAL, antecedent.ConditionalEffect("t", "y", "x")
        )
    )
