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
