"""Bayesian intervention-level refuter state on a static DAG.

Bayesian ``InterventionResponse`` with ``refute="cheap"`` or ``"full"`` runs the
Bayesian refuter state (plugin-level overlap on the posterior-mean level and the
prior and posterior predictive checks; full adds prior sensitivity), and a static
derivative runs its derivative refuter state; neither is licensed:
``evidence_status`` is not ``"licensed"``.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest

EDGES = [("z", "t"), ("z", "y"), ("t", "y")]


def _data(seed: int = 5, n: int = 1500):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.8 * z))).astype(float)
    y = 1.0 + 2.0 * t + 1.5 * z + rng.normal(scale=0.5, size=n)
    return {"t": t, "y": y, "z": z}, float(z.mean())


def _level_query():
    return antecedent.InterventionResponse("y", intervention=antecedent.intervention.Set("t", 1.0))


@pytest.mark.parametrize("suite", ["cheap", "full"])
def test_bayesian_level_refuter_state_runs_and_is_unlicensed(suite):
    data, mean_z = _data()
    result = antecedent.analyze(
        data,
        graph=EDGES,
        query=_level_query(),
        inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
        refute=suite,
        seed=3,
    )
    assert result.evidence_status != "licensed"
    refuters = [report.refuter for report in result.validation.reports]
    assert any(name.startswith("overlap") for name in refuters), refuters
    assert "prior_predictive" in refuters and "posterior_predictive" in refuters, refuters
    assert any(name.startswith("prior_sensitivity") for name in refuters) == (suite == "full")
    assert abs(float(result.effect) - (3.0 + 1.5 * mean_z)) < 0.15


def test_bayesian_level_without_refute_stays_licensed():
    data, _ = _data(seed=7)
    result = antecedent.analyze(
        data,
        graph=EDGES,
        query=_level_query(),
        inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
        refute="none",
        seed=3,
    )
    assert result.evidence_status == "licensed"
    assert result.validation.count == 0


def test_derivative_cheap_runs_its_refuter_state():
    rng = np.random.default_rng(9)
    z = rng.normal(size=1500)
    t = 0.6 * z + rng.normal(size=1500)
    y = 1.0 + 2.0 * t + 1.5 * z + rng.normal(scale=0.5, size=1500)
    result = antecedent.analyze(
        {"t": t, "y": y, "z": z},
        graph=EDGES,
        query=antecedent.AverageDerivative("t", "y"),
        refute="cheap",
        seed=3,
    )
    assert result.evidence_status != "licensed"
    refuters = [report.refuter for report in result.validation.reports]
    assert "derivative.support" in refuters and "derivative.finite_difference" in refuters
