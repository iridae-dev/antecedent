"""Python selection and result lifecycle for the Bayesian estimators."""

from __future__ import annotations

import antecedent
import numpy as np
from antecedent.outcomes import require_success


def test_new_bayesian_estimator_ids_are_public_and_basis_result_keeps_metadata():
    assert antecedent.Estimator.BAYESIAN_BASIS_GCOMP.value == "bayesian.basis.gcomp"
    assert antecedent.Estimator.BAYESIAN_ROBUST_ATE.value == "bayesian.robust_ate"
    assert antecedent.Estimator.IV_BAYESIAN_JOINT_LINEAR.value == "iv.bayesian_joint_linear"
    assert antecedent.Estimator.RD_BAYESIAN_LOCAL_LINEAR.value == "rd.bayesian_local_linear"

    rng = np.random.default_rng(213)
    n = 120
    x = rng.normal(size=n)
    treatment = (rng.random(n) < 0.5).astype(float)
    outcome = 0.4 + 1.25 * treatment + 0.6 * x + 0.35 * treatment * x + rng.normal(size=n)
    result = require_success(
        antecedent.analyze(
            {"t": treatment, "y": outcome, "x": x},
            graph=[("t", "y"), ("x", "t"), ("x", "y")],
            query=antecedent.AverageEffect("t", "y"),
            estimator=antecedent.Estimator.BAYESIAN_BASIS_GCOMP,
            inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
            refute=False,
            execution=antecedent.Execution(seed=19),
        )
    )

    assert result.estimate.estimator_id == "bayesian.basis.gcomp"
    assert result.evidence_status == "licensed"
    assert result.posterior is not None
    assert result.posterior.n_draws == 256
    assert result.posterior.backend
    assert result.posterior.interval_type == "equal_tailed_95"
    assert result.posterior.q025 is not None
    assert result.posterior.q975 is not None
    assert result.posterior.q025 < result.posterior.q975
    assert result.posterior.interval() == (result.posterior.q025, result.posterior.q975)
    # The sample average effect of the design is 1.25 + 0.35 mean(x) = 1.2547; the equal-tailed
    # interval covers it and the posterior mean is within 0.2 of it.
    sample_ate = 1.25 + 0.35 * float(np.mean(x))
    assert result.posterior.q025 < sample_ate < result.posterior.q975
    assert abs(result.ate - sample_ate) < 0.2
    exported = result.export(quantity="posterior")
    assert bytes(exported)
    assert any(
        "zero-centered Gaussian prior" in assumption for assumption in result.assumptions or []
    )

    artifact = require_success(antecedent.load(exported))
    assert artifact.backend_id == result.posterior.backend
    assert "ate" in artifact.quantity_names
    assert artifact.n_draws == 256
    assert artifact.quantity("ate").q025 == result.posterior.q025
    assert artifact.quantity("ate").q975 == result.posterior.q975


def _bayesian_plan(estimator, *, seed=213):
    rng = np.random.default_rng(seed)
    n = 120
    x = rng.normal(size=n)
    treatment = (rng.random(n) < 0.5).astype(float)
    outcome = 0.4 + 1.25 * treatment + 0.6 * x + rng.normal(size=n)
    prepared = antecedent.prepare(
        {"t": treatment, "y": outcome, "x": x},
        graph=[("t", "y"), ("x", "t"), ("x", "y")],
        query=antecedent.AverageEffect("t", "y"),
        estimator=estimator,
        inference=antecedent.Bayesian(backend="conjugate", n_draws=64, prior_scale=30.0),
        refute=False,
        execution=antecedent.Execution(seed=19),
    )
    result = antecedent.estimate(prepared)
    require_success(result)
    return result.study.plan_view().operation, n


def test_the_bayesian_basis_route_publishes_its_target_adjustment_and_prior():
    operation, _ = _bayesian_plan(antecedent.Estimator.BAYESIAN_BASIS_GCOMP)
    assert operation["kind"] == "bayesian_basis_ate"
    assert (operation["treatment"], operation["outcome"]) == ("t", "y")
    assert operation["adjustment_set"] == ["x"]
    assert operation["posterior_draws"] == 64
    assert operation["prior_scale"] == 30.0


def test_the_bayesian_robust_route_publishes_its_rows_and_cross_fit_folds():
    operation, n = _bayesian_plan(antecedent.Estimator.BAYESIAN_ROBUST_ATE)
    assert operation["kind"] == "bayesian_robust_ate"
    assert (operation["treatment"], operation["outcome"]) == ("t", "y")
    assert operation["adjustment_set"] == ["x"]
    assert operation["posterior_draws"] == 64
    assert len(operation["row_ids"]) == len(operation["fold_ids"]) == n
    assert len(set(operation["row_ids"])) == n, "every row is planned exactly once"
    assert len(set(operation["fold_ids"])) >= 2, "cross-fitting uses more than one fold"
