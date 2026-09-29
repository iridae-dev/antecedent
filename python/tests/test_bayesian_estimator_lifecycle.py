"""Python selection and result lifecycle for the 2.1 Bayesian estimators."""

from __future__ import annotations

import antecedent
import numpy as np


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
    result = antecedent.analyze(
        {"t": treatment, "y": outcome, "x": x},
        graph=[("t", "y"), ("x", "t"), ("x", "y")],
        query=antecedent.AverageEffect("t", "y"),
        estimator=antecedent.Estimator.BAYESIAN_BASIS_GCOMP,
        inference=antecedent.Bayesian(backend="conjugate", n_draws=256),
        refute=False,
        seed=19,
        return_posterior_artifact=True,
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
    assert result.posterior.artifact is not None
    assert any(
        "zero-centered Gaussian prior" in assumption for assumption in result.assumptions or []
    )

    artifact = antecedent.inference.decode_posterior_artifact(result.posterior.artifact)
    assert artifact.backend_id == result.posterior.backend
    assert "ate" in artifact.quantity_names
    assert artifact.n_draws == 256
    assert artifact.q025[artifact.quantity_names.index("ate")] == result.posterior.q025
    assert artifact.q975[artifact.quantity_names.index("ate")] == result.posterior.q975
