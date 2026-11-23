"""analyze() identifier/estimator kwargs and nested result schema.

IPW ATE dual: shares structural confounded SCM (true ATE=2) and acceptance band
with Rust `end_to_end_propensity_weighting_recovers_confounded_effect`
(`crates/causal/src/lib.rs`). The adjusted estimators land within 0.05 of 2 on these
seeds; the band is 0.15, well below the 0.66 bias of the unadjusted contrast.
"""

from __future__ import annotations

import random

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _scm import confounded_scm


def test_analyze_default_pair_schema_and_fields():
    data, edges = confounded_scm(n=800, seed=5)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=10),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.identification.method == "backdoor.adjustment"
    assert result.estimate.estimator_id in ("", "linear.adjustment.ate")
    assert result.estimate.overlap_ess is None
    assert result.validation.count >= 0
    # The structural effect is 2; the unadjusted contrast is pulled well above it by z.
    assert abs(result.ate - 2.0) < 0.15, result.ate


def test_analyze_propensity_weighting_recovers_ate_and_overlap():
    # Shared dual band with Rust IPW ATE≈2 (see module docstring).
    data, edges = confounded_scm(n=800, seed=5)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            identifier="backdoor.adjustment",
            estimator="propensity.weighting",
            inference=antecedent.Frequentist(bootstrap=10),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert abs(result.ate - 2.0) < 0.15, result.ate
    assert result.estimate.estimator_id == "propensity.weighting"
    assert result.estimate.overlap_ess is not None
    assert result.estimate.overlap_propensity_min is not None
    assert result.validation.count == 0


@pytest.mark.parametrize(
    "estimator",
    [
        "propensity.stratification",
        "aipw",
        "propensity.matching",
        "distance.matching",
    ],
)
def test_analyze_estimate_estimators_recover_the_confounded_effect(estimator):
    data, edges = confounded_scm(n=1000, seed=9)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            estimator=estimator,
            inference=antecedent.Frequentist(bootstrap=5),
            execution=antecedent.Execution(seed=1),
            refute=False,
        )
    )
    assert abs(result.ate - 2.0) < 0.15, (estimator, result.ate)


def test_bayesian_robust_ate_recovers_the_confounded_effect_of_a_binary_treatment():
    data, edges = confounded_scm(n=800, seed=5)
    result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            estimator=antecedent.Estimator.BAYESIAN_ROBUST_ATE,
            inference=antecedent.Bayesian(backend="conjugate", n_draws=200),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.estimate.estimator_id == "bayesian.robust_ate"
    assert result.posterior is not None and result.posterior.n_draws == 200
    assert abs(result.posterior.effect_mean - 2.0) < 0.15, result.posterior.effect_mean


def test_analyze_iv_2sls_recovers_the_structural_effect():
    rng = random.Random(2)
    n = 800
    z = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float64)
    u = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float64)
    t = (0.8 * z + 0.5 * u + np.array([rng.gauss(0, 0.3) for _ in range(n)]) > 0).astype(np.float64)
    y = 1.5 * t + u + np.array([rng.gauss(0, 0.3) for _ in range(n)], dtype=np.float64)
    result = require_success(
        antecedent.analyze(
            {"t": t, "y": y, "z": z},
            graph=[("z", "t"), ("t", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            identifier="iv",
            estimator="iv.2sls",
            inference=antecedent.Frequentist(bootstrap=5),
            execution=antecedent.Execution(seed=1),
            refute=False,
        )
    )
    # Structural effect 1.5; the confounder u biases the naive contrast, 2SLS removes it.
    assert abs(result.ate - 1.5) < 0.15, result.ate


def _frontdoor_interaction_table() -> dict[str, np.ndarray]:
    """Exact 4000-row table of U~Bern(.5), P(T=1|U)=.1+.5U, P(M=1|T)=.1+.7T,
    P(Y=1|M,U)=.05+.9MU with U dropped; the enumerated effect is 0.7*0.9*0.5 = 0.315."""
    rows: list[tuple[float, float, float]] = []
    for u in (0.0, 1.0):
        for t in (0.0, 1.0):
            pt = 0.1 + 0.5 * u if t else 0.9 - 0.5 * u
            for m in (0.0, 1.0):
                pm = 0.1 + 0.7 * t if m else 0.9 - 0.7 * t
                for y in (0.0, 1.0):
                    py1 = 0.05 + 0.9 * m * u
                    py = py1 if y else 1.0 - py1
                    rows += [(t, m, y)] * round(0.5 * pt * pm * py * 4000)
    assert len(rows) == 4000
    arr = np.array(rows, dtype=np.float64)
    return {"t": arr[:, 0].copy(), "m": arr[:, 1].copy(), "y": arr[:, 2].copy()}


def _analyze_frontdoor(estimator: str):
    return require_success(
        antecedent.analyze(
            _frontdoor_interaction_table(),
            graph=[("t", "m"), ("m", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            identifier="frontdoor",
            estimator=estimator,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
            refute=False,
        )
    )


def test_analyze_frontdoor_functional_matches_enumerated_truth():
    result = _analyze_frontdoor(str(antecedent.Estimator.FRONTDOOR_FUNCTIONAL))
    assert abs(result.ate - 0.315) < 1e-12
    assert result.estimate.se_analytic > 0.0
    assert any("frontdoor.functional.saturated_cells" in a for a in result.assumptions or [])


def test_analyze_frontdoor_linear_two_stage_converges_to_its_own_limit():
    # The latent modifies the mediator effect, so the product of coefficients lands on
    # 0.3631 (variance-weighted within-arm slope times 0.7), not the effect 0.315.
    result = _analyze_frontdoor(str(antecedent.Estimator.FRONTDOOR_LINEAR_TWO_STAGE))
    assert abs(result.ate - 0.3631) < 1e-3
    assert any("frontdoor.linear_path_product" in a for a in result.assumptions or [])


def test_an_identifier_the_estimator_cannot_use_is_refused_by_the_pair_allowlist():
    data, edges = confounded_scm(n=300, seed=5)
    with pytest.raises(antecedent.errors.CausalError) as caught:
        require_success(
            antecedent.analyze(
                data,
                graph=edges,
                query=antecedent.AverageEffect(treatment="t", outcome="y"),
                identifier="frontdoor",
                estimator="propensity.weighting",
                inference=antecedent.Frequentist(bootstrap=0),
                execution=antecedent.Execution(seed=1),
                refute=False,
            )
        )
    assert caught.value.reason_code == "strategy_incompatible"
    assert "frontdoor" in str(caught.value) and "propensity.weighting" in str(caught.value)


def test_a_frontdoor_identifier_over_a_direct_edge_is_refused_as_not_identified():
    # The mediator does not intercept the direct treatment-outcome edge, so the front-door
    # criterion fails and the estimator is never reached.
    with pytest.raises(antecedent.errors.EffectNotIdentified) as caught:
        require_success(
            antecedent.analyze(
                _frontdoor_interaction_table(),
                graph=[("t", "m"), ("m", "y"), ("t", "y")],
                query=antecedent.AverageEffect(treatment="t", outcome="y"),
                identifier="frontdoor",
                estimator=str(antecedent.Estimator.FRONTDOOR_FUNCTIONAL),
                inference=antecedent.Frequentist(bootstrap=0),
                execution=antecedent.Execution(seed=1),
                refute=False,
            )
        )
    assert caught.value.reason_code == "effect_not_identified"


def test_lagged_prediction_is_labelled_conditional_not_interventional():
    # U_t drives X_t and Y_{t+1}; X has no effect on Y, so E[Y | do(X=1)] = 0 while the
    # association E[Y_t | X_{t-1}=1] = Var(U)/Var(X) = 0.8. The helper can only see the latter.
    from antecedent import model

    rng = np.random.default_rng(11)
    n = 20_000
    u = rng.normal(size=n)
    x = u + 0.5 * rng.normal(size=n)
    y = np.concatenate([[0.0], u[:-1]]) + 0.5 * rng.normal(size=n)
    summary = model.predict_conditional_summary(["x", "y"], [x, y], "y", "x", level=1.0)
    assert abs(summary.mean_prediction - 0.8) < 0.03
    assert not hasattr(model, "predict_intervened_summary")
    with pytest.raises(Exception, match="finite"):
        model.predict_conditional_summary(["x", "y"], [x, y], "y", "x", level=float("nan"))


def test_a_propensity_weighting_study_publishes_its_adjustment_rows_and_uncertainty_plan():
    data, edges = confounded_scm(n=800, seed=5)
    prepared = antecedent.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        identifier="backdoor.adjustment",
        estimator="propensity.weighting",
        inference=antecedent.Frequentist(bootstrap=10),
        execution=antecedent.Execution(seed=1),
    )
    outcome = antecedent.estimate(prepared)
    require_success(outcome)
    operation = outcome.study.plan_view().operation
    assert operation["kind"] == "propensity"
    assert operation["estimator"] == "propensity.weighting"
    assert operation["adjustment_set"] == ["z"], "z is the only back-door confounder"
    assert operation["bootstrap_replicates"] == 10
    assert operation["uncertainty"]
    assert len(operation["source_rows"]) == 800, "every complete row is planned once"
