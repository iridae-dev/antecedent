"""analyze() identifier/estimator kwargs and nested result schema.

IPW ATE dual: shares structural confounded SCM (true ATE=2) and acceptance band
with Rust `end_to_end_propensity_weighting_recovers_confounded_effect`
(`crates/causal/src/lib.rs`). Cross-language floor: |ate − 2| < 0.4
(Rust unit test uses a tighter 0.3 on its RNG stream).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _confounded_scm(n: int = 800, seed: int = 5):
    """Confounded Z→T, Z→Y, T→Y with structural ATE=2 (Python dual of Rust IPW fixture)."""
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    data = {"t": t, "y": y, "z": z}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    return data, edges


def test_analyze_default_pair_schema_and_fields():
    data, edges = _confounded_scm()
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        bootstrap=10,
        seed=1,
    )
    assert result.identification.method == "backdoor.adjustment"
    assert result.estimate.estimator_id in ("", "linear.adjustment.ate")
    assert result.estimate.overlap_ess is None
    assert result.validation.count >= 0


def test_analyze_propensity_weighting_recovers_ate_and_overlap():
    # Shared dual band with Rust IPW ATE≈2 (see module docstring).
    data, edges = _confounded_scm(n=800, seed=5)
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        identifier="backdoor.adjustment",
        estimator="propensity.weighting",
        bootstrap=10,
        seed=1,
    )
    assert abs(result.ate - 2.0) < 0.4, result.ate
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
def test_analyze_estimate_estimators_smoke(estimator):
    data, edges = _confounded_scm(n=1000, seed=9)
    result = antecedent.analyze(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator=estimator,
        bootstrap=5,
        seed=1,
        refute=False,
    )
    assert np.isfinite(result.ate)


def test_analyze_iv_2sls_smoke():
    rng = random.Random(2)
    n = 800
    z = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float64)
    u = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=np.float64)
    t = (0.8 * z + 0.5 * u + np.array([rng.gauss(0, 0.3) for _ in range(n)]) > 0).astype(np.float64)
    y = 1.5 * t + u + np.array([rng.gauss(0, 0.3) for _ in range(n)], dtype=np.float64)
    result = antecedent.analyze(
        {"t": t, "y": y, "z": z},
        graph=[("z", "t"), ("t", "y")],
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        identifier="iv",
        estimator="iv.2sls",
        bootstrap=5,
        seed=1,
        refute=False,
    )
    assert np.isfinite(result.ate)


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
    return antecedent.analyze(
        _frontdoor_interaction_table(),
        graph=[("t", "m"), ("m", "y")],
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        identifier="frontdoor",
        estimator=estimator,
        bootstrap=0,
        seed=1,
        refute=False,
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
