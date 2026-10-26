"""The temporal GLM link contrast through ``antecedent.analyze`` over a panel.

Each unit contributes one decision time (its earliest row with a full lag window), so rows are
independent subjects. References are computed here with numpy from the raw first rows: the Newton
logistic maximum-likelihood fit, the probability-scale contrast averaged over the observed lag,
its Fisher delta-method standard error, and for the Bayesian route the Newton mode and a Laplace
sample under the stated Gaussian prior. Both routes are unlicensed with calibration unmeasured.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.data import panel
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal

UNITS = 300
STEPS = 3
GRAPH = [("c", 1, "x", 1), ("c", 1, "y", 0), ("x", 1, "y", 0)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_units(seed: int = 5):
    rng = np.random.default_rng(seed)
    units = []
    for _ in range(UNITS):
        c = rng.standard_normal(STEPS)
        x = (rng.random(STEPS) < sigmoid(0.5 * c)).astype(float)
        y = np.zeros(STEPS)
        for t in range(1, STEPS):
            y[t] = float(rng.random() < sigmoid(-0.5 + x[t - 1] + 0.8 * c[t - 1]))
        units.append({"x": x, "c": c, "y": y})
    return units


def first_rows(units):
    rows = np.array([[1.0, u["x"][0], u["c"][0]] for u in units])
    return rows, np.array([u["y"][1] for u in units])


def newton(rows, y, prec):
    beta = np.zeros(3)
    for _ in range(100):
        p = sigmoid(rows @ beta)
        gradient = rows.T @ (y - p) - prec * beta
        hessian = rows.T @ (rows * (p * (1 - p))[:, None]) + prec * np.eye(3)
        beta = beta + np.linalg.solve(hessian, gradient)
    p = sigmoid(rows @ beta)
    return beta, rows.T @ (rows * (p * (1 - p))[:, None]) + prec * np.eye(3)


def contrast(rows, beta):
    lag = rows[:, 2] * beta[..., 2:3] if beta.ndim == 2 else rows[:, 2] * beta[2]
    return (sigmoid(beta[..., 0:1] + beta[..., 1:2] + lag) - sigmoid(beta[..., 0:1] + lag)).mean(
        axis=-1
    )


def delta_se(rows, beta, information):
    lag = rows[:, 2] * beta[2]
    p1, p0 = sigmoid(beta[0] + beta[1] + lag), sigmoid(beta[0] + lag)
    d1, d0 = p1 * (1 - p1), p0 * (1 - p0)
    gradient = np.array([(d1 - d0).mean(), d1.mean(), ((d1 - d0) * rows[:, 2]).mean()])
    return float(np.sqrt(gradient @ np.linalg.solve(information, gradient)))


def run(units, **kwargs):
    return require_success(
        antecedent.analyze(
            panel(units),
            graph=GRAPH,
            query=antecedent.PulseEffect(
                treatment="x", outcome="y", treatment_lag=1, horizon_steps=1, active_level=1.0
            ),
            refute=False,
            bootstrap=0,
            seed=4,
            **kwargs,
        )
    )


def test_the_frequentist_contrast_matches_newton_and_the_delta_method() -> None:
    units = make_units()
    rows, y = first_rows(units)
    beta, information = newton(rows, y, 0.0)
    result = run(units, estimator=antecedent.Estimator.TEMPORAL_GLM_LINK_CONTRAST)
    assert result.estimate.estimator_id == "temporal.glm_link_contrast"
    assert result.ate == pytest.approx(float(contrast(rows, beta)), abs=1e-8)
    assert result.estimate.se_analytic == pytest.approx(delta_se(rows, beta, information), abs=1e-8)
    assert abs(result.ate - beta[1]) > 0.1  # the contrast is not the link-scale coefficient
    assert result.evidence_status != "licensed"


def test_the_bayesian_contrast_matches_the_mode_and_a_laplace_sample() -> None:
    units = make_units()
    rows, y = first_rows(units)
    mode, hessian = newton(rows, y, 1.0)
    sample = np.random.default_rng(1).multivariate_normal(mode, np.linalg.inv(hessian), 200_000)
    result = run(
        units,
        estimator=antecedent.Estimator.BAYESIAN_TEMPORAL_GLM_LINK_CONTRAST,
        inference=antecedent.Bayesian(
            backend="laplace", likelihood="logit", prior_scale=1.0, n_draws=4000
        ),
    )
    assert result.estimate.estimator_id == "bayesian.temporal.glm_link_contrast"
    assert result.posterior is not None
    assert result.posterior.effect_mean == pytest.approx(
        float(contrast(rows, sample).mean()), abs=0.01
    )
    assert result.posterior.n_draws == 4000
    assert abs(result.posterior.effect_mean - float(contrast(rows, mode[None, :])[0])) < 0.02


def test_a_single_dependent_series_is_refused_with_its_next_attempt() -> None:
    rng = np.random.default_rng(2)
    series = {
        "x": (rng.random(60) < 0.5).astype(float),
        "c": rng.standard_normal(60),
        "y": (rng.random(60) < 0.5).astype(float),
    }
    with pytest.raises(CausalUnsupportedError) as info:
        require_success(
            antecedent.analyze(
                series,
                graph=GRAPH,
                query=antecedent.PulseEffect(
                    treatment="x", outcome="y", treatment_lag=1, horizon_steps=1, active_level=1.0
                ),
                estimator=antecedent.Estimator.TEMPORAL_GLM_LINK_CONTRAST,
                refute=False,
                bootstrap=0,
            )
        )
    text = str(info.value)
    assert info.value.reason_code == "route_not_supported"
    assert "single dependent series" in text and "Next attempt" in text
    assert_registered_refusal(info.value)


def test_the_estimators_name_their_inference_mode() -> None:
    units = make_units()
    with pytest.raises(CausalUnsupportedError) as info:
        run(units, estimator=antecedent.Estimator.BAYESIAN_TEMPORAL_GLM_LINK_CONTRAST)
    assert info.value.reason_code == "estimator_inference_mismatch"
