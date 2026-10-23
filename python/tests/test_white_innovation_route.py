"""The declared white-innovation temporal posterior through ``antecedent.analyze``.

``Bayesian(white_innovation=(max_lag, level))`` declares serially independent innovations of a
single-step temporal linear effect and swaps the tempered (generalized) posterior for the
untempered full-likelihood posterior of the same effect. The premise is declared, never inferred;
a Ljung-Box test of the residuals can refute it. References are numpy least squares and a
numpy Ljung-Box statistic. The route is an unlicensed estimator with calibration unmeasured.
"""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalUnsupportedError

from _refusal import assert_registered_refusal

N = 400
DRAWS = 4000
ID = "bayesian.temporal.white_innovation_linear"


def series(rho_x: float, rho_e: float, seed: int):
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    y = np.zeros(N)
    previous = e = 0.0
    for t in range(N):
        previous = rho_x * previous + rng.standard_normal()
        x[t] = previous
        e = rho_e * e + rng.standard_normal()
        y[t] = (0.8 * x[t - 1] if t > 0 else 0.0) + e
    return x, y


def ols(x, y):
    design = np.column_stack([np.ones(N - 1), x[:-1]])
    target = y[1:]
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    residuals = target - design @ beta
    sigma2 = residuals @ residuals / (N - 1 - 2)
    se = math.sqrt(sigma2 * np.linalg.inv(design.T @ design)[1, 1])
    return beta[1], se, residuals


def ljung_box_p(residuals, lags: int) -> float:
    n = len(residuals)
    total = float(residuals @ residuals)
    q = sum((residuals[k:] @ residuals[:-k] / total) ** 2 / (n - k) for k in range(1, lags + 1))
    half = n * (n + 2) * q / 2.0
    return math.exp(-half) * sum(half**k / math.factorial(k) for k in range(lags // 2))


def run(x, y, **bayes):
    return antecedent.analyze(
        {"x": x, "y": y},
        graph=[("x", 1, "y", 0)],
        query=antecedent.PulseEffect(
            treatment="x", outcome="y", treatment_lag=1, horizon_steps=1, active_level=1.0
        ),
        inference=antecedent.Bayesian(
            n_draws=DRAWS, prior_scale=1.0e3, backend="conjugate", **bayes
        ),
        refute=False,
        bootstrap=0,
        seed=11,
    )


def test_a_declared_premise_gives_the_untempered_least_squares_posterior() -> None:
    x, y = series(0.5, 0.0, 2)
    beta, se, residuals = ols(x, y)
    assert ljung_box_p(residuals, 6) > 0.01
    result = run(x, y, white_innovation=(6, 0.01))
    assert result.estimate.estimator_id == ID
    assert result.posterior is not None
    assert result.posterior.effect_mean == pytest.approx(beta, abs=3 * se / math.sqrt(DRAWS) + 2e-3)
    assert result.posterior.effect_sd / se == pytest.approx(1.0, abs=0.06)
    # An unlicensed construction: the cell's license does not carry to it.
    assert result.evidence_status != "licensed"


def test_without_the_declaration_the_tempered_posterior_keeps_its_id() -> None:
    x, y = series(0.5, 0.0, 2)
    result = run(x, y)
    assert result.estimate.estimator_id == "bayesian.temporal.gcomp"


def test_autocorrelated_innovations_refuse_the_declaration_and_name_the_next_attempt() -> None:
    x, y = series(0.8, 0.8, 1)
    _, _, residuals = ols(x, y)
    assert ljung_box_p(residuals, 6) < 1e-6
    with pytest.raises(CausalUnsupportedError) as info:
        run(x, y, white_innovation=(6, 0.01))
    text = str(info.value)
    assert "white_innovation.residual_autocorrelation" in text
    assert "tempered posterior" in text and "Next attempt" in text
    assert_registered_refusal(info.value)
    assert run(x, y).posterior is not None


def test_a_route_that_does_not_read_the_premise_refuses_it() -> None:
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 200).astype(float)
    y = 1.0 + 2.0 * t + rng.standard_normal(200)
    with pytest.raises(CausalUnsupportedError) as info:
        antecedent.analyze(
            {"t": t, "y": y},
            graph=[("t", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(white_innovation=(6, 0.01)),
            refute=False,
            bootstrap=0,
        )
    assert info.value.reason_code == "option_not_applicable"
    assert "Next attempt" in str(info.value)


def test_the_declaration_is_validated() -> None:
    x, y = series(0.5, 0.0, 2)
    for bad in ((0, 0.01), (6, 0.0), (6, 1.0)):
        with pytest.raises(antecedent.errors.CausalValueError):
            run(x, y, white_innovation=bad)


def run_curve(x, y, horizons=(1,), **bayes):
    return antecedent.analyze(
        {"x": x, "y": y},
        graph=[("x", 1, "y", 0)],
        query=antecedent.ResponseCurve(
            "x", "y", grid=[0.0, 1.0, 2.0], horizons=list(horizons), policy="pulse", treatment_lag=1
        ),
        inference=antecedent.Bayesian(
            n_draws=DRAWS, prior_scale=1.0e3, backend="conjugate", **bayes
        ),
        refute=False,
        bootstrap=0,
        seed=12,
    )


def test_a_declared_premise_gives_the_untempered_curve_at_one_step() -> None:
    x, y = series(0.5, 0.0, 2)
    beta, se, _ = ols(x, y)
    declared = run_curve(x, y, white_innovation=(6, 0.01))
    values = np.asarray([row[0] for row in declared.response.values], dtype=float)
    assert values[1] - values[0] == pytest.approx(beta, abs=3.0 * se / math.sqrt(DRAWS) + 5e-3)
    ids = {d.id for d in declared.support.diagnostics}
    assert "response.temporal_bayesian.white_innovation" in ids
    assert "response.temporal_bayesian.tempering" not in ids
    default_ids = {d.id for d in run_curve(x, y).support.diagnostics}
    assert "response.temporal_bayesian.tempering" in default_ids
    assert "response.temporal_bayesian.white_innovation" not in default_ids


def test_the_curve_refuses_the_premise_on_autocorrelation_and_beyond_one_step() -> None:
    x, y = series(0.8, 0.8, 1)
    with pytest.raises(CausalUnsupportedError) as info:
        run_curve(x, y, white_innovation=(6, 0.01))
    text = str(info.value)
    assert "tempered posterior" in text and "Next attempt" in text
    assert run_curve(x, y).response is not None
    x, y = series(0.5, 0.0, 2)
    with pytest.raises(CausalUnsupportedError) as multi:
        run_curve(x, y, horizons=(1, 2), white_innovation=(6, 0.01))
    assert multi.value.reason_code == "option_not_applicable"
    assert "Next attempt" in str(multi.value)
