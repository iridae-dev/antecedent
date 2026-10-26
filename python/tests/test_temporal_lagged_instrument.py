"""The lagged-instrument effect through ``antecedent.analyze``.

``z_{t-2} -> x_{t-1} -> y_t`` with a confounder ``u`` of treatment and outcome: back-door
adjustment cannot identify the effect, a lagged cause of the treatment can. The identifier
certifies the instrument on the unfolded window graph, never from a time column; relevance is
checked by the first-stage F. References are numpy: the closed-form Wald ratio with its
homoskedastic standard error, and a numpy Gaussian reduced-form posterior for the ratio.
Both estimators are unlicensed with calibration unmeasured.
"""

from __future__ import annotations

import math

import antecedent
import numpy as np
import pytest
from antecedent import errors
from antecedent.errors import CausalUnsupportedError
from antecedent.outcomes import require_success

N = 800
GRAPH = [("z", 2, "x", 1), ("u", 1, "x", 1), ("u", 1, "y", 0), ("x", 1, "y", 0)]


def make_series(seed: int = 31):
    rng = np.random.default_rng(seed)
    z, u = rng.standard_normal(N), rng.standard_normal(N)
    x = np.zeros(N)
    y = np.zeros(N)
    x[1:] = 0.9 * z[:-1] + 0.8 * u[1:] + 0.3 * rng.standard_normal(N - 1)
    x[0] = 0.8 * u[0] + 0.3 * rng.standard_normal()
    y[1:] = 2.0 * x[:-1] + u[:-1] + 0.3 * rng.standard_normal(N - 1)
    return {"x": x, "y": y, "z": z, "u": u}


def aligned(series):
    """Rows i = 1..N-2 pairing z_{i-1}, x_i and y_{i+1}."""
    return series["z"][: N - 2], series["x"][1 : N - 1], series["y"][2:]


def wald(series):
    z, x, y = aligned(series)
    n = len(z)
    zc = z - z.mean()
    effect = (zc @ (y - y.mean())) / (zc @ (x - x.mean()))
    slope = (zc @ x) / (zc @ zc)
    first_stage = x - x.mean() - slope * zc
    structural = y - (y.mean() - effect * x.mean()) - effect * x
    se = math.sqrt(structural @ structural / (n - 2) * (zc @ zc) / (zc @ (x - x.mean())) ** 2)
    f_stat = slope**2 * (zc @ zc) / (first_stage @ first_stage / (n - 2))
    return effect, se, f_stat


def ratio_posterior_reference(series, prior_sd, draws=400_000, seed=0):
    z, t, y = aligned(series)
    n = len(z)
    x1 = np.column_stack([np.ones(n), z])
    rt = t - x1 @ np.linalg.lstsq(x1, t, rcond=None)[0]
    ry = y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]
    df = n - 2
    vt, vy, c = max(rt @ rt / df, 1e-8), max(ry @ ry / df, 1e-8), rt @ ry / df
    weight = np.linalg.inv(np.array([[vt, c], [c, vy]]))
    precision, rhs = np.eye(4) / prior_sd**2, np.zeros(4)
    for zi, ti, yi in zip(z, t, y, strict=True):
        design = np.array([[1.0, zi, 0.0, 0.0], [0.0, 0.0, 1.0, zi]])
        precision += design.T @ weight @ design
        rhs += design.T @ weight @ np.array([ti, yi])
    covariance = np.linalg.inv(precision)
    sample = np.random.default_rng(seed).multivariate_normal(covariance @ rhs, covariance, draws)
    return sample[:, 3] / sample[:, 1]


def run(series, graph=GRAPH, **kwargs):
    return require_success(
        antecedent.analyze(
            series,
            graph=graph,
            query=antecedent.PulseEffect(
                treatment="x", outcome="y", treatment_lag=1, horizon_steps=1, active_level=1.0
            ),
            identifier=antecedent.Identifier.TEMPORAL_IV_UNFOLDED,
            refute=False,
            bootstrap=0,
            seed=2,
            **kwargs,
        )
    )


def test_the_frequentist_route_matches_the_wald_ratio() -> None:
    series = make_series()
    effect, se, f_stat = wald(series)
    assert f_stat > 10.0
    result = run(series, estimator=antecedent.Estimator.TEMPORAL_LAGGED_INSTRUMENT)
    assert result.estimate.estimator_id == "temporal.lagged_instrument"
    assert result.ate == pytest.approx(effect, abs=1e-8)
    assert result.estimate.se_analytic == pytest.approx(se, abs=1e-8)
    assert abs(result.ate - 2.0) < 3 * se
    assert result.evidence_status != "licensed"


def test_the_bayesian_route_matches_a_numpy_reduced_form_posterior() -> None:
    series = make_series()
    effect, se, _ = wald(series)
    for prior_sd in (100.0, 0.05):
        result = run(
            series,
            estimator=antecedent.Estimator.BAYESIAN_TEMPORAL_LAGGED_INSTRUMENT,
            inference=antecedent.Bayesian(backend="conjugate", prior_scale=prior_sd, n_draws=4000),
        )
        reference = ratio_posterior_reference(series, prior_sd)
        spread = float(np.std(reference))
        assert result.estimate.estimator_id == "bayesian.temporal.lagged_instrument"
        assert result.posterior.effect_mean == pytest.approx(
            float(np.mean(reference)), abs=max(0.12 * spread, 1e-3)
        )
    assert abs(result.posterior.effect_mean - effect) > 0.5  # the tight prior moved the ratio


def test_a_lagged_confounder_of_the_instrument_is_refused_with_its_next_attempt() -> None:
    series = make_series()
    confounded = [*GRAPH, ("u", 2, "z", 2), ("u", 2, "y", 0)]
    with pytest.raises(errors.CausalError) as info:
        run(series, graph=confounded, estimator=antecedent.Estimator.TEMPORAL_LAGGED_INSTRUMENT)
    text = str(info.value)
    assert "exogeneity" in text and "Next attempt" in text


def test_the_identifier_and_the_estimators_go_together() -> None:
    series = make_series()
    with pytest.raises(CausalUnsupportedError) as info:
        require_success(
            antecedent.analyze(
                series,
                graph=GRAPH,
                query=antecedent.PulseEffect(
                    treatment="x", outcome="y", treatment_lag=1, horizon_steps=1, active_level=1.0
                ),
                estimator=antecedent.Estimator.TEMPORAL_LAGGED_INSTRUMENT,
                refute=False,
                bootstrap=0,
            )
        )
    assert info.value.reason_code == "option_not_applicable"
