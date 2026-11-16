"""Exposure-error corrections through the Python facade.

Every asserted value has an oracle computed here in pure Python or numpy, never by native code:

* exposure misclassification: exact recorded shares ``p_y = Se_y pi_y + (1 - Sp_y)(1 - pi_y)`` of
  planted true shares, inverted by hand, and the delta-method variance as a numerical gradient of
  the explicit corrected log odds ratio;
* soft exposures: least squares on ``(1, q, z)`` with ``numpy.linalg.lstsq``, the HC0 sandwich
  written out, and Rubin's rules over per-draw least-squares fits;
* reliability: bin means computed by hand;
* detection: the planted prevalence recovered from detections ``pi d(c)`` per unit, the bounds of
  an envelope by hand;
* repeat codings: the planted latent-class law recovered from rounded exact pattern counts.

Tolerances are written in the test: ``1e-9`` for exact algebra and the stated amounts for planted
recovery.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import measurement_exposure as mx
from antecedent.errors import CausalValueError
from antecedent.measurement_models import BinaryRates, MeasurementModelRefusal

from _refusal import assert_registered_refusal

CASES = dict(exposed_cases=480, cases=1000, exposed_controls=180, controls=1000)


def rates(se, sp, vse=0.0, vsp=0.0):
    return BinaryRates(se, sp, vse, vsp)


def log_or(p1, p0, r):
    pi1 = (p1 + r[1] - 1) / (r[0] + r[1] - 1)
    pi0 = (p0 + r[3] - 1) / (r[2] + r[3] - 1)
    return math.log(pi1 / (1 - pi1)) - math.log(pi0 / (1 - pi0))


def test_outcome_specific_rates_recover_the_planted_odds_ratio():
    fit = mx.correct_exposure_misclassification(
        **CASES,
        error=mx.ExposureError.by_outcome(cases=rates(0.9, 0.8), controls=rates(0.7, 0.95)),
    )
    assert fit.prevalence_cases == pytest.approx(0.4, abs=1e-12)
    assert fit.prevalence_controls == pytest.approx(0.2, abs=1e-12)
    assert fit.log_odds_ratio == pytest.approx(math.log(8 / 3), abs=1e-12)
    assert fit.odds_ratio == pytest.approx(8 / 3, abs=1e-12)
    assert fit.differential
    assert fit.parameter_uncertainty == "declared_exact"
    assert fit.interval.calibration["status"] == "not_measured"
    # One pair of rates misses the truth when the error differs by outcome.
    shared = mx.correct_exposure_misclassification(
        **CASES, error=mx.ExposureError.shared(rates(0.8, 0.9))
    )
    assert abs(shared.log_odds_ratio - math.log(8 / 3)) > 0.3


def test_the_rate_variance_equals_a_numerical_gradient():
    variances = [4e-4, 9e-4, 6e-4, 2e-4]
    fit = mx.correct_exposure_misclassification(
        **CASES,
        error=mx.ExposureError.by_outcome(
            cases=rates(0.9, 0.8, variances[0], variances[1]),
            controls=rates(0.7, 0.95, variances[2], variances[3]),
        ),
    )
    base = [0.9, 0.8, 0.7, 0.95]
    h = 1e-6
    expected = 0.0
    for slot, variance in enumerate(variances):
        up, down = list(base), list(base)
        up[slot] += h
        down[slot] -= h
        d = (log_or(0.48, 0.18, up) - log_or(0.48, 0.18, down)) / (2 * h)
        expected += d * d * variance
    by_name = {c.name: c.variance for c in fit.variance_components}
    assert by_name["sensitivity_specificity_uncertainty"] == pytest.approx(expected, rel=1e-6)
    assert fit.parameter_uncertainty == "propagated"
    assert fit.standard_error == pytest.approx(math.sqrt(sum(by_name.values())), rel=1e-12)


def test_validation_counts_propagate_their_binomial_variance():
    fit = mx.correct_exposure_misclassification(
        **CASES,
        error=mx.ExposureError.by_outcome(
            cases=mx.ValidationCounts(90, 10, 80, 20),
            controls=mx.ValidationCounts(70, 30, 190, 10),
        ),
    )
    by_name = {c.name: c.variance for c in fit.variance_components}
    assert by_name["sensitivity_specificity_uncertainty"] > 0.0
    assert fit.parameter_uncertainty == "propagated"


@pytest.mark.parametrize("weak", [(0.55, 0.5), (0.5, 0.5)])
def test_rates_that_do_not_identify_the_exposure_refuse(weak):
    with pytest.raises(MeasurementModelRefusal) as caught:
        mx.correct_exposure_misclassification(
            **CASES,
            error=mx.ExposureError.by_outcome(cases=rates(0.9, 0.8), controls=rates(*weak)),
        )
    assert_registered_refusal(caught.value)
    assert caught.value.reason_code == "measurement_not_identified"
    assert caught.value.detail == reason_details.measurement_model.YOUDEN_TOO_SMALL


def test_error_must_be_declared_consistently():
    with pytest.raises(CausalValueError):
        mx.ExposureError.by_outcome(cases=rates(0.9, 0.8), controls=mx.ValidationCounts(1, 1, 1, 1))


# --------------------------------------------------------------------------- soft exposures


def soft_data(n=40):
    q = np.linspace(0.05, 0.95, n)
    z = np.cos(np.arange(n) * 1.3)
    noise = np.sin(np.arange(n) * 2.1)
    return 1.0 + 3.0 * q + 0.5 * z + 2.0 * noise, q, z


def test_posterior_mean_substitution_is_least_squares_with_an_hc0_sandwich():
    y, q, z = soft_data()
    fit = mx.soft_exposure_effect(y, q, z)
    design = np.column_stack([np.ones_like(q), q, z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    np.testing.assert_allclose(fit.coefficients, beta, atol=1e-9)
    resid = y - design @ beta
    bread = np.linalg.inv(design.T @ design)
    sandwich = bread @ (design.T * resid**2) @ design @ bread
    np.testing.assert_allclose(fit.covariance, sandwich, atol=1e-9)
    by_name = {c.name: c.variance for c in fit.variance_components}
    assert by_name["exposure_uncertainty"] >= 0.0
    assert by_name["exposure_uncertainty"] + by_name["sampling_given_exposure"] == pytest.approx(
        sandwich[1, 1], rel=1e-9
    )
    assert fit.mean_posterior_variance == pytest.approx(float(np.mean(q * (1 - q))), abs=1e-12)


def test_supplied_draws_are_combined_by_rubins_rules():
    y, q, z = soft_data()
    rng = np.random.default_rng(3)
    draws = (rng.random((4, len(y))) < q).astype(float)
    fit = mx.soft_exposure_effect(y, q, z, method="supplied_draws", draws=draws)
    betas, withins = [], []
    for draw in draws:
        design = np.column_stack([np.ones_like(q), draw, z])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        bread = np.linalg.inv(design.T @ design)
        betas.append(beta[1])
        withins.append((bread @ (design.T * resid**2) @ design @ bread)[1, 1])
    within, between = np.mean(withins), np.var(betas, ddof=1)
    assert fit.coefficient == pytest.approx(np.mean(betas), abs=1e-9)
    assert fit.standard_error**2 == pytest.approx(within + (1 + 1 / 4) * between, rel=1e-9)
    assert fit.imputations == 4
    with pytest.raises(CausalValueError):
        mx.soft_exposure_effect(y, q, z, method="supplied_draws", draws=draws[:, :-1])


def test_generated_imputations_are_reproducible_from_the_seed():
    y, q, z = soft_data(120)
    first = mx.soft_exposure_effect(y, q, z, method="multiple_imputation", draw_count=8, seed=5)
    again = mx.soft_exposure_effect(y, q, z, method="multiple_imputation", draw_count=8, seed=5)
    assert first.coefficient == again.coefficient
    assert first.imputations == 8


def test_reliability_bins_match_hand_computed_means_and_never_gate():
    q = [0.05, 0.1, 0.55, 0.6, 0.95]
    codings = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    report = mx.soft_exposure_reliability(q, codings, bins=2)
    assert report.gates is False
    low, high = report.bins
    assert low.units == 2
    assert low.mean_expected == pytest.approx(0.075, abs=1e-12)
    assert low.mean_observed == pytest.approx(0.125, abs=1e-12)
    assert low.gap == pytest.approx(0.05, abs=1e-12)
    assert high.mean_observed == pytest.approx((0.75 + 0.75 + 1.0) / 3, abs=1e-12)
    assert report.max_abs_gap == pytest.approx(max(abs(b.gap) for b in report.bins), abs=1e-12)
    with pytest.raises(CausalValueError):
        mx.soft_exposure_reliability(q, codings[:-1])


# --------------------------------------------------------------------------- detection


def detections(per, hits):
    detected = [k < h for h in hits for k in range(per)]
    covariate = [float(c) for c, _ in enumerate(hits) for _ in range(per)]
    return detected, covariate


def test_a_declared_detection_function_recovers_the_planted_prevalence():
    detected, covariate = detections(1000, [120, 180, 240, 300])
    fit = mx.detection_dependent_prevalence(
        detected, covariate, mx.TableDetection([0.0, 3.0], [0.4, 1.0])
    )
    assert fit.kind == "point"
    assert fit.prevalence == pytest.approx(0.3, abs=1e-12)
    assert fit.observed_rate == pytest.approx(0.21, abs=1e-12)
    by_hand = sum(1000 * (0.3 * d) * (1 - 0.3 * d) for d in (0.4, 0.6, 0.8, 1.0)) / 2800**2
    by_name = {c.name: c.variance for c in fit.variance_components}
    assert by_name["sampling_of_detections"] == pytest.approx(by_hand, rel=1e-12)
    assert fit.interval is not None and fit.interval.lower < 0.3 < fit.interval.upper


def test_a_partially_declared_monotone_function_gives_sharp_bounds():
    detected, covariate = detections(1000, [120, 180, 240, 300])
    fit = mx.detection_dependent_prevalence(
        detected,
        covariate,
        mx.MonotoneDetection("nondecreasing", [(0.0, 0.3, 0.5), (3.0, 0.9, 1.0)]),
    )
    assert fit.kind == "bounds"
    assert fit.lower == pytest.approx(840 / 3500, abs=1e-12)
    assert fit.upper == pytest.approx(840 / 1800, abs=1e-12)
    assert fit.lower < 0.3 < fit.upper
    assert fit.prevalence is None


def test_detection_that_cannot_identify_the_prevalence_refuses():
    detected, covariate = detections(100, [10, 10, 10, 10])
    with pytest.raises(MeasurementModelRefusal) as caught:
        mx.detection_dependent_prevalence(
            detected, covariate, mx.TableDetection([0.0, 3.0], [0.05, 0.08])
        )
    assert caught.value.reason_code == "measurement_not_identified"
    assert caught.value.detail == reason_details.measurement_model.YOUDEN_TOO_SMALL


# --------------------------------------------------------------------------- repeat codings


def pattern_rows(n, pi, theta, categories):
    rows = []
    k = len(theta)
    for pattern in itertools.product(range(categories), repeat=k):
        p = sum(
            pi[level] * np.prod([theta[c][level][pattern[c]] for c in range(k)])
            for level in range(len(pi))
        )
        rows.extend([pattern] * round(n * p))
    return np.array(rows, dtype=np.int64)


def test_three_binary_codings_recover_the_planted_latent_class_model():
    se, sp = [0.9, 0.8, 0.85], [0.95, 0.9, 0.85]
    theta = [[[sp[k], 1 - sp[k]], [1 - se[k], se[k]]] for k in range(3)]
    rows = pattern_rows(30000, [0.7, 0.3], theta, 2)
    fit = mx.repeat_codings_latent_class(rows, levels=2)
    assert fit.kruskal_identified
    assert fit.prevalence[1] == pytest.approx(0.3, abs=3e-3)
    for k in range(3):
        assert fit.response_of(k, 1, 1) == pytest.approx(se[k], abs=5e-3)
        assert fit.response_of(k, 0, 0) == pytest.approx(sp[k], abs=5e-3)
    assert fit.prevalence_standard_error is not None
    assert fit.posterior.shape == (len(rows), 2)
    np.testing.assert_allclose(fit.posterior.sum(axis=1), 1.0, atol=1e-9)
    assert fit.probabilities().shape == (len(rows),)


def test_two_codings_need_declared_constraints():
    law = [[0.95, 0.05], [0.1, 0.9]]
    rows = pattern_rows(40000, [0.7, 0.3], [law, law], 2)
    for kwargs in (
        {},
        {"exchangeable": True},
        {"lowest_level": [0.95, 0.05]},
    ):
        with pytest.raises(MeasurementModelRefusal) as caught:
            mx.repeat_codings_latent_class(rows, levels=2, **kwargs)
        assert caught.value.reason_code == "measurement_not_identified"
        assert caught.value.detail == reason_details.measurement_model.INSUFFICIENT_REPLICATES
    fit = mx.repeat_codings_latent_class(
        rows, levels=2, exchangeable=True, lowest_level=[0.95, 0.05]
    )
    assert fit.constrained
    assert fit.prevalence[1] == pytest.approx(0.3, abs=5e-3)
    assert fit.response_of(0, 1, 1) == pytest.approx(0.9, abs=5e-3)


def test_ordinal_codings_recover_a_three_level_facet():
    law = [[0.8, 0.15, 0.05], [0.15, 0.7, 0.15], [0.05, 0.15, 0.8]]
    rows = pattern_rows(30000, [0.5, 0.3, 0.2], [law, law, law], 3)
    fit = mx.repeat_codings_latent_class(rows, levels=3, categories=3)
    np.testing.assert_allclose(fit.prevalence, [0.5, 0.3, 0.2], atol=1.5e-2)
