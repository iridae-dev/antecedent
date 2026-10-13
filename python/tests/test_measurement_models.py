"""Measurement models through the Python facade.

Every asserted value has an oracle computed here in pure Python or numpy, never by native code:

* binary routes: the algebra of misclassification ``p* = Se*pi + (1 - Sp)*(1 - pi)`` inverted by
  hand, and its delta-method variance derived term by term (``d pi / d p* = 1/J``,
  ``d pi / d Se = -pi/J``, ``d pi / d Sp = (1 - pi)/J`` with ``J = Se + Sp - 1``);
* the categorical route: the exact observed-class summaries of a confusion model, inverted with
  ``numpy.linalg.solve``;
* the classical route: a simulated DGP with known coefficient 2 and the closed-form attenuation
  factor ``lambda = 0.8 / 1.4``, plus regression calibration computed with ``numpy.linalg.lstsq``;
* the repeated route: the pooled within-unit mean square of the replicates;
* the posterior route: Rubin's pooled point as the mean of per-draw least-squares coefficients.

Tolerances are written in the test: ``1e-12`` for exact algebra, ``4`` reported standard errors
for simulations with fixed seeds. Calibration of any interval is deliberately not measured: the
interval route is closed and these tests assert that it refuses.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from antecedent import _native
from antecedent import measurement_models as mm
from antecedent.errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from antecedent.measurement_models import MeasurementModelRefusal

from _refusal import assert_registered_refusal

SCALAR = mm.RepresentationVersion("scalar", 1)
COHORT = "cohort"
REGIME = "observational"
CONSTRUCT = "construct:latent"
INDICATOR = "schema:indicator"


# --------------------------------------------------------------------------- fixtures


def analysis(**overrides) -> mm.AnalysisBinding:
    fields = dict(construct_id=CONSTRUCT, population_id=COHORT, regime_id=REGIME)
    fields.update(overrides)
    return mm.AnalysisBinding(**fields)


def model(
    parameters,
    indicator_kind="noisy_misclassified",
    structure="non_differential",
    evidence=None,
    **overrides,
) -> mm.MeasurementModel:
    fields = dict(
        indicator=mm.MeasurementVariable(INDICATOR, indicator_kind, "unit"),
        construct=mm.MeasurementVariable(CONSTRUCT, "latent_scientific", "unit"),
        parameters=parameters,
        population_id=COHORT,
        regime_id=REGIME,
        evidence=evidence or mm.ParameterEvidence.declared("test"),
        structure=structure,
        representation=SCALAR,
    )
    fields.update(overrides)
    return mm.MeasurementModel(**fields)


def classical_model(error_variance, standard_error=0.0, **overrides) -> mm.MeasurementModel:
    return model(
        mm.ClassicalErrorParameters(error_variance, standard_error, min_reliability=0.1),
        **overrides,
    )


def rates(se, sp, vse=0.0, vsp=0.0, arm=None) -> mm.BinaryRates:
    return mm.BinaryRates(se, sp, vse, vsp, arm_id=arm)


def binary_model(rate_list, structure="non_differential", **overrides) -> mm.MeasurementModel:
    return model(
        mm.BinaryParameters(tuple(rate_list), min_youden=0.1), structure=structure, **overrides
    )


def simulate(n: int, seed: int):
    """``X ~ N(0,1)``, ``Z = 0.5 X + N(0,1)``, ``W = X + N(0, 0.6)``,
    ``Y = 1 + 2 X + 0.5 Z + N(0,1)``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = 0.5 * x + rng.standard_normal(n)
    w = x + np.sqrt(0.6) * rng.standard_normal(n)
    y = 1.0 + 2.0 * x + 0.5 * z + rng.standard_normal(n)
    return y, w, z


def classical_data(y, w, z) -> mm.ClassicalLinearData:
    return mm.ClassicalLinearData(outcome=y, regressor=w, covariates=z)


def prepare_classical(error_variance, standard_error, sim, **overrides):
    return mm.prepare_measurement_model(
        classical_model(error_variance, standard_error, **overrides),
        analysis(),
        classical_data(*sim),
    )


def assert_refusal(error, code, detail):
    assert_registered_refusal(error)
    assert error.reason_code == code, (error.reason_code, str(error))
    assert error.detail == detail, (error.detail, str(error))


# --------------------------------------------------------------------------- binary algebra


def test_binary_prevalence_inverts_the_misclassification_algebra():
    se, sp, vse, vsp = 0.9, 0.8, 1e-4, 4e-4
    n, events = 1000, 410
    prepared = mm.prepare_measurement_model(
        binary_model([rates(se, sp, vse, vsp)]), analysis(), mm.BinaryPrevalenceData(events, n)
    )
    estimate = prepared.estimate()

    # Oracle: pi = (p* + Sp - 1) / J, and the delta method term by term.
    j = se + sp - 1.0
    p_star = events / n
    pi = (p_star + sp - 1.0) / j
    assert pi == pytest.approx(0.3, abs=1e-15)
    variance = (p_star * (1 - p_star) / n + (1 - pi) ** 2 * vsp + pi**2 * vse) / j**2

    assert estimate.point == pytest.approx(pi, abs=1e-12)
    assert estimate.naive_point == pytest.approx(p_star, abs=1e-15)
    assert estimate.standard_error**2 == pytest.approx(variance, abs=1e-12)
    assert estimate.variance == pytest.approx(variance, abs=1e-12)
    assert estimate.operation == "latent_prevalence"
    assert estimate.route == "binary_misclassification"
    assert estimate.parameter_uncertainty == "propagated"
    assert estimate.inference == "uncalibrated_standard_error"
    assert estimate.rows == n
    # There is no coefficient vector to take a functional of.
    with pytest.raises(MeasurementModelRefusal) as caught:
        estimate.downstream_linear_functional([1.0])
    assert caught.value.reason_code in {"cell_not_licensed", "invalid_argument"}


def test_risk_difference_keeps_the_arm_covariance_that_split_rates_miss():
    se, sp, vse, vsp = 0.9, 0.8, 4e-4, 9e-4
    table = mm.RiskDifferenceData("treated", "control", 480, 1000, 340, 1000)
    prepared = mm.prepare_measurement_model(
        binary_model([rates(se, sp, vse, vsp)]), analysis(), table
    )
    estimate = prepared.estimate()
    j = se + sp - 1.0
    p1, p0 = 0.48, 0.34
    rd = (p1 - p0) / j
    sampling = (p1 * (1 - p1) / 1000 + p0 * (1 - p0) / 1000) / j**2
    # Shared rates enter both arms: d RD / d Se = d RD / d Sp = -RD / J.
    rate_part = (rd / j) ** 2 * (vse + vsp)
    assert estimate.point == pytest.approx(rd, abs=1e-12)
    assert estimate.point == pytest.approx(0.2, abs=1e-12)
    assert estimate.standard_error**2 == pytest.approx(sampling + rate_part, abs=1e-12)
    assert estimate.operation == "risk_difference"


def test_a_differential_model_applies_each_arms_own_rates_and_is_never_pooled():
    table = mm.RiskDifferenceData("treated", "control", 480, 1000, 180, 1000)
    differential = binary_model(
        [rates(0.9, 0.8, arm="treated"), rates(0.7, 0.95, arm="control")],
        structure="differential_arm",
    )
    # Analysed as non-differential: refused, not pooled.
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(differential, analysis(), table)
    assert_refusal(
        caught.value, "differential_error_not_pooled", "measurement_model.differential_declared"
    )

    right = mm.prepare_measurement_model(
        differential, analysis(assumed_structure="differential_arm"), table
    ).estimate()
    # Oracle: per-arm inversion pi_a = (p_a + Sp_a - 1) / (Se_a + Sp_a - 1).
    pi_treated = (0.48 + 0.8 - 1.0) / (0.9 + 0.8 - 1.0)
    pi_control = (0.18 + 0.95 - 1.0) / (0.7 + 0.95 - 1.0)
    assert right.point == pytest.approx(pi_treated - pi_control, abs=1e-12)
    assert right.point == pytest.approx(0.2, abs=1e-12)
    assert right.naive_point == pytest.approx(0.48 - 0.18, abs=1e-12)

    # The shared-rate model is a different identity and a different answer.
    pooled = mm.prepare_measurement_model(
        binary_model([rates(0.9, 0.8)]), analysis(), table
    ).estimate()
    assert pooled.point == pytest.approx((0.48 - 0.18) / 0.7, abs=1e-12)
    assert pooled.model_identity != right.model_identity

    # An arm the differential model has no rates for refuses; there is no fallback to pooled.
    unknown = mm.RiskDifferenceData("treated", "placebo", 480, 1000, 180, 1000)
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            differential, analysis(assumed_structure="differential_arm"), unknown
        ).estimate()
    assert caught.value.detail == "measurement_model.arm_mismatch"

    # Differential by time has no bounded route: refused, not pooled.
    by_time = classical_model(0.6, structure="differential_time")
    sim = simulate(60, 3)
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            by_time, analysis(assumed_structure="differential_time"), classical_data(*sim)
        )
    assert caught.value.reason_code == "differential_error_not_pooled"


@pytest.mark.parametrize("se,sp", [(0.5, 0.5), (0.3, 0.6), (0.55, 0.5)])
def test_a_youden_index_at_or_below_the_minimum_refuses(se, sp):
    # J = 0, -0.1 and 0.05: all at or below the declared minimum 0.1.
    prepared = mm.prepare_measurement_model(
        binary_model([rates(se, sp)]), analysis(), mm.BinaryPrevalenceData(40, 100)
    )
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.estimate()
    assert_refusal(caught.value, "measurement_not_identified", "measurement_model.youden_too_small")
    assert caught.value.remedy is None or isinstance(caught.value.remedy, str)


# --------------------------------------------------------------------------- categorical


CONFUSION = (0.8, 0.1, 0.0, 0.1, 0.8, 0.2, 0.1, 0.1, 0.8)  # [observed][true], columns sum to 1
LEVELS = ("low", "mid", "high")
PI = np.array([0.5, 0.3, 0.2])
MU = np.array([1.0, 2.0, 4.0])


def exact_classes(n=10_000):
    c = np.array(CONFUSION).reshape(3, 3)
    p = c @ PI
    ybar = (c * PI * MU).sum(axis=1) / p
    return [(int(round(p[j] * n)), float(ybar[j]), 1.0) for j in range(3)]


def categorical_model(counts=(400, 400, 400)) -> mm.MeasurementModel:
    return model(
        mm.CategoricalParameters(LEVELS, CONFUSION, counts, 50.0),
        indicator_kind="probabilistic_feature_assignment",
    )


def test_categorical_inversion_recovers_proportions_and_outcome_means():
    classes = exact_classes()
    c = np.array(CONFUSION).reshape(3, 3)
    # Oracle: observed class proportions solve C pi = p_obs.
    observed = np.array([count for count, _, _ in classes], dtype=float)
    pi = np.linalg.solve(c, observed / observed.sum())
    assert pi == pytest.approx(PI, abs=1e-9)
    for level, truth in zip(LEVELS, PI, strict=True):
        estimate = mm.prepare_measurement_model(
            categorical_model(),
            analysis(),
            mm.CategoricalData(classes, mm.CategoricalTarget.proportion(level)),
        ).estimate()
        assert estimate.point == pytest.approx(truth, abs=1e-9)
        assert estimate.operation == "latent_prevalence"
        assert estimate.standard_error > 0.0

    contrast = mm.prepare_measurement_model(
        categorical_model(),
        analysis(),
        mm.CategoricalData(classes, mm.CategoricalTarget.outcome_contrast("high", "low")),
    ).estimate()
    assert contrast.point == pytest.approx(MU[2] - MU[0], abs=1e-6)
    assert contrast.naive_point < 2.5  # the observed-label contrast is attenuated
    assert contrast.operation == "category_outcome_means"
    assert contrast.parameter_uncertainty == "propagated"

    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            categorical_model(),
            analysis(),
            mm.CategoricalData(classes, mm.CategoricalTarget.proportion("absent")),
        ).estimate()
    assert_refusal(caught.value, "invalid_argument", "measurement_model.unknown_level")
    assert isinstance(caught.value, ValueError)  # an invalid argument is also a ValueError


# --------------------------------------------------------------------------- classical


def test_classical_error_recovers_the_coefficient_and_the_attenuation_is_closed_form():
    y, w, z = simulate(20_000, 424_242)
    prepared = prepare_classical(0.6, 0.0, (y, w, z))
    assert prepared.operation == "linear_adjusted_effect"
    estimate = prepared.estimate()

    se = estimate.standard_error
    assert abs(estimate.point - 2.0) < 4.0 * se
    assert se < 0.08
    lam = 0.8 / 1.4  # Var(X | Z) / Var(W | Z)
    detail = estimate.detail
    dim = detail["dimension"]
    naive_se = np.sqrt(detail["naive_covariance"][dim + 1])
    assert abs(estimate.naive_point - 2.0 * lam) < 4.0 * naive_se
    assert (2.0 - estimate.naive_point) > 10.0 * naive_se

    # Oracle: regression calibration with numpy. X_hat = lam_hat W + (1 - lam_hat) E[W | Z],
    # lam_hat = 1 - sigma_u^2 / Var(W | Z); the coefficient on X_hat is the corrected slope.
    design = np.column_stack([np.ones_like(z), z])
    beta_w = np.linalg.lstsq(design, w, rcond=None)[0]
    fitted = design @ beta_w
    resid = w - fitted
    lam_hat = 1.0 - 0.6 / resid.var()
    x_hat = lam_hat * w + (1.0 - lam_hat) * fitted
    full = np.column_stack([np.ones_like(z), x_hat, z])
    rc = np.linalg.lstsq(full, y, rcond=None)[0]
    # The divisor convention differs by O(1/n): predeclared tolerance 5e-4 at n = 20 000.
    assert abs(estimate.point - rc[1]) < 5e-4

    # The closed interval route and the labels.
    assert estimate.inference == "uncalibrated_standard_error"
    assert estimate.parameter_uncertainty == "declared_exact"
    assert estimate.route == "classical_continuous_error"
    assert estimate.model_identity == prepared.model_identity
    for target in (estimate, prepared):
        with pytest.raises(MeasurementModelRefusal) as caught:
            target.interval()
        assert caught.value.reason_code == "cell_not_licensed"
        assert isinstance(caught.value, CausalUnsupportedError)
    with pytest.raises(MeasurementModelRefusal) as caught:
        estimate.interval()
    assert caught.value.detail == "measurement_model.interval_closed"


def test_the_measurement_parameter_uncertainty_enters_the_variance_and_the_functional():
    sim = simulate(5_000, 7)
    exact = prepare_classical(0.6, 0.0, sim).estimate()
    uncertain = prepare_classical(0.6, 0.06, sim).estimate()
    assert uncertain.point == exact.point  # the same bits: uncertainty never moves the point
    assert uncertain.standard_error > exact.standard_error
    assert uncertain.parameter_uncertainty == "propagated"
    # Added variance (d beta / d sigma_u^2)^2 Var(sigma_u^2) with
    # d beta / d sigma_u^2 = beta / (s_r^2 - sigma_u^2), s_r^2 the sample residual variance of W
    # given (1, Z) (divisor n). The population value of the denominator is 0.8; the sample value
    # differs by sampling error, so the oracle uses the sample quantity.
    _y, w, z = sim
    design = np.column_stack([np.ones_like(z), z])
    resid = w - design @ np.linalg.lstsq(design, w, rcond=None)[0]
    s_r2 = float(resid @ resid / len(w))
    assert abs(s_r2 - 1.4) < 0.15  # near the population 0.8 + 0.6
    added = uncertain.standard_error**2 - exact.standard_error**2
    expected = (uncertain.point / (s_r2 - 0.6)) ** 2 * 0.06**2
    assert added == pytest.approx(expected, rel=1e-6)
    # A downstream linear functional is the quadratic form under the joint covariance.
    value, variance = uncertain.downstream_linear_functional([0.0, 1.5, 0.0])
    assert value == pytest.approx(1.5 * uncertain.point, abs=1e-12)
    assert variance == pytest.approx(2.25 * uncertain.standard_error**2, abs=1e-12)
    cov = np.array(uncertain.detail["covariance"]).reshape(3, 3)
    weights = np.array([0.3, -1.0, 2.0])
    value, variance = uncertain.downstream_linear_functional(weights)
    assert value == pytest.approx(
        float(weights @ np.array(uncertain.detail["coefficients"])), abs=1e-12
    )
    assert variance == pytest.approx(float(weights @ cov @ weights), abs=1e-12)
    with pytest.raises(MeasurementModelRefusal):
        uncertain.downstream_linear_functional([1.0])  # wrong width


def test_row_order_is_invariant_and_the_same_inputs_give_the_same_bits():
    y, w, z = simulate(2_000, 99)
    forward = prepare_classical(0.6, 0.03, (y, w, z)).estimate()
    backward = prepare_classical(0.6, 0.03, (y[::-1], w[::-1], z[::-1])).estimate()
    assert abs(forward.point - backward.point) < 1e-9
    assert abs(forward.standard_error - backward.standard_error) < 1e-9
    again = prepare_classical(0.6, 0.03, (y, w, z)).estimate()
    assert again.point == forward.point
    assert again.standard_error == forward.standard_error
    assert again.data_identity == forward.data_identity
    # The permutation is a different arrangement of the same rows but not the same data bytes.
    assert backward.data_identity != forward.data_identity


def test_an_error_variance_that_leaves_no_latent_variance_refuses():
    prepared = prepare_classical(5.0, 0.0, simulate(500, 5))
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.estimate()
    assert_refusal(
        caught.value, "measurement_not_identified", "measurement_model.reliability_too_small"
    )


# --------------------------------------------------------------------------- repeated


def repeated_world(units: int, seed: int):
    """``X ~ N(0,1)``; replicates ``W_ij = X_i + N(0, 0.5)`` (2 or 3 per unit): ICC = 1/1.5."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(units)
    y = 1.0 + 1.5 * x + rng.standard_normal(units)
    counts = np.array([2 + (i % 2) for i in range(units)])
    values = np.concatenate(
        [x[i] + np.sqrt(0.5) * rng.standard_normal(counts[i]) for i in range(units)]
    )
    return y, values, counts


def test_repeated_measurements_estimate_the_icc_and_recover_the_coefficient():
    y, values, counts = repeated_world(6_000, 2024)
    model_ = model(mm.RepeatedParameters(50, 0.1))
    prepared = mm.prepare_measurement_model(
        model_, analysis(), mm.RepeatedLinearData(y, values, counts)
    )
    estimate = prepared.estimate()
    se = estimate.standard_error
    assert abs(estimate.point - 1.5) < 4.0 * se
    assert estimate.naive_point < 1.5 - 5.0 * se  # the mean of 2.5 replicates still attenuates
    assert estimate.parameter_uncertainty == "propagated"
    assert estimate.route == "repeated_measurements"

    components = estimate.detail["components"]
    assert abs(components["intraclass_correlation"] - 1.0 / 1.5) < 0.03
    assert abs(components["within_variance"] - 0.5) < 0.03
    # Oracle: the pooled within-unit mean square, computed directly.
    offsets = np.concatenate([[0], np.cumsum(counts)])
    ss = sum(
        float(((values[a:b] - values[a:b].mean()) ** 2).sum())
        for a, b in zip(offsets[:-1], offsets[1:], strict=True)
    )
    within = ss / (len(values) - len(counts))
    assert components["within_variance"] == pytest.approx(within, abs=1e-9)

    # Too few replicated units for the model's declared minimum.
    y2, values2, counts2 = repeated_world(200, 5)
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            model(mm.RepeatedParameters(10_000, 0.1)),
            analysis(),
            mm.RepeatedLinearData(y2, values2, counts2),
        ).estimate()
    assert_refusal(
        caught.value, "measurement_not_identified", "measurement_model.insufficient_replicates"
    )


# --------------------------------------------------------------------------- posterior


def provider(request_id="req-1", **overrides) -> mm.ProviderObject:
    fields = dict(
        provider_id="vendor",
        object_id="latent-1",
        version_id="v3",
        snapshot_id="snap",
        request_id=request_id,
        variable_id=CONSTRUCT,
        variable_name="latent",
        units="unit",
        population_id=COHORT,
        regime_id=REGIME,
        trust="externally_attested",
        attestor="vendor",
    )
    fields.update(overrides)
    return mm.ProviderObject(**fields)


def posterior_model(conditioned: bool, min_draws=20) -> mm.MeasurementModel:
    reference = mm.PosteriorReference("vendor", "latent-1", "v3", "snap", "req-1")
    return model(
        mm.PosteriorParameters(reference, conditioned, min_draws),
        indicator_kind="learned_vector_representation",
    )


def posterior_world(n: int, draws: int, seed: int):
    """``X ~ N(0,1)``, ``W = X + N(0,1)``, ``Y = 1 + 2 X + N(0,1)``; the posterior of ``X`` given
    ``W`` alone is N(W/2, 1/2), given ``(W, Y)`` it is N((W + 2(Y - 1))/6, 1/6)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    w = x + rng.standard_normal(n)
    y = 1.0 + 2.0 * x + rng.standard_normal(n)
    marginal = w / 2.0 + np.sqrt(0.5) * rng.standard_normal((draws, n))
    conditional = (w + 2.0 * (y - 1.0)) / 6.0 + np.sqrt(1.0 / 6.0) * rng.standard_normal((draws, n))
    return y, marginal, conditional


def test_a_posterior_route_needs_the_exact_attested_provider_object():
    y, marginal, _ = posterior_world(500, 30, 31)
    data = mm.LatentPosteriorData(y, marginal)
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(posterior_model(False), analysis(), data)
    assert caught.value.reason_code == "measurement_posterior_not_trusted"
    # Metadata cannot confer native trust.
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            posterior_model(False), analysis(), data, provider=provider(trust="native_licensed")
        )
    assert_refusal(
        caught.value,
        "measurement_posterior_not_trusted",
        "measurement_model.posterior_native_trust",
    )
    # A verified claim without the receipt the library issues is not a receipt.
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            posterior_model(False),
            analysis(),
            data,
            provider=provider(trust="exact_request_verified"),
        )
    assert caught.value.reason_code == "measurement_posterior_not_trusted"
    # Another request fingerprint is another object.
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            posterior_model(False), analysis(), data, provider=provider("req-2")
        )
    assert_refusal(
        caught.value, "measurement_posterior_not_trusted", "measurement_model.posterior_identity"
    )
    # An attestation without an attestor names no one.
    with pytest.raises(MeasurementModelRefusal):
        mm.prepare_measurement_model(
            posterior_model(False), analysis(), data, provider=provider(attestor=None)
        )
    # The exact object, attested, is consumed and labelled as such.
    prepared = mm.prepare_measurement_model(
        posterior_model(False), analysis(), data, provider=provider()
    )
    assert prepared.provider_trust == "externally_attested"
    # A learned representation's coordinates are not intervenable.
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.check_operation("intervene_on_representation")
    assert caught.value.detail == "measurement_model.learned_coordinates_not_intervenable"


def test_posterior_substitution_and_rubins_rules_replay_independently():
    n, draws = 4_000, 100
    y, marginal, conditional = posterior_world(n, draws, 31)

    # Substitution: the posterior does not condition on the outcome.
    prepared = mm.prepare_measurement_model(
        posterior_model(False),
        analysis(),
        mm.LatentPosteriorData(y, marginal),
        provider=provider(),
    )
    estimate = prepared.estimate()
    se = estimate.standard_error
    assert abs(estimate.point - 2.0) < 4.0 * se + 0.03
    assert estimate.parameter_uncertainty == "not_propagated"
    assert estimate.provider_trust == "externally_attested"
    consumed = mm.consume_measurement_model_artifact(
        estimate.export("posterior-substitution"), prepared.model_identity
    )
    assert consumed.provider_trust == "externally_attested"
    assert consumed.point == pytest.approx(estimate.point, abs=1e-9)
    assert consumed.standard_error == pytest.approx(se, abs=1e-9)

    # Outcome-conditioned draws combine by Rubin's rules.
    prepared = mm.prepare_measurement_model(
        posterior_model(True),
        analysis(),
        mm.LatentPosteriorData(y, conditional),
        provider=provider(),
    )
    estimate = prepared.estimate()
    se = estimate.standard_error
    assert abs(estimate.point - 2.0) < 4.0 * se + 0.02
    assert estimate.parameter_uncertainty == "propagated"
    # Oracle: the pooled point is the mean of the per-draw least-squares slopes.
    slopes = np.array(
        [
            np.linalg.lstsq(np.column_stack([np.ones(n), conditional[s]]), y, rcond=None)[0][1]
            for s in range(draws)
        ]
    )
    assert estimate.point == pytest.approx(slopes.mean(), abs=1e-9)
    names = [c.name for c in estimate.variance_components]
    assert names == ["within_imputation", "between_imputation"]
    parts = {c.name: c.variance for c in estimate.variance_components}
    assert sum(parts.values()) == pytest.approx(estimate.standard_error**2, abs=1e-12)
    between = (1.0 + 1.0 / draws) * slopes.var(ddof=1)
    assert parts["between_imputation"] == pytest.approx(between, rel=1e-6)
    consumed = mm.consume_measurement_model_artifact(
        estimate.export("posterior-imputation"), prepared.model_identity
    )
    assert consumed.point == pytest.approx(estimate.point, abs=1e-9)

    # Fewer draws than the model declares refuses before a claim.
    y2, marginal2, _ = posterior_world(200, 10, 4)
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            posterior_model(False),
            analysis(),
            mm.LatentPosteriorData(y2, marginal2),
            provider=provider(),
        ).estimate()
    assert caught.value.detail == "measurement_model.insufficient_draws"


# --------------------------------------------------------------------------- prepare checks


def test_population_regime_arm_and_representation_are_checked_before_estimation():
    sim = simulate(60, 3)
    data = classical_data(*sim)
    m = classical_model(0.6)

    def refuse(model_, binding):
        with pytest.raises(MeasurementModelRefusal) as caught:
            mm.prepare_measurement_model(model_, binding, data)
        assert isinstance(caught.value, CausalUnsupportedError)
        assert_registered_refusal(caught.value)
        return caught.value

    error = refuse(m, analysis(population_id="elsewhere"))
    assert_refusal(error, "measurement_binding_mismatch", "measurement_model.population_mismatch")
    assert error.remedy is not None or error.message

    assert refuse(m, analysis(regime_id="do(a=1)")).detail == "measurement_model.regime_mismatch"

    arm_model = classical_model(0.6, arm_id="treated")
    assert refuse(arm_model, analysis(arm_id="control")).detail == "measurement_model.arm_mismatch"

    error = refuse(m, analysis(representation=mm.RepresentationVersion("scalar", 2)))
    assert_refusal(
        error, "representation_mapping_required", "measurement_model.representation_unmapped"
    )

    # A proxy is never the construct: the measured column named as the construct, and a proxy kind
    # declared as the construct.
    error = refuse(m, analysis(construct_id=INDICATOR))
    assert_refusal(error, "proxy_offered_as_construct", "measurement_model.indicator_as_construct")
    proxy = classical_model(
        0.6, construct=mm.MeasurementVariable(CONSTRUCT, "learned_vector_representation", "unit")
    )
    error = refuse(proxy, analysis())
    assert_refusal(error, "proxy_offered_as_construct", "measurement_model.proxy_as_construct")

    # An observed column is given no error model.
    observed = classical_model(0.6, indicator=mm.MeasurementVariable(INDICATOR, "observed", "unit"))
    assert refuse(observed, analysis()).reason_code == "measurement_model_invalid"

    # A route that does not match the parameters is refused by name.
    mismatched = classical_model(0.6, route="binary_misclassification")
    error = refuse(mismatched, analysis())
    assert error.reason_code == "measurement_model_invalid"

    # A binary input sent at a classical model is not an operation of its route.
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(m, analysis(), mm.BinaryPrevalenceData(10, 100))
    assert caught.value.reason_code == "cell_not_licensed"


def test_the_route_licenses_only_its_own_operations():
    prepared = prepare_classical(0.6, 0.0, simulate(60, 3))
    prepared.check_operation("linear_functional_of_coefficients")  # licensed: returns
    for refused in (
        "nonlinear_outcome_model",
        "condition_on_construct",
        "transport_to_other_population",
        "interval",
    ):
        with pytest.raises(MeasurementModelRefusal) as caught:
            prepared.check_operation(refused)
        assert_refusal(
            caught.value, "cell_not_licensed", "measurement_model.operation_not_licensed"
        )
        assert caught.value.remedy is not None
    assert prepared.route_card["route"] == "classical_continuous_error"
    assert {item["operation"] for item in prepared.route_card["refused"]} >= {
        "nonlinear_outcome_model",
        "interval",
    }
    with pytest.raises(CausalValueError):
        prepared.check_operation("not_an_operation")


def test_a_mapping_carries_values_and_intervention_meaning_only_when_declared():
    sim = simulate(2_000, 11)
    data = classical_data(*sim)
    m = classical_model(0.6)
    v2 = mm.RepresentationVersion("scalar", 2)
    relabel = mm.RepresentationMapping(SCALAR, v2, "map-relabel")
    prepared = mm.prepare_measurement_model(m, analysis(representation=v2, mapping=relabel), data)
    estimate = prepared.estimate()
    plain = prepare_classical(0.6, 0.0, sim).estimate()
    assert estimate.representation_mapping_id == "map-relabel"
    # Values crossed unchanged: the same correction as the unmapped one.
    assert estimate.point == plain.point
    assert estimate.standard_error == plain.standard_error
    assert estimate.data_identity == plain.data_identity
    # A relabeling carries values only.
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.check_operation("intervene_on_representation")
    assert_refusal(
        caught.value,
        "representation_mapping_required",
        "measurement_model.intervention_meaning_not_carried",
    )
    # A mapping between other versions is refused outright.
    wrong = mm.RepresentationMapping(mm.RepresentationVersion("scalar", 7), v2, "map-relabel")
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(m, analysis(representation=v2, mapping=wrong), data)
    assert caught.value.detail == "measurement_model.mapping_endpoints"
    # Declared evidence of intervention preservation licenses the meaning; the route itself
    # still refuses an intervention on the recorded indicator.
    preserving = mm.RepresentationMapping(SCALAR, v2, "map-relabel", "ev:trial-9")
    ok = mm.prepare_measurement_model(m, analysis(representation=v2, mapping=preserving), data)
    with pytest.raises(MeasurementModelRefusal) as caught:
        ok.check_operation("intervene_on_representation")
    assert_refusal(caught.value, "cell_not_licensed", "measurement_model.operation_not_licensed")


# --------------------------------------------------------------------------- lifecycle


def test_refresh_keeps_the_model_recomputes_the_numbers_and_refuses_other_questions():
    first, second = simulate(3_000, 21), simulate(3_000, 22)
    prepared = prepare_classical(0.6, 0.05, first)
    before = prepared.estimate()

    refreshed = prepared.refresh(classical_data(*second))
    # Kept: the model and its identities.
    assert refreshed.prepared.model == prepared.model
    assert refreshed.prepared.model_identity == prepared.model_identity
    assert refreshed.estimate.model_identity == before.model_identity
    # Recomputed: the point, the standard error and the data identity ...
    assert refreshed.estimate.point != before.point
    assert refreshed.estimate.data_identity != before.data_identity
    # ... to exactly what a fresh preparation on the new data gives.
    direct = prepare_classical(0.6, 0.05, second).estimate()
    assert refreshed.estimate.point == direct.point
    assert refreshed.estimate.standard_error == direct.standard_error

    # Refused: another question; and the object it was called on is untouched.
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.refresh(mm.BinaryPrevalenceData(1, 10))
    assert caught.value.reason_code == "route_not_supported"
    assert prepared.estimate().point == before.point
    # A refresh that fails inside the estimator leaves the prior state alone as well.
    tiny = mm.ClassicalLinearData(np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(MeasurementModelRefusal):
        prepared.refresh(tiny)
    again = prepared.estimate()
    assert again.point == before.point
    assert again.standard_error == before.standard_error
    assert again.data_identity == before.data_identity


def test_changed_validation_parameters_rebuild_the_identities_and_nothing_else():
    sim = simulate(3_000, 21)
    prepared = prepare_classical(0.6, 0.05, sim)
    before = prepared.estimate()
    rebuilt = prepared.with_validation_parameters(
        mm.ClassicalErrorParameters(0.5, 0.05, 0.1),
        mm.ParameterEvidence.validation_sample("val-2", 400),
    )
    old, new = prepared.identities, rebuilt.prepared.identities
    assert new.model != old.model
    assert new.measurement != old.measurement
    assert new.evidence != old.evidence
    # The population, regime, arm and representation identities are untouched.
    assert new.population == old.population
    assert new.regime == old.regime
    assert new.arm == old.arm
    assert new.representation == old.representation
    assert rebuilt.estimate.model_identity == new.model
    assert rebuilt.estimate.point != before.point
    # Oracle: a smaller error variance corrects less, so the corrected slope is smaller.
    assert abs(rebuilt.estimate.point) < abs(before.point)
    # Parameters of another route are not a validation update; the original is untouched.
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.with_validation_parameters(
            mm.BinaryParameters((rates(0.9, 0.8),), 0.1), mm.ParameterEvidence.declared("test")
        )
    assert caught.value.reason_code == "route_not_supported"
    assert prepared.estimate().point == before.point


def test_cancellation_stops_before_a_claim_and_is_typed():
    prepared = prepare_classical(0.6, 0.0, simulate(200, 5))
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.estimate(cancel=token)
    assert caught.value.reason_code == "cancelled_no_claim"
    with pytest.raises(MeasurementModelRefusal) as caught:
        prepared.refresh(classical_data(*simulate(200, 6)), cancel=token)
    assert caught.value.reason_code == "cancelled_no_claim"
    with pytest.raises(MeasurementModelRefusal) as caught:
        mm.prepare_measurement_model(
            classical_model(0.6), analysis(), classical_data(*simulate(200, 5)), cancel=token
        )
    assert caught.value.reason_code == "cancelled_no_claim"


# --------------------------------------------------------------------------- artifact


def test_the_artifact_replays_independently_and_refuses_semantic_mutation():
    sim = simulate(3_000, 8)
    m = classical_model(0.6, 0.05)
    prepared = mm.prepare_measurement_model(m, analysis(), classical_data(*sim))
    estimate = prepared.estimate()
    artifact = estimate.export("replay-1")
    assert artifact == estimate.export("replay-1")  # the bytes are canonical
    identity = prepared.model_identity

    consumed = mm.consume_measurement_model_artifact(artifact, identity)
    assert consumed.point == pytest.approx(estimate.point, abs=1e-10)
    assert consumed.standard_error == pytest.approx(estimate.standard_error, abs=1e-10)
    assert consumed.stored_point == estimate.point
    assert consumed.rows == estimate.rows
    assert consumed.model.to_wire() == m.to_wire()
    assert consumed.model_identity == identity
    assert consumed.covers_data_identity(estimate.data_identity)
    assert not consumed.covers_data_identity("0" * 64)
    # The consumer applies its own binding to the stored result.
    consumed.check_use(analysis(), "linear_adjusted_effect")
    with pytest.raises(MeasurementModelRefusal) as caught:
        consumed.check_use(analysis(population_id="elsewhere"), "linear_adjusted_effect")
    assert caught.value.detail == "measurement_model.population_mismatch"

    # Semantic mutations are different identities: a result carried by a mutated model is refused
    # by a consumer that retained the original identity.
    mutations = {
        "population": classical_model(0.6, 0.05, population_id="elsewhere"),
        "parameters": classical_model(0.7, 0.05),
        "representation": classical_model(
            0.6, 0.05, representation=mm.RepresentationVersion("scalar", 2)
        ),
    }
    for name, mutated in mutations.items():
        binding = analysis(
            population_id="elsewhere" if name == "population" else COHORT,
            representation=mutated.representation,
        )
        forged = (
            mm.prepare_measurement_model(mutated, binding, classical_data(*sim))
            .estimate()
            .export("forged")
        )
        with pytest.raises(MeasurementModelRefusal) as caught:
            mm.consume_measurement_model_artifact(forged, identity)
        assert caught.value.reason_code == "measurement_binding_mismatch", name
        assert mutated.to_wire() != m.to_wire()
        assert mm.model_identities(mutated).model != identity

    # Damaged bytes and a malformed identity are refused, not ignored.
    damaged = bytearray(artifact)
    damaged[len(damaged) // 2] ^= 0x55
    with pytest.raises(MeasurementModelRefusal):
        mm.consume_measurement_model_artifact(bytes(damaged), identity)
    with pytest.raises(MeasurementModelRefusal):
        mm.consume_measurement_model_artifact(artifact, "not-an-identity")
    with pytest.raises(MeasurementModelRefusal):
        mm.consume_measurement_model_artifact(artifact[: len(artifact) // 2], identity)
    with pytest.raises(CausalTypeError):
        mm.consume_measurement_model_artifact("not bytes", identity)  # type: ignore[arg-type]


def test_every_route_exports_a_result_a_consumer_recomputes():
    # Binary prevalence.
    prepared = mm.prepare_measurement_model(
        binary_model([rates(0.9, 0.8, 1e-4, 4e-4)]), analysis(), mm.BinaryPrevalenceData(410, 1000)
    )
    consumed = mm.consume_measurement_model_artifact(
        prepared.estimate().export("binary"), prepared.model_identity
    )
    assert consumed.point == pytest.approx(0.3, abs=1e-12)
    assert consumed.route == "binary_misclassification"

    # Categorical contrast.
    prepared = mm.prepare_measurement_model(
        categorical_model(None),
        analysis(),
        mm.CategoricalData(exact_classes(), mm.CategoricalTarget.outcome_contrast("high", "low")),
    )
    consumed = mm.consume_measurement_model_artifact(
        prepared.estimate().export("categorical"), prepared.model_identity
    )
    assert consumed.point == pytest.approx(3.0, abs=1e-6)
    assert consumed.route == "probabilistic_categorical"

    # Repeated measurements.
    y, values, counts = repeated_world(500, 3)
    prepared = mm.prepare_measurement_model(
        model(mm.RepeatedParameters(50, 0.1)),
        analysis(),
        mm.RepeatedLinearData(y, values, counts),
    )
    estimate = prepared.estimate()
    consumed = mm.consume_measurement_model_artifact(
        estimate.export("repeated"), prepared.model_identity
    )
    assert consumed.point == pytest.approx(estimate.point, abs=1e-9)


def test_a_fresh_process_consumes_the_artifact_against_the_retained_identity(tmp_path):
    prepared = mm.prepare_measurement_model(
        binary_model([rates(0.9, 0.8, 1e-4, 4e-4)]), analysis(), mm.BinaryPrevalenceData(410, 1000)
    )
    path = tmp_path / "artifact.bin"
    path.write_bytes(prepared.estimate().export("fresh"))
    script = (
        "import json, sys\n"
        "from antecedent import measurement_models as mm\n"
        "consumed = mm.consume_measurement_model_artifact(open(sys.argv[1], 'rb').read(), sys.argv[2])\n"
        "print(json.dumps({'point': consumed.point, 'se': consumed.standard_error, "
        "'identity': consumed.model_identity, 'route': consumed.route}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), prepared.model_identity],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["point"] == pytest.approx(0.3, abs=1e-12)
    assert wire["identity"] == prepared.model_identity
    assert wire["route"] == "binary_misclassification"
    # The wrong retained identity is refused in the fresh process too.
    refused = subprocess.run(
        [sys.executable, "-c", script, str(path), "0" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refused.returncode != 0
    assert "measurement_binding_mismatch" in refused.stderr


def test_the_identities_ignore_the_label_and_move_separately():
    base = classical_model(0.6, 0.05)
    identities = mm.model_identities(base)
    relabeled = mm.model_identities(classical_model(0.6, 0.05, model_id="another label"))
    assert relabeled == identities

    moved = mm.model_identities(classical_model(0.6, 0.05, population_id="clinic-b"))
    assert moved.population != identities.population
    assert moved.regime == identities.regime
    assert moved.arm == identities.arm
    assert moved.model != identities.model
    assert moved.measurement == identities.measurement
    assert moved.representation == identities.representation
    assert moved.evidence == identities.evidence

    versioned = mm.model_identities(
        classical_model(0.6, 0.05, representation=mm.RepresentationVersion("scalar", 2))
    )
    assert versioned.representation != identities.representation
    assert versioned.measurement == identities.measurement

    # Swapping sensitivity and specificity is a different measurement.
    a = mm.model_identities(binary_model([rates(0.9, 0.8, 1e-4, 4e-4)]))
    b = mm.model_identities(binary_model([rates(0.8, 0.9, 4e-4, 1e-4)]))
    assert a.measurement != b.measurement
    assert a.population == b.population
    assert a.regime == b.regime
    assert a.arm == b.arm

    evidence = mm.model_identities(
        classical_model(0.6, 0.05, evidence=mm.ParameterEvidence.declared("lab"))
    )
    assert evidence.evidence != identities.evidence or evidence.model != identities.model


def test_declarations_are_validated_before_the_native_layer_sees_them():
    with pytest.raises(CausalValueError):
        mm.MeasurementVariable("x", "not_a_kind", "u")
    with pytest.raises(CausalValueError):
        mm.AnalysisBinding(CONSTRUCT, COHORT, REGIME, assumed_structure="sometimes")
    with pytest.raises(CausalValueError):
        classical_model(float("nan")).to_wire()
    with pytest.raises(CausalTypeError):
        mm.ParameterEvidence.declared("")
    with pytest.raises(CausalValueError):
        mm.ClassicalLinearData([1.0, np.inf], [0.0, 0.0]).to_wire()
    with pytest.raises(CausalValueError):
        mm.ClassicalLinearData([1.0, 2.0], [0.0, 0.0], covariates=np.zeros((3, 1))).to_wire()
