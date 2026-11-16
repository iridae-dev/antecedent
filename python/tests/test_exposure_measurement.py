"""Exposure-error declarations consumed by the causal analysis.

Every asserted value has an oracle written here with numpy or the standard library, never taken
from the code under test:

* Rubin's rules: within, between and total variance and the degrees of freedom by hand, and the
  ``t`` quantile against published table values;
* known-truth recovery: a planted exposure effect (linear outcome 2, logistic risk contrast from
  a closed-form integral) with the exposure misclassified, soft, under-detected or recoded, read
  through ``analyze`` / ``prepare`` with the declaration on the exposure variable. The recorded
  column is shown to miss the truth, the pooled estimate to recover it, the pooled interval to
  contain it;
* the exposure draws: export, consumption, completed tables and the identities they carry.

Tolerances are written in each test: four pooled standard errors for planted recovery.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent import exposure_measurement as em
from antecedent import measurement_exposure as mx
from antecedent._posterior_artifact import consume_posterior
from antecedent.errors import CausalValueError
from antecedent.measurement_models import BinaryRates, MeasurementModelRefusal
from antecedent.outcomes import require_success

GRAPH = [("z", "x"), ("z", "y"), ("x", "y")]
QUERY = ant.AverageEffect(treatment="x", outcome="y")


def expit(v):
    return 1.0 / (1.0 + np.exp(-v))


def planted(seed, n=3000):
    """Exposure ``x`` caused by ``z``; ``y = 1 + 2 x + 0.5 z + N(0, 1)``: the effect is 2."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = (rng.random(n) < expit(0.3 * z)).astype(float)
    y = 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=n)
    return rng, z, x, y


def ols_exposure_coefficient(y, x, z):
    design = np.column_stack([np.ones(len(y)), x, z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    cov = np.linalg.inv(design.T @ design) * (resid @ resid / (len(y) - 3))
    return float(beta[1]), float(cov[1, 1])


def analyse(data, measurement, draws=12, seed=3):
    return require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            measurement=measurement,
            measurement_draws=draws,
            measurement_seed=seed,
        )
    )


def assert_recovers(result, truth, naive_gap=0.15):
    assert abs(result.effect - truth) < 4.0 * result.standard_error + 0.02
    interval = result.interval(0.95)
    assert interval.lower <= truth <= interval.upper
    assert interval.calibration["status"] == "not_measured"
    assert result.naive_effect is None or abs(result.naive_effect - truth) > naive_gap


# --------------------------------------------------------------------------- Rubin's rules


def test_rubin_pooling_matches_the_hand_computation():
    estimates = [1.9, 2.1, 2.3, 2.0]
    variances = [0.04, 0.05, 0.03, 0.04]
    pool = em.rubin_pool(estimates, variances)
    m = len(estimates)
    mean = sum(estimates) / m
    within = sum(variances) / m
    between = sum((q - mean) ** 2 for q in estimates) / (m - 1)
    total = within + (1 + 1 / m) * between
    df = (m - 1) * (1 + within / ((1 + 1 / m) * between)) ** 2
    assert pool.estimate == pytest.approx(mean, abs=1e-12)
    assert pool.within == pytest.approx(within, abs=1e-12)
    assert pool.between == pytest.approx(between, abs=1e-12)
    assert pool.total == pytest.approx(total, abs=1e-12)
    assert pool.degrees_of_freedom == pytest.approx(df, abs=1e-9)
    assert pool.standard_error == pytest.approx(math.sqrt(total), abs=1e-12)


def test_the_pooled_interval_uses_students_t_quantiles():
    # Two-sided 95% quantiles of Student's t from published tables.
    for df, quantile in ((3, 3.182446), (10, 2.228139), (30, 2.042272)):
        assert em._student_t_quantile(0.975, df) == pytest.approx(quantile, abs=1e-5)
    assert em._student_t_quantile(0.975, math.inf) == pytest.approx(1.959964, abs=1e-5)
    pool = em.rubin_pool([1.0, 1.0, 1.0], [0.25, 0.25, 0.25])
    interval = pool.interval(0.95)
    assert pool.between == 0.0 and math.isinf(pool.degrees_of_freedom)
    assert interval.upper - interval.lower == pytest.approx(2 * 1.959964 * 0.5, abs=1e-4)


def test_pooling_needs_two_finite_estimates():
    with pytest.raises(CausalValueError):
        em.rubin_pool([1.0], [0.1])
    with pytest.raises(CausalValueError):
        em.rubin_pool([1.0, float("nan")], [0.1, 0.1])


# --------------------------------------------------------------------------- recovery


def test_misclassified_exposure_recovers_the_planted_effect_through_analyze():
    rng, z, x, y = planted(11)
    se, sp = 0.85, 0.9
    w = np.where(x == 1, rng.random(len(x)) < se, rng.random(len(x)) >= sp).astype(float)
    declaration = em.MisclassifiedExposure(
        "x", mx.ExposureError.shared(BinaryRates(se, sp, 1e-4, 1e-4)), covariates=("z",)
    )
    result = analyse({"x": w, "y": y, "z": z}, declaration).value
    assert result.exposure.kind == "misclassified_exposure"
    assert_recovers(result, 2.0)
    # The recorded column attenuates the effect well below the planted value.
    assert result.naive_effect < 1.7


def test_outcome_differential_misclassification_recovers_the_risk_contrast():
    rng = np.random.default_rng(21)
    n = 6000
    z = rng.normal(size=n)
    x = (rng.random(n) < expit(0.3 * z)).astype(float)
    y = (rng.random(n) < expit(-0.5 + 1.0 * x + 0.4 * z)).astype(float)
    cases, controls = (0.9, 0.8), (0.7, 0.95)
    sens = np.where(y == 1, cases[0], controls[0])
    spec = np.where(y == 1, cases[1], controls[1])
    w = np.where(x == 1, rng.random(n) < sens, rng.random(n) >= spec).astype(float)
    # Planted risk difference of the exposure, integrated over z ~ N(0, 1) by Gauss-Hermite.
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / weights.sum()
    truth = float(np.sum(weights * (expit(-0.5 + 1.0 + 0.4 * nodes) - expit(-0.5 + 0.4 * nodes))))
    declaration = em.MisclassifiedExposure(
        "x",
        mx.ExposureError.by_outcome(
            cases=BinaryRates(cases[0], cases[1], 1e-4, 1e-4),
            controls=BinaryRates(controls[0], controls[1], 1e-4, 1e-4),
        ),
        covariates=("z",),
    )
    result = analyse({"x": w, "y": y, "z": z}, declaration).value
    assert abs(result.effect - truth) < 4.0 * result.standard_error + 0.02
    interval = result.interval()
    assert interval.lower <= truth <= interval.upper
    # One pair of rates for both outcome levels misses the planted contrast.
    shared = analyse(
        {"x": w, "y": y, "z": z},
        em.MisclassifiedExposure(
            "x", mx.ExposureError.shared(BinaryRates(0.8, 0.9, 1e-4, 1e-4)), covariates=("z",)
        ),
    ).value
    assert abs(shared.effect - truth) > abs(result.effect - truth)


def test_validation_counts_widen_the_pooled_interval_over_exact_rates():
    rng, z, x, y = planted(31)
    se, sp = 0.85, 0.9
    w = np.where(x == 1, rng.random(len(x)) < se, rng.random(len(x)) >= sp).astype(float)
    data = {"x": w, "y": y, "z": z}
    exact = analyse(
        data,
        em.MisclassifiedExposure(
            "x", mx.ExposureError.shared(BinaryRates(se, sp)), covariates=("z",)
        ),
        draws=20,
    ).value
    validated = analyse(
        data,
        em.MisclassifiedExposure(
            "x",
            mx.ExposureError.shared(mx.ValidationCounts(17, 3, 18, 2)),
            covariates=("z",),
        ),
        draws=20,
    ).value
    assert validated.pool.between > exact.pool.between


def test_soft_exposure_probabilities_recover_the_planted_effect():
    rng, z, x, y = planted(41)
    n = len(x)
    signal = rng.normal(size=n) + 1.5 * (2 * x - 1)
    prior = expit(0.3 * z)
    odds = prior / (1 - prior) * np.exp(3.0 * signal)  # exact likelihood ratio of N(+-1.5, 1)
    q = odds / (1 + odds)
    declaration = em.SoftExposure("x", "q", covariates=("z",))
    result = analyse({"q": q, "y": y, "z": z}, declaration).value
    assert result.exposure.kind == "soft_exposure"
    assert result.naive_effect is None
    assert_recovers(result, 2.0)


def test_covariate_dependent_detection_recovers_the_planted_effect():
    rng, z, x, y = planted(51)
    n = len(x)
    count = rng.poisson(3, n).astype(float)
    detected = (x * (rng.random(n) < expit(-1.0 + 0.8 * count))).astype(float)
    declaration = em.DetectedExposure(
        "x", "count", mx.LogisticDetection(-1.0, 0.8, 1e-3, 0.0, 1e-3), covariates=("z",)
    )
    result = analyse({"x": detected, "count": count, "y": y, "z": z}, declaration).value
    assert_recovers(result, 2.0)
    # The fully declared function point-identifies the prevalence; the standalone correction
    # reports it from the same declaration.
    prevalence = declaration.core({"x": detected, "count": count, "y": y, "z": z}, outcome="y")
    assert prevalence.kind == "point"


def test_repeat_codings_recover_the_planted_effect_of_a_latent_facet():
    rng, z, x, y = planted(61)
    n = len(x)
    codings = np.column_stack(
        [np.where(x == 1, rng.random(n) < 0.8, rng.random(n) < 0.15).astype(int) for _ in range(3)]
    )
    data = {"c1": codings[:, 0], "c2": codings[:, 1], "c3": codings[:, 2], "y": y, "z": z}
    declaration = em.RepeatCodedExposure("x", ("c1", "c2", "c3"), covariates=("z",))
    result = analyse(data, declaration).value
    assert result.exposure.kind == "repeat_coded_exposure"
    assert result.naive_effect is None
    assert_recovers(result, 2.0)


# --------------------------------------------------------------------------- composition


def test_prepare_and_estimate_equal_analyze_and_carry_the_declaration():
    rng, z, x, y = planted(71, n=1500)
    w = np.where(x == 1, rng.random(len(x)) < 0.85, rng.random(len(x)) >= 0.9).astype(float)
    data = {"x": w, "y": y, "z": z}
    declaration = em.MisclassifiedExposure(
        "x", mx.ExposureError.shared(BinaryRates(0.85, 0.9, 1e-4, 1e-4)), covariates=("z",)
    )
    study = ant.prepare(
        data,
        graph=GRAPH,
        query=QUERY,
        measurement=declaration,
        measurement_draws=6,
        measurement_seed=3,
    )
    assert isinstance(study, em.PreparedMeasuredAnalysis)
    assert study.refusal is None and len(study.studies) == 6
    outcome = ant.estimate(study)
    direct = analyse(data, declaration, draws=6, seed=3)
    assert outcome.value.effect == direct.value.effect
    assert outcome.answer.kind == "point" and outcome.answer.value == outcome.value.effect
    assert outcome.value.identities["declaration_identity"] == declaration.identity()
    again = ant.estimate(ant.rebind(study, data=data))
    assert again.value.effect == direct.value.effect


def test_the_pooled_point_equals_rubin_over_the_per_draw_analyses():
    rng, z, x, y = planted(81, n=1500)
    w = np.where(x == 1, rng.random(len(x)) < 0.85, rng.random(len(x)) >= 0.9).astype(float)
    declaration = em.MisclassifiedExposure(
        "x", mx.ExposureError.shared(BinaryRates(0.85, 0.9)), covariates=("z",)
    )
    result = analyse({"x": w, "y": y, "z": z}, declaration, draws=8).value
    points = [a.estimate.ate for a in result.analyses]
    variances = [a.estimate.se_analytic**2 for a in result.analyses]
    pool = em.rubin_pool(points, variances)
    assert result.effect == pytest.approx(pool.estimate, abs=1e-12)
    assert result.pool.total == pytest.approx(pool.total, abs=1e-12)
    # The library's own multiple-imputation pooling agrees on the point when given the same draws.
    native = mx.soft_exposure_effect(
        y,
        result.exposure.probability,
        z.reshape(-1, 1),
        method="supplied_draws",
        draws=result.exposure.draws,
    )
    per_draw = [ols_exposure_coefficient(y, result.exposure.column(k), z)[0] for k in range(8)]
    assert native.coefficient == pytest.approx(float(np.mean(per_draw)), abs=1e-8)


def test_the_exposure_draws_export_as_a_posterior_artifact_and_read_back():
    rng, z, x, y = planted(91, n=400)
    w = np.where(x == 1, rng.random(len(x)) < 0.85, rng.random(len(x)) >= 0.9).astype(float)
    data = {"x": w, "y": y, "z": z}
    declaration = em.MisclassifiedExposure(
        "x", mx.ExposureError.shared(BinaryRates(0.85, 0.9, 1e-4, 1e-4)), covariates=("z",)
    )
    draws = declaration.realize(data, outcome="y", covariates=("z",), draws=5, seed=2)
    assert draws.draws.shape == (5, 400) and set(np.unique(draws.draws)) <= {0.0, 1.0}
    assert np.all((draws.probability >= 0) & (draws.probability <= 1))
    artifact = draws.export()
    back = em.consume_exposure_draws(
        artifact, variable="x", expected_declaration_identity=declaration.identity()
    )
    assert np.array_equal(back.draws, draws.draws)
    assert back.declaration_identity == declaration.identity()
    assert back.data_identity == draws.data_identity
    assert consume_posterior(artifact).n_draws == 5
    with pytest.raises(CausalValueError):
        em.consume_exposure_draws(artifact, variable="other")
    with pytest.raises(CausalValueError):
        em.consume_exposure_draws(artifact, variable="x", expected_declaration_identity="0" * 64)
    tables = draws.completed_tables(data)
    assert len(tables) == 5 and np.array_equal(tables[2]["x"], draws.column(2))
    assert np.array_equal(tables[2]["y"], y)
    # A different seed gives different draws; the same seed reproduces them.
    same = declaration.realize(data, outcome="y", covariates=("z",), draws=5, seed=2)
    other = declaration.realize(data, outcome="y", covariates=("z",), draws=5, seed=3)
    assert np.array_equal(same.draws, draws.draws) and not np.array_equal(other.draws, draws.draws)


# --------------------------------------------------------------------------- refusals


def test_an_unidentified_error_declaration_is_refused_before_any_draw():
    rng, z, x, y = planted(101, n=300)
    declaration = em.MisclassifiedExposure(
        "x", mx.ExposureError.shared(BinaryRates(0.55, 0.5)), covariates=("z",)
    )
    with pytest.raises(MeasurementModelRefusal) as refusal:
        declaration.realize({"x": x, "y": y, "z": z}, outcome="y", covariates=("z",), draws=3)
    assert refusal.value.detail == "measurement_model.youden_too_small"
    assert refusal.value.code == "measurement_not_identified"


def test_an_outcome_specific_error_needs_a_binary_outcome():
    rng, z, x, y = planted(111, n=300)
    declaration = em.MisclassifiedExposure(
        "x",
        mx.ExposureError.by_outcome(cases=BinaryRates(0.9, 0.8), controls=BinaryRates(0.7, 0.95)),
    )
    with pytest.raises(MeasurementModelRefusal) as refusal:
        declaration.realize({"x": x, "y": y, "z": z}, outcome="y", draws=3)
    assert refusal.value.detail == "measurement_model.binary_outcome_continuous_error"


def test_a_partially_declared_detection_function_gives_no_unit_draws():
    with pytest.raises(MeasurementModelRefusal) as refusal:
        em.DetectedExposure(
            "x",
            "count",
            mx.MonotoneDetection("nondecreasing", [(0.0, 0.1, 0.3), (5.0, 0.6, 0.9)]),
        )
    assert refusal.value.code == "measurement_not_identified"


def test_the_declaration_must_name_the_queried_exposure():
    rng, z, x, y = planted(121, n=300)
    declaration = em.MisclassifiedExposure(
        "z", mx.ExposureError.shared(BinaryRates(0.9, 0.9)), covariates=()
    )
    with pytest.raises(CausalValueError):
        ant.prepare({"x": x, "y": y, "z": z}, graph=GRAPH, query=QUERY, measurement=declaration)
