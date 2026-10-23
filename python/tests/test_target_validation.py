"""Target-native refutation and sensitivity ranges from Python: published numerical examples,
the refuter catalogue, derivative diagnostics, Bayesian target checks and a prepared study
refuting its own target."""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent import validation as v


def test_evalue_reproduces_vanderweele_and_ding():
    # RR 3.9 (1.8, 8.7): E-value 7.26, confidence-limit E-value 3.0.
    result = v.evalue(3.9, lower=1.8, upper=8.7)
    assert result.summary["evalue"] == pytest.approx(7.26, abs=0.005)
    assert result.summary["evalue_limit"] == pytest.approx(3.0, abs=1e-12)
    assert result.range is not None
    assert result.range.assumption == "unmeasured_confounding_risk_ratio"
    assert result.range.break_even == pytest.approx(result.summary["evalue"])
    assert result.range.calibration == "unmeasured"


def test_evalue_scales():
    common = v.evalue(4.0, lower=2.0, upper=8.0, scale="odds_ratio_common")
    assert common.summary["risk_ratio"] == pytest.approx(2.0)
    smd = v.evalue(0.5, standard_error=0.1, scale="standardized_mean_difference")
    assert smd.summary["risk_ratio"] == pytest.approx(math.exp(0.91 * 0.5))
    with pytest.raises(ValueError, match="unknown effect scale"):
        v.evalue(2.0, scale="not_a_scale")


def test_omitted_variable_bias_reproduces_the_darfur_example():
    result = v.omitted_variable_bias(0.09731582, 0.02325654, 783.0)
    assert result.summary["partial_r2_treatment"] == pytest.approx(0.0219, abs=5e-5)
    assert result.summary["robustness_value"] == pytest.approx(0.1388, abs=5e-5)
    assert result.summary["robustness_value_alpha"] == pytest.approx(0.0763, abs=5e-5)
    bounded = v.omitted_variable_bias(
        0.09731582, 0.02325654, 783.0, benchmarks=[("female", 0.0, 0.1)], kd=2.0
    )
    assert bounded.summary["benchmarks"][0]["covariate"] == "female"


def test_omitted_variable_bias_from_data_benchmarks_every_covariate():
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    t = 0.6 * x1 + rng.normal(size=n)
    y = 1.0 + 0.5 * t + x1 - 0.3 * x2 + rng.normal(size=n)
    result = v.omitted_variable_bias_from_data(
        {"y": y, "t": t, "x1": x1, "x2": x2}, outcome="y", treatment="t", covariates=["x1", "x2"]
    )
    names = [b["covariate"] for b in result.summary["benchmarks"]]
    assert names == ["x1", "x2"]
    assert result.summary["estimate"] == pytest.approx(0.5, abs=0.15)


def test_rosenbaum_bounds_and_amplification():
    result = v.rosenbaum_bounds([1.0, 2.0, 3.0, 4.0, 5.0], gammas=[1.0, 2.0])
    grid = result.summary["grid"]
    assert grid[0]["p_value"] == pytest.approx(1.0 / 32.0)
    assert grid[1]["p_value"] == pytest.approx((2.0 / 3.0) ** 5)
    assert result.summary["exact"] is True
    assert v.amplify(1.25, 2.0) == pytest.approx(2.0)
    assert v.amplify(2.0, 3.0) == pytest.approx(5.0)
    assert v.amplify(2.0, 1.5) is None


def test_matched_sets_have_a_break_even_gamma():
    sets = [([b + 1.5, b, b + 0.2, b - 0.1], 0) for b in [0.0, 1.0, 2.0, 3.0, 4.0] * 6]
    result = v.matched_sets_sensitivity(sets)
    assert result.summary["gamma_break_even"] > 1.0


def test_mediation_sensitivity_breaks_even_at_the_residual_correlation():
    rng = np.random.default_rng(5)
    n = 500
    t = rng.integers(0, 2, size=n).astype(float)
    m = 0.7 * t + rng.normal(size=n)
    y = 0.2 * t + 0.9 * m + rng.normal(size=n)
    result = v.mediation_sensitivity(
        {"t": t, "m": m, "y": y}, treatment="t", mediator="m", outcome="y"
    )
    assert result.range is not None
    assert result.range.break_even == pytest.approx(result.summary["rho_tilde"])
    assert result.range.target.family == "mediation"


def test_the_catalogue_covers_every_quadrant():
    for temporal in (False, True):
        for bayesian in (False, True):
            plan = v.target_refutation_plan(
                "world_valued", temporal=temporal, bayesian=bayesian, suite="full"
            )
            assert "target.placebo_treatment" in plan
            assert ("target.pre_trend" in plan) == temporal
            assert ("target.prior_sensitivity" in plan) == bayesian
    assert "target.sequential_ignorability" in v.target_refutation_plan("mediation")
    assert "target.finite_difference_consistency" in v.target_refutation_plan(
        "response_derivative", suite="cheap"
    )


def test_derivative_diagnostics():
    points = [-1.0, 0.0, 0.5, 1.2]
    good = v.derivative_consistency(math.sin, points, [math.cos(x) for x in points])
    assert good.passed
    bad = v.derivative_consistency(
        math.sin, points, [math.cos(x) + 0.1 for x in points], tolerance=1e-3
    )
    assert not bad.passed
    assert not v.local_linearity(lambda x: math.exp(5 * x), [0.0, 0.5], step=0.5).passed
    assert v.local_linearity(lambda x: 2 * x + 1, [0.0, 0.5], step=0.5).passed


def test_bayesian_target_checks():
    rng = np.random.default_rng(9)
    base = rng.normal(size=(200, 2))
    near = base + np.array([0.05, 0.0])
    far = base + np.array([3.0, 0.0])
    ok = v.prior_sensitivity_target(["a", "b"], base, {"wide": near})
    assert ok.passed
    shifted = v.prior_sensitivity_target(["a", "b"], base, {"tight": far})
    assert not shifted.passed
    draws = rng.normal(scale=0.01, size=(200, 1))
    replicated = draws + rng.normal(size=(200, 1))
    assert v.posterior_predictive_target(["a"], [0.0], draws, replicated).passed
    assert not v.posterior_predictive_target(["a"], [25.0], draws, replicated).passed


def test_a_prepared_study_refutes_its_own_target():
    rng = np.random.default_rng(11)
    n = 600
    z = rng.normal(size=n)
    t = 0.5 * z + rng.normal(size=n)
    y = 1.0 + 2.0 * t + z + rng.normal(size=n)
    data = {"t": t, "y": y, "z": z}
    study = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        bootstrap=0,
        refute="none",
    )
    outcome = study.refute_target(data, suite="cheap", replicates=12)
    assert isinstance(outcome, v.TargetRefutation)
    assert outcome.target.family == "scalar_average_effect"
    refuters = {report.refuter for report in outcome.reports}
    assert "target.placebo_treatment" in refuters
    placebo = next(r for r in outcome.reports if r.refuter == "target.placebo_treatment")
    assert placebo.passed
    assert all(report.target == outcome.target.identity for report in outcome.reports)
    assert outcome.calibration == "unmeasured"


def test_constructed_refuters_pass_on_a_sound_design_and_flag_a_planted_confounder():
    def sample(seed: int, planted: bool) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        n = 1200
        z = rng.normal(size=n)
        u = rng.normal(size=n) if planted else np.zeros(n)
        t = (z + 1.5 * u + rng.normal(size=n) > 0).astype(float)
        x = (z + 1.5 * u + rng.normal(size=n) > 0).astype(float)
        w = z + rng.normal(size=n)
        y = 1.0 + 2.0 * t + z + 1.5 * u + rng.normal(size=n)
        return {"t": t, "y": y, "z": z, "x": x, "w": w}

    graph = [("z", "t"), ("z", "y"), ("t", "y"), ("z", "x"), ("z", "w")]
    clean = sample(1, False)
    study = ant.prepare(
        clean, graph=graph, query=ant.AverageEffect("t", "y"), bootstrap=0, refute="none"
    )
    outcome = study.refute_target(clean, suite="full", replicates=12, common_cause_strength=0.5)
    passed = {r.refuter: r.passed for r in outcome.reports}
    for refuter in (
        "target.random_common_cause",
        "target.negative_control_exposure",
        "target.negative_control_outcome",
    ):
        assert passed[refuter], refuter
    planted = sample(2, True)
    study = ant.prepare(
        planted, graph=graph, query=ant.AverageEffect("t", "y"), bootstrap=0, refute="none"
    )
    outcome = study.refute_target(planted, suite="full", replicates=12)
    flagged = {r.refuter: r.passed for r in outcome.reports}
    assert flagged["target.negative_control_exposure"] is False
