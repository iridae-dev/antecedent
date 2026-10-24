"""Target-native refutation and sensitivity ranges from Python: published numerical examples,
the refuter catalogue, derivative diagnostics, Bayesian target checks and a prepared study
refuting its own target."""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent import validation as v
from antecedent.errors import CausalError


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


def _log_spread(center: float, half: float, n: int = 101) -> list[float]:
    return [math.exp(math.log(center) + half * (2.0 * i / (n - 1) - 1.0)) for i in range(n)]


def test_vector_evalue_is_read_at_its_weakest_coordinate():
    strata = ["young", "old"]
    result = v.evalue_vector([3.9, 2.0], strata, lowers=[1.8, 1.2], uppers=[8.7, 3.3])
    scalar = [v.evalue(3.9, lower=1.8, upper=8.7), v.evalue(2.0, lower=1.2, upper=3.3)]
    assert result.summary["evalue"] == pytest.approx([s.summary["evalue"] for s in scalar])
    assert result.range is not None
    assert result.range.target.labels == ("young", "old")
    assert result.range.break_even == pytest.approx(min(s.summary["evalue"] for s in scalar))
    assert result.range.break_even_significance == pytest.approx(
        min(s.summary["evalue_limit"] for s in scalar)
    )
    # A scalar range cannot be cited for the vector target, and the labels must match.
    assert scalar[0].range.target.identity != result.range.target.identity
    with pytest.raises((ValueError, CausalError), match="one input per coordinate"):
        v.evalue_vector([3.9, 2.0], ["only_one_label"], lowers=[1.8, 1.2], uppers=[8.7, 3.3])


def test_posterior_evalue_carries_the_draws_and_reports_joint_retention():
    draws = [_log_spread(3.0, 0.15), _log_spread(1.6, 0.25)]
    result = v.evalue_posterior(draws, ["h1", "h2"], temporal=True)
    assert result.range is not None
    assert result.range.target.quadrant == "temporal.bayesian"
    assert result.range.parameter == "risk_ratio_posterior"
    assert result.summary["median_risk_ratio"] == pytest.approx([3.0, 1.6])
    retention = [r["probability"] for r in result.summary["retention"]]
    assert retention[0] == pytest.approx(1.0)
    assert all(b <= a + 1e-12 for a, b in zip(retention, retention[1:], strict=False))
    assert retention[-1] == pytest.approx(0.0)
    # The joint probability is the share of draws on the reported side after the adjustment.
    for point in result.summary["retention"]:
        s = point["strength"]
        factor = s * s / (2.0 * s - 1.0)
        kept = sum(all(c[d] / factor > 1.0 for c in draws) for d in range(101))
        assert point["probability"] == pytest.approx(kept / 101)
    with pytest.raises((ValueError, CausalError), match="positive and finite"):
        v.evalue_posterior([[1.0, -2.0]], ["x"], level=0.9)


def test_vector_and_posterior_partial_r2_ranges_match_the_closed_forms():
    vector = v.omitted_variable_bias_vector(
        [0.0973, -0.04], [0.0233, 0.02], [783.0, 500.0], ["a", "b"]
    )
    singles = [
        v.omitted_variable_bias(0.0973, 0.0233, 783.0),
        v.omitted_variable_bias(-0.04, 0.02, 500.0),
    ]
    assert vector.range.break_even == pytest.approx(
        min(s.summary["robustness_value"] for s in singles)
    )
    assert vector.range.tested[0].target == pytest.approx(1.0)
    n = 400
    normal = [float(x) for x in np.sort(np.random.default_rng(5).normal(size=n))]
    coefficients = [[0.5 + 0.05 * z for z in normal], [-0.3 + 0.05 * z for z in normal]]
    scale = [[1.2] * n, [0.8] * n]
    posterior = v.omitted_variable_bias_posterior(coefficients, scale, ["h1", "h2"], level=0.9)

    def reaches(median: float, bias_scale: float) -> float:
        c = (median / bias_scale) ** 2
        return 0.5 * (-c + math.sqrt(c * c + 4.0 * c))

    expected = min(reaches(0.5, 1.2), reaches(0.3, 0.8))
    assert posterior.range.break_even == pytest.approx(expected, abs=0.02)
    assert posterior.range.break_even_significance <= posterior.range.break_even + 1e-9
    assert posterior.range.tested[0].test == pytest.approx(1.0, abs=1e-3)


def test_vector_rosenbaum_reports_the_smallest_break_even():
    strong = [1.0 + 0.2 * i for i in range(1, 15)]
    null = [1.0 if i % 2 == 0 else -1.0 for i in range(1, 15)]
    both = v.rosenbaum_bounds_vector([strong, strong], ["a", "b"], gammas=[1.0, 2.0, 3.0])
    single = v.rosenbaum_bounds(strong, gammas=[1.0, 2.0, 3.0])
    assert both.range.break_even_significance == pytest.approx(single.range.break_even_significance)
    flat = v.rosenbaum_bounds_vector([strong, null], ["a", "b"], gammas=[1.0, 2.0, 3.0])
    assert flat.range.break_even_significance is None


def _mediation_columns(n: int = 160):
    t = np.array([float(i % 2) for i in range(n)])
    x = np.sin(0.3 * np.arange(n))
    m = 0.8 * t + 0.5 * x + 0.4 * np.sin(1.7 * np.arange(n))
    y1 = 0.3 * t + 1.2 * m - 0.7 * x + 0.5 * np.cos(0.9 * np.arange(n))
    y2 = -0.4 * t + 0.5 * m + 0.2 * x + 0.3 * np.sin(2.3 * np.arange(n))
    return t, m, y1, y2, x


def test_mediation_vector_reads_the_weakest_outcome_and_the_total_effect_identity():
    t, m, y1, y2, x = _mediation_columns()
    labels = ["y1", "y2"]
    indirect = v.mediation_sensitivity_vector(
        t.tolist(), m.tolist(), [y1.tolist(), y2.tolist()], [x.tolist()], labels
    )
    direct = v.mediation_sensitivity_vector(
        t.tolist(),
        m.tolist(),
        [y1.tolist(), y2.tolist()],
        [x.tolist()],
        labels,
        component="direct",
    )
    assert indirect.range.assumption == "mediator_outcome_confounding"
    # Closed form: the indirect effect of each outcome vanishes at its residual correlation.
    rhos = [c["rho_tilde"] for c in indirect.summary["coordinates"]]
    assert indirect.range.break_even == pytest.approx(min(rhos, key=abs), abs=1e-12)
    # Total = direct + indirect, and the direct effect at rho = 0 is the structural coefficient.
    for i, y in enumerate([y1, y2]):
        design = np.column_stack([np.ones_like(t), t, m, x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        coords = direct.summary["coordinates"][i]
        assert coords["direct_at_zero"] == pytest.approx(beta[1], abs=1e-9)
        assert coords["indirect_at_zero"] + coords["direct_at_zero"] == pytest.approx(
            coords["total_effect"], abs=1e-9
        )
    first = indirect.range.tested[0]
    assert first.parameter < 0
    zero = next(p for p in indirect.range.tested if p.parameter == 0.0)
    assert zero.target == pytest.approx(1.0)


def test_mediation_posterior_carries_each_draw_through_the_formula():
    n = 101
    spread = np.linspace(-1.0, 1.0, n)
    beta = [(0.8 + 0.1 * spread).tolist(), (0.5 + 0.1 * spread).tolist()]
    result = v.mediation_sensitivity_posterior(
        beta,
        [[1.1] * n, [0.9] * n],
        [[0.7] * n, [0.6] * n],
        [[0.2] * n, [0.35] * n],
        [(1.4 + 0.05 * spread).tolist(), (0.4 + 0.05 * spread).tolist()],
        ["h1", "h2"],
    )
    assert result.range.assumption == "mediator_outcome_confounding"
    assert result.range.break_even == pytest.approx(0.2, abs=1e-6)
    by_rho = {p.parameter: p for p in result.range.tested}
    assert by_rho[0.0].test == pytest.approx(1.0)
    assert by_rho[0.1].test > 0.999
    assert by_rho[0.3].test < 1e-12
    with pytest.raises((ValueError, CausalError), match="one column of draws"):
        v.mediation_sensitivity_posterior(
            beta, [[1.1] * n], [[0.7] * n], [[0.2] * n], beta, ["h1", "h2"]
        )


def test_the_vector_ranges_bind_to_the_assumption_of_their_setting():
    plain = v.evalue_vector([3.9], ["e"], lowers=[1.8], uppers=[8.7])
    mediation = v.evalue_vector([3.9], ["e"], lowers=[1.8], uppers=[8.7], model="mediation")
    msm = v.evalue_vector([3.9], ["e"], lowers=[1.8], uppers=[8.7], model="marginal_structural")
    assert mediation.range.assumption == "mediator_outcome_confounding_risk_ratio"
    assert msm.range.assumption == "sequential_exchangeability_risk_ratio"
    assert mediation.range.break_even == plain.range.break_even
    assert len({plain.range.identity, mediation.range.identity, msm.range.identity}) == 3
    ovb = v.omitted_variable_bias_vector(
        [0.0973], [0.0233], [783.0], ["a"], model="marginal_structural"
    )
    assert ovb.range.assumption == "sequential_exchangeability_partial_r2"
    with pytest.raises(ValueError, match="unknown model"):
        v.evalue_vector([3.9], ["e"], lowers=[1.8], uppers=[8.7], model="nonsense")


def test_msm_parameters_are_fitted_with_a_robust_error_and_read_at_the_weakest():
    n = 400
    i = np.arange(n)
    dose = (i % 5).astype(float)
    trend = np.sin(0.37 * i)
    outcome = 1.0 + 0.5 * dose + 0.05 * trend + 0.8 * np.sin(1.3 * i)
    weights = 0.5 + (i % 3) * 0.5
    result = v.msm_parameter_sensitivity(
        outcome.tolist(),
        [dose.tolist(), trend.tolist()],
        weights.tolist(),
        ["dose", "trend"],
    )
    assert result.range.assumption == "sequential_exchangeability_partial_r2"
    estimate = result.summary["estimate"]
    # Independent weighted least squares.
    design = np.column_stack([np.ones(n), dose, trend])
    sw = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], outcome * sw, rcond=None)
    assert estimate == pytest.approx(beta[1:], abs=1e-9)
    # HC1 sandwich computed independently.
    bread = np.linalg.inv(design.T @ (design * weights[:, None]))
    resid = outcome - design @ beta
    meat = design.T @ (design * (weights * resid)[:, None] ** 2)
    se = np.sqrt(np.diag(bread @ meat @ bread) * n / (n - 3))[1:]
    assert result.summary["standard_error"] == pytest.approx(se, rel=1e-9)
    neff = weights.sum() ** 2 / (weights**2).sum()
    assert result.summary["dof"][0] == pytest.approx(neff - 3)
    assert result.range.break_even == pytest.approx(min(result.summary["robustness_value"]))
    with pytest.raises((ValueError, CausalError), match="non-negative weights"):
        v.msm_parameter_sensitivity(
            outcome.tolist(), [dose.tolist()], (-weights).tolist(), ["dose"]
        )


def test_mediation_paths_have_closed_form_robustness_values_and_a_product_for_the_indirect():
    t, m, y1, _, x = _mediation_columns()
    args = (t.tolist(), m.tolist(), [y1.tolist()], [x.tolist()], ["y1"])
    ranges = {
        path: v.mediation_path_sensitivity(*args, path=path)
        for path in ("treatment_mediator", "mediator_outcome", "treatment_outcome", "indirect")
    }
    assert ranges["treatment_mediator"].range.assumption == (
        "treatment_mediator_confounding_partial_r2"
    )
    assert ranges["treatment_outcome"].range.assumption == (
        "treatment_outcome_confounding_partial_r2"
    )
    assert ranges["indirect"].range.assumption == "indirect_effect_confounding_partial_r2"
    coordinate = ranges["indirect"].summary["coordinates"][0]
    for path in ("treatment_mediator", "mediator_outcome", "treatment_outcome"):
        rv = ranges[path].summary["coordinates"][0][path]["robustness_value"]
        assert ranges[path].range.break_even == pytest.approx(rv, abs=1e-12)
    # With q = 1 the product vanishes when either path does.
    smaller = min(
        coordinate["treatment_mediator"]["robustness_value"],
        coordinate["mediator_outcome"]["robustness_value"],
    )
    assert ranges["indirect"].range.break_even == pytest.approx(smaller, abs=1e-9)
    # The treatment-mediator path is the OLS coefficient of the mediator on the treatment.
    design = np.column_stack([np.ones_like(t), t, x])
    beta, *_ = np.linalg.lstsq(design, m, rcond=None)
    assert coordinate["treatment_mediator"]["estimate"] == pytest.approx(beta[1], abs=1e-9)
    with pytest.raises(ValueError, match="unknown path"):
        v.mediation_path_sensitivity(*args, path="sideways")


def test_msm_weight_refit_standard_error_carries_the_weight_estimation():
    rng = np.random.default_rng(5)
    n = 500
    history = rng.normal(size=n)
    a = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-history))).astype(float)
    y = 1.0 + 2.0 * a + history + rng.normal(size=n)
    refit = v.msm_parameter_sensitivity_refit(
        y.tolist(), [a.tolist()], [a.tolist()], [[]], [[history.tolist()]], ["a"], replicates=99
    )
    assert refit.range.assumption == "sequential_exchangeability_partial_r2"
    assert refit.summary["estimate"][0] == pytest.approx(2.0, abs=0.5)
    assert refit.summary["standard_error"][0] > 0
    assert refit.summary["standard_error"][0] != pytest.approx(
        refit.summary["fixed_weight_standard_error"][0], abs=1e-6
    )
    assert sum(refit.summary["replicates"]) == 99
    with pytest.raises((ValueError, CausalError), match="at least 99"):
        v.msm_parameter_sensitivity_refit(
            y.tolist(), [a.tolist()], [a.tolist()], [[]], [[history.tolist()]], ["a"], replicates=10
        )
