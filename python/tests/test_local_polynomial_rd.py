from __future__ import annotations

import numpy as np
import pytest
from antecedent import analyze
from antecedent.estimation import PreparedAnalysis
from antecedent.errors import CausalValueError
from antecedent.quasi import (
    FuzzyRegressionDiscontinuity,
    RegressionKink,
    estimate_fuzzy_rd,
    estimate_regression_kink,
)


def _running_grid() -> list[float]:
    return [sign * step / 20 for sign in (-1, 1) for step in range(1, 20)]


def test_fuzzy_rd_local_quadratic_recovers_known_local_effect():
    running: list[float] = []
    treatment: list[float] = []
    outcome: list[float] = []
    for step in range(1, 80):
        for sign, assignments in ((-1, (1, 0, 0, 0)), (1, (1, 1, 1, 0))):
            score = sign * step / 80
            for assignment, noise in zip(assignments, (-0.02, -0.01, 0.01, 0.02), strict=True):
                treated = float(assignment)
                running.append(score)
                treatment.append(treated)
                outcome.append(1.0 + 2.0 * score + 0.5 * score**2 + 3.0 * treated + noise)
    result = estimate_fuzzy_rd(
        {"x": running, "t": treatment, "y": outcome},
        FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 1.0),
    )
    assert result.estimate == pytest.approx(3.0, abs=1e-10)
    assert result.first_stage_discontinuity == pytest.approx(0.5, abs=1e-10)
    assert result.reduced_form_discontinuity == pytest.approx(1.5, abs=1e-10)
    assert result.observations_left == 316
    assert result.observations_right == 316
    assert result.standard_error > 0
    assert result.ci_lower < result.estimate < result.ci_upper
    assert result.uncertainty == "local_quadratic_rbc_hc0_delta_normal_unvalidated"
    assert result.support_status == "unlicensed_point_utility"
    assert "exclusion_restriction_for_threshold_instrument" in result.assumptions
    assert "cubic_pilot_bias_correction_at_same_bandwidth" in result.diagnostics
    assert "normal_approximation_interval_not_calibrated_or_licensed" in result.diagnostics


def test_fuzzy_regression_kink_recovers_known_local_effect():
    running = _running_grid()
    treatment = [1.0 + 0.2 * x + 0.8 * max(x, 0.0) for x in running]
    outcome = [2.0 + 1.5 * x + 0.5 * x**2 + 3.0 * t for x, t in zip(running, treatment, strict=True)]
    result = estimate_regression_kink(
        {"x": running, "dose": treatment, "y": outcome},
        RegressionKink("y", "dose", "x", 0.0, 1.0),
    )
    assert result.estimate == pytest.approx(3.0, abs=1e-10)
    assert result.first_stage_discontinuity == pytest.approx(0.8, abs=1e-10)
    assert result.design == "fuzzy_regression_kink_local_quadratic"
    assert "potential_outcome_derivatives_are_smooth_at_cutoff" in result.assumptions
    assert result.uncertainty == "local_quadratic_rbc_hc0_delta_normal_unvalidated"


@pytest.mark.parametrize("kink", [False, True])
def test_retained_local_ratio_uses_main_analyze_flow_without_interval(kink: bool):
    running: list[float] = []
    treatment: list[float] = []
    outcome: list[float] = []
    for step in range(1, 80):
        for sign in (-1, 1):
            score = sign * step / 80
            for replicate in range(4):
                dose = (1.0 + 0.2 * score + 0.8 * max(score, 0.0)) if kink else float(
                    replicate == 0 if sign < 0 else replicate != 3
                )
                running.append(score)
                treatment.append(dose)
                outcome.append(1.0 + 2.0 * score + 0.5 * score**2 + 3.0 * dose)
    rows = {"x": running, "t": treatment, "y": outcome}
    query = (RegressionKink if kink else FuzzyRegressionDiscontinuity)("y", "t", "x", 0.0, 1.0)
    result = analyze(rows, query=query)
    fit = result.local_polynomial_ratio
    assert fit is not None
    assert fit.estimate == pytest.approx(3.0, abs=1e-7)
    assert fit.ci_lower is None and fit.ci_upper is None
    assert np.isnan(result.estimate.se_analytic)
    direct = (estimate_regression_kink if kink else estimate_fuzzy_rd)(rows, query)
    assert fit.standard_error == pytest.approx(direct.standard_error)
    assert fit.reduced_form_standard_error == pytest.approx(direct.reduced_form_standard_error)
    assert fit.first_stage_standard_error == pytest.approx(direct.first_stage_standard_error)
    assert (fit.cutoff, fit.bandwidth, fit.kink) == (query.cutoff, query.bandwidth, kink)
    assert fit.uncertainty == "rbc_point_with_unvalidated_hc0_standard_error_no_interval"
    assert fit.observations_left == fit.observations_right == 316
    assert "no_calibrated_interval" in fit.diagnostics
    assert "cubic_pilot_bias_correction_at_same_bandwidth" in fit.diagnostics
    assert PreparedAnalysis.prepare(rows, query=query).estimate(rows).local_polynomial_ratio == fit


def test_local_polynomial_rd_refuses_weak_first_stage_and_sparse_side_support():
    running = _running_grid()
    treatment = [0.4 + 0.2 * x for x in running]
    outcome = [1.0 + 2.0 * x + treatment_value for x, treatment_value in zip(running, treatment, strict=True)]
    with pytest.raises(CausalValueError, match="discontinuity is too small"):
        estimate_fuzzy_rd(
            {"x": running, "t": treatment, "y": outcome},
            FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 1.0),
        )
    rng = np.random.default_rng(4)
    weak_probability = 0.45 + 0.02 * np.asarray(running) + 0.01 * (np.asarray(running) > 0)
    weak_treatment = rng.binomial(1, weak_probability).astype(float)
    with pytest.raises(CausalValueError, match="weak first stage"):
        estimate_fuzzy_rd(
            {"x": running, "t": weak_treatment, "y": outcome},
            FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 1.0),
        )
    sparse = {"x": [-0.2, -0.1, 0.1, 0.2], "t": [0, 0, 1, 1], "y": [0, 0, 1, 1]}
    with pytest.raises(CausalValueError, match="lacks full-rank support"):
        estimate_fuzzy_rd(
            sparse, FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 1.0)
        )
    with pytest.raises(CausalValueError, match="lacks full-rank support"):
        estimate_fuzzy_rd(
            {"x": running, "t": [float(x > 0) for x in running], "y": [float(x > 0) for x in running]},
            FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 0.04),
        )


def test_fuzzy_rd_rbc_interval_has_nominal_coverage_in_seeded_known_truth_fixture():
    rng = np.random.default_rng(20260926)
    running = np.linspace(-1.0, 1.0, 602, dtype=np.float64)[1:-1]
    truth = 2.0
    covered = 0
    accepted = 0
    repetitions = 350
    for _ in range(repetitions):
        probability = 0.1 + 0.02 * running + 0.8 * (running > 0.0)
        treatment = rng.binomial(1, probability).astype(np.float64)
        outcome = (
            0.7
            + 0.5 * running
            + 0.3 * running**2
            + 0.1 * running**3
            + truth * treatment
            + rng.normal(0.0, 1.0, len(running))
        )
        try:
            result = estimate_fuzzy_rd(
                {"x": running, "t": treatment, "y": outcome},
                FuzzyRegressionDiscontinuity("y", "t", "x", 0.0, 0.5),
            )
        except CausalValueError as error:
            assert "weak first stage" in str(error)
            continue
        accepted += 1
        covered += result.ci_lower <= truth <= result.ci_upper
    assert accepted >= repetitions * 0.9
    coverage = covered / accepted
    assert 0.90 <= coverage <= 0.99
