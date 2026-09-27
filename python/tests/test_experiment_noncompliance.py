import antecedent as ant
import numpy as np
import pytest
from antecedent._native import estimate_complier_effect as native_estimate_complier_effect
from antecedent.errors import CausalValueError
from antecedent.experiment import (
    ComplierEffect,
    ExperimentDesign,
    TreatmentOnTreated,
    estimate_complier_effect,
    estimate_cuped_effect,
)
from antecedent.interference import BernoulliAssignment


def test_randomized_noncompliance_recovers_known_complier_effect_and_itt():
    assignment = [False, True] * 8
    # Four always-takers, eight never-takers, and four compliers; the sequence
    # is balanced over encouragement arms within compliance types.
    receipt = [True] * 4 + [False] * 8 + [False, True, False, True]
    outcome = 4.0 * np.asarray(receipt, dtype=float)
    # Baseline is balanced within assignment; treatment receipt changes only
    # among compliers, so ITT = 1 and first stage = 0.25.
    result = estimate_complier_effect(outcome, assignment, receipt, 0.5)
    assert result.intention_to_treat_effect == pytest.approx(1.0)
    assert result.first_stage_effect == pytest.approx(0.25)
    assert result.complier_average_causal_effect == pytest.approx(4.0)
    assert result.standard_error >= 0.0
    # Sixteen rows are far below the calibrated interval support, so the direct
    # utility withholds the interval and reports only the standard error.
    assert result.interval_95 is None
    assert result.uncertainty == "asymptotic_influence_function_standard_error"
    assert "exclusion restriction" in result.assumptions[1]
    native = native_estimate_complier_effect(outcome, assignment, receipt, np.array([0.5]))
    assert native[4] is None
    assert tuple(native[:4]) == pytest.approx(
        (result.intention_to_treat_effect, result.first_stage_effect,
         result.complier_average_causal_effect, result.standard_error)
    )


def test_randomized_noncompliance_refuses_zero_first_stage():
    with pytest.raises(CausalValueError, match="first stage must be positive"):
        estimate_complier_effect([1.0, 2.0], [False, True], [False, False], 0.5)
    with pytest.raises(CausalValueError, match="first stage must be positive"):
        estimate_complier_effect(
            [1.0, 0.0, 1.0, 0.0],
            [False, True, False, True],
            [True, False, True, False],
            0.5,
        )


def test_cuped_adjustment_removes_preassignment_signal_and_reports_se():
    assignment = [True, False, False, True] * 4
    covariate = np.arange(16, dtype=float)
    outcome = 2.0 * np.asarray(assignment, dtype=float) + 5.0 * covariate
    result = estimate_cuped_effect(outcome, covariate, assignment, 0.5)
    assert result.effect == pytest.approx(2.0)
    assert result.adjustment_coefficient == pytest.approx(5.0)
    assert result.standard_error >= 0.0
    assert "pre-treatment" in result.assumptions[1]


def _complier_route_data(n: int, *, one_sided: bool):
    assigned = [i % 2 == 0 for i in range(n)]
    if one_sided:
        received = [assigned[i] and i % 5 != 0 for i in range(n)]
    else:
        received = [(i % 5 != 0) if assigned[i] else (i % 7 == 0) for i in range(n)]
    outcome = np.asarray(
        [1.0 + 2.0 * float(received[i]) + 0.5 * np.sin(0.3 * i) for i in range(n)],
        dtype=float,
    )
    design = ExperimentDesign(
        BernoulliAssignment(0.5), assigned,
        [f"unit-{i}" for i in range(n)], [f"row-{i}" for i in range(n)],
    )
    return outcome, assigned, received, design


def test_retained_complier_route_licenses_interval_matching_direct_utility():
    outcome, assigned, received, design = _complier_route_data(400, one_sided=False)
    result = ant.analyze(
        {"outcome": outcome}, query=ComplierEffect("outcome", design, received), refute="none"
    )
    fit = result.randomized_effect
    assert fit is not None
    assert fit.estimand == "cace_late"
    assert fit.interval_95 is not None
    assert fit.interval_95[0] < fit.effect < fit.interval_95[1]
    assert fit.support_status == "licensed"
    assert result.evidence_status == "licensed"
    # The direct utility reconciles to the exact licensed interval.
    direct = estimate_complier_effect(outcome, assigned, received, 0.5)
    assert direct.interval_95 is not None
    assert direct.interval_95 == pytest.approx(fit.interval_95)
    assert direct.support_status == "unlicensed_direct_estimator_utility"


def test_retained_treatment_on_treated_route_licenses_interval():
    outcome, assigned, received, design = _complier_route_data(400, one_sided=True)
    result = ant.analyze(
        {"outcome": outcome}, query=TreatmentOnTreated("outcome", design, received), refute="none"
    )
    fit = result.randomized_effect
    assert fit is not None
    assert fit.estimand == "treatment_on_treated"
    assert fit.interval_95 is not None
    assert fit.support_status == "licensed"
    direct = estimate_complier_effect(outcome, assigned, received, 0.5)
    assert direct.interval_95 == pytest.approx(fit.interval_95)
