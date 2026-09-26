import numpy as np
import pytest
from antecedent._native import estimate_complier_effect as native_estimate_complier_effect
from antecedent.errors import CausalValueError
from antecedent.experiment import estimate_complier_effect, estimate_cuped_effect


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
    assert result.uncertainty == "asymptotic_influence_function_standard_error"
    assert "exclusion restriction" in result.assumptions[1]
    native = native_estimate_complier_effect(outcome, assignment, receipt, np.array([0.5]))
    assert tuple(native) == pytest.approx(
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
