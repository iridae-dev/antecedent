from __future__ import annotations

import pytest
from antecedent.errors import CausalValueError
from antecedent.quasi import SyntheticControl, estimate_synthetic_control


def _synthetic_truth() -> dict[str, list[object]]:
    donor_trajectories = {
        "a": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "b": [0.0, 1.0, 0.0, 1.0, 2.0, 2.0],
        "c": [1.0, 0.0, 0.0, 1.0, 0.0, 2.0],
    }
    weights = {"a": 0.2, "b": 0.3, "c": 0.5}
    treated = [
        sum(weights[unit] * donor_trajectories[unit][period] for unit in donor_trajectories)
        + (5.0 if period >= 4 else 0.0)
        for period in range(6)
    ]
    rows: dict[str, list[object]] = {"y": [], "unit": [], "period": []}
    outcomes = {**donor_trajectories, "treated": treated}
    for unit, trajectory in outcomes.items():
        for period, value in enumerate(trajectory, start=1):
            rows["y"].append(value)
            rows["unit"].append(unit)
            rows["period"].append(period)
    return rows


def test_native_synthetic_control_recovers_known_effect_and_reports_diagnostics():
    result = estimate_synthetic_control(
        _synthetic_truth(), SyntheticControl("y", "unit", "period", "treated", 5)
    )
    assert result.estimate == pytest.approx(5.0, abs=1e-5)
    assert result.pre_treatment_rmse < 1e-5
    weights = dict(result.donor_weights)
    assert weights == pytest.approx({"a": 0.2, "b": 0.3, "c": 0.5}, abs=1e-4)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert result.n_donors == 3
    assert result.n_pre_periods == 4
    assert result.n_post_periods == 2
    assert result.effective_donors == pytest.approx(1 / (0.2**2 + 0.3**2 + 0.5**2), abs=1e-3)
    assert len(result.placebo_effects) == 3
    assert 0.0 <= result.placebo_rank_p_value <= 1.0
    assert result.uncertainty == "point_only_with_unlicensed_placebo_rank"
    assert result.support_status == "unlicensed_point_utility"
    assert "convex_donor_combination_is_a_valid_counterfactual" in result.assumptions
    assert "placebo_rank_assumes_exchangeable_donors_and_is_not_calibrated" in result.diagnostics


def test_synthetic_control_refuses_insufficient_donor_pool():
    rows = _synthetic_truth()
    rows = {key: [value for value, unit in zip(values, rows["unit"], strict=True) if unit != "c"]
            for key, values in rows.items()}
    with pytest.raises(CausalValueError, match="at least three donor units"):
        estimate_synthetic_control(
            rows, SyntheticControl("y", "unit", "period", "treated", 5)
        )


def test_synthetic_control_refuses_unbalanced_panel_and_no_pre_support():
    rows = _synthetic_truth()
    short = {key: values[:-1] for key, values in rows.items()}
    with pytest.raises(CausalValueError, match="balanced panel"):
        estimate_synthetic_control(
            short, SyntheticControl("y", "unit", "period", "treated", 5)
        )
    with pytest.raises(CausalValueError, match="at least two pre-periods"):
        estimate_synthetic_control(
            rows, SyntheticControl("y", "unit", "period", "treated", 1)
        )
