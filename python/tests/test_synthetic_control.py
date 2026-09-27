from __future__ import annotations

import math
import pytest
from antecedent import analyze
from antecedent.errors import CausalCompileError, CausalUnsupportedError, CausalValueError
from antecedent.estimation import PreparedAnalysis
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


def test_retained_analyze_and_prepare_match_direct_point_result():
    rows = _synthetic_truth()
    query = SyntheticControl("y", "unit", "period", "treated", 5)
    direct = estimate_synthetic_control(rows, query)
    result = analyze(rows, query=query)
    fit = result.synthetic_control
    assert fit is not None
    assert fit.estimate == pytest.approx(direct.estimate, abs=1e-5)
    assert dict(fit.donor_weights) == pytest.approx(dict(direct.donor_weights))
    assert fit.placebo_effects == pytest.approx(direct.placebo_effects)
    assert fit.support_status == "unlicensed_point_utility"
    assert fit.uncertainty == "point_only_with_unlicensed_placebo_rank"
    assert math.isnan(result.estimate.se_analytic)
    prepared = PreparedAnalysis.prepare(rows, query=query)
    refreshed = prepared.estimate(rows)
    assert refreshed.synthetic_control == fit


def test_retained_synthetic_control_refuses_changed_design_and_bad_donors():
    rows = _synthetic_truth()
    query = SyntheticControl("y", "unit", "period", "treated", 5)
    prepared = PreparedAnalysis.prepare(rows, query=query)
    changed = {key: list(value) for key, value in rows.items()}
    changed["unit"][0] = "different"
    with pytest.raises(CausalUnsupportedError, match="prepared unit and period row order"):
        prepared.estimate(changed)
    short = {key: [value for value, unit in zip(values, rows["unit"], strict=True) if unit != "c"]
             for key, values in rows.items()}
    with pytest.raises((CausalValueError, CausalCompileError), match="at least three donor units"):
        analyze(short, query=query)


def test_uniform_unit_randomization_reports_exact_sharp_null_p_value_only():
    rows = _synthetic_truth()
    query = SyntheticControl("y", "unit", "period", "treated", 5, uniform_unit_randomization=True)
    result = analyze(rows, query=query)
    fit = result.synthetic_control
    assert fit is not None
    assert fit.estimate == pytest.approx(5.0, abs=1e-5)
    assert len(fit.randomization_statistics) == 4
    observed = dict(fit.randomization_statistics)["treated"]
    assert fit.randomization_p_value == pytest.approx(
        sum(statistic >= observed for _, statistic in fit.randomization_statistics) / 4
    )
    assert fit.uncertainty == "point_only_with_exact_unit_randomization_p_value_no_interval"
    assert "exact_uniform_unit_randomization_sharp_null_test" in fit.diagnostics
    assert math.isnan(result.estimate.se_analytic)
    assert estimate_synthetic_control(rows, query) == fit
    assert PreparedAnalysis.prepare(rows, query=query).estimate(rows).synthetic_control == fit


def test_augmented_synthetic_control_uses_retained_donor_model_and_refuses_invalid_options():
    rows: dict[str, list[object]] = {"y": [], "unit": [], "period": []}
    for unit, position in (("a", 0.0), ("b", 1.0), ("c", 2.0), ("treated", 3.0)):
        for period in range(1, 5):
            rows["unit"].append(unit)
            rows["period"].append(period)
            rows["y"].append(position * period if period < 4 else 4 * position + (5 if unit == "treated" else 0))
    query = SyntheticControl("y", "unit", "period", "treated", 4, augmentation_ridge=1e-8)
    result = analyze(rows, query=query)
    fit = result.synthetic_control
    assert fit is not None
    assert fit.unadjusted_effect == pytest.approx(9.0, abs=1e-5)
    assert fit.outcome_model_correction == pytest.approx(4.0, abs=1e-4)
    assert fit.estimate == pytest.approx(5.0, abs=1e-4)
    assert fit.uncertainty == "point_only_augmented_no_interval"
    assert fit.support_status == "unlicensed_point_utility"
    assert "donor_ridge_outcome_model_transports" in fit.assumptions
    assert math.isnan(result.estimate.se_analytic)
    assert estimate_synthetic_control(rows, query) == fit
    assert PreparedAnalysis.prepare(rows, query=query).estimate(rows).synthetic_control == fit
    with pytest.raises(CausalValueError, match="positive"):
        SyntheticControl("y", "unit", "period", "treated", 4, augmentation_ridge=0.0)
    exact_query = SyntheticControl("y", "unit", "period", "treated", 4,
                                   uniform_unit_randomization=True, augmentation_ridge=1.0)
    exact = analyze(rows, query=exact_query).synthetic_control
    assert exact is not None
    assert exact.uncertainty == "point_only_augmented_with_exact_unit_randomization_p_value_no_interval"
    assert len(exact.randomization_statistics) == 4
    assert dict(exact.randomization_statistics)["treated"] == pytest.approx(abs(exact.estimate))
    assert exact.randomization_p_value == pytest.approx(
        sum(statistic >= abs(exact.estimate) for _, statistic in exact.randomization_statistics) / 4
    )
    assert estimate_synthetic_control(rows, exact_query) == exact
