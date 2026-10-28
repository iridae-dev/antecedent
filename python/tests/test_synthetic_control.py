from __future__ import annotations

import math

import antecedent
import pytest
from antecedent import analyze
from antecedent.errors import (
    CausalCompileError,
    CausalError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.outcomes import require_success
from antecedent.quasi import SyntheticControl


def _analyzed(rows, query):
    return require_success(analyze(rows, query=query))


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
    result = _analyzed(
        _synthetic_truth(), SyntheticControl("y", "unit", "period", "treated", 5)
    ).synthetic_control
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
    rows = {
        key: [value for value, unit in zip(values, rows["unit"], strict=True) if unit != "c"]
        for key, values in rows.items()
    }
    with pytest.raises(CausalCompileError, match="at least three donor units"):
        analyze(rows, query=SyntheticControl("y", "unit", "period", "treated", 5))


def test_synthetic_control_refuses_unbalanced_panel_and_no_pre_support():
    rows = _synthetic_truth()
    short = {key: values[:-1] for key, values in rows.items()}
    with pytest.raises(CausalCompileError, match="balanced panel"):
        analyze(short, query=SyntheticControl("y", "unit", "period", "treated", 5))
    with pytest.raises(CausalCompileError, match="at least two pre-periods"):
        analyze(rows, query=SyntheticControl("y", "unit", "period", "treated", 1))


def _unbalanced_truth(gone: set[tuple[str, int]]) -> dict[str, list[object]]:
    """The exact-weights panel plus an unrelated donor ``d``, without the listed unit-periods."""
    rows = _synthetic_truth()
    extra = {"d": [4.0, 1.0, 3.0, 2.0, 0.0, 5.0]}
    for unit, trajectory in extra.items():
        for period, value in enumerate(trajectory, start=1):
            rows["y"].append(value)
            rows["unit"].append(unit)
            rows["period"].append(period)
    keep = [
        index
        for index in range(len(rows["y"]))
        if (rows["unit"][index], rows["period"][index]) not in gone
    ]
    return {key: [values[index] for index in keep] for key, values in rows.items()}


def test_observed_window_support_recovers_the_effect_from_an_unbalanced_panel():
    # The treated unit misses period 2 and donor d misses period 3 (inside the window), so
    # d is excluded; donor a misses period 2, which the treated unit does not have.
    rows = _unbalanced_truth({("treated", 2), ("d", 3), ("a", 2)})
    query = SyntheticControl("y", "unit", "period", "treated", 5, observed_window_support=True)
    result = _analyzed(rows, query).synthetic_control
    assert result.estimate == pytest.approx(5.0, abs=1e-5)
    assert result.n_pre_periods == 3 and result.n_post_periods == 2
    weights = dict(result.donor_weights)
    assert set(weights) == {"a", "b", "c"}
    assert weights == pytest.approx({"a": 0.2, "b": 0.3, "c": 0.5}, abs=1e-4)
    window = result.observed_window
    assert window is not None
    assert window.pre_periods == (1, 3, 4) and window.post_periods == (5, 6)
    assert window.donors_used == 3 and window.donors_dropped == (("d", 1),)
    balanced = _analyzed(
        _synthetic_truth(), SyntheticControl("y", "unit", "period", "treated", 5)
    ).synthetic_control
    assert balanced.observed_window is None
    augmented = _analyzed(
        rows,
        SyntheticControl(
            "y",
            "unit",
            "period",
            "treated",
            5,
            augmentation_ridge=1e-8,
            observed_window_support=True,
        ),
    ).synthetic_control
    assert augmented.estimate == pytest.approx(5.0, abs=1e-4)
    # Without the declaration the same panel is still refused as unbalanced.
    with pytest.raises(CausalCompileError, match="balanced panel"):
        analyze(rows, query=SyntheticControl("y", "unit", "period", "treated", 5))


def _refused(rows, **options):
    query = SyntheticControl("y", "unit", "period", "treated", 5, **options)
    return require_success(analyze(rows, query=query))


def test_observed_window_support_refuses_designs_without_enough_support():
    thin = _unbalanced_truth({("treated", 1), ("treated", 2), ("treated", 3)})
    with pytest.raises(CausalError, match="(?i)at least two observed pre-intervention"):
        _refused(thin, observed_window_support=True)
    few = _unbalanced_truth({("a", 4), ("d", 6)})
    with pytest.raises(CausalError, match="(?i)only 2 donors"):
        _refused(few, observed_window_support=True)
    with pytest.raises(CausalError, match="(?i)next attempt"):
        _refused(
            _unbalanced_truth(set()),
            observed_window_support=True,
            uniform_unit_randomization=True,
        )
    with pytest.raises(CausalValueError):
        SyntheticControl("y", "unit", "period", "treated", 5, observed_window_support="yes")


def test_retained_analyze_and_prepare_match_direct_point_result():
    rows = _synthetic_truth()
    query = SyntheticControl("y", "unit", "period", "treated", 5)
    direct = _analyzed(rows, query).synthetic_control
    result = _analyzed(rows, query)
    fit = result.synthetic_control
    assert fit is not None
    assert fit.estimate == pytest.approx(direct.estimate, abs=1e-5)
    assert dict(fit.donor_weights) == pytest.approx(dict(direct.donor_weights))
    assert fit.placebo_effects == pytest.approx(direct.placebo_effects)
    assert fit.support_status == "unlicensed_point_utility"
    assert fit.uncertainty == "point_only_with_unlicensed_placebo_rank"
    assert math.isnan(result.estimate.se_analytic)
    prepared = antecedent.prepare(rows, query=query)
    executed = require_success(antecedent.estimate(prepared))
    assert executed.synthetic_control == fit
    rebound = antecedent.rebind(prepared, data=rows)
    assert require_success(antecedent.estimate(rebound)).synthetic_control == fit


def test_retained_synthetic_control_refuses_changed_design_and_bad_donors():
    rows = _synthetic_truth()
    query = SyntheticControl("y", "unit", "period", "treated", 5)
    prepared = antecedent.prepare(rows, query=query)
    changed = {key: list(value) for key, value in rows.items()}
    changed["unit"][0] = "different"
    with pytest.raises(CausalUnsupportedError, match="prepared unit and period row order"):
        antecedent.rebind(prepared, data=changed)
    short = {
        key: [value for value, unit in zip(values, rows["unit"], strict=True) if unit != "c"]
        for key, values in rows.items()
    }
    with pytest.raises(CausalCompileError, match="at least three donor units"):
        _analyzed(short, query)


def test_uniform_unit_randomization_reports_exact_sharp_null_p_value_only():
    rows = _synthetic_truth()
    query = SyntheticControl("y", "unit", "period", "treated", 5, uniform_unit_randomization=True)
    result = _analyzed(rows, query)
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
    assert _analyzed(rows, query).synthetic_control == fit
    assert (
        require_success(
            antecedent.estimate(antecedent.prepare(rows, query=query))
        ).synthetic_control
        == fit
    )


def test_nonzero_constant_effect_sharp_null_is_explicit_in_public_result():
    rows = _synthetic_truth()
    with pytest.raises(CausalValueError, match="uniform_unit_randomization"):
        SyntheticControl("y", "unit", "period", "treated", 5, sharp_null_effect=5.0)
    with pytest.raises(CausalValueError, match="finite"):
        SyntheticControl(
            "y",
            "unit",
            "period",
            "treated",
            5,
            uniform_unit_randomization=True,
            sharp_null_effect=float("nan"),
        )
    query = SyntheticControl(
        "y", "unit", "period", "treated", 5, uniform_unit_randomization=True, sharp_null_effect=5.0
    )
    fit = _analyzed(rows, query).synthetic_control
    assert fit is not None
    assert fit.randomization_null_effect == 5.0
    assert fit.randomization_p_value == 1.0
    assert dict(fit.randomization_statistics)["treated"] == pytest.approx(0.0, abs=1e-5)
    assert fit.uncertainty == "point_only_with_exact_unit_randomization_p_value_no_interval"


def test_augmented_synthetic_control_uses_retained_donor_model_and_refuses_invalid_options():
    rows: dict[str, list[object]] = {"y": [], "unit": [], "period": []}
    for unit, position in (("a", 0.0), ("b", 1.0), ("c", 2.0), ("treated", 3.0)):
        for period in range(1, 5):
            rows["unit"].append(unit)
            rows["period"].append(period)
            rows["y"].append(
                position * period if period < 4 else 4 * position + (5 if unit == "treated" else 0)
            )
    query = SyntheticControl("y", "unit", "period", "treated", 4, augmentation_ridge=1e-8)
    result = _analyzed(rows, query)
    fit = result.synthetic_control
    assert fit is not None
    assert fit.unadjusted_effect == pytest.approx(9.0, abs=1e-5)
    assert fit.outcome_model_correction == pytest.approx(4.0, abs=1e-4)
    assert fit.estimate == pytest.approx(5.0, abs=1e-4)
    assert fit.uncertainty == "point_only_augmented_no_interval"
    assert fit.support_status == "unlicensed_point_utility"
    assert "donor_ridge_outcome_model_transports" in fit.assumptions
    assert math.isnan(result.estimate.se_analytic)
    assert _analyzed(rows, query).synthetic_control == fit
    assert (
        require_success(
            antecedent.estimate(antecedent.prepare(rows, query=query))
        ).synthetic_control
        == fit
    )
    with pytest.raises(CausalValueError, match="positive"):
        SyntheticControl("y", "unit", "period", "treated", 4, augmentation_ridge=0.0)
    exact_query = SyntheticControl(
        "y", "unit", "period", "treated", 4, uniform_unit_randomization=True, augmentation_ridge=1.0
    )
    exact = _analyzed(rows, exact_query).synthetic_control
    assert exact is not None
    assert (
        exact.uncertainty
        == "point_only_augmented_with_exact_unit_randomization_p_value_no_interval"
    )
    assert len(exact.randomization_statistics) == 4
    assert dict(exact.randomization_statistics)["treated"] == pytest.approx(abs(exact.estimate))
    assert exact.randomization_p_value == pytest.approx(
        sum(statistic >= abs(exact.estimate) for _, statistic in exact.randomization_statistics) / 4
    )
    assert _analyzed(rows, exact_query).synthetic_control == exact
