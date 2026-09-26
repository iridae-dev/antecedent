from __future__ import annotations

import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.quasi import (
    AugmentedPanelDiD,
    SyntheticDifferenceInDifferences,
    estimate_augmented_panel_did,
    estimate_synthetic_did,
)


def _synthetic_did_panel():
    units = ["treated", "d0", "d1", "d2"]
    unit_effect = {"treated": 10.0, "d0": 2.0, "d1": 10.0, "d2": 18.0}
    common = {1: 1.0, 2: 3.0, 3: -2.0, 4: 5.0}
    rows = [(unit, period) for unit in units for period in (1, 2, 3, 4)]
    return {
        "unit": [unit for unit, _ in rows],
        "period": [period for _, period in rows],
        "y": [
            unit_effect[unit] + common[period] + (7.0 if unit == "treated" and period == 4 else 0.0)
            for unit, period in rows
        ],
    }


def test_synthetic_did_recovers_known_effect_with_additive_unit_and_time_effects():
    result = estimate_synthetic_did(
        _synthetic_did_panel(),
        SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 4),
    )
    assert result.estimate == pytest.approx(7.0)
    assert result.n_donors == 3
    assert result.n_pre_periods == 3
    assert result.n_post_periods == 1
    assert sum(weight for _, weight in result.donor_weights) == pytest.approx(1.0)
    assert sum(weight for _, weight in result.time_weights) == pytest.approx(1.0)
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert "no_concurrent_treated_unit_specific_shock" in result.assumptions


def test_synthetic_did_refuses_unbalanced_and_insufficient_pre_support():
    query = SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 4)
    rows = _synthetic_did_panel()
    with pytest.raises(CausalValueError, match="balanced panel"):
        estimate_synthetic_did({key: value[:-1] for key, value in rows.items()}, query)
    with pytest.raises(CausalValueError, match="two pre-periods"):
        estimate_synthetic_did(
            rows,
            SyntheticDifferenceInDifferences("y", "unit", "period", "treated", 2),
        )


def test_augmented_panel_did_recovers_known_effect_and_reports_overlap():
    subjects = [f"s{i}" for i in range(6)]
    treated = [True, True, True, False, False, False]
    baseline = np.array([10.0, 12.0, 13.0, 9.0, 15.0, 11.0])
    changes = np.array([5.0, 5.0, 5.0, 2.0, 2.0, 2.0])
    result = estimate_augmented_panel_did(
        {
            "id": subjects,
            "pre": baseline,
            "post": baseline + changes,
            "treated": treated,
            "p": np.full(6, 0.5),
            "m0": np.full(6, 2.0),
        },
        AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0", True),
    )
    assert result.estimate == pytest.approx(3.0)
    assert result.treated_subjects == 3
    assert result.control_subjects == 3
    assert result.propensity_min == pytest.approx(0.5)
    assert result.propensity_max == pytest.approx(0.5)
    assert result.effective_control_sample_size == pytest.approx(3.0)
    assert result.nuisance_predictions_cross_fitted
    assert result.uncertainty == "point_only"
    assert "strict_propensity_overlap" in result.assumptions


def test_augmented_panel_did_refuses_nonoverlap_and_duplicate_subjects():
    query = AugmentedPanelDiD("pre", "post", "id", "treated", "p", "m0")
    data = {
        "id": ["a", "b", "c", "d"],
        "pre": [0.0] * 4,
        "post": [1.0, 1.0, 0.0, 0.0],
        "treated": [True, True, False, False],
        "p": [0.5, 0.5, 0.0, 0.5],
        "m0": [0.0] * 4,
    }
    with pytest.raises(CausalValueError, match="strictly between zero and one"):
        estimate_augmented_panel_did(data, query)
    with pytest.raises(CausalValueError, match="unique subject"):
        estimate_augmented_panel_did({**data, "id": ["a", "a", "c", "d"]}, query)
