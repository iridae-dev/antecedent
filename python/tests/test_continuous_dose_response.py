from __future__ import annotations

import numpy as np
import pytest
from antecedent import policy
from antecedent.errors import CausalValueError


def _dose_data():
    doses = [-1.0, -0.5, 0.0, 0.5, 1.0]
    groups = [group for group in ("control", "treated") for _ in doses]
    dose = doses * 2
    outcome = [
        (1.0 if group == "control" else 10.0) + 2.0 * value
        for group, value in zip(groups, dose, strict=True)
    ]
    return {
        "y": np.asarray(outcome),
        "dose": np.asarray(dose),
        "group": groups,
        "density": np.full(len(dose), 0.5),
    }


def test_stratified_continuous_dose_response_recovers_known_linear_truth():
    result = policy.estimate_continuous_dose_response(
        _dose_data(),
        outcome="y",
        dose="dose",
        baseline_group="group",
        dose_density="density",
        target_doses=[0.0, 0.5],
        bandwidth=0.6,
        density_provenance="known",
    )
    expected = {
        ("control", 0.0): 1.0,
        ("control", 0.5): 2.0,
        ("treated", 0.0): 10.0,
        ("treated", 0.5): 11.0,
    }
    assert {(point.baseline_group, point.target_dose): point.response for point in result.points} == pytest.approx(expected)
    assert all(point.local_rows == 3 for point in result.points)
    assert all(point.effective_sample_size == pytest.approx(1682 / 769) for point in result.points)
    assert all(point.minimum_dose_density == pytest.approx(0.5) for point in result.points)
    assert result.policy_value_estimated is False
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert "no interference" in " ".join(result.assumptions).lower()


def test_continuous_dose_response_refuses_zero_density_and_unsupported_targets():
    data = _dose_data()
    with pytest.raises(CausalValueError, match="densit.*positive"):
        policy.estimate_continuous_dose_response(
            {**data, "density": [0.0] * 10},
            outcome="y", dose="dose", baseline_group="group", dose_density="density",
            target_doses=[0.0], bandwidth=0.6, density_provenance="known",
        )
    with pytest.raises(CausalValueError, match="support failure"):
        policy.estimate_continuous_dose_response(
            data,
            outcome="y", dose="dose", baseline_group="group", dose_density="density",
            target_doses=[4.0], bandwidth=0.1, density_provenance="known",
        )


def test_continuous_dose_response_refuses_bad_density_source_and_bandwidth():
    with pytest.raises(CausalValueError, match="density_provenance"):
        policy.estimate_continuous_dose_response(
            _dose_data(),
            outcome="y", dose="dose", baseline_group="group", dose_density="density",
            target_doses=[0.0], bandwidth=0.6, density_provenance="guessed",
        )
    with pytest.raises(CausalValueError, match="bandwidth"):
        policy.estimate_continuous_dose_response(
            _dose_data(),
            outcome="y", dose="dose", baseline_group="group", dose_density="density",
            target_doses=[0.0], bandwidth=0.0, density_provenance="known",
        )
