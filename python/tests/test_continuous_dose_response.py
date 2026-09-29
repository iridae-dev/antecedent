from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalCompileError, CausalValueError
from antecedent.policy import ConditionalDoseResponse


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
    result = antecedent.analyze(
        _dose_data(),
        query=ConditionalDoseResponse(
            outcome="y",
            dose="dose",
            baseline_group="group",
            dose_density="density",
            target_doses=(0.0, 0.5),
            bandwidth=0.6,
            density_provenance="known",
        ),
    ).continuous_dose_response
    expected = {
        ("control", 0.0): 1.0,
        ("control", 0.5): 2.0,
        ("treated", 0.0): 10.0,
        ("treated", 0.5): 11.0,
    }
    assert {
        (point.baseline_group, point.target_dose): point.response for point in result.points
    } == pytest.approx(expected)
    assert all(point.local_rows == 3 for point in result.points)
    assert all(point.effective_sample_size == pytest.approx(1682 / 769) for point in result.points)
    assert all(point.minimum_dose_density == pytest.approx(0.5) for point in result.points)
    assert result.policy_value_estimated is False
    assert result.uncertainty == "point_only_no_interval"
    assert result.support_status == "unlicensed_point_utility"
    assert "no interference" in " ".join(result.assumptions).lower()


def test_continuous_dose_response_refuses_zero_density_and_unsupported_targets():
    data = _dose_data()
    with pytest.raises(CausalCompileError, match="densit.*positive"):
        antecedent.analyze(
            {**data, "density": [0.0] * 10},
            query=ConditionalDoseResponse(
                outcome="y",
                dose="dose",
                baseline_group="group",
                dose_density="density",
                target_doses=(0.0,),
                bandwidth=0.6,
                density_provenance="known",
            ),
        )
    with pytest.raises(CausalCompileError, match="support failure"):
        antecedent.analyze(
            data,
            query=ConditionalDoseResponse(
                outcome="y",
                dose="dose",
                baseline_group="group",
                dose_density="density",
                target_doses=(4.0,),
                bandwidth=0.1,
                density_provenance="known",
            ),
        )


def test_continuous_dose_response_refuses_bad_density_source_and_bandwidth():
    with pytest.raises(CausalValueError, match="density_provenance"):
        ConditionalDoseResponse(
            outcome="y",
            dose="dose",
            baseline_group="group",
            dose_density="density",
            target_doses=(0.0,),
            bandwidth=0.6,
            density_provenance="guessed",
        )
    with pytest.raises(CausalValueError, match="bandwidth"):
        ConditionalDoseResponse(
            outcome="y",
            dose="dose",
            baseline_group="group",
            dose_density="density",
            target_doses=(0.0,),
            bandwidth=0.0,
            density_provenance="known",
        )


def test_retained_continuous_dose_matches_direct_kernel_and_artifact():
    data = _dose_data()
    query = antecedent.policy.ConditionalDoseResponse(
        outcome="y",
        dose="dose",
        baseline_group="group",
        dose_density="density",
        target_doses=(0.0, 0.5),
        bandwidth=0.6,
        density_provenance="known",
    )
    direct = antecedent.analyze(data, query=query).continuous_dose_response
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    grid = result.continuous_dose_response
    assert grid is not None
    assert grid.points == direct.points
    assert grid.uncertainty == "point_only_no_interval"
    assert result.estimate.ate is None
    assert prepared.refresh(data).continuous_dose_response == grid
    assert antecedent.analyze(data, query=query).continuous_dose_response == grid
    loaded = antecedent.load(prepared.export(artifact_id="continuous-dose-study"))
    assert loaded.answer == result.answer
    assert loaded.answer.structured["graphless_support_status"] == "unlicensed_point_utility"
    assert loaded.answer.kind == "structured"
    assert loaded.answer.structured["points"][3]["response"] == pytest.approx(11.0)


def test_retained_continuous_dose_refuses_support_and_row_rebinding():
    data = _dose_data()
    query = antecedent.policy.ConditionalDoseResponse(
        "y",
        "dose",
        "group",
        "density",
        (4.0,),
        0.1,
        "known",
    )
    with pytest.raises(Exception, match="support failure"):
        antecedent.analyze(data, query=query)
    query = antecedent.policy.ConditionalDoseResponse(
        "y",
        "dose",
        "group",
        "density",
        (0.0,),
        0.6,
        "known",
    )
    prepared = antecedent.prepare(data, query=query)
    with pytest.raises(Exception, match="baseline-group row order"):
        prepared.refresh({**data, "group": list(reversed(data["group"]))})
