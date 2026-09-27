from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError


def _data(rows: int = 800) -> dict[str, np.ndarray]:
    index = np.arange(rows)
    dose = ((index * 37) % 997) / 997.0
    group = np.where(index < rows // 2, "a", "b")
    outcome = (
        1.0 + (index >= rows // 2).astype(float)
        + 0.35 * np.sin(0.13 * index)
        + np.where(index < rows // 2, 1.5, 2.0) * dose
    )
    return {"outcome": outcome, "dose": dose, "density": np.ones(rows), "group": group}


def _query(provenance: str = "known") -> ant.ConditionalDoseResponse:
    return ant.ConditionalDoseResponse(
        outcome="outcome", dose="dose", baseline_group="group", dose_density="density",
        target_doses=(), bandwidth=0.2, density_provenance=provenance,
        min_local_support=80,
        policy_doses={"a": 0.7, "b": 0.8},
        reference_doses={"a": 0.3, "b": 0.4},
    )


def test_fixed_dose_policy_uses_retained_analyze_and_prepare():
    data = _data()
    query = _query()
    result = ant.analyze(data, query=query, refute="none")
    response = result.continuous_dose_response
    assert response is not None
    assert response.policy_value_estimated
    assert response.points == ()
    value = response.fixed_policy
    assert value is not None
    assert value.incremental_value == pytest.approx(0.7, abs=0.15)
    assert value.policy_interval_95 is not None
    assert value.reference_interval_95 is not None
    assert value.incremental_interval_95 is not None
    assert result.answer.detail == "fixed_group_kernel_smoothed_dose_policy_value"
    assert ant.prepare(data, query=query, refute="none").estimate(data).continuous_dose_response == response


def test_fixed_dose_policy_refuses_malformed_rules_and_withholds_external_density_interval():
    with pytest.raises(CausalValueError, match="both policy_doses and reference_doses"):
        ant.ConditionalDoseResponse(
            "outcome", "dose", "group", "density", (), 0.2, "known",
            policy_doses={"a": 0.7},
        )
    with pytest.raises(Exception, match="every observed baseline group"):
        ant.analyze(_data(), query=ant.ConditionalDoseResponse(
            "outcome", "dose", "group", "density", (), 0.2, "known",
            policy_doses={"a": 0.7}, reference_doses={"a": 0.3},
        ), refute="none")
    result = ant.analyze(_data(), query=_query("externally_estimated"), refute="none")
    value = result.continuous_dose_response.fixed_policy
    assert value is not None
    assert value.incremental_interval_95 is None
    assert value.policy_interval_95 is None
    assert value.reference_interval_95 is None
