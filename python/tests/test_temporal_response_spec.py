"""Temporal-response query policy is owned by Rust, not mirrored in Python."""

from __future__ import annotations

import antecedent
import pytest
from antecedent._native import temporal_response_spec as native_spec
from antecedent.errors import CausalValueError
from antecedent.query import temporal_response_spec as spec


def test_python_license_is_the_native_license():
    raw = native_spec()
    assert spec.max_horizons == raw["max_horizons"]
    assert spec.max_cells == raw["max_cells"]
    assert list(spec.allowed_policies) == list(raw["allowed_policies"])
    assert spec.default_policy == raw["default_policy"]
    assert spec.default_treatment_lag == raw["default_treatment_lag"]
    assert spec.default_policy in spec.allowed_policies


def test_query_defaults_come_from_the_license():
    assert antecedent.PulseEffect("t", "y").treatment_lag == spec.default_treatment_lag
    assert antecedent.SustainedEffect("t", "y").treatment_lag == spec.default_treatment_lag
    curve = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1])
    assert curve.treatment_lag == spec.default_treatment_lag
    assert curve.policy == spec.default_policy
    path = antecedent.InterventionResponse("y", intervention={"t": 1.0}, horizons=[1])
    assert path.treatment_lag == spec.default_treatment_lag
    assert path.policy == spec.default_policy


def test_policy_list_is_native_and_horizon_count_has_no_fixed_cap():
    # The native license states no fixed horizon count: execution memory bounds materialization.
    assert spec.max_horizons == native_spec()["max_horizons"]
    with pytest.raises(CausalValueError, match=spec.allowed_policies[0]):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], policy="dynamic")
    long = list(range(1, 513))
    curve = antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=long)
    assert curve.horizons == tuple(long)
    with pytest.raises(CausalValueError, match="strictly increasing"):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[*long, 3])


def test_joint_cell_count_has_no_fixed_cap_at_construction():
    # More cells than the former fixed ceiling are a valid declaration; whether they fit is
    # decided by the execution budget when the response is materialized.
    horizons = list(range(1, 1001))
    doses = [float(i) for i in range(1001)]
    assert spec.max_cells == native_spec()["max_cells"]
    curve = antecedent.ResponseCurve("t", "y", grid=doses, horizons=horizons)
    assert len(curve.grid) * len(curve.horizons) == 1_001_000
