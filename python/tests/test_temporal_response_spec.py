"""Temporal-response query policy is owned by Rust, not mirrored in Python."""

from __future__ import annotations

import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent._native import temporal_response_spec as native_spec
from antecedent.errors import CausalValueError
from antecedent.query import temporal_response_spec as spec


def test_python_license_is_the_native_license():
    raw = native_spec()
    assert spec.max_horizons == raw["max_horizons"]
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


def test_horizon_cap_and_policy_list_are_not_hardcoded():
    too_many = list(range(1, spec.max_horizons + 2))
    with pytest.raises(CausalValueError, match=str(spec.max_horizons)):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=too_many)
    with pytest.raises(CausalValueError, match=spec.allowed_policies[0]):
        antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], policy="dynamic")
    ok = list(range(1, spec.max_horizons + 1))
    antecedent.ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=ok)
