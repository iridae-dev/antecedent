"""Soft overlays act on propagated means, not clipped stochastic draws."""

import json
from pathlib import Path

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.intervention import Soft


@pytest.mark.parametrize(
    "family,params", [("multiplicative", [2.0]), ("truncated_shift", [3.0, 0.0, 2.0])]
)
def test_extra_soft_families_consume_native_mean_dynamics(family, params):
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "conformance/response/temporal_soft_mean_mechanisms/expected.json"
        ).read_text()
    )
    n = fixture["generation"]["n"]
    t = 1.0 + np.sin(np.arange(n) * 1.719)
    y = 1.0 + 2.0 * np.roll(t, 1) + 3.0 * np.roll(t, 2)
    result = antecedent.analyze(
        {"t": t, "y": y},
        graph=[("t", 1, "y", 0), ("t", 2, "y", 0)],
        query=antecedent.InterventionResponse(
            "y",
            intervention=Soft("t", family, parameters=params),
            horizons=[1],
            policy="pulse",
            treatment_lag=1,
        ),
        refute=False,
        bootstrap=0,
        seed=17,
    )
    assert result.response.values[0][0] == pytest.approx(
        fixture["python_reduced_graph"]["mean"], abs=fixture["python_reduced_graph"]["atol"]
    )


@pytest.mark.parametrize(
    "family,params",
    [
        ("multiplicative", []),
        ("multiplicative", [1.0, 2.0]),
        ("multiplicative", [float("nan")]),
        ("truncated_shift", [1.0]),
        ("truncated_shift", [1.0, 3.0, 2.0]),
        ("truncated_shift", [1.0, 0.0, float("inf")]),
    ],
)
def test_extra_soft_malformed_parameters_fail_at_construction(family, params):
    with pytest.raises(CausalValueError):
        Soft("t", family, parameters=params)
