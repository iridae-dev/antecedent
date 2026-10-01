"""Refusals name a remedy in a structured, optional ``remedy`` attribute.

The remedy is additive: it never changes the message or the reason code, and
every exception that names none reads ``remedy is None``.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import intervention
from antecedent.errors import (
    CausalError,
    CausalEstimateError,
    CausalUnsupportedError,
    CausalValueError,
)


def _three_treatment_data():
    rng = np.random.default_rng(31)
    x = rng.normal(size=200)
    a, b, c = (0.3 * x + rng.normal(size=200) for _ in range(3))
    y = 1.0 + a - 0.5 * b + 0.25 * c + x + rng.normal(scale=0.2, size=200)
    data = {"x": x, "a": a, "b": b, "c": c, "y": y}
    graph = [("x", t) for t in "abc"] + [("x", "y")] + [(t, "y") for t in "abc"]
    return data, graph


def test_jacobian_over_two_treatments_names_a_remedy():
    data, graph = _three_treatment_data()
    with pytest.raises(CausalEstimateError) as raised:
        antecedent.analyze(
            data,
            query=antecedent.ResponseJacobian(["a", "b", "c"], ["y"], at=[0.0, 0.0, 0.0]),
            graph=graph,
        )
    err = raised.value
    # The message and (absent) reason code are unchanged; the remedy is separate.
    assert str(err) == "plug-in response Jacobian supports at most two treatments"
    assert getattr(err, "reason_code", None) is None
    assert err.remedy is not None
    assert "at most two treatments" in err.remedy
    assert "AverageDerivative" in err.remedy
    assert err.remedy not in str(err)


def test_directional_derivative_over_two_treatments_names_a_remedy():
    data, graph = _three_treatment_data()
    with pytest.raises(CausalEstimateError) as raised:
        antecedent.analyze(
            data,
            query=antecedent.DirectionalDerivative(
                ["a", "b", "c"], ["y"], at=[0.0, 0.0, 0.0], direction=[1.0, 0.0, 0.0]
            ),
            graph=graph,
        )
    err = raised.value
    assert str(err) == "plug-in directional derivative supports at most two treatments"
    assert err.remedy is not None
    assert "ResponseJacobian" in err.remedy


def test_remedy_defaults_to_none_on_every_error():
    # Class default on the native root: every subclass reads it.
    assert CausalError.remedy is None
    assert CausalEstimateError("x").remedy is None
    refusal = CausalUnsupportedError("refused", reason_code="route_not_supported")
    assert refusal.remedy is None
    assert refusal.reason_code == "route_not_supported"
    assert CausalValueError("bad").remedy is None


def test_a_native_refusal_without_a_remedy_reads_none():
    data = {"a": np.arange(40, dtype=float), "y": np.arange(40, dtype=float)}
    with pytest.raises(CausalUnsupportedError, match="temporal response cell") as raised:
        antecedent.analyze(
            data,
            query=antecedent.InterventionResponse(
                "y", intervention=intervention.Soft("a", "replacement")
            ),
            graph=[("a", "y")],
        )
    assert raised.value.remedy is None


def test_python_refusal_accepts_a_remedy_without_changing_its_message():
    plain = CausalUnsupportedError("refused", reason_code="route_not_supported")
    remedied = CausalUnsupportedError(
        "refused", reason_code="route_not_supported", remedy="do this instead"
    )
    assert remedied.remedy == "do this instead"
    assert str(remedied) == str(plain)
    assert remedied.reason_code == plain.reason_code
