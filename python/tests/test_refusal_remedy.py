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
from antecedent.outcomes import require_success


def _three_treatment_data(n=600):
    rng = np.random.default_rng(31)
    x = rng.normal(size=n)
    a, b, c = (0.3 * x + rng.normal(size=n) for _ in range(3))
    y = 1.0 + a - 0.5 * b + 0.25 * c + x + rng.normal(scale=0.2, size=n)
    data = {"x": x, "a": a, "b": b, "c": c, "y": y}
    graph = [("x", t) for t in "abc"] + [("x", "y")] + [(t, "y") for t in "abc"]
    return data, graph


def test_jacobian_over_three_treatments_is_estimated_not_refused():
    # The plug-in gradient has no fixed treatment count: three treatments are estimated,
    # and the additive truth y = 1 + a - 0.5 b + 0.25 c + x has Jacobian (1, -0.5, 0.25).
    data, graph = _three_treatment_data()
    result = require_success(
        antecedent.analyze(
            data,
            query=antecedent.ResponseJacobian(["a", "b", "c"], ["y"], at=[0.0, 0.0, 0.0]),
            graph=graph,
        )
    )
    # A derivative-family answer is the native vector or matrix, not a point grid.
    jacobian = np.asarray(result.estimate, dtype=float).ravel()
    assert jacobian.shape == (3,)
    assert np.allclose(jacobian, [1.0, -0.5, 0.25], atol=0.05)


def test_directional_derivative_over_three_treatments_is_estimated_not_refused():
    data, graph = _three_treatment_data()
    result = require_success(
        antecedent.analyze(
            data,
            query=antecedent.DirectionalDerivative(
                ["a", "b", "c"], ["y"], at=[0.0, 0.0, 0.0], direction=[1.0, 0.0, 0.0]
            ),
            graph=graph,
        )
    )
    # The derivative along the first coordinate is the additive slope of y in a.
    derivative = float(np.asarray(result.estimate, dtype=float).ravel()[0])
    assert derivative == pytest.approx(1.0, abs=0.05)


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
        require_success(
            antecedent.analyze(
                data,
                query=antecedent.InterventionResponse(
                    "y", intervention=intervention.Soft("a", "replacement")
                ),
                graph=[("a", "y")],
            )
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


def test_conditional_effect_accepts_a_modifier_set_and_refuses_malformed_modifiers():
    # A modifier set is accepted and normalized; empty, repeated or non-string
    # modifiers are invalid arguments (typed errors, not a route refusal).
    several = antecedent.ConditionalEffect("t", "y", ["w", "v"])
    assert several.modifiers == ("w", "v")
    assert antecedent.ConditionalEffect("t", "y", ("w", "v")) == several
    for bad in ([], ["w", "w"], ["w", 3], 3):
        with pytest.raises(CausalValueError):
            antecedent.ConditionalEffect("t", "y", bad)
    # A single column name is unchanged.
    single = antecedent.ConditionalEffect("t", "y", "w")
    assert single.modifier == "w"
    assert single.modifiers == ("w",)
