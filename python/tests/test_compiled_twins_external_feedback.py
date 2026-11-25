"""The compiled external feedback fit reproduces the deprecated direct fit.

``ExternalFeedbackModel.fit`` warns with ``AntecedentDeprecationWarning`` and still returns the
fit ``fit_external_feedback`` computes; ``ExternalFeedbackModel.compiled_fit`` binds the same
request into the ``ExternalFeedbackFit`` program family and returns the identical fit. The
reference mechanism is ``gain * tanh(x1)`` with ``x1`` hard-clamped in every row, so the
maximum-likelihood gain is the least-squares slope of ``x0`` on ``tanh(x1)``, computed here.
"""

from __future__ import annotations

import math
import warnings

import pytest
from antecedent import external_mechanisms as em
from antecedent._deprecation import AntecedentDeprecationWarning
from antecedent.cyclic import Equation
from antecedent.errors import CausalValueError

CONTENT = "gain * tanh(x1) :: v1 :: reference mechanism"
CLAMPED = (-2.0, -1.0, 0.5, 1.0, 2.0)
TARGETS = [{"kind": "parameter", "equation": 0, "parameter": 0}]


def make_registry() -> em.MechanismRegistry:
    registry = em.MechanismRegistry()
    em.register_mechanism(
        "lab/saturating_gain",
        "1.0",
        lambda x, p: p[0] * math.tanh(x[0]),
        reads=[1],
        parameters=["gain"],
        content=CONTENT,
        input_gradient=lambda x, p: [p[0] * (1.0 - math.tanh(x[0]) ** 2)],
        parameter_gradient=lambda x, p: [math.tanh(x[0])],
        input_lipschitz=[0.9],
        registry=registry,
    )
    return registry


def model(registry: em.MechanismRegistry) -> em.ExternalFeedbackModel:
    implementation = registry.resolve(
        {"implementation_id": "lab/saturating_gain", "version": "1.0"}
    )
    return em.ExternalFeedbackModel(
        equations=(
            em.ExternalEquation("response", implementation, parameters=(0.0,)),
            Equation("feedback", 0.0, (0.2, 0.0), "tanh"),
        ),
        noise_covariance=((1.0, 0.0), (0.0, 1.0)),
    )


def regime_rows() -> list[dict]:
    return [
        {"values": [0.7 * math.tanh(x) + noise, x], "clamps": [{"variable": 1, "value": x}]}
        for x in CLAMPED
        for noise in (-0.1, 0.1)
    ]


def least_squares_gain() -> float:
    rows = regime_rows()
    t = [math.tanh(r["values"][1]) for r in rows]
    y = [r["values"][0] for r in rows]
    return sum(a * b for a, b in zip(t, y, strict=True)) / sum(a * a for a in t)


def test_the_direct_fit_warns_and_equals_the_compiled_fit():
    registry = make_registry()
    with pytest.warns(AntecedentDeprecationWarning, match="ExternalFeedbackModel.fit"):
        direct = model(registry).fit(TARGETS, regime_rows(), resolver=registry)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntecedentDeprecationWarning)
        compiled = model(registry).compiled_fit(TARGETS, regime_rows(), resolver=registry)
    assert compiled["fit"] == direct
    gain = compiled["fit"]["system"]["equations"][0]["parameters"][0]
    assert gain == direct["system"]["equations"][0]["parameters"][0]
    assert compiled["route"] == "ExternalFeedbackFit"
    assert compiled["claim"] == "model_conditional_point"
    assert len(compiled["lowered_program_identity"]) == 64


def test_the_compiled_fit_recovers_the_least_squares_gain():
    registry = make_registry()
    compiled = model(registry).compiled_fit(TARGETS, regime_rows(), resolver=registry)
    fit = compiled["fit"]
    assert fit["converged"]
    assert fit["trust"] == "verified_extension"
    assert fit["system"]["equations"][0]["parameters"][0] == pytest.approx(
        least_squares_gain(), abs=1e-6
    )


def test_the_compiled_fit_without_a_resolver_is_refused_like_the_direct_fit():
    registry = make_registry()
    with pytest.raises(CausalValueError) as compiled:
        model(registry).compiled_fit(TARGETS, regime_rows(), resolver=None)
    assert compiled.value.reason_code == "mechanism_resolver_missing"
    with (
        pytest.warns(AntecedentDeprecationWarning),
        pytest.raises(CausalValueError) as direct,
    ):
        model(registry).fit(TARGETS, regime_rows(), resolver=None)
    assert direct.value.reason_code == compiled.value.reason_code
