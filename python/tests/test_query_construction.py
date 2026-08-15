"""Query dataclass construction contract: identifier prefix stays positional,
everything after is keyword-only (``KW_ONLY``), and ``kind`` is a read-only
discriminator (``init=False``) on every query type.
"""

from __future__ import annotations

import dataclasses

import pytest
from antecedent.query import (
    AverageDerivative,
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionalDistribution,
    InterventionResponse,
    MediationEffect,
    PathSpecificEffect,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)

# (class, positional identifier args, expected kind, extra required kwargs)
_CASES = [
    (AverageEffect, ("t", "y"), "average", {}),
    (PulseEffect, ("t", "y"), "pulse", {}),
    (SustainedEffect, ("t", "y"), "sustained", {}),
    (InterventionalDistribution, ("y",), "distribution", {}),
    (PathSpecificEffect, ("t", "y"), "path_specific", {}),
    (ConditionalEffect, ("t", "y", "m"), "conditional", {}),
    (MediationEffect, ("t", "y"), "mediation", {"mediators": ["m1"]}),
    (Counterfactual, ("t", "y"), "counterfactual", {}),
    (TemporalMediationEffect, ("t", "m", "y"), "temporal_mediation", {}),
    (ResponseCurve, ("t", "y"), "response_curve", {"grid": [0.0, 1.0]}),
    (AverageDerivative, ("t", "y"), "average_derivative", {}),
    (PointDerivative, ("t", "y"), "point_derivative", {"at": 0.0}),
    (Elasticity, ("t", "y"), "elasticity", {"at": 1.0}),
    (SemiElasticity, ("t", "y"), "semi_elasticity", {"at": 1.0}),
    (
        DirectionalDerivative,
        (["t1", "t2"], ["y"]),
        "directional_derivative",
        {"at": [0.0, 1.0], "direction": [1.0, 0.0]},
    ),
    (
        ResponseJacobian,
        (["t1", "t2"], ["y1", "y2"]),
        "response_jacobian",
        {"at": [0.0, 1.0]},
    ),
    (
        InterventionResponse,
        ("y",),
        "intervention_response",
        {"intervention": {"t": 1.0}},
    ),
]
_IDS = [c[0].__name__ for c in _CASES]


@pytest.mark.parametrize("cls, positional, kind, extra", _CASES, ids=_IDS)
def test_positional_prefix_still_works(cls, positional, kind, extra):
    """The identifier prefix (treatment/outcome/mediator/modifier/…) stays positional."""
    instance = cls(*positional, **extra)
    assert instance.kind == kind


@pytest.mark.parametrize("cls, positional, kind, extra", _CASES, ids=_IDS)
def test_extra_positional_raises_type_error(cls, positional, kind, extra):
    """Anything past the identifier prefix is keyword-only: one extra positional arg fails."""
    del kind, extra
    with pytest.raises(TypeError):
        cls(*positional, "unexpected_extra_positional")


@pytest.mark.parametrize("cls, positional, kind, extra", _CASES, ids=_IDS)
def test_kind_not_accepted_as_init_kwarg(cls, positional, kind, extra):
    """``kind`` is a discriminator (``init=False``); a caller can never set it."""
    with pytest.raises(TypeError):
        cls(*positional, kind=kind, **extra)


@pytest.mark.parametrize("cls, positional, kind, extra", _CASES, ids=_IDS)
def test_kind_reads_back_as_expected_string(cls, positional, kind, extra):
    instance = cls(*positional, **extra)
    assert instance.kind == kind


def test_average_effect_keyword_only_fields_still_settable():
    q = AverageEffect("t", "y", control_level=0.5, active_level=2.0, target_population="all")
    assert q.control_level == 0.5
    assert q.active_level == 2.0
    assert q.target_population == "all"


def test_mediation_effect_mediators_is_keyword_only_and_required():
    q = MediationEffect("t", "y", mediators=["m1", "m2"], contrast="direct")
    assert list(q.mediators) == ["m1", "m2"]
    assert q.contrast == "direct"
    with pytest.raises(TypeError):
        MediationEffect("t", "y")  # mediators is required, no default


def test_temporal_mediation_effect_three_identifier_prefix():
    q = TemporalMediationEffect("t", "m", "y", contrast="direct", control_level=0.1)
    assert (q.treatment, q.mediator, q.outcome) == ("t", "m", "y")
    assert q.contrast == "direct"
    assert q.control_level == 0.1


@pytest.mark.parametrize("cls, positional, kind, extra", _CASES, ids=_IDS)
def test_query_dataclasses_are_frozen(cls, positional, kind, extra):
    del kind
    instance = cls(*positional, **extra)
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.kind = "tampered"  # type: ignore[misc]
