"""Query dataclass construction contract: identifier prefix stays positional,
everything after is keyword-only (``KW_ONLY``), and ``kind`` is a read-only
discriminator (``init=False``) on every query type.
"""

from __future__ import annotations

import dataclasses

import pytest
from antecedent.errors import CausalValueError
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
    temporal_response_spec,
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
    assert list(q.horizons) == [1]
    multi = TemporalMediationEffect("t", "m", "y", horizons=[1, 2])
    assert list(multi.horizons) == [1, 2]
    with pytest.raises(CausalValueError, match="non-empty"):
        TemporalMediationEffect("t", "m", "y", horizons=())
    with pytest.raises(CausalValueError, match="strictly increasing"):
        TemporalMediationEffect("t", "m", "y", horizons=[2, 2])


@pytest.mark.parametrize("cls, positional, kind, extra", _CASES, ids=_IDS)
def test_query_dataclasses_are_frozen(cls, positional, kind, extra):
    del kind
    instance = cls(*positional, **extra)
    with pytest.raises(dataclasses.FrozenInstanceError):
        instance.kind = "tampered"  # type: ignore[misc]


def test_derivative_coordinates_accept_aligned_mappings():
    DirectionalDerivative(
        ["a", "b"],
        ["y"],
        at={"a": 1.0, "b": 0.0},
        direction={"b": 2.0, "a": 0.0},
    )
    ResponseJacobian(["a", "b"], ["y"], at={"b": 0.5, "a": 1.5})
    with pytest.raises(CausalValueError, match="exactly match treatments"):
        ResponseJacobian(["a", "b"], ["y"], at={"a": 1.0})
    with pytest.raises(CausalValueError, match="one value per treatment"):
        ResponseJacobian(["a", "b"], ["y"], at=[1.0])


def test_query_value_guards():
    with pytest.raises(ValueError, match="pair of integer"):
        SustainedEffect("t", "y", window=(0.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="from <= until"):
        SustainedEffect("t", "y", window=(1, 0))
    SustainedEffect("t", "y", window=(-2, -1))

    with pytest.raises(CausalValueError, match="at least two"):
        ResponseCurve("t", "y", grid=[0.0])
    with pytest.raises(CausalValueError, match="finite"):
        ResponseCurve("t", "y", grid=[0.0, float("nan")])
    with pytest.raises(CausalValueError, match="non-empty variable"):
        AverageDerivative(" ", "y")
    with pytest.raises(CausalValueError, match="order must be 1 or 2"):
        PointDerivative("t", "y", at=0.0, order=3)
    with pytest.raises(CausalValueError, match="positive"):
        Elasticity("t", "y", at=0.0)
    with pytest.raises(CausalValueError, match="log_scale"):
        SemiElasticity("t", "y", at=1.0, log_scale="both")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError, match="positive"):
        SemiElasticity("t", "y", at=0.0, log_scale="treatment")
    SemiElasticity("t", "y", at=-1.0, log_scale="outcome")

    with pytest.raises(CausalValueError, match="must not be None"):
        InterventionResponse("y", intervention=None)
    with pytest.raises(CausalValueError, match="at least one"):
        DirectionalDerivative([], ["y"], at=[], direction=[])
    curve = ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1, 2])
    assert curve.is_temporal
    assert InterventionResponse("y", intervention={"t": 1.0}, horizons=[1]).is_temporal
    assert not ResponseCurve("t", "y", grid=[0.0, 1.0]).is_temporal

    spec = temporal_response_spec
    with pytest.raises(CausalValueError, match="non-empty"):
        ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=())
    with pytest.raises(CausalValueError, match="at most"):
        ResponseCurve(
            "t",
            "y",
            grid=[0.0, 1.0],
            horizons=list(range(1, spec.max_horizons + 2)),
        )
    with pytest.raises(CausalValueError, match="positive integers"):
        ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[True])
    with pytest.raises(CausalValueError, match="strictly increasing"):
        ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[2, 2])
    with pytest.raises(CausalValueError, match="policy"):
        ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], policy="unknown")
    with pytest.raises(CausalValueError, match="treatment_lag"):
        ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], treatment_lag=-1)
    with pytest.raises(CausalValueError, match="max_history_lag"):
        ResponseCurve("t", "y", grid=[0.0, 1.0], horizons=[1], max_history_lag=-1)
