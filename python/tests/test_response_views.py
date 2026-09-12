"""Constructor contracts for response result views (no native calls)."""

from __future__ import annotations

import pytest
from antecedent.errors import CausalValueError
from antecedent.results import IdentificationView
from antecedent.results.response import (
    CausalResponseView,
    ResponseEnvelopeView,
    ResponseUncertainty,
    ResponseValidationCheck,
    ResponseValidationView,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
)


def _ident() -> IdentificationView:
    return IdentificationView(
        status="NonparametricallyIdentified",
        method="response.backdoor",
        adjustment_set=["z"],
        assumption_count=1,
        derivation_step_count=1,
    )


def test_response_view_rejects_empty_and_nonfinite_geometry():
    with pytest.raises(CausalValueError, match="must not be empty"):
        ResponseView([], ["y"], [[0.0]], [[1.0]])
    with pytest.raises(CausalValueError, match="same number of rows"):
        ResponseView(["a"], ["y"], [[0.0]], [[1.0], [2.0]])
    with pytest.raises(CausalValueError, match="at least one value per treatment"):
        ResponseView(["a"], ["y"], [[]], [[1.0]])
    with pytest.raises(CausalValueError, match="one value per treatment"):
        ResponseView(["a", "b"], ["y"], [[0.0]], [[1.0]])
    with pytest.raises(CausalValueError, match="points must be finite"):
        ResponseView(["a"], ["y"], [[float("nan")]], [[1.0]])
    with pytest.raises(CausalValueError, match="one value per outcome"):
        ResponseView(["a"], ["y", "v"], [[0.0]], [[1.0]])
    with pytest.raises(CausalValueError, match="values must be finite"):
        ResponseView(["a"], ["y"], [[0.0]], [[float("inf")]])
    view = ResponseView(["a"], ["y"], [[0.0], [1.0]], [[2.0], [3.0]])
    assert "2 points" in repr(view)


def test_envelope_and_support_and_uncertainty_guards():
    with pytest.raises(CausalValueError, match="same number of rows"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0]], [], 1.0, 0.0, 1)
    with pytest.raises(CausalValueError, match="identified_mass"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0]], [[1.0]], 1.5, -0.5, 1)
    with pytest.raises(CausalValueError, match="unidentified_mass"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0]], [[1.0]], 0.4, 1.2, 1)
    with pytest.raises(CausalValueError, match="sum to one"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0]], [[1.0]], 0.4, 0.4, 1)
    with pytest.raises(CausalValueError, match="completion"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0]], [[1.0]], 1.0, 0.0, 0)
    with pytest.raises(CausalValueError, match="examined completions"):
        ResponseEnvelopeView(
            ["a"],
            ["y"],
            [[0.0]],
            [[0.0]],
            [[1.0]],
            1.0,
            0.0,
            1,
            enumeration_capped=True,
        )
    with pytest.raises(CausalValueError, match="one value per outcome"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0, 1.0]], [[1.0]], 1.0, 0.0, 1)
    with pytest.raises(CausalValueError, match="must not exceed"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[2.0]], [[1.0]], 1.0, 0.0, 1)
    envelope = ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0]], [[1.0]], 0.7, 0.3, 2)
    assert len(envelope) == 1

    with pytest.raises(CausalValueError, match="unknown support status"):
        SupportReport("maybe", {"a": (0.0, 1.0)})
    with pytest.raises(CausalValueError, match="unknown support status"):
        SupportReport("supported", {"a": (0.0, 1.0)}, point_status=["maybe"])
    with pytest.raises(CausalValueError, match="invalid support query region"):
        SupportReport("supported", {"": (0.0, 1.0)})
    with pytest.raises(CausalValueError, match="invalid support query region"):
        SupportReport("supported", {"a": (1.0, 0.0)})
    supported = SupportReport(
        "supported",
        {"a": (0.0, 1.0)},
        point_status=["supported", "extrapolative"],
    )
    assert supported
    assert "cells=2" in repr(supported)
    assert not SupportReport("extrapolative", {"a": (0.0, 1.0)})
    with pytest.raises(CausalValueError, match="non-empty string"):
        SupportDiagnostic("  ", [1.0], "x")
    with pytest.raises(CausalValueError, match="must be finite"):
        SupportDiagnostic("ess", [float("nan")], "x")

    with pytest.raises(CausalValueError, match="unknown uncertainty kind"):
        ResponseUncertainty("bootstrap")
    with pytest.raises(CausalValueError, match="both be provided"):
        ResponseUncertainty("pointwise", lower=[[0.0]])
    with pytest.raises(CausalValueError, match="same number of rows"):
        ResponseUncertainty("pointwise", lower=[[0.0]], upper=[[0.0], [1.0]])
    with pytest.raises(CausalValueError, match="strictly between"):
        ResponseUncertainty("pointwise", lower=[[0.0]], upper=[[1.0]], level=1.0)
    with pytest.raises(CausalValueError, match="cannot carry"):
        ResponseUncertainty("none", standard_error=0.1)
    with pytest.raises(CausalValueError, match="non-negative"):
        ResponseUncertainty("pointwise", lower=[[0.0]], upper=[[1.0]], standard_error=-1.0)
    with pytest.raises(CausalValueError, match="at least 1"):
        ResponseUncertainty("pointwise", lower=[[0.0]], upper=[[1.0]], replicates=0)
    with pytest.raises(CausalValueError, match="invalid PAG completion"):
        ResponseEnvelopeView(["a"], ["y"], [[0.0]], [[0.0]], [[1.0]], 1.0, 0.0, 1, 2)
    band = ResponseUncertainty("pointwise", lower=[[0.0]], upper=[[1.0]], level=0.9)
    assert "level=90.0%" in repr(band)


def test_validation_and_causal_response_repr_branches():
    failed = ResponseValidationCheck("placebo", "failed", 0.01, 0.05, "too close")
    skipped = ResponseValidationCheck("rcc", "skipped", None, None, "n/a")
    view = ResponseValidationView(checks=(failed, skipped))
    assert view.passed is False
    assert view.skipped == [skipped]
    assert ResponseValidationView().passed is True

    response = ResponseView(["a"], ["y"], [[0.0]], [[2.0]])
    support = SupportReport("supported", {"a": (0.0, 1.0)}, warnings=("thin overlap",))
    none = ResponseUncertainty("none")
    scalar = CausalResponseView(
        estimand="curve",
        response=response,
        estimate=2.0,
        uncertainty=none,
        support=support,
        identification=_ident(),
        evidence_status="allowed_unlicensed",
    )
    text = repr(scalar)
    assert "estimate=2.000" in text
    assert "unlicensed" in text
    assert "warnings=1" in text

    structured = CausalResponseView(
        estimand="curve",
        response=response,
        estimate=None,
        uncertainty=none,
        support=support,
        identification=_ident(),
    )
    assert "1 response points" in repr(structured)

    no_curve = CausalResponseView(
        estimand="curve",
        response=None,
        estimate=[1.0, 2.0],
        uncertainty=none,
        support=support,
        identification=_ident(),
    )
    assert "structured" in repr(no_curve)


@pytest.mark.parametrize("standard_error", [float("nan"), float("inf")])
def test_uncertainty_rejects_nonfinite_standard_error(standard_error: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ResponseUncertainty("pointwise", standard_error=standard_error)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [([[0.0, 1.0]], [[2.0]]), ([[2.0]], [[1.0]]), ([[float("nan")]], [[1.0]])],
)
def test_uncertainty_rejects_malformed_bounds(
    lower: list[list[float]], upper: list[list[float]]
) -> None:
    with pytest.raises(ValueError):
        ResponseUncertainty("pointwise", lower=lower, upper=upper)


def test_identified_set_keeps_unbounded_intervals() -> None:
    uncertainty = ResponseUncertainty(
        "identified_set", lower=[[-float("inf")]], upper=[[float("inf")]]
    )
    assert uncertainty.lower == [[-float("inf")]]
