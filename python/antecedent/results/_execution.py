"""Shared analyst-facing behavior for immutable execution results."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, fields, is_dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..estimation import PreparedAnalysis

from .._api import describe_refusal
from ..errors import CausalUnsupportedError
from ._slots import ReasoningSlots, SlotView


@dataclass(frozen=True, slots=True)
class CalibrationInfo:
    """Whether a coverage artifact is bound to this execution.

    1.10 does not bind the 1.9 weekly coverage gate to individual executions.
    ``unavailable`` means no artifact is attached, not that the interval was
    measured and failed. A supplied certificate can be retained as
    ``scope_not_assessed``; neither a licensed matrix cell nor a passing
    refuter is promoted into a calibration claim.
    """

    status: str = "unavailable"
    reason: str = "No calibration evidence has been bound to this execution."
    evidence: tuple[Any, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Answer:
    """Safe consumption shape: ``point``, ``bounds``, ``partial``, or ``unavailable``.

    This is the interface that withholds an unrestricted scalar when
    identification is partial or leftover mass remains. Historical fields
    such as ``effect``, ``ate``, ``posterior``, and ``response`` stay on the
    result for existing callers. ``effect`` / ``ate`` warn when a point
    display would misrepresent; ``ANTECEDENT_STRICT_ANSWER=1`` raises.
    """

    kind: str
    value: float | None = None
    bounds: tuple[float, float] | None = None
    detail: str | None = None


def _payload(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: getattr(value, f.name)
            for f in fields(value)
            if f.name not in {"artifact", "score_table", "score_inference", "envelope"}
        }
    return {"value": value}


class ResultAPI:
    """Common API; implementing dataclasses retain their historical public fields."""

    @property
    @describe_refusal
    def study(self) -> PreparedAnalysis:
        """Reusable study, when this execution retained one; never re-prepares lazily."""
        prepared = getattr(self, "_prepared", None)
        if prepared is None:
            raise CausalUnsupportedError(
                "This execution route did not retain a reusable study. "
                "Ordinary prepared tabular/temporal scalar, class, posterior-mixture, "
                "and response routes retain one; callbacks, custom estimator settings, "
                "RD, panel/event/multi-environment, and some discovery paths do not. "
                "Accessing study never reruns discovery."
            )
        return prepared

    def _scalar_effect(self) -> float | None:
        """Historical scalar without the legacy-field warning."""
        getter = getattr(self, "estimate", None)
        if getter is not None and hasattr(getter, "ate"):
            value = getter.ate
            return value if isinstance(value, (int, float)) else None
        if isinstance(getter, (int, float)):
            return float(getter)
        return None

    def _warn_legacy_scalar(self, name: str) -> None:
        limitation = getattr(self, "rendering_limitation", lambda: None)()
        if limitation is None:
            return
        message = (
            f"result.{name} is a historical field; result.answer is the safe "
            f"interface (this execution is limited: {limitation}). "
            "Set ANTECEDENT_STRICT_ANSWER=1 to raise."
        )
        if os.environ.get("ANTECEDENT_STRICT_ANSWER") == "1":
            raise CausalUnsupportedError(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    @property
    def answer(self) -> Answer:
        limitation = getattr(self, "rendering_limitation", lambda: None)()
        bounds = getattr(self, "structural_identified_set", None)
        if bounds is not None:
            return Answer("bounds", bounds=tuple(bounds), detail=limitation)
        if limitation is not None:
            return Answer("partial", detail=limitation)
        if (
            getattr(self, "response", None) is not None
            or getattr(self, "mediation_grid", None) is not None
        ):
            return Answer("response")
        value = self._scalar_effect()
        if value is None:
            return Answer("structured")
        if not isfinite(value):
            return Answer("unavailable", detail="non_finite_effect")
        return Answer("point", value=float(value))

    @property
    def calibration(self) -> CalibrationInfo:
        # Keep supplied evidence intact. Absence of execution-bound evidence
        # must not be upgraded from a passing refuter or a licensed matrix cell.
        certificate = getattr(self, "certificate", None) or {}
        evidence = certificate.get("calibration")
        return CalibrationInfo(
            status="scope_not_assessed" if evidence is not None else "unavailable",
            reason="Evidence retained; applicability to this execution is not established."
            if evidence is not None
            else "No calibration evidence has been bound to this execution.",
            evidence=() if evidence is None else (evidence,),
            limitations=tuple(getattr(self, "diagnostics", ())),
        )

    def inspect(self) -> ReasoningSlots:
        """Inspect this execution, including uncertainty and all available evidence."""
        slots = getattr(self, "reasoning", None)
        if slots is None:
            slots = ReasoningSlots.from_contract({})
        identification = getattr(self, "identification", None)
        ident_payload = {} if identification is None else _payload(identification)
        ident_payload.update(slots.identification.payload)
        for field in (
            "structural_unidentified_mass",
            "structural_unevaluable_mass",
            "structural_identified_mass",
        ):
            value = getattr(self, field, None)
            if value is not None:
                ident_payload[field.removeprefix("structural_")] = value
        status = (
            slots.identification.summary
            if slots.identification.available
            else getattr(identification, "status", "unavailable")
        )
        uncertainty = getattr(self, "uncertainty", None)
        estimate = getattr(self, "estimate", None)
        posterior = getattr(self, "posterior", None)
        uncertainty_payload = dict(slots.uncertainty.payload)
        if uncertainty is not None:
            uncertainty_payload["execution"] = _payload(uncertainty)
        else:
            if estimate is not None:
                uncertainty_payload["estimate"] = _payload(estimate)
            if posterior is not None:
                uncertainty_payload["posterior"] = _payload(posterior)
            for name in (
                "structural_identified_set_interval",
                "structural_identified_set_interval_level",
                "structural_identified_set_interval_method",
                "structural_identified_set_interval_truncated",
            ):
                value = getattr(self, name, None)
                if value is not None:
                    uncertainty_payload[name.removeprefix("structural_")] = value
        uncertainty_available = (
            slots.uncertainty.available
            or (uncertainty is not None and getattr(uncertainty, "kind", "none") != "none")
            or (posterior is not None and getattr(posterior, "effect_sd", None) is not None)
        )
        if estimate is not None:
            uncertainty_available |= any(
                isinstance(v := getattr(estimate, name, None), (int, float)) and isfinite(v)
                for name in ("se_analytic", "se_bootstrap")
            )
        assumptions = getattr(self, "assumptions", None)
        assumptions_available = slots.assumptions.available or assumptions is not None
        assumption_payload = dict(slots.assumptions.payload)
        if assumptions is not None:
            assumption_payload["assumptions"] = assumptions
        certificate = getattr(self, "certificate", None)
        if certificate is not None:
            assumption_payload["certificate"] = certificate
        support_payload = dict(slots.support.payload)
        support_payload["execution"] = _payload(getattr(self, "support", None))
        support_payload["validation"] = _payload(getattr(self, "validation", None))
        return replace(
            slots,
            identification=SlotView(
                identification is not None or slots.identification.available,
                None if identification is not None or slots.identification.available else "missing",
                status,
                ident_payload,
            ),
            support=replace(slots.support, payload=support_payload),
            uncertainty=SlotView(
                uncertainty_available,
                None if uncertainty_available else "not_evaluated",
                slots.uncertainty.summary
                if slots.uncertainty.available
                else "execution_specific"
                if uncertainty_available
                else "unavailable",
                uncertainty_payload,
            ),
            assumptions=SlotView(
                assumptions_available,
                None if assumptions_available else "not_retained",
                slots.assumptions.summary
                if slots.assumptions.available
                else "retained"
                if assumptions_available
                else "unavailable",
                assumption_payload,
            ),
            answer=self.answer,
            calibration=self.calibration,
            diagnostics=tuple(getattr(self, "diagnostics", ())),
        )

    @describe_refusal
    def export(self, *, artifact_id: str = "analysis-result") -> bytes:
        """Export this exact execution, even after another estimate or refresh."""
        execution = getattr(self, "_execution", None)
        if execution is None:
            raise CausalUnsupportedError(
                "This result has no retained execution artifact; export cannot reconstruct a contract."
            )
        return execution.export_contracted_artifact(artifact_id=artifact_id)
