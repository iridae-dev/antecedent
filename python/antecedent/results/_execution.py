"""Shared analyst-facing behavior for immutable execution results."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, fields, is_dataclass, replace
from math import isfinite
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..estimation import PreparedAnalysis

from .._api import describe_refusal
from ..errors import CausalUnsupportedError
from ._slots import ReasoningSlots, SlotView


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(frozen=True, slots=True)
class CalibrationInfo:
    """Calibration slot projected from ``contract["claim"]["calibration"]``."""

    status: str = "unavailable"
    record_id: str | None = None
    reason: str | None = None
    scope_n: int | None = None
    scope_dependence: str | None = None
    calibration_sha: str | None = None

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any] | None) -> CalibrationInfo:
        if not isinstance(contract, Mapping):
            return cls(status="unavailable", reason="not_executed")
        claim = contract.get("claim")
        slot = claim.get("calibration") or {} if isinstance(claim, Mapping) else {}
        if not slot and "calibration_status" not in contract:
            return cls(status="unavailable", reason="not_executed")
        return cls(
            status=str(slot.get("status") or contract.get("calibration_status") or "unavailable"),
            record_id=slot.get("record_id") or contract.get("calibration_record_id"),
            reason=slot.get("reason") or contract.get("calibration_reason"),
            scope_n=slot.get("scope_n") or _as_optional_int(contract.get("calibration_scope_n")),
            scope_dependence=slot.get("scope_dependence") or contract.get("calibration_scope_dependence"),
            calibration_sha=slot.get("calibration_sha") or contract.get("calibration_sha"),
        )


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
                "This result has no retained study.",
                reason_code="not_executed",
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
        contract = getattr(self, "_contract", None)
        if not isinstance(contract, Mapping):
            slots = getattr(self, "reasoning", None)
            contract = getattr(slots, "contract", None) if slots is not None else None
        if isinstance(contract, Mapping):
            return CalibrationInfo.from_contract(contract)
        return CalibrationInfo(status="unavailable", reason="not_executed")

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
            calibration=(
                CalibrationInfo.from_contract(contract)
                if isinstance((contract := getattr(self, "_contract", None) or getattr(slots, "contract", None)), Mapping)
                else CalibrationInfo(status="unavailable", reason="not_executed")
            ),
            diagnostics=tuple(getattr(self, "diagnostics", ())),
        )

    @describe_refusal
    def export(self, *, artifact_id: str = "analysis-result") -> bytes:
        """Export this exact execution, even after another estimate or refresh."""
        execution = getattr(self, "_execution", None)
        if execution is None:
            raise CausalUnsupportedError(
                "This result has no retained execution artifact.",
                reason_code="not_executed",
            )
        return execution.export_contracted_artifact(artifact_id=artifact_id)
