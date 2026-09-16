"""Shared analyst-facing behavior for immutable execution results."""

from __future__ import annotations

import os
import struct
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from math import isfinite
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args

if TYPE_CHECKING:
    from ..estimation import PreparedAnalysis

from .._api import describe_refusal
from ..errors import CausalUnsupportedError
from ._slots import ReasoningSlots, SlotView


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


#: Calibration fields with a typed attribute. Any other field the claim's
#: calibration slot carries is kept, in order, on :attr:`CalibrationInfo.extra`.
_CALIBRATION_FIELDS = (
    "status",
    "record_id",
    "reason",
    "scope_n",
    "scope_dependence",
    "calibration_sha",
    "level",
    "observed_coverage",
    "replicates",
)


@dataclass(frozen=True, slots=True)
class CalibrationInfo:
    """Calibration slot projected from ``contract["claim"]["calibration"]``.

    ``status`` is the native string (``calibrated``, ``boundary``,
    ``scope_not_assessed``, ``unavailable``, or any status a newer claim
    carries); it is never normalized. ``reason`` is the reason code when the
    interval is not calibrated. ``level`` / ``observed_coverage`` /
    ``replicates`` are the record's measured coverage fields when the claim
    carries them, and ``extra`` keeps every other slot field.
    """

    status: str = "unavailable"
    record_id: str | None = None
    reason: str | None = None
    scope_n: int | None = None
    scope_dependence: str | None = None
    calibration_sha: str | None = None
    level: float | None = None
    observed_coverage: float | None = None
    replicates: int | None = None
    extra: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any] | None) -> CalibrationInfo:
        if not isinstance(contract, Mapping):
            return cls(status="unavailable", reason="not_executed")
        claim = contract.get("claim")
        raw_slot = claim.get("calibration") if isinstance(claim, Mapping) else None
        slot: Mapping[str, Any] = raw_slot if isinstance(raw_slot, Mapping) else {}
        if not slot and "calibration_status" not in contract:
            return cls(status="unavailable", reason="not_executed")

        def pick(name: str) -> Any:
            value = slot.get(name)
            if value is None:
                flat = "calibration_sha" if name == "calibration_sha" else f"calibration_{name}"
                value = contract.get(flat)
            return None if value == "" else value

        extra = tuple((str(k), v) for k, v in slot.items() if k not in _CALIBRATION_FIELDS)
        if not slot:
            extra = tuple(
                (k.removeprefix("calibration_"), v)
                for k, v in contract.items()
                if isinstance(k, str)
                and k.startswith("calibration_")
                and k.removeprefix("calibration_") not in _CALIBRATION_FIELDS
                and k != "calibration_sha"
            )
        return cls(
            status=str(pick("status") or "unavailable"),
            record_id=pick("record_id"),
            reason=pick("reason"),
            scope_n=_as_optional_int(pick("scope_n")),
            scope_dependence=pick("scope_dependence"),
            calibration_sha=pick("calibration_sha"),
            level=_as_optional_float(pick("level")),
            observed_coverage=_as_optional_float(pick("observed_coverage")),
            replicates=_as_optional_int(pick("replicates")),
            extra=extra,
        )

    def describe(self) -> str:
        """One line: status, then reason code, record id, and measured fields when present.

        Absent fields are omitted, never printed as ``None``; an unrecognized
        status renders as itself.
        """
        bits = [self.status]
        if self.reason:
            bits.append(f"reason {self.reason}")
        if self.record_id:
            bits.append(f"record {self.record_id}")
        if self.level is not None:
            bits.append(f"level {self.level:g}")
        if self.observed_coverage is not None:
            bits.append(f"observed coverage {self.observed_coverage:g}")
        if self.replicates is not None:
            bits.append(f"{self.replicates} replicates")
        bits.extend(f"{key} {value}" for key, value in self.extra if value not in (None, ""))
        return " · ".join(bits)


AnswerKind = Literal["point", "bounds", "partial", "response", "structured", "unavailable"]

#: The closed :attr:`Answer.kind` vocabulary, shared by live and loaded results.
ANSWER_KINDS: tuple[AnswerKind, ...] = get_args(AnswerKind)

#: Portable claim kind (``contract["claim"]["kind"]``) → :attr:`Answer.kind`.
#: A loaded result and the live result of the same execution give one kind.
CLAIM_KIND_ANSWERS: dict[str, AnswerKind] = {
    "point": "point",
    "bounds": "bounds",
    "mixture": "partial",
    "response": "response",
    "incomplete": "unavailable",
    "refusal": "unavailable",
}


@dataclass(frozen=True, slots=True)
class Answer:
    """Safe consumption shape. ``kind`` is one of :data:`ANSWER_KINDS`:

    - ``point``: a complete scalar claim; ``value`` holds it.
    - ``bounds``: a set-identified scalar; ``bounds`` is the identified set
      ``(lower, upper)`` over identified completions.
    - ``partial``: identification is partial or leftover structural mass
      remains, so there is no scalar (or, for a function-valued claim, no
      unrestricted curve: read ``result.envelope``); ``bounds`` carries the
      identified set whenever the execution computed one, and ``detail`` names
      the limitation.
    - ``response``: a function-valued claim (response curve, intervention
      response, derivative or Jacobian); read ``result.response`` /
      ``result.estimate``.
    - ``structured``: an executed claim with no single scalar, such as a
      multi-horizon temporal mediation grid; read its structured fields
      (``result.mediation_grid``).
    - ``unavailable``: no claim (not identified, refused, not executed,
      non-finite, or not semantically accepted); ``detail`` names why.

    This is the interface that withholds an unrestricted scalar when
    identification is partial or leftover mass remains. Historical fields
    such as ``effect``, ``ate``, ``posterior``, and ``response`` stay on the
    result for existing callers. ``effect`` / ``ate`` warn when a point
    display would misrepresent; ``ANTECEDENT_STRICT_ANSWER=1`` raises.
    """

    kind: AnswerKind
    value: float | None = None
    bounds: tuple[float, float] | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ANSWER_KINDS:
            raise ValueError(f"Answer.kind must be one of {ANSWER_KINDS}; got {self.kind!r}")


def _scalar_bounds(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    lower, upper = value
    return (float(lower), float(upper))


def answer_from_artifact(contract: Mapping[str, Any], payload: Mapping[str, Any]) -> Answer:
    """The :class:`Answer` of a verified portable execution, from its claim kind."""
    claim = contract.get("claim")
    claim = claim if isinstance(claim, Mapping) else {}
    claim_kind = claim.get("kind")
    kind = CLAIM_KIND_ANSWERS.get(str(claim_kind)) if claim_kind is not None else None
    if kind is None:
        return Answer("unavailable", detail=f"unrecognized_claim_kind:{claim_kind}")
    limitation = ReasoningSlots.from_result_section(contract, payload).rendering_limitation()
    structural = payload.get("structural_response")
    envelope = structural.get("identified_set") if isinstance(structural, Mapping) else None
    bounds = None
    if (
        isinstance(envelope, Mapping)
        and len(envelope.get("lower", [])) == len(envelope.get("upper", [])) == 1
    ):
        bounds = (float(envelope["lower"][0]), float(envelope["upper"][0]))
    if kind == "point":
        bits = claim.get("value_bits")
        if bits is None:
            return Answer("structured")
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if not isfinite(value):
            return Answer("unavailable", detail="non_finite_effect")
        return Answer("point", value=value)
    if kind == "unavailable":
        return Answer("unavailable", detail=limitation or str(claim_kind))
    if kind == "response":
        # A limited function-valued claim is partial: its envelope, not a curve.
        return Answer("partial" if limitation else "response", detail=limitation)
    return Answer(kind, bounds=bounds, detail=limitation)


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

    #: ``True`` on function-valued result families, whose claim kind is ``response``.
    _function_valued: ClassVar[bool] = False

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
        """Claim kind of this execution; :data:`CLAIM_KIND_ANSWERS` gives the loaded twin."""
        limitation = getattr(self, "rendering_limitation", lambda: None)()
        bounds = _scalar_bounds(getattr(self, "structural_identified_set", None))
        if self._function_valued:
            # A limited function-valued claim is partial: its envelope, not a curve.
            return Answer("partial" if limitation else "response", detail=limitation)
        if limitation == "identification_unavailable":
            return Answer("unavailable", detail=limitation)
        if bounds is not None:
            return Answer("bounds", bounds=bounds, detail=limitation)
        if limitation is not None:
            return Answer("partial", detail=limitation)
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
                if isinstance(
                    (
                        contract := getattr(self, "_contract", None)
                        or getattr(slots, "contract", None)
                    ),
                    Mapping,
                )
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
