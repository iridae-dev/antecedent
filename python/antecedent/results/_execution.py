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
    from .response import CausalResponseView

from .._api import describe_refusal
from ..errors import CausalUnsupportedError
from ._report import InspectionReport, as_inspection
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
    skip = {"artifact", "score_table", "score_inference", "envelope"}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in fields(value) if f.name not in skip}
    try:
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            return {
                name: getattr(value, name)
                for name, field in type(value).model_fields.items()
                if field.exclude is not True and name not in skip
            }
    except ImportError:
        pass  # pydantic is optional; fall through to the generic value wrapper
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

    def refresh(
        self,
        data: Mapping[str, Any] | Any,
        *,
        seed: int | None = None,
        threads: int | None = None,
    ) -> Any:
        """Re-estimate on new data via the retained prepared handle.

        The five-line second click is ``result = analyze(...); result.refresh(new_data)``.
        A second ``analyze()`` still re-prepares. Equivalent to ``result.study.refresh``.
        """
        return self.study.refresh(data, seed=seed, threads=threads)

    def refute(
        self,
        data: Mapping[str, Any] | Any,
        suite: Any = "placebo",
        *,
        seed: int | None = None,
        threads: int | None = None,
        cancel: Any | None = None,
    ) -> Any:
        """Second-click refute via the retained prepared handle."""
        from ..ids import Refute

        if isinstance(suite, Refute):
            suite = str(suite)
        study = self.study
        execution = getattr(self, "_execution", None)
        if execution is None:
            raise CausalUnsupportedError(
                "This result has no retained execution snapshot.",
                reason_code="not_executed",
            )
        frozen = study._frozen(execution.snapshot())
        token = {} if cancel is None else {"cancel": cancel}
        return frozen.refute(data, suite, seed=seed, threads=threads, **token)

    def _scalar_effect(self) -> float | None:
        """Historical scalar without the legacy-field warning."""
        getter = getattr(self, "estimate", None)
        if getter is not None and hasattr(getter, "ate"):
            value = getter.ate
            return value if isinstance(value, (int, float)) else None
        if isinstance(getter, (int, float)):
            return float(getter)
        return None

    @property
    def effect(self) -> float | None:
        """Historical scalar. Prefer :attr:`answer` / :meth:`as_point`."""
        self._warn_legacy_scalar("effect")
        return self._scalar_effect()

    @property
    def ate(self) -> float | None:
        """Alias for :attr:`effect`."""
        self._warn_legacy_scalar("ate")
        return self._scalar_effect()

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

    def claim(self) -> str:
        """One paragraph: identification, answer, calibration. Same table as HTML."""
        from .._claim import result_claim

        identification = getattr(self, "identification", None)
        return result_claim(
            query=getattr(self, "query", None) or getattr(self, "estimand", None),
            status=getattr(identification, "status", "NotIdentified"),
            method=getattr(identification, "method", None),
            adjustment_set=tuple(getattr(identification, "adjustment_set", ()) or ()),
            answer=self.answer,
            calibration=self.calibration.describe(),
        )

    def as_point(self) -> float:
        """The scalar when :attr:`answer` is ``point``; refuse any other kind."""
        answer = self.answer
        if answer.kind != "point" or answer.value is None:
            raise CausalUnsupportedError(
                f"result.as_point() requires answer.kind='point'; got {answer.kind!r}"
                + (f" ({answer.detail})" if answer.detail else ""),
                reason_code="invalid_argument",
            )
        return float(answer.value)

    def as_response(self) -> CausalResponseView:
        """This result when it is function-valued; refuse a scalar analysis."""
        from .response import CausalResponseView

        if isinstance(self, CausalResponseView):
            return self
        raise CausalUnsupportedError(
            "result.as_response() requires a function-valued analysis "
            f"(answer.kind={self.answer.kind!r})",
            reason_code="invalid_argument",
        )

    @property
    def calibration(self) -> CalibrationInfo:
        contract = getattr(self, "_contract", None)
        if not isinstance(contract, Mapping):
            slots = getattr(self, "reasoning", None)
            contract = getattr(slots, "contract", None) if slots is not None else None
        if isinstance(contract, Mapping):
            return CalibrationInfo.from_contract(contract)
        return CalibrationInfo(status="unavailable", reason="not_executed")

    def inspect(self) -> InspectionReport:
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
        report = replace(
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
        return as_inspection(self._with_portable_record(report))

    def _with_portable_record(self, report: ReasoningSlots) -> ReasoningSlots:
        """Report the execution through the record its export carries.

        ``contract`` becomes the portable contract section and the portable
        evidence (identification method and adjustment set, validation verdict,
        unit effects) is read from the exported body, so this report and
        ``load(export()).inspect()`` agree on them. An execution that exports no
        record (none retained, or a cancelled click) has no portable contract.
        """
        from dataclasses import replace

        from .. import artifacts
        from ..errors import CausalError
        from ._slots import portable_evidence, with_portable_evidence

        execution = getattr(self, "_execution", None)
        if execution is None:
            return replace(report, contract=None)
        try:
            decoded = artifacts.loads(execution.export_contracted_artifact())
        except (CausalError, ValueError):
            return replace(report, contract=None)
        if not isinstance(decoded.contract, Mapping):
            return replace(report, contract=None)
        section = dict(decoded.contract)
        return replace(
            with_portable_evidence(report, portable_evidence(section, decoded.payload)),
            contract=section,
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
