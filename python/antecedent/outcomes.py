"""One retained outcome for scientific answers and scientific refusals.

Operational failures remain exceptions. These wrappers retain the original typed
read model and native handles; they never reconstruct scientific authority from JSON.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import wraps
from inspect import Signature, signature
from math import isfinite
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
    get_args,
    overload,
)

if TYPE_CHECKING:
    from ._artifact_protocol import ArtifactAcceptance, ExportedArtifact
    from ._batch import PreparedBatch
    from .prepared import PreparedAnalysis
    from .results._report import InspectionReport, SlotModel

from ._freeze import freeze
from .errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalSerializationError,
    CausalStateError,
    CausalUnsupportedError,
    EffectNotIdentified,
    ReviewRequired,
    StructuredRefusal,
)

T = TypeVar("T")
P = ParamSpec("P")
AnswerKind = Literal[
    "point",
    "bounds",
    "partial",
    "response",
    "structured",
    "policy_value",
    "associational",
    "distribution",
    "unavailable",
    "refused",
]
ANSWER_KINDS: tuple[AnswerKind, ...] = get_args(AnswerKind)


@dataclass(frozen=True, slots=True, repr=False)
class Answer:
    """The native claim's safe consumption shape; uncertainty stays on its read model."""

    kind: AnswerKind
    value: float | None = None
    bounds: tuple[float, float] | None = None
    detail: str | None = None
    structured: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ANSWER_KINDS:
            raise ValueError(f"Answer.kind must be one of {ANSWER_KINDS}; got {self.kind!r}")
        object.__setattr__(self, "structured", freeze(self.structured))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kind={self.kind!r}, value={self.value!r}, bounds={self.bounds!r}, detail={self.detail[:96] if self.detail else None!r})"


@dataclass(frozen=True, slots=True, repr=False)
class Refusal:
    """A scientific refusal, retaining its original exception and cause chain."""

    code: str
    message: str
    detail: str | None = None
    stage: str | None = None
    remedy: str | None = None
    offending: Any = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    operation: str | None = None
    _error: BaseException = field(default_factory=RuntimeError, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", freeze(self.fields))
        object.__setattr__(self, "offending", freeze(self.offending))

    def __repr__(self) -> str:
        return f"Refusal(code={self.code!r}, detail={self.detail!r})"

    @property
    def causes(self) -> tuple[Mapping[str, Any], ...]:
        retained = self.fields.get("causes", self.fields.get("refusals", ()))
        return (
            tuple(cause for cause in retained if isinstance(cause, Mapping))
            if isinstance(retained, (list, tuple))
            else ()
        )

    @property
    def obligations(self) -> tuple[Mapping[str, Any], ...]:
        retained = self.fields.get(
            "obligations",
            self.fields.get("evidence_obligations", self.fields.get("unmet_obligations", ())),
        )
        return (
            tuple(item for item in retained if isinstance(item, Mapping))
            if isinstance(retained, (list, tuple))
            else ()
        )

    @property
    def next_actions(self) -> tuple[Mapping[str, Any], ...]:
        retained = self.fields.get("next_actions", ())
        return (
            tuple(item for item in retained if isinstance(item, Mapping))
            if isinstance(retained, (list, tuple))
            else ()
        )

    @property
    def cause(self) -> BaseException | None:
        return self._error.__cause__

    def raise_error(self) -> None:
        """Raise the original object, preserving its structured fields and cause."""
        raise self._error


@dataclass(frozen=True, slots=True, repr=False)
class Refused(Answer):
    """An explicit refused answer; never a scalar, bounds, or a missing execution."""

    kind: Literal["refused"] = field(default="refused", init=False)
    refusal: Refusal | None = None


# These are operational/protocol outcomes even when an older family reports them
# through StructuredRefusal. Unknown generic unsupported errors stay exceptions.
_OPERATIONAL_CODES = frozenset(
    {
        "invalid_argument",
        "not_executed",
        "cancelled_no_claim",
        "transport_budget_cancel",
        "callback_protocol_invalid",
        "callback_reuse_ineligible",
        "callback_side_effect_forbidden",
        "callback_output_invalid",
        "callback_provider_unavailable",
        "callback_limit_exceeded",
        "policy_callback_failed",
        "policy_callback_output_invalid",
        "policy_callback_not_portable",
        "policy_callback_unrecoverable",
    }
)


def is_scientific_refusal(error: BaseException) -> bool:
    """Explicit typed classification; no exception-text matching or broad catch-all."""
    if getattr(error, "_antecedent_operational_callback", False):
        return False
    if isinstance(
        error,
        (
            TypeError,
            ValueError,
            CausalResourceError,
            CausalCancelledError,
            CausalSerializationError,
            CausalStateError,
        ),
    ):
        return False
    # Callback causes are operational. Do not convert them just because a native
    # domain error happens to carry a registered unsupported/refusal code.
    if error.__cause__ is not None or (
        error.__context__ is not None and not error.__suppress_context__
    ):
        return False
    if getattr(error, "reason_code", None) in _OPERATIONAL_CODES:
        return False
    return isinstance(error, (EffectNotIdentified, ReviewRequired, StructuredRefusal)) or (
        isinstance(error, CausalUnsupportedError)
        and getattr(error, "reason_code", None)
        in {
            "effect_not_identified",
            "cell_not_licensed",
            "route_not_supported",
            "row_weights_bound_to_snapshot",
            "policy_dependency_circular",
            "policy_dependency_unavailable",
            "policy_directive_conflict",
            "policy_output_type_invalid",
            "policy_history_unavailable",
            "policy_route_not_licensed",
        }
    )


def outcome_from_error(error: BaseException, *, operation: str | None = None) -> Outcome[Any]:
    """Convert a known scientific refusal; re-raise every operational error unchanged."""
    if not is_scientific_refusal(error):
        raise error
    native_fields = getattr(error, "fields", None)
    fields = dict(native_fields) if isinstance(native_fields, Mapping) else {}
    fields.update(
        {
            key: value
            for key, value in vars(error).items()
            if not key.startswith("_") and key not in {"study", "report", "fields"}
        }
    )
    refusal = Refusal(
        code=getattr(error, "reason_code", None) or type(error).__name__,
        message=getattr(error, "message", None) or str(error),
        detail=getattr(error, "detail", None),
        stage=getattr(error, "stage", None),
        remedy=getattr(error, "remedy", None),
        offending=getattr(error, "offending", None),
        fields=fields,
        operation=operation,
        _error=error,
    )
    return Outcome(refusal=refusal)


@dataclass(frozen=True, slots=True, repr=False)
class Outcome(Generic[T]):
    """A successful typed reading or an explicit scientific refusal."""

    value: T | None = None
    refusal: Refusal | None = None
    _study: PreparedAnalysis | PreparedBatch | None = field(default=None, repr=False, compare=False)
    _acceptance: ArtifactAcceptance | None = field(default=None, repr=False, compare=False)
    associational: Outcome[Any] | None = field(default=None, repr=False, compare=False)
    evidence_plan: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.refusal is not None and not isinstance(self.refusal, Refusal):
            raise TypeError("Outcome.refusal must be a Refusal or None")
        if self.refusal is not None and self.value is not None:
            raise ValueError("an outcome cannot contain both a result and a refusal")

    @property
    def accepted(self) -> bool:
        return self.refusal is None

    @property
    def answer(self) -> Answer:
        if self.refusal is not None:
            return Refused(detail=self.refusal.code, refusal=self.refusal)
        reading = getattr(self.value, "answer", None)
        if isinstance(reading, Answer):
            return reading
        # A scalar returned by an explicitly decorated scientific operation is
        # its native point answer. Booleans and arbitrary scalar-like objects
        # are not scientific effect values.
        if type(self.value) in (int, float):
            value = float(cast(int | float, self.value))
            return (
                Answer("point", value=value)
                if isfinite(value)
                else Answer(
                    "unavailable",
                    detail="non_finite_effect",
                )
            )
        return Answer("structured")

    def inspect(self) -> InspectionReport:
        if self.refusal is None:
            inspect = getattr(self.value, "inspect", None)
            if callable(inspect):
                return inspect()
        from .results._report import InspectionReport, SlotModel

        unavailable = SlotModel(available=False, reason=self.answer.detail, summary="unavailable")
        return InspectionReport(
            identification=unavailable,
            support=unavailable,
            uncertainty=unavailable,
            assumptions=unavailable,
            answer=self.answer,
            family=type(self.value).__name__,
            stage="refused" if self.refusal is not None else "executed",
            details=self.refusal.fields if self.refusal is not None else None,
            evidence_obligations=() if self.refusal is None else self.refusal.obligations,
            next_actions=() if self.refusal is None else self.refusal.next_actions,
            operations=("inspect", "export") if self.refusal is not None else ("inspect",),
        )

    @property
    def acceptance(self) -> ArtifactAcceptance | None:
        return (
            self._acceptance
            if self._acceptance is not None
            else getattr(self.value, "acceptance", None)
        )

    @property
    def uncertainty(self) -> SlotModel:
        """The retained native uncertainty slot, including unavailable/partial reasons."""
        return self.inspect().uncertainty

    @property
    def support(self) -> SlotModel:
        return self.inspect().support

    @property
    def assumptions(self) -> SlotModel:
        return self.inspect().assumptions

    @property
    def diagnostics(self) -> tuple[Any, ...]:
        return tuple(self.inspect().diagnostics)

    @property
    def provenance(self) -> Mapping[str, Any]:
        retained = getattr(self.value, "provenance", None)
        if isinstance(retained, Mapping):
            return freeze(retained)
        if isinstance(retained, (list, tuple)) and all(
            isinstance(step, Mapping) for step in retained
        ):
            return freeze({"steps": retained})
        contract = self.inspect().contract
        if isinstance(contract, Mapping) and isinstance(contract.get("provenance"), Mapping):
            return freeze(contract["provenance"])
        return freeze({})

    @property
    def identities(self) -> Mapping[str, Any]:
        report = self.inspect()
        names = (
            "program_id",
            "claim_id",
            "target_id",
            "identification_id",
            "identification_product_id",
            "inference_binding_id",
            "observation_id",
            "data_snapshot_id",
            "execution_id",
            "score_reuse_id",
            "target_weights_id",
        )
        domain = getattr(self.value, "identities", None)
        retained = dict(domain) if isinstance(domain, Mapping) else {}
        retained.update(
            {name: getattr(report, name) for name in names if getattr(report, name) is not None}
        )
        identity = getattr(self.value, "identity", None)
        if identity is not None:
            retained["identity"] = identity
        return freeze(retained)

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.inspect().operations
            if name not in {"refresh", "explore"} or self.study is not None
        )

    @property
    def study(self) -> PreparedAnalysis | PreparedBatch | None:
        """The explicitly retained canonical study; independent loads have none."""
        return self._study if self._study is not None else getattr(self.value, "_prepared", None)

    @property
    def response(self) -> Any | None:
        if self.answer.kind not in {"response", "partial"}:
            return None
        return getattr(self.value, "response", None)

    @property
    def distribution(self) -> Any | None:
        if self.answer.kind != "distribution":
            return None
        law = getattr(self.value, "distribution", self.value)
        return law() if callable(law) else law

    def as_mean_source(self, **coordinates: Any) -> Outcome[Any]:
        """Adapt the actual producer, preserving its native quantity and license checks."""
        from .results.sources import adapt_mean_source

        return capture_outcome(adapt_mean_source, self, **coordinates)

    def export(self, *args: Any, **kwargs: Any) -> ExportedArtifact[Any]:
        from ._artifact_protocol import exported_artifact

        if self.associational is not None:
            selection = kwargs.pop("quantity", None)
            if selection == "associational":
                return self.associational.export(*args, **kwargs)
            if selection != "refusal":
                from .errors import CausalValueError

                raise CausalValueError(
                    "a descriptive fallback contains two results; select quantity='refusal' "
                    "or quantity='associational'"
                )
        if self.refusal is not None:
            if args or kwargs:
                raise TypeError("a refusal export takes no selection or artifact-id arguments")
            return _export_refusal(self.refusal)
        return exported_artifact(require_success(self), *args, **kwargs)

    def __repr__(self) -> str:
        if self.refusal is not None:
            return f"Outcome(refused={self.refusal.code!r})"
        return f"Outcome(value={type(self.value).__name__})"

    def _repr_html_(self) -> str:
        return self.inspect()._repr_html_()


def as_outcome(value: T | Outcome[T]) -> Outcome[T]:
    return value if isinstance(value, Outcome) else Outcome(value=value)


@overload
def capture_outcome(
    fn: Callable[P, Outcome[T]], /, *args: P.args, **kwargs: P.kwargs
) -> Outcome[T]: ...


@overload
def capture_outcome(fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Outcome[T]: ...


def capture_outcome(
    fn: Callable[P, T | Outcome[T]], /, *args: P.args, **kwargs: P.kwargs
) -> Outcome[T]:
    """Shared boundary for existing family operations; never catches BaseException."""
    try:
        return as_outcome(fn(*args, **kwargs))
    except Exception as error:
        return outcome_from_error(error, operation=getattr(fn, "__name__", None))


@overload
def require_success(value: Outcome[T]) -> T: ...


@overload
def require_success(value: T) -> T: ...


def require_success(value: object) -> object:
    """Unwrap a successful reading for a family adapter, preserving its backing object.

    Adapters still enforce their quantity/support/uncertainty compatibility rules.
    An explicit refusal re-raises its original exception rather than becoming data.
    """
    if not isinstance(value, Outcome):
        return value
    if value.refusal is not None:
        value.refusal.raise_error()
    return value.value


def outcome_operation(fn: Callable[P, T]) -> Callable[P, Outcome[T]]:
    """Migrate an existing public execution boundary to the shared outcome contract."""

    @wraps(fn)
    def call(*args: P.args, **kwargs: P.kwargs) -> Outcome[T]:
        return capture_outcome(fn, *args, **kwargs)

    public = signature(fn)
    annotation = public.return_annotation
    result = (
        "Outcome[" + annotation + "]"
        if isinstance(annotation, str)
        else Outcome[Any]
        if annotation is Signature.empty
        else GenericAlias(Outcome, annotation)
    )
    call.__dict__["__signature__"] = public.replace(return_annotation=result)
    call.__annotations__ = {**fn.__annotations__, "return": result}
    return call


_REFUSAL_FORMAT = "scientific_refusal_v1"
_REFUSAL_MAX_BYTES = 16 * 1024 * 1024


def _export_refusal(refusal: Refusal) -> ExportedArtifact[Refused]:
    import json

    from ._artifact_protocol import ArtifactExpectation, ExportedArtifact
    from ._digest import canonical_digest
    from ._freeze import thaw

    body = {
        "format": _REFUSAL_FORMAT,
        "version": 1,
        "code": refusal.code,
        "message": refusal.message,
        "detail": refusal.detail,
        "stage": refusal.stage,
        "remedy": refusal.remedy,
        "offending": thaw(refusal.offending),
        "fields": thaw(refusal.fields),
        "operation": refusal.operation,
    }
    try:
        identity = canonical_digest("scientific_refusal.v1", body)
        encoded = json.dumps(
            {**body, "identity": identity}, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CausalSerializationError("refusal contains a non-portable field") from error
    if len(encoded) > _REFUSAL_MAX_BYTES:
        raise CausalSerializationError("refusal artifact exceeds its byte bound")
    return ExportedArtifact(
        encoded, _REFUSAL_FORMAT, ArtifactExpectation(_REFUSAL_FORMAT, identity)
    )


def _consume_refusal(artifact: bytes, expected_identity: str | None = None) -> Outcome[Refused]:
    import json

    from ._artifact_protocol import ArtifactAcceptance, container_bytes
    from ._digest import canonical_digest

    raw = container_bytes(artifact)
    if len(raw) > _REFUSAL_MAX_BYTES:
        raise CausalSerializationError("refusal artifact exceeds its byte bound")
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise CausalSerializationError("malformed refusal artifact") from error
    if not isinstance(record, dict):
        raise CausalSerializationError("refusal artifact is not an object")
    identity = record.pop("identity", None)
    required = {
        "format",
        "version",
        "code",
        "message",
        "detail",
        "stage",
        "remedy",
        "offending",
        "fields",
        "operation",
    }
    if (
        set(record) != required
        or record.get("format") != _REFUSAL_FORMAT
        or record.get("version") != 1
        or not isinstance(record.get("code"), str)
        or not isinstance(record.get("message"), str)
        or not record["code"]
        or not isinstance(record.get("fields"), dict)
        or any(
            record[name] is not None and not isinstance(record[name], str)
            for name in ("detail", "stage", "remedy", "operation")
        )
    ):
        raise CausalSerializationError("invalid refusal artifact fields or layout")
    try:
        actual = canonical_digest("scientific_refusal.v1", record)
    except (TypeError, ValueError) as error:
        raise CausalSerializationError("invalid refusal artifact values") from error
    if identity != actual or expected_identity is not None and expected_identity != actual:
        raise CausalSerializationError("refusal artifact identity mismatch")
    wire = {
        **record["fields"],
        "code": record["code"],
        "message": record["message"],
        "detail": record["detail"] or record["code"],
        "stage": record["stage"],
        "remedy": record["remedy"],
        "offending": record["offending"],
    }
    original = StructuredRefusal(wire)
    refusal = Refusal(
        code=record["code"],
        message=record["message"],
        detail=record["detail"],
        stage=record["stage"],
        remedy=record["remedy"],
        offending=record["offending"],
        fields=record["fields"],
        operation=record["operation"],
        _error=original,
    )
    acceptance = ArtifactAcceptance(
        "sealed",
        expected_identity is not None,
        False,
        False,
        proof="canonical refusal record identity; the scientific refusal is retained, not reexecuted",
    )
    return Outcome(refusal=refusal, _acceptance=acceptance)
