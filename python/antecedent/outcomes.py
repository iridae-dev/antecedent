"""Outcomes: the one return type for scientific answers and scientific refusals.

Every operation that answers a scientific question returns an :class:`Outcome`. It holds
either a typed result (``outcome.value``) or a :class:`Refusal` (``outcome.refusal``), never
both. A refusal states that the question, as declared, cannot be answered from the data and
assumptions given (for example the effect is not identified); it is a finding, so it is
returned, not raised. Operational failures (invalid arguments, cancellation, exceeded resource
limits, serialization errors, failing callbacks) are raised as
:class:`~antecedent.errors.CausalError` subclasses.

Example::

    outcome = antecedent.analyze(data, graph=graph, query=query)
    if outcome.refusal is not None:
        print(outcome.refusal.code, outcome.refusal.remedy)
    else:
        answer = outcome.answer            # Answer(kind="point", value=...)
        print(answer.kind, answer.value, outcome.uncertainty.summary)

:func:`require_success` unwraps an outcome and re-raises a refusal's original error for code
that prefers exceptions.
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
    CausalEstimateError,
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
    """The answer of a result in a form that is safe to consume without knowing the result type.

    Read ``kind`` first; it says which other field carries the answer:

    * ``"point"`` -- a finite scalar estimate in ``value``.
    * ``"bounds"`` -- an identified set ``(lower, upper)`` in ``bounds``; the data and assumptions
      do not determine a single value.
    * ``"response"`` -- a function-valued answer (a curve or surface); read it from
      :attr:`Outcome.response`.
    * ``"partial"`` -- an answer limited as ``detail`` states (for example an envelope instead of a
      curve); no scalar is offered.
    * ``"distribution"`` -- a law; read it from :attr:`Outcome.distribution`.
    * ``"policy_value"`` -- the value of a policy, with its record in ``structured``.
    * ``"associational"`` -- a noncausal description of the observed data.
    * ``"structured"`` -- a family-specific record, in ``structured`` when one is provided,
      otherwise on the result object itself.
    * ``"unavailable"`` -- no answer could be published; ``detail`` gives the reason, for example
      ``"non_finite_effect"``.
    * ``"refused"`` -- the question was refused; see :class:`Refused`.

    Fields:

    * ``kind`` -- one of the values above. Any other value raises ``ValueError``.
    * ``value`` -- the point estimate, or ``None`` when ``kind`` is not ``"point"``.
    * ``bounds`` -- ``(lower, upper)``, or ``None``.
    * ``detail`` -- a short machine-readable qualifier of the answer, or ``None``.
    * ``structured`` -- a read-only mapping, or ``None``.

    An answer carries no uncertainty. Read intervals, standard errors and their calibration status
    from :attr:`Outcome.uncertainty` or the result's ``inspect()`` report.
    """

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
    """A scientific refusal: why a question could not be answered, and what would change that.

    Carried by :attr:`Outcome.refusal`. Read ``code`` to branch, ``message`` for the explanation
    and ``remedy`` for the next attempt.

    * ``code`` -- the registered refusal code, for example ``"effect_not_identified"`` or
      ``"positivity_violated"``. When the refusing error has no reason code, the name of its class.
    * ``message`` -- human-readable explanation.
    * ``detail`` -- a namespaced qualifier of the code (``family.slot``), or ``None``.
    * ``stage`` -- the stage that refused, or ``None``.
    * ``remedy`` -- what the caller can change to proceed, when the refusal names it.
    * ``offending`` -- the field, variable or identity the refusal is about, when there is one.
    * ``fields`` -- read-only mapping of every structured field the refusal carries.
    * ``operation`` -- name of the operation that refused, or ``None``.

    :meth:`raise_error` raises the original exception for code that prefers exceptions.
    """

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
        """The underlying refusals this refusal aggregates, as read-only mappings.

        Read from the ``causes`` (or ``refusals``) entry of :attr:`fields`; empty when there is
        none.
        """
        retained = self.fields.get("causes", self.fields.get("refusals", ()))
        return (
            tuple(cause for cause in retained if isinstance(cause, Mapping))
            if isinstance(retained, (list, tuple))
            else ()
        )

    @property
    def obligations(self) -> tuple[Mapping[str, Any], ...]:
        """The evidence or assumption obligations that were not met, as read-only mappings.

        Read from the ``obligations``, ``evidence_obligations`` or ``unmet_obligations`` entry of
        :attr:`fields`, in that order of preference; empty when there is none.
        """
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
        """Suggested next actions named by the refusal, as read-only mappings; never executed."""
        retained = self.fields.get("next_actions", ())
        return (
            tuple(item for item in retained if isinstance(item, Mapping))
            if isinstance(retained, (list, tuple))
            else ()
        )

    @property
    def cause(self) -> BaseException | None:
        """The exception the original error was raised from (its ``__cause__``), or ``None``."""
        return self._error.__cause__

    def raise_error(self) -> None:
        """Raise the original error, with its structured fields and cause chain intact.

        The error is a :class:`~antecedent.errors.CausalError` subclass such as
        :class:`~antecedent.errors.EffectNotIdentified`.
        """
        raise self._error


@dataclass(frozen=True, slots=True, repr=False)
class Refused(Answer):
    """The :class:`Answer` of a refused outcome.

    ``kind`` is always ``"refused"``, ``detail`` is the refusal code and ``refusal`` is the
    :class:`Refusal`. ``value`` and ``bounds`` are ``None``: a refusal is never a number, a set or
    a missing execution.
    """

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
        "mechanism_declaration_invalid",
        "mechanism_resolver_missing",
        "mechanism_resolver_failed",
        "mechanism_implementation_unavailable",
        "mechanism_implementation_changed",
        "policy_callback_failed",
        "policy_callback_output_invalid",
        "policy_callback_not_portable",
        "policy_callback_unrecoverable",
    }
)


# Codes an estimator names when its fixed refusal says the request, as declared, cannot be
# answered by this estimator or route (an estimand, population, option or inference mode it does
# not implement, a companion option it needs, an overlap declaration, a positivity requirement).
# An estimate error without one of these codes is an operational failure and raises.
_ESTIMATE_REFUSAL_CODES = frozenset(
    {
        "route_not_supported",
        "strategy_incompatible",
        "required_option_missing",
        "option_not_applicable",
        "estimator_inference_mismatch",
        "positivity_violated",
        "transport_target_overlap_failed",
    }
)


def is_scientific_refusal(error: BaseException) -> bool:
    """Return whether ``error`` is a scientific refusal, as opposed to an operational failure.

    The classification reads the error's type and registered ``reason_code``; it never matches
    message text. These are operational, and return ``False``: ``TypeError`` and ``ValueError``
    (including :class:`~antecedent.errors.CausalTypeError` and
    :class:`~antecedent.errors.CausalValueError`), resource, cancellation, serialization and state
    errors, any error raised from another exception (a callback failure), and protocol codes such
    as ``invalid_argument`` or ``not_executed``.

    These are scientific, and return ``True``: an error carrying native structured refusal
    fields; :class:`~antecedent.errors.EffectNotIdentified`,
    :class:`~antecedent.errors.ReviewRequired` and
    :class:`~antecedent.errors.StructuredRefusal`; an estimate error whose code says the request
    cannot be answered by the estimator or route (``route_not_supported``,
    ``strategy_incompatible``, ``required_option_missing``, ``option_not_applicable``,
    ``estimator_inference_mismatch``, ``positivity_violated``); and a
    :class:`~antecedent.errors.CausalUnsupportedError` with one of a fixed set of codes such as
    ``effect_not_identified``, ``not_supported``, ``route_not_supported``,
    ``population_not_estimable`` or ``stratum_too_sparse``.
    """
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
    # A native refusal that carries its structured stage/subject/reason/remedy fields is a
    # typed statement that the data cannot answer the question, not a failed implementation.
    fields = getattr(error, "refusal_fields", None)
    if isinstance(fields, Mapping) and fields:
        return True
    if (
        isinstance(error, CausalEstimateError)
        and getattr(error, "reason_code", None) in _ESTIMATE_REFUSAL_CODES
    ):
        return True
    return isinstance(error, (EffectNotIdentified, ReviewRequired, StructuredRefusal)) or (
        isinstance(error, CausalUnsupportedError)
        and getattr(error, "reason_code", None)
        in {
            "effect_not_identified",
            "not_supported",
            "route_not_supported",
            "row_weights_bound_to_snapshot",
            "treatment_level_undeclared",
            "population_not_estimable",
            "transport_target_overlap_failed",
            "stratum_too_sparse",
            "policy_dependency_circular",
            "policy_dependency_unavailable",
            "policy_directive_conflict",
            "policy_output_type_invalid",
            "policy_history_unavailable",
            "policy_route_not_supported",
            "adaptive_policy_unobserved_feature",
            "adaptive_policy_identification_unsupported",
            "adaptive_policy_heldout_unsupported",
        }
    )


def outcome_from_error(error: BaseException, *, operation: str | None = None) -> Outcome[Any]:
    """Convert a scientific refusal into a refused :class:`Outcome`; re-raise anything else.

    ``error`` is classified by :func:`is_scientific_refusal`. An operational error is raised
    unchanged. A refusal becomes ``Outcome(refusal=Refusal(...))`` whose fields are read from the
    error (``reason_code``, ``message``, ``detail``, ``stage``, ``remedy``, ``offending``) and
    which keeps the original error for :meth:`Refusal.raise_error`. ``operation`` is recorded as
    :attr:`Refusal.operation`.
    """
    if not is_scientific_refusal(error):
        raise error
    native_fields = getattr(error, "fields", None)
    fields = dict(native_fields) if isinstance(native_fields, Mapping) else {}
    fields.update(
        {
            key: value
            for key, value in vars(error).items()
            if not key.startswith("_") and key not in {"study", "report", "fields", "fallback"}
        }
    )
    # A native refusal carries its structured diagnostics only in `refusal_fields`; read stage
    # and detail from there when the error has no attribute of its own.
    native = getattr(error, "refusal_fields", None)
    native = native if isinstance(native, Mapping) else {}
    refusal = Refusal(
        code=getattr(error, "reason_code", None) or type(error).__name__,
        message=getattr(error, "message", None) or str(error),
        detail=getattr(error, "detail", None) or native.get("reason"),
        stage=getattr(error, "stage", None) or native.get("stage"),
        remedy=getattr(error, "remedy", None),
        offending=getattr(error, "offending", None),
        fields=fields,
        operation=operation,
        _error=error,
    )
    return Outcome(refusal=refusal)


@dataclass(frozen=True, slots=True, repr=False)
class Outcome(Generic[T]):
    """The result of a scientific operation: a typed value, or an explicit refusal.

    Returned by :func:`antecedent.analyze`, :func:`antecedent.estimate` and the other operations
    that answer a question. Exactly one of ``value`` and ``refusal`` is set (constructing an
    outcome with both raises ``ValueError``).

    Read in this order:

    1. :attr:`refusal` -- ``None`` on success. Otherwise a :class:`Refusal` with ``code``,
       ``message`` and ``remedy``; the remaining readings below then report "unavailable".
    2. :attr:`answer` -- the estimate as an :class:`Answer` (``kind``, then ``value`` or
       ``bounds``).
    3. :attr:`uncertainty`, :attr:`support`, :attr:`assumptions` -- what qualifies the answer:
       the interval or posterior summary and its calibration status, whether the analysis cell is
       supported, and the assumptions the claim rests on.
    4. :meth:`inspect` -- the full report, including identification and diagnostics.
    5. :attr:`value` -- the typed result object (for example
       :class:`~antecedent.AnalysisResult`) for family-specific fields.

    Fields:

    * ``value`` -- the typed result, or ``None`` for a refusal.
    * ``refusal`` -- the :class:`Refusal`, or ``None``.
    * ``associational`` -- set only when the study was prepared with ``on_refusal="describe"`` and
      identification was refused: an outcome holding a noncausal description of the observed
      data. It is not a causal answer.
    * ``evidence_plan`` -- with the same opt-in, the evidence that would resolve the refusal.

    Example::

        outcome = antecedent.analyze(data, graph=graph, query=query)
        if outcome.accepted:
            print(outcome.answer, outcome.uncertainty.summary)
            artifact = outcome.export()
        else:
            print(outcome.refusal)
    """

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
        """``True`` when the operation produced a result, ``False`` when it was refused."""
        return self.refusal is None

    @property
    def answer(self) -> Answer:
        """The answer in consumption form; see :class:`Answer`.

        A refusal gives a :class:`Refused` answer. Otherwise the result's own ``answer`` is
        returned; a bare finite number is a ``"point"`` answer, a non-finite one is
        ``"unavailable"`` with detail ``"non_finite_effect"``, and any other value is
        ``"structured"``.
        """
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
        """Return the full inspection report of the result or of the refusal.

        The report (an ``InspectionReport``) has four slots, ``identification``, ``support``,
        ``uncertainty`` and ``assumptions``, each with ``available``, ``summary``, ``reason`` and a
        ``payload``; the ``answer``; ``calibration``; ``diagnostics``; the contract identities; and
        the ``operations`` the outcome supports. ``to_dict()`` gives a JSON-safe copy.

        For a refusal, or a value without its own report, every slot is unavailable, ``stage`` is
        ``"refused"`` (or ``"executed"``), and the refusal's fields, obligations and next actions
        are reported under ``details``, ``evidence_obligations`` and ``next_actions``.
        """
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
        """How a loaded artifact was accepted, or ``None`` for a freshly computed outcome.

        Set on outcomes returned by :func:`antecedent.load`. Its ``status`` is ``"loaded"`` (the
        artifact is intact; nothing was recomputed), ``"replayed"`` (see :meth:`replay`) or
        ``"unavailable"``, and ``identity_bound`` says whether the load was checked against an
        expected identity. Acceptance does not certify the assumptions of the loaded claim.
        """
        return (
            self._acceptance
            if self._acceptance is not None
            else getattr(self.value, "acceptance", None)
        )

    def replay(self) -> Outcome[T]:
        """Recompute the published numbers of a loaded analysis result, on request.

        Loading checks that an artifact is intact and recomputes nothing. Replay returns the same
        outcome with its :attr:`acceptance` updated: ``replayed`` when every part was recomputed and
        agrees; otherwise ``integrity``, ``not_replayable`` and ``mismatches`` say why not. A
        refusal replays to itself.
        """
        if self.refusal is not None:
            return self
        # A loaded analysis result: its value is a result with `replay`.
        return capture_outcome(cast(Any, self.value).replay)

    @property
    def uncertainty(self) -> SlotModel:
        """The uncertainty slot of :meth:`inspect`.

        A slot with ``available``, ``summary``, ``reason`` and ``payload``. When no interval or
        posterior summary is published, ``available`` is ``False`` and ``reason`` says why.
        """
        return self.inspect().uncertainty

    @property
    def support(self) -> SlotModel:
        """The support slot of :meth:`inspect`: whether the analysis cell that ran is supported."""
        return self.inspect().support

    @property
    def assumptions(self) -> SlotModel:
        """The assumptions slot of :meth:`inspect`: the assumptions the claim rests on."""
        return self.inspect().assumptions

    @property
    def diagnostics(self) -> tuple[Any, ...]:
        """The diagnostics of :meth:`inspect`, as a tuple; empty for a refusal."""
        return tuple(self.inspect().diagnostics)

    @property
    def provenance(self) -> Mapping[str, Any]:
        """The provenance of the result, as a read-only mapping.

        The result's own ``provenance`` when it has one (a sequence of steps is returned under the
        key ``"steps"``); otherwise the ``provenance`` section of the inspected contract; otherwise
        empty. Distinct from :attr:`method_provenance`.
        """
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
    def method_provenance(self) -> Any:
        """Provenance records behind the route this outcome executed.

        A :class:`~antecedent.citations.MethodProvenance` with the sources, documented
        deviations, review scope limits, id and digest of each record; empty for a refusal.
        Descriptive only; distinct from :attr:`provenance`.
        """
        from .citations import provenance_of

        return provenance_of(self)

    def citations(self) -> tuple[Any, ...]:
        """This outcome's sources, de-duplicated, each with its records' caveats.

        :func:`antecedent.citations.bibliography` de-duplicates across a study's outcomes.
        """
        return self.method_provenance.citations()

    @property
    def identities(self) -> Mapping[str, Any]:
        """The content identities that pin this outcome, as a read-only mapping.

        Includes whichever of ``program_id``, ``lowered_program_id``, ``claim_id``, ``target_id``,
        ``identification_id``, ``identification_product_id``, ``inference_binding_id``,
        ``observation_id``, ``data_snapshot_id``, ``execution_id``, ``score_reuse_id`` and
        ``target_weights_id`` the inspection report carries, merged over the result's own
        ``identities``, plus ``identity`` when the result has one.
        """
        report = self.inspect()
        names = (
            "program_id",
            "lowered_program_id",
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
        """Names of the operations this outcome supports, from the inspection report.

        ``"refresh"`` and ``"explore"`` are listed only when the outcome retains its study
        (:attr:`study` is not ``None``).
        """
        return tuple(
            name
            for name in self.inspect().operations
            if name not in {"refresh", "explore"} or self.study is not None
        )

    @property
    def study(self) -> PreparedAnalysis | PreparedBatch | None:
        """The prepared study this outcome was computed from, or ``None``.

        Pass it to :func:`antecedent.rebind`, :func:`antecedent.reconfigure` or
        :func:`antecedent.prepared.refute` to continue from this result. An outcome loaded from an
        artifact has no study.
        """
        return self._study if self._study is not None else getattr(self.value, "_prepared", None)

    @property
    def response(self) -> Any | None:
        """The function-valued answer (a response curve or surface), or ``None``.

        Set only when :attr:`answer` has kind ``"response"`` or ``"partial"``.
        """
        if self.answer.kind not in {"response", "partial"}:
            return None
        return getattr(self.value, "response", None)

    @property
    def distribution(self) -> Any | None:
        """The law this outcome answers with, or ``None``.

        Set only when :attr:`answer` has kind ``"distribution"``.
        """
        if self.answer.kind != "distribution":
            return None
        law = getattr(self.value, "distribution", self.value)
        return law() if callable(law) else law

    def as_mean_source(self, **coordinates: Any) -> Outcome[Any]:
        """Adapt this outcome's mean for use as an input to a downstream composition or decision.

        ``coordinates`` are the keyword arguments of
        :func:`antecedent.results.sources.adapt_mean_source` (``quantity``, ``field``,
        ``contrast``): they declare the units, population or functional the producer does not state
        itself. The adapter carries the mean only, never an interval. Returns an :class:`Outcome`
        holding the mean source; an incompatible producer is refused or raises as that function
        documents.
        """
        from .results.sources import adapt_mean_source

        return capture_outcome(adapt_mean_source, self, **coordinates)

    def export(self, *args: Any, **kwargs: Any) -> ExportedArtifact[Any]:
        """Export this outcome as a portable artifact that :func:`antecedent.load` reads back.

        For a result, positional and keyword arguments are passed to the result's own ``export``;
        for example ``quantity="posterior"`` selects a Bayesian posterior for
        ``Bayesian(prior_from=...)`` and ``quantity="query"`` the frozen query. A refusal exports
        as a refusal record and takes no arguments. An outcome produced with
        ``on_refusal="describe"`` holds two results, so it requires ``quantity="refusal"`` or
        ``quantity="associational"``.

        Returns an :class:`~antecedent.artifacts.ExportedArtifact` with the bytes in ``data``, the
        artifact ``kind``, an ``expectation`` that a consumer can pass to
        :func:`antecedent.load` as ``expected=`` to check identity, and ``metadata``.

        Raises:
            CausalValueError: when a described refusal is exported without ``quantity``.
            TypeError: when arguments are passed to a plain refusal export.
            CausalSerializationError: when a refusal holds a field that cannot be serialized or
                exceeds the size bound.
        """
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
    """Return ``value`` unchanged when it is an :class:`Outcome`; otherwise wrap it as a success."""
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
    """Call ``fn(*args, **kwargs)`` and return its result as an :class:`Outcome`.

    A returned value is wrapped (an :class:`Outcome` is passed through). An exception that is a
    scientific refusal (:func:`is_scientific_refusal`) becomes a refused outcome whose
    ``operation`` is the name of ``fn``; every other exception propagates unchanged.
    ``BaseException`` subclasses such as ``KeyboardInterrupt`` are never caught.
    """
    try:
        return as_outcome(fn(*args, **kwargs))
    except Exception as error:
        return outcome_from_error(error, operation=getattr(fn, "__name__", None))


@overload
def require_success(value: Outcome[T]) -> T: ...


@overload
def require_success(value: T) -> T: ...


def require_success(value: object) -> object:
    """Return the value of a successful outcome; raise the original error of a refusal.

    Use it where an exception is more convenient than a branch. A value that is not an
    :class:`Outcome` is returned unchanged.

    Example::

        result = antecedent.outcomes.require_success(antecedent.analyze(data, graph=g, query=q))

    Raises:
        CausalError: the refusal's original error (for example
            :class:`~antecedent.errors.EffectNotIdentified`) when ``value`` is a refused outcome.
    """
    if not isinstance(value, Outcome):
        return value
    if value.refusal is not None:
        value.refusal.raise_error()
    return value.value


def outcome_operation(fn: Callable[P, T]) -> Callable[P, Outcome[T]]:
    """Decorate ``fn`` so that it returns an :class:`Outcome` instead of raising refusals.

    The wrapped function calls ``fn`` through :func:`capture_outcome`. Its signature is that of
    ``fn`` with the return annotation wrapped in ``Outcome[...]``.
    """

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
        "loaded",
        expected_identity is not None,
        False,
        False,
        proof="canonical refusal record identity; the scientific refusal is retained, not reexecuted",
    )
    return Outcome(refusal=refusal, _acceptance=acceptance)
