"""Composition-aware uncertainty: propagate uncertain inputs under the dependence they need.

Unrelated posterior or bootstrap draws are never paired by index. One rule (in Rust) decides
what a composition needs:

* linear transforms, contrasts, sums and weighted sums take a **covariance** (when evidence is
  shared, or an explicit joint moment exists), or distinct evidence with a declared
  independence (:func:`linear_moments`), or an aligned joint law (:func:`aligned_linear`);
* products and nonlinear functionals need an **aligned joint law**: columns of one
  :class:`~antecedent.joint_distribution.JointDistributionArtifact`, which share one alignment
  identity (:func:`aligned_functional`). A covariance is not enough, and the rows of unrelated
  artifacts or of independently sampled marginals are refused with ``joint_law_required``;
* mixtures need known weights (:func:`mixture`);
* a transported interval composed with another claim needs an aligned law or a covariance
  (pass ``role="transported_interval"`` on a :class:`MomentInput`);
* a functional of **unpaired marginals** has a dependence envelope
  (:func:`dependence_envelope`), a Frechet-Hoeffding ``structural_envelope`` -- never a point,
  an interval or a posterior and never a substitute for an aligned joint law -- and a declared
  correlation range has an ``assumption_range`` (:func:`correlation_range`).

A value under a supported route is a labelled ``point_only`` value. Its **spread** (a variance, a
Monte Carlo standard error, an exceedance probability with its Monte Carlo band and a verdict) is
read through :func:`spread`, :func:`aligned_linear_spread`, :func:`linear_moments_spread`,
:func:`composed_exceedance`, :func:`mixture_exceedance`, :func:`transported_spread` and
:func:`transported_spread_aligned`; it is labelled ``uncalibrated_spread`` and carries no coverage
statement. A spread of a transported claim needs all three pieces (source sampling, target
sampling, transport functional) and their dependence (an aligned law or a covariance).

:meth:`PropagationClaim.interval` and :meth:`PropagationClaim.posterior` always raise
``uncertainty_representation_insufficient``. An interval at a composed coordinate is supported only by
:meth:`PropagationClaim.check_composed_interval`, which resolves the claim's coordinate
(:meth:`PropagationClaim.coordinate_id`), construction and declared evidence size against the
shipped coverage records; the caller supplies no record, and the interval carries the resulting
calibration status. :meth:`PropagationClaim.export` writes
``joint_law_propagation_v1``; :func:`consume_joint_law_propagation_artifact` re-runs the rule and
replays the result to identical bits and refuses any edit.

Every rule, number, identity and refusal is computed in Rust.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._mean_binding import bind_mean_reading
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from ._validate import finite_float_protocol as _real
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .joint_distribution import JointDistributionArtifact
from .outcomes import Answer, Outcome, outcome_operation, require_success

__all__ = [
    "ComposedInterval",
    "ConsumedJointLawPropagation",
    "EnvelopeQuery",
    "JointLawPropagationRefusal",
    "MixtureComponent",
    "MixtureDraws",
    "MomentInput",
    "PropagationClaim",
    "PropagationDraws",
    "PropagationPlan",
    "aligned_functional",
    "aligned_linear",
    "aligned_linear_spread",
    "check_propagation",
    "check_rows_pairable",
    "composed_exceedance",
    "consume_joint_law_propagation_artifact",
    "correlation_range",
    "dependence_envelope",
    "linear_moments",
    "linear_moments_spread",
    "mixture",
    "mixture_exceedance",
    "spread",
    "transported_spread",
    "transported_spread_aligned",
]

_FUNCTIONALS = ("product", "ratio", "minimum", "maximum", "squared_difference")
_ROLES = (
    "estimate",
    "transported_interval",
    "source_sampling",
    "target_sampling",
    "transport_functional",
)


class JointLawPropagationRefusal(StructuredRefusal):
    """A typed refusal of a composition.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``joint_law_required``, ``sampling_dependence_unknown``, ``evidence_double_counted``,
    ``not_supported``, ``invalid_argument``, ``route_not_supported``) and a
    ``joint_law_propagation.*`` ``detail``; ``remedy`` names the dependence information that would
    support the request.
    """


def _refuse(payload: str | None) -> None:
    raise_refusal(payload, JointLawPropagationRefusal)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{what} must be a non-empty string")
    return value


def _floats(values: Sequence[float] | np.ndarray, what: str) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise CausalValueError(f"{what} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise CausalValueError(f"{what} must be finite")
    return [float(v) for v in array]


def _cancel(execution: Execution | None) -> Any:
    """The cancellation token of ``execution`` (the default execution has none)."""
    return resolve_execution(execution).cancel


@dataclass(repr=False, frozen=True)
class PropagationDraws(ScientificResult):
    """The draws of one uncertain input with the identities the rule reads.

    Build with :meth:`from_artifact_column` (a column of a joint distribution artifact: its
    alignment identity is computed in Rust) or :meth:`independent_marginal` (draws of a marginal
    with no joint law; only the dependence envelope accepts it).
    """

    id: str
    evidence_id: str
    role: str
    alignment: str
    alignment_id: str | None
    values: np.ndarray = field(repr=False, compare=False)
    _artifact: JointDistributionArtifact | None = field(default=None, repr=False, compare=False)
    _column: int | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_artifact_column(
        cls, input_id: str, artifact: JointDistributionArtifact, column: int
    ) -> PropagationDraws:
        """One column of ``artifact``; the evidence identity is its data snapshot."""
        _text(input_id, "input_id")
        artifact = require_success(artifact)
        if not isinstance(artifact, JointDistributionArtifact):
            raise CausalTypeError("artifact must be a JointDistributionArtifact")
        if isinstance(column, bool) or not isinstance(column, int) or column < 0:
            raise CausalValueError("column must be a non-negative integer")
        parsed, refusal = _native.joint_law_column(input_id, artifact._native, column)
        _refuse(refusal)
        assert parsed is not None
        info = json.loads(parsed)
        values = np.asarray(artifact)[:, column].copy()
        return cls(
            id=input_id,
            evidence_id=info["evidence_id"],
            role="estimate",
            alignment=info["alignment"],
            alignment_id=info["alignment_id"],
            values=values,
            _artifact=artifact,
            _column=column,
        )

    @classmethod
    def independent_marginal(
        cls, input_id: str, values: Sequence[float] | np.ndarray, *, evidence_id: str
    ) -> PropagationDraws:
        """Independently produced draws of one marginal; they have no alignment identity."""
        return cls(
            id=_text(input_id, "input_id"),
            evidence_id=_text(evidence_id, "evidence_id"),
            role="estimate",
            alignment="independent_marginals",
            alignment_id=None,
            values=np.asarray(_floats(values, "values"), dtype=np.float64),
        )

    def with_evidence_source(self, source: str) -> PropagationDraws:
        """The same draws resting on the evidence-ledger source ``source`` (its identity).

        A check that has the ledger (``check_propagation(..., ledger=ledger)``) then decides
        whether this input shares observations with the others from the ledger's rows, rather
        than from the equality of data-snapshot identity strings.
        """
        return replace(self, evidence_id=_text(source, "source"))

    def _require_artifact(self) -> tuple[Any, int]:
        if self._artifact is None or self._column is None:
            raise CausalValueError(
                f"input {self.id!r} is not a joint distribution artifact column; an aligned "
                "route pairs rows only under an alignment identity, so build it with "
                "PropagationDraws.from_artifact_column"
            )
        return self._artifact._native, self._column


@dataclass(frozen=True)
class MomentInput:
    """A summary-only input: its mean (its variance is on the covariance diagonal)."""

    id: str
    evidence_id: str
    mean: float
    role: str = "estimate"

    def _wire(self) -> dict[str, Any]:
        if self.role not in _ROLES:
            raise CausalValueError(f"role must be one of {_ROLES}; got {self.role!r}")
        return {
            "id": _text(self.id, "id"),
            "evidence_id": _text(self.evidence_id, "evidence_id"),
            "role": self.role,
            "mean": _real(self.mean, "mean"),
        }


@dataclass(frozen=True)
class MixtureComponent:
    """One component of a finite mixture with its known weight."""

    id: str
    evidence_id: str
    weight: float
    mean: float
    variance: float

    def _wire(self) -> dict[str, Any]:
        return {
            "id": _text(self.id, "id"),
            "evidence_id": _text(self.evidence_id, "evidence_id"),
            "weight": _real(self.weight, "weight"),
            "mean": _real(self.mean, "mean"),
            "variance": _real(self.variance, "variance"),
        }


@dataclass(frozen=True)
class MixtureDraws:
    """One component of a finite mixture given as draws, with its known weight."""

    id: str
    evidence_id: str
    weight: float
    draws: Sequence[float] | np.ndarray

    def _wire(self) -> dict[str, Any]:
        return {
            "id": _text(self.id, "id"),
            "evidence_id": _text(self.evidence_id, "evidence_id"),
            "weight": _real(self.weight, "weight"),
            "draws": _floats(self.draws, "draws"),
        }


@dataclass(repr=False, frozen=True)
class ComposedInterval(ScientificResult):
    """A composed interval annotated with the calibration of its composed coordinate.

    ``lower`` and ``upper`` are present when the construction is built from the claim's own mean
    and variance (``normal_approximation``); otherwise the support names the construction the
    caller builds from the aligned draws. ``calibration`` is the slot the registry resolves
    (``status`` is ``calibrated`` only under a currently attesting record, which ``record_id``
    names); the interval is reported whatever the status.
    ``declared_evidence_rows`` is the caller's declared evidence size the record was resolved at;
    the sealed claim does not authenticate it, so the support holds only if it is true.
    """

    coordinate_id: str
    record_id: str | None
    calibration: Mapping[str, Any]
    construction: str
    nominal_level: float
    label: str
    declared_evidence_rows: int
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class EnvelopeQuery:
    """What a dependence envelope is asked for. Build with the class methods."""

    kind: str
    threshold: float | None = None
    a: float | None = None
    b: float | None = None
    functional: str | None = None

    @classmethod
    def sum_cdf(cls, threshold: float) -> EnvelopeQuery:
        """``P(X + Y <= threshold)`` over every coupling (Makarov bounds)."""
        return cls("sum_cdf", threshold=_real(threshold, "threshold"))

    @classmethod
    def difference_cdf(cls, threshold: float) -> EnvelopeQuery:
        """``P(X - Y <= threshold)`` over every coupling."""
        return cls("difference_cdf", threshold=_real(threshold, "threshold"))

    @classmethod
    def product_moment(cls) -> EnvelopeQuery:
        """``E[X Y]`` over every coupling (comonotone and countermonotone)."""
        return cls("product_moment")

    @classmethod
    def covariance(cls) -> EnvelopeQuery:
        """``Cov(X, Y)`` over every coupling."""
        return cls("covariance")

    @classmethod
    def linear_variance(cls, a: float, b: float) -> EnvelopeQuery:
        """``Var(a X + b Y)`` over every coupling."""
        return cls("linear_variance", a=_real(a, "a"), b=_real(b, "b"))

    @classmethod
    def of_functional(cls, functional: str) -> EnvelopeQuery:
        """``E[g(X, Y)]`` for a closed functional whose sign structure is verified."""
        if functional not in _FUNCTIONALS:
            raise CausalValueError(f"functional must be one of {_FUNCTIONALS}")
        return cls("functional", functional=functional)

    def _json(self) -> str:
        body = {
            k: v
            for k, v in (
                ("kind", self.kind),
                ("threshold", self.threshold),
                ("a", self.a),
                ("b", self.b),
                ("functional", self.functional),
            )
            if v is not None
        }
        return json.dumps(body)


@dataclass(repr=False, frozen=True)
class PropagationClaim(ScientificResult):
    """A composed uncertain quantity: the supported route, the labelled values and the artifact."""

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    identity: str
    operation: str
    route: str
    label: str
    is_envelope: bool
    alignment_id: str | None
    n_rows: int | None
    values: Mapping[str, float]
    verdict: str | None
    _bytes: bytes = field(repr=False, compare=False)

    @property
    def answer(self) -> Answer:
        return Answer(
            "partial" if self.is_envelope else "structured",
            detail=self.label,
            structured=self.values,
        )

    def inspect(self) -> Any:
        from .results._report import SlotModel

        report = super().inspect()
        return report.model_copy(
            update={
                "uncertainty": SlotModel(
                    available=self.label != "point_only",
                    reason="no_interval_claim",
                    summary=self.label,
                    payload={"route": self.route, "values": self.values}
                    if self.label != "point_only"
                    else {},
                )
            }
        )

    @classmethod
    def _from(cls, parts: tuple[bytes, str]) -> PropagationClaim:
        data, summary = parts
        info = json.loads(summary)
        return cls(
            identity=info["identity"],
            operation=info["operation"],
            route=info["route"],
            label=info["label"],
            is_envelope=info["is_envelope"],
            alignment_id=info["alignment_id"],
            n_rows=info["n_rows"],
            values={name: value for name, value in info["values"]},
            verdict=info.get("verdict"),
            _bytes=bytes(data),
        )

    def value(self, name: str) -> float:
        """A named value (``mean``, ``variance``, ``..._lower`` / ``..._upper``)."""
        try:
            return self.values[name]
        except KeyError as error:
            raise CausalValueError(
                f"no value {name!r}; this claim carries {sorted(self.values)}"
            ) from error

    @artifact_export
    def export(self) -> ExportedArtifact[ConsumedJointLawPropagation]:
        """The sealed ``joint_law_propagation_v1`` artifact."""
        return artifact_from_receipt(self, self._bytes)

    def interval(self) -> None:
        """Always raises ``uncertainty_representation_insufficient``: a component interval does not cover the composition.

        An interval at the composed coordinate is read through :meth:`check_composed_interval`.
        """
        _refuse(_native.joint_law_unsupported_reading("interval"))

    def coordinate_id(self) -> str:
        """The composed coordinate (``operation|route|roles``) a coverage record must be measured at."""
        coordinate, refusal = _native.joint_law_coordinate_id(self._bytes)
        _refuse(refusal)
        assert coordinate is not None
        return coordinate

    @outcome_operation
    def check_composed_interval(
        self, construction: str, *, evidence_rows: int, execution: Execution | None = None
    ) -> ComposedInterval:
        """The interval reading, annotated with the registry's calibration at this coordinate.

        The sealed artifact is verified and replayed first, then the claim's coordinate,
        ``construction`` and ``evidence_rows`` (the declared number of observations behind the
        least-supported input) are resolved against the shipped coverage records. The caller
        supplies no record. Without an attesting record the interval is still reported and its
        ``calibration`` says so (``outside_measured_scope`` or ``not_measured``). A claim whose
        dependence route supports no spread (an envelope) is refused with its own detail.
        """
        if (
            isinstance(evidence_rows, bool)
            or not isinstance(evidence_rows, int)
            or not 1 <= evidence_rows < 2**63
        ):
            raise CausalValueError("evidence_rows must be an integer of at least one")
        rows = evidence_rows
        support, refusal = _native.joint_law_check_composed_interval(
            self._bytes,
            _text(construction, "construction"),
            rows,
            _cancel(execution),
        )
        _refuse(refusal)
        assert support is not None
        info = json.loads(support)
        return ComposedInterval(
            coordinate_id=info["coordinate_id"],
            record_id=info["record_id"],
            calibration=dict(info["calibration"]),
            construction=info["construction"],
            nominal_level=info["nominal_level"],
            label=info["label"],
            declared_evidence_rows=info["declared_evidence_rows"],
            lower=info["lower"],
            upper=info["upper"],
        )

    def posterior(self) -> None:
        """Always raises ``uncertainty_representation_insufficient``."""
        _refuse(_native.joint_law_unsupported_reading("posterior"))


@dataclass(repr=False, frozen=True)
class ConsumedJointLawPropagation(ScientificResult):
    """A replayed artifact; every value equals the stored bits."""

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    identity: str
    operation: str
    route: str
    label: str
    is_envelope: bool
    alignment_id: str | None
    n_rows: int | None
    values: Mapping[str, float]
    verdict: str | None = None

    @property
    def answer(self) -> Answer:
        return Answer(
            "partial" if self.is_envelope else "structured",
            detail=self.label,
            structured=self.values,
        )

    def inspect(self) -> Any:
        from .results._report import SlotModel

        report = super().inspect()
        return report.model_copy(
            update={
                "uncertainty": SlotModel(
                    available=self.label != "point_only",
                    reason="no_interval_claim",
                    summary=self.label,
                    payload={"route": self.route, "values": self.values}
                    if self.label != "point_only"
                    else {},
                )
            }
        )


@bind_mean_reading
def _claim(result: tuple[tuple[bytes, str] | None, str | None]) -> PropagationClaim:
    parts, refusal = result
    _refuse(refusal)
    assert parts is not None
    return PropagationClaim._from(parts)


_DEPENDENCES = (
    "unknown",
    "declared_independent",
    "covariance",
    "aligned_joint_law",
    "known_weights",
    "correlation_range",
    "marginals_only",
)
_READINGS = ("point", "spread", "interval", "posterior", "envelope")
_OPERATIONS = (
    "linear_transform",
    "contrast",
    "sum",
    "weighted_sum",
    "product",
    "nonlinear_functional",
    "chained_sampling",
    "mixture",
    "transported_composition",
)


@dataclass(repr=False, frozen=True)
class PropagationPlan(ScientificResult):
    """What a passed check of a composition recorded, before any number is formed.

    ``route`` is the computation the rule would run (``linear_moments``, ``aligned_joint_law``,
    ``known_weight_mixture`` or an envelope), ``alignment_id`` the identity the inputs share for
    an aligned route, ``covariance_id`` the declared covariance for a covariance route and
    ``shared_evidence`` the evidence identities two or more inputs share. ``label`` is the
    label the result would carry; nothing here is an interval or a posterior.
    """

    operation: str
    route: str
    alignment_id: str | None
    covariance_id: str | None
    shared_evidence: tuple[str, ...]
    label: str
    missing_transport_pieces: tuple[str, ...]


def _checked_inputs(inputs: Sequence[PropagationDraws | MomentInput]) -> str:
    rows = []
    for item in inputs:
        if isinstance(item, PropagationDraws):
            rows.append(
                {
                    "id": item.id,
                    "evidence_id": item.evidence_id,
                    "role": item.role,
                    "alignment": item.alignment,
                    "alignment_id": item.alignment_id,
                }
            )
        elif isinstance(item, MomentInput):
            wire = item._wire()
            rows.append(
                {
                    "id": wire["id"],
                    "evidence_id": wire["evidence_id"],
                    "role": wire["role"],
                    "alignment": "moments_only",
                    "alignment_id": None,
                }
            )
        else:
            raise CausalTypeError("inputs must be PropagationDraws or MomentInput")
    return json.dumps(rows)


def check_propagation(
    operation: str,
    inputs: Sequence[PropagationDraws | MomentInput],
    *,
    dependence: str = "unknown",
    covariance_id: str | None = None,
    correlation_range: tuple[float, float] | None = None,
    reading: str = "point",
    ledger: Any = None,
) -> PropagationPlan:
    """Check a composition against the propagation rule before any number is formed.

    ``dependence`` is the information supplied: ``"unknown"`` (never independence),
    ``"declared_independent"`` (distinct evidence only), ``"covariance"`` (name it with
    ``covariance_id``), ``"aligned_joint_law"``, ``"known_weights"`` (a mixture),
    ``"correlation_range"`` (pass ``correlation_range=(low, high)``) or ``"marginals_only"``.

    With ``ledger`` (an :class:`~antecedent.evidence_ledger.EvidenceLedger`) whether inputs share
    evidence is decided from the ledger's observation rows, each input's ``evidence_id`` read as
    a ledger source identity (:meth:`PropagationDraws.with_evidence_source`): two sources that
    share observations are shared evidence however their identities are spelled, and a pair whose
    overlap the ledger cannot compute is unknown dependence, so a declared independence of it is
    refused (``joint_law_propagation.evidence_overlap_unknown``).

    Raises:
        JointLawPropagationRefusal: the first failed check, with a registered code, a
            ``joint_law_propagation.*`` detail and the dependence information that would support
            the request.
    """
    if operation not in _OPERATIONS:
        raise CausalValueError(f"operation must be one of {_OPERATIONS}")
    if dependence not in _DEPENDENCES:
        raise CausalValueError(f"dependence must be one of {_DEPENDENCES}")
    if reading not in _READINGS:
        raise CausalValueError(f"reading must be one of {_READINGS}")
    declared: dict[str, Any] = {"kind": dependence}
    if dependence == "covariance":
        declared["id"] = _text(covariance_id, "covariance_id")
    if dependence == "correlation_range":
        if correlation_range is None or len(correlation_range) != 2:
            raise CausalValueError("correlation_range is a (low, high) pair")
        declared["rho_lo"] = _real(correlation_range[0], "correlation_range[0]")
        declared["rho_hi"] = _real(correlation_range[1], "correlation_range[1]")
    handle = None
    if ledger is not None:
        handle = getattr(ledger, "_handle", None)
        if handle is None:
            raise CausalTypeError("ledger must be an EvidenceLedger")
    wire, refusal = _native.joint_law_check_propagation(
        operation, _checked_inputs(inputs), json.dumps(declared), reading, handle
    )
    _refuse(refusal)
    assert wire is not None
    plan = json.loads(wire)
    return PropagationPlan(
        operation=plan["operation"],
        route=plan["route"],
        alignment_id=plan["alignment_id"],
        covariance_id=plan["covariance_id"],
        shared_evidence=tuple(plan["shared_evidence"]),
        label=plan["label"],
        missing_transport_pieces=tuple(plan["missing_transport_pieces"]),
    )


def check_rows_pairable(inputs: Sequence[PropagationDraws | MomentInput]) -> None:
    """Whether the rows of several inputs may be paired row by row.

    Returns ``None`` when they may. The inputs' own alignment identities decide: draws of one
    joint law share an identity and pair; draws of different artifacts (other draws, another
    snapshot) are refused ``joint_law_propagation.alignment_mismatch``, and independently
    produced marginals or a summary are refused ``joint_law_propagation.index_pairing_refused``.
    A covariance or a declared independence is never a route to row pairing.

    Raises:
        JointLawPropagationRefusal: ``joint_law_required``.
    """
    _refuse(_native.joint_law_check_rows_pairable(_checked_inputs(inputs)))


@outcome_operation
def aligned_functional(
    operation: str,
    x: PropagationDraws,
    y: PropagationDraws,
    functional: str,
    *,
    execution: Execution | None = None,
) -> PropagationClaim:
    """``E[g(X, Y)]`` under the aligned joint law of two artifact columns.

    ``operation`` is ``"product"`` or ``"nonlinear_functional"``. The rows are paired only
    when both columns carry one alignment identity.

    Raises:
        JointLawPropagationRefusal: ``joint_law_required`` for independent marginals or columns
            of different artifacts.
    """
    if functional not in _FUNCTIONALS:
        raise CausalValueError(f"functional must be one of {_FUNCTIONALS}")
    xa, xc = x._require_artifact()
    ya, yc = y._require_artifact()
    return _claim(
        _native.joint_law_aligned_functional(
            operation, x.id, xa, xc, y.id, ya, yc, functional, _cancel(execution)
        )
    )


@outcome_operation
def aligned_linear(
    operation: str,
    coefficients: tuple[float, float],
    x: PropagationDraws,
    y: PropagationDraws,
    *,
    execution: Execution | None = None,
) -> PropagationClaim:
    """Mean and variance of ``a X + b Y`` from two aligned artifact columns."""
    a = _real(coefficients[0], "coefficients[0]")
    b = _real(coefficients[1], "coefficients[1]")
    xa, xc = x._require_artifact()
    ya, yc = y._require_artifact()
    return _claim(
        _native.joint_law_aligned_linear(
            operation, a, b, x.id, xa, xc, y.id, ya, yc, _cancel(execution)
        )
    )


@outcome_operation
def linear_moments(
    operation: str,
    inputs: Sequence[MomentInput],
    coefficients: Sequence[float],
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    dependence: str,
    covariance_id: str | None = None,
    execution: Execution | None = None,
) -> PropagationClaim:
    """Mean and variance of ``sum a_i X_i`` from means and a covariance (delta method, exact).

    ``dependence`` is ``"covariance"`` (the covariance includes the cross terms; name it with
    ``covariance_id``), ``"declared_independent"`` (distinct evidence only) or ``"unknown"``
    (always refused with ``sampling_dependence_unknown``).
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    k = len(inputs)
    if matrix.shape != (k, k):
        raise CausalValueError("covariance must be a k by k matrix for k inputs")
    return _claim(
        _native.joint_law_linear_moments(
            operation,
            json.dumps([item._wire() for item in inputs]),
            _floats(list(coefficients), "coefficients"),
            [float(v) for v in matrix.ravel()],
            dependence,
            covariance_id,
            _cancel(execution),
        )
    )


@outcome_operation
def mixture(
    components: Sequence[MixtureComponent],
    *,
    weights_declared: bool = True,
    execution: Execution | None = None,
) -> PropagationClaim:
    """Mean and variance of a finite mixture with known weights (law of total variance).

    ``weights_declared=False`` states that the weights are not declared; the rule refuses
    (``joint_law_propagation.mixture_weights_unknown``).
    """
    items = [c._wire() for c in components]
    if weights_declared:
        return _claim(_native.joint_law_mixture(json.dumps(items), _cancel(execution)))
    return _claim(
        _native.joint_law_mixture_undeclared(
            json.dumps([i["id"] for i in items]),
            [i["weight"] for i in items],
            [i["mean"] for i in items],
            [i["variance"] for i in items],
        )
    )


@outcome_operation
def dependence_envelope(
    operation: str,
    query: EnvelopeQuery,
    x: PropagationDraws,
    y: PropagationDraws,
    *,
    execution: Execution | None = None,
) -> PropagationClaim:
    """The structural envelope of a functional of two unpaired marginals.

    The result is labelled ``structural_envelope``: the extremes over every coupling of the two
    marginals (Frechet-Hoeffding). It is never a point, an interval or a posterior.
    """
    return _claim(
        _native.joint_law_envelope(
            operation,
            query._json(),
            x.id,
            x.evidence_id,
            [float(v) for v in x.values],
            y.id,
            y.evidence_id,
            [float(v) for v in y.values],
            _cancel(execution),
        )
    )


@outcome_operation
def correlation_range(
    operation: str,
    x: tuple[str, str],
    y: tuple[str, str],
    *,
    x_mean: float,
    x_sd: float,
    y_mean: float,
    y_sd: float,
    rho_lo: float,
    rho_hi: float,
    a: float = 1.0,
    b: float = 1.0,
    execution: Execution | None = None,
) -> PropagationClaim:
    """Ranges of ``E[XY]``, ``Cov`` and ``Var(aX + bY)`` over a declared correlation range.

    ``x`` and ``y`` are ``(input id, evidence id)``. Labelled ``assumption_range``.
    """
    return _claim(
        _native.joint_law_correlation_range(
            operation,
            x[0],
            x[1],
            y[0],
            y[1],
            _real(x_mean, "x_mean"),
            _real(x_sd, "x_sd"),
            _real(y_mean, "y_mean"),
            _real(y_sd, "y_sd"),
            _real(a, "a"),
            _real(b, "b"),
            _real(rho_lo, "rho_lo"),
            _real(rho_hi, "rho_hi"),
            _cancel(execution),
        )
    )


@outcome_operation
def spread(
    operation: str,
    x: PropagationDraws,
    y: PropagationDraws,
    functional: str,
    *,
    execution: Execution | None = None,
) -> PropagationClaim:
    """The spread of ``g(X, Y)`` under the aligned joint law of two artifact columns.

    Carries ``mean``, ``variance`` (sample, divisor ``n - 1``) and ``mean_standard_error`` of the
    composed rows, labelled ``uncalibrated_spread``: no coverage statement.
    """
    if functional not in _FUNCTIONALS:
        raise CausalValueError(f"functional must be one of {_FUNCTIONALS}")
    xa, xc = x._require_artifact()
    ya, yc = y._require_artifact()
    return _claim(
        _native.joint_law_spread(
            operation, x.id, xa, xc, y.id, ya, yc, functional, _cancel(execution)
        )
    )


@outcome_operation
def aligned_linear_spread(
    operation: str,
    coefficients: tuple[float, float],
    x: PropagationDraws,
    y: PropagationDraws,
    *,
    execution: Execution | None = None,
) -> PropagationClaim:
    """The variance (and covariance) of ``a X + b Y`` from aligned columns, as a spread."""
    a = _real(coefficients[0], "coefficients[0]")
    b = _real(coefficients[1], "coefficients[1]")
    xa, xc = x._require_artifact()
    ya, yc = y._require_artifact()
    return _claim(
        _native.joint_law_aligned_linear_spread(
            operation, a, b, x.id, xa, xc, y.id, ya, yc, _cancel(execution)
        )
    )


@outcome_operation
def linear_moments_spread(
    operation: str,
    inputs: Sequence[MomentInput],
    coefficients: Sequence[float],
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    dependence: str,
    covariance_id: str | None = None,
    execution: Execution | None = None,
) -> PropagationClaim:
    """The variance of ``sum a_i X_i`` under a covariance, as a spread (``uncalibrated_spread``)."""
    matrix = np.asarray(covariance, dtype=np.float64)
    k = len(inputs)
    if matrix.shape != (k, k):
        raise CausalValueError("covariance must be a k by k matrix for k inputs")
    return _claim(
        _native.joint_law_linear_moments_spread(
            operation,
            json.dumps([item._wire() for item in inputs]),
            _floats(list(coefficients), "coefficients"),
            [float(v) for v in matrix.ravel()],
            dependence,
            covariance_id,
            _cancel(execution),
        )
    )


@outcome_operation
def composed_exceedance(
    operation: str,
    x: PropagationDraws,
    y: PropagationDraws,
    functional: str,
    *,
    threshold: float,
    cutoff: float,
    band_level: float = 0.95,
    execution: Execution | None = None,
) -> PropagationClaim:
    """``P(g(X, Y) > threshold)`` from aligned draws with its Monte Carlo band and a verdict.

    ``verdict`` is ``"act"`` when the probability exceeds ``cutoff`` beyond the band,
    ``"withhold"`` when it is below beyond the band, ``"unresolved"`` when the cutoff is inside
    the band. The verdict is about the Monte Carlo error of the probability only. Values:
    ``probability``, ``standard_error``, ``band``.
    """
    if functional not in _FUNCTIONALS:
        raise CausalValueError(f"functional must be one of {_FUNCTIONALS}")
    xa, xc = x._require_artifact()
    ya, yc = y._require_artifact()
    return _claim(
        _native.joint_law_composed_exceedance(
            operation,
            x.id,
            xa,
            xc,
            y.id,
            ya,
            yc,
            functional,
            _real(threshold, "threshold"),
            _real(cutoff, "cutoff"),
            _real(band_level, "band_level"),
            _cancel(execution),
        )
    )


@outcome_operation
def mixture_exceedance(
    components: Sequence[MixtureDraws], *, threshold: float, execution: Execution | None = None
) -> PropagationClaim:
    """``P(X > threshold)`` of a known-weight mixture of draw sets (components never paired)."""
    items = [c._wire() for c in components]
    return _claim(
        _native.joint_law_mixture_exceedance(
            json.dumps(items), _real(threshold, "threshold"), _cancel(execution)
        )
    )


def transported_spread(
    pieces: Sequence[MomentInput],
    coefficients: Sequence[float],
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    dependence: str = "covariance",
    covariance_id: str | None = None,
    execution: Execution | None = None,
) -> Outcome[PropagationClaim]:
    """The spread of a transported claim from its pieces and their covariance.

    ``pieces`` carry the roles ``source_sampling``, ``target_sampling`` and
    ``transport_functional``; a missing piece is refused
    (``joint_law_propagation.transport_pieces_incomplete``) and so is a declared independence
    (``joint_law_propagation.transported_dependence_required``): the pieces are dependent when
    evidence is shared.
    """
    return linear_moments_spread(
        "transported_composition",
        pieces,
        coefficients,
        covariance,
        dependence=dependence,
        covariance_id=covariance_id,
        execution=execution,
    )


@outcome_operation
def transported_spread_aligned(
    source_sampling: PropagationDraws,
    target_sampling: PropagationDraws,
    transport_functional: PropagationDraws,
    *,
    coefficients: Sequence[float] = (1.0, 1.0, 1.0),
    execution: Execution | None = None,
) -> PropagationClaim:
    """The spread of a transported claim from the aligned draws of its three pieces.

    The three columns must share one alignment identity (columns of one joint law); the
    covariance is the population covariance of the aligned rows.
    """
    sa, sc = source_sampling._require_artifact()
    ta, tc = target_sampling._require_artifact()
    fa, fc = transport_functional._require_artifact()
    return _claim(
        _native.joint_law_transported_spread_aligned(
            source_sampling.id,
            sa,
            sc,
            target_sampling.id,
            ta,
            tc,
            transport_functional.id,
            fa,
            fc,
            _floats(list(coefficients), "coefficients"),
            _cancel(execution),
        )
    )


@outcome_operation
@bind_mean_reading
@artifact_consumer(
    proof="joint_law_propagation.consume_joint_law_propagation_artifact: independent semantic consumer checks declarations, identities and stored result; recomputes the scientific result",
    replayable=True,
    input="data",
)
def consume_joint_law_propagation_artifact(
    data: bytes, *, expected_identity: str | None = None, execution: Execution | None = None
) -> ConsumedJointLawPropagation:
    """Verify an artifact, re-run the rule and replay the result to identical bits.

    Raises:
        JointLawPropagationRefusal: an edited, truncated or foreign artifact, a retained
            identity that differs, or a stored declaration the rule no longer supports.
    """
    data = container_bytes(data)
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CausalTypeError("data must be bytes")
    summary, refusal = _native.joint_law_consume(bytes(data), expected_identity, _cancel(execution))
    _refuse(refusal)
    assert summary is not None
    info = json.loads(summary)
    return ConsumedJointLawPropagation(
        identity=info["identity"],
        operation=info["operation"],
        route=info["route"],
        label=info["label"],
        is_envelope=info["is_envelope"],
        alignment_id=info["alignment_id"],
        n_rows=info["n_rows"],
        values={name: value for name, value in info["values"]},
        verdict=info.get("verdict"),
    )
