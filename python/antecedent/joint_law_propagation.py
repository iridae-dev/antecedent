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

A value under a licensed route is a labelled ``point_only`` value. :meth:`PropagationClaim.interval`
and :meth:`PropagationClaim.posterior` always raise ``cell_not_licensed``: no composed coordinate
has a measured coverage record. :meth:`PropagationClaim.export` writes
``joint_law_propagation_v1``; :func:`consume_joint_law_propagation_artifact` re-runs the rule and
replays the result to identical bits and refuses any edit.

Every rule, number, identity and refusal is computed in Rust.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from .errors import CausalTypeError, CausalValueError
from .joint_distribution import JointDistributionArtifact

__all__ = [
    "ConsumedJointLawPropagation",
    "EnvelopeQuery",
    "JointLawPropagationRefusal",
    "MixtureComponent",
    "MomentInput",
    "PropagationClaim",
    "PropagationDraws",
    "aligned_functional",
    "aligned_linear",
    "consume_joint_law_propagation_artifact",
    "correlation_range",
    "dependence_envelope",
    "linear_moments",
    "mixture",
]

_FUNCTIONALS = ("product", "ratio", "minimum", "maximum", "squared_difference")
_ROLES = ("estimate", "transported_interval", "source_sampling", "target_sampling")


class JointLawPropagationRefusal(StructuredRefusal):
    """A typed refusal of a composition.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code``
    (``joint_law_required``, ``sampling_dependence_unknown``, ``evidence_double_counted``,
    ``cell_not_licensed``, ``invalid_argument``, ``route_not_supported``) and a
    ``joint_law_propagation.*`` ``detail``; ``remedy`` names the dependence information that would
    license the request.
    """


def _refuse(payload: str | None) -> None:
    raise_refusal(payload, JointLawPropagationRefusal)


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not hasattr(value, "__float__"):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


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


@dataclass(frozen=True)
class PropagationDraws:
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


@dataclass(frozen=True)
class PropagationClaim:
    """A composed uncertain quantity: the licensed route, the labelled values and the artifact."""

    identity: str
    operation: str
    route: str
    label: str
    is_envelope: bool
    alignment_id: str | None
    n_rows: int | None
    values: Mapping[str, float]
    _bytes: bytes = field(repr=False, compare=False)

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

    def export(self) -> bytes:
        """The sealed ``joint_law_propagation_v1`` artifact."""
        return self._bytes

    def interval(self) -> None:
        """Always raises ``cell_not_licensed``: a component interval does not cover the composition."""
        _refuse(_native.joint_law_closed_reading("interval"))

    def posterior(self) -> None:
        """Always raises ``cell_not_licensed``: no composed coordinate has measured calibration."""
        _refuse(_native.joint_law_closed_reading("posterior"))


@dataclass(frozen=True)
class ConsumedJointLawPropagation:
    """A replayed artifact; every value equals the stored bits."""

    identity: str
    operation: str
    route: str
    label: str
    is_envelope: bool
    alignment_id: str | None
    n_rows: int | None
    values: Mapping[str, float]


def _claim(result: tuple[tuple[bytes, str] | None, str | None]) -> PropagationClaim:
    parts, refusal = result
    _refuse(refusal)
    assert parts is not None
    return PropagationClaim._from(parts)


def aligned_functional(
    operation: str,
    x: PropagationDraws,
    y: PropagationDraws,
    functional: str,
    *,
    cancel: Any = None,
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
            operation, x.id, xa, xc, y.id, ya, yc, functional, cancel
        )
    )


def aligned_linear(
    operation: str,
    coefficients: tuple[float, float],
    x: PropagationDraws,
    y: PropagationDraws,
    *,
    cancel: Any = None,
) -> PropagationClaim:
    """Mean and variance of ``a X + b Y`` from two aligned artifact columns."""
    a = _real(coefficients[0], "coefficients[0]")
    b = _real(coefficients[1], "coefficients[1]")
    xa, xc = x._require_artifact()
    ya, yc = y._require_artifact()
    return _claim(
        _native.joint_law_aligned_linear(operation, a, b, x.id, xa, xc, y.id, ya, yc, cancel)
    )


def linear_moments(
    operation: str,
    inputs: Sequence[MomentInput],
    coefficients: Sequence[float],
    covariance: Sequence[Sequence[float]] | np.ndarray,
    *,
    dependence: str,
    covariance_id: str | None = None,
    cancel: Any = None,
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
            cancel,
        )
    )


def mixture(
    components: Sequence[MixtureComponent], *, weights_declared: bool = True, cancel: Any = None
) -> PropagationClaim:
    """Mean and variance of a finite mixture with known weights (law of total variance).

    ``weights_declared=False`` states that the weights are not declared; the rule refuses
    (``joint_law_propagation.mixture_weights_unknown``).
    """
    items = [c._wire() for c in components]
    if weights_declared:
        return _claim(_native.joint_law_mixture(json.dumps(items), cancel))
    return _claim(
        _native.joint_law_mixture_undeclared(
            json.dumps([i["id"] for i in items]),
            [i["weight"] for i in items],
            [i["mean"] for i in items],
            [i["variance"] for i in items],
        )
    )


def dependence_envelope(
    operation: str,
    query: EnvelopeQuery,
    x: PropagationDraws,
    y: PropagationDraws,
    *,
    cancel: Any = None,
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
            cancel,
        )
    )


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
    cancel: Any = None,
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
            cancel,
        )
    )


def consume_joint_law_propagation_artifact(
    data: bytes, *, expected_identity: str | None = None, cancel: Any = None
) -> ConsumedJointLawPropagation:
    """Verify an artifact, re-run the rule and replay the result to identical bits.

    Raises:
        JointLawPropagationRefusal: an edited, truncated or foreign artifact, a retained
            identity that differs, or a stored declaration the rule no longer licenses.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CausalTypeError("data must be bytes")
    summary, refusal = _native.joint_law_consume(bytes(data), expected_identity, cancel)
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
    )
