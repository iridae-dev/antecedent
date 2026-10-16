"""Observation-conditioned scientific queries.

Conditioning on an observation and intervening are different claims, also when both are computed
from the same causal model. On a composed finite model (see :mod:`antecedent.mechanism_composition`)
``P(Y | S = s)`` keeps every association between ``S`` and ``Y``, including the one a common cause
creates, while ``P(Y | do(S = s))`` cuts the arrows into ``S``::

    from antecedent import mechanism_composition as mc
    from antecedent import observation_query as oq

    claim = oq.condition_versus_intervene(composition, "Y", {"S": "1"})
    claim.paired.conditioning        # {"0": 0.22, "1": 0.78}   (hand-computed confounded model)
    claim.paired.intervention        # {"0": 0.4,  "1": 0.6}

What this is, and is not:

* **Exact given a declared model.** A value from mechanisms the analyst wrote down is exact for
  that declaration (``claim.inference_basis == "exact_given_declared_model"``). A value from
  declared Bayes models states whether its numbers were written down or fitted to a sample
  (``basis="sample_fitted"`` -> ``"point_only_sample_fitted"``). No interval and no calibration is
  made: :meth:`ObservationClaim.interval` and :meth:`ObservationClaim.calibrated_probability`
  raise ``cell_not_licensed``.
* **Operation is part of the identity.** Conditioning, intervention and the paired comparison of
  the same variables and levels on the same model have three different identities.
* **A recorded value is not the construct it measures.** :class:`Recorded` values go in
  conditioning slots; an :class:`Assigned` value there is refused as an intervention smuggled into
  a conditioning slot. A :class:`Measured` indicator value used as the construct is refused
  (``observation_query.measured_as_construct``) unless its measurement model
  (:mod:`antecedent.measurement_models`) is declared and applies in the model's population,
  regime, arm and representation version; the construct's posterior is then the Bayes inversion of
  its confusion matrix against the model's own law.
* **A level no evidence reaches is refused**, with the supremum (raising) or infimum (lowering)
  that is reachable stated in ``supplied``.

Every refusal is an :class:`ObservationQueryRefusal` with a registered ``reason_code``, an
``observation_query.*`` ``detail`` and a remedy. Every number, identity and the artifact is
computed in Rust; this module builds the request and presents the result.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import _native
from ._artifact_protocol import ExportedArtifact
from ._freeze import thaw
from ._mean_binding import bind_mean_reading
from ._measured_inference import MeasuredInference
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "Assigned",
    "ConsumedObservationClaim",
    "CountFittedObservation",
    "Hypothetical",
    "Measured",
    "ObservationClaim",
    "ObservationQueryRefusal",
    "PairedLaw",
    "Recorded",
    "ThresholdRow",
    "condition_on",
    "condition_versus_intervene",
    "conjugate_posterior",
    "consume_observation_query_artifact",
    "count_fitted_observation",
    "finite_posterior",
    "intervene_on",
    "required_bayes_factor",
    "required_observations_binary_signal",
    "required_observations_conjugate",
    "threshold_crossing",
]

_BASES = ("declared_model", "sample_fitted")
_DIRECTIONS = ("raise", "lower")


class ObservationQueryRefusal(StructuredRefusal):
    """A typed refusal of an observation query.

    A :class:`~antecedent.errors.CausalUnsupportedError` carrying the first refusal's registered
    ``reason_code``, namespaced ``detail``, ``offending`` subject and ``remedy``. ``refusals``
    holds every refusal; :attr:`details` lists their details.
    """

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        listed = refusal.get("refusals")
        self.refusals: tuple[Mapping[str, Any], ...] = tuple(listed) if listed else (refusal,)
        #: Expected semantics (the requested level of an unreachable-level refusal), if any.
        self.expected: str | None = refusal.get("expected")
        #: Supplied semantics (the reachable extremum of an unreachable-level refusal), if any.
        self.supplied: str | None = refusal.get("supplied")

    @property
    def details(self) -> tuple[str, ...]:
        """The namespaced detail of every refusal."""
        return tuple(str(item["detail"]) for item in self.refusals)


# --------------------------------------------------------------------------- helpers


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, (bool, str)) or not hasattr(value, "__float__"):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CausalTypeError(f"{what} must be a non-negative integer")
    return int(value)


def _choice(value: str, what: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise CausalValueError(f"{what} must be one of {', '.join(allowed)}; got {value!r}")
    return value


def _dumps(value: object) -> str:
    return json.dumps(thaw(value), allow_nan=False, separators=(",", ":"))


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, ObservationQueryRefusal)
    return value


def _assignment(values: Mapping[str, Any], what: str) -> list[list[str]]:
    if not isinstance(values, Mapping) or not values:
        raise CausalTypeError(f"{what} must be a non-empty mapping of variable to level")
    return [[_name(v, f"{what} variable"), str(level)] for v, level in values.items()]


# --------------------------------------------------------------------------- slots


@dataclass(frozen=True, slots=True)
class Recorded:
    """A value of a model variable recorded without error: the only value a slot may hold."""

    variable: str
    level: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "variable": _name(self.variable, "variable"),
            "level": str(self.level),
            "provenance": "recorded",
            "measured": None,
        }


@dataclass(frozen=True, slots=True)
class Assigned:
    """A value assigned by an action. Placing it in a conditioning slot is refused."""

    variable: str
    level: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "variable": _name(self.variable, "variable"),
            "level": str(self.level),
            "provenance": "assigned",
            "measured": None,
        }


@dataclass(frozen=True, slots=True)
class Hypothetical:
    """A what-if value that was never recorded. Refused in a conditioning slot."""

    variable: str
    level: str

    def to_wire(self) -> dict[str, Any]:
        return {
            "variable": _name(self.variable, "variable"),
            "level": str(self.level),
            "provenance": "hypothetical",
            "measured": None,
        }


@dataclass(frozen=True, slots=True)
class Measured:
    """The recorded value of an indicator of a construct (a model variable).

    ``construct`` is the model variable the value is offered for and ``level`` the recorded level
    in its level coding. ``model`` is the :class:`antecedent.measurement_models.MeasurementModel`
    relating the indicator to the construct; ``None`` states that the recorded value is being used
    as the construct, which is refused. ``positive_level`` names the positive class of a binary
    indicator. ``kind`` is the indicator's measurement kind.
    """

    construct: str
    level: str
    indicator_id: str
    model: Any = None
    positive_level: str | None = None
    kind: str = "noisy_misclassified"
    coordinate_system_id: str = "scalar"
    version: int = 1

    def to_wire(self) -> dict[str, Any]:
        return {
            "variable": _name(self.construct, "construct"),
            "level": str(self.level),
            "provenance": "recorded",
            "measured": {
                "indicator_id": _name(self.indicator_id, "indicator_id"),
                "kind": _name(self.kind, "kind"),
                "model": None if self.model is None else self.model.to_wire(),
                "positive_level": self.positive_level,
                "representation": {
                    "coordinate_system_id": _name(
                        self.coordinate_system_id, "coordinate_system_id"
                    ),
                    "version": _count(self.version, "version"),
                },
            },
        }


Slot = Recorded | Assigned | Hypothetical | Measured


def _slots(slots: Sequence[Slot] | Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if slots is None:
        return []
    if isinstance(slots, Mapping):
        return [Recorded(v, str(level)).to_wire() for v, level in slots.items()]
    out = []
    for slot in slots:
        if not isinstance(slot, (Recorded, Assigned, Hypothetical, Measured)):
            raise CausalTypeError("a slot must be Recorded, Assigned, Hypothetical or Measured")
        out.append(slot.to_wire())
    return out


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class PairedLaw:
    """Both laws of one assignment from one model."""

    conditioning: Mapping[str, float]
    intervention: Mapping[str, float]
    max_abs_difference: float
    evidence_probability: float


@dataclass(frozen=True, slots=True)
class ThresholdRow:
    """One observable level of a signal and what it does to the outcome probability."""

    level: str
    probability: float
    conditional: float
    crosses: bool


def _levels(pairs: Any) -> dict[str, float]:
    return {str(level): float(p) for level, p in pairs}


@dataclass(repr=False, frozen=True, slots=True, eq=False)
class ObservationClaim(ScientificResult):
    """One answered query: the value, how to read it, and the identity binding the operation.

    ``operation`` is one of ``conditioning``, ``intervention``, ``conditioning_and_intervention``,
    ``threshold_crossing``, ``posterior_update`` or ``evidence_requirement``. ``artifact`` is the
    sealed ``observation_query_v1`` artifact, replayed by
    :func:`consume_observation_query_artifact`. Only the accessors of the claim's own kind carry a
    value; the others are ``None``.
    """

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    identity: str
    operation: str
    inference_basis: str
    _artifact_bytes: bytes
    _result: Mapping[str, Any] = field(repr=False, compare=False)

    def artifact(self) -> ExportedArtifact[ConsumedObservationClaim]:
        """Bytes and producer metadata; load returns the typed consumed result."""
        from ._artifact_protocol import exported_artifact

        return exported_artifact(self)

    def to_dict(self) -> dict[str, Any]:
        from ._freeze import thaw

        return {
            "identity": self.identity,
            "operation": self.operation,
            "inference_basis": self.inference_basis,
            **thaw(self._result),
        }

    @property
    def kind(self) -> str:
        """The result kind (``conditional``, ``interventional``, ``paired``, ...)."""
        return str(self._result["kind"])

    @property
    def evidence_probability(self) -> float | None:
        """Probability of the observed evidence under the model, when the claim has one."""
        value = self._result.get("evidence_probability", self._result.get("conditioning_evidence"))
        return None if value is None else float(value)

    @property
    def distribution(self) -> dict[str, float] | None:
        """The outcome distribution of a conditioning or interventional claim."""
        if self.kind in ("conditional", "interventional"):
            return _levels(self._result["levels"])
        return None

    @property
    def construct_posterior(self) -> dict[str, float] | None:
        """The construct's posterior given a measured value."""
        posterior = self._result.get("construct_posterior")
        return None if posterior is None else _levels(posterior)

    @property
    def paired(self) -> PairedLaw | None:
        """Conditioning and intervention side by side, for a paired claim."""
        if self.kind != "paired":
            return None
        return PairedLaw(
            conditioning=_levels(self._result["conditioning"]),
            intervention=_levels(self._result["intervention"]),
            max_abs_difference=float(self._result["max_abs_difference"]),
            evidence_probability=float(self._result["conditioning_evidence"]),
        )

    @property
    def threshold_rows(self) -> tuple[ThresholdRow, ...] | None:
        """The observed signal levels of a threshold-crossing claim."""
        if self.kind != "threshold_crossing":
            return None
        return tuple(
            ThresholdRow(
                r["level"], float(r["probability"]), float(r["conditional"]), bool(r["crosses"])
            )
            for r in self._result["rows"]
        )

    @property
    def crossing_probability(self) -> float | None:
        """Total probability of observing a level that crosses the threshold."""
        value = self._result.get("crossing_probability")
        return None if value is None else float(value)

    @property
    def posterior(self) -> dict[str, float] | None:
        """Posterior probabilities of a finite hypothesis space."""
        if self.kind != "finite_posterior":
            return None
        return _levels(self._result["posterior"])

    @property
    def predictive(self) -> dict[str, float] | None:
        """Posterior predictive distribution of the next observation."""
        if self.kind != "finite_posterior":
            return None
        return _levels(self._result["predictive"])

    @property
    def beta_posterior(self) -> tuple[float, float, float] | None:
        """``(alpha, beta, predictive success probability)`` of a conjugate claim."""
        if self.kind != "conjugate_posterior":
            return None
        r = self._result
        return float(r["alpha"]), float(r["beta"]), float(r["predictive_success"])

    @property
    def required_factor(self) -> float | None:
        """The Bayes factor of a required-factor claim."""
        return float(self._result["factor"]) if self.kind == "required_bayes_factor" else None

    @property
    def required_observations(self) -> int | None:
        """The minimal observation count of a required-observations claim."""
        return int(self._result["observations"]) if self.kind == "required_observations" else None

    @property
    def value_at_requirement(self) -> float | None:
        """The probability at the minimal observation count."""
        return float(self._result["value"]) if self.kind == "required_observations" else None

    @property
    def already_satisfied(self) -> bool | None:
        """Whether no evidence was needed."""
        value = self._result.get("already_satisfied")
        return None if value is None else bool(value)

    def interval(self) -> None:
        """An interval for any observation-conditioned value. Always refused.

        Raises:
            ObservationQueryRefusal: ``cell_not_licensed`` with
                ``observation_query.declared_model_has_no_sampling_interval``.
        """
        raise_refusal(_native.observation_query_closed("interval"), ObservationQueryRefusal)

    def calibrated_probability(self) -> None:
        """A calibration claim for a posterior or decision probability. Always refused.

        Raises:
            ObservationQueryRefusal: ``cell_not_licensed`` with
                ``observation_query.calibrated_probability_closed``.
        """
        raise_refusal(
            _native.observation_query_closed("calibrated_probability"), ObservationQueryRefusal
        )


@dataclass(repr=False, frozen=True, slots=True)
class ConsumedObservationClaim(ScientificResult):
    """A consumed artifact: the verified identity, operation, reading and replayed result."""

    _mean_binding: Any = field(default=None, init=False, repr=False, compare=False)

    identity: str
    operation: str
    inference_basis: str
    claim: ObservationClaim


@bind_mean_reading
def _claim(pair: tuple[Any, str | None]) -> ObservationClaim:
    payload, artifact = _unwrap(pair)
    wire = json.loads(payload)
    return ObservationClaim(
        identity=wire["identity"],
        operation=wire["operation"],
        inference_basis=wire["inference_basis"],
        _artifact_bytes=bytes(artifact),
        _result=wire["result"],
    )


def _handle(composition: Any) -> Any:
    handle = getattr(composition, "_handle", None)
    if handle is None or getattr(composition, "family", None) is None:
        raise CausalTypeError("composition must be a PreparedMechanismComposition")
    return handle


# --------------------------------------------------------------------------- composed queries


def condition_on(
    composition: Any,
    outcome: str,
    given: Sequence[Slot] | Mapping[str, Any] | None = None,
    *,
    cancel: Any = None,
) -> ObservationClaim:
    """``P(outcome | recorded values)`` on a composed finite model.

    ``given`` is a mapping of variable to recorded level, or a sequence of :class:`Recorded` and
    :class:`Measured` slots. An :class:`Assigned` or :class:`Hypothetical` slot, and a
    :class:`Measured` slot without an applicable model, are refused.

    Raises:
        ObservationQueryRefusal: the slot refusals above, an unknown variable or level, evidence
            of probability zero, or a linear-Gaussian composition.
    """
    wire = {
        "operation": "conditioning",
        "outcome": _name(outcome, "outcome"),
        "conditioning": _slots(given),
        "interventions": [],
    }
    return _claim(
        _native.observation_query_observe(_handle(composition), _dumps(wire), cancel=cancel)
    )


def intervene_on(
    composition: Any,
    outcome: str,
    interventions: Mapping[str, Any],
    *,
    cancel: Any = None,
) -> ObservationClaim:
    """``P(outcome | do(interventions))`` on a composed finite model.

    Raises:
        ObservationQueryRefusal: an unknown variable or level, or a linear-Gaussian composition.
    """
    wire = {
        "operation": "intervention",
        "outcome": _name(outcome, "outcome"),
        "conditioning": [],
        "interventions": _assignment(interventions, "interventions"),
    }
    return _claim(
        _native.observation_query_observe(_handle(composition), _dumps(wire), cancel=cancel)
    )


def condition_versus_intervene(
    composition: Any,
    outcome: str,
    assignment: Mapping[str, Any],
    *,
    cancel: Any = None,
) -> ObservationClaim:
    """Both laws of one assignment from one model, never substituted for each other."""
    return _claim(
        _native.observation_query_pair(
            _handle(composition),
            _name(outcome, "outcome"),
            _dumps(_assignment(assignment, "assignment")),
            cancel=cancel,
        )
    )


def threshold_crossing(
    composition: Any,
    outcome: str,
    outcome_level: str,
    signal: str,
    *,
    threshold: float,
    direction: str = "raise",
    background: Sequence[Slot] | Mapping[str, Any] | None = None,
    cancel: Any = None,
) -> ObservationClaim:
    """The observed levels of ``signal`` that move ``P(outcome = outcome_level)`` to a level.

    Raises:
        ObservationQueryRefusal: ``evidence_cannot_reach_level`` when no observed level crosses,
            with the most extreme conditional probability attained in ``supplied``.
    """
    wire = {
        "outcome": _name(outcome, "outcome"),
        "outcome_level": str(outcome_level),
        "signal": _name(signal, "signal"),
        "background": _slots(background),
        "threshold": _number(threshold, "threshold"),
        "direction": _choice(direction, "direction", _DIRECTIONS),
    }
    return _claim(
        _native.observation_query_threshold(_handle(composition), _dumps(wire), cancel=cancel)
    )


# --------------------------------------------------------------------------- declared models


def finite_posterior(
    hypotheses: Sequence[str],
    prior: Sequence[float],
    observation_levels: Sequence[str],
    likelihood: Sequence[Sequence[float]],
    observations: Sequence[str],
    *,
    basis: str = "declared_model",
    cancel: Any = None,
) -> ObservationClaim:
    """Posterior of a declared finite hypothesis space after exchangeable observations.

    ``likelihood[h][o]`` is ``P(observation o | hypothesis h)``. ``basis`` states whether the
    prior and likelihood were written down (``declared_model``) or fitted to a sample
    (``sample_fitted``, a labelled point value).
    """
    rows = [[_number(p, "likelihood") for p in row] for row in likelihood]
    wire = {
        "hypotheses": [_name(h, "hypothesis") for h in hypotheses],
        "prior": [_number(p, "prior") for p in prior],
        "observation_levels": [_name(o, "observation level") for o in observation_levels],
        "likelihood": [p for row in rows for p in row],
    }
    return _claim(
        _native.observation_query_finite_posterior(
            _dumps(wire),
            [_name(o, "observation") for o in observations],
            _choice(basis, "basis", _BASES),
            cancel=cancel,
        )
    )


def conjugate_posterior(
    alpha: float,
    beta: float,
    *,
    successes: int,
    failures: int,
    basis: str = "declared_model",
    cancel: Any = None,
) -> ObservationClaim:
    """Beta-Bernoulli posterior and the predictive probability of the next success."""
    return _claim(
        _native.observation_query_conjugate_posterior(
            _number(alpha, "alpha"),
            _number(beta, "beta"),
            _count(successes, "successes"),
            _count(failures, "failures"),
            _choice(basis, "basis", _BASES),
            cancel=cancel,
        )
    )


def required_bayes_factor(
    prior: float, level: float, *, direction: str = "raise", cancel: Any = None
) -> ObservationClaim:
    """The Bayes factor that moves a probability from ``prior`` to ``level``."""
    return _claim(
        _native.observation_query_required_bayes_factor(
            _number(prior, "prior"),
            _number(level, "level"),
            _choice(direction, "direction", _DIRECTIONS),
            cancel=cancel,
        )
    )


def required_observations_binary_signal(
    prior: float,
    level: float,
    *,
    rate_given_hypothesis: float,
    rate_given_complement: float,
    direction: str = "raise",
    max_observations: int = 10_000,
    basis: str = "declared_model",
    cancel: Any = None,
) -> ObservationClaim:
    """The fewest conditionally independent readings of a binary signal that reach ``level``.

    Raises:
        ObservationQueryRefusal: ``evidence_cannot_reach_level`` when the signal does not move the
            probability that way or needs more than ``max_observations``; the reachable extremum
            is in ``supplied``.
    """
    return _claim(
        _native.observation_query_required_observations_binary(
            _number(prior, "prior"),
            _number(level, "level"),
            _choice(direction, "direction", _DIRECTIONS),
            _number(rate_given_hypothesis, "rate_given_hypothesis"),
            _number(rate_given_complement, "rate_given_complement"),
            _count(max_observations, "max_observations"),
            _choice(basis, "basis", _BASES),
            cancel=cancel,
        )
    )


def required_observations_conjugate(
    alpha: float,
    beta: float,
    level: float,
    *,
    direction: str = "raise",
    max_observations: int = 10_000,
    basis: str = "declared_model",
    cancel: Any = None,
) -> ObservationClaim:
    """The fewest consecutive successes (raise) or failures (lower) that reach a predictive level."""
    return _claim(
        _native.observation_query_required_observations_conjugate(
            _number(alpha, "alpha"),
            _number(beta, "beta"),
            _number(level, "level"),
            _choice(direction, "direction", _DIRECTIONS),
            _count(max_observations, "max_observations"),
            _choice(basis, "basis", _BASES),
            cancel=cancel,
        )
    )


# --------------------------------------------------------------------------- consume


@bind_mean_reading
def consume_observation_query_artifact(
    artifact: bytes,
    expected_identity: str | None = None,
    *,
    cancel: Any = None,
) -> ConsumedObservationClaim:
    """Independently consume an ``observation_query_v1`` artifact.

    Recomputes the identity, recompiles the premises under the coherence checker and replays the
    query to identical result bits. ``expected_identity`` is the identity the consumer holds
    independently of the bytes; with it a consistently resealed edit is refused.

    Raises:
        ObservationQueryRefusal: ``observation_query.artifact_refused`` for an identity that is
            not the stored or the retained one or a result that does not replay,
            ``observation_query.artifact_invalid`` for corrupt bytes, or the structured defects of
            premises that no longer compose.
    """
    if not isinstance(artifact, (bytes, bytearray, memoryview)):
        raise CausalTypeError("artifact must be bytes")
    raw = _unwrap(
        _native.observation_query_consume(bytes(artifact), expected_identity, cancel=cancel)
    )
    wire = json.loads(raw)
    claim = ObservationClaim(
        identity=wire["identity"],
        operation=wire["operation"],
        inference_basis=wire["inference_basis"],
        _artifact_bytes=bytes(artifact),
        _result=wire["result"],
    )
    return ConsumedObservationClaim(
        identity=wire["identity"],
        operation=wire["operation"],
        inference_basis=wire["inference_basis"],
        claim=claim,
    )


# ------------------------------------------------------------------ count-fitted tables


def _bits(value: float) -> int:
    return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


@dataclass(repr=False, frozen=True, slots=True)
class CountFittedObservation(ScientificResult):
    """A conditional probability computed from tables fitted to counts.

    The retained per-parent-configuration counts and the row count give every fitted
    conditional an independent Dirichlet posterior. :attr:`point` is the labelled value under
    the posterior-mean tables. :meth:`interval` returns the equal-tailed posterior interval as
    a measured scalar bound to a currently attesting coverage record, or raises the typed
    refusal. A declared model's probability, by contrast, is exact and has no sampling
    interval (:meth:`ObservationClaim.interval`).
    """

    _source: bytes = field(repr=False)

    @property
    def point(self) -> float:
        """The probability under the posterior-mean tables (uniform Dirichlet prior)."""
        value, refusal = _native.count_fitted_observation_point(self._source)
        raise_refusal(refusal, ObservationQueryRefusal)
        if value is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native point value returned neither value nor refusal")
        return float(value)

    def export(self) -> bytes:
        """The retained counts and question (canonical bytes)."""
        return self._source

    def interval(self, *, level: float = 0.95, cancel: Any = None) -> MeasuredInference:
        """The measured posterior interval at ``level`` (0.90 and 0.95 are measured).

        Raises:
            ObservationQueryRefusal: ``cell_not_licensed`` when no currently attesting coverage
                record governs this construction (level, prior, draws, rows, sample size).
        """
        measured, refusal = _native.count_fitted_observation_interval(
            self._source, _number(level, "level"), cancel=cancel
        )
        raise_refusal(refusal, ObservationQueryRefusal)
        if measured is None:  # pragma: no cover - the native contract
            raise CausalValueError("the native interval returned neither result nor refusal")
        return MeasuredInference._from_native(measured)

    def calibrated_probability(
        self, *, level: float = 0.95, cancel: Any = None
    ) -> MeasuredInference:
        """The calibration claim of the posterior probability: the same measured interval."""
        return self.interval(level=level, cancel=cancel)


def count_fitted_observation(
    levels: Mapping[str, Sequence[str]],
    *,
    parents: Mapping[str, Sequence[str]] | None = None,
    counts: Mapping[str, Sequence[int]] | None = None,
    declared: Mapping[str, Sequence[float]] | None = None,
    rows: int,
    target: str,
    target_level: str,
    evidence: Mapping[str, str] | None = None,
    seed: int | None = None,
) -> CountFittedObservation:
    """Retain tables fitted to counts and ask ``P(target = target_level | evidence)``.

    ``levels`` maps each variable, in topological order, to its level names. ``counts[v]`` is
    the flattened count table of a fitted variable (rows are parent configurations with the
    first parent varying slowest, columns its levels); ``declared[v]`` is a fixed conditional
    (for example a measured signal with known sensitivity and specificity), flattened the same
    way. Every fitted variable must have been counted from the same ``rows`` complete rows.
    """
    parents = parents or {}
    counts = counts or {}
    declared = declared or {}
    order = list(levels)
    nodes = []
    for name in order:
        in_counts, in_declared = name in counts, name in declared
        if in_counts == in_declared:
            raise CausalValueError(f"{name}: give exactly one of counts= and declared=")
        nodes.append(
            {
                "name": _name(name, "variable"),
                "levels": [_name(level, "level") for level in levels[name]],
                "parents": [order.index(p) for p in parents.get(name, ())],
                "counts": [_count(c, "count") for c in counts[name]] if in_counts else None,
                "declared_bits": [_bits(p) for p in declared[name]] if in_declared else None,
            }
        )
    source, refusal = _native.count_fitted_observation_build(
        _dumps(nodes),
        _count(rows, "rows"),
        _name(target, "target"),
        _name(target_level, "target_level"),
        _dumps([[k, v] for k, v in (evidence or {}).items()]),
        seed,
    )
    raise_refusal(refusal, ObservationQueryRefusal)
    if source is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native builder returned neither a source nor a refusal")
    return CountFittedObservation(bytes(source))
