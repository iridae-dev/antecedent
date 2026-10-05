"""Learned continuous-outcome trial transport with an estimator menu (2.2A cell X4).

One cell: a randomized binary source treatment with known probabilities, a
continuous outcome, complete baseline covariates equal to the certified
standardizers, and an overlap-supported target population. The estimand is
``E_target[E(Y | X, A=1, S=1) - E(Y | X, A=0, S=1)]`` read through a direct or
baseline-standardization certificate. The target is the nonparticipants of one IID
cohort (``nested_cohort``) or a separately sampled representative IID target
(``independent_samples``); the two designs are distinct. Sampling is IID only.

The outcome regressions and the source-membership model are learned through the
``antecedent.learners`` specs and every nuisance is cross-fitted with no
preprocessing outside the folds. The robustness claimed is model double robustness
of the point estimate: it is consistent when either the outcome regressions or the
participation model is. No efficiency, rate or CATE claim is made.

The point estimate and the estimator menu are licensed. The whole-estimator
interval (joint outer refit percentile bootstrap of the cross-fitted estimator) is
not: its route is closed with ``cell_not_licensed`` until its coverage records are
measured, so an estimate reports the interval withheld and never publishes one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .. import _native
from ..errors import CausalTypeError, CausalValueError
from ..learners import Logistic, Ridge, _learner_wire
from ._impl import TrialAipwData, TrialAipwQuery

#: The one supported target of a request.
TARGET = "target_mean_contrast"


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class LearnedContinuousOptions:
    """Learners, folds, overlap thresholds and bootstrap request of one estimate.

    ``min_membership_probability`` is the smallest out-of-fold source-membership
    probability accepted on any row; ``min_treatment_probability`` bounds the known
    randomization probabilities. Both lie in ``(0, 0.5)``; insufficient overlap
    refuses rather than extrapolates. ``bootstrap`` requests interval replicates
    (at most 2000, floor 199): the percentile route failed calibration, so the
    request only changes the reported interval status. Set it to zero and call
    ``PreparedLearnedContinuous.interval()`` for the analytic influence interval.
    """

    outcome: Any = field(default_factory=Ridge)
    membership: Any = field(default_factory=Logistic)
    folds: int = 5
    min_membership_probability: float = 0.05
    min_treatment_probability: float = 0.05
    bootstrap: int = 0
    coverage_level: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.folds, int) or isinstance(self.folds, bool) or self.folds < 2:
            raise CausalValueError("folds must be an integer of at least 2")
        if not isinstance(self.bootstrap, int) or isinstance(self.bootstrap, bool):
            raise CausalTypeError("bootstrap must be an integer")
        if self.bootstrap < 0:
            raise CausalValueError("bootstrap must be non-negative")
        for label in ("min_membership_probability", "min_treatment_probability"):
            value = getattr(self, label)
            if not isinstance(value, (int, float)) or not 0.0 < float(value) < 0.5:
                raise CausalValueError(f"{label} must lie strictly between 0 and 0.5")
        if not isinstance(self.coverage_level, (int, float)) or not (
            0.0 < float(self.coverage_level) < 1.0
        ):
            raise CausalValueError("coverage_level must lie strictly between 0 and 1")

    def _json(self) -> str:
        return json.dumps(
            dict(
                outcome=_learner_wire(self.outcome),
                membership=_learner_wire(self.membership),
                folds=self.folds,
                min_membership_probability=float(self.min_membership_probability),
                min_treatment_probability=float(self.min_treatment_probability),
                bootstrap=self.bootstrap,
                coverage_level=float(self.coverage_level),
            ),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class EstimatorMenuEntry:
    """One estimator: eligibility, requirements, uncertainty status and any refusal."""

    estimator: str
    eligible: bool
    required_laws: tuple[str, ...]
    required_graph_conditions: tuple[str, ...]
    nuisance_tasks: tuple[str, ...]
    support_requirements: tuple[str, ...]
    sampling_design: tuple[str, ...]
    uncertainty_status: str
    #: Fields (by name) that are fixed descriptions, not computed from the graph, query,
    #: learners or options in force. Every other requirement field is derived.
    static_fields: tuple[str, ...]
    refusal: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class EstimatorMenu:
    """The estimators for one graph, query and learner configuration.

    Selection is manual: no entry is ranked or recommended, because no licensed
    comparison criterion exists between the estimators.
    """

    selection: str
    entries: tuple[EstimatorMenuEntry, ...]

    def __getitem__(self, name: str) -> EstimatorMenuEntry:
        for entry in self.entries:
            if entry.estimator == name:
                return entry
        raise KeyError(name)

    @property
    def eligible(self) -> tuple[str, ...]:
        return tuple(entry.estimator for entry in self.entries if entry.eligible)


def _menu(payload: str) -> EstimatorMenu:
    raw = json.loads(payload)
    return EstimatorMenu(
        raw["selection"],
        tuple(
            EstimatorMenuEntry(
                entry["estimator"],
                entry["eligible"],
                tuple(entry["required_laws"]),
                tuple(entry["required_graph_conditions"]),
                tuple(entry["nuisance_tasks"]),
                tuple(entry["support_requirements"]),
                tuple(entry["sampling_design"]),
                entry["uncertainty_status"],
                tuple(entry["static_fields"]),
                _frozen(entry["refusal"]) if entry["refusal"] else None,
            )
            for entry in raw["entries"]
        ),
    )


def estimator_menu(
    query: TrialAipwQuery, *, outcome: Any = None, membership: Any = None
) -> EstimatorMenu:
    """List the estimators for a binary trial-to-target contrast on this graph and query.

    Inspection only: nothing is fitted. Each entry names its eligibility, required
    laws, graph conditions, nuisance tasks, support and sampling design, its
    uncertainty status and, when refused, why. Give ``outcome`` and ``membership``
    learners to make the learned entry provider-specific.
    """
    if not isinstance(query, TrialAipwQuery):
        raise CausalTypeError("estimator_menu requires a TrialAipwQuery")
    learners = None
    if outcome is not None or membership is not None:
        learners = json.dumps(
            dict(
                outcome=_learner_wire(outcome if outcome is not None else Ridge()),
                membership=_learner_wire(membership if membership is not None else Logistic()),
            )
        )
    return _menu(
        _native.learned_continuous_estimator_menu(
            query.graph,
            list(query.diagram.selections),
            query.diagram.source,
            query.diagram.target,
            query.treatment,
            query.outcome,
            learners,
        )
    )


@dataclass(frozen=True, slots=True)
class LearnedContinuousEstimate:
    """The estimate with provenance, diagnostics and interval status.

    ``overlap`` keeps source membership (``selection``) and treatment overlap apart.
    ``uncertainty`` is ``point_only`` when no interval was requested and ``withheld``
    (``cell_not_licensed``, or ``estimator_inference_mismatch`` below the replicate
    floor) for a legacy bootstrap request. ``interval()`` returns an ``available``
    analytic influence interval and its standard error.
    """

    estimate: float
    interval: tuple[float, float] | None
    standard_error: float | None
    uncertainty: Mapping[str, Any]
    overlap: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]
    folds: Mapping[str, Any]
    sampling: str
    certificate: Mapping[str, Any]
    seed: int
    premises_digest: str
    data_digest: str
    #: Digest of the executed result (point, nuisance predictions, provenance, folds).
    evidence_digest: str
    execution_id: str
    variable_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "interval": self.interval,
            "standard_error": self.standard_error,
            "uncertainty": _thaw(self.uncertainty),
            "overlap": _thaw(self.overlap),
            "diagnostics": _thaw(self.diagnostics),
            "provenance": [_thaw(p) for p in self.provenance],
            "folds": _thaw(self.folds),
            "sampling": self.sampling,
            "certificate": _thaw(self.certificate),
            "seed": self.seed,
            "premises_digest": self.premises_digest,
            "data_digest": self.data_digest,
            "evidence_digest": self.evidence_digest,
            "execution_id": self.execution_id,
        }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _estimate(payload: str) -> LearnedContinuousEstimate:
    raw = json.loads(payload)
    return LearnedContinuousEstimate(
        raw["estimate"],
        tuple(raw["interval"]) if raw["interval"] is not None else None,
        raw["standard_error"],
        _frozen(raw["uncertainty"]),
        _frozen(raw["overlap"]),
        _frozen(raw["diagnostics"]),
        tuple(_frozen(p) for p in raw["provenance"]),
        _frozen(raw["folds"]),
        raw["sampling"],
        _frozen(raw["certificate"]),
        raw["seed"],
        raw["premises_digest"],
        raw["data_digest"],
        raw["evidence_digest"],
        raw["execution_id"],
        tuple(raw["variable_names"]),
    )


class PreparedLearnedContinuous:
    """A prepared learned continuous transport.

    The certificate is derived once at preparation; ``estimate`` cross-fits every
    nuisance on the retained rows under the frozen seed and never identifies again.
    """

    __slots__ = ("_last", "_native")

    def __init__(self, native: Any) -> None:
        self._native = native
        self._last: LearnedContinuousEstimate | None = None

    def estimate(self, *, cancel: Any = None) -> LearnedContinuousEstimate:
        """Cross-fit and report the point and any legacy-bootstrap refusal status."""
        self._last = _estimate(self._native.estimate(cancel))
        return self._last

    def refresh(self, data: TrialAipwData, *, cancel: Any = None) -> None:
        """Replace the rows with a compatible snapshot; the certificate is kept.

        A changed feature schema or sampling design needs a new preparation.
        """
        if not isinstance(data, TrialAipwData):
            raise CausalTypeError("refresh requires TrialAipwData")
        self._native.refresh(data, cancel)
        self._last = None

    def interval(self, *, cancel: Any = None) -> LearnedContinuousEstimate:
        """Fit the design-specific analytic influence interval and keep its artifact."""
        self._last = _estimate(self._native.interval(cancel))
        return self._last

    def estimator_menu(self) -> EstimatorMenu:
        """The estimator menu for this prepared graph, query and learners."""
        return _menu(self._native.estimator_menu())

    def export(self) -> bytes:
        """The last estimate as an independently consumable artifact."""
        return bytes(self._native.export())


def prepare_learned_continuous(
    query: TrialAipwQuery,
    data: TrialAipwData,
    *,
    options: LearnedContinuousOptions | None = None,
    target: str = TARGET,
    seed: int = 1,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedLearnedContinuous:
    """Prepare the certified, learner-backed continuous-outcome mean contrast.

    Baseline covariates must equal the certificate's standardizers. ``target`` is the
    population mean contrast; a conditional, heterogeneous or simultaneous target is
    refused (``learned_transport.cate_requested``), as is any sampling design but the
    two declared IID ones (``learned_transport.non_iid_design``).
    """
    if not isinstance(query, TrialAipwQuery) or not isinstance(data, TrialAipwData):
        raise CausalTypeError(
            "prepare_learned_continuous requires TrialAipwQuery and TrialAipwData"
        )
    options = options or LearnedContinuousOptions()
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise CausalValueError("seed must be a non-negative integer")
    if memory_bytes is not None and (not isinstance(memory_bytes, int) or memory_bytes < 0):
        raise CausalValueError("memory_bytes must be a non-negative integer or None")
    native = _native.prepare_learned_continuous(
        query.graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        query.treatment,
        query.outcome,
        data,
        options._json(),
        target,
        seed=seed,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedLearnedContinuous(native)


def consume_learned_continuous(
    artifact: bytes, *, max_rows: int | None = None, max_features: int | None = None
) -> LearnedContinuousEstimate:
    """Independently verify an exported estimate and replay its point.

    The consumer re-derives the certificate from the stored graph and query, recomputes
    the fold assignment, replays the augmented inverse-odds point and held-out
    diagnostics bit for bit, re-applies the declared overlap thresholds and re-derives
    the interval status. It never fits a learner or resamples. The premises digest binds
    graph, query, learner specs, folds, seed, thresholds, limits and variable names; any
    edit is refused with a typed error.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    for label, value in (("max_rows", max_rows), ("max_features", max_features)):
        if value is not None and (not isinstance(value, int) or value < 0 or math.isnan(value)):
            raise CausalValueError(f"{label} must be a non-negative integer or None")
    return _estimate(
        _native.consume_learned_continuous(artifact, max_rows=max_rows, max_features=max_features)
    )


__all__ = [
    "EstimatorMenu",
    "EstimatorMenuEntry",
    "LearnedContinuousEstimate",
    "LearnedContinuousOptions",
    "PreparedLearnedContinuous",
    "consume_learned_continuous",
    "estimator_menu",
    "prepare_learned_continuous",
]
