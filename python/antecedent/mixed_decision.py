"""One decision from a cluster trial, a temporal period effect, a cost model and an uncertain bridge.

The decision rests on four routes the library fits itself: a cluster-randomized trial, a checked
two-step temporal history, a conjugate Gaussian cost posterior and an affine endpoint-to-margin
bridge with declared mapping uncertainty and slope-source covariance. The three estimates are
combined in one linear combination whose mean and variance follow the declared dependence of the
three routes; the action is the expected-utility comparison of acting and declining on that mean::

    from antecedent import mixed_decision as md

    workflow = md.run_mixed_decision(md.Dependence.independent())
    decision = workflow.decision()
    decision.mean, decision.variance, decision.evaluation.verdict

* A new utility recomputes the decision and refits no trial, mechanism or posterior
  (:meth:`MixedDecisionWorkflow.utility_only`); a new cost prior refits only the cost posterior
  (:meth:`MixedDecisionWorkflow.prior_change`); a new observation period refits only the
  temporal mechanisms (:meth:`MixedDecisionWorkflow.period_change`). Each reports the work it did.
* The decision is authorised by the current standing of its sources, held in an evidence ledger.
  :meth:`MixedDecisionWorkflow.withdraw_cluster` retires the trial: the margin and the decision
  are withdrawn, the temporal and cost claims stand, and a decision exported earlier no longer
  loads. :func:`replay_decision` replays the numbers of an exported decision without granting any
  current authority.
* Refusals are :class:`MixedDecisionRefusal` (a
  :class:`~antecedent.errors.CausalUnsupportedError`): a withdrawn or superseded source, an
  artifact whose identity or version does not match, a covariance that is not lawful. A corrupt
  artifact raises :class:`~antecedent.errors.CausalSerializationError`.

Every fit, the combination, the decision, the artifact and its independent consumer are Rust's.
The bounds on the combination are not uncertainty intervals: nothing here reports a coverage
statement or a calibration record.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import _native
from ._refusal import StructuredRefusal, raise_refusal
from ._scientific import ScientificResult
from .composition_verification import (
    CompositionVerification,
    consume_composition_verification_artifact,
)
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "ActionValue",
    "Dependence",
    "DecisionEvaluation",
    "DecisionExport",
    "Gaussian",
    "JointDecision",
    "LoadedDecision",
    "MixedDecisionRefusal",
    "MixedDecisionWorkflow",
    "PeriodReplay",
    "Policy",
    "PriorReplay",
    "RoutesSummary",
    "SharedParameterModel",
    "UtilityReplay",
    "replay_decision",
    "run_mixed_decision",
]


class MixedDecisionRefusal(StructuredRefusal):
    """A typed refusal of the mixed-evidence decision (registered ``reason_code``, ``remedy``)."""


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CausalValueError(f"{what} must be a finite number")
    return float(value)


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class Gaussian:
    """A Gaussian law by its mean and a positive variance."""

    mean: float
    variance: float

    def __post_init__(self) -> None:
        _finite(self.mean, "mean")
        if _finite(self.variance, "variance") <= 0.0:
            raise CausalValueError("variance must be positive")


@dataclass(frozen=True, slots=True)
class SharedParameterModel:
    """Two modules that share one parameter ``theta``.

    The first observes ``y1 = theta + e1`` (the cluster margin); the second observes
    ``y2 = theta + phi + e2`` with a module-specific ``phi`` (the cost contrast). ``pooling_weight``
    in ``[0, 1]`` weights the first prior in the log-linear pool of the two priors of ``theta`` and
    ``feedback_weight`` in ``(0, 1]`` is what a declared feedback policy lets the second module
    return to ``theta``.
    """

    margin_prior: Gaussian
    cost_prior: Gaussian
    pooling_weight: float
    margin_observation: Gaussian
    cost_observation: Gaussian
    specific_prior: Gaussian
    feedback_weight: float

    def __post_init__(self) -> None:
        for what in ("margin_prior", "cost_prior", "margin_observation", "cost_observation"):
            if not isinstance(getattr(self, what), Gaussian):
                raise CausalTypeError(f"{what} must be a Gaussian")
        if not isinstance(self.specific_prior, Gaussian):
            raise CausalTypeError("specific_prior must be a Gaussian")
        _finite(self.pooling_weight, "pooling_weight")
        _finite(self.feedback_weight, "feedback_weight")

    def _wire(self) -> dict[str, Any]:
        def law(value: Gaussian) -> dict[str, float]:
            return {"mean": value.mean, "variance": value.variance}

        return {
            "margin_prior": law(self.margin_prior),
            "cost_prior": law(self.cost_prior),
            "pooling_weight": self.pooling_weight,
            "margin_observation": law(self.margin_observation),
            "cost_observation": law(self.cost_observation),
            "specific_prior": law(self.specific_prior),
            "feedback_weight": self.feedback_weight,
        }


@dataclass(frozen=True, slots=True)
class Policy:
    """How the second module's data is allowed to inform a shared parameter.

    ``coherent_joint`` is one joint model. A ``cut`` keeps the second module from informing the
    shared parameter and a ``feedback`` lets it through at the model's feedback weight; each
    states the ``reconciliation`` of the shared parameter and priors it rests on.
    """

    kind: str = "coherent_joint"
    reconciliation: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("coherent_joint", "cut", "feedback"):
            raise CausalValueError("kind must be coherent_joint, cut or feedback")
        if (self.kind == "coherent_joint") != (self.reconciliation is None):
            raise CausalValueError("only a cut or feedback policy states a reconciliation")
        if self.reconciliation is not None:
            _name(self.reconciliation, "reconciliation")

    @classmethod
    def coherent_joint(cls) -> Policy:
        return cls()

    @classmethod
    def cut(cls, reconciliation: str) -> Policy:
        return cls("cut", reconciliation)

    @classmethod
    def feedback(cls, reconciliation: str) -> Policy:
        return cls("feedback", reconciliation)

    def _wire(self) -> str:
        wire: dict[str, Any] = {"kind": self.kind}
        if self.reconciliation is not None:
            wire["reconciliation"] = self.reconciliation
        return json.dumps(wire)


@dataclass(frozen=True, slots=True)
class Dependence:
    """How the cluster margin, the temporal effect and the cost contrast depend on one another.

    :meth:`independent`: estimated apart, every cross-term zero. :meth:`covariance`: a declared
    joint covariance (the diagonal stays what each route published; these are the off-diagonal
    terms) that must be symmetric and positive semidefinite. :meth:`correlated`: as a covariance,
    with the margin-cost term given as a correlation of the published standard deviations.
    """

    kind: str
    id: str | None = None
    margin_temporal: float = 0.0
    margin_cost: float = 0.0
    margin_cost_correlation: float = 0.0
    temporal_cost: float = 0.0
    model: SharedParameterModel | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("independent", "covariance", "correlated", "shared_parameter"):
            raise CausalValueError(
                "kind must be independent, covariance, correlated or shared_parameter"
            )
        if (self.kind == "shared_parameter") != (self.model is not None):
            raise CausalValueError("only a shared_parameter dependence declares a model")
        if self.model is not None and not isinstance(self.model, SharedParameterModel):
            raise CausalTypeError("model must be a SharedParameterModel")
        if self.kind != "independent":
            _name(self.id, "id")
        for what in (
            "margin_temporal",
            "margin_cost",
            "margin_cost_correlation",
            "temporal_cost",
        ):
            _finite(getattr(self, what), what)

    @classmethod
    def independent(cls) -> Dependence:
        return cls("independent")

    @classmethod
    def covariance(
        cls,
        id: str,  # noqa: A002
        *,
        margin_temporal: float,
        margin_cost: float,
        temporal_cost: float,
    ) -> Dependence:
        return cls(
            "covariance",
            id,
            margin_temporal=margin_temporal,
            margin_cost=margin_cost,
            temporal_cost=temporal_cost,
        )

    @classmethod
    def correlated(
        cls,
        id: str,  # noqa: A002
        *,
        margin_temporal: float,
        margin_cost_correlation: float,
        temporal_cost: float,
    ) -> Dependence:
        return cls(
            "correlated",
            id,
            margin_temporal=margin_temporal,
            margin_cost_correlation=margin_cost_correlation,
            temporal_cost=temporal_cost,
        )

    @classmethod
    def shared_parameter(
        cls,
        id: str,  # noqa: A002
        model: SharedParameterModel,
        *,
        margin_temporal: float = 0.0,
        temporal_cost: float = 0.0,
    ) -> Dependence:
        """The margin and the cost contrast share a parameter: their correlation is the posterior
        correlation of the shared parameter and the contrast under the run's :class:`Policy`."""
        return cls(
            "shared_parameter",
            id,
            margin_temporal=margin_temporal,
            temporal_cost=temporal_cost,
            model=model,
        )

    def _wire(self) -> str:
        wire: dict[str, Any] = {"kind": self.kind}
        if self.kind == "shared_parameter":
            assert self.model is not None
            wire.update(
                id=self.id,
                model=self.model._wire(),
                margin_temporal=self.margin_temporal,
                temporal_cost=self.temporal_cost,
            )
        elif self.kind == "covariance":
            wire.update(
                id=self.id,
                margin_temporal=self.margin_temporal,
                margin_cost=self.margin_cost,
                temporal_cost=self.temporal_cost,
            )
        elif self.kind == "correlated":
            wire.update(
                id=self.id,
                margin_temporal=self.margin_temporal,
                margin_cost_correlation=self.margin_cost_correlation,
                temporal_cost=self.temporal_cost,
            )
        return json.dumps(wire, allow_nan=False)


@dataclass(frozen=True, slots=True)
class ActionValue:
    """One action's admissibility and expected utility."""

    id: str
    admissible: bool
    expected_utility: float


@dataclass(repr=False, frozen=True, slots=True)
class DecisionEvaluation(ScientificResult):
    """The expected-utility comparison of acting and declining at the combined mean.

    ``verdict`` is ``uniquely_optimal`` (``chosen`` holds the action), ``indistinguishable`` or
    ``no_admissible_action``. ``evpi`` is the value of perfect information over the admissible
    actions when any is admissible.
    """

    contract_identity: str
    criterion_label: str
    verdict: str
    chosen: tuple[str, ...]
    evpi: float | None
    actions: tuple[ActionValue, ...]


def _evaluation(wire: Mapping[str, Any]) -> DecisionEvaluation:
    return DecisionEvaluation(
        contract_identity=str(wire["contract_identity"]),
        criterion_label=str(wire["criterion_label"]),
        verdict=str(wire["verdict"]),
        chosen=tuple(wire["chosen"]),
        evpi=None if wire["evpi"] is None else float(wire["evpi"]),
        actions=tuple(
            ActionValue(str(a["id"]), bool(a["admissible"]), float(a["expected_utility"]))
            for a in wire["actions"]
        ),
    )


@dataclass(repr=False, frozen=True, slots=True)
class JointDecision(ScientificResult):
    """The combined decision: the linear combination's moments and the decision on its mean.

    ``mean`` and ``variance`` are those of ``benefit * margin + temporal effect - cost``;
    ``margin_mean`` and ``margin_variance`` are the bridge's delta-method moments of the margin;
    ``dependence`` is the recorded dependence (``independent`` or a ``covariance`` with its id).
    """

    mean: float
    variance: float
    identity: str
    bridge_value: float
    margin_mean: float
    margin_variance: float
    temporal_variance: float
    cost_variance: float
    dependence: str
    dependence_id: str | None
    evaluation: DecisionEvaluation


@dataclass(repr=False, frozen=True, slots=True)
class LoadedDecision(ScientificResult):
    """A decision rebuilt by replaying an exported artifact against current source authority."""

    mean: float
    variance: float
    evaluation: DecisionEvaluation


@dataclass(repr=False, frozen=True, slots=True)
class RoutesSummary(ScientificResult):
    """What the three routes estimated."""

    cluster_trials: int
    cluster_effect: float
    cluster_variance: float
    cluster_uncertainty: str
    cluster_design: str
    temporal_effect: float
    temporal_means: int
    cost_mean: float
    cost_draws: int


@dataclass(frozen=True, slots=True)
class DecisionExport:
    """An exported decision: its bytes and the identity to retain independently of them."""

    artifact: bytes
    identity: str


@dataclass(frozen=True, slots=True)
class UtilityReplay:
    """Work a utility change measured: no temporal or cost refit."""

    temporal_factor_builds: int
    temporal_identifications: int
    bayesian_model_fits: int
    bayesian_posterior_draws: int
    bayesian_identifications: int


@dataclass(frozen=True, slots=True)
class PriorReplay:
    """What a prior change did to the cost model."""

    bayesian_model_fits: int
    bayesian_identifications: int
    score_recomputed: bool
    identification_reused: bool


@dataclass(frozen=True, slots=True)
class PeriodReplay:
    """What a period change did to the temporal route."""

    temporal_factor_builds: int
    temporal_identifications: int
    score_recomputed: bool
    identification_reused: bool


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, MixedDecisionRefusal)
    return value


def _decision(wire: Mapping[str, Any]) -> JointDecision:
    margin = wire["margin"]
    dependence = wire["dependence"]
    return JointDecision(
        mean=float(wire["mean"]),
        variance=float(wire["variance"]),
        identity=str(wire["identity"]),
        bridge_value=float(wire["bridge_value"]),
        margin_mean=float(margin["mean"]),
        margin_variance=float(margin["variance"]),
        temporal_variance=float(wire["temporal_variance"]),
        cost_variance=float(wire["cost_variance"]),
        dependence=str(dependence["kind"]),
        dependence_id=dependence["id"],
        evaluation=_evaluation(wire["evaluation"]),
    )


class MixedDecisionWorkflow:
    """The fitted routes, the combined decision and the current authority of its sources.

    Build one with :func:`run_mixed_decision`. Execution controls (seed, threads, memory limit,
    cancellation) are fixed when it is built and apply to every later replay.
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def __repr__(self) -> str:
        return "<MixedDecisionWorkflow>"

    @property
    def routes(self) -> RoutesSummary:
        """What the cluster trial, the temporal route and the cost model estimated."""
        wire = json.loads(self._handle.summary_json())
        return RoutesSummary(
            cluster_trials=int(wire["cluster_trials"]),
            cluster_effect=float(wire["cluster_effect"]),
            cluster_variance=float(wire["cluster_variance"]),
            cluster_uncertainty=str(wire["cluster_uncertainty"]),
            cluster_design=str(wire["cluster_design"]),
            temporal_effect=float(wire["temporal_effect"]),
            temporal_means=int(wire["temporal_means"]),
            cost_mean=float(wire["cost_mean"]),
            cost_draws=int(wire["cost_draws"]),
        )

    def decision(self) -> JointDecision:
        """The current decision.

        Raises:
            MixedDecisionRefusal: a source was withdrawn, superseded or lost its trust.
        """
        return _decision(json.loads(_unwrap(self._handle.decision_json())))

    def export_decision(self) -> DecisionExport:
        """Export the combination bound to the current source authority.

        Raises:
            MixedDecisionRefusal: the source authority is no longer current.
        """
        artifact, identity = _unwrap(self._handle.export_decision())
        return DecisionExport(bytes(artifact), str(identity))

    def export_historical_replay(self) -> DecisionExport:
        """Export the numerical inputs for a replay that grants no current authority."""
        artifact, identity = _unwrap(self._handle.export_historical_replay())
        return DecisionExport(bytes(artifact), str(identity))

    def load(self, artifact: bytes, expected_identity: str | None = None) -> LoadedDecision:
        """Consume an exported decision against this workflow's current source authority.

        The combination is replayed numerically and the decision evaluated again on the replayed
        mean. A decision whose sources were withdrawn or replaced since it was exported is refused.

        Raises:
            MixedDecisionRefusal: stale, retired or mismatched source authority, an identity that
                differs from ``expected_identity``, or an unsupported artifact version.
            CausalSerializationError: corrupt bytes.
        """
        if expected_identity is not None:
            _name(expected_identity, "expected_identity")
        wire = json.loads(_unwrap(self._handle.load(_bytes(artifact), expected_identity)))
        return LoadedDecision(
            mean=float(wire["mean"]),
            variance=float(wire["variance"]),
            evaluation=_evaluation(wire["evaluation"]),
        )

    def utility_only(self, benefit: float) -> UtilityReplay:
        """Recompute the decision at a new benefit per unit; refit no trial, mechanism or posterior.

        Raises:
            CausalValueError: a non-finite benefit.
            MixedDecisionRefusal: the source authority is no longer current.
        """
        wire = json.loads(_unwrap(self._handle.utility_only(_finite(benefit, "benefit"))))
        return UtilityReplay(**{k: int(v) for k, v in wire.items()})

    def prior_change(self, prior_scale: float) -> PriorReplay:
        """Replace the cost prior scale and refit only the cost posterior.

        Raises:
            MixedDecisionRefusal: a scale that is not a finite positive number, or a source
                authority that is no longer current.
        """
        wire = json.loads(_unwrap(self._handle.prior_change(_finite(prior_scale, "prior_scale"))))
        return PriorReplay(
            bayesian_model_fits=int(wire["bayesian_model_fits"]),
            bayesian_identifications=int(wire["bayesian_identifications"]),
            score_recomputed=bool(wire["score_recomputed"]),
            identification_reused=bool(wire["identification_reused"]),
        )

    def period_change(self, start: int, end: int) -> PeriodReplay:
        """Replace the temporal observation period and refit only the temporal mechanisms.

        Raises:
            MixedDecisionRefusal: a period that does not still contain the retained two-step
                histories, or a source authority that is no longer current.
        """
        for what, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CausalValueError(f"{what} must be a non-negative integer")
        wire = json.loads(_unwrap(self._handle.period_change(start, end)))
        return PeriodReplay(
            temporal_factor_builds=int(wire["temporal_factor_builds"]),
            temporal_identifications=int(wire["temporal_identifications"]),
            score_recomputed=bool(wire["score_recomputed"]),
            identification_reused=bool(wire["identification_reused"]),
        )

    def withdraw_cluster(self) -> CompositionVerification:
        """Withdraw the cluster trial and classify every claim.

        The margin and the decision depend on the trial and are withdrawn; the temporal effect and
        the cost model do not and stand. The classification is sealed and consumed again
        independently, so the returned verification is the replayed one. Afterwards the workflow's
        decision is refused and an earlier export no longer loads.
        """
        artifact, identity = _unwrap(self._handle.withdraw_cluster())
        return consume_composition_verification_artifact(bytes(artifact), str(identity))


def _bytes(artifact: object) -> bytes:
    from ._artifact_protocol import container_bytes

    return container_bytes(artifact)


def run_mixed_decision(
    dependence: Dependence | None = None,
    *,
    policy: Policy | None = None,
    seed: int = 1,
    threads: int | None = None,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> MixedDecisionWorkflow:
    """Fit every route and the decision at benefit one under ``dependence`` and ``policy``.

    ``policy`` (default coherent joint) says whether a shared-parameter dependence lets the cost
    data inform the margin; a cut or feedback policy over an independent dependence is the same
    decision for a cut and a refusal for a feedback, which declares no covariance to carry it.

    Raises:
        CausalValueError: malformed controls.
        MixedDecisionRefusal: the declared covariance is not symmetric and positive semidefinite.
    """
    chosen = Dependence.independent() if dependence is None else dependence
    if not isinstance(chosen, Dependence):
        raise CausalTypeError("dependence must be a mixed_decision.Dependence")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise CausalValueError("seed must be a non-negative 64-bit integer")
    if policy is not None and not isinstance(policy, Policy):
        raise CausalTypeError("policy must be a mixed_decision.Policy")
    handle = _unwrap(
        _native.mixed_decision_execute(
            chosen._wire(),
            seed,
            threads=threads,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
            policy_json=None if policy is None else policy._wire(),
        )
    )
    return MixedDecisionWorkflow(handle)


def replay_decision(
    artifact: bytes, expected_identity: str | None = None, *, seed: int = 1
) -> tuple[float, float]:
    """Replay the combined ``(mean, variance)`` of an exported decision without current authority.

    A historical joint-law artifact is replayed through the same path. The numbers are those the
    artifact was sealed with; nothing here says the decision is still licensed.

    Raises:
        MixedDecisionRefusal: an identity that differs from ``expected_identity`` or an
            unsupported artifact version.
        CausalSerializationError: corrupt bytes.
    """
    if expected_identity is not None:
        _name(expected_identity, "expected_identity")
    mean, variance = _unwrap(
        _native.mixed_decision_replay(_bytes(artifact), expected_identity, seed)
    )
    return float(mean), float(variance)
