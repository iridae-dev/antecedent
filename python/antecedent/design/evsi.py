"""Decision problems, candidate signals and the durable value-of-information artifact.

A :class:`DesignDecision` names its terminal actions and a utility affine in one scalar
state; a candidate study names the signal it would produce. Each candidate's signal comes
from a :class:`SignalProvider` declaration: a native :class:`GaussianMeanSignal` or
:class:`BinomialSignal`, or an :class:`ExternalSignal` that carries *attested* values (a
predictive likelihood Antecedent updates natively, a posterior computed elsewhere, or
per-branch decision values) and the name of the party attesting them::

    decision = design.DesignDecision(
        contract=contract,                       # a decision.Contract or its identity string
        actions=(ActionUtility("guess0", 1.0, -1.0), ActionUtility("guess1", 0.0, 1.0)),
        prior=design.StatePrior.draws([0.0, 1.0]),
        utility_units="utility",
    )
    ranked = design.rank_designs(
        [Candidate("cand-1", 1, ExternalSignal(...), cost=0.1, cost_unit="utility")],
        decision=decision,
        signal=design.SignalSpec(...),
        cost_map=design.CostMap("utility", "utility", 1.0),
    )
    ranked.candidates[0].net_value               # EVSI minus the study cost in utility units
    data = ranked.export()                       # the durable design_ranking_v1 artifact
    design.consume(data, expected=ranked.expectation())

Rust owns the exact signal request fingerprint, the provider binding, the preposterior
integration, source-overlap and cost-unit checks, the artifact identity, independent
recomputation and every refusal; this module builds declarations and raises each structured
refusal as :class:`DesignRankingRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered ``reason_code`` and a
namespaced ``detail`` (``signal_provider.*``, ``evsi.*`` or ``design_ranking.*``).

An external signal is at most ``externally_attested`` and is never reported as natively
replayed: ``natively_replayed`` is ``True`` only for a native signal with exact integration
whose law the consumer rebuilt and recomputed. A study cost is compared with a value only
through an explicit :class:`CostMap` into the decision's utility unit; without one, value and
cost are reported separately and an incompatible unit refuses. An exact-integration EVSI is a
point value; the calibration of Monte Carlo error coverage and rank guarantees is
``unmeasured`` and no coverage is claimed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .._native import composition_lineage as _composition_lineage
from .._native import consume_design_ranking as _consume
from ..decision import Contract
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..external import LineageLink
from ..joint_distribution import JointDistributionArtifact, ScientificQuantity
from .plans import PLAN_TYPES, DesignPlan

#: Distribution meanings that are a belief about a quantity (usable as a state prior).
_BELIEF_MEANINGS = frozenset(
    {
        "parameter_posterior",
        "causal_functional_posterior",
        "posterior_predictive",
        "interventional_predictive",
    }
)

CALIBRATION: Literal["unmeasured"] = "unmeasured"
#: Identity of the reported ranking within the result's ``lineage``.
RESULT_LINK_ID = "design_ranking_result"
ARTIFACT_KIND = "design_ranking_v1"
TrustLabel = Literal["native_licensed", "externally_attested", "exact_request_verified"]
UpdateMode = Literal["native_update", "external_posterior", "external_decision_values"]
Integration = Literal["exact", "monte_carlo", "externally_computed"]


class DesignRankingRefusal(CausalUnsupportedError):
    """A structured refusal from signal providers, EVSI or the ranking artifact.

    ``reason_code`` and ``remedy`` are inherited; the code is registered. ``stage`` is the
    stage that refused, ``detail`` the namespaced ``family.slot``; ``offending``, ``expected``
    and ``supplied`` carry the compared semantics when there are any.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        text = str(refusal["detail"]) + (f" at {offending}" if offending else "")
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        self.stage: str = refusal.get("stage", "")
        self.detail: str = refusal["detail"]
        self.offending: str | None = offending
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


class SignalProviderRefusal(DesignRankingRefusal):
    """A candidate signal or posterior update was refused (``signal_provider.*``)."""


class CostUnitsRefusal(DesignRankingRefusal):
    """A cost unit is incompatible with the cost mapping or the decision's utility unit."""


class SourceOverlapRefusal(DesignRankingRefusal):
    """A study reuses observations the prior already summarizes."""


def _refusal_type(refusal: Mapping[str, Any]) -> type[DesignRankingRefusal]:
    detail = str(refusal["detail"])
    if refusal["code"] == "design_cost_units_mismatch":
        return CostUnitsRefusal
    if detail.endswith(".source_overlap"):
        return SourceOverlapRefusal
    if detail.split(".", 1)[0] == "signal_provider":
        return SignalProviderRefusal
    return DesignRankingRefusal


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        wire = json.loads(refusal)
        raise _refusal_type(wire)(wire)


def _floats(values: Sequence[float], what: str) -> list[float]:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError) as error:
        raise CausalValueError(f"{what} must be a sequence of numbers") from error


def _matrix(rows: Sequence[Sequence[float]], what: str) -> list[list[float]]:
    return [_floats(row, what) for row in rows]


# -- declarations ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionUtility:
    """One terminal action with utility ``intercept + slope * state``.

    This is the design-ranking declaration of an action's utility; the action itself is
    declared in the decision contract (:class:`antecedent.decision.Action`).
    """

    id: str
    intercept: float
    slope: float = 0.0


@dataclass(frozen=True, slots=True)
class StatePrior:
    """The belief about the scalar decision state: equally weighted draws or a normal.

    Not to be confused with :class:`~antecedent.design.StructurePrior`, the weights over
    causal structures used by the identification ranking.
    """

    kind: Literal["draws", "normal"]
    states: tuple[float, ...] = ()
    mean: float | None = None
    variance: float | None = None

    @classmethod
    def draws(cls, states: Sequence[float]) -> StatePrior:
        """Equally weighted draws; any signal family is supported."""
        return cls("draws", states=tuple(_floats(states, "prior draws")))

    @classmethod
    def normal(cls, mean: float, variance: float) -> StatePrior:
        """Conjugate normal belief; needs a :class:`GaussianMeanSignal` and no constraints."""
        return cls("normal", mean=float(mean), variance=float(variance))

    @classmethod
    def from_distribution(
        cls, distribution: JointDistributionArtifact, coordinate: int | ScientificQuantity
    ) -> StatePrior:
        """Equally weighted draws of one coordinate of a joint distribution artifact.

        Only a distribution that is a *belief* about the state qualifies
        (``parameter_posterior``, ``causal_functional_posterior``, ``posterior_predictive``
        or ``interventional_predictive``): the sampling distribution of an estimator, a
        bootstrap and an empirical outcome are not beliefs and refuse. Unequal weights and
        draws marked unsupported refuse too, because the prior is equally weighted. A mean
        grid (a claim or mean source) has no draws and is not offered here.

        Args:
            coordinate: the column index, or the :class:`ScientificQuantity` identifying it.

        Raises:
            CausalTypeError: ``distribution`` is not a ``JointDistributionArtifact``.
            CausalValueError: the distribution is not a belief, has unequal weights or
                unsupported draws, or the coordinate is not one of its quantities.
        """
        if not isinstance(distribution, JointDistributionArtifact):
            raise CausalTypeError(
                "StatePrior.from_distribution needs a JointDistributionArtifact, not "
                f"{type(distribution).__name__}"
            )
        if distribution.semantic not in _BELIEF_MEANINGS:
            raise CausalValueError(
                f"a {distribution.semantic!r} distribution is not a belief about the state; "
                f"a state prior needs one of {sorted(_BELIEF_MEANINGS)}"
            )
        weights = distribution.weights
        if weights is not None and (max(weights) - min(weights)) > 1e-12 * max(weights):
            raise CausalValueError("a state prior is equally weighted; weighted draws refuse")
        supported = distribution.supported
        if supported is not None and not all(supported):
            raise CausalValueError("draws marked unsupported cannot form a state prior")
        quantities = distribution.quantities
        if isinstance(coordinate, ScientificQuantity):
            if coordinate not in quantities:
                raise CausalValueError("the coordinate is not one of the distribution's quantities")
            index = quantities.index(coordinate)
        elif isinstance(coordinate, int) and not isinstance(coordinate, bool):
            if not 0 <= coordinate < len(quantities):
                raise CausalValueError(
                    f"coordinate index {coordinate} is outside 0..{len(quantities) - 1}"
                )
            index = coordinate
        else:
            raise CausalTypeError("coordinate must be a column index or a ScientificQuantity")
        return cls.draws([float(v) for v in np.asarray(distribution)[:, index]])

    def _wire(self) -> dict[str, Any]:
        if self.kind == "draws":
            return {"kind": "draws", "states": list(self.states)}
        return {"kind": "normal", "mean": self.mean, "variance": self.variance}


def _affine_in_state(action: Any, state: ScientificQuantity) -> tuple[float, float]:
    """``(intercept, slope)`` of an action's utility in the state, or a typed refusal."""

    def refuse(reason: str) -> CausalValueError:
        return CausalValueError(
            f"action {action.id!r} utility is not affine in the state quantity: {reason}"
        )

    def walk(node: Any) -> tuple[float, float]:
        if "const" in node:
            return float(node["const"]), 0.0
        if "input" in node:
            index = node["input"]
            if index >= len(action.inputs):
                raise refuse(f"it reads input {index} but the action declares {len(action.inputs)}")
            if action.inputs[index] != state:
                raise refuse(
                    f"input {index} is {action.inputs[index].variable_id!r} "
                    f"({action.inputs[index].regime_id}), not the state"
                )
            return 0.0, 1.0
        ((op, args),) = node.items()
        if op == "neg":
            a, b = walk(args)
            return -a, -b
        left, right = walk(args[0]), walk(args[1])
        if op == "add":
            return left[0] + right[0], left[1] + right[1]
        if op == "sub":
            return left[0] - right[0], left[1] - right[1]
        if op == "mul":
            if left[1] == 0.0:
                return left[0] * right[0], left[0] * right[1]
            if right[1] == 0.0:
                return right[0] * left[0], right[0] * left[1]
            raise refuse("it multiplies the state by the state")
        if op in ("max", "min") and left[1] == 0.0 and right[1] == 0.0:
            pick = max if op == "max" else min
            return pick(left[0], right[0]), 0.0
        raise refuse(f"it takes a {op} that depends on the state")

    return walk(action.utility._wire_value)


@dataclass(frozen=True, slots=True)
class DesignDecision:
    """The decision problem a study's information is valued for.

    ``contract`` is a :class:`antecedent.decision.Contract` (its ``identity`` and
    ``utility_units`` are used) or a contract identity string, in which case
    ``utility_units`` is required. The terminal action set is the same before and after the
    information; an action-set change is a separate problem.
    """

    contract: Contract | str
    actions: tuple[ActionUtility, ...]
    prior: StatePrior
    utility_units: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract, (Contract, str)):
            raise CausalTypeError("contract must be a decision.Contract or its identity string")
        if not isinstance(self.prior, StatePrior):
            raise CausalTypeError("prior must be a design.StatePrior")
        object.__setattr__(self, "actions", tuple(self.actions))
        if not all(isinstance(action, ActionUtility) for action in self.actions):
            raise CausalTypeError("actions must contain design.ActionUtility values")

    @classmethod
    def from_contract(
        cls,
        contract: Contract,
        *,
        prior: StatePrior,
        state: ScientificQuantity | None = None,
    ) -> DesignDecision:
        """The study-ranking decision of a :class:`antecedent.decision.Contract`.

        Study ranking values information about one scalar state on which every action's
        utility is affine, ``intercept + slope * state``. This reads each action's
        utility expression and extracts exactly that pair, so the declaration cannot drift
        from the contract it names. ``state`` is the one quantity the utilities read; when
        omitted it is the single distinct input quantity across the contract's actions.

        Raises:
            CausalTypeError: ``contract`` is not a ``decision.Contract`` or ``prior`` is not
                a :class:`StatePrior`.
            CausalValueError: the criterion is not ``expected_utility``; the contract has hard
                constraints (study ranking does not carry them); ``state`` cannot be inferred;
                or an action's utility is not affine in the state (the message names the
                action and why: a nonlinear product, a ``max``/``min`` of the state, or an
                input that is not the state quantity).
        """
        if not isinstance(contract, Contract):
            raise CausalTypeError(
                "DesignDecision.from_contract needs a decision.Contract, "
                f"not {type(contract).__name__}"
            )
        if contract.criterion.kind != "posterior_expected_utility":
            raise CausalValueError(
                "study ranking values expected utility; the contract's criterion is "
                f"{contract.criterion.kind!r}"
            )
        if contract.constraints:
            raise CausalValueError(
                "study ranking does not carry hard constraints; the contract declares "
                f"{[c.id for c in contract.constraints]}"
            )
        if state is None:
            distinct = {q for action in contract.actions for q in action.inputs}
            if len(distinct) != 1:
                raise CausalValueError(
                    f"state= is required: the actions read {len(distinct)} distinct input "
                    "quantities, not exactly one"
                )
            (state,) = distinct
        elif not isinstance(state, ScientificQuantity):
            raise CausalTypeError("state must be a ScientificQuantity")
        actions = []
        for action in contract.actions:
            intercept, slope = _affine_in_state(action, state)
            actions.append(ActionUtility(action.id, intercept, slope))
        return cls(contract=contract, actions=tuple(actions), prior=prior)

    def _identity(self) -> str:
        return self.contract if isinstance(self.contract, str) else str(self.contract.identity)

    def _units(self) -> str:
        units = self.utility_units
        if units is None and not isinstance(self.contract, str):
            units = self.contract.utility_units
        if not units:
            raise CausalValueError(
                "utility_units is required when the contract is an identity string"
            )
        return str(units)

    def _wire(self) -> dict[str, Any]:
        if not isinstance(self.contract, str):
            declared = {action.id for action in self.contract.actions}
            if declared != {action.id for action in self.actions}:
                raise CausalValueError("the decision's actions must be the contract's actions")
        return {
            "contract_identity": self._identity(),
            "utility_unit": self._units(),
            "action_ids": [a.id for a in self.actions],
            "intercepts": [float(a.intercept) for a in self.actions],
            "slopes": [float(a.slope) for a in self.actions],
            "prior": self.prior._wire(),
        }

    def export_rollout(
        self,
        source: JointDistributionArtifact,
        state: ScientificQuantity,
        ranking: DesignRankingResult,
        *,
        interpretation: Literal["posterior_state", "interventional_state"] = "posterior_state",
        artifact_id: str = "rollout",
    ) -> RolloutResult:
        """Bind a finite source state law to this full decision and its study ranking.

        Posterior state uses actual posterior draws. Interventional state requires an
        explicitly named ``state`` functional. The source's standing and calibration
        remain descriptive; replay issues no native execution authority.
        """
        from .._native import export_rollout as _export_rollout
        from ..decision import source_digest

        if not isinstance(source, JointDistributionArtifact) or not isinstance(
            state, ScientificQuantity
        ):
            raise CausalTypeError("rollout needs a joint distribution and named ScientificQuantity")
        if not isinstance(ranking, DesignRankingResult):
            raise CausalTypeError("rollout needs an independently consumed DesignRankingResult")
        if self.prior.kind != "draws":
            raise CausalValueError(
                "rollout.finite_draw_state_required", reason_code="route_not_supported"
            )
        states = list(self.prior.states)
        if len(states) > 65_536 or len(self.actions) * len(states) > 4 * 1024 * 1024:
            raise CausalValueError("rollout decision table exceeds its finite bound")
        declared = self._wire()
        decision = {
            "contract_identity": declared["contract_identity"],
            "utility_unit": declared["utility_unit"],
            "action_ids": declared["action_ids"],
            "prior": {"draws": {"states": states}},
            "utility": {
                "table": {
                    "rows": [
                        [float(a.intercept) + float(a.slope) * float(value) for value in states]
                        for a in self.actions
                    ]
                }
            },
            "admissible": [True] * len(self.actions),
        }
        binding = {
            "source": {
                "identity": source.identity._wire(),
                "trust": source.trust,
                "calibration": source.calibration,
            },
            "state": state._wire(),
            "interpretation": interpretation,
            "source_digest": source_digest(source),
            "decision": decision,
            "ranking_identity": ranking.identity,
        }
        report, data = _export_rollout(
            source.export("rollout-source"),
            ranking.export(),
            json.dumps(binding, allow_nan=False),
            artifact_id,
            source.source_evidence.export() if source.source_evidence is not None else None,
        )
        return _rollout_result(report, bytes(data))


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Independently checked finite source-state/decision/ranking binding."""

    identity: str
    source_trust: str
    calibration: str
    state: ScientificQuantity
    decision_contract_identity: str
    ranking_identity: str
    _bytes: bytes = field(repr=False)
    _expectation_json: str = field(repr=False)
    _source_evidence: Mapping[str, Any] = field(repr=False)

    @property
    def source_evidence(self) -> dict[str, Any]:
        """Original resolved source, standing and typed ancestry; other leaves remain explicit."""
        return json.loads(json.dumps(self._source_evidence))

    @property
    def lineage(self) -> tuple[LineageLink, ...]:
        return tuple(
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
            for item in self._source_evidence["lineage"]
        )

    @property
    def ranking(self) -> ConsumedRanking:
        """Independently consumed original ranking behind this verified source handoff."""
        from .._native import rollout_ranking

        return consume(bytes(rollout_ranking(self._bytes, self._expectation_json)))

    @property
    def native_execution_authority(self) -> Literal[False]:
        """Historical numerical replay does not issue native producing authority."""
        return False

    def export(self) -> bytes:
        """The bounded ``rollout_state_v1`` artifact."""
        return self._bytes

    def expectation(self) -> dict[str, Any]:
        """Copy of the full scientific and terminal inputs to retain independently."""
        return dict(json.loads(self._expectation_json))


def _rollout_result(report: str, data: bytes) -> RolloutResult:
    value = json.loads(report)
    binding = value["expectation"]
    return RolloutResult(
        value["identity"],
        binding["source"]["trust"],
        binding["source"]["calibration"],
        ScientificQuantity._from_wire(binding["state"]),
        binding["decision"]["contract_identity"],
        binding["ranking_identity"],
        data,
        json.dumps(binding, allow_nan=False),
        value["source_evidence"],
    )


def consume_rollout(data: bytes, *, expected: Mapping[str, Any]) -> RolloutResult:
    """Replay the original source/ranking under independently retained full inputs."""
    from .._native import consume_rollout as _consume_rollout

    if len(data) > 32 * 1024 * 1024:
        raise CausalValueError("rollout artifact exceeds its byte bound")
    report = _consume_rollout(data, json.dumps(dict(expected), allow_nan=False))
    return _rollout_result(report, data)


@dataclass(frozen=True, slots=True)
class CostMap:
    """Positive linear map from a study-cost unit to the decision's utility unit."""

    cost_unit: str
    utility_unit: str
    utility_per_cost: float

    def _wire(self) -> dict[str, Any]:
        return {
            "cost_unit": self.cost_unit,
            "utility_unit": self.utility_unit,
            "utility_per_cost": float(self.utility_per_cost),
        }


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """The exact scientific request a candidate's signal answers.

    ``state`` and ``observation`` are the coordinates of the decision state and of the
    possible observation; ``evidence_lineage`` names the evidence the signal depends on and
    ``conditional_independence`` the assumption on the observations given the state.
    """

    prior_id: str
    state: ScientificQuantity
    observation: ScientificQuantity
    evidence_lineage: tuple[str, ...]
    conditional_independence: str = "iid_given_state"
    rng_seed: int = 0
    max_sample_size: int | None = None
    max_support: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_lineage", tuple(self.evidence_lineage))

    def _wire(self) -> dict[str, Any]:
        return {
            "prior_id": self.prior_id,
            "state_quantity": self.state._wire(),
            "observation_quantity": self.observation._wire(),
            "rng_seed": int(self.rng_seed),
            "evidence_lineage": list(self.evidence_lineage),
            "conditional_independence": self.conditional_independence,
            "max_sample_size": self.max_sample_size,
            "max_support": self.max_support,
        }


class SignalProvider:
    """Declaration of a candidate-specific predictive observation law and posterior update."""

    __slots__ = ()

    #: Trust the declaration carries; an external provider is never ``native_licensed``.
    trust: TrustLabel = "native_licensed"

    def _wire(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class GaussianMeanSignal(SignalProvider):
    """Native Gaussian sample mean ``ybar ~ N(theta, noise_variance / n)``."""

    noise_variance: float

    def _wire(self) -> dict[str, Any]:
        return {"kind": "gaussian_mean", "noise_variance": float(self.noise_variance)}


@dataclass(frozen=True, slots=True)
class BinomialSignal(SignalProvider):
    """Native binomial: successes in ``n`` Bernoulli trials with success probability ``theta``."""

    def _wire(self) -> dict[str, Any]:
        return {"kind": "binomial"}


@dataclass(frozen=True, slots=True)
class ExternalLaw:
    """The values an external provider attests, one constructor per update mode."""

    mode: Literal["likelihood", "posterior", "decision_values"]
    payload: Mapping[str, Any]

    @classmethod
    def likelihood(
        cls,
        states: Sequence[float],
        statistics: Sequence[float],
        probabilities: Sequence[Sequence[float]],
    ) -> ExternalLaw:
        """Predictive law ``probabilities[y][k] = P(statistic_y | state_k)``; updated natively."""
        return cls(
            "likelihood",
            {
                "states": _floats(states, "states"),
                "statistics": _floats(statistics, "statistics"),
                "probabilities": _matrix(probabilities, "probabilities"),
            },
        )

    @classmethod
    def posterior(
        cls,
        states: Sequence[float],
        statistics: Sequence[float],
        predictive: Sequence[float],
        posterior: Sequence[Sequence[float]],
    ) -> ExternalLaw:
        """Posterior ``posterior[y][k] = P(state_k | statistic_y)`` computed externally."""
        return cls(
            "posterior",
            {
                "states": _floats(states, "states"),
                "statistics": _floats(statistics, "statistics"),
                "predictive": _floats(predictive, "predictive"),
                "posterior": _matrix(posterior, "posterior"),
            },
        )

    @classmethod
    def decision_values(
        cls,
        branch_probabilities: Sequence[float],
        action_ids: Sequence[str],
        values: Sequence[Sequence[float]],
    ) -> ExternalLaw:
        """Posterior expected utility ``values[branch][action]`` computed externally."""
        return cls(
            "decision_values",
            {
                "branch_probabilities": _floats(branch_probabilities, "branch_probabilities"),
                "action_ids": [str(a) for a in action_ids],
                "values": _matrix(values, "values"),
            },
        )

    def _wire(self) -> dict[str, Any]:
        return {"mode": self.mode, **self.payload}


@dataclass(frozen=True, slots=True)
class ExternalSignal(SignalProvider):
    """An external provider's attested candidate signal and update.

    The exact request fingerprint is bound by Antecedent to the request evaluated, never
    supplied by the provider. ``attested_candidate_id``, ``attested_prior_id`` and
    ``attested_sample_size`` state what the supplier's object was produced for (default: the
    candidate's own); a different candidate, prior or sample size refuses.
    """

    provider_id: str
    object_id: str
    version_id: str
    snapshot_id: str
    attestor: str
    law: ExternalLaw
    attested_candidate_id: str | None = None
    attested_prior_id: str | None = None
    attested_sample_size: int | None = None

    @property
    def trust(self) -> TrustLabel:  # type: ignore[override]
        """Never ``native_licensed``; the supplier's assertion is not verified by Antecedent."""
        return "externally_attested"

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "external",
            "provider_id": self.provider_id,
            "object_id": self.object_id,
            "version_id": self.version_id,
            "snapshot_id": self.snapshot_id,
            "attestor": self.attestor,
            "law": self.law._wire(),
            "attested_candidate_id": self.attested_candidate_id,
            "attested_prior_id": self.attested_prior_id,
            "attested_sample_size": self.attested_sample_size,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate study: what it collects, what it costs and which signal it produces.

    ``cost`` is in ``cost_unit``; it is compared with the EVSI only through a
    :class:`CostMap`. ``reused_observations`` names existing observations the study would
    reuse; any shared with the prior's observations refuses. ``plan`` optionally ties the
    study to a structural plan (:class:`~antecedent.design.Measurement` and the like): with a
    ``prior=`` structure prior, :func:`~antecedent.design.rank_designs` reports the plan's
    identification probability as a gate on the value ranking.
    """

    id: str
    sample_size: int
    provider: SignalProvider
    cost: float = 0.0
    cost_unit: str = "utility"
    signal: SignalSpec | None = None
    reused_observations: tuple[str, ...] = ()
    plan: DesignPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reused_observations", tuple(self.reused_observations))
        if self.plan is not None and not isinstance(self.plan, PLAN_TYPES):
            raise CausalTypeError(
                "plan must be a design.Measurement, Experiment, Environment or Sampling"
            )


@dataclass(frozen=True, slots=True)
class MonteCarlo:
    """Monte Carlo configuration, used when a value or probability has no exact integral."""

    min_batches: int = 4
    max_batches: int = 64
    batch_size: int = 8
    rank_uncertainty_threshold: float = 0.05

    def _wire(self) -> dict[str, Any]:
        return {
            "min_batches": self.min_batches,
            "max_batches": self.max_batches,
            "batch_size": self.batch_size,
            "rank_uncertainty_threshold": float(self.rank_uncertainty_threshold),
        }


# -- results ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntegrationReport:
    """How one candidate's EVSI integral was evaluated and its error."""

    method: Integration
    replicates: int
    stderr: float
    ess: float | None
    converged: bool
    early_stopped: bool


@dataclass(frozen=True, slots=True)
class SourceOverlapDiagnostics:
    """Observations compared between the prior and the study; empty overlap for any result."""

    observations_checked: int
    overlapping: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Exact external provider object identity retained on a receipt."""

    provider_id: str
    object_id: str
    version_id: str
    snapshot_id: str
    request_id: str


@dataclass(frozen=True, slots=True)
class CandidateValue:
    """One candidate's value of information with its error, trust and diagnostics."""

    id: str
    rank: int
    evsi: float
    evpi: float
    sample_size: int
    study_cost: float
    cost_unit: str
    study_cost_utility: float | None
    net_value: float | None
    integration: IntegrationReport
    rank_uncertain: bool
    provider_trust: TrustLabel
    update_mode: UpdateMode
    attestor: str | None
    provider: ProviderIdentity | None
    signal_family: str
    request_fingerprint: str
    signal_identity: str
    source_overlap: SourceOverlapDiagnostics
    assumptions: tuple[str, ...]
    replay: str
    natively_replayed: bool
    trust_limit: str

    @property
    def score(self) -> float:
        """The ranking score: the net value under a cost map, otherwise the EVSI."""
        return self.net_value if self.net_value is not None else self.evsi

    @property
    def claim(self) -> str:
        """``point_only`` for an exact or externally computed value; no coverage is claimed."""
        return "monte_carlo_estimate" if self.integration.method == "monte_carlo" else "point_only"


@dataclass(frozen=True, slots=True)
class DesignSearchReceipt:
    """How much of the catalog was evaluated; a truncated search never claims the rest."""

    supplied: int
    evaluated: int
    truncated: bool
    unevaluated_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Expectation:
    """Identities a consumer retained independently of the artifact bytes."""

    artifact_identity: str | None = None
    decision_contract_identity: str | None = None
    signal_identities: Mapping[str, str] | None = None
    source_digests: Sequence[str] | None = None
    cost_map: CostMap | None = None
    no_cost_map: bool = False

    def _wire(self) -> dict[str, Any]:
        return {
            "artifact_identity": self.artifact_identity,
            "decision_contract_identity": self.decision_contract_identity,
            "signal_identities": dict(self.signal_identities)
            if self.signal_identities is not None
            else None,
            "source_digests": list(self.source_digests)
            if self.source_digests is not None
            else None,
            "cost_mapping": self.cost_map._wire() if self.cost_map is not None else None,
            "assert_no_cost_mapping": self.no_cost_map,
        }


def _candidate_from_wire(c: Mapping[str, Any], replay: Mapping[str, Any]) -> CandidateValue:
    provider = c["provider"]
    return CandidateValue(
        id=c["semantic_id"],
        rank=int(c["rank"]),
        evsi=float(c["evsi"]),
        evpi=float(c["evpi"]),
        sample_size=int(c["sample_size"]),
        study_cost=float(c["cost_amount"]),
        cost_unit=c["cost_unit"],
        study_cost_utility=c["study_cost_utility"],
        net_value=c["net_value"],
        integration=IntegrationReport(
            method=c["integration"],
            replicates=int(c["replicates"]),
            stderr=float(c["stderr"]),
            ess=c["ess"],
            converged=bool(c["converged"]),
            early_stopped=bool(c["early_stopped"]),
        ),
        rank_uncertain=bool(c["rank_uncertain"]),
        provider_trust=c["trust"],
        update_mode=c["update_mode"],
        attestor=c["attestor"],
        provider=ProviderIdentity(**provider) if provider is not None else None,
        signal_family=c["family"],
        request_fingerprint=c["request_fingerprint"],
        signal_identity=c["signal_identity"],
        source_overlap=SourceOverlapDiagnostics(
            observations_checked=int(c["overlap_checked"]),
            overlapping=tuple(c["overlapping"]),
        ),
        assumptions=tuple(c["assumptions"]),
        replay=str(replay["replay"]),
        natively_replayed=bool(replay["natively_replayed"]),
        trust_limit=str(replay["trust_limit"]),
    )


@dataclass(frozen=True, slots=True)
class ConsumedEntry:
    """One ranked candidate as recovered and checked by an independent consumer."""

    id: str
    rank: int
    evsi: float
    evpi: float
    net_value: float | None
    mc_stderr: float
    replicates: int
    integration: Integration
    sample_size: int
    rank_uncertain: bool
    signal_identity: str
    request_fingerprint: str
    update_mode: UpdateMode
    provider_trust: TrustLabel
    replay: str
    natively_replayed: bool
    trust_limit: str


@dataclass(frozen=True, slots=True)
class ConsumedRanking:
    """A consumed ranking. ``natively_replayed`` is ``False`` for every external signal."""

    identity: str
    ranking_identity: str
    basis: Literal["evsi", "net_value"]
    decision_contract_identity: str
    utility_unit: str
    cost_map: CostMap | None
    rng_seed: int
    source_digests: tuple[str, ...]
    entries: tuple[ConsumedEntry, ...]
    search: DesignSearchReceipt
    calibration: Literal["unmeasured"]
    #: Checked derivation chain, retained independently of the original provider.
    lineage: tuple[LineageLink, ...] = ()


def _search(wire: Mapping[str, Any]) -> DesignSearchReceipt:
    return DesignSearchReceipt(
        supplied=int(wire["supplied"]),
        evaluated=int(wire["evaluated"]),
        truncated=bool(wire["truncated"]),
        unevaluated_ids=tuple(wire["unevaluated_ids"]),
    )


def _cost_map(wire: Mapping[str, Any] | None) -> CostMap | None:
    if wire is None:
        return None
    return CostMap(wire["cost_unit"], wire["utility_unit"], float(wire["utility_per_cost"]))


def _links(rows: Any) -> tuple[LineageLink, ...]:
    return tuple(
        LineageLink(
            item["id"],
            item["stage"],
            tuple(item["parents"]),
            item["digest"],
            tuple(item["parent_digests"]),
        )
        for item in json.loads(_composition_lineage(json.dumps(rows)))
    )


def _consumed_from_wire(wire: Mapping[str, Any]) -> ConsumedRanking:
    entries = tuple(
        ConsumedEntry(
            id=e["semantic_id"],
            rank=int(e["rank"]),
            evsi=float(e["evsi"]),
            evpi=float(e["evpi"]),
            net_value=e["net_value"],
            mc_stderr=float(e["mc_stderr"]),
            replicates=int(e["replicates"]),
            integration=e["integration"],
            sample_size=int(e["sample_size"]),
            rank_uncertain=bool(e["rank_uncertain"]),
            signal_identity=e["signal_identity"],
            request_fingerprint=e["request_fingerprint"],
            update_mode=e["update_mode"],
            provider_trust=e["provider_trust"],
            replay=str(e["replay"]),
            natively_replayed=bool(e["natively_replayed"]),
            trust_limit=str(e["trust_limit"]),
        )
        for e in wire["entries"]
    )
    return ConsumedRanking(
        identity=wire["identity"],
        ranking_identity=wire["ranking_identity"],
        basis=wire["basis"],
        decision_contract_identity=wire["decision_contract_identity"],
        utility_unit=wire["utility_unit"],
        cost_map=_cost_map(wire["cost_mapping"]),
        rng_seed=int(wire["rng_seed"]),
        source_digests=tuple(wire["source_digests"]),
        entries=entries,
        search=_search(wire["search"]),
        calibration=wire["calibration"],
        lineage=_links(wire["lineage"]),
    )


def consume(
    data: bytes,
    *,
    expected: Expectation | None = None,
    skip_expectation_check: bool = False,
) -> ConsumedRanking:
    """Consume an exported ranking by independent recomputation.

    A native exact-integration signal has its law rebuilt and its EVSI, EVPI and net value
    recomputed; an external likelihood or posterior has only the arithmetic recomputed from
    the retained attested table; external decision values are only checked for coherence and
    combined; a Monte Carlo value is bound but not re-simulated. Only the first is
    ``natively_replayed``. Corruption, truncation and unknown versions raise
    :class:`~antecedent.errors.CausalSerializationError`.

    ``expected`` carries identities retained independently of the bytes (build one with
    ``result.expectation()`` or :class:`Expectation`); a changed signal, update mode, source
    digest, cost mapping or contract refuses even when the artifact was resealed. Without it
    the bytes are only checked against themselves, so a resealed artifact would be accepted:
    pass ``skip_expectation_check=True`` to say that is intended. Supplying neither, or both,
    raises :class:`~antecedent.errors.CausalValueError`.
    """
    if expected is None and not skip_expectation_check:
        raise CausalValueError(
            "consume needs expected=<design.Expectation> (e.g. result.expectation()) to check "
            "the artifact against identities you retained; pass skip_expectation_check=True "
            "to accept the artifact on its own seal"
        )
    if expected is not None and skip_expectation_check:
        raise CausalValueError("pass expected= or skip_expectation_check=True, not both")
    expectation = expected._wire() if expected is not None else None
    consumed, refusal = _consume(
        bytes(data), json.dumps(expectation, allow_nan=False) if expectation is not None else None
    )
    _raise(refusal)
    assert consumed is not None
    return _consumed_from_wire(json.loads(consumed))


def _request_wire(
    decision: DesignDecision,
    candidates: Sequence[Candidate],
    *,
    signal: SignalSpec | None,
    cost_map: CostMap | None,
    require_net_value: bool,
    prior_observations: Sequence[str],
    source_digests: Sequence[str],
    rng_seed: int,
    mc_error_tolerance: float,
    tie_tolerance: float,
    max_candidates: int,
    monte_carlo: MonteCarlo | None,
) -> dict[str, Any]:
    wires = []
    for candidate in candidates:
        spec = candidate.signal or signal
        if spec is None:
            raise CausalValueError(
                f"candidate {candidate.id!r} has no signal request: pass signal=SignalSpec(...)"
            )
        wires.append(
            {
                "semantic_id": candidate.id,
                "sample_size": int(candidate.sample_size),
                "cost": {"amount": float(candidate.cost), "unit": candidate.cost_unit},
                "signal": spec._wire(),
                "provider": candidate.provider._wire(),
                "reused_observation_ids": list(candidate.reused_observations),
            }
        )
    return {
        "decision": decision._wire(),
        "candidates": wires,
        "cost_map": cost_map._wire() if cost_map is not None else None,
        "require_net_value": bool(require_net_value),
        "prior_observation_ids": list(prior_observations),
        "source_digests": list(source_digests),
        "rng_seed": int(rng_seed),
        "mc_error_tolerance": float(mc_error_tolerance),
        "tie_tolerance": float(tie_tolerance),
        "max_candidates": int(max_candidates),
        "monte_carlo": monte_carlo._wire() if monte_carlo is not None else None,
    }


__all__ = [
    "RolloutResult",
    "consume_rollout",
    "ARTIFACT_KIND",
    "CALIBRATION",
    "RESULT_LINK_ID",
    "ActionUtility",
    "BinomialSignal",
    "Candidate",
    "CandidateValue",
    "ConsumedEntry",
    "ConsumedRanking",
    "CostMap",
    "CostUnitsRefusal",
    "DesignDecision",
    "DesignRankingRefusal",
    "DesignSearchReceipt",
    "Expectation",
    "ExternalLaw",
    "ExternalSignal",
    "GaussianMeanSignal",
    "IntegrationReport",
    "MonteCarlo",
    "ProviderIdentity",
    "SignalProvider",
    "SignalProviderRefusal",
    "SignalSpec",
    "SourceOverlapDiagnostics",
    "SourceOverlapRefusal",
    "StatePrior",
    "consume",
]
