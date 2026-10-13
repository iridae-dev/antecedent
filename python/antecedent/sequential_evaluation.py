"""Sequential evaluation of a caller-supplied contingent policy.

A policy is evaluated, not planned: a finite-horizon problem declares discrete latent states,
actions, delayed outcomes, a resource budget, stopping boundaries and a terminal utility, and
every observation enters a Bayes belief update through a typed measurement model::

    from antecedent import sequential_evaluation as se

    model = se.FiniteModel(
        population="plant", regime="operations", period="2026", state_variable="unit_condition",
        states=("sound", "faulty"), initial=(0.5, 0.5), horizon=2,
        actions=(
            se.Action("inspect", "continue", transition=((1, 0), (0, 1)), utility_cost=0.25,
                      channel=alarm_panel),            # a measurement_models.MeasurementModel
            se.Action("ship", "commit", transition=((1, 0), (0, 1)), immediate_utility=(10, -10)),
            se.Action("repair", "commit", transition=((1, 0), (0, 1)), immediate_utility=(6, -4)),
        ),
        terminal_utility=(0.0, 0.0),
    )
    policy = se.Policy.deterministic("inspect_then_act_v1", {
        (): "inspect",
        (("inspect", 1),): "repair",
        (("inspect", 0),): "ship",
    })
    evaluation = se.evaluate(model, policy)
    evaluation.value                      # 1.5, exact given the declared model

What is supplied and what is refused:

* :func:`evaluate` is the sequential g-formula: exact enumeration of the histories the policy
  reaches, under the declared exchangeability, consistency and positivity premises
  (:attr:`PolicyEvaluation.assumptions`). A policy that takes an action in a history the model
  gives no support is refused (``history_support_violated``), as is a reachable history without a
  decision.
* :func:`evaluate_off_policy` weights logged finite trajectories by their logged assignment
  probabilities. It is a point estimate; its standard error is an uncalibrated diagnostic.
* :func:`rollout` simulates on common random streams keyed by ``(seed, rollout, time, role)``;
  :func:`stream_uniform` exposes one stream identity.
* :func:`compare` evaluates named act-now, wait, collect-evidence and choose-after-observation
  policies; :func:`value_of_collecting_evidence` scores an observation through the design ranker.
* :class:`BeliefSession` follows one realized history and records each action and observation
  once in an evidence ledger.
* :func:`seal` / :func:`consume_artifact` export and re-evaluate a ``sequential_evaluation_v1``
  artifact bit for bit.

A reweighted static score table (:class:`ReweightedStaticScores`) and a policy learned from the
data under evaluation (:class:`LearnedFromEvaluationData`) are refused by name
(``route_not_supported``), a value for another population or period is refused naming the missing
selection-diagram and invariance contract (``transport_missing_evidence``), and
:func:`request_interval` always raises ``cell_not_licensed``. Rust owns every number and
refusal; this module builds declarations and raises :class:`SequentialEvaluationRefusal`
(a :class:`~antecedent.errors.CausalUnsupportedError`) with the registered ``reason_code`` and
the ``detail`` slot.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from ._refusal import raise_refusal
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .measurement_models import MeasurementModel

__all__ = [
    "Action",
    "BeliefSession",
    "Comparison",
    "ComparisonReport",
    "ComparisonRow",
    "DecisionReceipt",
    "Delayed",
    "EvaluationArtifact",
    "FiniteModel",
    "HistoryReceipt",
    "LearnedFromEvaluationData",
    "LoggedStep",
    "LoggedTrajectory",
    "ObservationReceipt",
    "OffPolicyEstimate",
    "Policy",
    "PolicyEvaluation",
    "ReweightedStaticScores",
    "RolloutEstimate",
    "SequentialEvaluationRefusal",
    "StoppingBoundary",
    "SupportOverride",
    "ValueOfEvidence",
    "check_request",
    "compare",
    "consume_artifact",
    "evaluate",
    "evaluate_off_policy",
    "request_interval",
    "resolve_policy",
    "rollout",
    "seal",
    "stream_uniform",
    "value_of_collecting_evidence",
]

_History = tuple[tuple[int, int], ...]

_ROLES = ("continue", "commit")
_CENSORING = ("refuse", "forfeit")
_COMPARISON_KINDS = ("act_now", "wait", "collect_evidence", "choose_after_observation")
_STREAM_ROLES = ("initial_state", "transition", "observation", "policy_choice")


class SequentialEvaluationRefusal(StructuredRefusal):
    """A typed refusal of the sequential-evaluation route.

    A :class:`~antecedent.errors.CausalUnsupportedError` with a registered ``reason_code`` and a
    namespaced ``detail`` (the module's own slot for a policy or model refusal, the measurement
    layer's for an observation channel); ``remedy`` names the declaration, contract or evidence
    that would license the request.
    """


# --------------------------------------------------------------------------- validation


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _count(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{what} must be a non-negative integer")
    if value < 0:
        raise CausalValueError(f"{what} must be a non-negative integer")
    return value


def _reals(values: object, what: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{what} must be a sequence of numbers")
    return tuple(_number(v, what) for v in values)


def _dumps(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _unwrap(pair: tuple[Any, str | None]) -> Any:
    value, refusal = pair
    raise_refusal(refusal, SequentialEvaluationRefusal)
    return value


def _result(pair: tuple[str | None, str | None]) -> dict[str, Any]:
    text, refusal = pair
    raise_refusal(refusal, SequentialEvaluationRefusal)
    assert text is not None
    return dict(json.loads(text))


def _history(value: object, what: str) -> _History:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CausalTypeError(f"{what} must be a sequence of (action, observation) pairs")
    steps: list[tuple[int, int]] = []
    for step in value:
        if not isinstance(step, Sequence) or len(step) != 2:
            raise CausalTypeError(f"{what} must be a sequence of (action, observation) pairs")
        steps.append((_count(step[0], what), _count(step[1], what)))
    return tuple(steps)


# --------------------------------------------------------------------------- declarations


@dataclass(frozen=True, slots=True)
class Delayed:
    """An outcome realized ``delay`` decision times after its action, utility by latent state."""

    delay: int
    utility: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "delay", _count(self.delay, "delay"))
        object.__setattr__(self, "utility", _reals(self.utility, "delayed utility"))

    def _wire(self) -> dict[str, Any]:
        return {"delay": self.delay, "utility": list(self.utility)}


@dataclass(frozen=True, slots=True)
class Action:
    """One declared action: ``continue`` (state moves, an observation may be emitted) or ``commit``.

    ``transition`` is a row-stochastic ``states x states`` matrix. ``channel`` is the
    :class:`~antecedent.measurement_models.MeasurementModel` that produces the observation: a
    binary misclassification or categorical confusion model whose construct is the model's state
    variable, non-differential, for the model's own population and regime.
    """

    label: str
    role: str
    transition: tuple[tuple[float, ...], ...]
    immediate_utility: tuple[float, ...] | None = None
    utility_cost: float = 0.0
    resource_use: float = 0.0
    channel: MeasurementModel | None = None
    delayed: Delayed | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _name(self.label, "action label"))
        if self.role not in _ROLES:
            raise CausalValueError(f"action role must be one of {', '.join(_ROLES)}")
        rows = tuple(_reals(row, "transition row") for row in self.transition)
        object.__setattr__(self, "transition", rows)
        if self.immediate_utility is not None:
            object.__setattr__(
                self, "immediate_utility", _reals(self.immediate_utility, "immediate utility")
            )
        object.__setattr__(self, "utility_cost", _number(self.utility_cost, "utility cost"))
        object.__setattr__(self, "resource_use", _number(self.resource_use, "resource use"))
        if self.channel is not None and not isinstance(self.channel, MeasurementModel):
            raise CausalTypeError("channel must be a measurement_models.MeasurementModel")
        if self.delayed is not None and not isinstance(self.delayed, Delayed):
            raise CausalTypeError("delayed must be a Delayed outcome")

    def _wire(self, states: int) -> dict[str, Any]:
        immediate = self.immediate_utility
        return {
            "label": self.label,
            "role": self.role,
            "utility_cost": self.utility_cost,
            "resource_use": self.resource_use,
            "transition": [v for row in self.transition for v in row],
            "channel": None if self.channel is None else self.channel.to_wire(),
            "immediate_utility": list(immediate) if immediate is not None else [0.0] * states,
            "delayed": None if self.delayed is None else self.delayed._wire(),
        }


@dataclass(frozen=True, slots=True)
class StoppingBoundary:
    """Force the commit action ``stop_action`` when the posterior mass on ``states`` crosses a bound."""

    label: str
    states: tuple[int, ...]
    stop_action: int
    upper: float | None = None
    lower: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _name(self.label, "boundary label"))
        object.__setattr__(self, "states", tuple(_count(s, "boundary state") for s in self.states))
        object.__setattr__(self, "stop_action", _count(self.stop_action, "stop action"))
        if self.upper is not None:
            object.__setattr__(self, "upper", _number(self.upper, "upper threshold"))
        if self.lower is not None:
            object.__setattr__(self, "lower", _number(self.lower, "lower threshold"))

    def _wire(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "states": list(self.states),
            "upper": self.upper,
            "lower": self.lower,
            "stop_action": self.stop_action,
        }


@dataclass(frozen=True, slots=True)
class SupportOverride:
    """Which actions are supported in one history (positivity), one flag per action."""

    history: _History
    allowed: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", _history(self.history, "history"))
        object.__setattr__(self, "allowed", tuple(bool(a) for a in self.allowed))

    def _wire(self) -> dict[str, Any]:
        return {"history": [list(s) for s in self.history], "allowed": list(self.allowed)}


@dataclass(frozen=True, slots=True)
class FiniteModel:
    """A finite-horizon dynamic finite model.

    The per-action transition and observation laws are the interventional laws given the
    information set: supplying them is the declared exchangeability premise. ``support_default``
    is one flag per action (all supported when omitted); ``censoring`` is ``refuse`` or
    ``forfeit`` for outcomes due after the horizon.
    """

    population: str
    regime: str
    period: str
    state_variable: str
    states: tuple[str, ...]
    initial: tuple[float, ...]
    horizon: int
    actions: tuple[Action, ...]
    terminal_utility: tuple[float, ...]
    label: str = "sequential_model"
    censoring: str = "refuse"
    resource_budget: float | None = None
    stopping: tuple[StoppingBoundary, ...] = ()
    support_default: tuple[bool, ...] | None = None
    support_overrides: tuple[SupportOverride, ...] = ()

    def __post_init__(self) -> None:
        for field in ("population", "regime", "period", "state_variable", "label"):
            _name(getattr(self, field), field)
        object.__setattr__(self, "states", tuple(_name(s, "state") for s in self.states))
        object.__setattr__(self, "initial", _reals(self.initial, "initial law"))
        object.__setattr__(self, "horizon", _count(self.horizon, "horizon"))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "terminal_utility", _reals(self.terminal_utility, "terminal utility"))
        if any(not isinstance(a, Action) for a in self.actions):
            raise CausalTypeError("actions must be Action declarations")
        if self.censoring not in _CENSORING:
            raise CausalValueError(f"censoring must be one of {', '.join(_CENSORING)}")
        if self.resource_budget is not None:
            object.__setattr__(self, "resource_budget", _number(self.resource_budget, "budget"))
        object.__setattr__(self, "stopping", tuple(self.stopping))
        object.__setattr__(self, "support_overrides", tuple(self.support_overrides))

    def action_index(self, label: str) -> int:
        """The index of the action with this label."""
        for index, action in enumerate(self.actions):
            if action.label == label:
                return index
        raise CausalValueError(f"no action named {label!r}")

    def to_wire(self) -> dict[str, Any]:
        """The declaration in the plain-number JSON form the native layer reads."""
        n = len(self.states)
        default = (
            list(self.support_default)
            if self.support_default is not None
            else [True] * len(self.actions)
        )
        return {
            "label": self.label,
            "population": self.population,
            "regime": self.regime,
            "period": self.period,
            "state_variable": self.state_variable,
            "states": list(self.states),
            "initial": list(self.initial),
            "horizon": self.horizon,
            "actions": [a._wire(n) for a in self.actions],
            "terminal_utility": list(self.terminal_utility),
            "censoring": self.censoring,
            "resource_budget": self.resource_budget,
            "stopping": [b._wire() for b in self.stopping],
            "support_default": default,
            "support_overrides": [o._wire() for o in self.support_overrides],
        }

    @property
    def identity(self) -> str:
        """The model identity (the display label is not part of it); validates the model."""
        return str(_result(_native.sequential_model_check(_dumps(self.to_wire())))["identity"])


def _model_from_wire(wire: Mapping[str, Any]) -> FiniteModel:
    from .measurement_models import _model_from_wire as channel_from_wire

    actions = tuple(
        Action(
            label=a["label"],
            role=a["role"],
            transition=tuple(
                tuple(a["transition"][i : i + len(wire["states"])])
                for i in range(0, len(a["transition"]), len(wire["states"]))
            ),
            immediate_utility=tuple(a["immediate_utility"]),
            utility_cost=a["utility_cost"],
            resource_use=a["resource_use"],
            channel=None if a["channel"] is None else channel_from_wire(a["channel"]),
            delayed=None if a["delayed"] is None else Delayed(**a["delayed"]),
        )
        for a in wire["actions"]
    )
    return FiniteModel(
        label=wire["label"],
        population=wire["population"],
        regime=wire["regime"],
        period=wire["period"],
        state_variable=wire["state_variable"],
        states=tuple(wire["states"]),
        initial=tuple(wire["initial"]),
        horizon=wire["horizon"],
        actions=actions,
        terminal_utility=tuple(wire["terminal_utility"]),
        censoring=wire["censoring"],
        resource_budget=wire["resource_budget"],
        stopping=tuple(
            StoppingBoundary(
                label=b["label"],
                states=tuple(b["states"]),
                stop_action=b["stop_action"],
                upper=b["upper"],
                lower=b["lower"],
            )
            for b in wire["stopping"]
        ),
        support_default=tuple(wire["support_default"]),
        support_overrides=tuple(
            SupportOverride(tuple(tuple(s) for s in o["history"]), tuple(o["allowed"]))
            for o in wire["support_overrides"]
        ),
    )


@dataclass(frozen=True, slots=True)
class Policy:
    """A fixed contingent policy: a table from observed histories to action distributions.

    A history is a tuple of ``(action, observation)`` pairs, actions by index. The table never
    changes during an evaluation, and it carries a version label that every receipt names.
    """

    version: str
    decisions: tuple[tuple[_History, tuple[tuple[int, float], ...]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _name(self.version, "policy version"))

    @staticmethod
    def deterministic(
        version: str, table: Mapping[_History, int | str], model: FiniteModel | None = None
    ) -> Policy:
        """Take one action for certain in each history.

        An action may be an index, or a label when ``model`` is given.
        """
        entries = []
        for history, action in table.items():
            if isinstance(action, str):
                if model is None:
                    raise CausalTypeError("an action label needs the model it names")
                action = model.action_index(action)
            entries.append((_history(history, "history"), ((_count(action, "action"), 1.0),)))
        return Policy(version, tuple(sorted(entries)))

    @staticmethod
    def stochastic(
        version: str, table: Mapping[_History, Mapping[int, float]]
    ) -> Policy:
        """Take actions at random in each history with the given probabilities."""
        entries = []
        for history, distribution in table.items():
            row = tuple(
                sorted((_count(a, "action"), _number(p, "probability")) for a, p in distribution.items())
            )
            entries.append((_history(history, "history"), row))
        return Policy(version, tuple(sorted(entries)))

    def to_wire(self) -> dict[str, Any]:
        """The declaration in the JSON form the native layer reads."""
        return {
            "version": self.version,
            "decisions": [
                {"history": [list(s) for s in history], "actions": [[a, p] for a, p in actions]}
                for history, actions in self.decisions
            ],
        }


def _policy_from_wire(wire: Mapping[str, Any]) -> Policy:
    return Policy(
        wire["version"],
        tuple(
            sorted(
                (
                    tuple((int(s[0]), int(s[1])) for s in d["history"]),
                    tuple((int(a), float(p)) for a, p in d["actions"]),
                )
                for d in wire["decisions"]
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ReweightedStaticScores:
    """A static table of scores with weights: no information sets, so not a sequential policy."""

    scores: tuple[float, ...]
    weights: tuple[float, ...]

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "reweighted_static_scores",
            "scores": list(_reals(self.scores, "scores")),
            "weights": list(_reals(self.weights, "weights")),
        }


@dataclass(frozen=True, slots=True)
class LearnedFromEvaluationData:
    """A policy produced by a learning rule from the data being evaluated: not evaluation."""

    rule: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "learned_from_evaluation_data", "rule": _name(self.rule, "rule")}


def _spec_wire(policy: Policy | ReweightedStaticScores | LearnedFromEvaluationData) -> dict[str, Any]:
    if isinstance(policy, Policy):
        return {"kind": "contingent", "policy": policy.to_wire()}
    if isinstance(policy, (ReweightedStaticScores, LearnedFromEvaluationData)):
        return policy._wire()
    raise CausalTypeError("a policy is a Policy, ReweightedStaticScores or LearnedFromEvaluationData")


def resolve_policy(policy: Policy | ReweightedStaticScores | LearnedFromEvaluationData) -> Policy:
    """The fixed policy to evaluate; a static score table or a learning rule raises its refusal."""
    wire = _result(_native.sequential_resolve_policy(_dumps(_spec_wire(policy))))
    return _policy_from_wire(wire)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class HistoryReceipt:
    """One reachable history: reach probability, belief and the utility it accounts for."""

    history: _History
    time: int
    kind: str
    boundary: int | None
    reach_probability: float
    belief: tuple[float, ...]
    realized_utility: float
    continuation_value: float
    forfeited_outcomes: int


@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    """One decision: the information set, the action distribution and what forced it."""

    history: _History
    time: int
    forced_by: int | None
    actions: tuple[tuple[int, float], ...]
    resource_used: float


def _histories(rows: Sequence[Mapping[str, Any]]) -> tuple[HistoryReceipt, ...]:
    return tuple(
        HistoryReceipt(
            history=tuple((int(a), int(o)) for a, o in r["history"]),
            time=r["time"],
            kind=r["kind"],
            boundary=r["boundary"],
            reach_probability=r["reach_probability"],
            belief=tuple(r["belief"]),
            realized_utility=r["realized_utility"],
            continuation_value=r["continuation_value"],
            forfeited_outcomes=r["forfeited_outcomes"],
        )
        for r in rows
    )


def _decisions(rows: Sequence[Mapping[str, Any]]) -> tuple[DecisionReceipt, ...]:
    return tuple(
        DecisionReceipt(
            history=tuple((int(a), int(o)) for a, o in r["history"]),
            time=r["time"],
            forced_by=r["forced_by"],
            actions=tuple((int(a), float(p)) for a, p in r["actions"]),
            resource_used=r["resource_used"],
        )
        for r in rows
    )


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """The value of a fixed policy by exact enumeration, with its receipts."""

    value: float
    policy_version: str
    policy_identity: str
    model_identity: str
    histories: tuple[HistoryReceipt, ...]
    decisions: tuple[DecisionReceipt, ...]
    nodes_visited: int
    accounting_residual: float
    measurement_identities: tuple[tuple[int, str], ...]
    assumptions: Mapping[str, str]
    inference: str

    @staticmethod
    def _from_wire(wire: Mapping[str, Any]) -> PolicyEvaluation:
        return PolicyEvaluation(
            value=wire["value"],
            policy_version=wire["policy_version"],
            policy_identity=wire["policy_identity"],
            model_identity=wire["model_identity"],
            histories=_histories(wire["histories"]),
            decisions=_decisions(wire["decisions"]),
            nodes_visited=wire["nodes_visited"],
            accounting_residual=wire["accounting_residual"],
            measurement_identities=tuple((int(a), str(i)) for a, i in wire["measurement_identities"]),
            assumptions=dict(wire["assumptions"]),
            inference=wire["inference"],
        )


@dataclass(frozen=True, slots=True)
class OffPolicyEstimate:
    """A point estimate from logged trajectories; the standard error is uncalibrated."""

    ordinary: float
    self_normalized: float
    effective_sample_size: float
    max_weight: float
    trajectories: int
    uncalibrated_standard_error: float | None
    policy_version: str
    policy_identity: str
    behaviour_version: str
    behaviour_identity: str
    model_identity: str
    positivity_histories_checked: int
    records_counted: int
    reused_recorded_log: bool
    ledger_state: str
    inference: str


@dataclass(frozen=True, slots=True)
class RolloutEstimate:
    """A seeded Monte Carlo diagnostic on common random streams."""

    mean: float
    monte_carlo_standard_error: float
    rollouts: int
    seed: int
    stream_scheme: str
    error_label: str
    policy_version: str
    policy_identity: str
    model_identity: str
    inference: str


@dataclass(frozen=True, slots=True)
class ValueOfEvidence:
    """The value of one observation scored by the design ranker and checked by enumeration."""

    collect_action: int
    prior_value: float
    expected_value_of_sample_information: float
    cost: float
    net_value_of_collecting: float
    predicted_policy_value: float
    enumerated_policy_value: float
    ranking_evaluation: str
    implemented_functional: str
    bayes_after_observation: Policy

    @staticmethod
    def _from_wire(wire: Mapping[str, Any]) -> ValueOfEvidence:
        return ValueOfEvidence(
            collect_action=wire["collect_action"],
            prior_value=wire["prior_value"],
            expected_value_of_sample_information=wire["expected_value_of_sample_information"],
            cost=wire["cost"],
            net_value_of_collecting=wire["net_value_of_collecting"],
            predicted_policy_value=wire["predicted_policy_value"],
            enumerated_policy_value=wire["enumerated_policy_value"],
            ranking_evaluation=wire["ranking_evaluation"],
            implemented_functional=wire["implemented_functional"],
            bayes_after_observation=_policy_from_wire(wire["bayes_after_observation"]),
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    """One named comparison: its kind (act_now, wait, collect_evidence,
    choose_after_observation) and the policy offered."""

    label: str
    kind: str
    policy: Policy | ReweightedStaticScores | LearnedFromEvaluationData

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _name(self.label, "comparison label"))
        if self.kind not in _COMPARISON_KINDS:
            raise CausalValueError(f"kind must be one of {', '.join(_COMPARISON_KINDS)}")


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """An evaluated comparison."""

    label: str
    kind: str
    policy_version: str
    policy_identity: str
    value: float
    difference_from_first: float
    evaluation: PolicyEvaluation


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Evaluated comparisons, in the order supplied; ``best`` ranks them, it is no optimum."""

    rows: tuple[ComparisonRow, ...]
    best: int
    value_of_information: ValueOfEvidence | None
    model_identity: str
    inference: str


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    """What one recorded observation did to the belief."""

    record_id: str
    time: int
    action: int
    symbol: int
    symbol_label: str
    probability: float
    belief: tuple[float, ...]
    measurement_identity: str | None
    action_record: str
    observation_record: str
    parameter_uncertainty_propagated: bool


# --------------------------------------------------------------------------- entry points


def evaluate(
    model: FiniteModel,
    policy: Policy | ReweightedStaticScores | LearnedFromEvaluationData,
    *,
    max_nodes: int | None = None,
    cancel: Any = None,
) -> PolicyEvaluation:
    """Evaluate a fixed policy by exact enumeration (the sequential g-formula).

    Raises :class:`SequentialEvaluationRefusal` for a static score table, a learned policy, a
    reachable history without a decision, an unsupported action, a resource budget exceeded or an
    outcome due after the horizon under the ``refuse`` rule.
    """
    fixed = resolve_policy(policy) if not isinstance(policy, Policy) else policy
    wire = _result(
        _native.sequential_evaluate(
            _dumps(model.to_wire()), _dumps(fixed.to_wire()), max_nodes, cancel=cancel
        )
    )
    return PolicyEvaluation._from_wire(wire)


@dataclass(frozen=True, slots=True)
class LoggedStep:
    """One logged decision cycle; ``observation`` is ``None`` for a commit action."""

    action: int
    observation: int | None
    probability: float


@dataclass(frozen=True, slots=True)
class LoggedTrajectory:
    """One logged trajectory: its steps and the realized state-dependent outcome utility."""

    id: str
    steps: tuple[LoggedStep, ...]
    outcome_utility: float

    def _wire(self) -> dict[str, Any]:
        return {
            "id": _name(self.id, "trajectory id"),
            "steps": [
                {"action": s.action, "observation": s.observation, "probability": s.probability}
                for s in self.steps
            ],
            "outcome_utility": _number(self.outcome_utility, "outcome utility"),
        }


def evaluate_off_policy(
    model: FiniteModel,
    policy: Policy,
    behaviour: Policy,
    trajectories: Sequence[LoggedTrajectory],
    *,
    max_nodes: int | None = None,
    cancel: Any = None,
) -> OffPolicyEstimate:
    """Sequential importance weighting of logged trajectories (point-only).

    Positivity is checked on every history the target policy reaches; a trajectory identity
    repeated in the log, or inconsistent with the model or logging policy, is refused.
    """
    wire = _result(
        _native.sequential_off_policy(
            _dumps(model.to_wire()),
            _dumps(policy.to_wire()),
            _dumps(behaviour.to_wire()),
            _dumps([t._wire() for t in trajectories]),
            max_nodes,
            cancel=cancel,
        )
    )
    return OffPolicyEstimate(**wire)


def rollout(
    model: FiniteModel, policy: Policy, *, seed: int, rollouts: int, cancel: Any = None
) -> RolloutEstimate:
    """Simulate the policy on common random streams; the error is an uncalibrated diagnostic."""
    wire = _result(
        _native.sequential_rollout(
            _dumps(model.to_wire()),
            _dumps(policy.to_wire()),
            _count(seed, "seed"),
            _count(rollouts, "rollouts"),
            cancel=cancel,
        )
    )
    return RolloutEstimate(**wire)


def stream_uniform(seed: int, rollout_index: int, time: int, role: str) -> float:
    """The uniform of one semantic stream identity ``(seed, rollout, time, role)``."""
    if role not in _STREAM_ROLES:
        raise CausalValueError(f"role must be one of {', '.join(_STREAM_ROLES)}")
    return float(
        _native.sequential_stream_uniform(
            _count(seed, "seed"), _count(rollout_index, "rollout"), _count(time, "time"), role
        )
    )


def compare(
    model: FiniteModel,
    comparisons: Sequence[Comparison],
    *,
    collect_action: int | None = None,
    commit_actions: Sequence[int] = (),
    target_population: str | None = None,
    target_period: str | None = None,
    max_nodes: int | None = None,
    cancel: Any = None,
) -> ComparisonReport:
    """Evaluate named comparisons; with ``collect_action``, score the observation's value.

    The value of collecting is scored by the existing design ranker and checked against the
    enumeration. A request for another population or period is refused naming the missing
    transport contract.
    """
    wire_comparisons = [
        {"label": c.label, "kind": c.kind, "policy": _spec_wire(c.policy)} for c in comparisons
    ]
    voi = (
        None
        if collect_action is None
        else _dumps(
            {
                "collect_action": _count(collect_action, "collect action"),
                "commit_actions": [_count(a, "commit action") for a in commit_actions],
            }
        )
    )
    request = (
        None
        if target_population is None and target_period is None
        else _dumps({"target_population": target_population, "target_period": target_period})
    )
    wire = _result(
        _native.sequential_compare(
            _dumps(model.to_wire()), _dumps(wire_comparisons), voi, request, max_nodes, cancel=cancel
        )
    )
    return ComparisonReport(
        rows=tuple(
            ComparisonRow(
                label=r["label"],
                kind=r["kind"],
                policy_version=r["policy_version"],
                policy_identity=r["policy_identity"],
                value=r["value"],
                difference_from_first=r["difference_from_first"],
                evaluation=PolicyEvaluation._from_wire(r["evaluation"]),
            )
            for r in wire["rows"]
        ),
        best=wire["best"],
        value_of_information=(
            None
            if wire["value_of_information"] is None
            else ValueOfEvidence._from_wire(wire["value_of_information"])
        ),
        model_identity=wire["model_identity"],
        inference=wire["inference"],
    )


def value_of_collecting_evidence(
    model: FiniteModel,
    collect_action: int,
    commit_actions: Sequence[int],
    *,
    max_nodes: int | None = None,
    cancel: Any = None,
) -> ValueOfEvidence:
    """Score one observation by the design ranker (value of sample information)."""
    wire = _result(
        _native.sequential_value_of_evidence(
            _dumps(model.to_wire()),
            _dumps(
                {
                    "collect_action": _count(collect_action, "collect action"),
                    "commit_actions": [_count(a, "commit action") for a in commit_actions],
                }
            ),
            max_nodes,
            cancel=cancel,
        )
    )
    return ValueOfEvidence._from_wire(wire)


def check_request(
    model: FiniteModel, *, target_population: str | None = None, target_period: str | None = None
) -> None:
    """Raise unless the request is about the model's own population and period."""
    refusal = _native.sequential_check_request(
        _dumps(model.to_wire()),
        _dumps({"target_population": target_population, "target_period": target_period}),
    )
    raise_refusal(refusal, SequentialEvaluationRefusal)


def request_interval() -> None:
    """Always raises ``cell_not_licensed``: no interval is licensed for a policy value."""
    raise_refusal(_native.sequential_request_interval(), SequentialEvaluationRefusal)


class BeliefSession:
    """Follows one realized history: Bayes updates, each record counted once in a ledger."""

    def __init__(self, model: FiniteModel, session: str) -> None:
        self._handle = _unwrap(_native.SequentialBeliefSession.start(_dumps(model.to_wire()), session))

    def observe(self, action: int, symbol: int, record: str) -> ObservationReceipt:
        """Record one action and the observation it produced; a repeat is refused."""
        wire = _result(self._handle.observe(_count(action, "action"), _count(symbol, "symbol"), record))
        return ObservationReceipt(
            record_id=wire["record_id"],
            time=wire["time"],
            action=wire["action"],
            symbol=wire["symbol"],
            symbol_label=wire["symbol_label"],
            probability=wire["probability"],
            belief=tuple(wire["belief"]),
            measurement_identity=wire["measurement_identity"],
            action_record=wire["action_record"],
            observation_record=wire["observation_record"],
            parameter_uncertainty_propagated=wire["parameter_uncertainty_propagated"],
        )

    @property
    def belief(self) -> tuple[float, ...]:
        """The current posterior over the latent states."""
        return tuple(self._handle.belief)

    @property
    def history(self) -> _History:
        """The realized history so far."""
        return tuple((int(a), int(o)) for a, o in json.loads(self._handle.history_json))

    @property
    def time(self) -> int:
        """The decision time of the next action."""
        return int(self._handle.time)

    @property
    def resource_used(self) -> float:
        """Resource consumed so far."""
        return float(self._handle.resource_used)

    @property
    def model_identity(self) -> str:
        """Identity of the model the session updates under."""
        return str(self._handle.model_identity)

    @property
    def ledger_state(self) -> str:
        """Digest of the evidence ledger state."""
        return str(self._handle.ledger_state)

    @property
    def recorded(self) -> int:
        """Number of recorded observations."""
        return int(self._handle.recorded)


class EvaluationArtifact:
    """A sealed, replayable ``sequential_evaluation_v1`` evaluation."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    @property
    def value(self) -> float:
        """The exact policy value."""
        return float(self._handle.value)

    @property
    def identities(self) -> dict[str, Any]:
        """``model``, ``policy``, ``behaviour`` and ``evaluation`` identities."""
        return dict(json.loads(self._handle.identities_json))

    @property
    def summary(self) -> dict[str, Any]:
        """Receipts and estimates as plain numbers."""
        return dict(json.loads(self._handle.summary_json))

    @property
    def histories(self) -> tuple[HistoryReceipt, ...]:
        """The history receipts."""
        return _histories(self.summary["histories"])

    @property
    def decisions(self) -> tuple[DecisionReceipt, ...]:
        """The decision receipts."""
        return _decisions(self.summary["decisions"])

    @property
    def model(self) -> FiniteModel:
        """The model the artifact replays."""
        return _model_from_wire(json.loads(self._handle.model_json))

    @property
    def policy(self) -> Policy:
        """The policy (with its version) the artifact replays."""
        return _policy_from_wire(json.loads(self._handle.policy_json))

    def export(self, artifact_id: str = "sequential_evaluation") -> bytes:
        """The bounded checksummed container bytes."""
        return bytes(self._handle.export(artifact_id))


def seal(
    model: FiniteModel,
    policy: Policy,
    *,
    behaviour: Policy | None = None,
    trajectories: Sequence[LoggedTrajectory] = (),
    rollouts: tuple[int, int] | None = None,
    cancel: Any = None,
) -> EvaluationArtifact:
    """Evaluate and seal a policy, optionally with its off-policy estimate and a rollout.

    ``rollouts`` is ``(seed, count)``.
    """
    off = None
    if behaviour is not None:
        off = _dumps(
            {"behaviour": behaviour.to_wire(), "trajectories": [t._wire() for t in trajectories]}
        )
    elif trajectories:
        raise CausalValueError("logged trajectories need the logging policy that generated them")
    roll = None if rollouts is None else _dumps({"seed": rollouts[0], "rollouts": rollouts[1]})
    handle = _unwrap(
        _native.SequentialEvaluationArtifact.seal(
            _dumps(model.to_wire()), _dumps(policy.to_wire()), off, roll, cancel=cancel
        )
    )
    return EvaluationArtifact(handle)


def consume_artifact(
    data: bytes,
    *,
    model: str | None = None,
    policy: str | None = None,
    evaluation: str | None = None,
    cancel: Any = None,
) -> EvaluationArtifact:
    """Re-evaluate an artifact and require every stored number to replay bit for bit.

    ``model``, ``policy`` and ``evaluation`` are identities the consumer retained independently
    of the bytes; one that differs is refused.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise CausalTypeError("an artifact is bytes")
    expected = None
    if model is not None or policy is not None or evaluation is not None:
        expected = _dumps({"model": model, "policy": policy, "evaluation": evaluation})
    handle = _unwrap(
        _native.SequentialEvaluationArtifact.consume(bytes(data), expected, cancel=cancel)
    )
    return EvaluationArtifact(handle)
