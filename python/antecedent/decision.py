"""Declare a decision problem and evaluate it on aligned joint draws.

A decision names its actions by stable ID, the scientific quantities each
action's utility reads, and a closed utility expression you build with ordinary
operators. Hard constraints exclude actions; they never become penalties::

    p, q = decision.x(0), decision.x(1)
    contract = decision.Contract(
        actions=[
            decision.Action("treat", inputs=(benefit, cost), utility=p * q),
            decision.Action("wait", inputs=(baseline,), utility=decision.x(0)),
        ],
        utility_units="qaly",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )
    result = contract.evaluate(joint_distribution)   # aligned draws, not marginals
    print(result.explain())

The draws are a :class:`~antecedent.joint_distribution.JointDistributionArtifact`:
a nonlinear utility over several quantities needs genuine joint rows, and
independent marginals refuse with ``joint_law_required``. Rust owns the
identity, validation, evaluation, artifacts and refusal rules; this module
builds declarations and raises each refusal as
:class:`DecisionRefusal`, a :class:`~antecedent.errors.CausalUnsupportedError`
with its registered ``reason_code``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ._native import decision_contract_normalize as _normalize
from ._native import decision_source_digest as _source_digest
from ._native import evaluate_decision as _evaluate
from ._native import export_decision_contract as _export_contract
from ._native import export_decision_result as _export_result
from ._native import load_decision_contract as _load_contract
from ._native import replay_decision_result as _replay
from .errors import CausalUnsupportedError, CausalValueError
from .joint_distribution import JointDistributionArtifact, ScientificQuantity

ActionKind = Literal["intervention", "policy", "regime", "study", "external"]
StructuralPolicy = Literal[
    "require_invariant_best_action", "maximin", "bayes_over_structures", "report_only"
]


class DecisionRefusal(CausalUnsupportedError):
    """A decision refusal carrying the structured Rust fields.

    ``reason_code`` is inherited and registered. ``detail`` is the namespaced
    ``family.slot``; ``offending`` names the action input (``action[input]``)
    when there is one.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        text = str(refusal["detail"]) + (f" at {offending}" if offending else "")
        super().__init__(text, reason_code=refusal["code"])
        self.detail: str = refusal["detail"]
        self.offending: str | None = offending


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise DecisionRefusal(json.loads(refusal))


@dataclass(frozen=True, slots=True)
class Expr:
    """A closed utility expression over an action's inputs; build it with operators."""

    _wire_value: Any

    def __add__(self, other: Expr | float) -> Expr:
        return Expr({"add": [self._wire_value, _coerce(other)._wire_value]})

    def __radd__(self, other: float) -> Expr:
        return _coerce(other) + self

    def __sub__(self, other: Expr | float) -> Expr:
        return Expr({"sub": [self._wire_value, _coerce(other)._wire_value]})

    def __rsub__(self, other: float) -> Expr:
        return _coerce(other) - self

    def __mul__(self, other: Expr | float) -> Expr:
        return Expr({"mul": [self._wire_value, _coerce(other)._wire_value]})

    def __rmul__(self, other: float) -> Expr:
        return _coerce(other) * self

    def __neg__(self) -> Expr:
        return Expr({"neg": self._wire_value})

    def __repr__(self) -> str:
        return f"Expr({json.dumps(self._wire_value, sort_keys=True)})"


def _coerce(value: Expr | float) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalValueError("a utility expression combines Expr values and numbers")
    return const(float(value))


def x(index: int) -> Expr:
    """The ``index``th input quantity of the action."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CausalValueError("an input index is a non-negative integer")
    return Expr({"input": index})


def const(value: float) -> Expr:
    """A constant."""
    return Expr({"const": float(value)})


def maximum(left: Expr | float, right: Expr | float) -> Expr:
    """Pointwise maximum."""
    return Expr({"max": [_coerce(left)._wire_value, _coerce(right)._wire_value]})


def minimum(left: Expr | float, right: Expr | float) -> Expr:
    """Pointwise minimum."""
    return Expr({"min": [_coerce(left)._wire_value, _coerce(right)._wire_value]})


@dataclass(frozen=True, slots=True)
class Criterion:
    """How actions are ranked. Build with the classmethods; CVaR is not offered."""

    kind: str
    threshold: float | None = None
    p: float | None = None

    @classmethod
    def expected_utility(cls) -> Criterion:
        return cls("posterior_expected_utility")

    @classmethod
    def expected_loss(cls) -> Criterion:
        """Minimize posterior expected loss; the contract's utilities are losses."""
        return cls("posterior_expected_loss")

    @classmethod
    def threshold_probability(cls, threshold: float) -> Criterion:
        return cls("threshold_probability", threshold=float(threshold))

    @classmethod
    def quantile(cls, p: float) -> Criterion:
        return cls("quantile", p=float(p))

    @classmethod
    def minimax_over_identified_set(cls) -> Criterion:
        return cls("minimax_over_identified_set")

    @classmethod
    def maximin_over_structures(cls) -> Criterion:
        return cls("maximin_over_structures")

    @classmethod
    def regret(cls) -> Criterion:
        return cls("regret")

    @classmethod
    def expected_regret(cls) -> Criterion:
        return cls("expected_regret")

    def _wire(self) -> Any:
        if self.kind == "threshold_probability":
            return {"threshold_probability": {"threshold": self.threshold}}
        if self.kind == "quantile":
            return {"quantile": {"p": self.p}}
        return self.kind

    @classmethod
    def _from_wire(cls, wire: Any) -> Criterion:
        if isinstance(wire, str):
            return cls(wire)
        ((kind, args),) = wire.items()
        return cls(kind, threshold=args.get("threshold"), p=args.get("p"))


@dataclass(frozen=True, slots=True)
class Action:
    """One action: a stable ID, the quantities its utility reads, and the utility."""

    id: str
    inputs: tuple[ScientificQuantity, ...]
    utility: Expr
    kind: ActionKind = "intervention"

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "inputs": [q._wire() for q in self.inputs],
            "utility": self.utility._wire_value,
        }


@dataclass(frozen=True, slots=True)
class Constraint:
    """A hard constraint ``P(expr <= bound) >= min_probability``.

    A violated constraint excludes the action; it never enters the utility.
    ``applies_to`` empty applies to every action.
    """

    id: str
    expr: Expr
    bound: float
    units: str
    min_probability: float = 1.0
    applies_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "applies_to", tuple(self.applies_to))

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "expr": self.expr._wire_value,
            "bound": float(self.bound),
            "min_probability": float(self.min_probability),
            "units": self.units,
            "applies_to": list(self.applies_to),
        }


@dataclass(frozen=True, slots=True)
class ConstraintExclusion:
    """A hard constraint an action failed, with the probability it reached."""

    constraint_id: str
    probability: float
    required: float


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """One action's outcome under the contract."""

    id: str
    admissible: bool
    exclusions: tuple[ConstraintExclusion, ...]
    expected_utility: float
    value: float
    standard_error: float | None
    expected_regret: float | None
    max_regret: float | None


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the evaluation can claim about the choice.

    ``kind`` is ``uniquely_optimal``, ``indistinguishable`` (the leader cannot be
    told apart from the others within the declared error) or
    ``no_admissible_action``. ``actions`` are the leader, then those it ties.
    """

    kind: Literal["uniquely_optimal", "indistinguishable", "no_admissible_action"]
    actions: tuple[str, ...]


def _expr_from_wire(wire: Any) -> Expr:
    return Expr(wire)


@dataclass(frozen=True, slots=True)
class Contract:
    """A durable decision problem."""

    actions: tuple[Action, ...]
    utility_units: str
    criterion: Criterion
    target_population: str
    horizon: int = 0
    constraints: tuple[Constraint, ...] = ()
    structural_policy: StructuralPolicy = "report_only"

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "constraints", tuple(self.constraints))

    def _wire(self) -> dict[str, Any]:
        return {
            "version": 1,
            "actions": [a._wire() for a in self.actions],
            "utility_units": self.utility_units,
            "criterion": self.criterion._wire(),
            "constraints": [c._wire() for c in self.constraints],
            "target_population": self.target_population,
            "horizon": self.horizon,
            "structural_policy": self.structural_policy,
        }

    def _normalized(self) -> dict[str, Any]:
        normalized, refusal = _normalize(json.dumps(self._wire()))
        _raise(refusal)
        assert normalized is not None
        return dict(json.loads(normalized))

    @property
    def identity(self) -> str:
        """Canonical digest: unchanged by reordering actions, changed by any semantic edit."""
        return str(self._normalized()["identity"])

    def evaluate(self, source: JointDistributionArtifact) -> Decision:
        """Evaluate on aligned joint draws, or refuse with a registered reason code."""
        result, refusal = _evaluate(json.dumps(self._wire()), source._native)
        _raise(refusal)
        assert result is not None
        return Decision(self, source, json.loads(result))

    def export(self, *, artifact_id: str = "decision-contract") -> bytes:
        return bytes(_export_contract(json.dumps(self._wire()), artifact_id))

    @classmethod
    def load(cls, data: bytes, *, expected_identity: str) -> Contract:
        """Load only under the identity the consumer retained independently."""
        return cls._from_wire(json.loads(_load_contract(data, expected_identity)))

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> Contract:
        return cls(
            actions=tuple(
                Action(
                    id=a["id"],
                    kind=a["kind"],
                    inputs=tuple(ScientificQuantity._from_wire(q) for q in a["inputs"]),
                    utility=_expr_from_wire(a["utility"]),
                )
                for a in wire["actions"]
            ),
            utility_units=wire["utility_units"],
            criterion=Criterion._from_wire(wire["criterion"]),
            target_population=wire["target_population"],
            horizon=wire["horizon"],
            constraints=tuple(
                Constraint(
                    id=c["id"],
                    expr=_expr_from_wire(c["expr"]),
                    bound=c["bound"],
                    units=c["units"],
                    min_probability=c["min_probability"],
                    applies_to=tuple(c["applies_to"]),
                )
                for c in wire["constraints"]
            ),
            structural_policy=wire["structural_policy"],
        )


class Decision:
    """A decision result with the evidence behind it; exportable and replayable."""

    def __init__(
        self, contract: Contract, source: JointDistributionArtifact, body: Mapping[str, Any]
    ) -> None:
        self._contract = contract
        self._source = source
        self._body = body

    @property
    def contract_identity(self) -> str:
        return str(self._body["contract_identity"])

    @property
    def source_digest(self) -> str:
        return str(self._body["source_digest"])

    @property
    def actions(self) -> tuple[ActionOutcome, ...]:
        def number(value: float | None) -> float:
            return float("nan") if value is None else float(value)

        return tuple(
            ActionOutcome(
                id=a["id"],
                admissible=a["admissible"],
                exclusions=tuple(
                    ConstraintExclusion(e["constraint_id"], e["probability"], e["required"])
                    for e in a["exclusions"]
                ),
                expected_utility=float(a["expected_utility"]),
                value=number(a["value"]),
                standard_error=a["standard_error"],
                expected_regret=a["expected_regret"],
                max_regret=a["max_regret"],
            )
            for a in self._body["actions"]
        )

    @property
    def verdict(self) -> Verdict:
        wire = self._body["verdict"]
        if wire == "no_admissible_action":
            return Verdict("no_admissible_action", ())
        ((kind, value),) = wire.items()
        if kind == "uniquely_optimal":
            return Verdict("uniquely_optimal", (value,))
        return Verdict("indistinguishable", tuple(value))

    @property
    def selected(self) -> tuple[str, ...]:
        """The leader, and any actions it cannot be told apart from; empty if none admissible."""
        return self.verdict.actions

    @property
    def evpi(self) -> float | None:
        """Expected value of perfect information over the admissible actions."""
        value = self._body["evpi"]
        return None if value is None else float(value)

    @property
    def n_draws(self) -> int:
        return int(self._body["n_draws"])

    @property
    def effective_draws(self) -> float:
        return float(self._body["effective_draws"])

    @property
    def assumptions(self) -> tuple[str, ...]:
        return tuple(self._body["assumptions"])

    def explain(self) -> str:
        """Why this action, or why none: the choice, its evidence and its limits."""
        verdict = self.verdict
        by_id = {a.id: a for a in self.actions}
        criterion = self._contract.criterion.kind.replace("_", " ")
        if verdict.kind == "no_admissible_action":
            reasons = "; ".join(
                f"{a.id} failed {e.constraint_id} (held {e.probability:.3g}, required {e.required:.3g})"
                for a in self.actions
                for e in a.exclusions
            )
            return f"No action satisfies every hard constraint: {reasons}."
        leader = by_id[verdict.actions[0]]
        text = f"{leader.id!r} ranks first by {criterion} (value {leader.value:.4g}"
        if leader.standard_error is not None:
            text += f", standard error {leader.standard_error:.3g}"
        text += ")"
        if verdict.kind == "indistinguishable":
            others = ", ".join(repr(a) for a in verdict.actions[1:])
            text += f"; it cannot be separated from {others} within the sampling error"
        excluded = [a for a in self.actions if not a.admissible]
        if excluded:
            text += "; excluded by a hard constraint: " + ", ".join(repr(a.id) for a in excluded)
        if self.evpi is not None:
            text += f"; perfect information would be worth {self.evpi:.4g}"
        return text + "."

    def export(self, *, artifact_id: str = "decision-result") -> bytes:
        """The result bound to its contract identity and source digest, replayable."""
        return bytes(
            _export_result(json.dumps(self._contract._wire()), self._source._native, artifact_id)
        )

    def __repr__(self) -> str:
        return f"<Decision {self.verdict.kind} {self.selected}>"


def replay(data: bytes, *, contract: Contract, source: JointDistributionArtifact) -> Decision:
    """Recompute a stored decision from its inputs and require an exact match.

    The contract and source are the consumer's own: a result computed under a
    different contract or from different draws, or one whose numbers were edited,
    refuses.
    """
    result, refusal = _replay(data, json.dumps(contract._wire()), source._native)
    _raise(refusal)
    assert result is not None
    return Decision(contract, source, json.loads(result))


def source_digest(source: JointDistributionArtifact) -> str:
    """Digest of a source's aligned draws, to retain alongside a result."""
    return str(_source_digest(source._native))


__all__ = [
    "Action",
    "ActionOutcome",
    "Constraint",
    "ConstraintExclusion",
    "Contract",
    "Criterion",
    "Decision",
    "DecisionRefusal",
    "Expr",
    "Verdict",
    "const",
    "maximum",
    "minimum",
    "replay",
    "source_digest",
    "x",
]
