"""Declare a decision problem and evaluate it on aligned joint draws.

A decision names its actions by stable ID, the scientific quantities each
action's utility reads, and a closed utility expression you build with ordinary
operators. Hard constraints exclude actions; they never become penalties::

    from antecedent import require_success

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
    outcome = contract.evaluate(joint_distribution)
    result = require_success(outcome)  # explicit refusal handling; aligned joint draws
    print(result.explain())

The draws are a :class:`~antecedent.joint_distribution.JointDistributionArtifact`:
a nonlinear utility over several quantities needs genuine joint rows, and
independent marginals refuse with ``joint_law_required``. Rust owns the
identity, validation, evaluation, artifacts and refusal rules; this module
builds declarations and returns scientific refusals as Outcome readings backed by
:class:`DecisionRefusal`, a :class:`~antecedent.errors.CausalUnsupportedError`
with its registered ``reason_code``.

A decision that weighs several outcomes (benefit, harm, cost) can state its value over named
attributes instead, with the trade-offs between them either as plain numbers or as declared
ranges::

    benefit = decision.attribute("benefit", units="qaly")
    harm = decision.attribute("harm", units="events")
    w = decision.tradeoff("harm_weight", units="qaly/event", lower=0.5, upper=4.0,
                          assumption="clinical_panel")
    contract = decision.Contract(
        value=decision.Value(benefit - w * harm, units="qaly"),
        actions=[
            decision.Action("treat", inputs={"benefit": treat_benefit, "harm": treat_harm}),
            decision.Action("wait", inputs={"benefit": wait_benefit, "harm": wait_harm}),
        ],
        target_population="target",
    )
    result = require_success(contract.evaluate(joint_distribution))
    print(result.choice, result.standing, result.statement)

A value whose trade-offs are plain numbers is an ordinary contract: the same identity,
result type, ties and replay as the contract written with :func:`x`. A value with a named
trade-off is decided across every admissible value of its trade-offs and returns a
:class:`TradeoffDecision`; a range is a declared assumption, never a probability. Every result
reads the same way: ``choice``, ``kind``, ``standing`` and ``statement``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args

from antecedent import _reason_details

from . import _retained
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._columns import coerce_fields
from ._decision_result import CompiledReceipt, DecisionResult, Standing, _standing
from ._decision_value import (
    ActionKind,
    Attribute,
    Curve,
    Measure,
    Relation,
    Structures,
    Tradeoff,
    TradeoffPolicy,
    Value,
    ValueExpr,
    _Algebra,
    _constraint_node,
    _constraint_node_of,
    attribute,
    both,
    excess,
    exp,
    interaction,
    ln,
    relation,
    segment,
    shortfall,
    tradeoff,
)
from ._decision_value import StructuralPolicy as _StructuralPolicyClass
from ._decision_value import StructuralPolicyKind as StructuralPolicy
from ._decision_value import maximum as _value_maximum
from ._decision_value import minimum as _value_minimum
from ._native import composition_lineage as _composition_lineage
from ._native import decision_contract_normalize as _normalize
from ._native import decision_source_digest as _source_digest
from ._native import evaluate_decision as _evaluate
from ._native import evaluate_decision_means as _evaluate_means
from ._native import export_decision_contract as _export_contract
from ._native import export_decision_result as _export_result
from ._native import load_decision_contract as _load_contract
from ._native import replay_decision_result as _replay
from ._native import score_published_decision as _score_published
from ._refusal import raise_refusal
from ._validate import real, text
from .errors import (
    CausalCancelledError,
    CausalTypeError,
    CausalUnsupportedError,
    CausalValueError,
    StructuredRefusal,
)
from .external import BoundExternalClaim, LineageLink
from .joint_distribution import JointDistributionArtifact, ScientificQuantity
from .outcomes import Answer, outcome_operation, require_success
from .results.sources import MeanSource

if TYPE_CHECKING:
    from ._decision_compile import (
        CompiledDecision,
        CompiledQuantity,
        DesignLawPlan,
        JointLawInput,
        JointLawPlan,
        TradeOffRange,
    )
    from ._decision_sources import (
        MultiArmEvidence,
        chain_law,
        experiment_means,
        experiment_quantity,
        joint_law,
        multi_arm_means,
        multi_arm_quantity,
    )
    from ._decision_tradeoff_results import (
        ActionRecord,
        ClassSampling,
        Region,
        SampledMargin,
        SampledMass,
        StructureOutcome,
        Switch,
        SwitchThreshold,
        TradeoffDecision,
    )
    from ._native import AteAnalysisResult
    from .decision_robust import AdmissibilityRules, RobustDecision
    from .program_claims import ProgramBinding
    from .source_evidence import SourceEvidence

RESULT_LINK_ID = "decision_result"


class DecisionRefusal(StructuredRefusal):
    """A decision refusal carrying the structured Rust fields.

    A :class:`~antecedent.errors.StructuredRefusal`: ``code`` (alias ``reason_code``) and
    ``remedy`` are the registered fields. ``stage`` is the stage that refused. ``detail`` is
    the namespaced ``family.slot``; ``offending`` names the action input (``action[input]``) or
    action when there is one; ``expected`` and ``supplied`` carry the compared
    semantics (for example ``joint`` against ``independent_marginals``).
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        text = str(refusal["detail"]) + (f" at {offending}" if offending else "")
        super().__init__(refusal, text=text)
        self.expected: str | None = refusal.get("expected")
        self.supplied: str | None = refusal.get("supplied")


class PreferenceRefusal(DecisionRefusal):
    """A refusal of a value, a contract across trade-offs or a decision across them.

    A :class:`DecisionRefusal`, so ``except DecisionRefusal`` catches it too: ``code`` (alias
    ``reason_code``) is the registered refusal code and ``detail`` the namespaced
    ``family.slot`` (for example ``preference_decision.measure_required``); ``offending`` names
    the trade-off, action or record when there is one, and ``expected`` / ``supplied`` carry the
    compared values.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        offending = refusal.get("offending")
        message = refusal.get("message")
        detail = str(refusal["detail"])
        text_ = (
            detail + (f": {message}" if message else "") + (f" at {offending}" if offending else "")
        )
        StructuredRefusal.__init__(self, refusal, text=text_)
        self.expected = refusal.get("expected")
        self.supplied = refusal.get("supplied")


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


def _leaf_kinds(expr: Expr) -> frozenset[str]:
    """The leaf kinds an expression reads: ``input``, ``quantity``, ``attribute``,
    ``parameter`` and/or ``const``.

    A decision utility reads ``input`` leaves (:func:`x`); a sensitivity utility reads
    ``quantity`` leaves (:func:`antecedent.sensitivity_decision.quantity`); a hard constraint of
    a value contract reads ``attribute`` leaves (:func:`attribute`). The same :class:`Expr` type
    carries them, and each declaration refuses the others' leaves.
    """
    found: set[str] = set()
    stack: list[Any] = [expr._wire_value]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("input", "quantity", "const", "attribute", "parameter"):
                    found.add(key)
                else:
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return frozenset(found)


def _coerce(value: Expr | float) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalValueError("a utility expression combines Expr values and numbers")
    return const(float(value))


_real = partial(real, type_message="{what} must be a number, got {got}")

_text = partial(text, type_message="{what} must be a string, got {got}")


def x(index: int) -> Expr:
    """The ``index``th input quantity of the action."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CausalValueError("an input index is a non-negative integer")
    return Expr({"input": index})


def const(value: float) -> Expr:
    """A constant."""
    return Expr({"const": float(value)})


def maximum(left: Any, right: Any) -> Any:
    """Pointwise maximum.

    Of two utility expressions (:func:`x`, numbers) it is an :class:`Expr`; when either side
    reads an attribute or a trade-off it is a :class:`ValueExpr`.
    """
    if isinstance(left, _Algebra) or isinstance(right, _Algebra):
        return _value_maximum(left, right)
    return Expr({"max": [_coerce(left)._wire_value, _coerce(right)._wire_value]})


def minimum(left: Any, right: Any) -> Any:
    """Pointwise minimum.

    Of two utility expressions (:func:`x`, numbers) it is an :class:`Expr`; when either side
    reads an attribute or a trade-off it is a :class:`ValueExpr`.
    """
    if isinstance(left, _Algebra) or isinstance(right, _Algebra):
        return _value_minimum(left, right)
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
        """Maximize ``P(utility >= threshold)``; ``threshold`` is a number in utility units."""
        return cls("threshold_probability", threshold=_real(threshold, "threshold"))

    @classmethod
    def quantile(cls, p: float) -> Criterion:
        """Maximize the ``p`` quantile of the utility; ``p`` is a number."""
        return cls("quantile", p=_real(p, "p"))

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


_VALUE_UTILITY = (
    "Action.utility reads inputs by position (decision.x(i)); a value over attributes goes in "
    "Contract(value=decision.Value(...))"
)


@dataclass(frozen=True, slots=True)
class Action:
    """One action: a stable ID, the quantities it reads, and how they are read.

    Two forms, told apart by ``inputs``:

    * a sequence of :class:`~antecedent.joint_distribution.ScientificQuantity` read by position
      (:func:`x`), with this action's own ``utility``;
    * a mapping from attribute ids (or :class:`Attribute` objects) to quantities, or to finite
      numbers in the attribute's units for an input known without uncertainty. The action then
      takes no ``utility``: the contract's ``value=`` reads every action's inputs by name.

    ``utility`` defaults to ``None`` only for the mapping form: with positional inputs it is
    required, and ``None`` raises :class:`~antecedent.errors.CausalTypeError`; with a mapping
    it must stay ``None``. With a mapping, a kind given third by position is read as ``kind``.
    """

    id: str
    inputs: tuple[ScientificQuantity, ...] | Mapping[str, ScientificQuantity | float]
    utility: Expr | None = None
    kind: ActionKind = "intervention"

    def __post_init__(self) -> None:
        _text(self.id, "Action.id")
        if isinstance(self.inputs, Mapping):
            self._named_inputs()
        else:
            self._positional_inputs()
        if self.kind not in get_args(ActionKind):
            raise CausalValueError(f"Action.kind must be one of {get_args(ActionKind)}")

    def _positional_inputs(self) -> None:
        if isinstance(self.inputs, (str, bytes)) or not isinstance(self.inputs, Iterable):
            raise CausalTypeError("Action.inputs must be a sequence of ScientificQuantity")
        inputs = tuple(self.inputs)
        if any(not isinstance(item, ScientificQuantity) for item in inputs):
            raise CausalTypeError("Action.inputs must hold only ScientificQuantity values")
        if isinstance(self.utility, _Algebra):
            raise CausalTypeError(_VALUE_UTILITY)
        if not isinstance(self.utility, Expr):
            raise CausalTypeError(
                "Action.utility must be an Expr built from decision.x(i), decision.const(v) "
                f"and operators, not {type(self.utility).__name__}; wrap a number with "
                "decision.const(value)"
            )
        leaves = _leaf_kinds(self.utility)
        if "quantity" in leaves:
            raise CausalTypeError(
                "Action.utility reads inputs by position (decision.x(i)); a "
                "sensitivity_decision.quantity(...) leaf belongs to "
                "sensitivity_decision.SensitivityAction"
            )
        if "attribute" in leaves or "parameter" in leaves:
            raise CausalTypeError(_VALUE_UTILITY)
        coerce_fields(self, inputs=inputs)

    def _named_inputs(self) -> None:
        if (
            isinstance(self.utility, str)
            and self.utility in get_args(ActionKind)
            and self.kind == "intervention"
        ):
            # Named inputs take no utility, so a kind given third by position is the kind:
            # Action(id, inputs, kind).
            object.__setattr__(self, "kind", self.utility)
            object.__setattr__(self, "utility", None)
        if self.utility is not None:
            raise CausalTypeError(
                f"Action {self.id!r} names its inputs by attribute; its value is the contract's "
                "value=, so it takes no utility=; to give this action its own utility pass "
                "inputs as a sequence read by decision.x(i)"
            )
        assert isinstance(self.inputs, Mapping)
        inputs: dict[str, ScientificQuantity | float] = {}
        for key, value in self.inputs.items():
            name = key.id if isinstance(key, Attribute) else key
            if not isinstance(name, str):
                raise CausalTypeError("Action.inputs keys are attribute ids or Attributes")
            if not isinstance(value, ScientificQuantity):
                value = real(value, f"Action {self.id!r} input {name!r}", finite=True)
            inputs[name] = value
        coerce_fields(self, inputs=MappingProxyType(inputs))

    @property
    def _named(self) -> bool:
        return isinstance(self.inputs, Mapping)

    @property
    def quantities(self) -> tuple[ScientificQuantity, ...]:
        """The scientific quantities the action reads, in input order."""
        values = self.inputs.values() if isinstance(self.inputs, Mapping) else self.inputs
        return tuple(v for v in values if isinstance(v, ScientificQuantity))

    def _wire(self) -> dict[str, Any]:
        if isinstance(self.inputs, Mapping):
            raise CausalTypeError(
                f"Action {self.id!r} names its inputs by attribute; it is read through the "
                "value of a Contract(value=...)"
            )
        assert self.utility is not None
        return {
            "id": self.id,
            "kind": self.kind,
            "inputs": [q._wire() for q in self.inputs],
            "utility": self.utility._wire_value,
        }

    def _value_wire(self, attributes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """The action of a value contract: one input per attribute of the value, in order."""
        attributes = list(attributes)
        if not isinstance(self.inputs, Mapping):
            raise CausalTypeError(
                f"Contract(value=...) reads every action's inputs by attribute name; action "
                f"{self.id!r} gives positional inputs"
            )
        names = [a["id"] for a in attributes]
        missing = [n for n in names if n not in self.inputs]
        extra = [n for n in self.inputs if n not in names]
        if missing or extra:
            raise CausalValueError(
                f"Action {self.id!r} supplies every attribute of the value exactly once"
                + (f"; missing {missing}" if missing else "")
                + (f"; unknown {extra}" if extra else "")
            )
        wire = []
        for attr in attributes:
            value = self.inputs[attr["id"]]
            if isinstance(value, ScientificQuantity):
                wire.append({"quantity": value._wire()})
            else:
                wire.append({"fixed": {"value": float(value), "units": attr["units"]}})
        return {"id": self.id, "kind": self.kind, "inputs": wire}


@dataclass(frozen=True, slots=True)
class Constraint:
    """A hard constraint ``P(expr <= bound) >= min_probability``.

    A violated constraint excludes the action; it never enters the utility.
    ``applies_to`` empty applies to every action. In an ordinary contract ``expr`` reads an
    action's inputs by position (:func:`x`); in a contract with ``value=`` it reads attributes
    by name (an :class:`Attribute`, or an expression or :class:`Curve` over attributes) and
    never a trade-off, so whether it excludes an action does not depend on the trade-offs.
    """

    id: str
    expr: Expr
    bound: float
    units: str
    min_probability: float = 1.0
    applies_to: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.id, "Constraint.id")
        if isinstance(self.expr, _Algebra):
            coerce_fields(self, expr=Expr(_constraint_node_of(self.expr)))
        elif isinstance(self.expr, Expr):
            if "parameter" in _leaf_kinds(self.expr):
                raise CausalValueError("a hard constraint reads attributes only, never a trade-off")
        else:
            raise CausalTypeError(
                "Constraint.expr must be an Expr built from decision.x(i), or an expression "
                "over attributes"
            )
        _real(self.bound, "Constraint.bound")
        _real(self.min_probability, "Constraint.min_probability")
        _text(self.units, "Constraint.units")
        if isinstance(self.applies_to, (str, bytes)):
            raise CausalTypeError("Constraint.applies_to must be a sequence of action ids")
        coerce_fields(self, applies_to=tuple(self.applies_to))

    @property
    def _reads_attributes(self) -> bool:
        return "attribute" in _leaf_kinds(self.expr)

    @property
    def _reads_positions(self) -> bool:
        return "input" in _leaf_kinds(self.expr)

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

    ``kind`` is ``uniquely_optimal`` under the supplied finite law,
    ``indistinguishable`` for equal criterion values, or ``no_admissible_action``.
    Sampled-law rankings carry no statistical ranking guarantee.
    ``actions`` are the leader, then those it ties.
    """

    kind: Literal["uniquely_optimal", "indistinguishable", "no_admissible_action"]
    actions: tuple[str, ...]


def _not_replayable(supplied: str, remedy: str) -> dict[str, Any]:
    """The refusal to export a decision that was not evaluated on aligned joint draws."""
    return {
        "code": "route_not_supported",
        "stage": "export",
        "detail": _reason_details.decision_evaluation.MEAN_SOURCE_NOT_REPLAYABLE,
        "offending": None,
        "expected": "joint_draws",
        "supplied": supplied,
        "remedy": remedy,
    }


def _expr_from_wire(wire: Any) -> Expr:
    return Expr(wire)


_TUNING = ("measure", "tie_tolerance", "vertex_cap", "max_boxes", "min_width_fraction")


def _not_applicable(message: str) -> CausalValueError:
    return CausalValueError(message, reason_code="option_not_applicable")


@dataclass(frozen=True, slots=True)
class Contract:
    """A durable decision problem: its actions, how each is valued and how they are ranked.

    Two forms, told apart by ``value``:

    * **ordinary** (``value=None``): each :class:`Action` reads its quantities by position and
      carries its own ``utility``; ``utility_units`` and ``criterion`` are required;
    * **value** (``value=``): a :class:`Value` (or accepted preferences, anything exposing
      ``_preference_wire()``) reads every action's inputs by attribute name. ``utility_units``
      may be omitted (it is the value's units, and must equal them when given), ``criterion``
      defaults to expected utility, and ``policy`` (a :class:`TradeoffPolicy`, default
      :meth:`TradeoffPolicy.invariant_best`) says how a result across trade-off ranges is
      consumed. Hard constraints read attributes by name.

    A value whose trade-offs are plain numbers is decided as an ordinary contract: the same
    identity, result type, exact ties and exact replay as the contract written by position.
    A value with a named trade-off (:func:`tradeoff`) is decided across every admissible value
    of its trade-offs. Which engine decides is settled natively; :attr:`form` reports it.

    Three fields default to ``None`` only so the value form can leave them out; construction
    checks them:

    * ``target_population`` is required in both forms: ``None`` raises
      :class:`~antecedent.errors.CausalTypeError`.
    * ``utility_units`` is required for an ordinary contract (``None`` raises
      :class:`~antecedent.errors.CausalTypeError`). With ``value=`` it may stay ``None``: the
      contract is in the value's units, and a string given here must equal them.
    * ``criterion`` is required for an ordinary contract (``None`` raises
      :class:`~antecedent.errors.CausalTypeError`). With ``value=`` it may stay ``None``, which
      decides by expected utility.

    ``value`` and ``policy`` are keyword-only and default to ``None``: no value is the ordinary
    form, and an ordinary contract refuses ``policy=`` (``option_not_applicable``). An ordinary
    contract is therefore written exactly as before ``value=`` existed, with ``actions``,
    ``utility_units``, ``criterion`` and ``target_population`` all given.
    """

    actions: tuple[Action, ...]
    utility_units: str | None = None
    criterion: Criterion | None = None
    target_population: str | None = None
    horizon: int = 0
    constraints: tuple[Constraint, ...] = ()
    structural_policy: StructuralPolicy = "report_only"
    value: Value | Any | None = field(default=None, kw_only=True)
    policy: TradeoffPolicy | None = field(default=None, kw_only=True)
    _described: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.actions, (str, bytes)) or not isinstance(self.actions, Iterable):
            raise CausalTypeError("Contract.actions must be a sequence of Action")
        actions = tuple(self.actions)
        if any(not isinstance(action, Action) for action in actions):
            raise CausalTypeError("Contract.actions must hold only Action values")
        if isinstance(self.constraints, (str, bytes)) or not isinstance(self.constraints, Iterable):
            raise CausalTypeError("Contract.constraints must be a sequence of Constraint")
        constraints = tuple(self.constraints)
        if any(not isinstance(constraint, Constraint) for constraint in constraints):
            raise CausalTypeError("Contract.constraints must hold only Constraint values")
        if self.target_population is None:
            raise CausalTypeError("Contract needs target_population=")
        _text(self.target_population, "Contract.target_population")
        if self.criterion is not None and not isinstance(self.criterion, Criterion):
            raise CausalTypeError(
                "Contract.criterion must be a Criterion such as Criterion.expected_utility(), "
                f"not {type(self.criterion).__name__}"
            )
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise CausalTypeError("Contract.horizon must be an integer")
        structural = self.structural_policy
        if isinstance(structural, _StructuralPolicyClass):
            structural = structural.kind
        if structural not in get_args(StructuralPolicy):
            raise CausalValueError(
                f"Contract.structural_policy must be one of {get_args(StructuralPolicy)}"
            )
        if self.value is None:
            self._check_ordinary(actions, constraints)
        else:
            self._check_value(actions, constraints)
        coerce_fields(self, actions=actions)
        coerce_fields(self, constraints=constraints)
        coerce_fields(self, structural_policy=structural)

    def _check_ordinary(
        self, actions: tuple[Action, ...], constraints: tuple[Constraint, ...]
    ) -> None:
        if self.utility_units is None:
            raise CausalTypeError(
                "Contract needs utility_units= (or a value=decision.Value(...) that states them)"
            )
        _text(self.utility_units, "Contract.utility_units")
        if self.criterion is None:
            raise CausalTypeError(
                "Contract needs criterion=, such as Criterion.expected_utility() (a "
                "Contract(value=...) defaults to expected utility)"
            )
        for action in actions:
            if action._named:
                raise CausalTypeError(
                    f"action {action.id!r} names its inputs by attribute; pass the value that "
                    "reads them as Contract(value=decision.Value(...))"
                )
        for constraint in constraints:
            if constraint._reads_attributes:
                raise CausalTypeError(
                    f"constraint {constraint.id!r} reads attributes by name; pass the value "
                    "that reads them as Contract(value=decision.Value(...))"
                )
        if self.policy is not None:
            raise _not_applicable(
                "policy= says how a decision across trade-off ranges is consumed; an ordinary "
                "contract has no trade-offs (declare them in Contract(value=decision.Value(...)))"
            )

    def _check_value(
        self, actions: tuple[Action, ...], constraints: tuple[Constraint, ...]
    ) -> None:
        if isinstance(self.value, (_Algebra, Expr)):
            raise CausalTypeError(
                "Contract.value is a decision.Value: wrap the expression in "
                "decision.Value(expr, units=...)"
            )
        if not isinstance(self.value, Value) and not callable(
            getattr(self.value, "_preference_wire", None)
        ):
            raise CausalTypeError(
                "Contract.value is a decision.Value or accepted preferences (an object with "
                f"_preference_wire()), not {type(self.value).__name__}"
            )
        for action in actions:
            if not action._named:
                raise CausalTypeError(
                    "Contract(value=...) reads every action's inputs by attribute name; action "
                    f"{action.id!r} gives positional inputs"
                )
        for constraint in constraints:
            if constraint._reads_positions:
                raise CausalValueError(
                    "in a value contract a constraint reads attributes by name; decision.x(k) "
                    "reads an action's k-th positional input, which a value contract does not "
                    f"have (constraint {constraint.id!r})"
                )
        model = self._preferences()["model"]
        if self.utility_units is not None:
            _text(self.utility_units, "Contract.utility_units")
            if self.utility_units != model["utility_units"]:
                raise CausalValueError(
                    f"Contract.utility_units {self.utility_units!r} differs from the value's "
                    f"units {model['utility_units']!r}"
                )
        if self.policy is not None:
            if not isinstance(self.policy, TradeoffPolicy):
                raise CausalTypeError(
                    "Contract.policy is a TradeoffPolicy such as TradeoffPolicy.maximin()"
                )
            if not model["parameters"]:
                raise _not_applicable(
                    "policy= says how a decision across trade-off ranges is consumed; this "
                    "value declares no trade-off, so it is decided as an ordinary contract"
                )

    # -- the two wires --------------------------------------------------------------

    def _preferences(self) -> dict[str, Any]:
        """The value as the native wire: ``{"model": ..., "measure": ...}``."""
        if self.value is None:
            raise CausalTypeError("an ordinary contract has no value=; it declares no trade-offs")
        return dict(self.value._preference_wire())

    def _tradeoff_wire(self) -> dict[str, Any]:
        """The contract as the native contract across trade-offs (a value contract only)."""
        model = self._preferences()["model"]
        attributes = {a["id"]: k for k, a in enumerate(model["attributes"])}
        return {
            "model": model,
            "actions": [a._value_wire(model["attributes"]) for a in self.actions],
            "criterion": (self.criterion or Criterion.expected_utility())._wire(),
            "constraints": [
                {**c._wire(), "expr": _constraint_node(c.expr._wire_value, attributes)}
                for c in self.constraints
            ],
            "target_population": self.target_population,
            "horizon": self.horizon,
            "structural_policy": self.structural_policy,
            "preference_policy": (self.policy or TradeoffPolicy.invariant_best()).kind,
        }

    def _describe(self) -> dict[str, Any]:
        """The native description of a value contract, read once."""
        cached = self._described
        if cached is None:
            from ._decision_tradeoff import describe

            cached = describe(self)
            object.__setattr__(self, "_described", cached)
        return cached

    def _engine(self) -> Literal["ordinary", "tradeoff"]:
        if self.value is None:
            return "ordinary"
        described = self._describe()
        if described.get("ordinary") is not None:
            return "ordinary"
        # Natively decided: a value that names a trade-off, or one the ordinary utility language
        # cannot state (exp or ln of an attribute), is decided across trade-offs. Any other
        # refusal of the ordinary form is a fault of the declaration and is raised.
        if described.get("decided_across_preferences"):
            return "tradeoff"
        refusal = described.get("ordinary_refusal")
        if isinstance(refusal, str):
            refusal = json.loads(refusal)
        if isinstance(refusal, Mapping):
            raise PreferenceRefusal(refusal)
        raise CausalUnsupportedError(
            "this value contract has no ordinary form", reason_code="route_not_supported"
        )

    _compile_form = _engine

    def _wire(self) -> dict[str, Any]:
        """The ordinary contract wire: the contract itself, or the ordinary contract its value
        binds to. A value with no ordinary form refuses natively."""
        if self.value is None:
            return {
                "version": 1,
                "actions": [a._wire() for a in self.actions],
                "utility_units": self.utility_units,
                "criterion": self._criterion()._wire(),
                "constraints": [c._wire() for c in self.constraints],
                "target_population": self.target_population,
                "horizon": self.horizon,
                "structural_policy": self.structural_policy,
            }
        described = self._describe()
        ordinary = described.get("ordinary")
        if ordinary is not None:
            return dict(ordinary)
        refusal = described.get("ordinary_refusal")
        if refusal is None:
            raise CausalUnsupportedError(
                "this value contract has no ordinary form", reason_code="route_not_supported"
            )
        raise PreferenceRefusal(json.loads(refusal) if isinstance(refusal, str) else refusal)

    def _criterion(self) -> Criterion:
        return self.criterion or Criterion.expected_utility()

    def _ordinary(self) -> Contract:
        """The ordinary contract this contract decides as."""
        return self if self.value is None else Contract._from_wire(self._wire())

    def _normalized(self) -> dict[str, Any]:
        normalized, refusal = _normalize(json.dumps(self._wire()))
        raise_refusal(refusal, DecisionRefusal)
        assert normalized is not None
        return dict(json.loads(normalized))

    # -- what the contract states ---------------------------------------------------

    @property
    def form(self) -> Literal["ordinary", "tradeoff"]:
        """``ordinary`` when the contract is decided as an ordinary contract (no value, or a
        value whose trade-offs are plain numbers), ``tradeoff`` when it is decided across the
        admissible values of its trade-offs."""
        return self._engine()

    @property
    def route(self) -> Literal["exact", "certified"] | None:
        """How a decision across trade-offs is answered: ``exact`` (linear programs) or
        ``certified`` (interval subdivision); ``None`` for an ordinary contract."""
        if self._engine() == "ordinary":
            return None
        return self._describe()["route"]

    @property
    def trade_offs(self) -> tuple[TradeOffRange, ...]:
        """Each named trade-off with its declared range and assumption (empty without one)."""
        if self.value is None:
            return ()
        from ._decision_compile import trade_off_ranges

        return trade_off_ranges(self._preferences()["model"]["parameters"])

    @property
    def identity(self) -> str:
        """Canonical digest: unchanged by reordering actions, changed by any semantic edit.

        A value contract decided as an ordinary contract has the identity of that ordinary
        contract; one decided across trade-offs has the identity of the trade-off contract.
        """
        if self._engine() == "tradeoff":
            return str(self._describe()["identity"])
        return str(self._normalized()["identity"])

    # -- deciding -------------------------------------------------------------------

    @outcome_operation
    def evaluate(
        self,
        source: JointDistributionArtifact | BoundExternalClaim | MeanSource | Any,
        *,
        measure: Measure | None = None,
        admissibility: AdmissibilityRules | None = None,
        tie_tolerance: float | None = None,
        vertex_cap: int | None = None,
        max_boxes: int | None = None,
        min_width_fraction: float | None = None,
        program: ProgramBinding | None = None,
        outcome_units: str | None = None,
        dose_units: str | None = None,
        population: str = "target",
        execution: Any = None,
    ) -> Decision | RobustDecision | TradeoffDecision:
        """Evaluate on aligned joint draws, a bound external response grid, a mean source,
        an analysis result or several causal structures.

        ``source`` is one of:

        * a :class:`~antecedent.joint_distribution.JointDistributionArtifact` (aligned draws:
          every criterion, replayable), a sequence of them whose rows pair (:func:`joint_law`),
          or a :class:`~antecedent.propagation_chain.PropagationChain` with joint outputs
          (:func:`chain_law`);
        * a :class:`~antecedent.external.BoundExternalClaim` or a :class:`MeanSource` (one
          mean per coordinate: the expectation of an affine utility only), or the
          :class:`~antecedent.decision_evidence.JointEvidence` of one experiment (its arm
          means, :func:`experiment_means`), or the :class:`MultiArmEvidence` of one
          multi-arm randomized experiment (one ``RandomizedEffect`` result per outcome,
          :func:`multi_arm_means`): one mean per arm, variances left behind, and the
          results must share one assignment. Passing one multi-arm result directly still
          reads its scalar contrast and refuses;
        * a checked static AverageEffect result: its single ``mean_difference`` contrast
          coordinate is an affine mean input, never two reconstructed outcome means;
        * a response-curve analysis result (the :class:`~antecedent.results.CausalResponseView`
          that ``antecedent.analyze(..., query=ResponseCurve(...))`` returns). Its response
          values are turned into a native mean claim through
          :func:`antecedent.program_claims.native_claim` and evaluated as a mean source, so
          the same restrictions hold and the result is not replayable. Units are declared once: ``analyze`` can retain a program binding, or the
          contract supplies an unambiguous outcome/unit/population declaration. Undeclared
          dose units mean ``native_numeric_scale``: the original numeric intervention
          scale, with no physical unit claim or conversion. Override with ``program=`` (a
          :class:`~antecedent.program_claims.ProgramBinding`) or ``outcome_units=`` and
          ``dose_units=`` (``population=`` names the target population). A result that is not
          a point-identified response curve refuses with a typed
          :class:`~antecedent.external.ExternalRefusal` or
          :class:`~antecedent.errors.CausalUnsupportedError` naming what is missing, and a
          contract that needs a joint law (a probability, a quantile, a nonlinear utility, a
          hard constraint) refuses ``decision_contract_unsatisfied`` because a mean
          response retains no outcome draws;
        * :class:`Structures`, to decide across several causal structures as well.

        An ordinary contract returns a :class:`Decision`, or across :class:`Structures` a
        :class:`~antecedent.decision_robust.RobustDecision` under ``admissibility=`` (a
        :class:`~antecedent.decision_robust.AdmissibilityRules`, none by default). A contract
        across trade-offs returns a :class:`TradeoffDecision`; ``measure`` weighs trade-off
        values for :meth:`TradeoffPolicy.weighted` and reports shares, and
        ``tie_tolerance``, ``vertex_cap``, ``max_boxes`` and ``min_width_fraction`` tune its
        routes. An option the engine does not read refuses with ``option_not_applicable``.

        Raises:
            CausalTypeError: ``source`` is none of the accepted types.
            CausalValueError: scientific declarations are ambiguous or conflict with a binding.

        A :class:`~antecedent.external.BoundExternalClaim` supplies one mean per
        coordinate, so it answers only the expectation of an affine utility: a
        nonlinear utility, a hard constraint or any other criterion refuses with
        ``decision_contract_unsatisfied`` (``decision_evaluation.mean_source_insufficient``).
        Regret and EVPI are unavailable for such a decision.
        """
        if self._engine() == "tradeoff":
            if admissibility is not None:
                raise _not_applicable(
                    "admissibility= applies to an ordinary contract across Structures; a "
                    "decision across trade-offs reads its structural_policy"
                )
            from ._decision_tradeoff import evaluate_tradeoff

            return evaluate_tradeoff(
                self,
                source,
                measure=measure,
                tie_tolerance=tie_tolerance,
                vertex_cap=vertex_cap,
                max_boxes=max_boxes,
                min_width_fraction=min_width_fraction,
                program=program,
                outcome_units=outcome_units,
                dose_units=dose_units,
                population=population,
                execution=execution,
            )
        tuning = zip(
            _TUNING,
            (measure, tie_tolerance, vertex_cap, max_boxes, min_width_fraction),
            strict=True,
        )
        given = [f"{name}=" for name, value in tuning if value is not None]
        if given:
            raise _not_applicable(
                f"{', '.join(given)} apply to a decision across trade-off ranges; this "
                "contract is decided as an ordinary contract"
            )
        binding = {
            "program": program,
            "outcome_units": outcome_units,
            "dose_units": dose_units,
            "population": population,
        }
        return self._evaluate_ordinary(source, admissibility, binding, execution)

    def _evaluate_ordinary(
        self,
        source: Any,
        admissibility: AdmissibilityRules | None,
        binding: Mapping[str, Any],
        execution: Any,
    ) -> Decision | RobustDecision:
        from ._decision_sources import resolve_source

        contract = self._ordinary()
        declared = None if contract is self else self
        source = resolve_source(
            source,
            quantities=tuple(q for action in self.actions for q in action.quantities),
            decision_source=lambda claim: claim.as_decision_source(contract),
            execution=execution,
            **binding,
        )
        _refuse_if_cancelled(execution, "before the decision was evaluated")
        if isinstance(source, Structures):
            from .decision_robust import AdmissibilityRules, AdmissibleContract, robust

            decision = require_success(
                robust(
                    AdmissibleContract(contract, admissibility or AdmissibilityRules()),
                    source.claims,
                    kind=source.kind,
                )
            )
            decision._declared = self
            return decision
        if admissibility is not None:
            raise _not_applicable(
                "admissibility= applies to a decision across Structures; pass "
                "decision.Structures(...) as the source"
            )
        wire = json.dumps(contract._wire())
        if isinstance(source, BoundExternalClaim):
            identity = source.identity_fields
            result, refusal = _evaluate_means(
                wire,
                json.dumps([q._wire() for q in source.quantities]),
                [float(v) for v in source.values],
                str(identity["provider_id"]),
                str(identity["snapshot_id"]),
                str(identity["causal_contract_id"]),
            )
            raise_refusal(refusal, DecisionRefusal)
            assert result is not None
            return Decision(contract, source, json.loads(result), declared=declared)
        if isinstance(source, MeanSource):
            original = source._validated_original()
            effect = None
            if source._original_effect is not None:
                _, value, units, population = source._original_effect
                effect = (original, value, units, population)
            result, refusal = _evaluate_means(
                wire,
                json.dumps([q._wire() for q in source.coordinates]),
                list(source.means),
                source.provider_id,
                source.snapshot_id,
                source.causal_contract_id,
                effect,
                getattr(source._original_native, "_native", None),
            )
            raise_refusal(refusal, DecisionRefusal)
            assert result is not None
            return Decision(contract, source, json.loads(result), declared=declared)
        if not isinstance(source, JointDistributionArtifact):
            hint = (
                "; a NativeClaim is not a source itself: pass "
                "claim.as_decision_source(contract).source"
                if type(source).__name__ == "NativeClaim"
                else ""
            )
            raise CausalTypeError(
                "Contract.evaluate needs a JointDistributionArtifact (aligned draws), a "
                "BoundExternalClaim or MeanSource (means), or a supported response/effect analysis "
                f"result (antecedent.analyze(...)), not {type(source).__name__}{hint}"
            )
        result, refusal = _evaluate(wire, source._native)
        raise_refusal(refusal, DecisionRefusal)
        assert result is not None
        return Decision(contract, source, json.loads(result), declared=declared)

    @outcome_operation
    def compile(
        self,
        source: Any,
        *,
        draws: int | None = None,
        seed: int | None = None,
        shared: Mapping[tuple[int, int], Any] | None = None,
        refits: int | None = None,
        sampling: str | None = None,
        execution: Any = None,
    ) -> CompiledDecision:
        """Bind this contract to the analysis that estimates its quantities and compile it.

        ``source`` is the analysis, prepared with :func:`antecedent.prepare`: one prepared
        study (its mean response curve, under its structure or each structure of its graph
        posterior), a sequence of prepared studies (one outcome each), or a compiled design of
        several outcomes (``joint_panel.prepare_native_joint_design``). Every quantity an
        action reads must be a coordinate the analysis targets. The decision is lowered into
        the analysis's own plan, so the :class:`CompiledDecision` states the quantities, the
        source the contract needs, the trade-off ranges (when there are any) and the
        identification before anything is fitted; what cannot be answered (a quantity not
        identified, a source the analysis cannot supply, studies that share observations
        without a declared covariance) is refused here. :meth:`CompiledDecision.execute` then
        runs each plan once. An ordinary contract (or a value whose trade-offs are plain
        numbers) compiles as an ordinary contract; a value with named trade-offs compiles as
        a decision across them.

        Several studies meet by their means when means suffice. A contract that needs one law
        of their outcomes needs ``draws=`` (the aligned rows the propagation chain draws)
        and, for studies that share observations, ``shared=``: the declared covariance
        between the estimates of each such pair, keyed ``(i, j)`` by the studies' positions
        in ``source``. Each block is a 2-D array-like with one row per coordinate of study
        ``i`` and one column per coordinate of study ``j``, both in grid order; a block keyed
        ``(j, i)`` is read as its transpose. A static-regime design reads its arm means, or
        with ``refits=`` (at least 99) its aligned whole-subject refits. A cluster-randomized
        design or a matched-pair or switchback experiment reads its arm means; randomization
        alone does not identify the dependence of its arm-mean estimators across arms, so a
        contract that needs one law of its outcomes also needs ``sampling=``, the assumption
        id of the premise that its independent units (clusters, pairs or sequences) were
        sampled independently from a larger population, with ``refits=`` for the
        independent-unit bootstrap. The premise enters the plan and the result's assumptions.

        ``seed`` seeds the chain's draws and the refits; by default it is the seed of
        ``execution`` (``Execution().seed`` when none is given), so ``execution=`` alone
        reproduces a compile. An argument the source does not read refuses with
        ``option_not_applicable`` rather than being dropped: ``draws=`` and ``shared=`` are
        read only for several studies, ``refits=`` and ``sampling=`` only for a design, and
        ``seed=`` only when something is drawn (``draws=``, ``shared=`` or ``refits=``).
        """
        from ._decision_compile import compile_contract

        return compile_contract(
            self,
            source,
            draws=draws,
            seed=seed,
            shared=shared,
            refits=refits,
            sampling=sampling,
            execution=execution,
        )

    @outcome_operation
    def evaluate_transported(
        self,
        response: Any,
        *,
        analysis: Any,
        demand: Literal["expected_utility", "law", "interval"] = "expected_utility",
        support_policy: Any = None,
    ) -> Any:
        """Decide on a :class:`~antecedent.transported_response.TransportedResponse`.

        The result is a :class:`~antecedent.transported_response.TransportedDecision` that names
        the transport beside every number and is never presented as a decision on native
        target data. See :func:`antecedent.transported_response.evaluate_transported`. A
        contract across trade-off ranges has no ordinary form and refuses.
        """
        from .transported_response import evaluate_transported

        contract = self._ordinary()
        return evaluate_transported(
            contract,
            require_success(response),
            analysis=require_success(analysis),
            demand=demand,
            support_policy=support_policy,
        )

    @artifact_export
    def export(self, *, artifact_id: str = "decision-contract") -> ExportedArtifact:
        """The ordinary contract as a portable artifact. A value contract decided as an ordinary
        contract exports that ordinary contract, byte for byte; a contract across trade-off
        ranges has no ordinary form and refuses."""
        return artifact_from_receipt(
            self, bytes(_export_contract(json.dumps(self._wire()), artifact_id))
        )

    @classmethod
    def load(
        cls, data: bytes, *, expected_identity: str, retained_identity: str | None = None
    ) -> Contract:
        """Load only under the identity the consumer retained independently."""
        _retained.admit(
            data,
            retained_identity,
            "antecedent.decision.Contract.load",
            "antecedent.decision.Contract.load(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
            bound=True,
        )
        data = container_bytes(data)
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


class Decision(DecisionResult):
    """A decision result with the evidence behind it; exportable and replayable.

    ``verdict``, ``selected`` and the per-action table are the ordinary engine's own; ``choice``,
    ``kind``, ``standing`` and ``statement`` read it the way every decision result is read.
    """

    def __init__(
        self,
        contract: Contract,
        source: JointDistributionArtifact | BoundExternalClaim | MeanSource | Any,
        body: Mapping[str, Any],
        *,
        declared: Contract | None = None,
    ) -> None:
        self._contract = contract
        self._source = source
        self._body = body
        self._declared = declared

    @property
    def kind(self) -> str:
        """The verdict word: ``uniquely_optimal``, ``indistinguishable`` or
        ``no_admissible_action``."""
        return self.verdict.kind

    @property
    def choice(self) -> str | None:
        """The leader when it is uniquely optimal; ``None`` for a tie or no admissible action."""
        verdict = self.verdict
        return verdict.actions[0] if verdict.kind == "uniquely_optimal" else None

    @property
    def standing(self) -> Standing:
        """The verdict in the vocabulary every decision result shares."""
        return _standing(self._body)

    @property
    def statement(self) -> str:
        """The verdict in plain words."""
        return str(self._body["statement"])

    @property
    def effective_tie_tolerance(self) -> float:
        """Ordinary criterion values tie only when exactly equal."""
        return 0.0

    @property
    def contract(self) -> Contract:
        """The contract as declared (a value contract when it was declared with ``value=``)."""
        return self._contract if self._declared is None else self._declared

    @property
    def source(self) -> Any:
        """The immutable scientific source, including retained adapter provenance."""
        return self._source

    @property
    def answer(self) -> Answer:
        """The native decision verdict, without treating a chosen action as an effect."""
        return Answer(
            "structured",
            detail="decision." + self.verdict.kind,
            structured={"selected": self.selected, "contract_identity": self.contract_identity},
        )

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

        declared = (
            frozenset(c.id for c in self._declared.constraints)
            if self._declared is not None
            else frozenset()
        )

        def constraint_id(action: str, bound: str) -> str:
            # A value contract's constraint that reads an input one action knows exactly is
            # bound per action as "{id}@{action}"; it is reported under the id it was declared
            # with, as the trade-off routes report it.
            base, _, suffix = bound.rpartition("@")
            split = suffix == action and base in declared and bound not in declared
            return base if split else bound

        return tuple(
            ActionOutcome(
                id=a["id"],
                admissible=a["admissible"],
                exclusions=tuple(
                    ConstraintExclusion(
                        constraint_id(a["id"], e["constraint_id"]), e["probability"], e["required"]
                    )
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

    @property
    def original_execution(self) -> AteAnalysisResult | None:
        """Original native scalar execution behind a contrast mean, or ``None``.

        Keeps its checked identification, diagnostics, support and uncertainty
        available for inspection. The decision is an affine point ranking, not an
        interval for utility; the original effect interval is never transferred.
        """
        if isinstance(self._source, MeanSource) and self._source._original_effect is not None:
            self._source._validated_original()
            return self._source._original_effect[0]._raw
        return None

    @property
    def source_evidence(self) -> tuple[SourceEvidence, ...]:
        """Original external support/request ancestry; no inferred numeric diagnostic or native authority."""
        sources: list[SourceEvidence] = []
        if isinstance(self._source, BoundExternalClaim):
            sources.append(self._source.source_evidence)
        elif isinstance(self._source, MeanSource) and self._source._original_native is not None:
            self._source._validated_original()
            sources.append(self._source._original_native.source_evidence)
        elif (
            isinstance(self._source, JointDistributionArtifact)
            and self._source.source_evidence is not None
        ):
            sources.append(self._source.source_evidence)
        return tuple(
            source.project(self._contract, [action.id for action in self.actions])
            for source in sources
        )

    @property
    def lineage(self) -> tuple[LineageLink, ...]:
        """Derivation chain behind the decision, parents before children.

        For a joint-draw source Rust derives it (contract, provider, distribution,
        decision contract, ``decision_result``). For an external claim it is the
        claim's own lineage plus the decision contract and ``decision_result``,
        re-chained natively so every link, including the two appended ones, has
        its Merkle digest (Python cannot hash).
        """
        if isinstance(self._source, (BoundExternalClaim, MeanSource)) and self._source.lineage:
            decision_id = f"decision:{self.contract_identity}"
            rows = [
                [link["id"], link["stage"], list(link["parents"])]
                if isinstance(link, Mapping)
                else [link.id, link.stage, list(link.parents)]
                for link in self._source.lineage
            ]
            rows.append([decision_id, "decision_contract", []])
            rows.append([RESULT_LINK_ID, "claim", [rows[-2][0], decision_id]])
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
        return tuple(
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
            for item in self._body["lineage"]
        )

    def stages_behind(self, link: str = RESULT_LINK_ID) -> frozenset[str]:
        """Stages standing behind ``link`` (default: the reported decision)."""
        by_id = {item.id: item for item in self.lineage}
        if link not in by_id:
            raise CausalValueError(f"unknown lineage link {link!r}")
        seen: set[str] = set()
        stack = [link]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(by_id[current].parents)
        return frozenset(by_id[item].stage for item in seen)

    def explain(self) -> str:
        """Why this action, or why none: the choice, its evidence and its limits."""
        verdict = self.verdict
        by_id = {a.id: a for a in self.actions}
        criterion = self._contract._criterion().kind.replace("_", " ")
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
            text += f"; it ties {others} at the same criterion value"
        excluded = [a for a in self.actions if not a.admissible]
        if excluded:
            text += "; excluded by a hard constraint: " + ", ".join(repr(a.id) for a in excluded)
        if self.evpi is not None:
            text += f"; perfect information would be worth {self.evpi:.4g}"
        if isinstance(self._source, BoundExternalClaim):
            text += (
                "; values came from an external mean grid supplied by "
                f"{self._source.provenance_label}, not estimated natively, and carry no "
                "sampling error"
            )
        if any("point ranking" in assumption for assumption in self.assumptions):
            text += "; sampled-law point ranking only; Monte Carlo error is not certified"
        return text + "."

    @artifact_export
    def export(self, *, artifact_id: str = "decision-result") -> ExportedArtifact:
        """The result bound to its contract identity and source digest, replayable.

        Only a joint-draw source can be exported and replayed; a decision computed
        from an external or native mean grid refuses with ``route_not_supported``.
        """
        if not isinstance(self._source, JointDistributionArtifact):
            raise DecisionRefusal(
                _not_replayable(
                    "mean"
                    if isinstance(self._source, (BoundExternalClaim, MeanSource))
                    else type(self._source).__name__,
                    "export a decision evaluated on aligned joint draws",
                )
            )
        return artifact_from_receipt(
            self,
            bytes(
                _export_result(
                    json.dumps(self._contract._wire()), self._source._native, artifact_id
                )
            ),
        )

    def __repr__(self) -> str:
        return f"<Decision {self.verdict.kind} {self.selected}>"


@outcome_operation
@artifact_consumer(
    proof="Independent native decision replay rebuilds declared inputs and compares stored identity, numbers and standing.",
    replayable=True,
    identity=("contract", "source"),
    input="data",
)
def replay(
    data: bytes,
    *,
    contract: Any,
    source: JointDistributionArtifact,
    execution: Any = None,
    retained_identity: str | None = None,
) -> Decision | TradeoffDecision:
    """Recompute a stored decision from the consumer's own contract and draws.

    ``retained_identity`` (``antecedent.artifacts.artifact_identity`` of the bytes the producer
    wrote) binds the bytes themselves before they are read; bytes other than the retained
    artifact are refused with ``external_binding_mismatch``. A result computed under a different
    contract or from different draws, or one whose numbers were edited, refuses. An ordinary decision (including a value contract decided as
    an ordinary contract) is recomputed and must match exactly. A decision across trade-off
    ranges reuses its stored measure and options and must match within the relative tolerance
    its linear-program and interval numerics need; its recomputation reads the execution's
    cancellation token at its work boundaries, and a cancelled replay is refused with
    ``cancelled_no_claim``.
    """
    _retained.admit(
        data,
        retained_identity,
        "antecedent.decision.replay",
        "antecedent.decision.replay(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
        bound=True,
    )
    data = container_bytes(data)
    source = require_success(source)
    if not isinstance(contract, Contract):
        raise CausalTypeError(f"replay needs a decision.Contract, not {type(contract).__name__}")
    if not isinstance(source, JointDistributionArtifact):
        raise CausalTypeError(
            "replay needs the JointDistributionArtifact of aligned draws the result was computed "
            f"from, not {type(source).__name__}; only a joint-draw decision replays"
        )
    if contract._engine() == "tradeoff":
        from ._decision_tradeoff import replay_tradeoff

        return replay_tradeoff(
            data, contract, source, execution, retained_identity=_retained.native(retained_identity)
        )
    ordinary = contract._ordinary()
    _refuse_if_cancelled(execution, "before the decision was replayed")
    result, refusal = _replay(data, json.dumps(ordinary._wire()), source._native)
    raise_refusal(refusal, DecisionRefusal)
    assert result is not None
    return Decision(
        ordinary, source, json.loads(result), declared=None if ordinary is contract else contract
    )


def _refuse_if_cancelled(execution: Any, where: str) -> None:
    """The ordinary evaluators run in one native call that reads no cancellation token, so a
    cancelled execution is refused once before they run."""
    cancel = getattr(execution, "cancel", None)
    if cancel is not None and cancel.is_cancelled():
        raise CausalCancelledError(f"cancelled {where}")


def source_digest(source: JointDistributionArtifact) -> str:
    """Digest of a source's aligned draws, to retain alongside a result."""
    if not isinstance(source, JointDistributionArtifact):
        raise CausalTypeError(
            f"source_digest needs a JointDistributionArtifact, not {type(source).__name__}"
        )
    return str(_source_digest(source._native))


@dataclass(frozen=True, slots=True)
class PublishedScore:
    """An affine decision score and the carrier of the published answer that produced it."""

    value: float
    """``slope * location + intercept``."""
    carrier: str
    """``parameter_posterior`` or ``mean``; never a confidence interval."""


@outcome_operation
def score_published_answer(
    demand: Literal["affine_posterior", "affine_mean", "nonlinear", "risk"],
    *,
    mean: float,
    posterior_mean: float | None = None,
    slope: float | None = None,
    intercept: float | None = None,
) -> PublishedScore:
    """Score a decision demand from what an answer published.

    An ``affine_posterior`` demand (utility ``slope * answer + intercept``) reads the posterior
    mean and needs the answer to be a posterior; an ``affine_mean`` demand reads the published
    point answer ``mean`` and needs no posterior. A ``nonlinear`` utility or an individual outcome
    ``risk`` cannot be carried by a mean.

    Raises:
        DecisionRefusal: the answer cannot publish the carrier the demand needs; the refusal
            names the missing carrier (``parameter_posterior`` or ``outcome_law``).
    """
    scored, missing = _score_published(
        demand,
        real(mean, "mean", finite=True),
        None if posterior_mean is None else real(posterior_mean, "posterior_mean", finite=True),
        None if slope is None else real(slope, "slope", finite=True),
        None if intercept is None else real(intercept, "intercept", finite=True),
    )
    if scored is None:
        raise DecisionRefusal(
            {
                "code": "distribution_meaning_mismatch",
                "stage": "score_published_answer",
                "detail": "decision.carrier_missing",
                "message": (
                    f"the {demand} demand needs the {missing} carrier, which the published "
                    "answer does not carry"
                ),
                "offending": missing,
                "remedy": (
                    "demand a parameter posterior or an outcome law, or declare the utility's "
                    "affine form"
                ),
            }
        )
    value, carrier = scored
    return PublishedScore(value=value, carrier=carrier)


_LAZY: dict[str, tuple[str, str]] = {
    "TradeoffDecision": ("_decision_tradeoff_results", "TradeoffDecision"),
    "ActionRecord": ("_decision_tradeoff_results", "ActionRecord"),
    "Switch": ("_decision_tradeoff_results", "Switch"),
    "SwitchThreshold": ("_decision_tradeoff_results", "SwitchThreshold"),
    "Region": ("_decision_tradeoff_results", "Region"),
    "StructureOutcome": ("_decision_tradeoff_results", "StructureOutcome"),
    "ClassSampling": ("_decision_tradeoff_results", "ClassSampling"),
    "SampledMass": ("_decision_tradeoff_results", "SampledMass"),
    "SampledMargin": ("_decision_tradeoff_results", "SampledMargin"),
    "CompiledDecision": ("_decision_compile", "CompiledDecision"),
    "CompiledQuantity": ("_decision_compile", "CompiledQuantity"),
    "TradeOffRange": ("_decision_compile", "TradeOffRange"),
    "JointLawInput": ("_decision_compile", "JointLawInput"),
    "JointLawPlan": ("_decision_compile", "JointLawPlan"),
    "DesignLawPlan": ("_decision_compile", "DesignLawPlan"),
    "joint_law": ("_decision_sources", "joint_law"),
    "chain_law": ("_decision_sources", "chain_law"),
    "MultiArmEvidence": ("_decision_sources", "MultiArmEvidence"),
    "experiment_means": ("_decision_sources", "experiment_means"),
    "experiment_quantity": ("_decision_sources", "experiment_quantity"),
    "multi_arm_means": ("_decision_sources", "multi_arm_means"),
    "multi_arm_quantity": ("_decision_sources", "multi_arm_quantity"),
}


def __getattr__(name: str) -> Any:
    """Load the result, compile and source names on first use, so importing this module does
    not load the trade-off engine."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute_name = target
    value = getattr(importlib.import_module(f"{__package__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "Action",
    "ActionOutcome",
    "ActionRecord",
    "Attribute",
    "ClassSampling",
    "CompiledDecision",
    "CompiledQuantity",
    "CompiledReceipt",
    "Constraint",
    "ConstraintExclusion",
    "Contract",
    "Criterion",
    "Curve",
    "Decision",
    "DecisionRefusal",
    "DecisionResult",
    "DesignLawPlan",
    "Expr",
    "JointLawInput",
    "JointLawPlan",
    "MeanSource",
    "Measure",
    "MultiArmEvidence",
    "PreferenceRefusal",
    "PublishedScore",
    "Region",
    "Relation",
    "SampledMargin",
    "SampledMass",
    "Standing",
    "StructureOutcome",
    "Structures",
    "Switch",
    "SwitchThreshold",
    "TradeOffRange",
    "Tradeoff",
    "TradeoffDecision",
    "TradeoffPolicy",
    "Value",
    "ValueExpr",
    "Verdict",
    "attribute",
    "both",
    "chain_law",
    "const",
    "excess",
    "exp",
    "experiment_means",
    "experiment_quantity",
    "interaction",
    "joint_law",
    "ln",
    "maximum",
    "minimum",
    "multi_arm_means",
    "multi_arm_quantity",
    "relation",
    "replay",
    "score_published_answer",
    "segment",
    "shortfall",
    "source_digest",
    "tradeoff",
    "x",
]
