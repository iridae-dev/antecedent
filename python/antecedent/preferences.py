"""Decide when the trade-offs between outcomes are stated as ranges rather than single numbers.

The objects here are those of :mod:`antecedent.decision`, under the names this module has
always used (``ValueModel`` for :class:`~antecedent.decision.Value`, ``strength`` and ``fixed``
for :func:`~antecedent.decision.tradeoff`, ``Parameter`` for
:class:`~antecedent.decision.Tradeoff`, ``Expr`` for :class:`~antecedent.decision.ValueExpr`,
``Policy`` for :class:`~antecedent.decision.TradeoffPolicy`, ``PreferenceDecision`` for
:class:`~antecedent.decision.TradeoffDecision`). Each trade-off is a named parameter with a
declared range (:func:`strength`) or a fixed value (:func:`fixed`), and every range names the
judgment it stands on. The decision is answered across every admissible value of the
trade-offs at once, and, when the analysis supplies several causal structures, across those
structures too::

    from antecedent import preferences as pf

    benefit = pf.attribute("benefit", units="qaly")
    harm = pf.attribute("harm", units="events")
    w = pf.strength("harm_weight", units="qaly/event", lower=0.5, upper=4.0,
                    assumption="clinical_panel_2026")
    model = pf.ValueModel(benefit - w * harm, utility_units="qaly")
    contract = pf.Contract(
        model=model,
        actions=[
            pf.Action("treat", inputs={"benefit": treat_benefit, "harm": treat_harm}),
            pf.Action("wait", inputs={"benefit": wait_benefit, "harm": 0.0}),
        ],
        target_population="target",
    )
    decision = antecedent.require_success(contract.evaluate(joint_draws))
    print(decision.statement)      # e.g. "Action treat is preferred for every admissible value ..."
    for switch in decision.switches:
        print(switch.first, switch.second, switch.threshold)

What a result is and is not:

* A range of trade-off values is a declared assumption, not a probability. A verdict over the
  range says what holds for every admissible value, or where the choice changes; it never says
  how likely a value is.
* A share is reported only for a supplied :class:`Measure` and is that measure's weight, not a
  probability of anyone's preferences unless the measure is one.
* When the value is affine in the trade-offs and the criterion is an expectation, the verdict is
  exact (linear programs). Otherwise the set is subdivided into boxes and each box is certified
  with interval arithmetic; part of the set may be left undecided, and the result says how much.

A :class:`Contract` here is always decided across its trade-offs, even when every one is fixed;
in it ``decision.x(k)`` in a hard constraint reads the value's ``k``-th attribute.
``decision.Contract(value=...)`` is the same declaration in the decision module, where a value
whose trade-offs are plain numbers is decided as an ordinary contract.

Stating the trade-offs is the whole capability. Learning them from recorded choices is optional
and lives in :mod:`antecedent.decision_learning`; anything it produces plugs in here as
``Contract(preferences=...)``. This module does not depend on it.

Rust owns validation, identity, the route, every verdict, artifacts and refusals; a refusal is a
:class:`PreferenceRefusal` carrying the registered ``code`` and ``detail``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from ._decision_compile import (
    CompiledDecision,
    CompiledQuantity,
    DesignLawPlan,
    JointLawInput,
    JointLawPlan,
    TradeOffRange,
    compile_contract,
    trade_off_ranges,
)
from ._decision_sources import chain_law, experiment_means, experiment_quantity, joint_law
from ._decision_tradeoff import (
    describe,
    evaluate_tradeoff,
    resolve_tradeoff_source,
    tradeoff_measure,
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
)
from ._decision_tradeoff_results import TradeoffDecision as PreferenceDecision
from ._decision_value import (
    Attribute,
    Curve,
    Measure,
    Relation,
    StructuralPolicy,
    StructuralPolicyKind,
    Structures,
    ValueModel,
    _constraint_node,
    _constraint_node_of,
    attribute,
    both,
    excess,
    exp,
    interaction,
    ln,
    maximum,
    minimum,
    relation,
    segment,
    shortfall,
    tradeoff,
)
from ._decision_value import Tradeoff as Parameter
from ._decision_value import TradeoffPolicy as Policy
from ._decision_value import ValueExpr as Expr
from ._validate import text
from .decision import Action, Constraint, Criterion, PreferenceRefusal, replay
from .decision import Expr as _DecisionExpr
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .outcomes import outcome_operation

if TYPE_CHECKING:
    from .program_claims import ProgramBinding


def strength(id: str, *, units: str, lower: float, upper: float, assumption: str) -> Parameter:
    """A trade-off whose admissible values are ``[lower, upper]``
    (:func:`antecedent.decision.tradeoff` with a range).

    ``units`` is utility per unit of what it multiplies (``qaly/event``); ``assumption`` names
    the judgment the range stands on. Every value in the range is treated as admissible; none
    is more likely than another.
    """
    return tradeoff(id, units=units, lower=lower, upper=upper, assumption=assumption)


def fixed(id: str, *, units: str, value: float, assumption: str = "declared_value") -> Parameter:
    """A trade-off known as one value, a range of width zero
    (:func:`antecedent.decision.tradeoff` with ``value=``)."""
    return tradeoff(id, units=units, value=value, assumption=assumption)


def constraint(
    id: str,
    expr: Any,
    *,
    at_most: float,
    units: str,
    min_probability: float = 1.0,
    applies_to: Sequence[str] = (),
) -> Constraint:
    """A hard constraint ``P(expr <= at_most) >= min_probability`` over attributes.

    It never reads a trade-off parameter, so whether it excludes an action does not depend on
    preferences. Returns a :class:`antecedent.decision.Constraint` to pass to ``constraints=``.
    """
    return Constraint(
        id=id,
        expr=_DecisionExpr(_constraint_node_of(expr)),
        bound=at_most,
        units=units,
        min_probability=min_probability,
        applies_to=tuple(applies_to),
    )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Contract:
    """A decision whose value comes from a model with explicit trade-offs, always decided across
    them.

    Give the model as ``model=`` (a :class:`ValueModel`), or as ``preferences=``: any object
    exposing ``_preference_wire()``, such as a :class:`ValueModel` or the accepted preferences
    of :mod:`antecedent.decision_learning` (which then also supply their measure, if any).
    ``criterion`` and ``constraints`` are those of :mod:`antecedent.decision`; a constraint reads
    attributes (build one with :func:`constraint`, or with ``decision.x(k)`` for attribute
    ``k``). ``policy`` says how the answer across the trade-offs is consumed. The same
    declaration is ``decision.Contract(value=...)``.
    """

    actions: tuple[Action, ...]
    target_population: str
    model: ValueModel | None = None
    preferences: Any = None
    criterion: Criterion = field(default_factory=Criterion.expected_utility)
    policy: Policy = field(default_factory=Policy.invariant_best)
    constraints: tuple[Constraint, ...] = ()
    horizon: int = 0
    structural_policy: StructuralPolicy | StructuralPolicyKind = field(
        default_factory=StructuralPolicy.report_only
    )

    def __post_init__(self) -> None:
        if (self.model is None) == (self.preferences is None):
            raise CausalValueError("a Contract takes model= or preferences=, exactly one")
        if self.model is not None and not isinstance(self.model, ValueModel):
            raise CausalTypeError("Contract.model is a ValueModel")
        if self.preferences is not None and not callable(
            getattr(self.preferences, "_preference_wire", None)
        ):
            raise CausalTypeError(
                "Contract.preferences is a ValueModel or accepted preferences "
                "(an object with _preference_wire())"
            )
        if isinstance(self.actions, (str, bytes)) or not isinstance(self.actions, Iterable):
            raise CausalTypeError("Contract.actions is a sequence of Action")
        actions = tuple(self.actions)
        if any(not isinstance(a, Action) for a in actions):
            raise CausalTypeError("Contract.actions holds only Action values")
        for action in actions:
            if not action._named:
                raise CausalTypeError(
                    "Contract reads every action's inputs by attribute name; action "
                    f"{action.id!r} gives positional inputs"
                )
        constraints = tuple(self.constraints)
        if any(not isinstance(c, Constraint) for c in constraints):
            raise CausalTypeError("Contract.constraints holds decision.Constraint values")
        if not isinstance(self.criterion, Criterion):
            raise CausalTypeError("Contract.criterion is a decision.Criterion")
        if not isinstance(self.policy, Policy):
            raise CausalTypeError("Contract.policy is a Policy such as Policy.maximin()")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise CausalTypeError("Contract.horizon is an integer")
        structural = self.structural_policy
        if isinstance(structural, str):
            if structural not in get_args(StructuralPolicyKind):
                raise CausalValueError(
                    f"Contract.structural_policy is a StructuralPolicy or one of "
                    f"{get_args(StructuralPolicyKind)}"
                )
            structural = StructuralPolicy(structural)
        elif not isinstance(structural, StructuralPolicy):
            raise CausalTypeError(
                "Contract.structural_policy is a StructuralPolicy such as "
                "StructuralPolicy.maximin(), or its kind as a string"
            )
        object.__setattr__(self, "structural_policy", structural)
        text(self.target_population, "Contract.target_population", non_empty=True)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "constraints", constraints)

    @property
    def _structural_policy(self) -> StructuralPolicy:
        policy = self.structural_policy
        return policy if isinstance(policy, StructuralPolicy) else StructuralPolicy(policy)

    def _preferences(self) -> dict[str, Any]:
        source = self.model if self.model is not None else self.preferences
        return dict(source._preference_wire())

    def _wire(self) -> dict[str, Any]:
        model = self._preferences()["model"]
        attributes = {a["id"]: k for k, a in enumerate(model["attributes"])}
        return {
            "model": model,
            "actions": [a._value_wire(model["attributes"]) for a in self.actions],
            "criterion": self.criterion._wire(),
            "constraints": [
                {**c._wire(), "expr": _constraint_node(c.expr._wire_value, attributes)}
                for c in self.constraints
            ],
            "target_population": self.target_population,
            "horizon": self.horizon,
            "structural_policy": self._structural_policy.kind,
            "preference_policy": self.policy.kind,
        }

    def _tradeoff_wire(self) -> dict[str, Any]:
        return self._wire()

    def _compile_form(self) -> Literal["tradeoff"]:
        return "tradeoff"

    @property
    def identity(self) -> str:
        """Canonical digest of the contract, to retain for replay."""
        return str(describe(self)["identity"])

    @property
    def route(self) -> Literal["exact", "certified"]:
        """``exact`` (linear programs) or ``certified`` (interval subdivision)."""
        return describe(self)["route"]

    @property
    def parameter_ids(self) -> tuple[str, ...]:
        """The trade-off parameters, in model order."""
        return tuple(p["id"] for p in self._preferences()["model"]["parameters"])

    @property
    def trade_offs(self) -> tuple[TradeOffRange, ...]:
        """Each trade-off with its declared range and assumption."""
        return trade_off_ranges(self._preferences()["model"]["parameters"])

    def _measure(self, measure: Measure | None) -> dict[str, Any] | None:
        return tradeoff_measure(self, measure)

    def _source(self, source: Any, **binding: Any) -> Any:
        return resolve_tradeoff_source(self, source, **binding)

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
        execution: Execution | None = None,
    ) -> CompiledDecision:
        """Bind this contract to the analysis that estimates its quantities and compile it.

        The arguments and what is refused are those of
        :meth:`antecedent.decision.Contract.compile`; the :class:`CompiledDecision` states the
        quantities, the source the contract needs, the trade-off ranges and the identification
        before anything is fitted, and :meth:`CompiledDecision.execute` returns the
        :class:`PreferenceDecision`.
        """
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
    def evaluate(
        self,
        source: Any,
        *,
        measure: Measure | None = None,
        tie_tolerance: float | None = None,
        vertex_cap: int | None = None,
        max_boxes: int | None = None,
        min_width_fraction: float | None = None,
        program: ProgramBinding | None = None,
        outcome_units: str | None = None,
        dose_units: str | None = None,
        population: str = "target",
        execution: Execution | None = None,
    ) -> PreferenceDecision:
        """Decide across every admissible value of the trade-offs, on a source already in hand.

        ``source`` and the options are those of :meth:`antecedent.decision.Contract.evaluate`
        for a contract across trade-offs: aligned joint draws (or laws whose rows pair, or a
        propagation chain), a mean source, bound external claim or the evidence of one
        experiment (means answer only the exact route), a supported analysis result, or
        :class:`Structures`. ``measure`` weighs trade-off values for :meth:`Policy.weighted` and
        reports shares; it is refused when the accepted preferences already carry one. The
        decision reads the execution's cancellation token at its work boundaries; a cancelled
        run is refused with ``cancelled_no_claim``.
        """
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


__all__ = [
    "Action",
    "ActionRecord",
    "Attribute",
    "ClassSampling",
    "CompiledDecision",
    "CompiledQuantity",
    "Contract",
    "Criterion",
    "Constraint",
    "Curve",
    "DesignLawPlan",
    "Expr",
    "JointLawInput",
    "JointLawPlan",
    "Measure",
    "Parameter",
    "Policy",
    "PreferenceDecision",
    "PreferenceRefusal",
    "Region",
    "Relation",
    "SampledMargin",
    "SampledMass",
    "StructureOutcome",
    "StructuralPolicy",
    "Structures",
    "Switch",
    "SwitchThreshold",
    "TradeOffRange",
    "ValueModel",
    "attribute",
    "both",
    "chain_law",
    "constraint",
    "excess",
    "exp",
    "experiment_means",
    "experiment_quantity",
    "fixed",
    "interaction",
    "joint_law",
    "ln",
    "maximum",
    "minimum",
    "relation",
    "replay",
    "segment",
    "shortfall",
    "strength",
]
