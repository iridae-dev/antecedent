"""Decide when the trade-offs between outcomes are stated as ranges rather than single numbers.

A decision that weighs several outcomes (benefit, harm, cost) needs to say how much of one is
worth how much of another. Here you state those trade-offs explicitly: each one is a named
parameter with a declared range (:func:`strength`) or a fixed value (:func:`fixed`), and every
range names the judgment it stands on. The decision is then answered across every admissible
value of the trade-offs at once, and, when the analysis supplies several causal structures,
across those structures too::

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

Stating the trade-offs is the whole capability. Learning them from recorded choices is optional
and lives in :mod:`antecedent.preference_learning`; anything it produces plugs in here as
``Contract(preferences=...)``. This module does not depend on it.

Rust owns validation, identity, the route, every verdict, artifacts and refusals; a refusal is a
:class:`PreferenceRefusal` carrying the registered ``code`` and ``detail``.
"""

from __future__ import annotations

from ._preference_contracts import (
    CompiledDecision,
    Contract,
    Criterion,
    chain_law,
    experiment_means,
    experiment_quantity,
    joint_law,
)
from ._preference_declarations import (
    Action,
    Attribute,
    Constraint,
    Curve,
    Expr,
    Measure,
    Parameter,
    Policy,
    PreferenceRefusal,
    Relation,
    StructuralPolicy,
    Structures,
    ValueModel,
    attribute,
    both,
    constraint,
    excess,
    exp,
    fixed,
    interaction,
    ln,
    maximum,
    minimum,
    relation,
    segment,
    shortfall,
    strength,
)
from ._preference_results import (
    ActionRecord,
    ClassSampling,
    CompiledQuantity,
    DesignLawPlan,
    JointLawInput,
    JointLawPlan,
    PreferenceDecision,
    Region,
    SampledMargin,
    SampledMass,
    StructureOutcome,
    Switch,
    SwitchThreshold,
    TradeOffRange,
    replay,
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
