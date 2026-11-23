# Decide when trade-offs are a range

Decide between actions that move several outcomes at once (benefit, harm, cost) when you can
state how those outcomes trade off only as a range. You declare a value model: the outcomes it
reads, the trade-offs between them as ranges, fixed values, orderings and curve shapes, and the
assumption each declaration stands on. Antecedent answers the decision across every trade-off
you allowed. It says which conclusions follow from the evidence and your declarations, which
depend on a judgment you have not settled, and which cannot be justified. Use this page when a
[decision contract](decisions.md) would need one exact utility and you cannot defend one.

The method is causal decision analysis under partially identified preferences. The evidence
fixes what each action does to each outcome. Your declarations fix a set of admissible
trade-offs, not one point. The answer is a statement about that whole set.

For a worked story from simulated data to a threshold, see
[Choose a discount when the cost of a return is a range](tutorials/decide-across-trade-offs.md).

## A first decision

Two actions are compared on benefit (in quality-adjusted life years) and harm (adverse events),
both supplied by the source. How much benefit one adverse event costs is stated as a range.

```python
from antecedent import preferences as pf
from antecedent.outcomes import require_success

benefit = pf.attribute("benefit", units="qaly")
harm = pf.attribute("harm", units="events")
harm_weight = pf.strength(
    "harm_weight", units="qaly/event", lower=0.5, upper=4.0, assumption="clinical_panel"
)
model = pf.ValueModel(benefit - harm_weight * harm, utility_units="qaly")

actions = (
    pf.Action("treat", inputs={"benefit": treat_benefit, "harm": treat_harm}),
    pf.Action("wait", inputs={"benefit": wait_benefit, "harm": wait_harm}),
)
contract = pf.Contract(model=model, actions=actions, target_population="target")
decided = require_success(contract.evaluate(source))
decided.kind         # "necessarily_best", "never_beaten", "preference_dependent", ...
decided.selected     # the chosen action, or None
decided.statement    # the verdict in plain words
```

`treat_benefit`, `treat_harm`, `wait_benefit` and `wait_harm` are `ScientificQuantity`
coordinates, and `source` supplies them: aligned joint draws (a `JointDistributionArtifact`), a
mean source, or a supported analysis result. A plain number in `inputs` is an input known
without uncertainty, stated in the attribute's units. Every attribute of the model gets exactly
one input.

If the treatment raises benefit by 1 and harm by 0.5 relative to waiting, the margin of `treat`
over `wait` is `1 − 0.5 × harm_weight`. Over 0.5 to 4 the verdict is `preference_dependent`,
with one switch at `harm_weight = 2`. Over 0.5 to 1.5 it is `necessarily_best` for `treat`.

What this answer is and is not:

- A range is a declared assumption. It is never a probability. Nothing on this page treats
  the middle of a range as more likely than its ends.
- The verdict is about the set you declared. Widen a range and a verdict that held can stop
  holding; narrow it and a dependent verdict can settle.
- The answer does not say what the trade-offs should be. It says what follows from the
  evidence once you have said what you are willing to assume.
- A probability over trade-offs exists only when you supply one (a [measure](#supply-your-own-weighted-viewpoints))
  or when the optional [statistical learner](preference-learning.md#two-methods) estimates one.
  Its shares and intervals are a different kind of object from the ranges.

## Declare a value model

A value model has three parts: attributes, trade-off parameters and the value expression that
combines them.

- An **attribute** is one outcome the value reads, with its units
  (`pf.attribute(id, units=...)`). Its value for each action comes from a scientific quantity or
  a fixed input.
- A **trade-off parameter** is a number the value multiplies or shapes, with units, an
  admissible range and the id of the assumption the range stands on. A parameter whose lower
  and upper ends are equal is fixed.
- The **value** is a closed expression over attributes and parameters, in the model's
  `utility_units`. Write it with `+`, `-`, `*`, division by a number, `pf.maximum`,
  `pf.minimum`, `pf.exp` and `pf.ln`.

`pf.ValueModel(value, utility_units=..., relations=...)` reads the attributes and parameters
from the expression in order of first appearance. The admissible set is every parameter point
inside the declared ranges that also satisfies the declared relations. This page calls a point
in it an *admissible trade-off*.

### Rates, fixed values and units

`pf.strength(id, units=..., lower=..., upper=..., assumption=...)` declares a ranged trade-off.
`pf.fixed(id, units=..., value=...)` declares a settled one. Units are required: `qaly/event`
reads as quality-adjusted life years per adverse event. A fixed parameter is not searched and
not learned; it is part of the value like a constant.

| Declaration | Use it for | Example |
| --- | --- | --- |
| `pf.strength(...)` | a trade-off you can bound but not pin | 0.5 to 4 qaly per adverse event |
| `pf.fixed(...)` | a trade-off that is settled | 20 usd of margin per order |
| `pf.relation(...)` | a judgment about which trade-off is larger | harm weighs at least twice cost |
| `pf.Curve(...)` with shape rules | value that changes with the level of an outcome | each further unit of benefit is worth no more than the last |
| `pf.excess`, `pf.shortfall`, `pf.segment` | value that starts or stops at a level | waiting time costs nothing up to 14 days |
| `pf.interaction`, `pf.both` | value that depends on two outcomes together | revenue counts only when stock and staff are both available |

### Orderings between trade-offs

A relation restricts parameters jointly:
`coefficient_1 × parameter_1 + coefficient_2 × parameter_2 + ... (>=, <= or ==) bound`.
Write it with `>=` or `<=` between parameter expressions; `pf.relation` gives it a stable id,
and the four-argument form states an equality.

```python
cost = pf.attribute("cost", units="usd")
cost_weight = pf.strength("cost_weight", units="qaly/usd", lower=0.0, upper=1.0, assumption="payer")
rule = pf.relation("harm_outweighs_cost", 4 * cost_weight <= harm_weight)

model = pf.ValueModel(
    benefit - harm_weight * harm - cost_weight * cost,
    utility_units="qaly",
    relations=(rule,),
)
```

The relation cuts the box of ranges: here `cost_weight` can be at most a quarter of
`harm_weight`. A relation reads trade-off parameters only, never an attribute, and is linear.
If the relations leave no admissible point, evaluation is refused
(`preference_decision.preference_set_empty`). A relation is an assumption like a range, and the
result lists it among its assumptions.

### Curves, shape rules and kinks

A curve is piecewise linear in one attribute between breakpoints you declare. Each segment has
its own slope parameter; the curve is zero at the first breakpoint and flat beyond the last.
The shape rules are relations between the slopes, which you pass to the model:

| Rule | Relation added | Shape |
| --- | --- | --- |
| `curve.increasing()` | every slope `>= 0` | more is never worse |
| `curve.decreasing()` | every slope `<= 0` | more is never better |
| `curve.diminishing()` | each slope `>=` the next | concave: each further unit is worth no more |
| `curve.accelerating()` | each slope `<=` the next | convex |

```python
low = pf.strength("slope_low", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape")
high = pf.strength("slope_high", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape")
curve = pf.Curve(benefit, breakpoints=(0.0, 1.0, 3.0), slopes=(low, high))

model = pf.ValueModel(
    curve - harm_weight * harm,
    utility_units="qaly",
    relations=(*curve.increasing(), *curve.diminishing()),
)
```

A kink at a declared level uses a parameter-free shape: `pf.excess(x, above)` is
`max(x − above, 0)`, `pf.shortfall(x, below)` is `max(below − x, 0)` and
`pf.segment(x, start, end)` is the part of `x` between two levels. `pf.interaction(a, b)` is
`a × b` and `pf.both(a, b)` is `min(a, b)`. A parameter that multiplies one of these shapes, as
the slopes of a curve do, keeps the value affine in the parameters. A parameter placed inside a
shape (the location of a kink, a parameter squared, a rate inside `exp`) is a *shape
parameter*: the value is no longer affine in it, `model.shape_parameters` names it, and the
decision takes the [certified route](#certified-route).

The curve follows the piecewise-linear marginal value functions of Jacquet-Lagrèze and Siskos
(1982). It is parameterized by slopes rather than by values at the breakpoints, imposes no
normalization, and does not require the value to be additive across attributes.

## Bind actions

`pf.Action(id, inputs={...})` maps each attribute id to a `ScientificQuantity` whose law the
source supplies, or to a fixed number in the attribute's units. At least one input of each
action is a quantity. Every quantity must lie in the contract's target population and horizon
and be stated in the attribute's units; a mismatch is refused
(`preference_contract.quantity_units_mismatch`, `preference_contract.quantity_outside_scope`).

`pf.Contract(actions=..., target_population=..., model=...)` also takes:

| Argument | Default | Meaning |
| --- | --- | --- |
| `criterion` | `decision.Criterion.expected_utility()` | expected utility, expected loss, threshold probability, quantile, regret or expected regret, as in [decision contracts](decisions.md#what-a-contract-declares) |
| `policy` | `pf.Policy.invariant_best()` | how the answer across the trade-offs is consumed; see [policies](#choose-a-policy) |
| `constraints` | none | hard constraints over attributes |
| `horizon` | `0` | the horizon every quantity must be at |
| `structural_policy` | `"report_only"` | how [structures](#decide-across-uncertain-causal-structures) combine |
| `preferences` | | in place of `model`: reviewed [accepted preferences](preference-learning.md#accept-review-and-reuse), which also carry their measure when there is one |

`contract.route` says which route will answer and `contract.identity` is the digest to retain
for replay.

### Hard constraints never become trade-offs

A hard constraint reads attributes, never trade-off parameters:
`P(expression <= bound) >= min_probability`.

```python
cap = pf.constraint("harm_cap", harm, at_most=1.0, units="events", min_probability=0.9)
```

`pf.constraint` refuses an expression that reads a trade-off. A constraint excludes an action
before any trade-off is considered. An excluded action keeps its exclusions, each with the
probability it reached and the probability required. No range of trade-offs can buy an action
back into the comparison, and no constraint is turned into a penalty. When every action is
excluded the verdict is `no_admissible_action`. A mean cannot answer a constraint probability,
so a contract with hard constraints needs aligned joint draws.

## Choose a policy

The policy says how a result across the admissible set may be consumed.

| Policy | Chooses | Needs |
| --- | --- | --- |
| `pf.Policy.invariant_best()` | an action only when it is never beaten anywhere in the set | nothing more |
| `pf.Policy.maximin()` | the action with the best worst-case value over the set | nothing more |
| `pf.Policy.minimax_regret()` | the action with the smallest worst-case regret | an expectation criterion |
| `pf.Policy.weighted()` | the action with the best value under a measure | a [measure](#supply-your-own-weighted-viewpoints) |
| `pf.Policy.report_only()` | nothing; every table is computed | nothing more |

The policy changes the verdict, not the tables. Every result reports value ranges, possibly
best actions, worst-case regret and switches whichever policy you pick.

## Read the result

### Verdicts

`decided.kind` is the verdict in snake case; the Rust variant is in the second column.

| `kind` | Rust | What it says |
| --- | --- | --- |
| `necessarily_best` | `NecessarilyBest` | One action is ahead of every rival by more than the tie tolerance at every admissible trade-off: the unique winner. |
| `never_beaten` | `NeverBeaten { action, ties_with }` | One action is never behind anywhere in the set but ties the listed rivals somewhere, usually at the edge of the set. |
| `preference_dependent` | `PreferenceDependent` | Different admissible trade-offs favor different actions. The possibly best actions are listed, and `switches` says where each pair changes order. |
| `tied` | `Tied` | Several actions share the best score within the tie tolerance, or are equal at every admissible trade-off. Declaration order is not a choice. |
| `worst_case_choice` | `WorstCaseChoice` | Under `maximin`, the action whose lowest value over the set is highest. |
| `minimax_regret_choice` | `MinimaxRegretChoice` | Under `minimax_regret`, the action whose largest shortfall from the best rival is smallest. |
| `measure_choice` | `MeasureChoice` | Under `weighted`, the action with the best measure-weighted value, with the weight of points where it leads. |
| `report_only` | `ReportOnly` | Everything was computed; you asked for no choice. |
| `no_admissible_action` | `NoAdmissibleAction` | Every action failed a hard constraint. Nothing is allowed. |
| `undetermined` | `CertifiedVerdict::Undetermined` | Certified route only: the enclosures did not settle the policy within the budget. The candidates and the undecided share are listed. |
| `structures_disagree`, `insufficient_science` | `StructuralPreferenceVerdict` | Across structures only; see [structures](#decide-across-uncertain-causal-structures). |

`decided.selected` is the chosen action, or `None` when the verdict chooses none.
`decided.statement` gives the verdict in one or two plain sentences built from the result's own
facts. For a dependent verdict that turns on one trade-off it names the trade-off, the value at
which the choice changes and the admissible range. A range is stated as a range and a measure
share as a weight; neither is called a probability. `decided.explain()` adds the route and the
assumptions.

### Per-action facts

`decided.actions` holds one record per action, in declaration order.

| Field | Meaning | What it is not |
| --- | --- | --- |
| `value_range` | Lowest and highest criterion value over the admissible set | A confidence interval |
| `possibly_best` | Best (ties allowed) at some admissible trade-off | A share of the set |
| `necessarily_best` | Never beaten: never behind any rival by more than the tie tolerance anywhere in the set (ties allowed) | A unique winner |
| `uniquely_best` | Ahead of every rival by more than the tie tolerance everywhere; at most one action has it | A statement about trade-offs you did not allow |
| `dominated_by` | Rivals strictly ahead of this action at every admissible trade-off | The full list of actions that beat it somewhere |
| `worst_case_regret` | The largest shortfall from the best rival over the set, floored at zero, in the criterion's units | An expected loss |
| `measure_share`, `measure_value` | The weight of measure points where the action leads, and its measure-weighted value, when a measure was supplied | A probability, unless your measure is one |
| `exclusions` | The hard constraints the action failed | A penalty |

An action can be best nowhere without being dominated by a single rival: different rivals can
lead it in different parts of the set. An action that is not possibly best can be dropped under
any trade-off you allowed. An action that is possibly best but not never beaten is a live
candidate whose case rests on a judgment you have not settled.

`decided.parameter_ranges` gives each trade-off's lowest and highest admissible value: the
declared range cut by the relations.

### Switches and thresholds

A switch is a pair of actions whose order changes inside the set, listed in declaration order.
Each reports `margin_range` (the range of the first action's margin over the second), and
`favours_first_at` and `favours_second_at` (one admissible trade-off point on each side).

When the margin depends on a single trade-off, the switch also has a `threshold`: the
`parameter`, the `value` at which the two actions are equal, and `first_leads_above`, which says
whether the first action leads above that value. A margin counts as depending on one parameter
when every other parameter's coefficient times the width of its declared range is within the tie
tolerance. When the margin moves with several trade-offs, `threshold` is `None`; the boundary is
a plane through the set, and the two reported points show the trade-offs on each side.

```python
for switch in decided.switches:
    if switch.threshold is not None:
        print(switch.first, switch.second, switch.threshold.parameter, switch.threshold.value)
```

The necessary and possible readings follow Greco, Mousseau and Słowiński (2008), applied to the
expected values of causal quantities rather than to fixed criterion scores, and stated about the
best action rather than about pairs. Worst-case regret over a set of utility parameters follows
Boutilier, Patrascu, Poupart and Schuurmans (2006).

On aligned joint draws, the Rust result of a unique winner also reports
`draw_share_best_everywhere`: the weight of draws in which that action has the highest per-draw
utility at every vertex of the set. It is a point diagnostic of the supplied law, not a
calibrated probability. The Python result does not expose it.

## Supply your own weighted viewpoints

When you want shares as well as ranges, supply a measure: a finite set of admissible trade-off
points with weights. Named points are viewpoints.

```python
views = pf.Measure.viewpoints(
    {"clinician": {"harm_weight": 1.0}, "patient": {"harm_weight": 3.0}},
    weights={"clinician": 3.0, "patient": 1.0},
    assumption="stakeholder_weights",
)
weighted = pf.Contract(model=model, actions=actions, target_population="target",
                       policy=pf.Policy.weighted())
decided = require_success(weighted.evaluate(source, measure=views))
```

Viewpoints are equally weighted unless you give weights. `pf.Measure(points=..., weights=...,
labels=..., assumption=...)` builds an unnamed measure; `unresolved=` names trade-offs the
measure says nothing about, whose declared ranges still apply. Every point must lie inside the
admissible set; a point outside it is refused
(`preference_decision.measure_point_outside_set`).

A share is the weight of the measure's points at which an action leads; points where several
actions tie share their weight equally. It is a probability only if the measure you supplied is
one. The share is the first-rank acceptability index of Lahdelma, Hokkanen and Salminen (1998)
for a discrete measure, not their integral over a density. Under `Policy.weighted()` the measure
chooses, and without one the policy is refused (`preference_decision.measure_required`). Under
any other policy the shares are reported beside the verdict.

## Several estimated outcomes

A value model usually reads several outcomes, so the source must supply all of them for every
action. What it must supply depends on the route: the exact route reads only the expectation of
each part of the value, and the certified route reads aligned joint draws.

| Where the outcomes come from | What reaches the decision | What it answers |
| --- | --- | --- |
| One experiment that measured several outcomes on every unit | Its arm means, as one mean source (`experiment_mean_source`, with `experiment_arm_quantity` naming each coordinate) | Expectation trade-offs only: a value affine in the outcomes, on the exact route. A kinked term, a probability limit or a quantile is refused by the mean-source rule. Arm means are read only when the exposure is declared free of between-unit interference. |
| Outcomes fitted separately | One aligned joint law, built by a [propagation chain](scientific-bridges.md) that joins the fits under a dependence you declare (`chain_joint_law`) | Everything the certified route answers, kinked terms and probability limits included. The law is unverified and its calibration not measured. |
| Several laws whose rows are realizations of one joint law | Their columns side by side (`joint_law_of`) | As for a joint law. Rows of separately produced draws are never paired: they refuse with `joint_law_required`. |
| One study with several outcomes, wanting a joint law | Not yet available | Kinked terms and probability limits on such a study are refused. |

These adapters are in Rust (`antecedent::analysis::preference_decision`). From Python today,
`Contract.evaluate` reads one source: a `JointDistributionArtifact` of aligned draws, a
`MeanSource`, a bound external claim, `pf.Structures`, or one supported analysis result, read as
its mean source. An analysis result supplies one outcome, so a contract over several outcomes
cannot be evaluated directly on it. The experiment and chain adapters above are not exposed to
Python. You can build a `MeanSource` or a `JointDistributionArtifact` over several outcomes
yourself; it is then labelled as yours (externally attested or unverified), and the
[tutorial](tutorials/decide-across-trade-offs.md) does this with randomized arm means. A
`ChainJointLaw` from `PropagationChain.joint_output_law` is not itself an artifact that
`evaluate` accepts.

## Decide across uncertain causal structures

When the causal structure is uncertain, pass the admissible structures in place of the source:

```python
structures = pf.Structures("weighted_graph_posterior", claims)
contract = pf.Contract(model=model, actions=actions, target_population="target",
                       structural_policy="maximin")
decided = require_success(contract.evaluate(structures))
decided.structures       # each structure's status and its own verdict
```

`claims` are `decision_robust.Claim` values (evaluated with draws, unidentified, or unevaluated
with a reason), read as the [structural decision routes](sensitivity-and-robustness.md) read
them; the kind is `point`, `graph_dependent`, `weighted_graph_posterior` or `finite_scenarios`.

The admissible combinations are **every pairing of an admissible structure with an admissible
trade-off**. A conclusion that holds across structures holds for each structure at each
trade-off. You cannot state that a trade-off goes only with some structures: linked pairs are
not supported.

| `structural_policy` | How structures combine |
| --- | --- |
| `"require_invariant_best_action"` | Every structure is evaluated and every structure's verdict must agree; otherwise `structures_disagree` lists each structure's verdict. |
| `"maximin"` | Each action is scored by its worst value over structures at each trade-off, then the preference policy is applied. Every structure must be evaluated. |
| `"bayes_over_structures"` | Structure probabilities you supply weight the per-structure values; they are never renormalized. |
| `"report_only"` | Each structure's result, with no combination. |

Unidentified and unevaluated structure mass is reported (`unidentified_mass`,
`unevaluated_mass`). A claim that needs every structure returns `insufficient_science` when one
is unresolved. An action excluded by a hard constraint in any evaluated structure is excluded
everywhere. These rules are those of [structural uncertainty](decisions.md#structural-uncertainty)
in a decision contract; see also [graph uncertainty](graph_uncertainty.md).

## Two routes and what each guarantees

Antecedent picks the route from the contract. `decided.route` is `exact` or `certified`, and
`decided.route_reason` names why (`preference_route.*`).

| | Exact route | Certified route |
| --- | --- | --- |
| Used when | the value is affine in the trade-off parameters and the criterion is expected utility, expected loss or expected regret | any other supported value or criterion |
| Method | linear programs over the admissible set | interval subdivision of the declared box |
| Source | means (when every part is affine in the attributes) or aligned joint draws | aligned joint draws only |
| Undecided share | none | reported when the budget runs out |

`evaluate` takes `tie_tolerance=`, `vertex_cap=`, `max_boxes=` and `min_width_fraction=` to
tune the routes.

### Exact route

An action's expected value at trade-off point `theta` is `c_0 + theta_1 c_1 + ... + theta_K c_K`,
where each `c` is the expectation of one part of the value under the supplied law. Every
question on this page is then a small linear program: value ranges, margins, never beaten,
unique winner, possibly best, domination, worst-case regret and switches.

What it guarantees: the answers are exact statements about the supplied law, to the solver's
checked numerical tolerance. Every linear-program outcome is rechecked before it is returned:
feasibility and a non-negative least-squares fit of the objective by the tight rows for an
optimum, Farkas multipliers for infeasibility, an improving ray for unboundedness. The solver
uses Bland's (1977) smallest-index rule and the non-negative least-squares method of Lawson
and Hanson. Margins within the tie tolerance (`1e-9` criterion units by default) count as ties.

What it does not guarantee: the law itself. If the source is a posterior or a set of sampled
draws, its own sampling error is not part of the ranges. The ranges are over trade-offs only.

### Certified route

The certified route covers the declared box of ranges with sub-boxes and bounds every action's
criterion on each sub-box with outward-rounded interval arithmetic (Moore, Kearfott and Cloud,
2009). It answers shape parameters, threshold probabilities, quantiles and largest per-draw
regret.

What it guarantees:

- **Enclosures.** Every bound contains the exact real range. Each floating-point result is
  moved outward by one representable step (two for `exp` and `ln`), and weighted sums are
  widened to cover any summation order.
- **Supported function class.** The closed value expressions: sums, differences, products,
  minimum, maximum, `exp` and `ln` of attributes and parameters. A logarithm whose argument can
  reach zero or below on a box is refused (`preference_certified.logarithm_domain`).
- **Bounded declared ranges.** Every parameter has a finite range, so the box to cover is
  bounded.
- **No point labels.** A box is certified for an action only when its enclosures prove that
  action better than every admissible rival by more than the tie tolerance on the whole box.
- **Boundary boxes counted in full.** A box that straddles a relation is kept, flagged as
  touching the boundary and counted in full, although part of it lies outside the set.
- **An explicit undecided share.** Splitting stops when every box is decided, when the box
  budget is spent (4,096 boxes by default) or when every width is at most a minimum fraction
  of its range (`1e-6` by default). Volume left undecided is reported as `undecided_fraction`
  and, when it prevents a verdict, as `undetermined` with the remaining candidates.

On this route `value_range`, `worst_case_range` and `worst_case_regret_range` are enclosures,
`best_fraction` is the volume fraction on which an action is certified best, and
`decided.regions` lists the kept boxes. The fractions are volume fractions of the declared
ranges. They are not probabilities and not acceptability indices. Switches and thresholds are
reported on the exact route only.

## Export and replay

A decision on aligned joint draws exports as `preference_decision_v1`: the contract, the
measure, the evaluation options, the route that answered, the result and the digest of the
draws. A decision on a mean source or across structures does not export
(`route_not_supported`).

```python
exported = decided.export()
replayed = require_success(pf.replay(exported, contract=contract, source=source))
```

Replay loads only under the contract and draws you retained, recomputes the decision and
compares every stored field; other draws or an edited number refuse.

## Refusals you are likely to meet

Each refusal is a `pf.PreferenceRefusal` with a registered `code` and a namespaced `detail`. The
full tables are in
[preference sets and recorded choices](reference/refusal-codes.md#preference-sets-and-recorded-choices).

| Detail | Code | Cause | Remedy |
| --- | --- | --- | --- |
| `preference_model.parameter_range_empty` | `invalid_argument` | A range has `lower > upper` | Declare lower <= upper |
| `preference_model.parameter_unused` | `invalid_argument` | A declared parameter does not enter the value | Remove it or use it |
| `preference_contract.too_few_actions` | `invalid_argument` | Fewer than two actions | Declare at least two |
| `preference_contract.quantity_units_mismatch` | `quantity_semantics_mismatch` | A quantity is not in its attribute's units | Read a quantity in the attribute's units |
| `preference_decision.preference_set_empty` | `invalid_argument` | The relations leave no admissible trade-off | Widen a range or remove a relation |
| `preference_evaluation.joint_draws_required` | `joint_law_required` | The certified route was needed but the source holds only means | Supply aligned joint draws |
| `preference_decision.measure_required` | `decision_contract_unsatisfied` | `weighted` policy without a measure | Supply a measure or choose another policy |
| `preference_decision.measure_point_outside_set` | `invalid_argument` | A viewpoint lies outside the declared set | Move the point inside, or widen the set if the viewpoint is admissible |
| `preference_decision.structure_probabilities_required` | `decision_contract_unsatisfied` | Bayes over structures without probabilities | Supply one per structure or choose another structural policy |
| `preference_certified.minimax_regret_needs_expectation` | `route_not_supported` | Minimax regret with a non-expectation criterion | Use an expectation criterion or another policy |
| `preference_certified.logarithm_domain` | `invalid_argument` | A logarithm's argument can reach zero inside the box | Narrow the ranges or change the value |

## What stays out of reach

- A probability over trade-offs that nobody supplied or estimated. Ranges stay ranges.
- Linked pairs of structure and trade-off. Every admissible structure is paired with every
  admissible trade-off.
- A certified answer from means. The certified route reads aligned joint draws.
- A joint law from one study with several outcomes, so kinked terms and probability limits on
  such a study.
- Hard constraints that depend on trade-offs, and penalties in place of constraints.
- Unbounded ranges. Every parameter needs finite ends.
- The worst case over trade-offs of expected regret on the exact route; it is referred to the
  certified route.
- A judgment of which trade-offs are right. The verdict follows from what you declared.

## Entry points

| Language | Entry |
| --- | --- |
| Python | `antecedent.preferences`: `attribute`, `strength`, `fixed`, `relation`, `Curve`, the shapes, `ValueModel`, `Action`, `constraint`, `Policy`, `Measure`, `Structures`, `Contract(...).evaluate(...)`, `PreferenceDecision`, `replay` |
| Rust, analysis surface | `antecedent::analysis::preference_decision`: `decide`, `decide_claim`, `structure_claims`, `decide_across_structures`, `experiment_mean_source`, `experiment_arm_quantity`, `chain_joint_law`, `joint_law_of`, `record_decision`, `export_decision`, `consume_decision` |
| Rust, declarations and routes | `antecedent_design::preference_model` (`ValueModel`, `PreferenceSet`, `Curve`, `shape`), `preference_contract` (`PreferenceContract`, `PreferencePolicy`), `preference_evaluation` (`preference_route`, `decide_preferences`, `decide_preferences_on_means`, `decide_preferences_structural`) |

## Sources

Jacquet-Lagrèze, E. and Siskos, J. (1982), assessing a set of additive utility functions, the
UTA method, doi:10.1016/0377-2217(82)90155-2. Greco, S., Mousseau, V. and Słowiński, R. (2008),
ordinal regression revisited, doi:10.1016/j.ejor.2007.08.013. Boutilier, C., Patrascu, R.,
Poupart, P. and Schuurmans, D. (2006), constraint-based optimization and utility elicitation
using the minimax decision criterion, doi:10.1016/j.artint.2006.02.003. Lahdelma, R.,
Hokkanen, J. and Salminen, P. (1998), SMAA, doi:10.1016/S0377-2217(97)00163-X. Moore, R. E.,
Kearfott, R. B. and Cloud, M. J. (2009), *Introduction to Interval Analysis*,
doi:10.1137/1.9780898717716. Bland, R. G. (1977), new finite pivoting rules for the simplex
method, doi:10.1287/moor.2.2.103. Lawson, C. L. and Hanson, R. J., *Solving Least Squares
Problems*, doi:10.1137/1.9781611971217. Each result names these through its method
provenance, with the documented deviations; see [citing a study](guides/citing-a-study.md).

## Where next

- [Narrow trade-offs from past decisions and stated comparisons](preference-learning.md):
  optional, for when you want past choices to inform the ranges
- [Choose a discount when the cost of a return is a range](tutorials/decide-across-trade-offs.md):
  the worked story
- [Evaluate actions with a decision contract](decisions.md) and
  [probe sensitivity to assumptions](sensitivity-and-robustness.md)
