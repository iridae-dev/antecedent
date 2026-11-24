# Trade-off ranges in depth

This page is the deep dive behind the trade-off sections of the [Decisions guide](decisions.md).
Decide between actions that move several outcomes at once (benefit, harm, cost) when you can
state how those outcomes trade off only as a range. You declare a value: the outcomes it
reads, the trade-offs between them as ranges, fixed values, orderings and curve shapes, and the
assumption each declaration stands on. Antecedent answers the decision across every trade-off
you allowed. It says which conclusions follow from the evidence and your declarations, which
depend on a judgment you have not settled, and which cannot be justified. Use this page when a
[decision contract](decisions.md) would need one exact utility and you cannot defend one. The
same `decision.Contract(value=...)` covers a weight you can state exactly (an ordinary contract)
and a weight you can only bound.

The method is causal decision analysis under partially identified preferences. The evidence
fixes what each action does to each outcome. Your declarations fix a set of admissible
trade-offs, not one point. The answer is a statement about that whole set.

For a worked story from simulated data to a threshold, see
[Choose a discount when the cost of a return is a range](tutorials/decide-across-trade-offs.md).

## A first decision

Two actions are compared on benefit (in quality-adjusted life years) and harm (adverse events),
both estimated by the analysis you prepared. How much benefit one adverse event costs is stated
as a range. The contract is compiled with that analysis, its plan is inspected, then it is
executed.

```python
from antecedent import decision
from antecedent.outcomes import require_success

benefit = decision.attribute("benefit", units="qaly")
harm = decision.attribute("harm", units="events")
harm_weight = decision.tradeoff(
    "harm_weight", units="qaly/event", lower=0.5, upper=4.0, assumption="clinical_panel"
)
value = decision.Value(benefit - harm_weight * harm, units="qaly")

actions = (
    decision.Action("treat", inputs={"benefit": treat_benefit, "harm": treat_harm}),
    decision.Action("wait", inputs={"benefit": wait_benefit, "harm": wait_harm}),
)
contract = decision.Contract(value=value, actions=actions, target_population="target")
compiled = require_success(contract.compile([benefit_study, harm_study]))
compiled.quantities       # what each action reads, and the study that targets it
compiled.representation   # "mean": the source the contract needs; nothing is fitted yet
decided = require_success(compiled.execute())
decided.kind         # "necessarily_best", "never_beaten", "preference_dependent", ...
decided.choice       # the chosen action, or None
decided.standing     # "best", "depends", "policy_choice", ...
decided.statement    # the verdict in plain words
```

`treat_benefit`, `treat_harm`, `wait_benefit` and `wait_harm` are `ScientificQuantity`
coordinates of the mean responses `benefit_study` and `harm_study` estimate (studies prepared
with `antecedent.prepare` and a `ResponseCurve` query). A plain number in `inputs` is an input
known without uncertainty, stated in the attribute's units. Every attribute of the model gets
exactly one input. [Compile with the analysis](#compile-with-the-analysis) says what the compile
checks and what the plan shows; [evaluate on a source in hand](#evaluate-on-a-source-in-hand) is
the lower layer it calls.

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

## Compile with the analysis

`Contract.compile(source)` binds the contract to the analysis that estimates its quantities and
lowers the decision into that analysis's own compiled plan. `source` is one prepared study, a
sequence of prepared studies (one outcome each), or a compiled design of several outcomes
(`joint_panel.prepare_native_joint_design`). Nothing is fitted until `execute()`.

At compile:

- every quantity an action reads must be a coordinate the analysis targets: its outcome, a
  regime on its dose grid, the contract's population, the horizon, the mean on the identity
  scale, in the outcome's units. A quantity that names anything else is refused at bind time,
  naming the action and the attribute;
- the contract's requirement fixes the source it needs: the means when the value is affine in
  the outcomes and the criterion is an expectation, aligned joint draws otherwise. The plan
  demands the matching carrier of its statistical program, and a realization that cannot supply
  it (a frequentist study asked for draws) is refused;
- identification is checked under every structure the study carries. A graph posterior's
  structures are compiled one by one: DAG and ADMG atoms as studies of their own structure,
  CPDAG and PAG atoms member by member, weighed by a declared class prior when one is given
  (see [weigh the members of an equivalence class](#weigh-the-members-of-an-equivalence-class)).
  Mass under which the quantities are not identified is reported, never renormalized, and a
  quantity no structure identifies is refused (`effect_not_identified`);
- separately estimated studies are checked for shared observations (see
  [which studies share evidence](#which-studies-share-evidence)), and a value that needs one
  law of their outcomes gets its propagation chain compiled here: the chain rule checks its
  inputs, dependence and joint outputs, the chain's program is lowered with its family table
  deferred, and the table's work budget and declared covariance blocks are checked before
  anything is fitted.

The `CompiledDecision` states, before anything is fitted:

| Property | What it states |
| --- | --- |
| `contract_identity`, `bound_identity` | the contract, and the contract bound to the analysis's causal contracts |
| `quantities` | each quantity an action reads (`CompiledQuantity`: the `ScientificQuantity`, the position of the study that targets it, and the `(action, attribute)` pairs that read it) |
| `representation`, `carrier`, `route` | the source the contract needs (`"mean"` or `"joint_draws"`), what each plan demands of its analysis, and the route that will answer |
| `trade_offs`, `parameter_ranges` | each trade-off parameter with its units, declared range and assumption (`TradeOffRange`), and the ranges by id |
| `joint_law` | for separately estimated outcomes joined into one law, a `JointLawPlan`: the dependence, the number of aligned draws, their seed, the study pairs with a declared cross covariance and the columns the chain reads; `None` otherwise |
| `design_law` | for a compiled design that needs one law of its outcomes, a `DesignLawPlan`: the resampling law, the unit resampled, the refits, their seed and the sampling premise; `None` otherwise |
| `unidentified_mass`, `class_sampling` | across structures, the mass under which the quantities are not identified and the draw of every sampled class |

`compiled.inspect()` returns everything the compile states as nested mappings, every compiled
plan included. Each plan carries the decision in its evidence layer:
`evidence.decision.preference` lists every quantity (canonical identity, schema variable,
regime, units, population, horizon, functional, scale), the representation the route reads and
the ones the contract accepts, the trade-off parameters with their declared ranges, the bound
contract identity and, for separately estimated outcomes, the joint law the propagation chain
will produce: its inputs with their row factors and how their overlap was judged, the
dependence, the declared blocks and `program`, the identity of the chain's compiled program.
Another range or another chain is another decision, so another plan identity.

`compile` takes only the arguments its source reads, and refuses any other with
`option_not_applicable` rather than dropping it: `draws=` and `shared=` apply to several
studies, `refits=` and `sampling=` to a compiled design, and `seed=` only when something is
drawn (`draws=`, `shared=` or `refits=`). `seed=` defaults to the seed of `execution=`, so
`execution=ant.Execution(seed=11)` alone fixes the chain's draws and a design's refits.

`compiled.execute(measure=None, execution=None)` runs each plan once and returns the same
`TradeoffDecision` as `evaluate`. When a chain was compiled, execution fits each study, binds
its posterior moments to the chain's deferred family table and runs the compiled program; the
receipt names that program's identity. The receipt lists the compiled program of every plan
(`programs`) and, position by position beside it, the identity the executed result reported
(`executed`); an execution whose result reports another program is refused
(`preference_compile.executed_program_mismatch`), for studies and compiled designs alike. An
`executed` entry is `None` (`null` in Python) where no result reports one: a structure whose
estimation failed without ending the run, which the decision counts as unevaluated, or a
runtime-callback program, which has no semantic identity. The decision's `source` is the aligned
law it read, when it read one, so the decision exports and replays as one evaluated on draws.

### Evaluate on a source in hand

`Contract.evaluate(source)` is the lower layer the compiled route calls. It reads a source you
already hold: aligned joint draws (a `JointDistributionArtifact`), a `MeanSource`, a bound
external claim, `decision.Structures`, the arm means of one experiment, or a propagation chain with
joint outputs. Use it when the numbers come from outside a compiled analysis; it checks the
coordinates and the representation but cannot check identification, the carrier or the overlap
of the evidence behind the source.

## Declare a value model

A value has three parts: attributes, trade-off parameters and the value expression that
combines them.

- An **attribute** is one outcome the value reads, with its units
  (`decision.attribute(id, units=...)`). Its value for each action comes from a scientific quantity or
  a fixed input.
- A **trade-off parameter** is a number the value multiplies or shapes, with units, an
  admissible range and the id of the assumption the range stands on. A parameter whose lower
  and upper ends are equal is fixed.
- The **value** is a closed expression over attributes and parameters, in the value's
  `units`. Write it with `+`, `-`, `*`, division by a number, `decision.maximum`,
  `decision.minimum`, `decision.exp` and `decision.ln`.

`decision.Value(expr, units=..., relations=...)` reads the attributes and parameters
from the expression in order of first appearance. The admissible set is every parameter point
inside the declared ranges that also satisfies the declared relations. This page calls a point
in it an *admissible trade-off*.

### Rates, fixed values and units

`decision.tradeoff(id, units=..., lower=..., upper=..., assumption=...)` declares a ranged trade-off.
`decision.tradeoff(id, units=..., value=...)` declares a settled, named one; a plain number in
the expression is a settled trade-off with no name (see
[fixed trade-offs](#fixed-trade-offs-and-the-tie-tolerance)). Units are required: `qaly/event`
reads as quality-adjusted life years per adverse event. A fixed parameter is not searched and
not learned; it is part of the value like a constant.

| Declaration | Use it for | Example |
| --- | --- | --- |
| `decision.tradeoff(..., lower=, upper=)` | a trade-off you can bound but not pin | 0.5 to 4 qaly per adverse event |
| `decision.tradeoff(..., value=)` | a trade-off that is settled | 20 usd of margin per order |
| `decision.relation(...)` | a judgment about which trade-off is larger | harm weighs at least twice cost |
| `decision.Curve(...)` with shape rules | value that changes with the level of an outcome | each further unit of benefit is worth no more than the last |
| `decision.excess`, `decision.shortfall`, `decision.segment` | value that starts or stops at a level | waiting time costs nothing up to 14 days |
| `decision.interaction`, `decision.both` | value that depends on two outcomes together | revenue counts only when stock and staff are both available |

### Orderings between trade-offs

A relation restricts parameters jointly:
`coefficient_1 × parameter_1 + coefficient_2 × parameter_2 + ... (>=, <= or ==) bound`.
Write it with `>=` or `<=` between parameter expressions; `decision.relation` gives it a stable id,
and the four-argument form states an equality.

```python
cost = decision.attribute("cost", units="usd")
cost_weight = decision.tradeoff("cost_weight", units="qaly/usd", lower=0.0, upper=1.0, assumption="payer")
rule = decision.relation("harm_outweighs_cost", 4 * cost_weight <= harm_weight)

value = decision.Value(
    benefit - harm_weight * harm - cost_weight * cost,
    units="qaly",
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
low = decision.tradeoff("slope_low", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape")
high = decision.tradeoff("slope_high", units="qaly/qaly", lower=0.0, upper=2.0, assumption="shape")
curve = decision.Curve(benefit, breakpoints=(0.0, 1.0, 3.0), slopes=(low, high))

value = decision.Value(
    curve - harm_weight * harm,
    units="qaly",
    relations=(*curve.increasing(), *curve.diminishing()),
)
```

A kink at a declared level uses a parameter-free shape: `decision.excess(x, above)` is
`max(x − above, 0)`, `decision.shortfall(x, below)` is `max(below − x, 0)` and
`decision.segment(x, start, end)` is the part of `x` between two levels. `decision.interaction(a, b)` is
`a × b` and `decision.both(a, b)` is `min(a, b)`. A parameter that multiplies one of these shapes, as
the slopes of a curve do, keeps the value affine in the parameters. A parameter placed inside a
shape (the location of a kink, a parameter squared, a rate inside `exp`) is a *shape
parameter*: the value is no longer affine in it, `model.shape_parameters` names it, and the
decision takes the [certified route](#certified-route).

The curve follows the piecewise-linear marginal value functions of Jacquet-Lagrèze and Siskos
(1982). It is parameterized by slopes rather than by values at the breakpoints, imposes no
normalization, and does not require the value to be additive across attributes.

## Bind actions

`decision.Action(id, inputs={...})` maps each attribute id to a `ScientificQuantity` whose law the
source supplies, or to a fixed number in the attribute's units. At least one input of each
action is a quantity. Every quantity must lie in the contract's target population and horizon
and be stated in the attribute's units; a mismatch is refused
(`preference_contract.quantity_units_mismatch`, `preference_contract.quantity_outside_scope`).

`decision.Contract(actions=..., target_population=..., value=...)` also takes:

| Argument | Default | Meaning |
| --- | --- | --- |
| `criterion` | `decision.Criterion.expected_utility()` | expected utility, expected loss, threshold probability, quantile, regret or expected regret, as in [decision contracts](decisions.md#what-a-contract-declares) |
| `policy` | `decision.TradeoffPolicy.invariant_best()` | how the answer across the trade-offs is consumed; see [policies](#choose-a-policy) |
| `constraints` | none | hard constraints over attributes |
| `horizon` | `0` | the horizon every quantity must be at |
| `structural_policy` | `"report_only"` | how [structures](#decide-across-uncertain-causal-structures) combine, as a string |
| `value` | | a `decision.Value`, or reviewed [accepted preferences](preference-learning.md#accept-review-and-reuse), which also carry their measure when there is one |

`contract.route` says which route will answer and `contract.identity` is the digest to retain
for replay.

### Hard constraints never become trade-offs

A hard constraint reads attributes, never trade-off parameters:
`P(expression <= bound) >= min_probability`.

```python
cap = decision.Constraint("harm_cap", harm, bound=1.0, units="events", min_probability=0.9)
```

`decision.Constraint` refuses an expression that reads a trade-off. A constraint excludes an action
before any trade-off is considered. An excluded action keeps its exclusions, each under the
constraint's own id with the probability it reached and the probability required. No range of trade-offs can buy an action
back into the comparison, and no constraint is turned into a penalty. When every action is
excluded the verdict is `no_admissible_action`. A mean cannot answer a constraint probability,
so a contract with hard constraints needs aligned joint draws.

## Choose a policy

The policy says how a result across the admissible set may be consumed.

| Policy | Chooses | Needs |
| --- | --- | --- |
| `decision.TradeoffPolicy.invariant_best()` | an action only when it is never beaten anywhere in the set | nothing more |
| `decision.TradeoffPolicy.maximin()` | the action with the best worst-case value over the set | nothing more |
| `decision.TradeoffPolicy.minimax_regret()` | the action with the smallest worst-case regret | an expectation criterion |
| `decision.TradeoffPolicy.weighted()` | the action with the best value under a measure | a [measure](#supply-your-own-weighted-viewpoints) |
| `decision.TradeoffPolicy.report_only()` | nothing; every table is computed | nothing more |

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

`decided.choice` is the chosen action, or `None` when the verdict chooses none; `decided.standing`
places the verdict in the vocabulary every [decision result](decisions.md#read-any-decision-result) shares.
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

On aligned joint draws, a result with a unique winner also reports
`draw_share_best_everywhere` (`decided.draw_share_best_everywhere` in Python): the weight of
draws in which that action has the highest per-draw utility at every vertex of the set. It is a
point diagnostic of the supplied law, not a calibrated probability. It is `None` for a mean
source, on the certified route and for a combination across structures.

## Supply your own weighted viewpoints

When you want shares as well as ranges, supply a measure: a finite set of admissible trade-off
points with weights. Named points are viewpoints.

```python
views = decision.Measure.viewpoints(
    {"clinician": {"harm_weight": 1.0}, "patient": {"harm_weight": 3.0}},
    weights={"clinician": 3.0, "patient": 1.0},
    assumption="stakeholder_weights",
)
weighted = decision.Contract(value=value, actions=actions, target_population="target",
                       policy=decision.TradeoffPolicy.weighted())
decided = require_success(weighted.evaluate(source, measure=views))
```

Viewpoints are equally weighted unless you give weights. `decision.Measure(points=..., weights=...,
labels=..., assumption=...)` builds an unnamed measure; `unresolved=` names trade-offs the
measure says nothing about, whose declared ranges still apply. Every point must lie inside the
admissible set; a point outside it is refused
(`preference_decision.measure_point_outside_set`).

A share is the weight of the measure's points at which an action leads; points where several
actions tie share their weight equally. It is a probability only if the measure you supplied is
one. The share is the first-rank acceptability index of Lahdelma, Hokkanen and Salminen (1998)
for a discrete measure, not their integral over a density. Under `TradeoffPolicy.weighted()` the measure
chooses, and without one the policy is refused (`preference_decision.measure_required`). Under
any other policy the shares are reported beside the verdict.

## Several estimated outcomes

A value model usually reads several outcomes, so the analysis must supply all of them for every
action. What it must supply depends on the route: the exact route reads only the expectation of
each part of the value, and the certified route reads one aligned joint law.

| Where the outcomes come from | How to compile | What it answers |
| --- | --- | --- |
| Separate studies, one outcome each | `contract.compile([study_a, study_b])` | Means side by side for an affine expectation. A value that needs a joint law needs a declared dependence: `draws=` (and `seed=`) under distinct evidence, so the propagation chain draws one aligned law from a family table matched to each study's posterior moments. The chain is declared, checked and compiled at compile with its family table deferred: the plan states its inputs, dependence, joint outputs and program identity, the table's work budget is checked before fitting, and execution binds each study's posterior moments to the table. |
| Separate studies that share observations | `contract.compile([study_a, study_b], draws=..., shared={(0, 1): block})` | Studies that share observations (see [which studies share evidence](#which-studies-share-evidence)) are not distinct evidence and are refused (`evidence_double_counted`) unless the covariance between their estimates is declared. Each block is a 2-D array keyed by the studies' positions: one row per coordinate of study `i` and one column per coordinate of study `j`, both in grid order (a block keyed `(j, i)` is read as its transpose). A block of another shape raises `CausalValueError`. The block is the cross-study part of the chain's declared covariance, which the drawn law carries exactly. Only the pairs that share observations need a block. |
| One static-regime design of several outcomes | `contract.compile(design, refits=99)` | Its arm means, or with `refits=` its aligned whole-subject bootstrap refits as one law of every outcome under both arms. |
| One cluster-randomized design, or one matched-pair or switchback experiment | `contract.compile(design)`, or `contract.compile(design, refits=99, seed=..., sampling="assumption.units_sampled")` | Its arm means, under an exposure declared free of interference. A value that needs a joint law reads the independent-unit bootstrap of the arm means, and only under a declared premise (`sampling=`, with `refits=`) that the clusters, pairs or sequences were sampled independently from a larger population; see [a joint law of a randomized design](#a-joint-law-of-a-randomized-design). |
| Laws you already hold whose rows pair | `contract.evaluate(decision.joint_law(a, b))` | As for a joint law. Rows of separately produced draws are never paired: they refuse with `joint_law_required`. |

The compiled design carries the decision in its own program's evidence layer, so its plan view
shows the decision and the design executes the decision-bearing program. When the decision reads
refits, the plan states the design's resampling law (`design_law`: its kind, unit, refits, seed
and premise), and all of them enter the decision program's identity.

### A joint law of a randomized design

Randomization identifies the arm means of a cluster-randomized design, a matched-pair experiment
or a switchback experiment, and a conservative covariance bound for any linear combination of
them. It does not identify how the arm-mean estimators vary together across arms: that depends
on the joint distribution of each unit's outcomes under both arms, which no assignment reveals.
A value that needs a joint law of these outcomes therefore needs a further premise, declared by
you: that the independent units were sampled independently from a larger population. Under it,
the sampling law of the arm means is identified and an independent-unit bootstrap estimates it.

```python
compiled = contract.compile(design, refits=199, seed=11,
                            sampling="assumption.clusters_sampled_from_population")
```

- Clusters are resampled within each arm, so each arm keeps its size; matched pairs and
  switchback sequences are resampled whole. Every refit reruns the complete estimator.
- The bootstrap needs at least 99 refits and at least 30 independent units: 30 clusters in each
  arm, 30 pairs or 30 sequences. Fewer units refuse with `preference_compile.design_units_too_few`.
- Without `sampling=`, a value that needs a joint law refuses at compile with
  `preference_compile.design_sampling_undeclared`; `sampling=` without `refits=` raises
  `CausalValueError`. Arm means alone, for an affine expectation, need neither.
- `seed=` seeds the refits; without it they are seeded by `execution=` (see
  [compile with the analysis](#compile-with-the-analysis)).
- A refit the estimator refuses keeps its slot and refuses the decision at execute
  (`native_claims.joint_law_invalid`); no refit is dropped.
- The premise's identity is listed in the decision's assumptions. The law is labelled a
  bootstrap law, unverified and with no calibration measured. The studies of a
graph posterior cannot be combined with other studies in one compile.

### Which studies share evidence

Before it joins separately estimated outcomes, the compile judges which studies observe the
same units. Two studies that do are not distinct evidence: their estimates are correlated, and
a law that treated them as independent would understate the spread of every margin.

- **Studies of one table share its rows.** `antecedent.prepare` keeps only the columns an
  analysis reads, but a prepared static `ResponseCurve` names the whole table it was given by
  its content, whether or not columns were dropped, and its rows keep their positions in that
  table. Two studies of different outcomes in one table, or a study of every column and a study
  of a few, therefore share every row. A table with any other value is another table.
- **Studies of different tables are compared through a unit key.** Declare the column that
  identifies the unit each row observes, and the identifier space it belongs to:

  ```python
  key = ant.UnitKey("patient", namespace="hospital_mrn")
  benefit_study = ant.prepare(trial_a, graph=..., query=..., unit_key=key)
  harm_study = ant.prepare(registry_b, graph=..., query=..., unit_key=key)
  ```

  Two studies keyed in the same namespace share exactly the units both observe, each unit
  counted once; rows excluded from the analysis are not units the study observes. The column is
  read from the data as supplied, so it need not be an analysis variable, and identifiers keep
  their type (the text `"1"` and the integer `1` are different units). A key applies to a static
  `ResponseCurve` on one table and is refused elsewhere (`option_not_applicable`).
- **Keys that cannot be compared leave the overlap unknown.** Keys in different namespaces, or
  a key on one study and none on a study of another table, say nothing about which units the
  two share, so the compile refuses with `evidence_overlap_unknown`. Key both studies in one
  namespace, or declare their cross-study covariance with `shared=`.
- **Unkeyed studies of different tables are read as distinct evidence.** A subset of a table
  passed to `prepare` is a table of its own. If two tables may hold the same units, declare a
  unit key: without one, the compile has nothing to compare.

A refusal for shared evidence says how many observations (`200 shared observations`) or units
(`100 shared units in namespace hospital_mrn`) the pair shares, and only those pairs need a
declared block. In Rust, a unit key is `StudyBuilder::unit_key(UnitKey::column(id, namespace))`,
`UnitKey::text_labels(namespace, ids)` for one text identifier per row (any text, a `:`
included: `2023:017` is the text `2023:017`), or `UnitKey::canonical_labels(namespace, labels)`
for labels already in the canonical type-tagged form (`text:<id>`, `int64:<decimal>`,
`float64:<16 hex digits>`, `bool:<0|1>`), each checked when the study is built. Where a
study's rows came from is carried beside its table, never in it: a study of a column
projection declares its source with
`StudyBuilder::row_provenance(RowProvenance::of_table(&source)?)`, and
`RowProvenance::select_rows(&table, &rows)` returns the selected rows together with their
provenance (their positions in the first source table), to declare the same way. Two
projections of one table then share every row, and row selections share exactly the rows both
took; `RowProvenance::new(table, Some(rows))` names rows of a table derived some other way. A
study without a declaration is a study of its own table's rows. A key column that is the
study's treatment or outcome is refused (`preference_compile.unit_key_invalid`).

## Decide across uncertain causal structures

When a study carries a graph posterior, `contract.compile(study)` compiles each of its
structures and decides across them under the contract's `structural_policy`; see
[compile with the analysis](#compile-with-the-analysis). The members of a CPDAG or PAG atom are
read as graph-dependent claims unless you declare a measure over them; see
[weigh the members of an equivalence class](#weigh-the-members-of-an-equivalence-class).

On structures you already hold, pass them to `evaluate` in place of the source:

```python
structures = decision.Structures.weighted_atoms(claims)
contract = decision.Contract(value=value, actions=actions, target_population="target",
                       structural_policy="maximin")
decided = require_success(contract.evaluate(structures))
decided.structures       # each structure's status and its own verdict
```

`claims` are `decision_robust.Claim` values (evaluated with draws, unidentified, or unevaluated
with a reason), read as the [structural decision routes](sensitivity-and-robustness.md) read
them. Build the set by how the claims are read: `Structures.point(claim)`,
`Structures.graph_dependent(claims)`, `Structures.weighted_atoms(claims)` (graph-posterior atoms
whose probabilities weigh them) or `Structures.finite_scenarios(claims)`. The same kinds are
accepted as a string, `Structures("weighted_graph_posterior", claims)`.

The admissible combinations are **every pairing of an admissible structure with an admissible
trade-off**. A conclusion that holds across structures holds for each structure at each
trade-off. You cannot state that a trade-off goes only with some structures: linked pairs are
not supported.

| `structural_policy` | How structures combine |
| --- | --- |
| `"require_invariant_best_action"` | Every structure is evaluated and every structure's verdict must agree; otherwise `structures_disagree` lists each structure's verdict. |
| `"maximin"` | Each action is scored by its worst value over structures at each trade-off, then the preference policy is applied. Every structure must be evaluated. |
| `"bayes_over_structures"` | Structure probabilities you supply weight the per-structure values; they are never renormalized. |
| `"report_only"` (the default) | Each structure's result, with no combination. |

`structural_policy=` takes the kind as a string; a kind that does not exist raises
`CausalValueError`, as an unknown `TradeoffPolicy` kind does.

Unidentified and unevaluated structure mass is reported (`unidentified_mass`,
`unevaluated_mass`). A claim that needs every structure returns `insufficient_science` when one
is unresolved. An action excluded by a hard constraint in any evaluated structure is excluded
everywhere; under `bayes_over_structures` only a structure with positive probability excludes.
These rules are those of [structural uncertainty](decisions.md#structural-uncertainty) in a
decision contract, with one difference on the exact route, stated in
[fixed trade-offs](#fixed-trade-offs-and-the-tie-tolerance); see also
[graph uncertainty](graph_uncertainty.md).

### Weigh the members of an equivalence class

A CPDAG or PAG atom of a graph posterior stands for every member of its equivalence class. The
compile evaluates each member as its own structure. Without a declared measure over the members
they are graph-dependent claims: `"maximin"`, `"require_invariant_best_action"` and
`"report_only"` read them, and `"bayes_over_structures"` is refused at compile
(`decision_adapters.completion_count_not_probability`), because a count of members is not a
probability.

Every member of a Markov equivalence class fits the data equally well, so no amount of data
says how likely one member is relative to another. A measure over members is a declaration you
make, never an estimate. Declare it with `class_prior=` beside the posterior:

```python
from antecedent.inference import ClassPrior

study = ant.prepare(
    table,
    discovery=posterior,                       # a GraphPosterior of CPDAG or PAG atoms
    query=ant.ResponseCurve("t", "y", grid=[0.5, 1.5]),
    inference=ant.Bayesian(backend="conjugate"),
    class_prior=ClassPrior.uniform("assumption.uniform_within_class"),
)
compiled = require_success(contract.compile(study))
```

A member's probability is then its atom's posterior weight times its mass within the class,
and every structural policy reads it, `"bayes_over_structures"` included. Two measures are
available:

- **Uniform** (`ClassPrior.uniform(assumption)`): equal mass on every member, under the named
  assumption. For a PAG, uniform is over the maximal ancestral graphs it represents, not over
  DAGs with latent variables.
- **Keyed** (`ClassPrior.from_pairs([(fingerprint, mass), ...])`): masses keyed by the member
  fingerprints the compile reports (`compiled.inspect()["structures"][i]["fingerprint"]`),
  normalized within each class. A graph that is a member of two class atoms has one
  fingerprint and is keyed once. A missing or unknown fingerprint, masses that sum to zero over
  a class, or ordered masses, which name no member, are refused
  (`preference_compile.class_prior_keys_mismatch`).

A class with at most 32 members (or the limit you declare with `.exact_limit(n)`) is enumerated
exactly. A larger class, or one whose member search stops at its budget, is refused
(`preference_compile.class_completions_truncated`) unless a uniform measure declares a sample:

```python
prior = ClassPrior.uniform("assumption.uniform_within_class").sample(400, seed=9)
```

`sample(members, seed=...)` draws that many members of each class above the limit, uniformly
and with replacement. A CPDAG's members are drawn exactly uniformly by clique-picking from an
exact count of the class (Wienöbst, Bannach and Liśkiewicz, 2021). A PAG's are drawn by setting
each circle mark to a tail or an arrowhead at random and keeping the draws that are members,
within a search budget, so the draws are exactly uniform over its members; its class size is
exact only when its enumeration completed, and otherwise a lower bound (the members enumerated
before the search stopped, or the distinct members drawn, whichever is larger). Only a uniform measure is sampled, since keyed masses need every
member's key (`preference_compile.class_prior_sampling_needs_uniform`), and a draw the budget
cannot complete is refused (`preference_compile.class_sampling_budget`). `compiled.class_sampling`
and `decided.sampling` state each draw: the scheme, the class size and whether it is exact, the
members drawn, the distinct members and the seed.

A sample never gives a verdict over every member, so a sampled decision is never
`necessarily_best`:

| `structural_policy` | Verdict on sampled classes | What comes with it |
| --- | --- | --- |
| `"bayes_over_structures"` | `best_on_sampled_measure` or `undetermined_on_sampled_measure`: the verdict on the sampled measure, a Monte Carlo estimate of the declared one | `decided.margins`: each pairwise expected-value margin at each vertex of the admissible trade-offs, with its Monte Carlo standard error (exact route) |
| `"maximin"`, `"require_invariant_best_action"` | `best_on_every_sampled_member` or `per_sampled_member`: each sampled member's own verdict, with no combined verdict | per sampled class, a one-sided 95% Clopper–Pearson upper bound on the declared mass where another action is best (about `3/m` of the class mass when none of `m` draws disagree), in the structural verdict's `other_best` |

For every sampled class, `decided.sampled_masses` gives the identified, unidentified and
unevaluated mass, each as an estimate with an exact 95% Clopper–Pearson interval (Clopper and
Pearson, 1934), scaled by the class mass. Unidentified and unevaluated mass is reported, never
renormalized.

The class prior is read only by a decision compiled across the posterior's structures. An
ordinary estimate of the same prepared study treats each class atom as a whole and refuses the
prior rather than ignore it (`structure_posterior.class_prior_not_read`).

## Fixed trade-offs and the tie tolerance

A plain number as a weight is not a trade-off at all: `benefit - 1.0 * harm` is the ordinary
[decision contract](decisions.md#a-fixed-trade-off-is-an-ordinary-contract) whose utility is the
expression, with its identity, its `Decision` result, its exact ties and its exact replay.

A trade-off that is named and fixed (`decision.tradeoff("harm_weight", units=..., value=1.0)`)
stays in the trade-off engine, so the result reports its assumption. Its verdict, admissible
actions, exclusions (under your constraint ids) and criterion values are those of the ordinary
contract, under every criterion and structural policy. The words differ: `uniquely_optimal` is
`necessarily_best`, and `indistinguishable` is `tied` on the exact route and `undetermined`
among the same actions on the certified route. The results differ only where the rules below
say so.

| Where | Ordinary contract | Contract with a trade-off |
| --- | --- | --- |
| Ties, exact route | criterion values equal to the last bit | margins within the effective tie tolerance (the declared `tie_tolerance`, default `1e-9`, or the numerical error bound of the margins when larger); with `tie_tolerance=0` the two agree whenever the margin exceeds that bound |
| Ties, certified route | criterion values equal to the last bit | a choice only when the outward-rounded enclosures separate by more than `tie_tolerance`; a utility exactly on a threshold probability's threshold, or values that close, leave the choice `undetermined` |
| Criterion values | `E[sum of terms]` | `sum of E[term]` on the exact route; equal up to floating rounding |
| Expected regret | regret amounts, `max_regret` and `evpi` | ranked by expected-utility margins (the same ranking); no regret amounts |
| Invariance across structures, exact route | each structure's own leader | an action excluded in one structure is excluded in every structure before each is analysed (the certified route keeps each structure's own leader) |
| Replay | exact | within `1e-9` relative |
| `effective_tie_tolerance` | `0.0` | the tolerance applied |

## Two routes and what each guarantees

Antecedent picks the route from the contract. `decided.route` is `exact` or `certified`, and
`decided.route_reason` names why (`preference_route.*`).

| | Exact route | Certified route |
| --- | --- | --- |
| Used when | the value is affine in the trade-off parameters, the criterion is expected utility, expected loss or expected regret, and the preference set holds no value relations | any other supported value or criterion, and any set with value relations |
| Method | linear programs over the admissible set | interval subdivision of the declared box |
| Source | means (when every part is affine in the attributes) or aligned joint draws | aligned joint draws only |
| Undecided share | none | reported when the budget runs out |

`evaluate` takes `tie_tolerance=`, `vertex_cap=`, `max_boxes=` and `min_width_fraction=` to
tune the routes.

Preferences [learned from recorded choices](preference-learning.md#shape-parameters) under a
value with shape parameters hold *value relations*: each says that a chosen option's value was
at least a rejected option's, at the trade-off point. They are not linear in the parameters, so
a set that holds them is decided on the certified route (`route_reason` is
`preference_route.value_relations`), which places every box against every value relation as it
places boxes against linear relations. The exact route refuses such a set
(`preference_decision.value_relations_need_certified_route`).

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
and Hanson. Margins within the tie tolerance count as ties: the declared tolerance (`1e-9`
criterion units by default) or, when larger, the numerical error bound of the margins (the
rounding of the expected values, which grows with their magnitude, plus the linear-program
tolerances times the margin's slope). The tolerance applied is reported as
`effective_tie_tolerance`.

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
- **A non-empty set.** The ranges and linear relations are checked by linear program before any
  box is placed. Relations that conflict jointly by more than its feasibility tolerance are
  refused as an empty set (`preference_certified.preference_set_empty`), and so is a set whose
  every box is proven outside some relation.
- **A witnessed point behind every choice.** The linear program works to a tolerance, so its
  point is not proof that the set holds one. A choosing verdict (`necessarily_best`,
  `worst_case_choice`, `minimax_regret_choice`) needs a kept box wholly inside every relation, or
  a point at which every relation is proven to hold with no tolerance (linear relations in
  arithmetic that widens only an inexact operation, so touching bounds such as `t >= 0.5` and
  `t <= 0.5` are proven at `t = 0.5`). Without one the verdict is `undetermined`: relations
  that miss each other by less than the feasibility tolerance (`t >= 0.5 + 1e-10` with
  `t <= 0.5`) are neither a choice nor called empty.
- **Equalities solved.** Each equality relation is solved for one parameter, so boxes lie inside
  the equality and the fractions are over the set's own dimensions.
- **Value relations met.** The witness above also holds every value relation: at a point, its
  outward-rounded gap enclosure is at least minus its tolerance.
- **No point labels.** A box is certified for an action only when its enclosures prove that
  action better than every admissible rival by more than the tie tolerance on the whole box.
- **Boundary boxes counted in full.** A box that straddles a relation is kept, flagged as
  touching the boundary and counted in full, although part of it lies outside the set.
- **Leads at measure points read from enclosures.** Under a weighted policy each measure point
  is also evaluated in outward-rounded interval arithmetic, and a lead counts toward an action's
  share only when the enclosure of its margin there exceeds the tie tolerance; the choice reads
  the measure-weighted margin enclosures the same way. A value that cancels large terms (a
  benefit and a price times a cost near `1e9` leaving a value near `1`) can lead by pure rounding
  in floating point, and such a lead does not count. `effective_tie_tolerance` reports the tie
  tolerance plus the widest measure-weighted margin enclosure; on boxes it is the tie tolerance.
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
replayed = require_success(decision.replay(exported, contract=contract, source=source))
```

Replay loads only under the contract and draws you retained, recomputes the decision and
compares every stored field; other draws or an edited number refuse.

## Refusals you are likely to meet

Each refusal is a `decision.PreferenceRefusal` (a `decision.DecisionRefusal`) with a registered `code` and a namespaced `detail`. The
compiled route reports a cancelled compile or execution through it too, as
`preference_compile.cancelled` (`cancelled_no_claim`). An error of the analysis behind the
decision (the study or design refusing to compile or fit) raises the analysis's own exception,
as the analysis would alone, with its registered reason code. A malformed
argument (a `TradeoffPolicy` kind that does not exist, a `shared=` block of the wrong shape, an argument
`compile` does not read) raises `CausalValueError` before anything runs, and exceeding the
execution's memory limit raises `CausalResourceError`, as elsewhere. A refusal of the ordinary
decision evaluator (a quantity the source does not hold, a non-finite utility) is that refusal
exactly: the same code, detail and `evaluate` stage as in a decision contract. A declaration
check reached by an evaluation keeps its code and detail and is reported at `evaluate`, as a
decision contract reports one, and a structure check stays at `structural`. The full tables are
in [preference sets and recorded choices](reference/refusal-codes.md#preference-sets-and-recorded-choices).

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
| `native_claims.response_not_point_identified` | `effect_not_identified` | At compile, no structure the study carries identifies a quantity | Declare a structure or design under which it is identified |
| `preference_decision.value_relations_need_certified_route` | `route_not_supported` | The exact route was asked to decide a set learned under shape parameters, which holds value relations | Decide on the certified route, which the route choice does by itself |
| `preference_compile.studies_share_evidence` | `evidence_double_counted` | Compiled studies share observations (rows of one table, or units of one key namespace) and no cross-study covariance is declared; the refusal names how many | Declare it with `shared=`, or estimate the outcomes jointly in one design |
| `decision_evidence.dependence_unknown` | `evidence_overlap_unknown` | Studies of different tables whose overlap cannot be judged: unit keys in different namespaces, or a key on one study only | Key both studies in one namespace, or declare their covariance with `shared=` |
| `preference_compile.unit_key_invalid` | `invalid_argument` | A study's unit key column is its treatment or its outcome | Name a column that identifies the units the rows observe |
| `preference_compile.cross_study_covariance_invalid` | `invalid_argument` | A declared block has the wrong shape, a non-finite term or only zero terms, or names studies that share nothing | Declare one finite block, not all zero, per overlapping pair |
| `preference_compile.unit_label_types_differ` | `evidence_overlap_unknown` | Two studies keyed in one identifier space carry unit labels of different types (for example text in one, integers in the other) | Key both studies by labels of one type |
| `preference_compile.executed_program_mismatch` | `route_not_lowered` | An executed result reports a program other than the decision program its plan states | Execute the compiled decision's own prepared studies |
| `preference_compile.sampled_mixture_not_exact` | `route_not_supported` | A mixture of sampled class members was read under a policy other than report-only | Report the sampled shares, or enumerate the class |
| `preference_compile.design_sampling_undeclared` | `joint_law_required` | A cluster-randomized, matched-pair or switchback design asked for a joint law with no declared sampling premise | Declare `sampling=` with `refits=` (at least 99), or decide with an affine value and an expectation, which its arm means answer |
| `preference_compile.design_units_too_few` | `too_few_clusters` | The design has fewer than 30 clusters in an arm, 30 matched pairs or 30 switchback sequences for its independent-unit bootstrap | Add units, or decide with an affine value and an expectation |
| `preference_compile.design_exposure_bounded` | `effect_not_identified` | The design's exposure declares bias bounds | Declare no interference, or decide over the bounded set robustly |
| `preference_compile.design_not_supported` | `route_not_supported` | The compiled design has no arm coordinates the decision reads | Use a static-regime, cluster-randomized, matched-pair or switchback design |
| `preference_compile.cancelled` | `cancelled_no_claim` | The execution's cancellation token was set during compile or execute | Run again without cancelling; the stop is not a verdict |
| `preference_compile.atom_kind_not_compiled` | `route_not_supported` | Lagged temporal graph-posterior atoms, or a structure the single-graph route does not support | Decide over a posterior whose atoms the single-graph compiler compiles |
| `preference_compile.class_completions_truncated` | `route_not_supported` | An equivalence-class atom has more members than the exact limit (32 unless declared), or its member search hit its budget, and no uniform class prior declares a sample | Declare `ClassPrior.uniform(...).sample(...)`, raise the limit, orient edges or declare the structure |
| `preference_compile.class_prior_keys_mismatch` | `invalid_argument` | A keyed class prior misses a member fingerprint, names one that is no member, sums to zero over a class, or gives ordered masses | Key the masses by the fingerprints the compile reports, or declare a uniform measure |
| `preference_compile.class_prior_sampling_needs_uniform` | `invalid_argument` | A keyed or ordered class prior declares a sample | Sample under a uniform measure, or key every member and raise the exact limit |
| `preference_compile.class_sampling_budget` | `route_not_supported` | The declared draws of a class do not fit the search budget (a PAG whose draws are rarely members, or a count beyond its bound) | Draw fewer members, orient edges or declare the structure |
| `decision_adapters.completion_count_not_probability` | `decision_contract_unsatisfied` | `bayes_over_structures` over CPDAG or PAG members with no class prior declared | Declare a class prior, use another structural policy, or supply genuine structure probabilities |
| `structure_posterior.class_prior_not_read` | `option_not_applicable` | An ordinary estimate of a graph posterior of CPDAG or PAG atoms was given a class prior, which only a compiled decision reads | Compile the decision, or drop the class prior for this estimate |

## What stays out of reach

- A probability over trade-offs that nobody supplied or estimated. Ranges stay ranges.
- Linked pairs of structure and trade-off. Every admissible structure is paired with every
  admissible trade-off.
- A certified answer from means. The certified route reads aligned joint draws.
- A joint law of a cluster-randomized design or a matched-pair or switchback experiment under
  randomization alone. Randomization does not identify the cross-arm dependence of their
  arm-mean estimators, so a joint law needs your declared premise that the units were sampled
  from a larger population.
- Graph posteriors with lagged temporal atoms in the compiled route: the decision binds a static
  mean response curve of tabular data, and lagged atoms need a temporal query on temporal data.
- A graph posterior among several separately fitted studies in one compile.
- A measure over the members of an equivalence class learned from data. The data cannot tell
  the members apart, so the measure is always your declaration.
- A verdict over every member of a sampled class. A sample supports a Monte Carlo estimate on
  the sampled measure, or each sampled member's verdict with a bound on the mass that disagrees.
- The overlap of studies of different tables without a unit key in one namespace. Unkeyed, they
  are read as distinct evidence; keyed in different namespaces, the overlap is unknown.
- Hard constraints that depend on trade-offs, and penalties in place of constraints.
- Unbounded ranges. Every parameter needs finite ends.
- The worst case over trade-offs of expected regret on the exact route; it is referred to the
  certified route.
- A judgment of which trade-offs are right. The verdict follows from what you declared.

## Entry points

| Language | Entry |
| --- | --- |
| Python | `antecedent.decision`: `attribute`, `tradeoff`, `relation`, `Curve`, the shapes, `Value`, `Action`, `Constraint`, `TradeoffPolicy`, `Measure`, `Structures`, `Contract(value=...).compile(...)` with `CompiledDecision` (`quantities`, `representation`, `carrier`, `trade_offs`, `parameter_ranges`, `joint_law`, `design_law`, `contract_identity`, `.class_sampling`, `.inspect()` and `.execute()`; `CompiledQuantity`, `TradeOffRange`, `JointLawPlan`, `JointLawInput`, `DesignLawPlan`), `Contract(value=...).evaluate(...)`, `TradeoffDecision` (with `sampling`, `sampled_masses`, `margins`), `ClassSampling`, `SampledMass`, `SampledMargin`, `replay`; `antecedent.UnitKey` with `antecedent.prepare(unit_key=...)`; `antecedent.inference.ClassPrior` (`uniform`, `from_pairs`, `exact_limit`, `sample`) with `prepare(class_prior=...)` |
| Rust, compiled route | `Study::compile_for_preference_decision`, `PreparedStudy::compile_for_preference_decision`, `PreparedNativeJointDesign::compile_for_preference_decision`, `compile_preference_decision_across_studies`, `compile_preference_decision_with_shared_evidence`, `compile_preference_decision_across_prepared_studies`, `CompiledPreferenceDecision::execute`, `CompiledPreferenceDecision::chain_plan_view`, `CompiledPreferenceDecision::class_sampling`; `UnitKey`, `RowProvenance`, `StudyBuilder::unit_key`, `StudyBuilder::row_provenance`, `RowProvenance::of_table` and `RowProvenance::select_rows`; `antecedent::ClassPrior` (`uniform`, `from_pairs`, `exact_limit`, `sample`) with `StudyBuilder::class_prior` |
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
Problems*, doi:10.1137/1.9781611971217. Wienöbst, M., Bannach, M. and Liśkiewicz, M. (2021),
polynomial-time algorithms for counting and sampling Markov equivalent DAGs, AAAI 35. Clopper,
C. J. and Pearson, E. S. (1934), the use of confidence or fiducial limits illustrated in the
case of the binomial, doi:10.1093/biomet/26.4.404. Each result names these through its method
provenance, with the documented deviations; see [citing a study](guides/citing-a-study.md).

## Where next

- [Decisions](decisions.md): the guide this page deepens
- [Learn trade-off ranges](preference-learning.md): optional, for when you want past choices to
  inform the ranges
- [Choose a discount when the cost of a return is a range](tutorials/decide-across-trade-offs.md):
  the worked story
- [Evaluate actions with a decision contract](decisions.md) and
  [probe sensitivity to assumptions](sensitivity-and-robustness.md)
