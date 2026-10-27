# Ask which actions reach a target

`antecedent.inverse.inverse_outcome` asks the forward question backwards over a list
you enumerate: *which of these interventions reach a target mean?* The answer is a point classification of each action you list; nothing outside the list is excluded. The goal is
`E[Y^do(a)] >= threshold` (or `<=`) for the outcome, population and horizon of an
existing, licensed forward response route. The claim is `point_only`.

```python
import antecedent as ant
from antecedent.inverse import Action, TargetMean, inverse_outcome

grid = [0.0, 0.5, 1.0, 1.5, 2.0]
study = ant.prepare(
    data, graph=edges, query=ant.ResponseCurve("t", "y", grid=grid),
    estimator="response.kennedy_dr", refute="none",
    inference=ant.Frequentist(bootstrap=0),
)
forward = ant.estimate(study)      # an Outcome; its .value is the forward response view

report = inverse_outcome(
    forward.value,
    query=TargetMean(threshold=3.0),                      # E[Y^do(a)] >= 3.0
    actions=[Action(f"dose_{a}", a, cost=a, constraints={"in_stock": a <= 1.5}) for a in grid],
    budget=1.6,
)
report.feasible, report.cheapest_feasible, report.enumerated
report.action("dose_1.0").status, report.action("dose_1.0").margin
```

The forward response is never re-estimated here. Run the existing route on a grid that
contains every action's point; the inverse query classifies each action against what that
route published. An `allowed_unlicensed` compatibility result cannot license this inverse
claim and is refused.

## Three kinds of answer

1. **Feasible interventions** (this cell). Every enumerated action is returned as
   `feasible`, `infeasible`, `unsupported` or `unevaluated`, with its cost, constraints,
   forward estimate, margin to the threshold, support label and the forward response's
   assumptions.
2. **Observational scenarios compatible with an outcome**. Closed:
   `ObservationalScenarios()` refuses with `cell_not_licensed`
   (`inverse.observational_scenarios`). `P(X | Y)` or a nearest-feature change is not an
   action, and nothing here recommends one.
3. **Missing evidence and study options**. Not computed here. Pass the study planner's plan
   as `study_options=` and a sensitivity result as `sensitivity=`; they are attached as
   `report.missing_evidence` and `report.sensitivity`, separate views that never change
   the classification or the report identity. A study option is a hypothetical catalog delta
   (`evidence_status="hypothetical_catalog_delta"`), not observed evidence and not an
   estimate of this outcome. A sensitivity range is an assumption range with tipping points,
   never a confidence interval (`is_confidence_interval` is `False`).

## How an action is classified

In order:

| Status | Reason | When |
| --- | --- | --- |
| `infeasible` | `constraint_violated` | a declared constraint is `False` (names listed) |
| `infeasible` | `over_budget` | its cost exceeds `budget` |
| `unevaluated` | `not_on_evaluated_grid` | its point is not a point of the forward evaluation; never interpolated |
| `unsupported` | `outside_empirical_support` / `missing_evidence` | the forward route's support label at that point |
| `unevaluated` | `non_finite_estimate` | the forward estimate is not finite |
| `unevaluated` | `numerical_margin_overflow` | finite estimate and threshold have a difference that cannot be represented as a finite margin |
| `feasible` / `infeasible` | `meets_target` / `misses_target` | `margin >= -tolerance` |

`margin` is the signed distance to the goal side of the threshold (positive meets it).
`tolerance` (default `1e-9`) is purely numerical: it makes an action exactly on the
threshold feasible and flags it `within_tolerance`. It is not a statistical margin.
`extrapolative` and `weak_overlap` support are reported on the action, not hidden and not
blocking.

On a static curve the forward route publishes one support label for the whole surface (the
worst over the grid), so the report copies it to every point and says so
(`support_basis="surface_worst_case"`). One out-of-support dose then marks every action
unsupported; split the grid and run the forward route per sub-grid to separate them. A
temporal dose-by-horizon surface has a label per point (`support_basis="per_point"`), and
actions on it are `(dose, horizon)` pairs.
One inverse report uses a single horizon declared by the forward temporal query;
split actions at different horizons into separate reports so their target means
retain one time coordinate.

## What "feasible" means

* **Relative to the enumerated set.** `enumerated` is `reachable` (some action is
  feasible), `unreachable_within_set` (every enumerated action is infeasible) or
  `undetermined` (none feasible, some unsupported or unevaluated). An action you did not
  list is unresolved, never excluded. "Necessary" and "sufficient" claims about actions
  hold only for this list and the forward response's declared assumptions
  (`report.assumptions`).
* **Point-based.** Classification uses the forward point estimate. If the forward route
  published a pointwise or simultaneous interval, each feasible action also carries
  `robustly_feasible`: `True` when that interval, used exactly as published, lies on the
  feasible side. It adds no coverage claim: the interval keeps its own level and
  `confidence`/`credible` reading (`report.interval`), a pointwise band gives no joint
  statement over the grid, and picking an action by its estimate is not adjusted for.
  `None` means no interval, or an action that is not point-feasible. There is no flag for
  an interval-free route, never a `False`.
  When a temporal result publishes both pointwise intervals and a separate simultaneous
  band, the inverse report uses the simultaneous band and labels its scope accordingly.
* **Identity.** `report.identity` is an order-invariant digest of the forward response as
  published (including its claim, program and data-snapshot IDs when present), the action grid,
  the target, budget and tolerance. `report.verify()` recomputes and compares the
  public classification against those stored inputs; it does not authenticate the
  forward model or protect against replacement of all stored inputs. Reordering
  actions or forward rows changes neither
  results nor identity; changing an assumption, a cost, a constraint or a support label does.

## Typed refusals

| Reason code | Detail | When |
| --- | --- | --- |
| `cell_not_licensed` | `inverse.probability_target` | `ChanceConstraint`: a mean cannot establish `P(Y^do(a) >= y) >= q`; that needs a separately identified, calibrated interventional-distribution cell |
| `cell_not_licensed` | `inverse.quantile_target` | `TargetQuantile` is unsupported by this mean-only route; the separate functional inverse query (`antecedent.inverse_query`, see [decision routes](../decisions.md#routes-at-a-glance)) requires a source supplying the requested functional |
| `cell_not_licensed` | `inverse.observational_scenarios` | `ObservationalScenarios` |
| `route_not_supported` | `inverse.forward_not_mean_response` | the forward estimand is not a scalar-outcome mean curve (`ResponseCurve`) |
| `route_not_supported` | `inverse.forward_not_point_identified` | the forward response is an identified set, a class mixture or unidentified |
| `cell_not_licensed` | `inverse.forward_not_licensed` | the forward result is marked `allowed_unlicensed` for compatibility |
| `invalid_argument` | `inverse.invalid_target`, `inverse.invalid_action`, `inverse.invalid_forward` | a non-finite threshold, negative tolerance or budget; an empty grid, duplicate label, wrong-dimension point, bad cost or constraint name; a repeated forward point or malformed interval |
| `transport_budget_cancel` | `inverse.cancelled` | cancelled mid-enumeration; never a verdict |

The probability, quantile and observational refusals are raised before any input is read.

## Inverse queries on composed, cohort and transported sources

Inverse queries and decisions can also consume a composed mechanism, a cohort-mean
contrast and a transported response. Each source states what it cannot supply
before the engine runs, and its identity travels into the result.

| Source | Rust entry | Supplies |
| --- | --- | --- |
| Composed mechanisms | `antecedent_design::inverse_forward_source::composed_forward_source` | Linear-Gaussian: per-action interventional mean (point). Finite table: the exact finite law |
| Cohort-mean contrast | `cohort_forward_source` | A point; under a declared covariate shift it is a transported response |
| Transported response | `transported_forward_source`, `decision_transported_source::evaluate_transported_decision` | A point, or a transported law |
| Observational scenarios | `inverse_observational_scenarios::evaluate_observational_scenarios` | `P(S)`, `P(S and E)`, `P(E | S)`, `P(S | E)`, lift |

Python entries: `antecedent.inverse_sources` (`composed_forward_source`,
`cohort_forward_source`, `transported_forward_source`, `ForwardSource.evaluate`,
`.seal`, `.seal_per_action`, `observational_scenarios`),
`antecedent.transported_response` (`TransportedResponse`,
`TransportDeclaration`, `evaluate_transported`) and
`Contract.evaluate_transported`.

### Point-only means point-only

A mean supplies the expectation of an affine utility and nothing else. A
quantile, a probability, a nonlinear utility, a hard constraint, regret or an
outcome-law input needs a law and is refused. No source supplies an interval
region: composition variance, a cohort model's standard error and the transport
step have no measured coverage. A source's own variance is kept as a diagnostic
and never becomes a feasibility statement. A Gaussian composition does not
supply a law the engine can integrate.

### Transported acceptance

A transported response is labelled at best extrapolative. A consumer that does
not accept a transported input refuses it with one code, `cell_not_licensed`,
whichever source carried it: an inverse query reading a cohort, composed or
response source under the native-target-only policy
(`inverse_forward_source.transported_not_accepted`), and a decision under the
default support policy, which admits only supported coordinates
(`transported_decision.transported_not_accepted`). Accepting is explicit: the
inverse query is told to accept transported inputs
(`accepts_transported=True` in Python), and a decision uses a policy that names
the weaker status. The receipt (source populations, target, diagram identities,
assumptions) travels into the result and the sealed artifact. A support rule of
the contract stronger than the label refuses `transport_support_failure`
(`transported_decision.support_rule_unmet`); declared exclusions remove actions
before the comparison and are part of the bound decision's identity.

A carried composition is a transported claim. A composed model whose
mechanisms were admitted into the target (carried from another population)
reaches an inverse query only with the receipt that carried it and an explicit
accept: its origin is labelled extrapolative with the receipt's support, never
supported. A composition with no such receipt, or a receipt of another route,
is refused.

### Typed sources and exported decisions

A transported point enters a decision through the typed constructors, which
take populations and identities from the producing result. The generic
`declared_point` / `declared_law` constructors are for a declared transport and
refuse the typed routes (policy, shifted cohort, synthesis); every typed route
requires digest transport identities, so a label typed by hand cannot stand in
for one. A decision on a transported point can be exported and consumed
(`transported_decision_v1`): the consumer rebuilds the contract, validates the
receipt, re-evaluates, and requires the identity and the action values to
match. A law is not exportable (`law_not_exportable`).

### Identity

`ForwardOrigin` records the source kind, identity, representation, support
label, assumed mechanisms or the whole receipt, and a digest of the claim handed
to the engine. `seal_on_source` binds it into the inverse-query artifact, and a
consumer re-derives the digest and refuses an origin that does not describe the
embedded claim. Replay checks the numbers against the origin; it does not
re-execute the producing route.

### Observational scenarios

`P(X in scenario | Y event)` from an aligned observational law over declared
scenario cells. It is associational: it ranks no action and recommends none.
The mass of the event no scenario explains is reported, not renormalized away.

### What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Quantile, probability or nonlinear utility on a point source | `PointOnlyLawRequested` / `decision_contract_unsatisfied` | Supply a law (draws or a finite table) |
| Quantile or probability on a Gaussian composition | `ComposedLawNotMaterialized` | Draw the intervened law |
| Interval region from any source | `IntervalNotLicensed` / `cell_not_licensed` | Use the point as labelled |
| Transported input to a consumer that does not accept one (inverse query, cohort forward claim, decision under the default policy) | `cell_not_licensed` (`inverse_forward_source.transported_not_accepted`, `transported_decision.transported_not_accepted`) | Accept transported inputs explicitly, naming the weaker support status in the policy |
| Support rule stronger than the transported label | `transport_support_failure` (`transported_decision.support_rule_unmet`) | Weaken the rule for this input or measure in the target |
| Typed route through the declared constructor, or a hand-typed identity | `transported_decision.route_needs_typed_source`, `transport_identity_not_digest` | Use the typed constructor of the route |
| Native or calibrated label on a transported value | `transport_not_certified` | Keep the transport label |
| Scenario query on an interventional law or independent marginals | `joint_law_required` | Supply an observational aligned law |
| Event with no mass | typed refusal | Choose a supported event |

Implementation limits are listed in [supported input bounds](../supported-bounds.md).
