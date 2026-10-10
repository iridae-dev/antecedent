# Finite-action inverse-outcome query (2.2 E6)

`antecedent.inverse.inverse_outcome` asks the forward question backwards over a list
you enumerate: *which of these interventions reach a target mean?* The goal is
`E[Y^do(a)] >= threshold` (or `<=`) for the outcome, population and horizon of an
existing, licensed forward response route. The claim is `point_only`. The record is
`2.2E.E6.finite_action_inverse_outcome` in `parity/promotion_2_2.toml`.

```python
import antecedent as ant
from antecedent.inverse import Action, TargetMean, inverse_outcome

grid = [0.0, 0.5, 1.0, 1.5, 2.0]
forward = ant.prepare(
    data, graph=edges, query=ant.ResponseCurve("t", "y", grid=grid),
    estimator="response.kennedy_dr", refute="none", bootstrap=0,
).estimate(data)

report = inverse_outcome(
    forward,
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
| `cell_not_licensed` | `inverse.quantile_target` | `TargetQuantile` is unsupported by this mean-only route; the separate [functional inverse query](../2_3-decisions-breadth.md) requires a source supplying the requested functional |
| `cell_not_licensed` | `inverse.observational_scenarios` | `ObservationalScenarios` |
| `route_not_supported` | `inverse.forward_not_mean_response` | the forward estimand is not a scalar-outcome mean curve (`ResponseCurve`) |
| `route_not_supported` | `inverse.forward_not_point_identified` | the forward response is an identified set, a class mixture or unidentified |
| `cell_not_licensed` | `inverse.forward_not_licensed` | the forward result is marked `allowed_unlicensed` for compatibility |
| `route_not_supported` | `inverse.bounds_exceeded` | more than 4096 actions or 65536 forward points |
| `invalid_argument` | `inverse.invalid_target`, `inverse.invalid_action`, `inverse.invalid_forward` | a non-finite threshold, negative tolerance or budget; an empty grid, duplicate label, wrong-dimension point, bad cost or constraint name; a repeated forward point or malformed interval |
| `transport_budget_cancel` | `inverse.cancelled` | cancelled mid-enumeration; never a verdict |

The probability, quantile and observational refusals are raised before any input is read.
