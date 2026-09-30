# Joint mechanism deviations on the z route (2.2B X3)

This note derives the exact joint assumption range that
`antecedent_validate::z_transport_joint_mechanism_sensitivity` computes, states
what it claims and what it does not, and records the sampling composition that
is wired but closed. Record: `2.2B.X3.joint_mechanism_sensitivity` in
`parity/promotion_2_2.toml`. The derivation is our own; it is not taken from a
source.

## Scope

The only formula is the registered surrogate z formula, the one 2.1
`z_transport_mechanism_sensitivity` accepts:

    psi = sum_w P_W(w) [m(w, 1) - m(w, 0)],   m(w, x) = sum_y y P_Y(y | w, x, do(z)),

with one outcome, one binary treatment, one shared parent `W` and one cited
source regime. Two factors can deviate: the outcome kernel `P_Y` (contaminated
per `(w, x)` stratum, as in 2.1) and the shared parent marginal `P_W`. The
treatment is set by intervention and is never a factor.

Refused by design, each with a `joint_sensitivity.*` detail:

- a total budget coupling the fractions (`sum eps <= B`): `budget_coupling`;
- a fixed-graph parent-mechanism perturbation: `fixed_graph_parent`. The 2.1
  fixed-graph route requires the supplied target parent law, so perturbing it
  would contradict supplied target evidence;
- the fixed-graph conditional-mechanism joint problem: `fixed_graph_conditional`;
- source-target discrepancy diagnostics: `source_target_discrepancy` (2.3B);
- no factor or more than three factors: `factor_count`. Only two are supported.

## Declared bounds

At most 3 factors declared (only 2 are supported), 64 parent levels, 32 outcome categories,
a frontier grid of 1 to 33 points, at most 100000 search operations, a bisection depth of
64 iterations per frontier line, and a bootstrap request cap of 2000 (internal; the interval
route is closed). A request above a bound refuses as a typed `joint_sensitivity.*` detail.

## Contamination class

Each declared factor `i` is box-independently contaminated,

    Q_i = (1 - eps_i) P_i + eps_i R_i,   0 <= eps_i <= e_i,   R_i in its simplex,

and `R_Y` is free separately in each `(w, x)` stratum.

## Derivation

**Containment.** For `e > 0`, the set `{(1 - eps) P + eps R : eps <= e}` equals
the set at `eps = e`. Write
`(1 - eps) P + eps R = (1 - e) P + e [((e - eps) P + eps R) / e]`; the bracket is
a distribution. So the extremal range over the box is attained at the corner
`eps = e`, and there is no continuous fraction optimization.

**Multilinearity.** Given every other replacement, `psi` is linear in each `R_i`.
Its extrema over the product of simplices are therefore attained at
product-polytope vertices. For a fixed `Q_W >= 0`, the kernel optimum decouples
by stratum. The maximum puts all replacement mass on `y_max` in the active arm
and on `y_min` in the control arm:

    D+(w) = (1 - e_Y) Delta(w) + e_Y (y_max - y_min),   Delta(w) = m(w,1) - m(w,0) at the source,

and then the parent optimum puts `R_W` on `argmax_w D+(w)`:

    U = (1 - e_W) sum_w P_W(w) D+(w) + e_W max_w D+(w).

The lower end is symmetric, with `D-(w) = (1 - e_Y) Delta(w) - e_Y (y_max - y_min)`
and a min over `w`. The cost is O(strata x categories). The code computes the
kernel stage with the unchanged 2.1 `DiscreteKernelSensitivity` evaluator, which
gives `A = sum_w P_W(w) D+(w)`, and then `U = A + e_W max(max_w D+(w) - A, 0)`.
The clamp only removes rounding, since `max >= average` holds exactly. At
`e_W = 0` the kernel stage is returned unchanged, so a single kernel factor is
bit-equal to the 2.1 output.

**Numerical check (not a proof).** The rule was checked in a throwaway script
before it was relied on: 200 random models, with up to 3 parent levels and 3
outcome categories, were enumerated over every vertex (fraction corners, every
parent vertex, every per-stratum outcome vertex). The largest disagreement was
8.9e-16. No random interior point left the range, and nested boxes nested.

The committed test `exact_range_equals_brute_force_vertex_enumeration`
(`crates/antecedent-validate/tests/joint_mechanism_sensitivity.rs`) repeats the
enumeration at 1e-12 on 40 seeded models. It also checks 8000 random interior
points.

**Monotonicity and nesting.** `max_w D+ >= sum_w P_W D+` and
`y_max - y_min >= Delta(w)`. So `U` is non-decreasing in each fraction, and `L`
is symmetrically non-increasing. Nested boxes give nested ranges
(`nested_boxes_give_nested_ranges`, 480 seeded comparisons).

**Tipping.** Take a threshold above the baseline. The set of deviations that
reach it is an up-set of the box. With the parent fraction fixed, `U` is convex
and piecewise linear in the kernel fraction; with the kernel fraction fixed, it
is linear in the parent fraction.

The frontier holds the parent fraction on a grid of 1 to 33 points and bisects
the kernel fraction. Each resolved line carries a certified bracket
`[lower, upper]`: the response does not reach the threshold at `lower` and does
at `upper`, and `upper - lower <= tolerance` (between 1e-12 and 1e-2). The
bracket is the unresolved region.

On the axes the 2.1 one-factor analytic values are reported:

- the kernel axis uses `DiscreteKernelSensitivity` on the kernel;
- the parent axis uses it on `P_W` with the outcome values `Delta(w)`, which is
  the 2.1 root-mechanism construction.

Tests check that each value lies in its bracket. A threshold below the baseline
is symmetric, using `L`.

**Budget.** The range stage is charged once per parent level. Each bisection
step is charged at depth equal to its iteration. All charges go to one
`antecedent_core::SearchBudget` with cumulative live-state bytes, a memory cap
that is always present, and cancellation.

A stop before the range refuses (`transport_budget_cancel` /
`joint_sensitivity.budget`) with its receipt. A stop during bracketing keeps the
exact range and marks the remaining lines `unevaluated`, with the receipt. A
cancelled frontier is not exported, because it does not replay.

## What is claimed

- The range is exact within the declared box-independent contamination class:
  every value in it is attained and nothing outside it is. It is an assumption
  range. It is not a sharp identification claim beyond those assumptions, and it
  is never a confidence interval.
- The claim is conditional on the 2.1 checked z derivation and catalog binding,
  which the existing route verifies.

## Sampling uncertainty (wired, closed)

There is one composition: the percentile bootstrap of the exact endpoints,

    [ q_{(1-a)/2}(L*_b),  q_{(1+a)/2}(U*_b) ],

where each replicate is an iid row bootstrap of the cited count table, on the
same stream as the 2.1 nominal z interval, and gets its own exact range.

Fix any true deviation `delta0` in the box. On every replicate,
`L*_b <= psi*_b(delta0) <= U*_b`. So the interval contains the pointwise
percentile interval of `psi(delta0)`, and its coverage of `psi(delta0)` is at
least that interval's coverage.

The coverage target is therefore one-sided: asymptotic coverage of at least the
nominal level whenever the pointwise percentile bootstrap is consistent. That
consistency is **paper-inherited** and unverified here. At the zero box the
composition is exactly the ordinary percentile bootstrap.

References, from abstracts only (the theorem statements were not verified):

- Zhao, Small and Bhattacharya, arXiv 1711.11286, on percentile-bootstrap
  sensitivity intervals with at-least-nominal coverage;
- Fang and Santos, arXiv 1404.3763, on bootstrap failure for maps that are not
  fully differentiable, such as a tied max over `w`. The dominance argument
  above does not need the endpoints themselves to be consistent.

The estimator `joint_sensitivity_bootstrap_interval_internal` is compiled only
under the `calibration-internal` feature. Every public route refuses the
interval with `cell_not_licensed` / `joint_sensitivity.interval_withheld`, and a
consumer refuses an artifact that carries one.

The coverage harness (`crates/antecedent/tests/common/calibration.rs`) gates a
two-sided band with a precision ceiling. That band cannot express a one-sided
target: an interval that is conservative by design over-covers and fails the
ceiling. The zero-box record is two-sided nominal. The positive-box record
asserts only a one-sided floor and is emitted as a named boundary, so it never
counts as a nominal pass. The route stays closed for the 2.2 release.

## Artifact (version 3)

The artifact is a separate wire from the v2 one-factor artifact. The v2 artifact
is unchanged and refuses version 3; the v3 reader refuses version 2.

**Contents:**

- the embedded baseline point artifact;
- the perturbation definition and limits;
- the exact range with its vertex witnesses;
- the axis tipping points and frontier brackets;
- the optimization receipt;
- the interpretation and the inference claim;
- the declared sampling method (withheld, no interval);
- a premises digest and a separate data digest.

**What the consumer checks, in order:**

1. Stored limits against its own maxima and hard memory limit, before any work.
2. Both digests.
3. The interval and inference-claim labels.
4. It replays the baseline, recomputes the body under the stored limits and
   effective memory cap, and compares it bit for bit.

**What replay does not protect against:**

- a producer that forges the baseline laws consistently (the point artifact's
  own consumer checks only that its laws reproduce its point);
- a self-consistent but scientifically wrong choice of declared perturbation.

Replay proves the numbers follow from the stored premises, not that the premises
are right.
