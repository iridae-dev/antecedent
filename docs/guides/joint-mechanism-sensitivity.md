# Joint mechanism deviations on the z route (2.2B X3)

This note derives the exact joint assumption range that
`antecedent_validate::z_transport_joint_mechanism_sensitivity` computes and
states what it claims and what it does not. It also records the one sampling
composition that was designed for the range. That composition is **not offered
in 2.2**: its record is carried forward and every route to it refuses.

Two records in `parity/promotion_2_2.toml` cover this page:

- `2.2B.X3.joint_mechanism_sensitivity` (promoted): the exact assumption range
  only. It offers no sampling uncertainty.
- `2.2B.X3.joint_sensitivity_uncertainty` (carried forward): the conservative
  endpoint bootstrap, its closed routes and its two coverage ids, which are not
  measured at the 2.2 cut.

The derivation is our own; it is not taken from a source.

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
- no factor or more than two factors: `factor_count`. The formula has two
  factors, so a third declaration is always a duplicate or an out-of-scope
  factor; it refuses by count before its kind is examined.

## Declared bounds

At most 2 factors declared (the outcome kernel and the shared parent marginal), 64 parent levels, 32 outcome categories,
a frontier grid of 1 to 33 points, at most 100000 search operations, a bisection depth of
64 iterations per frontier line, and a declared memory cap of at most 512 MiB (default
64 MiB). The internal estimator of the carried-forward uncertainty record has a
bootstrap request cap of 2000 (no public route reaches it). A request above a bound refuses as
`joint_sensitivity.bounds_exceeded`.

The parent-level and outcome-category bounds are checked on the law's shape,
after the proof, formula and provider checks and before any cell of the law is
read. The other bounds are checked on the declared spec before anything else.

## Contamination class

Each declared factor `i` is box-independently contaminated,

    Q_i = (1 - eps_i) P_i + eps_i R_i,   0 <= eps_i <= e_i,   R_i in its simplex,

and `R_Y` is free separately in each `(w, x)` stratum.

The outcome values `y` are the numeric values of the law's `Y` axis, the
outcome levels the cited joint lists. They are not a declared outcome domain.
So `y_max - y_min` below is the spread of those listed levels: a replacement
kernel can move mass only among them. An outcome level with no mass in the
source still counts if the law lists it.

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
and piecewise linear in the kernel fraction: it is a max over `w` of functions
linear in `e_Y`. With the kernel fraction fixed, `U` is linear in the parent
fraction.

The frontier is a family of axis-parallel lines, not a ray from the origin. Each
line holds the parent fraction at one value of a grid of 1 to 33 points and
searches the kernel fraction on `[0, e_Y]`. No closed-form root is used. Each
line's first crossing is bracketed by bisection, and a resolved line carries a
bracket `[lower, upper]`. The response does not reach the threshold at `lower`
and does at `upper`. "Reaches" is `U >= threshold` above the baseline and
`L <= threshold` below it. `upper - lower <= tolerance`, which lies between
1e-12 and 1e-2. The bracket is the unresolved region.

The bracket is certified against the floating-point evaluation of `U` (or `L`)
by the closed form, the same evaluation that reports the range. It is not
certified in exact arithmetic. When the true crossing lies within a few
rounding errors of `lower` or `upper`, exact arithmetic could place it just
outside the bracket.

On the axes the 2.1 one-factor analytic values are reported:

- the kernel axis uses `DiscreteKernelSensitivity` on the kernel;
- the parent axis uses it on `P_W` with the outcome values `Delta(w)`, which is
  the 2.1 root-mechanism construction.

Tests check that each value lies in its bracket. A threshold below the baseline
is symmetric, using `L`.

**Budget.** All charges go to one `antecedent_core::SearchBudget`. It carries
cumulative live-state bytes, a memory cap that is always present, and
cancellation. What is charged:

- **Reading the law.** One operation is charged per parent level, before that
  level's three passes over every cell of the law. The charge carries the byte
  estimate of the factors held for a law of that shape. This is the only
  metered work before the range.
- **The range.** The closed-form range itself is not charged. It costs
  O(levels x categories) on factors already read.
- **Bracketing.** Each bracketing line is charged once at depth 0 on entry.
  Each bisection step is charged at depth equal to its iteration, and each
  resolved line adds its bytes to the cumulative live state.

The proof, formula and provider checks, and the shape bounds, run before the
read and are not charged. The effective memory cap is the smaller of the
declared cap (default 64 MiB, at most 512 MiB) and the execution context's hard
limit.

A stop while reading the law refuses (`transport_budget_cancel` /
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
- The guarantee `exact_range_complete_within_declared_contamination_class`
  covers the range only. Sampling uncertainty is not offered in 2.2, and the
  sampling method below is paper-inherited.

## Sampling uncertainty (not offered in 2.2; record carried forward)

Record `2.2B.X3.joint_sensitivity_uncertainty` is carried forward. Every route
to this composition refuses. Its two coverage ids are allocated, but its tests
are not registered in `scripts/gate_calibration.sh`, so nothing is measured at
the 2.2 cut. This section records what the composition is and why it cannot be
opened yet.

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

**Why the cell is carried forward.** Opening it is blocked on two things.

1. **A one-sided coverage role in the harness.** Today no layer can express a
   one-sided target:
   - `crates/antecedent/tests/common/calibration.rs` gates a two-sided band,
     level +- 3 MCSE;
   - `scripts/gate_parity_schema.sh` fails over-coverage;
   - `scripts/collect_coverage_records.py` knows only the `gated` and
     `named_boundary` roles.

   An interval that is conservative by design over-covers, so it fails that
   band.
2. **A truth at the extremal vertex.** The tight side of the target is
   `psi(delta0)` at the vertex that attains `U`, not the interior point
   `delta0 = 0`, which every box contains and which is the easiest point to
   cover.

The calibration tests in `crates/antecedent/tests/joint_sensitivity_calibration.rs`
now use the extremal-vertex truth. The non-ignored test
`the_calibration_truths_are_the_extremal_vertex_of_the_true_law` checks that
truth against the exact range on the true law. The two coverage tests stay
`#[ignore]`d and unregistered until the harness has a one-sided role.

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

1. Stored limits against its own maxima, the producer's 512 MiB memory-cap
   ceiling and its own hard memory limit, and `effective <= declared` for the
   memory cap, all before any work.
2. Both digests.
3. The interval and inference-claim labels.
4. It replays the baseline, recomputes the body under the stored limits and
   effective memory cap, and compares it bit for bit.

**What replay does not protect against:**

- a producer that forges the baseline laws consistently (the point artifact's
  own consumer checks only that its laws reproduce its point);
- a self-consistent but scientifically wrong choice of declared perturbation;
- a re-sealed change of the declared memory cap that stays at or above the
  stored effective cap and within the consumer's maxima. The numbers depend
  only on the effective cap. A producer whose context has a hard limit writes
  an effective cap below the declared one, so replay cannot re-derive the
  declared cap. The premises digest binds it against unsealed edits. The test
  `a_resealed_declared_memory_cap_is_bound_by_the_premises_digest_not_by_replay`
  pins this behaviour.

Replay proves the numbers follow from the stored premises, not that the premises
are right.
