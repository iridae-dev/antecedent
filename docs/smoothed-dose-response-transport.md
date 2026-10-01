# Smoothed dose-response transport (2.2B X4)

One cell: a randomized continuous dose from a trial, transported to a covariate-shifted
target population, reported on a grid of at most 16 doses at one declared bandwidth.
Record `2.2B.X4.smoothed_dose_response_transport` in `parity/promotion_2_2.toml`;
refusal details live in the `dose_response` namespace. The scope summary is in
[the transport scope guide](guides/transport-scope.md#smoothed-dose-response-transport-22b-x4).

This page states the estimand, derives the estimator's score and its robustness, and
lists the numerical checks that verify the derivation. The derivation was written for
this cell. The kernel-smoothed doubly robust form is related to Kennedy, Ma, McHugh and
Small (arXiv:1507.00747) and Colangelo and Lee (arXiv:2004.03036), but only their abstracts
were read, and it is not theirs: Kennedy et al. target the unsmoothed curve, whose
pseudo-outcome needs the marginal dose density `m(A)`, while the smoothed target here needs
no `m(A)` because `pi` is known and the kernel is part of the estimand. The transported
composition is not taken from any paper, so every claim below is marked `paper-inherited`
or is checked by an executed test.

## Setting

- `S = 1` marks a trial (source) row and `S = 0` a target row.
- `X` holds the baseline covariates, which equal the certified standardizers.
- `A` is the dose and `Y` the outcome. Both are observed on source rows only.
- **Randomization with a known density.** In the source, `A | X` has a conditional
  density `pi(t | x)` that is known by design, is supplied at each source row, and is
  positive on a declared support `[lo, hi]`.
- **Sampling designs.** Two are allowed:
  - `nested_cohort`: one IID cohort, whose nonparticipants are the target.
  - `independent_samples`: an IID trial sample and a separate representative IID target
    sample, each of fixed size.

## Estimand

With `K(u) = 0.75 (1 - u^2)` on `[-1, 1]` (Epanechnikov), `K_h(u) = K(u/h)/h` and
`mu(t, x) = E(Y | X = x, A = t, S = 1)`:

```text
psi_h(a) = E[ nu_h(X; a) | S = 0 ],     nu_h(x; a) = integral K_h(a - t) mu(t, x) dt.
```

### Identification

Assume four things:

1. **Consistency:** `Y = Y^A`.
2. **Source randomization:** `A` is independent of `Y^t` given `X` and `S = 1`, and the
   density is `pi`.
3. **Mean exchangeability over `X`:** `E(Y^t | X, S = 1) = E(Y^t | X, S = 0)` for `t` in
   every kernel window.
4. **Positivity:** `P(S = 1 | X) > 0` on the target support, and `pi(t | x) > 0` on every
   window.

Under these assumptions, `mu(t, x) = E(Y^t | X = x, S = 0)`, so

```text
psi_h(a) = E_target[ integral K_h(a - t) Y^t dt ]
```

This is the kernel-smoothed dose response of the target population.

### What changes with h

The bandwidth and the kernel belong to the estimand: `psi_h` is a different target for
each `h`, and `h` is never tuned. Expanding `mu` to second order in `t` around `a` gives
`psi_h(a) = psi_0(a) + (h^2 / 10) E_target[d^2 mu / dt^2 (a, X)] + O(h^4)`. The factor
comes from the kernel's second moment, `integral u^2 K(u) du = 1/5`. The cell reports
`psi_h` and keeps smoothing bias apart from it (see below).

### A related functional

`psi_h(a)` equals the target mean under a stochastic dose `T ~ K_h(a - .)` drawn
independently of `X`. The functional is the same. The routes stay distinct because this
cell has its own query type (`SmoothedDoseTransportQuery`), its own declared smoothing
target, and its own refusals of point-curve, stochastic, coarsened, incremental and
derivative targets (`dose_response.target_not_smoothed`).

### Refused inputs

- Every window `[a - h, a + h]` must lie inside `[lo, hi]`. There are no boundary
  kernels, and an extrapolative grid dose is refused
  (`dose_response.grid_outside_dose_support`).
- An estimated density is refused (`dose_response.estimated_dose_density`).

## Score and estimator

Write:

- `p(x) = P(S = 1 | x)`;
- `omega(x) = (1 - p(x)) / p(x)`;
- `w_a(t, x) = K_h(a - t) / pi(t | x)`.

The composed score, averaged over rows and divided by `P(S = 0)`, is

```text
phi(O) = (1 - S) (nu_h(X; a) - psi_h(a)) + S omega(X) w_a(A, X) (Y - mu(A, X)).
```

**It has mean zero at the truth.** The target term is centred by definition. For the
source term, take the expectation given `A`, `X` and `S = 1`: the residual `Y - mu(A, X)`
then has mean zero, and the weight is a known function of `(A, X)`.

**Why the residual uses `mu(A, X)` rather than `nu_h(X)`.** The alternative
`S omega(X) (w_a Y - nu_h(X))` also has mean zero, because
`E(w_a(A, X) mu(A, X) | X, S = 1) = nu_h(X)`. The two forms differ by
`S omega (w_a mu(A, X) - nu_h(X))`, which is a function of `(A, X)` with mean zero given
`X`. That is a score of the density of `A` given `X`, and since `pi` is known that score
lies outside the model's tangent space. Removing it gives the residual form above.

Whether the residual form is the efficient influence function of the known-`pi` model is
`paper-inherited` and **not claimed**. No efficiency claim is made.

The estimator uses out-of-fold nuisances, with `n_0` target rows:

```text
psi_hat(a) = (1/n_0) [ sum_{S=0} nu_hat(X_i; a)
                     + sum_{S=1} omega_hat(X_i) w_a(A_i, X_i) (Y_i - mu_hat(A_i, X_i)) ].
```

This is the unnormalized form of the learned-trial cells, with `(1 - p_hat)/p_hat`
estimated in the pooled data. For independent samples the pooled odds equal the density
ratio times `n_0 / n_1`, and dividing by `n_0` cancels that factor.

## Model double robustness of the point

Let `mu_bar` and `omega_bar` be the limits of the fitted nuisances, and
`nu_bar(x) = integral K_h(a - t) mu_bar(t, x) dt`. The limit of `psi_hat(a)` is

```text
E[nu_bar(X) | S = 0] + E[ S omega_bar(X) (nu_h(X) - nu_bar(X)) ] / P(S = 0),
```

The second term uses
`E(w_a(A, X)(mu - mu_bar)(A, X) | X, S = 1) = nu_h(X) - nu_bar(X)`, which holds because
`pi` is the true density.

- **Outcome correct** (`mu_bar = mu`): then `nu_bar = nu_h`, the second term vanishes,
  and the limit is `psi_h`.
- **Membership correct** (`omega_bar = omega`): then
  `E[S omega(X) g(X)] = E[(1 - S) g(X)]` for any `g`, so the limit is
  `E[nu_bar | S = 0] + E[nu_h - nu_bar | S = 0] = psi_h`.

This is the whole claim. Both families wrong at once is outside it. No rate,
asymptotic normality or efficiency is claimed.

The argument needs `nu_bar` to be the exact integral of `mu_bar`. With quadrature, the
limit is off by at most the largest quadrature error over the rows. For a linear-family
fit that error is zero up to rounding (the windows are split at the knots, next section);
for any other fit it is estimated, not bounded, and reported separately.

## Numerical error and smoothing bias, kept apart

### Quadrature

`nu_hat(x; a) = sum_q v_q K(u_q) mu_hat(a + h u_q, x)` over Gauss-Legendre nodes and
weights `(u_q, v_q)` (`antecedent_stats::special::gauss_legendre`).

- **Splitting at the knots.** The basis knots are declared, so every window
  `[a - h, a + h]` (and every half-bandwidth window of the bias diagnostic) is split at
  the knots strictly inside it, and the `Q`-node rule is applied on each piece
  (`QuadratureRecord::pieces`).
- **Exactness.** A linear-family learner (linear, ridge, elastic net) on the dose basis
  fits a curve that is a polynomial of degree at most 3 in the dose on every piece; times
  the quadratic kernel that is degree at most 5, which a 16-node rule integrates exactly.
  So both rules are exact up to rounding, the record says `exact = true`, and no
  tolerance is gated. A hinge basis is covered: the kink sits on a piece boundary.
- **Other fitted curves.** A tree learner's curve is a step function whose steps are
  not at the knots, so no rule is exact. Each grid dose then compares the `Q`-node and
  `2Q`-node rules on every target row and records:
  - `max_row_error`, the largest `|nu_hat_Q - nu_hat_2Q|` over target rows (the gated
    quantity);
  - `estimate_error = |psi_hat_Q - psi_hat_2Q|`, its grid average.
  Both are **estimates**, not bounds: a sweep of 20,000 hinge-knot positions over an
  unsplit window (`splitting_at_the_knot_makes_the_hinge_exact_where_the_doubling_estimate_misses`)
  finds `|I_Q - I_2Q|` below the `2Q` rule's own true error at about 7% of positions
  (1440 of 20,000 at `Q = 16`, 1368 at `Q = 32`), and passing a `1e-6` tolerance while
  the true error exceeds it at 56 and 76 positions. That is why a polynomial fit is split
  rather than gated.
- **Which rule the point uses.** The `2Q` rule.
- **Refusal.** A non-exact grid dose whose `max_row_error` exceeds the declared
  tolerance is refused (`dose_response.quadrature_tolerance`).
- **Relation to sampling error.** Numerical error is never mixed into sampling error.

### Smoothing bias

The diagnostic is the plug-in difference

```text
Delta = mean_T nu_hat_h - mean_T nu_hat_{h/2}
```

When the fitted curve is locally quadratic, `Delta = (3/4)(h^2/10) E[mu'']`, so
`(4/3) Delta` estimates `psi_h - psi_0`. Both are reported, and neither is ever added to
the estimate or to an interval.

The diagnostic is computed from the fitted curve alone, so it inherits the fit's
misspecification. For a fit linear in the dose it is exactly zero, because
`integral K_h(a - t) t dt = a` for every `h`, whatever the true curvature (asserted in
the misspecification test). It says how much the *fitted* curve bends, not the true one.

### Diagnostic standard error

The analytic influence-function standard error is
`sqrt(sum_T (nu_hat - psi_hat)^2 + sum_S (omega w (Y - mu_hat))^2) / n_0`. It is a
diagnostic only and never a licensed claim.

## Numerical verification (executed)

All in `crates/antecedent/tests/smoothed_dose_lifecycle.rs`, on the structural model in
`crates/antecedent/tests/smoothed_dose_dgp/mod.rs`:

- **The structural model.**
  - Source `X ~ N(0, 1)`.
  - Dose density `pi(a | x) = (1 + 0.6 tanh(x)(a - 2)/2)/4` on `[0, 4]`, drawn exactly
    by rejection sampling.
  - Target `X ~ N(m_T, 1)`.
  - Outcome `Y = A^2 + X(1 + A/2) + e`.
  - Closed form: `psi_h(a) = a^2 + h^2/5 + m_T (1 + a/2)`.
- **`quadrature_matches_exact_integration_and_the_oracle_score_is_centred`** checks three
  things:
  - The closed form matches independent numerical integration of the model (a midpoint
    rule in the dose, Gauss-Hermite in `X`) to `1e-7`.
  - The estimator's quadrature of every stored fold model equals that model's
    closed-form kernel integral to `1e-10`, for degree-2 and degree-3 bases and for hinge
    bases with one or two knots inside every window (split into that many pieces plus
    one), including the `h/2` diagnostic, at a tolerance of `1e-300` that is never
    gated. The kernel moments are `1, a, a^2 + h^2/5, a^3 + 3 a h^2/5` and the hinge
    moment is given below.
  - The oracle score (true `mu`, true membership odds, known `pi`) on 70,000 rows is
    within four standard errors of `psi_h` at every grid dose.
- **`known_truth_curve_under_both_designs`**: the grid matches the closed form under both
  designs.
- **`smoothing_bias_is_reported_apart_and_the_estimate_targets_psi_h`**: with `h = 1.2`,
  the estimate is within 0.08 of `psi_h` and more than 0.2 from `psi_0`. The diagnostic
  matches `3h^2/20`, and `(4/3)` of it matches `h^2/5`.
- **`misspecified_nuisance_cases_match_only_the_claimed_robustness`**: one family wrong at a
  time, at the off-centre grid doses `0.75, 1.5, 2.5, 3.25` (where both the covariate
  shift and the dose-density tilt act), with a tolerance of four influence-function
  standard errors, under both designs.
  - Outcome wrong, membership right: the truth `Y = A^2 + X(1 + A/2) + X A^2 + e`
    (target `X ~ N(0.8, 1)`, 105,000 rows) and a fit linear in the dose. The fit's error
    is `(1 + X)` times the missed curvature, which varies with `X`. The estimator lands
    within tolerance; the plug-in misses by more than 2.5 tolerances at three or more
    doses; the smoothing-bias diagnostic is exactly zero. On the same fitted nuisances,
    the point with a constant odds weight `n_0/n_1` (membership ignored) leaves the
    tolerance at three or more doses, and the point with a constant (uniform) dose
    density in the residual weight leaves it at one or more; the test asserts both.
  - Membership wrong, outcome right: the same outcome, target `X ~ N(0.5, 1.4^2)` (log odds
    quadratic in `X`) against a linear logistic. The estimator lands within tolerance
    while the weighting-only estimator with the same fitted odds is more than two
    tolerances off at every dose, so the odds really are wrong.
- **`known_truth_curve_under_both_designs`** also recomputes the augmentation from the
  fitted out-of-fold nuisances and the design's density function and matches it to
  `1e-9`.
- **`a_kinked_fitted_curve_surfaces_its_quadrature_error_and_refuses_at_a_tight_tolerance`**:
  a linear fit on a hinge basis with its kink inside the window is integrated exactly
  (two pieces, point equal to the closed-form hinge moment
  `h * 0.75 (1/4 - 2d/3 + d^2/2 - d^4/12)`, `d = (c - a)/h`, at a tolerance of `1e-300`);
  a step-shaped tree fit is gated on the largest row difference, refused at a tolerance
  between the grid-averaged and the largest row difference, and accepted at the largest.
- **`a_target_row_below_the_membership_floor_refuses_even_when_every_source_row_clears_it`**:
  overlap is gated on target rows too.
- **`a_bootstrap_replicate_reuses_the_point_run_fold_of_every_drawn_row`**: a replicate keeps
  each drawn row's point-run fold (a duplicated row stays in one fold).

## Bounds

At most 16 grid doses, 20 cross-fitting folds, 200,000 rows, 256 covariates, basis degree 1 to 3,
at most 8 knots, and an interval bootstrap request of 199 to 2000 replicates (or none). A
request above a cap refuses as `dose_response.bounds_exceeded`; a request of 1 to 198
replicates is below the floor, which is not a bounds refusal: the point is kept and the
interval status is withheld (`estimator_inference_mismatch`,
`dose_response.bootstrap_below_floor`). Each cap is tested at its value and one above (the
covariate cap on a certificate that standardizes over 256 covariates).

**Memory.** A mandatory cap of 512 MiB (536,870,912 bytes) on the estimated workspace, lowered (never raised)
by a context hard memory limit, is checked before any fit (`transport_budget_cancel`, a
resource refusal). The estimate adds up the stored rows, every design the producer
actually allocates (membership and outcome designs, one training-design copy and normal
matrix per concurrently fitted fold, the per-fold evaluation designs) and one quadrature
chunk (at most 8 MiB of design) per concurrently integrated grid dose. Learner internals
beyond that copy (tree storage, boosting state) are not modelled. A request at every cap
at once (width 3084) is far above the cap and is refused. The calibration-internal
bootstrap multiplies the estimate by the number of concurrent replicates.

**Stored bounds.** The bounds are stored in the artifact and bound into its premises
digest. A consumer compares them with its own field by field and refuses only a looser
stored bound (a larger cap, a lower replicate floor, or a node count it does not offer),
naming it, as a limits refusal. An equal or tighter stored bound is accepted, so a v1
artifact stays readable when a later build raises a cap.

## Inference

The one interval method is the pointwise joint outer refit percentile bootstrap of the
whole cross-fitted composed estimator. It is grouped per design and has a replicate floor
of 199 and a cap of 2000. It is an interval for `psi_h` at each grid dose: smoothing bias
is excluded and there is no simultaneous band.

The estimator is `antecedent_estimate::smoothed_dose_interval_internal`. It is compiled
only under the `calibration-internal` feature, which only dev-dependencies enable, and
the calibration harness `crates/antecedent/tests/smoothed_dose_calibration.rs` measures
it. The public route is closed (`cell_not_licensed`, `dose_response.interval_withheld`)
until the two coverage records are measured at the 2.2 cut.

A replicate:

- reuses the point run's fold label for each resampled row;
- reruns the whole fit, the integration and every refusal.

Any failed replicate withholds every interval.

## Artifact

`checked_smoothed_dose_transport_v1` stores:

- the graph and the smoothed-dose query;
- the certificate;
- the rows, including doses and known densities;
- the options and the frozen bounds;
- every fold's portable outcome and membership predictor, with provider provenance;
- the fold assignment;
- the grid with its quadrature, bias and support records;
- the interval status.

It has no interval field.

### What a consumer does

A consumer never fits. In order, it:

1. refuses stored bounds looser than its own compiled bounds, a request above the stored
   bounds or above its own row and covariate limits, and a replay workspace estimate above
   its memory limit (the 512 MiB cap, lowered by its own limit or context), before any
   other work;
2. re-derives the certificate;
3. re-validates the request;
4. recomputes the folds;
5. checks the models against the learner specs;
6. re-predicts every nuisance from the stored models;
7. re-integrates the quadrature;
8. replays every recorded number bit for bit, polling its cancellation token before every
   quadrature chunk.

### What replay does not establish

Replay does not show that the stored models were fitted by the producer on the other
folds. A forger who replaces a model, recomputes everything it implies and re-seals the
digests produces an artifact that consumes; a test asserts exactly this.

The consumer also uses the producer's evaluator. Replay is therefore an integrity check,
and the evaluator's correctness rests on the verification tests above.

Some stored premises do not enter the replayed point, so a re-sealed edit of them
consumes (a test asserts each):

- the master **seed**: the fold assignment is deterministic in the source flags, and the
  seed reaches only the fits, which are not replayed;
- the **sampling design**: it only groups the bootstrap, which is not replayed;
- the **support thresholds** (`min_local_ess`, `min_distinct_doses`, `min_dose_density`,
  `min_membership_probability`): they are re-validated against the stored rows, so a
  re-sealed threshold the rows fail is refused, but one they still pass consumes.

The premises digest binds all of them against unsealed edits only. The replayed
smoothing-bias diagnostic is only as good as the fitted curve (see above).

## Not in this cell

- An estimated conditional dose density (the observational generalized-propensity path).
- Boundary kernels.
- A simultaneous band.
- Conditional (CATE) and derivative targets.
- Bandwidth selection.
- ADMG structure beyond the certified standardization contract.
- Any efficiency or rate claim.
