# Penalized propensity for AIPW (2.2 E2)

The tuning grid accepts at most 64 penalty candidates and at most 20 inner folds.

`Aipw(propensity_penalty=PropensityPenalty(...))` (Rust:
`AipwAte::with_propensity_nuisance(PropensityNuisance::ridge_logistic(..))` or
`::lasso_with(..)`) fits the binary propensity of the cross-fitted AIPW with an explicit
ridge-logistic or lasso-logistic penalty. It is a *declared nuisance choice*, distinct from
`GlmOptions.ridge_on_separation` (a rescue that estimation paths refuse to keep because a
rescued fit is not an MLE). The route publishes the cross-fitted point estimate, the score
table, and two interval constructions whose assumptions are stated below: the cross-fitted
influence-function SE of the out-of-fold scores, and (with `bootstrap > 0`) a refit bootstrap
that repeats penalty selection and every nuisance fit on each resample. A declared
GLM-to-penalized fallback runs the same route when the plain GLM propensity fit fails. The
record is `2.2E.E2.penalized_propensity_aipw` in `parity/promotion_2_2.toml`; its claim is
`calibrated` at the four measured ridge/lasso 95% analytic and refit-bootstrap coordinates (see "Calibration").

```python
from antecedent.estimators import Aipw, PropensityPenalty

# Cross-fitted influence-function SE (bootstrap=0) ...
result = ant.analyze(
    data, graph=graph, query=query,
    estimator=Aipw(bootstrap=0, propensity_penalty=PropensityPenalty(
        lambdas=[0.5, 5.0, 50.0], inner_folds=3,
    )),
)
# ... or a refit bootstrap that repeats the penalty selection in every replicate.
result = ant.analyze(
    data, graph=graph, query=query,
    estimator=Aipw(bootstrap=200, propensity_penalty=PropensityPenalty(
        kind="lasso", lambdas=[2.0, 10.0, 40.0], inner_folds=3,
    )),
)
result.estimate.penalized_support      # the covariates the lasso kept on each fold
result.estimate.penalized_bootstrap    # per-replicate penalties, replicate accounting
```

The shown grids match measured 95% coordinates. The analytic ridge record covers 300–1200
rows; the lasso refit-bootstrap record covers 600–2400 rows with at least 40 bootstrap
replicates. The default tuning grids are useful for estimation but do not match these
coverage records.

## What is fixed

- **Scope.** The untrimmed `AllObserved` mean ATE with binary treatment. Other
  functionals, other target populations and trimmed fits refit full-sample nuisances and
  are refused: `route_not_supported` (`penalized_propensity.scope`) for ridge, and
  `selection_inference_not_licensed` (`penalized_propensity.selection_closed`) for a lasso,
  because a full-sample lasso would select its support on the rows it scores.
- **Folds.** Five folds from the established seeded, arm-stratified, unit-level plan
  (`fold_seed` is the analysis seed); duplicated rows of one unit share a fold. Fold `k`'s
  rows are scored by nuisances fit on the other folds only.
- **Penalty rule (tuning on training rows only).** On fold `k`'s *training rows*, the
  penalty is the member of a fixed grid with the smallest `inner_folds`-fold (default 5)
  cross-validated log loss, using a seeded, arm-stratified, unit-level inner plan that
  depends on the seed, the fold index and the training units. Defaults: ridge `0.01, 0.1, 1,
  10, 100, 1000`; lasso `0.5, 1, 2, 5, 10, 20, 50, 100` (sum-scale). Ties go to the larger
  penalty; a grid value whose inner fit fails is not a candidate, and if none is usable the
  fold fails with the last failure recorded. The evaluation fold never informs its own
  penalty, nor, for the lasso, its own selected support, so both replay bit for bit from the
  seed and are recorded on the result (`learner_provenance`, one entry per fold) and on the
  score table's provenance (`selected_lambda=`, and for a lasso `selected_support=`, per
  fold).
- **Scale.** The objective is `-loglik + (lambda / 2) * ||beta||^2` (ridge) or
  `-loglik + lambda * ||beta||_1` (lasso) on a sum log likelihood, intercept unpenalized,
  each non-intercept column standardized by the training rows' mean and standard deviation (a
  column with no variance is zeroed). Ridge reuses `antecedent-learn`'s `RidgeLogisticLearner`;
  the lasso is `antecedent-learn`'s `LassoLogisticLearner` (proximal Newton with coordinate
  descent, accepted only when the KKT conditions of the exact objective hold). Both go through
  the cross-fit driver; there is no second learner stack. Outcome models stay arm-wise OLS,
  so a rank-deficient design is still refused by the outcome stage.
- **Preserved.** The score table (scores, out-of-fold propensities, fold ids, row identity,
  adjustment set), the overlap report and clip, and retargeting, including the retarget
  covariance (a penalized table retargets exactly like an unpenalized one). The
  configuration is part of the estimator identity (`canonical_key`, hashed with
  `canonical_bytes`) and of the score table's provenance, so a different grid, fold count or
  fallback never shares a score-reuse identity or a batch nuisance fit; penalized fits are
  never served from the shared nuisance cache. The calibration key's functional label carries
  the nuisance (`...+propensity=ridge_logistic:k5:g...`), so an interval measured under one
  configuration never binds to another.

## The cross-fitted influence-function interval

Write `psi(W; eta)` for the AIPW score with nuisances `eta = (e, mu0, mu1)`, `eta0` for the
truth, and `eta_k` for the nuisances fit without fold `F_k` (propensity penalty, and for a
lasso the support, chosen on that complement). The estimate is
`theta_hat = n^-1 sum_i psi(W_i; eta_{k(i)})`, which equals the mean of the stored scores
exactly (the score-mean identity, tested). Then

    theta_hat - theta0 = n^-1 sum_i (psi(W_i; eta0) - theta0)      (S: the influence term)
                       + n^-1 sum_k sum_{i in F_k} [psi(W_i; eta_k) - psi(W_i; eta0)]   (R)

Conditional on the rows outside `F_k`, the rows of `F_k` are independent of `eta_k`. This is
exactly what training-only tuning protects: the penalty, the selected support and the fit are
functions of the complement, so no evaluation row leaks into its own nuisance, and the
post-selection problem of an in-sample sandwich does not arise for the score average. Split
`R` into a centered empirical-process part and a conditional-mean part.

*Centered part.* Its conditional variance is `O(||psi(eta_k) - psi(eta0)||_2^2 / n)`, so it
is `o_p(n^-1/2)` when the nuisances are `L2`-consistent and the fitted propensity stays in
`[c, 1 - c]` (the clip does this).

*Conditional-mean part.* A direct calculation with `e0 = P(T = 1 | X)` gives

    E[psi(W; eta) | X] - (mu1 - mu0)
        = (e_hat - e0) * [ (mu1_hat - mu1) / e_hat + (mu0_hat - mu0) / (1 - e_hat) ],

so by Cauchy-Schwarz its size is bounded by `c^-1 ||e_hat - e0||_2 (||mu1_hat - mu1||_2 +
||mu0_hat - mu0||_2)`. The **remainder condition** is

    (RC)   ||e_k - e0||_2 * ||mu_{t,k} - mu_t||_2 = o_p(n^-1/2) for t = 0, 1.

Under (RC) `sqrt(n)(theta_hat - theta0) = n^-1/2 sum_i (psi(W_i; eta0) - theta0) + o_p(1)`:
the usual influence-function variance `Var psi(eta0)` applies and its plug-in, the sample
variance of the stored scores, is the published `se_analytic`:

    se = sqrt( sum_i (d_i - mean d)^2 / (n (n - 1)) ),   d_i = phi1_i - phi0_i

(`crossfit_influence_se`, equal to the table's contrast standard error and to this hand formula
in the tests). The joint covariance, the influence values and the retarget covariance of the
table are the same iid objects as for the unpenalized route.

### When (RC) is assumed to hold

(RC) is a rate condition on unknown nuisance errors; the library cannot check it, and the
interval is conditional on it. Sufficient regimes, stated as assumptions:

- **Ridge.** The penalty grid is fixed, so the ridge shrinkage is `O(lambda_max / n)` and the
  propensity error is that of the logistic MLE, `O_p(sqrt(p / n))`; the arm-wise OLS outcome
  error is `O_p(sqrt(p / n))` as well. The product is `O_p(p / n)`, which is `o(n^-1/2)` iff
  `p = o(n^1/2)`. So the ridge interval is the established AIPW interval extended to the
  finite-sample failures a penalty cures (separation, near-collinearity, a design where the
  plain fit does not converge) in the regime where `p` is small against `sqrt(n)`. It is
  **not** claimed in the `p` comparable to `n` regime: there the OLS error is of order
  `sqrt(p / n)` and the ridge error at least as large, the product is of order `p / n`
  (for `n = 2000`, `p = 175`: about `0.1` against `0.02`), and a prediction-optimal
  cross-validated penalty does not control bias.
- **Lasso.** Under the usual compatibility (restricted-eigenvalue) condition with `s` active
  propensity covariates the lasso's prediction error is `O_p(sqrt(s log p / n))`, and the
  literature shows a cross-validated penalty attaining it up to a logarithmic factor (an
  external premise, not re-derived here). Against the OLS outcome error `O_p(sqrt(p / n))`
  the product is `sqrt(s p log p) / n`, which is `o(n^-1/2)` iff `s p log p = o(n)`. No
  minimum-signal (beta-min) condition is needed: the cross-fit removes the dependence
  between the selected support and the scored rows, and (RC) only needs the lasso fit to
  predict well, not to recover the true support. The outcome model is not penalized, so the
  lasso's sparsity helps the propensity only.

### What is not shown

- (RC) for any particular data set; nothing in the data certifies it, and the penalty
  chosen by cross-validation is prediction-optimal, not bias-controlling.
- Validity when `p` is comparable to `n` (see above) or when the outcome OLS is
  ill-conditioned within an arm.
- Uniformity of the lasso interval over designs with unstable selection (support that
  changes across folds is recorded in `penalized_support`, not tested for).
- The fold plan is stratified on the treatment arm, so the rows of one fold are not
  exchangeable with a simple random split; the argument above treats the plan as ancillary
  given the arms (the standard practical reading).
- Any coverage statement: the calibration suite below measures coverage on designs where
  (RC) holds; those design-specific records have been measured.

## The refit bootstrap

With `bootstrap_replicates = B > 0` each replicate draws `n` rows with replacement (copies of
one row keep that row's unit identity, so a unit is never in a training set and in its own
validation fold) and re-runs the **whole pipeline** on the resample: the arm-stratified unit
fold plan, each fold's inner-CV penalty selection (and a lasso's support selection) and every
nuisance fit. Penalty selection therefore varies from replicate to replicate and its variability
enters the SE. The replicate's contrast is the mean of its own out-of-fold scores; the published
`se_bootstrap` is the sample standard deviation of the successful replicates, under the shared
failure policy (at least two successes and at most half of the attempted replicates failing,
otherwise no SE). A replicate that cannot be fit (a fold missing an arm, a failed fit, a
boundary propensity) is counted as failed, never replaced; a cancelled run is reported as a
stop (`bootstrap_cancelled`) and publishes no SE from the partial run.

What is recorded beside the scalar (`EffectEstimate::penalized`, Python
`estimate.penalized_bootstrap`): the replicates requested, successful and failed; the penalty
every successful replicate selected on each of its folds, in replicate order (so the record
replays from the seed under any thread count); and the cross-fitted influence-function SE for
comparison. `bootstrap = 0` publishes the influence-function SE only.

**Justification.** Under (RC) the estimator is asymptotically linear in the influence term,
so the nonparametric bootstrap of the sample mean of that term is consistent and the refit
resamples add the variability of the (cross-validated) nuisance fits, which is `o_p(n^-1/2)`
under (RC) and finite-sample material otherwise. This is a statement conditional on (RC), not
a proof: bootstrap validity for regularized, cross-fitted, selected nuisances is not
established here, and a resample re-centers on the penalized pseudo-truth of the resample, so
**it does not recover a penalty bias that violates (RC)**. When (RC) fails, neither interval
is supported and the bootstrap offers no repair. The two SEs are reported together so a large
disagreement is visible; it is a symptom, not a test.

## Lasso

`PropensityPenalty(kind="lasso")` is a real route, cross-fitted: for each fold the penalty and
the support (covariates with a nonzero coefficient) are chosen on the training rows by the
same seeded inner CV as for ridge, the support is recorded per fold (names `V<id>` after the
adjustment variable ids, and the count) in `penalized_support`, in the table's
`selected_support=` provenance and in the estimate. The interval is the cross-fitted
influence-function SE or the refit bootstrap above under (RC). A lasso outside the
cross-fitted untrimmed `AllObserved` mean ATE is refused with
`selection_inference_not_licensed`. Where the plain logistic has no maximizer (complete
separation), the L1-penalized objective still has a finite minimizer, so the lasso fits.

## GLM-to-penalized fallback

`nuisance_fallback="ridge_logistic"` / `"lasso"` (default tuning) or
`nuisance_fallback=PropensityPenalty(...)` (its kind and tuning) declares the destination of a
failed GLM propensity fit. If every GLM fold fits, the result is the GLM result bit for bit
and nothing is recorded. If any fold's GLM fit fails, the partial table is discarded and the
**whole** cross-fitted route is rebuilt with the destination (no fold mixes the two
nuisances), with the same estimand (`AllObserved` mean ATE), the same folds and the
destination's claim and intervals. The result records both identities:
`estimate.penalized_fallback` (Rust `PenalizedReport::fallback`) carries the failed fit
(`stage="propensity_fit"`, the fold, the class `separated` / `non_converged` /
`boundary_saturated` / `rank_deficient` / `fit_failed`, and the failure's message) and the
destination's canonical key; the score table's provenance names the destination, the failure
and the declared configuration (`;nuisance_fallback=glm_failed:<class>@fold<k>;declared=...`);
the estimator-spec identity (`treatment_config`, `canonical_key`) includes the declared
destination and tuning, so a different fallback is a different estimator. With
`bootstrap > 0` after a fallback, the replicates run the destination (the claim is the
destination's); after a GLM success they run the GLM with no fallback, so a failing replicate
is counted rather than replaced. A fallback is refused beside a declared penalty
(`invalid_argument`, `penalized_propensity.fallback_with_penalty`): it replaces a failed GLM
fit, and a penalized primary is not one. On the full-sample routes (trimmed, ATT/ATC,
predicate) there is no fallback; a failed fit there is recorded in the refusal
(`nuisance_fallback_not_licensed`).

`nuisance_fallback="ml"` (a flexible learner) stays closed: its destination has no
cross-fitted support or uncertainty license, so a failed GLM fit is refused with
`nuisance_fallback_not_licensed` and the failed fit recorded, and nothing is silently
substituted. The fallback is never chosen after seeing a favorable result: it is declared in
advance and triggered only by a fit failure.

## Refusals

| code | detail | when |
| --- | --- | --- |
| `invalid_argument` | `penalized_propensity.invalid_penalty` | empty or oversized grid, non-positive or non-finite penalty, inner folds outside 2 to 20 |
| `invalid_argument` | `penalized_propensity.fallback_with_penalty` | a fallback declared beside a penalized primary |
| `route_not_supported` | `penalized_propensity.scope` | a ridge penalty with a non-mean functional, a population other than `AllObserved`, or trimming |
| `selection_inference_not_licensed` | `penalized_propensity.selection_closed` | a lasso outside the cross-fitted untrimmed `AllObserved` mean ATE |
| `nuisance_fallback_not_licensed` | `penalized_propensity.fallback_closed` | a GLM fit fails under an ML fallback, or under any fallback on a full-sample route |
| `cancelled_no_claim` | `penalized_propensity.cancelled` | penalty selection was cancelled; no estimate is reported |

Selection observes cancellation once per penalty of every outer fold, so a cancel is seen
within one penalty's inner cross-fit; a cancelled call returns no estimate or score table. A
cancelled bootstrap reports the stop and publishes no SE.

## Calibration

The four 95% ridge/lasso analytic and refit-bootstrap coverage records are measured and
licensed at their exact calibration coordinates. The tests in
`crates/antecedent-estimate/src/calibration_coverage.rs` are registered in
`scripts/gate_calibration.sh`; `crates/antecedent/tests/calibration_binding.rs`
checks that each record's nuisance label and interval method match the public facade.

| Propensity grid (3 inner folds) | Analytic interval rows | Refit bootstrap rows |
| --- | ---: | ---: |
| Ridge: 0.5, 5, 50 | 300–1200 | 600–2400; at least 40 replicates |
| Lasso: 2, 10, 40 | 150–600 | 600–2400; at least 40 replicates |

A different penalty grid, sample-size range, level or bootstrap count is reported as
`scope_not_assessed`, not covered by these records. The fallback
route has no coverage test of its own: its destination's interval is the ridge or lasso
interval, but its calibration key (`propensity=glm_fallback_...`) matches no record, so a
fallback interval stays unassessed until one is measured. An in-repo, fixed-seed repeated
sampling test (`the_influence_interval_covers_the_known_effect_in_repeated_sampling`) is a
guard, not a record.

## What these records do not measure

No coverage record licenses the fallback interval. The estimate is a point under the identification assumptions of the
AIPW average effect plus consistency of the penalized and OLS nuisances; none of that, and not
(RC), is checked by the library.
