# Estimate with a penalized propensity (ridge or lasso AIPW)

Use a penalized propensity when the plain logistic propensity of a cross-fitted AIPW
average effect fails or is unstable. That covers separation, near-collinear adjustment
covariates and many covariates. Declare the penalty before you estimate. Do not apply it as
a rescue after a failure. Call `Aipw(propensity_penalty=PropensityPenalty(...))`. In Rust,
call `AipwAte::with_propensity_nuisance` with `PropensityNuisance::ridge_logistic` or
`PropensityNuisance::lasso_with`. The call fits the binary propensity with an explicit
ridge-logistic or lasso-logistic penalty. It is a *declared nuisance choice*.
`GlmOptions.ridge_on_separation` is a rescue. Estimation refuses to keep a rescued fit,
because a rescued fit is not an MLE.

The estimand is unchanged: the average treatment effect `E[Y(1) - Y(0)]` of a binary
treatment over all observed rows, under the identification assumptions of the AIPW average
effect for your graph. You get the cross-fitted point estimate and the score table. You also
get two interval constructions whose assumptions are stated below. One is the cross-fitted
influence-function SE of the out-of-fold scores. With `bootstrap > 0`, the other is a refit
bootstrap that repeats penalty selection and every nuisance fit on each resample. A declared
GLM-to-penalized fallback runs the same call when the plain GLM propensity fit fails. Each
interval reports its calibration status. It is `calibrated` only at a coordinate a ridge or
lasso coverage record measures (see [Calibration](#calibration)).

The snippet assumes `data`, `graph` and an `ant.AverageEffect` `query` as in
[your first analysis](../start/first-analysis.md).

```python
import antecedent as ant
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
result.value.estimate.penalized_support      # the covariates the lasso kept on each fold
result.value.estimate.penalized_bootstrap    # per-replicate penalties, replicate accounting
```

The shown grids are the configurations the 95% calibration design measures (see
[Calibration](#calibration)). The default tuning grids are useful for estimation. They are a
different configuration. Their intervals report `outside_measured_scope`.

The constructor limits bound what is admitted. They do not imply measured coverage for every admitted configuration. The constructor requires a nonempty grid of finite positive penalties and at least {{bound:estimate.penalized_propensity_aipw.runtime_min_inner_folds}} folds. It accepts at most 65,536 input penalties per declaration, counting duplicates before deduplication. This is a conservative 256 MiB configuration policy (4,096 bytes per entry). It is not a shared reservation or an execution-context admission. It does not charge downstream copies or concurrent declarations. The fold plan separately limits folds by the training-row count and its {{bound:estimate.penalized_propensity_aipw.runtime_fold_index_capacity}}-value index representation. Calibrated intervals remain qualified by the exact recorded configuration.

## What is fixed

- **Scope.** The untrimmed `AllObserved` mean ATE with a binary treatment (two declared levels). Other functionals, other target populations and trimmed fits refit full-sample nuisances and are refused. Ridge refuses with `route_not_supported` (`penalized_propensity.scope`). A lasso refuses with `selection_inference_not_supported` (`penalized_propensity.selection_not_supported`). A full-sample lasso would select its support on the rows it scores.
- **Folds.** Five folds from the established seeded, arm-stratified, unit-level plan. `fold_seed` is the analysis seed. Duplicated rows of one unit share a fold. Fold `k`'s rows are scored by nuisances fit on the other folds only.
- **Penalty rule (tuning on training rows only).** On fold `k`'s *training rows*, the penalty is the member of a fixed grid with the smallest `inner_folds`-fold (default 5) cross-validated log loss. The inner plan is seeded, arm-stratified and unit-level. It depends on the seed, the fold index and the training units. Ridge defaults are `0.01, 0.1, 1, 10, 100, 1000`. Lasso defaults are `0.5, 1, 2, 5, 10, 20, 50, 100` (sum-scale). Ties go to the larger penalty. A grid value whose inner fit fails is not a candidate. If none is usable the fold fails with the last failure recorded. The evaluation fold never informs its own penalty. For the lasso, it never informs its own selected support. Both replay bit for bit from the seed. They are recorded on the result (`learner_provenance`, one entry per fold) and on the score table's provenance (`selected_lambda=` per fold, and for a lasso `selected_support=` per fold).
- **Scale.** The ridge objective is `-loglik + (lambda / 2) * ||beta||^2`. The lasso objective is `-loglik + lambda * ||beta||_1`. Both use a sum log likelihood. The intercept is unpenalized. Each non-intercept column is standardized by the training rows' mean and standard deviation. A column with no variance is zeroed. Ridge uses `antecedent-learn`'s `RidgeLogisticLearner`. The lasso is `antecedent-learn`'s `LassoLogisticLearner`. That fit is proximal Newton with coordinate descent. It is accepted only when the KKT conditions of the exact objective hold. Both use the same cross-fit. There is no second learner stack. Outcome models stay arm-wise OLS. A rank-deficient design is still refused by the outcome stage.
- **Preserved.** You still get the score table: scores, out-of-fold propensities, fold ids, row identity and the adjustment set. You still get the overlap report and clip, and retargeting, including the retarget covariance. A penalized table retargets exactly like an unpenalized one. The configuration is part of the estimator identity (`canonical_key`, digested by `identity_digest`) and of the score table's provenance. A different grid, fold count or fallback never shares a score-reuse identity or a batch nuisance fit. Penalized fits are never served from the shared nuisance cache. The calibration key's functional label carries the nuisance (`...+propensity=ridge_logistic:k5:g...`). An interval measured under one configuration never binds to another.

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
- Any coverage statement beyond the calibration design: the calibration suite below measures
  coverage on designs where (RC) holds, and a record speaks only for its exact configuration.

## The refit bootstrap

With `bootstrap_replicates = B > 0` each replicate draws `n` rows with replacement. Copies of
one row keep that row's unit identity. A unit is never in a training set and in its own
validation fold. Each replicate re-runs the whole fit on the resample: the arm-stratified unit
fold plan, each fold's inner-CV penalty selection, a lasso's support selection, and every
nuisance fit. Penalty selection therefore varies from replicate to replicate. Its variability
enters the SE. The replicate's contrast is the mean of its own out-of-fold scores. The published
`se_bootstrap` is the sample standard deviation of the successful replicates. The shared
failure policy requires at least two successes and at most half of the attempted replicates
failing. Otherwise no SE is published. A replicate that cannot be fit is counted as failed.
That includes a fold missing an arm, a failed fit or a boundary propensity. It is never
replaced. A cancelled run is reported as a stop (`bootstrap_cancelled`). It publishes no SE
from the partial run.

Read the record beside the scalar. In Rust, that is `EffectEstimate.diagnostics.penalized`.
In Python, that is `estimate.penalized_bootstrap`. It holds the replicates requested,
successful and failed. It holds the penalty every successful replicate selected on each of
its folds, in replicate order. The record replays from the seed under any thread count. It
also holds the cross-fitted influence-function SE for comparison. `bootstrap = 0` publishes
the influence-function SE only.

**Justification.** Under (RC) the estimator is asymptotically linear in the influence term.
The nonparametric bootstrap of the sample mean of that term is consistent. The refit
resamples add the variability of the cross-validated nuisance fits. That variability is
`o_p(n^-1/2)` under (RC) and finite-sample material otherwise. This is a statement conditional
on (RC). It is not a proof. Bootstrap validity for regularized, cross-fitted, selected
nuisances is not established here. A resample re-centers on the penalized pseudo-truth of the
resample. **It does not recover a penalty bias that violates (RC)**. When (RC) fails, neither
interval is supported. The bootstrap offers no repair. The two SEs are reported together. A
large disagreement is visible. It is a symptom. It is not a test.

## Lasso

Call `PropensityPenalty(kind="lasso")`. The fit is cross-fitted. For each fold the penalty and
the support are chosen on the training rows by the same seeded inner CV as for ridge. The
support is the covariates with a nonzero coefficient. It is recorded per fold, as names
`V<id>` after the adjustment variable ids, and the count. Read it in `penalized_support`, in
the table's `selected_support=` provenance and in the estimate. The interval is the
cross-fitted influence-function SE or the refit bootstrap above under (RC). A lasso outside
the cross-fitted untrimmed `AllObserved` mean ATE is refused with
`selection_inference_not_supported`. Where the plain logistic has no maximizer (complete
separation), the L1-penalized objective still has a finite minimizer. The lasso fits.

## GLM-to-penalized fallback

Pass `nuisance_fallback="ridge_logistic"` or `"lasso"` for default tuning. Or pass
`nuisance_fallback=PropensityPenalty(...)` with its kind and tuning. That declares the
destination of a failed GLM propensity fit. If every GLM fold fits, the result is the GLM
result bit for bit. Nothing is recorded. If any fold's GLM fit fails, the partial table is
discarded. The whole cross-fit is rebuilt with the destination. No fold mixes the two
nuisances. The estimand stays the `AllObserved` mean ATE. The folds stay the same. The claim
and intervals are the destination's. The result records both identities. Read
`estimate.penalized_fallback`. In Rust, read `PenalizedReport::fallback`. It carries the
failed fit: `stage="propensity_fit"`, the fold, the class (`separated`, `non_converged`,
`boundary_saturated`, `rank_deficient` or `fit_failed`) and the failure's message. It also
carries the destination's canonical key. The score table's provenance names the destination,
the failure and the declared configuration
(`;nuisance_fallback=glm_failed:<class>@fold<k>;declared=...`). The estimator-spec identity
(`treatment_config`, `canonical_key`) includes the declared destination and tuning. A
different fallback is a different estimator. With `bootstrap > 0` after a fallback, the
replicates run the destination. The claim is the destination's. After a GLM success they run
the GLM with no fallback. A failing replicate is counted. It is not replaced. A fallback is
refused beside a declared penalty (`invalid_argument`,
`penalized_propensity.fallback_with_penalty`). It replaces a failed GLM fit. A penalized
primary is not one. On the full-sample analyses (trimmed, ATT/ATC, predicate) there is no
fallback. A failed fit there is recorded in the refusal (`nuisance_fallback_not_supported`).

`nuisance_fallback="ml"` (a flexible learner) is refused. Its destination has no cross-fitted
or uncertainty evidence. A failed GLM fit is refused with `nuisance_fallback_not_supported`
and the failed fit recorded. Nothing is silently substituted. The fallback is never chosen
after seeing a favorable result. It is declared in advance. It is triggered only by a fit
failure.

## Refusals

| code | detail | when |
| --- | --- | --- |
| `invalid_argument` | `penalized_propensity.invalid_penalty` | empty grid, non-positive or non-finite penalty, or fewer than two inner folds; the fold plan separately enforces its row-count and index-representation limits |
| `invalid_argument` | `penalized_propensity.fallback_with_penalty` | a fallback declared beside a penalized primary |
| `route_not_supported` | `penalized_propensity.scope` | a ridge penalty with a non-mean functional, a population other than `AllObserved`, or trimming |
| `selection_inference_not_supported` | `penalized_propensity.selection_not_supported` | a lasso outside the cross-fitted untrimmed `AllObserved` mean ATE |
| `nuisance_fallback_not_supported` | `penalized_propensity.fallback_not_supported` | a GLM fit fails under an ML fallback, or under any fallback on a full-sample route |
| `cancelled_no_claim` | `penalized_propensity.cancelled` | penalty selection was cancelled; no estimate is reported |
| `execution_resource_limit` | `penalized_propensity.grid_allocation` | the declared penalty grid exceeds the configuration policy or cannot be allocated |

Each refusal carries the fields described in
[refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies);
`selection_inference_not_supported`, `nuisance_fallback_not_supported` and
`execution_resource_limit` are listed under
[other registered refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes).

Selection observes cancellation once per penalty of every outer fold. A cancel is seen
within one penalty's inner cross-fit. A cancelled call returns no estimate or score table. A
cancelled bootstrap reports the stop and publishes no SE.

## Calibration

The calibration design measures the 95% ridge and lasso analytic and refit-bootstrap
intervals at the coordinates below. An interval reports `calibrated` only when a current
coverage record covers its exact coordinate. Read `result.inspect().calibration`. The
coverage tests in
`crates/antecedent-estimate/src/calibration_coverage.rs` and the binding check in
`crates/antecedent/tests/calibration_binding.rs` confirm that each record's nuisance label
and interval method match the call you make.

| Propensity grid (3 inner folds) | Analytic interval rows | Refit bootstrap rows |
| --- | ---: | ---: |
| Ridge: 0.5, 5, 50 | 300–1200 | 600–2400; at least 40 replicates |
| Lasso: 2, 10, 40 | 150–600 | 600–2400; at least 40 replicates |

A different penalty grid, sample-size range, level or bootstrap count is reported as
`outside_measured_scope`. It is not covered by these records. The fallback has no coverage
test of its own. Its destination's interval is the ridge or lasso interval. Its calibration
key (`propensity=glm_fallback_...`) matches no record. A fallback interval reports
`not_measured`. An in-repo, fixed-seed repeated sampling test
(`the_influence_interval_covers_the_known_effect_in_repeated_sampling`) is a guard. It is
not a record.

## What these records do not measure

No coverage record supports the fallback interval. The estimate is a point under the identification assumptions of the
AIPW average effect plus consistency of the penalized and OLS nuisances. None of that is checked by the library. (RC) is not checked either.

Implementation limits are listed in [supported input bounds](../supported-bounds.md).
