# Penalized propensity for AIPW (2.2 E2)

`Aipw(propensity_penalty=PropensityPenalty(...))` (Rust:
`AipwAte::with_propensity_nuisance(PropensityNuisance::ridge_logistic(..))`) fits the
binary propensity of the cross-fitted AIPW with an explicit ridge-logistic penalty. It is
a *declared nuisance choice*, distinct from `GlmOptions.ridge_on_separation` (a rescue
that estimation paths refuse to keep because a rescued fit is not an MLE). The claim is
`point_only`: the route publishes the cross-fitted point estimate and the score table and
**no interval**. The record is `2.2E.E2.penalized_propensity_aipw` in
`parity/promotion_2_2.toml`.

```python
from antecedent.estimators import Aipw, PropensityPenalty

# The point estimate and the score table are reported; no interval is.
result = ant.analyze(
    data, graph=graph, query=query,
    estimator=Aipw(bootstrap=0, propensity_penalty=PropensityPenalty()),
)
```

## What is fixed

- **Scope.** The untrimmed `AllObserved` mean ATE with binary treatment. Other
  functionals, other target populations and trimmed fits refit full-sample nuisances and
  are refused (`route_not_supported`, `penalized_propensity.scope`).
- **Folds.** Five folds from the established seeded, arm-stratified, unit-level plan
  (`fold_seed` is the analysis seed); duplicated rows of one unit share a fold. Fold `k`'s
  rows are scored by nuisances fit on the other folds only.
- **Penalty rule (tuning on training rows only).** On fold `k`'s *training rows*, the
  penalty is the member of a fixed grid (default `0.01, 0.1, 1, 10, 100, 1000`, sum-scale)
  with the smallest `inner_folds`-fold (default 5) cross-validated log loss, using a
  seeded, arm-stratified, unit-level inner plan that depends on the seed, the fold index
  and the training units. Ties go to the larger penalty; a grid value whose inner fit
  fails is not a candidate, and if none is usable the fold fails with the last failure
  recorded. The evaluation fold never informs its own penalty, so the chosen penalties
  replay bit for bit from the seed and are recorded on the result
  (`learner_provenance`, one entry per fold) and on the score table's provenance
  (`selected_lambda=`, per fold).
- **Scale.** The objective is `-loglik + (lambda / 2) * ||beta||^2` on a sum log
  likelihood, intercept unpenalized, each non-intercept column standardized by the
  training rows' mean and standard deviation (a column with no variance is zeroed). It
  reuses `antecedent-learn`'s `RidgeLogisticLearner` and its cross-fit driver; there is no
  second learner stack. Outcome models stay arm-wise OLS, so a rank-deficient design is
  still refused by the outcome stage.
- **Preserved.** The score table (scores, out-of-fold propensities, fold ids, row
  identity, adjustment set), the overlap report and clip, and retargeting. The
  configuration is part of the estimator identity (`canonical_key`, hashed with
  `canonical_bytes`) and of the score table's provenance, so a different grid, fold count
  or fallback never shares a score-reuse identity or a batch nuisance fit; penalized fits
  are never served from the shared nuisance cache.

## Why no interval (the variance derivation)

Write `psi(W; eta)` for the AIPW score with nuisances `eta = (e, mu0, mu1)`, `eta0` for the
truth, and `eta_k` for the nuisances fit without fold `F_k` (propensity penalty chosen on
that complement). The estimate is `theta_hat = n^-1 sum_i psi(W_i; eta_{k(i)})`, which
equals the mean of the stored scores exactly (the score-mean identity, tested). Then

    theta_hat - theta0 = n^-1 sum_i (psi(W_i; eta0) - theta0)      (S: the influence term)
                       + n^-1 sum_k sum_{i in F_k} [psi(W_i; eta_k) - psi(W_i; eta0)]   (R)

Conditional on the rows outside `F_k`, the rows of `F_k` are independent of `eta_k`. This
is exactly what training-only tuning protects: the penalty and the fit are functions of the
complement, so no evaluation row leaks into its own nuisance. Split `R` into a centered
empirical-process part and a conditional-mean part.

*Centered part.* Its conditional variance is `O(||psi(eta_k) - psi(eta0)||_2^2 / n)`, so it
is `o_p(n^-1/2)` when the nuisances are `L2`-consistent and the fitted propensity stays in
`[c, 1 - c]` (the clip does this).

*Conditional-mean part.* A direct calculation with `e0 = P(T = 1 | X)` gives

    E[psi(W; eta) | X] - (mu1 - mu0)
        = (e_hat - e0) * [ (mu1_hat - mu1) / e_hat + (mu0_hat - mu0) / (1 - e_hat) ],

so by Cauchy-Schwarz its size is bounded by `c^-1 ||e_hat - e0||_2 (||mu1_hat - mu1||_2 +
||mu0_hat - mu0||_2)`. If

    (RC)   ||e_k - e0||_2 * ||mu_{t,k} - mu_t||_2 = o_p(n^-1/2) for t = 0, 1

then `sqrt(n)(theta_hat - theta0) = n^-1/2 sum_i (psi(W_i; eta0) - theta0) + o_p(1)`, the
usual influence-function variance `Var psi(eta0)` applies, and its plug-in is the
variance of the stored scores.

**The remainder condition (RC) is not justified for this route, so the interval is
withheld.** (RC) is a rate condition on unknown nuisance errors. It holds in the
low-dimensional regime (a fixed number of covariates, a correct logistic model, a penalty
that stays bounded), where the unpenalized route already applies. The regime that motivates
a penalty is the opposite one: `p` comparable to `n`. There the arm-wise OLS outcome error
is of order `sqrt(p / n)` and the ridge propensity error is at least as large (a penalty
trades variance for a bias that does not vanish at the `n^-1/2` rate), so the product is of
order `p / n`, far above `n^-1/2` (for `n = 2000`, `p = 175`: about `0.1` against `0.02`).
Nothing in the data certifies (RC) either, and a penalty chosen by cross-validation is a
prediction-optimal choice, not a bias-controlling one. The plug-in variance of the scores
estimates `Var psi(eta_k)`, not the sampling variance of `theta_hat` when `R` is not
negligible, so an interval built from it has no coverage statement. Accordingly:

- an analytic interval, a score covariance, a simultaneous band and the influence values
  are **not published** for a penalized table: `se_analytic` is `NaN`, `joint_covariance`,
  `score_inference` and `influence` are absent, and a retarget of the table reports the
  point only (`estimate.aipw.penalized_interval_withheld`);
- requesting one (a bootstrap count above zero, or a non-default `se_kind`) is refused
  with `penalized_interval_not_licensed` (`penalized_propensity.interval_withheld`)
  rather than ignored.

The fold plan is stratified on the treatment arm, so the rows of one fold are not exchangeable
with a simple random split; the argument above treats the plan as ancillary given the arms
(the standard practical reading), a further assumption an interval would owe.

## Bootstrap

A row bootstrap would have to repeat penalty selection and nuisance fitting in every
replicate. The score-table builder does, by construction (selection is inside the per-fold
fit), but the bootstrap cannot supply what (RC) lacks: it re-centers on the penalized
pseudo-truth of each resample, so it does not recover the penalty bias, and bootstrap
validity for regularized high-dimensional nuisances is not established here. It is closed
with the interval route until a coverage record at this construction is measured; no
calibration is part of this change.

## Closed options

- **Lasso** (`PropensityPenalty(kind="lasso")`) is declarable and refused at execution with
  `selection_inference_not_licensed` (`penalized_propensity.selection_closed`). Selecting
  covariates from the data changes the inference contract, and an ordinary
  influence-function interval is not valid after selection.
- **GLM-to-ML fallback** (`nuisance_fallback="ml"`) is declarable and closed. If the GLM
  nuisance fit succeeds the result is the GLM result and nothing changes. If it fails, the
  fit is refused with `nuisance_fallback_not_licensed`
  (`penalized_propensity.fallback_closed`) and the message records the failed fit: the
  destination route has no support or uncertainty license, so no estimator is silently
  switched.
- An invalid grid or fold count refuses with `invalid_argument`
  (`penalized_propensity.invalid_penalty`); a cancelled selection is a
  `cancelled_no_claim` stop, never a verdict.

## What was not measured

No coverage record exists or was run for any penalized construction. The estimate is a
point under the identification assumptions of the AIPW average effect plus consistency of
the penalized and OLS nuisances; none of that is checked by the library.
