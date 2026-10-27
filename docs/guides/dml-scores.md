# Reuse cross-fitted DML scores to retarget with joint covariance

`DmlAte` (the AIPW score), `DrLearner` and `CausalForest` keep a row-identified, per-arm
cross-fitted AIPW **score table** of the average effect they report, so a prepared retarget
and a batch retarget (see [the batch guide](batch-retarget.md)) reweight those scores and
report the joint plug-in score covariance of the retargeted claims. The claim is
`point_only`: values and a plug-in score covariance (a standard error is not an interval).

```rust
let prepared = BatchStudy::new(data, graph)
    .estimator_spec(DmlAte::new())
    .prepare(&queries, &ctx)?;
let (results, scores) = prepared.estimate_scored(&data, &ctx)?;
let report = prepared.retarget(&scores, &request, &ctx)?;
```

## What the table is

With out-of-fold `mu_a(x)` and `e(x)` from `antecedent-learn` (the same cached bundle the
fit uses, the propensity clipped at the overlap clip) the table holds, for every row `i`,

    phi1_i = mu1_i + T_i (Y_i - mu1_i) / e_i,    phi0_i = mu0_i + (1 - T_i)(Y_i - mu0_i) / (1 - e_i)

in exactly the formulas of the licensed AIPW table, so `phi1_i - phi0_i` is the DML AIPW score
`mu1 - mu0 + T (Y - mu1)/e - (1 - T)(Y - mu0)/(1 - e)` row by row, the mean of the contrast
is the fitted ATE, and uniform-weight retarget returns the route's own point and iid standard
error. The table carries the original-row index of each complete-case row, the fold plan,
the raw out-of-fold propensities, and a provenance tag
`dml.crossfit.v1;route=<dml|dr_learner>;outcome=<spec>;treatment=<spec>`. It is built once by
`dml::aipw_score_table`; there is no second score implementation.

`DrLearner` and `CausalForest` report the marginal AIPW score of the same nuisances (the
DR-Learner's ATE is the mean of its pseudo-outcome; the forest's marginal ATE is the DML
AIPW score), so their tables are the marginal AIPW scores. **No score, table or interval is
derived from a CATE prediction**: the final-stage learner and the forest's CATE contribute no
scores, and the retargeted claim is a weighted marginal AIPW mean, not a CATE functional.

## Conditions under which a covariance is claimed

The covariance is the plug-in Gram matrix of the weighted influence values of these
scores. It is meaningful only when all of these hold; the library checks the first two.

1. The fit is the **untrimmed `AllObserved` AIPW score**. The partially linear (Robinson)
   score has no per-arm score, so a partially linear fit keeps no table; trimming redefines
   the estimand to the retained rows, so a trimmed fit keeps no table either. Either refuses
   a retarget with `score_table_unavailable`
   (`batch_retarget.scores_unavailable` / `batch_retarget.scores_unavailable_after_estimate`)
   instead of reweighting scores of another construction.
2. The scores are finite (a propensity of exactly 0 or 1 without a clip refuses).
3. **Cross-fit assumptions** (unchecked): iid rows; each row is scored by nuisances fit on
   other folds only; positivity (the overlap clip and the weighted overlap gate of a
   retarget); the product of the outcome and propensity nuisance errors is `o_p(n^-1/2)` on
   the training folds; the retarget weights are fixed functions of certified covariates.
   Flexible learners are not shown to satisfy the rate, which is why the claim stays
   `point_only` and no interval is published.
4. A cluster or panel dependence structure is not handled: use the cluster-DML route for
   whole-cluster folds.

The table has no wire change: it is the existing `ScoreTable` artifact with another
provenance tag, so its wire round trip is the AIPW one.

## Inspect the treatment model on its own

When the question is about the treatment model rather than the effect (how spread out are the propensity scores, and how likely is a row to fall outside a positivity band), ask a prepared average-effect study for the posterior of its treatment model. `treatment_score_posterior` fits the Laplace posterior of the logistic treatment model on exactly the study's certified adjustment set, under a stated `N(0, prior_sd**2)` prior on every coefficient, the intercept included. It lives with `antecedent.nuisance_scores`.

The fixture is two cells: 100 rows at `z = 0` with 30 treated and 100 rows at `z = 1` with 60 treated.

```python
import antecedent
import numpy as np
from antecedent import nuisance_scores
from antecedent.execution import Execution
from antecedent.outcomes import require_success

z = np.r_[np.zeros(100), np.ones(100)]
t = np.r_[np.ones(30), np.zeros(70), np.ones(60), np.zeros(40)]
y = 1.0 + 2.0 * t + 0.1 * (np.arange(200) % 7)
study = antecedent.prepare(
    {"z": z, "t": t, "y": y},
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=antecedent.AverageEffect("t", "y"),
    inference=antecedent.Frequentist(bootstrap=0),
    execution=Execution(seed=0),
    refute=False,
)
product = require_success(
    study.treatment_score_posterior(
        prior_sd=2.0, draws=4000, execution=Execution(seed=13), clip=0.35, summary_rows=[0, 150]
    )
)
product["row0.at_mode"], product["row0.mean"], product["row0.sd"], product["row0.outside_band"]
```

For each summarized row the product reports the score at the posterior mode, its posterior mean and spread, and the probability that the score falls outside the band set by `clip`. It is a posterior of a nuisance, never of an effect: `product.estimator_id` is `bayesian.propensity.coefficient_posterior`, `product.licensing` is `unlicensed` and `product.calibration` is `unmeasured`. A posterior of the weighting effect cannot be built from these draws. Asking for a Bayesian `PropensityWeighting` estimator is refused with `route_not_supported`, and the message names the posteriors that do answer that effect: the Bayesian-bootstrap posterior of the cross-fitted ridge-orthogonal score, or the Bayesian g-computation posterior of an outcome model.

`product.export()` embeds the design the product was fitted on; `nuisance_scores.load_artifact(bytes(product.export()))` refits from it and refuses any edited, truncated or resealed artifact with `CausalSerializationError`. `temporal_treatment_score` is the temporal counterpart for a `TemporalDag` study.

## What stays closed

- A partially linear DML fit, a trimmed DML fit, a DR-Learner or CausalForest fit under a
  non-`AllObserved` target or trimming: no table, `score_table_unavailable`.
- Quantile and exceedance functionals read the AIPW table only; the learner routes keep no
  functional grid, so those refuse as before.
- DR-Learner CATE profiles and CausalForest CATE intervals are separate cells and are not
  extended by this one.

## Tests

`crates/antecedent-estimate/src/dml.rs` (inline): the score-mean identity and the iid
covariance against raw sums, a hand calculation of the table formulas on a fixed nuisance
bundle (`mu0 = 1`, `mu1 = 3`, `e = 0.5`), seeded replay, partially linear and trimmed fits
keeping no table, and the DR-Learner and CausalForest tables.
`crates/antecedent/tests/dml_scores_retarget.rs`: retarget after estimate with the
independent sums for points and the cross-plan covariance, the prepare-time table equal to
the estimate's, the typed refusals, and DR-Learner / CausalForest.
