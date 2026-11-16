# Reuse cross-fitted DML scores to retarget with joint covariance

Use this page when you have fitted an average effect with `DmlAte` (the AIPW score),
`DrLearner` or `CausalForest` and want the effect in other declared populations without
refitting. Each of these estimators keeps a row-identified, per-arm cross-fitted AIPW
**score table** of the average effect it reports, so a prepared retarget and a batch
retarget (see [the batch guide](batch-retarget.md)) reweight those scores and report the
joint plug-in score covariance of the retargeted claims. The claim is `point_only`: values
and a plug-in score covariance (a standard error is not an interval).

```rust
let prepared = BatchStudy::new(data.clone(), graph)
    .estimator_spec(DmlAte::new())
    .prepare(&queries, &ctx)?;
let (results, scores) = prepared.estimate_scored(&data, &ctx)?;
let report = prepared.retarget(&scores, &request, &ctx)?;
```

`request` is a `BatchRetargetRequest`: the named claims, each with row-aligned target
weights, and the named contrasts. `report` carries each retargeted point, the joint score
covariance and the members that failed. In Python, retargeting belongs to the completed
batch (`batch.retarget(...)`); the [batch guide](batch-retarget.md) shows the call and how
to read the report.

## Estimand

With row weights `w_i` that are fixed functions of certified adjustment covariates, a
retargeted claim is the weighted mean of the per-row score contrast,
`theta_w = sum_i w_i (phi1_i - phi0_i) / sum_i w_i`, which estimates the average effect
`E_w[Y(1) - Y(0)]` in the population the weights describe. Uniform weights return the fitted
average effect. `phi1_i` and `phi0_i` are defined next.

## What the table is

With out-of-fold `mu_a(x)` and `e(x)` from `antecedent-learn` (the same cached bundle the
fit uses, the propensity clipped at the overlap clip) the table holds, for every row `i`,

    phi1_i = mu1_i + T_i (Y_i - mu1_i) / e_i,    phi0_i = mu0_i + (1 - T_i)(Y_i - mu0_i) / (1 - e_i)

in exactly the formulas of the supported AIPW table, so `phi1_i - phi0_i` is the DML AIPW score
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

The table is the `ScoreTable` artifact with its own provenance tag, so it serializes and
round-trips exactly as the AIPW table does.

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

For each summarized row the product reports the score at the posterior mode, its posterior mean and spread, and the probability that the score falls outside the band set by `clip`. It is a posterior of a nuisance, never of an effect: `product.estimator_id` is `bayesian.propensity.coefficient_posterior`, `product.support` is `unsupported` and `product.calibration` is `not_measured`. A posterior of the weighting effect cannot be built from these draws. Asking for a Bayesian `PropensityWeighting` estimator is refused with `route_not_supported`, and the message names the posteriors that do answer that effect: the Bayesian-bootstrap posterior of the cross-fitted ridge-orthogonal score, or the Bayesian g-computation posterior of an outcome model.

`product.export()` embeds the design the product was fitted on; `nuisance_scores.load_artifact(bytes(product.export()))` refits from it and refuses any edited, truncated or resealed artifact with `CausalSerializationError`. `temporal_treatment_score` is the temporal counterpart for a `TemporalDag` study.

## A prior on the outcome nuisance

`DML(prior_from=result.posterior.artifact)` and `Aipw(bootstrap=0, prior_from=...)` regularize the per-arm outcome regressions of
the AIPW score with the coefficient summaries of a fitted posterior. In every cross-fit fold
each arm's outcome model is the posterior mode of the linear regression under the Gaussian
coefficient prior,
`(X'X + V0^-1)^-1 (X'y + V0^-1 m)`, fitted on that fold's training rows of the arm, instead
of the `outcome` learner. A dense source (draws retained) carries its coefficient correlation
into `V0`; a source that records no residual variance is converted to the conjugate scale with
the training rows' own residual variance. The doubly robust correction, the cross-fitting, the
propensity nuisance and the influence-function interval are those of the unregularized score,
so the prior shapes the efficiency of the estimate and never its identification: a correct
propensity keeps the estimate consistent under any prior, however wrong. The prior (source
digest, mapping and every hydration restriction) is recorded in the result's assumptions as
`aipw_outcome_nuisance_prior`, and the score table's provenance names it.

The per-arm outcome regression has the columns `intercept` and `coef_<covariate>` per
adjustment variable, with no treatment coefficient. `mapping=None` binds an artifact whose
coefficients are exactly those columns (names equal when the source names them);
`PriorMapping.named_parameters([...])` pairs a source that also carries other coefficients
(a treatment coefficient, for example) with the outcome columns, and design columns left
unpaired keep the weakly informative default. A prior whose coordinates do not map, or an
effect-functional mapping, refuses `quantity_semantics_mismatch` naming the declaration that
would map it. `Aipw` takes the prior on its cross-fitted untrimmed all-observed mean ATE under
IID analytic inference only: `bootstrap=0` is required (the bootstrap refits the unregularized
nuisances), and a trimmed, ATT/ATC, quantile or exceedance fit, a penalized propensity,
`cluster_dml` or a declared `se` refuses `route_not_supported`; it keeps no score table, so a
retarget refuses `score_table_unavailable`. `DRLearner` and `CausalForest` take no prior. A
`ComposedPrior` is accepted when its coordinates are named
(`ExternalPriorSourceSpec(coefficient_names=...)`): a column of the outcome regression takes the
coordinate of its name, a column with no coordinate keeps the weakly informative baseline, and a
coordinate with no column (the treatment coefficient) is dropped, all recorded as prior
assumptions. A kernel- or row-weighted target composition and a prior at a data-selected
power-prior discount map the same way. A `PriorMapping.named_parameters` renames coordinates; an
unnamed prior is positional over a design of the same width; a prior none of whose names is an
outcome column refuses `quantity_semantics_mismatch`. The partially linear score, whole-cluster cross-fitting, caller-supplied learner
factories and the population-linear interval refuse a prior (`route_not_supported`); a
prior-regularized fit keeps no cached nuisance bundle and is not replayable through the
recalculation session.

## What is refused

- A partially linear DML fit, a trimmed DML fit, a DR-Learner or CausalForest fit under a
  non-`AllObserved` target or trimming: no table, `score_table_unavailable`.
- Quantile and exceedance functionals read the AIPW table only; the learner routes keep no
  functional grid, so those refuse.
- DR-Learner CATE profiles and CausalForest CATE intervals are separate cells and are not
  extended by this one.

The `score_table_unavailable` code and the fields every refusal carries are described in
[refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies).
For a retarget of a single adjusted fit (a GLM adjustment fit rather than a DML table), see the runnable example
[`examples/python/retarget_adjusted_fit.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/retarget_adjusted_fit.py).

## Tests

`crates/antecedent-estimate/src/dml.rs` (inline): the score-mean identity and the iid
covariance against raw sums, a hand calculation of the table formulas on a fixed nuisance
bundle (`mu0 = 1`, `mu1 = 3`, `e = 0.5`), seeded replay, partially linear and trimmed fits
keeping no table, and the DR-Learner and CausalForest tables.
`crates/antecedent-estimate/tests/dml_outcome_prior_reference.rs`: the per-fold posterior
mode of the arm regressions against an independent 2 x 2 solve, the AIPW point and per-arm
scores against a hand-computed doubly robust estimate from the published out-of-fold
propensity, consistency under a badly misspecified prior with a correct propensity, a very wide
prior against the unpenalized fit, and the refusals. `crates/antecedent/tests/dml_outcome_prior_calibration.rs`
measures the interval's coverage under a weak, an informative and a misspecified prior.
`crates/antecedent/tests/dml_scores_retarget.rs`: retarget after estimate with the
independent sums for points and the cross-plan covariance, the prepare-time table equal to
the estimate's, the typed refusals, and DR-Learner / CausalForest.
