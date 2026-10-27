# Sampling weights

A sampling weight `s_i` says how many population units row `i` stands for (an inverse
inclusion probability, possibly calibrated). Declaring weights changes **what is estimated**
and **how uncertain it is**, so every route that reads them follows one contract, and a route
that cannot honour the contract refuses the weights (`option_not_applicable`,
`sampling_weights.route_unsupported`) instead of ignoring them.

## Declaring weights

| Route | Declaration | Reads |
|---|---|---|
| `LinearAdjustment` | `weights="w"` | weighted least squares, sampling sandwich |
| `PropensityWeighting` | `weights="w"` | weighted logistic propensity, weighted Hajek means |
| `PropensityStratification` | `weights="w"` | weighted logistic propensity, weighted strata |
| `PropensityMatching` | `weights="w"` | weighted logistic propensity, weighted matched mean |
| `Aipw` | `weights="w"` | weighted nuisances, weighted score (cross-fitted ATE, ATT, ATC, trimmed ATE, cluster-DML) |
| `DistanceMatching` | `weights="w"` | weighted diagnostic propensity, weighted matched mean |
| `GlmAdjustment` | `weights="w"` | prior-weighted IRLS (logit, probit, Poisson, Gaussian), weighted g-computation |
| `DML` | `weights="w"` | prior-weighted nuisance learners, weighted AIPW or partially linear score |
| `DRLearner` | `weights="w"` | prior-weighted nuisances and final stage, weighted marginal ATE |
| `RandomizedEffect` | `weights="w"` | weighted Hajek means per arm (Bernoulli and complete designs) |
| `PanelDifferenceInDifferences` | `weights="w"` | weighted group means of the change scores (balanced panel) |

`weights` names a numeric column of the table passed to `analyze` / `prepare`, or gives one
weight per row. The weights belong to the original rows: a row dropped as incomplete or outside
the target population takes its weight with it, so a weight is never shifted onto another
row. A named column is frozen at prepare like a cluster column, and a rebound table whose
weights changed is refused (`ant.prepare(data, ...)` declares the new ones).

Rust: `with_sampling_weights(SamplingWeights::new(weights)?)` on `LinearAdjustmentAte`,
`PropensityWeighting`, `PropensityStratification`, `PropensityMatching` and `AipwAte`; the
randomized and panel routes read the table's observation weights
(`TabularData::with_weights`).

## What is estimated

The **weighted population**: the population the declared weights represent. An average
effect under weights is `Σ s_i τ_i / Σ s_i` for the weighted finite population and the
corresponding functional of the super-population when rows are independent draws from a
design with inclusion probabilities `1 / s_i`. The estimate does not speak for the sampled
units themselves: with informative sampling the two differ, and the unweighted estimator of
the same route is biased for the population (the known-truth tests sample a super-population
with inclusion probabilities that rise with a confounder and check exactly this).

**Composition with a target population.** A target-population declaration (a custom
distribution or row weights) multiplies the sampling weights, row by row: the target law is
reweighted *within* the weighted population. A predicate target restricts the rows first and
the weights follow the surviving rows.

## Normalization

Estimates, standard errors and intervals depend on the weights only through their ratios.
Weights are normalized to mean one over the rows an estimator analyses; rescaling the declared
weights by any positive factor changes nothing, and the `weight_sum` that is reported is the
declared one before normalization. It is **never read as a population size**: no
finite-population correction is made.

## Zero weights and invalid weights

A row of weight zero stands for no population unit. It is **excluded and counted**
(`zero_weight_rows`), and the estimate equals the one computed on the table with those rows
deleted (exactly for the deterministic routes, except the temporal window: its block bootstrap resamples the time axis, and deleting a window would join windows that are not adjacent, so zero-weight windows stay in the sequence, contribute no score and are counted — the point estimate is unchanged, its bootstrap SE reflects the resampled time axis; the cross-fitted AIPW assigns folds from the
original row index, so it agrees up to fold noise). A negative, NaN or infinite weight is
refused, naming the first offending row (`sampling_weights_invalid`,
`sampling_weights.invalid`); so are all-zero weights, a declaration that does not cover the
table's rows, and weights with no positive row left after the other selections.

## Variance

The standard error is the **linearization (Taylor-series) estimator of the with-replacement
design**: a sandwich over weighted estimating equations. It treats rows, or declared clusters,
as the independent primary sampling units.

- Linear adjustment: the HC1 sandwich of the scores `s_i x_i e_i` (the classical homoskedastic
  request is refused with weights — a model-based variance treats weights as precision; Python
  defaults to HC1 when `weights=` is set and `se=` is not), or the cluster sandwich of the same scores.
- Propensity weighting: the stacked weighted-logistic and weighted-Hajek estimating
  equations, so the estimated propensity is in the variance.
- Stratification: the weighted per-stratum arm-mean variances pooled by weighted stratum
  mass, conditional on the strata.
- Matching: `σ̂² Σ c_i²`, where `c_i` is a row's own weight plus the weights of the matched
  units it donates to and `σ̂²` is the weighted pooled matched-effect variance with the
  reliability-weights correction `1 / (1 − Σ w̃²)`. At constant weights, of any scale, it is
  exactly the unweighted Abadie-Imbens variance. A sampling weight is one unit standing for
  many, not repeated rows, so the variance differs from that of replicated rows (the point
  estimate does not).
- AIPW: the influence function of the weighted mean of the cross-fitted scores; the in-sample
  ATT, ATC and trimmed ATE use the weighted efficient influence function with the weighted
  stacked-nuisance correction; cluster-DML is the cluster sandwich of the weighted influence.
- Distance matching: the weighted matched-mean form used for propensity matching.
- GLM adjustment: the sandwich of the weighted scores `m_i u_i x_i` and information
  `m_i w_i x_i x_iᵀ`, delta-method onto the weighted g-computation average. The Fisher
  delta-method standard error treats weights as precision and is refused.
- DML and DR-learner: the influence function of the weighted mean of the orthogonal score
  (the partially linear score uses `m_i (ΔT ΔY − θ ΔT²)` over the weighted Jacobian). The
  cross-fit estimates the nuisances out of fold, so their estimation is second order.
- Randomized trial and panel DiD: the linearized Hajek arm-mean variances (Neyman's
  conservative bound for equal weights) and the cluster score sandwich of the weighted change
  scores.

**Intervals.** Weighted and unweighted constructions have separate calibration
keys. A sampling-weight declaration or raw standard error does not carry an
interval license by itself. The exact estimator, population target, variance,
dependence, reference distribution and functional must match a current record.

The published scalar endpoint uses its reported analytic reference: normal
when no positive reference degrees of freedom are supplied, Student when they
are. Weighted cluster AIPW, DML and DR-learner records retain the actual Student
reference. The ordinary raw fits and their normal-reference recipes remain
distinct from the opt-in model methods.

Rust exposes separate interval producers:

- `PropensityWeighting::population_linear_score_interval`: balanced positive
  population weights and independent homoskedastic Gaussian arm-linear outcomes,
  with conditional outcome-noise and random-target uncertainty in the population
  score. `population_score_interval` has a separately measured cluster scope;
  that evidence does not license its unmeasured iid working-quadratic variant.
- `AipwAte`, `DmlAte` and `DrLearner::population_linear_interval`: independent
  homoskedastic Gaussian arm-linear outcomes, complete training-fit uncertainty
  and the declared projection/ridge correction.
- `AipwAte::population_linear_atc_interval`: a control-population target under
  arm-linear means, with complete treated-training HC3 noise variance and target
  heterogeneity; this is a distinct heteroskedastic asymptotic construction.

Each method owns its exact collected records and assumptions. The explicit
model intervals do not replace the default robust fit, and this list does not
claim an equivalent Python method. Matching's opt-in population-linear variance
likewise has its own donor-reuse and shared-outcome-fit construction. The
measurement suite is `weighted_interval_calibration`; its recipes are not all
`estimate ± 1.96 se`.

## Effective sample size

`result.estimate.sampling_weights` (Rust: `EffectEstimate::diagnostics.sampling_weights`)
reports `rows_used`, `zero_weight_rows`, the Kish `effective_sample_size`
`(Σ s)² / Σ s²` of the rows used, the `weight_sum` and the Kish `design_effect`
`rows_used / effective_sample_size`. The randomized and panel routes also carry it in a
diagnostic.

## Where weights are refused

- AIPW refuses non-mean outcome functionals. A penalized propensity (ridge, lasso, or a
  penalized fallback) reads weights: each fold's penalty is chosen by the weighted inner
  cross-validated log loss (columns standardized by weighted training moments) and every fit
  is the prior-weighted penalized likelihood.
- `LinearAdjustment` reads weights in its OLS fit; ridge, lasso and Huber refuse them.
- `GlmAdjustment` refuses the Fisher delta-method standard error; the negative-binomial
  dispersion is updated from the weighted data (weighted Pearson moment or weighted profile
  likelihood).
- DML and DR-learner need nuisance and final-stage learners that carry prior weights through
  every fold (linear, ridge, logistic and gradient-boosted); elastic net, forests and neural
  nets refuse, naming the learner. A weighted fit keeps no row-identified score table, so it
  is not retargetable. The DR-learner's `cate_se` and pointwise profiles are the HC0 sandwich of
  the weighted least squares. A `cluster_dml` declaration with weights takes the cluster
  sandwich of the weighted influence.
- A time series with zero-weight lag windows keeps them in the design so the windows stay
  consecutive; they contribute no score, and the block length is sized on the windows of
  positive weight.
- `RandomizedEffect` reads weights for unadjusted Bernoulli and complete randomization.
  Cluster, stratified, factorial, multi-arm, switchback, receipt-adjusted, CUPED, ANCOVA and
  exact-randomization designs, and outcomes with missing values under a missing-data policy,
  refuse them.
- `PanelDifferenceInDifferences` reads weights on the balanced two-period panel; repeated
  cross sections, staggered and augmented designs refuse them. Both rows of a subject must
  carry its weight.

## Weighted estimation on the analysis part of a split

A sample split keeps each row's weight unnormalized, its original row index and the Kish size
of the analysis part's weights (`SampleSplit.analysis_weights`, `analysis_row_ids`,
`analysis_effective_sample_size`). `SplitDiscovery::estimate_analysis_effect` (Rust) estimates
a linear effect on the analysis part with those weights as sampling weights, adjusting for a
declared set (typically the treatment's parents in the discovered graph). Its estimand is the
effect in the weighted population the analysis part represents, **conditional on the selected
graph** (discovery saw none of these rows); weights are normalized to mean one over the rows
used; the standard error defaults to the sampling-weighted HC3 sandwich (explicit HC1 remains available for historical replay), or the cluster sandwich over
declared analysis-part clusters; rows of weight zero are excluded and counted; and the Kish
size is reported. The result carries the binding that ties it to the split receipt and the
discovery. Its interval method returns a computed Student interval labelled unmeasured; whole-procedure records belong to the exact declared discovery/split/variance recipe and do not supply automatic calibration authority to an arbitrary split.

In Python, estimate on the analysis part with `weights=split.analysis_weights`:

```python
result = ant.analyze(
    split.analysis, graph=graph, query=query, estimator=LinearAdjustment(weights=split.analysis_weights)
)
```
