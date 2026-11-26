# Estimate with sampling weights

Use this guide when your rows come from a survey or another sample with unequal inclusion
probabilities. You want the effect in the population the sample represents. A sampling weight
`s_i` says how many population units row `i` stands for. That weight is an inverse inclusion
probability, possibly calibrated. You need a numeric weight column, or one weight per row.
You also need an analysis that reads weights. The table below lists them.

```python
import antecedent as ant
from antecedent.estimators import LinearAdjustment

result = ant.analyze(
    data,                                   # holds columns t, y, z and the weight column w
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=ant.AverageEffect("t", "y"),
    estimator=LinearAdjustment(weights="w"),
)
result.answer.value                         # the effect in the weighted population
weights = result.value.estimate.sampling_weights
weights.rows_used, weights.zero_weight_rows
weights.effective_sample_size, weights.design_effect
weights.target_population
```

Declaring weights changes **what is estimated** ([the weighted population](#what-is-estimated)).
It also changes **how uncertain it is** ([the variance](#variance)). Every analysis that reads
weights follows one contract. An analysis that cannot honour the contract refuses the weights
(`option_not_applicable`, detail `sampling_weights.route_unsupported`). It does not ignore
them. The identifying assumptions of the query are unchanged. The weights add the assumption
that they are the inverse inclusion probabilities of the design, up to a common scale.

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

The first nine rows are estimators (`antecedent.estimators`). `RandomizedEffect` and
`PanelDifferenceInDifferences` are queries and take `weights=` themselves. So does the temporal
`PulseEffect` (see [where weights are refused](#where-weights-are-refused)).
`PanelDifferenceInDifferences` and `PulseEffect` accept a column name only.

`weights` names a numeric column of the table passed to `analyze` or `prepare`, or gives one
weight per row. The weights belong to the original rows. A row dropped as incomplete or outside
the target population takes its weight with it. A weight is never shifted onto another row. A
named column is frozen at prepare like a cluster column. A rebound table whose weights changed
is refused. Call `ant.prepare(data, ...)` to declare the new ones.

In Rust, call `with_sampling_weights` with `SamplingWeights::new(weights)?` on these estimators:

- `LinearAdjustmentAte`
- `PropensityWeighting`
- `PropensityStratification`
- `PropensityMatching`
- `AipwAte`

The randomized and panel analyses read the table's observation weights. Call
`TabularData::with_weights`.

Only the average effect reads weights declared on its estimator. The other static effect
analyses have no weighted construction:

- conditional effect
- natural-direct mediation
- counterfactual
- mean curve
- average derivative
- intervention response
- continuous dose response
- regression discontinuity
- the prepared temporal designs
- synthetic control
- observational survival
- the policy value

A table that carries observation weights is refused with `option_not_applicable`
(`sampling_weights.route_unsupported`) when the study is prepared or a table is rebound.
Weights declared on an estimator such an analysis does not run are refused the same way. The
rows are never estimated as if they were unweighted. The average effect on a Dag or an ADMG
with weights declared on `LinearAdjustmentAte` equals the same fit on rows repeated by integer
weights. A weighted table given to it without that declaration is refused too. Declare the
weights on the estimator, or drop them from the table.

## What is estimated

The **weighted population**: the population the declared weights represent. With `τ_i` the
unit-level effect of row `i` (the difference of its potential outcomes under the two treatment
values of the query), an average
effect under weights is `Σ s_i τ_i / Σ s_i` for the weighted finite population and the
corresponding functional of the super-population when rows are independent draws from a
design with inclusion probabilities `1 / s_i`. The estimate does not speak for the sampled
units themselves: with informative sampling the two differ, and the unweighted estimator of
the same route is biased for the population (the known-truth tests sample a super-population
with inclusion probabilities that rise with a confounder and check exactly this).

**Composition with a target population.** A target-population declaration multiplies the
sampling weights, row by row. That declaration is a custom distribution or row weights. The
target law is reweighted *within* the weighted population. A predicate target restricts the
rows first. The weights follow the surviving rows.

## Normalization

Estimates, standard errors and intervals depend on the weights only through their ratios.
Weights are normalized to mean one over the rows an estimator analyses. Rescaling the declared
weights by any positive factor changes nothing. The `weight_sum` that is reported is the
declared one before normalization. It is **never read as a population size**. No
finite-population correction is made.

## Zero weights and invalid weights

A row of weight zero stands for no population unit. It is **excluded and counted**
(`zero_weight_rows`). The estimate equals the one computed on the table with those rows
deleted. The equality is exact for the deterministic analyses, with two qualifications:

- The temporal window's block bootstrap resamples the time axis, and deleting a window would
  join windows that are not adjacent. Zero-weight windows therefore stay in the sequence,
  contribute no score and are counted; the point estimate is unchanged and the bootstrap
  standard error reflects the resampled time axis.
- The cross-fitted AIPW assigns folds from the original row index, so it agrees up to fold noise.

A negative, NaN or infinite weight is refused, naming the first offending row (code
`sampling_weights_invalid`, detail `sampling_weights.invalid`; see
[other registered refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes));
so are all-zero weights, a declaration that does not cover the table's rows, and weights with
no positive row left after the other selections.

## Variance

The standard error is the **linearization (Taylor-series) estimator of the with-replacement
design**: a sandwich over weighted estimating equations. It treats rows, or declared clusters,
as the independent primary sampling units.

- Linear adjustment: the HC1 sandwich of the scores `s_i x_i e_i`, or the cluster sandwich of the same scores. The classical homoskedastic request is refused with weights. A model-based variance treats weights as precision. Python defaults to HC1 when `weights=` is set and `se=` is not.
- Propensity weighting: the stacked weighted-logistic and weighted-Hajek estimating
  equations, so the estimated propensity is in the variance.
- Stratification: the weighted per-stratum arm-mean variances pooled by weighted stratum
  mass, conditional on the strata.
- Matching: `σ̂² Σ c_i²`. `c_i` is a row's own weight plus the weights of the matched units it donates to. `σ̂²` is the weighted pooled matched-effect variance with the reliability-weights correction `1 / (1 − Σ w̃²)`. At constant weights, of any scale, it is exactly the unweighted Abadie-Imbens variance. A sampling weight is one unit standing for many. It is not repeated rows. The variance differs from that of replicated rows. The point estimate does not.
- AIPW: the influence function of the weighted mean of the cross-fitted scores. The in-sample ATT, ATC and trimmed ATE use the weighted efficient influence function with the weighted stacked-nuisance correction. Cluster-DML is the cluster sandwich of the weighted influence.
- Distance matching: the weighted matched-mean form used for propensity matching.
- GLM adjustment: the sandwich of the weighted scores `m_i u_i x_i` and information `m_i w_i x_i x_iᵀ`, delta-method onto the weighted g-computation average. The Fisher delta-method standard error treats weights as precision. It is refused.
- DML and DR-learner: the influence function of the weighted mean of the orthogonal score. The partially linear score uses `m_i (ΔT ΔY − θ ΔT²)` over the weighted Jacobian. The cross-fit estimates the nuisances out of fold. Their estimation is second order.
- Randomized trial and panel DiD: the linearized Hajek arm-mean variances. For equal weights that is Neyman's conservative bound. The cluster score sandwich of the weighted change scores is the other form.

**Intervals.** Weighted and unweighted constructions have separate calibration
keys. A sampling-weight declaration or raw standard error does not carry an
interval support by itself. The exact estimator, population target, variance,
dependence, reference distribution and functional must match a current record.

The published scalar endpoint uses its reported analytic reference. That reference is normal
when no positive reference degrees of freedom are supplied. It is Student when they are.
Weighted cluster AIPW, DML and DR-learner records retain the actual Student reference. The
ordinary raw fits and their normal-reference recipes remain distinct from the opt-in model
methods.

Rust exposes separate interval producers:

- `PropensityWeighting::population_linear_score_interval`: balanced positive population weights and independent homoskedastic Gaussian arm-linear outcomes. The population score includes conditional outcome-noise and random-target uncertainty. `population_score_interval` has a separately measured cluster scope. That evidence does not cover its iid working-quadratic variant.
- `AipwAte`, `DmlAte` and `DrLearner::population_linear_interval`: independent
  homoskedastic Gaussian arm-linear outcomes, complete training-fit uncertainty
  and the declared projection/ridge correction.
- `AipwAte::population_linear_atc_interval`: a control-population target under
  arm-linear means, with complete treated-training HC3 noise variance and target
  heterogeneity; this is a distinct heteroskedastic asymptotic construction.

Each method owns its exact collected records and assumptions. The explicit model intervals
do not replace the default robust fit. This list does not claim an equivalent Python method.
Matching's opt-in population-linear variance has its own donor-reuse and shared-outcome-fit
construction. The measurement suite is `weighted_interval_calibration`. Its recipes are not
all `estimate ± 1.96 se`.

## Effective sample size

Read `result.value.estimate.sampling_weights`. In Rust, read
`EffectEstimate::diagnostics.sampling_weights`. It reports `rows_used` and
`zero_weight_rows`. It reports the Kish `effective_sample_size` `(Σ s)² / Σ s²` of the rows
used, the `weight_sum` and the Kish `design_effect` `rows_used / effective_sample_size`.
The randomized and panel analyses also carry it in a diagnostic.

## Where weights are refused

- Static effect routes other than the average effect refuse table weights and estimator
  weights (see "Declaring weights"); so does the average effect for table weights it was not
  told to read.
- AIPW refuses non-mean outcome functionals. A penalized propensity (ridge, lasso, or a
  penalized fallback) reads weights: each fold's penalty is chosen by the weighted inner
  cross-validated log loss (columns standardized by weighted training moments) and every fit
  is the prior-weighted penalized likelihood.
- `LinearAdjustment` reads weights in its OLS fit; ridge, lasso and Huber refuse them.
- `GlmAdjustment` refuses the Fisher delta-method standard error; the negative-binomial
  dispersion is updated from the weighted data (weighted Pearson moment or weighted profile
  likelihood).
- DML and DR-learner need nuisance and final-stage learners that carry prior weights through every fold. Those learners are linear, ridge, logistic and gradient-boosted. Elastic net, forests and neural nets refuse, naming the learner. A weighted fit keeps no row-identified score table. It is not retargetable. The DR-learner's `cate_se` and pointwise profiles are the HC0 sandwich of the weighted least squares. A `cluster_dml` declaration with weights takes the cluster sandwich of the weighted influence.
- The frequentist temporal linear adjustment reads weights on a single window and on a sustained window of any length. Each lag window is weighted by the observation it ends at. A multi-step sustained contrast is the weighted least squares of every mechanism on its unfolded windows. The block bootstrap carries each window's weight with it. Conditional effects, Bayesian fits, refuters and series with gaps refuse. So do the prepared contrast designs, modifiers and level evaluations of the temporal analyses. Weights are read by the frequentist analysis only. In Python, `PulseEffect(weights="w")` declares them. `SustainedEffect` has no `weights` argument. Weight a multi-step sustained window from Rust.
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
of the analysis part's weights. Read `SampleSplit.analysis_weights`, `analysis_row_ids` and
`analysis_effective_sample_size`. In Rust, call `SplitDiscovery::estimate_analysis_effect`.
It estimates a linear effect on the analysis part with those weights as sampling weights. It
adjusts for a declared set, typically the treatment's parents in the discovered graph. Its
estimand is the effect in the weighted population the analysis part represents, **conditional
on the selected graph**. Discovery saw none of these rows. Weights are normalized to mean one
over the rows used. The standard error defaults to the sampling-weighted HC3 sandwich. Call
`estimate_analysis_effect_hc1` for HC1. You can also take the cluster sandwich over declared
analysis-part clusters. Rows of weight zero are excluded and counted. The Kish size is
reported. The result carries the binding that ties it to the split receipt and the discovery.
Its interval method returns a computed Student interval. The calibration status is
`not_measured`. Coverage records belong to the exact declared discovery, split and variance
recipe. They do not extend to an arbitrary split.

In Python, estimate on the analysis part with `weights=split.analysis_weights`:

```python
from antecedent.estimators import LinearAdjustment

result = ant.analyze(
    split.analysis, graph=graph, query=query, estimator=LinearAdjustment(weights=split.analysis_weights)
)
```

## Where next

- [Clusters](clustered-dml.md): rows that share a sampling or assignment unit.
- [Missing values](missing-data.md): how dropped rows are accounted for.
- [Discover, then estimate](../selective-discovery.md): the sample split whose analysis part is weighted above.
- [Refusal codes and remedies](../reference/refusal-codes.md): reading `code`, `detail` and `remedy` on a refusal.
