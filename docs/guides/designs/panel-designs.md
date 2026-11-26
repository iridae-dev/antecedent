# Estimate a policy effect from panel data

Estimate what a rollout, launch or policy did when you observed treated and untreated units before and after the change and nothing randomized who was treated. This guide covers difference-in-differences on a two-period panel, event studies for staggered adoption, and the synthetic control family for one or a few treated units. For a threshold rule on a running variable, see [Regression discontinuity and kink](discontinuity-and-kink.md).

These designs identify an effect only through assumptions you declare. Antecedent cannot test them. You declare that the treated units would have followed the comparison units' trend without the change. You declare that nothing responded before adoption. You declare that units do not affect each other.

The [graphless support matrix](../../graphless-support-matrix.md) has a row for each of these:

- the two-period difference-in-differences, and its Normal-Inverse-Gamma posterior
- the never-treated event study
- the overall effect of a balanced staggered rollout's simple aggregation, and its Bayesian bootstrap
- the covariate-adjusted two-period effect under bounded overlap
- the augmented panel's bootstrap interval, when the supplied predictions are known functions rather than estimates
- for synthetic control: the conformal sets, the placebo-variance interval of a staggered rollout, and the donor-weight posterior with its predictive path

Each row states the sizes, options, and replicate or draw counts it was measured at. A fit outside its row runs. The result says that fit is unsupported. Every other interval on this page runs and is unsupported. Those intervals are dynamic aggregations, group aggregations, and calendar aggregations. They are also group-time cells, pre-trend leads, simultaneous bands, and unbalanced panels. The plain synthetic control publishes a point, a descriptive placebo rank, and an exact randomization tail. It is supported as exactly that. Every result carries its support status. Every interval carries its calibration status in `result.inspect().calibration`. The result lists the assumptions it rests on. A good pre-period fit or a flat pre-trend is a reason to look harder, not a verification of the assumption.

## Two-period difference-in-differences

`PanelDifferenceInDifferences` takes a balanced two-period panel with one pre and one post outcome per subject.

```python
import antecedent
from antecedent.quasi import PanelDifferenceInDifferences
```

```python
query = PanelDifferenceInDifferences("outcome", "subject", "treated", "post", cluster="site")
result = antecedent.analyze(data, query=query)
result.value.panel_did.estimate, result.value.panel_did.standard_error
```

The estimand is the average effect on the treated in the post period, `ATT = E[Y_post(1) - Y_post(0) | D = 1]`, where `D` is the stable treatment indicator. It is identified by parallel trends: `E[Y_post(0) - Y_pre(0) | D = 1] = E[Y_post(0) - Y_pre(0) | D = 0]`, that is, without treatment the treated group's mean outcome would have changed by the same amount as the comparison group's.

The query names the outcome, the subject, the stable treatment indicator, and the pre/post indicator. You can also name a higher-level cluster column. The result reports the difference in mean subject-level changes under `result.value.panel_did`. The standard error is a cluster score-sandwich. At least two clusters must contribute to each treatment group. Clusters must nest within treatment groups. A pointwise 95% normal interval appears when each group has at least 30 independent clusters. Repeated cross sections use the query's explicit sampling mode. This query refuses staggered adoption. This query refuses missing waves. Use the event study below.

When parallel trends hold only after adjusting for baseline covariates, use `CovariateAdjustedDiD`. Set `nuisance` to `outcome_regression`, `inverse_probability_weighting`, or `doubly_robust`.

[`examples/python/quasi_difference_in_differences.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/quasi_difference_in_differences.py) runs a repeated cross section and a balanced panel on simulated data with a known effect of 4.

### Augmented panel DiD with your own nuisance models

`AugmentedPanelDiD` takes one row per subject with pre and post outcomes, a binary treatment indicator, propensity scores and predictions of the untreated outcome change. The treated mean change minus the predicted counterfactual change is corrected by propensity-odds-weighted control residuals.

```python
import antecedent
from antecedent.quasi import AugmentedPanelDiD

query = AugmentedPanelDiD("pre", "post", "subject", "treated", "propensity", "untreated_change")
prepared = antecedent.prepare(data, query=query)
result = prepared.estimate()
assert result.value.panel_did.uncertainty == "point_only_no_standard_error"
```

You supply the nuisance models. Antecedent does not fit them. `predictions_cross_fitted=True` records your declaration. It is not checked against fold ownership. Propensities must lie strictly inside (0, 1). The result reports the propensity range and the effective weighted control count as support diagnostics. Without `resampling` the result is point-only, as above. With `resampling="cluster_bootstrap"` and `replicates`, whole clusters are resampled. `"cluster_multiplier"` does the same. The result carries `interval_95` and its standard error. The supplied propensities and predictions are held fixed in every replicate. The interval leaves out their own estimation error.

## Event study with staggered adoption

When units adopt at different times, an event study reports one contrast per cohort and event time instead of one number.

```python
import antecedent
from antecedent.quasi import StaggeredAdoption

query = StaggeredAdoption("outcome", "subject", "period", "cohort", event_study=True)
result = antecedent.analyze(data, query=query)
effects = result.value.panel_did.effects
```

The columns are the outcome, the subject, the calendar period, and the first-treatment cohort. Cohort 0 is reserved for units never treated. The reference period is the one just before adoption (`g - 1`). Cohort 0 is the comparison group. Every unit needs one observation at every common period. Each treated cohort needs its baseline and adoption periods. A design with no never-treated units is refused. Unsupported timing is refused.

`effects` lists cohort, calendar period, event time, effect, and treated and control counts for every comparison, including descriptive pre-adoption contrasts. The top-level scalar is the first post-adoption contrast. Read the whole curve.

Uncertainty is a pointwise cluster-robust standard error from cluster-aggregated influence contributions. The finite-cluster multiplier is `G/(G - 1)`. A post-adoption contrast gets a pointwise 95% interval when each side has at least 24 independent clusters. These intervals are not a simultaneous band. The pre-period diagnostic is the largest absolute standardized lead among non-reference leads. It has no calibrated cutoff. A large value directs scrutiny to pre-treatment differences. A small value does not establish parallel trends.

### Heterogeneity-robust rollouts

A static two-way fixed-effects coefficient can use already-treated units as controls. `StaggeredRollout` estimates the effect for each adoption cohort and period without doing that. It reports its answer under `result.quasi_rollout`.

```python
from antecedent.quasi import StaggeredRollout

query = StaggeredRollout(
    "outcome", "subject", "period", "cohort",
    estimator="callaway_santanna", aggregation="dynamic",
)
```

- `estimator` is one of three. `callaway_santanna` uses 2x2 comparisons. It can adjust for baseline `covariates` by outcome regression, inverse-probability weighting, or a doubly robust fit. `sun_abraham` uses interaction-weighted cohort-by-event-time contrasts. On an unbalanced panel it is the saturated cohort-by-period regression fitted on the observed unit-periods. `imputation` imputes untreated outcomes from unit and period effects fitted on untreated rows.
- `comparison` is `never_treated` or `not_yet_treated`.
- `aggregation` is one of four. `simple` is the cohort-size-weighted average of post-adoption cells. `dynamic` is the event-time profile. `group` is one number per cohort. `calendar` is one number per period.
- `anticipation=delta` moves the base period to `g - 1 - delta`.
- `resampling` is `cluster_bootstrap` or `cluster_multiplier`, with `replicates`. It redraws or reweights whole clusters. Each replicate refits every cell, nuisance model, and aggregation. A declaration of dependent time-series resampling is refused.

Pre-adoption contrasts are reported as falsification leads with a simultaneous band. They never enter an aggregation. The static two-way fixed-effects coefficient is reported beside the summary as a diagnostic.

### Augmented panel with missing subjects

The augmented panel is one row per subject. An unbalanced panel there means subjects missing an outcome or a supplied prediction. Under `missing=available_case`, such subjects are dropped whole. `balanced_panel` is identical here. The row-aligned outcome, propensity, and prediction columns are restricted together. Nothing is imputed. The missing-data accounting counts the dropped subjects. `attrition_ignorable_given_treatment_group_and_predictors` is recorded as an untestable assumption. Without a declaration the first missing cell is named. A policy is required.

### Unbalanced panels

Every estimator reads an unbalanced panel as it is. `callaway_santanna` uses the units observed at both periods of each 2x2. `imputation` uses every observed untreated row. `sun_abraham` fits unit effects, period effects, and one coefficient per cohort and period other than the period before adoption. The fit is least squares on the observed unit-periods only. No observation is invented. No unrelated period is joined.

For `sun_abraham` a cohort-period coefficient is reported only where a never-treated unit is observed at that period. Without a never-treated unit, the last-treated cohort before its adoption is the comparison. That period must also be linked to the cohort's base period through the comparison units' own observed periods and through the cohort's own units. Unit-periods without that support are dropped and counted. A cohort with no unit observed at its base period contributes no cell. It is reported in the dropped cohorts. It is not silently reweighted. The estimand stays cohort specific. The result carries `dropped_rows` and `dropped_units`. For every cohort `g` it also carries `cohort_g_units`, `cohort_g_rows`, and `cohort_g_units_at_base`. It also carries `cohort_g_post_cells`, `cohort_g_pre_cells`, and `cohort_g_dropped_rows`. On a balanced panel the regression reduces exactly to the 2x2 contrasts.

`observation` declares the process that decides which unit-periods are observed. `ignorable_given_cohort` is the default. It says that being observed is unrelated to the unit's untreated outcome trend within its cohort. It cannot be tested from the observed rows. It is recorded with the result as the untestable assumption `attrition_ignorable_given_cohort`. `informative` says attrition or late entry depends on outcomes or trends within cohort. With no bound nothing identifies the effect. The call refuses and names the next attempt. Set `attrition_shift_bound=B` with `observation="informative"`. That declares that the treated units a cohort fails to observe change by at most `B` outcome units more or less than those it observes. Comparison units are missing at random. The summary is linear in that shift. The result reports `attrition_slope`, the summary's weights times the share of each cohort unobserved at each cell. It reports `assumption_range_lower` and `assumption_range_upper`, the estimate minus and plus `B` times the slope. It reports `attrition_tipping_shift`, the shift at which the estimate crosses zero. The range is not a confidence interval. It is separate from the resampling intervals. Those intervals are conditional on a zero shift. The bound is recorded as the untestable assumption `attrition_shift_within_declared_bound`. A design with no comparison unit observed in two periods is refused. A single treated cohort with no unit at its base period is refused too.

Under Bayesian inference the rollout designs return a Bayesian-bootstrap posterior. The weights are flat-Dirichlet weights on whole independent units. Those units are clusters, never rows. Each draw refits every cell, nuisance model, and aggregation. Draws are joint across cells. The result carries posterior means, equal-tailed credible intervals, and simultaneous credible bands. A frequentist `resampling` declared beside it is refused.

## Synthetic control

`SyntheticControl` fits one treated unit against a donor pool, in a balanced panel by default or in an unbalanced panel over the treated unit's observed window.

```python
from antecedent import analyze
from antecedent.quasi import SyntheticControl

result = analyze(panel, query=SyntheticControl("outcome", "unit", "period", "treated", 5))
result.value.synthetic_control.estimate
```

Donor weights are nonnegative and sum to one. They are chosen by projected gradient descent on pre-period squared error. The effect is the treated unit's mean post-period outcome minus the weighted donors' mean. The design needs one treated unit. It needs at least three donors. It needs two or more pre-periods. It needs an observed intervention period. Every unit needs identical periods.

When the panel is read over the observed window the result carries `observed_window` (`ObservedWindowSupport`). That report has the pre- and post-periods read. It has `donors_used`. It has the excluded units with the window periods each misses. It has `unread_rows`. The same report is stored in the result artifact. On load it is checked against the frozen unit and period labels.

The result reports donor weights, pre-fit RMSE, the effective donor count, and leave-one-donor-out placebo effects. The placebo tail fraction compares post/pre RMSE ratios. It assumes exchangeable units. It is neither a confidence interval nor a supported test. Identification needs no anticipation. It needs stable treatment after adoption. It needs no concurrent treated-unit shock. It needs no interference. It needs a convex combination of donors to be a valid untreated counterfactual. Pre-fit RMSE has no acceptance threshold.

The prepared study freezes unit and period row order. Refreshing with changed design labels is refused.

### Options

- **Unbalanced panels.** `observed_window_support=True` fits the weights on the treated unit's observed pre-periods. It averages the treated unit's observed post-periods. The donors are the units observed in every one of those periods. Units missing part of the window are excluded and listed, with the unread unit-periods, in the `estimate.quasi.synthetic_control.observed_window_support` diagnostic. Nothing is imputed. A period only the donors have is never read. The design is refused with a next attempt when the treated unit has fewer than two observed pre-periods. It is refused when the treated unit has no observed post-period. It is refused when fewer than three donors are observed through the window. Whether a unit is observed being unrelated to its untreated outcome is an untestable assumption, recorded as `attrition_unrelated_to_untreated_outcomes`. It applies to the simplex point fit and the ridge-augmented point fit. Unit randomization reads a balanced panel and is refused with this option. A conformal interval is refused with it. A staggered interval is refused with it. The Bayesian posterior is refused with it.
- **Randomization inference.** If the treated unit was drawn uniformly from the observed units before outcomes were seen, set `uniform_unit_randomization=True`. Antecedent refits every possible treated-unit assignment against all other units. It reports the exact two-sided tail fraction as `randomization_p_value`. That is a Fisher sharp-null test under the declared assignment. It has no interval for the effect. Enumeration is limited to 32 candidate units.
- **Ridge augmentation.** Set `augmentation_ridge` to a finite positive penalty. It corrects the simplex gap with a donor-trained ridge model when the treated unit lies outside the donors' convex hull. The result reports `unadjusted_effect`, `outcome_model_correction`, and the adjusted `estimate`. The extra assumption is that the donor outcome model transports to the treated unit. The penalty is yours to set. It is not tuned from the panel. With uniform assignment the Fisher test refits weights and correction for every candidate. The displayed placebo rank still describes the unadjusted fit.
- **Conformal intervals.** `conformal=True` inverts a permutation test over time. Weights are refit under each hypothesized effect. You get one interval per post period. You also get a constant-effect interval. It needs at least 19 pre-periods for a 5% test. It is exact when residuals are exchangeable over time. It is approximate under weak dependence.
- **Several treated units.** Pass `adoptions=[(unit, period), ...]` with `placebo_replicates`. Each adopting unit is fit against never-treated donors at its own adoption period. The overall effect is the post-period-weighted average. The interval comes from reassigning adoptions to never-treated units and refitting. Conformal and staggered adoptions are alternatives.
- **Bayesian.** With staggered adoptions, `bayesian.synthetic_control` averages the per-unit simplex posteriors. With the conformal declaration it reports a posterior predictive interval at each post period. That interval is a model-based statement about the counterfactual path. It is a different object from the permutation set.

### What 95% means for the Bayesian average effect

For one treated unit, the donor-weight posterior reports the 2.5th and 97.5th percentiles of draws of the average post-treatment effect. Those draws include uncertainty in the donor weights and in the untreated outcome. The predictive path reports an interval for the same average. It averages the effects **within each simulated path**. It then takes the 2.5th and 97.5th percentiles of those averages. A single-period interval answers a separate question.

To interpret either interval, compare your donor count, pre-treatment periods, and post-treatment periods with the measured design. The following 3.0.0 primary-run measurements used five post-treatment periods, independent Gaussian noise, and a treated unit whose untreated outcome depended on three of the available donors. The predictive-average column describes the primary run’s implementation. That implementation sampled the average separately using the same weights and error model. Coverage for the revised within-path average awaits measurement. Each fraction counts how often a **returned** interval contained the known average effect. Skips are additional simulated panels for which the study had no interval to score.

| Donors | Pre-treatment periods | Posterior interval | Skips | Predictive path average | Skips |
|---:|---:|---:|---:|---:|---:|
| 20 | 30 | 374/400 (93.5%) | 0 | 385/400 (96.3%) | 0 |
| 20 | 40 | 382/400 (95.5%) | 0 | 372/400 (93.0%) | 0 |
| 20 | 80 | 373/400 (93.3%) | 0 | 382/400 (95.5%) | 0 |
| 40 | 30 | 1,821/1,994 (91.3%) | 6 | 1,807/1,993 (90.7%) | 7 |
| 40 | 40 | 1,828/1,993 (91.7%) | 7 | 1,806/1,993 (90.6%) | 7 |
| 40 | 80 | 1,854/2,000 (92.7%) | 0 | 384/400 (96.0%) | 0 |

For example, with 40 donors and 40 pre-treatment periods, 95% is the nominal model level. The predictive interval for the five-period average covered the truth in 90.6% of returned intervals in this synthetic design. These frequencies have Monte Carlo uncertainty. For another donor count, history, or horizon, an exact-design frequency requires a measurement at that design. Equal ratios of history to donors can give different results.

### Synthetic difference-in-differences

`SyntheticDifferenceInDifferences` weights both donor units and pre-periods on the simplex. It then computes the weighted difference-in-differences contrast. It needs a balanced panel. It needs one treated unit. It needs at least two donors. It needs two pre-periods. It needs one post-period. It reports `result.value.synthetic_did`, including both weight vectors. With one treated unit it returns no interval or standard error. It refuses a fabricated one in a saved artifact. As for synthetic control, `adoptions=[(unit, period), ...]` with `placebo_replicates` fits several adopting units. It reports placebo-variance intervals. `uniform_unit_randomization=True` adds the exact Fisher tail fraction as `randomization_p_value`, as for synthetic control. Ridge augmentation is not available here.

Runnable example: [`examples/python/panel_designs.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/panel_designs.py) runs synthetic control, synthetic difference-in-differences and a staggered event study on one simulated panel with a known, growing effect.

## Boundaries

- Covariate adjustment at the unit level for synthetic control is not available. Use the covariate-adjusted DiD or a rollout estimator.
- Intervals on every call above are pointwise. Read each result's calibration status before relying on coverage.
- If treatment turns on and off, the call refuses and names the next option. It also refuses when units influence each other. It also refuses when the panel is unbalanced where the estimator requires balance. See [Refusals](../../concepts/refusals.md) for how a refusal reaches your code and [refusal codes](../../reference/refusal-codes.md) for each code and its remedy.

## Related

- [Regression discontinuity and kink](discontinuity-and-kink.md) for threshold rules.
- [Sequential regimes](sequential-regimes.md) and [temporal designs](../temporal-designs.md) when treatment varies over time within units.
- [Uncertainty](../../concepts/uncertainty.md) for how to read an interval's calibration status.
