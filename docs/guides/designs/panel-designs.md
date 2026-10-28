# Estimate a policy effect from panel data

Estimate what a rollout, launch or policy did when you observed treated and untreated units before and after the change and nothing randomized who was treated. This guide covers difference-in-differences on a two-period panel, event studies for staggered adoption, and the synthetic control family for one or a few treated units. For a threshold rule on a running variable, see [Regression discontinuity and kink](discontinuity-and-kink.md).

These designs identify an effect only through assumptions you declare and Antecedent cannot test: that the treated units would have followed the comparison units' trend without the change, that nothing responded before adoption, and that units do not affect each other. Every result below executes and keeps its status as an unlicensed point utility or an unlicensed interval with calibration `unmeasured`; read `result.inspect().calibration` and the assumptions on the result. A good pre-period fit or a flat pre-trend is a reason to look harder, not a verification of the assumption.

## Two-period difference-in-differences

`PanelDifferenceInDifferences` takes a balanced two-period panel with one pre and one post outcome per subject.

```python
import antecedent
from antecedent.quasi import PanelDifferenceInDifferences
```

The query names the outcome, subject, stable treatment indicator, pre/post indicator and, optionally, a higher-level cluster column. The result reports the difference in mean subject-level changes under `result.value.panel_did`, with a cluster score-sandwich standard error. At least two clusters must contribute to each treatment group and clusters must nest within treatment groups. A pointwise 95% normal interval appears when each group has at least 30 independent clusters. Repeated cross sections use the query's explicit sampling mode. Staggered adoption and missing waves are refused by this query; use the event study below.

When parallel trends hold only after adjusting for baseline covariates, use `CovariateAdjustedDiD` with `nuisance` set to `outcome_regression`, `inverse_probability_weighting` or `doubly_robust`.

`examples/python/quasi_difference_in_differences.py` runs a repeated cross section and a balanced panel on simulated data with a known effect of 4.

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

You supply the nuisance models; Antecedent does not fit them. `predictions_cross_fitted=True` records your declaration and is not checked against fold ownership. Propensities must lie strictly inside (0, 1), and the result reports the propensity range and the effective weighted control count as support diagnostics. The result is point-only.

## Event study with staggered adoption

When units adopt at different times, an event study reports one contrast per cohort and event time instead of one number.

```python
import antecedent
from antecedent.quasi import StaggeredAdoption

query = StaggeredAdoption("outcome", "subject", "period", "cohort", event_study=True)
result = antecedent.analyze(data, query=query)
effects = result.value.panel_did.effects
```

The columns are the outcome, subject, calendar period and first-treatment cohort, with cohort 0 reserved for units never treated. The reference period is the one just before adoption (`g - 1`) and cohort 0 is the comparison group. Every unit needs one observation at every common period, and each treated cohort needs its baseline and adoption periods. Designs with no never-treated units or unsupported timing are refused.

`effects` lists cohort, calendar period, event time, effect and treated and control counts for every comparison, including descriptive pre-adoption contrasts. The top-level scalar is the first post-adoption contrast; read the whole curve.

Uncertainty is a pointwise cluster-robust standard error from cluster-aggregated influence contributions, with the finite-cluster multiplier `G/(G - 1)`. A post-adoption contrast gets a pointwise 95% interval when each side has at least 24 independent clusters; these are not a simultaneous band. The pre-period diagnostic is the largest absolute standardized lead among non-reference leads. It has no calibrated cutoff: a large value directs scrutiny to pre-treatment differences, and a small value does not establish parallel trends.

### Heterogeneity-robust rollouts

A static two-way fixed-effects coefficient can use already-treated units as controls. `StaggeredRollout` estimates the effect for each adoption cohort and period without doing that, and reports its answer under `result.quasi_rollout`.

```python
from antecedent.quasi import StaggeredRollout

query = StaggeredRollout(
    "outcome", "subject", "period", "cohort",
    estimator="callaway_santanna", aggregation="dynamic",
)
```

- `estimator`: `callaway_santanna` (2x2 comparisons, with outcome regression, inverse-probability weighting or doubly robust adjustment for baseline `covariates`), `sun_abraham` (interaction-weighted cohort-by-event-time contrasts; on an unbalanced panel it is the saturated cohort-by-period regression fitted on the observed unit-periods) or `imputation` (untreated outcomes imputed from unit and period effects fitted on untreated rows).
- `comparison`: `never_treated` or `not_yet_treated`.
- `aggregation`: `simple` (cohort-size-weighted average of post-adoption cells), `dynamic` (event-time profile), `group` (per cohort) or `calendar` (per period).
- `anticipation=delta` moves the base period to `g - 1 - delta`.
- `resampling` (`cluster_bootstrap` or `cluster_multiplier`, with `replicates`) redraws or reweights whole clusters and refits every cell, nuisance model and aggregation in each replicate. A declaration of dependent time-series resampling is refused.

Pre-adoption contrasts are reported as falsification leads with a simultaneous band and never enter an aggregation. The static two-way fixed-effects coefficient is reported beside the summary as a diagnostic.

### Augmented panel with missing subjects

The augmented panel is one row per subject, so an unbalanced panel there means subjects missing an outcome or a supplied prediction. Under `missing=available_case` (or `balanced_panel`, identical here) such subjects are dropped whole and the row-aligned outcome, propensity and prediction columns are restricted together; nothing is imputed, the missing-data accounting counts the dropped subjects, and `attrition_ignorable_given_treatment_group_and_predictors` is recorded as an untestable assumption. Without a declaration the first missing cell is named and a policy is required.

### Unbalanced panels

Every estimator reads an unbalanced panel as it is: `callaway_santanna` uses the units observed at both periods of each 2x2, `imputation` uses every observed untreated row, and `sun_abraham` fits unit effects, period effects and one coefficient per cohort and period (other than the period before adoption) by least squares on the observed unit-periods only. No observation is invented and no unrelated period is joined.

For `sun_abraham` a cohort-period coefficient is reported only where a never-treated unit (or, without one, the last-treated cohort before its adoption) is observed at that period and that period is linked to the cohort's base period through the comparison units' own observed periods and through the cohort's own units. Unit-periods without that support are dropped and counted, and a cohort with no unit observed at its base period contributes no cell: it is reported in the dropped cohorts rather than silently reweighted, and the estimand stays cohort specific. The result carries `dropped_rows`, `dropped_units` and, for every cohort `g`, `cohort_g_units`, `cohort_g_rows`, `cohort_g_units_at_base`, `cohort_g_post_cells`, `cohort_g_pre_cells` and `cohort_g_dropped_rows`. On a balanced panel the regression reduces exactly to the 2x2 contrasts.

`observation` declares the process that decides which unit-periods are observed. `ignorable_given_cohort` (the default) says that being observed is unrelated to the unit's untreated outcome trend within its cohort. It cannot be tested from the observed rows, so it is recorded with the result as the untestable assumption `attrition_ignorable_given_cohort`. `informative` says attrition or late entry depends on outcomes or trends within cohort. With no bound nothing identifies the effect, so the route refuses and names the next attempt. `attrition_shift_bound=B` (with `observation="informative"`) declares that the treated units a cohort fails to observe change by at most `B` outcome units more or less than those it observes, comparison units being missing at random. The summary is linear in that shift, so the result reports `attrition_slope` (the summary's weights times the share of each cohort unobserved at each cell), the assumption-dependent range `assumption_range_lower` and `assumption_range_upper` (estimate minus and plus `B` times the slope) and `attrition_tipping_shift`, the shift at which the estimate crosses zero. The range is not a confidence interval and is separate from the resampling intervals, which are conditional on a zero shift. The bound is recorded as the untestable assumption `attrition_shift_within_declared_bound`. A design with no comparison unit observed in two periods, or a single treated cohort with no unit at its base period, is refused too.

Under Bayesian inference the rollout designs return a Bayesian-bootstrap posterior: flat-Dirichlet weights on whole independent units (clusters, never rows), refitting every cell, nuisance model and aggregation per draw, so draws are joint across cells. The result carries posterior means, equal-tailed credible intervals and simultaneous credible bands. A frequentist `resampling` declared beside it is refused.

## Synthetic control

`SyntheticControl` fits one treated unit against a donor pool, in a balanced panel by default or in an unbalanced panel over the treated unit's observed window.

```python
from antecedent import analyze
from antecedent.quasi import SyntheticControl

result = analyze(panel, query=SyntheticControl("outcome", "unit", "period", "treated", 5))
result.value.synthetic_control.estimate
```

Donor weights are nonnegative and sum to one, chosen by projected gradient descent on pre-period squared error. The effect is the treated unit's mean post-period outcome minus the weighted donors' mean. The design needs one treated unit, at least three donors, two or more pre-periods, an observed intervention period and identical periods for every unit.

When the panel is read over the observed window the result carries `observed_window` (`ObservedWindowSupport`): the pre- and post-periods read, `donors_used`, the excluded units with the window periods each misses, and `unread_rows`; the same typed report is sealed into the result artifact and checked against the frozen unit and period labels on load.

The result reports donor weights, pre-fit RMSE, the effective donor count and leave-one-donor-out placebo effects. The placebo tail fraction compares post/pre RMSE ratios, assumes exchangeable units and is neither a confidence interval nor a licensed test. Identification needs no anticipation, stable treatment after adoption, no concurrent treated-unit shock, no interference, and a convex combination of donors being a valid untreated counterfactual. Pre-fit RMSE has no acceptance threshold.

The prepared study freezes unit and period row order; refreshing with changed design labels is refused.

### Options

- **Unbalanced panels.** `observed_window_support=True` fits the weights on the treated unit's observed pre-periods and averages its observed post-periods. The donors are the units observed in every one of those periods; units missing part of the window are excluded and listed, with the unread unit-periods, in the `estimate.quasi.synthetic_control.observed_window_support` diagnostic. Nothing is imputed and a period only the donors have is never read. The design is refused with a next attempt when the treated unit has fewer than two observed pre-periods or no observed post-period, or when fewer than three donors are observed through the window. Whether a unit is observed being unrelated to its untreated outcome is an untestable assumption, recorded as `attrition_unrelated_to_untreated_outcomes`. It applies to the simplex and ridge-augmented point fit; unit randomization, conformal or staggered intervals and the Bayesian posterior read a balanced panel and are refused with it.
- **Randomization inference.** If the treated unit was drawn uniformly from the observed units before outcomes were seen, set `uniform_unit_randomization=True`. Antecedent refits every possible treated-unit assignment against all other units and reports the exact two-sided tail fraction as `randomization_p_value`: a Fisher sharp-null test under the declared assignment, with no interval for the effect. Enumeration is limited to 32 candidate units.
- **Ridge augmentation.** `augmentation_ridge` set to a finite positive penalty corrects the simplex gap with a donor-trained ridge model when the treated unit lies outside the donors' convex hull. The result reports `unadjusted_effect`, `outcome_model_correction` and the adjusted `estimate`; the extra assumption is that the donor outcome model transports to the treated unit. The penalty is yours to set and is not tuned from the panel. With uniform assignment the Fisher test refits weights and correction for every candidate. The displayed placebo rank still describes the unadjusted fit.
- **Conformal intervals.** `conformal=True` inverts a permutation test over time, with weights refit under each hypothesized effect, and reports one interval per post period plus a constant-effect interval. It needs at least 19 pre-periods for a 5% test; it is exact when residuals are exchangeable over time and approximate under weak dependence.
- **Several treated units.** `adoptions=[(unit, period), ...]` with `placebo_replicates` fits each adopting unit against never-treated donors at its own adoption period. The overall effect is the post-period-weighted average, with an interval from reassigning adoptions to never-treated units and refitting. Conformal and staggered adoptions are alternatives.
- **Bayesian.** With staggered adoptions, `bayesian.synthetic_control` averages the per-unit simplex posteriors. With the conformal declaration it reports a posterior predictive interval at each post period, which is a model-based statement about the counterfactual path and a different object from the permutation set.

### Synthetic difference-in-differences

`SyntheticDifferenceInDifferences` weights both donor units and pre-periods on the simplex, then computes the weighted difference-in-differences contrast. It needs a balanced panel with one treated unit, at least two donors, two pre-periods and one post-period, and reports `result.value.synthetic_did` including both weight vectors. It returns no interval or standard error, and refuses a fabricated one in a saved artifact. The randomization test option is refused here, and ridge augmentation cannot be combined with it.

## Boundaries

- Covariate adjustment at the unit level for synthetic control is not available; use the covariate-adjusted DiD or a rollout estimator.
- Intervals on every route above are pointwise. Read each result's calibration status before relying on coverage.
- If treatment turns on and off, units influence each other, or the panel is unbalanced where the estimator requires balance, the route refuses and names the next option. See [Refusals](../../concepts/refusals.md).

## Related

- [Regression discontinuity and kink](discontinuity-and-kink.md) for threshold rules.
- [Sequential regimes](sequential-regimes.md) and [temporal designs](../temporal-designs.md) when treatment varies over time within units.
- [Uncertainty](../../concepts/uncertainty.md) for how to read intervals whose calibration is unmeasured.
