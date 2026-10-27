# Estimate a treatment effect on time to an event

Compare event times between a treated and a control group when follow-up ends early for some subjects (right-censoring). You get arm-specific Kaplan-Meier curves, the restricted mean survival time (RMST) up to a horizon you choose, and the survival probability at that horizon. The same guide covers delayed entry, competing events, and the case where treatment was observed rather than randomized.

Name the horizon `tau` first. A survival curve is not an effect until a time is named, so every route here asks for one.

## Randomized two-arm studies

With individual randomization, the `prepare` / `analyze` flow reports the curves and contrasts on `result.value.survival`. Declare how censoring relates to the data through an observation assumption: `IndependentGiven(())` states that censoring is marginally independent of the event time within each arm.

```python
import antecedent
from antecedent.observation import IndependentGiven

query = antecedent.survival.SurvivalOutcome(
    duration="follow_up_days",
    event_observed="event",
    treatment="treated",
    tau=180,
    randomized=True,
    observation_assumption=IndependentGiven(()),
)
result = antecedent.analyze(data, query=query)
result.value.survival.rmst_difference
result.value.survival.times, result.value.survival.control_survival, result.value.survival.treated_survival
```

A row with `event_observed=False` is right-censored: it stays in the risk set through its recorded duration and does not count as an event. Both arms need observed follow-up through `tau`. Antecedent reports, and cannot verify, the assumptions of individual random assignment, independent right censoring within each arm, consistency and no interference. There is no scalar average treatment effect: the contrasts are the RMST difference and the survival difference at `tau`.

`examples/python/survival_rmst.py` is the maintained executing example.

### Intervals and a simultaneous band

The default result is point-only. A bootstrap gives pointwise 95% percentile intervals for the RMST contrast and for the survival contrast at `tau`:

```python
result = antecedent.analyze(
    data, query=query,
    inference=antecedent.Frequentist(bootstrap=399),
    execution=antecedent.Execution(seed=17),
)
result.value.survival.rmst_difference_interval
result.value.survival.survival_at_tau_difference_interval
result.value.survival.bootstrap_replicates_ok
```

The bootstrap resamples whole subjects within each arm and recomputes the full risk-set estimator. It needs at least eight subjects per arm and 199 to 100,000 requested replicates, and at least 90% of replicates must satisfy the estimator's support contract. If that gate fails the interval is refused rather than silently thinned. The two intervals cover the two scalar contrasts separately. Read `result.value.survival.uncertainty` for the method and valid replicate count, and note that the interval is available but unlicensed.

With at least 80 subjects per arm and at least 399 valid replicates, the same call gives a simultaneous 95% band over the whole survival-difference curve:

```python
band = result.value.survival.difference_band
band.times, band.difference, band.lower, band.upper
```

The band uses one critical radius from the bootstrap distribution of the maximum curve deviation, and its grid, endpoints and accepted replicate count are frozen in the artifact. With fewer than 80 subjects per arm, delayed entry or caller-supplied censoring weights, `band_unavailable_reason` names why, and the scalar intervals remain available where eligible. The band is distinct from the pointwise intervals.

## Censoring that depends on covariates

If censoring depends on baseline covariates, supply the probability that each subject is still uncensored at each time (the censoring survival `G`) and the estimator weights by its inverse. You must fit and check `G`; Antecedent does not estimate censoring models or verify where the probabilities came from.

```python
weighted = antecedent.survival.estimate_survival_ipcw(
    data,
    query,
    times=[0.0, 30.0, 60.0, 180.0],
    censoring_survival=subject_survival_of_censoring,
    minimum_probability=0.01,
)
weighted.rmst_difference
weighted.censoring_survival_provenance
```

The grid must be strictly increasing from zero through `tau` and include every recorded event and censoring time. Each matrix entry is a subject's probability of remaining uncensored immediately before that time. The estimator uses inverse-`G` weighted event and risk counts at each event time and integrates the right-continuous curve for RMST. It refuses probabilities outside (0, 1], below the declared floor, non-monotone within a row, or implying weighted event counts above the risk set. The minimum `G` and event-time risk-set sizes are diagnostics, not inference.

The same kernel runs through `prepare` / `analyze`. Put the probabilities in data columns, one per time point, and name the conditioning variables in the observation assumption:

```python
query = antecedent.survival.SurvivalOutcome(
    "follow_up_days", "event", "treated", tau=180, randomized=True,
    observation_assumption=IndependentGiven(("baseline_risk",)),
    known_censoring=antecedent.survival.KnownCensoringSurvival(
        times=(0, 30, 60, 180),
        columns=("g_0", "g_30", "g_60", "g_180"),
        minimum_probability=0.01,
    ),
)
result = antecedent.analyze(data, query=query)
```

The artifact freezes the grid, column identities, floor and conditioning variables; a rebound study reads the columns from the new table in the same row order. With `Frequentist(bootstrap=399)` the pointwise intervals keep each subject's probabilities with that subject in every resample. The probabilities are held fixed and never refit, which the result labels `caller_supplied_fixed_not_fitted_or_verified`, so the intervals leave out uncertainty from fitting `G`. Assumptions include correct supplied probabilities, censoring positivity, independent censoring given the supplied history, random assignment, consistency and no interference. This path has no simultaneous band.

## Delayed entry

A study that enrols people after the clock starts (left truncation) needs a `delayed_entry` column. The supported claim is marginal: `IndependentGiven(())` asserts that entry and right censoring are independent of the event time, and Antecedent does not verify it.

```python
query = antecedent.survival.SurvivalOutcome(
    duration="follow_up_days",
    event_observed="event",
    treatment="treated",
    tau=180,
    randomized=True,
    delayed_entry="entry_day",
    observation_assumption=antecedent.observation.IndependentGiven(()),
)
summary = antecedent.analyze(data, query=query).survival
```

Risk sets follow counting-process intervals `(entry, duration]`: a subject entering exactly at an event time does not join that time's risk set, and a censored subject stays through the censoring time. Negative or non-finite entry times, and entry times not strictly earlier than the event or censoring time, are refused. Because RMST starts at time zero, each arm must include at least one time-zero entrant and observed follow-up through `tau`; otherwise the estimate would extrapolate from a left-truncated sample as if it were the RMST from the origin. Conditional delayed entry and interval censoring are refused. Delayed entry combines with known censoring survival under `IndependentGiven(())`; the bootstrap keeps each subject's entry, exit, cause and censoring row together, and the delayed-entry curves have no simultaneous band.

## Competing events

When subjects can fail from different causes, use a cause-code column: zero is right censoring and each positive integer is one event type. Choose one target cause. Other causes are competing events that remove a subject from the event-free risk set; they are never treated as censoring.

```python
query = antecedent.survival.CompetingRisksOutcome(
    duration="follow_up_days",
    event_cause="event_code",  # 0 = censored; 1, 2, ... = distinct event causes
    treatment="treated",
    target_cause=1,
    tau=180,
    randomized=True,
    observation_assumption=antecedent.observation.IndependentGiven(()),
)
cif = antecedent.analyze(data, query=query).survival
cif.incidence_difference
```

The target cumulative incidence uses the Aalen-Johansen risk-set recursion. At least two distinct causes must appear, the target must be present and both arms need follow-up through `tau`. The route is point-only by default; with `bootstrap` it gives the pointwise incidence-difference interval and, for the unweighted route, the simultaneous difference band under the same 80-subject, 399-draw gate. Assumptions: random assignment, complete and distinct cause coding, independent censoring within each arm, consistency and no interference. Cause misclassification and covariate-adjusted censoring are unsupported.

With supplied `G`, `estimate_cumulative_incidence_ipcw(data, competing_query, times=..., censoring_survival=...)` weights target-cause failures and all-cause event-free survival by inverse `G` inside the same recursion. It reports the minimum supplied `G` and event-time risk-set sizes. Fixed known-`G` weights and delayed entry have no simultaneous band.

### Declare the hazard partition for a posterior

For a Bayesian answer, `design_models.PiecewiseHazards` states the model: hazards are constant on each interval of a partition and independent a priori, with a Gamma prior. `cuts` is the partition (its last cut is the restriction horizon and must equal `tau`), `prior_shape` is the prior event count and `prior_rate` the prior time at risk, in the unit of the durations. The partition and prior are part of the answer. The twelve rows below are a small hand-checkable fixture.

```python
import antecedent
from antecedent.design_models import PiecewiseHazards
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success
from antecedent.survival import SurvivalOutcome

data = {
    "duration": [0.5, 1.2, 2.5, 3.5, 1.8, 0.9, 0.7, 1.5, 2.2, 3.2, 3.8, 1.1],
    "event": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    "treatment": [0.0] * 6 + [1.0] * 6,
}
query = SurvivalOutcome(
    "duration", "event", "treatment", 3.0,
    randomized=True, observation_assumption=IndependentGiven(()),
)
inference = antecedent.Bayesian(
    backend="conjugate",
    n_draws=40_000,
    design_model=PiecewiseHazards(cuts=(1.0, 2.0, 3.0), prior_shape=1.0, prior_rate=1.0),
)
result = require_success(
    antecedent.analyze(data, query=query, inference=inference, execution=antecedent.Execution(seed=11))
)
row = result.posterior_row
row.posterior_mean("survival_difference"), row.credible_interval("survival_difference")
```

The answer is on `result.posterior_row` (estimator `bayesian.piecewise_survival`), with coordinates `survival_difference` and `rmst_difference`; there is no `result.survival` and no scalar `ate`. The interval is a posterior statement under the declared partition and prior, not a calibrated interval: `row.calibration` is `"unmeasured"` and `row.licensed` is `False`. With a nearly flat prior and a fine partition the posterior tracks the product-limit estimate; with twelve rows the prior matters a great deal and the interval is wide. See [Declare a likelihood and prior](randomized-experiments.md#declare-a-likelihood-and-prior-for-a-bayesian-answer) for the other declarations.

## When treatment was observed, not randomized

For an observed binary treatment, `observational_survival.survival_effect` estimates the effect on a right-censored event time at a horizon you declare. It needs a declared adjustment set and a declared censoring assumption instead of randomization. Rust: `antecedent::analysis::observational_survival::ObservationalSurvivalEffect`.

```python
from antecedent import observational_survival

effect = observational_survival.survival_effect(
    duration, event, treatment,
    active=1.0, control=0.0,
    horizon=2.0,                          # t*
    censoring="at_random_given_covariates",
    adjust={"age": age, "stage": stage},
)
effect.survival_difference.estimate, effect.survival_difference.se
effect.rmst_difference.estimate, effect.rmst_difference.se
effect.assumptions        # the list the numbers rest on
```

The estimands are the survival difference `S_1(t*) - S_0(t*)`, with `S_a(t) = P(T(a) > t)`, and the RMST difference `R_1(t*) - R_0(t*)`, with `R_a(t*) = E[min(T(a), t*)]`.

`censoring` has no default; choose one:

- `independent_within_arm`: censoring is independent of the event time and the covariates within each arm. The censoring survival is one Nelson-Aalen curve per arm.
- `at_random_given_covariates`: given the arm and the declared covariates, the censoring hazard does not depend on the event time. The censoring survival is a Cox model on the declared covariates within each arm.

Informative censoring, where the hazard depends on the unobserved event time, is not identified and is not offered.

| Assumption | Who owns it |
| --- | --- |
| Conditional exchangeability given the declared `adjust` columns | You. No graph is identified or checked. |
| Consistency, no interference | You. |
| Censoring is ignorable as declared above | You; it cannot be tested from the data. |
| Positivity of both treatment levels | Checked: a propensity below `min_propensity` (default 0.01) at any row refuses. |
| Censoring positivity | Checked: a probability of remaining uncensored through the horizon below `min_censoring_survival` (default 0.05) in an arm refuses. |
| Nuisance models | The estimator is consistent if the outcome model is right, or if the propensity and censoring models are. |

The estimator is a cross-fitted augmented inverse-probability and censoring-weighted score. For each fold it fits a multinomial-logit propensity, a Cox event model with a Breslow baseline per arm, and a Cox censoring model per arm on the other folds. With `G_a` the censoring survival and `M` the event counting-process martingale, the score of a unit for arm `a` is

```text
psi_S = S_a(t*|Z) - 1{A = a}/e_a(Z) * S_a(t*|Z) * integral_0^{t*} dM(u) / (S_a(u|Z) G_a(u-|Z))
psi_R = R_a(Z)    - 1{A = a}/e_a(Z) * integral_0^{t*} [integral_u^{t*} S_a(s|Z) ds] dM(u) / (S_a(u|Z) G_a(u-|Z))
```

The estimate is the mean score and its standard error is the standard deviation of the scores over the square root of the row count. The scores are returned with `row_index`, so a consumer can replay the estimate and its standard error. A `NaN` is a missing cell: the row leaves the complete cases and is counted in `rows_dropped_missing`.

The standard errors are influence-function standard errors with `calibration == "unmeasured"`. No interval is produced until a coverage record exists. Delayed entry, competing risks and time-varying covariates are not covered by this route. Tests compare it to closed-form survival and RMST for exponential event times with a proportional-hazards effect and six confounder cells, and show that the unadjusted Kaplan-Meier contrast is biased under the same confounding (`crates/antecedent/tests/observational_survival_effects.rs`, `python/tests/test_observational_survival.py`).

### Refusals on the observational route

| Reason code | Detail | When |
| --- | --- | --- |
| `required_option_missing` | `survival_effect.horizon_missing` | No horizon declared |
| `required_option_missing` | `survival_effect.censoring_assumption_missing` | No censoring assumption declared |
| `invalid_argument` | `survival_effect.censoring_assumption_unknown` | An assumption other than the two above |
| `invalid_argument` | `survival_effect.censoring_covariates_missing` | At-random censoring with no declared covariates |
| `positivity_violated` | `survival_effect.horizon_beyond_follow_up` | No unit of an arm is followed to the horizon |
| `positivity_violated` | `survival_effect.treatment_positivity` | A propensity falls below the floor |
| `positivity_violated` | `survival_effect.censoring_support` | The probability of remaining uncensored through the horizon falls below the floor |
| `treatment_level_undeclared` | `survival_effect.undeclared_level` | A row at neither the active nor the control level |
| `arm_not_populated` | `survival_effect.arm_not_populated`, `survival_effect.training_fold_without_events` | An arm has no row, or a training fold has no event |
| `invalid_argument` | `survival_effect.invalid_*` | Malformed levels, horizon, folds, floors, columns or event codes |

## Related

- [Sequentially randomized designs and regime survival](../temporal-designs.md) for survival under whole treatment regimes over time.
- [Uncertainty](../../concepts/uncertainty.md) for the calibration status on each interval.
- [Analyze a randomized or factorial experiment](randomized-experiments.md) for non-survival outcomes.
