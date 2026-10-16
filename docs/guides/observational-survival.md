# Observational survival effects at a declared horizon

`antecedent.observational_survival.survival_effect` (Rust:
`antecedent::analysis::observational_survival::ObservationalSurvivalEffect`) estimates the
effect of an **observed** binary treatment on a right-censored event time at a horizon the
caller declares. The randomized two-arm route in [Survival outcomes](../survival-outcomes.md)
needs randomization; this one needs a declared adjustment set and a declared censoring
assumption.

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

## Estimands

For potential event times `T(a)`:

- the survival probability difference `S_1(t*) - S_0(t*)`, with `S_a(t) = P(T(a) > t)`;
- the restricted mean survival time difference `R_1(t*) - R_0(t*)`, with
  `R_a(t*) = E[min(T(a), t*)]`, the area under `S_a` through `t*`.

The horizon `t*` is required: the survival curve is not an effect until a time is named. A
missing horizon refuses `required_option_missing` (`survival_effect.horizon_missing`).

## Assumptions

| Assumption | Where it is owned |
|---|---|
| Conditional exchangeability given the declared `adjust` columns | The caller. No graph is identified or checked. |
| Positivity of both treatment levels | Checked: a propensity below `min_propensity` (default 0.01) at any row refuses. |
| Consistency, no interference | The caller. |
| Censoring is ignorable (below) | The caller declares it; it cannot be tested from the data. |
| Censoring positivity | Checked: a probability of remaining uncensored through the horizon below `min_censoring_survival` (default 0.05) in an arm refuses. |
| Nuisance model correctness | Double robustness: consistent if the outcome model is right, or if the propensity and censoring models are. |

`censoring` has no default and is one of:

- `independent_within_arm`: censoring is independent of the event time and of the covariates
  within each arm. The censoring survival is one Nelson-Aalen curve per arm.
- `at_random_given_covariates`: coarsening at random. Given the arm and the declared
  covariates the censoring hazard does not depend on the event time. The censoring survival is
  a Cox model on the declared covariates within each arm.

Informative censoring (a hazard that depends on the unobserved event time) is not identified
and is not offered. An absent assumption refuses `required_option_missing`
(`survival_effect.censoring_assumption_missing`).

## Estimator

Cross-fitted augmented inverse-probability and censoring weighting. For each fold a
multinomial-logit propensity `e_a(Z)`, a Cox event model with a Breslow baseline per arm and a
Cox censoring model per arm are fitted on the other folds. For arm `a` the score of a unit is

```text
psi_S = S_a(t*|Z) - 1{A = a}/e_a(Z) * S_a(t*|Z) * integral_0^{t*} dM(u) / (S_a(u|Z) G_a(u-|Z))
psi_R = R_a(Z)    - 1{A = a}/e_a(Z) * integral_0^{t*} [integral_u^{t*} S_a(s|Z) ds] dM(u) / (S_a(u|Z) G_a(u-|Z))
```

with `G_a` the censoring survival and `M` the event counting-process martingale. The estimate
is the mean score, and its standard error the standard deviation of the scores over the square
root of the rows. The scores are returned (`row_index`, `*_scores`) so a consumer can replay
the estimate and its standard error.

## Refusals

| Reason code | Detail | When |
|---|---|---|
| `required_option_missing` | `survival_effect.horizon_missing` | No horizon declared. |
| `required_option_missing` | `survival_effect.censoring_assumption_missing` | No censoring assumption declared. |
| `invalid_argument` | `survival_effect.censoring_assumption_unknown` | An assumption other than the two above. |
| `invalid_argument` | `survival_effect.censoring_covariates_missing` | At-random censoring with no declared covariates. |
| `positivity_violated` | `survival_effect.horizon_beyond_follow_up` | No unit of an arm is followed to the horizon. |
| `positivity_violated` | `survival_effect.treatment_positivity` | A treatment propensity falls below the floor. |
| `positivity_violated` | `survival_effect.censoring_support` | The probability of remaining uncensored through the horizon falls below the floor. |
| `treatment_level_undeclared` | `survival_effect.undeclared_level` | A row at neither the active nor the control level. |
| `arm_not_populated` | `survival_effect.arm_not_populated`, `survival_effect.training_fold_without_events` | An arm has no row, or a training fold has no event in it. |
| `invalid_argument` | `survival_effect.invalid_*` | Malformed levels, horizon, folds, floors, columns or event codes. |

## What is not claimed

The standard errors are influence-function standard errors, labelled `calibration ==
"unmeasured"`. No interval is produced and no coverage claim is made until a coverage record
exists: the repeated-sampling designs are written as ignored calibration tests and are
measured at the end of the release. A `NaN` is a missing cell; its row is dropped from the
complete cases and counted in `rows_dropped_missing`. Delayed entry, competing risks and
time-varying covariates are not covered by this route.

Verification against known truth: exponential event times with a proportional-hazards
treatment effect and six confounder cells have closed-form survival and restricted mean
survival, which the tests compare to (`observational_survival_effects.rs`,
`python/tests/test_observational_survival.py`). The same tests show the unadjusted
Kaplan-Meier contrast is biased under the same confounding.
