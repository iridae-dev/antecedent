# Evaluate a treatment regime over several periods

Estimate the mean reward you would get if every subject followed a treatment regime over several periods: a fixed schedule ("treat in period 2 only") or a history-adaptive rule ("treat if the last reward was low"). You supply the conditional predictions and the assignment and censoring probabilities from your own models; Antecedent combines them into a regime value and reports the positivity diagnostics.

Two estimators are available in `antecedent.regimes`. The sequential g-formula sums period-specific conditional reward predictions. The sequential doubly robust estimator adds a backward-recursive correction that uses assignment and censoring probabilities as well. Both read subject-level matrices with one row per unique subject and one column per period.

If you want Antecedent to fit the treatment and outcome models, see [Sequentially randomized designs and regime survival](../temporal-designs.md): its observational regime contrast fits them and puts the fits inside the interval.

## Sequential g-formula value

```python
from antecedent.regimes import evaluate_sequential_gformula

value = evaluate_sequential_gformula(
    period_outcome_predictions=q_predictions,  # subjects x periods
    treatment_history=treatment_history,
    regime=[False, True],  # or a callback(time, past_treatments, covariates)
    treatment_probabilities=propensity_by_subject_period,
    censoring_survival=censoring_survival_by_subject_period,
    subject_ids=subject_ids,
    fold_ids=fold_ids,
)
```

Each prediction must already be the conditional mean reward under the regime's action for that history, including the continuation you want counted downstream. The estimate sums each subject's period predictions and averages those sums.

Treatment probabilities and censoring survival are checked against the positivity floor, and the minimum regime-action and censoring probabilities are returned as diagnostics. They are not multiplied into the plug-in value. You are responsible for predictions that handle the censoring or dropout model.

## Sequential doubly robust value

```python
from antecedent.regimes import evaluate_sequential_doubly_robust
```

You supply terminal outcomes, observation history, sequential treatment and censoring probabilities, and one cross-fitted conditional prediction `Q` per subject and period. For each subject the recursion starts at the terminal outcome and runs backward:

```text
V[t] = Q[t] + I(observed through t and A[t] == regime[t])
                 * (V[t+1] - Q[t]) / (P(A[t] == regime[t]) * G[t])
```

The reported value is the subject mean of `V[0]`. `G[t]` is the conditional probability of remaining observed from the previous period through period `t`, not cumulative survival. Once a subject is censored, `observation_history` stays false at later periods, and a missing terminal outcome is ignored when the final observation flag is false. The augmentation divides by the assignment and censoring probabilities, so probabilities near zero inflate variance.

## What is checked and what you own

| Checked by Antecedent | Declared by you |
| --- | --- |
| Subject IDs are unique and each subject has one fold ID | That predictions were cross-fitted: fold IDs are recorded as provenance, and Antecedent does not check that a model was trained without its held-out subjects |
| Each prediction's fold matches its subject's fold | Sequential exchangeability, sequential positivity and consistency |
| Treatment and censoring probabilities meet the positivity floor | That the supplied predictions and probabilities are valid for the regime and the censoring model |

Both results are point-only. They carry no standard error or interval, no support-matrix licence, and no nuisance-model training.

## Next step

If you need an interval, move to the observational regime contrast in [temporal designs](../temporal-designs.md), whose unit bootstrap refits the nuisance models in every replicate. See [Uncertainty](../../concepts/uncertainty.md) for how to read the calibration status on that interval.

## Related

- [Sequentially randomized designs and regime survival](../temporal-designs.md)
- [Observational policy value](../heterogeneous-effects-and-policies.md#observational-policy-value) for static, single-decision policies.
