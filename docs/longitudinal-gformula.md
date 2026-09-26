# Sequential g-formula value: caller-supplied prediction path

`antecedent.regimes.evaluate_sequential_gformula` evaluates a static or
history-adaptive binary regime from caller-supplied period-specific conditional
reward predictions. The input matrix has one row per unique subject and one
column per treatment/reward period. Each prediction must already represent the
conditional mean reward under the regime action for that history, including
the caller's downstream continuation. The estimator sums each subject's
period predictions and averages those sums over subjects.

```python
from antecedent.regimes import evaluate_sequential_gformula

value = evaluate_sequential_gformula(
    period_outcome_predictions=q_predictions,  # subjects × periods
    treatment_history=treatment_history,
    regime=[False, True],  # or a callback(time, past_treatments, covariates)
    treatment_probabilities=propensity_by_subject_period,
    censoring_survival=censoring_survival_by_subject_period,
    subject_ids=subject_ids,
    fold_ids=fold_ids,
)
```

Subject IDs must be unique because the input is subject-level rather than
long-form; exactly one fold ID is retained for each subject. This preserves
fold ownership metadata. It does **not** fit outcome regressions, verify that
predictions are held out, or claim cross-fitting. Fold IDs are provenance only.

Treatment probabilities and censoring survival are checked against the
configured positivity floor, and the minimum regime-action and censoring
probabilities are returned. They are diagnostics only and are not multiplied
into the plug-in g-formula value. The caller must supply predictions that
handle the stated censoring/dropout model and justify sequential exchangeability,
consistency, treatment support, and valid conditional reward predictions.

The estimate is **point-only**, has no standard error or interval, and does not
add a support-matrix license. Supplied predictions and assumptions are not
validated by the library.
