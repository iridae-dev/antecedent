# Sequential doubly robust regime value

`antecedent.regimes.evaluate_sequential_doubly_robust` computes a backward
recursive augmented score for a static or history-adaptive binary regime.
The caller supplies terminal outcomes, observation history, sequential
treatment and censoring probabilities, and one cross-fitted conditional Q
prediction per subject and period. For each subject, starting with terminal
outcome `Y`, the native recursion is

```text
V[t] = Q[t] + I(observed through t and A[t] == regime[t])
                 * (V[t+1] - Q[t]) / (P(A[t] == regime[t]) * G[t])
```

and the reported value is the subject mean of `V[0]`. `G[t]` must be the
conditional probability of remaining observed from the preceding period
through period `t`, not cumulative survival. Once a subject is censored,
`observation_history` must stay false at later periods. A missing terminal
outcome is ignored when the final observation-history flag is false.

The Python API checks that subject IDs are unique, that each subject has one
fold ID, and that each supplied Q prediction's fold ID matches the subject's
fold. This preserves subject-level fold ownership. These checks do not verify
that the Q models were actually trained without their held-out subjects.
Treatment and conditional censoring probabilities must satisfy the declared
positivity floor. The score uses their inverse values in its augmentation.

The result is point-only, with no interval, calibration, support-matrix
license, or nuisance-model training. The caller remains responsible for
cross-fitting Q predictions, supplying valid conditional Q means and
probabilities, and defending consistency, sequential exchangeability,
sequential positivity, and the conditions required for the sequential
doubly-robust argument.
