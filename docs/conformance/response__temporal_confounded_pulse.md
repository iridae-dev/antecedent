# Confounded temporal pulse (identification + estimand + estimate)

**Suite path:** `conformance/response/temporal_confounded_pulse`

The stationary linear SCM is

```text
Z_s = V_s
A_s = Z_s + U_s
Y_s = 1 + 2 A_{s-1} + 5 Z_{s-1}
```

with template edges `Z@lag0 -> A@lag0`, `Z@lag1 -> Y@lag0`, and `A@lag1 -> Y@lag0`.
`V` and `U` are the period-four exogenous sequences `[1,1,-1,-1]` and `[1,-1,1,-1]`,
so `A_s = [2,0,0,-2]`. Treatment is noisy given `Z`; the outcome is exact given
`(A, Z)`. `n = 241` makes the lag-aligned `Z` window an integer number of periods,
so `E[Z]=0` in that window.

The policy is `Pulse { at: -1 }` at horizon 1. Therefore

```text
E[Y_0 | do(A_{-1}=d)] = 1 + 2d
```

This fixture pins three things, not the number alone:

1. **Adjustment set.** Temporal backdoor identification returns `{Z@-1}`. Empty
   `Z` with method `temporal.backdoor.unfolded` is the schedule-ID relabel bug
   (general ID, then a backdoor label, then unadjusted OLS).
2. **Identified estimand.** Status `NonparametricallyIdentified`, method
   `temporal.backdoor.unfolded`, treatment `t@-1`, outcome `y@0`. Pulse and
   single-step Sustained share that estimand.
3. **Estimate.** Doses `[0, 1]` give the surface `[1, 3]`. Pulse / single-step
   Sustained recover the two-point contrast `2`.

The same estimator on the subgraph that omits `Z` is still
`temporal.backdoor.unfolded`, but with empty `Z`, and returns `[1, 5.5]`
(slope 4.5 / contrast 4.5): the confounded association, not the interventional
response. That negative control is why an unconfounded DGP cannot be the only
scientific line of defence.

This fixture does not pin bands, multi-horizon surfaces, or nonlinear response.
Horizon is 1 only: identify-once at `max(horizons)` can drop `Z` when
confounding does not reach a longer outcome.

## Expected summary

Top-level keys: `claim, contract, estimator_contract, fixture_id, generation, tolerance` (6 fields).
