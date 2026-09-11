# Temporal response observation pairs

**Suite path:** `conformance/response/temporal_observation`

Licensed 1.3 `ObservationSpec` × `ObservationAssumption` pairs riding temporal
`ResponseCurve` / `InterventionResponse` on `TemporalDag` (Frequentist `none`).
Each pair consumes this fixture, not `conformance/response/observation_primitives`.

The stationary linear SCM is

```text
T_s ~ Uniform(-1, 1)
Y_s = 5 + 2 T_{s-1} + 3 T_{s-2} + U_s
```

with `U_s ~ Uniform(-0.5, 0.5)` and template edges `T@lag1 -> Y@lag0`,
`T@lag2 -> Y@lag0`. Pulse at `-1`. Horizons `[1, 2]`. Therefore

```text
E[Y_0 | do(T_{-1}=d)] = 5 + 2d
E[Y_1 | do(T_{-1}=d)] = 5 + 3d
```

Dose-major surface on `[-0.5, 0, 0.5]` is `[4, 3.5, 5, 5, 6, 6.5]`.
`Set(0.5)` is the path `[6, 6.5]`.

| Pair | Assumption | Estimator |
|---|---|---|
| Selected | OutcomeIndependentGiven(T) | Cross-fitted logistic AIPW |
| RightCensored / LeftCensored | IndependentGiven([]) | Marginal KM IPCW |
| RightCensored / LeftCensored | IndependentGiven(T) | Cox IPCW, T lag-aligned at the policy offset |

A Complete query on the recorded proxy must miss the surface by more than
`naive_gap_min`. Conditional Cox must beat marginal KM on the informative-censoring
DGP. Interval/truncation and Bayesian non-Complete stay refused.

Delayed entry, cheap/full, and TemporalCpdag/Pag observation rides are unavailable.
Uncertainty is omitted (`none`), same as the static 1.3 contract.

## Expected summary

Top-level keys: `atol, dgp, fixture_id, grid, horizons, intervention_set_0_5, naive_gap_min, pairs, rows, seed, stream, surface, treatment_lag` (13 fields).
