# Static response observation contract (1.3)

This table is part of the licensed `ResponseCurve × Dag × explicit/accepted ×
Frequentist × none` cell in the [support matrix](support-matrix.md).
Observation is an additional contract, not another matrix axis. Every mechanism
requires exactly one explicit compatible assumption. Available columns do not
imply independence.

| ObservationSpec | ObservationAssumption | Estimation | Consuming fixture |
|---|---|---|---|
| Complete | None | Complete-data response | Existing response conformance |
| Selected | OutcomeIndependentGiven(Z), including treatment and every causal adjustment variable | Cross-fitted logistic observation propensity and linear outcome AIPW | `conformance/response/observation_pairs/selected.json` |
| RightCensored | IndependentGiven([]) | Marginal Kaplan–Meier IPCW | `conformance/response/observation_pairs/right_km.json` |
| LeftCensored | IndependentGiven([]) | Marginal KM IPCW after sign reversal | `conformance/response/observation_pairs/left_km.json` |
| RightCensored | IndependentGiven(Z), nonempty and including treatment and every causal adjustment variable | Cox IPCW | `conformance/response/conditional_ipcw/response_truth.json`, right test |
| LeftCensored | IndependentGiven(Z), same requirement | Cox IPCW after sign reversal | Same conditional fixture, left test |

The historical `OutcomeIndependentGiven([])` marginal censoring alias remains
accepted. Selected IPW remains an explicit lower-level correction option; AIPW
is the default licensed response primitive. Its double robustness requires a
correct observation-propensity or outcome nuisance model, as well as positivity.

Containment of treatment and the causal adjustment set in a nonempty
`IndependentGiven` claim is a composition requirement for the downstream
response regression, not a test that independent censoring holds. The declared
covariates are the Cox model. Conditional censoring assumes a proportional-hazards
nuisance model with log-linear effects of those completely observed numeric
covariates. The fit uses Breslow ties and baseline cumulative hazard; each observed row receives
`1 / G(T-|Z)`, with zero weight on censored rows. The curve is fit to the
Horvitz–Thompson transform `Y* = Y · W`; the zeros stay in the sample and are
offset by the upweighted observed rows. Dropping them would be complete-case
analysis. Left censoring reverses the
recorded outcome and censoring time before this computation. Neither weights nor
hazards are silently clipped: observed-row survival below the declared floor
refuses. Existing floor diagnostics remain available. Singular or nonconvergent
Cox fits refuse, with no marginal fallback.

The oracle is executing R `survival` 3.8.6: `coxph(..., ties="breslow")` and
`basehaz(centered=FALSE)`, evaluated strictly before each observed time.
The [Cox API](https://www.stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html)
and [survival-curve API](https://www.stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.coxph.html)
document the reference conventions. Regenerate from the repository root:

```sh
Rscript conformance/response/conditional_ipcw/generate.R
python3 conformance/response/conditional_ipcw/freeze.py
cargo test -p antecedent-stats cox_ipcw
cargo test -p antecedent --test conditional_ipcw
```

Marginal delayed entry retains its existing contract. Conditional delayed entry,
interval-censored/truncated response MLE, and joint observation/curve intervals
remain unavailable. Observation-adjusted results omit uncertainty rather than
reuse complete-data intervals. These cells do not extend to derivatives,
Bayesian responses, temporal responses, or partial graphs.
