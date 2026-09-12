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
is the default licensed response primitive. Unselected AIPW rows receive the
outcome-regression prediction `m(X)`, not a Horvitz–Thompson zero. Its double
robustness requires a correct observation-propensity or outcome nuisance model,
as well as positivity.

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
Bayesian responses, or partial graphs.

# Temporal response observation contract (1.6)

The same pairs ride Frequentist `TemporalDag` `ResponseCurve` /
`InterventionResponse` at validation `none`. Each pair consumes
`conformance/response/temporal_observation`, not
`conformance/response/observation_primitives`.

| ObservationSpec | ObservationAssumption | Estimation | Consuming fixture |
|---|---|---|---|
| Complete | None | Existing temporal g-comp | `conformance/response/temporal_dose_horizon` |
| Selected | OutcomeIndependentGiven(Z), including treatment and every causal adjustment process | Cross-fitted logistic AIPW on the lag-aligned series | `conformance/response/temporal_observation` |
| RightCensored | IndependentGiven([]) | Marginal KM IPCW | same |
| LeftCensored | IndependentGiven([]) | Marginal KM IPCW after sign reversal | same |
| RightCensored | IndependentGiven(Z), nonempty and including treatment and every causal adjustment process | Cox IPCW, Z lag-aligned at the policy treatment offset | same |
| LeftCensored | IndependentGiven(Z), same requirement | Cox IPCW after sign reversal, same lag alignment | same |

The historical empty `OutcomeIndependentGiven([])` marginal-censoring alias
remains accepted. Containment uses contemporaneous process ids (unfolded
adjustment nodes map back through the temporal indexer). Nonempty Z is never
the contemporaneous column: the Cox/AIPW design uses Z at
`TemporalResponseSpec::treatment_offset()` (typically −1).

Until a pair is licensed, `compile_logical_temporal_response` refuses
non-`Complete` with `temporal response observation pair is not licensed`.
Delayed entry, interval/truncation, cheap/full, and `TemporalCpdag` / `TemporalPag` observation rides stay refused. Bayesian temporal response uses the separate observed-data likelihood contract below.
When bootstrap replicates are requested, observation-adjusted temporal
surfaces — including Sequence overlays — report pointwise outer circular-block
bootstrap intervals. Every replicate resamples the original series, refits the
selected/KM/Cox observation nuisance, reconstructs the pseudo-outcome, and
refits every horizon or sequential overlay. Complete-data analytic bands and
Bayesian IPCW/KM/Cox bands are not substituted.


### Bayesian temporal observed-data likelihood

The same five selected/right-/left-censored temporal pairs support Gaussian
SEM latent-trajectory Gibbs sampling, including multi-step Sequence. This
route integrates missing and censored outcomes through the observed-data
likelihood; it does not use fitted IPCW weights as a Bayesian likelihood.

Its declared ignorability assumption concerns the entire selection-indicator
trajectory or censoring-bound trajectory: conditional on fully observed Z,
that trajectory is independent of the latent outcome trajectory. Censoring
event indicators are deterministic comparisons of outcomes with bounds and
are not assumed independent of the outcome. Distinct observation/outcome
priors permit the ignorable observation factor to be omitted. Intervals are
pointwise posterior bands under the Gaussian mechanism model and this
trajectory assumption, not a guarantee of repeated-sampling coverage.

Consuming evidence: `crates/antecedent/tests/temporal_observed_bayesian.rs`;
backend provenance: `provenance/estimate.temporal_observed_bayes.toml`.
