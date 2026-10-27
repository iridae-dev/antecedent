# Estimate responses, derivatives and outcome distributions

For retargeting, outcome distributions and non-additive joint interventions, see [Retarget, outcome distributions and joint interventions](#retarget-outcome-distributions-and-joint-interventions) below.

Antecedent treats a continuous causal response as more than a collection of
binary contrasts. The scalar Python constructors take the treatment first and the outcome second:

```python
query = antecedent.ResponseCurve(
    "dose",
    "outcome",
    grid=[0.5, 1.0, 1.5, 2.0],
)
result = antecedent.analyze(data, graph=dag, query=query)
```

The same complete-observation curve can use the staged workflow
when identification should be inspected before the data are estimated:

```python
identified = antecedent.identify(graph=dag, query=query)
print(identified.status, identified.adjustment_set)
result = identified.estimate(data)
```

The first positional name is the treatment and the second is the outcome.
Scientific and numerical options are keyword-only. The same convention applies
to `PointDerivative`, `Elasticity`, `SemiElasticity`, and
`AverageDerivative`. `DirectionalDerivative` and `ResponseJacobian` accept
treatment and outcome sequences in that order.

## Read the result on separate axes

A response result deliberately keeps four judgments separate:

1. `identification` says whether the causal functional follows from the graph
   and declared structural assumptions. On a temporal dose × horizon surface
   that argument can itself depend on horizon: `I(h)` is recorded on
   `identification.horizon_adjustment_sets` (and on the Rust
   `CausalResponse.horizon_identification`). `adjustment_set` is the first
   requested horizon. A union of those sets is not a valid shared `Z`.
2. `support` describes empirical overlap in the requested treatment region.
   An identified response may still be extrapolative or outside empirical
   support. On a static curve, `support.status` is the worst label over
   requested points. On a temporal dose × horizon surface it summarizes
   `support.point_status` over the same geometry as the mean: fully
   supported, partially extrapolative, or outside empirical support. A dose
   inside the union of lag-aligned treatment ranges can still sit outside a
   long-horizon window. Estimator-regime failures (heavy-tailed outcomes,
   floored conditional densities) are warnings and diagnostics on this
   object; they do not change `evidence_status`, which is the support-matrix
   cell.
3. `uncertainty.kind` says what the interval means. Pointwise intervals cover
   one grid point at a time. The Frequentist Kennedy-DR estimator can instead request a fixed-grid
   simultaneous band with `response_options(...)`; Python uses
   `estimator_config={"bandwidth": ..., "simultaneous_replicates": ...}`.
   The explicit bandwidth is required: Antecedent does not silently invent an
   undersmoothing rule. The band uses a deterministic Rademacher multiplier
   construction conditional on the cross-fitted pseudo-outcome. It does not
   refit nuisance models or claim unconditional coverage without the caller's
   bandwidth/regularity contract. An identified
   set is uncertainty about what the assumptions determine, not a confidence
   interval. Temporal dose × horizon surfaces keep the pointwise band in
   `uncertainty` and publish a simultaneous band next to it (see
   [Temporal simultaneous bands](#temporal-simultaneous-bands)).
   `uncertainty.interpretation` says what the level means: `"confidence"` is
   repeated-sampling coverage, `"credible"` is posterior probability under the
   model and prior (a credible interval's `standard_error` is a posterior
   standard deviation, and it makes no frequentist coverage claim). Bayesian
   inference publishes credible intervals; an interval built from influence
   scores or a bootstrap is a confidence interval even inside a Bayesian
   study. Coverage measurements are keyed on `inference`, so a `Bayesian`
   record measures a credible interval's frequentist coverage and a
   `Frequentist` record a confidence interval's.
4. `assumptions` and `provenance` record the claims and algorithm used. Do not
   infer an observation assumption merely from an observation-mechanism column.

Inspect warnings and diagnostics even when a numerical estimate is present:

```python
view = result.value                    # the specialized response view of the Outcome
print(view.identification)
print(view.identification.horizon_adjustment_sets)
print(view.support.status)
print(view.support.point_status)
print(view.support.warnings)
print(view.uncertainty.kind)
print(view.provenance)
print(view.evidence_status)
```

Every successful analysis has `evidence_status == "licensed"`.
A licensed cell can still carry
a scientifically untrustworthy number; read `support.warnings` before treating
the curve as a claim.

## Least-squares Kennedy-DR regularity

`response.kennedy_dr` fits additive cubic GAMs by least squares and smooths a
doubly robust pseudo-outcome with a local quadratic. That construction needs
finite residual moments. It does not inherit Kennedy et al.'s rates for
heavy-tailed outcomes, and the influence sandwich is not a Cauchy-resistant
interval.

Every Kennedy-DR mean-curve result reports `response.outcome_tail_ratio`:
`max |Y - median| / (1.4826 MAD)` of the retained outcome, followed by the
warning bound (20). Gaussian samples stay near sqrt(2 log n) (about 5 at a
million rows). Above the bound the result emits
`response.heavy_tailed_outcome`. A second diagnostic,
`response.pseudo_outcome_winsor_shift`, is the absolute movement of the
fitted level after clipping the pseudo-outcome at its 1st and 99th
percentiles; a large relative shift emits
`response.pseudo_outcome_tail_sensitivity`.

Neither diagnostic changes `support.status` or `evidence_status`. Overlap can
be honest while the outcome is outside the estimator's regularity. Transform
or winsorize before fitting if the warning fires; the library will not do
that silently.

Grid points are not silently trimmed to the sample range. A point outside the
observed treatment range remains visible and is labelled unsupported. This
makes the scientific request auditable, but it does not make the extrapolated
number credible.

## Compose discovery artifacts and response curves

Discovery remains outside the response estimate call: passing `discovery=` with
a response query is refused. Discover and review once, then reuse the accepted
artifact without rediscovery:

```python
accepted = antecedent.discovery.GES().accept(data)
result = antecedent.analyze(data, graph=accepted, query=query)
# Equivalent session-oriented spelling:
result = accepted.analyze(data, query)
```

When the accepted graph is a DAG, this is the same response identification and
estimation path as a hand-authored DAG. The artifact version does not change on
an estimate click.

`Cpdag` and `Pag` response queries are licensed through the same completion
envelope as ATE: `analyze()` and prepared studies mass-weight identified
completions and keep the runtime class. Each completion is identified by
(generalized) adjustment first. A single-treatment response on a completion
with no adjustment set then falls back to Shpitser–Pearl ID: on a `Cpdag`
completion, which is a DAG; on a `Pag` completion, on the MAG with every
invisible directed edge also read as latent-confounded, so an edge such as a
lone `T -> Y` that a latent common cause could explain is refused rather than
read as `P(y | t)`. That fallback is sound, not complete (the complete PAG
algorithm, IDP, is not implemented), and a completion it refuses keeps its mass
as unidentified, which makes the class answer graph-dependent. Influence
is mixed only when every completion is identified and every contributing
atom supplies an aligned IF; unidentified mass NaNs the envelope SE
rather than publishing a primary-atom interval. That is not
PAG-native response identification. `Admg` response, `TemporalCpdag`/`TemporalPag` response, and mixtures over
graph posteriors are licensed on the cells the
[support matrix](support-matrix.md) lists, each with its own evidence row;
everything not listed there is refused.

## Validation on a function-valued response

Response queries license validation `none` only. Leave the query's default
unchanged or pass `refute="none"`. Requesting `cheap`, `full`, or another scalar
refuter suite raises `CausalUnsupportedError`; these suite names do not denote
function-valued validation. The returned support diagnostics still describe
empirical treatment support. They do not establish exchangeability, positivity,
or model correctness.

## Bayesian responses

Estimate a Gaussian response posterior with the one-call workflow and retain its study:

```python
result = antecedent.analyze(
    data,
    graph=dag,
    query=query,
    inference=antecedent.Bayesian(backend="conjugate", n_draws=2048),
    refute="none",
)
study = result.study
report = result.inspect().to_dict()
response_bytes = result.export()
loaded = antecedent.load(bytes(response_bytes), expected=response_bytes.expectation)
assert loaded.acceptance.verified or loaded.acceptance.sealed
assert loaded.answer == result.answer
repeated = antecedent.estimate(study)
```

Explicit and accepted DAGs, TemporalDAGs, CPDAGs, and PAGs are licensed for
`ResponseCurve` and `InterventionResponse` under the documented Gaussian
additive forms. Static responses use `response.bayesian`; temporal responses
use `response.temporal.bayesian` and retain identification and support per
horizon. Class-aware Cpdag/Pag Bayesian response mixes identified-mass means
only; posterior draws are not mixed. The temporal posterior band in
`uncertainty` is pointwise; a simultaneous credible band is published next to
it (see [Temporal simultaneous bands](#temporal-simultaneous-bands)). Each
horizon is a separate conjugate fit, so across horizons that band describes the
product of per-horizon posteriors, not a joint horizon posterior. Both bands
condition on the observed covariate (and, for shifts, treatment) average. Each
horizon's Gaussian likelihood is tempered by its REML autoregressive variance
ratio `κ̂_h ≥ 1` (a generalized power posterior, the same correction as the Bayesian
Pulse / Sustained cells; support diagnostic
`response.temporal_bayesian.tempering`): lag-aligned `h`-step residuals are
MA(h−1) whenever the outcome or treatment is persistent, and serially dependent
at every horizon under autocorrelated innovations. Kennedy-DR regularity and row-influence
diagnostics below describe the Frequentist estimator, not these Bayesian
posteriors.

## Temporal simultaneous bands

A pointwise band covers one (dose, horizon) cell at a time. Temporal
`ResponseCurve` / `InterventionResponse` surfaces also publish a band that
covers the whole grid at once, as three support diagnostics:

| id | values |
|----|--------|
| `response.simultaneous_band.lower` | lower edge, same dose-major layout as the mean |
| `response.simultaneous_band.upper` | upper edge, same layout |
| `response.simultaneous_band.critical` | `[level, critical value c, joint replicates or draws]` |

The construction is a max-studentized deviation (sup-t) from joint replicates
of the whole surface, the replicate analogue of the Kennedy-DR multiplier band
above. With cell scales `s_j` (the SD of the replicates of cell `j`) and
`B` replicates, `c` is the `ceil(level·(B+1))`-th smallest of
`max_j |θ*_j − θ̂_j| / s_j`, and the band is `θ̂_j ± c·s_j` (the static
multiplier band and the Gaussian max-t critical value read the same rank). At
least 40 joint replicates are required; otherwise
`response.simultaneous_band_withheld` explains why no band was published. The
pointwise band needs only two surviving replicates, but below 40 it carries
`response.temporal.pointwise_band_few_replicates`: its SD is then itself noisy.

The critical value is floored at the one-cell normal quantile, so the
simultaneous band is never narrower than the pointwise band.

Every Frequentist temporal response band (curves, Set/Shift/Soft, Sequence
overlays, observation-adjusted surfaces, and TemporalCpdag/Pag completion atoms)
uses one construction:

1. a joint circular-block bootstrap of lag-aligned outcome-time tuples over the
   `n` series times every horizon can evaluate; every replicate row keeps its own
   intact lag window, and every horizon of a replicate is refit on its rows at the
   same resampled calendar times;
2. block length `ℓ = max(max(unfolded span, ceil(sqrt(n))), min(ceil(b_PW·n^{1/6}), n/3))`,
   with `b_PW` the largest Politis–White (2004) block length over the level's
   estimating scores (every fitted regression's normal-equation scores and the
   centered covariate columns whose averages the level reads) — the lengthening
   of the scalar temporal effects, on the `sqrt(n)` floor. The support diagnostic
   `response.temporal.block_length` records `[ℓ, floor, uncapped testing length,
   n, dispersion factor]`, and `response.temporal.block_length_capped` warns when
   the `n/3` cap binds;
3. replicate deviations from `θ̂` scaled by the fixed-b critical-value ratio of the
   circular Bartlett estimator at `b = ℓ/n` (simulated circular-Bartlett 97.5%
   quantiles; the correction the plain TemporalDag Pulse / Sustained SE uses) and by the HC1 factor
   `sqrt(n/(n − p))`;
4. each cell's deviations further scaled by that cell's kernel-bias factor
   `1/sqrt(f)` (support diagnostic `response.temporal.kernel_bias_factor`, in the
   mean layout): `f` is the share of the long-run variance of the autoregression
   fitted to the cell's influence series (a Kendall-corrected AR(1), or the
   BIC-selected AR(q ≤ 4) when it reads more) that the Bartlett kernel at `ℓ`
   keeps, `f = (1 + 2 Σ_{k<ℓ} (1 − k/ℓ) ρ_k) / (1 + 2 Σ_k ρ_k)`. The influence is
   the delta-method score of the level: the row's weight in `c'β̂` times its
   residual, plus the centered design columns whose sample means the level reads
   (every column but the treatment for a dose; the treatment too for a shift),
   each weighted by its coefficient. A Sequence level, which composes several
   mechanisms, adds the sample means of the exogenous root nodes it reads to its
   influence (`Σ_r ∂level/∂x̄_r · (x_r − x̄_r)`) and takes the largest factor over
   its estimating scores. A fit with no
   positive long-run excess keeps the factor at 1; on iid and AR(1) `ρ = 0.5`
   cells it stays within 1% of 1;
5. pointwise band `θ̂ ± 1.96·SE` with SE the scaled replicate SD, and the
   simultaneous band above from the same scaled replicates (a per-cell scale
   leaves the studentized maximum, hence the critical value, unchanged). The
   fixed-b ratio is the two-sided 95% correction for one pointwise interval;
   carrying it into the sup-t band is a heuristic extension, checked by the
   calibration rather than derived.

The block length is chosen for interval coverage, not for the mean-squared error
of the variance. The circular-block variance behaves like a Bartlett-kernel
long-run variance with bandwidth `ℓ`, and the testing-optimal Bartlett bandwidth
grows like `n^{1/2}` (Sun, Phillips & Jin 2008). With the `n^{1/3}` rule of the
scalar temporal effect resamplers, the kernel truncation bias left
observation-adjusted bands under AR(1) `ρ = 0.5` residuals under the band for
nominal 95% (that run was not retained), and neither a `t_ν/z` batch-means
factor nor longer blocks under that factor repaired it. The fixed-b factor carries the extra estimation noise of the longer
block, so independent rows are not over-covered.

The kernel-bias factor is what the block length cannot buy. The Bartlett kernel
misses `O(1/ℓ)` of a persistent influence's long-run variance at any licensed
length (46% kept at `ℓ = 13`, 82% at `ℓ = n/3` for an AR(1) ρ = 0.9 influence at
n = 160), and the fixed-b ratio corrects the estimation noise of the block
variance, not its truncation. Lengthening alone does not close the gap: on the
mean of an AR(1) ρ = 0.9 series at n = 160 the fixed-b interval covers 0.803 at
`ℓ = 13` and 0.903 at `ℓ = n/3`, while the fitted-AR(1) factor at `ℓ = 13` covers
0.954 (0.948 for ρ = 0.5, 0.956 for iid; 4000 series); and long blocks degrade the
sup-t band, which needs many blocks per replicate. The factor is model-based, so
it carries no wide-bandwidth sampling noise, and it corrects only what the
autoregression fits.

Each band also reads every cell's influence for the family's short-series
statistic (`response.temporal.effective_rows`: the smaller of the lag-1 AR(1)
reading `n(1 − r₁)/(1 + r₁)` and the block-length Bartlett reading, the readings
of the scalar temporal effects) and warns `response.temporal.block.short_series`
when any cell reads fewer than 15 effective rows.

- **Frequentist complete data.** Curves and single Set/Shift/Soft responses refit
  every horizon and recompute the covariate (and, for shifts, treatment) averages
  on each replicate. Multi-step and joint Sequence overlays refit every unfolded
  sequential mechanism of every horizon on the resampled tuples and recompute the
  root-node means. Both target the population level `E[Y_h | do(A)]`. With zero
  replicates no band is published: the analytic OLS band would treat lag-aligned
  rows as independent, which temporal rows are not, so the surface keeps its
  point values and warns `estimate.temporal_response.band_withheld`.
- **Frequentist observation-adjusted.** Each outer replicate additionally refits
  the observation nuisance on exactly the resampled tuples and replaces the
  outcome-time column with the replicate pseudo-outcome. Sequence overlays whose
  unfolded design carries the outcome at a nonzero lag are refused (the lagged
  outcome would be a pseudo-outcome regressor). The same guard covers curve and
  Set / Shift designs; there a lagged outcome in the adjustment set would need
  outcome ancestry that repeats at every earlier slice, which temporal backdoor
  identification does not certify, so it refuses first. No replicate row pairs values
  across a block junction. The raw-series replicate that an earlier release used for Sequence
  overlays measured 31–81% coverage for a nominal 95% band and is gone.
- **Bayesian.** The simultaneous credible band uses posterior draws of the
  grid around the posterior mean: joint Gibbs draws for
  `estimate.temporal_observed_bayes`, and index-paired independent per-horizon
  draws (a product posterior) for `response.temporal.bayesian`.
- **Not published.** Bayesian Sequence overlays (each horizon is sampled
  separately; pointwise bands only) and the class-level TemporalCpdag/Pag
  identified set (no class band; each completion atom keeps its own pointwise and
  simultaneous band, calibrated against that atom's own probability limit).

In Python, `Frequentist(bootstrap=...)` on `analyze(...)` and `prepare(...)` is
the replicate count for Frequentist temporal `ResponseCurve` /
`InterventionResponse` (TemporalDag surfaces and TemporalCpdag/Pag completion
atoms). An explicit count always wins. Omitted, it follows
`Frequentist(profile=...)` as prepared Pulse / Sustained do: `interactive` 0,
`standard` 199, `report` 200. Declare the count explicitly when you need a band.
`bootstrap=0` returns the point surface with `uncertainty.kind == "none"`, the
withheld-band message in `support.warnings`, and
`estimate.temporal_response.band_withheld: …` in `diagnostics`. The seed on
`Execution(seed=...)` drives the replicates, so a fixed seed reproduces the band.
The simultaneous band is `result.value.simultaneous_band` (`None` when none
was published), rebuilt from the support diagnostics above. Its `lower` / `upper`
rows match `uncertainty.lower` / `uncertainty.upper`; `critical` and `replicates`
come from `response.simultaneous_band.critical`:

```python
result = antecedent.analyze(
    data, graph=lagged_edges, query=surface,
    inference=antecedent.Frequentist(bootstrap=200),
    execution=antecedent.Execution(seed=7),
)
result.value.uncertainty.lower, result.value.uncertainty.upper   # pointwise 95% band
band = result.value.simultaneous_band               # whole-grid 95% band
band.lower, band.upper, band.critical, band.replicates
```

Bayesian responses (posterior intervals), static responses, and `cell.aipw`
refuse a positive `Frequentist(bootstrap=...)` rather than ignoring it.

Coverage of these bands was measured on linear-Gaussian simulations with iid and
AR(1) residuals (`crates/antecedent/tests/temporal_response_calibration.rs`,
400 replicates, `n = 160`). With iid and AR(1) `rho = 0.5` residuals on every
Frequentist cell, and with an AR(1) `phi = 0.9` treatment, the dose curve covers
at 0.945-0.948 pointwise and 0.948 simultaneous, and the shift response at 0.943
pointwise and 0.948 simultaneous. The shift level is essentially the sample mean
of a series with about ten effective rows, so its band also carries the
short-series warning on most replicates; at `n = 100` it covers 0.885 / 0.890
and warns on every replicate, at `n = 400` 0.938 / 0.940, and at `n = 1000`
0.965 / 0.968.

Strong residual persistence is outside the measured range and is disclosed on
every band as `response.temporal.block.persistence_boundary`. With AR(1)
`rho = 0.9` residuals the dose-by-horizon curve covers at 0.887-0.943 pointwise
and 0.877 simultaneous at `n = 160` (0.902-0.943 / 0.905 at `n = 1000`). The
persistent part of the residual is a small share of its variance yet carries
most of the level's long-run variance, and nothing in the sample reads it at
these sizes: the lag-by-lag autocorrelations stay under the block-length
threshold and no short-series warning is raised. The block-length rule catches
this component only once `n` is large enough for its autocorrelations to clear
the threshold (about `n = 2000`). Read a band under strongly persistent
residuals as under-covering.

Prepared responses require complete observations and the AllObserved empirical
population. Unsupported observation mechanisms, observation assumptions, or
population specifications are refused explicitly. Observation-adjusted
Frequentist curves use the separate path described under
[Observation is not outcome](#observation-is-not-outcome). Bayesian derivative
responses are described under
[Curves, derivatives, and elasticities](#curves-derivatives-and-elasticities).
Graph-posterior response mixtures are licensed only on the cells listed in the
[support matrix](support-matrix.md); multi-step temporal response policies are
refused here.

## Row-diagnostic export contract

`response_options={"export_row_diagnostics": True}` (Rust:
`ContinuousResponseOptions::export_row_diagnostics`) adds three channels to
`support.diagnostics`. Off by default; the estimate is identical either way.
Once exported, these are scientific objects users will compute on, so their
contract is stated exactly. Everything below is derived from the
implementation, not from the cited literature.

**Channels.** For `N` retained rows and a grid of `G` points:

- `response.row_index` — `N` values; the 0-based row position of each retained
  row (exact integers stored as `f64`).
- `response.row_pseudo_outcome` — `N` values; the cross-fitted Kennedy
  pseudo-outcome `phi_i`, in outcome units, aligned with `row_index`.
- `response.row_influence` — `G * N` values, **grid-major**:
  `value[g * N + i]` is row `i`'s influence at grid point `g`. `g` follows the
  same order as the returned curve's grid points; `i` aligns with the other two
  channels.

**What an influence value is.** At grid point `g` with bandwidth `h`, the curve
value `m(g)` is the level of a Gaussian-kernel local-quadratic weighted least
squares fit of the pseudo-outcomes: with `dx_i = a_i - g`,
`x_i = (1, dx_i, dx_i^2)`, `w_i = exp(-(dx_i/h)^2 / 2)`, and residual
`r_i = phi_i - x_i' beta(g)`,

```
psi[g, i] = w_i * [(X'WX)^{-1}]_{row 0} . x_i * r_i / sqrt(v_i) + (c_i - mean(c)) / N
v_i = 1 - 2*S_ii + sum_j S_ij^2; S = X (X'WX)^{-1} X'W
```

The first term is the WLS influence of row `i` on the fitted level, in outcome
units. The second is the marginalization term of Kennedy et al. (2017,
Theorem 3): the pseudo-outcome averages the outcome regression over every
row's covariates, `∫ mu(a, x) dP_n(x)`, so each row also moves the level through
its covariates. With the additive outcome nuisance this is the per-row
constant `c_i = mu(a_i, x_i) - ∫ mu(a_i, x) dP_n(x) = h(x_i) - mean h(x)` from
the row's own cross-fit fold, identical at every grid point. Without it the
pointwise band under-covered at nominal 0.90 on a confounded linear law
(`crates/antecedent/tests/static_calibration.rs`; that pre-fix run was not
retained). The values
still condition on the fitted nuisances and bandwidth: second-order nuisance
error and bandwidth selection are not inside them.

**Centering and scaling.** The raw WLS residual term and centered covariate term
sum to zero. Correcting each residual by its local smoother variance factor can
break exact centering; joint influence covariance centers its supplied scores.
There is no additional `1/n` scaling: these are per-row contributions.

**Exact relationship to reported uncertainty.** The reported pointwise standard
error is `SE(g) = sqrt(sum_i psi[g, i]^2)`. Its band is `m(g) ± t_df * SE(g)`:
`df = tr(A)^2 / tr(A^2)` matches the first two Gaussian moments of the
leverage-corrected local residual variance quadratic form. The per-grid df are
exported as `response.local_variance_degrees_of_freedom`. This is a conditional
Gaussian working approximation; it excludes nuisance and bandwidth selection
uncertainty. The simultaneous band's critical value uses the upper `level`
quantile of the Rademacher max statistic and is floored at the largest
pointwise Student critical, preserving nesting.
Both bands can be reconstructed from this export bit-for-bit; if your
reconstruction disagrees, that is a bug report, not a tolerance issue.

**Row indices and preprocessing.** Indices are positions in the data exactly as
the estimator received it — after any preprocessing your pipeline did, before
the one row filter Antecedent applies: rows with a non-finite outcome,
treatment, or adjustment value are dropped (at least 20 complete rows must
remain). Order is preserved. If you filtered or reordered rows before calling
`analyze`, the indices refer to your filtered frame, not your original one.

**Cross-fitting.** Fold assignment is deterministic: retained-row *position*
modulo `folds` (default 5) — not the original row index. `phi_i` is computed
from nuisances fit on the folds excluding row `i`'s. Two consequences for
downstream use: `phi` values on either side of a fold boundary come from
different nuisance fits, so finite-sample fold effects are real; and any
user-side resampling of rows breaks the sample-splitting these values were
constructed under — a bootstrap over `(phi_i)` is a bootstrap conditional on
the fitted nuisances, nothing stronger.

**Stability.** The three channel ids, their alignment, and the grid-major
layout are a documented contract and will not change silently; the values are
diagnostics conditional on the estimator's internal construction (nuisance
families, bandwidth rule, fold rule), so they may change when that construction
changes, without an artifact-format bump. They serialize into response
artifacts like any other support diagnostic. A non-finite influence is refused
at export rather than published.

## Curves, derivatives, and elasticities

The six derivative query types are licensed on explicit or accepted DAGs at
validation `none`, under Frequentist and Bayesian inference; see the
[support matrix](support-matrix.md), which carries each cell's evidence.
Derivatives on a `Cpdag`, `Pag`, latent-confounded `Admg` or `TemporalAdmg` (the
contemporaneous projection over rows, or the unrolled window of its template over
a series) execute through the compiler and stay unlicensed: each identified
completion, or the ADMG itself, is identified by one adjustment set valid for every
treatment/outcome pair; where no adjustment set exists the front-door criterion is
searched and the derivative is estimated with the outcome additive in the treatment, the
mediators and the covariates (an interval that refits both stages on each replicate); a
pair with a hedge is refused with its certificate, and a target that general ID identifies
through neither an adjustment nor a front door is refused. Completions of a class that
disagree publish their identified set with an Imbens-Manski band formed from their joint
row influence, and a TemporalAdmg derivative over a series is identified per horizon on the
unrolled window of its template and may be prepared as a handle.
Graph-posterior, observation-adjusted, and cheap/full coordinates remain refused.
Mean curves on `Cpdag` / `Pag` are a separate class-aware cell. The definitions below are the
licensed forms, not a license to run every constructor argument. Repeated-
sampling coverage of every published derivative interval is measured in
`crates/antecedent/tests/derivative_calibration.rs`.

- `ResponseCurve(treatment, outcome, grid=...)` evaluates
  `a -> E[Y | do(A=a)]` on an explicit, increasing grid.
- `PointDerivative(..., at=a)` estimates the true slope `m'(a)` of that curve,
  not a bandwidth-smoothed surrogate. It requires an explicit `bandwidth` in
  `estimator_config`; Silverman's rule is a level/KDE rate and is refused
  rather than silently oversmoothing `m'`. The point value is the
  Kennedy-DR local-quadratic slope. The interval is robust bias-corrected
  (Calonico–Cattaneo–Titiunik with the pilot bandwidth equal to the caller
  bandwidth, which is numerically the local-cubic slope and its own
  Eicker–White standard error). It is therefore centered at the bias-corrected
  slope rather than at the point value, and `standard_error` is the
  bias-corrected standard error; the diagnostic
  `response.derivative_interval_bias_corrected` says so on every result that
  publishes one. The conventional local-quadratic interval ignores the
  `O(h² m‴)` smoothing bias of the slope and covered about 82% at a nominal
  90% at an MSE-sized bandwidth on a curved response. For `order=2` the
  leading curvature bias of a local quadratic is `O(h² m⁗)`, which a local
  cubic does not reduce, so the second-derivative interval is centered at the
  local-quartic curvature with its own Eicker–White standard error. The
  Frequentist interval treats the cross-fitted pseudo-outcome as data and
  conditions on the caller bandwidth.
- `AverageDerivative(...)` averages a derivative over an explicit weighting
  law; only observed-law weighting is licensed, via the Gaussian-score Riesz
  representer. The representer assumes the treatment is homoskedastic Gaussian
  given the adjustment set (additive mean). That assumption is load-bearing:
  on a skewed, heteroskedastic treatment law with a nonlinear outcome the
  interval's measured coverage collapses (shown by
  `ade_skewed_heteroskedastic_treatment_probe`, which is not a licensed coordinate).
- `Elasticity(..., at=a)` is `a m'(a) / μ(a)`. The treatment point must be
  positive, and the fitted response at that point must be positive.
  Like `PointDerivative`, it requires an explicit bandwidth. The Frequentist
  result publishes a Fieller confidence interval from the full joint covariance
  of the bias-corrected local level and slope; its reported standard error is
  the corresponding delta-method approximation. A finite scalar interval is
  unavailable when the level is not statistically separated from zero, which
  is reported as `response.derivative_interval_unbounded`. The interval
  conditions on the caller bandwidth and fitted nuisances; its nominal 90% coverage was measured
  on the curved additive, Gaussian-treatment calibration design at sample sizes
  500, 1000, and 1500 for explicit and accepted DAGs.
- `SemiElasticity(..., at=a, log_scale=...)` is either `a m'`
  (`log_scale="treatment"`, the default) or `m'/μ` (`log_scale="outcome"`).
  Both ride the same matrix cell. Frequentist `log_scale="treatment"` publishes
  `|a|` times the bias-corrected slope interval. `log_scale="outcome"` and
  transformed second derivatives publish joint-covariance delta-method
  intervals, whose coverage is not calibrated for those scales.
- `ResponseJacobian(...)` and `DirectionalDerivative(...)` are additive-GAM
  plug-in gradients with at most two treatments, a common adjustment set, and
  a shared complete-case row set. The target surface is an unpenalized cubic
  regression spline on the nuisance basis: a roughness penalty shrinks the
  published gradient at first order (with quantile knots, even a linear
  component), so `nuisance_lambda` regularizes nuisances only. They are not
  doubly robust. A Jacobian is row-major outcomes × treatments. A directional
  query is the unnormalized inner product `∇m · d`. Differential missingness
  across outcomes is refused.

  The Frequentist result is point-only: `uncertainty` is empty and the warning
  `response.derivative_interval_withheld` says why. Each gradient coordinate
  is a linear functional of the target fit's coefficients, so a
  coefficient-sandwich confidence band exists. Its analytic variance uses an
  HC2 correction based on each complete row's leverage in the penalized spline
  fit; the Bayesian band below uses a separate average-leverage correction.
  Its 90% coverage is measured at three sample sizes in
  `derivative_calibration.rs`. The public route stays point-only while those
  records are reviewed: one directional coordinate narrowly missed an
  exploratory 8,000-repetition threshold. For an interval today, run the query
  with Bayesian inference. A Rust `Jacobian` on a transformed
  `DerivativeScale` (for example an elasticity Jacobian) has no interval
  construction at all, wired or published. The scalar `Elasticity` query is a
  different route (a Kennedy-DR point derivative) and does publish a calibrated
  Fieller interval.

Bayesian derivatives keep the Frequentist estimator identities and replace
the analytic interval with a Rubin Dirichlet(1,…,1) row-weight posterior in
which every draw refits its nuisances:

- `AverageDerivative` refits the additive-GAM outcome and the Gaussian
  treatment law under the draw's weights and evaluates the weighted Riesz
  score mean.
- `PointDerivative`, `Elasticity`, and `SemiElasticity` rebuild the
  cross-fitted Kennedy pseudo-outcome per draw (outcome and treatment
  nuisances refit on each weighted training fold), then evaluate the weighted
  local quadratic and its bias-corrected coordinate at the caller bandwidth
  (local-cubic slope; local-quartic level and curvature).
  The reported value is the posterior mean of the local-quadratic coordinate;
  the credible interval and `standard_error` come from the bias-corrected
  draws. Scale transforms are applied per draw. Held fixed across draws: the caller bandwidth, fold
  assignment, spline knots, and penalty.
- `ResponseJacobian` and `DirectionalDerivative` refit the unpenalized
  additive regression spline per draw with fixed knots and publish pointwise
  quantile bands. The band covers the derivative when the additive
  components lie in (or are well approximated by) that spline space; sieve
  approximation bias is not inside it.

Coefficient priors and prior transfer are refused for derivatives: no mapping
to the Riesz / local-polynomial / GAM parameterization exists.

These are distinct estimands. A curve estimate does not automatically justify a
derivative estimate, and a pointwise curve interval is not automatically valid
for an elasticity transformation.

## Retarget, outcome distributions and joint interventions

The blocks in this section run in order in one interpreter with NumPy and Antecedent; see [Install](start/installation.md). They ask three things of one simulated table: the effect in a different population, the outcome distribution under each arm, and the response to a coordinated pair of interventions.

### Prepare once, change the target population

The synthetic effect is `2 + 0.5*z`. Weighting toward larger `z` therefore changes
which population's average effect we estimate. The DAG asserts that `z` suffices
for adjustment; data alone do not establish that assumption.

```python
import numpy as np
import antecedent
from antecedent.intervention import Set
from antecedent.query import ExceedanceGrid, InterventionResponse, Quantile

rng = np.random.default_rng(150)
n = 2000
z = rng.normal(size=n)
probability = 1.0 / (1.0 + np.exp(-0.5 * z))
t = (rng.uniform(size=n) < probability).astype(float)
y = (2.0 + 0.5 * z) * t + z + rng.normal(size=n)
data = {"z": z, "t": t, "y": y}
graph = antecedent.Dag.from_edges(
    ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")]
)
study = antecedent.prepare(
    data,
    graph=graph,
    query=antecedent.AverageEffect("t", "y"),
    estimator="aipw",
    refute="none",
    inference=antecedent.Frequentist(bootstrap=0),
)
weights = np.exp(-0.5 * ((z - 0.5) / 0.8) ** 2)
executed = antecedent.estimate(study)  # fits the nuisances once and freezes the scores
completed = executed.study
all_observed = completed.retarget(np.ones(n), depends_on=[])
local = completed.retarget(weights, depends_on=["z"])
print("AllObserved ATE:", all_observed.value.estimate.ate)
print("Weighted ATE:", local.value.estimate.ate)
print("Weighted analytic SE:", local.value.estimate.se_analytic)
assert np.isfinite(local.value.estimate.ate)
```

`retarget` averages frozen cross-fitted scores with the supplied weights. It does
not refit nuisances. Nonconstant weights require declared parents in the certified
adjustment set; treatment and directed descendants are not allowed. This is
standardization within one population, not cross-environment transport or an
individual treatment-effect model. Inference assumes positivity and appropriate
nuisance convergence rates. `refute="none"` keeps this example focused on the
estimator; it does not establish that the causal assumptions hold.

Weights must align with the retained score rows. Here all rows are complete. If
rows were dropped, inspect `executed.value.estimate.score_table.row_index`
and align weights to those original row indices. `antecedent.estimate(antecedent.rebind(study, data=new_data))`
refits on new data under the prepared identification contract and leaves `study` and its
scores unchanged. A one-call `antecedent.analyze(...)` outcome offers the same handle as
`outcome.study`.

A nonconstant-weight retarget exports under a `RowWeights` target whose
`target_weights` identity binds the weight bits, row count, data snapshot, score
table and `depends_on`; the weights travel in the artifact and consume re-derives
the identity. Re-executing those weights on a different snapshot raises
`row_weights_bound_to_snapshot` (`study.reexecute_retarget`, Rust
`PreparedStudy::reexecute_retarget`); the prepared plan itself keeps
re-estimating, since the binding belongs to the retargeted result.
Calibration availability is explicit in that report.

### Read a CDF and its supported bands

An exceedance grid estimates the outcome distribution under each treatment arm.
It does not reduce several thresholds to one scalar ATE. The last threshold below
is deliberately beyond all observed outcomes to demonstrate unsupported tails.

```python
cdf_study = antecedent.prepare(
    data,
    graph=graph,
    query=antecedent.AverageEffect(
        "t", "y", outcome_functional=ExceedanceGrid([0.0, 2.0, 1_000_000.0])
    ),
    estimator="aipw",
    refute="none",
    inference=antecedent.Frequentist(bootstrap=0),
)
cdf_result = antecedent.estimate(cdf_study).study.retarget(weights, depends_on=["z"])
estimate = cdf_result.value.estimate
assert cdf_result.answer.value is None  # a CDF grid is not one scalar effect
assert estimate.score_table is not None
assert estimate.score_inference is not None
assert estimate.exceedance_cdf is not None
bands = estimate.score_inference
for (arm, threshold), projected, raw, lower, upper, supported in zip(
    estimate.score_table.columns,
    estimate.exceedance_cdf,
    bands.raw_means,
    bands.lower,
    bands.upper,
    bands.threshold_supported,
    strict=True,
):
    print(arm, threshold, "CDF:", projected, "raw CDF:", 1.0 - raw)
    if supported:
        print("raw-CDF simultaneous band:", 1.0 - upper, 1.0 - lower)
    else:
        print("band unavailable: insufficient tail or overlap support")
        # Do not present the numerical bounds when support is false.
assert not all(bands.threshold_supported)
```

For this binary query, arm 0 is control and arm 1 is active. Use the column metadata
rather than assuming an array order. `exceedance_cdf` reports `F_a(c)=P(Y(a)≤c)`;
`monotone_rearranged` indicates whether projection changed the raw estimates.
For this AIPW score-table path, `score_inference.raw_means` and its bounds are
**exceedance** coordinates, `P(Y(a)>c)`. Convert a raw mean with `1-mean` and an
interval with `[1-upper, 1-lower]`, as above, to obtain CDF coordinates. Complementing
all coordinates leaves their joint covariance unchanged. The conditional-grid
path already publishes CDF-coordinate bands and must not be complemented again.
Neither path supplies uncertainty for the projected CDF. Always honor
`threshold_supported`: unsupported bounds may be NaN or finite placeholders, and
must not be displayed as confidence intervals.

For a single `Exceedance(c)` AverageEffect query, the scalar effect is the active
minus control exceedance probability. For a quantile effect, use `Quantile(tau)`
with explicit Frequentist AIPW on a DAG or CoDetermined closure. Its uncertainty
conditions on the finite CDF grid and excludes interpolation bias and grid
selection. Inversion refuses unsupported, flat, or projection-altered crossings;
the request must use `refute="none"` because mean-effect refuters do not validate
quantiles. The licensed query families are:

| Query | Quantile meaning | Estimation contract |
| --- | --- | --- |
| `AverageEffect` | Active minus control quantile | Explicit Frequentist `aipw`, DAG or CoDetermined |
| Binary `ConditionalEffect` with one modifier or a modifier set | Difference of quantiles of arm CDFs standardized over the retained modifier distribution | Frequentist DAG, CPDAG, or PAG; aligned atom influences and zero unidentified mass |
| Discrete joint `InterventionResponse` | Quantile level in the requested joint cell | Explicit Frequentist `cell.aipw`, DAG or CoDetermined |

These start from AllObserved. Average and joint score-table plans also support
prepared retargeting; conditional plans do not export retargetable score tables.
The conditional `estimate.functional.quantile_grid` diagnostic lists the actual
thresholds; CDF coordinates are threshold-major, control then active.
Conditional quantiles are not a pointwise modifier surface or an average of
individual conditional quantiles. Graph mixtures combine CDFs before inversion.
Other estimator families remain refused: reusing a mean regression, matching
estimate, IV estimate, or Bayesian mean posterior would not supply the required
CDF influence function and density. Supporting those would require a separate
validated distributional estimator contract.

```python
conditional_median = antecedent.analyze(
    data,
    graph=graph,
    query=antecedent.ConditionalEffect("t", "y", "z", outcome_functional=Quantile(0.5)),
    refute="none",
    inference=antecedent.Frequentist(bootstrap=0),
)
print("Modifier-standardized median effect:", conditional_median.value.estimate.ate)
```

### Estimate a non-additive joint response

A joint intervention requests a response level, not an ATE. This model contains
an interaction, so an additive response model would miss part of its structure.
The two treatments below are binary and all four joint cells have observations.
This Python example uses a DAG with a common observed adjustment variable.

```python
t2 = (rng.uniform(size=n) < probability).astype(float)
y_joint = 1.2 * t + 0.8 * t2 + 1.5 * t * t2 + 0.5 * z + rng.normal(size=n)
joint_data = {"z": z, "t1": t, "t2": t2, "y": y_joint}
joint_graph = antecedent.Dag.from_edges(
    ["z", "t1", "t2", "y"],
    [("z", "t1"), ("z", "t2"), ("z", "y"), ("t1", "y"), ("t2", "y")],
)
joint = antecedent.analyze(
    joint_data,
    graph=joint_graph,
    query=InterventionResponse("y", intervention=[Set("t1", 1.0), Set("t2", 1.0)]),
    estimator="cell.aipw",
    refute="none",
    inference=antecedent.Frequentist(bootstrap=0),
)
print("E[Y | do(t1=1, t2=1)]:", joint.value.estimate)
print("Analytic SE:", joint.value.uncertainty.standard_error)
assert np.isfinite(joint.value.estimate)
assert joint.value.uncertainty.standard_error is not None
```

For a joint quantile level, pass the functional on the response query:

```python
joint_median = antecedent.analyze(
    joint_data,
    graph=joint_graph,
    query=InterventionResponse(
        "y", intervention=[Set("t1", 1.0), Set("t2", 1.0)],
        outcome_functional=Quantile(0.5),
    ),
    estimator="cell.aipw",
    refute="none",
    inference=antecedent.Frequentist(bootstrap=0),
)
print("Median under the joint intervention:", joint_median.value.estimate)
```

This is a response level in the requested cell, including cells containing a
zero treatment level; it is not an active-minus-control quantile contrast.

The population response in this simulation is 3.5 because `E[z]=0`; a finite
sample estimate need not equal it. `cell.aipw` fits joint-cell propensities and
per-cell outcome models. Empty training cells and inadequate support refuse.
CoDetermined tier backgrounds also license joint cells when their closure is
certified; Unknown tiers do not. `continuous_cell` does not license a point
intervention on a continuous mediator.

See the [support matrix](support-matrix.md) for the current scope and per-cell evidence, and the [artifact guide](artifacts.md#score-tables-and-distribution-payloads) for saved payload semantics.

## Intervention responses

Typed intervention laws live in the `antecedent.intervention` stage namespace:

```python
query = antecedent.InterventionResponse(
    "outcome",
    intervention=antecedent.intervention.Gaussian("dose", mean=1.0, variance=0.04),
)
result = antecedent.analyze(data, graph=dag, query=query)
```

Under Frequentist inference, `Set`, `Shift`, `Bernoulli`, `Gaussian`, and
`Categorical` evaluate plug-in mean responses. A sequence of specifications requests a joint intervention. The
estimator fits an additive outcome model and averages predictions under the
requested policy. `Bernoulli` and `Categorical` laws are summed exactly over
their support; a `Gaussian` law is integrated by 48-node Gauss-Hermite
quadrature per policy (deterministic, exact for polynomial dose effects of
degree up to 95), which is valid because the outcome model is additive, so each
policy enters only through its own marginal law. Its canonical strategy is
`response.intervention_gcomp`, distinct from the Kennedy curve estimator. The result is deliberately
marked extrapolative: observed marginal bounds are reported, but joint policy
support and statistical uncertainty are not certified. Soft mechanism
replacements and temporally sequenced policies require a structural model and
fail closed in this estimator.

## Observation is not outcome

The scientific outcome may differ from the recorded variable. For example,
sales limited by inventory are a censored measurement of demand:

```python
mechanism = antecedent.observation.RightCensored(
    "latent_demand",
    "observed_sales",
    "inventory",
    "demand_observed",
)
assumption = antecedent.observation.IndependentGiven(["price", "season_index"])

query = antecedent.ResponseCurve(
    "price",
    "latent_demand",
    grid=[8.0, 10.0, 12.0],
    observation=mechanism,
    observation_assumptions=[assumption],
)
```

`RightCensored` describes what was recorded. `IndependentGiven` is a separate,
contestable identifying claim. Antecedent does not derive the second from the
first. This example declares censoring independent of residual demand given
price and season. A nonempty `IndependentGiven(Z)` that contains treatment and
every causal adjustment variable is the licensed Cox IPCW pair; marginal
Kaplan–Meier IPCW remains the licensed pair for `IndependentGiven([])`. See
the [observation pair contract](observation-contract.md). Selected-outcome AIPW
and both censoring IPCW estimators can be composed into a `ResponseCurve` when
their explicit assumption contract is satisfied. Selected-outcome AIPW
cross-fits both its nuisance models, and refuses a fold whose training rows
cannot support them rather than falling back to an in-sample fit. These paths
return a point curve with `uncertainty.kind == "none"`
and a `joint_uncertainty_unavailable` warning: correcting observation and then
smoothing a curve does not make the component standard errors a valid joint
band. Observation-aware subset validation likewise fails closed until both
stages can be refit jointly. Observation-adjusted derivatives, interval
censoring, and other unsupported mechanism/assumption/estimator combinations
fail closed rather than returning the response of the observed proxy.

See the runnable, deterministic notebooks for a
[complete-observation response](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/continuous_causal_response.ipynb)
and the [pricing/availability distinction](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/pricing_availability_latent_demand.ipynb).

### Static intervention-level uncertainty

Additive GAM intervention responses use a fitted-basis sandwich that includes
coefficient-fitting uncertainty and empirical covariate-average uncertainty.
Knots and smoothing penalties are held fixed. The interval excludes knot
selection, smoothing bias, and policy-integration error; it is not a
nonparametric efficient-influence interval. A Set response can therefore have
sampling uncertainty even when all counterfactual fitted row means are equal.
Class-aware static responses combine these scores using frozen graph weights
and original row IDs. Curve joint-covariance scores are `N` times the local-WLS
contributions documented above; the diagnostic export itself is unchanged.
