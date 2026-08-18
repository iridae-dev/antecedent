# Causal responses

Antecedent 0.5 treats a continuous causal response as more than a collection of
binary contrasts. The scalar Python constructors retain the established
positional convention:

```python
query = antecedent.ResponseCurve(
    "dose",
    "outcome",
    grid=[0.5, 1.0, 1.5, 2.0],
)
result = antecedent.analyze(data, graph=dag, query=query)
```

The same complete-observation curve can use the established staged workflow
when identification should be inspected before data are estimated:

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
   and declared structural assumptions.
2. `support` describes empirical overlap in the requested treatment region.
   An identified response may still be extrapolative or outside empirical
   support.
3. `uncertainty.kind` says what the interval means. Pointwise intervals cover
   one grid point at a time. The Rust estimator can instead request a fixed-grid
   simultaneous band with `response_options(...)`; Python uses
   `estimator_config={"bandwidth": ..., "simultaneous_replicates": ...}`.
   The explicit bandwidth is required: Antecedent does not silently invent an
   undersmoothing rule. The band uses a deterministic Rademacher multiplier
   construction conditional on the cross-fitted pseudo-outcome. It does not
   refit nuisance models or claim unconditional coverage without the caller's
   bandwidth/regularity contract. An identified
   set is uncertainty about what the assumptions determine, not a confidence
   interval.
4. `assumptions` and `provenance` record the claims and algorithm used. Do not
   infer an observation assumption merely from an observation-mechanism column.

Inspect warnings and diagnostics even when a numerical estimate is present:

```python
print(result.identification)
print(result.support.status)
print(result.support.warnings)
print(result.uncertainty.kind)
print(result.provenance)
```

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

A `Pag` with unresolved marks has a different contract. For `ResponseCurve`
only, Antecedent streams a bounded set of compatible MAG completions, applies
generalized adjustment in each completion, and estimates the curve in every
identified case. `result.envelope` contains pointwise lower/upper identified
bounds plus normalized `identified_mass` and `unidentified_mass`; unidentified
completions are never discarded or renormalized away. Those bounds are the
pointwise min/max of completion-specific **point** curves: they encode structural
uncertainty across the examined class, not within-completion sampling
uncertainty, and are not a confidence band. A capped enumeration is
reported separately through `enumeration_capped`; in that case `mass_scope` is
`"examined_completions"`, so the normalized fractions are never presented as
full-class probability mass. Per-completion adjustment-search truncation remains
separate in `truncated_completions`. Derivatives and graph-
posterior mixtures over curves remain outside this PAG path.

## Validate a curve without pretending it is an ATE

`refute="cheap"` (and the other non-empty refutation spellings) runs only checks
that have function-valued meaning. `result.validation` records the existing
support/overlap status and a deterministic ten-replicate 80% row-subset curve
refit diagnostic. The subset statistic is the maximum absolute shift between
the full-data curve (or PAG envelope) and the mean subset result; it is reported
without inventing a universal pass threshold. Scalar ATE refuters—placebo
treatment, dummy outcome, random common cause, and scalar sensitivity—appear as
an explicit `skipped` check.

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
psi[g, i] = w_i * [(X'WX)^{-1}]_{row 0} . x_i * r_i
```

— the WLS influence of row `i` on the fitted level, in outcome units. It is
**not** the semiparametric efficient influence function of the Kennedy
estimator: the pseudo-outcomes are treated as fixed data, so nuisance-estimation
uncertainty is not inside these values, and neither is bandwidth selection.

**Centering and scaling.** At every grid point the influences sum to zero
exactly (up to float roundoff) — this is the first WLS normal equation, not a
convention applied afterwards. There is no `1/n` scaling: these are per-row
contributions, sized so the identities below hold.

**Exact relationship to reported uncertainty.** The reported pointwise standard
error is `SE(g) = sqrt(sum_i psi[g, i]^2)`, and the pointwise band is
`m(g) ± z * SE(g)` with `z = Phi^{-1}(0.5 + level/2)`. The simultaneous band's
critical value is the `level` empirical quantile, over
`simultaneous_replicates` Rademacher draws (SplitMix64 stream from
`multiplier_seed`, so deterministic given the seed), of
`max_g | sum_i eps_i * psi[g, i] | / SE(g)`; the band is `m(g) ± c * SE(g)`.
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

- `ResponseCurve(treatment, outcome, grid=...)` evaluates
  `a -> E[Y | do(A=a)]` on an explicit, increasing grid.
- `PointDerivative(..., at=a)` is a local slope and is typically more sensitive
  to smoothing and local support than the curve itself. It requires an explicit
  `bandwidth` in response options / `estimator_config`; Silverman's rule is a
  level/KDE rate and is refused here rather than silently oversmoothing `m'`.
- `AverageDerivative(...)` averages a derivative over an explicit weighting
  law; the default observed-law weighting describes the sampled population.
- `Elasticity(..., at=a)` is a log-outcome/log-treatment derivative. The
  treatment point must be positive, and scientific interpretation also requires
  a meaningful positive outcome scale. Like `PointDerivative`, it requires an
  explicit bandwidth.

These are distinct estimands. A curve estimate does not automatically justify a
derivative estimate, and a pointwise curve interval is not automatically valid
for an elasticity transformation.

## Intervention responses

Typed intervention laws live in the `antecedent.intervention` stage namespace:

```python
query = antecedent.InterventionResponse(
    "outcome",
    intervention=antecedent.intervention.Gaussian("dose", mean=1.0, variance=0.04),
)
result = antecedent.analyze(data, graph=dag, query=query)
```

`Set`, `Shift`, `Bernoulli`, `Gaussian`, and `Categorical` evaluate plug-in mean
responses. A sequence of specifications requests a joint intervention. The
estimator fits an additive outcome model and averages predictions under the
requested policy; stochastic laws use fixed-seed independent-coordinate
inverse-CDF Monte Carlo integration. Its canonical strategy is
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
first. This example deliberately declares conditional censoring; the current
marginal Kaplan–Meier IPCW estimator refuses it because its licensed contract
requires `IndependentGiven([])`. Selected-outcome AIPW and *unconditional*
right/left-censoring IPCW can be composed into a `ResponseCurve` when their
explicit assumption contract is satisfied. Selected-outcome AIPW cross-fits both
its nuisance models, and refuses a fold whose training rows cannot support them
rather than falling back to an in-sample fit. These paths return a point curve with
`uncertainty.kind == "none"`
and a `joint_uncertainty_unavailable` warning: correcting observation and then
smoothing a curve does not make the component standard errors a valid joint
band. Observation-aware subset validation likewise fails closed until both
stages can be refit jointly. Unsupported mechanism/assumption/estimator
combinations fail closed rather than returning the response of the observed
proxy.

See the runnable, deterministic notebooks for a
[complete-observation response](../examples/notebooks/continuous_causal_response.ipynb)
and the [pricing/availability distinction](../examples/notebooks/pricing_availability_latent_demand.ipynb).
