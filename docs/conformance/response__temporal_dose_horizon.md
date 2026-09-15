# Temporal dose × horizon known-truth pin

**Suite path:** `conformance/response/temporal_dose_horizon`

The stationary linear temporal SCM is

```text
T_s = U_s
Y_s = 1 + 2 T_{s-1} + 3 T_{s-2}
```

with template edges `T@lag1 -> Y@lag0` and `T@lag2 -> Y@lag0`.
The frozen deterministic realization uses the period-four exogenous sequence
`U_s = [0, 1, 0, -1]` repeated for 242 observations. This sequence is not a
treatment autoregression: each `T_s` is an exogenous input. Over the exact
lag-aligned windows used by the estimator, the adjacent treatment columns have
zero mean and zero cross-product, so the two OLS fits recover the structural
coefficients exactly.

The policy is `Pulse { at: -1 }`. Horizon 1 evaluates `Y_0`, and horizon 2
evaluates `Y_1`. Therefore

```text
E[Y_0 | do(T_{-1}=d)] = 1 + 2d
E[Y_1 | do(T_{-1}=d)] = 1 + 3d
```

For doses `[0, 1]` and horizons `[1, 2]`, the dose-major surface is
`[1, 1, 3, 4]`: `value[dose_index * 2 + horizon_index]`. The flattened grid
stores coordinate pairs `[dose, horizon]` in the same order.

This truth matches `TemporalResponseEstimator`: each horizon is re-anchored as
a lagged OLS design, then linear g-computation replaces the treatment column
with the requested dose and averages the fitted rows. Multi-step `Sequence`
uses the same unfolded sequential g-computation as multi-step Sustained
(`temporal.backdoor.unfolded`), not a second identifier.

`Soft(constant=1)` has the same path `[3, 4]` as `Set(1)`.
`Soft(additive_shift=1)` averages g-computation at each observed treatment plus
one; the aligned treatment means are zero here, so its path is also `[3, 4]`.
A single-step, one-variable `Sequence` resolves to the same overlay.
A two-step `Sequence([Set(1), Set(1)])` at consecutive times ending at the
pulse origin (`T_{-2}` then `T_{-1}`) is

```text
E[Y_0 | do(T_{-2}=1, T_{-1}=1)] = 1 + 2 + 3 = 6
E[Y_1 | do(T_{-2}=1, T_{-1}=1)] = 1 + 2 E[T_0] + 3 = 4
```

so the path is `[6, 4]`. Last-step-only `Set(1)` remains `[3, 4]`; the two
must not agree. Nested `Sequence` and Soft families other than
`constant` / `additive_shift` stay refused.

At horizon 1, the two-point surface contrast
`mean(dose=1) - mean(dose=0) = 2` matches the `PulseEffect` value for
active `1` versus control `0`; both dispatch through `TemporalLinearAdjustment`
and agree numerically on this fixture. The licensed single-step
`SustainedEffect` window at offset `-1` recovers the same contrast. This is
observed numerical agreement via shared adjustment machinery, not a
derivation of one from the other and not a separate response estimand;
multi-step Sustained is not evidenced here.

## Bands

The pre-1.9 fixture pinned an analytic delta-method band (`surface.lower` /
`surface.upper`, removed in 1.9; see git history at `0b907cb5`). That band
treated lag-aligned rows as independent and is no longer published: with zero
bootstrap replicates the surface carries no band and an
`estimate.temporal_response.band_withheld` warning.

`block_band` pins what requested replicates publish instead: the joint
circular-block bootstrap of the dose × horizon surface, produced through the
public Study API with `bootstrap_replicates(60)` and
`ExecutionContext::for_tests(21)` (seed 21) at commit `3330a2c4`, by
`cargo test -p antecedent --test temporal_response_facade
temporal_dose_horizon_point_and_block_bands_match_fixture` (the test itself
recomputes the run; the numbers were captured from the same run). The Python
facade (`analyze(..., bootstrap=60, seed=21)`) runs the production context
with the same seed and reproduces the values bit for bit, as do a serial
context and a four-thread bounded context: the bootstrap loop is serial and
its RNG stream is seeded from the context, so the pin does not depend on the
thread count.

Pinned fields, all in the dose-major layout of `surface.mean`:

- `pointwise_lower` / `pointwise_upper`: the 95% pointwise band
  (`ResponseUncertainty::PointwiseBand`), mean ± 1.96 × replicate SD after the
  dispersion inflation.
- `simultaneous_lower` / `simultaneous_upper` / `simultaneous_critical`: the
  sup-t band carried by the `response.simultaneous_band.{lower,upper,critical}`
  support diagnostics; the critical value is the max-studentized-deviation
  order statistic over the 60 joint replicates.
- `block_length`: the `response.temporal.block_length` diagnostic
  `[length, rule, testing, rows, dispersion_factor]`; the block stays at the
  rule `max(span, ceil(sqrt(240))) = 16` because every estimating score on
  this noiseless period-4 DGP is degenerate or negatively dependent
  (`testing = 0`). 16 is a multiple of the period, so every circular resample
  stays orthogonal and the band collapses to the point surface. The
  dispersion factor is the circular-Bartlett fixed-b ratio times HC1.
- `kernel_bias_factor` / `effective_rows`: the per-cell
  `response.temporal.kernel_bias_factor` and `response.temporal.effective_rows`
  diagnostics.

These values are compared at relative tolerance `tolerance.band_rtol`
(`1e-9`) by `crates/antecedent/tests/temporal_response_facade.rs` and
`python/tests/test_temporal_response_api.py`. They are a **determinism pin**,
not a coverage claim: a change in the resampler, the block-length rule, the
dispersion factors or the RNG stream moves them and must be re-recorded
deliberately. The coverage evidence for the band is the weekly gate
`crates/antecedent/tests/v19_temporal_response_calibration.rs`. There is no
`reference.py` for this fixture: the point surface is derived analytically
above, and the band is a seeded bootstrap that cannot be re-derived
independently in numpy without reimplementing the resampler, its RNG stream
and the dispersion factors, so no independent numeric cross-check of the band
is claimed. Widths sit at GEMM scale because the rule block preserves
orthogonality; they must not vanish only at dose zero (the old
|dose|-scaled analytic-band regression).

Empirical support on this fixture is fully `supported` at every cell: the
period-4 treatment lives in `{-1,0,1}`, so the union of horizon ranges equals
the intersection. Mixed-horizon support is pinned on
`conformance/response/temporal_horizon_support`.

## Expected summary

Top-level keys: `claim, contract, estimator_contract, fixture_id, generation, tolerance` (6 fields).
