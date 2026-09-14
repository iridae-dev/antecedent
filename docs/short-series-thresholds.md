# Short-series thresholds for circular-block intervals

Every Frequentist temporal interval on one series is a fixed-b circular-block
bootstrap over lag-aligned rows
([capabilities](capabilities.md#serially-dependent-rows-circular-block-uncertainty)).
When the series is short for the serial dependence of its estimating score,
the interval can under-cover, and the result carries
`estimate.temporal.circular_block_se.short_series`. This page records the
measurement behind when that warning fires.

## Statistic

The warning reads the *score effective rows*
(`antecedent_estimate::score_effective_rows`): over every estimating-score
series of the interval, the smaller of two readings,

* the lag-1 reading `n(1 − r₁)/(1 + r₁)` (an AR(1) variance-inflation
  equivalent, `r₁` floored at zero), and
* the block-length reading `n·γ̂₀ / ĝ_b`, with `ĝ_b` the Bartlett long-run
  variance at the block length the interval resamples with.

The scores are the treatment-coefficient influence (TemporalDag Pulse and
single-step Sustained), the Total, Direct and Mediated scores (temporal
mediation), the contrast influence (multi-step Sustained sequential
g-computation), and every atom's influence plus the weighted mixture score
(class envelopes and DBN posteriors). The provenance diagnostic prints the
statistic; the warning fires below the threshold of the interval's family.

| family | SE path | threshold |
|---|---|---|
| single-window adjustment | TemporalDag Pulse / single-step Sustained | 45 |
| temporal mediation | Total, Direct, Mediated (shared replicate) | 35 |
| multi-step sequential | TemporalDag multi-step Sustained | 40 |
| multi-atom mixture | TemporalCpdag / TemporalPag envelopes, DBN posteriors | 155 |

## How the thresholds were set

`crates/antecedent/tests/v19_short_series_measurement.rs` (an ignored
measurement, not part of the calibration gate) sweeps AR(1) persistence
ρ ∈ {0.5, 0.8, 0.9, 0.95} and series length n for each family, on a
short-memory design and a persistent one:

* the short-memory designs are the calibration DGPs (an MA(3) treatment driven
  through an observed variable, AR(1) residual; or a confounded lag DGP whose
  adjusted treatment innovation is iid). Their scores forget within a few lags
  whatever the residual persistence;
* the persistent designs make the treatment AR(1)(ρ) as well as the residual,
  so the score is itself close to AR(1)(ρ²).

The effective-row count alone does not tell the two apart: at 25–35 effective
rows the short-memory designs cover nominally (n = 40–60) while the persistent
ones fail (n = 60–160). The thresholds therefore follow the persistent designs.
A threshold is the smallest multiple of 5 at which every cell covering below
0.855 (the lower edge of the 400-replicate gate band) warns on at least 90% of
its replicates, taking the larger of the sweep below and an independent
replication (`ANTECEDENT_SHORT_SERIES_SEED_OFFSET=5000`, 1000 replicates):

| family | sweep | replication | threshold |
|---|---|---|---|
| single-window adjustment | 45 | 45 | 45 |
| temporal mediation | 30 | 35 | 35 |
| multi-step sequential | 35 | 40 | 40 |
| multi-atom mixture | 155 | 155 | 155 |

### What the failure line tolerates

The failure line 0.855 is the calibration gate's tolerance, not a coverage
target. "Nominal" in the gate means the empirical coverage is within ±3 Monte
Carlo SEs of the level at the gate's replicate count: at 400 replicates and a
90% level that is ±4.5 points, the band [0.855, 0.945]. The thresholds are set
so that cells *below that band* warn; a cell between 0.855 and roughly 0.88
is not caught by the warning and can still pass a 400-replicate gate, even
though at 2000 replicates (band [0.880, 0.920], ±2 points) it measures
significantly low. Such cells are in the table below: the AR(1)-treatment
Pulse at ρ = 0.8–0.9, n = 400 (0.875), the AR(1)-treatment mediation Total at
ρ = 0.9–0.95, n = 400 (0.879–0.882), and the TemporalCpdag mixture at ρ = 0.9,
n = 400 (0.873, warned on 76%). The last is also a gated design; its gate
(`frequentist_temporal_cpdag_pulse_ar1_rho09_n400_boundary_within_band`) is
named as a boundary cell and asserts only the 400-replicate band.

The mixture threshold is set by a different failure. Its SE-driven cells (the
six-completion TemporalPag envelope at ρ ≥ 0.9, n ≤ 160) warn from about 30.
The TemporalCpdag and DBN mixtures fail at ρ ≥ 0.9 through bias, not the SE:
the non-causal completion omits a persistent confounder, and its finite-sample
estimate sits 0.2–0.8 of an SD from its probability limit. Only that
completion's score shows it, as a weak but slowly decaying component that the
block-length reading sees; catching it at ρ = 0.95, n = 400 takes 155, which
also warns on mixtures that cover nominally below about n = 400.

The previous rule (one floor of 100 on the lag-1 reading of the single score,
or of the weighted mixture score) warned, in a 1000-replicate run of the same
sweep, on every short-memory replicate at n ≤ 100 and on 30–88% at n = 160
while those designs covered nominally, and on at most 11% of the TemporalCpdag
bias cells at n = 160 and none at n = 400.

## Table

2000 replicates per cell, nominal 0.90 (band at this count [0.880, 0.920]);
100 bootstrap replicates per fit for one-series designs, 199 for mixtures.

Measurement conditions. The table was measured with blocks sized on every
estimating score: the target influence(s), the mixture score, and every
normal-equation score of every fitted regression (residuals included), the
rule every circular-block path uses. It was measured before two later changes
that move individual numbers without changing the rule:

* the fixed-b factor then was the non-circular Kiefer–Vogelsang polynomial;
  the circular-block SEs now carry the circular-Bartlett factor
  (`antecedent_estimate::circular_fixed_b_scale`). At the block-to-row ratios
  in this table (median `blocks` 3–31, so b ≈ 0.03–0.33) the new factor is
  0.3–0.7% smaller for b ≤ 0.15 and up to 4% larger at b = 1/3, so the cells
  with the fewest blocks (about 3 blocks, b ≈ 0.31; nearly all warned) gain
  about 1.2 coverage points and the rest move by a few tenths of a point;
* the persistent fixtures (`fixtures::chain_pag_series`,
  `persistent_dgp::mediation_series`, `persistent_dgp::sequential_series`)
  now seed their noise streams independently per replicate; the previous
  seeding reused one replicate's treatment path as another replicate's
  residual path, so in the AR(1)-treatment and TemporalPag rows pairs of
  replicates were not independent.

Re-running the command below measures the table under the current code.
Mediation lists Total / Direct / Mediated; the other columns describe the
headline contrast. `bias/SD` and `SE/SD` divide the mean error and the mean
reported SE by the Monte-Carlo SD of the estimate; `blocks` is the median
`rows / block length`; `warned` is the share of replicates carrying the
short-series warning at the thresholds above.

```text
ANTECEDENT_CALIBRATION_NSIM=2000 cargo test --release -p antecedent \
  --test v19_short_series_measurement -- --ignored --nocapture
```

| family | design | ρ | n | coverage | bias/SD | SE/SD | eff. rows q10 / median / q90 | blocks | warned |
|---|---|---|---|---|---|---|---|---|---|
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 40 | 0.876 | -0.00 | 1.02 | 20 / 31 / 39 | 6.5 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 40 | 0.868 | +0.05 | 1.01 | 16 / 26 / 39 | 3.9 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 40 | 0.890 | -0.03 | 1.02 | 15 / 24 / 39 | 3.2 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 40 | 0.870 | +0.02 | 0.96 | 14 / 23 / 38 | 3.2 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 60 | 0.889 | +0.01 | 1.03 | 30 / 43 / 59 | 9.8 | 0.56 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 60 | 0.874 | -0.00 | 1.03 | 23 / 36 / 55 | 4.2 | 0.75 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 60 | 0.886 | -0.01 | 1.00 | 21 / 32 / 50 | 3.7 | 0.82 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 60 | 0.885 | +0.03 | 0.96 | 20 / 32 / 50 | 3.1 | 0.83 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 100 | 0.882 | +0.01 | 1.03 | 49 / 69 / 93 | 11.0 | 0.05 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 100 | 0.905 | -0.00 | 1.04 | 36 / 53 / 78 | 5.0 | 0.27 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 100 | 0.894 | -0.00 | 1.03 | 34 / 49 / 72 | 3.8 | 0.38 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 100 | 0.901 | -0.02 | 1.04 | 33 / 48 / 69 | 3.2 | 0.42 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 160 | 0.897 | +0.01 | 1.04 | 80 / 106 / 138 | 15.9 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 160 | 0.897 | +0.02 | 1.06 | 56 / 82 / 111 | 5.7 | 0.02 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 160 | 0.898 | +0.04 | 1.06 | 53 / 76 / 105 | 4.0 | 0.04 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 160 | 0.911 | +0.01 | 1.06 | 50 / 71 / 98 | 3.5 | 0.06 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 400 | 0.887 | -0.00 | 0.99 | 206 / 255 / 304 | 18.1 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 400 | 0.900 | -0.02 | 1.06 | 148 / 195 / 242 | 8.1 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 400 | 0.901 | +0.00 | 1.08 | 134 / 178 / 225 | 5.2 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 400 | 0.902 | -0.03 | 1.08 | 125 / 170 / 213 | 4.4 | 0.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 40 | 0.886 | -0.02 | 1.05 | 20 / 31 / 39 | 9.8 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 40 | 0.847 | +0.03 | 0.95 | 11 / 19 / 33 | 6.5 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 40 | 0.810 | -0.02 | 0.86 | 9 / 16 / 28 | 3.9 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 40 | 0.770 | -0.04 | 0.79 | 8 / 15 / 26 | 3.9 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 60 | 0.871 | +0.04 | 0.98 | 29 / 43 / 59 | 9.8 | 0.58 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 60 | 0.858 | +0.01 | 0.97 | 15 / 23 / 37 | 5.9 | 0.97 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 60 | 0.794 | -0.01 | 0.86 | 11 / 18 / 29 | 4.2 | 0.99 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 60 | 0.765 | -0.00 | 0.79 | 9 / 16 / 26 | 3.7 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 100 | 0.867 | +0.00 | 0.96 | 47 / 66 / 89 | 11.0 | 0.07 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 100 | 0.868 | +0.01 | 0.99 | 21 / 32 / 48 | 5.5 | 0.86 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 100 | 0.847 | -0.00 | 0.97 | 13 / 21 / 34 | 4.1 | 0.98 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 100 | 0.821 | +0.00 | 0.87 | 10 / 17 / 28 | 3.5 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 160 | 0.892 | -0.01 | 1.01 | 76 / 101 / 130 | 15.9 | 0.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 160 | 0.862 | -0.02 | 0.99 | 31 / 44 / 63 | 6.1 | 0.53 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 160 | 0.852 | -0.01 | 0.99 | 18 / 27 / 41 | 4.2 | 0.95 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 160 | 0.842 | +0.05 | 0.91 | 12 / 19 / 31 | 3.8 | 0.99 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 400 | 0.895 | -0.02 | 1.04 | 193 / 237 / 285 | 18.1 | 0.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 400 | 0.875 | +0.01 | 1.01 | 70 / 94 / 121 | 8.1 | 0.01 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 400 | 0.875 | +0.03 | 1.02 | 37 / 51 / 69 | 5.4 | 0.32 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 400 | 0.853 | -0.02 | 0.98 | 20 / 30 / 43 | 4.6 | 0.92 |
| mediation | mediation, MA(3) treatment | 0.50 | 40 | Total 0.884, Direct 0.904, Mediated 0.911 | -0.04 | 1.06 | 19 / 28 / 39 | 6.5 | 0.78 |
| mediation | mediation, MA(3) treatment | 0.80 | 40 | Total 0.875, Direct 0.891, Mediated 0.912 | +0.02 | 1.03 | 16 / 24 / 36 | 3.9 | 0.88 |
| mediation | mediation, MA(3) treatment | 0.90 | 40 | Total 0.889, Direct 0.901, Mediated 0.919 | +0.01 | 1.05 | 15 / 23 / 35 | 3.2 | 0.90 |
| mediation | mediation, MA(3) treatment | 0.95 | 40 | Total 0.906, Direct 0.898, Mediated 0.922 | -0.03 | 1.09 | 14 / 22 / 35 | 3.2 | 0.91 |
| mediation | mediation, MA(3) treatment | 0.50 | 60 | Total 0.894, Direct 0.893, Mediated 0.898 | +0.02 | 1.07 | 30 / 42 / 56 | 9.8 | 0.26 |
| mediation | mediation, MA(3) treatment | 0.80 | 60 | Total 0.883, Direct 0.893, Mediated 0.903 | -0.04 | 1.05 | 22 / 34 / 50 | 4.2 | 0.54 |
| mediation | mediation, MA(3) treatment | 0.90 | 60 | Total 0.893, Direct 0.901, Mediated 0.923 | +0.01 | 1.04 | 21 / 32 / 48 | 3.3 | 0.63 |
| mediation | mediation, MA(3) treatment | 0.95 | 60 | Total 0.902, Direct 0.896, Mediated 0.912 | +0.04 | 1.05 | 21 / 31 / 46 | 3.1 | 0.66 |
| mediation | mediation, MA(3) treatment | 0.50 | 100 | Total 0.886, Direct 0.900, Mediated 0.899 | -0.03 | 1.04 | 48 / 66 / 86 | 11.0 | 0.01 |
| mediation | mediation, MA(3) treatment | 0.80 | 100 | Total 0.898, Direct 0.897, Mediated 0.908 | -0.01 | 1.10 | 37 / 53 / 74 | 5.0 | 0.08 |
| mediation | mediation, MA(3) treatment | 0.90 | 100 | Total 0.887, Direct 0.911, Mediated 0.912 | -0.02 | 1.06 | 33 / 49 / 70 | 3.5 | 0.13 |
| mediation | mediation, MA(3) treatment | 0.95 | 100 | Total 0.903, Direct 0.905, Mediated 0.911 | +0.01 | 1.08 | 32 / 48 / 68 | 3.0 | 0.15 |
| mediation | mediation, MA(3) treatment | 0.50 | 160 | Total 0.890, Direct 0.896, Mediated 0.911 | +0.02 | 1.05 | 77 / 103 / 132 | 11.4 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.80 | 160 | Total 0.898, Direct 0.903, Mediated 0.916 | +0.01 | 1.07 | 58 / 80 / 107 | 5.7 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.90 | 160 | Total 0.909, Direct 0.891, Mediated 0.916 | +0.03 | 1.08 | 52 / 75 / 103 | 4.0 | 0.01 |
| mediation | mediation, MA(3) treatment | 0.95 | 160 | Total 0.896, Direct 0.901, Mediated 0.910 | +0.02 | 1.02 | 51 / 72 / 101 | 3.4 | 0.01 |
| mediation | mediation, MA(3) treatment | 0.50 | 400 | Total 0.904, Direct 0.898, Mediated 0.906 | +0.02 | 1.04 | 201 / 252 / 302 | 18.1 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.80 | 400 | Total 0.901, Direct 0.896, Mediated 0.901 | -0.04 | 1.10 | 146 / 194 / 239 | 7.7 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.90 | 400 | Total 0.897, Direct 0.899, Mediated 0.898 | -0.01 | 1.07 | 132 / 177 / 221 | 5.2 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.95 | 400 | Total 0.902, Direct 0.895, Mediated 0.922 | +0.04 | 1.08 | 127 / 172 / 215 | 4.4 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.50 | 40 | Total 0.876, Direct 0.889, Mediated 0.907 | +0.04 | 1.05 | 18 / 28 / 39 | 6.5 | 0.79 |
| mediation | mediation, AR(1) treatment | 0.80 | 40 | Total 0.833, Direct 0.841, Mediated 0.921 | +0.00 | 0.95 | 11 / 18 / 28 | 3.9 | 0.97 |
| mediation | mediation, AR(1) treatment | 0.90 | 40 | Total 0.791, Direct 0.796, Mediated 0.933 | -0.03 | 0.87 | 8 / 14 / 23 | 3.9 | 0.99 |
| mediation | mediation, AR(1) treatment | 0.95 | 40 | Total 0.791, Direct 0.755, Mediated 0.935 | -0.04 | 0.85 | 7 / 12 / 21 | 3.2 | 0.99 |
| mediation | mediation, AR(1) treatment | 0.50 | 60 | Total 0.873, Direct 0.893, Mediated 0.901 | -0.02 | 0.99 | 28 / 40 / 56 | 9.8 | 0.33 |
| mediation | mediation, AR(1) treatment | 0.80 | 60 | Total 0.856, Direct 0.865, Mediated 0.914 | -0.01 | 0.99 | 13 / 21 / 33 | 4.2 | 0.93 |
| mediation | mediation, AR(1) treatment | 0.90 | 60 | Total 0.826, Direct 0.823, Mediated 0.926 | +0.02 | 0.94 | 9 / 15 / 26 | 3.7 | 0.98 |
| mediation | mediation, AR(1) treatment | 0.95 | 60 | Total 0.807, Direct 0.804, Mediated 0.933 | +0.00 | 0.85 | 7 / 13 / 22 | 3.3 | 0.99 |
| mediation | mediation, AR(1) treatment | 0.50 | 100 | Total 0.889, Direct 0.899, Mediated 0.908 | +0.03 | 1.05 | 44 / 62 / 83 | 11.0 | 0.02 |
| mediation | mediation, AR(1) treatment | 0.80 | 100 | Total 0.865, Direct 0.880, Mediated 0.909 | -0.00 | 1.01 | 20 / 30 / 44 | 5.0 | 0.70 |
| mediation | mediation, AR(1) treatment | 0.90 | 100 | Total 0.856, Direct 0.863, Mediated 0.925 | +0.02 | 0.98 | 12 / 19 / 31 | 3.8 | 0.95 |
| mediation | mediation, AR(1) treatment | 0.95 | 100 | Total 0.815, Direct 0.814, Mediated 0.929 | -0.03 | 0.90 | 8 / 14 / 24 | 3.2 | 0.99 |
| mediation | mediation, AR(1) treatment | 0.50 | 160 | Total 0.900, Direct 0.899, Mediated 0.912 | +0.01 | 1.04 | 73 / 97 / 124 | 11.4 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.80 | 160 | Total 0.867, Direct 0.875, Mediated 0.920 | +0.02 | 1.01 | 30 / 43 / 60 | 5.7 | 0.22 |
| mediation | mediation, AR(1) treatment | 0.90 | 160 | Total 0.863, Direct 0.857, Mediated 0.910 | -0.01 | 0.99 | 16 / 25 / 38 | 4.0 | 0.85 |
| mediation | mediation, AR(1) treatment | 0.95 | 160 | Total 0.850, Direct 0.856, Mediated 0.929 | -0.03 | 0.97 | 11 / 18 / 28 | 3.5 | 0.98 |
| mediation | mediation, AR(1) treatment | 0.50 | 400 | Total 0.908, Direct 0.901, Mediated 0.905 | -0.00 | 1.04 | 190 / 234 / 280 | 18.1 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.80 | 400 | Total 0.894, Direct 0.890, Mediated 0.910 | +0.07 | 1.06 | 70 / 92 / 120 | 7.7 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.90 | 400 | Total 0.882, Direct 0.886, Mediated 0.908 | +0.03 | 1.06 | 35 / 49 / 66 | 5.2 | 0.10 |
| mediation | mediation, AR(1) treatment | 0.95 | 400 | Total 0.879, Direct 0.881, Mediated 0.903 | +0.03 | 1.06 | 19 / 28 / 42 | 4.6 | 0.77 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 40 | 0.899 | -0.02 | 1.08 | 22 / 32 / 38 | 6.3 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 40 | 0.902 | +0.04 | 1.08 | 18 / 27 / 38 | 6.3 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 40 | 0.896 | +0.02 | 1.01 | 16 / 24 / 38 | 3.8 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 40 | 0.888 | -0.01 | 1.02 | 15 / 23 / 36 | 3.8 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 60 | 0.909 | +0.02 | 1.09 | 33 / 47 / 58 | 9.7 | 0.28 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 60 | 0.889 | +0.01 | 1.06 | 26 / 38 / 56 | 5.8 | 0.58 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 60 | 0.909 | -0.00 | 1.06 | 22 / 33 / 48 | 3.6 | 0.74 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 60 | 0.919 | +0.01 | 1.06 | 21 / 30 / 45 | 3.6 | 0.82 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 100 | 0.894 | +0.00 | 1.06 | 55 / 75 / 98 | 10.9 | 0.01 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 100 | 0.893 | +0.04 | 1.07 | 42 / 58 / 80 | 4.9 | 0.06 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 100 | 0.917 | +0.05 | 1.07 | 36 / 51 / 70 | 3.8 | 0.19 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 100 | 0.906 | -0.01 | 1.02 | 33 / 46 / 62 | 3.2 | 0.29 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 160 | 0.890 | -0.02 | 1.02 | 92 / 117 / 150 | 15.8 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 160 | 0.901 | +0.02 | 1.07 | 67 / 90 / 117 | 6.1 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 160 | 0.903 | -0.04 | 1.07 | 58 / 76 / 100 | 4.2 | 0.01 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 160 | 0.904 | +0.01 | 1.04 | 52 / 69 / 90 | 3.5 | 0.01 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 400 | 0.906 | -0.02 | 1.05 | 236 / 289 / 338 | 18.1 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 400 | 0.890 | -0.00 | 1.05 | 173 / 217 / 259 | 8.1 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 400 | 0.900 | +0.04 | 1.07 | 146 / 181 / 218 | 5.4 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 400 | 0.906 | -0.03 | 1.07 | 130 / 161 / 195 | 4.6 | 0.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 40 | 0.888 | -0.02 | 1.05 | 15 / 23 / 37 | 6.3 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 40 | 0.831 | +0.06 | 0.98 | 8 / 14 / 24 | 3.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 40 | 0.795 | -0.04 | 0.87 | 6 / 11 / 20 | 3.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 40 | 0.748 | -0.02 | 0.76 | 6 / 10 / 18 | 3.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 60 | 0.868 | -0.04 | 0.99 | 21 / 32 / 49 | 9.7 | 0.74 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 60 | 0.844 | -0.00 | 0.95 | 10 / 17 / 27 | 5.8 | 0.99 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 60 | 0.817 | +0.01 | 0.91 | 7 / 12 / 21 | 3.6 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 60 | 0.758 | -0.00 | 0.79 | 6 / 10 / 18 | 3.6 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 100 | 0.893 | +0.02 | 1.04 | 36 / 50 / 71 | 10.9 | 0.20 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 100 | 0.875 | +0.01 | 1.05 | 15 / 23 / 34 | 4.9 | 0.95 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 100 | 0.863 | +0.00 | 1.00 | 9 / 15 / 24 | 3.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 100 | 0.792 | -0.02 | 0.85 | 7 / 11 / 19 | 3.2 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 160 | 0.876 | -0.03 | 1.01 | 59 / 78 / 104 | 15.8 | 0.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 160 | 0.877 | +0.01 | 1.03 | 22 / 32 / 45 | 5.6 | 0.80 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 160 | 0.870 | -0.05 | 1.02 | 12 / 19 / 30 | 4.2 | 0.99 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 160 | 0.862 | -0.01 | 0.97 | 8 / 13 / 22 | 3.5 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 400 | 0.893 | -0.02 | 1.04 | 148 / 184 / 222 | 18.1 | 0.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 400 | 0.897 | +0.01 | 1.05 | 54 / 71 / 91 | 8.1 | 0.01 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 400 | 0.883 | -0.01 | 1.02 | 28 / 38 / 52 | 5.4 | 0.58 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 400 | 0.855 | +0.02 | 0.97 | 15 / 22 / 33 | 4.6 | 0.98 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 60 | 0.877 | +0.06 | 1.01 | 26 / 37 / 53 | 9.8 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 60 | 0.865 | +0.01 | 1.00 | 12 / 19 / 30 | 4.2 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 60 | 0.827 | -0.03 | 0.90 | 8 / 14 / 22 | 3.7 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 60 | 0.803 | -0.01 | 0.82 | 7 / 12 / 19 | 3.1 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 100 | 0.889 | +0.03 | 1.04 | 42 / 59 / 76 | 11.0 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 100 | 0.864 | -0.03 | 1.02 | 18 / 27 / 39 | 5.0 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 100 | 0.843 | -0.00 | 0.96 | 11 / 17 / 27 | 3.5 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 100 | 0.833 | -0.03 | 0.90 | 8 / 13 / 21 | 3.2 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 160 | 0.898 | +0.02 | 1.05 | 69 / 91 / 115 | 11.4 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 160 | 0.884 | -0.01 | 1.05 | 27 / 39 / 54 | 5.7 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 160 | 0.870 | +0.05 | 1.01 | 15 / 22 / 33 | 4.0 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 160 | 0.850 | -0.02 | 0.94 | 10 / 15 / 24 | 3.5 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 400 | 0.885 | -0.01 | 1.01 | 177 / 224 / 267 | 18.1 | 0.03 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 400 | 0.899 | -0.01 | 1.07 | 62 / 85 / 109 | 7.7 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 400 | 0.888 | +0.05 | 1.05 | 32 / 45 / 60 | 5.1 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 400 | 0.860 | +0.01 | 0.96 | 17 / 26 / 37 | 4.4 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 800 | 0.893 | +0.01 | 1.01 | 362 / 443 / 510 | 21.6 | 0.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 800 | 0.903 | -0.02 | 1.07 | 127 / 166 / 200 | 10.0 | 0.35 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 800 | 0.878 | +0.08 | 1.04 | 61 / 83 / 106 | 6.4 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 800 | 0.883 | -0.03 | 1.03 | 32 / 44 / 59 | 5.3 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 1600 | 0.894 | -0.01 | 1.00 | 740 / 901 / 999 | 30.8 | 0.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 1600 | 0.908 | +0.01 | 1.06 | 252 / 329 / 380 | 13.7 | 0.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 1600 | 0.906 | +0.04 | 1.08 | 119 / 159 / 192 | 8.5 | 0.45 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 1600 | 0.878 | -0.01 | 1.02 | 59 / 82 / 102 | 6.6 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 40 | 0.900 | -0.05 | 1.08 | 21 / 31 / 39 | 6.5 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 40 | 0.880 | -0.22 | 1.05 | 16 / 25 / 39 | 3.9 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 40 | 0.822 | -0.44 | 0.99 | 15 / 24 / 37 | 3.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 40 | 0.665 | -0.83 | 0.89 | 15 / 25 / 39 | 3.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 60 | 0.906 | -0.06 | 1.07 | 33 / 45 / 59 | 9.8 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 60 | 0.883 | -0.14 | 1.05 | 22 / 34 / 51 | 4.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 60 | 0.821 | -0.34 | 0.97 | 19 / 31 / 49 | 3.7 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 60 | 0.716 | -0.64 | 0.90 | 19 / 31 / 49 | 3.1 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 100 | 0.893 | -0.02 | 1.04 | 56 / 75 / 98 | 11.0 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 100 | 0.889 | -0.17 | 1.07 | 32 / 51 / 74 | 5.0 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 100 | 0.855 | -0.24 | 1.01 | 26 / 44 / 67 | 3.8 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 100 | 0.779 | -0.51 | 0.94 | 23 / 41 / 68 | 3.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 160 | 0.903 | -0.04 | 1.05 | 89 / 117 / 148 | 11.4 | 0.94 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 160 | 0.903 | -0.10 | 1.08 | 50 / 76 / 107 | 5.7 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 160 | 0.882 | -0.16 | 1.04 | 34 / 60 / 93 | 4.0 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 160 | 0.808 | -0.38 | 0.94 | 27 / 51 / 89 | 3.5 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 400 | 0.901 | -0.06 | 1.04 | 221 / 286 / 345 | 18.1 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 400 | 0.900 | -0.07 | 1.06 | 106 / 168 / 226 | 7.7 | 0.40 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 400 | 0.873 | -0.16 | 1.04 | 62 / 113 / 185 | 5.2 | 0.76 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 400 | 0.850 | -0.20 | 0.97 | 42 / 79 / 151 | 4.4 | 0.91 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 800 | 0.904 | -0.00 | 1.03 | 457 / 573 / 656 | 23.5 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 800 | 0.910 | -0.07 | 1.07 | 205 / 323 / 415 | 10.4 | 0.02 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 800 | 0.895 | -0.11 | 1.05 | 116 / 198 / 319 | 6.5 | 0.29 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 800 | 0.868 | -0.17 | 1.04 | 67 / 126 / 244 | 5.3 | 0.65 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 1600 | 0.902 | -0.02 | 1.02 | 942 / 1152 / 1273 | 30.8 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 1600 | 0.905 | -0.03 | 1.04 | 410 / 608 / 784 | 14.2 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 1600 | 0.894 | -0.11 | 1.06 | 212 / 350 / 576 | 8.8 | 0.02 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 1600 | 0.893 | -0.11 | 1.04 | 117 / 204 / 388 | 6.7 | 0.26 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 40 | 0.887 | -0.05 | 1.06 | 16 / 25 / 37 | 6.3 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 40 | 0.877 | -0.08 | 1.01 | 11 / 18 / 28 | 3.8 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 40 | 0.875 | -0.21 | 1.00 | 10 / 16 / 25 | 3.8 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 40 | 0.823 | -0.40 | 0.94 | 10 / 16 / 24 | 3.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 60 | 0.897 | -0.04 | 1.05 | 25 / 36 / 51 | 9.7 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 60 | 0.893 | -0.09 | 1.04 | 15 / 23 / 35 | 4.1 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 60 | 0.886 | -0.13 | 1.01 | 13 / 20 / 31 | 3.6 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 60 | 0.841 | -0.32 | 0.95 | 12 / 20 / 29 | 3.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 100 | 0.901 | +0.01 | 1.07 | 42 / 58 / 77 | 10.9 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 100 | 0.902 | -0.03 | 1.08 | 24 / 35 / 51 | 4.9 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 100 | 0.897 | -0.11 | 1.08 | 18 / 28 / 42 | 3.8 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 100 | 0.862 | -0.21 | 0.96 | 16 / 26 / 38 | 3.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 160 | 0.891 | +0.01 | 1.03 | 69 / 91 / 117 | 11.3 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 160 | 0.906 | -0.01 | 1.09 | 35 / 52 / 71 | 5.6 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 160 | 0.890 | -0.08 | 1.09 | 26 / 39 / 56 | 4.0 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 160 | 0.870 | -0.22 | 1.01 | 21 / 33 / 51 | 3.5 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 400 | 0.895 | +0.01 | 1.03 | 178 / 222 / 267 | 18.1 | 0.02 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 400 | 0.912 | -0.10 | 1.09 | 87 / 116 / 148 | 7.7 | 0.94 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 400 | 0.886 | -0.04 | 1.04 | 52 / 81 / 108 | 5.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 400 | 0.900 | -0.12 | 1.04 | 35 / 60 / 92 | 4.4 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 800 | 0.887 | -0.04 | 0.99 | 373 / 445 / 510 | 23.5 | 0.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 800 | 0.912 | -0.01 | 1.08 | 172 / 228 / 275 | 10.4 | 0.05 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 800 | 0.894 | -0.01 | 1.09 | 94 / 148 / 193 | 6.7 | 0.57 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 800 | 0.895 | -0.06 | 1.06 | 54 / 99 / 150 | 5.3 | 0.92 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 1600 | 0.911 | +0.00 | 1.03 | 759 / 892 / 991 | 30.7 | 0.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 1600 | 0.904 | -0.06 | 1.06 | 345 / 447 / 518 | 14.1 | 0.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 1600 | 0.900 | -0.08 | 1.07 | 180 / 282 / 354 | 8.8 | 0.05 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 1600 | 0.894 | -0.04 | 1.08 | 98 / 173 / 265 | 6.7 | 0.41 |
