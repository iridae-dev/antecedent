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
single-step Sustained), the Total, Direct and Mediated persistence probes
(temporal mediation: each mechanism residual times the centred treatment, and
for the mediated path the centred mediator; the partialled influence functions
size the blocks but predict under-coverage poorly, because partialling on
lagged design columns removes the treatment's persistence from them while the
interval still under-covers — on the same sweep they would need a threshold of
70, which also warns on about 20 nominally covering cells), the contrast influence (multi-step Sustained sequential
g-computation), and every atom's influence plus the weighted mixture score
(class envelopes and DBN posteriors). The provenance diagnostic prints the
statistic; the warning fires below the threshold of the interval's family.

| family | SE path | threshold |
|---|---|---|
| single-window adjustment | TemporalDag Pulse / single-step Sustained | 45 |
| temporal mediation | Total, Direct, Mediated (shared replicate) | 40 |
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
| single-window adjustment | 45 | 40 | 45 |
| temporal mediation | 35 | 40 | 40 |
| multi-step sequential | 25 | 30 | 40 |
| multi-atom mixture | 155 | 90 | 155 |

The multi-step sequential threshold is not the rule's 30: it stays at the 40
an earlier measurement of the same sweep set (before the SE construction
changes listed under the table), which warns on more replicates, not fewer.
The mediation threshold rose from 35 to 40 with that re-measurement: in the
replication, the AR(1)-treatment Total at ρ = 0.9, n = 160 covered 0.846 and
warned on fewer than 90% of its replicates at 35.

### What the failure line tolerates

The failure line 0.855 is the calibration gate's tolerance, not a coverage
target. "Nominal" in the gate means the empirical coverage is within ±3 Monte
Carlo SEs of the level at the gate's replicate count: at 400 replicates and a
90% level that is ±4.5 points, the band [0.855, 0.945]. The thresholds are set
so that cells *below that band* warn; a cell between 0.855 and roughly 0.88
is not caught by the warning and can still pass a 400-replicate gate, even
though at 2000 replicates (band [0.880, 0.920], ±2 points) it measures
significantly low. Such cells are in the table below, for example the
AR(1)-treatment Pulse at ρ = 0.9, n = 400 (0.876, warned on 32%), the
AR(1)-treatment mediation Direct at ρ = 0.95, n = 400 (0.864), the
AR(1)-treatment multi-step Sustained at ρ = 0.5, n = 100–160 (0.867–0.870,
warned on at most 19%), and the TemporalCpdag mixture at ρ = 0.9, n = 400
(0.875, warned on 76%). The last is also a gated design; its gate
(`frequentist_temporal_cpdag_pulse_ar1_rho09_n400_boundary_within_band`) is
named as a boundary cell and asserts only the 400-replicate band.

The mixture threshold is set by a different failure. Its SE-driven cells (the
six-completion TemporalPag envelope at ρ ≥ 0.9, n ≤ 160, and the DBN mixture at
ρ = 0.95, n ≤ 60) warn from about 30.
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

Measurement conditions: blocks sized on every estimating score (the target
influence(s), the mixture score, and every normal-equation score of every
fitted regression, residuals included), the circular-Bartlett fixed-b factor
(`antecedent_estimate::circular_fixed_b_scale`), and fixtures whose noise
streams are seeded independently per replicate. An earlier run of the same
sweep, with the non-circular Kiefer–Vogelsang factor and a fixture seeding
that let pairs of replicates share a noise path, differed from this table by
at most 1.4 coverage points (0.45 on average) in the designs that draw the
same datasets under both (MA(3) treatment, confounded DAG, TemporalCpdag,
DBN), where only the SE changed, and by at most 3.2 points (1.1 on average)
in the re-seeded persistent designs (AR(1) treatment, TemporalPag).

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
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 40 | 0.875 | -0.00 | 1.02 | 20 / 31 / 39 | 6.5 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 40 | 0.871 | +0.05 | 1.03 | 16 / 26 / 39 | 3.9 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 40 | 0.896 | -0.03 | 1.05 | 15 / 24 / 39 | 3.2 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 40 | 0.878 | +0.02 | 0.99 | 14 / 23 / 38 | 3.2 | 1.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 60 | 0.885 | +0.01 | 1.03 | 30 / 43 / 59 | 9.8 | 0.56 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 60 | 0.879 | -0.00 | 1.05 | 23 / 36 / 55 | 4.2 | 0.75 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 60 | 0.892 | -0.01 | 1.02 | 21 / 32 / 50 | 3.7 | 0.82 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 60 | 0.899 | +0.03 | 0.99 | 20 / 32 / 50 | 3.1 | 0.83 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 100 | 0.878 | +0.01 | 1.02 | 49 / 69 / 93 | 11.0 | 0.05 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 100 | 0.908 | -0.00 | 1.05 | 36 / 53 / 78 | 5.0 | 0.27 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 100 | 0.905 | -0.00 | 1.06 | 34 / 49 / 72 | 3.8 | 0.38 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 100 | 0.912 | -0.02 | 1.08 | 33 / 48 / 69 | 3.2 | 0.42 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 160 | 0.895 | +0.01 | 1.03 | 80 / 106 / 138 | 15.9 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 160 | 0.898 | +0.02 | 1.07 | 56 / 82 / 111 | 5.7 | 0.02 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 160 | 0.902 | +0.04 | 1.08 | 53 / 76 / 105 | 4.0 | 0.04 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 160 | 0.919 | +0.01 | 1.09 | 50 / 71 / 98 | 3.5 | 0.06 |
| single-window | Pulse h=1, MA(3) treatment | 0.50 | 400 | 0.885 | -0.00 | 0.98 | 206 / 255 / 304 | 18.1 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.80 | 400 | 0.898 | -0.02 | 1.05 | 148 / 195 / 242 | 8.1 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.90 | 400 | 0.901 | +0.00 | 1.08 | 134 / 178 / 225 | 5.2 | 0.00 |
| single-window | Pulse h=1, MA(3) treatment | 0.95 | 400 | 0.907 | -0.03 | 1.09 | 125 / 170 / 213 | 4.4 | 0.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 40 | 0.879 | -0.02 | 1.02 | 20 / 30 / 39 | 9.8 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 40 | 0.832 | +0.05 | 0.94 | 12 / 19 / 32 | 6.5 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 40 | 0.806 | +0.04 | 0.87 | 9 / 16 / 27 | 3.9 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 40 | 0.750 | +0.01 | 0.78 | 9 / 15 / 26 | 3.9 | 1.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 60 | 0.875 | +0.04 | 1.01 | 29 / 42 / 59 | 9.8 | 0.58 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 60 | 0.846 | +0.02 | 0.98 | 14 / 23 / 38 | 5.9 | 0.96 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 60 | 0.808 | +0.01 | 0.87 | 10 / 18 / 30 | 4.2 | 0.99 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 60 | 0.786 | +0.01 | 0.84 | 9 / 16 / 26 | 3.7 | 0.99 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 100 | 0.886 | +0.02 | 1.03 | 48 / 66 / 90 | 11.0 | 0.07 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 100 | 0.867 | -0.05 | 1.03 | 21 / 32 / 47 | 5.0 | 0.87 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 100 | 0.841 | -0.01 | 0.95 | 14 / 21 / 33 | 4.1 | 0.98 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 100 | 0.825 | +0.03 | 0.89 | 10 / 17 / 28 | 3.5 | 0.99 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 160 | 0.883 | -0.02 | 1.01 | 75 / 100 / 130 | 15.9 | 0.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 160 | 0.889 | -0.01 | 1.02 | 30 / 44 / 62 | 6.1 | 0.54 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 160 | 0.854 | +0.01 | 0.99 | 18 / 27 / 41 | 4.2 | 0.94 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 160 | 0.844 | -0.01 | 0.94 | 12 / 19 / 30 | 3.8 | 0.99 |
| single-window | Pulse h=1, AR(1) treatment | 0.50 | 400 | 0.894 | -0.03 | 1.03 | 192 / 239 / 287 | 18.1 | 0.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.80 | 400 | 0.891 | -0.01 | 1.06 | 71 / 94 / 121 | 8.1 | 0.00 |
| single-window | Pulse h=1, AR(1) treatment | 0.90 | 400 | 0.876 | -0.01 | 1.04 | 36 / 50 / 69 | 5.4 | 0.32 |
| single-window | Pulse h=1, AR(1) treatment | 0.95 | 400 | 0.853 | +0.01 | 0.97 | 21 / 30 / 44 | 4.6 | 0.91 |
| mediation | mediation, MA(3) treatment | 0.50 | 40 | Total 0.881, Direct 0.903, Mediated 0.911 | -0.04 | 1.06 | 19 / 28 / 39 | 6.5 | 1.00 |
| mediation | mediation, MA(3) treatment | 0.80 | 40 | Total 0.880, Direct 0.896, Mediated 0.917 | +0.02 | 1.05 | 16 / 24 / 36 | 3.9 | 1.00 |
| mediation | mediation, MA(3) treatment | 0.90 | 40 | Total 0.894, Direct 0.907, Mediated 0.928 | +0.01 | 1.08 | 15 / 23 / 35 | 3.2 | 1.00 |
| mediation | mediation, MA(3) treatment | 0.95 | 40 | Total 0.914, Direct 0.909, Mediated 0.928 | -0.03 | 1.13 | 14 / 22 / 35 | 3.2 | 1.00 |
| mediation | mediation, MA(3) treatment | 0.50 | 60 | Total 0.895, Direct 0.892, Mediated 0.898 | +0.02 | 1.08 | 30 / 42 / 56 | 9.8 | 0.45 |
| mediation | mediation, MA(3) treatment | 0.80 | 60 | Total 0.889, Direct 0.894, Mediated 0.908 | -0.04 | 1.07 | 22 / 34 / 50 | 4.2 | 0.71 |
| mediation | mediation, MA(3) treatment | 0.90 | 60 | Total 0.900, Direct 0.910, Mediated 0.928 | +0.01 | 1.07 | 21 / 32 / 48 | 3.3 | 0.78 |
| mediation | mediation, MA(3) treatment | 0.95 | 60 | Total 0.908, Direct 0.902, Mediated 0.924 | +0.04 | 1.08 | 21 / 31 / 46 | 3.1 | 0.80 |
| mediation | mediation, MA(3) treatment | 0.50 | 100 | Total 0.886, Direct 0.897, Mediated 0.899 | -0.03 | 1.04 | 48 / 66 / 86 | 11.0 | 0.02 |
| mediation | mediation, MA(3) treatment | 0.80 | 100 | Total 0.902, Direct 0.900, Mediated 0.913 | -0.01 | 1.11 | 37 / 53 / 74 | 5.0 | 0.17 |
| mediation | mediation, MA(3) treatment | 0.90 | 100 | Total 0.898, Direct 0.918, Mediated 0.919 | -0.02 | 1.09 | 33 / 49 / 70 | 3.5 | 0.25 |
| mediation | mediation, MA(3) treatment | 0.95 | 100 | Total 0.914, Direct 0.913, Mediated 0.921 | +0.01 | 1.12 | 32 / 48 / 68 | 3.0 | 0.28 |
| mediation | mediation, MA(3) treatment | 0.50 | 160 | Total 0.885, Direct 0.893, Mediated 0.909 | +0.02 | 1.04 | 77 / 103 / 132 | 11.4 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.80 | 160 | Total 0.897, Direct 0.904, Mediated 0.920 | +0.01 | 1.08 | 58 / 80 / 107 | 5.7 | 0.01 |
| mediation | mediation, MA(3) treatment | 0.90 | 160 | Total 0.913, Direct 0.897, Mediated 0.919 | +0.03 | 1.10 | 52 / 75 / 103 | 4.0 | 0.01 |
| mediation | mediation, MA(3) treatment | 0.95 | 160 | Total 0.904, Direct 0.907, Mediated 0.920 | +0.02 | 1.05 | 51 / 72 / 101 | 3.4 | 0.02 |
| mediation | mediation, MA(3) treatment | 0.50 | 400 | Total 0.902, Direct 0.896, Mediated 0.903 | +0.02 | 1.03 | 202 / 252 / 302 | 18.1 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.80 | 400 | Total 0.900, Direct 0.894, Mediated 0.901 | -0.04 | 1.09 | 146 / 194 / 239 | 7.7 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.90 | 400 | Total 0.898, Direct 0.900, Mediated 0.901 | -0.01 | 1.07 | 132 / 177 / 221 | 5.2 | 0.00 |
| mediation | mediation, MA(3) treatment | 0.95 | 400 | Total 0.906, Direct 0.899, Mediated 0.925 | +0.04 | 1.09 | 127 / 172 / 215 | 4.4 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.50 | 40 | Total 0.889, Direct 0.890, Mediated 0.917 | +0.01 | 1.04 | 18 / 27 / 39 | 6.5 | 1.00 |
| mediation | mediation, AR(1) treatment | 0.80 | 40 | Total 0.853, Direct 0.864, Mediated 0.928 | -0.01 | 0.98 | 11 / 17 / 28 | 3.9 | 1.00 |
| mediation | mediation, AR(1) treatment | 0.90 | 40 | Total 0.816, Direct 0.825, Mediated 0.933 | -0.02 | 0.92 | 8 / 13 / 23 | 3.9 | 1.00 |
| mediation | mediation, AR(1) treatment | 0.95 | 40 | Total 0.791, Direct 0.773, Mediated 0.945 | -0.01 | 0.84 | 7 / 12 / 22 | 3.2 | 1.00 |
| mediation | mediation, AR(1) treatment | 0.50 | 60 | Total 0.874, Direct 0.892, Mediated 0.897 | -0.02 | 1.02 | 27 / 38 / 54 | 9.8 | 0.55 |
| mediation | mediation, AR(1) treatment | 0.80 | 60 | Total 0.857, Direct 0.869, Mediated 0.913 | +0.02 | 0.98 | 13 / 21 / 33 | 4.2 | 0.97 |
| mediation | mediation, AR(1) treatment | 0.90 | 60 | Total 0.849, Direct 0.849, Mediated 0.935 | -0.03 | 0.97 | 9 / 15 / 26 | 3.7 | 0.99 |
| mediation | mediation, AR(1) treatment | 0.95 | 60 | Total 0.805, Direct 0.809, Mediated 0.943 | -0.03 | 0.86 | 7 / 13 / 23 | 3.3 | 1.00 |
| mediation | mediation, AR(1) treatment | 0.50 | 100 | Total 0.891, Direct 0.896, Mediated 0.905 | +0.01 | 1.03 | 46 / 62 / 83 | 11.0 | 0.05 |
| mediation | mediation, AR(1) treatment | 0.80 | 100 | Total 0.882, Direct 0.888, Mediated 0.912 | +0.02 | 1.07 | 20 / 30 / 45 | 5.0 | 0.83 |
| mediation | mediation, AR(1) treatment | 0.90 | 100 | Total 0.863, Direct 0.855, Mediated 0.932 | -0.00 | 1.01 | 12 / 19 / 30 | 3.8 | 0.98 |
| mediation | mediation, AR(1) treatment | 0.95 | 100 | Total 0.845, Direct 0.846, Mediated 0.938 | -0.00 | 0.94 | 8 / 14 / 24 | 3.2 | 0.99 |
| mediation | mediation, AR(1) treatment | 0.50 | 160 | Total 0.895, Direct 0.882, Mediated 0.897 | +0.02 | 1.04 | 73 / 97 / 125 | 11.4 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.80 | 160 | Total 0.873, Direct 0.894, Mediated 0.906 | +0.01 | 1.06 | 30 / 42 / 60 | 5.7 | 0.44 |
| mediation | mediation, AR(1) treatment | 0.90 | 160 | Total 0.870, Direct 0.874, Mediated 0.926 | -0.05 | 1.04 | 16 / 25 / 37 | 4.0 | 0.93 |
| mediation | mediation, AR(1) treatment | 0.95 | 160 | Total 0.846, Direct 0.853, Mediated 0.938 | +0.02 | 0.98 | 10 / 17 / 27 | 3.5 | 0.99 |
| mediation | mediation, AR(1) treatment | 0.50 | 400 | Total 0.896, Direct 0.908, Mediated 0.919 | -0.00 | 1.02 | 184 / 235 / 282 | 18.1 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.80 | 400 | Total 0.886, Direct 0.901, Mediated 0.897 | -0.03 | 1.04 | 70 / 93 / 119 | 8.1 | 0.00 |
| mediation | mediation, AR(1) treatment | 0.90 | 400 | Total 0.884, Direct 0.885, Mediated 0.916 | -0.01 | 1.05 | 36 / 49 / 67 | 5.2 | 0.21 |
| mediation | mediation, AR(1) treatment | 0.95 | 400 | Total 0.870, Direct 0.864, Mediated 0.912 | +0.00 | 1.02 | 20 / 29 / 42 | 4.6 | 0.87 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 40 | 0.900 | -0.02 | 1.08 | 22 / 32 / 38 | 6.3 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 40 | 0.906 | +0.04 | 1.09 | 18 / 27 / 38 | 6.3 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 40 | 0.898 | +0.02 | 1.03 | 16 / 24 / 38 | 3.8 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 40 | 0.895 | -0.01 | 1.04 | 15 / 23 / 36 | 3.8 | 1.00 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 60 | 0.909 | +0.02 | 1.09 | 33 / 47 / 58 | 9.7 | 0.28 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 60 | 0.891 | +0.01 | 1.07 | 26 / 38 / 56 | 5.8 | 0.58 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 60 | 0.913 | -0.00 | 1.09 | 22 / 33 / 48 | 3.6 | 0.74 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 60 | 0.926 | +0.01 | 1.09 | 21 / 30 / 45 | 3.6 | 0.82 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 100 | 0.891 | +0.00 | 1.05 | 55 / 75 / 98 | 10.9 | 0.01 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 100 | 0.898 | +0.04 | 1.09 | 42 / 58 / 80 | 4.9 | 0.06 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 100 | 0.922 | +0.05 | 1.09 | 36 / 51 / 70 | 3.8 | 0.19 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 100 | 0.918 | -0.01 | 1.05 | 33 / 46 / 62 | 3.2 | 0.29 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 160 | 0.888 | -0.02 | 1.01 | 92 / 117 / 150 | 15.8 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 160 | 0.904 | +0.02 | 1.08 | 67 / 90 / 117 | 6.1 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 160 | 0.910 | -0.04 | 1.09 | 58 / 76 / 100 | 4.2 | 0.01 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 160 | 0.906 | +0.01 | 1.07 | 52 / 69 / 90 | 3.5 | 0.01 |
| sequential | multi-step Sustained, confounded DAG | 0.50 | 400 | 0.903 | -0.02 | 1.04 | 236 / 289 / 338 | 18.1 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.80 | 400 | 0.887 | -0.00 | 1.04 | 173 / 217 / 259 | 8.1 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.90 | 400 | 0.902 | +0.04 | 1.07 | 146 / 181 / 218 | 5.4 | 0.00 |
| sequential | multi-step Sustained, confounded DAG | 0.95 | 400 | 0.909 | -0.03 | 1.08 | 130 / 161 / 195 | 4.6 | 0.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 40 | 0.874 | +0.02 | 1.04 | 15 / 23 / 37 | 6.3 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 40 | 0.829 | +0.00 | 0.96 | 8 / 14 / 24 | 4.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 40 | 0.806 | +0.00 | 0.88 | 6 / 11 / 20 | 3.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 40 | 0.764 | -0.03 | 0.82 | 6 / 10 / 18 | 3.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 60 | 0.875 | +0.01 | 1.00 | 22 / 33 / 50 | 9.7 | 0.72 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 60 | 0.858 | +0.00 | 0.99 | 10 / 16 / 27 | 5.8 | 0.99 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 60 | 0.813 | +0.01 | 0.90 | 7 / 12 / 20 | 3.6 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 60 | 0.784 | +0.01 | 0.85 | 6 / 10 / 18 | 3.6 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 100 | 0.870 | +0.00 | 1.01 | 35 / 51 / 70 | 10.9 | 0.20 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 100 | 0.869 | -0.03 | 1.03 | 14 / 22 / 34 | 4.9 | 0.96 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 100 | 0.852 | +0.05 | 1.03 | 9 / 15 / 24 | 3.8 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 100 | 0.813 | +0.04 | 0.92 | 6 / 11 / 19 | 3.2 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 160 | 0.867 | +0.01 | 0.98 | 58 / 77 / 102 | 13.2 | 0.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 160 | 0.875 | -0.02 | 1.02 | 22 / 32 / 45 | 5.6 | 0.78 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 160 | 0.857 | -0.01 | 1.00 | 12 / 20 / 30 | 4.2 | 0.99 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 160 | 0.853 | -0.02 | 0.97 | 8 / 13 / 21 | 3.5 | 1.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.50 | 400 | 0.896 | +0.01 | 1.03 | 151 / 184 / 225 | 18.1 | 0.00 |
| sequential | multi-step Sustained, AR(1) treatment | 0.80 | 400 | 0.889 | -0.02 | 1.05 | 54 / 71 / 91 | 8.1 | 0.01 |
| sequential | multi-step Sustained, AR(1) treatment | 0.90 | 400 | 0.892 | +0.04 | 1.07 | 27 / 37 / 51 | 5.4 | 0.62 |
| sequential | multi-step Sustained, AR(1) treatment | 0.95 | 400 | 0.881 | +0.03 | 1.04 | 15 / 22 / 32 | 4.6 | 0.98 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 60 | 0.877 | +0.01 | 1.03 | 26 / 37 / 52 | 9.8 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 60 | 0.863 | -0.03 | 1.01 | 12 / 19 / 29 | 4.2 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 60 | 0.839 | -0.01 | 0.93 | 8 / 14 / 22 | 3.7 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 60 | 0.804 | -0.01 | 0.83 | 7 / 12 / 19 | 3.3 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 100 | 0.875 | -0.03 | 1.02 | 43 / 59 / 78 | 11.0 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 100 | 0.875 | +0.04 | 1.03 | 18 / 27 / 39 | 5.0 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 100 | 0.858 | -0.01 | 1.01 | 11 / 17 / 27 | 3.5 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 100 | 0.822 | +0.01 | 0.89 | 8 / 13 / 21 | 3.2 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 160 | 0.898 | +0.03 | 1.03 | 69 / 91 / 116 | 11.4 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 160 | 0.890 | +0.02 | 1.06 | 27 / 38 / 53 | 5.1 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 160 | 0.883 | -0.00 | 1.06 | 15 / 22 / 33 | 3.8 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 160 | 0.852 | +0.03 | 0.96 | 10 / 16 / 24 | 3.5 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 400 | 0.891 | +0.02 | 1.03 | 176 / 223 / 266 | 18.1 | 0.03 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 400 | 0.893 | +0.00 | 1.05 | 65 / 86 / 110 | 7.7 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 400 | 0.871 | +0.00 | 1.00 | 32 / 45 / 60 | 5.1 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 400 | 0.883 | -0.01 | 1.04 | 17 / 25 / 37 | 4.4 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 800 | 0.899 | -0.00 | 1.01 | 362 / 448 / 510 | 21.6 | 0.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 800 | 0.894 | +0.06 | 1.03 | 127 / 167 / 199 | 10.0 | 0.34 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 800 | 0.897 | -0.02 | 1.05 | 61 / 82 / 104 | 6.4 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 800 | 0.878 | -0.04 | 1.02 | 32 / 44 / 58 | 5.3 | 1.00 |
| mixture | TemporalPag Pulse, six completions | 0.50 | 1600 | 0.883 | +0.01 | 0.99 | 732 / 901 / 1005 | 30.8 | 0.00 |
| mixture | TemporalPag Pulse, six completions | 0.80 | 1600 | 0.900 | +0.04 | 1.05 | 254 / 333 / 381 | 13.7 | 0.00 |
| mixture | TemporalPag Pulse, six completions | 0.90 | 1600 | 0.893 | +0.03 | 1.05 | 121 / 160 / 192 | 8.5 | 0.42 |
| mixture | TemporalPag Pulse, six completions | 0.95 | 1600 | 0.889 | +0.02 | 1.02 | 60 / 81 / 101 | 6.6 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 40 | 0.899 | -0.05 | 1.08 | 21 / 31 / 39 | 6.5 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 40 | 0.886 | -0.22 | 1.07 | 16 / 25 / 39 | 3.9 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 40 | 0.831 | -0.44 | 1.02 | 15 / 24 / 37 | 3.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 40 | 0.675 | -0.83 | 0.92 | 15 / 25 / 39 | 3.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 60 | 0.904 | -0.06 | 1.07 | 33 / 45 / 59 | 9.8 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 60 | 0.888 | -0.14 | 1.06 | 22 / 34 / 51 | 4.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 60 | 0.832 | -0.34 | 0.99 | 19 / 31 / 49 | 3.7 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 60 | 0.726 | -0.64 | 0.93 | 19 / 31 / 49 | 3.1 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 100 | 0.890 | -0.02 | 1.03 | 56 / 75 / 98 | 11.0 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 100 | 0.892 | -0.17 | 1.09 | 32 / 51 / 74 | 5.0 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 100 | 0.864 | -0.24 | 1.04 | 26 / 44 / 67 | 3.8 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 100 | 0.787 | -0.51 | 0.98 | 23 / 41 / 68 | 3.2 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 160 | 0.899 | -0.04 | 1.05 | 89 / 117 / 148 | 11.4 | 0.94 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 160 | 0.906 | -0.10 | 1.09 | 50 / 76 / 107 | 5.7 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 160 | 0.887 | -0.16 | 1.06 | 34 / 60 / 93 | 4.0 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 160 | 0.821 | -0.38 | 0.97 | 27 / 51 / 89 | 3.5 | 1.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 400 | 0.899 | -0.06 | 1.03 | 221 / 286 / 345 | 18.1 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 400 | 0.896 | -0.07 | 1.06 | 106 / 168 / 226 | 7.7 | 0.40 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 400 | 0.875 | -0.16 | 1.04 | 62 / 113 / 185 | 5.2 | 0.76 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 400 | 0.853 | -0.20 | 0.98 | 42 / 79 / 151 | 4.4 | 0.91 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 800 | 0.900 | -0.00 | 1.02 | 457 / 573 / 656 | 23.5 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 800 | 0.904 | -0.07 | 1.06 | 205 / 323 / 415 | 10.4 | 0.02 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 800 | 0.893 | -0.11 | 1.05 | 116 / 198 / 319 | 6.5 | 0.29 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 800 | 0.868 | -0.17 | 1.04 | 67 / 126 / 244 | 5.3 | 0.65 |
| mixture | TemporalCpdag Pulse, two completions | 0.50 | 1600 | 0.898 | -0.02 | 1.02 | 942 / 1152 / 1273 | 30.8 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.80 | 1600 | 0.901 | -0.03 | 1.03 | 410 / 608 / 784 | 14.2 | 0.00 |
| mixture | TemporalCpdag Pulse, two completions | 0.90 | 1600 | 0.894 | -0.11 | 1.05 | 212 / 350 / 576 | 8.8 | 0.02 |
| mixture | TemporalCpdag Pulse, two completions | 0.95 | 1600 | 0.891 | -0.11 | 1.04 | 117 / 204 / 388 | 6.7 | 0.26 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 40 | 0.886 | -0.05 | 1.06 | 16 / 25 / 37 | 6.3 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 40 | 0.878 | -0.08 | 1.03 | 11 / 18 / 28 | 3.8 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 40 | 0.879 | -0.21 | 1.02 | 10 / 16 / 25 | 3.8 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 40 | 0.833 | -0.40 | 0.96 | 10 / 16 / 24 | 3.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 60 | 0.895 | -0.04 | 1.05 | 25 / 36 / 51 | 9.7 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 60 | 0.898 | -0.09 | 1.06 | 15 / 23 / 35 | 4.1 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 60 | 0.891 | -0.13 | 1.03 | 13 / 20 / 31 | 3.6 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 60 | 0.851 | -0.32 | 0.98 | 12 / 20 / 29 | 3.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 100 | 0.901 | +0.01 | 1.06 | 42 / 58 / 77 | 10.9 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 100 | 0.904 | -0.03 | 1.10 | 24 / 35 / 51 | 4.9 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 100 | 0.901 | -0.11 | 1.11 | 18 / 28 / 42 | 3.8 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 100 | 0.872 | -0.21 | 0.99 | 16 / 26 / 38 | 3.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 160 | 0.889 | +0.01 | 1.03 | 69 / 91 / 117 | 11.3 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 160 | 0.908 | -0.01 | 1.10 | 35 / 52 / 71 | 5.6 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 160 | 0.894 | -0.08 | 1.11 | 26 / 39 / 56 | 4.0 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 160 | 0.879 | -0.22 | 1.03 | 21 / 33 / 51 | 3.5 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 400 | 0.892 | +0.01 | 1.02 | 178 / 222 / 267 | 18.1 | 0.02 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 400 | 0.911 | -0.10 | 1.08 | 87 / 116 / 148 | 7.7 | 0.94 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 400 | 0.885 | -0.04 | 1.04 | 52 / 81 / 108 | 5.2 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 400 | 0.901 | -0.12 | 1.05 | 35 / 60 / 92 | 4.4 | 1.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 800 | 0.887 | -0.04 | 0.99 | 373 / 445 / 510 | 23.5 | 0.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 800 | 0.910 | -0.01 | 1.07 | 172 / 228 / 275 | 10.4 | 0.05 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 800 | 0.894 | -0.01 | 1.08 | 94 / 148 / 193 | 6.7 | 0.57 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 800 | 0.897 | -0.06 | 1.07 | 54 / 99 / 150 | 5.3 | 0.92 |
| mixture | DBN multi-step Sustained, two atoms | 0.50 | 1600 | 0.907 | +0.00 | 1.03 | 759 / 892 / 991 | 30.7 | 0.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.80 | 1600 | 0.902 | -0.06 | 1.05 | 345 / 447 / 518 | 14.1 | 0.00 |
| mixture | DBN multi-step Sustained, two atoms | 0.90 | 1600 | 0.898 | -0.08 | 1.07 | 180 / 282 / 354 | 8.8 | 0.05 |
| mixture | DBN multi-step Sustained, two atoms | 0.95 | 1600 | 0.893 | -0.04 | 1.08 | 98 / 173 / 265 | 6.7 | 0.41 |
