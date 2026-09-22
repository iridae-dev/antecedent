# Temporal Pulse / Sustained without replicates — 1.9.0

Established: 2026-09-14 (the first measurement date the note below states)
Machine class: macOS arm64 development host, as the measurement note below states; the chip model was not written down
Commit: cb15e6c1 (the commit that added this file; the measured commit was not written down)

Measured 2026-09-14 on this macOS arm64 development host with the release
profile, min of 5 runs, three interleaved rounds against the pre-fix build
(other workloads were running; the interleaving keeps the ratios honest, the
absolute values are within about 10%). One `TemporalDag` (`x_{t-1} -> y_t`),
Frequentist, `Study::run` end to end. Tiers: `I` is `LatencyMode::Interactive`
(zero replicates, cheap validators); `B0` is zero replicates with the default
placebo + RCC suite; `B199` is 199 replicates with that suite. These are local
measurements, not portable CI thresholds.

| Workload | Before | After |
|---|---|---|
| `pulse_n2000\|I` | 0.595 ms | **0.114 ms** |
| `pulse_n2000\|B0` | 4.01 ms | **3.49 ms** |
| `pulse_n2000\|B199` | 9.57 ms | **9.30 ms** |
| `sustained3_n2000\|I` | 0.632 ms | **0.145 ms** |
| `sustained3_n2000\|B0` | 21.2 ms | **5.24 ms** |
| `sustained3_n2000\|B199` | 28.5 ms | **11.2 ms** |
| `pulse_n100k\|I` | 178 ms | **8.9 ms** |
| `pulse_n100k\|B0` | 379 ms | **205 ms** |
| `pulse_n100k\|B199` | 798 ms | **730 ms** |
| `sustained3_n100k\|I` | 178 ms | **10.4 ms** |
| `sustained3_n100k\|B0` | 5100 ms | **317 ms** |
| `sustained3_n100k\|B199` | 5495 ms | **785 ms** |

What moved: the dependence-aware circular-block length (a refit for every
normal-equation score plus a Politis–White scan per score, O(n^1.5) in the
series length) was computed before the `replicates > 0` branch on the
single-window, sequential and mediation paths, so a zero-replicate estimate
paid for an interval it never published, and every placebo / RCC refit paid
again. The length is now only derived when replicates are drawn (the rule
length is reported otherwise, and the diagnostic says so), the Politis–White
scan takes autocorrelations lag by lag as the bandwidth search asks for them
(a short-memory score needs `K_n` lags, not `√n + K_n`), and the
`bootstrap.ci_coverage` refuter reuses the length carried on a bootstrapped
estimate (`EffectEstimate::block_resampling`) instead of re-deriving it.

Run `cargo bench -p antecedent --bench temporal_zero_replicates`
(`ANTECEDENT_BENCH_CELLS` selects cells, `ANTECEDENT_BENCH_REPEATS` the min-of
count). The release gate runs it with `--test`, which times the two Interactive
cells at n = 20 000 (`pulse_n20k|I` 1.29 ms, `sustained3_n20k|I` 1.63 ms after
the fix) against a soft budget of 5 ms: about 3× the fixed values, and about a
third of what the eager O(n^1.5) score scan alone costs at that length (the
178 ms it cost at n = 100 000 scales to roughly 16 ms).
