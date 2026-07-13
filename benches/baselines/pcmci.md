# PCMCI benchmark baseline (Phase 2)

Workload: `pcmci_n500_p4_lag2` — lagged PCMCI (PC parents + MCI, FDR off,
`max_cond_size=1`, `max_lag=2`) on a synthetic 4-variable series of length 500.

Established: 2026-07-21
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 10

## Accepted measurement

| Metric | Value |
|--------|-------|
| mean wall time | **3.16 ms** |
| CI (lower / upper) | 3.15 ms / 3.17 ms |

## Acceptance

Regressions exceeding **20%** wall-time vs the last accepted Criterion run on
the same machine class require an approved explanation and replacement baseline
(DESIGN.md §28.9). Gate: mean ≤ **3.79 ms** (20% over 3.16 ms).

## How to refresh

```bash
cargo +1.85 bench -p causal-discovery --bench pcmci
```
