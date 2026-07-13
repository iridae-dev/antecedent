# Partial-correlation batch baseline (Phase 2)

Workload: `parcorr_batch64_n2k_p8` — 64 partial-correlation queries on
`n=2000` rows with up to 3 conditioning columns drawn from 8 series columns,
via the batch API with reusable [`ParCorrWorkspace`].

Established: 2026-07-21
Machine class: Apple M1 Max (arm64), 64 GB
Path: portable
Criterion sample size: 20

## Accepted measurement

| Metric | Value |
|--------|-------|
| mean wall time | **3.06 ms** |
| CI (lower / upper) | 3.03 ms / 3.12 ms |

## Acceptance

Regressions exceeding **20%** wall-time vs the last accepted Criterion run on
the same machine class require an approved explanation and replacement baseline
(DESIGN.md §28.9). Gate: mean ≤ **3.68 ms** (20% over 3.06 ms).

## How to refresh

```bash
cargo +1.85 bench -p causal-kernels --bench partial_correlation
```
