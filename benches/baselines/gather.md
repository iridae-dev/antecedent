# Sample gather benchmark baseline (Phase 0)

Workload: `gather_stride10_n100k` — gather every 10th index from a contiguous
`f64` vector of length 100_000 via the public dispatch entry.

Established: 2026-07-21
Machine class: local developer (Apple Silicon / comparable)
Policy: default (`portable-optimized` allowed)

## Acceptance

Regressions exceeding **20%** wall-time vs the last accepted Criterion run on
the same machine class require an approved explanation and replacement baseline
(DESIGN.md §28.9).

## How to refresh

```bash
cargo +1.85 bench -p causal-kernels --bench gather
```

Record mean time and commit an update to this file with hardware notes.
