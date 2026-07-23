# Kernel reductions benchmark baseline

Workloads (public dispatch, default `KernelPolicy` / portable-optimized):

| Workload | Description |
|----------|-------------|
| `masked_covariance_n10k` | Population covariance, contiguous `f64` length 10_000 |
| `standardize_inplace_n10k` | Sample-SD standardize in place, `n = 10_000` |
| `weighted_sum_n10k` | Bootstrap weighted sum, `n = 10_000` |
| `pairwise_l1_n256` | Pairwise L1 fill into `256×256` |
| `accumulate_contingency_n10k` | Contingency accumulate, 8×5 levels, `n = 10_000` |

Established: 2026-07-22
Machine class: Apple M1 Max (arm64), 64 GB
Policy: default (portable-optimized; `allow_arch_simd` wired but no `simd-runtime` path)
Criterion sample size: 30

## Accepted measurement

| Metric | masked_covariance | standardize | weighted_sum | pairwise_l1 | contingency |
|--------|-------------------|-------------|--------------|-------------|-------------|
| mean wall time | **28.15 µs** | **21.84 µs** | **9.56 µs** | **7.66 µs** | **6.42 µs** |
| CI (lower / upper) | 28.12 / 28.19 µs | 21.78 / 21.90 µs | 9.53 / 9.57 µs | 7.17 / 8.38 µs | 6.41 / 6.42 µs |

## Arch SIMD decision

Portable contiguous paths (covariance / standardize / weighted_sum / pairwise) rely on
safe auto-vectorization. Contingency is scatter-add and shares the scalar body.
No kernel cleared the ≥10% median win vs portable at `n ≥ 10_000` needed to justify
`unsafe` arch SIMD + `simd-runtime`. `KernelPolicy::allow_arch_simd`
is consulted by `select_impl`; `arch_simd_available()` remains false until justified.

## Acceptance

Regressions exceeding **10%** wall-time vs the last accepted Criterion run on the
same machine class require an approved explanation and replacement baseline
. Gates (mean ≤ 1.10×):

| Workload | Gate |
|----------|------|
| masked_covariance_n10k | ≤ **30.97 µs** |
| standardize_inplace_n10k | ≤ **24.02 µs** |
| weighted_sum_n10k | ≤ **10.51 µs** |
| pairwise_l1_n256 | ≤ **8.43 µs** |
| accumulate_contingency_n10k | ≤ **7.06 µs** |

## How to refresh

```bash
cargo bench -p causal-kernels --bench reductions -- --sample-size 30
```

Record mean times and commit an update to this file with hardware notes.
