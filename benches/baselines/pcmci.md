# PCMCI benchmark baseline

Workload: `pcmci_n500_p4_lag2` — lagged PCMCI (PC parents + MCI, FDR off,
`max_cond_size=1`, `max_lag=2`) on a synthetic 4-variable series of length 500.

Established: 2026-07-21 (refreshed after LaggedFrame / hot-path rewrite)
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 10

## Accepted measurement

| Metric | Value |
|--------|-------|
| mean wall time | **1.59 ms** |
| CI (lower / upper) | 1.58 ms / 1.61 ms |

## Acceptance

Regressions exceeding **20%** wall-time vs the last accepted Criterion run on
the same machine class require an approved explanation and replacement baseline
. Gate: mean ≤ **1.91 ms** (20% over 1.59 ms).

## Declared allocation budget

Steady-state candidate loop (after one warmup CI):

- one `LaggedFrame` per `run` (`p * (max_lag+1) * n_effective * 8` bytes);
- `DiscoveryWorkspace` scratch (`col_idxs`, `z_flat`, `ci.parcorr`) must not
 grow capacity across repeated CI calls;
- no per-CI `SamplePlan` / `Arc<[LaggedColumn]>` rebuild.

Gate: `ci_hot_path_no_scratch_growth` in `causal-discovery`.

## Target-wise parallel scaling

Workload: `pcmci_target_parallel/threads_{1,2,4}` on `n=400`, `p=8`, same
algorithm knobs. Threads come from `ExecutionContext.parallelism` (scoped
workers; no global pool).

| Threads | mean wall time |
|---------|----------------|
| 1 | **10.28 ms** |
| 2 | **5.62 ms** (~1.83×) |
| 4 | **6.77 ms** (overhead-dominated vs 2 on this size) |

Refresh after algorithm changes:

```bash
cargo +1.85 bench -p antecedent-discovery --bench pcmci -- pcmci_target_parallel
```

## How to refresh

```bash
cargo +1.85 bench -p antecedent-discovery --bench pcmci
```
