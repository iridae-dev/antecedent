# PCMCI benchmark baseline

Workload: `pcmci_n500_p4_lag2` — lagged PCMCI (PC parents + MCI, FDR off,
`max_cond_size=1`, `max_lag=2`) on a synthetic 4-variable series of length 500.

Established: 2026-08-18 (0.5.2 performance pass)
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 100

## Accepted measurement

| Metric | Value |
|--------|-------|
| mean wall time | **6.73 ms** |
| CI (lower / upper) | 6.72 ms / 6.74 ms |

## Acceptance

Regressions exceeding **20%** wall-time vs the last accepted Criterion run on
the same machine class require an approved explanation and replacement baseline
. Gate: mean ≤ **8.08 ms** (20% over 6.73 ms).

**Replacement note (2026-08-18).** The previously recorded 1.59 ms baseline is
not reproducible on the reference machine at any commit: re-running this bench
at 9aa3ce2 (2026-07-19), 9d54254 (2026-07-25), 3acced0 (2026-07-29), and
current HEAD measures 6.0–7.1 ms throughout. The recorded number therefore did
not describe this workload on this machine class, and no code regression
matches it. Measured drift over that commit range is +11% (6.07 → 6.73 ms),
attributable to the 0.4.0 correctness fixes (bounded MCI conditioning, FDR
fail-closed) — an explained, accepted increase.

## Declared allocation budget

Steady-state candidate loop (after one warmup CI):

- one `LaggedFrame` per `run` (`p * (max_lag+1) * n_effective * 8` bytes);
- `DiscoveryWorkspace` scratch (`col_idxs`, `z_flat`, `ci.parcorr`) must not
 grow capacity across repeated CI calls;
- no per-CI `SamplePlan` / `Arc<[LaggedColumn]>` rebuild.

Gate: `ci_hot_path_no_scratch_growth` in `antecedent-discovery`.

## Target-wise parallel scaling

Workload: `pcmci_target_parallel/threads_{1,2,4}` on `n=400`, `p=8`, same
algorithm knobs. Threads come from `ExecutionContext.parallelism` (scoped
workers; no global pool).

| Threads | mean wall time |
|---------|----------------|
| 1 | **2.98 ms** |
| 2 | **1.75 ms** (~1.71×) |
| 4 | **1.05 ms** (~2.84×) |

(Refreshed 2026-08-18 alongside the headline baseline; the prior recorded
numbers — 10.28 / 5.62 / 6.77 ms — came from the same unreproducible run.)

Refresh after algorithm changes:

```bash
cargo +1.85 bench -p antecedent-discovery --bench pcmci -- pcmci_target_parallel
```

## How to refresh

```bash
cargo +1.85 bench -p antecedent-discovery --bench pcmci
```
