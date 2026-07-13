# Phase 5 CI / orientation / kNN baselines

Established: 2026-07-21
Machine class: Apple M1 (arm64), 64 GB
Criterion: `--quick` sample (refresh with full Criterion for gate decisions)

## CI batch (`causal-stats` bench `ci_phase5`)

Workload: `PartialCorrelation` analytic batches on `n=400`, `p=6`, conditioning
sizes `z∈{0,1,2,4}`, with full sample and 20% complete-case drop (`missing20`).

| Workload | mean wall time |
|----------|----------------|
| z0_full | **636 ns** |
| z0_missing20 | **537 ns** |
| z1_full | **5.01 µs** |
| z1_missing20 | **3.86 µs** |
| z2_full | **7.91 µs** |
| z2_missing20 | **6.61 µs** |
| z4_full | **14.9 µs** |
| z4_missing20 | **11.3 µs** |

```bash
cargo +1.85 bench -p causal-stats --bench ci_phase5 -- ci_batch_parcorr
```

Acceptance: no unexplained >20% regression vs last accepted Criterion mean on
the same machine class (DESIGN.md §28.9).

## kNN index reuse

Workload: `knn_cmi_reuse_batch8` — eight identical kNN CMI queries; after warmup,
`CiWorkspace.knn.index_generation` must stay constant (asserted inside the
bench). Gate also covered by unit test
`knn_reuses_permutation_plan_across_queries`.

| Workload | mean wall time |
|----------|----------------|
| knn_cmi_reuse_batch8 | **2.18 ms** |

```bash
cargo +1.85 bench -p causal-stats --bench ci_phase5 -- knn_cmi_reuse
```

## Orientation local-delta vs global rescan (`causal-discovery` bench `orientation`)

Workload: chain CPDAG of size `n∈{16,64,128}`; `local_delta_*` uses
`run_orientation_to_fixed_point` (neighbor enqueue); `global_rescan_*` reseeds
the full node set every rule application.

| Workload | mean wall time |
|----------|----------------|
| local_delta_n16 | **21.6 µs** |
| global_rescan_n16 | **24.5 µs** |
| local_delta_n64 | **168 µs** |
| global_rescan_n64 | **177 µs** |
| local_delta_n128 | **500 µs** |
| global_rescan_n128 | **524 µs** |

Local-delta remains faster than global rescan at each size.

```bash
cargo +1.85 bench -p causal-discovery --bench orientation
```

## Calibration suite

Unit gate: `calibrate_parcorr_like` Type I < 0.20 and power > 0.50 at α=0.05
(`causal-stats` `ci::calibration` tests).
