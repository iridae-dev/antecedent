# CI / orientation / kNN baselines

Established: 2026-07-21
Machine class: Apple M1 (arm64), 64 GB
Criterion: `--quick` sample (refresh with full Criterion for gate decisions)

## CI batch (`causal-stats` bench `ci_phase5`)

Workload: analytic CI batches on `n=400` (kNN `n=120`), conditioning sizes as
noted, with full sample and 20% mask-based complete-case drop (`missing20`).

Groups: `ci_batch_parcorr`, `ci_batch_robust`, `ci_batch_gsquared`, `ci_batch_knn`.

| Workload | mean wall time |
|----------|----------------|
| parcorr z0_full | **~0.7 µs** |
| parcorr z1_full | **~5 µs** |
| parcorr z4_full | **~17 µs** |
| robust z1_full | **~14 µs** |
| gsquared z1_full | **~11 µs** |
| knn z1_full | **~8 ms** |

```bash
cargo +1.85 bench -p causal-stats --bench ci_phase5
```

Acceptance: no unexplained >20% regression vs last accepted Criterion mean on
the same machine class.

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

Workload: chain CPDAG of size `n∈{16,64,128}`; rules =
`OrientCollider + Meek R1–R4`. `local_delta_*` uses
`run_orientation_to_fixed_point` (neighbor enqueue); `global_rescan_*` reseeds
the full node set every rule application.

| Workload | mean wall time |
|----------|----------------|
| local_delta_n16 | **21.7 µs** |
| global_rescan_n16 | **29.4 µs** |
| local_delta_n64 | **168 µs** |
| global_rescan_n64 | **194 µs** |
| local_delta_n128 | **509 µs** |
| global_rescan_n128 | **558 µs** |

Local-delta remains clearly faster than global rescan at each size (~15–25% win).

```bash
cargo +1.85 bench -p causal-discovery --bench orientation
```

## Calibration suite

Unit gate: `calibrate_parcorr_like` Type I < 0.20 and power > 0.50 at α=0.05
(`causal-stats` `ci::calibration` tests).
