# Prepared handle — 1.3.0

Measured 2026-09-08 on this macOS arm64 development host with the release
profile, Criterion 10 samples, one-second warmup and measurement. Preparation
is outside the timed loop; each iteration estimates against supplied data.
These are local measurements, not portable CI thresholds.

| Workload | Criterion estimate | 95% interval |
|---|---|---|
| `prepared_point_derivative_n400` | **704.31 µs** | 700.86–709.77 µs |
| `prepared_counterfactual_n400` | **22.895 µs** | 22.822–22.961 µs |
| `inspect_n400` | none published | smoke only |
| `capability_n400` | none published | smoke only |
| `metadata_only_artifact_read` | none published | smoke only |

Run `cargo bench -p antecedent --bench staged_handle`. The release gate runs
these with `--test` alongside the designated response and GCM workloads.
Inspection and metadata-only reads must not identify; `--test` covers that
path, not a portable wall-time gate.
