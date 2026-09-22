# Linear-Gaussian counterfactual baselines

Established: 2026-08-18 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 59295cec (the commit that added this file; the measured commit was not written down)

Owner: `antecedent-counterfactual` / `CounterfactualEngine::predict`

## Criteria

- Two-node linear-Gaussian SCM, n=100, abduct then predict `do(T := 1)` on Y.
  Streaming output must match retained (`streaming_matches_retained`).
- Bench targets:
  - `counterfactual_predict_n100` — `unit_rows: None` (full column).
  - `counterfactual_predict_n100_unit_rows` — `unit_rows: [0..10]`. Nested CF /
    LGSSM is out of scope (separate P1).

## Measured mean

- `counterfactual_predict_n100`: **0.985 µs** (Criterion mean, CI 0.983–0.986 µs).
- `counterfactual_predict_n100_unit_rows`: **1.159 µs** (Criterion mean, CI 1.153–1.164 µs).
- Established: 2026-08-18. Machine class: Apple M1 Max.

## How to refresh

```bash
cargo bench -p antecedent-counterfactual --bench counterfactual_batch
```

Record the Criterion means and update this file with the run date and machine class.

Gates (mean ≤ 1.20×):

| Workload | Gate |
|----------|------|
| counterfactual_predict_n100 | ≤ **1.18 µs** |
| counterfactual_predict_n100_unit_rows | ≤ **1.39 µs** |
