<!-- generated from benches/baselines/baselines.toml (noise_replay); do not edit -->
# Batch coalition noise replay baselines

Established: 2026-10-17 (the date this file was first committed; no timing has been measured yet)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 9f2a661e4 (the commit the benchmark was written against; no timing has been measured yet)

Owner: `antecedent-model` / `NoiseReplayer`

## Criteria

- Chain `a -> b -> c`, 4096 coalitions replayed from two retained noise tables of 100,000
  rows. The batch route borrows the model and reuses one replayer and one execution context; the
  baseline clones the compiled model and starts from empty scratch for every coalition.
- Bench targets: `coalitions_4096_n1000_batch` and `coalitions_4096_n1000_cloning_baseline`
  (Criterion, 1,000-row tables), plus the 100,000-row regression gate in the bench body.
- Regression bound (`MAX_REGRESSION = 1.1`, asserted in the bench): the fastest of three batch
  runs takes at most 1.1 times the fastest cloning run.
- Reuse (asserted in the bench): `scratch_footprint` is constant across all coalitions, and the
  two routes replay identical values.

## How to refresh

```bash
cargo bench -p antecedent-model --bench noise_replay
```

Numeric wall-time gate: none published. The regression bound is relative and asserted in the
bench itself, so it holds on any machine and in the `--test` smoke.
