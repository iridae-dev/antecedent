<!-- generated from benches/baselines/baselines.toml (policy_sampling); do not edit -->
# Program-executed interventional sampling baselines

Established: 2026-10-17 (the date this file was first committed; no timing has been measured yet)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 9f2a661e4 (the commit the benchmark was written against; no timing has been measured yet)

Owner: `antecedent-model` / `sample_interventional`

## Criteria

- Pricing chain `price -> demand -> revenue` (and `price -> revenue`) with a lognormal price
  (location 3.0, scale 0.4), n = 1,000,000 rows. The policy `price := max(10, 0.85 * price)`
  is one extra value-program pass over the price column.
- Bench targets: `sample_plain_n1000000` and `sample_price_floor_program_n1000000`.
- Regression bound (`MAX_REGRESSION = 1.1`, asserted in the bench): the fastest of five
  interleaved program-executed draws takes at most 1.1 times the fastest plain draw.
- Reuse (asserted in the bench): one `ExecutionContext` and one `MechanismWorkspace` serve every
  draw, and `ws.grow_count` stays flat after the warm-up draw.

## How to refresh

```bash
cargo bench -p antecedent-model --bench policy_sampling
```

Numeric wall-time gate: none published. The regression bound is relative and asserted in the
bench itself, so it holds on any machine and in the `--test` smoke.
