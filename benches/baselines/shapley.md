# Shapley attribution baselines

Established: 2026-07-14 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 9a26b51a (the commit that added this file; the measured commit was not written down)

Criterion bench `antecedent-attribution` / `shapley` (gated with `--test`):

- `shapley_mc_8p_200_cached`
- `shapley_mc_8p_200_uncached`
- `shapley_exact_10p_cached`

**Budgets (local regression, Apple M1 class):**

| Case | Soft latency budget |
|------|---------------------|
| shapley_mc_8p_200_cached | < 500 ms / iter |
| shapley_exact_10p_cached | < 200 ms / iter |

Cache hits must reduce coalition re-evaluation vs uncached MC on additive games
(see unit tests in `coalition` / `shapley` modules). Exact Shapley rejects above
`max_exact_components` unless `allow_exact_override` is set.
