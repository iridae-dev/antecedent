# design / state baselines

Established: 2026-07-14 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 2ef9a173 (the commit that added this file; the measured commit was not written down)

Criterion smokes (gated with `--test`):

- `antecedent-design` / `design_rank` — `design_rank_eig_8_candidates`
- `antecedent-state` / `state_append` — `state_append_invalidate_ols`

**Budgets (local regression, Apple M1 class):**

| Case | Soft latency budget |
|------|---------------------|
| design_rank_eig_8_candidates | < 50 ms / iter |
| state_append_invalidate_ols | < 20 ms / iter |

Ranking always reports `MonteCarloBudget` + per-candidate stderr. State caches
refuse inserts over `CacheBudget` without changing semantics.
