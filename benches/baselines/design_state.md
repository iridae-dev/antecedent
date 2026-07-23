# design / state baselines

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
