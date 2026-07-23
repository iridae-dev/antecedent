# Shapley attribution baselines

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
