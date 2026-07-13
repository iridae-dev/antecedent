# Matching index baselines (Phase 4)

Owner: `causal-stats` / `MatchingIndex::exact`

## Criteria

- Exact brute-force path for `n ≤ EXACT_MATCHING_ROW_LIMIT` (10_000).
- Bench target: `matching_exact_n500_d4` — 500 donors × 500 queries, dim=4.

## Notes

- Larger-n approximate indexes are out of Phase 4 scope.
- Differential tests compare `nearest` against `nearest_euclidean_scalar`.
