# Matching index baselines (Phase 4)

Owner: `causal-stats` / `MatchingIndex::exact` and
`causal-estimate` / `PropensityEstimationWorkspace`

## Criteria

- Exact brute-force path for `n ≤ EXACT_MATCHING_ROW_LIMIT` (10_000).
- Bench target: `matching_exact_n500_d4` — 500 donors × 500 queries, dim=4.
- Point-estimate fits retain `MatchingIndex` across compatible donor geometries
  (`matching_index_builds` stays flat on a second identical fit).
- Bootstrap replicates rebuild the index whenever resampled donors change the
  geometry key (DESIGN §14.6).

## Notes

- Larger-n approximate indexes are out of Phase 4 scope.
- Differential tests compare `nearest` against `nearest_euclidean_scalar`.
