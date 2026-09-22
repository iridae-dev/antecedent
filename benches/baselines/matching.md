# Matching index baselines

Established: 2026-07-13 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: d798e2f1 (the commit that added this file; the measured commit was not written down)

Owner: `antecedent-stats` / `MatchingIndex::exact` and
`antecedent-estimate` / `PropensityEstimationWorkspace`

## Criteria

- Exact brute-force path for `n ≤ EXACT_MATCHING_ROW_LIMIT` (10_000).
- Bench target: `matching_exact_n500_d4` — 500 donors × 500 queries, dim=4.
- Point-estimate fits retain `MatchingIndex` across compatible donor geometries
 (`matching_index_builds` stays flat on a second identical fit).
- Bootstrap replicates rebuild the index whenever resampled donors change the
 geometry key .

## Notes

- Larger-n approximate indexes are out of estimate matching scope.
- Differential tests compare `nearest` against `nearest_euclidean_scalar`.

Numeric wall-time gate: none published (reuse / `--test` smoke only).
