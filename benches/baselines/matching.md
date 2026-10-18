<!-- generated from benches/baselines/baselines.toml (matching); do not edit -->
# Matching index baselines

Established: 2026-07-13 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: d798e2f1 (the commit that added this file; the measured commit was not written down)

Owner: `antecedent-stats` / `MatchingIndex::exact` and
`antecedent-estimate` / `PropensityEstimationWorkspace`

## Criteria

- Exact results at any donor count (no row limit): a sorted search for 1-d absolute distance and a k-d tree for Euclidean distance, both reproducing the linear scan's lowest-donor-index tie rule bit for bit; index memory is checked against `ExecutionContext` `memory.hard_limit_bytes` and queries observe cancellation.
- Bench target: `matching_exact_n500_d4` — 500 donors × 500 queries, dim=4.
- Point-estimate fits retain `MatchingIndex` across compatible donor geometries
 (`matching_index_builds` stays flat on a second identical fit).
- Bootstrap replicates rebuild the index whenever resampled donors change the
 geometry key .

## Notes

- No approximate index is used; the k-d tree degrades toward a full scan only as the feature dimension grows.
- Differential tests compare `nearest` against `nearest_euclidean_scalar` and a retained linear-scan oracle on tied, duplicated and signed-zero inputs; a 20k-donor test runs by default and a 100k-donor acceptance test is `#[ignore]`.

Numeric wall-time gate: none published (reuse / `--test` smoke only).
