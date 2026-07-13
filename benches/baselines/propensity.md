# Propensity bootstrap baselines (Phase 4)

Owner: `causal-estimate` / `PropensityWeighting::fit`

## Criteria

- Bootstrap SE refits the logistic propensity model on every replicate (documented as the
  "honest" choice in `causal-estimate/src/propensity.rs`), reusing
  `PropensityEstimationWorkspace::propensity` (`causal_stats::PropensityWorkspace`) IRLS scratch
  across replicates rather than reallocating it.
- Bench target: `propensity_weighting_ipw_bootstrap50_n800` — n=800, 1 adjustment covariate,
  50 bootstrap replicates, `PropensityWeighting::fit` end to end (propensity fit + Hajek point
  estimate + bootstrap SE).

## Notes

- `PropensityMatching` / `DistanceMatching` / `PropensityStratification` bootstraps follow the
  same refit-per-replicate pattern; only the IPW-weighting path is currently benched.
- Matching-based bootstraps additionally rebuild a `MatchingIndex` per replicate (donor rows
  change with resampling), so they are expected to be slower per-replicate than IPW weighting
  at comparable n; not yet benched separately.
