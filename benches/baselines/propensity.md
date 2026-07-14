# Propensity bootstrap baselines 

Owner: `causal-estimate` / `PropensityWeighting::fit`

## Criteria

- Bootstrap SE refits the logistic propensity model on every replicate (documented as the
  "honest" choice in `causal-estimate/src/propensity.rs`), reusing
  `PropensityEstimationWorkspace::propensity` (`causal_stats::PropensityWorkspace`) IRLS scratch
  across replicates rather than reallocating it.
- After a warm fit, `ols.grow_count` and `scores_grow_count` must stay flat across further
  fits of the same `n` (asserted in the Criterion bench and in
  `bootstrap_reuses_propensity_workspace_buffers`).
- Bench target: `propensity_weighting_ipw_bootstrap50_n800` — n=800, 1 adjustment covariate,
  50 bootstrap replicates, `PropensityWeighting::fit` end to end (propensity fit + Hajek point
  estimate + bootstrap SE).
- PR CI gate: `scripts/gate_estimate_reuse.sh`.

## Notes

- `PropensityMatching` / `DistanceMatching` / `PropensityStratification` bootstraps follow the
  same refit-per-replicate pattern; only the IPW-weighting path is currently benched.
- Matching-based bootstraps rebuild a `MatchingIndex` when resampled donors change the
  geometry key; point estimates retain the index across compatible fits.
