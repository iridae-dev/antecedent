# Propensity bootstrap baselines 

Established: 2026-07-13 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 3ec75735 (the commit that added this file; the measured commit was not written down)

Owner: `antecedent-estimate` / `PropensityWeighting::fit`

## Criteria

- Bootstrap SE refits the logistic propensity model on every replicate (documented as the
  "honest" choice in `antecedent-estimate/src/propensity.rs`), reusing
  `PropensityEstimationWorkspace::propensity` (`antecedent_stats::PropensityWorkspace`) IRLS scratch
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

Numeric wall-time gate: none published (reuse / `--test` smoke only).
