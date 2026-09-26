# Bayesian trial-to-target transport oracle

**Suite path:** `conformance/estimate/bayesian_trial_transport`

Two observed trial rows have treated/control outcomes 3 and 1. With selection
and treatment probabilities each fixed at 0.5, the source-row Bayesian
bootstrap puts mass `U ~ Beta(1,1)` on the treated row and `1-U` on the control
row. Holding the two target rows and supplied probabilities fixed yields an IPW
effect draw `8U-2`, with mean 2 and variance 16/3. This checks the estimator's
posterior law, not causal identification or uncertainty in the supplied
probabilities.

## Expected summary

Top-level keys: `expected_mean, expected_variance, mean_tolerance, oracle, schema_version, variance_tolerance` (6 fields).
