# prior_bank_transport

**Suite path:** `conformance/bayesian/prior_bank_transport`

TransportPolicy is required when source and target population tags differ.
With an explicit policy, composition succeeds and records
`external_transport_prior`. Propensity without weights forces α → 0.

## Expected summary

Top-level keys: `alpha, baseline_mean, baseline_variance, error_code, notes, propensity_missing_weights, source_mean, source_population, source_variance, target_population, tol, with_policy` (12 fields).
