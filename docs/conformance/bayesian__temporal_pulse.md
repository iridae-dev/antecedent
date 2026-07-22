# Bayesian temporal pulse conformance

**Suite path:** `conformance/bayesian/temporal_pulse`

Lag-1 SCM `y_t = 0.9 * x_{t-1}`; conjugate Bayesian g-comp on the unfolded
temporal design via the `CausalAnalysis` facade. Posterior mean ≈ 0.9; finite
`P(effect < 0)`; posterior artifact round-trip.

## Expected summary

Top-level keys: `backend, expected_ate, horizon_steps, n, n_draws, outcome, require_artifact_round_trip, require_finite_p_below_zero, scm, tolerance, treatment, treatment_lag, true_effect_per_unit` (13 fields).
