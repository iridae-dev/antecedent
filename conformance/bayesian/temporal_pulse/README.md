# Bayesian temporal pulse conformance

Lag-1 SCM `y_t = 0.9 * x_{t-1}`; conjugate Bayesian g-comp on the unfolded
temporal design via the `CausalAnalysis` facade. Posterior mean ≈ 0.9; finite
`P(effect < 0)`; posterior artifact round-trip.
