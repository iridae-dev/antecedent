# prior_bank_power_mixture

**Suite path:** `conformance/bayesian/prior_bank_power_mixture`

Analytic 1-D Gaussian power-prior composition.

Baseline: mean `0`, conjugate scale `V0 = 4` (precision `Λ₀ = 0.25`).
Source: mean `2`, scale `V = 1` (precision `Λ = 1`), α = `0.5`.

Expected composed precision `Λ = Λ₀ + α Λ = 0.75`, mean
`μ = (Λ₀·0 + α Λ·2) / Λ = 4/3`, scale `V = 1/Λ = 4/3`.

## Expected summary

Top-level keys: `alpha, baseline_mean, baseline_variance, expected_mean, expected_precision, expected_variance, notes, required_assumption_ids, source_mean, source_variance, tol` (11 fields).
