# laplace_glm conformance fixture

**Suite path:** `conformance/bayesian/laplace_glm`

Clean-room noiseless linear design; Laplace MAP must recover OLS under a
weakly informative prior. Workspace reuse is gated by the laplace_glm bench.

## Expected summary

Top-level keys: `backend, likelihood, notes, tolerance, tolerance_class, true_coefficients` (6 fields).
