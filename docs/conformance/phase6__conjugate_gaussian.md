# Phase 6 conjugate_gaussian conformance fixture

**Suite path:** `conformance/phase6/conjugate_gaussian`

Clean-room noiseless linear design `y = 1 + 2x` with known tiny residual
variance. Expected posterior MAP / mean recovers OLS coefficients.
Exercised by `causal-prob` unit/integration tests and Phase 6 gate.

## Expected summary

Top-level keys: `backend, notes, prior, tolerance, tolerance_class, true_coefficients` (6 fields).
