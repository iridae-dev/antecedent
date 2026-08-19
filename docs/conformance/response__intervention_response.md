# Intervention response known-truth pin

**Suite path:** `conformance/response/intervention_response`

`Y = 1 + 2T + 0.8Z` (no outcome noise) with `Z = sin(i/17)` for a fixed
deterministic row index `i`, `T = Z + cos(i/11)`, `Z -> T`, `Z -> Y`, `T -> Y`.

`response.intervention_gcomp` estimates `E[Y | do(T := 0.25)]` by fitting the
outcome nuisance on `{T, Z}` and integrating out the observed `Z` exactly
(a `Set` intervention needs no Monte Carlo). Because the structural mean is
linear in `T` and `Z` and additive, the intervention response has the closed
form `1 + 2*0.25 + 0.8*E[Z]`. `E[Z]` over the fixed 240-row deterministic
sequence is computed directly from the same `z_i = sin(i/17)` formula the
Rust conformance test regenerates, giving `true_response = 1.553878239227273`.

The tolerance is a statistical one, not a bitwise-identity claim: the
estimator's outcome nuisance is an additive-GAM fit with a ridge-style
`nuisance_lambda` penalty, which can bias a penalized fit slightly away from
the exact linear coefficients even with zero outcome noise.

## Expected summary

Top-level keys: `claim, contract, estimator_contract, fixture_id, generation, tolerance` (6 fields).
