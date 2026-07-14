# IV / two-stage-least-squares conformance fixture

**Suite path:** `conformance/estimate/iv_2sls`

Generated inline by `crates/causal/tests/estimate_conformance.rs` — no
CSV, no DoWhy install. Clean-room synthetic SCM, deterministic from a fixed
`ExecutionContext` seed.

Comparison: `|estimate.ate - true_effect| < tolerance` (finite-sample Monte
Carlo check; not a `StableFloat` bitwise/analytic comparison).

## Expected summary

Top-level keys: `estimator, identifier, notes, tolerance, true_effect` (5 fields).
