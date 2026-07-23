# IV / two-stage-least-squares conformance fixture

**Suite path:** `conformance/estimate/iv_2sls`

Generated inline by `crates/antecedent/tests/estimate_conformance.rs` — no
CSV, no pinned baseline install. Clean-room synthetic SCM, deterministic from a fixed
`ExecutionContext` seed.

Comparison: `|estimate.ate - true_effect| < tolerance` (finite-sample Monte
Carlo check; not a `StableFloat` bitwise/analytic comparison).

## Expected summary

Top-level keys: `estimator, generation, identifier, notes, reference, tolerance, true_effect` (7 fields).
