# Phase 4 IV / two-stage-least-squares conformance fixture

Generated inline by `crates/causal/tests/phase4_conformance.rs` — no
CSV, no DoWhy install. Clean-room synthetic SCM, deterministic from a fixed
`ExecutionContext` seed.

Comparison: `|estimate.ate - true_effect| < tolerance` (finite-sample Monte
Carlo check; not a `StableFloat` bitwise/analytic comparison).
