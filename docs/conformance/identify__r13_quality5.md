# Identify: R13 tests-quality-5 numeric gaps

**Suite path:** `conformance/identify/r13_quality5`

Hand-derived binary SCMs consumed by
`crates/antecedent-identify/tests/r13_quality5.rs`.

These three cases are the defects named by R13 `tests-quality-5`: a numeric
identification query whose treatment has an observed parent (and whose
functional differs from the parent-free graph), a napkin with a pinned
interventional mean, and a MAG whose treatment→outcome edge is visible only
through a discriminating collider path — each checked against truncated
factorization, not against `identify() == Ok`.

| Case | Why it is here |
| --- | --- |
| `treatment_with_observed_parent` | Transport/ID numeric sweeps treated a root; empty-backdoor `P(y\|x)` must not pass when the treatment has a parent. |
| `napkin` | Status-only napkin pins miss the free-`z` ratio and the enumerated ATE. |
| `discriminating_path_visibility` | `mag_visibility_id` has a direct witness and a length-2 collider path, but no multi-collider discriminating path with a numeric effect. |

## Expected summary

Top-level keys: `cases, oracle, schema_version` (3 fields).
