# Distribution-change attribution — Y intercept shift

**Suite path:** `conformance/attribution/distribution_change_y_shift`

Synthetic linear SCM `X → Y`. Baseline (rows 0..40): `Y = 1 + 2X`.
Comparison (rows 40..80): `Y = 6 + 2X`. Exact Shapley should attribute the
mean shift primarily to the Y mechanism.

## Expected summary

Top-level keys: `allocation, dominant_component, notes, outcome, total_change_min` (5 fields).
