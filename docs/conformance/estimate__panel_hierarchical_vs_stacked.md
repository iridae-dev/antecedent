# Panel hierarchical vs stacked posterior width

**Suite path:** `conformance/estimate/panel_hierarchical_vs_stacked`

This fixture freezes the qualitative contract that Bayesian panel g-computation
must not treat repeated measures on the same unit as independent rows. On a DGP
with an additive unit intercept and treatment held constant within unit, the
stacked conjugate fit sees each within-unit row as a fresh look at the same
contrast and reports a posterior that is too narrow; random-intercept GLS
whitening must widen it.

The pin is a minimum ratio of posterior effect SDs (hierarchical over stacked),
not a numeric SD. It is a direction-and-magnitude floor on a seeded synthetic
DGP, not an external package parity claim and not a known-truth SD.

## Expected summary

Top-level keys: `dgp, min_posterior_sd_ratio_hierarchical_over_stacked, tolerance` (3 fields).
