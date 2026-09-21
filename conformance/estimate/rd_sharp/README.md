# rd_sharp conformance fixture

Clean-room synthetic SCM generated inline by
`crates/antecedent/tests/estimate_conformance.rs`.

`true_effect` is the effect for units at the cutoff, the only effect a sharp
design identifies (`TargetPopulation::LocalAtCutoff`). The jump fixture has a
constant effect, so it cannot tell that target from a window or population
average; `rd_sharp_reports_the_cutoff_effect_and_labels_it` in the same test file
uses a closed-form design where the three differ (2, 2.32 and 8) and checks both
the estimate and its label. The treatment column is the threshold rule and the
graph is `r -> t -> y`, `r -> y`: the estimator verifies the first and the
identifier requires the second.

The `se_reference` block pins the analytic standard errors. `reference.py`
freezes a small deterministic design whose outcome noise grows with the
distance to the cutoff, and computes the local-linear jump, its HC1 residual
sandwich SE (the `rd.sharp` default), and the classical homoskedastic SE
(explicit opt-in) from the textbook formulas with numpy.
`estimate_rd_sharp_analytic_se_matches_reference` checks both SEs at a
relative tolerance of 1e-9. Run
`python3 conformance/estimate/rd_sharp/reference.py --check`.
