# rd_sharp conformance fixture

**Suite path:** `conformance/estimate/rd_sharp`

Clean-room synthetic SCM generated inline by
`crates/antecedent/tests/estimate_conformance.rs`.

The `se_reference` block pins the analytic standard errors. `reference.py`
freezes a small deterministic design whose outcome noise grows with the
distance to the cutoff, and computes the local-linear jump, its HC1 residual
sandwich SE (the `rd.sharp` default), and the classical homoskedastic SE
(explicit opt-in) from the textbook formulas with numpy.
`estimate_rd_sharp_analytic_se_matches_reference` checks both SEs at a
relative tolerance of 1e-9. Run
`python3 conformance/estimate/rd_sharp/reference.py --check`.

## Expected summary

Top-level keys: `estimator, generation, identifier, notes, reference, se_reference, tolerance, true_effect` (8 fields).
