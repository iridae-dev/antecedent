# PAG ATE envelope numeric pin

**Suite path:** `conformance/estimate/pag_ate_envelope`

This fixture upgrades the generalized-adjustment identification pin to an
effect pin.  The supplied PAG has three valid MAG completions for the
`Z o-o T` edge while keeping `Z -> Y` and `T -> Y` fixed.  All three
completions identify the ATE, but they do not imply one invariant estimand:
the empirical completion effects are `0.40`, `0.52`, and `0.40`.  Equal
completion weights therefore make the Frequentist envelope return `0.44`.

The contingency table is the entire frozen input law; the test expands its
integer counts without sampling.  The Bayesian value is the deterministic
64-draw conjugate envelope mean at seed 1.  It is an output pin (not a claim
that posterior shrinkage must equal the empirical Frequentist functional).

Both explicit and accepted PAGs consume this fixture through every licensed
validation level (`none`, `cheap`, and `full`) in
`crates/antecedent/tests/pag_admg_numeric_pins.rs`, which also pins prepared
estimate and same-schema refresh reuse, and in
`python/tests/test_pag_admg_numeric_pins.py`.

## Expected summary

Top-level keys: `case, columns, contingency_table, graph, identification, query, schema_version, validity, withdrawn_numeric_outputs` (9 fields).
