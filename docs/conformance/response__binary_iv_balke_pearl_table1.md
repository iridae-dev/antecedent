# Binary-IV Balke–Pearl Table 1 oracle

**Suite path:** `conformance/response/binary_iv_balke_pearl_table1`

This fixture compares Antecedent's exact 16-response-type linear program with
the public `bpbounds()` API from the immutable CRAN `bpbounds` 0.1.8 source
release. The input is the conditional law from Balke and Pearl's Table 1. The
comparison covers the average causal effect without monotonicity.

The R array is ordered `(D, Y, Z)` by `bpbounds`; `expected.json` records the
equivalent Antecedent order `(Y, D)` within each instrument arm. The frozen
reference command installs the exact source release and prints only the public
result fields. No upstream source code or tests are stored in this repository.

## Expected summary

Top-level keys: `cell_order, cells, estimand, expected, fixture_id, reference, tolerance` (7 fields).
