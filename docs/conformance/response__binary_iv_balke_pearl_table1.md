# Binary-IV Balke–Pearl Table 1 oracle

**Suite path:** `conformance/response/binary_iv_balke_pearl_table1`

This fixture compares Antecedent's exact 16-response-type linear program with
the public `bpbounds()` API from the immutable CRAN `bpbounds` 0.1.8 source
release. The input is the conditional law from Balke and Pearl's Table 1. The
comparison covers the average causal effect without monotonicity.

The R array is ordered `(D, Y, Z)` by `bpbounds`; `expected.json` records the
equivalent Antecedent order `(Y, D)` within each instrument arm. The historical
import command installed the exact source release; it is preserved separately.
The retained harness reads the same public result fields in its isolated lock. No upstream source code or tests are stored in this repository.

## Retained reference environment

The generator and complete R 4.6.1 dependency lock are shared with the other R
response fixture. From the repository root:

```sh
Rscript --vanilla conformance/response/causaleffect_transport_subset/prepare.R
Rscript --vanilla conformance/response/causaleffect_transport_subset/reference.R --check --self-test
```

Preparation restores into `target/reference-environments/r-oracles/library`;
`ANTECEDENT_R_ORACLE_LIBRARY` selects another isolated tooling library. The
reference verifies R and every locked package version and excludes the global
site library. Upstream GPL packages remain tooling dependencies; ordinary CI
consumes frozen results and imports no upstream implementation or tests.

The original inputs, outputs and historical command metadata are preserved.
Fault controls reject changed graphs/laws, both copies of a corrupted formula,
wrong bounds/verdicts, widened tolerances, empty selections and wrong versions.
This regeneration is fixed-input conformance, not calibration or a performance
acceptance campaign.

Only the lower/upper average-effect bounds without monotonicity and the IV
inequality on the original conditional law are compared, at absolute 1e-12.
The harness explicitly maps the fixture's (Y,D) cells into R's (D,Y,Z) array
order. Sampling uncertainty, arbitrary instruments and general robustness are
outside this single-law comparison.

## Expected summary

Top-level keys: `cell_order, cells, estimand, expected, fixture_id, generation, historical_oracle, historical_reference, oracle, reference, tolerance` (11 fields).
