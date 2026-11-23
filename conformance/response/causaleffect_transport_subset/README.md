# causaleffect transport on the supported sID subset

Black-box parity for Antecedent's sound single-source transport rules against
R `causaleffect` 1.3.15 `transport()`.

Only cases inside Antecedent's implemented subset are compared:

- direct transport when selection does not reach the outcome;
- exogenous pre-treatment standardization (`transport.sid.standardize`).

General multi-node c-component cases remain `NotCertified` in Antecedent and
are not part of this claim. The frozen oracle stores the rendered formula
string; the consuming test checks Antecedent's certificate rule and formula
kind, not string identity with R's TeX-like rendering.

## What is and is not oracle-backed

The two cases in `expected.json` above are oracle-backed: causaleffect 1.3.15
was actually run on them (see `reference.command`), and the consuming test
(`matches_frozen_causaleffect_supported_sid_subset` in
`crates/antecedent-identify/tests/causaleffect_transport_subset.rs`) checks
Antecedent's output against that frozen run.

The same test file also contains
`multinode_c_component_outside_certified_subset_is_not_certified`, which
covers the fail-closed `NotCertified` path for a selection diagram outside
Antecedent's implemented subset (a two-node c-component reached by
selection). That case is **not oracle-backed**: causaleffect was never run
against it, because this fixture schema has no way to express "the diagram
is outside the certified subset" without inventing a causaleffect output we
did not observe. It is a plain Rust assertion against Antecedent's own
conservative refusal, included here only because it is the counterpart to
the positive cases above and belongs next to them.

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

The product assertion now reads the actual R formula to determine its class and
summed variables. A local class label plus two equal strings is insufficient.
This remains a two-graph structural comparison: R returns target-observational
expressions, while the product's source-experimental forms agree under the fixed
graph invariances. It establishes no general experiment-catalog equivalence,
numeric transported-law agreement or recursive completeness. The separate
out-of-subset refusal stays a software assertion, without a fabricated R output.
