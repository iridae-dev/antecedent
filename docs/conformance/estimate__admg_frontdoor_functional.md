# ADMG front-door FunctionalEffect numeric pin

**Suite path:** `conformance/estimate/admg_frontdoor_functional`

This fixture is the numeric companion to
`conformance/identify/general_id_frontdoor`.  It uses the same projected ADMG,
`T -> M -> Y` with `T <-> Y`, and freezes a fully enumerated binary empirical
law.  There is no random data generation.

For the table, `P(T=1)=0.5`, `P(M=1|T=0)=0.2`, and
`P(M=1|T=1)=0.8`.  Averaging `P(Y=1|M,T)` over the observed treatment law gives
`g(0)=0.2` and `g(1)=0.7`.  The front-door functional is therefore

`[0.2*g(0) + 0.8*g(1)] - [0.8*g(0) + 0.2*g(1)] = 0.3`.

`crates/antecedent/tests/pag_admg_numeric_pins.rs` and
`python/tests/test_pag_admg_numeric_pins.py` pin that number for explicit and
accepted ADMGs through every licensed Frequentist validation level; the Rust
test also pins prepared estimate and same-schema refresh reuse.

## Expected summary

Top-level keys: `case, columns, contingency_table, frequentist, graph, identification, query, schema_version` (8 fields).
