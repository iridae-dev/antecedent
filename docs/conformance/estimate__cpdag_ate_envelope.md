# CPDAG ATE envelope numeric pin

**Suite path:** `conformance/estimate/cpdag_ate_envelope`

This fixture is the CPDAG sibling of `pag_ate_envelope`. The supplied CPDAG
has one undirected edge `Z — T` while keeping `Z -> Y` and `T -> Y` fixed.
The two MEC DAG completions both identify the ATE, but they do not share one
invariant estimand: `Z -> T` adjusts for `Z` (empirical ATE `0.40`) and
`T -> Z` is unadjusted (empirical ATE `0.52`). Equal completion weights
therefore make the Frequentist envelope return `0.46`.

The contingency table is the entire frozen input law — the same table as the
PAG envelope fixture — and the test expands its integer counts without
sampling. The Bayesian value is the deterministic 64-draw conjugate envelope
mean at seed 1 (`0.4615641945101853`). It is an output pin, not a claim that posterior shrinkage
must equal the empirical Frequentist functional.

Both explicit and accepted CPDAGs consume this fixture through every licensed
validation level (`none`, `cheap`, and `full`) in
`crates/antecedent/tests/cpdag_ate_numeric_pins.rs`, which also pins prepared
estimate and same-schema refresh reuse, and in
`python/tests/test_cpdag_ate_numeric_pins.py`.

Class-aware `ConditionalEffect` on Cpdag/Pag reuses this table. The modifier
`z` stays in the outcome model, so both completions estimate the
Z-conditional effect at Ē[Z] (`0.40`), not the ATE envelope mean.
Consumers: `class_aware_conditional_numeric_pins`.

## Expected summary

Top-level keys: `bayesian, case, columns, conditional, contingency_table, frequentist, graph, identification, query, schema_version` (10 fields).
