# `antecedent`

The Rust facade for Antecedent's typed causal workflow. Build a graph and
query, identify the claim, estimate through a supported route, inspect the
contract, and export durable artifacts. The facade covers the established
static, temporal, response, Bayesian, discovery, validation, attribution,
design, and state workflows alongside learner-backed estimation and
structural transport.

<!-- generated:release:crate-dep:begin -->
```toml
[dependencies]
antecedent = "3.0"
```
<!-- generated:release:crate-dep:end -->

A route runs when it has provenance and an independent scientific check;
otherwise it returns a typed `not_supported` refusal that names the missing
evidence. Every reported interval carries its calibration status, and every
result names the provenance records behind the route it executed
(`StudyResult::method_provenance()`, `antecedent::bibliography`), so the
published methods and their documented deviations travel with the answer.

The default `file-source` feature reads Parquet and Arrow IPC tables.
`ml-gbdt`, `ml-forest`, `ml-full` and `ml-neural` enable optional learner
backends; they grant no new scientific claims.

See the [documentation](https://antecedent.readthedocs.io/en/stable/) for
tutorials, guides and the support matrix, and the
[API reference](https://docs.rs/antecedent/) for this crate.
