# antecedent-core

The shared vocabulary of the workspace: identifiers, schemas, assumptions, queries,
interventions, provenance, diagnostics, errors, execution policy, domain-separated contract
identities and common artifact envelopes. It depends on no numerical, graph-algorithm, Arrow
or Python crate.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports the common types (`CausalSchemaBuilder`, `ExecutionContext`, the main queries) in `antecedent::prelude` and the rest under `antecedent::query`, `antecedent::schema` and `antecedent::execution`. Depend on this crate directly only when you build a component that must not depend on the estimators.

Main types: `CausalSchema`, `CausalSchemaBuilder`, `CausalQuery` and its query types, `Intervention`, `ExecutionContext`, `Diagnostic`. API reference: <https://docs.rs/antecedent-core>.

## License

MIT OR Apache-2.0.
