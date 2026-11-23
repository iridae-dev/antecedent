# antecedent-state

Incremental causal state: data events and invalidation tracking, bounded result caches,
retained least-squares and covariance summaries that refresh with new rows, conjugate
refresh, and spill-to-disk retention.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which wraps this crate under `antecedent::state`, publishing results only against the data version they were computed from. Depend on this crate directly only to embed the state machinery without the facade.

Main types: `CausalState`, `StateEvent`, `OlsRefresh`, `ConjugateOlsPosterior`, `SpillRegistry`. API reference: <https://docs.rs/antecedent-state>.

## License

MIT OR Apache-2.0.
