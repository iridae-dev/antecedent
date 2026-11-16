# antecedent-graph

Causal graph types, endpoint semantics, dense node indexes and reusable traversal workspaces.
Each graph class (`Dag`, `Admg`, `Cpdag`, `Pag`, their temporal forms, `CyclicDigraph`,
`CyclicMixedGraph`) is its own type, so edge semantics are never confused. `CyclicDigraph` and
`CyclicMixedGraph` admit directed cycles; `Dag` and `Admg` do not. User-facing names never
enter algorithmic paths.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports the graph types under `antecedent::graph` (and `Dag`, `TemporalDag` at its root). Depend on this crate directly only to manipulate graphs without the analysis layers.

Main types: `Dag`, `Admg`, `Cpdag`, `Pag`, `TemporalDag`, `DenseNodeId`. API reference: <https://docs.rs/antecedent-graph>.

## License

MIT OR Apache-2.0.
