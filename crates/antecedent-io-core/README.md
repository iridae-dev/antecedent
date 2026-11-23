# antecedent-io-core

The estimator-free layer of the artifact IO: the versioned container and its reader, CBOR and
Arrow section codecs, graph interchange, the query, response and identification wires,
canonical digests, the shared sectioned-artifact codec, the analysis-result artifact and its
consumer, and the artifacts that replay without an estimator. `antecedent-io` re-exports it
and adds the artifacts that replay through an estimator or validator.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports the artifact and interchange functions under `antecedent::io`. Depend on this crate directly only for a tool that reads or writes artifacts without linking the estimators.

Main types: the artifact container, `IoError`, and the graph interchange functions (`dag_to_json`, `dag_from_dot` and their siblings). API reference: <https://docs.rs/antecedent-io-core>.

## License

MIT OR Apache-2.0.
