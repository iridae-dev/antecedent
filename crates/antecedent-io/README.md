# antecedent-io

Versioned artifact containers with BLAKE3 checksums and stable CBOR metadata
([ADR 0002](https://github.com/iridae-dev/antecedent/blob/v3.0.0/adr/0002-artifact-encoding.md)). The container, wire types, canonical digests
and estimator-free artifacts live in `antecedent-io-core` and are re-exported here at the same
paths; this crate adds the artifacts that replay through an estimator or validator.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which re-exports the artifact functions under `antecedent::io`. Depend on this crate directly only to encode, decode or replay artifacts without the facade.

Main types: `IoError`, `encode_analysis_result_artifact`, `decode_analysis_result_artifact`, `replay_analysis_result`. API reference: <https://docs.rs/antecedent-io>.

## License

MIT OR Apache-2.0.
