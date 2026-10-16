# antecedent-io

Stable CBOR metadata serialization and versioned
artifact containers with BLAKE3 checksums (ADR 0002).

The container, wire types, canonical digests and estimator-free artifacts live
in `antecedent-io-core` and are re-exported here at the same paths. This crate
adds the artifacts that replay through an estimator or validator.
