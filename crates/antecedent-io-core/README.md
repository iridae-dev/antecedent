# antecedent-io-core

The estimator-free layer of the artifact IO: the versioned container and its
reader, CBOR sections, graph interchange, the query, response and
identification wires, canonical digests, the shared sectioned-artifact codec, the
analysis-result artifact and its consumer, and the artifacts that replay without
an estimator. It sits below the estimators; `antecedent-io` re-exports it and adds
the artifacts that replay through an estimator or validator.
