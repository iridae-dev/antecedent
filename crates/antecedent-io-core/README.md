# antecedent-io-core

The estimator-free layer of the artifact IO: the versioned container and its
reader, CBOR and Arrow IPC sections, graph interchange, the query, response and
identification wires, canonical digests, and the shared sectioned-artifact
codec. It sits below the estimators; `antecedent-io` re-exports it and adds the
artifacts that replay through an estimator or validator.
