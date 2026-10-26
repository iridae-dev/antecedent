Reviewer handoff record. [Integrated verification](validation.md) records executed results.

# Final IO-core bridge and observation strict-Clippy review

Owner: methods_review agent. Only bridge_distribution.rs and observation_query_artifact.rs edited. Root owns serialized builds/tests/fmt/commits; none were run by reviewer. Source frozen.

## Structured refusal ownership and compatibility

ExternalRefusal (contract/external_refusal.rs) is the shared public by-value structured refusal: code/stage, owned detail/offending/expected/supplied, optional capability and remedy. BridgeRefusal is its public alias. Replacing it with a string would lose negotiation/refusal structure; changing bridge_draws or BridgedDistribution::chain_stage to return Box would change a public Result error type. Concrete consumers include bridge_artifact.rs bridge_draws(...).map_err(refused), where refused takes ExternalRefusal by value, and Python scientific_bridge_api.rs chain_stage matching the structured error. The current public fields/result signatures are compatibility constraints.

Choice: precisely function-scoped clippy::result_large_err allowances on these two existing public methods, with reasons naming the compatibility contract. No file/module/crate-wide allowance; no error-field/layout/API change. Private map_affine/map_correction instead return Box<BridgeRefusal>; bridge_draws unboxes at its existing boundary. Their cold error branches now allocate one small box; successful draw allocation, RNG calls/order, arithmetic, domain tests/refusal selection, canonical identity and output bytes are unchanged. A law.innovation refusal uses the standard From<T> for Box<T>; all original structured fields are retained and unboxed intact at the public return. No artificial new public error wrapper or new public API.

## Observation retained rows and digest conversion

RetainedMechanismWire::Table retains its existing public tuple field verbatim: (parent configuration, level/count table), consumed by retained_from_wire's TableCountRow conversion. Review found a private alias would appear only once and obscure the public wire field shape without giving a concrete consuming caller a useful new interface. Chosen disposition: field-scoped clippy::type_complexity allowance with a reason naming this versioned artifact contract. No alias, nominal type, new export, field/type identity, serde representation or encoded ordering change. The source API renderer continues to see the original tuple.

digest_error now borrows IoError; four map_err adapters borrow their incoming owned error while rendering the same ObservationQueryArtifactError::Encode string. The original errors are dropped normally after conversion. No scientific values, refusal kind/text, identity fields or digest inputs change.

Suggested root checks: strict IO-core/workspace Clippy and existing bridge_distribution/bridge_artifact/observation_query_artifact unit suites, already in serialized queue. No new test was added to mirror these representation-preserving ownership changes. No golden or manifest changes.
