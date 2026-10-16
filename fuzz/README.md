# Fuzz targets

Build with nightly + cargo-fuzz:

```bash
cargo +nightly install cargo-fuzz
cargo +nightly fuzz run schema_names
```

Existing targets:

- `schema_names` — schema construction
- `dag_ops` — DAG insert / d-separation
- `kernel_reductions` — masked reductions

P3.5 additions:

- `artifact_container` — `EncodedArtifact::read_from` (length-cap / checksum paths)
- `expression_arena` — `CausalExprArena` intern + estimand method `FromStr`
- `sample_plan` — `LaggedFrame::from_series` with capped dims
- `python_boundary` — DOT/`dag_from_dot` + schema names (Rust side of FFI)
- `arrow_metadata` — malformed Arrow `RecordBatch` → `tabular_from_record_batch`

IO decoder additions:

- `query_wire` — `from_cbor::<CausalQueryWire>` then `causal_query_from_wire`
- `contract_section` — `decode_analysis_result_contract` on a container, and `from_cbor::<AnalysisResultContractWire>` plus `validate_contract_section`
- `artifact_reader` — `MappedArtifactReader::from_bytes` (mmap is file-only)
- `causal_payload` — `decode_causal_payload_artifact` (dispatches payload kinds)

Artifact and declaration parsers (each takes arbitrary bytes and must return a typed error, never panic or allocate past its cap):

- `transport_handoff` — `TransportHandoffArtifact::inspect` / `from_bytes`
- `joint_law_propagation` — `JointLawPropagationArtifactWire::decode`, then `consume` on whatever decodes (replay cost)
- `composition_verification` — `CompositionVerificationArtifact::from_bytes`
- `interface_contract` — `InterfaceContractArtifact::from_bytes` (callback declarations travel inside this schema and inside the handoff and result artifacts; there is no free-standing declaration byte parser)
- `callback_result` — `CallbackResultArtifact::from_bytes`
- `callback_envelope` — `ExternalCallbackArtifact::inspect` and the provider envelope readers
- `inverse_query` — `decode_parts` and `InverseQueryArtifact::from_bytes`
- `design_ranking` — `DesignRankingArtifactWire::from_bytes`
- `decision_artifacts` — decision contract, result, robust and structural result artifacts (first 32 bytes seed the expected identity)
- `adaptive_artifacts` — adaptive inference result and monitoring contract artifacts (checked and unchecked decode)
- `counterfactual_transport` — counterfactual transport wire, transported counterfactual and sampled transport artifacts
- `evidence_ledger` — `EvidenceLedgerArtifact::from_bytes`
- `cohort_decision` — `CohortDecisionSource::consume` (first 32 bytes seed the expected identity)
- `measurement_model` — `MeasurementModelArtifact::from_bytes` (first 32 bytes seed the expected identity)
- `provider_verification_receipt` — `receipt_from_bytes` and `ProviderVerificationReceiptWire::into_receipt` (first 32 bytes seed the retained digest)
- `external_claims` — `ExternalClaimArtifact::from_bytes` and `ExternalColumnClaimArtifact::from_bytes` (first byte is the length of a CBOR expected identity, the rest is the artifact)
- `selective_discovery` — `SelectiveDiscoveryArtifact::inspect` and `from_bytes` (first 32 bytes seed the expected identity)
- `cross_world_execution` — `CrossWorldExecutionArtifact::from_bytes` with and without an expected identity (length-prefixed CBOR, as above)
- `source_evidence` — `SourceEvidence::consume` / `from_result` and `SourceProjectionArtifact::inspect` / `consume`
- `repair_report_artifact` — `RepairReportArtifact::from_bytes`, then `consume` under the default limits
- `rollout_artifact` — `RolloutArtifact::from_bytes` and `consume` (length-prefixed CBOR expectation)
- `sequential_evaluation_artifact` — `SequentialEvaluationArtifact::from_bytes` and `consume`
- `study_plan_artifact` — `StudyPlanArtifactWire` CBOR decode, then `consume_with_limits`
- `transport_artifacts` — joint Bayesian, ADMG conditional (and obstruction), mz and z transport artifacts: `decode` and `consume`
- `decision_json` — `contract_from_json` and `contract_from_json_refusal` (there is no `result_from_json`; results export JSON only)

Corpus notes: seed each target from a valid exported artifact of its kind (the golden fixtures
used by the artifact round-trip tests are the natural seeds), plus truncations and single-byte
flips of them. Semantic-mutation coverage (a resealed edit that must be refused on re-derivation)
is in the per-artifact tests, not the fuzzer; the fuzzer's job is the no-panic and bounded-allocation
property. Targets pass `None` for the expected identity so that decoding, not the identity
comparison, is what is exercised. These targets depend on `antecedent-io-core` and `antecedent-design`
by path; `cargo +nightly fuzz build` refreshes `fuzz/Cargo.lock`.

Not reached: Arrow C Data Interface. The safe ingest is `tabular_from_arrow_c_columns`, and `ArrowCColumn` is only built from an existing Arrow array or an `unsafe` FFI pair. There is no byte-slice parser for that boundary. `arrow_metadata` already fuzzes `tabular_from_record_batch`. The per-artifact `decode` methods are not one public dispatcher.

CI builds every target and does not run the fuzzer:

```bash
cargo +nightly fuzz build
```

Targets exercise core construction; they must not import pinned external baseline source.
