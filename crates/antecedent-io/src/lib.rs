//! Versioned artifact IO for Antecedent.
//!
//! This crate holds the artifacts that replay through an estimator or a
//! validator, and re-exports `antecedent-io-core` (the container, wires,
//! digests and estimator-free artifacts) at its root, so every public path is
//! the one it was before the split.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use antecedent_io_core::{
    AUTO_COMPRESS_MAX_RATIO, AUTO_COMPRESS_MIN_BYTES, AdaptiveBudgetWire, AdmgJson, AdmgWire,
    AnalysisResultConsumption, AnalysisResultContractWire, AnalysisResultHeader,
    AnalysisResultWire, AnalysisTraceWire, ArtifactCheckError, ArtifactContainer, ArtifactKind,
    ArtifactManifest, ArtifactReader, AssignmentDesignWire, AssumptionRecordWire,
    AssumptionSlotWire, AssumptionTagWire, AttestedEvidenceWire, BayesianBindingWire,
    COMPRESSION_ZSTD, CONTAINER_VERSION, CONTENT_BYTES, CONTENT_CBOR, CONTRACT_SECTION,
    CONTRACT_SECTION_FORMAT, CalibrationSlotWire, CalibrationStatus, CausalPayloadHeader,
    CausalPayloadKind, CausalPayloadWire, CausalPosteriorWire, CausalQueryWire, CausalResponseWire,
    CheckedAipwLoweringWire, CheckedAipwRowsWire, CheckedFrontDoorLoweringWire,
    CheckedFunctionalResponseGridWire, CheckedFunctionalResponseMemberWire, CheckedIvLoweringWire,
    CheckedLinearAdjustmentLoweringWire, CheckedNestedCounterfactualWire, ClaimDomainsWire,
    ClaimHostProjection, ClaimIdentityWire, ClaimSectionWire, ClassPriorIdentityWire, CodecError,
    CompatibilityRejectReason, CompatibilityReport, CompressPolicy, ConsumeLimits,
    ContinuousDomainWire, ContractIdentitiesWire, ContractReason, ContrastBundleWire, CpdagJson,
    CpdagWire, DagJson, DagWire, DataPartitionIdentityWire, DataSnapshotIdentityWire,
    DerivationStepWire, DerivativeScaleWire, DerivativeWeightingWire, DesignVariableRole,
    DesignVariableSummary, DiagnosticWire, DiscoveryHeaderWire, DistributionAtomWire,
    DistributionFactorDomainWire, DistributionFactorKeyWire, DistributionFactorLawsWire,
    DistributionFactorRowWire, DistributionFactorTableWire, EdgeEvidenceWire, EncodedArtifact,
    EndpointWire, EstimandFingerprint, EstimatorConfigWire, EstimatorSpecWire,
    ExecutionIdentityWire, ExecutionPerformanceWire, ExposureLevelWire, ExposureMappingWire,
    ExprArenaWire, ExprNodeWire, ExternalComposeIdentityWire, ExternalEstimateAttach,
    ExternalPriorSourceIdentityWire, FormatVersion, FunctionalProgramWire, GlmOptionsWire,
    GraphIdentityWire, GridSpecWire, HedgeCertificateWire, HorizonAdjustmentNodeWire,
    IdentificationEnvelopeWire, IdentificationIdentityWire, IdentificationProductWire,
    IdentificationResultWire, IdentificationSlotWire, IdentificationStatusWire,
    IdentifiedEstimandWire, IdentifiedSetIntervalMethodWire, IdentifiedSetIntervalWire,
    InferenceBindingWire, InferentialCommitmentsWire, InterferenceFunctionalWire,
    InterferenceInferenceWire, InterferencePointwiseIntervalWire, InterferenceQueryWire,
    InterferenceSnapshotWire, IntervalInterpretationWire, InterventionWire,
    InterventionalDistributionQueryWire, InterventionalDistributionWire, IoError, KernelPolicyWire,
    LaggedLinkWire, LinearFitMomentsWire, LocalPolynomialRatioWire, LogicalAnalysisPlanWire,
    LongitudinalRegimeWire, MAGIC, MappedArtifactReader, MappedSection, MarkedEdgeWire,
    MeasurementSpecWire, MechanismSlotWire, MechanismStoreWire, MediationPosteriorSummaryWire,
    ModelKindWire, NestedCounterfactualFitWire, NetworkXAdjacency, NetworkXNodeLink,
    ObligationSectionWire, ObservationAssumptionWire, ObservationIdentityWire,
    ObservationOptionsWire, ObservationSpecWire, OutSection, OverlapPolicyWire,
    PRIOR_SOURCE_META_SECTION, PagJson, PagWire, PanelDidWire, PathSpecificEffectQueryWire,
    PayloadDigestWire, PhysicalExecutionPlanWire, PolicyValueWire, PosteriorAtomGraphWire,
    PosteriorAtomIdentityWire, PosteriorQuantityWire, PriorArtifactBody, PriorCatalog,
    PriorMapping, PriorMappingIdentityWire, PriorSetIdentityWire, PriorSourceMeta, PriorSourceRef,
    PriorSpecIdentityWire, ProgramIdentityWire, ProvenanceGraphWire, ProvenanceNodeWire,
    ProvenanceWire, ROW_WEIGHTS_PAYLOAD, RandomizationDesignWire, RandomizedEffectQueryWire,
    RandomizedEffectWire, RandomizedEstimandWire, RdConfigWire, RdDesignWire, ReasoningSectionWire,
    RecordedContrastWire, RefutationReportWire, ResponseEnvelopeWire, ResponseFunctionalWire,
    ResponseIdentificationWire, ResponseOptionsWire, ResponseQueryWire, ResponseUncertaintyWire,
    ResponseValueWire, STABLE_FORMAT, SUPPORTED_SOURCE_FORMATS, SchemaWire, SchemaWireV01,
    ScoreReuseIdentityWire, ScoreTableKey, SectionAccess, SectionBytes, SectionDescriptor,
    SectionIndexEntry, SectionLoadStats, SemanticVersion, SetInterventionWire, SlotSectionWire,
    SplitIdentityWire, StaticResultWire, StructuralResponseAtomWire, StructuralResponseMixtureWire,
    StructuralWeightBasisWire, SupportDiagnosticWire, SupportRegionWire, SupportReportWire,
    SupportSlotWire, SupportStatusWire, SurvivalDifferenceBandWire, SurvivalWire,
    SyntheticControlWire, SyntheticDidWire, TargetDesign, TargetIdentityWire, TargetPopulationWire,
    TargetWeightsIdentityWire, TargetWeightsSectionWire, TemporalClassIdentityWire,
    TemporalCoordinates, TemporalGraphWire, TemporalIdentificationWire, TemporalMediationGridWire,
    TemporalMediationSliceWire, TemporalMediationUncertaintyWire, TemporalNodeKeyWire,
    TemporalPolicyWire, TemporalResponseSpecWire, TransportFactorView, TransportIdentityWire,
    TransportProofStepView, TransportProofView, TransportQueryWire, UncertaintyComponentWire,
    UncertaintySlotWire, UnitEffectIntervalsWire, UnitEffectsWire, UpliftBinWire, ValueTypeWire,
    ValueWire, VariableSchemaWire, accept_claim, admg_from_dot, admg_from_gml, admg_from_json,
    admg_from_networkx_node_link, admg_from_wire, admg_identity, admg_to_dot, admg_to_gml,
    admg_to_json, admg_to_networkx_node_link, admg_to_wire, analysis_result_artifact,
    artifact_codec, assumptions_to_wire, attach_prior_source_meta, attested_external_estimate,
    bayesian_prior_recalc_artifact, calibration, canonical_dag_wire, canonical_json,
    canonical_temporal_dag_wire, causal_artifact, causal_query_from_wire, causal_query_to_wire,
    causal_query_to_wire_with_registry, causal_response_from_wire, causal_response_to_wire,
    checked_aipw_rows_digest, claim_digest, claim_domains, claim_kind_name, compact_export,
    consume_analysis_result, container, container_kind, contract_seal, contract_section,
    contract_wire, contrast_wire, convert, counterfactual_id_artifact,
    counterfactual_transport_wire, coverage_records_data, cpdag_from_dot, cpdag_from_gml,
    cpdag_from_json, cpdag_from_networkx_node_link, cpdag_from_wire, cpdag_identity, cpdag_to_dot,
    cpdag_to_gml, cpdag_to_json, cpdag_to_networkx_node_link, cpdag_to_wire, cross_world_artifact,
    dag_from_dot, dag_from_gml, dag_from_json, dag_from_networkx_adjacency,
    dag_from_networkx_node_link, dag_from_wire, dag_identity, dag_json_from_str, dag_to_dot,
    dag_to_gml, dag_to_json, dag_to_networkx_adjacency, dag_to_networkx_node_link, dag_to_wire,
    dag_wire_from_dot, dag_wire_from_gml, dag_wire_to_dot, dag_wire_to_gml, data_snapshot_digest,
    decode_analysis_result_artifact, decode_analysis_result_contract,
    decode_causal_payload_artifact, decode_container, decode_external_estimate_claim,
    decode_posterior_artifact, decode_posterior_meta_from_path, decode_posterior_meta_from_seek,
    decode_prior_source_meta, decode_with_version, diagnostic_from_wire, diagnostic_to_wire,
    digest_canonical, digest_hex, digest_wire, discovery_wire, distribution_artifact,
    distribution_replay, encode_analysis_result_artifact,
    encode_analysis_result_artifact_with_contract, encode_causal_payload_artifact,
    encode_container, encode_external_estimate_claim, encode_posterior_artifact,
    encode_prior_source_meta, error, exact_law_wire, executed_functional_labels, execution_digest,
    execution_identity_from_context, expr_arena_from_wire, expr_arena_to_wire, expr_wire,
    external_binding_wire, external_callback_artifact, external_claim_artifact,
    external_column_claim_artifact, external_compose_identity, external_estimate,
    extract_prior_source_meta, from_cbor, frozen_scores_artifact, functional_program_from_wire,
    functional_program_to_wire, graph_dot, graph_gml, graph_json, graph_mixed, graph_networkx,
    graph_posterior_atom_identities, graphless_support, identification_digest,
    identification_from_wire, identification_product_digest, identification_product_digest_wire,
    identification_product_wire, identification_to_wire, identification_to_wire_with_registry,
    identified_set_interval_from_wire, identified_set_interval_to_wire, identity,
    inference_binding_digest, interference_query_from_wire, interference_query_to_wire,
    interventional_distribution_from_wire, interventional_distribution_to_wire,
    is_supported_source, logical_plan_from_wire, logical_plan_to_wire, map_artifact_check,
    mechanism_wire, mechanisms_from_wire, mechanisms_to_wire, migrate, migrate_artifact, mmap_file,
    observation_identity_wire, pack_section, pack_section_shared, pag_from_dot, pag_from_gml,
    pag_from_json, pag_from_networkx_node_link, pag_from_wire, pag_identity, pag_to_dot,
    pag_to_gml, pag_to_json, pag_to_networkx_node_link, pag_to_wire, parse_digest_hex,
    path_specific_from_wire, path_specific_to_wire, payload_digest, peek_version,
    performance_from_wire, performance_to_wire, physical_plan_from_wire, physical_plan_to_wire,
    plan_wire, posterior, posterior_has_named_effect, prior_bank, prior_set_identity,
    program_digest, project_claim_host, project_lossy_scalar, provenance_from_wire,
    provenance_to_wire, provenance_wire, provider_envelope, provider_verification_artifact,
    quantity_wire, query_wire, read_and_migrate, reader, recalc_receipt_artifact, require_digest,
    response_query_from_wire, response_query_to_wire, response_value_from_wire,
    response_value_to_wire, response_wire, result_digest, schema_from_wire, schema_to_wire,
    schema_wire_from_v01, score_reuse_digest, score_wire, search_bounds, section_descriptor,
    section_descriptor_with_policy, sectioned_artifact_enum, sectioned_artifact_error,
    stored_law_bounds, support_empirical, target_weights_digest, temporal_admg_identity,
    temporal_counterfactual_artifact, temporal_cpdag_identity, temporal_dag_from_wire,
    temporal_dag_identity, temporal_dag_to_wire, temporal_pag_identity, temporal_recalc_artifact,
    to_cbor, trace, transport_catalog_wire, transport_certificate, transport_errors,
    transport_identity, transport_point_wire, transport_proof, transport_query_from_wire,
    transport_query_to_wire, transport_query_wire, transported_counterfactual_artifact,
    validate_contract_section, validate_mixture_masses, verify_contract_against_body,
    weight_basis_name, wire,
};

mod artifact_kinds;
mod width;

/// Analysis / estimate / identification / refutation / diagnostic wire types.
pub mod analysis_wire;
mod estimate_wire;
mod transport_effect_wire;

pub use analysis_wire::{
    CandidateSelectionWire, ClipSensitivityWire, EffectEstimateWire, FirstStageDiagnosticsWire,
    JointCovarianceWire, OverlapReportWire, PropensityIntervalWire, ScoreColumnWire,
    ScoreInferenceWire, ScoreTableWire, effect_estimate_from_wire, effect_estimate_to_wire,
    refutation_from_wire, refutation_to_wire,
};

pub mod adaptive_inference_result_artifact;
pub mod admg_conditional_transport_artifact;
pub mod counterfactual_transport_artifact;
pub mod cpdag_completion_artifact;
mod distribution_factor_laws;
pub use distribution_factor_laws::distribution_factor_laws_to_wire;
pub mod joint_bayesian_transport_artifact;
pub mod learned_joint_transport_artifact;
pub mod measured_inference;
pub mod mixed_source_artifact;
pub mod model_bundle;
pub mod model_codes;
pub mod mz_transport_artifact;
pub mod nested_markov_artifact;
pub mod nested_markov_bayesian_artifact;
pub mod posterior_convert;
pub mod recovery_artifact;
pub mod sampled_counterfactual_transport_artifact;
pub mod sampled_recovery_artifact;
pub mod scenario_covariance_artifact;
pub mod temporal_initial_state_artifact;
pub mod temporal_interval_artifact;
pub mod temporal_refresh_artifact;
pub mod temporal_transport_artifact;
pub mod transport_interference_wire;
pub mod transport_scenario_artifact;
pub mod z_transport_artifact;
pub use antecedent_io_core::adaptive_monitoring_contract_artifact;
pub use antecedent_io_core::transport_handoff_artifact;
pub use antecedent_io_core::transport_revalidation_artifact;
pub use antecedent_io_core::callback_result_artifact;
pub use antecedent_io_core::cohort_decision;
pub use antecedent_io_core::composition_verification_artifact;
pub use antecedent_io_core::cross_world_execution_artifact;
pub use antecedent_io_core::joint_law_propagation_artifact;
pub use antecedent_io_core::observation_query_artifact;
pub use antecedent_io_core::selective_discovery_artifact;
pub use antecedent_io_core::{
    bridge_artifact, evidence_ledger_artifact, interface_contract_artifact, measurement_model_wire,
    mechanism_composition_artifact,
};
pub use model_bundle::{
    ModelBundle, ModelBundleEncode, ModelBundleHeaderWire, decode_model_bundle, encode_model_bundle,
};
pub use posterior_convert::{
    PosteriorPayload, decode_causal_posterior_bytes, encode_causal_posterior,
    encode_causal_posterior_bytes, encode_causal_posterior_bytes_with_payload,
    encode_causal_posterior_with_payload,
};
pub use transport_interference_wire::{
    ExposureProbabilityMethodWire, InterferenceEstimateWire, NotCertifiedCertificateWire,
    PopulationFactorWire, RandomizationContrastWire, TransportCertificateWire,
    TransportEffectEstimateWire, TransportFormulaWire, TransportIdentificationWire,
    TransportOverlapDiagnosticWire, interference_estimate_from_wire, interference_estimate_to_wire,
    transport_effect_from_wire, transport_effect_to_wire, transport_identification_from_wire,
    transport_identification_to_wire,
};

#[cfg(test)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, VERSION, ValueType,
    };
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;

    #[test]
    fn schema_and_dag_artifact_round_trip() {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();

        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

        let schema_bytes = to_cbor(&schema_to_wire(&schema)).unwrap();
        let dag_bytes = to_cbor(&dag_to_wire(&dag).unwrap()).unwrap();

        let schema_desc = section_descriptor("schema", "application/cbor", &schema_bytes);
        let dag_desc = section_descriptor("dag", "application/cbor", &dag_bytes);

        let artifact = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: STABLE_FORMAT,
                minimum_reader_version: STABLE_FORMAT,
                artifact_kind: ArtifactKind::SchemaGraph,
                library_version: SemanticVersion::from_crate_version(VERSION)
                    .expect("CARGO_PKG_VERSION"),
                artifact_id: "test-schema-graph".into(),
                sections: vec![schema_desc, dag_desc],
                provenance: ProvenanceWire { note: "roundtrip".into() },
            },
            sections: vec![
                SectionBytes::new("schema", schema_bytes),
                SectionBytes::new("dag", dag_bytes),
            ],
        };

        let mut buf = Vec::new();
        artifact.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        assert_eq!(decoded.manifest.artifact_id, "test-schema-graph");

        let schema_wire: SchemaWire = from_cbor(&decoded.sections[0].data).unwrap();
        assert_eq!(schema_wire.variable_names(), vec!["x", "y"]);
        let dag_wire: DagWire = from_cbor(&decoded.sections[1].data).unwrap();
        let dag2 = dag_from_wire(&dag_wire).unwrap();
        assert!(dag2.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn analysis_trace_artifact_round_trips_assumptions_and_derivation() {
        use std::sync::Arc;

        use antecedent_core::{
            Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
            AssumptionStatus,
        };

        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::CausalMarkov,
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("backdoor") },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        let trace = AnalysisTraceWire {
            assumptions: assumptions_to_wire(&assumptions),
            derivation: vec![DerivationStepWire {
                rule: "backdoor.criterion".into(),
                detail: "Z blocks all backdoor paths".into(),
            }],
            method: "backdoor.adjustment".into(),
            adjustment_set: vec![2],
            support_status: None,
            allowlist_reason: None,
            allowlist_parent: None,
        };
        let bytes = to_cbor(&trace).unwrap();
        let desc = section_descriptor("analysis.trace", "application/cbor", &bytes);
        let artifact = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: STABLE_FORMAT,
                minimum_reader_version: STABLE_FORMAT,
                artifact_kind: ArtifactKind::AnalysisTrace,
                library_version: SemanticVersion::from_crate_version(VERSION)
                    .expect("CARGO_PKG_VERSION"),
                artifact_id: "test-analysis-trace".into(),
                sections: vec![desc],
                provenance: ProvenanceWire { note: "trace".into() },
            },
            sections: vec![SectionBytes::new("analysis.trace", bytes)],
        };
        let mut buf = Vec::new();
        artifact.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        let round: AnalysisTraceWire = from_cbor(&decoded.sections[0].data).unwrap();
        assert_eq!(round.method, "backdoor.adjustment");
        assert_eq!(round.adjustment_set, vec![2]);
        assert_eq!(round.assumptions.len(), 1);
        assert_eq!(round.assumptions[0].assumption, AssumptionTagWire::CausalMarkov);
        assert_eq!(round.derivation[0].rule, "backdoor.criterion");
    }
}

/// Transport grid artifact records and encoding.
pub mod transport_grid_wire;

/// Categorical-treatment artifact (`categorical_treatment_v1`) and its recomputing consumer.
pub mod categorical_treatment_artifact;
/// Dose-grid functional artifact (`dose_grid_functional_row_v1`) and its recomputing consumer.
pub mod dose_grid_functional_artifact;
/// Effect-constancy artifact (`effect_constancy_v1`) and its recomputing consumer.
pub mod effect_constancy_artifact;
/// Latent-class regime-effects artifact (`latent_class_effects_v1`) and its refitting consumer.
pub mod latent_class_artifact;
/// Learned continuous-outcome trial transport artifact verification.
pub mod learned_continuous_artifact;
/// Learner-backed trial transport artifact verification.
pub mod learned_trial_wire;
/// Mechanism discrepancy artifact (`mechanism_discrepancy_v1`) and its recomputing consumer.
pub mod mechanism_discrepancy_artifact;
/// Adapter from a marginal sensitivity model result into the sensitivity artifact.
pub mod msm_sensitivity_adapter;
/// Nonlinear-mediation artifact (`nonlinear_mediation_v1`) and its re-estimating consumer.
pub mod nonlinear_mediation_artifact;
/// Predictive assessment artifact (`predictive_assessment_v1`) and its re-scoring consumer.
pub mod predictive_assessment_artifact;
/// Ordered-response recovery artifact (`ordered_response_recovery_v1`) and its re-deciding consumer.
pub mod recovery_chain_artifact;
/// Sensitivity-surface artifact (`sensitivity_decision_v1`) and its recomputing consumer.
pub mod sensitivity_artifact;
/// Smoothed dose-response transport grid artifact verification.
pub mod smoothed_dose_artifact;
/// Joint vector-treatment artifact (`vector_treatment_v1`) and its recomputing consumer.
pub mod vector_treatment_artifact;

#[cfg(test)]
mod closed_wire_fields {
    use serde::Deserialize;

    fn zeros() -> serde_json::Value {
        serde_json::Value::Array(vec![serde_json::json!(0); 32])
    }

    fn accept_then_reject_unknown<T: for<'de> Deserialize<'de>>(historical: &serde_json::Value) {
        serde_json::from_value::<T>(historical.clone()).expect("historical payload");
        let mut with_unknown = historical.clone();
        with_unknown
            .as_object_mut()
            .expect("object payload")
            .insert("not_a_historical_field".to_owned(), serde_json::Value::Null);
        assert!(
            serde_json::from_value::<T>(with_unknown).is_err(),
            "unknown field must be rejected"
        );
    }

    #[test]
    fn historical_payloads_decode_and_unknown_fields_do_not() {
        let digest = zeros();
        accept_then_reject_unknown::<crate::ContractIdentitiesWire>(&serde_json::json!({
            "target": digest.clone(),
            "identification": digest.clone(),
            "program": digest.clone(),
            "inference_binding": digest.clone(),
            "observation": digest.clone(),
            "data_snapshot": digest,
        }));
        accept_then_reject_unknown::<crate::wire::GraphCpdagWire>(&serde_json::json!({
            "node_count": 2,
            "directed": [[0, 1]],
            "undirected": [],
        }));
        accept_then_reject_unknown::<crate::analysis_wire::ScoreColumnWire>(&serde_json::json!({
            "arm": 1,
            "threshold": null,
        }));
        accept_then_reject_unknown::<crate::plan_wire::LogicalAnalysisPlanWire>(
            &serde_json::json!({
                "plan_id": "p",
                "data_classification": "tabular",
                "discovery_algorithm": null,
                "graph_review_required": false,
                "identifier": null,
                "estimator": null,
                "validation_suite": null,
                "query_variables": [],
            }),
        );
        assert!(
            serde_json::from_value::<crate::plan_wire::LogicalAnalysisPlanWire>(
                serde_json::json!({
                    "plan_id": "p",
                    "data_classification": "public",
                    "discovery_algorithm": null,
                    "graph_review_required": false,
                    "identifier": null,
                    "estimator": null,
                    "validation_suite": null,
                    "query_variables": [],
                })
            )
            .is_err(),
            "a classification outside the historical labels is not a plan"
        );
        accept_then_reject_unknown::<crate::trace::DerivationStepWire>(&serde_json::json!({
            "rule": "backdoor",
            "detail": "adjust",
        }));
        let temporal: crate::prior_bank::TemporalCoordinates =
            serde_json::from_value(serde_json::json!({
                "treatment_lags": [1],
                "horizon": 1,
            }))
            .expect("omitted coefficient_names");
        assert!(temporal.coefficient_names.is_empty());
        accept_then_reject_unknown::<crate::prior_bank::TemporalCoordinates>(&serde_json::json!({
            "treatment_lags": [1],
            "horizon": 1,
        }));
        accept_then_reject_unknown::<crate::mechanism_wire::ParentBasisWire>(&serde_json::json!({
            "n_parents": 0,
            "centers": [],
            "scales": [],
            "knots": [],
            "terms": [],
        }));
        accept_then_reject_unknown::<crate::response_wire::SupportRegionWire>(&serde_json::json!({
            "minima": [0.0],
            "maxima": [1.0],
        }));
        let diagnostic: crate::response_wire::SupportDiagnosticWire =
            serde_json::from_value(serde_json::json!({
                "id": "overlap",
                "values": [1.0],
                "detail": "ok",
            }))
            .expect("omitted scope");
        assert!(diagnostic.scope.is_none());
        accept_then_reject_unknown::<crate::response_wire::SupportDiagnosticWire>(
            &serde_json::json!({
                "id": "overlap",
                "values": [1.0],
                "detail": "ok",
            }),
        );
    }

    #[test]
    fn artifact_kind_keeps_the_forward_compat_other_tag() {
        let known: crate::wire::ArtifactKind = serde_json::from_str(r#""schema_graph""#).unwrap();
        assert_eq!(known, crate::wire::ArtifactKind::SchemaGraph);
        let future: crate::wire::ArtifactKind =
            serde_json::from_str(r#"{"other":"future_kind"}"#).unwrap();
        assert_eq!(future, crate::wire::ArtifactKind::Other("future_kind".into()));
    }
}
