//! Graph interchange (JSON, DOT, GML, `NetworkX`) and durable, versioned artifacts of results,
//! posteriors and models, with their replay.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io::{
    AnalysisResultConsumption, AnalysisResultContractWire, AnalysisResultLoad,
    AssignmentDesignWire, CONTRACT_SECTION, ExposureLevelWire, ExposureMappingWire,
    ExposureProbabilityMethodWire, InterferenceEstimateWire, InterferenceFunctionalWire,
    InterferenceQueryWire, NotCertifiedCertificateWire, PopulationFactorWire,
    RandomizationContrastWire, ReplayOptions, ReplayReport, TransportCertificateWire,
    TransportEffectEstimateWire, TransportFormulaWire, TransportIdentificationWire,
    TransportOverlapDiagnosticWire, TransportQueryWire, admg_from_dot, admg_from_gml,
    admg_from_json, admg_from_networkx_node_link, admg_to_dot, admg_to_gml, admg_to_json,
    admg_to_networkx_node_link, assumptions_to_wire, cpdag_from_dot, cpdag_from_gml,
    cpdag_from_json, cpdag_from_networkx_node_link, cpdag_to_dot, cpdag_to_gml, cpdag_to_json,
    cpdag_to_networkx_node_link, dag_from_dot, dag_from_gml, dag_from_json,
    dag_from_networkx_adjacency, dag_from_networkx_node_link, dag_to_dot, dag_to_gml, dag_to_json,
    dag_to_networkx_adjacency, dag_to_networkx_node_link,
    decode_analysis_result_artifact_expecting, decode_analysis_result_contract,
    decode_causal_posterior_bytes_expecting, encode_analysis_result_artifact,
    encode_analysis_result_artifact_with_contract, encode_causal_posterior_bytes,
    interference_estimate_from_wire, interference_estimate_to_wire, interference_query_from_wire,
    interference_query_to_wire, load_analysis_result_expecting, pag_from_dot, pag_from_gml,
    pag_from_json, pag_from_networkx_node_link, pag_to_dot, pag_to_gml, pag_to_json,
    pag_to_networkx_node_link, pag_to_wire, replay_analysis_result_expecting, replay_loaded_result,
    transport_effect_from_wire, transport_effect_to_wire, transport_identification_from_wire,
    transport_identification_to_wire, transport_query_from_wire, transport_query_to_wire,
};

pub use antecedent_io::graph_dot::dag_with_names_from_dot;
pub use antecedent_io::graph_gml::dag_with_names_from_gml;
pub use antecedent_io::graph_json::dag_with_names_from_json;
pub use antecedent_io::graph_mixed::{
    admg_with_names_from_dot, admg_with_names_from_gml, admg_with_names_from_json,
    admg_with_names_from_networkx_node_link, cpdag_with_names_from_dot, cpdag_with_names_from_gml,
    cpdag_with_names_from_json, cpdag_with_names_from_networkx_node_link, pag_with_names_from_dot,
    pag_with_names_from_gml, pag_with_names_from_json, pag_with_names_from_networkx_node_link,
};
pub use antecedent_io::graph_networkx::{
    dag_with_names_from_networkx_adjacency, dag_with_names_from_networkx_node_link,
};
/// Admit an artifact only as the artifact the consumer retained: every loader's `*_expecting`
/// form binds the bytes to this identity before it reads them.
pub use antecedent_io::{
    RetainedArtifactIdentity, artifact_identity, bind_retained_artifact, load_expecting,
};
/// Format 0.6 compiled study and portable estimator replay.
pub mod compiled_study {
    pub use crate::analysis::compiled_study::*;
}
/// Accept an exported claim only as the claim the consumer retained.
pub use antecedent_io::{RetainedClaimIdentity, accept_claim_expecting};

use crate::error::CausalError;

/// Encode a model bundle to durable bytes.
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn encode_model_bundle_bytes(
    input: &antecedent_io::ModelBundleEncode<'_>,
) -> Result<Vec<u8>, CausalError> {
    let art = antecedent_io::encode_model_bundle(input).map_err(CausalError::from)?;
    let mut buf = Vec::new();
    art.write_to(&mut buf).map_err(CausalError::from)?;
    Ok(buf)
}

/// Decode a model bundle from durable bytes (migrates format if needed), without a retained
/// identity; [`decode_model_bundle_bytes_expecting`] admits only the retained bundle.
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
#[deprecated(
    since = "3.1.0",
    note = "use decode_model_bundle_bytes_expecting with a RetainedArtifactIdentity"
)]
pub fn decode_model_bundle_bytes(bytes: &[u8]) -> Result<antecedent_io::ModelBundle, CausalError> {
    decode_model_bundle(bytes)
}

fn decode_model_bundle(bytes: &[u8]) -> Result<antecedent_io::ModelBundle, CausalError> {
    let art = antecedent_io::read_and_migrate(bytes).map_err(CausalError::from)?;
    antecedent_io::decode_model_bundle(&art).map_err(CausalError::from)
}

/// [`decode_model_bundle_bytes`], admitting only the bundle the consumer retained: the bytes
/// bind to `retained` ([`antecedent_io::bind_retained_artifact`]) before they are read.
///
/// # Errors
///
/// As [`decode_model_bundle_bytes`], and `external_binding_mismatch` when the bytes are not
/// the retained bundle.
pub fn decode_model_bundle_bytes_expecting(
    bytes: &[u8],
    retained: &antecedent_io::RetainedArtifactIdentity,
) -> Result<antecedent_io::ModelBundle, CausalError> {
    antecedent_io::bind_retained_artifact(bytes, retained).map_err(CausalError::from)?;
    decode_model_bundle(bytes)
}

/// Deprecated: admits the artifact without an identity its consumer retained.
///
/// [`load_analysis_result_expecting`] admits only the retained artifact. This form calls
/// `antecedent_io::load_analysis_result` unchanged and is removed in 4.0.
///
/// # Errors
///
/// As `antecedent_io::load_analysis_result`.
#[deprecated(
    since = "3.1.0",
    note = "use load_analysis_result_expecting with a RetainedArtifactIdentity"
)]
pub fn load_analysis_result(
    bytes: &[u8],
) -> Result<antecedent_io::AnalysisResultLoad, antecedent_io::IoError> {
    antecedent_io::load_analysis_result(bytes)
}

/// Deprecated: admits the artifact without an identity its consumer retained.
///
/// [`replay_analysis_result_expecting`] admits only the retained artifact. This form calls
/// `antecedent_io::replay_analysis_result` unchanged and is removed in 4.0.
///
/// # Errors
///
/// As `antecedent_io::replay_analysis_result`.
#[deprecated(
    since = "3.1.0",
    note = "use replay_analysis_result_expecting with a RetainedArtifactIdentity"
)]
pub fn replay_analysis_result(
    bytes: &[u8],
) -> Result<antecedent_io::AnalysisResultConsumption, antecedent_io::IoError> {
    antecedent_io::replay_analysis_result(bytes)
}

/// Deprecated: admits the artifact without an identity its consumer retained.
///
/// [`decode_analysis_result_artifact_expecting`] admits only the retained artifact. This form calls
/// `antecedent_io::decode_analysis_result_artifact` unchanged and is removed in 4.0.
///
/// # Errors
///
/// As `antecedent_io::decode_analysis_result_artifact`.
#[deprecated(
    since = "3.1.0",
    note = "use decode_analysis_result_artifact_expecting with a RetainedArtifactIdentity"
)]
pub fn decode_analysis_result_artifact(
    bytes: &[u8],
) -> Result<
    (
        antecedent_io::EncodedArtifact,
        antecedent_io_core::AnalysisResultHeader,
        antecedent_io::AnalysisResultWire,
    ),
    antecedent_io::IoError,
> {
    antecedent_io::decode_analysis_result_artifact(bytes)
}

/// Deprecated: admits the artifact without an identity its consumer retained.
///
/// [`decode_causal_posterior_bytes_expecting`] admits only the retained artifact. This form calls
/// `antecedent_io::decode_causal_posterior_bytes` unchanged and is removed in 4.0.
///
/// # Errors
///
/// As `antecedent_io::decode_causal_posterior_bytes`.
#[deprecated(
    since = "3.1.0",
    note = "use decode_causal_posterior_bytes_expecting with a RetainedArtifactIdentity"
)]
pub fn decode_causal_posterior_bytes(
    bytes: &[u8],
) -> Result<(antecedent_io::CausalPosteriorWire, Vec<f64>), antecedent_io::IoError> {
    antecedent_io::decode_causal_posterior_bytes(bytes)
}

/// Deprecated: admits the artifact without an identity its consumer retained.
///
/// [`decode_causal_payload_artifact_expecting`] admits only the retained artifact. This form calls
/// `antecedent_io::decode_causal_payload_artifact` unchanged and is removed in 4.0.
///
/// # Errors
///
/// As `antecedent_io::decode_causal_payload_artifact`.
#[deprecated(
    since = "3.1.0",
    note = "use decode_causal_payload_artifact_expecting with a RetainedArtifactIdentity"
)]
pub fn decode_causal_payload_artifact(
    bytes: &[u8],
) -> Result<
    (
        antecedent_io::EncodedArtifact,
        antecedent_io_core::causal_artifact::CausalPayloadHeader,
        antecedent_io::CausalPayloadWire,
    ),
    antecedent_io::IoError,
> {
    antecedent_io::decode_causal_payload_artifact(bytes)
}

/// Deprecated: admits the artifact without an identity its consumer retained.
///
/// [`decode_prior_source_meta_expecting`] admits only the retained artifact. This form calls
/// `antecedent_io::decode_prior_source_meta` unchanged and is removed in 4.0.
///
/// # Errors
///
/// As `antecedent_io::decode_prior_source_meta`.
#[deprecated(
    since = "3.1.0",
    note = "use decode_prior_source_meta_expecting with a RetainedArtifactIdentity"
)]
pub fn decode_prior_source_meta(
    bytes: &[u8],
) -> Result<antecedent_io::PriorSourceMeta, antecedent_io::IoError> {
    antecedent_io::decode_prior_source_meta(bytes)
}

/// Portable checked transport proof and supplied exact-law records.
pub use antecedent_io::{exact_law_wire::ExactLawWire, transport_proof::TransportProofWire};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Portable policy programs: the items of `antecedent-io::policy_program_artifact`, reached as
/// `io::policy_program_artifact::<Item>`.
pub mod policy_program_artifact {
    pub use antecedent_io::policy_program_artifact::{
        POLICY_PROGRAM_MAGIC, POLICY_PROGRAM_MAX_BYTES, POLICY_PROGRAM_MAX_INTERVENTIONS,
        POLICY_PROGRAM_RECOVERABLE_VERSION, POLICY_PROGRAM_SUPPORTED_VERSIONS,
        POLICY_PROGRAM_VERSION, decode_policy_program_expecting,
        decode_policy_program_with_expecting, encode_policy_program, interventions_from_named_json,
        named_json_from_interventions, policy_program_callback_records, policy_program_identity,
        value_expression_identity,
    };

    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`decode_policy_program_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_io::policy_program_artifact::decode_policy_program` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_io::policy_program_artifact::decode_policy_program`.
    #[deprecated(
        since = "3.1.0",
        note = "use decode_policy_program_expecting with a RetainedArtifactIdentity"
    )]
    pub fn decode_policy_program(
        bytes: &[u8],
    ) -> Result<Vec<antecedent_core::Intervention>, antecedent_io::IoError> {
        antecedent_io::policy_program_artifact::decode_policy_program(bytes)
    }

    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`decode_policy_program_with_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_io::policy_program_artifact::decode_policy_program_with` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_io::policy_program_artifact::decode_policy_program_with`.
    #[deprecated(
        since = "3.1.0",
        note = "use decode_policy_program_with_expecting with a RetainedArtifactIdentity"
    )]
    pub fn decode_policy_program_with(
        bytes: &[u8],
        supply: &dyn Fn(
            &antecedent_core::RecoveryRecord,
        ) -> Option<std::sync::Arc<dyn antecedent_core::BatchCallback>>,
    ) -> Result<Vec<antecedent_core::Intervention>, antecedent_io::IoError> {
        antecedent_io::policy_program_artifact::decode_policy_program_with(bytes, supply)
    }
}
/// Estimate conversions for the transportability and randomized-interference wires.
pub use antecedent_io::try_interference_estimate_from_wire;
/// `antecedent-io` items at the crate root.
pub use antecedent_io::{
    AnalysisResultWire, CausalPayloadKind, CausalPayloadWire, CausalPosteriorWire, CausalQueryWire,
    CausalResponseWire, CompatibilityRejectReason, CompatibilityReport, DagWire,
    DesignVariableRole, DesignVariableSummary, EncodedArtifact, EstimandFingerprint, ExprArenaWire,
    ExternalEstimateAttach, FunctionalProgramWire, IdentifiedSetIntervalMethodWire,
    IdentifiedSetIntervalWire, IoError, ModelKindWire, ParticipantBindingWire,
    PosteriorQuantityWire, PriorCatalog, PriorMapping, PriorSourceMeta, PriorSourceRef,
    STABLE_FORMAT, StaticResultWire, TargetDesign, admg_from_wire, admg_to_wire,
    causal_query_from_wire, causal_query_to_wire, causal_query_to_wire_with_registry,
    causal_response_from_wire, causal_response_to_wire, dag_from_wire, dag_to_wire,
    decode_causal_payload_artifact_expecting, decode_posterior_artifact,
    decode_prior_source_meta_expecting, encode_causal_payload_artifact,
    encode_external_estimate_claim, encode_posterior_artifact, encode_prior_source_meta,
    executed_functional_labels, execution_digest, execution_identity_from_context,
    expr_arena_from_wire, expr_arena_to_wire, from_cbor, functional_program_from_wire,
    functional_program_to_wire, identification_to_wire_with_registry,
    identified_set_interval_from_wire, parse_digest_hex, temporal_dag_to_wire, to_cbor,
};
/// Model bundle artifact encode/decode.
pub use antecedent_io::{ModelBundleEncode, ModelBundleHeaderWire};
/// Wire types for durable artifacts (not internal Rust structs).
pub use antecedent_io_core::{FormatVersion, SemanticVersion};
/// Adaptive inference result artifact (`adaptive_inference_result_v1`).
pub mod adaptive_inference_result_artifact {
    pub use antecedent_io::adaptive_inference_result_artifact::AdaptiveInferenceResultArtifact;
}
/// Items of `antecedent-io::adaptive_monitoring_contract_artifact`.
pub mod adaptive_monitoring_contract_artifact {
    pub use antecedent_io::adaptive_monitoring_contract_artifact::{
        AdaptiveMonitoringContractArtifact, DeclarationWire, RuleWire, contract_boundaries,
        declaration_from_wire, declaration_to_wire,
    };
}
/// Independent point-result artifacts for ADMG conditional transport.
pub mod admg_conditional_transport_artifact {
    pub use antecedent_io::admg_conditional_transport_artifact::{
        AdmgConditionalArtifactWire, AdmgConditionalConsumeLimits, AdmgConditionalObstructionWire,
    };
}
/// Exact funded JSON encoding of borrowed native read models.
pub mod admitted_json {
    pub use antecedent_io_core::admitted_json::AdmittedJson;
}
/// Replay artifact for the treatment-propensity nuisance product of a prepared study.
pub mod array_fit_artifact {
    pub use antecedent_io::array_fit_artifact::inspect_array_fit_header;
}
/// Items of `antecedent-io::bayesian_prior_recalc_artifact`.
pub mod bayesian_prior_recalc_artifact {
    pub use antecedent_io::bayesian_prior_recalc_artifact::{
        BOUND_PRIOR_MAGIC, BoundBayesianPriorWire,
    };
}
/// Calibration match: one reported interval against the coverage records.
pub mod calibration {
    pub use antecedent_io_core::calibration::CONSTRUCTION_NOT_MEASURED;
}
/// Items of `antecedent-io::canonical_json`.
pub mod canonical_json {
    pub use antecedent_io::canonical_json::canonical_json_digest;
}
/// Choice inference result artifact (`choice_inference_result_v1`).
pub mod choice_inference_result_artifact {
    pub use antecedent_io::choice_inference_result_artifact::{
        ChoiceDeclarationWire, ChoiceInferenceResultArtifact, ChoiceSpecWire,
        MAX_CHOICE_INFERENCE_RESULT_BYTES, declaration_from_wire, spec_from_wire,
    };
}
/// Items of `antecedent-io::composition_verification_artifact`.
pub mod composition_verification_artifact {
    pub use antecedent_io::composition_verification_artifact::{
        CompositionVerificationArtifact, CompositionVerificationArtifactError,
    };
}
/// Items of `antecedent-io::container`.
pub mod container {
    pub use antecedent_io::container::MAGIC;
}
/// Items of `antecedent-io::contract_section`.
pub mod contract_section {
    pub use antecedent_io::contract_section::CalibrationSlotWire;
    pub use antecedent_io_core::contract_section::CalibrationStatus;
    /// Items of `antecedent-io::contract_section::cyclic`.
    pub mod cyclic {
        pub use antecedent_io::contract_section::cyclic::{
            CyclicContractWire, CyclicIdentificationWire, CyclicTargetWire,
        };
    }
}
/// Items of `antecedent-io::convert`.
pub mod convert {
    pub use antecedent_io::convert::from_cbor;
}
/// Items of `antecedent-io::counterfactual_id_artifact`.
pub mod counterfactual_id_artifact {
    pub use antecedent_io::counterfactual_id_artifact::CounterfactualIdArtifactError;
}
/// Independent replay of general finite counterfactual transport artifacts (kind.
pub mod counterfactual_transport_artifact {
    pub use antecedent_io::counterfactual_transport_artifact::CounterfactualTransportConsumeLimits;
}
/// Items of `antecedent-io::counterfactual_transport_wire`.
pub mod counterfactual_transport_wire {
    pub use antecedent_io::counterfactual_transport_wire::{
        CounterfactualTransportArtifactWire, CounterfactualTransportExpectedIdentity,
    };
}
/// Independent artifacts for the DAG completions of a supplied CPDAG.
pub mod cpdag_completion_artifact {
    pub use antecedent_io::cpdag_completion_artifact::{
        CpdagCompletionArtifactWire, CpdagConsumeLimits,
    };
}
/// Items of `antecedent-io::cross_world_execution_artifact`.
pub mod cross_world_execution_artifact {
    pub use antecedent_io::cross_world_execution_artifact::{CrossWorldRequestWire, request_of};
}
/// Items of `antecedent-io::cyclic_family_artifact`.
pub mod cyclic_family_artifact {
    pub use antecedent_io::cyclic_family_artifact::{
        CyclicWorldOutcomeAnswer, ExperimentalRecoveryReport, RecoveredDrawAnswer,
        RecoveredDrawStatusWire, RecoveredWorldAnswer,
    };
}
/// Items of `antecedent-io::cyclic_graph`.
pub mod cyclic_graph {
    pub use antecedent_io::cyclic_graph::CyclicGraphReport;
}
/// Items of `antecedent-io::cyclic_system_artifact`.
pub mod cyclic_system_artifact {
    pub use antecedent_io::cyclic_system_artifact::{
        CyclicArtifactError, MAX_CYCLIC_ARTIFACT_BYTES,
    };
}
/// Items of `antecedent-io::distribution_artifact`.
pub mod distribution_artifact {
    pub use antecedent_io::distribution_artifact::{
        DistributionCalibration, DistributionIdentity, DistributionMetadata, DistributionTrust,
        DrawAlignment, MAX_DISTRIBUTION_ARTIFACT_BYTES, MAX_DISTRIBUTION_COORDINATES,
        MAX_DISTRIBUTION_DRAWS,
    };
}
/// Items of `antecedent-io::external_binding_wire`.
pub mod external_binding_wire {
    pub use antecedent_io::external_binding_wire::{
        ColumnResponseWire, ContractWire, MAX_BINDING_WIRE_BYTES, ProviderWire, RefusalWire,
        ResponseWire, bind_column, bind_response, bind_response_to_program, core_contract,
        native_contract_identity, response_from_wire,
    };
}
/// Items of `antecedent-io::external_callback_artifact`.
pub mod external_callback_artifact {
    pub use antecedent_io::external_callback_artifact::ExternalCallbackArtifact;
}
/// Items of `antecedent-io::external_claim_artifact`.
pub mod external_claim_artifact {
    pub use antecedent_io::external_claim_artifact::{
        ExternalClaimArtifact, ExternalClaimIdentity, ExternalClaimTrust,
        MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES, VerificationProbeWire, lineage_wire,
    };
}
/// Items of `antecedent-io::external_column_claim_artifact`.
pub mod external_column_claim_artifact {
    pub use antecedent_io::external_column_claim_artifact::{
        ExternalColumnClaimArtifact, ExternalColumnIdentity,
        MAX_EXTERNAL_COLUMN_CLAIM_ARTIFACT_BYTES, MAX_EXTERNAL_COLUMN_ROWS,
    };
}
/// Items of `antecedent-io::external_cyclic_artifact`.
pub mod external_cyclic_artifact {
    pub use antecedent_io::external_cyclic_artifact::{
        ExternalCyclicRequest, MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES,
        consume_external_cyclic_artifact, consume_external_cyclic_artifact_expecting,
        inspect_external_cyclic_artifact,
    };
}
/// Feature and family-payload versions, separate from package `SemVer`.
pub mod feature_version {
    pub use antecedent_io_core::feature_version::{
        FeatureRead, FeatureRefusal, FeatureVersion, RequestedVersion, classify_feature_read,
    };
}
/// Items of `antecedent-io::frozen_scores_artifact`.
pub mod frozen_scores_artifact {
    pub use antecedent_io::frozen_scores_artifact::{FrozenScoreTable, FrozenScoresArtifactError};
}
/// Canonical encoding and domain-separated digests for contract identities.
pub mod identity {
    pub use antecedent_io::identity::digest_wire;
}
/// Items of `antecedent-io::interface_contract_artifact`.
pub mod interface_contract_artifact {
    pub use antecedent_io::interface_contract_artifact::{
        ContractWire, contract_from_wire, contract_to_wire,
    };
}
/// Independent artifact of the joint Bayesian source-target transport engine.
pub mod joint_bayesian_transport_artifact {
    pub use antecedent_io::joint_bayesian_transport_artifact::{
        DataIdentityWire, GaussianPriorWire, JointBayesianArtifactWire, JointBayesianConsumeLimits,
        JointBayesianExpectation, PriorProvenanceWire,
    };
}
/// Items of `antecedent-io::joint_law_propagation_artifact`.
pub mod joint_law_propagation_artifact {
    pub use antecedent_io::joint_law_propagation_artifact::{PropagationRequestWire, float_bits};
}
/// Exact frozen joint-panel execution, replayed by the existing model engine.
pub mod joint_panel_artifact {
    pub use antecedent_io::joint_panel_artifact::{
        PanelArtifactError, consume_panel_execution_expecting,
    };
    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`consume_panel_execution_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_io::joint_panel_artifact::consume_panel_execution` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_io::joint_panel_artifact::consume_panel_execution`.
    #[deprecated(
        since = "3.1.0",
        note = "use consume_panel_execution_expecting with a RetainedArtifactIdentity"
    )]
    pub fn consume_panel_execution(
        bytes: &[u8],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<
        antecedent_io::joint_panel_artifact::HistoricalPanelExecution,
        antecedent_io::joint_panel_artifact::PanelArtifactError,
    > {
        antecedent_io::joint_panel_artifact::consume_panel_execution(bytes, ctx)
    }
}
/// Independent point and analytic-interval artifacts for learned continuous-outcome trial.
pub mod learned_continuous_artifact {
    pub use antecedent_io::learned_continuous_artifact::{
        LearnedContinuousArtifactError, LearnedContinuousConsumeLimits,
    };
}
/// Independent artifact of the learned joint source-target transport row.
pub mod learned_joint_transport_artifact {
    pub use antecedent_io::learned_joint_transport_artifact::{
        LearnedJointArtifactWire, LearnedJointConsumeLimits, LearnedJointExpectation,
    };
}
/// Native measured interval authority, separate from unchanged candidate artifacts.
pub mod measured_inference {
    pub use antecedent_io::measured_inference::{
        MAX_MEASURED_ARTIFACT_BYTES, MeasuredExpectation, MeasuredInferenceReport,
    };
    /// Whole-method measured interval for an effect estimated on a sample split after discovery.
    pub mod split_effect {
        pub use antecedent_io::measured_inference::split_effect::SplitEffectSource;
    }
    /// Whole-method measured interval for an effect estimated on a sample split after discovery.
    pub mod split_parent_set_effect {
        pub use antecedent_io::measured_inference::split_parent_set_effect::ParentSetEffectSource;
    }
}
/// Items of `antecedent-io::measurement_model_wire`.
pub mod measurement_model_wire {
    pub use antecedent_io::measurement_model_wire::model_from_wire;
}
/// Items of `antecedent-io::method_provenance`.
pub mod method_provenance {
    pub use antecedent_io::method_provenance::{
        MethodProvenanceWire, bibliography, capability_provenance,
        decode_method_provenance_section, method_provenance,
    };
}
/// Independent point-result artifacts for mixed-source proof search.
pub mod mixed_source_artifact {
    pub use antecedent_io::mixed_source_artifact::{
        MixedSourceArtifactError, MixedSourceArtifactWire, MixedSourceConsumeLimits,
    };
}
/// Source-bound reconstruction of the declared-shift MNAR candidate.
pub mod mnar_candidate_source {
    pub use antecedent_io::mnar_candidate_source::decode_source_expecting;
    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`decode_source_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_io::mnar_candidate_source::decode_source` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_io::mnar_candidate_source::decode_source`.
    #[deprecated(
        since = "3.1.0",
        note = "use decode_source_expecting with a RetainedArtifactIdentity"
    )]
    pub fn decode_source(
        bytes: &[u8],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<antecedent_io::mnar_candidate_source::ReconstructedMnarSource, antecedent_io::IoError>
    {
        antecedent_io::mnar_candidate_source::decode_source(bytes, ctx)
    }
}
/// Independent point-result artifacts for multi-source limited-experiment transport.
pub mod mz_transport_artifact {
    pub use antecedent_io::mz_transport_artifact::{
        MzContrastWire, MzTransportArtifactError, MzTransportArtifactWire,
        MzTransportConsumeLimits, MzUncertaintyWire,
    };
}
/// Independent artifact of the bounded binary nested-Markov likelihood pilot (record.
pub mod nested_markov_artifact {
    pub use antecedent_io::nested_markov_artifact::{
        NestedFisherArtifact, NestedMarkovConsumeLimits, NestedMarkovExpectation,
    };
}
/// Independently replayed, calibration-internal continuous Verma posterior candidate.
pub mod nested_markov_bayesian_artifact {
    pub use antecedent_io::nested_markov_bayesian_artifact::{Artifact, Expectation, Limits};
}
/// Replayable artifact for an observation-conditioned scientific query.
pub mod observation_query_artifact {
    pub use antecedent_io_core::observation_query_artifact::ObservationQueryArtifactWire;
}
/// Independent artifact for an autoregressive mechanism refreshed across observation periods.
pub mod period_refresh_artifact {
    pub use antecedent_io::period_refresh_artifact::{
        PeriodRefreshArtifactWire, PriorWire, product_label,
    };
}
/// Wire types and encode/decode for columnar posterior artifacts.
pub mod posterior {
    pub use antecedent_io_core::posterior::validate_posterior_meta;
}
/// Items of `antecedent-io::provider_envelope`.
pub mod provider_envelope {
    pub use antecedent_io::provider_envelope::{
        ProviderEnvelope, ProviderEnvelopeHeader, inspect_provider_envelope, is_provider_envelope,
        open_provider_envelope_expecting, open_verified_provider_envelope,
        open_verified_provider_envelope_expecting, seal_provider_envelope,
    };
    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`open_provider_envelope_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_io::provider_envelope::open_provider_envelope` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_io::provider_envelope::open_provider_envelope`.
    #[deprecated(
        since = "3.1.0",
        note = "use open_provider_envelope_expecting with a RetainedArtifactIdentity"
    )]
    pub fn open_provider_envelope(
        bytes: &[u8],
    ) -> Result<antecedent_io::provider_envelope::ProviderEnvelope, String> {
        antecedent_io::provider_envelope::open_provider_envelope(bytes)
    }
}
/// Items of `antecedent-io::provider_verification_artifact`.
pub mod provider_verification_artifact {
    pub use antecedent_io::provider_verification_artifact::{
        ExecutionVerificationWire, MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES,
        ProviderVerificationReceiptWire, receipt_from_bytes, receipt_from_bytes_expecting,
        receipt_to_bytes, verify_execution_requests,
    };
}
/// Items of `antecedent-io::quantity_wire`.
pub mod quantity_wire {
    pub use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
}
/// Full `CausalQuery` and Intervention wire forms.
pub mod query_wire {
    pub use antecedent_io_core::query_wire::{
        CausalQueryWire, RandomizedEffectQueryWire, causal_query_from_wire,
    };
}
/// Items of `antecedent-io::reader`.
pub mod reader {
    pub use antecedent_io::reader::ArtifactReader;
}
/// Items of `antecedent-io::recalc_receipt_artifact`.
pub mod recalc_receipt_artifact {
    pub use antecedent_io::recalc_receipt_artifact::{
        CapabilitiesWire, CountsWire, DeclaredStageWire, PlanEntryWire, PlanWire,
        RecalcReceiptArtifact, RecalcReceiptArtifactError, ReceiptEntryWire, ResumeWire,
        identities_from_wire, identities_to_wire, plan_from_wire, plan_to_wire, stage_from_label,
    };
}
/// Independent point-result artifacts for exact binary observation recovery.
pub mod recovery_artifact {
    pub use antecedent_io::recovery_artifact::{RecoveryArtifactWire, RecoveryConsumeLimits};
}
/// Migration from the previous release onto the current contracts.
pub mod release_migration {
    pub use antecedent_io::release_migration::{
        CurrentMethod, MethodAssignment, PreviousReleaseError, RouteRecordError,
        RouteRecordOutcome, consume_previous_release_artifact_expecting,
        consume_previous_route_records_expecting, current_score_row_digest,
        narrow_score_row_digest,
    };
    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`consume_previous_release_artifact_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_io::release_migration::consume_previous_release_artifact` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_io::release_migration::consume_previous_release_artifact`.
    #[deprecated(
        since = "3.1.0",
        note = "use consume_previous_release_artifact_expecting with a RetainedArtifactIdentity"
    )]
    pub fn consume_previous_release_artifact(
        bytes: &[u8],
        target: &antecedent_core::ScientificTarget,
    ) -> Result<
        antecedent_io::release_migration::AdoptedRelease,
        antecedent_io::release_migration::PreviousReleaseError,
    > {
        antecedent_io::release_migration::consume_previous_release_artifact(bytes, target)
    }

    /// Deprecated: admits the artifact without an identity its consumer retained.
    ///
    /// [`consume_previous_route_records_expecting`] admits only the retained artifact. This form calls
    /// `antecedent_io::release_migration::consume_previous_route_records` unchanged and is removed in 4.0.
    ///
    /// # Errors
    ///
    /// As `antecedent_io::release_migration::consume_previous_route_records`.
    #[deprecated(
        since = "3.1.0",
        note = "use consume_previous_route_records_expecting with a RetainedArtifactIdentity"
    )]
    pub fn consume_previous_route_records(
        bytes: &[u8],
        current: &[antecedent_io::release_migration::CurrentMethod],
    ) -> Result<
        Vec<antecedent_io::release_migration::RouteRecordMigration>,
        antecedent_io::release_migration::PreviousReleaseError,
    > {
        antecedent_io::release_migration::consume_previous_route_records(bytes, current)
    }
}
/// Independent replay of sampled counterfactual transport artifacts (kind.
pub mod sampled_counterfactual_transport_artifact {
    pub use antecedent_io::sampled_counterfactual_transport_artifact::{
        SampledCounterfactualTransportArtifactWire, SampledCounterfactualTransportConsumeLimits,
        SampledCounterfactualTransportExpectedIdentity, consume_sampled_counterfactual_transport,
        consume_sampled_counterfactual_transport_expecting,
    };

    /// Estimate and seal: runs the sampled estimator on the stored derivation and
    /// binds counts, derivation, estimate, uncertainty method and overlap.
    ///
    /// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
    /// names. This forwards to [`antecedent_io::sampled_counterfactual_transport_artifact::sampled_counterfactual_transport_artifact`].
    ///
    /// # Errors
    /// The estimator's typed refusal (as `Error::Replay`) or a shape failure.
    #[deprecated(
        since = "3.1.0",
        note = "use the compiled SampledCounterfactualTransport route, which calls this kernel function with the same inputs"
    )]
    pub fn sampled_counterfactual_transport_artifact(derivation: &antecedent_identify::CounterfactualTransportDerivation, request: &antecedent_estimate::counterfactual_transport_sampled::SampledTransportRequest, variable_names: &[String], ctx: &antecedent_core::ExecutionContext) -> Result<(antecedent_io::sampled_counterfactual_transport_artifact::SampledCounterfactualTransportArtifactWire, antecedent_estimate::counterfactual_transport_sampled::SampledCounterfactualTransportEstimate), antecedent_io::sampled_counterfactual_transport_artifact::SampledCounterfactualTransportArtifactError>{
        antecedent_io::sampled_counterfactual_transport_artifact::sampled_counterfactual_transport_artifact(derivation, request, variable_names, ctx)
    }
}
/// Independent artifact of sampled observation recovery (`recovery.sampled_observation`).
pub mod sampled_recovery_artifact {
    pub use antecedent_io::sampled_recovery_artifact::{
        SampledRecoveryArtifactInput, SampledRecoveryArtifactWire, SampledRecoveryConsumeLimits,
        SampledRecoveryExpectation,
    };
}
/// Independent artifacts for the shared-data covariance of scenario estimates.
pub mod scenario_covariance_artifact {
    pub use antecedent_io::scenario_covariance_artifact::{
        CovarianceSpec, ScenarioCovarianceArtifactWire, ScenarioCovarianceConsumeLimits,
    };
}
/// Items of `antecedent-io::scientific_target_wire`.
pub mod scientific_target_wire {
    pub use antecedent_io::scientific_target_wire::ScientificTargetWire;
}
/// Items of `antecedent-io::selective_discovery_artifact`.
pub mod selective_discovery_artifact {
    pub use antecedent_io::selective_discovery_artifact::{
        SelectiveDiscoveryArtifact, SelectiveDiscoveryArtifactError,
    };
}
/// Independent point-result artifacts for the smoothed dose-response transport grid.
pub mod smoothed_dose_artifact {
    pub use antecedent_io::smoothed_dose_artifact::{
        SmoothedDoseArtifactError, SmoothedDoseConsumeLimits,
    };
}
/// Independent artifact for the unsupported Bayesian smoothed dose-response transport grid.
pub mod smoothed_dose_posterior_artifact {
    pub use antecedent_io::smoothed_dose_posterior_artifact::SmoothedDosePosteriorModelWire;
}
/// Portable staleness baseline: the summary of a plan's data that new data is compared with.
pub mod staleness_artifact {
    pub use antecedent_io_core::staleness_artifact::StalenessBaselineWire;
}
/// Independent artifact for the target-marginal initial-state estimand of the finite two-step.
pub mod temporal_initial_state_artifact {
    pub use antecedent_io::temporal_initial_state_artifact::{
        TemporalInitialStateConsumeLimits, TemporalPremisesWire,
    };
}
/// Internal replay artifact for the dependence-preserving temporal interval.
pub mod temporal_interval_artifact {
    pub use antecedent_io::temporal_interval_artifact::{
        IntervalConfigWire, IntervalResultWire, TemporalIntervalArtifactWire,
        TemporalIntervalConsumeLimits,
    };
}
/// Independent artifact for the semantic refresh of a prepared horizon-two result onto a new.
pub mod temporal_refresh_artifact {
    pub use antecedent_io::temporal_refresh_artifact::{ReceiptWire, WindowIdentityWire};
}
/// Independent artifacts for one finite two-step temporal transport sequence.
pub mod temporal_transport_artifact {
    pub use antecedent_io::temporal_transport_artifact::{
        TemporalSequenceArtifactWire, TemporalTransportArtifactError,
        TemporalTransportConsumeLimits, temporal_refusal,
    };
}
/// Items of `antecedent-io::transport_catalog_wire`.
pub mod transport_catalog_wire {
    pub use antecedent_io::transport_catalog_wire::EvidenceCatalogWire;
}
/// Items of `antecedent-io::transport_certificate`.
pub mod transport_certificate {
    pub use antecedent_io::transport_certificate::{CertificateOutcome, TransportCertificateWire};
}
/// Durable transport grid and statistical-provider records.
pub mod transport_grid_wire {
    pub use antecedent_io::transport_grid_wire::TransportGridFailureWire;
}
/// Independent artifacts for finite transport scenario sets.
pub mod transport_scenario_artifact {
    pub use antecedent_io::transport_scenario_artifact::{
        TransportScenarioArtifactError, TransportScenarioArtifactWire,
        TransportScenarioConsumeLimits, scenario_refusal,
    };
}
/// Wire types for durable artifacts (not internal Rust structs).
pub mod wire {
    pub use antecedent_io_core::wire::AdmgWire;
}
/// Independent point-result artifacts for single-source z transport.
pub mod z_transport_artifact {
    pub use antecedent_io::z_transport_artifact::{
        ZTransportArtifactError, ZTransportArtifactWire, ZTransportConsumeLimits,
    };
}
