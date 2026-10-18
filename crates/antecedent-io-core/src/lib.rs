//! Versioned artifact container, wire types and digests for Antecedent, below
//! the estimators.
//!
//! This crate holds everything that needs no estimator: the container and its
//! reader, the CBOR and Arrow section codecs, the graph, query, response and
//! identification wires, the canonical digests, the shared sectioned-artifact
//! codec, and the artifacts that replay without an estimator. The estimator
//! bound artifacts live in `antecedent-io`, which re-exports this crate at its
//! own root so every path stays valid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod adaptive_monitoring_contract_artifact;
pub mod analysis_result_artifact;
pub mod analysis_wire;
pub mod artifact_codec;
mod artifact_kinds;
pub mod bayesian_prior_recalc_artifact;
pub mod bridge_artifact;
pub mod calibration;
pub mod callback_result_artifact;
pub mod canonical_json;
pub mod causal_artifact;
mod claim_identity;
pub mod cohort_decision;
pub mod compact_export;
pub mod composition_verification_artifact;
pub mod container;
mod contract_reason;
pub mod contract_section;
pub mod contract_wire;
pub mod contrast_wire;
pub mod convert;
pub mod counterfactual_id_artifact;
pub mod counterfactual_transport_wire;
pub mod coverage_records_data;
pub mod cross_world_artifact;
pub mod cross_world_execution_artifact;
pub mod discovery_wire;
pub mod distribution_artifact;
pub mod distribution_replay;
pub mod error;
pub mod evidence_ledger_artifact;
pub mod exact_law_wire;
pub mod expr_wire;
pub mod external_binding_wire;
pub mod external_callback_artifact;
pub mod external_claim_artifact;
pub mod external_column_claim_artifact;
pub mod external_estimate;
pub mod frozen_scores_artifact;
pub mod graph_dot;
pub mod graph_gml;
pub mod graph_json;
pub mod graph_mixed;
pub mod graph_networkx;
pub mod graphless_support;
pub mod identity;
pub mod interface_contract_artifact;
pub mod joint_law_propagation_artifact;
pub mod measurement_model_wire;
pub mod mechanism_composition_artifact;
pub mod mechanism_wire;
pub mod migrate;
/// `Mmap::map` lives here. [`crate::reader::MappedArtifactReader`] has the
/// mapped constructors (`open_path_mapped`, `from_mmap`) under the same
/// file-immutability contract.
///
/// SPDX-License-Identifier: MIT OR Apache-2.0
#[allow(
    unsafe_code,
    reason = "memory-mapped reader; the caller guarantees the file is not mutated while mapped"
)]
pub mod mmap_file;
pub mod observation_query_artifact;
pub mod plan_wire;
pub mod policy_program_artifact;
pub mod posterior;
pub mod prior_bank;
pub mod provenance_wire;
pub mod provider_envelope;
pub mod provider_verification_artifact;
pub mod quantity_wire;
pub mod query_wire;
pub mod reader;
pub mod recalc_receipt_artifact;
pub mod response_wire;
pub mod score_wire;
pub mod selective_discovery_artifact;
pub mod sensitivity_artifact;
pub mod temporal_recalc_artifact;
pub mod trace;
pub mod transport_catalog_wire;
pub mod transport_certificate;
pub mod transport_point_wire;
pub mod transport_proof;
pub mod transport_query_wire;
pub mod wire;
pub use analysis_result_artifact::{
    AnalysisResultHeader, AnalysisResultWire, DistributionAtomWire,
    IdentifiedSetIntervalMethodWire, IdentifiedSetIntervalWire, InterferenceInferenceWire,
    InterferencePointwiseIntervalWire, InterventionalDistributionWire, LocalPolynomialRatioWire,
    LongitudinalRegimeWire, MediationPosteriorSummaryWire, PanelDidWire, PolicyValueWire,
    RandomizedEffectWire, StructuralResponseAtomWire, StructuralResponseMixtureWire,
    StructuralWeightBasisWire, SurvivalDifferenceBandWire, SurvivalWire, SyntheticControlWire,
    SyntheticDidWire, TemporalIdentificationWire, TemporalMediationGridWire,
    TemporalMediationSliceWire, TemporalMediationUncertaintyWire, UnitEffectIntervalsWire,
    UnitEffectsWire, UpliftBinWire, decode_analysis_result_artifact,
    encode_analysis_result_artifact, encode_analysis_result_artifact_with_contract,
    identified_set_interval_from_wire, identified_set_interval_to_wire,
};
pub use analysis_wire::{
    DiagnosticWire, HedgeCertificateWire, IdentificationResultWire, IdentifiedEstimandWire,
    RdDesignWire, RefutationReportWire, diagnostic_from_wire, diagnostic_to_wire,
    identification_from_wire, identification_to_wire, identification_to_wire_with_registry,
};
pub use artifact_codec::{
    ArtifactCheckError, ArtifactContainer, CONTENT_BYTES, CONTENT_CBOR, CodecError, ConsumeLimits,
    OutSection, container_kind, decode_container, decode_with_version, digest_hex,
    encode_container, peek_version, require_digest, search_bounds, stored_law_bounds,
};
pub use causal_artifact::{
    CausalPayloadHeader, CausalPayloadKind, CausalPayloadWire, StaticResultWire,
    decode_causal_payload_artifact, encode_causal_payload_artifact,
};
pub use claim_identity::{ClaimIdentityWire, claim_digest};
pub use container::{
    AUTO_COMPRESS_MAX_RATIO, AUTO_COMPRESS_MIN_BYTES, ArtifactManifest, COMPRESSION_ZSTD,
    CONTAINER_VERSION, CompressPolicy, EncodedArtifact, MAGIC, SectionBytes, pack_section,
    pack_section_shared, section_descriptor, section_descriptor_with_policy,
};
pub use contract_section::TargetWeightsSectionWire;
pub use contract_section::{
    AnalysisResultConsumption, AnalysisResultContractWire, AssumptionSlotWire,
    AttestedEvidenceWire, CONTRACT_SECTION, CONTRACT_SECTION_FORMAT, CalibrationSlotWire,
    CalibrationStatus, CheckedAipwLoweringWire, CheckedAipwRowsWire, CheckedIvLoweringWire,
    ClaimDomainsWire, ClaimHostProjection, ClaimSectionWire, ContractIdentitiesWire,
    ContractReason, IdentificationSlotWire, ObligationSectionWire, ReasoningSectionWire,
    SlotSectionWire, SupportSlotWire, UncertaintyComponentWire, UncertaintySlotWire, accept_claim,
    checked_aipw_rows_digest, claim_domains, claim_kind_name, consume_analysis_result,
    contract_seal, decode_analysis_result_contract, project_claim_host, project_lossy_scalar,
    result_digest, support_empirical, validate_contract_section, validate_mixture_masses,
    verify_contract_against_body, weight_basis_name,
};
pub use contrast_wire::{ContrastBundleWire, RecordedContrastWire};
pub use convert::{
    admg_from_wire, admg_to_wire, cpdag_from_wire, cpdag_to_wire, dag_from_wire, dag_to_wire,
    from_cbor, pag_from_wire, pag_to_wire, schema_from_wire, schema_to_wire, schema_wire_from_v01,
    to_cbor,
};
pub use discovery_wire::{
    DiscoveryHeaderWire, EdgeEvidenceWire, LaggedLinkWire, TemporalGraphWire, TemporalNodeKeyWire,
    temporal_dag_from_wire, temporal_dag_to_wire,
};
pub use error::IoError;
pub use expr_wire::{
    ExprArenaWire, ExprNodeWire, FunctionalProgramWire, expr_arena_from_wire, expr_arena_to_wire,
    functional_program_from_wire, functional_program_to_wire,
};
pub use external_estimate::{
    ExternalEstimateAttach, attested_external_estimate, decode_external_estimate_claim,
    encode_external_estimate_claim, parse_digest_hex,
};
pub use graph_dot::{dag_from_dot, dag_to_dot, dag_wire_from_dot, dag_wire_to_dot};
pub use graph_gml::{dag_from_gml, dag_to_gml, dag_wire_from_gml, dag_wire_to_gml};
pub use graph_json::{DagJson, dag_from_json, dag_json_from_str, dag_to_json};
pub use graph_mixed::{
    AdmgJson, CpdagJson, PagJson, admg_from_dot, admg_from_gml, admg_from_json,
    admg_from_networkx_node_link, admg_to_dot, admg_to_gml, admg_to_json,
    admg_to_networkx_node_link, cpdag_from_dot, cpdag_from_gml, cpdag_from_json,
    cpdag_from_networkx_node_link, cpdag_to_dot, cpdag_to_gml, cpdag_to_json,
    cpdag_to_networkx_node_link, pag_from_dot, pag_from_gml, pag_from_json,
    pag_from_networkx_node_link, pag_to_dot, pag_to_gml, pag_to_json, pag_to_networkx_node_link,
};
pub use graph_networkx::{
    NetworkXAdjacency, NetworkXNodeLink, dag_from_networkx_adjacency, dag_from_networkx_node_link,
    dag_to_networkx_adjacency, dag_to_networkx_node_link,
};
pub use identity::{
    AdaptiveBudgetWire, BayesianBindingWire, CheckedFrontDoorLoweringWire,
    CheckedFunctionalResponseGridWire, CheckedFunctionalResponseMemberWire,
    CheckedLinearAdjustmentLoweringWire, CheckedNestedCounterfactualWire, ClassPriorIdentityWire,
    DataPartitionIdentityWire, DataSnapshotIdentityWire, DistributionFactorDomainWire,
    DistributionFactorKeyWire, DistributionFactorLawsWire, DistributionFactorRowWire,
    DistributionFactorTableWire, EstimatorConfigWire, EstimatorSpecWire, ExecutionIdentityWire,
    ExternalComposeIdentityWire, ExternalPriorSourceIdentityWire, GlmOptionsWire,
    GraphIdentityWire, IdentificationEnvelopeWire, IdentificationIdentityWire,
    IdentificationProductWire, InferenceBindingWire, InferentialCommitmentsWire,
    InterferenceSnapshotWire, KernelPolicyWire, LinearFitMomentsWire, MissingPolicyWire,
    NestedCounterfactualFitWire, ObservationIdentityWire, ObservationOptionsWire,
    OverlapPolicyWire, ParticipantBindingWire, PayloadDigestWire, PosteriorAtomGraphWire,
    PosteriorAtomIdentityWire, PriorMappingIdentityWire, PriorSetIdentityWire,
    PriorSpecIdentityWire, ProgramIdentityWire, ROW_WEIGHTS_PAYLOAD, RdConfigWire,
    ResponseOptionsWire, ScoreReuseIdentityWire, ScoreTableKey, SplitIdentityWire,
    TargetIdentityWire, TargetWeightsIdentityWire, TemporalClassIdentityWire,
    TransportIdentityWire, admg_identity, canonical_dag_wire, canonical_temporal_dag_wire,
    cpdag_identity, dag_identity, data_snapshot_digest, digest_canonical, digest_wire,
    executed_functional_labels, execution_digest, execution_identity_from_context,
    external_compose_identity, graph_posterior_atom_identities, identification_digest,
    identification_product_digest, identification_product_digest_wire, identification_product_wire,
    inference_binding_digest, observation_identity_wire, pag_identity, payload_digest,
    prior_set_identity, program_digest, score_reuse_digest, target_weights_digest,
    temporal_admg_identity, temporal_cpdag_identity, temporal_dag_identity, temporal_pag_identity,
    transport_identity,
};
pub use joint_law_propagation_artifact::{
    ConsumedJointLawPropagation, EnvelopeQueryWire, JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE,
    JOINT_LAW_PROPAGATION_ARTIFACT_VERSION, JOINT_LAW_PROPAGATION_MAX_INPUTS,
    JointLawPropagationArtifactError, JointLawPropagationArtifactWire, MarginalWire,
    PropagationInputWire, PropagationRequestWire, PropagationRuleWire, float_bits,
    joint_law_propagation_identity,
};
pub use mechanism_composition_artifact::{
    CompositionQueryWire, CompositionResultWire, ConsumedMechanismComposition,
    MECHANISM_COMPOSITION_ARTIFACT_FEATURE, MECHANISM_COMPOSITION_ARTIFACT_VERSION,
    MechanismCompositionArtifactError, MechanismCompositionArtifactWire,
    MechanismCompositionPremisesWire, MechanismNumbers, evaluate_composition_query, lower_plan,
    mechanism_composition_identity, refusals_for_findings,
};
pub use mechanism_wire::{
    MechanismSlotWire, MechanismStoreWire, ModelKindWire, mechanisms_from_wire, mechanisms_to_wire,
};
pub use migrate::{
    STABLE_FORMAT, SUPPORTED_SOURCE_FORMATS, is_supported_source, migrate_artifact,
    read_and_migrate,
};
pub use observation_query_artifact::{
    ConsumedObservationQuery, MeasuredSlotWire, OBSERVATION_QUERY_ARTIFACT_FEATURE,
    OBSERVATION_QUERY_ARTIFACT_VERSION, ObservationQueryArtifactError,
    ObservationQueryArtifactWire, ObservationQueryWire, ObservationResultWire,
    evaluate_observation_query, inference_basis_of, observation_query_identity,
    query_error_refusals,
};
pub use plan_wire::{
    ExecutionPerformanceWire, LogicalAnalysisPlanWire, PhysicalExecutionPlanWire,
    logical_plan_from_wire, logical_plan_to_wire, performance_from_wire, performance_to_wire,
    physical_plan_from_wire, physical_plan_to_wire,
};
pub use posterior::{
    CausalPosteriorWire, PosteriorQuantityWire, decode_causal_posterior_bytes,
    decode_posterior_artifact, decode_posterior_meta_from_path, decode_posterior_meta_from_seek,
    encode_posterior_artifact,
};
pub use prior_bank::{
    CompatibilityRejectReason, CompatibilityReport, DesignVariableRole, DesignVariableSummary,
    EstimandFingerprint, PRIOR_SOURCE_META_SECTION, PriorArtifactBody, PriorCatalog, PriorMapping,
    PriorSourceMeta, PriorSourceRef, TargetDesign, TemporalCoordinates, attach_prior_source_meta,
    decode_prior_source_meta, encode_prior_source_meta, extract_prior_source_meta,
    posterior_has_named_effect,
};
pub use provenance_wire::{
    ProvenanceGraphWire, ProvenanceNodeWire, provenance_from_wire, provenance_to_wire,
};
pub use query_wire::{
    AssignmentDesignWire, CausalQueryWire, ExposureLevelWire, ExposureMappingWire,
    InterferenceFunctionalWire, InterferenceQueryWire, InterventionWire,
    InterventionalDistributionQueryWire, PathSpecificEffectQueryWire, RandomizationDesignWire,
    RandomizedEffectQueryWire, RandomizedEstimandWire, SetInterventionWire, TargetPopulationWire,
    TemporalPolicyWire, TransportQueryWire, ValueWire, causal_query_from_wire,
    causal_query_to_wire, causal_query_to_wire_with_registry, interference_query_from_wire,
    interference_query_to_wire, interventional_distribution_from_wire,
    interventional_distribution_to_wire, path_specific_from_wire, path_specific_to_wire,
    transport_query_from_wire, transport_query_to_wire,
};
pub use reader::{
    ArtifactReader, MappedArtifactReader, MappedSection, SectionAccess, SectionIndexEntry,
    SectionLoadStats,
};
pub use response_wire::{
    CausalResponseWire, ContinuousDomainWire, DerivativeScaleWire, DerivativeWeightingWire,
    GridSpecWire, HorizonAdjustmentNodeWire, IdentificationStatusWire, IntervalInterpretationWire,
    ObservationAssumptionWire, ObservationSpecWire, ResponseEnvelopeWire, ResponseFunctionalWire,
    ResponseIdentificationWire, ResponseQueryWire, ResponseUncertaintyWire, ResponseValueWire,
    SupportDiagnosticWire, SupportRegionWire, SupportReportWire, SupportStatusWire,
    TemporalResponseSpecWire, causal_response_from_wire, causal_response_to_wire,
    response_query_from_wire, response_query_to_wire, response_value_from_wire,
    response_value_to_wire,
};
pub use trace::{
    AnalysisTraceWire, AssumptionRecordWire, AssumptionTagWire, DerivationStepWire,
    assumptions_to_wire,
};
pub use transport_interference_wire::{
    ExposureProbabilityMethodWire, InterferenceEstimateWire, NotCertifiedCertificateWire,
    PopulationFactorWire, RandomizationContrastWire, TransportEffectEstimateWire,
    TransportFormulaWire, TransportIdentificationWire, TransportOverlapDiagnosticWire,
    transport_identification_from_wire, transport_identification_to_wire,
};
pub use transport_proof::{TransportFactorView, TransportProofStepView, TransportProofView};
pub use wire::{
    AdmgWire, ArtifactKind, CpdagWire, DagWire, EndpointWire, FormatVersion, MarkedEdgeWire,
    MeasurementSpecWire, PagWire, ProvenanceWire, SchemaWire, SchemaWireV01, SectionDescriptor,
    SemanticVersion, ValueTypeWire, VariableSchemaWire,
};

pub mod temporal_counterfactual_artifact;
pub mod transport_errors;
pub mod transport_handoff_artifact;
pub mod transport_interference_wire;
pub mod transport_revalidation_artifact;
pub mod transported_counterfactual_artifact;
