//! Claim, external-evidence and program-binding contracts built on `antecedent-core`.
//!
//! `antecedent-contract` owns the portable claim envelope and its composition
//! rules, the evidence envelope, the external provider contracts and their
//! verification and binding to a checked program, evidence obligations,
//! composition provenance, the reasoning view, the capability report and the
//! physical-plan records. No
//! estimator, identifier or statistics crate consumes these; they sit between
//! the shared vocabulary in `antecedent-core` and the facade, artifact and
//! validation layers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod adaptive_design;
pub mod bridge;
pub mod callback_protocol;
pub mod capability;
pub mod claim;
pub mod composition_provenance;
pub mod composition_verification;
pub mod evidence_envelope;
pub mod evidence_ledger;
pub mod evidence_obligation;
pub mod evidence_ports;
pub mod external;
pub mod external_binding;
pub mod external_refusal;
pub mod interface_contract;
pub mod joint_law_rule;
pub mod measurement;
pub mod mechanism_composition;
pub mod obligation;
pub mod observation_query;
pub mod plan;
pub mod program_binding;
pub mod provenance;
pub mod provider_transport;
pub mod provider_trust;
pub mod provider_verification;
pub mod quantity;
pub mod reasoning;
pub mod response_coordinates;
pub mod support_labels;
pub mod transform;
pub mod transport_result;
pub mod transport_revalidation;

pub use capability::{
    BlockedOperation, LicensedNeighbor, NextAction, OperationKind, OperationReadiness,
    OperationReport, PremiseChange, SemanticApplicability,
};
pub use claim::{
    AcceptanceReport, AttestedEvidence, CalibrationView, ClaimCompatibility, ClaimDomainAxis,
    ClaimDomains, ClaimEnvelope, ClaimFreshness, ClaimKind, ClaimOperation, ConsumerProfile,
    DerivedClaim, DerivedClaimOutcome, DomainStatus, EvidenceDependence, HandoffReceipt,
    HostOperation, SharedEvidenceRef, claim_compatibility, compose_claims,
};
pub use composition_provenance::{
    CompositionLink, CompositionStage, MAX_CHAIN_LINKS, ProvenanceChain, ProvenanceChainError,
};
pub use evidence_envelope::{
    AxisInterval, EvidenceBinding, EvidenceCalibration, EvidenceEnvelope, EvidenceEnvelopeError,
    EvidenceOrigins, EvidenceSupport, UncertaintyAxes, UncertaintyAxis, WorldTime,
};
pub use evidence_obligation::{
    EvidenceObligation, EvidenceObligationError, EvidenceObligationKind, EvidenceObligationSpec,
    EvidenceOffer, MAX_OBLIGATION_COORDINATES, ObligationProvenance, ObligationRegime,
    unresolved_assumption_obligations,
};
pub use evidence_ports::{DecisionRequest, EvidencePort, EvidencePortError};
pub use external::{
    EvidenceProviderContract, ExternalCapability, ExternalCapabilityRequest, ExternalContractError,
    ExternalEvidenceKind, ExternalPosteriorKind, ExternalScientificObject, ExternalTrustState,
    ExternalVerificationError, LawProviderContract, NegotiatedExternalOperation,
    PosteriorProviderContract, ProviderObjectIdentity, SignalProviderContract, UtilityMonotonicity,
    UtilityProviderContract, VerificationProbe, VerificationProbeKind,
};
pub use external_binding::{
    BoundExternalClaim, BoundTrustLevel, CheckedCausalContract, CheckedEquivalence, ColumnClaim,
    ColumnSupport, EquivalenceScope, ExternalBindingError, ExternalColumn, ExternalDistribution,
    ExternalPosterior, ExternalResponse, ExternalResult, ExternalResultHeader,
    ExternalUncertaintyMeaning, OBSERVATIONAL_REGIME, bind_external_result, external_values_digest,
};
pub use external_refusal::{
    ExternalRefusal, StructuredRefusal, capability_from_name, capability_name, probe_from_name,
    probe_name,
};
pub use joint_law_rule::{
    ASSUMPTION_RANGE_LABEL, ComposedCoordinate, ComposedIntervalPlan, CoverageRecord,
    InputAlignment, InputRole, JOINT_LAW_STAGE, MEASURED_INTERVAL_LABEL, POINT_ONLY_LABEL,
    PropagationClass, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationPlan, PropagationRequest, PropagationRoute, RequestedReading,
    STRUCTURAL_ENVELOPE_LABEL, UNCALIBRATED_SPREAD_LABEL, check_composed_interval,
    check_propagation, check_propagation_with_ledger, check_rows_pairable,
    check_rows_pairable_across,
};
pub use mechanism_composition::{
    AdmissionBasis, AdmissionCoordinate, DependenceDeclaration, LatentCauseDeclaration,
    MAX_DECLARED_MECHANISMS, MECHANISM_COMPOSITION_STAGE, MECHANISM_OUTPUT_PORT,
    MECHANISM_VALUE_FUNCTIONAL, MechanismCompositionDeclaration, MechanismCompositionPlan,
    MechanismDeclaration, MechanismInterfaceBasis, MechanismOriginKind, PlannedMechanism,
    PlannedParent, TransportAdmission, UncertaintyRoute,
    check_declared_independence_against_ledger, check_mechanism_composition, composition_refusal,
    reason_code_for_kind,
};
pub use obligation::{ObligationKind, ObligationRecord, ObligationScope};
pub use observation_query::{
    InferenceBasis, MeasuredIndicator, OBSERVATION_QUERY_STAGE, ObservationContext,
    ObservationOperation, ObservationSlotsRequest, ObservedSlot, SlotProvenance, SlotValue,
    calibration_refusal, check_observation_slots, interval_refusal, level_unreachable_refusal,
    measured_likelihood, zero_probability_evidence_refusal,
};
pub use plan::{
    BufferMaterialization, DataClassification, ExecutionPerformanceRecord, KernelSelection,
    LogicalAnalysisPlanRecord, ParallelTaskSpec, PhysicalExecutionPlanRecord,
};
pub use program_binding::{
    ExternalProgramClaim, ProgramBinding, ProgramBoundClaim, ProgramCheck,
    bind_external_result_to_program, check_external_against_program, dose_label,
};
pub use provenance::{ArtifactId, ProvenanceError, ProvenanceGraph, ProvenanceNode};
pub use provider_trust::ProviderTrust;
pub use provider_verification::{
    ExecutionContract, ProviderVerificationReceipt, VerificationSubject, VerifiedContractKind,
    VerifiedRequest, verify_external_object, verify_provider_requests,
};
pub use quantity::{
    DistributionMeaning, QuantityCondition, QuantityMismatch, QuantityRole, ScientificQuantity,
};
pub use reasoning::{
    AssumptionSlot, IdentificationSlot, ReasoningView, SlotAvailability, SupportSlot,
    UncertaintyComponent, UncertaintySlot, UncertaintySource,
};
pub use response_coordinates::{
    ResponseCoordinateLabels, ResponseCoordinateSkeleton, label_coordinates,
    response_coordinate_skeleton, response_coordinates,
};
pub use support_labels::check_static_point_labels;
pub use transform::{
    LayerEffect, SemanticLayer, TransformEffect, TransformIntent, TransformationReport,
    intent_effects,
};
pub use transport_result::TransportGridFailure;

/// Library crate version string from Cargo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
