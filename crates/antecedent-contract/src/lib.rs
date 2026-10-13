//! Claim, external-evidence and program-binding contracts built on `antecedent-core`.
//!
//! `antecedent-contract` owns the portable claim envelope and its composition
//! rules, the external provider contracts and their verification and binding
//! to a checked program, evidence obligations, composition provenance, the
//! reasoning view, the capability report and the physical-plan records. No
//! estimator, identifier or statistics crate consumes these; they sit between
//! the shared vocabulary in `antecedent-core` and the facade, artifact and
//! validation layers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod capability;
pub mod claim;
pub mod composition_provenance;
pub mod evidence_ledger;
pub mod evidence_obligation;
pub mod external;
pub mod external_binding;
pub mod external_refusal;
pub mod measurement;
pub mod mechanism_composition;
pub mod obligation;
pub mod plan;
pub mod program_binding;
pub mod provenance;
pub mod provider_trust;
pub mod provider_verification;
pub mod quantity;
pub mod reasoning;
pub mod response_coordinates;
pub mod support_labels;
pub mod transform;
pub mod transport_result;

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
pub use evidence_obligation::{
    EvidenceObligation, EvidenceObligationError, EvidenceObligationKind, EvidenceObligationSpec,
    EvidenceOffer, MAX_OBLIGATION_COORDINATES, ObligationProvenance, ObligationRegime,
    unresolved_assumption_obligations,
};
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
pub use mechanism_composition::{
    AdmissionBasis, AdmissionCoordinate, DependenceDeclaration, LatentCauseDeclaration,
    MAX_DECLARED_MECHANISMS, MECHANISM_COMPOSITION_STAGE, MECHANISM_VALUE_FUNCTIONAL,
    MechanismCompositionDeclaration, MechanismCompositionPlan, MechanismDeclaration,
    MechanismOriginKind, PlannedMechanism, PlannedParent, TransportAdmission, UncertaintyRoute,
    check_mechanism_composition, composition_refusal, reason_code_for_kind,
};
pub use obligation::{ObligationKind, ObligationRecord, ObligationScope};
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
