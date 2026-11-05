//! Claim, provider and obligation records of the contract layer: what a result may claim, which
//! provider produced a quantity, which obligations remain, and the estimand bridges between claims.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Stable semantic coordinates for scientific values and distributions.
pub use antecedent_contract::DistributionMeaning;
/// Binding of externally supplied scientific results to a checked causal contract.
pub use antecedent_contract::ExternalResponse;
/// One provider-trust vocabulary for every surface that records who stands behind a number.
pub use antecedent_contract::ProviderTrust;
/// Transformation-effect reports for incremental compilation.
pub use antecedent_contract::TransformationReport;
/// `antecedent-contract` items at the crate root.
pub use antecedent_contract::response_coordinates;
/// Semantic contract for composing independently fitted causal mechanisms.
pub use antecedent_contract::{
    AdmissionBasis, AdmissionCoordinate, DependenceDeclaration, LatentCauseDeclaration,
    MechanismCompositionDeclaration, MechanismDeclaration, MechanismInterfaceBasis,
    MechanismOriginKind, TransportAdmission, check_declared_independence_against_ledger,
};
/// Four typed reasoning slots: Identification, Support, Uncertainty, Assumptions.
pub use antecedent_contract::{AssumptionSlot, ReasoningView, SlotAvailability, SupportSlot};
/// Portable claim envelope over a compiled contract and one result.
pub use antecedent_contract::{
    CheckedWorldPointError, CheckedWorldPointLimits, ClaimDomains, ClaimEnvelope, ClaimKind,
    ClaimOperation, DomainStatus,
};
/// Typed declarations for scientific objects supplied by another system.
pub use antecedent_contract::{
    ExternalCapability, ExternalPosteriorKind, ExternalScientificObject, ExternalTrustState,
    LawProviderContract, PosteriorProviderContract, ProviderObjectIdentity, VerificationProbe,
};
/// Binding of an external claim to the identified program and request it answers.
pub use antecedent_contract::{
    ExternalProgramClaim, ProgramBinding, check_external_against_program,
};
/// Structured refusals for external-object negotiation, verification and binding.
pub use antecedent_contract::{ExternalRefusal, capability_from_name, probe_from_name};
/// Semantic contract for observation-conditioned scientific queries.
pub use antecedent_contract::{
    InferenceBasis, MeasuredIndicator, ObservationOperation, ObservedSlot, SlotProvenance,
    SlotValue, calibration_refusal, interval_refusal,
};
/// The joint-law propagation rule.
pub use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, check_propagation, check_propagation_with_ledger,
    check_rows_pairable_across,
};
/// Typed obligation references that extend `antecedent_core::AssumptionRecord`.
pub use antecedent_contract::{ObligationKind, ObligationRecord, ObligationScope};
/// One verification receipt for every external provider.
pub use antecedent_contract::{ProviderVerificationReceipt, verify_external_object};
/// Scientific coordinates of a native response, read from the response itself.
pub use antecedent_contract::{
    ResponseCoordinateLabels, ResponseCoordinateSkeleton, label_coordinates,
    response_coordinate_skeleton,
};
/// External providers and consumers of transported evidence, through the one interface contract.
pub mod provider_transport {
    pub use antecedent_contract::provider_transport::{
        AdmittedProviderSource, ProviderTransportSource, SourceTrustRequirement,
        connect_bound_provider_source, connect_provider_source,
    };
}
/// Typed bridges between scientific quantities.
pub mod bridge {
    pub use antecedent_contract::bridge::BridgeInterfaceBasis;
    pub use antecedent_contract::bridge::{
        AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind,
        BridgeMapping, BridgeScope, BridgeSource, BridgedClaim, EstimandBridge, ExposureSetting,
        MappingUncertainty, NonlinearMap, PointPropagation, QuantityClass, SourceMappingDependence,
        TransportabilityPremise, TrialEffectPair, TrialLevelValidation, ValueKind,
        point_propagation_unavailable,
    };
}
/// Portable claim envelope over a compiled contract and one result.
pub mod claim {
    pub use antecedent_contract::claim::ClaimFreshness;
}
/// The runtime evidence ledger.
pub mod evidence_ledger {
    pub use antecedent_contract::evidence_ledger::{EffectiveObservations, parse_digest_hex};
}
/// Semantic contract for measurement models.
pub mod measurement {
    pub use antecedent_contract::measurement::{
        LatentPosteriorTrust, MeasurementInterfaceBasis, MeasurementOperation,
        measurement_interface_contract,
    };
}
/// Mixed-module policy for a release workflow.
pub mod mixed_module {
    pub use antecedent_contract::mixed_module::{
        CutFeedback, GaussianLaw, MixedModulePolicy, SharedParameterModel,
    };
}
/// Items of `antecedent-contract::propagation_chain`.
pub mod propagation_chain {
    pub use antecedent_contract::propagation_chain::FIX_CHAIN_DECLARATION;
}
/// Items of `antecedent-contract::rewrite_legality`.
pub mod rewrite_legality {
    pub use antecedent_contract::rewrite_legality::{
        FrameContents, NuisanceEffect, PreparationEffect, PreservationAxis, PreservedIdentities,
        PushdownEffect, Rewrite, RewriteRefusal, SummaryEffect, TileEffect,
    };
}
/// Items of `antecedent-contract::summary_observation`.
pub mod summary_observation {
    pub use antecedent_contract::summary_observation::{
        Dependence, FrequentistSummary, FrequentistSummaryKind, LikelihoodUpdateAdmission, Origin,
        REPORT_AS_FREQUENTIST_INTERVAL, SamplingApproximation, Selection, SummaryObservationError,
        SummaryObservationModel, SummaryUse, USE_AS_LIKELIHOOD_UPDATE,
    };
}
