//! Causal identification algorithms.
//!
//! Identify first; estimate second. Primary entry points include
//! [`BackdoorIdentifier`], [`FrontDoorIdentifier`], [`IdIdentifier`],
//! [`IdcIdentifier`], and [`AutoIdentifier`].
//!
//! ```
//! use antecedent_identify::BackdoorIdentifier;
//!
//! let id = BackdoorIdentifier::new();
//! let _ = id;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod admg_adjustment;
pub mod assumptions;
pub mod auto;
pub mod backdoor;
pub mod bounds;
mod class_envelope_cache;
mod class_transport;
pub mod compiled_product;
pub mod counterfactual_id;
pub mod cross_world;
pub mod cyclic_calculus;
pub mod cyclic_id;
pub mod efficient;
pub(crate) mod enum_masks;
pub mod envelope;
pub mod error;
pub(crate) mod exact_search;
/// Actual finite identification checker counters.
pub mod execution_counts;
pub mod experiment_moments;
pub mod experiment_regime;
pub mod experimental_recovery;
pub mod frontdoor;
pub mod generalized;
pub mod hedge;
pub mod id;
pub mod idc;
pub(crate) mod identification_run;
pub mod identifier;
pub(crate) mod intervention_support;
pub mod iv;
pub(crate) mod node_map;
mod pag_path_specific;
pub mod path_specific;
mod pow2;
pub mod prepared;
pub mod rd;
pub mod recovery;
pub mod recovery_chain;
pub mod response;
pub(crate) mod response_id;
pub mod result;
pub(crate) mod selection_separation;
pub mod sid;
pub mod temporal_admg;
pub mod temporal_backdoor;
pub mod temporal_generalized;
pub mod temporal_iv;
mod temporal_mag;
pub mod temporal_mediation;
mod temporal_pag_path_specific;
mod temporal_window;
pub mod tiered;
pub mod transport;
pub use recovery::{
    ObservationRecoveryQuery, PartiallyObserved, RECOVERY_DEFAULT_LIMITS,
    RECOVERY_MAX_OBSERVED_CELLS, RECOVERY_RULE_VERSION, RecoveredEffect, RecoveredEffectQuery,
    RecoveredEffectRecord, RecoveryDecision, RecoveryDerivation, RecoveryDerivationRecord,
    RecoveryDetail, RecoveryError, RecoveryFactorRecord, RecoveryLimits, RecoveryMarginRecord,
    RecoveryReceiptRecord, RecoveryWitness, StoredRecovery, WitnessCheck, WitnessMechanism,
    decide_observation_recovery, verify_observation_recovery, verify_recovery_witness,
};
pub use recovery_chain::{
    CHAIN_RECOVERY_LIMITS, CHAIN_RECOVERY_RULE_VERSION, ChainPartial, ChainRecoveryDecision,
    ChainRecoveryDetail, ChainRecoveryError, ChainRecoveryPlan, ChainRecoveryQuery,
    ChainRecoveryWitness, ChainWitnessCheck, ChainWitnessMechanism, decide_chain_recovery,
    verify_chain_witness,
};
#[allow(deprecated, reason = "the retained released name stays reachable")]
pub use sid::MZ_TRANSPORT_MAX_OBSERVED;
pub use sid::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, ADMG_CONDITIONAL_MAX_CONDITIONED,
    ADMG_CONDITIONAL_MAX_OBSERVED, ADMG_CONDITIONAL_MAX_TREATMENTS,
    BoundConditionalTransportFunctional, ConditionalObstructionCandidate,
    ConditionalObstructionRecord, ConditionalStageRecord, ConditionalTransportDecision,
    ConditionalTransportDerivation, ConditionalTransportInspection, ConditionalTransportQuery,
    ConditionalTransportRecord, admg_conditional_refusal, decide_admg_conditional_transport,
};
pub use sid::{
    BoundMixedSourceFunctional, MIXED_SOURCE_DEFAULT_LIMITS, MIXED_SOURCE_MAX_DISTRIBUTIONS,
    MIXED_SOURCE_MAX_DO, MIXED_SOURCE_MAX_MOVE, MIXED_SOURCE_MAX_OBSERVED, MIXED_SOURCE_RULE_SET,
    MixedExclusion, MixedInput, MixedMissingEvidence, MixedMissingLeaf, MixedQuantity, MixedRule,
    MixedSearchInspection, MixedSearchSummary, MixedSourceDecision, MixedSourceDerivation,
    MixedSourceDerivationRecord, MixedSourceLeaf, MixedSourceQuery, MixedStageRecord, MixedStep,
    MixedStepRecord, ValidatedMixedSourceQuery, bind_mixed_source_catalog, decide_mixed_source,
    mixed_source_rule_names, render_quantity, validate_mixed_source_query,
    verify_mixed_source_derivation,
};
pub use sid::{
    BoundMzTransportFunctional, BoundTransportFunctional, BoundZTransportFunctional,
    CatalogTransportResult, ClassicalTransportDerivation, ClassicalTransportQuery,
    ClassicalTransportResult, ComponentFactorization, MZ_TRANSPORT_DEFAULT_LIMITS,
    MZ_TRANSPORT_MAX_CANDIDATE_REGIMES, MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE,
    MZ_TRANSPORT_MAX_SOURCES, MZ_TRANSPORT_MEMORY_BYTES, MetaSource, MetaTransportQuery,
    MzSearchInspection, MzSearchRecord, MzStageRecord, MzTransportDecision, MzTransportDerivation,
    MzTransportDerivationRecord, MzTransportObstruction, MzTransportQuery, MzTransportRoute,
    SidLimits, TwoSourceZTransportComponent, TwoSourceZTransportDecision, TwoSourceZTransportQuery,
    ValidatedMzTransportQuery, ZExperimentFamilyError, ZFactorObligation, ZProofOperation,
    ZTransportBudgetKind, ZTransportDecision, ZTransportDerivation, ZTransportDerivationRecord,
    ZTransportLimitsReceipt, ZTransportMissingEvidence, ZTransportNotCertifiedInspection,
    ZTransportNotCertifiedKind, ZTransportObstruction, ZTransportObstructionRecord,
    ZTransportOutcome, ZTransportProofInspection, ZTransportQuery, ZTransportResult,
    ZTransportSourceSpec, ZTransportTerminalRecord, bind_mz_transport_catalog,
    bind_z_transport_catalog, decide_mz_transport, decide_two_source_z_transport,
    decide_z_transport_inspecting, decide_z_transport_with_catalog, identify_catalog_transport,
    identify_catalog_transport_selection_standardization_first,
    identify_catalog_transport_standardization_first, identify_classical_transport,
    identify_meta_catalog, identify_meta_transport, identify_z_transport,
    identify_z_transport_reporting, intervention_ancestral_treatments,
    intervention_levels_conflict, intervention_mutilated_admg, mz_transport_refusal,
    validate_mz_transport_query, validate_z_experiment_family, validate_z_transport_query,
    verify_classical_transport, verify_meta_s_hedge, verify_meta_transport,
    verify_mz_transport_obstruction, verify_z_transport_derivation,
};
pub use sid::{
    CONDITIONAL_WITNESS_MAX_LATENT_LEVELS, ConditionalNonTransportabilityProof,
    ConditionalNonTransportabilityRecord, ConditionalWitnessCheck, ConditionalWitnessRecord,
    WITNESS_SEARCH_DEFAULT_OPERATIONS, WITNESS_VERIFY_DEFAULT_OPERATIONS,
    WITNESS_WORK_PER_OPERATION, WitnessKernelRecord, WitnessLatentRecord, WitnessMeter,
    WitnessModelRecord, WitnessSearch, search_conditional_witness,
    search_conditional_witness_within, verify_conditional_witness,
    verify_conditional_witness_within,
};
pub use sid::{
    COUNTERFACTUAL_TRANSPORT_CONTRACT, COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
    COUNTERFACTUAL_TRANSPORT_MASK_VARIABLES, COUNTERFACTUAL_TRANSPORT_MAX_ATOMS,
    COUNTERFACTUAL_TRANSPORT_MAX_CONTROLLABLE, COUNTERFACTUAL_TRANSPORT_MAX_LIMITS,
    COUNTERFACTUAL_TRANSPORT_MAX_NODES, COUNTERFACTUAL_TRANSPORT_MAX_REGIMES,
    COUNTERFACTUAL_TRANSPORT_MAX_SOURCES, COUNTERFACTUAL_TRANSPORT_MAX_SUM_CELLS,
    COUNTERFACTUAL_TRANSPORT_MAX_WORLDS, COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
    CounterfactualCitedRegime, CounterfactualFactor, CounterfactualFunctional, CounterfactualItem,
    CounterfactualItemSource, CounterfactualLevel, CounterfactualLevelIndex,
    CounterfactualMissingRegime, CounterfactualMovedAtom, CounterfactualSum, CounterfactualSymbol,
    CounterfactualSymbolId, CounterfactualTransportDecision, CounterfactualTransportDerivation,
    CounterfactualTransportInspection, CounterfactualTransportMissing,
    CounterfactualTransportObstruction, CounterfactualTransportProblem,
    CounterfactualTransportQuery, CounterfactualTransportSearchRecord,
    CounterfactualTransportSource, CounterfactualTransportUnsupported,
    CounterfactualTransportWitness, CounterfactualTransportWitnessCheck,
    CounterfactualVariableIndex, counterfactual_transport_refusal,
    identify_counterfactual_transport, replay_counterfactual_transport,
    verify_counterfactual_transport_witness,
};
pub use sid::{
    STUDY_PLAN_DEFAULT_LIMITS, STUDY_PLAN_MAX_CANDIDATES, STUDY_PLAN_MAX_PROPOSALS,
    STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE, STUDY_PLAN_MAX_SUBSET, STUDY_PLAN_MEMORY_BYTES,
    STUDY_PLAN_RANKING, StudyBaseFailure, StudyFactor, StudyPlan, StudyPlanCandidate,
    StudyPlanLimits, StudyPlanRefusal, StudyPlanRoute, StudyPlanStop, StudyProposal,
    StudyProposalDerivation, StudyRepair, StudySubsetOutcome, StudySubsetRecord,
    plan_study_additions,
};
mod transport_lower;

#[cfg(test)]
mod test_support {
    use antecedent_core::VariableId;
    use antecedent_graph::DenseNodeId;

    /// `VariableId::from_raw`, shared by the crate's tests.
    pub(crate) fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    /// `DenseNodeId::from_raw`, shared by the crate's tests.
    pub(crate) fn d(raw: u32) -> DenseNodeId {
        DenseNodeId::from_raw(raw)
    }
}
pub mod discovered_transport;
pub mod discovered_transport_temporal;
#[cfg(test)]
mod experiment_moments_noisy;
#[cfg(test)]
mod id_scm_property;
#[cfg(test)]
mod mag_id_bruteforce;
/// Hidden parser for the frozen external `graph_dot` oracles used by tests.
///
/// Compiled only for this crate's tests and under the `test-util` feature: it panics on
/// malformed input, which is fine for frozen fixtures and wrong for a library API.
#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
pub mod oracle_dot;

pub use admg_adjustment::{
    FrontDoorSearch, TemporalAdmgAdjustment, default_front_door_limits,
    identify_admg_front_door_derivative, identify_admg_response_adjustment,
    identify_cpdag_derivative_pairs, identify_pag_derivative_pairs,
    identify_temporal_admg_front_door_derivative, identify_temporal_admg_response_adjustment,
    search_admg_front_door_derivative, upgrade_id_only_derivative_cases,
};
pub use auto::{AutoIdentifier, PreparedAutoGraph};
pub use backdoor::{AdjustmentSearchConfig, BackdoorIdentifier, PreparedIdentificationGraph};
pub use bounds::{BinaryIvLaw, binary_iv_ate_bounds};
pub use class_transport::{
    CLASS_TRANSPORT_CANDIDATES_BOUNDED_REASON, CLASS_TRANSPORT_METHOD, ClassTransportEnvelope,
    PAG_TRANSPORT_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
};
pub use compiled_product::{
    CompiledIdentificationProduct, CompiledProductParts, IdentificationClaimClass,
    IdentificationOutcome, ProductRefusal, RequiredFactor, UnresolvedObligation,
};
pub use efficient::EfficientBackdoorIdentifier;
pub use envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
    carries_identified_mass, search_truncated,
};
pub use error::{IdentificationBudget, IdentificationError};
pub use experiment_regime::{
    AcceptedRegimeContract, DependencePremise, ExperimentRegime, PopulationId,
    REPAIR_MASK_OR_PREMISE, REPAIR_MASK_OR_PREMISE_ATTEMPT, RegimeContract, RegimeContractDecision,
    RegimeIntervention, RegimeRefusal, RegimeRefusalReason, SourceIdentity, UnitMask,
    validate_regime_contract,
};
pub use experimental_recovery::{
    CLAMP_EXPERIMENT_RECIPE, CLAMP_EXPERIMENT_RECIPE_ATTEMPT, ClampExogeneity, ClampExperimentSet,
    CoefficientRecovery, DeclaredClampExperiment, DeclaredEquationRank, DeclaredNoiseLaw,
    DesignFault, DirectCoefficient, ExperimentDesign, ExperimentalRecoveryDecision,
    LATENT_VARIABLE_EXPANSION_ATTEMPT, NotThisRecipe, PRIOR_REGULARIZATION_ATTEMPT,
    PhysicalDynamicalStability, RecipeRefusal, RecoveryAttempt, RefusalReason, RegimeAssumption,
    UnresolvedDirection, UnresolvedReason, WeakStability, decide_experimental_recovery,
};
pub use frontdoor::{FrontDoorIdentifier, FrontDoorSearchConfig};
pub use generalized::{
    CAPPED_COMPLETION_DIAGNOSTIC_CODE, GeneralizedAdjustmentConfig,
    GeneralizedAdjustmentIdentifier, MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE, invisible_directed_edges,
    mag_to_confounded_admg,
};
pub use hedge::{HedgeCertificate, HedgeProblem};
pub use id::{IdIdentifier, ScheduleContrast};
pub use idc::IdcIdentifier;
pub use identifier::{IdentificationWorkspace, Identifier};
pub use iv::{InstrumentSearchConfig, InstrumentalVariableIdentifier};
pub use joint_response::JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE;
pub use pag_path_specific::{AdmgCandidates, PAG_ADMG_SEARCH_BOUNDED_DIAGNOSTIC_CODE};
pub use path_specific::PathSpecificIdentifier;
pub use prepared::{PreparedAdmg, dag_to_admg};
pub use rd::{SharpRdConfig, SharpRdIdentifier};
pub use response::ResponseIdentifier;
pub use response_id::{identify_cpdag_response_general, identify_pag_response_general};
pub use result::{
    DerivationStep, DerivationTrace, EstimandClaim, EstimandSelectionError,
    IdentificationPerformanceRecord, IdentificationResult, IdentificationResultParts,
    IdentificationStatus, IdentifiedEstimand,
};
pub use temporal_backdoor::{
    PARENT_ADJUSTMENT_RULE, TemporalBackdoorIdentifier, TemporalIdentificationResult,
};
pub use temporal_generalized::{TemporalClassEnvelope, TemporalCompletionGraph};
pub use temporal_iv::{TemporalIvIdentifier, TemporalIvResult};
pub use temporal_mediation::TemporalMediationIdentifier;
pub use temporal_pag_path_specific::{
    TEMPORAL_PAG_WINDOW_NOT_MAG_DIAGNOSTIC_CODE, TemporalPagMediationWindow,
    temporal_pag_mediation_window,
};
pub use tiered::{
    NO_LATENT_TO_OUTCOME, TIERED_JOINT_ADJUSTMENT_REFUSE, identify_tiered, identify_tiered_joint,
    identify_tiered_joint_on, identify_tiered_on,
};
pub use transport::{
    MissingEvidenceCertificate, NotCertifiedCertificate, PopulationFactor, TransportCertificate,
    TransportFormula, TransportIdentification, TransportIdentifier,
};
pub use transport_lower::{
    bind_transport_derivation, lower_transport_formula, lower_transport_mean,
};

mod joint_response;
