//! Identification stage types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_identify::{
    BinaryIvLaw, GeneralizedAdjustmentConfig, GeneralizedAdjustmentIdentifier,
    GraphIdentificationCase, IdentificationEnvelope, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, NotCertifiedCertificate, PopulationFactor, ProbabilityMass,
    TemporalMediationIdentifier, TransportCertificate, TransportFormula, TransportIdentification,
    TransportIdentifier, binary_iv_ate_bounds,
};

pub use antecedent_identify::sid::{
    SHedgeCertificate, SHedgeRecord, SelectionForest, SelectionForestRecord, SidDerivationRecord,
    SidStepRecord, verify_s_hedge,
};
pub use antecedent_identify::{
    BoundTransportFunctional, ClassicalTransportDerivation, ClassicalTransportQuery,
    ClassicalTransportResult, SidLimits, identify_classical_transport, verify_classical_transport,
};

pub use antecedent_identify::{CatalogTransportResult, identify_catalog_transport};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Conservative structural transportability identification.
pub use antecedent_identify::MissingEvidenceCertificate;
/// `TemporalCpdag` / `TemporalPag` identification envelopes.
pub use antecedent_identify::TemporalCompletionGraph;
/// Recursive z-transport search and catalog binding.
pub use antecedent_identify::bind_z_transport_catalog;
/// Decide items of the parent module.
pub use antecedent_identify::decide_mixed_source;
/// Derive items of the parent module.
pub use antecedent_identify::identify_z_transport;
/// Self-censoring witness and stored-recovery verification.
pub use antecedent_identify::verify_recovery_witness;
/// Contract validation for the bounded single-source z-transport setting.
pub use antecedent_identify::{
    BoundZTransportFunctional, ComponentFactorization, TwoSourceZTransportDecision,
    TwoSourceZTransportQuery, ZTransportDecision, ZTransportDerivation, ZTransportLimitsReceipt,
    ZTransportMissingEvidence, ZTransportObstruction, ZTransportOutcome, ZTransportQuery,
    ZTransportResult, ZTransportSourceSpec,
};
/// General finite counterfactual transport on a static semi-Markovian graph.
pub use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, CounterfactualTransportMissing,
    CounterfactualTransportProblem, CounterfactualTransportQuery, CounterfactualTransportSource,
    counterfactual_transport_refusal, identify_counterfactual_transport,
};
/// ADMG conditional transport on the classical complete-source family.
pub use antecedent_identify::{
    ConditionalObstructionCandidate, ConditionalTransportDecision, ConditionalTransportQuery,
    admg_conditional_refusal, decide_admg_conditional_transport,
};
/// Self-certifying two-model witnesses for ADMG conditional transport.
pub use antecedent_identify::{ConditionalWitnessRecord, WitnessKernelRecord, WitnessModelRecord};
/// Identification result types.
pub use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};
/// Identification errors.
pub use antecedent_identify::{IdentificationBudget, IdentificationError};
/// Classical meta-transportability (Bareinboim & Pearl 2013, Figure 5).
pub use antecedent_identify::{MetaTransportQuery, identify_meta_transport};
/// Bounded mixed-source proof search.
pub use antecedent_identify::{
    MixedQuantity, MixedSourceDerivation, MixedSourceQuery, MixedStageRecord, render_quantity,
};
/// Mixed-source rule engine and the decision that runs it.
pub use antecedent_identify::{MixedSourceDecision, bind_mixed_source_catalog};
/// The bounded multi-source limited-experiment (mz-transportability) contract.
pub use antecedent_identify::{
    MzSearchRecord, MzStageRecord, MzTransportDecision, MzTransportQuery, MzTransportRoute,
    bind_mz_transport_catalog, decide_mz_transport, mz_transport_refusal,
};
/// Exact binary causal observation recovery.
pub use antecedent_identify::{
    ObservationRecoveryQuery, PartiallyObserved, RECOVERY_DEFAULT_LIMITS, RecoveredEffectQuery,
    RecoveryDecision, RecoveryDerivation, RecoveryDetail, RecoveryError, RecoveryLimits,
    RecoveryWitness, WitnessMechanism, decide_observation_recovery,
};
/// Study planning over the mz-transport and mixed-source catalogs.
pub use antecedent_identify::{
    STUDY_PLAN_MEMORY_BYTES, StudyPlan, StudyPlanLimits, StudyPlanRoute, StudyPlanStop,
    StudySubsetOutcome,
};
/// Lower `TransportFormula` views into the shared expression arena.
pub use antecedent_identify::{bind_transport_derivation, lower_transport_formula};
/// Decide items of the parent module.
pub use antecedent_identify::{decide_two_source_z_transport, decide_z_transport_inspecting};
/// Tier-closure identification as a fast path over generalized adjustment.
pub use antecedent_identify::{identify_tiered, identify_tiered_joint};
/// Transport under graph uncertainty.
pub mod discovered_transport {
    pub use antecedent_identify::discovered_transport::DiscoveredTransportOperationalKind;
}
/// General ID of a whole-program joint intervention on an unrolled panel graph.
pub mod panel_unrolled {
    pub use antecedent_identify::panel_unrolled::{
        PanelNode, UnrolledPanelGraph, UnrolledPanelSpec,
    };
}
/// Exact binary causal observation recovery.
pub mod recovery {
    pub use antecedent_identify::recovery::MISSING_LEVEL;
}
/// Classical single-source sID, scoped to access to all source experiments.
pub mod sid {
    /// Bounded DAG completions of a supplied CPDAG, decided as a finite scenario set.
    pub mod cpdag_completion {
        pub use antecedent_identify::sid::cpdag_completion::{
            CPDAG_EVIDENCE_MISMATCH_DETAIL, CompletionEvidenceBinding, CpdagCompletionInput,
            CpdagEvidence, enumerate_cpdag_completions,
        };
    }
    /// Per-scenario selection differences and invariances.
    pub mod scenario_invariance {
        pub use antecedent_identify::sid::scenario_invariance::{
            InvarianceBody, InvarianceReport, ObstructionWitness,
        };
    }
    /// Finite, explicitly supplied graph/selection scenarios for one transport question.
    pub mod scenarios {
        pub use antecedent_identify::sid::scenarios::{
            ScenarioCoordinate, TransportScenario, TransportScenarioSet,
        };
    }
    /// One finite temporal transport sequence: exact and point-only.
    pub mod temporal_sequence {
        pub use antecedent_identify::sid::temporal_sequence::{
            TemplateRoles, TemporalOutcome, TemporalSequenceSpec, TemporalSlots, unroll_two_slice,
        };
    }
}
