//! Validation / refutation types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_validate::{
    PosteriorPredictiveCheck, PredictiveCheckKind, PredictiveCheckReport, PredictiveNoise,
    PriorPredictiveCheck, SerialDiscrepancy,
};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// A sensitivity analysis reported as an assumption range tied to its target.
pub use antecedent_validate::AssumptionRange;
/// Object-safe custom effect validators.
pub use antecedent_validate::CustomEffectValidator;
/// Environment holdout validation via J-PCMCI+.
pub use antecedent_validate::EnvironmentHoldout;
/// Orientation stability via PCMCI+ block bootstrap.
pub use antecedent_validate::OrientationStability;
/// Regime stability via RPCMCI block bootstrap.
pub use antecedent_validate::RegimeStability;
/// Synthetic-null discovery calibration.
pub use antecedent_validate::SyntheticNullCalibration;
/// Validation errors.
pub use antecedent_validate::ValidationError;
/// PCMCI link-frequency stability and parameter grids.
pub use antecedent_validate::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity,
    DiscoveryStabilityReport, LagWindowSensitivity,
};
/// Confounding sensitivity of the components of a mediation analysis and of the parameters of.
pub use antecedent_validate::{
    ComponentModel, MediationComponent, MediationComponents, MediationPath, MediationPosterior,
    MsmWeightPeriod, component_evalue_posterior_range, component_evalue_range,
    component_ovb_posterior_range, component_ovb_range, mediation_components_fit,
    mediation_path_ovb_range, mediation_paths_fit, mediation_posterior_range,
    mediation_vector_range, msm_parameter_fit, msm_parameter_fit_refitting_weights,
};
/// Conflict policy: shrink external prior α from prior-PPC / KL signals.
pub use antecedent_validate::{ConflictPolicy, ConflictSignals, apply_conflict_and_compose};
/// E-values on the risk-ratio.
pub use antecedent_validate::{EffectInterval, EffectScale, evalue_range};
/// False-positive checks via permute / phase-randomize surrogates.
pub use antecedent_validate::{FalsePositiveCheck, NullTransform};
/// Exact sensitivity ranges for finite discrete outcome mechanisms.
pub use antecedent_validate::{
    FixedGraphMechanismSensitivitySpec, FixedGraphSensitivityError, SourceOutcomeKernelRow,
    SourceParentLawRow, TargetParentLawRow, ZTransportSensitivityError,
    z_transport_mechanism_sensitivity,
};
/// Joint mechanism deviations for the registered surrogate z-transport formula.
pub use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityLimits, TippingBracket,
};
/// Omitted-variable-bias sensitivity for linear adjustment.
pub use antecedent_validate::{
    LinearAdjustmentSummary, benchmark_bound, linear_adjustment_fit, ovb_range,
};
/// Confounding sensitivity of a whole target vector, and of a posterior.
pub use antecedent_validate::{
    OvbPosterior, evalue_posterior_range, evalue_vector_range, ovb_posterior_range,
    ovb_vector_range, rosenbaum_vector_range,
};
/// Shared refuter types and data transforms.
pub use antecedent_validate::{RefutationProblem, RefutationReport, RefutationScope};
/// Rosenbaum sensitivity bounds for matched designs.
pub use antecedent_validate::{
    amplify, matched_sets_break_even, matched_sets_upper_p, rosenbaum_range,
};
/// Sensitivity of a linear mediation effect to mediator-outcome confounding (Imai.
pub use antecedent_validate::{mediation_range, mediation_sensitivity_fit};
/// Bounded, fit-free descriptions of suspicious data characteristics.
pub mod data_diagnostics {
    pub use antecedent_validate::data_diagnostics::DATA_DIAGNOSTIC_CHECKS;
}
/// Assessment of predictive laws.
pub mod predictive_assessment {
    pub use antecedent_validate::predictive_assessment::{PredictiveAssessmentError, score_case};
}
/// Rosenbaum sensitivity bounds for matched designs.
pub mod rosenbaum {
    pub use antecedent_validate::rosenbaum::{DEFAULT_GAMMA_GRID, MatchedSet, MatchedSetScore};
}
/// Target-native refutation: refuters that test the quantity a claim actually reports.
pub mod target_native {
    pub use antecedent_validate::target_native::{
        DerivativeCheckPoint, PosteriorPredictiveTarget, PriorSensitivityTarget,
        finite_difference_consistency, local_linearity, posterior_predictive_target_check,
        prior_sensitivity_target_check,
    };
}
