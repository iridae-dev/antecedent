//! Effect refuters and validation diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod assumption_range;
pub mod bayesian_checks;
pub mod bootstrap_refute;
pub mod common;
pub mod conflict;
pub mod custom;
pub mod data_subset;
pub mod dummy_outcome;
pub mod error;
pub mod evalue;
pub mod evalue_scales;
pub mod functional;
pub mod graph_refute;
pub mod joint_mechanism_sensitivity;
pub mod mechanism_sensitivity;
pub mod mediation;
pub mod mediation_sensitivity;
pub mod msm_sensitivity;
pub mod omitted_variable_bias;
pub mod overlap;
pub mod overlap_rule;
mod panel_slice;
pub mod placebo;
pub mod predictive_assessment;
pub mod rcc;
pub mod riesz;
pub mod rosenbaum;
pub mod sensitivity;
pub mod stability;
pub mod staleness;
mod streams;
pub mod suite;
pub mod target_native;
mod tipping;
pub mod unobserved_common_cause;
pub mod validator;

pub use assumption_range::{AssumptionRange, RangePoint, SensitivityAssumption};
pub use bayesian_checks::{
    ExternalAlphaSensitivity, PosteriorPredictiveCheck, PredictiveCheckKind, PredictiveCheckReport,
    PredictiveNoise, PriorPredictiveCheck, PriorSensitivity, SbcReport, SerialDiscrepancy,
    SimulationBasedCalibration, with_prior_sensitivity,
};
pub use bootstrap_refute::BootstrapRefute;
pub use common::{RefutationProblem, RefutationReport, TemporalRefitContext, stack_panel_tabular};
pub use conflict::{
    ConflictPolicy, ConflictSignals, apply_conflict_and_compose, compose_with_conflict_policy,
    with_conflict_summary,
};
pub use custom::CustomEffectValidator;
pub use data_subset::DataSubsetRefuter;
pub use dummy_outcome::DummyOutcome;
pub use error::ValidationError;
pub use evalue::{DEFAULT_EVALUE_THRESHOLD, EValue};
pub use evalue_scales::{
    EValueBounds, EffectInterval, EffectScale, evalue_bounds, evalue_of_risk_ratio, evalue_range,
};
pub use graph_refute::GraphRefuter;
pub use joint_mechanism_sensitivity::{
    AxisTipping, FrontierPoint, JointDeviationSpec, JointFactor, JointFactorBound,
    JointMechanismSensitivityResult, JointSensitivityError, JointSensitivityInterval,
    JointSensitivityLimits, JointSensitivityRange, JointSensitivityReceipt,
    JointSensitivityUncertainty, TippingBracket, TippingStatus,
    joint_sensitivity_bootstrap_interval_internal, z_transport_joint_mechanism_sensitivity,
};
pub use mechanism_sensitivity::{
    DiscreteKernelOptimizationReceipt, DiscreteKernelSensitivity, DiscreteKernelSensitivityError,
    DiscreteKernelSensitivityResult, FixedGraphConditionalMechanismSensitivityResult,
    FixedGraphMechanismSensitivityResult, FixedGraphMechanismSensitivitySpec,
    FixedGraphRootMechanismSensitivityResult, FixedGraphSensitivityError, SourceOutcomeKernelRow,
    SourceParentLawRow, TargetParentLawRow, ZTransportMechanismSensitivityResult,
    ZTransportSensitivityError, fixed_graph_conditional_mechanism_sensitivity,
    fixed_graph_mechanism_sensitivity, fixed_graph_root_mechanism_sensitivity,
    fixed_graph_treatment_level_sensitivity, z_transport_mechanism_sensitivity,
};
pub use mediation::QueryRefutationPlan;
pub use mediation_sensitivity::{
    MediationSensitivitySummary, mediation_range, mediation_sensitivity_fit,
};
pub use msm_sensitivity::{
    MsmOutcomeLaw, MsmPoint, MsmSensitivityError, MsmSensitivityResult, MsmSensitivitySpec,
    MsmSensitivityUncertainty, MsmStratum, MsmTipping, MsmTippingDirection, msm_ate_bounds_at,
    msm_ate_sensitivity, msm_dirichlet_posterior,
};
pub use omitted_variable_bias::{
    BenchmarkBound, LinearAdjustmentFit, LinearAdjustmentSummary, benchmark_bound,
    linear_adjustment_fit, ovb_range,
};
pub use overlap::OverlapRefuter;
pub use overlap_rule::OverlapRuleRefuter;
pub use panel_slice::PanelSliceTemplate;
pub use placebo::{PlaceboMode, PlaceboTreatment};
pub use rcc::RandomCommonCause;
pub use riesz::RieszSensitivity;
pub use rosenbaum::{
    MatchedSet, MatchedSetBound, MatchedSetScore, SignedRankSensitivity, amplify,
    matched_sets_break_even, matched_sets_upper_p, rosenbaum_range, signed_rank_sensitivity,
};
pub use sensitivity::{
    LinearSensitivity, NeighborhoodSensitivity, NonparametricSensitivity, PartialLinearSensitivity,
};
pub use stability::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity,
    DiscoveryStabilityReport, EnvironmentHoldout, EnvironmentHoldoutReport, FalsePositiveCheck,
    FalsePositiveCheckReport, LagWindowSensitivity, LinkStability, NullCalibrationReport,
    NullTransform, OrientationStability, OrientationStabilityReport, RegimeStability,
    RegimeStabilityReport, SyntheticNullCalibration, UndirectedLinkStability,
};
pub use staleness::{
    FindingKind, Multiplicity, Recommendation, StalenessBaseline, StalenessError, StalenessFinding,
    StalenessPlan, StalenessPolicy, StalenessReport, assess_exported_baseline,
};
pub use suite::{
    BayesianSuiteContext, ConfiguredValidator, ValidationOutcome, ValidationSuite, ValidatorId,
};
pub use target_native::{
    RefutationTarget, TargetCertificate, TargetCertificationError, TargetFamily, TargetQuadrant,
    TargetRefit, TargetRefutationPlan, TargetRefutationReport, TargetRefuterKind, TargetSuite,
    certify_target_refutations, refuse_scalar_reports_for_target, run_target_plan,
};
pub use unobserved_common_cause::UnobservedCommonCause;
pub use validator::{PreparedRefutation, Validator};

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
