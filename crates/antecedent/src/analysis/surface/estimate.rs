//! Estimation stage types and strategy identifiers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_data::{NetworkData, NetworkEdge};

pub use antecedent_estimate::{
    CausalPosterior, ConditionalLinearAdjustment, ContinuousResponseOptions, EffectEstimate,
    EstimationError, InterferenceEstimate, OverlapPolicy, ResponseGridInputs,
    TemporalEffectSurface, TemporalLinearPredictor, TemporalMediationEstimator,
    TemporalMediationGrid, TemporalMediationIdentifiedSet, TemporalMediationSlice,
    TemporalMediationUncertainty, TransportEffectEstimate, TransportOverlapDiagnostic,
    TransportOverlapReport, TransportResponseGridEstimate, own_treatment_level,
    transport_augmented_response_grid,
};

pub use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, EstimatorId, IdentifierId, StaticEstimateWorkspaces,
    estimand_compatible_with_estimator, identification_status_acceptable, identify_static_query,
    identify_static_query_with_rd, require_identified, select_estimand, validate_distribution_pair,
    validate_path_specific_pair, validate_static_pair,
};

pub use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits,
    ExactSupportRecord, ExactTransportData, LawTolerance,
};

pub use antecedent_estimate::white_innovation_posterior::{
    InnovationWhitenessGate, InnovationWhitenessReport,
};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Augmented inverse-probability weighting (AIPW / doubly robust) ATE estimator.
pub use antecedent_estimate::AipwAte;
/// Shared analytic SE policy for ATE estimators.
pub use antecedent_estimate::AnalyticSeKind;
/// Honest causal forest CATE (`CausalForest`).
pub use antecedent_estimate::CausalForest;
/// Cell-saturated AIPW for discrete joint `Set` interventions.
pub use antecedent_estimate::CellSaturatedAipw;
/// The one checked nuisance-reuse store.
pub use antecedent_estimate::CrossfitNuisanceCache;
/// Covariate-distance matching estimator.
pub use antecedent_estimate::DistanceMatching;
/// Doubly robust CATE learner (`DrLearner`).
pub use antecedent_estimate::DrLearner;
/// Linear front-door two-stage (product-of-coefficients) regression estimator.
pub use antecedent_estimate::FrontDoorTwoStage;
/// Generalized linear model (logistic) adjustment ATE estimator for binary outcomes.
pub use antecedent_estimate::GlmAdjustmentAte;
/// Learned continuous-outcome trial-to-target transport.
pub use antecedent_estimate::LearnedContinuousOptions;
/// Declared treatment levels of a propensity problem.
pub use antecedent_estimate::OtherLevels;
/// Propensity stratification estimator.
pub use antecedent_estimate::PropensityStratification;
/// Inverse-probability weighting estimator.
pub use antecedent_estimate::PropensityWeighting;
/// Exact evaluation of a checked observation-recovery formula and of the downstream effect.
pub use antecedent_estimate::RecoveredLaw;
/// Estimation errors.
pub use antecedent_estimate::RefusalFields;
/// Descriptive association of an outcome with predictors.
pub use antecedent_estimate::SPREAD_LEVEL;
/// Declared sampling weights.
pub use antecedent_estimate::SamplingWeights;
/// Cross-fitted AIPW score tables for retargeting and joint inference.
pub use antecedent_estimate::ScoreTable;
/// Joint outer bootstrap for catalog-bound empirical transport tables.
pub use antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED;
/// Propensity problem preparation and shared workspace.
pub use antecedent_estimate::default_propensity_overlap;
/// Design-based estimation under network interference.
pub use antecedent_estimate::estimate_interference_with_context;
/// Exact exposure probabilities under two-stage cluster saturation.
pub use antecedent_estimate::estimate_saturation_interference;
/// Whole `antecedent-estimate` modules re-exported as namespaces (their items are
/// reached as `estimate::<module>::<Item>`).
pub use antecedent_estimate::joint_panel;
/// Supported temporal intervention plans.
pub use antecedent_estimate::plan_from_response_query;
/// Trial-to-target transport estimators under the caller's execution policy.
pub use antecedent_estimate::trial_to_target_effect_with_context;
/// Descriptive raw-versus-adjusted comparison and declared reporting-scale transforms.
pub use antecedent_estimate::{
    AdjustedEstimate, ArmSummary, Availability, MeanPair, RawContrast, ReportingScale,
    ReportingTransform, Unavailable, compare_raw_adjusted, raw_contrast, transform_mean_pair,
};
/// Finite categorical frequency plug-in for transport joints.
pub use antecedent_estimate::{
    BayesianTransportLawProvider, EmpiricalTableEstimator, EmpiricalTableOptions, RegimeSample,
    StatisticalTransportInput,
};
/// Propensity-score nearest-neighbor matching.
pub use antecedent_estimate::{CaliperScale, PropensityMatching};
/// Factorized-propensity AIPW for bounded discrete joint cells.
pub use antecedent_estimate::{
    CellStatus, FactorizedJointConfig, FactorizedJointFit, JointCellReport, JointContrast,
    declared_joint_nuisance, orderings_for,
};
/// Cluster-aware cross-fitted AIPW: the independence unit is a declared cluster.
pub use antecedent_estimate::{
    ClusterDml, DEFAULT_MIN_CLUSTERS, DEFAULT_MIN_COMPONENTS_PER_FOLD, DeclaredUnits,
};
/// Explicit penalized binary-propensity configuration for cross-fitted AIPW.
pub use antecedent_estimate::{
    DEFAULT_RIDGE_GRID, NuisanceFallback, PropensityNuisance, RidgeTuning,
};
/// Partially linear DML and cross-fitted AIPW via `antecedent-learn`.
pub use antecedent_estimate::{DmlAte, DmlScore};
/// `antecedent-estimate` items at the crate root.
pub use antecedent_estimate::{FittedEffect, IdentifiedSetIntervalMethod, LearnerSpec};
/// Bootstrap orchestration: standard errors.
pub use antecedent_estimate::{IntervalRequest, ResamplingReport};
/// Nonparametric estimation of identified interventional distributions via discrete empirical.
pub use antecedent_estimate::{InterventionalDistributionEstimate, ProbabilityInterval};
/// Linear adjustment ATE estimator.
pub use antecedent_estimate::{LinearAdjustmentAte, LinearFitKind};
/// Estimator-side handling of explicit outcome-observation mechanisms.
pub use antecedent_estimate::{
    ObservationEstimatorOptions, ObservationMechanismEstimator, SelectedOutcomeCorrection,
};
/// Sampled observation recovery.
pub use antecedent_estimate::{
    ObservationPattern, ObservationRow, SampledObservationInput, SampledRecoveryConfig,
    estimate_sampled_recovery,
};
/// Supplied-propensity observational exposure contrasts on a fixed network.
pub use antecedent_estimate::{ObservationalExposureSpec, estimate_observational_exposure};
/// Smoothed dose-response transport grid.
pub use antecedent_estimate::{SmoothedDoseInput, SmoothedDoseOptions};
/// Cross-fitted trial-to-nonparticipant transport with explicit sampling design.
pub use antecedent_estimate::{TrialAipwInput, TrialAipwOptions};
/// Instrumental-variable estimators: Wald ratio and two-stage least squares.
pub use antecedent_estimate::{TwoStageLeastSquares, WaldIv};
/// Conditional odds ratio from matched case-control sets.
pub use antecedent_estimate::{conditional_odds_ratio, parse_matched_sampling};
/// Validated finite-discrete laws.
pub use antecedent_expr::ExactLawError;
/// `antecedent-expr` items at the crate root.
pub use antecedent_expr::{CausalExprArena, ExprId, ExprNode, InterventionAssignment};
/// Distribution providers for compiled expression evaluation.
pub use antecedent_expr::{DistributionProvider, EvalContext, EvalError, FactorSpec};
/// Immutable, structurally checked expression programs.
pub use antecedent_expr::{
    FunctionalProgram, ProgramError, ProgramLimits, ProgramSchema, ProgramVariable,
};
/// Deterministic adaptive resampling in waves, with cancellation and failure accounting.
pub mod adaptive_resample_waves {
    pub use antecedent_estimate::adaptive_resample_waves::StopReason;
}
/// Retained adjusted regression.
pub mod adjustment_resume {
    pub use antecedent_estimate::adjustment_resume::count_adjusted_model_fits;
}
/// Independent-unit randomized ANCOVA point and HC0 sandwich variance.
pub mod ancova {
    pub use antecedent_estimate::ancova::fit_ancova;
}
/// Categorical treatments: ordered and unordered regimes.
pub mod categorical_treatment {
    pub use antecedent_estimate::categorical_treatment::CategoricalTreatmentSpec;
}
/// Sampled inference for general counterfactual transport.
pub mod counterfactual_transport_sampled {
    pub use antecedent_estimate::counterfactual_transport_sampled::{
        DatasetOverlap, OverlapDeclaration, SampledAxis, SampledCounterfactualTransportEstimate,
        SampledIntervalOutcome, SampledRegimeCounts, SampledRegimeCountsParts, SampledTableKey,
        SampledTransportLimits, SampledTransportRequest, SamplingDesign, UnseenCellRule,
    };
}
/// Preparation and evaluation of the DAG completions of a supplied CPDAG.
pub mod cpdag_scenarios {
    pub use antecedent_estimate::cpdag_scenarios::CpdagScenarioReport;
}
/// Items of `antecedent-estimate::cyclic_likelihood`.
pub mod cyclic_likelihood {
    pub use antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior;
}
/// Conjugate Gaussian regression of an outcome on a fixed dose basis.
pub mod dose_basis_regression {
    pub use antecedent_estimate::dose_basis_regression::ResidualPrior;
}
/// Items of `antecedent-estimate::experimental_fit`.
pub mod experimental_fit {
    pub use antecedent_estimate::experimental_fit::ExperimentalFitError;
}
/// Items of `antecedent-estimate::external_cyclic_fit`.
pub mod external_cyclic_fit {
    pub use antecedent_estimate::external_cyclic_fit::{ExternalFitPolicy, fit_external_feedback};
}
/// Independent-Bernoulli 2×2 factorial Horvitz–Thompson contrasts.
pub mod factorial_bernoulli {
    pub use antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2;
}
/// Scoped instrumentation of factories resolved through the public learner specification.
pub mod fit_counts {
    pub use antecedent_learn::fit_counts::count_resolved_fits;
}
/// Joint Bayesian source-target transport.
pub mod joint_bayesian_transport {
    pub use antecedent_estimate::joint_bayesian_transport::{
        DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
        PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData,
        TransportGraphClass, VaryingBlock,
    };
}
/// Design-native joint response functionals.
pub mod joint_design {
    /// Finite-history adaptive regimes evaluated against treatment-affected state transitions.
    pub mod adaptive_regime {
        pub use antecedent_estimate::joint_design::adaptive_regime::JointAdaptiveRegimeRecipe;
    }
    /// Compiler-owned joint cluster randomization with declared preassignment adjustment.
    pub mod cluster_decision {
        pub use antecedent_estimate::joint_design::cluster_decision::JointClusterRecipe;
    }
    /// Versioned executable design proofs retaining identifying functionals and source data.
    pub mod compiled {
        pub use antecedent_estimate::joint_design::compiled::{
            JointDesignInference, JointDesignRecipe,
        };
    }
    /// Whole-program randomization over independently assigned units or clusters.
    pub mod randomized {
        pub use antecedent_estimate::joint_design::randomized::{
            JointRandomizedDesign, JointRandomizedUnit,
        };
    }
}
/// Learned joint source-target transport (model-provider row `learned_joint_transport`).
pub mod learned_joint_transport {
    pub use antecedent_estimate::learned_joint_transport::LearnedJointModel;
}
/// Items of `antecedent-estimate::longitudinal_regime`.
pub mod longitudinal_regime {
    pub use antecedent_estimate::longitudinal_regime::{
        SequentialDrInputs, SubjectHistories, evaluate_g_formula_value, evaluate_regime_value,
        evaluate_sequential_dr_value,
    };
}
/// Corrected functionals under a declared measurement model.
pub mod measurement {
    pub use antecedent_estimate::measurement::PosteriorCombination;
}
/// Multiple imputation: impute, fit, pool as one uncertainty path.
pub mod multiple_imputation {
    pub use antecedent_estimate::multiple_imputation::PooledVectorEstimate;
}
/// Continuous eleven-dimensional Bayesian Verma nested-Markov candidate.
pub mod nested_markov_bayesian {
    pub use antecedent_learn::nested_markov_bayesian::{Options, Prior};
}
/// Bounded binary nested-Markov likelihood pilot (record `identify.binary_nested_markov`).
pub mod nested_markov_binary {
    pub use antecedent_estimate::nested_markov_binary::{
        AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
    };
}
/// Items of `antecedent-estimate::ou_mechanisms`.
pub mod ou_mechanisms {
    pub use antecedent_estimate::ou_mechanisms::OuMechanismOutcome;
}
/// Frozen-score doubly robust policy value utility.
pub mod policy_value {
    pub use antecedent_estimate::policy_value::{
        BinaryIpwDesign, BinaryPolicyDesign, BinaryPolicyLimits, MultiActionIpwDesign,
        MultiActionLimits, evaluate_binary_ipw_policy, evaluate_multi_action_ipw_policy,
        evaluate_policy_value_scores, uplift_by_score_bin,
    };
}
/// Design-based inference for complete randomization of independent clusters.
pub mod randomized_neyman {
    pub use antecedent_estimate::randomized_neyman::stratified_neyman_effect;
}
/// Bayesian counterpart of the randomized-trial arm contrasts.
pub mod randomized_posterior {
    pub use antecedent_estimate::randomized_posterior::{
        ArmBetaPrior, ArmNormalPrior, ArmOutcomeModel,
    };
}
/// Conservative score inference for independently randomized action contrasts.
pub mod randomized_scores {
    pub use antecedent_estimate::randomized_scores::{
        bernoulli_exact_randomization_test, cuped_ht_effect,
    };
}
/// Whole-procedure resampling shared by the rollout designs.
pub mod rollout_resampling {
    pub use antecedent_estimate::rollout_resampling::DesignEffect;
}
/// Shared-data covariance of scenario-specific estimates over one row sample.
pub mod scenario_covariance {
    pub use antecedent_estimate::scenario_covariance::ScenarioCovariance;
}
/// Selection differences and invariances behind each scenario answer.
pub mod scenario_invariance_report {
    pub use antecedent_estimate::scenario_invariance_report::{
        EnvelopeSide, ScenarioSetInvarianceReport,
    };
}
/// Bayesian-bootstrap posterior of the mean of per-subject scores.
pub mod score_posterior {
    pub use antecedent_estimate::score_posterior::ScorePosteriorOptions;
}
/// Cross-fitted AIPW score tables for retargeting and joint inference.
pub mod scores {
    pub use antecedent_estimate::scores::ScoreInference;
}
/// Smoothed dose-response transport grid.
pub mod smoothed_dose {
    pub use antecedent_estimate::smoothed_dose::parse_smoothing_kernel;
}
/// Cohort-specific event-time contrasts against never-treated subjects.
pub mod staggered_event_study {
    pub use antecedent_estimate::staggered_event_study::group_time_att;
}
/// Native risk-set summaries for randomized survival outcomes.
pub mod survival {
    pub use antecedent_estimate::survival::{
        KnownCensoring, SurvivalEndpoint, SurvivalRows, randomized_survival_ipcw_summary,
    };
}
/// Sequence-level inference for randomized switchback experiments.
pub mod switchback {
    pub use antecedent_estimate::switchback::switchback_itt;
}
/// Bayesian counterpart of the switchback treatment effect.
pub mod switchback_posterior {
    pub use antecedent_estimate::switchback_posterior::{
        CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel,
        SwitchbackPrior,
    };
}
/// Dependence-preserving sampling interval for the finite action sequence.
pub mod temporal_dependent_interval {
    pub use antecedent_estimate::temporal_dependent_interval::{
        DependentIntervalConfig, IntervalMethod, SequenceHistory, TemporalUnitPanel, UnitHistories,
    };
}
/// Uncertain pre-action state of the finite action sequence.
pub mod temporal_initial_state {
    pub use antecedent_estimate::temporal_initial_state::{
        InitialStateLaw, InitialStatePopulation, InitialStateSpec,
    };
}
/// Semantic refresh of a prepared horizon-two temporal result onto a new observation period.
pub mod temporal_refresh {
    pub use antecedent_estimate::temporal_refresh::{
        MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
    };
}
/// Exact point evaluation of one finite temporal transport sequence.
pub mod temporal_transport {
    pub use antecedent_estimate::temporal_transport::TemporalSequenceReport;
}
/// Reports over the identified members of a finite transport scenario set.
pub mod transport_scenarios {
    pub use antecedent_estimate::transport_scenarios::{PreparedScenarioSet, ScenarioSetReport};
}
