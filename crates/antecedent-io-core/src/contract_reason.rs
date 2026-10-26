//! Stable unresolved-reason codes for the analysis-result contract verifier.
//!
//! The spelling returned to callers is [`ContractReason::as_str`]. Control flow
//! matches the enum, not the string.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// One independent reason [`crate::verify_contract_against_body`] can report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContractReason {
    /// `body.estimate`.
    BodyEstimate,
    /// `body.frontdoor_assumption`.
    BodyFrontdoorAssumption,
    /// `body.identification.query`.
    BodyIdentificationQuery,
    /// `body.identification_product`.
    BodyIdentificationProduct,
    /// `body.interventional_distribution`.
    BodyInterventionalDistribution,
    /// `body.nested_counterfactual_estimate`.
    BodyNestedCounterfactualEstimate,
    /// `body.query`.
    BodyQuery,
    /// `body.response`.
    BodyResponse,
    /// `body.standard_error`.
    BodyStandardError,
    /// `body.temporal_nested_point`.
    BodyTemporalNestedPoint,
    /// `checked_aipw.row_binding`.
    CheckedAipwRowBinding,
    /// `claim.calibration`.
    ClaimCalibration,
    /// `claim.calibration.basis`.
    ClaimCalibrationBasis,
    /// `claim.calibration.off_axis`.
    ClaimCalibrationOffAxis,
    /// `claim.domains`.
    ClaimDomains,
    /// `claim.id`.
    ClaimId,
    /// `claim.kind`.
    ClaimKind,
    /// `claim.value`.
    ClaimValue,
    /// `contract.seal`.
    ContractSeal,
    /// `data_snapshot.linear_fit_moments`.
    DataSnapshotLinearFitMoments,
    /// `data_snapshot.linear_fit_moments_rank`.
    DataSnapshotLinearFitMomentsRank,
    /// `data_snapshot.observation`.
    DataSnapshotObservation,
    /// `dependencies.checked_admg_graph_posterior_response_operation`.
    DependenciesCheckedAdmgGraphPosteriorResponseOperation,
    /// `dependencies.checked_attribution_operation`.
    DependenciesCheckedAttributionOperation,
    /// `dependencies.checked_batch_family_operation`.
    DependenciesCheckedBatchFamilyOperation,
    /// `dependencies.checked_bayesian_basis_ate_operation`.
    DependenciesCheckedBayesianBasisAteOperation,
    /// `dependencies.checked_bayesian_basis_cate_operation`.
    DependenciesCheckedBayesianBasisCateOperation,
    /// `dependencies.checked_bayesian_class_conditional_operation`.
    DependenciesCheckedBayesianClassConditionalOperation,
    /// `dependencies.checked_bayesian_conditional_operation`.
    DependenciesCheckedBayesianConditionalOperation,
    /// `dependencies.checked_bayesian_dag_ate_operation`.
    DependenciesCheckedBayesianDagAteOperation,
    /// `dependencies.checked_bayesian_graph_posterior_ate_operation`.
    DependenciesCheckedBayesianGraphPosteriorAteOperation,
    /// `dependencies.checked_bayesian_iv_operation`.
    DependenciesCheckedBayesianIvOperation,
    /// `dependencies.checked_bayesian_mediation_operation`.
    DependenciesCheckedBayesianMediationOperation,
    /// `dependencies.checked_bayesian_local_ratio_operation`.
    DependenciesCheckedBayesianLocalRatioOperation,
    /// `dependencies.checked_bayesian_survival_operation`.
    DependenciesCheckedBayesianSurvivalOperation,
    /// `dependencies.checked_bayesian_quantile_operation`.
    DependenciesCheckedBayesianQuantileOperation,
    /// `dependencies.checked_bayesian_dose_policy_operation`.
    DependenciesCheckedBayesianDosePolicyOperation,
    /// `dependencies.checked_bayesian_randomized_operation`.
    DependenciesCheckedBayesianRandomizedOperation,
    /// `dependencies.checked_bayesian_panel_did_operation`.
    DependenciesCheckedBayesianPanelDidOperation,
    /// `dependencies.checked_bayesian_synthetic_control_operation`.
    DependenciesCheckedBayesianSyntheticControlOperation,
    /// `dependencies.checked_bayesian_rd_operation`.
    DependenciesCheckedBayesianRdOperation,
    /// `dependencies.checked_bayesian_robust_ate_operation`.
    DependenciesCheckedBayesianRobustAteOperation,
    /// `dependencies.checked_bayesian_temporal_class_effect_operation`.
    DependenciesCheckedBayesianTemporalClassEffectOperation,
    /// `dependencies.checked_bayesian_temporal_dag_effect_operation`.
    DependenciesCheckedBayesianTemporalDagEffectOperation,
    /// `dependencies.checked_class_graph_posterior_effect_operation`.
    DependenciesCheckedClassGraphPosteriorEffectOperation,
    /// `dependencies.checked_conditional_effect_operation`.
    DependenciesCheckedConditionalEffectOperation,
    /// `dependencies.checked_derivative_response_operation`.
    DependenciesCheckedDerivativeResponseOperation,
    /// `dependencies.checked_glm_operation`.
    DependenciesCheckedGlmOperation,
    /// `dependencies.checked_graph_posterior_effect_operation`.
    DependenciesCheckedGraphPosteriorEffectOperation,
    /// `dependencies.checked_graph_posterior_response_operation`.
    DependenciesCheckedGraphPosteriorResponseOperation,
    /// `dependencies.checked_interference_operation`.
    DependenciesCheckedInterferenceOperation,
    /// `dependencies.checked_intervention_response_operation`.
    DependenciesCheckedInterventionResponseOperation,
    /// `dependencies.checked_mediation_operation`.
    DependenciesCheckedMediationOperation,
    /// `dependencies.checked_propensity_operation`.
    DependenciesCheckedPropensityOperation,
    /// `dependencies.checked_rd_operation`.
    DependenciesCheckedRdOperation,
    /// `dependencies.checked_response_grid_operation`.
    DependenciesCheckedResponseGridOperation,
    /// `dependencies.checked_static_class_effect_operation`.
    DependenciesCheckedStaticClassEffectOperation,
    /// `dependencies.checked_static_class_response_operation`.
    DependenciesCheckedStaticClassResponseOperation,
    /// `dependencies.checked_temporal_class_effect_operation`.
    DependenciesCheckedTemporalClassEffectOperation,
    /// `dependencies.checked_temporal_class_response_operation`.
    DependenciesCheckedTemporalClassResponseOperation,
    /// `dependencies.checked_temporal_dag_effect_operation`.
    DependenciesCheckedTemporalDagEffectOperation,
    /// `dependencies.checked_temporal_graph_posterior_effect_operation`.
    DependenciesCheckedTemporalGraphPosteriorEffectOperation,
    /// `dependencies.checked_temporal_graph_posterior_response_operation`.
    DependenciesCheckedTemporalGraphPosteriorResponseOperation,
    /// `dependencies.checked_temporal_mediation_operation`.
    DependenciesCheckedTemporalMediationOperation,
    /// `dependencies.checked_temporal_response_operation`.
    DependenciesCheckedTemporalResponseOperation,
    /// `dependencies.checked_transport_trial_operation`.
    DependenciesCheckedTransportTrialOperation,
    /// `dependencies.checked_unknown_tiered_average_operation`.
    DependenciesCheckedUnknownTieredAverageOperation,
    /// `dependencies.distribution_factor_laws`.
    DependenciesDistributionFactorLaws,
    /// `dependencies.distribution_posterior_factor_draws`.
    DependenciesDistributionPosteriorFactorDraws,
    /// `dependencies.fitted_counterfactual_mechanisms`.
    DependenciesFittedCounterfactualMechanisms,
    /// `dependencies.functional_effect_factor_laws`.
    DependenciesFunctionalEffectFactorLaws,
    /// `dependencies.functional_effect_posterior_draws`.
    DependenciesFunctionalEffectPosteriorDraws,
    /// `dependencies.functional_effect_response_factor_laws`.
    DependenciesFunctionalEffectResponseFactorLaws,
    /// `dependencies.functional_effect_response_posterior_draws`.
    DependenciesFunctionalEffectResponsePosteriorDraws,
    /// `dependencies.linear_fit_sufficient_statistics`.
    DependenciesLinearFitSufficientStatistics,
    /// `dependencies.nested_counterfactual_fit_moments`.
    DependenciesNestedCounterfactualFitMoments,
    /// `dependencies.nested_counterfactual_fit_rank`.
    DependenciesNestedCounterfactualFitRank,
    /// `distribution.atom_count`.
    DistributionAtomCount,
    /// `distribution.atom_probability`.
    DistributionAtomProbability,
    /// `distribution.atom_replay_mismatch`.
    DistributionAtomReplayMismatch,
    /// `distribution.conditioning_support`.
    DistributionConditioningSupport,
    /// `distribution.expression_evaluation`.
    DistributionExpressionEvaluation,
    /// `distribution.factor_domains`.
    DistributionFactorDomains,
    /// `distribution.factor_duplicate_row`.
    DistributionFactorDuplicateRow,
    /// `distribution.factor_law_coverage`.
    DistributionFactorLawCoverage,
    /// `distribution.factor_laws_format`.
    DistributionFactorLawsFormat,
    /// `distribution.factor_normalization`.
    DistributionFactorNormalization,
    /// `distribution.factor_requirements`.
    DistributionFactorRequirements,
    /// `distribution.factor_row`.
    DistributionFactorRow,
    /// `distribution.factor_row_outside_domain`.
    DistributionFactorRowOutsideDomain,
    /// `distribution.factor_table_incomplete`.
    DistributionFactorTableIncomplete,
    /// `distribution.free_variable_law`.
    DistributionFreeVariableLaw,
    /// `distribution.free_variable_support`.
    DistributionFreeVariableSupport,
    /// `distribution.functional_program`.
    DistributionFunctionalProgram,
    /// `distribution.intervention_kind`.
    DistributionInterventionKind,
    /// `distribution.intervention_outside_domain`.
    DistributionInterventionOutsideDomain,
    /// `distribution.no_evaluable_free_variable_mass`.
    DistributionNoEvaluableFreeVariableMass,
    /// `distribution.outcome_support`.
    DistributionOutcomeSupport,
    /// `distribution.population`.
    DistributionPopulation,
    /// `distribution.program_compile`.
    DistributionProgramCompile,
    /// `distribution.provider_provenance`.
    DistributionProviderProvenance,
    /// `distribution.query_kind`.
    DistributionQueryKind,
    /// `distribution.replay_resource_limit`.
    DistributionReplayResourceLimit,
    /// `distribution.symbolic_factor_intervention`.
    DistributionSymbolicFactorIntervention,
    /// `functional_effect.expression_evaluation`.
    FunctionalEffectExpressionEvaluation,
    /// `functional_effect.factor_domains`.
    FunctionalEffectFactorDomains,
    /// `functional_effect.factor_law_coverage`.
    FunctionalEffectFactorLawCoverage,
    /// `functional_effect.factor_laws`.
    FunctionalEffectFactorLaws,
    /// `functional_effect.factor_laws_format`.
    FunctionalEffectFactorLawsFormat,
    /// `functional_effect.factor_laws_resource_limit`.
    FunctionalEffectFactorLawsResourceLimit,
    /// `functional_effect.factor_normalization`.
    FunctionalEffectFactorNormalization,
    /// `functional_effect.factor_requirements`.
    FunctionalEffectFactorRequirements,
    /// `functional_effect.free_variable_law`.
    FunctionalEffectFreeVariableLaw,
    /// `functional_effect.free_variable_support`.
    FunctionalEffectFreeVariableSupport,
    /// `functional_effect.functional_program`.
    FunctionalEffectFunctionalProgram,
    /// `functional_effect.non_finite_value`.
    FunctionalEffectNonFiniteValue,
    /// `functional_effect.program_compile`.
    FunctionalEffectProgramCompile,
    /// `functional_effect.provider_provenance`.
    FunctionalEffectProviderProvenance,
    /// `functional_effect.query_kind`.
    FunctionalEffectQueryKind,
    /// `functional_effect.replay_resource_limit`.
    FunctionalEffectReplayResourceLimit,
    /// `functional_effect.response_grid`.
    FunctionalEffectResponseGrid,
    /// `functional_effect.response_grid_resource_limit`.
    FunctionalEffectResponseGridResourceLimit,
    /// `functional_effect.response_grid_result_mismatch`.
    FunctionalEffectResponseGridResultMismatch,
    /// `functional_effect.response_result_shape`.
    FunctionalEffectResponseResultShape,
    /// `functional_effect.result`.
    FunctionalEffectResult,
    /// `functional_effect.result_mismatch`.
    FunctionalEffectResultMismatch,
    /// `functional_effect.zero_free_variable_mass`.
    FunctionalEffectZeroFreeVariableMass,
    /// `header.variable_names`.
    HeaderVariableNames,
    /// `identification.graph_class`.
    IdentificationGraphClass,
    /// `identification.observation`.
    IdentificationObservation,
    /// `identification.population_depends_on`.
    IdentificationPopulationDependsOn,
    /// `identification.schema_names`.
    IdentificationSchemaNames,
    /// `identification.status`.
    IdentificationStatus,
    /// `identification.structure_source`.
    IdentificationStructureSource,
    /// `identification.target_question`.
    IdentificationTargetQuestion,
    /// `identities.data_snapshot`.
    IdentitiesDataSnapshot,
    /// `identities.execution`.
    IdentitiesExecution,
    /// `identities.identification`.
    IdentitiesIdentification,
    /// `identities.identification_product`.
    IdentitiesIdentificationProduct,
    /// `identities.inference_binding`.
    IdentitiesInferenceBinding,
    /// `identities.observation`.
    IdentitiesObservation,
    /// `identities.program`.
    IdentitiesProgram,
    /// `identities.score_reuse`.
    IdentitiesScoreReuse,
    /// `identities.target`.
    IdentitiesTarget,
    /// `identities.target_weights`.
    IdentitiesTargetWeights,
    /// `inference_binding.bayesian`.
    InferenceBindingBayesian,
    /// `inference_binding.inference`.
    InferenceBindingInference,
    /// `inference_binding.validation_suite`.
    InferenceBindingValidationSuite,
    /// `observation.schema`.
    ObservationSchema,
    /// `program.checked_aipw_binding`.
    ProgramCheckedAipwBinding,
    /// `program.checked_aipw_lowering`.
    ProgramCheckedAipwLowering,
    /// `program.checked_aipw_lowering.target`.
    ProgramCheckedAipwLoweringTarget,
    /// `program.checked_aipw_uncertainty`.
    ProgramCheckedAipwUncertainty,
    /// `program.checked_frontdoor_lowering`.
    ProgramCheckedFrontdoorLowering,
    /// `program.checked_frontdoor_lowering.target`.
    ProgramCheckedFrontdoorLoweringTarget,
    /// `program.checked_functional_response_grid.target`.
    ProgramCheckedFunctionalResponseGridTarget,
    /// `program.checked_iv_binding`.
    ProgramCheckedIvBinding,
    /// `program.checked_iv_lowering`.
    ProgramCheckedIvLowering,
    /// `program.checked_iv_lowering.target`.
    ProgramCheckedIvLoweringTarget,
    /// `program.checked_linear_adjustment_binding`.
    ProgramCheckedLinearAdjustmentBinding,
    /// `program.checked_linear_adjustment_lowering`.
    ProgramCheckedLinearAdjustmentLowering,
    /// `program.checked_linear_adjustment_lowering.target`.
    ProgramCheckedLinearAdjustmentLoweringTarget,
    /// `program.checked_nested_counterfactual`.
    ProgramCheckedNestedCounterfactual,
    /// `program.checked_nested_counterfactual.binding`.
    ProgramCheckedNestedCounterfactualBinding,
    /// `program.checked_nested_counterfactual.graph`.
    ProgramCheckedNestedCounterfactualGraph,
    /// `program.checked_nested_counterfactual.identification`.
    ProgramCheckedNestedCounterfactualIdentification,
    /// `program.checked_nested_counterfactual.scope`.
    ProgramCheckedNestedCounterfactualScope,
    /// `program.checked_nested_counterfactual.target`.
    ProgramCheckedNestedCounterfactualTarget,
    /// `program.checked_payload_conflict`.
    ProgramCheckedPayloadConflict,
    /// `program.commitments.estimator`.
    ProgramCommitmentsEstimator,
    /// `program.commitments.identifier`.
    ProgramCommitmentsIdentifier,
    /// `program.commitments.prior_required`.
    ProgramCommitmentsPriorRequired,
    /// `program.frontdoor_binding`.
    ProgramFrontdoorBinding,
    /// `program.functional_binding`.
    ProgramFunctionalBinding,
    /// `program.functional_effect_response_grid`.
    ProgramFunctionalEffectResponseGrid,
    /// `program.functional_effect_response_grid.member_count`.
    ProgramFunctionalEffectResponseGridMemberCount,
    /// `program.functional_effect_response_grid.member_functional`.
    ProgramFunctionalEffectResponseGridMemberFunctional,
    /// `program.functional_effect_response_grid.member_intervention`.
    ProgramFunctionalEffectResponseGridMemberIntervention,
    /// `program.functional_effect_response_grid.member_program`.
    ProgramFunctionalEffectResponseGridMemberProgram,
    /// `program.functional_effect_response_grid.member_query_kind`.
    ProgramFunctionalEffectResponseGridMemberQueryKind,
    /// `program.functional_effect_response_grid.member_value`.
    ProgramFunctionalEffectResponseGridMemberValue,
    /// `program.functional_effect_response_grid.target`.
    ProgramFunctionalEffectResponseGridTarget,
    /// `program.functional_effect_response_grid.target_binding`.
    ProgramFunctionalEffectResponseGridTargetBinding,
    /// `program.functional_effect_response_grid.target_kind`.
    ProgramFunctionalEffectResponseGridTargetKind,
    /// `program.functional_program`.
    ProgramFunctionalProgram,
    /// `program.functional_program.target`.
    ProgramFunctionalProgramTarget,
    /// `program.graph_posterior_programs`.
    ProgramGraphPosteriorPrograms,
    /// `program.graph_posterior_programs.binding`.
    ProgramGraphPosteriorProgramsBinding,
    /// `program.identification`.
    ProgramIdentification,
    /// `program.identification_product`.
    ProgramIdentificationProduct,
    /// `program.target`.
    ProgramTarget,
    /// `reasoning.assumptions`.
    ReasoningAssumptions,
    /// `reasoning.identification`.
    ReasoningIdentification,
    /// `reasoning.identification.masses`.
    ReasoningIdentificationMasses,
    /// `reasoning.identification.weight_basis`.
    ReasoningIdentificationWeightBasis,
    /// `reasoning.support`.
    ReasoningSupport,
    /// `reasoning.support.empirical`.
    ReasoningSupportEmpirical,
    /// `reasoning.uncertainty`.
    ReasoningUncertainty,
    /// `score_reuse.data_snapshot`.
    ScoreReuseDataSnapshot,
    /// `score_reuse.identification`.
    ScoreReuseIdentification,
    /// `target.population`.
    TargetPopulation,
    /// `target_payload`.
    TargetPayload,
    /// `target_weights.data_snapshot`.
    TargetWeightsDataSnapshot,
    /// `target_weights.row_count`.
    TargetWeightsRowCount,
    /// `target_weights.score_reuse`.
    TargetWeightsScoreReuse,
    /// `target_weights.values`.
    TargetWeightsValues,
}

impl ContractReason {
    /// Historical reason spelling stored on [`antecedent_contract::AcceptanceReport`].
    #[must_use]
    #[allow(
        clippy::too_many_lines,
        reason = "Each ledger code keeps its historical spelling in one match."
    )]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BodyEstimate => "body.estimate",
            Self::BodyFrontdoorAssumption => "body.frontdoor_assumption",
            Self::BodyIdentificationQuery => "body.identification.query",
            Self::BodyIdentificationProduct => "body.identification_product",
            Self::BodyInterventionalDistribution => "body.interventional_distribution",
            Self::BodyNestedCounterfactualEstimate => "body.nested_counterfactual_estimate",
            Self::BodyQuery => "body.query",
            Self::BodyResponse => "body.response",
            Self::BodyStandardError => "body.standard_error",
            Self::BodyTemporalNestedPoint => "body.temporal_nested_point",
            Self::CheckedAipwRowBinding => "checked_aipw.row_binding",
            Self::ClaimCalibration => "claim.calibration",
            Self::ClaimCalibrationBasis => "claim.calibration.basis",
            Self::ClaimCalibrationOffAxis => "claim.calibration.off_axis",
            Self::ClaimDomains => "claim.domains",
            Self::ClaimId => "claim.id",
            Self::ClaimKind => "claim.kind",
            Self::ClaimValue => "claim.value",
            Self::ContractSeal => "contract.seal",
            Self::DataSnapshotLinearFitMoments => "data_snapshot.linear_fit_moments",
            Self::DataSnapshotLinearFitMomentsRank => "data_snapshot.linear_fit_moments_rank",
            Self::DataSnapshotObservation => "data_snapshot.observation",
            Self::DependenciesCheckedAdmgGraphPosteriorResponseOperation => {
                "dependencies.checked_admg_graph_posterior_response_operation"
            }
            Self::DependenciesCheckedAttributionOperation => {
                "dependencies.checked_attribution_operation"
            }
            Self::DependenciesCheckedBatchFamilyOperation => {
                "dependencies.checked_batch_family_operation"
            }
            Self::DependenciesCheckedBayesianBasisAteOperation => {
                "dependencies.checked_bayesian_basis_ate_operation"
            }
            Self::DependenciesCheckedBayesianBasisCateOperation => {
                "dependencies.checked_bayesian_basis_cate_operation"
            }
            Self::DependenciesCheckedBayesianClassConditionalOperation => {
                "dependencies.checked_bayesian_class_conditional_operation"
            }
            Self::DependenciesCheckedBayesianConditionalOperation => {
                "dependencies.checked_bayesian_conditional_operation"
            }
            Self::DependenciesCheckedBayesianDagAteOperation => {
                "dependencies.checked_bayesian_dag_ate_operation"
            }
            Self::DependenciesCheckedBayesianGraphPosteriorAteOperation => {
                "dependencies.checked_bayesian_graph_posterior_ate_operation"
            }
            Self::DependenciesCheckedBayesianIvOperation => {
                "dependencies.checked_bayesian_iv_operation"
            }
            Self::DependenciesCheckedBayesianMediationOperation => {
                "dependencies.checked_bayesian_mediation_operation"
            }
            Self::DependenciesCheckedBayesianLocalRatioOperation => {
                "dependencies.checked_bayesian_local_ratio_operation"
            }
            Self::DependenciesCheckedBayesianSurvivalOperation => {
                "dependencies.checked_bayesian_survival_operation"
            }
            Self::DependenciesCheckedBayesianQuantileOperation => {
                "dependencies.checked_bayesian_quantile_operation"
            }
            Self::DependenciesCheckedBayesianDosePolicyOperation => {
                "dependencies.checked_bayesian_dose_policy_operation"
            }
            Self::DependenciesCheckedBayesianRandomizedOperation => {
                "dependencies.checked_bayesian_randomized_operation"
            }
            Self::DependenciesCheckedBayesianPanelDidOperation => {
                "dependencies.checked_bayesian_panel_did_operation"
            }
            Self::DependenciesCheckedBayesianSyntheticControlOperation => {
                "dependencies.checked_bayesian_synthetic_control_operation"
            }
            Self::DependenciesCheckedBayesianRdOperation => {
                "dependencies.checked_bayesian_rd_operation"
            }
            Self::DependenciesCheckedBayesianRobustAteOperation => {
                "dependencies.checked_bayesian_robust_ate_operation"
            }
            Self::DependenciesCheckedBayesianTemporalClassEffectOperation => {
                "dependencies.checked_bayesian_temporal_class_effect_operation"
            }
            Self::DependenciesCheckedBayesianTemporalDagEffectOperation => {
                "dependencies.checked_bayesian_temporal_dag_effect_operation"
            }
            Self::DependenciesCheckedClassGraphPosteriorEffectOperation => {
                "dependencies.checked_class_graph_posterior_effect_operation"
            }
            Self::DependenciesCheckedConditionalEffectOperation => {
                "dependencies.checked_conditional_effect_operation"
            }
            Self::DependenciesCheckedDerivativeResponseOperation => {
                "dependencies.checked_derivative_response_operation"
            }
            Self::DependenciesCheckedGlmOperation => "dependencies.checked_glm_operation",
            Self::DependenciesCheckedGraphPosteriorEffectOperation => {
                "dependencies.checked_graph_posterior_effect_operation"
            }
            Self::DependenciesCheckedGraphPosteriorResponseOperation => {
                "dependencies.checked_graph_posterior_response_operation"
            }
            Self::DependenciesCheckedInterferenceOperation => {
                "dependencies.checked_interference_operation"
            }
            Self::DependenciesCheckedInterventionResponseOperation => {
                "dependencies.checked_intervention_response_operation"
            }
            Self::DependenciesCheckedMediationOperation => {
                "dependencies.checked_mediation_operation"
            }
            Self::DependenciesCheckedPropensityOperation => {
                "dependencies.checked_propensity_operation"
            }
            Self::DependenciesCheckedRdOperation => "dependencies.checked_rd_operation",
            Self::DependenciesCheckedResponseGridOperation => {
                "dependencies.checked_response_grid_operation"
            }
            Self::DependenciesCheckedStaticClassEffectOperation => {
                "dependencies.checked_static_class_effect_operation"
            }
            Self::DependenciesCheckedStaticClassResponseOperation => {
                "dependencies.checked_static_class_response_operation"
            }
            Self::DependenciesCheckedTemporalClassEffectOperation => {
                "dependencies.checked_temporal_class_effect_operation"
            }
            Self::DependenciesCheckedTemporalClassResponseOperation => {
                "dependencies.checked_temporal_class_response_operation"
            }
            Self::DependenciesCheckedTemporalDagEffectOperation => {
                "dependencies.checked_temporal_dag_effect_operation"
            }
            Self::DependenciesCheckedTemporalGraphPosteriorEffectOperation => {
                "dependencies.checked_temporal_graph_posterior_effect_operation"
            }
            Self::DependenciesCheckedTemporalGraphPosteriorResponseOperation => {
                "dependencies.checked_temporal_graph_posterior_response_operation"
            }
            Self::DependenciesCheckedTemporalMediationOperation => {
                "dependencies.checked_temporal_mediation_operation"
            }
            Self::DependenciesCheckedTemporalResponseOperation => {
                "dependencies.checked_temporal_response_operation"
            }
            Self::DependenciesCheckedTransportTrialOperation => {
                "dependencies.checked_transport_trial_operation"
            }
            Self::DependenciesCheckedUnknownTieredAverageOperation => {
                "dependencies.checked_unknown_tiered_average_operation"
            }
            Self::DependenciesDistributionFactorLaws => "dependencies.distribution_factor_laws",
            Self::DependenciesDistributionPosteriorFactorDraws => {
                "dependencies.distribution_posterior_factor_draws"
            }
            Self::DependenciesFittedCounterfactualMechanisms => {
                "dependencies.fitted_counterfactual_mechanisms"
            }
            Self::DependenciesFunctionalEffectFactorLaws => {
                "dependencies.functional_effect_factor_laws"
            }
            Self::DependenciesFunctionalEffectPosteriorDraws => {
                "dependencies.functional_effect_posterior_draws"
            }
            Self::DependenciesFunctionalEffectResponseFactorLaws => {
                "dependencies.functional_effect_response_factor_laws"
            }
            Self::DependenciesFunctionalEffectResponsePosteriorDraws => {
                "dependencies.functional_effect_response_posterior_draws"
            }
            Self::DependenciesLinearFitSufficientStatistics => {
                "dependencies.linear_fit_sufficient_statistics"
            }
            Self::DependenciesNestedCounterfactualFitMoments => {
                "dependencies.nested_counterfactual_fit_moments"
            }
            Self::DependenciesNestedCounterfactualFitRank => {
                "dependencies.nested_counterfactual_fit_rank"
            }
            Self::DistributionAtomCount => "distribution.atom_count",
            Self::DistributionAtomProbability => "distribution.atom_probability",
            Self::DistributionAtomReplayMismatch => "distribution.atom_replay_mismatch",
            Self::DistributionConditioningSupport => "distribution.conditioning_support",
            Self::DistributionExpressionEvaluation => "distribution.expression_evaluation",
            Self::DistributionFactorDomains => "distribution.factor_domains",
            Self::DistributionFactorDuplicateRow => "distribution.factor_duplicate_row",
            Self::DistributionFactorLawCoverage => "distribution.factor_law_coverage",
            Self::DistributionFactorLawsFormat => "distribution.factor_laws_format",
            Self::DistributionFactorNormalization => "distribution.factor_normalization",
            Self::DistributionFactorRequirements => "distribution.factor_requirements",
            Self::DistributionFactorRow => "distribution.factor_row",
            Self::DistributionFactorRowOutsideDomain => "distribution.factor_row_outside_domain",
            Self::DistributionFactorTableIncomplete => "distribution.factor_table_incomplete",
            Self::DistributionFreeVariableLaw => "distribution.free_variable_law",
            Self::DistributionFreeVariableSupport => "distribution.free_variable_support",
            Self::DistributionFunctionalProgram => "distribution.functional_program",
            Self::DistributionInterventionKind => "distribution.intervention_kind",
            Self::DistributionInterventionOutsideDomain => {
                "distribution.intervention_outside_domain"
            }
            Self::DistributionNoEvaluableFreeVariableMass => {
                "distribution.no_evaluable_free_variable_mass"
            }
            Self::DistributionOutcomeSupport => "distribution.outcome_support",
            Self::DistributionPopulation => "distribution.population",
            Self::DistributionProgramCompile => "distribution.program_compile",
            Self::DistributionProviderProvenance => "distribution.provider_provenance",
            Self::DistributionQueryKind => "distribution.query_kind",
            Self::DistributionReplayResourceLimit => "distribution.replay_resource_limit",
            Self::DistributionSymbolicFactorIntervention => {
                "distribution.symbolic_factor_intervention"
            }
            Self::FunctionalEffectExpressionEvaluation => "functional_effect.expression_evaluation",
            Self::FunctionalEffectFactorDomains => "functional_effect.factor_domains",
            Self::FunctionalEffectFactorLawCoverage => "functional_effect.factor_law_coverage",
            Self::FunctionalEffectFactorLaws => "functional_effect.factor_laws",
            Self::FunctionalEffectFactorLawsFormat => "functional_effect.factor_laws_format",
            Self::FunctionalEffectFactorLawsResourceLimit => {
                "functional_effect.factor_laws_resource_limit"
            }
            Self::FunctionalEffectFactorNormalization => "functional_effect.factor_normalization",
            Self::FunctionalEffectFactorRequirements => "functional_effect.factor_requirements",
            Self::FunctionalEffectFreeVariableLaw => "functional_effect.free_variable_law",
            Self::FunctionalEffectFreeVariableSupport => "functional_effect.free_variable_support",
            Self::FunctionalEffectFunctionalProgram => "functional_effect.functional_program",
            Self::FunctionalEffectNonFiniteValue => "functional_effect.non_finite_value",
            Self::FunctionalEffectProgramCompile => "functional_effect.program_compile",
            Self::FunctionalEffectProviderProvenance => "functional_effect.provider_provenance",
            Self::FunctionalEffectQueryKind => "functional_effect.query_kind",
            Self::FunctionalEffectReplayResourceLimit => "functional_effect.replay_resource_limit",
            Self::FunctionalEffectResponseGrid => "functional_effect.response_grid",
            Self::FunctionalEffectResponseGridResourceLimit => {
                "functional_effect.response_grid_resource_limit"
            }
            Self::FunctionalEffectResponseGridResultMismatch => {
                "functional_effect.response_grid_result_mismatch"
            }
            Self::FunctionalEffectResponseResultShape => "functional_effect.response_result_shape",
            Self::FunctionalEffectResult => "functional_effect.result",
            Self::FunctionalEffectResultMismatch => "functional_effect.result_mismatch",
            Self::FunctionalEffectZeroFreeVariableMass => {
                "functional_effect.zero_free_variable_mass"
            }
            Self::HeaderVariableNames => "header.variable_names",
            Self::IdentificationGraphClass => "identification.graph_class",
            Self::IdentificationObservation => "identification.observation",
            Self::IdentificationPopulationDependsOn => "identification.population_depends_on",
            Self::IdentificationSchemaNames => "identification.schema_names",
            Self::IdentificationStatus => "identification.status",
            Self::IdentificationStructureSource => "identification.structure_source",
            Self::IdentificationTargetQuestion => "identification.target_question",
            Self::IdentitiesDataSnapshot => "identities.data_snapshot",
            Self::IdentitiesExecution => "identities.execution",
            Self::IdentitiesIdentification => "identities.identification",
            Self::IdentitiesIdentificationProduct => "identities.identification_product",
            Self::IdentitiesInferenceBinding => "identities.inference_binding",
            Self::IdentitiesObservation => "identities.observation",
            Self::IdentitiesProgram => "identities.program",
            Self::IdentitiesScoreReuse => "identities.score_reuse",
            Self::IdentitiesTarget => "identities.target",
            Self::IdentitiesTargetWeights => "identities.target_weights",
            Self::InferenceBindingBayesian => "inference_binding.bayesian",
            Self::InferenceBindingInference => "inference_binding.inference",
            Self::InferenceBindingValidationSuite => "inference_binding.validation_suite",
            Self::ObservationSchema => "observation.schema",
            Self::ProgramCheckedAipwBinding => "program.checked_aipw_binding",
            Self::ProgramCheckedAipwLowering => "program.checked_aipw_lowering",
            Self::ProgramCheckedAipwLoweringTarget => "program.checked_aipw_lowering.target",
            Self::ProgramCheckedAipwUncertainty => "program.checked_aipw_uncertainty",
            Self::ProgramCheckedFrontdoorLowering => "program.checked_frontdoor_lowering",
            Self::ProgramCheckedFrontdoorLoweringTarget => {
                "program.checked_frontdoor_lowering.target"
            }
            Self::ProgramCheckedFunctionalResponseGridTarget => {
                "program.checked_functional_response_grid.target"
            }
            Self::ProgramCheckedIvBinding => "program.checked_iv_binding",
            Self::ProgramCheckedIvLowering => "program.checked_iv_lowering",
            Self::ProgramCheckedIvLoweringTarget => "program.checked_iv_lowering.target",
            Self::ProgramCheckedLinearAdjustmentBinding => {
                "program.checked_linear_adjustment_binding"
            }
            Self::ProgramCheckedLinearAdjustmentLowering => {
                "program.checked_linear_adjustment_lowering"
            }
            Self::ProgramCheckedLinearAdjustmentLoweringTarget => {
                "program.checked_linear_adjustment_lowering.target"
            }
            Self::ProgramCheckedNestedCounterfactual => "program.checked_nested_counterfactual",
            Self::ProgramCheckedNestedCounterfactualBinding => {
                "program.checked_nested_counterfactual.binding"
            }
            Self::ProgramCheckedNestedCounterfactualGraph => {
                "program.checked_nested_counterfactual.graph"
            }
            Self::ProgramCheckedNestedCounterfactualIdentification => {
                "program.checked_nested_counterfactual.identification"
            }
            Self::ProgramCheckedNestedCounterfactualScope => {
                "program.checked_nested_counterfactual.scope"
            }
            Self::ProgramCheckedNestedCounterfactualTarget => {
                "program.checked_nested_counterfactual.target"
            }
            Self::ProgramCheckedPayloadConflict => "program.checked_payload_conflict",
            Self::ProgramCommitmentsEstimator => "program.commitments.estimator",
            Self::ProgramCommitmentsIdentifier => "program.commitments.identifier",
            Self::ProgramCommitmentsPriorRequired => "program.commitments.prior_required",
            Self::ProgramFrontdoorBinding => "program.frontdoor_binding",
            Self::ProgramFunctionalBinding => "program.functional_binding",
            Self::ProgramFunctionalEffectResponseGrid => "program.functional_effect_response_grid",
            Self::ProgramFunctionalEffectResponseGridMemberCount => {
                "program.functional_effect_response_grid.member_count"
            }
            Self::ProgramFunctionalEffectResponseGridMemberFunctional => {
                "program.functional_effect_response_grid.member_functional"
            }
            Self::ProgramFunctionalEffectResponseGridMemberIntervention => {
                "program.functional_effect_response_grid.member_intervention"
            }
            Self::ProgramFunctionalEffectResponseGridMemberProgram => {
                "program.functional_effect_response_grid.member_program"
            }
            Self::ProgramFunctionalEffectResponseGridMemberQueryKind => {
                "program.functional_effect_response_grid.member_query_kind"
            }
            Self::ProgramFunctionalEffectResponseGridMemberValue => {
                "program.functional_effect_response_grid.member_value"
            }
            Self::ProgramFunctionalEffectResponseGridTarget => {
                "program.functional_effect_response_grid.target"
            }
            Self::ProgramFunctionalEffectResponseGridTargetBinding => {
                "program.functional_effect_response_grid.target_binding"
            }
            Self::ProgramFunctionalEffectResponseGridTargetKind => {
                "program.functional_effect_response_grid.target_kind"
            }
            Self::ProgramFunctionalProgram => "program.functional_program",
            Self::ProgramFunctionalProgramTarget => "program.functional_program.target",
            Self::ProgramGraphPosteriorPrograms => "program.graph_posterior_programs",
            Self::ProgramGraphPosteriorProgramsBinding => {
                "program.graph_posterior_programs.binding"
            }
            Self::ProgramIdentification => "program.identification",
            Self::ProgramIdentificationProduct => "program.identification_product",
            Self::ProgramTarget => "program.target",
            Self::ReasoningAssumptions => "reasoning.assumptions",
            Self::ReasoningIdentification => "reasoning.identification",
            Self::ReasoningIdentificationMasses => "reasoning.identification.masses",
            Self::ReasoningIdentificationWeightBasis => "reasoning.identification.weight_basis",
            Self::ReasoningSupport => "reasoning.support",
            Self::ReasoningSupportEmpirical => "reasoning.support.empirical",
            Self::ReasoningUncertainty => "reasoning.uncertainty",
            Self::ScoreReuseDataSnapshot => "score_reuse.data_snapshot",
            Self::ScoreReuseIdentification => "score_reuse.identification",
            Self::TargetPopulation => "target.population",
            Self::TargetPayload => "target_payload",
            Self::TargetWeightsDataSnapshot => "target_weights.data_snapshot",
            Self::TargetWeightsRowCount => "target_weights.row_count",
            Self::TargetWeightsScoreReuse => "target_weights.score_reuse",
            Self::TargetWeightsValues => "target_weights.values",
        }
    }

    /// Replay-dependency gaps a producer may export while independent consume still names them.
    #[must_use]
    pub const fn omitted_when_encoding(self) -> bool {
        matches!(
            self,
            Self::DependenciesCheckedGlmOperation
                | Self::DependenciesCheckedRdOperation
                | Self::DependenciesCheckedPropensityOperation
                | Self::DependenciesCheckedDerivativeResponseOperation
                | Self::DependenciesCheckedResponseGridOperation
                | Self::DependenciesCheckedInterventionResponseOperation
                | Self::DependenciesFittedCounterfactualMechanisms
                | Self::DependenciesCheckedGraphPosteriorEffectOperation
                | Self::DependenciesCheckedClassGraphPosteriorEffectOperation
                | Self::DependenciesCheckedStaticClassEffectOperation
                | Self::DependenciesCheckedAdmgGraphPosteriorResponseOperation
                | Self::DependenciesCheckedConditionalEffectOperation
                | Self::DependenciesCheckedMediationOperation
                | Self::DependenciesCheckedBayesianMediationOperation
                | Self::DependenciesCheckedAttributionOperation
                | Self::DependenciesCheckedBayesianDagAteOperation
                | Self::DependenciesCheckedBatchFamilyOperation
                | Self::DependenciesCheckedBayesianBasisAteOperation
                | Self::DependenciesCheckedBayesianBasisCateOperation
                | Self::DependenciesCheckedBayesianRobustAteOperation
                | Self::DependenciesCheckedBayesianConditionalOperation
                | Self::DependenciesCheckedTemporalDagEffectOperation
                | Self::DependenciesCheckedBayesianTemporalDagEffectOperation
                | Self::DependenciesCheckedTemporalClassEffectOperation
                | Self::DependenciesCheckedTemporalResponseOperation
                | Self::DependenciesCheckedTemporalMediationOperation
                | Self::DependenciesCheckedInterferenceOperation
                | Self::DependenciesCheckedUnknownTieredAverageOperation
                | Self::DependenciesCheckedBayesianGraphPosteriorAteOperation
                | Self::DependenciesCheckedBayesianClassConditionalOperation
                | Self::DependenciesCheckedTemporalClassResponseOperation
                | Self::DependenciesCheckedTemporalGraphPosteriorResponseOperation
                | Self::DependenciesCheckedBayesianTemporalClassEffectOperation
                | Self::DependenciesCheckedTemporalGraphPosteriorEffectOperation
                | Self::DependenciesCheckedStaticClassResponseOperation
                | Self::DependenciesCheckedGraphPosteriorResponseOperation
                | Self::DependenciesCheckedBayesianIvOperation
                | Self::DependenciesCheckedBayesianLocalRatioOperation
                | Self::DependenciesCheckedBayesianSurvivalOperation
                | Self::DependenciesCheckedBayesianQuantileOperation
                | Self::DependenciesCheckedBayesianDosePolicyOperation
                | Self::DependenciesCheckedBayesianRandomizedOperation
                | Self::DependenciesCheckedBayesianPanelDidOperation
                | Self::DependenciesCheckedBayesianSyntheticControlOperation
                | Self::DependenciesCheckedBayesianRdOperation
                | Self::DependenciesCheckedTransportTrialOperation
        )
    }
}

impl AsRef<str> for ContractReason {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<str> for ContractReason {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for ContractReason {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<ContractReason> for str {
    fn eq(&self, other: &ContractReason) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<ContractReason> for &str {
    fn eq(&self, other: &ContractReason) -> bool {
        *self == other.as_str()
    }
}

pub(crate) fn publish_reasons(reasons: impl IntoIterator<Item = ContractReason>) -> Vec<Arc<str>> {
    reasons.into_iter().map(|reason| Arc::<str>::from(reason.as_str())).collect()
}

/// Every ledger code, in spelling order. The test locks the historical strings.
#[cfg(test)]
pub const ALL_CONTRACT_REASONS: &[ContractReason] = &[
    ContractReason::BodyEstimate,
    ContractReason::BodyFrontdoorAssumption,
    ContractReason::BodyIdentificationQuery,
    ContractReason::BodyIdentificationProduct,
    ContractReason::BodyInterventionalDistribution,
    ContractReason::BodyNestedCounterfactualEstimate,
    ContractReason::BodyQuery,
    ContractReason::BodyResponse,
    ContractReason::BodyStandardError,
    ContractReason::BodyTemporalNestedPoint,
    ContractReason::CheckedAipwRowBinding,
    ContractReason::ClaimCalibration,
    ContractReason::ClaimCalibrationBasis,
    ContractReason::ClaimCalibrationOffAxis,
    ContractReason::ClaimDomains,
    ContractReason::ClaimId,
    ContractReason::ClaimKind,
    ContractReason::ClaimValue,
    ContractReason::ContractSeal,
    ContractReason::DataSnapshotLinearFitMoments,
    ContractReason::DataSnapshotLinearFitMomentsRank,
    ContractReason::DataSnapshotObservation,
    ContractReason::DependenciesCheckedAdmgGraphPosteriorResponseOperation,
    ContractReason::DependenciesCheckedAttributionOperation,
    ContractReason::DependenciesCheckedBatchFamilyOperation,
    ContractReason::DependenciesCheckedBayesianBasisAteOperation,
    ContractReason::DependenciesCheckedBayesianBasisCateOperation,
    ContractReason::DependenciesCheckedBayesianClassConditionalOperation,
    ContractReason::DependenciesCheckedBayesianConditionalOperation,
    ContractReason::DependenciesCheckedBayesianDagAteOperation,
    ContractReason::DependenciesCheckedBayesianGraphPosteriorAteOperation,
    ContractReason::DependenciesCheckedBayesianIvOperation,
    ContractReason::DependenciesCheckedBayesianMediationOperation,
    ContractReason::DependenciesCheckedBayesianLocalRatioOperation,
    ContractReason::DependenciesCheckedBayesianSurvivalOperation,
    ContractReason::DependenciesCheckedBayesianQuantileOperation,
    ContractReason::DependenciesCheckedBayesianDosePolicyOperation,
    ContractReason::DependenciesCheckedBayesianRandomizedOperation,
    ContractReason::DependenciesCheckedBayesianPanelDidOperation,
    ContractReason::DependenciesCheckedBayesianSyntheticControlOperation,
    ContractReason::DependenciesCheckedBayesianRdOperation,
    ContractReason::DependenciesCheckedBayesianRobustAteOperation,
    ContractReason::DependenciesCheckedBayesianTemporalClassEffectOperation,
    ContractReason::DependenciesCheckedBayesianTemporalDagEffectOperation,
    ContractReason::DependenciesCheckedClassGraphPosteriorEffectOperation,
    ContractReason::DependenciesCheckedConditionalEffectOperation,
    ContractReason::DependenciesCheckedDerivativeResponseOperation,
    ContractReason::DependenciesCheckedGlmOperation,
    ContractReason::DependenciesCheckedGraphPosteriorEffectOperation,
    ContractReason::DependenciesCheckedGraphPosteriorResponseOperation,
    ContractReason::DependenciesCheckedInterferenceOperation,
    ContractReason::DependenciesCheckedInterventionResponseOperation,
    ContractReason::DependenciesCheckedMediationOperation,
    ContractReason::DependenciesCheckedPropensityOperation,
    ContractReason::DependenciesCheckedRdOperation,
    ContractReason::DependenciesCheckedResponseGridOperation,
    ContractReason::DependenciesCheckedStaticClassEffectOperation,
    ContractReason::DependenciesCheckedStaticClassResponseOperation,
    ContractReason::DependenciesCheckedTemporalClassEffectOperation,
    ContractReason::DependenciesCheckedTemporalClassResponseOperation,
    ContractReason::DependenciesCheckedTemporalDagEffectOperation,
    ContractReason::DependenciesCheckedTemporalGraphPosteriorEffectOperation,
    ContractReason::DependenciesCheckedTemporalGraphPosteriorResponseOperation,
    ContractReason::DependenciesCheckedTemporalMediationOperation,
    ContractReason::DependenciesCheckedTemporalResponseOperation,
    ContractReason::DependenciesCheckedTransportTrialOperation,
    ContractReason::DependenciesCheckedUnknownTieredAverageOperation,
    ContractReason::DependenciesDistributionFactorLaws,
    ContractReason::DependenciesDistributionPosteriorFactorDraws,
    ContractReason::DependenciesFittedCounterfactualMechanisms,
    ContractReason::DependenciesFunctionalEffectFactorLaws,
    ContractReason::DependenciesFunctionalEffectPosteriorDraws,
    ContractReason::DependenciesFunctionalEffectResponseFactorLaws,
    ContractReason::DependenciesFunctionalEffectResponsePosteriorDraws,
    ContractReason::DependenciesLinearFitSufficientStatistics,
    ContractReason::DependenciesNestedCounterfactualFitMoments,
    ContractReason::DependenciesNestedCounterfactualFitRank,
    ContractReason::DistributionAtomCount,
    ContractReason::DistributionAtomProbability,
    ContractReason::DistributionAtomReplayMismatch,
    ContractReason::DistributionConditioningSupport,
    ContractReason::DistributionExpressionEvaluation,
    ContractReason::DistributionFactorDomains,
    ContractReason::DistributionFactorDuplicateRow,
    ContractReason::DistributionFactorLawCoverage,
    ContractReason::DistributionFactorLawsFormat,
    ContractReason::DistributionFactorNormalization,
    ContractReason::DistributionFactorRequirements,
    ContractReason::DistributionFactorRow,
    ContractReason::DistributionFactorRowOutsideDomain,
    ContractReason::DistributionFactorTableIncomplete,
    ContractReason::DistributionFreeVariableLaw,
    ContractReason::DistributionFreeVariableSupport,
    ContractReason::DistributionFunctionalProgram,
    ContractReason::DistributionInterventionKind,
    ContractReason::DistributionInterventionOutsideDomain,
    ContractReason::DistributionNoEvaluableFreeVariableMass,
    ContractReason::DistributionOutcomeSupport,
    ContractReason::DistributionPopulation,
    ContractReason::DistributionProgramCompile,
    ContractReason::DistributionProviderProvenance,
    ContractReason::DistributionQueryKind,
    ContractReason::DistributionReplayResourceLimit,
    ContractReason::DistributionSymbolicFactorIntervention,
    ContractReason::FunctionalEffectExpressionEvaluation,
    ContractReason::FunctionalEffectFactorDomains,
    ContractReason::FunctionalEffectFactorLawCoverage,
    ContractReason::FunctionalEffectFactorLaws,
    ContractReason::FunctionalEffectFactorLawsFormat,
    ContractReason::FunctionalEffectFactorLawsResourceLimit,
    ContractReason::FunctionalEffectFactorNormalization,
    ContractReason::FunctionalEffectFactorRequirements,
    ContractReason::FunctionalEffectFreeVariableLaw,
    ContractReason::FunctionalEffectFreeVariableSupport,
    ContractReason::FunctionalEffectFunctionalProgram,
    ContractReason::FunctionalEffectNonFiniteValue,
    ContractReason::FunctionalEffectProgramCompile,
    ContractReason::FunctionalEffectProviderProvenance,
    ContractReason::FunctionalEffectQueryKind,
    ContractReason::FunctionalEffectReplayResourceLimit,
    ContractReason::FunctionalEffectResponseGrid,
    ContractReason::FunctionalEffectResponseGridResourceLimit,
    ContractReason::FunctionalEffectResponseGridResultMismatch,
    ContractReason::FunctionalEffectResponseResultShape,
    ContractReason::FunctionalEffectResult,
    ContractReason::FunctionalEffectResultMismatch,
    ContractReason::FunctionalEffectZeroFreeVariableMass,
    ContractReason::HeaderVariableNames,
    ContractReason::IdentificationGraphClass,
    ContractReason::IdentificationObservation,
    ContractReason::IdentificationPopulationDependsOn,
    ContractReason::IdentificationSchemaNames,
    ContractReason::IdentificationStatus,
    ContractReason::IdentificationStructureSource,
    ContractReason::IdentificationTargetQuestion,
    ContractReason::IdentitiesDataSnapshot,
    ContractReason::IdentitiesExecution,
    ContractReason::IdentitiesIdentification,
    ContractReason::IdentitiesIdentificationProduct,
    ContractReason::IdentitiesInferenceBinding,
    ContractReason::IdentitiesObservation,
    ContractReason::IdentitiesProgram,
    ContractReason::IdentitiesScoreReuse,
    ContractReason::IdentitiesTarget,
    ContractReason::IdentitiesTargetWeights,
    ContractReason::InferenceBindingBayesian,
    ContractReason::InferenceBindingInference,
    ContractReason::InferenceBindingValidationSuite,
    ContractReason::ObservationSchema,
    ContractReason::ProgramCheckedAipwBinding,
    ContractReason::ProgramCheckedAipwLowering,
    ContractReason::ProgramCheckedAipwLoweringTarget,
    ContractReason::ProgramCheckedAipwUncertainty,
    ContractReason::ProgramCheckedFrontdoorLowering,
    ContractReason::ProgramCheckedFrontdoorLoweringTarget,
    ContractReason::ProgramCheckedFunctionalResponseGridTarget,
    ContractReason::ProgramCheckedIvBinding,
    ContractReason::ProgramCheckedIvLowering,
    ContractReason::ProgramCheckedIvLoweringTarget,
    ContractReason::ProgramCheckedLinearAdjustmentBinding,
    ContractReason::ProgramCheckedLinearAdjustmentLowering,
    ContractReason::ProgramCheckedLinearAdjustmentLoweringTarget,
    ContractReason::ProgramCheckedNestedCounterfactual,
    ContractReason::ProgramCheckedNestedCounterfactualBinding,
    ContractReason::ProgramCheckedNestedCounterfactualGraph,
    ContractReason::ProgramCheckedNestedCounterfactualIdentification,
    ContractReason::ProgramCheckedNestedCounterfactualScope,
    ContractReason::ProgramCheckedNestedCounterfactualTarget,
    ContractReason::ProgramCheckedPayloadConflict,
    ContractReason::ProgramCommitmentsEstimator,
    ContractReason::ProgramCommitmentsIdentifier,
    ContractReason::ProgramCommitmentsPriorRequired,
    ContractReason::ProgramFrontdoorBinding,
    ContractReason::ProgramFunctionalBinding,
    ContractReason::ProgramFunctionalEffectResponseGrid,
    ContractReason::ProgramFunctionalEffectResponseGridMemberCount,
    ContractReason::ProgramFunctionalEffectResponseGridMemberFunctional,
    ContractReason::ProgramFunctionalEffectResponseGridMemberIntervention,
    ContractReason::ProgramFunctionalEffectResponseGridMemberProgram,
    ContractReason::ProgramFunctionalEffectResponseGridMemberQueryKind,
    ContractReason::ProgramFunctionalEffectResponseGridMemberValue,
    ContractReason::ProgramFunctionalEffectResponseGridTarget,
    ContractReason::ProgramFunctionalEffectResponseGridTargetBinding,
    ContractReason::ProgramFunctionalEffectResponseGridTargetKind,
    ContractReason::ProgramFunctionalProgram,
    ContractReason::ProgramFunctionalProgramTarget,
    ContractReason::ProgramGraphPosteriorPrograms,
    ContractReason::ProgramGraphPosteriorProgramsBinding,
    ContractReason::ProgramIdentification,
    ContractReason::ProgramIdentificationProduct,
    ContractReason::ProgramTarget,
    ContractReason::ReasoningAssumptions,
    ContractReason::ReasoningIdentification,
    ContractReason::ReasoningIdentificationMasses,
    ContractReason::ReasoningIdentificationWeightBasis,
    ContractReason::ReasoningSupport,
    ContractReason::ReasoningSupportEmpirical,
    ContractReason::ReasoningUncertainty,
    ContractReason::ScoreReuseDataSnapshot,
    ContractReason::ScoreReuseIdentification,
    ContractReason::TargetPopulation,
    ContractReason::TargetPayload,
    ContractReason::TargetWeightsDataSnapshot,
    ContractReason::TargetWeightsRowCount,
    ContractReason::TargetWeightsScoreReuse,
    ContractReason::TargetWeightsValues,
];

#[cfg(test)]
const ALL_CONTRACT_REASON_SPELLINGS: &[&str] = &[
    "body.estimate",
    "body.frontdoor_assumption",
    "body.identification.query",
    "body.identification_product",
    "body.interventional_distribution",
    "body.nested_counterfactual_estimate",
    "body.query",
    "body.response",
    "body.standard_error",
    "body.temporal_nested_point",
    "checked_aipw.row_binding",
    "claim.calibration",
    "claim.calibration.basis",
    "claim.calibration.off_axis",
    "claim.domains",
    "claim.id",
    "claim.kind",
    "claim.value",
    "contract.seal",
    "data_snapshot.linear_fit_moments",
    "data_snapshot.linear_fit_moments_rank",
    "data_snapshot.observation",
    "dependencies.checked_admg_graph_posterior_response_operation",
    "dependencies.checked_attribution_operation",
    "dependencies.checked_batch_family_operation",
    "dependencies.checked_bayesian_basis_ate_operation",
    "dependencies.checked_bayesian_basis_cate_operation",
    "dependencies.checked_bayesian_class_conditional_operation",
    "dependencies.checked_bayesian_conditional_operation",
    "dependencies.checked_bayesian_dag_ate_operation",
    "dependencies.checked_bayesian_dose_policy_operation",
    "dependencies.checked_bayesian_graph_posterior_ate_operation",
    "dependencies.checked_bayesian_iv_operation",
    "dependencies.checked_bayesian_local_ratio_operation",
    "dependencies.checked_bayesian_mediation_operation",
    "dependencies.checked_bayesian_panel_did_operation",
    "dependencies.checked_bayesian_quantile_operation",
    "dependencies.checked_bayesian_randomized_operation",
    "dependencies.checked_bayesian_rd_operation",
    "dependencies.checked_bayesian_robust_ate_operation",
    "dependencies.checked_bayesian_survival_operation",
    "dependencies.checked_bayesian_synthetic_control_operation",
    "dependencies.checked_bayesian_temporal_class_effect_operation",
    "dependencies.checked_bayesian_temporal_dag_effect_operation",
    "dependencies.checked_class_graph_posterior_effect_operation",
    "dependencies.checked_conditional_effect_operation",
    "dependencies.checked_derivative_response_operation",
    "dependencies.checked_glm_operation",
    "dependencies.checked_graph_posterior_effect_operation",
    "dependencies.checked_graph_posterior_response_operation",
    "dependencies.checked_interference_operation",
    "dependencies.checked_intervention_response_operation",
    "dependencies.checked_mediation_operation",
    "dependencies.checked_propensity_operation",
    "dependencies.checked_rd_operation",
    "dependencies.checked_response_grid_operation",
    "dependencies.checked_static_class_effect_operation",
    "dependencies.checked_static_class_response_operation",
    "dependencies.checked_temporal_class_effect_operation",
    "dependencies.checked_temporal_class_response_operation",
    "dependencies.checked_temporal_dag_effect_operation",
    "dependencies.checked_temporal_graph_posterior_effect_operation",
    "dependencies.checked_temporal_graph_posterior_response_operation",
    "dependencies.checked_temporal_mediation_operation",
    "dependencies.checked_temporal_response_operation",
    "dependencies.checked_transport_trial_operation",
    "dependencies.checked_unknown_tiered_average_operation",
    "dependencies.distribution_factor_laws",
    "dependencies.distribution_posterior_factor_draws",
    "dependencies.fitted_counterfactual_mechanisms",
    "dependencies.functional_effect_factor_laws",
    "dependencies.functional_effect_posterior_draws",
    "dependencies.functional_effect_response_factor_laws",
    "dependencies.functional_effect_response_posterior_draws",
    "dependencies.linear_fit_sufficient_statistics",
    "dependencies.nested_counterfactual_fit_moments",
    "dependencies.nested_counterfactual_fit_rank",
    "distribution.atom_count",
    "distribution.atom_probability",
    "distribution.atom_replay_mismatch",
    "distribution.conditioning_support",
    "distribution.expression_evaluation",
    "distribution.factor_domains",
    "distribution.factor_duplicate_row",
    "distribution.factor_law_coverage",
    "distribution.factor_laws_format",
    "distribution.factor_normalization",
    "distribution.factor_requirements",
    "distribution.factor_row",
    "distribution.factor_row_outside_domain",
    "distribution.factor_table_incomplete",
    "distribution.free_variable_law",
    "distribution.free_variable_support",
    "distribution.functional_program",
    "distribution.intervention_kind",
    "distribution.intervention_outside_domain",
    "distribution.no_evaluable_free_variable_mass",
    "distribution.outcome_support",
    "distribution.population",
    "distribution.program_compile",
    "distribution.provider_provenance",
    "distribution.query_kind",
    "distribution.replay_resource_limit",
    "distribution.symbolic_factor_intervention",
    "functional_effect.expression_evaluation",
    "functional_effect.factor_domains",
    "functional_effect.factor_law_coverage",
    "functional_effect.factor_laws",
    "functional_effect.factor_laws_format",
    "functional_effect.factor_laws_resource_limit",
    "functional_effect.factor_normalization",
    "functional_effect.factor_requirements",
    "functional_effect.free_variable_law",
    "functional_effect.free_variable_support",
    "functional_effect.functional_program",
    "functional_effect.non_finite_value",
    "functional_effect.program_compile",
    "functional_effect.provider_provenance",
    "functional_effect.query_kind",
    "functional_effect.replay_resource_limit",
    "functional_effect.response_grid",
    "functional_effect.response_grid_resource_limit",
    "functional_effect.response_grid_result_mismatch",
    "functional_effect.response_result_shape",
    "functional_effect.result",
    "functional_effect.result_mismatch",
    "functional_effect.zero_free_variable_mass",
    "header.variable_names",
    "identification.graph_class",
    "identification.observation",
    "identification.population_depends_on",
    "identification.schema_names",
    "identification.status",
    "identification.structure_source",
    "identification.target_question",
    "identities.data_snapshot",
    "identities.execution",
    "identities.identification",
    "identities.identification_product",
    "identities.inference_binding",
    "identities.observation",
    "identities.program",
    "identities.score_reuse",
    "identities.target",
    "identities.target_weights",
    "inference_binding.bayesian",
    "inference_binding.inference",
    "inference_binding.validation_suite",
    "observation.schema",
    "program.checked_aipw_binding",
    "program.checked_aipw_lowering",
    "program.checked_aipw_lowering.target",
    "program.checked_aipw_uncertainty",
    "program.checked_frontdoor_lowering",
    "program.checked_frontdoor_lowering.target",
    "program.checked_functional_response_grid.target",
    "program.checked_iv_binding",
    "program.checked_iv_lowering",
    "program.checked_iv_lowering.target",
    "program.checked_linear_adjustment_binding",
    "program.checked_linear_adjustment_lowering",
    "program.checked_linear_adjustment_lowering.target",
    "program.checked_nested_counterfactual",
    "program.checked_nested_counterfactual.binding",
    "program.checked_nested_counterfactual.graph",
    "program.checked_nested_counterfactual.identification",
    "program.checked_nested_counterfactual.scope",
    "program.checked_nested_counterfactual.target",
    "program.checked_payload_conflict",
    "program.commitments.estimator",
    "program.commitments.identifier",
    "program.commitments.prior_required",
    "program.frontdoor_binding",
    "program.functional_binding",
    "program.functional_effect_response_grid",
    "program.functional_effect_response_grid.member_count",
    "program.functional_effect_response_grid.member_functional",
    "program.functional_effect_response_grid.member_intervention",
    "program.functional_effect_response_grid.member_program",
    "program.functional_effect_response_grid.member_query_kind",
    "program.functional_effect_response_grid.member_value",
    "program.functional_effect_response_grid.target",
    "program.functional_effect_response_grid.target_binding",
    "program.functional_effect_response_grid.target_kind",
    "program.functional_program",
    "program.functional_program.target",
    "program.graph_posterior_programs",
    "program.graph_posterior_programs.binding",
    "program.identification",
    "program.identification_product",
    "program.target",
    "reasoning.assumptions",
    "reasoning.identification",
    "reasoning.identification.masses",
    "reasoning.identification.weight_basis",
    "reasoning.support",
    "reasoning.support.empirical",
    "reasoning.uncertainty",
    "score_reuse.data_snapshot",
    "score_reuse.identification",
    "target.population",
    "target_payload",
    "target_weights.data_snapshot",
    "target_weights.row_count",
    "target_weights.score_reuse",
    "target_weights.values",
];

#[cfg(test)]
mod tests {
    use super::{ALL_CONTRACT_REASON_SPELLINGS, ALL_CONTRACT_REASONS, ContractReason};

    #[test]
    fn reason_spellings_stay_unique_and_historical() {
        assert_eq!(ALL_CONTRACT_REASONS.len(), ALL_CONTRACT_REASON_SPELLINGS.len());
        let mut seen = std::collections::BTreeSet::new();
        for reason in ALL_CONTRACT_REASONS {
            let spelling = reason.as_str();
            assert!(
                ALL_CONTRACT_REASON_SPELLINGS.binary_search(&spelling).is_ok(),
                "{spelling} is not a historical spelling"
            );
            assert!(seen.insert(spelling), "duplicate {spelling}");
        }
        assert!(!ContractReason::BodyQuery.omitted_when_encoding());
        assert!(ContractReason::DependenciesCheckedGlmOperation.omitted_when_encoding());
    }
}
