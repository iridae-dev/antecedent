//! Artifact plan ids compiled in `compile.rs`.
//!
//! [`PlanId::as_str`] is the historical wire string. The enum is the source of
//! those ids so a new route cannot invent a literal beside this table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Logical-plan id written on a compiled record.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum PlanId {
    /// Static ADMG interventional distribution.
    StaticAdmgDistribution,
    /// Static DAG conditional effect.
    StaticConditional,
    /// Static CPDAG conditional effect.
    StaticCpdagConditional,
    /// Static PAG conditional effect.
    StaticPagConditional,
    /// Temporal DAG mediation.
    TemporalMediation,
    /// Temporal class mediation.
    TemporalMediationClass,
    /// Static mediation.
    StaticMediation,
    /// Trial transport.
    TransportTrial,
    /// Interference design.
    InterferenceDesign,
    /// Randomized ITT.
    RandomizedTrialItt,
    /// Multi-action policy value.
    RandomizedPolicyValueMultiActionIpw,
    /// IPW policy value.
    RandomizedPolicyValueIpw,
    /// Doubly robust policy value.
    RandomizedPolicyValueDr,
    /// Continuous dose response.
    PolicyContinuousDoseResponse,
    /// Randomized survival product limit.
    RandomizedSurvivalProductLimit,
    /// Longitudinal IPW regime.
    LongitudinalIpwRegime,
    /// Longitudinal g-formula regime.
    LongitudinalGFormulaRegime,
    /// Longitudinal sequential doubly robust regime.
    LongitudinalSequentialDrRegime,
    /// Longitudinal marginal structural model.
    LongitudinalMarginalStructuralModel,
    /// Bayesian posterior of a binary policy value.
    BayesianPolicyValue,
    /// Bayesian posterior of a sequential regime value.
    BayesianSequentialRegimeValue,
    /// Within-period history-matched contrast.
    PeriodHistoryMatching,
    /// Normal interval of the observational sequential contrast.
    SequentialCrossfitContrastWald,
    /// Modular posterior of the observational sequential contrast.
    BayesianSequentialCrossfitContrast,
    /// Bayesian piecewise-exponential survival posterior.
    BayesianPiecewiseSurvival,
    /// Bayesian group-polynomial dose policy posterior.
    BayesianDosePolicy,
    /// GCM tabular query.
    GcmQuery,
    /// GCM temporal query.
    GcmTemporalQuery,
    /// Temporal CPDAG cross-world.
    TemporalCpdagCrossWorld,
    /// Temporal PAG cross-world.
    TemporalPagCrossWorld,
    /// Temporal DAG cross-world linear adjustment.
    TemporalCrossWorldLinear,
    /// Static ADMG average effect.
    StaticAdmgAte,
    /// Temporal CPDAG effect.
    TemporalCpdagEffect,
    /// Temporal PAG effect.
    TemporalPagEffect,
    /// Panel CPDAG effect.
    PanelCpdagEffect,
    /// Panel PAG effect.
    PanelPagEffect,
    /// Panel CPDAG response.
    PanelCpdagResponse,
    /// Panel PAG response.
    PanelPagResponse,
    /// Panel class response that is neither CPDAG nor PAG.
    PanelClassResponse,
    /// Static ADMG response.
    StaticAdmgResponse,
    /// Static ADMG conditional effect.
    StaticAdmgConditional,
    /// Static structure search.
    StructureDiscovery,
    /// Temporal structure search (series, panel or multi-environment).
    TemporalStructureDiscovery,
    /// Attribution on a fitted structural model.
    GcmAttribute,
    /// Simulation on a fitted structural model.
    GcmSimulate,
    /// Excursion contrasts of a sequentially randomized unit-period design.
    TemporalDesignExcursion,
    /// Posterior of the excursion contrasts of a sequentially randomized design.
    BayesianTemporalDesignExcursion,
    /// Inverse-probability weighted regime survival on subject trajectories.
    TemporalRegimeSurvival,
    /// Joint-likelihood posterior of regime survival on subject trajectories.
    BayesianTemporalRegimeSurvival,
}

impl PlanId {
    /// Historical artifact id. Callers that digest `plan_id` keep these bytes.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StaticAdmgDistribution => "static_admg_distribution",
            Self::StaticConditional => "static_conditional",
            Self::StaticCpdagConditional => "static_cpdag_conditional",
            Self::StaticPagConditional => "static_pag_conditional",
            Self::TemporalMediation => "temporal_mediation",
            Self::TemporalMediationClass => "temporal_mediation_class",
            Self::StaticMediation => "static_mediation",
            Self::TransportTrial => "transport_trial",
            Self::InterferenceDesign => "interference_design",
            Self::RandomizedTrialItt => "randomized_trial_itt",
            Self::RandomizedPolicyValueMultiActionIpw => "randomized_policy_value_multi_action_ipw",
            Self::RandomizedPolicyValueIpw => "randomized_policy_value_ipw",
            Self::RandomizedPolicyValueDr => "randomized_policy_value_dr",
            Self::PolicyContinuousDoseResponse => "policy.continuous_dose_response",
            Self::RandomizedSurvivalProductLimit => "randomized.survival_product_limit",
            Self::LongitudinalIpwRegime => "longitudinal.ipw_regime",
            Self::LongitudinalGFormulaRegime => "longitudinal.g_formula_regime",
            Self::LongitudinalSequentialDrRegime => "longitudinal.sequential_dr_regime",
            Self::LongitudinalMarginalStructuralModel => "longitudinal.marginal_structural_model",
            Self::BayesianPolicyValue => "bayesian_policy_value",
            Self::BayesianPiecewiseSurvival => "bayesian_piecewise_survival",
            Self::BayesianDosePolicy => "bayesian_dose_policy",
            Self::BayesianSequentialRegimeValue => "bayesian.sequential_regime_value",
            Self::PeriodHistoryMatching => "matching.period_history_contrast",
            Self::SequentialCrossfitContrastWald => "sequential.crossfit_contrast.wald",
            Self::BayesianSequentialCrossfitContrast => "bayesian.sequential_crossfit_contrast",
            Self::GcmQuery => "gcm_query",
            Self::GcmTemporalQuery => "gcm_temporal_query",
            Self::TemporalCpdagCrossWorld => "temporal_cpdag_cross_world",
            Self::TemporalPagCrossWorld => "temporal_pag_cross_world",
            Self::TemporalCrossWorldLinear => "temporal.cross_world.linear",
            Self::StaticAdmgAte => "static_admg_ate",
            Self::TemporalCpdagEffect => "temporal_cpdag_effect",
            Self::TemporalPagEffect => "temporal_pag_effect",
            Self::PanelCpdagEffect => "panel_cpdag_effect",
            Self::PanelPagEffect => "panel_pag_effect",
            Self::PanelCpdagResponse => "panel_cpdag_response",
            Self::PanelPagResponse => "panel_pag_response",
            Self::PanelClassResponse => "panel_class_response",
            Self::StaticAdmgResponse => "static_admg_response",
            Self::StaticAdmgConditional => "static_admg_conditional",
            Self::StructureDiscovery => "structure_discovery",
            Self::TemporalStructureDiscovery => "temporal_structure_discovery",
            Self::GcmAttribute => "gcm_attribute",
            Self::GcmSimulate => "gcm_simulate",
            Self::TemporalDesignExcursion => "temporal_design.excursion",
            Self::BayesianTemporalDesignExcursion => "bayesian.temporal_design.excursion",
            Self::TemporalRegimeSurvival => "temporal_design.regime_survival",
            Self::BayesianTemporalRegimeSurvival => "bayesian.temporal_design.regime_survival",
        }
    }
}
