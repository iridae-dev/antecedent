//! Read-only inspection of a prepared handle: the retained operation, the plan
//! view, and the frozen study, schema and plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CachedTemporalIdentification, CheckedAipwOperation, CheckedBayesianGcompOperation,
    CheckedFunctionalEffectResponseMember, CheckedProgramBinding, CheckedUnknownTieredAverageInfo,
    ExecutionKind, Inspect, PreparedExecution, PreparedStudy,
};
use crate::accepted::GraphClass;
use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::EstimatorId;
use antecedent_core::{CausalQuery, CausalSchema};
use antecedent_data::TabularData;
use antecedent_estimate::ScoreTable;
use std::sync::Arc;

pub use crate::analysis::checked_bayesian_class_conditional::CheckedBayesianClassConditionalInfo;
pub use crate::analysis::checked_bayesian_graph_posterior::CheckedBayesianGraphPosteriorAteInfo;
pub use crate::analysis::checked_bayesian_temporal_class_effect::CheckedBayesianTemporalClassEffectInfo;
pub use crate::analysis::checked_bayesian_temporal_effect::CheckedBayesianTemporalDagEffectInfo;
pub use crate::analysis::checked_class_graph_posterior_effect::CheckedClassGraphPosteriorEffectInfo;
pub use crate::analysis::checked_conditional::CheckedConditionalEffectInfo;
pub use crate::analysis::checked_graph_posterior::{
    CheckedAdmgGraphPosteriorResponseInfo, CheckedGraphPosteriorEffectInfo,
    CheckedStaticClassEffectInfo,
};
pub use crate::analysis::checked_propensity::CheckedPropensityInfo;
pub use crate::analysis::checked_temporal_class_response::CheckedTemporalClassResponseInfo;
pub use crate::analysis::checked_temporal_graph_posterior_effect::CheckedTemporalGraphPosteriorEffectInfo;
pub use crate::analysis::checked_temporal_graph_posterior_response::CheckedTemporalGraphPosteriorResponseInfo;
pub use crate::analysis::execute::{
    CheckedAttributionInfo, CheckedBayesianBasisAteInfo, CheckedBayesianConditionalInfo,
    CheckedBayesianRobustAteInfo, CheckedCellAipwResponseInfo, CheckedDerivativeResponseInfo,
    CheckedGraphPosteriorResponseInfo, CheckedInterferenceInfo, CheckedStaticClassResponseInfo,
    CheckedStaticDagResponseInfo, CheckedStaticMediationInfo, CheckedTemporalClassEffectInfo,
    CheckedTemporalClassMediationInfo, CheckedTemporalDagEffectInfo,
    CheckedTemporalDagResponseInfo, CheckedTemporalMediationInfo,
    CheckedTemporalMediationTargetInfo,
};

/// The sealed operation a prepared handle retained, read without executing it.
///
/// One variant per sealed route. A route with a public inspection record carries
/// that record, built by the operation from the state it retains; a route that
/// retains a checked lowering or functional program borrows it; the remaining
/// design routes name themselves. [`PreparedStudy::plan_view`] names the variant
/// on the plan view; this enum is its typed detail.
#[non_exhaustive]
pub enum RetainedOperation<'a> {
    /// An estimator family sealed to the lowered program of its plan; the route runs
    /// from the frozen study inside that program's dispatch scope.
    EstimatorRoute,
    /// A run with no prepared handle, sealed to the lowered program of its plan.
    FreshRun,
    /// Checked general-ID functional program of an ADMG effect.
    FunctionalEffect(&'a antecedent_expr::FunctionalProgram),
    /// Checked functional program of a path-specific effect.
    PathSpecificEffect(&'a antecedent_expr::FunctionalProgram),
    /// Ordered checked members of an ADMG mean response curve or intervention response.
    AdmgResponseCurve(&'a [CheckedFunctionalEffectResponseMember]),
    /// ADMG graph-posterior response.
    AdmgGraphPosteriorResponse(CheckedAdmgGraphPosteriorResponseInfo),
    /// Checked linear adjustment lowering.
    CheckedLinear(&'a antecedent_estimate::CheckedLinearAdjustmentAte),
    /// Retained GLM target design of the checked DAG adjustment route.
    CheckedGlmAdjustment(&'a antecedent_estimate::PreparedGlmProblem),
    /// Retained local boundary and design of the checked sharp RD route.
    CheckedRd(&'a antecedent_estimate::CheckedRdPreparation),
    /// Target, row binding and uncertainty selected for propensity estimation.
    CheckedPropensity(CheckedPropensityInfo),
    /// Conditional target, procedure and bound design.
    CheckedConditional(CheckedConditionalEffectInfo),
    /// Bayesian conditional-effect plan.
    BayesianConditional(CheckedBayesianConditionalInfo),
    /// Frozen graph-posterior atoms and estimator procedure.
    GraphPosteriorEffect(CheckedGraphPosteriorEffectInfo),
    /// Frozen sample masses and completion envelopes of a class graph-posterior effect.
    ClassGraphPosteriorEffect(CheckedClassGraphPosteriorEffectInfo),
    /// Bayesian DAG posterior target, atoms and validation suite.
    BayesianGraphPosteriorAte(CheckedBayesianGraphPosteriorAteInfo),
    /// Static CPDAG/PAG average-effect target, procedure and completion envelope masses.
    StaticClassEffect(CheckedStaticClassEffectInfo),
    /// Checked AIPW lowering of a supported mean ATE.
    CheckedAipw(&'a antecedent_estimate::CheckedAipwPreparation),
    /// Retained graph and cross-world target of a checked counterfactual.
    Counterfactual(&'a crate::gcm::CheckedCounterfactualOperation),
    /// Checked shared-exogenous natural direct effect.
    NestedCounterfactual(&'a crate::gcm::NestedCounterfactualOperation),
    /// Static derivative response.
    DerivativeResponse(CheckedDerivativeResponseInfo),
    /// Cell AIPW intervention response.
    CellAipwResponse(CheckedCellAipwResponseInfo),
    /// Static DAG response.
    StaticDagResponse(CheckedStaticDagResponseInfo),
    /// Static DAG mediation target and procedure.
    StaticMediation(CheckedStaticMediationInfo),
    /// Bayesian DAG mediation target, model and validation.
    BayesianStaticMediation(CheckedStaticMediationInfo),
    /// Frequentist attribution target and supplied DAG.
    Attribution(CheckedAttributionInfo),
    /// Temporal DAG response.
    TemporalDagResponse(CheckedTemporalDagResponseInfo),
    /// Temporal DAG effect.
    TemporalDagEffect(CheckedTemporalDagEffectInfo),
    /// Bayesian temporal DAG target and model settings.
    BayesianTemporalDagEffect(CheckedBayesianTemporalDagEffectInfo),
    /// Temporal mediation target.
    TemporalMediation(CheckedTemporalMediationInfo),
    /// Fixed-network interference target and procedure.
    Interference(CheckedInterferenceInfo),
    /// Randomized assignment contrast.
    Randomized,
    /// Fixed policy value.
    PolicyValue,
    /// Continuous dose response.
    ContinuousDoseResponse,
    /// Panel difference in differences.
    PanelDid,
    /// Synthetic control.
    SyntheticControl,
    /// Local-polynomial ratio (fuzzy RD or regression kink).
    LocalPolynomialRatio,
    /// Survival contrast.
    Survival,
    /// Longitudinal regime value.
    LongitudinalRegime,
    /// Sequential regime contrast.
    SequentialContrast,
    /// History matching.
    HistoryMatch,
    /// Temporal design-based assignment contrast.
    TemporalDesign,
    /// Bounds on the average effect under a binary instrument.
    IvBounds,
    /// Source graph and complete temporal class envelope.
    TemporalClassEffect(CheckedTemporalClassEffectInfo),
    /// Checked functional program of an interventional distribution.
    Distribution(&'a antecedent_expr::FunctionalProgram),
    /// Bayesian g-computation of a static mean ATE, with its frozen validation.
    BayesianGcomp {
        /// The checked operation.
        operation: &'a CheckedBayesianGcompOperation,
        /// Validation procedure frozen with the route.
        validation: RefuteSuite,
    },
    /// Bayesian quadratic-basis ATE plan.
    BayesianBasisAte(CheckedBayesianBasisAteInfo),
    /// Target of the Bayesian basis CATE route.
    BayesianBasisCate(&'a antecedent_core::ConditionalEffectQuery),
    /// Target and row/fold plan of the Bayesian robust ATE.
    BayesianRobustAte(CheckedBayesianRobustAteInfo),
    /// Static response curve.
    StaticResponseCurve,
    /// Checked linear two-stage front-door lowering.
    FrontDoorLinear(&'a antecedent_estimate::CheckedFrontDoorPreparation),
    /// Checked Wald or single-binary-instrument 2SLS lowering.
    Iv(&'a antecedent_estimate::CheckedIvPreparation),
    /// Two-scenario Unknown-tier average effect.
    UnknownTieredAverage(CheckedUnknownTieredAverageInfo),
    /// Bayesian class conditional target, prior and completion proof.
    BayesianClassConditional(CheckedBayesianClassConditionalInfo),
    /// Source graph and per-horizon temporal class response envelopes.
    TemporalClassResponse(CheckedTemporalClassResponseInfo),
    /// Temporal graph-posterior response.
    TemporalGraphPosteriorResponse(CheckedTemporalGraphPosteriorResponseInfo),
    /// Class-envelope or graph-posterior temporal mediation target, proof mass and procedure.
    TemporalClassMediation(CheckedTemporalClassMediationInfo),
    /// Source class, completion proof and Bayesian settings.
    BayesianTemporalClassEffect(CheckedBayesianTemporalClassEffectInfo),
    /// Frozen temporal posterior samples, atom proofs and procedure.
    TemporalGraphPosteriorEffect(CheckedTemporalGraphPosteriorEffectInfo),
    /// Explicit or accepted CPDAG or PAG response.
    StaticClassResponse(CheckedStaticClassResponseInfo),
    /// DAG, CPDAG or PAG graph-posterior response.
    GraphPosteriorResponse(CheckedGraphPosteriorResponseInfo),
    /// Bayesian IV or sharp-RD model.
    BayesianSpecialist(crate::analysis::execute::CheckedBayesianSpecialistInfo),
    /// Certified trial-to-target transport.
    TransportTrial(crate::analysis::execute::CheckedTransportTrialInfo),
    /// Structure discovery, sealed to the lowered program of its search.
    Discovery,
    /// Panel temporal effect or response, sealed to its lowered program.
    PanelTemporal,
    /// Multi-environment temporal effect, sealed to its lowered program.
    MultiEnvTemporal,
    /// Temporal cross-world query, sealed to its lowered program.
    TemporalCrossWorld,
    /// Temporal ADMG effect or derivative response, sealed to its lowered program.
    TemporalAdmgEffect,
    /// Anomaly or change attribution over a structure class, sealed to its lowered program.
    ClassAttribution,
    /// Noncausal association of the observed data, sealed to its lowered program.
    Association,
    /// Mixed/class static cross-world target executed directly by its frozen family operation.
    StaticCrossWorld,
}

impl RetainedOperation<'_> {
    /// Stable `snake_case` name of the sealed route.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::EstimatorRoute => "estimator_route",
            Self::FreshRun => "fresh_run",
            Self::FunctionalEffect(_) => "functional_effect",
            Self::PathSpecificEffect(_) => "path_specific_effect",
            Self::AdmgResponseCurve(_) => "admg_response_curve",
            Self::AdmgGraphPosteriorResponse(_) => "admg_graph_posterior_response",
            Self::CheckedLinear(_) => "linear_adjustment",
            Self::CheckedGlmAdjustment(_) => "glm_adjustment",
            Self::CheckedRd(_) => "sharp_rd",
            Self::CheckedPropensity(_) => "propensity",
            Self::CheckedConditional(_) => "conditional_effect",
            Self::BayesianConditional(_) => "bayesian_conditional",
            Self::GraphPosteriorEffect(_) => "graph_posterior_effect",
            Self::ClassGraphPosteriorEffect(_) => "class_graph_posterior_effect",
            Self::BayesianGraphPosteriorAte(_) => "bayesian_graph_posterior_ate",
            Self::StaticClassEffect(_) => "static_class_effect",
            Self::CheckedAipw(_) => "aipw",
            Self::Counterfactual(_) => "counterfactual",
            Self::NestedCounterfactual(_) => "nested_counterfactual",
            Self::DerivativeResponse(_) => "derivative_response",
            Self::CellAipwResponse(_) => "cell_aipw_response",
            Self::StaticDagResponse(_) => "static_dag_response",
            Self::StaticMediation(_) => "static_mediation",
            Self::BayesianStaticMediation(_) => "bayesian_static_mediation",
            Self::Attribution(_) => "attribution",
            Self::TemporalDagResponse(_) => "temporal_dag_response",
            Self::TemporalDagEffect(_) => "temporal_dag_effect",
            Self::BayesianTemporalDagEffect(_) => "bayesian_temporal_dag_effect",
            Self::TemporalMediation(_) => "temporal_mediation",
            Self::Interference(_) => "interference",
            Self::Randomized => "randomized",
            Self::PolicyValue => "policy_value",
            Self::ContinuousDoseResponse => "continuous_dose_response",
            Self::PanelDid => "panel_did",
            Self::SyntheticControl => "synthetic_control",
            Self::LocalPolynomialRatio => "local_polynomial_ratio",
            Self::Survival => "survival",
            Self::LongitudinalRegime => "longitudinal_regime",
            Self::SequentialContrast => "sequential_contrast",
            Self::HistoryMatch => "history_match",
            Self::TemporalDesign => "temporal_design",
            Self::IvBounds => "iv_bounds",
            Self::TemporalClassEffect(_) => "temporal_class_effect",
            Self::Distribution(_) => "distribution",
            Self::BayesianGcomp { .. } => "bayesian_gcomp",
            Self::BayesianBasisAte(_) => "bayesian_basis_ate",
            Self::BayesianBasisCate(_) => "bayesian_basis_cate",
            Self::BayesianRobustAte(_) => "bayesian_robust_ate",
            Self::StaticResponseCurve => "static_response_curve",
            Self::FrontDoorLinear(_) => "frontdoor_linear",
            Self::Iv(_) => "iv",
            Self::UnknownTieredAverage(_) => "unknown_tiered_average",
            Self::BayesianClassConditional(_) => "bayesian_class_conditional",
            Self::TemporalClassResponse(_) => "temporal_class_response",
            Self::TemporalGraphPosteriorResponse(_) => "temporal_graph_posterior_response",
            Self::TemporalClassMediation(_) => "temporal_class_mediation",
            Self::BayesianTemporalClassEffect(_) => "bayesian_temporal_class_effect",
            Self::TemporalGraphPosteriorEffect(_) => "temporal_graph_posterior_effect",
            Self::StaticClassResponse(_) => "static_class_response",
            Self::GraphPosteriorResponse(_) => "graph_posterior_response",
            Self::BayesianSpecialist(_) => "bayesian_specialist",
            Self::TransportTrial(_) => "transport_trial",
            Self::Discovery => "discovery",
            Self::PanelTemporal => "panel_temporal",
            Self::MultiEnvTemporal => "multi_env_temporal",
            Self::TemporalCrossWorld => "temporal_cross_world",
            Self::TemporalAdmgEffect => "temporal_admg_effect",
            Self::ClassAttribution => "class_attribution",
            Self::Association => "association",
            Self::StaticCrossWorld => "static_cross_world",
        }
    }
}

impl std::fmt::Debug for RetainedOperation<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("RetainedOperation").field(&self.name()).finish()
    }
}

impl PreparedStudy {
    /// The sealed operation preparation retained, read without executing it.
    ///
    /// Inspection records are built from the retained state; nothing is
    /// identified, fitted or scanned.
    ///
    /// # Panics
    ///
    /// Panics only if a sealed temporal mediation operation holds a lag that does
    /// not fit an `i32`; preparation rejects such lags.
    #[must_use]
    pub fn retained_operation(&self) -> RetainedOperation<'_> {
        use PreparedExecution as E;
        use RetainedOperation as R;
        match &self.execution {
            E::EstimatorRoute(_) => R::EstimatorRoute,
            E::FreshRun(_) => R::FreshRun,
            E::FunctionalEffect(operation) => R::FunctionalEffect(operation.prepared.program()),
            E::PathSpecificEffect(operation) => R::PathSpecificEffect(operation.prepared.program()),
            E::AdmgResponseCurve(operation) => R::AdmgResponseCurve(operation.members.as_ref()),
            E::AdmgGraphPosteriorResponse(operation) => {
                R::AdmgGraphPosteriorResponse(operation.info())
            }
            E::CheckedLinear(operation) => R::CheckedLinear(&operation.preparation),
            E::CheckedGlmAdjustment(operation) => R::CheckedGlmAdjustment(&operation.preparation),
            E::CheckedRd(operation) => R::CheckedRd(&operation.preparation),
            E::CheckedPropensity(operation) => R::CheckedPropensity(operation.info()),
            E::CheckedConditional(operation) => R::CheckedConditional(operation.info()),
            E::BayesianConditional(operation) => R::BayesianConditional(operation.info()),
            E::GraphPosteriorEffect(operation) => R::GraphPosteriorEffect(operation.info()),
            E::ClassGraphPosteriorEffect(operation) => {
                R::ClassGraphPosteriorEffect(operation.info())
            }
            E::BayesianGraphPosteriorAte(operation) => {
                R::BayesianGraphPosteriorAte(operation.info())
            }
            E::StaticClassEffect(operation) => R::StaticClassEffect(operation.info()),
            E::CheckedAipw(operation) => R::CheckedAipw(&operation.preparation),
            E::Counterfactual(plan) => R::Counterfactual(plan.operation()),
            E::NestedCounterfactual(operation) => R::NestedCounterfactual(operation),
            E::DerivativeResponse(operation) => R::DerivativeResponse(operation.info()),
            E::CellAipwResponse(operation) => R::CellAipwResponse(operation.info()),
            E::StaticDagResponse(operation) => R::StaticDagResponse(operation.info()),
            E::StaticMediation(operation) => R::StaticMediation(operation.info()),
            E::BayesianStaticMediation(operation) => R::BayesianStaticMediation(operation.info()),
            E::Attribution(operation) => R::Attribution(operation.info()),
            E::TemporalDagResponse(operation) => R::TemporalDagResponse(operation.info()),
            E::TemporalDagEffect(operation) => R::TemporalDagEffect(operation.info()),
            E::BayesianTemporalDagEffect(operation) => {
                R::BayesianTemporalDagEffect(operation.info())
            }
            E::TemporalMediation(operation) => R::TemporalMediation(operation.info()),
            E::Interference(operation) => R::Interference(operation.info()),
            E::Randomized(_) => R::Randomized,
            E::PolicyValue(_) => R::PolicyValue,
            E::ContinuousDoseResponse(_) => R::ContinuousDoseResponse,
            E::PanelDid(_) => R::PanelDid,
            E::SyntheticControl(_) => R::SyntheticControl,
            E::LocalPolynomialRatio(_) => R::LocalPolynomialRatio,
            E::Survival(_) => R::Survival,
            E::LongitudinalRegime(_) => R::LongitudinalRegime,
            E::SequentialContrast(_) => R::SequentialContrast,
            E::HistoryMatch(_) => R::HistoryMatch,
            E::TemporalDesign(_) => R::TemporalDesign,
            E::IvBounds(_) => R::IvBounds,
            E::TemporalClassEffect(operation) => R::TemporalClassEffect(operation.info()),
            E::Distribution(operation) => R::Distribution(operation.prepared.program()),
            E::BayesianGcomp(operation) => R::BayesianGcomp {
                operation: operation.operation(),
                validation: operation.validation(),
            },
            E::BayesianBasisAte(operation) => R::BayesianBasisAte(operation.info()),
            E::BayesianBasisCate(operation) => R::BayesianBasisCate(operation.query()),
            E::BayesianRobustAte(operation) => R::BayesianRobustAte(operation.info()),
            E::StaticResponseCurve(_) => R::StaticResponseCurve,
            E::FrontDoorLinear(operation) => R::FrontDoorLinear(&operation.preparation),
            E::Iv(operation) => R::Iv(operation.preparation()),
            E::UnknownTieredAverage(operation) => R::UnknownTieredAverage(operation.info()),
            E::BayesianClassConditional(operation) => R::BayesianClassConditional(operation.info()),
            E::TemporalClassResponse(operation) => R::TemporalClassResponse(operation.info()),
            E::TemporalGraphPosteriorResponse(operation) => {
                R::TemporalGraphPosteriorResponse(operation.info())
            }
            E::TemporalClassMediation(operation) => R::TemporalClassMediation(operation.info()),
            E::BayesianTemporalClassEffect(operation) => {
                R::BayesianTemporalClassEffect(operation.info())
            }
            E::TemporalGraphPosteriorEffect(operation) => {
                R::TemporalGraphPosteriorEffect(operation.info())
            }
            E::StaticClassResponse(operation) => R::StaticClassResponse(operation.info()),
            E::GraphPosteriorResponse(operation) => R::GraphPosteriorResponse(operation.info()),
            E::BayesianSpecialist(operation) => R::BayesianSpecialist(operation.info()),
            E::TransportTrial(operation) => R::TransportTrial(operation.info()),
            E::Discovery(_) => R::Discovery,
            E::PanelTemporal(_) => R::PanelTemporal,
            E::MultiEnvTemporal(_) => R::MultiEnvTemporal,
            E::TemporalCrossWorld(_) => R::TemporalCrossWorld,
            E::TemporalAdmgEffect(_) => R::TemporalAdmgEffect,
            E::ClassAttribution(_) => R::ClassAttribution,
            E::Association(_) => R::Association,
            E::StaticCrossWorld(_) => R::StaticCrossWorld,
        }
    }

    /// One view over every layer of the program this handle executes: the lowered
    /// program compile attached, the identification preparation installed, the
    /// physical plan and the sealed operation preparation retained.
    ///
    /// Reading the view identifies, fits, scans and samples nothing and leases
    /// nothing from an admission pool.
    ///
    /// # Errors
    ///
    /// `route_not_lowered` when the handle's plan carries no lowered program.
    pub fn plan_view(&self) -> Result<crate::planner::PlanView, CausalError> {
        crate::planner::PlanView::of_plan(
            &self.plan,
            crate::planner::PlanStage::Prepared,
            Some(&self.analysis),
            None,
            Some(self.retained_operation().name()),
        )
    }

    /// The sealed route this handle retained.
    pub(crate) fn execution(&self) -> &PreparedExecution {
        &self.execution
    }

    /// Whether preparation retained the sealed operation of `kind`.
    pub(crate) fn retains(&self, kind: ExecutionKind) -> bool {
        self.execution.kind() == kind
    }
}

impl PreparedStudy {
    /// Read the cached identification certificate without identifying or fitting again.
    ///
    /// # Errors
    /// Returns a canonical serialization error when a retained certificate cannot be encoded.
    pub fn identification_details(&self) -> Result<serde_json::Value, CausalError> {
        let contract = self.contract()?;
        let mut details = contract.identification_details()?;
        let encode = |result: &antecedent_identify::IdentificationResult| {
            antecedent_io::identification_to_wire_with_registry(result, self.population_registry())
                .map_err(CausalError::from)
        };
        let study = self.study();
        if let Some(cache) = study.cpdag_cache() {
            let cases = cache.envelope.cases.iter().map(|case| {
                Ok(serde_json::json!({"identification": encode(&case.result)?, "weight": case.weight.0}))
            }).collect::<Result<Vec<_>, CausalError>>()?;
            details["cases"] = serde_json::json!(cases);
        } else if let Some(cache) = study.pag_cache() {
            let cases = cache.envelope.cases.iter().map(|case| {
                Ok(serde_json::json!({"identification": encode(&case.result)?, "weight": case.weight.0}))
            }).collect::<Result<Vec<_>, CausalError>>()?;
            details["cases"] = serde_json::json!(cases);
        }
        Ok(details)
    }

    /// Refute an execution on the retained tabular snapshot.
    ///
    /// # Errors
    /// Refuses absent tabular data or a suite unavailable for the retained execution.
    pub fn refute_bound(
        &self,
        result: &crate::StudyResult,
        suite: RefuteSuite,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<crate::StudyResult, CausalError> {
        let DataInput::Tabular(data) = &self.study().data else {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "retained refutation requires tabular data"
            ));
        };
        self.refute(result, data, suite, ctx)
    }
    pub(crate) fn rebind_checked_bayesian_inference(
        &self,
        config: crate::BayesianConfig,
    ) -> Result<Self, CausalError> {
        let execution = match &self.execution {
            PreparedExecution::BayesianGcomp(operation) => {
                PreparedExecution::BayesianGcomp(operation.with_inference_config(config.clone())?)
            }
            PreparedExecution::BayesianBasisAte(operation) => PreparedExecution::BayesianBasisAte(
                operation.with_inference_config(config.clone())?,
            ),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "retained Bayesian inference rebind requires its checked Gaussian operation",
                });
            }
        };
        let mut next = self.clone();
        next.execution = execution;
        let mut analysis = next.analysis.clone();
        analysis.settings.inference = InferenceMode::Bayesian(config);
        next.replace_study(analysis);
        next.seal_inspection_metadata()?;
        Ok(next)
    }

    pub(crate) fn program_binding(&self) -> CheckedProgramBinding<'_> {
        self.execution.program_binding()
    }

    pub(crate) fn has_sealed_glm_operation(&self) -> bool {
        matches!(&self.execution, PreparedExecution::CheckedGlmAdjustment(op) if op.sealed_for_direct_execution())
    }

    /// Whether this prepared handle owns the complete operation for a family
    /// whose one-shot facade must execute through preparation as well.
    pub(crate) fn has_complete_program_operation(&self, query: &CausalQuery) -> bool {
        matches!(
            (&self.execution, query),
            (PreparedExecution::Distribution(_), CausalQuery::Distribution(_))
                | (PreparedExecution::PathSpecificEffect(_), CausalQuery::PathSpecific(_))
                | (
                    PreparedExecution::StaticCrossWorld(_),
                    CausalQuery::PathSpecific(_) | CausalQuery::NestedCounterfactual(_)
                )
                | (
                    PreparedExecution::NestedCounterfactual(_),
                    CausalQuery::NestedCounterfactual(_)
                )
        )
    }

    pub(crate) fn has_sealed_aipw_operation(&self) -> bool {
        self.execution
            .aipw_operation()
            .is_some_and(CheckedAipwOperation::sealed_for_direct_execution)
    }

    /// Export the complete empirical factor laws for the retained effect program.
    pub(crate) fn functional_effect_factor_snapshot(
        &self,
        data: &TabularData,
    ) -> Result<
        Option<antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot>,
        CausalError,
    > {
        self.ensure_schema_compatible(data)?;
        if let Some(operation) = self.execution.functional_effect_operation() {
            if operation.factor_snapshot_digest == data.storage().content_digest() {
                return operation.prepared.factor_snapshot().map(Some).map_err(CausalError::from);
            }
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
        }
        if let Some(operation) = self.execution.path_specific_effect_operation() {
            if operation.factor_snapshot_digest == data.storage().content_digest() {
                return operation.prepared.factor_snapshot().map(Some).map_err(CausalError::from);
            }
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
        }
        if let PreparedExecution::StaticCrossWorld(operation) = &self.execution {
            return operation.functional_factor_snapshot(data);
        }
        // A path-restricted effect executed through the lowered route keeps no operation:
        // its factor laws are those of the identified functional on `data`.
        let route_roles = match &self.analysis.query {
            CausalQuery::PathSpecific(query) => {
                let mut roles = vec![query.treatment, query.outcome];
                roles.extend(query.path_nodes.iter().copied());
                Some(roles)
            }
            CausalQuery::NestedCounterfactual(query) => {
                Some(vec![query.treatment, query.mediator, query.outcome])
            }
            _ => None,
        };
        if let (PreparedExecution::EstimatorRoute(_), Some(extra), Some(cache)) =
            (&self.execution, route_roles, self.analysis.static_cache())
        {
            if cache.estimand.method_kind().ok()
                == Some(antecedent_expr::EstimandMethod::PathSpecificNatural)
            {
                let mut fitter = antecedent_estimate::FunctionalEffect::new();
                (fitter.row_resampling, fitter.sampling_design) =
                    self.analysis.functional_row_resampling()?;
                return fitter
                    .prepare(
                        data,
                        &cache.estimand,
                        &cache.identification.arena,
                        cache.identification.required_assumptions.clone(),
                        &extra,
                    )?
                    .factor_snapshot()
                    .map(Some)
                    .map_err(CausalError::from);
            }
        }
        Ok(None)
    }

    /// Rebound empirical factor snapshots, one for each response member in grid order.
    pub(crate) fn functional_effect_response_factor_snapshots(
        &self,
        data: &TabularData,
    ) -> Result<Option<Arc<[antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot]>>, CausalError>{
        let Some(operation) = self.execution.admg_response_curve() else { return Ok(None) };
        self.ensure_schema_compatible(data)?;
        // Compare the factors' actual producing snapshot, rather than the
        // study's retained data: generic refresh can replace only the latter.
        let rebound;
        let current = if operation.factor_snapshot_digest == data.storage().content_digest() {
            operation
        } else {
            rebound = operation.rebind(data)?;
            &rebound
        };
        let snapshots = current
            .members
            .iter()
            .map(|member| member.prepared.factor_snapshot().map_err(CausalError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Arc::from(snapshots)))
    }

    /// Export the complete provider laws for the retained distribution program,
    /// rebound to `data` after the normal semantic-schema check.
    pub(crate) fn distribution_factor_snapshot(
        &self,
        data: &TabularData,
    ) -> Result<
        Option<antecedent_estimate::functional_distribution::EmpiricalDistributionFactorSnapshot>,
        CausalError,
    > {
        let Some(operation) = self.execution.distribution() else {
            return Ok(None);
        };
        self.ensure_schema_compatible(data)?;
        if operation.factor_snapshot_digest == data.storage().content_digest() {
            return operation.prepared.factor_snapshot().map(Some).map_err(CausalError::from);
        }
        let prepared = operation.rebind(data)?;
        prepared.factor_snapshot().map(Some).map_err(CausalError::from)
    }

    /// Borrow the caller's frozen query, with original variable ids and query kind.
    #[must_use]
    pub fn query(&self) -> &CausalQuery {
        &self.analysis.query
    }

    /// Crate-visible frozen study.
    pub(crate) fn study(&self) -> &Study {
        &self.analysis
    }

    #[cfg(test)]
    pub(crate) fn study_mut(&mut self) -> Result<&mut Study, CausalError> {
        self.program_cache = std::sync::OnceLock::new();
        self.contract_cache = std::sync::OnceLock::new();
        self.inspection_header = std::sync::OnceLock::new();
        if self.execution.kind() == ExecutionKind::CheckedLinear {
            self.execution = PreparedExecution::EstimatorRoute(
                crate::analysis::execute::CheckedRouteOperation::lowered(
                    &self.analysis,
                    &self.plan,
                )?,
            );
        }
        Ok(&mut self.analysis)
    }

    /// Replace the frozen study after a successful refresh.
    pub(super) fn replace_study(&mut self, study: Study) {
        self.program_cache = std::sync::OnceLock::new();
        self.contract_cache = std::sync::OnceLock::new();
        self.inspection_header = std::sync::OnceLock::new();
        self.analysis = study;
    }

    pub(crate) fn posterior_programs(
        &self,
        query: &CausalQuery,
    ) -> Result<Option<&antecedent_io::GraphPosteriorProgramsWire>, CausalError> {
        self.posterior_programs.as_deref().map(|retained| {
            if retained.query != crate::analysis::contract_identity::canonical_query_bytes(query)? {
                return Err(CausalError::Compile {
                    message: "retained graph-posterior programs require their prepare-time query; re-prepare after changing the target population or query".into(),
                });
            }
            Ok(&retained.programs)
        }).transpose()
    }

    pub(crate) fn program_cache(
        &self,
    ) -> &std::sync::OnceLock<Arc<crate::analysis::contract::ProgramPayloads>> {
        &self.program_cache
    }

    pub(crate) fn inspection_header(
        &self,
    ) -> &std::sync::OnceLock<Arc<crate::analysis::contract::PreparedInspection>> {
        &self.inspection_header
    }

    pub(crate) fn contract_cache(
        &self,
    ) -> &std::sync::OnceLock<Arc<crate::analysis::contract::CompiledContract>> {
        &self.contract_cache
    }

    /// Replace custom validators. Incoming names must match the prepare-time set.
    ///
    /// # Errors
    ///
    /// Name mismatch.
    pub fn rebind_custom_validators(
        &mut self,
        validators: Vec<std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>>,
    ) -> Result<(), CausalError> {
        let expected: std::collections::BTreeSet<&str> =
            self.analysis.declared.custom_validators.iter().map(|v| v.name()).collect();
        let incoming: std::collections::BTreeSet<&str> =
            validators.iter().map(|v| v.name()).collect();
        if expected != incoming {
            return Err(crate::unsupported_reason!(
                "attested_not_reverifiable",
                "rebind_validators names must match attested names"
            ));
        }
        self.execution.set_custom_validators(&validators);
        self.analysis.declared.custom_validators = validators;
        Ok(())
    }

    /// Population bindings frozen at prepare, when the query names a predicate
    /// or a custom target distribution.
    #[must_use]
    pub fn population_registry(&self) -> Option<&antecedent_core::PopulationRegistry> {
        self.analysis.declared.population_registry.as_ref()
    }

    /// Borrow the caller validators frozen at preparation.
    ///
    /// Host bindings may wrap these on an execution-local clone before calling
    /// [`Self::rebind_custom_validators`]. The prepared handle and its callback
    /// declarations remain unchanged.
    #[must_use]
    pub fn custom_validators(
        &self,
    ) -> &[std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>] {
        &self.analysis.declared.custom_validators
    }

    /// Names of the caller custom validators frozen at prepare, in order.
    ///
    /// These are the names a claim attests and the names
    /// [`Self::rebind_custom_validators`] requires.
    #[must_use]
    pub fn custom_validator_names(&self) -> Vec<&str> {
        self.analysis.declared.custom_validators.iter().map(|validator| validator.name()).collect()
    }

    /// Stream progressive stages from later estimates of this handle.
    ///
    /// `None` stops streaming. The sink is an in-process execution control: it
    /// is not part of the contract, program, or claim identity.
    pub fn set_stage_sink(
        &mut self,
        sink: Option<Arc<dyn crate::analysis::stage::StageResultSink>>,
    ) {
        self.execution.set_stage_sink(sink.clone());
        Arc::make_mut(&mut self.analysis.config).stage_sink = sink;
    }

    /// Whether estimates of this handle stream identify → `estimate_point` →
    /// uncertainty → validate stage payloads.
    ///
    /// Stages describe one identification and one scalar effect estimate. They
    /// stream from the single-estimand static executors: a supplied or DAG-coerced
    /// DAG average effect (Frequentist or Bayesian, excluding `rd.sharp` and the
    /// general-ID functional plug-in) and the Bayesian DAG conditional effect.
    #[must_use]
    pub fn streams_stages(&self) -> bool {
        let analysis = &self.analysis;
        if analysis.graph_posterior.is_some()
            || analysis.declared.tiered.is_some()
            || !matches!(analysis.data, DataInput::Tabular(_))
        {
            return false;
        }
        let dag_like = match analysis.graph.class() {
            GraphClass::Dag => true,
            GraphClass::Admg => analysis
                .graph
                .as_admg()
                .is_some_and(|admg| !crate::analysis::execute::admg_has_bidirected(admg)),
            _ => false,
        };
        let estimator = self.plan.binding().estimator.named();
        match &analysis.query {
            CausalQuery::AverageEffect(_) => {
                dag_like
                    && !matches!(
                        estimator,
                        Some(EstimatorId::RdSharp | EstimatorId::FunctionalEffect)
                    )
            }
            CausalQuery::ConditionalEffect(_) => {
                analysis.graph.class() == GraphClass::Dag
                    && matches!(analysis.settings.inference, InferenceMode::Bayesian(_))
            }
            _ => false,
        }
    }

    /// Frozen horizon-specific identification and its exact unfolded variable namespace.
    ///
    /// A `TemporalDag` prepare caches this directly. A DBN posterior or a TemporalCpdag/Pag
    /// envelope caches a per-atom or per-completion result instead; this projects either
    /// onto the same shape (see `full_temporal_identification`), so an
    /// exported `analysis_result` artifact validates its identification against the same
    /// namespace the compiled contract already uses.
    #[must_use]
    pub fn temporal_identification(
        &self,
    ) -> Option<std::borrow::Cow<'_, CachedTemporalIdentification>> {
        crate::analysis::contract::full_temporal_identification(&self.analysis)
    }

    /// Borrow the frozen schema fingerprint.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Matrix structure-source axis frozen at prepare.
    #[must_use]
    pub const fn structure_source(&self) -> crate::support::StructureSource {
        self.state.analysis.structure_source()
    }

    /// Evidence contract frozen at prepare. `None` when the query is off-axis.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.state.analysis.support_status()
    }

    /// Borrow the ready physical plan retained from prepare.
    #[must_use]
    pub fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// Borrow the prepare-time AIPW score table, when the cell exported one.
    #[must_use]
    pub fn score_table(&self) -> Option<&ScoreTable> {
        self.score_table.as_ref()
    }

    /// Shared batch design attached when this plan was prepared inside a batch.
    #[must_use]
    pub fn shared_design(&self) -> Option<&crate::analysis::batch::SharedBatchDesign> {
        self.analysis.shared_batch_design.as_deref()
    }
}
