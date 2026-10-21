//! Read-only inspection records and accessors for a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CachedTemporalIdentification, CheckedAipwOperation, CheckedBayesianGcompOperation,
    CheckedFunctionalEffectResponseMember, CheckedIvOperation, CheckedLinearOperation,
    CheckedProgramBinding, CheckedUnknownTieredAverageInfo, ExecutionKind, Inspect,
    PreparedExecution, PreparedStudy,
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
};

/// Binds each public `checked_*_info` accessor to the sealed route whose
/// operation builds the record ([`Inspect`]). The route and the record type are
/// tied by the trait, so an accessor cannot read another route's operation.
macro_rules! inspection_accessors {
    ($($(#[doc = $doc:literal])* $name:ident -> $info:ty = $variant:ident;)+) => {
        impl PreparedStudy {
            $(
                $(#[doc = $doc])*
                #[must_use]
                pub fn $name(&self) -> Option<$info> {
                    match &self.execution {
                        PreparedExecution::$variant(operation) => Some(operation.info()),
                        _ => None,
                    }
                }
            )+
        }
    };
}

inspection_accessors! {
    /// Checked Bayesian quadratic-basis ATE plan, when this handle owns one.
    checked_bayesian_basis_ate_info -> CheckedBayesianBasisAteInfo = BayesianBasisAte;
    /// Inspect the retained target and row/fold plan for Bayesian robust ATE.
    checked_bayesian_robust_ate_info -> CheckedBayesianRobustAteInfo = BayesianRobustAte;
    /// Checked Bayesian conditional-effect plan, when this handle owns one.
    checked_bayesian_conditional_operation -> CheckedBayesianConditionalInfo = BayesianConditional;
    /// Inspect the sealed static DAG mediation target and procedure.
    checked_static_mediation_info -> CheckedStaticMediationInfo = StaticMediation;
    /// Inspect the retained Bayesian DAG mediation target, model, and validation.
    checked_bayesian_static_mediation_info -> CheckedStaticMediationInfo = BayesianStaticMediation;
    /// Inspect the sealed frequentist attribution target and supplied DAG.
    checked_attribution_info -> CheckedAttributionInfo = Attribution;
    /// Inspect the retained static CPDAG/PAG average-effect target, procedure,
    /// inference settings, and completion envelope masses.
    checked_static_class_effect_info -> CheckedStaticClassEffectInfo = StaticClassEffect;
    /// Inspect the retained Bayesian class conditional target, prior, and completion proof.
    checked_bayesian_class_conditional_info -> CheckedBayesianClassConditionalInfo = BayesianClassConditional;
    /// Inspect the frozen graph-posterior atoms and estimator procedure.
    checked_graph_posterior_effect_info -> CheckedGraphPosteriorEffectInfo = GraphPosteriorEffect;
    /// Inspect the frozen sample masses and completion envelopes for a class
    /// graph-posterior average effect.
    checked_class_graph_posterior_effect_info -> CheckedClassGraphPosteriorEffectInfo = ClassGraphPosteriorEffect;
    /// Inspect the retained Bayesian DAG posterior target, atoms, and validation suite.
    checked_bayesian_graph_posterior_ate_info -> CheckedBayesianGraphPosteriorAteInfo = BayesianGraphPosteriorAte;
    /// Inspect the retained ADMG graph-posterior response operation.
    checked_admg_graph_posterior_response_info -> CheckedAdmgGraphPosteriorResponseInfo = AdmgGraphPosteriorResponse;
    /// Inspect the retained explicit/accepted CPDAG or PAG response operation.
    checked_static_class_response_info -> CheckedStaticClassResponseInfo = StaticClassResponse;
    /// Inspect the retained DAG, CPDAG, or PAG graph-posterior response operation.
    checked_graph_posterior_response_info -> CheckedGraphPosteriorResponseInfo = GraphPosteriorResponse;
    /// Read-only target, row binding, and uncertainty selected for propensity estimation.
    checked_propensity_info -> CheckedPropensityInfo = CheckedPropensity;
    /// Inspect the retained conditional target, procedure, and bound design.
    checked_conditional_effect_info -> CheckedConditionalEffectInfo = CheckedConditional;
    /// Read-only view of the checked static derivative-response route, when
    /// preparation retained one.
    checked_derivative_response_info -> CheckedDerivativeResponseInfo = DerivativeResponse;
    /// Read-only view of the retained checked static DAG response route.
    checked_static_dag_response_info -> CheckedStaticDagResponseInfo = StaticDagResponse;
    /// Read-only view of the sealed Cell AIPW intervention-response route.
    checked_cell_aipw_response_info -> CheckedCellAipwResponseInfo = CellAipwResponse;
    /// Read-only view of the retained TemporalDag response route.
    checked_temporal_dag_response_info -> CheckedTemporalDagResponseInfo = TemporalDagResponse;
    /// Inspect the retained checked temporal mediation target before execution.
    ///
    /// # Panics
    ///
    /// Panics only if the sealed operation contains a temporal lag that cannot
    /// be represented as an `i32`; preparation rejects such lags.
    checked_temporal_mediation_info -> CheckedTemporalMediationInfo = TemporalMediation;
    /// Inspect the retained checked class-envelope or graph-posterior temporal
    /// mediation target, proof mass, and procedure before execution.
    checked_temporal_class_mediation_info -> CheckedTemporalClassMediationInfo = TemporalClassMediation;
    /// Inspect the retained fixed-network interference target and procedure.
    checked_interference_info -> CheckedInterferenceInfo = Interference;
    /// Inspect the sealed two-scenario Unknown-tier average-effect operation.
    checked_unknown_tiered_average_info -> CheckedUnknownTieredAverageInfo = UnknownTieredAverage;
    /// Inspect the retained Bayesian IV or sharp-RD model before execution.
    checked_bayesian_specialist_info -> crate::analysis::execute::CheckedBayesianSpecialistInfo = BayesianSpecialist;
    /// Inspect the retained certified trial-to-target transport operation.
    checked_transport_trial_info -> crate::analysis::execute::CheckedTransportTrialInfo = TransportTrial;
    /// Inspect the retained checked temporal effect before execution.
    checked_temporal_dag_effect_info -> CheckedTemporalDagEffectInfo = TemporalDagEffect;
    /// Inspect the retained Bayesian temporal DAG target and model settings.
    checked_bayesian_temporal_dag_effect_info -> CheckedBayesianTemporalDagEffectInfo = BayesianTemporalDagEffect;
    /// Inspect the retained source graph and complete temporal class envelope.
    checked_temporal_class_effect_info -> CheckedTemporalClassEffectInfo = TemporalClassEffect;
    /// Inspect the retained source class, completion proof, and Bayesian settings.
    checked_bayesian_temporal_class_effect_info -> CheckedBayesianTemporalClassEffectInfo = BayesianTemporalClassEffect;
    /// Inspect the frozen temporal posterior samples, atom proofs, and procedure.
    checked_temporal_graph_posterior_effect_info -> CheckedTemporalGraphPosteriorEffectInfo = TemporalGraphPosteriorEffect;
    /// Inspect the retained source graph and per-horizon temporal class
    /// response envelopes.
    checked_temporal_class_response_info -> CheckedTemporalClassResponseInfo = TemporalClassResponse;
    /// Inspect the retained temporal graph-posterior response operation.
    checked_temporal_graph_posterior_response_info -> CheckedTemporalGraphPosteriorResponseInfo = TemporalGraphPosteriorResponse;
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
        if let Some(cache) = &study.cpdag_identification_cache {
            let cases = cache.envelope.cases.iter().map(|case| {
                Ok(serde_json::json!({"identification": encode(&case.result)?, "weight": case.weight.0}))
            }).collect::<Result<Vec<_>, CausalError>>()?;
            details["cases"] = serde_json::json!(cases);
        } else if let Some(cache) = &study.pag_identification_cache {
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
        analysis.inference = InferenceMode::Bayesian(config);
        next.replace_study(analysis);
        Ok(next)
    }

    pub(crate) fn checked_program_binding(&self) -> CheckedProgramBinding<'_> {
        self.execution.program_binding()
    }

    /// Retained GLM target design for the checked DAG adjustment route.
    #[must_use]
    pub fn checked_glm_adjustment(&self) -> Option<&antecedent_estimate::PreparedGlmProblem> {
        if let PreparedExecution::CheckedGlmAdjustment(operation) = &self.execution {
            Some(&operation.preparation)
        } else {
            None
        }
    }

    /// Retained local boundary and design for the checked sharp RD route.
    #[must_use]
    pub fn checked_rd_preparation(&self) -> Option<&antecedent_estimate::CheckedRdPreparation> {
        if let PreparedExecution::CheckedRd(operation) = &self.execution {
            Some(&operation.preparation)
        } else {
            None
        }
    }

    pub(crate) fn has_sealed_glm_operation(&self) -> bool {
        matches!(&self.execution, PreparedExecution::CheckedGlmAdjustment(op) if op.sealed_for_direct_execution())
    }

    pub(crate) fn has_checked_rd_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::CheckedRd
    }

    /// Whether this prepared handle owns the complete operation for a family
    /// whose one-shot facade must execute through preparation as well.
    pub(crate) fn has_complete_program_operation(&self, query: &CausalQuery) -> bool {
        matches!(
            (&self.execution, query),
            (PreparedExecution::Distribution(_), CausalQuery::Distribution(_))
                | (PreparedExecution::PathSpecificEffect(_), CausalQuery::PathSpecific(_))
                | (
                    PreparedExecution::NestedCounterfactual(_),
                    CausalQuery::NestedCounterfactual(_)
                )
        )
    }

    /// Retained checked shared-exogenous natural direct effect operation.
    #[must_use]
    pub fn checked_nested_counterfactual_operation(
        &self,
    ) -> Option<&crate::gcm::NestedCounterfactualOperation> {
        self.execution.nested_counterfactual()
    }

    /// Retained static linear fit inputs for portable result replay.
    pub(crate) fn checked_linear_operation(&self) -> Option<&CheckedLinearOperation> {
        self.execution.linear_operation()
    }

    /// Checked Bayesian g-computation receipt for a static mean ATE, when this
    /// prepared handle uses that licensed route.
    #[must_use]
    pub fn checked_bayesian_gcomp_operation(&self) -> Option<&CheckedBayesianGcompOperation> {
        self.execution.bayesian_gcomp()
    }

    /// Inspect the retained target for the checked Bayesian basis CATE route.
    #[must_use]
    pub fn checked_bayesian_basis_cate_query(
        &self,
    ) -> Option<&antecedent_core::ConditionalEffectQuery> {
        match &self.execution {
            PreparedExecution::BayesianBasisCate(operation) => Some(operation.query()),
            _ => None,
        }
    }

    /// Validation procedure frozen with the checked Bayesian DAG effect route.
    #[must_use]
    pub fn checked_bayesian_gcomp_validation(&self) -> Option<RefuteSuite> {
        match &self.execution {
            PreparedExecution::BayesianGcomp(operation) => Some(operation.validation()),
            _ => None,
        }
    }

    /// Checked linear adjustment lowering retained by this prepared handle.
    ///
    /// `None` means this route carries no checked adjustment lowering;
    /// it does not imply that another estimator is unsupported.
    #[must_use]
    pub fn checked_linear_adjustment(
        &self,
    ) -> Option<&antecedent_estimate::CheckedLinearAdjustmentAte> {
        self.execution.checked_linear()
    }

    /// Checked AIPW lowering retained for a supported mean ATE.
    #[must_use]
    pub fn checked_aipw_ate(&self) -> Option<&antecedent_estimate::CheckedAipwPreparation> {
        self.execution.checked_aipw()
    }

    pub(crate) fn has_sealed_aipw_operation(&self) -> bool {
        self.execution
            .aipw_operation()
            .is_some_and(CheckedAipwOperation::sealed_for_direct_execution)
    }

    pub(crate) fn has_checked_counterfactual_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::Counterfactual
    }

    /// The retained graph and cross-world target for a checked counterfactual route.
    #[must_use]
    pub fn checked_counterfactual_operation(
        &self,
    ) -> Option<&crate::gcm::CheckedCounterfactualOperation> {
        if let PreparedExecution::Counterfactual(plan) = &self.execution {
            Some(plan.operation())
        } else {
            None
        }
    }

    pub(crate) fn has_checked_derivative_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::DerivativeResponse
    }

    pub(crate) fn has_checked_static_dag_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::StaticDagResponse
    }

    pub(crate) fn has_checked_temporal_dag_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalDagResponse
    }

    pub(crate) fn has_checked_temporal_class_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalClassResponse
    }

    pub(crate) fn has_checked_temporal_graph_posterior_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalGraphPosteriorResponse
    }

    pub(crate) fn has_checked_temporal_dag_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalDagEffect
    }

    pub(crate) fn has_checked_propensity_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::CheckedPropensity
    }

    /// Whether preparation retained the frequentist DAG graph-posterior effect plan.
    #[must_use]
    pub fn has_checked_graph_posterior_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::GraphPosteriorEffect
    }

    /// Whether preparation retained a frequentist CPDAG/PAG graph-posterior ATE plan.
    #[must_use]
    pub fn has_checked_class_graph_posterior_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::ClassGraphPosteriorEffect
    }

    /// Returns whether the static CPDAG/PAG average effect is bound to a
    /// completion envelope retained at preparation time.
    #[must_use]
    pub fn has_checked_static_class_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::StaticClassEffect
    }

    /// Whether preparation retained an ADMG graph-posterior response operation.
    #[must_use]
    pub fn has_checked_admg_graph_posterior_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::AdmgGraphPosteriorResponse
    }

    /// Whether preparation retained an explicit/accepted class response operation.
    #[must_use]
    pub fn has_checked_static_class_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::StaticClassResponse
    }

    /// Whether preparation retained a DAG, CPDAG, or PAG graph-posterior
    /// response operation.
    #[must_use]
    pub fn has_checked_graph_posterior_response_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::GraphPosteriorResponse
    }

    pub(crate) fn has_checked_temporal_class_mediation_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalClassMediation
    }

    /// Whether preparation retained the complete Bayesian IV or sharp-RD operation.
    #[must_use]
    pub fn has_checked_bayesian_specialist_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::BayesianSpecialist
    }

    /// Whether preparation retained the complete trial-to-target transport operation.
    #[must_use]
    pub fn has_checked_transport_trial_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TransportTrial
    }

    /// Whether preparation retained a Bayesian TemporalCpdag/TemporalPag effect plan.
    #[must_use]
    pub fn has_checked_bayesian_temporal_class_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::BayesianTemporalClassEffect
    }

    /// Whether preparation retained a temporal graph-posterior effect plan.
    #[must_use]
    pub fn has_checked_temporal_graph_posterior_effect_operation(&self) -> bool {
        self.execution.kind() == ExecutionKind::TemporalGraphPosteriorEffect
    }

    /// Compatibility spelling emphasizing the temporal effect operation family.
    #[must_use]
    pub fn checked_temporal_effect_info(&self) -> Option<CheckedTemporalDagEffectInfo> {
        self.checked_temporal_dag_effect_info()
    }

    /// Compatibility spelling emphasizing the response operation family.
    #[must_use]
    pub fn checked_temporal_response_info(&self) -> Option<CheckedTemporalDagResponseInfo> {
        self.checked_temporal_dag_response_info()
    }

    /// Checked linear two-stage front-door lowering retained by this handle.
    #[must_use]
    pub fn checked_frontdoor_linear(
        &self,
    ) -> Option<&antecedent_estimate::CheckedFrontDoorPreparation> {
        self.execution.frontdoor_linear().map(|operation| &operation.preparation)
    }

    /// Checked Wald or single-binary-instrument 2SLS lowering retained by this handle.
    #[must_use]
    pub fn checked_iv(&self) -> Option<&antecedent_estimate::CheckedIvPreparation> {
        self.execution.iv().map(CheckedIvOperation::preparation)
    }

    /// Checked functional program retained for a prepared distribution query.
    #[must_use]
    pub fn checked_distribution_program(&self) -> Option<&antecedent_expr::FunctionalProgram> {
        self.execution.distribution().map(|operation| operation.prepared.program())
    }

    /// Checked functional program retained for an ADMG general-ID effect.
    #[must_use]
    pub fn checked_functional_effect_program(&self) -> Option<&antecedent_expr::FunctionalProgram> {
        self.execution
            .functional_effect_operation()
            .map(|operation| operation.prepared.program())
            .or_else(|| {
                self.execution
                    .path_specific_effect_operation()
                    .map(|operation| operation.prepared.program())
            })
    }

    /// Ordered checked members for a prepared ADMG mean response curve or
    /// scalar intervention response.
    #[must_use]
    pub fn checked_functional_effect_response_members(
        &self,
    ) -> Option<&[CheckedFunctionalEffectResponseMember]> {
        self.execution.admg_response_curve().map(|operation| operation.members.as_ref())
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
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
        }
        if let Some(operation) = self.execution.path_specific_effect_operation() {
            return operation
                .rebind(data)?
                .prepared
                .factor_snapshot()
                .map(Some)
                .map_err(CausalError::from);
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
    pub(crate) fn study_mut(&mut self) -> &mut Study {
        self.program_cache = std::sync::OnceLock::new();
        self.contract_cache = std::sync::OnceLock::new();
        if self.execution.kind() == ExecutionKind::CheckedLinear {
            self.execution = PreparedExecution::OrdinaryDispatch;
        }
        &mut self.analysis
    }

    /// Replace the frozen study after a successful refresh.
    pub(super) fn replace_study(&mut self, study: Study) {
        self.program_cache = std::sync::OnceLock::new();
        self.contract_cache = std::sync::OnceLock::new();
        self.analysis = study;
    }

    pub(crate) fn program_cache(
        &self,
    ) -> &std::sync::OnceLock<Arc<crate::analysis::contract::ProgramPayloads>> {
        &self.program_cache
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
            self.analysis.custom_validators.iter().map(|v| v.name()).collect();
        let incoming: std::collections::BTreeSet<&str> =
            validators.iter().map(|v| v.name()).collect();
        if expected != incoming {
            return Err(crate::unsupported_reason!(
                "attested_not_reverifiable",
                "rebind_validators names must match attested names"
            ));
        }
        self.execution.set_custom_validators(&validators);
        self.analysis.custom_validators = validators;
        Ok(())
    }

    /// Population bindings frozen at prepare, when the query names a predicate
    /// or a custom target distribution.
    #[must_use]
    pub fn population_registry(&self) -> Option<&antecedent_core::PopulationRegistry> {
        self.analysis.population_registry.as_ref()
    }

    /// Names of the caller custom validators frozen at prepare, in order.
    ///
    /// These are the names a claim attests and the names
    /// [`Self::rebind_custom_validators`] requires.
    #[must_use]
    pub fn custom_validator_names(&self) -> Vec<&str> {
        self.analysis.custom_validators.iter().map(|validator| validator.name()).collect()
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
            || analysis.tiered.is_some()
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
                    && matches!(analysis.inference, InferenceMode::Bayesian(_))
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
