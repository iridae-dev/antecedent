//! Estimate clicks and sealed-operation dispatch for a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    PreparedExecution, PreparedStudy, overlay_prepared_score_functional, query_reads_score_table,
};
use crate::analysis::builder::DataInput;
use crate::analysis::{StaticClassGraph, StaticClassIdentification};
use crate::error::CausalError;
use crate::result::StudyResult;
use antecedent_core::{CausalQuery, ExecutionContext};
use antecedent_data::TabularData;
use std::sync::Arc;

impl PreparedStudy {
    /// Re-estimate on `data` without recompiling the physical plan.
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn estimate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let shared = self
            .analysis
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(data).map(Arc::new))
            .transpose()?;
        self.estimate_with_shared(data, shared, ctx)
    }

    pub(crate) fn estimate_with_shared(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        // A stats-layer refusal of a regression, GLM, IV or matching design fit learns here
        // which plan it was for and, for a rank refusal, which columns are dependent on the
        // table the fit read.
        self.estimate_with_shared_unnamed(data, shared, ctx).map_err(|error| {
            crate::analysis::preflight::name_stats_refusal(&self.analysis, Some(data), error, ctx)
        })
    }

    fn estimate_with_shared_unnamed(
        &self,
        data: &TabularData,
        shared: Option<Arc<crate::analysis::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        if let Some(result) = self.estimate_sealed_program_operation(data, ctx)? {
            return Ok(result);
        }
        if let Some(result) = self.estimate_sealed_static_operation(data, ctx)? {
            return Ok(result);
        }
        if let Some(result) = self.estimate_sealed_estimator_operation(data, ctx)? {
            return Ok(result);
        }
        if let Some(result) = self.estimate_sealed_class_operation(data, ctx)? {
            return Ok(result);
        }
        let mut click_analysis = self.analysis.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.interference =
            click_analysis.interference.as_ref().map(|spec| spec.bound_to(data)).transpose()?;
        click_analysis.shared_batch_design = shared;
        let rebound = self.execution.rebind_for_dispatch(data)?;
        let execution = rebound.as_ref().unwrap_or(&self.execution);
        let mut result = click_analysis.execute_tabular(data, &self.plan, execution, ctx)?;
        // `execute_tabular` bypasses `Study::execute_on`, which is where fresh runs
        // record which refutation reports are caller-attested. Without the names,
        // the claim would drop custom-validator evidence from `attested` and from
        // the claim identity.
        result.custom_validator_names = click_analysis
            .custom_validators
            .iter()
            .map(|validator| Arc::from(validator.name()))
            .collect();
        // Only a non-mean functional is read from the frozen scores; a mean click keeps
        // the estimator's own value, so refitting the cross-fit table would be discarded.
        let click_scores = if query_reads_score_table(&self.analysis.query) {
            click_analysis.prepare_score_table(ctx)?
        } else {
            None
        };
        overlay_prepared_score_functional(
            &self.analysis.query,
            click_scores.as_ref(),
            &mut result,
        )?;
        self.stamp(&click_analysis.data, result)
    }

    /// Sealed program routes: linear and AIPW adjustment, ADMG response
    /// curves, path-specific and functional effects, distributions,
    /// front-door, and IV.
    ///
    /// Each family of sealed routes executes from its own method so that a
    /// debug build of [`Self::estimate_with_shared`] does not hold a result
    /// slot for every route on one frame; that frame alone exceeded the
    /// default thread stack.
    fn estimate_sealed_program_operation(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Option<StudyResult>, CausalError> {
        if let Some(operation) = self.execution.linear_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let result = rebound.execute(data, ctx)?;
                return self
                    .stamp_linear(&DataInput::Tabular(data.clone()), &rebound, result)
                    .map(Some);
            }
        }
        if let Some(operation) = self.execution.admg_response_curve() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(data, ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                crate::analysis::execute::push_gaussian_likelihood_disclosure(
                    &mut result,
                    &rebound.inference,
                    &DataInput::Tabular(data.clone()),
                );
                result.executed_contract = Some(self.executed_contract(
                    &DataInput::Tabular(data.clone()),
                    rebound.refute,
                    None,
                )?);
                return Ok(Some(result));
            }
        }
        if let Some(operation) = self.execution.path_specific_effect_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(data, ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                crate::analysis::execute::push_gaussian_likelihood_disclosure(
                    &mut result,
                    &rebound.inference,
                    &DataInput::Tabular(data.clone()),
                );
                result.executed_contract = Some(self.executed_contract(
                    &DataInput::Tabular(data.clone()),
                    rebound.refute,
                    None,
                )?);
                return Ok(Some(result));
            }
        }
        if let Some(operation) = self.execution.aipw_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(data, ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                return self
                    .stamp_aipw(&DataInput::Tabular(data.clone()), &rebound, result)
                    .map(Some);
            }
        }
        if let Some(operation) = self.execution.distribution() {
            if operation.sealed_for_direct_execution() {
                let mut result = operation.execute(data, ctx)?;
                result.custom_validator_names = operation.custom_validator_names.to_vec();
                return self
                    .stamp_distribution(&DataInput::Tabular(data.clone()), operation, result)
                    .map(Some);
            }
        }
        if let Some(operation) = self.execution.functional_effect_operation() {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let mut result = rebound.execute(data, ctx)?;
                result.custom_validator_names = rebound.custom_validator_names.to_vec();
                crate::analysis::execute::push_gaussian_likelihood_disclosure(
                    &mut result,
                    &rebound.inference,
                    &DataInput::Tabular(data.clone()),
                );
                result.executed_contract = Some(self.executed_contract(
                    &DataInput::Tabular(data.clone()),
                    rebound.refute,
                    None,
                )?);
                return Ok(Some(result));
            }
        }
        if let Some(operation) = self.execution.frontdoor_linear() {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_frontdoor(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let Some(operation) = self.execution.iv() {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_iv(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        Ok(None)
    }

    /// Sealed static routes: counterfactual, tiered average, derivative and
    /// DAG responses, joint cells, mediation, attribution, and interference.
    /// See [`Self::estimate_sealed_program_operation`].
    fn estimate_sealed_static_operation(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Option<StudyResult>, CausalError> {
        if let PreparedExecution::Counterfactual(plan) = &self.execution {
            let result = plan.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::UnknownTieredAverage(operation) = &self.execution {
            let result = self.analysis.execute_checked_unknown_tiered_average(
                data,
                &operation.query,
                &operation.physical,
                ctx,
                &operation.background,
                operation.identification.clone(),
                operation.identifier,
                operation.estimator,
            )?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::DerivativeResponse(operation) = &self.execution {
            let result = self
                .analysis
                .execute_checked_derivative_response(data, &self.plan, ctx, operation)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::StaticDagResponse(operation) = &self.execution {
            let result = self
                .analysis
                .execute_checked_static_dag_response(data, &self.plan, ctx, operation)?;
            return self.stamp_with_likelihood(data, operation.inference(), result).map(Some);
        }
        if let PreparedExecution::CellAipwResponse(operation) = &self.execution {
            let rebound =
                operation.refresh(&self.analysis.graph, self.analysis.tiered.as_ref(), data)?;
            let result = self
                .analysis
                .execute_checked_cell_aipw_response(data, &self.plan, ctx, &rebound)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::StaticMediation(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::BayesianStaticMediation(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::Attribution(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::Interference(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::Randomized(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::PolicyValue(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::ContinuousDoseResponse(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::PanelDid(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::SyntheticControl(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::LocalPolynomialRatio(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::Survival(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::LongitudinalRegime(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        Ok(None)
    }

    /// Sealed estimator routes: Bayesian g-computation, basis, robust, and
    /// conditional effects, and GLM, RD, propensity, and conditional
    /// adjustment. See [`Self::estimate_sealed_program_operation`].
    fn estimate_sealed_estimator_operation(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Option<StudyResult>, CausalError> {
        if let PreparedExecution::BayesianGcomp(operation) = &self.execution {
            let mut result = operation.execute(data, ctx)?;
            result.custom_validator_names = operation.custom_validator_names();
            return self
                .stamp_with_likelihood(data, operation.operation().inference(), result)
                .map(Some);
        }
        if let PreparedExecution::BayesianBasisAte(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp_with_likelihood(data, &operation.inference(), result).map(Some);
        }
        if let PreparedExecution::BayesianBasisCate(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::BayesianRobustAte(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::BayesianConditional(operation) = &self.execution {
            let result = operation.execute(data, ctx)?;
            return self.stamp_with_likelihood(data, operation.inference(), result).map(Some);
        }
        if let PreparedExecution::CheckedGlmAdjustment(operation) = &self.execution {
            if operation.sealed_for_direct_execution() {
                let rebound = operation.rebind(data)?;
                let result = self.execute_checked_glm(data, &rebound, ctx)?;
                return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
            }
        }
        if let PreparedExecution::CheckedRd(operation) = &self.execution {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_rd(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::CheckedPropensity(operation) = &self.execution {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_propensity(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::CheckedConditional(operation) = &self.execution {
            let rebound = operation.rebind(data)?;
            let result = self.execute_checked_conditional(data, &rebound, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        Ok(None)
    }

    /// Sealed class routes: graph-posterior and class effects and responses,
    /// Bayesian specialists, and trial transport.
    /// See [`Self::estimate_sealed_program_operation`].
    fn estimate_sealed_class_operation(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Option<StudyResult>, CausalError> {
        if let PreparedExecution::GraphPosteriorEffect(operation) = &self.execution {
            let mut click_analysis = self.analysis.clone();
            click_analysis.data = DataInput::Tabular(data.clone());
            click_analysis.query = operation.target().causal_query();
            let result = if matches!(
                operation.posterior().atom_kind,
                antecedent_discovery::GraphPosteriorAtomKind::Admg
                    | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
            ) {
                click_analysis
                    .execute_checked_admg_graph_posterior(data, &self.plan, operation, ctx)?
            } else {
                click_analysis
                    .execute_checked_graph_posterior_frequentist(data, &self.plan, operation, ctx)?
            };
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::ClassGraphPosteriorEffect(operation) = &self.execution {
            let result = self
                .analysis
                .execute_checked_class_graph_posterior_effect(data, operation, &self.plan, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::BayesianGraphPosteriorAte(operation) = &self.execution {
            let result =
                self.analysis.execute_checked_bayesian_graph_posterior_ate(data, operation, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::StaticClassEffect(operation) = &self.execution {
            let mut click_analysis = self.analysis.clone();
            click_analysis.data = DataInput::Tabular(data.clone());
            click_analysis.query = CausalQuery::AverageEffect(operation.query().clone());
            click_analysis.inference = operation.inference().clone();
            click_analysis.estimator = Some(operation.procedure().id());
            click_analysis.estimator_spec = Some(operation.procedure().clone());
            click_analysis.refute = operation.validation();
            click_analysis.bootstrap_replicates = operation.bootstrap_replicates();
            click_analysis.overlap_policy = Some(operation.overlap());
            let result = match (operation.graph(), operation.identification()) {
                (StaticClassGraph::Cpdag(graph), StaticClassIdentification::Cpdag(cache)) => {
                    click_analysis.cpdag_identification_cache = Some(Arc::new(cache.clone()));
                    click_analysis.execute_cpdag(
                        data,
                        graph,
                        operation.query(),
                        operation.physical(),
                        ctx,
                    )?
                }
                (StaticClassGraph::Pag(graph), StaticClassIdentification::Pag(cache)) => {
                    click_analysis.pag_identification_cache = Some(Arc::new(cache.clone()));
                    click_analysis.execute_pag(
                        data,
                        graph,
                        operation.query(),
                        operation.physical(),
                        ctx,
                    )?
                }
                _ => {
                    return Err(CausalError::Compile {
                        message:
                            "checked static class effect graph and identification cache disagree"
                                .into(),
                    });
                }
            };
            return self.stamp_with_likelihood(data, operation.inference(), result).map(Some);
        }
        if let PreparedExecution::AdmgGraphPosteriorResponse(operation) = &self.execution {
            let result = self
                .analysis
                .execute_checked_admg_graph_posterior_response(data, operation, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::BayesianClassConditional(operation) = &self.execution {
            let result = operation.execute(&self.analysis, data, ctx)?;
            return self.stamp(&DataInput::Tabular(data.clone()), result).map(Some);
        }
        if let PreparedExecution::StaticClassResponse(operation) = &self.execution {
            let result =
                self.analysis.execute_checked_static_class_response(data, operation, ctx)?;
            return self.stamp_with_likelihood(data, operation.inference(), result).map(Some);
        }
        if let PreparedExecution::GraphPosteriorResponse(operation) = &self.execution {
            let result =
                self.analysis.execute_checked_graph_posterior_response(data, operation, ctx)?;
            return self.stamp_with_likelihood(data, operation.inference(), result).map(Some);
        }
        if matches!(
            self.execution,
            PreparedExecution::BayesianSpecialist(_) | PreparedExecution::TransportTrial(_)
        ) {
            return self.estimate_retained_single_operation(data, ctx).map(Some);
        }
        Ok(None)
    }

    /// Execute a retained Bayesian specialist or transport trial operation.
    /// Kept out of line so the shared tabular dispatcher's frame does not grow
    /// with every sealed single-operation family.
    #[inline(never)]
    fn estimate_retained_single_operation(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let result = match &self.execution {
            PreparedExecution::BayesianSpecialist(operation) => {
                operation.execute(&self.analysis, data, ctx)?
            }
            PreparedExecution::TransportTrial(operation) => {
                operation.execute(&self.analysis, data, ctx)?
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "prepared handle holds no retained single operation".into(),
                });
            }
        };
        self.stamp(&DataInput::Tabular(data.clone()), result)
    }

    /// Rebind the row-bound dispatch receipts to `data`. Boxed and out of
    /// line so the refresh frame never holds a second execution value.
    #[inline(never)]
    pub(super) fn rebound_dispatch_execution(
        &self,
        data: &DataInput,
    ) -> Result<Option<Box<PreparedExecution>>, CausalError> {
        let DataInput::Tabular(data) = data else {
            return Ok(None);
        };
        Ok(self.execution.rebind_for_dispatch(data)?.map(Box::new))
    }

    #[inline(never)]
    pub(super) fn install_execution(&mut self, execution: Box<PreparedExecution>) {
        self.execution = *execution;
    }
}
