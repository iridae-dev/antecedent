// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Admg, Arc, AverageEffectQuery, BootstrapAccounting, CausalError, CausalQuery, ClassAtomOutcome,
    ClassEnvelopeRun, ConditionalLinearAdjustment, Dag, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EnvelopeRefutation, EnvelopeRefuteAtom,
    EstimationWorkspace, EstimatorId, ExecuteRun, ExecutionContext, FunctionalDistribution,
    FunctionalDistributionWorkspace, GraphClass, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceMode, Instant, Intervention, LinearAdjustmentAte, OverlapPolicy,
    PhysicalExecutionPlan, RdWorkspace, RefutationRun, RefuteSuite, SharpRdConfig,
    SharpRdIdentifier, SharpRegressionDiscontinuity, StaticEstimateInput, StaticEstimateWorkspaces,
    Strategy, StudyResult, TableView, TabularData, bayesian_draw_count, bayesian_gcomp, elapsed_ns,
    envelope_se_omission_diagnostic, envelope_to_identification_result_for, estimate_static_effect,
    identification_from_cache_or, identification_status_ok_for_case, identify_static_query,
    identify_static_query_in_context, mix_static_envelope_se, mix_weighted_analytic_se,
    mixed_static_influence, overlap_diagnostic, overlap_limit_diagnostic, project_for_ate_estimate,
    projection_diagnostic, report_identify_compute, require_identified, scalar_identified_set,
    select_claim, select_estimand, static_aligned_influence,
};
mod conditional;
mod distribution;
mod mediation;
mod tiered;

pub(super) use distribution::distribution_interval_diagnostics;

use crate::analysis::lowering::scan::STREAMED_BATCH_ROWS;
use crate::estimator_spec::EstimatorSpec;
use crate::planner::RouteDefaults;

/// The estimator's refusal for an instrument outside the checked binary Wald slice. The
/// facade falls back to the generic 2SLS preparation on exactly this refusal; the estimate
/// crate exposes it only as a static message, so the pattern keeps one copy here.
const CHECKED_IV_NON_BINARY_INSTRUMENT: &str =
    "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional";

/// Borrowed inputs of one static ATE execution, shared by every phase.
#[derive(Clone, Copy)]
struct StaticRun<'a> {
    data: &'a TabularData,
    graph: &'a Dag,
    query: &'a AverageEffectQuery,
    physical: &'a PhysicalExecutionPlan,
    execution: &'a super::super::prepared::PreparedExecution,
    ctx: &'a ExecutionContext,
}

/// Fitters and receipts the checked static routes prepared for the point fit.
struct StaticCheckedFits {
    spec: EstimatorSpec,
    linear: Option<(LinearAdjustmentAte, antecedent_estimate::CheckedLinearAdjustmentAte)>,
    frontdoor_linear: Option<(
        antecedent_estimate::FrontDoorTwoStage,
        antecedent_estimate::CheckedFrontDoorPreparation,
    )>,
    frontdoor_functional: Option<(
        antecedent_estimate::FrontDoorFunctional,
        antecedent_estimate::CheckedFrontDoorPreparation,
    )>,
    wald: Option<(antecedent_estimate::WaldIv, antecedent_estimate::CheckedIvPreparation)>,
    two_sls: Option<(
        antecedent_estimate::TwoStageLeastSquares,
        antecedent_estimate::CheckedIvPreparation,
    )>,
}

/// Scratch the point and uncertainty stages of a static ATE share.
#[derive(Default)]
struct StaticWorkspaces {
    estimate: StaticEstimateWorkspaces,
    frontdoor: antecedent_estimate::FrontDoorWorkspace,
    iv: antecedent_estimate::TwoStageLeastSquaresWorkspace,
}

/// State a static ATE execution carries from identification to the result.
struct StaticAteExecution<'a> {
    study: &'a super::Study,
    run: StaticRun<'a>,
    identifier_id: IdentifierId,
    estimator_id: EstimatorId,
    clock: super::super::stage::StageClock,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    identify_cached: bool,
    data_est: TabularData,
    query_est: AverageEffectQuery,
    estimand_est: IdentifiedEstimand,
    full_cols: usize,
    projected_cols: usize,
}

impl<'a> StaticAteExecution<'a> {
    /// Identification stage plus projection of the table to the estimand's columns.
    fn identify(
        study: &'a super::Study,
        run: StaticRun<'a>,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
    ) -> Result<Self, CausalError> {
        let StaticRun { data, graph, query, ctx, .. } = run;
        let mut clock = super::super::stage::StageClock::new();
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query, rd), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) = identification_from_cache_or(
            ctx,
            study.reuse_static_identification(run.physical),
            || {
                let rd =
                    study.rd.map(|c| SharpRdConfig::new(c.running_variable, c.cutoff, c.bandwidth));
                let identification = identify_static_query_in_context(
                    identifier_id,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    rd,
                    Some(ctx),
                )?;
                select_claim(identification, estimator_id)
            },
        )?;
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            study.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );

        let full_cols = data.schema().len();
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            &data_est,
            query_est.outcome,
            &query_est.outcome_functional,
        )?;
        let projected_cols = data_est.schema().len();
        Ok(Self {
            study,
            run,
            identifier_id,
            estimator_id,
            clock,
            identification,
            estimand,
            identify_cached,
            data_est,
            query_est,
            estimand_est,
            full_cols,
            projected_cols,
        })
    }

    /// Opens the point stage, which refuses to start once cancellation is requested.
    fn begin_point_stage(&mut self) -> Result<(), CausalError> {
        // Point estimate first (no bootstrap); uncertainty stage fills SE separately.
        self.clock.begin(self.run.ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        if self.run.ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        Ok(())
    }

    /// Selects the estimator spec and prepares each checked receipt the point fit may use.
    fn prepare_checked(&self) -> Result<StaticCheckedFits, CausalError> {
        antecedent_estimate::prepare::with_preparation_budget(
            crate::planner::PhysicalExecutionPlan::preparation_budget(self.run.ctx),
            || self.prepare_checked_metered(),
        )
    }

    /// The body of [`Self::prepare_checked`], run under the plan's preparation budget.
    fn prepare_checked_metered(&self) -> Result<StaticCheckedFits, CausalError> {
        let StaticRun { data, execution, .. } = self.run;
        let prepared_linear = execution.linear_operation();
        let prepared_aipw = execution.aipw_operation();
        let prepared_frontdoor = execution.frontdoor_linear();
        let prepared_iv = execution.iv();
        let identification = &self.identification;
        // A prepared checked route has already selected its estimator and bound
        // its configuration. Do not let the copied Study select a competing
        // fitter while executing that retained operation.
        let spec = if prepared_linear.is_some()
            || prepared_aipw.is_some()
            || prepared_frontdoor.is_some()
            || prepared_iv.is_some()
        {
            EstimatorSpec::Default(self.estimator_id)
        } else {
            self.study
                .declared
                .estimator_spec
                .as_deref()
                .cloned()
                .unwrap_or(EstimatorSpec::Default(self.estimator_id))
        };
        // Prepare against the original semantic variable IDs. Projection remaps tabular
        // columns, while the identified expression still belongs to the original arena.
        let linear = self.prepare_checked_linear(&spec)?;
        let frontdoor_linear = if prepared_frontdoor.is_some() {
            None
        } else {
            match &spec {
                EstimatorSpec::Default(EstimatorId::FrontDoorTwoStage) => {
                    let mut fitter = antecedent_estimate::FrontDoorTwoStage::new();
                    fitter.inference.sampling_design = crate::estimator_spec::declared_sampling(
                        self.study.declared.sampling_design,
                    );
                    Some((fitter.clone(), fitter.prepare_checked(data, identification, 0)?))
                }
                EstimatorSpec::FrontDoorTwoStage(cfg) => {
                    let fitter = (**cfg).clone();
                    Some((fitter.clone(), fitter.prepare_checked(data, identification, 0)?))
                }
                _ => None,
            }
        };
        let frontdoor_functional =
            if matches!(spec, EstimatorSpec::Default(EstimatorId::FrontDoorFunctional)) {
                // Point stage only: the uncertainty stage below attaches the
                // study's replicate count, so the estimator's own default must
                // not bootstrap here.
                let mut fitter =
                    antecedent_estimate::FrontDoorFunctional::new().with_bootstrap_replicates(0);
                fitter.sampling_design =
                    crate::estimator_spec::declared_sampling(self.study.declared.sampling_design);
                Some((fitter.clone(), fitter.prepare_checked(data, identification, 0)?))
            } else {
                None
            };
        let wald = if prepared_iv.is_some() {
            None
        } else {
            match &spec {
                EstimatorSpec::Default(EstimatorId::IvWald) => {
                    let fitter = antecedent_estimate::WaldIv::new();
                    Some((fitter.clone(), fitter.prepare_checked(data, identification, 0)?))
                }
                EstimatorSpec::IvWald(cfg) => {
                    let fitter = (**cfg).clone();
                    Some((fitter.clone(), fitter.prepare_checked(data, identification, 0)?))
                }
                _ => None,
            }
        };
        let two_sls = self.prepare_checked_two_sls(&spec)?;
        Ok(StaticCheckedFits {
            spec,
            linear,
            frontdoor_linear,
            frontdoor_functional,
            wald,
            two_sls,
        })
    }

    /// Checked linear-adjustment receipt, taken from the prepared operation when one is retained.
    fn prepare_checked_linear(
        &self,
        spec: &EstimatorSpec,
    ) -> Result<
        Option<(LinearAdjustmentAte, antecedent_estimate::CheckedLinearAdjustmentAte)>,
        CausalError,
    > {
        let StaticRun { data, query, execution, physical, .. } = self.run;
        // A plan that streams its design never builds the checked design receipt.
        if physical.streams_design() {
            return Ok(None);
        }
        let prepared_linear = execution.linear_operation();
        if !(matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            && matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved))
        {
            return Ok(None);
        }
        let fitter = match spec {
            EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte) => {
                let mut fitter = LinearAdjustmentAte::new();
                fitter.inference.sampling_design =
                    crate::estimator_spec::declared_sampling(self.study.declared.sampling_design);
                Some(fitter)
            }
            EstimatorSpec::LinearAdjustmentAte(cfg) => Some((**cfg).clone()),
            _ => None,
        };
        let fitter = prepared_linear.map_or(fitter, |operation| Some(operation.fitter.clone()));
        fitter
            .map(|fitter| {
                let checked = match prepared_linear {
                    Some(operation) => {
                        if operation.preparation.source_functional() != self.estimand.functional
                            || operation.preparation.target().adjustment_set
                                != self.estimand.adjustment_set
                        {
                            return Err(CausalError::Compile {
                                message: "prepared adjustment lowering disagrees with selected identification".into(),
                            });
                        }
                        operation.preparation.clone()
                    }
                    None => fitter.prepare_checked(data, &self.identification, 0)?,
                };
                Ok::<_, CausalError>((fitter, checked))
            })
            .transpose()
    }

    /// Checked 2SLS receipt for the supported single binary-instrument slice.
    fn prepare_checked_two_sls(
        &self,
        spec: &EstimatorSpec,
    ) -> Result<
        Option<(
            antecedent_estimate::TwoStageLeastSquares,
            antecedent_estimate::CheckedIvPreparation,
        )>,
        CausalError,
    > {
        // A plan that streams its design reduces the rows in closed form and prepares nothing.
        if self.run.physical.streams_design() {
            return Ok(None);
        }
        let data = self.run.data;
        let identification = &self.identification;
        // Route the supported single binary-instrument slice through the checked IV
        // receipt. The generic 2SLS estimator still supports multiple/continuous
        // instruments and adjustment covariates; those designs remain on its ordinary
        // preparation path because the current receipt lowers the binary Wald functional.
        let checked_iv_roles_supported = identification.estimands.first().is_some_and(|target| {
            target.instruments.len() == 1
                && target.mediators.is_empty()
                && target.adjustment_set.is_empty()
        });
        if self.run.execution.iv().is_some() || !checked_iv_roles_supported {
            return Ok(None);
        }
        let fitter = match spec {
            EstimatorSpec::Default(EstimatorId::Iv2Sls) => {
                antecedent_estimate::TwoStageLeastSquares::new()
            }
            EstimatorSpec::Iv2Sls(cfg) => (**cfg).clone(),
            _ => return Ok(None),
        };
        match fitter.prepare_checked(data, identification, 0) {
            Ok(checked) => Ok(Some((fitter, checked))),
            Err(antecedent_estimate::EstimationError::Unsupported {
                message: CHECKED_IV_NON_BINARY_INSTRUMENT,
            }) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// The linear-adjustment fitter a streamed plan reads its rows through, when the plan
    /// streams its design and the selected estimator is the linear adjustment.
    fn streamed_fitter(&self, spec: &EstimatorSpec) -> Option<LinearAdjustmentAte> {
        if !self.run.physical.streams_design() {
            return None;
        }
        match spec {
            EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte) => {
                let mut fitter = LinearAdjustmentAte::new()
                    .with_bootstrap_replicates(self.study.settings.bootstrap_replicates);
                fitter.inference.sampling_design =
                    crate::estimator_spec::declared_sampling(self.study.declared.sampling_design);
                Some(fitter)
            }
            EstimatorSpec::LinearAdjustmentAte(config) => Some((**config).clone()),
            _ => None,
        }
    }

    /// The streamed reduction: the rows are read in bounded batches through the
    /// complete-case mask, and no design is built.
    fn fit_streamed(&self, fitter: &LinearAdjustmentAte) -> Result<EffectEstimate, CausalError> {
        let ctx = self.run.ctx;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "estimate_point" });
        }
        let ceiling = ctx.memory_limit_bytes().unwrap_or(u64::MAX / 4);
        let streamed = fitter.estimate_streamed(
            &self.data_est,
            &self.estimand_est,
            &self.query_est,
            STREAMED_BATCH_ROWS,
            ceiling,
        )?;
        Ok(EffectEstimate::new(
            streamed.ate,
            streamed.se_analytic,
            self.identification.required_assumptions.clone(),
            self.study.declared.overlap_policy.unwrap_or(OverlapPolicy::ExplicitOverride),
        )
        .with_n_obs(streamed.n_obs))
    }

    /// Point fit through the retained operation, the checked receipt, or the generic estimator.
    fn fit_point(
        &mut self,
        checked: &StaticCheckedFits,
        ws: &mut StaticWorkspaces,
    ) -> Result<EffectEstimate, CausalError> {
        let StaticRun { query, execution, ctx, .. } = self.run;
        let prepared_linear = execution.linear_operation();
        let prepared_aipw = execution.aipw_operation();
        let prepared_frontdoor = execution.frontdoor_linear();
        let prepared_iv = execution.iv();
        let streamed_nuisance = self.fit_streamed_nuisance(checked)?;
        let mut point = if let Some(operation) = prepared_iv {
            self.fit_prepared_iv(operation, &mut ws.iv)?
        } else if let Some((fitter, receipt)) = &checked.linear {
            if prepared_linear.is_some_and(|operation| operation.default_id)
                || (prepared_linear.is_none()
                    && matches!(
                        checked.spec,
                        EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte)
                    ))
            {
                // The progressive default route reports a genuine point stage.
                // Its bootstrap belongs to the uncertainty stage below, even
                // though the checked receipt retains the same bound design.
                fitter
                    .fit_point(
                        receipt.problem(),
                        &mut ws.estimate.linear,
                        receipt.required_assumptions().clone(),
                    )
                    .map_err(CausalError::from)?
            } else {
                fitter
                    .fit_checked(receipt, &mut ws.estimate.linear, ctx)
                    .map_err(CausalError::from)?
            }
        } else if let Some(estimate) = streamed_nuisance {
            estimate
        } else if let Some(fitter) = self.streamed_fitter(&checked.spec) {
            self.fit_streamed(&fitter)?
        } else if let Some(operation) = prepared_aipw {
            if operation.preparation.target().functional != self.estimand.functional
                || operation.preparation.target().adjustment_set != self.estimand.adjustment_set
            {
                return Err(CausalError::Compile {
                    message: "prepared AIPW lowering disagrees with selected identification".into(),
                });
            }
            operation
                .fitter
                .fit_checked(&operation.preparation, &mut ws.estimate.aipw, ctx)
                .map_err(CausalError::from)?
        } else if let Some(operation) = prepared_frontdoor {
            self.check_prepared_frontdoor_lowering(operation)?;
            operation
                .fitter
                .fit_checked(&operation.preparation, &mut ws.frontdoor, ctx)
                .map_err(CausalError::from)?
        } else if let Some((fitter, receipt)) = &checked.frontdoor_linear {
            fitter.fit_checked(receipt, &mut ws.frontdoor, ctx).map_err(CausalError::from)?
        } else if let Some((fitter, receipt)) = &checked.frontdoor_functional {
            fitter.fit_checked(receipt, ctx).map_err(CausalError::from)?
        } else if let Some((fitter, receipt)) = &checked.wald {
            fitter.fit_checked(receipt, ctx).map_err(CausalError::from)?
        } else if let Some((fitter, receipt)) = &checked.two_sls {
            fitter.fit_checked(receipt, &mut ws.iv, ctx).map_err(CausalError::from)?
        } else {
            estimate_static_effect(
                &checked.spec,
                StaticEstimateInput {
                    data: &self.data_est,
                    estimand: &self.estimand_est,
                    query: &self.query_est,
                    // Point stage: no bootstrap.
                    bootstrap_replicates: 0,
                    overlap_policy: self.study.declared.overlap_policy,
                    population_registry: self.study.declared.population_registry.as_ref(),
                    sampling_design: self.study.declared.sampling_design,
                },
                self.identification.required_assumptions.clone(),
                ctx,
                &mut ws.estimate,
            )?
        };
        // The estimator ran on the projected table, whose columns are renumbered: a score
        // table it kept names the projected treatment and adjustment ids. Restore the original
        // semantic ids (projection keeps the adjustment order) so the table agrees with the
        // identified estimand and with the prepare-time table of the same plan.
        if let Some(table) = point.scores.table.as_mut() {
            table.treatment = query.treatment;
            table.adjustment_set = Arc::clone(&self.estimand.adjustment_set);
        }
        // The same holds for a portable fitted effect: its feature ids are columns of the
        // projected table. Projection keeps the order of `columns_for_ate_estimand`, so
        // projected id `k` is the original id at position `k`; restore the original ids, which
        // are the ones the artifact's header names and every consumer resolve.
        let kept = super::super::helpers::columns_for_ate_estimand(query, &self.estimand);
        if kept.len() != self.full_cols
            && let Some(model) = point.heterogeneity.fitted_effect.as_mut()
        {
            let original: Option<Vec<u32>> = model
                .features
                .iter()
                .map(|&feature| kept.get(usize::try_from(feature).ok()?).map(|id| id.raw()))
                .collect();
            if let Some(original) = original {
                Arc::make_mut(model).features = original;
            }
        }
        self.clock.finish(super::super::stage::STAGE_ESTIMATE_POINT);
        super::super::stage::emit_stage(
            self.study.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Point { estimate: point.clone() },
        );
        Ok(point)
    }

    /// Streamed nuisance routes reduce rows before the retained point-fit dispatch.
    fn fit_streamed_nuisance(
        &self,
        checked: &StaticCheckedFits,
    ) -> Result<Option<EffectEstimate>, CausalError> {
        let ctx = self.run.ctx;
        Ok(
            if self.run.physical.streams_design()
                && !matches!(self.estimator_id, EstimatorId::LinearAdjustmentAte)
            {
                crate::strategy_table::estimate_static_effect_streamed(
                    &checked.spec,
                    self.estimator_id,
                    StaticEstimateInput {
                        data: &self.data_est,
                        estimand: &self.estimand_est,
                        query: &self.query_est,
                        // Point stage: no bootstrap.
                        bootstrap_replicates: 0,
                        overlap_policy: self.study.declared.overlap_policy,
                        population_registry: self.study.declared.population_registry.as_ref(),
                        sampling_design: self.study.declared.sampling_design,
                    },
                    self.identification.required_assumptions.clone(),
                    ctx.rng.master_seed(),
                    STREAMED_BATCH_ROWS,
                    ctx.memory_limit_bytes().unwrap_or(u64::MAX / 4),
                )?
            } else {
                None
            },
        )
    }

    /// Point fit of a retained checked IV operation after its lowering is matched to the query.
    #[expect(
        clippy::float_cmp,
        reason = "IV query intervention values are exact discrete declared values"
    )]
    fn fit_prepared_iv(
        &self,
        operation: &super::super::prepared::CheckedIvOperation,
        iv_workspace: &mut antecedent_estimate::TwoStageLeastSquaresWorkspace,
    ) -> Result<EffectEstimate, CausalError> {
        let StaticRun { query, ctx, .. } = self.run;
        let (preparation, procedure) = match operation {
            super::super::prepared::CheckedIvOperation::Wald { preparation, .. } => {
                (preparation, "Wald")
            }
            super::super::prepared::CheckedIvOperation::TwoSls { preparation, .. } => {
                (preparation, "2SLS")
            }
        };
        let lowering = preparation.lowering();
        let query_active = match &query.active {
            antecedent_core::Intervention::Set { variable, value }
                if *variable == query.treatment =>
            {
                value.as_f64()
            }
            _ => None,
        };
        let query_control = match &query.control {
            antecedent_core::Intervention::Set { variable, value }
                if *variable == query.treatment =>
            {
                value.as_f64()
            }
            _ => None,
        };
        if preparation.target().functional != self.estimand.functional
            || lowering.treatment != query.treatment
            || lowering.outcome != query.outcome
            || lowering.active != query_active.unwrap_or(f64::NAN)
            || lowering.control != query_control.unwrap_or(f64::NAN)
        {
            return Err(CausalError::Compile {
                message: format!(
                    "prepared IV {procedure} lowering disagrees with selected identification or query"
                ),
            });
        }
        Ok(match operation {
            super::super::prepared::CheckedIvOperation::Wald { fitter, preparation } => {
                fitter.fit_checked(preparation, ctx).map_err(CausalError::from)?
            }
            super::super::prepared::CheckedIvOperation::TwoSls { fitter, preparation } => {
                fitter.fit_checked(preparation, iv_workspace, ctx).map_err(CausalError::from)?
            }
        })
    }

    /// Refuses a retained front-door operation whose lowering disagrees with the query.
    fn check_prepared_frontdoor_lowering(
        &self,
        operation: &super::super::prepared::CheckedFrontDoorOperation,
    ) -> Result<(), CausalError> {
        let query = self.run.query;
        if operation.preparation.target().functional != self.estimand.functional
            || operation.preparation.lowering().treatment != query.treatment
            || operation.preparation.lowering().outcome != query.outcome
            || operation.preparation.lowering().population != query.target_population
            || !matches!(
                &query.active,
                antecedent_core::Intervention::Set { variable, value }
                    if *variable == query.treatment
                        && value.as_f64() == Some(operation.preparation.lowering().active)
            )
            || !matches!(
                &query.control,
                antecedent_core::Intervention::Set { variable, value }
                    if *variable == query.treatment
                        && value.as_f64() == Some(operation.preparation.lowering().control)
            )
        {
            return Err(CausalError::Compile {
                message:
                    "prepared front-door lowering disagrees with selected identification or query"
                        .into(),
            });
        }
        Ok(())
    }

    /// Uncertainty stage: fills the bootstrap onto the point estimate where the route allows it.
    fn attach_uncertainty(
        &mut self,
        checked: &StaticCheckedFits,
        ws: &mut StaticWorkspaces,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, CausalError> {
        let estimator_id = self.estimator_id;
        if self.skips_bootstrap_refill() {
            self.keep_point_as_uncertainty(point)
        } else if let (EstimatorId::FrontDoorTwoStage, Some((_, receipt))) =
            (estimator_id, &checked.frontdoor_linear)
        {
            self.attach_frontdoor_linear_bootstrap(receipt, ws, point)
        } else if let (EstimatorId::FrontDoorFunctional, Some((_, receipt))) =
            (estimator_id, &checked.frontdoor_functional)
        {
            self.attach_frontdoor_functional_bootstrap(receipt, point)
        } else if matches!(estimator_id, EstimatorId::LinearAdjustmentAte) {
            self.attach_linear_bootstrap(checked, ws, point)
        } else {
            self.attach_generic_bootstrap(ws, point)
        }
    }

    /// Whether the facade bootstrap must not refill this route's point estimate.
    fn skips_bootstrap_refill(&self) -> bool {
        let study = self.study;
        let execution = self.run.execution;
        let prepared_linear = execution.linear_operation();
        // Uncertainty: bootstrap fills (real work when replicates > 0).
        // IV and NN matching must not refill with the facade bootstrap: that
        // bypasses the weak-instrument gate (Wald/2SLS) and publishes an
        // Abadie–Imbens-invalid matching bootstrap SE. A configured estimator owns its
        // replicate count and already ran it in the point fit, and the double-ML, DR-learner
        // and causal-forest fits take no replicate count: refitting any of them here would
        // reproduce the same estimate at the cost of a second full nuisance fit.
        execution.aipw_operation().is_some()
            || execution.frontdoor_linear().is_some()
            || prepared_linear.is_some_and(|operation| !operation.default_id)
            || prepared_linear
                .is_some_and(|operation| operation.fitter.inference.bootstrap_replicates == 0)
            || (prepared_linear.is_none() && study.settings.bootstrap_replicates == 0)
            || (prepared_linear.is_none()
                && study
                    .declared
                    .estimator_spec
                    .as_deref()
                    .is_some_and(|spec| !matches!(spec, EstimatorSpec::Default(_))))
            || matches!(
                self.estimator_id,
                EstimatorId::IvWald
                    | EstimatorId::Iv2Sls
                    | EstimatorId::PropensityMatching
                    | EstimatorId::DistanceMatching
                    | EstimatorId::Dml
                    | EstimatorId::DrLearner
                    | EstimatorId::CausalForest
            )
    }

    /// Emits the uncertainty stage for an estimate that already carries its final SE.
    fn keep_point_as_uncertainty(
        &mut self,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, CausalError> {
        let ctx = self.run.ctx;
        if ctx.cancellation.is_cancelled() {
            self.clock.mark_cancelled();
            return Ok(point);
        }
        self.clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        self.clock.finish(super::super::stage::STAGE_UNCERTAINTY);
        super::super::stage::emit_uncertainty(self.study.stage_sink.as_ref(), &point);
        Ok(point)
    }

    /// Bootstraps through the checked front-door two-stage receipt.
    fn attach_frontdoor_linear_bootstrap(
        &mut self,
        receipt: &antecedent_estimate::CheckedFrontDoorPreparation,
        ws: &mut StaticWorkspaces,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, CausalError> {
        let ctx = self.run.ctx;
        self.clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        let mut fitter = antecedent_estimate::FrontDoorTwoStage::new();
        fitter.inference.sampling_design =
            crate::estimator_spec::declared_sampling(self.study.declared.sampling_design);
        fitter.inference.bootstrap_replicates = self.study.settings.bootstrap_replicates;
        let filled = fitter
            .attach_bootstrap(receipt.problem(), &mut ws.frontdoor, ctx, point)
            .map_err(CausalError::from)?;
        self.clock.finish(super::super::stage::STAGE_UNCERTAINTY);
        super::super::stage::emit_uncertainty(self.study.stage_sink.as_ref(), &filled);
        Ok(filled)
    }

    /// Bootstraps through the checked functional front-door receipt.
    fn attach_frontdoor_functional_bootstrap(
        &mut self,
        receipt: &antecedent_estimate::CheckedFrontDoorPreparation,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, CausalError> {
        let ctx = self.run.ctx;
        self.clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        let mut fitter = antecedent_estimate::FrontDoorFunctional::new()
            .with_bootstrap_replicates(self.study.settings.bootstrap_replicates);
        fitter.sampling_design =
            crate::estimator_spec::declared_sampling(self.study.declared.sampling_design);
        let filled =
            fitter.attach_bootstrap(receipt.problem(), ctx, point).map_err(CausalError::from)?;
        self.clock.finish(super::super::stage::STAGE_UNCERTAINTY);
        super::super::stage::emit_uncertainty(self.study.stage_sink.as_ref(), &filled);
        Ok(filled)
    }

    /// Re-prepares the linear adjustment on the warmed OLS workspace and attaches its bootstrap.
    fn attach_linear_bootstrap(
        &mut self,
        checked: &StaticCheckedFits,
        ws: &mut StaticWorkspaces,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, CausalError> {
        let ctx = self.run.ctx;
        let study = self.study;
        let prepared_linear = self.run.execution.linear_operation();
        // Reuse warmed OLS workspace: re-prepare + attach bootstrap without refitting point.
        let cancelled_before = ctx.cancellation.is_cancelled();
        if cancelled_before {
            self.clock.mark_cancelled();
            if let Some(p) = &ctx.progress {
                p.report(0.55, super::super::stage::STAGE_UNCERTAINTY);
            }
            return Ok(point);
        }
        self.clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        // A configured linear estimator never reaches here (it bootstrapped in the
        // point fit), so this is the id-selected default the point stage also used.
        let mut est = prepared_linear
            .map_or_else(LinearAdjustmentAte::new, |operation| operation.fitter.clone());
        if prepared_linear.is_none() {
            est.inference.sampling_design =
                crate::estimator_spec::declared_sampling(study.declared.sampling_design);
            est.inference.bootstrap_replicates = study.settings.bootstrap_replicates;
            est.overlap = OverlapPolicy::ExplicitOverride;
        }
        let prep = if let Some((_, receipt)) = &checked.linear {
            receipt.problem().clone()
        } else {
            est.prepare(&self.data_est, &self.estimand_est, &self.query_est)
                .map_err(CausalError::from)?
        };
        let filled = est
            .attach_bootstrap(&prep, &mut ws.estimate.linear, ctx, point)
            .map_err(CausalError::from)?;
        self.finish_filled_uncertainty(&filled);
        Ok(filled)
    }

    /// Attaches the strategy-table bootstrap for weighting, stratification, AIPW, GLM and front-door.
    fn attach_generic_bootstrap(
        &mut self,
        ws: &mut StaticWorkspaces,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, CausalError> {
        let ctx = self.run.ctx;
        let study = self.study;
        // Weighting, stratification, AIPW, GLM and front-door: their `fit` is the point fit
        // plus `attach_bootstrap`, so the uncertainty stage reuses the point estimate.
        let cancelled_before = ctx.cancellation.is_cancelled();
        if cancelled_before {
            self.clock.mark_cancelled();
            if let Some(p) = &ctx.progress {
                p.report(0.55, super::super::stage::STAGE_UNCERTAINTY);
            }
            return Ok(point);
        }
        self.clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        let filled = crate::strategy_table::attach_static_bootstrap(
            self.estimator_id,
            StaticEstimateInput {
                data: &self.data_est,
                estimand: &self.estimand_est,
                query: &self.query_est,
                bootstrap_replicates: study.settings.bootstrap_replicates,
                overlap_policy: study.declared.overlap_policy,
                population_registry: study.declared.population_registry.as_ref(),
                sampling_design: study.declared.sampling_design,
            },
            point,
            ctx,
            &mut ws.estimate,
        )?;
        self.finish_filled_uncertainty(&filled);
        Ok(filled)
    }

    /// Closes the uncertainty stage of a bootstrap-filled estimate and emits it.
    fn finish_filled_uncertainty(&mut self, filled: &EffectEstimate) {
        let cancelled =
            filled.uncertainty.bootstrap.cancelled || self.run.ctx.cancellation.is_cancelled();
        if cancelled {
            self.clock.mark_cancelled();
        } else {
            self.clock.finish(super::super::stage::STAGE_UNCERTAINTY);
        }
        super::super::stage::emit_uncertainty(self.study.stage_sink.as_ref(), filled);
    }

    /// Validation stage: runs the refutation suite unless the estimate was cancelled.
    fn validate(
        &mut self,
        ws: &mut StaticWorkspaces,
        estimate: &EffectEstimate,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let ctx = self.run.ctx;
        let cancelled = estimate.uncertainty.bootstrap.cancelled || self.clock.cancelled();
        if cancelled {
            return Ok((Vec::new(), Vec::new()));
        }
        self.clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let prop_scratch = match self.estimator_id {
            EstimatorId::Aipw => &mut ws.estimate.aipw.propensity,
            _ => &mut ws.estimate.propensity.propensity,
        };
        let (reports, na_diagnostics) = RefutationRun {
            data: &self.data_est,
            estimand: &self.estimand_est,
            query: &self.query_est,
            estimate,
            suite: self.study.settings.refute,
            estimator: self.estimator_id.as_str(),
            custom: &self.study.declared.custom_validators,
            temporal: None,
        }
        .reports(&mut ws.estimate.linear, Some(prop_scratch), ctx)?;
        self.clock.finish(super::super::stage::STAGE_VALIDATE);
        super::super::stage::emit_stage(
            self.study.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Validate {
                refutations: reports.clone(),
                predictive_checks: Vec::new(),
            },
        );
        Ok((reports, na_diagnostics))
    }

    /// Attaches the functional grid and assembles the identified result.
    fn finish(
        self,
        estimate: EffectEstimate,
        refutations: Vec<antecedent_validate::RefutationReport>,
        na_diagnostics: Vec<Diagnostic>,
    ) -> Result<StudyResult, CausalError> {
        let StaticRun { data, query, physical, ctx, .. } = self.run;
        let mut extra_diagnostics =
            if let Some(d) = projection_diagnostic(self.full_cols, self.projected_cols) {
                vec![d]
            } else {
                Vec::new()
            };
        extra_diagnostics.extend(na_diagnostics);
        let estimate = super::super::helpers::attach_average_functional_grid(
            estimate,
            super::super::helpers::AverageTarget { data, query, estimand: &self.estimand },
            &mut extra_diagnostics,
            self.estimator_id,
            self.study,
            ctx.rng.master_seed(),
        )?;
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;
        Ok(self.study.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                self.identification,
                self.estimand,
                estimate,
                Strategy::new(self.identifier_id, self.estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(self.identify_cached)
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(self.clock.wall_time_ns())
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(self.clock.cancelled())
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                stage_timings_ns: self.clock.timings(),
                ..Default::default()
            }),
        ))
    }
}

/// The retained background and two-scenario identification of a sealed Unknown-tier
/// average.
pub(crate) struct CheckedUnknownTiered<'a> {
    pub(crate) background: &'a antecedent_graph::TieredBackground,
    pub(crate) identification: IdentificationResult,
}

impl super::Study {
    pub(super) fn execute_static(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        execution: &super::super::prepared::PreparedExecution,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let run = StaticRun { data, graph, query, physical, execution, ctx };
        let (identifier_id, estimator_id) =
            physical.binding().resolve(RouteDefaults::STATIC_ATE)?;
        if let Some(result) = self.execute_static_specialist(&run, estimator_id) {
            return result;
        }

        let mut exec = StaticAteExecution::identify(self, run, identifier_id, estimator_id)?;
        exec.begin_point_stage()?;
        let mut workspaces = StaticWorkspaces::default();
        let checked = exec.prepare_checked()?;
        let point = exec.fit_point(&checked, &mut workspaces)?;
        let estimate = exec.attach_uncertainty(&checked, &mut workspaces, point)?;
        let (refutations, na_diagnostics) = exec.validate(&mut workspaces, &estimate)?;
        exec.finish(estimate, refutations, na_diagnostics)
    }

    /// Routes estimators that own a dedicated execution path (rd, Bayesian, functional).
    ///
    /// The tiered background never reaches here: the dispatcher routes it to
    /// `execute_tiered_average` before choosing a static graph class.
    fn execute_static_specialist(
        &self,
        run: &StaticRun<'_>,
        estimator_id: EstimatorId,
    ) -> Option<Result<StudyResult, CausalError>> {
        let StaticRun { data, graph, query, physical, execution, ctx } = *run;
        // rd.sharp identifies from the declared design, which the graph must agree
        // with; it has no adjustment search, so it takes its own path.
        if matches!(estimator_id, EstimatorId::RdSharp) {
            return Some(self.execute_rd(data, graph, query, physical, ctx));
        }
        if matches!(estimator_id, EstimatorId::BayesianIvJointLinear) {
            return Some(self.execute_bayesian_iv(data, graph, query, physical, ctx));
        }
        if matches!(estimator_id, EstimatorId::BayesianRdLocalLinear) {
            return Some(self.execute_bayesian_rd(data, graph, query, physical, ctx));
        }
        if matches!(estimator_id, EstimatorId::BayesianRandomizedQuantile) {
            return Some(self.execute_bayesian_quantile(data, graph, query, physical, ctx));
        }
        if matches!(estimator_id, EstimatorId::RandomizedQuantile) {
            return Some(self.execute_randomized_quantile(data, graph, query, physical, ctx));
        }
        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return Some(self.execute_bayesian(
                data,
                graph,
                query,
                physical,
                execution.bayesian_gcomp(),
                ctx,
            ));
        }
        if matches!(estimator_id, EstimatorId::BayesianBasisGcomp) {
            return Some(self.execute_bayesian_basis(data, graph, query, physical, ctx));
        }
        if matches!(estimator_id, EstimatorId::BayesianRobustAte) {
            return Some(self.execute_bayesian_robust_ate(data, graph, query, physical, ctx));
        }
        if matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Some(self.execute_functional_ate(data, graph, query, physical, ctx));
        }
        None
    }

    /// Identify + plug-in estimate for a path-specific natural effect.
    pub(super) fn execute_path_specific(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::PathSpecificEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "PathSpecific execute requires estimator functional.effect; got {estimator_id}"
                ),
            });
        }

        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query), all frozen there, so reuse is exact
        // and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                let cq = CausalQuery::PathSpecific(query.clone());
                let identification = crate::strategy_table::identify_static_query_in_context(
                    identifier_id,
                    graph,
                    &cq,
                    None,
                    Some(ctx),
                )?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;

        let mut extra = vec![query.treatment, query.outcome];
        extra.extend(query.path_nodes.iter().copied());
        let (estimate, posterior) =
            self.estimate_functional_effect(data, &estimand, &identification, &extra, ctx)?;

        let refutations = if self.settings.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_path(
                data,
                query,
                &identification,
                &estimand,
                estimate.ate,
                self.settings.refute == RefuteSuite::Full,
                ctx,
            )
            .map_err(CausalError::from)?
        };

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    /// Identify and plug-in estimate a path-restricted natural effect, or a natural direct
    /// effect, on a supplied ADMG.
    ///
    /// The path-specific identifier reads the graph's latent confounding: the surgical total
    /// effect when every directed path is selected, the edge g-formula when the outcome's
    /// ancestors are unconfounded, and Shpitser's district formula otherwise. The identified
    /// functional is evaluated by the same finite-discrete plug-in (frequentist bootstrap) or
    /// Bayesian-bootstrap posterior as the DAG cell.
    pub(super) fn execute_path_specific_mixed(
        &self,
        data: &TabularData,
        admg: &Admg,
        source: super::PathSpecificMixedSource<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_path_specific_mixed_prepared(data, admg, source, physical, ctx, None)
    }

    /// Execute with the immutable checked empirical program a retained click rebound.
    pub(super) fn execute_path_specific_mixed_prepared(
        &self,
        data: &TabularData,
        admg: &Admg,
        source: super::PathSpecificMixedSource<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        retained: Option<(
            &antecedent_estimate::FunctionalEffect,
            &antecedent_estimate::PreparedFunctionalEffect,
        )>,
    ) -> Result<StudyResult, CausalError> {
        ensure_admg_path_specific_supported(self.settings.refute)?;
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::PATH)?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "PathSpecific execute requires estimator functional.effect; got {estimator_id}"
                ),
            });
        }
        let (identified_query, treatment, outcome, roles) = path_specific_source_parts(source);

        // Prepared handles identify once at prepare time; identification reads only
        // (identifier, graph, query), all frozen there, so reuse is exact and observable via
        // the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                let mut identification = crate::strategy_table::identify_admg_query(
                    identifier_id,
                    admg,
                    &identified_query,
                )?;
                // The natural direct effect is identified as its mediation reading but
                // answers the nested query the study asked.
                identification.query = self.query.clone();
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;

        let (estimate, posterior) = match retained {
            Some((fitter, prepared)) => super::run_functional(
                fitter,
                prepared,
                &self.settings.inference,
                identification.status,
                ctx,
            )?,
            None => {
                self.estimate_functional_effect(data, &estimand, &identification, &roles, ctx)?
            }
        };
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    pub(super) fn execute_functional_ate(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = physical.binding().resolve(RouteDefaults::ADMG)?;
        // Compile retains the static claim preparation resolves; it is this route's
        // claim when the strategy matches, no design is declared and selecting the
        // estimand did not narrow it.
        let reused = self.reuse_static_identification_if(physical, |cache| {
            self.rd.is_none()
                && cache.identification.estimand_claims.is_empty()
                && physical.binding().resolves_to(
                    RouteDefaults::STATIC_ATE,
                    identifier_id,
                    estimator_id,
                )
        });
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, reused, || {
                let identification = crate::strategy_table::identify_static_query_in_context(
                    identifier_id,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    None,
                    Some(ctx),
                )?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;
        let (estimate, posterior) = self.estimate_functional_effect(
            data,
            &estimand,
            &identification,
            &[query.treatment, query.outcome],
            ctx,
        )?;
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra_diagnostics) = RefutationRun {
            data,
            estimand: &estimand,
            query,
            estimate: &estimate,
            suite: self.settings.refute,
            estimator: estimator_id.as_str(),
            custom: &self.declared.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    /// Bayesian g-computation execute path.
    pub(super) fn execute_rd(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let rd = self.rd.ok_or_else(|| CausalError::Compile {
            message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
        })?;
        // The design identifies once, at compile, and the program retains it; only a plan
        // that retained none (a design that does not identify) identifies here and reports
        // the refusal.
        let target = CausalQuery::AverageEffect(query.clone());
        let reused = self
            .reuse_static_identification_if(physical, |cache| cache.identification.query == target);
        let (identification, estimand, identify_cached) = if let Some((cache, prepared)) = reused {
            (cache.identification.clone(), cache.estimand.clone(), prepared)
        } else {
            report_identify_compute(ctx);
            let identification = SharpRdIdentifier::new(SharpRdConfig::new(
                rd.running_variable,
                rd.cutoff,
                rd.bandwidth,
            ))
            .identify_on(graph, target)
            .map_err(CausalError::from)?;
            if matches!(identification.status, IdentificationStatus::NotIdentified) {
                // Say why the design does not identify: the graph contradicts it, or the
                // requested population is not the one at the cutoff.
                let detail = identification
                    .diagnostics
                    .iter()
                    .find(|d| d.kind == antecedent_core::DiagnosticKind::Scientific)
                    .map_or_else(|| "sharp RD design".to_string(), |d| d.message.to_string());
                return Err(CausalError::not_identified(identification.status, false, &detail));
            }
            require_identified(&identification)?;
            let estimand = select_estimand(&identification, EstimatorId::RdSharp)?;
            (identification, estimand, false)
        };
        // The identified query names the population the design speaks for (units at the
        // cutoff); estimation, refutation and the result all use that query.
        let identified_query =
            identification.average_effect().cloned().unwrap_or_else(|| query.clone());
        let query = &identified_query;

        let mut est =
            SharpRegressionDiscontinuity::new(rd.running_variable, rd.cutoff, rd.bandwidth);
        est.inference.bootstrap_replicates = self.settings.bootstrap_replicates;
        est.inference.se_kind = rd.se_kind;
        let checked = est.prepare_checked(data, &identification, 0).map_err(CausalError::from)?;
        let mut ws = RdWorkspace::default();
        let estimate = est.fit_checked(&checked, &mut ws, ctx).map_err(CausalError::from)?;

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra_diagnostics) = RefutationRun {
            data,
            estimand: &estimand,
            query,
            estimate: &estimate,
            suite: self.settings.refute,
            estimator: "rd.sharp",
            custom: &self.declared.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::RdSharp, EstimatorId::RdSharp),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started)),
        ))
    }
}

/// Graph an interventional distribution is identified on.
#[derive(Clone, Copy)]
pub(crate) enum DistributionGraph<'a> {
    /// Supplied DAG: static identification by the selected identifier.
    Dag(&'a Dag),
    /// Supplied ADMG: general ID over the bidirected structure.
    Admg(&'a Admg),
}

impl DistributionGraph<'_> {
    /// Identify `query` on this graph with `identifier`.
    pub(crate) fn identify(
        self,
        identifier: IdentifierId,
        query: &antecedent_core::InterventionalDistributionQuery,
    ) -> Result<IdentificationResult, CausalError> {
        let cq = CausalQuery::Distribution(query.clone());
        match self {
            Self::Dag(graph) => identify_static_query(identifier, graph, &cq),
            Self::Admg(admg) => crate::strategy_table::identify_admg_query(identifier, admg, &cq),
        }
    }
}

/// The query a natural-effect source identifies (a natural direct effect as its mediation
/// reading), its treatment and outcome, and the variables the plug-in reads.
pub(super) fn path_specific_source_parts(
    source: super::PathSpecificMixedSource<'_>,
) -> (
    CausalQuery,
    antecedent_core::VariableId,
    antecedent_core::VariableId,
    Vec<antecedent_core::VariableId>,
) {
    match source {
        super::PathSpecificMixedSource::PathSpecific(query) => {
            let mut roles = vec![query.treatment, query.outcome];
            roles.extend(query.path_nodes.iter().copied());
            (CausalQuery::PathSpecific(query.clone()), query.treatment, query.outcome, roles)
        }
        super::PathSpecificMixedSource::Mediation(query) => {
            let mut roles = vec![query.treatment, query.outcome];
            roles.extend(query.mediators.iter().copied());
            (CausalQuery::Mediation(query.clone()), query.treatment, query.outcome, roles)
        }
        super::PathSpecificMixedSource::Nested(query) => (
            CausalQuery::Mediation(query.as_mediation_query()),
            query.treatment,
            query.outcome,
            vec![query.treatment, query.outcome, query.mediator],
        ),
    }
}

/// Gate for a path-specific effect on an ADMG, shared by compile and execute: the plug-in
/// reads the identified functional on the table. Cheap and full validation are the functional
/// refuter state, which runs the study at validation none and refits it; the route itself
/// never carries a scalar suite.
#[allow(
    clippy::unnecessary_wraps,
    reason = "the gate keeps the shape of the other support gates its callers share"
)]
pub(super) fn ensure_admg_path_specific_supported(refute: RefuteSuite) -> Result<(), CausalError> {
    let _ = refute;
    Ok(())
}

/// Support gate for an ADMG interventional distribution, shared by compile and
/// execute: unconditional finite-discrete tables at validation none.
pub(super) fn ensure_admg_distribution_supported(
    query: &antecedent_core::InterventionalDistributionQuery,
    refute: RefuteSuite,
) -> Result<(), CausalError> {
    if !query.conditioning.is_empty() {
        return Err(CausalError::Unsupported {
            message: "ADMG InterventionalDistribution is supported for unconditional \
                      finite-discrete tables; IDC conditionals are a follow-up",
        });
    }
    // Cheap and full validation are the functional refuter state (see the path-specific gate).
    let _ = refute;
    Ok(())
}
