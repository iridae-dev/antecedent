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
    identification_from_cache_or, identification_status_ok_for_case, identify_cpdag, identify_pag,
    identify_static, identify_static_query, identify_static_query_with_rd, mix_static_envelope_se,
    mix_weighted_analytic_se, mixed_static_influence, overlap_diagnostic, project_for_ate_estimate,
    projection_diagnostic, report_identify_compute, require_identified, scalar_identified_set,
    select_claim, select_estimand, static_aligned_influence,
};
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
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, study.identification_cache.as_deref(), || {
                let rd =
                    study.rd.map(|c| SharpRdConfig::new(c.running_variable, c.cutoff, c.bandwidth));
                let identification = identify_static_query_with_rd(
                    identifier_id,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    rd,
                )?;
                select_claim(identification, estimator_id)
            })?;
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
                    fitter.sampling_design =
                        crate::estimator_spec::declared_sampling(self.study.sampling_design);
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
                let fitter =
                    antecedent_estimate::FrontDoorFunctional::new().with_bootstrap_replicates(0);
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
        let StaticRun { data, query, execution, .. } = self.run;
        let prepared_linear = execution.linear_operation();
        if !(matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            && matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved))
        {
            return Ok(None);
        }
        let fitter = match spec {
            EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte) => {
                let mut fitter = LinearAdjustmentAte::new();
                fitter.sampling_design =
                    crate::estimator_spec::declared_sampling(self.study.sampling_design);
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
                    overlap_policy: self.study.overlap_policy,
                    population_registry: self.study.population_registry.as_ref(),
                    sampling_design: self.study.sampling_design,
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
        self.clock.finish(super::super::stage::STAGE_ESTIMATE_POINT);
        super::super::stage::emit_stage(
            self.study.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Point { estimate: point.clone() },
        );
        Ok(point)
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
            || prepared_linear.is_some_and(|operation| operation.fitter.bootstrap_replicates == 0)
            || (prepared_linear.is_none() && study.bootstrap_replicates == 0)
            || (prepared_linear.is_none()
                && study
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
        fitter.sampling_design =
            crate::estimator_spec::declared_sampling(self.study.sampling_design);
        fitter.bootstrap_replicates = self.study.bootstrap_replicates;
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
        let fitter = antecedent_estimate::FrontDoorFunctional::new()
            .with_bootstrap_replicates(self.study.bootstrap_replicates);
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
            est.sampling_design = crate::estimator_spec::declared_sampling(study.sampling_design);
            est.bootstrap_replicates = study.bootstrap_replicates;
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
                bootstrap_replicates: study.bootstrap_replicates,
                overlap_policy: study.overlap_policy,
                population_registry: study.population_registry.as_ref(),
                sampling_design: study.sampling_design,
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
            suite: self.study.refute,
            estimator: self.estimator_id.as_str(),
            custom: &self.study.custom_validators,
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

    /// Identify + plug-in estimate for an interventional distribution on a
    /// supplied DAG, or on a finite-discrete ADMG via general ID (bidirected
    /// edges stay; they are not dropped to coerce a DAG).
    pub(super) fn execute_distribution(
        &self,
        data: &TabularData,
        graph: DistributionGraph<'_>,
        query: &antecedent_core::InterventionalDistributionQuery,
        physical: &PhysicalExecutionPlan,
        distribution_operation: Option<&super::super::prepared::CheckedDistributionOperation>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if matches!(graph, DistributionGraph::Admg(_)) {
            ensure_admg_distribution_licensed(query, self.refute)?;
        }
        let started = Instant::now();
        let (identifier_id, estimator_id) =
            physical.binding().resolve(RouteDefaults::DISTRIBUTION)?;
        if !matches!(estimator_id, EstimatorId::FunctionalDistribution) {
            return Err(CausalError::Compile {
                message: format!(
                    "Distribution execute requires estimator functional.distribution; got {estimator_id}"
                ),
            });
        }

        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query), all frozen there, so reuse is exact
        // and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = graph.identify(identifier_id, query)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;

        let est = FunctionalDistribution {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalDistribution::new()
        };
        let prepared = Self::prepare_distribution_problem(
            &est,
            data,
            query,
            &identification,
            &estimand,
            distribution_operation,
        )?;
        let (dist, posterior) =
            self.estimate_distribution(&est, &prepared, identification.status, ctx)?;

        let estimate = EffectEstimate::from_parts(
            dist.mean,
            dist.se_analytic,
            BootstrapAccounting {
                se: dist.se_bootstrap,
                replicates_ok: dist.bootstrap_replicates_ok,
                replicates_failed: dist.bootstrap_replicates_failed,
                cancelled: dist.bootstrap_cancelled,
                early_stopped: dist.bootstrap_early_stopped,
            },
            dist.assumptions.clone(),
            dist.overlap,
            None,
            dist.retained_memory_bytes,
        );

        let treatment =
            query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                || CausalError::Compile {
                    message: "distribution query missing intervention target".into(),
                },
            )?;
        let outcome = *query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query missing outcome".into(),
        })?;
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let cancelled = estimate.uncertainty.bootstrap.cancelled;
        let early_stopped = estimate.uncertainty.bootstrap.early_stopped;

        // An ADMG distribution only reaches here at validation none (gated above).
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_distribution(
                data,
                query,
                &identification,
                &estimand,
                &dist,
                self.refute == RefuteSuite::Full,
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
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(distribution_interval_diagnostics(&dist))
            .refutations(refutations)
            .distribution(Some(dist))
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(cancelled)
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    /// Binds the distribution problem to `data`, through the retained operation when one exists.
    fn prepare_distribution_problem(
        est: &FunctionalDistribution,
        data: &TabularData,
        query: &antecedent_core::InterventionalDistributionQuery,
        identification: &IdentificationResult,
        estimand: &IdentifiedEstimand,
        distribution_operation: Option<&super::super::prepared::CheckedDistributionOperation>,
    ) -> Result<antecedent_estimate::PreparedFunctionalDistribution, CausalError> {
        if let Some(operation) = distribution_operation {
            let checked_arena =
                antecedent_io::expr_arena_to_wire(operation.prepared().program().arena())
                    .map_err(CausalError::rejected)?;
            let claim_arena = antecedent_io::expr_arena_to_wire(&identification.arena)
                .map_err(CausalError::rejected)?;
            if operation.query() != query
                || operation.prepared().estimand.functional != estimand.functional
                || checked_arena != claim_arena
            {
                return Err(CausalError::Compile {
                    message: "prepared distribution target disagrees with selected identification"
                        .into(),
                });
            }
            operation.rebind(data)
        } else {
            est.prepare(
                data,
                query,
                estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)
        }
    }

    /// Frequentist plug-in or Bayesian posterior estimate of the prepared distribution.
    fn estimate_distribution(
        &self,
        est: &FunctionalDistribution,
        prepared: &antecedent_estimate::PreparedFunctionalDistribution,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_estimate::InterventionalDistributionEstimate,
            Option<antecedent_estimate::CausalPosterior>,
        ),
        CausalError,
    > {
        if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.has_transfer() {
                return Err(CausalError::Unsupported {
                    message: "functional Bayesian prior transfer requires a declared \
                              functional mapping; a backdoor coefficient artifact cannot \
                              be applied as an isotropic CPT prior",
                });
            }
            let (dist, posterior) = est
                .estimate_bayesian(
                    prepared,
                    &[],
                    bayesian_draw_count(&self.inference)?,
                    status,
                    ctx,
                )
                .map_err(CausalError::from)?;
            Ok((dist, Some(posterior)))
        } else {
            let mut ws = FunctionalDistributionWorkspace::default();
            Ok((est.estimate(prepared, &[], &mut ws, ctx).map_err(CausalError::from)?, None))
        }
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
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let cq = CausalQuery::PathSpecific(query.clone());
                let identification = identify_static_query(identifier_id, graph, &cq)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok((identification, estimand))
            })?;

        let mut extra = vec![query.treatment, query.outcome];
        extra.extend(query.path_nodes.iter().copied());
        let (estimate, posterior) =
            self.estimate_functional_effect(data, &estimand, &identification, &extra, ctx)?;

        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::functional::refute_path(
                data,
                query,
                &identification,
                &estimand,
                estimate.ate,
                self.refute == RefuteSuite::Full,
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
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static_query(
                    identifier_id,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
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
            suite: self.refute,
            estimator: estimator_id.as_str(),
            custom: &self.custom_validators,
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
        // Sharp RD is the documented identify-per-click exception: prepare()
        // stores no cache for it, so every click computes identification.
        report_identify_compute(ctx);
        let identification = SharpRdIdentifier::new(SharpRdConfig::new(
            rd.running_variable,
            rd.cutoff,
            rd.bandwidth,
        ))
        .identify_on(graph, CausalQuery::AverageEffect(query.clone()))
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
        // The identified query names the population the design speaks for (units at the
        // cutoff); estimation, refutation and the result all use that query.
        let identified_query =
            identification.average_effect().cloned().unwrap_or_else(|| query.clone());
        let query = &identified_query;

        let mut est =
            SharpRegressionDiscontinuity::new(rd.running_variable, rd.cutoff, rd.bandwidth);
        est.bootstrap_replicates = self.bootstrap_replicates;
        est.se_kind = rd.se_kind;
        let checked = est.prepare_checked(data, &identification, 0).map_err(CausalError::from)?;
        let mut ws = RdWorkspace::default();
        let estimate = est.fit_checked(&checked, &mut ws, ctx).map_err(CausalError::from)?;

        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, extra_diagnostics) = RefutationRun {
            data,
            estimand: &estimand,
            query,
            estimate: &estimate,
            suite: self.refute,
            estimator: "rd.sharp",
            custom: &self.custom_validators,
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
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started)),
        ))
    }

    pub(super) fn execute_conditional(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.estimator == Some(EstimatorId::BayesianBasisGcomp) {
            return self.execute_bayesian_basis(data, graph, &query.inner, physical, ctx);
        }
        if self.inference.is_bayesian() {
            return self.execute_bayesian(data, graph, &query.inner, physical, None, ctx);
        }
        let started = Instant::now();
        let (identifier_id, _) = self.resolve_conditional_ids();
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query.inner), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static(identifier_id, graph, &query.inner)?;
                select_claim(identification, EstimatorId::ConditionalLinearAdjustment)
            })?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.inner.outcome,
            &query.inner.outcome_functional,
        )?;
        let est = ConditionalLinearAdjustment::new();
        let mut mean_query = query.clone();
        mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
        let estimate =
            est.estimate(&data_est, &estimand, &mean_query, ctx).map_err(CausalError::from)?;
        let estimate = super::super::helpers::attach_conditional_functional_grid(
            estimate, data, query, &estimand, ctx,
        )?;
        let uses_aipw_scores = super::super::helpers::conditional_uses_crossfit_aipw(query);
        let estimator_name =
            if uses_aipw_scores { "aipw" } else { "conditional.linear.adjustment" };
        let estimator_id = if uses_aipw_scores {
            EstimatorId::Aipw
        } else {
            EstimatorId::ConditionalLinearAdjustment
        };
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, mut extra_diagnostics) = RefutationRun {
            data,
            estimand: &estimand,
            query: &query.inner,
            estimate: &estimate,
            suite: self.refute,
            estimator: estimator_name,
            custom: &self.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;
        if let Some(diagnostic) = super::super::helpers::conditional_quantile_grid_diagnostic(
            data,
            query,
            estimand.adjustment_set.iter().copied(),
        )? {
            extra_diagnostics.push(diagnostic);
        }
        extra_diagnostics
            .extend(super::super::helpers::conditional_score_estimator_diagnostic(query));
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.inner.treatment,
                query.inner.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started)),
        ))
    }

    /// Cpdag/Pag `ConditionalEffect` via the same generalized-adjustment envelope as ATE.
    pub(in crate::analysis) fn execute_class_conditional(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier_id, estimator_id) = self.resolve_class_conditional_ids();
        match self.graph.class() {
            GraphClass::Pag => {
                let pag = physical.static_pag().ok_or_else(|| CausalError::Compile {
                    message: "PAG ConditionalEffect execute missing resolved static PAG".into(),
                })?;
                let (envelope, identification, identify_cached) =
                    if let Some(cache) = self.pag_identification_cache.as_ref() {
                        (Arc::clone(&cache.envelope), cache.identification.clone(), true)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = identify_pag(identifier_id, pag, &query.inner)?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (Arc::new(envelope), identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: super::pag_path::pag_envelope_diagnostic(&envelope),
                    },
                    identification,
                    Strategy::new(identifier_id, estimator_id),
                    super::pag_path::ClassTag::Pag,
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::Envelope {
                            envelope: Arc::unwrap_or_clone(envelope),
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            GraphClass::Cpdag => {
                let cpdag = self.graph.as_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "CPDAG ConditionalEffect execute missing supplied graph".into(),
                })?;
                let (envelope, identification, identify_cached) =
                    if let Some(cache) = self.cpdag_identification_cache.as_ref() {
                        (Arc::clone(&cache.envelope), cache.identification.clone(), true)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = identify_cpdag(identifier_id, cpdag, &query.inner)?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (Arc::new(envelope), identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    ClassEnvelopeRun {
                        run: ExecuteRun { physical, ctx, started, identify_cached },
                        envelope: &envelope,
                        envelope_diagnostic: super::pag_path::cpdag_envelope_diagnostic(&envelope),
                    },
                    identification,
                    Strategy::new(identifier_id, estimator_id),
                    super::pag_path::ClassTag::Cpdag,
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::CpdagEnvelope {
                            envelope: Arc::unwrap_or_clone(envelope),
                            strategy: identifier_id,
                            structure_version: self.graph.version(),
                        },
                    )
                })
            }
            _ => Err(CausalError::Unsupported {
                message: "class-aware ConditionalEffect execute requires a Cpdag or Pag",
            }),
        }
    }

    fn finish_class_conditional<G>(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        strategy: Strategy,
        class_tag: super::pag_path::ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let estimator_id = strategy.estimator;
        let envelope = run.envelope;
        if envelope.unidentified_weight.0 > 1e-12
            && (query.inner.outcome_functional.quantile_level().is_some()
                || query.inner.outcome_functional.thresholds().is_some())
        {
            return Err(CausalError::Unsupported {
                message: "class-aware ConditionalEffect grid requires zero unidentified completion mass",
            });
        }
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if self.inference.is_bayesian()
                || matches!(estimator_id, EstimatorId::BayesianConditional)
            {
                return self.execute_pag_nonidentified_prior(
                    &query.inner,
                    run,
                    identification,
                    class_tag.as_str(),
                );
            }
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                &format!(
                    "{class_tag} ConditionalEffect not identified (no identified mass in envelope)"
                ),
            ));
        }
        if self.inference.is_bayesian() || matches!(estimator_id, EstimatorId::BayesianConditional)
        {
            return self.execute_pag_bayesian(data, &query.inner, run, identification, class_tag);
        }
        self.finish_class_conditional_frequentist(
            data,
            query,
            run,
            identification,
            strategy,
            class_tag,
        )
    }

    /// Frequentist pooled conditional effect over the identified atoms of a class envelope.
    fn finish_class_conditional_frequentist<G>(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        strategy: Strategy,
        class_tag: super::pag_path::ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic: envelope_diag,
        } = run;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.inner.outcome,
            &query.inner.outcome_functional,
        )?;
        let mut diagnostics = vec![envelope_diag];
        let atoms =
            evaluate_conditional_atoms(data, &data_est, query, envelope, estimator_id, ctx)?;
        let (mut estimate, estimand, n_contributing) = atoms.pooled_estimate(class_tag.as_str())?;
        let ConditionalAtoms { refute_atoms, grid_atoms, outcomes, .. } = atoms;
        if query.inner.outcome_functional.thresholds().is_some()
            || query.inner.outcome_functional.quantile_level().is_some()
        {
            estimate = super::attach_class_conditional_functional_grid(
                estimate,
                data,
                query,
                &grid_atoms,
                ctx,
            )?;
        }
        if let Some(diagnostic) = super::super::helpers::conditional_quantile_grid_diagnostic(
            data,
            query,
            grid_atoms.iter().flat_map(|(_, e)| e.adjustment_set.iter().copied()),
        )? {
            diagnostics.push(diagnostic);
        }
        diagnostics.extend(super::super::helpers::conditional_score_estimator_diagnostic(query));
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = EnvelopeRefutation {
            data,
            query: &query.inner,
            suite: self.refute,
            estimator: estimator_id.as_str(),
            custom: &self.custom_validators,
            temporal: None,
        }
        .run(&refute_atoms, &mut refute_ws, ctx)?;
        diagnostics.extend(na_diagnostics);
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic()));
        // Completions that disagree publish their identified set, exactly as the
        // class ATE arm does; a point-identified envelope stays a point.
        let structural_response =
            super::pag_path::static_class_structural_mixture(envelope, &outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(super::pag_path::static_class_identified_set_diagnostic(
                mixture,
                class_tag.as_str(),
            ));
        }
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.inner.treatment,
                query.inner.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                structural_response,
                ..Default::default()
            }),
        ))
    }

    pub(super) fn execute_static_mediation_total(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static_query(
                    IdentifierId::PathSpecificNatural,
                    graph,
                    &CausalQuery::Mediation(query.clone()),
                )?;
                select_claim(identification, EstimatorId::StaticMediationLinear)
            })?;
        if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior.is_some() || cfg.external_compose.is_some() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
                });
            }
            if cfg.prior_artifact.is_some() && cfg.prior_mapping.is_none() {
                return Err(CausalError::Unsupported {
                    message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
                });
            }
        }
        let (mediation, posterior) = self.estimate_static_mediation_total(
            data,
            graph,
            query,
            identification.required_assumptions.clone(),
            identification.status,
            ctx,
        )?;
        let estimate = mediation.effect.clone();
        let bootstrap_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let (cancelled, early_stopped) = (
            estimate.uncertainty.bootstrap.cancelled,
            estimate.uncertainty.bootstrap.early_stopped,
        );
        let refutations = if self.refute == RefuteSuite::None {
            Vec::new()
        } else {
            antecedent_validate::mediation::refute_static_mediation(
                data,
                graph,
                query,
                &mediation,
                self.refute == RefuteSuite::Full,
                ctx,
            )?
        };
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::PathSpecificNatural,
                    EstimatorId::StaticMediationLinear,
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .mediation(Some(mediation))
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_ok)
            .cancelled(cancelled)
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            }),
        ))
    }

    /// Frequentist or Bayesian static mediation estimate for the total-effect route.
    fn estimate_static_mediation_total(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MediationQuery,
        assumptions: antecedent_core::AssumptionSet,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            antecedent_estimate::TemporalMediationEstimate,
            Option<antecedent_estimate::CausalPosterior>,
        ),
        CausalError,
    > {
        if let InferenceMode::Bayesian(cfg) = &self.inference {
            let est = bayesian_gcomp(cfg, ctx);
            let decoded = crate::inference::decode_prior_hydrate_source(cfg)?;
            let bridge = decoded.as_ref().map(|d| antecedent_estimate::MediationPriorBridge {
                mapping: &d.mapping,
                quantities: &d.quantities,
                mean: &d.mean,
                sd: &d.sd,
                source_contrast: d.source_contrast,
            });
            let (mediation, posterior) = antecedent_estimate::estimate_static_mediation_bayesian(
                &antecedent_estimate::StaticMediationProblem { data, graph, query, extra: &[] },
                assumptions,
                &est,
                status,
                bridge,
                ctx,
            )?;
            Ok((mediation, Some(posterior)))
        } else {
            Ok((
                antecedent_estimate::estimate_static_mediation(
                    data,
                    graph,
                    query,
                    assumptions,
                    self.bootstrap_replicates,
                    &[],
                    ctx,
                )?,
                None,
            ))
        }
    }

    pub(super) fn execute_tiered_average(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        background: &antecedent_graph::TieredBackground,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let started = Instant::now();
        let (identification, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = antecedent_identify::identify_tiered(background, query)?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "tiered identification returned no estimand".into(),
                    }
                })?;
                Ok((identification, estimand))
            })
            .map(|(id, _, cached)| (id, cached))?;
        if background.within_tier == antecedent_graph::WithinTier::Unknown {
            return self.finish_tiered_unknown(
                data,
                query,
                ExecuteRun { physical, ctx, started, identify_cached },
                identification,
                Strategy::new(identifier_id, estimator_id),
            );
        }
        let estimand = identification.estimands[0].clone();
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.outcome,
            &query.outcome_functional,
        )?;
        let mut ws = StaticEstimateWorkspaces::default();
        let spec =
            self.estimator_spec.as_deref().cloned().unwrap_or(EstimatorSpec::Default(estimator_id));
        let estimate = estimate_static_effect(
            &spec,
            StaticEstimateInput {
                data: &data_est,
                estimand: &estimand,
                query,
                bootstrap_replicates: self.bootstrap_replicates,
                overlap_policy: self.overlap_policy,
                population_registry: self.population_registry.as_ref(),
                sampling_design: self.sampling_design,
            },
            identification.required_assumptions.clone(),
            ctx,
            &mut ws,
        )?;
        let mut extra_diagnostics = Vec::new();
        let estimate = super::super::helpers::attach_average_functional_grid(
            estimate,
            super::super::helpers::AverageTarget { data, query, estimand: &estimand },
            &mut extra_diagnostics,
            estimator_id,
            self,
            ctx.rng.master_seed(),
        )?;
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = RefutationRun {
            data,
            estimand: &estimand,
            query,
            estimate: &estimate,
            suite: self.refute,
            estimator: estimator_id.as_str(),
            custom: &self.custom_validators,
            temporal: None,
        }
        .reports(&mut refute_ws, None, ctx)?;
        extra_diagnostics.extend(na_diagnostics);
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
            .wall_time_ns(elapsed_ns(started)),
        ))
    }

    /// Execute the prepared two-scenario Unknown-tier product without looking
    /// up or rebuilding its identification from the retained Study cache.
    pub(crate) fn execute_checked_unknown_tiered_average(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        retained: CheckedUnknownTiered<'_>,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let CheckedUnknownTiered { background, identification } = retained;
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        if matches!(self.inference, crate::inference::InferenceMode::Bayesian(_)) {
            // One posterior per canonical scenario, never a collapsed scalar.
            return self.execute_tiered_bayesian_average(data, query, physical, ctx, background);
        }
        if background.within_tier != antecedent_graph::WithinTier::Unknown
            || identification.query != CausalQuery::AverageEffect(query.clone())
            || identification.estimands.len() != 2
            || identification.status != IdentificationStatus::GraphDependent
        {
            return Err(CausalError::Conflict {
                what: "checked Unknown-tier operation",
                detail: "retained target, tier interpretation, or scenario envelope changed",
            });
        }
        self.finish_tiered_unknown(
            data,
            query,
            ExecuteRun { physical, ctx, started: Instant::now(), identify_cached: true },
            identification,
            Strategy::new(identifier_id, estimator_id),
        )
    }

    /// Fits every canonical scenario estimand of a tiered Unknown envelope.
    fn evaluate_tiered_scenarios(
        &self,
        data_est: &TabularData,
        data: &TabularData,
        query: &AverageEffectQuery,
        identification: &IdentificationResult,
        estimator_id: EstimatorId,
        ctx: &ExecutionContext,
    ) -> Result<TieredScenarios, CausalError> {
        let spec =
            self.estimator_spec.as_deref().cloned().unwrap_or(EstimatorSpec::Default(estimator_id));
        let mut atom_ifs = Vec::new();
        let mut atom_ws = Vec::new();
        let mut scenarios = Vec::new();
        let mut primary = None;
        let mut assumptions = identification.required_assumptions.clone();
        for estimand in &identification.estimands {
            let mut ws = StaticEstimateWorkspaces::default();
            let estimate = estimate_static_effect(
                &spec,
                StaticEstimateInput {
                    data: data_est,
                    estimand,
                    query,
                    bootstrap_replicates: self.bootstrap_replicates,
                    overlap_policy: self.overlap_policy,
                    population_registry: self.population_registry.as_ref(),
                    sampling_design: self.sampling_design,
                },
                identification.required_assumptions.clone(),
                ctx,
                &mut ws,
            )?;
            scenarios.push((
                scenarios.len() as u64,
                antecedent_core::ResponseValue::Scalar(estimate.ate),
            ));
            if let Some(inf) = estimate
                .scores
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, estimand, inf))
            {
                atom_ifs.push(inf);
                atom_ws.push(0.5);
            }
            if primary.is_none() {
                primary = Some(estimand.clone());
                assumptions = estimate.provenance.assumptions;
            }
        }
        Ok(TieredScenarios { atom_ifs, scenarios, primary, assumptions })
    }

    fn finish_tiered_unknown(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        run: ExecuteRun<'_>,
        identification: IdentificationResult,
        strategy: Strategy,
    ) -> Result<StudyResult, CausalError> {
        let Strategy { identifier: identifier_id, estimator: estimator_id } = strategy;
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.outcome,
            &query.outcome_functional,
        )?;
        let TieredScenarios { atom_ifs, scenarios, primary, assumptions } = self
            .evaluate_tiered_scenarios(
                &data_est,
                data,
                query,
                &identification,
                estimator_id,
                ctx,
            )?;
        let refs: Vec<&[f64]> = atom_ifs.iter().map(Vec::as_slice).collect();
        let mut extra_diagnostics = Vec::new();
        let covariance = if refs.len() == scenarios.len() && refs.len() >= 2 {
            if let Ok(cov) = antecedent_estimate::joint_influence_covariance(&refs, None) {
                Some(cov)
            } else {
                extra_diagnostics.push(Diagnostic::new(
                        "tiered.unknown.joint_if.unavailable",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Warning,
                        "tiered Unknown joint IF covariance could not be formed; scenario intervals are omitted",
                    ));
                None
            }
        } else {
            if !refs.is_empty() && refs.len() != scenarios.len() {
                extra_diagnostics.push(Diagnostic::new(
                    "tiered.unknown.joint_if.unavailable",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "tiered Unknown joint IF refused: every canonical scenario must supply an aligned influence function",
                ));
            }
            None
        };
        let estimate =
            EffectEstimate::new(f64::NAN, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride);
        let estimand = primary.ok_or_else(|| CausalError::Compile {
            message: "tiered Unknown envelope had no estimand".into(),
        })?;
        let mut estimate = estimate.with_joint_covariance(covariance.clone());
        let values: Vec<f64> = scenarios
            .iter()
            .filter_map(|(_, v)| {
                if let antecedent_core::ResponseValue::Scalar(v) = v { Some(*v) } else { None }
            })
            .collect();
        if let Some(cov) = &covariance {
            let c = antecedent_estimate::max_t_critical(cov, 0.95, 4096, 15)?;
            estimate.scenarios.intervals = Some(
                values
                    .iter()
                    .enumerate()
                    .map(|(j, v)| (v - c * cov.se(j), v + c * cov.se(j)))
                    .collect::<Vec<_>>()
                    .into(),
            );
        }
        // The within-tier order is unknown, so no canonical scenario is *the*
        // effect: the answer is the identified set over the scenarios, and each
        // keeps its own value. The reported scalar stays withheld (NaN).
        let structural_response = scenario_structural_mixture(&values, identification.status);
        if let Some(mixture) = structural_response.as_ref() {
            extra_diagnostics.extend(super::pag_path::static_class_identified_set_diagnostic(
                mixture,
                "tiered_unknown",
            ));
        }
        estimate.scenarios.effects = Some(values.into());
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
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras { structural_response, ..Default::default() }),
        ))
    }
}

/// Scenario values and aligned influence functions of a tiered Unknown envelope.
struct TieredScenarios {
    atom_ifs: Vec<Vec<f64>>,
    scenarios: Vec<(u64, antecedent_core::ResponseValue)>,
    primary: Option<IdentifiedEstimand>,
    assumptions: antecedent_core::AssumptionSet,
}

/// Per-atom results of a class-aware conditional envelope, accumulated in envelope order.
struct ConditionalAtoms {
    weighted_ate: f64,
    se_items: Vec<(f64, f64)>,
    atom_ifs: Vec<Vec<f64>>,
    atom_weights: Vec<f64>,
    total_w: f64,
    primary_estimand: Option<IdentifiedEstimand>,
    assumptions: antecedent_core::AssumptionSet,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
    grid_atoms: Vec<(f64, IdentifiedEstimand)>,
    outcomes: Vec<ClassAtomOutcome>,
}

/// Fits the conditional linear adjustment on each identified atom of the envelope.
fn evaluate_conditional_atoms<G>(
    data: &TabularData,
    data_est: &TabularData,
    query: &antecedent_core::ConditionalEffectQuery,
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
    estimator_id: EstimatorId,
    ctx: &ExecutionContext,
) -> Result<ConditionalAtoms, CausalError> {
    let mut atoms = ConditionalAtoms {
        weighted_ate: 0.0,
        se_items: Vec::new(),
        atom_ifs: Vec::new(),
        atom_weights: Vec::new(),
        total_w: 0.0,
        primary_estimand: None,
        assumptions: antecedent_core::AssumptionSet::default(),
        refute_atoms: Vec::new(),
        grid_atoms: Vec::new(),
        outcomes: vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()],
    };
    let est = ConditionalLinearAdjustment::new();
    for (i, case) in envelope.cases.iter().enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, estimator_id)?;
        let mut mean_query = query.clone();
        mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
        let estimate =
            est.estimate(data_est, &estimand, &mean_query, ctx).map_err(CausalError::from)?;
        let w = case.weight.0;
        atoms.outcomes[i] = ClassAtomOutcome::Evaluated(estimate.ate);
        atoms.weighted_ate += w * estimate.ate;
        atoms.se_items.push((w, estimate.se_analytic()));
        if let Some(inf) = estimate
            .scores
            .influence
            .as_deref()
            .and_then(|inf| static_aligned_influence(data, &query.inner, &estimand, inf))
        {
            atoms.atom_ifs.push(inf);
            atoms.atom_weights.push(w);
        }
        atoms.total_w += w;
        if atoms.primary_estimand.is_none() {
            atoms.primary_estimand = Some(estimand.clone());
            atoms.assumptions = estimate.provenance.assumptions.clone();
        }
        atoms.grid_atoms.push((w, estimand.clone()));
        atoms.refute_atoms.push(EnvelopeRefuteAtom {
            key: i as u64,
            weight: w,
            estimand,
            indexer: None,
            original: estimate,
        });
    }
    Ok(atoms)
}

impl ConditionalAtoms {
    /// Weight-pooled point estimate, primary estimand and contributing-atom count.
    fn pooled_estimate(
        &self,
        class_tag: &str,
    ) -> Result<(EffectEstimate, IdentifiedEstimand, usize), CausalError> {
        if !matches!(self.total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: format!("{class_tag} ConditionalEffect envelope had no estimable cases"),
            });
        }
        let estimand = self.primary_estimand.clone().ok_or_else(|| CausalError::Compile {
            message: format!("{class_tag} ConditionalEffect envelope missing estimand"),
        })?;
        let n_contributing = self.se_items.len();
        let se = if self.atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&self.atom_ifs, &self.atom_weights)
        } else {
            mix_weighted_analytic_se(self.se_items.iter().copied())
        };
        let mut estimate = EffectEstimate::new(
            self.weighted_ate / self.total_w,
            se,
            self.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if self.atom_ifs.len() == n_contributing {
            if let Some(inf) = mixed_static_influence(&self.atom_ifs, &self.atom_weights) {
                estimate.scores.influence = Some(inf);
            }
        }
        Ok((estimate, estimand, n_contributing))
    }
}

/// Structural uncertainty of a tiered `Unknown` within-tier closure: every
/// canonical scenario with equal enumeration weight, its own value, and the
/// identified set over them. `None` when no scenario produced a finite value.
fn scenario_structural_mixture(
    values: &[f64],
    status: IdentificationStatus,
) -> Option<crate::result::StructuralResponseMixture> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let (lo, hi) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    Some(crate::result::StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::CompletionEnumeration,
        atoms: values
            .iter()
            .enumerate()
            .map(|(index, value)| crate::result::StructuralResponseAtom {
                graph_key: u64::try_from(index).unwrap_or(u64::MAX),
                weight: 1.0,
                status,
                value: Some(antecedent_core::ResponseValue::Scalar(*value)),
                posterior: None,
                response: None,
            })
            .collect(),
        identified_mass: 1.0,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        subsampled_out_mass: 0.0,
        identified_set: Some(scalar_identified_set(lo, hi)),
        identified_set_interval: None,
        conditional_on_identified: None,
        full_mass_scope: true,
        truncated_atoms: 0,
    })
}

/// Name every interventional probability whose bounded interval could not be
/// formed (a boundary plug-in `p̂ ∈ {0, 1}`, a failed bootstrap, or zero
/// spread), so a missing interval is never mistaken for a computed one.
fn distribution_interval_diagnostics(
    dist: &antecedent_estimate::InterventionalDistributionEstimate,
) -> Vec<Diagnostic> {
    let missing: Vec<(usize, &'static str)> = dist
        .atom_uncertainty
        .iter()
        .enumerate()
        .filter_map(|(i, u)| match u.interval {
            antecedent_estimate::ProbabilityInterval::Unavailable(reason) => {
                Some((i, reason.as_str()))
            }
            antecedent_estimate::ProbabilityInterval::Bounded { .. } => None,
        })
        .collect();
    if missing.is_empty() {
        return Vec::new();
    }
    let atoms = missing.iter().map(|(i, r)| format!("{i}:{r}")).collect::<Vec<_>>().join(",");
    let mut diagnostic = Diagnostic::new(
        "estimate.distribution.interval_unavailable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        format!(
            "{} of {} interventional probabilities have no bounded interval (atom:reason \
             {atoms}); a plug-in probability of exactly 0 or 1 has no sampling spread to \
             build one from",
            missing.len(),
            dist.atom_uncertainty.len()
        ),
    );
    diagnostic.fields = Arc::from([(Arc::from("atoms"), Arc::from(atoms.as_str()))]);
    vec![diagnostic]
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

/// Licence gate for an ADMG interventional distribution, shared by compile and
/// execute: unconditional finite-discrete tables at validation none.
pub(super) fn ensure_admg_distribution_licensed(
    query: &antecedent_core::InterventionalDistributionQuery,
    refute: RefuteSuite,
) -> Result<(), CausalError> {
    if !query.conditioning.is_empty() {
        return Err(CausalError::Unsupported {
            message: "ADMG InterventionalDistribution is licensed for unconditional \
                      finite-discrete tables; IDC conditionals are a follow-up",
        });
    }
    if refute != RefuteSuite::None {
        return Err(CausalError::Unsupported {
            message: "ADMG InterventionalDistribution is licensed at validation none; \
                      cheap/full remain Dag-only",
        });
    }
    Ok(())
}
