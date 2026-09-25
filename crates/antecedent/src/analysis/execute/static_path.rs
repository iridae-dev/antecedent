// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::estimator_spec::EstimatorSpec;

impl super::Study {
    pub(super) fn execute_static(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        prepared_linear: Option<&super::super::prepared::CheckedLinearOperation>,
        prepared_aipw: Option<&super::super::prepared::CheckedAipwOperation>,
        prepared_frontdoor: Option<&super::super::prepared::CheckedFrontDoorOperation>,
        bayesian_gcomp_operation: Option<&super::super::prepared::CheckedBayesianGcompOperation>,
        prepared_iv: Option<&super::super::prepared::CheckedIvOperation>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut clock = super::super::stage::StageClock::new();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let estimator = physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;

        // rd.sharp identifies from the declared design, which the graph must agree
        // with; it has no adjustment search, so it takes its own path.
        if matches!(estimator_id, EstimatorId::RdSharp) {
            return self.execute_rd(data, graph, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::BayesianIvJointLinear) {
            return self.execute_bayesian_iv(data, graph, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::BayesianRdLocalLinear) {
            return self.execute_bayesian_rd(data, graph, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self.execute_bayesian(
                data,
                graph,
                query,
                physical,
                bayesian_gcomp_operation,
                ctx,
            );
        }
        if matches!(estimator_id, EstimatorId::BayesianBasisGcomp) {
            return self.execute_bayesian_basis(data, graph, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::BayesianRobustAte) {
            return self.execute_bayesian_robust_ate(data, graph, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return self.execute_functional_ate(data, graph, query, physical, ctx);
        }
        if let Some(background) = self.tiered.clone() {
            return self.execute_tiered_average(
                data,
                query,
                physical,
                ctx,
                &background,
                identifier_id,
                estimator_id,
            );
        }

        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query, rd), all frozen there, so reuse is
        // exact and observable via the `exec.identify.cached` diagnostic below.
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let rd =
                    self.rd.map(|c| SharpRdConfig::new(c.running_variable, c.cutoff, c.bandwidth));
                let identification = identify_static_query_with_rd(
                    identifier_id,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    rd,
                )?;
                select_claim(identification, estimator_id)
            })?;
        let assumptions = identification.required_assumptions.clone();
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
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

        // Point estimate first (no bootstrap); uncertainty stage fills SE separately.
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let mut estimate_ws = StaticEstimateWorkspaces::default();
        // A prepared checked route has already selected its estimator and bound
        // its configuration. Do not let the copied Study select a competing
        // fitter while executing that retained operation.
        let estimator_spec = if prepared_linear.is_some()
            || prepared_aipw.is_some()
            || prepared_frontdoor.is_some()
            || prepared_iv.is_some()
        {
            EstimatorSpec::Default(estimator_id)
        } else {
            self.estimator_spec.clone().unwrap_or(EstimatorSpec::Default(estimator_id))
        };
        // Prepare against the original semantic variable IDs. Projection remaps tabular
        // columns, while the identified expression still belongs to the original arena.
        let checked_linear = if matches!(
            query.outcome_functional,
            antecedent_core::OutcomeFunctional::Mean
        ) && matches!(
            query.target_population,
            antecedent_core::TargetPopulation::AllObserved
        ) {
            let fitter = match &estimator_spec {
                EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte) => {
                    Some(LinearAdjustmentAte::new())
                }
                EstimatorSpec::LinearAdjustmentAte(cfg) => Some((**cfg).clone()),
                _ => None,
            };
            let fitter = prepared_linear.map_or(fitter, |operation| Some(operation.fitter.clone()));
            fitter
                    .map(|fitter| {
                        let checked = match prepared_linear {
                            Some(operation) => {
                                if operation.preparation.source_functional() != estimand.functional
                                    || operation.preparation.target().adjustment_set != estimand.adjustment_set
                                {
                                    return Err(CausalError::Compile {
                                        message: "prepared adjustment lowering disagrees with selected identification".into(),
                                    });
                                }
                                operation.preparation.clone()
                            }
                            None => fitter.prepare_checked(data, &identification, 0)?,
                        };
                        Ok::<_, CausalError>((fitter, checked))
                    })
                    .transpose()?
        } else {
            None
        };
        let checked_frontdoor_linear = if prepared_frontdoor.is_some() {
            None
        } else {
            match &estimator_spec {
                EstimatorSpec::Default(EstimatorId::FrontDoorTwoStage) => {
                    let fitter = antecedent_estimate::FrontDoorTwoStage::new();
                    Some((fitter.clone(), fitter.prepare_checked(data, &identification, 0)?))
                }
                EstimatorSpec::FrontDoorTwoStage(cfg) => {
                    let fitter = (**cfg).clone();
                    Some((fitter.clone(), fitter.prepare_checked(data, &identification, 0)?))
                }
                _ => None,
            }
        };
        let checked_frontdoor_functional =
            if matches!(estimator_spec, EstimatorSpec::Default(EstimatorId::FrontDoorFunctional)) {
                let fitter = antecedent_estimate::FrontDoorFunctional::new();
                Some((fitter.clone(), fitter.prepare_checked(data, &identification, 0)?))
            } else {
                None
            };
        let checked_wald = if prepared_iv.is_some() {
            None
        } else {
            match &estimator_spec {
                EstimatorSpec::Default(EstimatorId::IvWald) => {
                    let fitter = antecedent_estimate::WaldIv::new();
                    Some((fitter.clone(), fitter.prepare_checked(data, &identification, 0)?))
                }
                EstimatorSpec::IvWald(cfg) => {
                    let fitter = (**cfg).clone();
                    Some((fitter.clone(), fitter.prepare_checked(data, &identification, 0)?))
                }
                _ => None,
            }
        };
        // Route the supported single binary-instrument slice through the checked IV
        // receipt. The generic 2SLS estimator still supports multiple/continuous
        // instruments and adjustment covariates; those designs remain on its legacy
        // preparation path because the current receipt lowers the binary Wald functional.
        let checked_iv_roles_supported = identification.estimands.first().is_some_and(|target| {
            target.instruments.len() == 1
                && target.mediators.is_empty()
                && target.adjustment_set.is_empty()
        });
        let checked_2sls = if prepared_iv.is_some() || !checked_iv_roles_supported {
            None
        } else {
            match &estimator_spec {
                EstimatorSpec::Default(EstimatorId::Iv2Sls) => {
                    let fitter = antecedent_estimate::TwoStageLeastSquares::new();
                    match fitter.prepare_checked(data, &identification, 0) {
                        Ok(checked) => Some((fitter, checked)),
                        Err(antecedent_estimate::EstimationError::Unsupported {
                            message:
                                "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
                        }) => None,
                        Err(error) => return Err(error.into()),
                    }
                }
                EstimatorSpec::Iv2Sls(cfg) => {
                    let fitter = (**cfg).clone();
                    match fitter.prepare_checked(data, &identification, 0) {
                        Ok(checked) => Some((fitter, checked)),
                        Err(antecedent_estimate::EstimationError::Unsupported {
                            message:
                                "checked IV currently requires a binary 0/1 instrument matching the checked Wald functional",
                        }) => None,
                        Err(error) => return Err(error.into()),
                    }
                }
                _ => None,
            }
        };
        let mut frontdoor_workspace = antecedent_estimate::FrontDoorWorkspace::default();
        let mut iv_workspace = antecedent_estimate::TwoStageLeastSquaresWorkspace::default();
        let point = if let Some(operation) = prepared_iv {
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
            if preparation.target().functional != estimand.functional
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
            match operation {
                super::super::prepared::CheckedIvOperation::Wald { fitter, preparation } => {
                    fitter.fit_checked(preparation, ctx).map_err(CausalError::from)?
                }
                super::super::prepared::CheckedIvOperation::TwoSls { fitter, preparation } => {
                    fitter
                        .fit_checked(preparation, &mut iv_workspace, ctx)
                        .map_err(CausalError::from)?
                }
            }
        } else if let Some((fitter, checked)) = &checked_linear {
            if prepared_linear.is_some_and(|operation| operation.default_id)
                || (prepared_linear.is_none()
                    && matches!(
                        estimator_spec,
                        EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte)
                    ))
            {
                // The progressive default route reports a genuine point stage.
                // Its bootstrap belongs to the uncertainty stage below, even
                // though the checked receipt retains the same bound design.
                fitter
                    .fit_point(
                        checked.problem(),
                        &mut estimate_ws.linear,
                        checked.required_assumptions().clone(),
                    )
                    .map_err(CausalError::from)?
            } else {
                fitter
                    .fit_checked(checked, &mut estimate_ws.linear, ctx)
                    .map_err(CausalError::from)?
            }
        } else if let Some(operation) = prepared_aipw {
            if operation.preparation.target().functional != estimand.functional
                || operation.preparation.target().adjustment_set != estimand.adjustment_set
            {
                return Err(CausalError::Compile {
                    message: "prepared AIPW lowering disagrees with selected identification".into(),
                });
            }
            operation
                .fitter
                .fit_checked(&operation.preparation, &mut estimate_ws.aipw, ctx)
                .map_err(CausalError::from)?
        } else if let Some(operation) = prepared_frontdoor {
            if operation.preparation.target().functional != estimand.functional
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
                    message: "prepared front-door lowering disagrees with selected identification or query".into(),
                });
            }
            operation
                .fitter
                .fit_checked(&operation.preparation, &mut frontdoor_workspace, ctx)
                .map_err(CausalError::from)?
        } else if let Some((fitter, checked)) = &checked_frontdoor_linear {
            fitter.fit_checked(checked, &mut frontdoor_workspace, ctx).map_err(CausalError::from)?
        } else if let Some((fitter, checked)) = &checked_frontdoor_functional {
            fitter.fit_checked(checked, ctx).map_err(CausalError::from)?
        } else if let Some((fitter, checked)) = &checked_wald {
            fitter.fit_checked(checked, ctx).map_err(CausalError::from)?
        } else if let Some((fitter, checked)) = &checked_2sls {
            fitter.fit_checked(checked, &mut iv_workspace, ctx).map_err(CausalError::from)?
        } else {
            estimate_static_effect(
                &estimator_spec,
                &data_est,
                &estimand_est,
                &query_est,
                assumptions,
                0, // point stage: no bootstrap
                self.overlap_policy,
                self.population_registry.as_ref(),
                ctx,
                &mut estimate_ws,
            )?
        };
        clock.finish(super::super::stage::STAGE_ESTIMATE_POINT);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Point { estimate: point.clone() },
        );

        // Uncertainty: bootstrap fills (real work when replicates > 0).
        // IV and NN matching must not refill with the facade bootstrap: that
        // bypasses the weak-instrument gate (Wald/2SLS) and publishes an
        // Abadie–Imbens-invalid matching bootstrap SE. A configured estimator owns its
        // replicate count and already ran it in the point fit, and the double-ML, DR-learner
        // and causal-forest fits take no replicate count: refitting any of them here would
        // reproduce the same estimate at the cost of a second full nuisance fit.
        let skip_bootstrap_refill = prepared_aipw.is_some()
            || prepared_frontdoor.is_some()
            || prepared_linear.is_some_and(|operation| !operation.default_id)
            || prepared_linear.is_some_and(|operation| operation.fitter.bootstrap_replicates == 0)
            || (prepared_linear.is_none() && self.bootstrap_replicates == 0)
            || (prepared_linear.is_none()
                && self
                    .estimator_spec
                    .as_ref()
                    .is_some_and(|spec| !matches!(spec, EstimatorSpec::Default(_))))
            || matches!(
                estimator_id,
                EstimatorId::IvWald
                    | EstimatorId::Iv2Sls
                    | EstimatorId::PropensityMatching
                    | EstimatorId::DistanceMatching
                    | EstimatorId::Dml
                    | EstimatorId::DrLearner
                    | EstimatorId::CausalForest
            );
        let estimate = if skip_bootstrap_refill {
            if ctx.cancellation.is_cancelled() {
                clock.mark_cancelled();
                point
            } else {
                clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
                clock.finish(super::super::stage::STAGE_UNCERTAINTY);
                super::super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    &super::super::stage::StageEvent::Uncertainty { estimate: point.clone() },
                );
                point
            }
        } else if matches!(estimator_id, EstimatorId::FrontDoorTwoStage)
            && checked_frontdoor_linear.is_some()
        {
            clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
            let (_, checked) = checked_frontdoor_linear.as_ref().expect("checked above");
            let mut fitter = antecedent_estimate::FrontDoorTwoStage::new();
            fitter.bootstrap_replicates = self.bootstrap_replicates;
            let filled = fitter
                .attach_bootstrap(checked.problem(), &mut frontdoor_workspace, ctx, point)
                .map_err(CausalError::from)?;
            clock.finish(super::super::stage::STAGE_UNCERTAINTY);
            super::super::stage::emit_stage(
                self.stage_sink.as_ref(),
                &super::super::stage::StageEvent::Uncertainty { estimate: filled.clone() },
            );
            filled
        } else if matches!(estimator_id, EstimatorId::FrontDoorFunctional)
            && checked_frontdoor_functional.is_some()
        {
            clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
            let (_, checked) = checked_frontdoor_functional.as_ref().expect("checked above");
            let fitter = antecedent_estimate::FrontDoorFunctional::new()
                .with_bootstrap_replicates(self.bootstrap_replicates);
            let filled = fitter
                .attach_bootstrap(checked.problem(), ctx, point)
                .map_err(CausalError::from)?;
            clock.finish(super::super::stage::STAGE_UNCERTAINTY);
            super::super::stage::emit_stage(
                self.stage_sink.as_ref(),
                &super::super::stage::StageEvent::Uncertainty { estimate: filled.clone() },
            );
            filled
        } else if matches!(estimator_id, EstimatorId::LinearAdjustmentAte) {
            // Reuse warmed OLS workspace: re-prepare + attach bootstrap without refitting point.
            let cancelled_before = ctx.cancellation.is_cancelled();
            if cancelled_before {
                clock.mark_cancelled();
                if let Some(p) = &ctx.progress {
                    p.report(0.55, super::super::stage::STAGE_UNCERTAINTY);
                }
                point
            } else {
                clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
                // A configured linear estimator never reaches here (it bootstrapped in the
                // point fit), so this is the id-selected default the point stage also used.
                let mut est = prepared_linear
                    .map_or_else(LinearAdjustmentAte::new, |operation| operation.fitter.clone());
                if prepared_linear.is_none() {
                    est.bootstrap_replicates = self.bootstrap_replicates;
                    est.overlap = OverlapPolicy::ExplicitOverride;
                }
                let prep = if let Some((_, checked)) = &checked_linear {
                    checked.problem().clone()
                } else {
                    est.prepare(&data_est, &estimand_est, &query_est).map_err(CausalError::from)?
                };
                let filled = est
                    .attach_bootstrap(&prep, &mut estimate_ws.linear, ctx, point)
                    .map_err(CausalError::from)?;
                let cancelled = filled.bootstrap_cancelled || ctx.cancellation.is_cancelled();
                if cancelled {
                    clock.mark_cancelled();
                } else {
                    clock.finish(super::super::stage::STAGE_UNCERTAINTY);
                }
                super::super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    &super::super::stage::StageEvent::Uncertainty { estimate: filled.clone() },
                );
                filled
            }
        } else {
            // Weighting, stratification, AIPW, GLM and front-door: their `fit` is the point fit
            // plus `attach_bootstrap`, so the uncertainty stage reuses the point estimate.
            let cancelled_before = ctx.cancellation.is_cancelled();
            if cancelled_before {
                clock.mark_cancelled();
                if let Some(p) = &ctx.progress {
                    p.report(0.55, super::super::stage::STAGE_UNCERTAINTY);
                }
                point
            } else {
                clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
                let filled = crate::strategy_table::attach_static_bootstrap(
                    estimator_id,
                    &data_est,
                    &estimand_est,
                    &query_est,
                    point,
                    self.bootstrap_replicates,
                    self.overlap_policy,
                    self.population_registry.as_ref(),
                    ctx,
                    &mut estimate_ws,
                )?;
                let cancelled = filled.bootstrap_cancelled || ctx.cancellation.is_cancelled();
                if cancelled {
                    clock.mark_cancelled();
                } else {
                    clock.finish(super::super::stage::STAGE_UNCERTAINTY);
                }
                super::super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    &super::super::stage::StageEvent::Uncertainty { estimate: filled.clone() },
                );
                filled
            }
        };

        let cancelled = estimate.bootstrap_cancelled || clock.cancelled();

        let (refutations, na_diagnostics) = if cancelled {
            (Vec::new(), Vec::new())
        } else {
            clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
            let prop_scratch = match estimator_id {
                EstimatorId::Aipw => &mut estimate_ws.aipw.propensity,
                _ => &mut estimate_ws.propensity.propensity,
            };
            let (reports, na_diagnostics) = run_refuters(
                &data_est,
                &estimand_est,
                &query_est,
                &estimate,
                &mut estimate_ws.linear,
                Some(prop_scratch),
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
                None,
            )?;
            clock.finish(super::super::stage::STAGE_VALIDATE);
            super::super::stage::emit_stage(
                self.stage_sink.as_ref(),
                &super::super::stage::StageEvent::Validate {
                    refutations: reports.clone(),
                    predictive_checks: Vec::new(),
                },
            );
            (reports, na_diagnostics)
        };

        let mut extra_diagnostics =
            if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
                vec![d]
            } else {
                Vec::new()
            };
        extra_diagnostics.extend(na_diagnostics);
        let estimate = super::super::helpers::attach_average_functional_grid(
            estimate,
            data,
            query,
            &estimand,
            &mut extra_diagnostics,
            estimator_id,
            self,
            ctx.rng.master_seed(),
        )?;
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: clock.wall_time_ns(),
            bootstrap_replicates_ok: bootstrap_ok,
            cancelled: clock.cancelled(),
            early_stopped,
            extras: IdentifiedExecuteExtras {
                stage_timings_ns: clock.timings(),
                ..Default::default()
            },
        }))
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
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if !matches!(estimator_id, EstimatorId::FunctionalDistribution) {
            return Err(CausalError::Compile {
                message: format!(
                    "Distribution execute requires estimator functional.distribution; got {estimator}"
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
        let prepared = if let Some(operation) = distribution_operation {
            let checked_arena =
                antecedent_io::expr_arena_to_wire(operation.prepared().program().arena())
                    .map_err(|err| CausalError::Compile { message: err.to_string() })?;
            let claim_arena = antecedent_io::expr_arena_to_wire(&identification.arena)
                .map_err(|err| CausalError::Compile { message: err.to_string() })?;
            if operation.query() != query
                || operation.prepared().estimand.functional != estimand.functional
                || checked_arena != claim_arena
            {
                return Err(CausalError::Compile {
                    message: "prepared distribution target disagrees with selected identification"
                        .into(),
                });
            }
            operation.rebind(data)?
        } else {
            est.prepare(
                data,
                query,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)?
        };
        let (dist, posterior) = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            if let InferenceMode::Bayesian(cfg) = &self.inference {
                if cfg.prior_artifact.is_some()
                    || cfg.external_compose.is_some()
                    || cfg.prior.is_some()
                {
                    return Err(CausalError::Unsupported {
                        message: "functional Bayesian prior transfer requires a declared \
                                  functional mapping; a backdoor coefficient artifact cannot \
                                  be applied as an isotropic CPT prior",
                    });
                }
            }
            let (dist, posterior) = est
                .estimate_bayesian(
                    &prepared,
                    &[],
                    bayesian_draw_count(&self.inference)?,
                    identification.status,
                    ctx,
                )
                .map_err(CausalError::from)?;
            (dist, Some(posterior))
        } else {
            let mut ws = FunctionalDistributionWorkspace::default();
            (est.estimate(&prepared, &[], &mut ws, ctx).map_err(CausalError::from)?, None)
        };

        let estimate = EffectEstimate::from_parts(
            dist.mean,
            dist.se_analytic,
            dist.se_bootstrap,
            dist.bootstrap_replicates_ok,
            dist.bootstrap_replicates_failed,
            dist.bootstrap_cancelled,
            dist.bootstrap_early_stopped,
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
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let cancelled = estimate.bootstrap_cancelled;
        let early_stopped = estimate.bootstrap_early_stopped;

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

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: distribution_interval_diagnostics(&dist),
            refutations,
            distribution: Some(dist),
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: bootstrap_ok,
            cancelled,
            early_stopped,
            extras: IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            },
        }))
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
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PATH_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_PATH_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if !matches!(estimator_id, EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "PathSpecific execute requires estimator functional.effect; got {estimator}"
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

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            },
        }))
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
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER);
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
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
        let (refutations, extra_diagnostics) = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
        )?;
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            },
        }))
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
        let (refutations, extra_diagnostics) = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            "rd.sharp",
            &self.custom_validators,
            None,
        )?;

        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::RdSharp,
            estimator_id: EstimatorId::RdSharp,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached: false,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }))
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
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            return self.execute_bayesian(data, graph, &query.inner, physical, None, ctx);
        }
        let started = Instant::now();
        let (identifier, _) = self.resolve_conditional_pair();
        let identifier_id: IdentifierId = identifier.parse()?;
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
        let (refutations, mut extra_diagnostics) = run_refuters(
            data,
            &estimand,
            &query.inner,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator_name,
            &self.custom_validators,
            None,
        )?;
        if let Some(diagnostic) = super::super::helpers::conditional_quantile_grid_diagnostic(
            data,
            query,
            estimand.adjustment_set.iter().copied(),
        )? {
            extra_diagnostics.push(diagnostic);
        }
        extra_diagnostics
            .extend(super::super::helpers::conditional_score_estimator_diagnostic(query));
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.inner.treatment,
            outcome: query.inner.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }))
    }

    /// Cpdag/Pag ConditionalEffect via the same generalized-adjustment envelope as ATE.
    pub(super) fn execute_class_conditional(
        &self,
        data: &TabularData,
        query: &antecedent_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identifier, estimator) = self.resolve_class_conditional_pair();
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        match self.graph.class() {
            GraphClass::Pag => {
                let pag = physical.static_pag().ok_or_else(|| CausalError::Compile {
                    message: "PAG ConditionalEffect execute missing resolved static PAG".into(),
                })?;
                let (envelope, identification, identify_cached) =
                    if let Some(cache) = self.pag_identification_cache.as_deref() {
                        (cache.envelope.clone(), cache.identification.clone(), true)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = identify_pag(identifier_id, pag, &query.inner)?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (envelope, identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identification,
                    identify_cached,
                    super::pag_path::pag_envelope_diagnostic(&envelope),
                    identifier_id,
                    estimator_id,
                    started,
                    "pag",
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::Envelope {
                            envelope,
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
                    if let Some(cache) = self.cpdag_identification_cache.as_deref() {
                        (cache.envelope.clone(), cache.identification.clone(), true)
                    } else {
                        report_identify_compute(ctx);
                        let envelope = identify_cpdag(identifier_id, cpdag, &query.inner)?;
                        let identification = envelope_to_identification_result_for(
                            &envelope,
                            CausalQuery::ConditionalEffect(query.clone()),
                        );
                        (envelope, identification, false)
                    };
                self.finish_class_conditional(
                    data,
                    query,
                    physical,
                    ctx,
                    &envelope,
                    identification,
                    identify_cached,
                    super::pag_path::cpdag_envelope_diagnostic(&envelope),
                    identifier_id,
                    estimator_id,
                    started,
                    "cpdag",
                )
                .map(|result| {
                    self.attach_certificate(
                        result,
                        crate::Identification::CpdagEnvelope {
                            envelope,
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
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<G>,
        identification: IdentificationResult,
        identify_cached: bool,
        envelope_diag: Diagnostic,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        started: Instant,
        class_tag: &str,
    ) -> Result<StudyResult, CausalError> {
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
            if matches!(self.inference, InferenceMode::Bayesian(_))
                || matches!(estimator_id, EstimatorId::BayesianConditional)
            {
                return self.execute_pag_nonidentified_prior(
                    &query.inner,
                    physical,
                    ctx,
                    envelope,
                    identification,
                    identify_cached,
                    started,
                    envelope_diag,
                    class_tag,
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
        if matches!(self.inference, InferenceMode::Bayesian(_))
            || matches!(estimator_id, EstimatorId::BayesianConditional)
        {
            return self.execute_pag_bayesian(
                data,
                &query.inner,
                physical,
                ctx,
                envelope,
                identification,
                identify_cached,
                started,
                envelope_diag,
                class_tag,
            );
        }

        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.inner.outcome,
            &query.inner.outcome_functional,
        )?;
        let mut diagnostics = vec![envelope_diag];
        let mut weighted_ate = 0.0;
        let mut se_items = Vec::new();
        let mut atom_ifs = Vec::new();
        let mut atom_weights = Vec::new();
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        let mut grid_atoms = Vec::new();
        let mut outcomes = vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()];
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
                est.estimate(&data_est, &estimand, &mean_query, ctx).map_err(CausalError::from)?;
            let w = case.weight.0;
            outcomes[i] = ClassAtomOutcome::Evaluated(estimate.ate);
            weighted_ate += w * estimate.ate;
            se_items.push((w, estimate.se_analytic));
            if let Some(inf) = estimate
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, &query.inner, &estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
            }
            grid_atoms.push((w, estimand.clone()));
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: None,
                original: estimate,
            });
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: format!("{class_tag} ConditionalEffect envelope had no estimable cases"),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: format!("{class_tag} ConditionalEffect envelope missing estimand"),
        })?;
        let n_contributing = se_items.len();
        let se = if atom_ifs.len() == n_contributing {
            mix_static_envelope_se(&atom_ifs, &atom_weights)
        } else {
            mix_weighted_analytic_se(se_items)
        };
        let mut estimate = EffectEstimate::new(
            weighted_ate / total_w,
            se,
            assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        if atom_ifs.len() == n_contributing {
            if let Some(inf) = mixed_static_influence(&atom_ifs, &atom_weights) {
                estimate.influence = Some(inf);
            }
        }
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
        let (refutations, na_diagnostics) = run_envelope_effect_refuters(
            data,
            &query.inner,
            &refute_atoms,
            &mut refute_ws,
            ctx,
            self.refute,
            estimator_id.as_str(),
            &self.custom_validators,
            None,
            self.split.as_ref(),
            None,
        )?;
        diagnostics.extend(na_diagnostics);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic));
        // Completions that disagree publish their identified set, exactly as the
        // class ATE arm does; a point-identified envelope stays a point.
        let structural_response =
            super::pag_path::static_class_structural_mixture(envelope, &outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(super::pag_path::static_class_identified_set_diagnostic(
                mixture, class_tag,
            ));
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.inner.treatment,
            outcome: query.inner.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                structural_response,
                ..Default::default()
            },
        }))
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
        let (mediation, posterior) = if let InferenceMode::Bayesian(cfg) = &self.inference {
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
                data,
                graph,
                query,
                identification.required_assumptions.clone(),
                &[],
                &est,
                identification.status,
                bridge,
                ctx,
            )?;
            (mediation, Some(posterior))
        } else {
            (
                antecedent_estimate::estimate_static_mediation(
                    data,
                    graph,
                    query,
                    identification.required_assumptions.clone(),
                    self.bootstrap_replicates,
                    &[],
                    ctx,
                )?,
                None,
            )
        };
        let estimate = mediation.effect.clone();
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let (cancelled, early_stopped) =
            (estimate.bootstrap_cancelled, estimate.bootstrap_early_stopped);
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
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::PathSpecificNatural,
            estimator_id: EstimatorId::StaticMediationLinear,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: bootstrap_ok,
            cancelled,
            early_stopped,
            extras: IdentifiedExecuteExtras {
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            },
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_tiered_average(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        background: &antecedent_graph::TieredBackground,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
    ) -> Result<StudyResult, CausalError> {
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
                physical,
                ctx,
                identification,
                identifier_id,
                estimator_id,
                identify_cached,
                started,
            );
        }
        let estimand = identification.estimands[0].clone();
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.outcome,
            &query.outcome_functional,
        )?;
        let mut ws = StaticEstimateWorkspaces::default();
        let spec = self.estimator_spec.clone().unwrap_or(EstimatorSpec::Default(estimator_id));
        let estimate = estimate_static_effect(
            &spec,
            &data_est,
            &estimand,
            query,
            identification.required_assumptions.clone(),
            self.bootstrap_replicates,
            self.overlap_policy,
            self.population_registry.as_ref(),
            ctx,
            &mut ws,
        )?;
        let mut extra_diagnostics = Vec::new();
        let estimate = super::super::helpers::attach_average_functional_grid(
            estimate,
            data,
            query,
            &estimand,
            &mut extra_diagnostics,
            estimator_id,
            self,
            ctx.rng.master_seed(),
        )?;
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator_id.as_str(),
            &self.custom_validators,
            None,
        )?;
        extra_diagnostics.extend(na_diagnostics);
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }))
    }

    /// Execute the prepared two-scenario Unknown-tier product without looking
    /// up or rebuilding its identification from the retained Study cache.
    pub(crate) fn execute_checked_unknown_tiered_average(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        background: &antecedent_graph::TieredBackground,
        identification: IdentificationResult,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
    ) -> Result<StudyResult, CausalError> {
        if background.within_tier != antecedent_graph::WithinTier::Unknown
            || identification.query != CausalQuery::AverageEffect(query.clone())
            || identification.estimands.len() != 2
            || identification.status != IdentificationStatus::GraphDependent
        {
            return Err(CausalError::Conflict {
                what: "checked Unknown-tier operation",
                detail: "retained target, tier interpretation, or scenario envelope changed".into(),
            });
        }
        self.finish_tiered_unknown(
            data,
            query,
            physical,
            ctx,
            identification,
            identifier_id,
            estimator_id,
            true,
            Instant::now(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_tiered_unknown(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        identification: IdentificationResult,
        identifier_id: IdentifierId,
        estimator_id: EstimatorId,
        identify_cached: bool,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            query.outcome,
            &query.outcome_functional,
        )?;
        let spec = self.estimator_spec.clone().unwrap_or(EstimatorSpec::Default(estimator_id));
        let mut atom_ifs = Vec::new();
        let mut atom_ws = Vec::new();
        let mut scenarios = Vec::new();
        let mut primary = None;
        let mut assumptions = identification.required_assumptions.clone();
        for estimand in &identification.estimands {
            let mut ws = StaticEstimateWorkspaces::default();
            let estimate = estimate_static_effect(
                &spec,
                &data_est,
                estimand,
                query,
                identification.required_assumptions.clone(),
                self.bootstrap_replicates,
                self.overlap_policy,
                self.population_registry.as_ref(),
                ctx,
                &mut ws,
            )?;
            scenarios.push((
                scenarios.len() as u64,
                antecedent_core::ResponseValue::Scalar(estimate.ate),
            ));
            if let Some(inf) = estimate
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, estimand, inf))
            {
                atom_ifs.push(inf);
                atom_ws.push(0.5);
            }
            if primary.is_none() {
                primary = Some(estimand.clone());
                assumptions = estimate.assumptions;
            }
        }
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
            estimate.scenario_intervals = Some(
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
        estimate.scenario_effects = Some(values.into());
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics,
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras { structural_response, ..Default::default() },
        }))
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
