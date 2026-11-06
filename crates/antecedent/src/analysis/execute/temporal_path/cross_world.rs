//! Temporal cross-world execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianGCompWorkspace, BayesianGComputationAte, CausalError, CausalPosterior,
    CausalQuery, CrossWorldContrast, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecutionContext, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    Intervention, OverlapPolicy, PhysicalExecutionPlan, RefuteSuite, RouteDefaults, SlopeSource,
    Strategy, StudyResult, TemporalBackdoorIdentifier, TemporalDag, TemporalEffectQuery,
    TemporalLinearAdjustment, TimeSeriesData, VariableId, bayesian_temporal_gcomp,
    cross_world_contrast, cross_world_point, effect_from_posterior, elapsed_ns, point_uncertainty,
    require_identified, require_path_chain, resolve_bayesian_prior_with_conflict,
    template_scoped_cross_world, with_conflict_summary,
};

impl super::super::Study {
    /// The isotropic conjugate prior scale a temporal cross-world fit uses
    /// under Bayesian inference; a mapped coefficient prior is refused.
    fn cross_world_prior_scale(&self) -> Result<Option<f64>, CausalError> {
        match &self.settings.inference {
            InferenceMode::Bayesian(cfg) => {
                if cfg.has_transfer() {
                    return Err(CausalError::Unsupported {
                        message: "temporal cross-world effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not supported",
                    });
                }
                Ok(Some(cfg.prior_scale))
            }
            InferenceMode::Frequentist => Ok(None),
        }
    }

    /// The identification of the cross-world probe, which depends on the query and the
    /// structure alone, for compile to retain with the program. `None` leaves identification
    /// (and its refusal) to execution.
    pub(crate) fn retain_cross_world_identification(
        &self,
        plan: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<Option<crate::planner::lowered::RetainedExecutionProof>, CausalError> {
        use crate::accepted::GraphRef;
        let Some(probe) =
            super::super::temporal_class_point::cross_world_identification_probe(&self.query)
        else {
            return Ok(None);
        };
        Ok(match self.graph.graph_ref() {
            GraphRef::TemporalDag(graph) => {
                let Ok(found) = TemporalBackdoorIdentifier::new().identify_temporal(graph, &probe)
                else {
                    return Ok(None);
                };
                if require_identified(&found.result).is_err() || found.result.estimands.is_empty() {
                    return Ok(None);
                }
                Some(crate::planner::lowered::RetainedExecutionProof::CrossWorld {
                    identification: found.result,
                    indexer: found.indexer,
                })
            }
            GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_) => {
                let Ok(identifier_id) = plan.binding().identifier_or(RouteDefaults::PAG) else {
                    return Ok(None);
                };
                let bundle = match self.identify_temporal_class(identifier_id, &probe, ctx) {
                    Ok(bundle) => bundle,
                    Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. })) => {
                        return Err(error);
                    }
                    Err(_) => return Ok(None),
                };
                Some(crate::planner::lowered::RetainedExecutionProof::CrossWorldClass {
                    envelope: Arc::clone(&bundle.envelope),
                })
            }
            _ => None,
        })
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one cross-world execution: the point, the draws, and the claim stay in one function"
    )]
    pub(in crate::analysis::execute) fn execute_temporal_cross_world(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "the temporal cross-world contrast",
        )?;
        let started = Instant::now();
        if self.settings.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not supported",
            });
        }
        let prior_scale = self.cross_world_prior_scale()?;
        let CrossWorldContrast { treatment, outcome, mediator, active, control } =
            cross_world_contrast(&self.query)?;
        if let CausalQuery::PathSpecific(path) = &self.query {
            require_path_chain(graph, &path.path_nodes)?;
        }
        let probe = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        // The probe depends only on the query and the graph, so it is identified once, at
        // compile, and the program retains it; a plan that retained none identifies here.
        let retained =
            match crate::analysis::lowering::plan_retained_execution_proof(self, physical) {
                Some(crate::planner::lowered::RetainedExecutionProof::CrossWorld {
                    identification,
                    indexer,
                }) => Some((identification.clone(), indexer.clone())),
                _ => None,
            };
        let (identification, indexer) = if let Some(retained) = retained {
            retained
        } else {
            let found = TemporalBackdoorIdentifier::new()
                .identify_temporal(graph, &probe)
                .map_err(CausalError::from)?;
            (found.result, found.indexer)
        };
        require_identified(&identification)?;
        let estimand = identification.estimands.first().cloned().ok_or_else(|| {
            CausalError::Compile { message: "temporal cross-world proof has no estimand".into() }
        })?;
        // Every route fits under the back-door set of the declared graph: a confounder of the
        // treatment and the outcome is adjusted for in the direct slope and in each edge of a
        // path, never dropped.
        let used = &estimand;
        let contrast = CrossWorldContrast { treatment, outcome, mediator, active, control };
        // Every regression the point reads is recorded, so the interval refits the
        // same procedure (see `temporal_point_uncertainty`).
        let atom = point_uncertainty::SlopeAtom::new(
            data,
            used,
            &indexer,
            (probe.try_treatment_offset()?, probe.outcome_offset()?),
            ctx.kernel_policy,
            prior_scale,
        );
        let point = {
            let fit =
                |outcome: VariableId, treatments: &[VariableId], contemporaneous: &[VariableId]| {
                    atom.slopes(outcome, treatments, contemporaneous)
                };
            cross_world_point(&self.query, &fit, contrast)?
        };
        let structural_span = super::super::temporal_class_block_span(std::iter::once(&indexer));
        let proof = identification.clone();
        let proof_estimand = estimand.clone();
        let identification = template_scoped_cross_world(
            identification,
            &indexer,
            self.query.clone(),
            CrossWorldContrast { treatment, outcome, mediator, active, control },
        );
        let estimand = identification.estimands[0].clone();
        if matches!(&self.query, CausalQuery::Counterfactual(_))
            && self.settings.inference.is_bayesian()
        {
            let mut contrast_query = probe.clone();
            contrast_query.control =
                Intervention::set(treatment, antecedent_core::Value::f64(control));
            contrast_query.active =
                Intervention::set(treatment, antecedent_core::Value::f64(active));
            let (estimate, posterior) = self.temporal_counterfactual_posterior(
                data,
                &proof,
                &proof_estimand,
                &indexer,
                &contrast_query,
                ctx,
            )?;
            let finish = IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::BayesianTemporalGcomp,
                ),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started));
            return Ok(self.finish_cross_world_posterior(finish, posterior));
        }
        let frozen = [atom.freeze()];
        let query = &self.query;
        let evaluate = |_: usize, source: &dyn SlopeSource| {
            let fit =
                |outcome: VariableId, treatments: &[VariableId], contemporaneous: &[VariableId]| {
                    source.slopes(outcome, treatments, contemporaneous)
                };
            cross_world_point(query, &fit, contrast).map(|value| vec![value])
        };
        let assumptions = identification.required_assumptions.clone();
        if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            let draws = point_uncertainty::point_posterior_draws(
                &frozen,
                point_uncertainty::PointPosteriorPlan {
                    prior_scale: cfg.prior_scale,
                    draws: cfg.n_draws,
                    structural_span,
                    stream_base: point_uncertainty::CROSS_WORLD_POSTERIOR_STREAM,
                },
                ctx,
                evaluate,
            )?;
            let column: Vec<f64> = draws.atom_draws[0].iter().map(|draw| draw[0]).collect();
            let posterior = point_uncertainty::scalar_posterior(
                &column,
                "cross_world_contrast",
                identification.status,
                assumptions.clone(),
                draws.tempering,
            )?;
            let sd = posterior.summaries.sd[0];
            let estimate =
                EffectEstimate::new(point, sd, assumptions, OverlapPolicy::ExplicitOverride);
            let n_draws = u32::try_from(column.len()).unwrap_or(u32::MAX);
            let finish = IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::TemporalLinearAdjustment,
                ),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started))
            .extra_diagnostics(vec![
                Diagnostic::new(
                    "estimate.temporal.cross_world",
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Info,
                    "published value is the linear unfolded cross-world contrast at the \
                     posterior-mean coefficients; the interval is the equal-tailed interval of \
                     its joint posterior draws",
                ),
                point_uncertainty::posterior_diagnostic(&draws, "temporal cross-world contrast"),
            ])
            .extras(IdentifiedExecuteExtras {
                n_draws: Some(n_draws),
                posterior: Some(posterior),
                ..Default::default()
            });
            return Ok(self.finish_identified_execute(finish));
        }
        let replicates = point_uncertainty::point_block_replicates(
            &frozen,
            point_uncertainty::PointBlockPlan {
                replicates: self.settings.bootstrap_replicates,
                stream_base: point_uncertainty::CROSS_WORLD_BLOCK_STREAM,
                structural_span,
            },
            false,
            ctx,
            evaluate,
            |values| values.first().cloned(),
        )?;
        let family = if matches!(&self.query, CausalQuery::PathSpecific(_)) {
            antecedent_estimate::CircularBlockFamily::Mediation
        } else {
            antecedent_estimate::CircularBlockFamily::SingleWindow
        };
        let estimate = point_uncertainty::block_scalar_estimate(
            point,
            replicates.as_ref(),
            assumptions,
            family,
        );
        let finish = IdentifiedExecuteFinish::new(
            physical,
            identification,
            estimand,
            estimate,
            Strategy::new(
                IdentifierId::TemporalBackdoorUnfolded,
                EstimatorId::TemporalLinearAdjustment,
            ),
            treatment,
            outcome,
        )
        .wall_time_ns(elapsed_ns(started))
        .bootstrap_replicates_ok(
            replicates.as_ref().map(point_uncertainty::PointBlockReplicates::completed),
        )
        .cancelled(replicates.as_ref().is_some_and(|r| r.draws.cancelled));
        Ok(self.finish_cross_world_block(finish, replicates.as_ref()))
    }

    /// Finishes the frequentist cross-world contrast with its shared circular-block
    /// interval (point only when no replicates were requested).
    fn finish_cross_world_block(
        &self,
        finish: IdentifiedExecuteFinish<'_>,
        replicates: Option<&point_uncertainty::PointBlockReplicates>,
    ) -> StudyResult {
        let family = if matches!(&self.query, CausalQuery::PathSpecific(_)) {
            antecedent_estimate::CircularBlockFamily::Mediation
        } else {
            antecedent_estimate::CircularBlockFamily::SingleWindow
        };
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal.cross_world",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the linear unfolded cross-world point; its interval refits every \
             regression of the contrast on shared circular blocks of series times",
        )];
        diagnostics.extend(point_uncertainty::block_diagnostics(
            replicates,
            self.settings.bootstrap_replicates,
            family,
            "every regression of the contrast is refit on each replicate",
        ));
        let mut result = self.finish_identified_execute(finish.extra_diagnostics(diagnostics));
        if replicates.is_none() {
            result.extras.interval = None;
        }
        result.posterior = None;
        result
    }

    /// Finishes the posterior cross-world contrast: the credible interval of the generalized
    /// posterior of the contrast slope.
    fn finish_cross_world_posterior(
        &self,
        finish: IdentifiedExecuteFinish<'_>,
        posterior: CausalPosterior,
    ) -> StudyResult {
        let n_draws = u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX);
        let mut result = self.finish_identified_execute(finish.extras(IdentifiedExecuteExtras {
            n_draws: Some(n_draws),
            posterior: Some(posterior),
            ..Default::default()
        }));
        result.diagnostics.push(Diagnostic::new(
            "estimate.temporal.cross_world",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the posterior mean of the linear unfolded cross-world \
             contrast; the interval is the credible interval of the long-run-tempered \
             generalized posterior of the contrast slope, parameter uncertainty only",
        ));
        result
    }

    /// Posterior of the unit counterfactual contrast of a linear additive `TemporalDag`.
    ///
    /// The structural equation `Y_t = a + b X_{t-1} + Z'g + e_t` has a noise `e_t` that is
    /// recovered exactly from the factual row and replayed unchanged in the counterfactual
    /// world, so it cancels: every unit's contrast is `b (active - control)` and the
    /// outcome carries no uncertainty beyond the parameters. The posterior is therefore the
    /// posterior of `b (active - control)` on the identified unfolded design, the same
    /// long-run-tempered generalized posterior (Gaussian likelihood tempered by the
    /// variance ratio of the treatment score, the prior at full weight) that the Bayesian
    /// pulse publishes. The coefficient prior is isotropic: a mapped prior is refused by the
    /// caller.
    fn temporal_counterfactual_posterior(
        &self,
        data: &TimeSeriesData,
        proof: &IdentificationResult,
        estimand: &IdentifiedEstimand,
        indexer: &antecedent_core::TemporalIndexer,
        contrast_query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, CausalPosterior), CausalError> {
        let InferenceMode::Bayesian(cfg) = &self.settings.inference else {
            return Err(CausalError::unsupported_inference(
                "a temporal counterfactual posterior requires Bayesian inference",
            ));
        };
        let mut temporal = TemporalLinearAdjustment::new();
        temporal.inner.inference.bootstrap_replicates = 0;
        temporal.inner.overlap = OverlapPolicy::ExplicitOverride;
        let prepared = temporal
            .prepare(data, estimand, contrast_query, indexer, None, &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        let names = antecedent_estimate::temporal_coefficient_names(
            data,
            estimand,
            contrast_query,
            indexer,
        )
        .map_err(CausalError::from)?;
        let problem = BayesianGComputationAte::from_prepared_temporal(&prepared, names)
            .map_err(CausalError::from)?;
        let (prior, conflict) = resolve_bayesian_prior_with_conflict(cfg, &problem, Some(ctx))?;
        let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
        bayes.inner.prior = prior;
        let mut workspace = BayesianGCompWorkspace::default();
        let mut posterior =
            bayes.fit(&problem, proof.status, &mut workspace, ctx).map_err(CausalError::from)?;
        if let Some(summary) = conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        posterior.assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: Arc::from("temporal.counterfactual.shared_additive_noise"),
                    description: Arc::from(
                        "linear additive-noise structural equation for the outcome: the factual \
                         noise is recovered exactly and replayed in the counterfactual world, so \
                         it cancels and every unit's contrast equals the slope times the \
                         contrast. The posterior is parameter uncertainty only; the outcome adds \
                         none. A nonlinear, interacting or categorical mechanism is outside this \
                         route",
                    ),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal.counterfactual.posterior"),
            },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        let estimate = effect_from_posterior(&posterior)?
            .with_n_obs(u64::try_from(prepared.design.nrows).unwrap_or(u64::MAX));
        Ok((estimate, posterior))
    }
}
