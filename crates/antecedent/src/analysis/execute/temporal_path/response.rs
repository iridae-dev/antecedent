//! Fixed-graph temporal response and sequence execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianGComputationAte, CausalError, CausalPosterior, CausalQuery, CausalResponse,
    DerivativeResponseInputs, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate,
    EstimatorId, ExecuteRun, ExecutionContext, IdentificationBasis, IdentificationResult,
    IdentificationStatus, IdentifiedClaim, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant, ObservationBootstrap,
    ObservationMechanismEstimator, ObservationSpec, OverlapPolicy, PhysicalExecutionPlan,
    RefuteSuite, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    SEQUENCE_LINEAR_SEM, SequenceMechanisms, SequenceResponseInputs, Strategy, StudyResult,
    TemporalDag, TemporalIndexer, TemporalResponseEstimator, TimeSeriesData, TupleBandSource,
    VariableId, aggregate_temporal_horizon_evidence, append_temporal_observation_assumptions,
    apply_observation_bootstrap, apply_temporal_observation_result, apply_tuple_bootstrap_band,
    bayesian_gcomp, bootstrap_observation_adjusted_temporal_response, bootstrap_sequence_response,
    contemporaneous_adjustment_variables, elapsed_ns, enforce_temporal_response_memory_budget,
    horizon_adjustment_sets_differ, identified_horizon_entries, identify_cached_diagnostic,
    lag_aligned_analysis_rows, named_adjustment_keys, posterior_note_diagnostics, provenance_ids,
    push_conflict_diagnostics, push_unique_diagnostic, report_identify_compute, require_identified,
    resolve_temporal_response_prior, sequential_linear_sem_assumption,
    temporal_observation_outcome_regressors,
};

/// What the observation preparation of a temporal response reads.
#[derive(Clone, Copy)]
struct ResponseObservationInput<'a> {
    data: &'a TimeSeriesData,
    graph: &'a TemporalDag,
    query: &'a ResponseQuery,
    outcome: VariableId,
    aligned: &'a [&'a crate::analysis::prepared::CachedTemporalHorizonIdentification],
    observed_bayes: bool,
}

/// The observation-adjusted series and working query of a temporal response;
/// the series and adjustment are absent when the data is complete or the
/// observation is modelled inside the Bayesian fit.
struct ResponseObservation {
    series: Option<TimeSeriesData>,
    adjusted: Option<antecedent_estimate::ObservationAdjustedOutcome>,
    working_query: ResponseQuery,
}

/// Everything the temporal response surface estimation reads.
#[derive(Clone, Copy)]
struct TemporalResponseEstimation<'a> {
    data: &'a TimeSeriesData,
    source_data: &'a TimeSeriesData,
    graph: &'a TemporalDag,
    query: &'a ResponseQuery,
    working_query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    aligned: &'a [&'a crate::analysis::prepared::CachedTemporalHorizonIdentification],
    identifications: &'a [(&'a IdentifiedEstimand, &'a TemporalIndexer)],
    status: IdentificationStatus,
    assumptions: &'a antecedent_core::AssumptionSet,
    observation_adjusted: Option<&'a antecedent_estimate::ObservationAdjustedOutcome>,
    observed_bayes: bool,
    treatment: VariableId,
    outcome: VariableId,
}

/// A temporal response surface with the prior-conflict summary and observed
/// posterior its estimation produced.
struct EstimatedTemporalResponse {
    response: CausalResponse,
    conflict_summary: Option<antecedent_prob::ConflictSummary>,
    observed_posterior: Option<antecedent_estimate::CausalPosterior>,
}

/// The diagnostics a temporal response result carries, in publication order.
fn temporal_response_diagnostics(
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    response: &CausalResponse,
    conflict_summary: Option<&antecedent_prob::ConflictSummary>,
    identify_cached: bool,
) -> Vec<Diagnostic> {
    let scalar = super::super::response_path::response_scalar_summary(response).0;
    let mut diagnostics = Vec::new();
    for entry in aligned {
        diagnostics.extend(entry.identification.diagnostics.iter().cloned());
    }
    if horizon_adjustment_sets_differ(aligned) {
        diagnostics.push(Diagnostic::new(
            "identify.temporal_response.horizon_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "adjustment sets differ across requested horizons; each cell uses I(h) \
             identified for that horizon, not a shared max-horizon set",
        ));
    }
    if identify_cached {
        diagnostics.push(identify_cached_diagnostic());
    }
    if let Some(summary) = conflict_summary {
        push_conflict_diagnostics(&mut diagnostics, summary);
    }
    diagnostics.push(Diagnostic::new(
        "refute.temporal_response.skipped",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "scalar ATE refuters are not applicable to a function-valued temporal response",
    ));
    if !scalar.is_finite() {
        diagnostics.push(Diagnostic::new(
            "estimate.response.no_scalar_summary",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "this response is function-valued; the scalar effect summary is not applicable \
             and the result is carried by the response payload",
        ));
    }
    for warning in &response.support.warnings {
        diagnostics.push(warning.clone());
    }
    diagnostics
}

/// The per-horizon surface of a Sequence response: pointwise means and bands,
/// each horizon's identification record, and the posteriors kept for
/// disclosure.
struct SequenceHorizonSurface {
    mean: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    horizons: Vec<antecedent_core::HorizonIdentification>,
    uncertainty_complete: bool,
    last_posterior: Option<CausalPosterior>,
    /// Every horizon's posterior notes are kept (not just the scalar artifact's),
    /// so a `tempering_capped` / `tempering_inestimable` disclosure raised at an
    /// earlier horizon still reaches the result even when only the last horizon's
    /// posterior is attached as the scalar summary.
    horizon_posteriors: Vec<CausalPosterior>,
    bootstrap_cancelled: bool,
}

/// What a Sequence response is finished from once its band is final.
struct SequenceFinish {
    response: CausalResponse,
    horizon_posteriors: Vec<CausalPosterior>,
    last_posterior: Option<CausalPosterior>,
    bootstrap_cancelled: bool,
    attach_scalar_posterior: bool,
}

/// Fits the sequential mechanisms at every requested horizon and collects the
/// pointwise surface (Wald bounds, or posterior quantiles under Bayesian
/// inference).
fn estimate_sequence_horizons(
    inputs: &SequenceResponseInputs<'_>,
    outcome: VariableId,
    bayes: Option<&BayesianGComputationAte>,
    assumptions: &antecedent_core::AssumptionSet,
    ctx: &ExecutionContext,
) -> Result<SequenceHorizonSurface, CausalError> {
    let SequenceResponseInputs { data, graph, temporal, overlays, aligned, .. } = *inputs;
    let horizon_count = temporal.horizons.len();
    let mut surface = SequenceHorizonSurface {
        mean: Vec::with_capacity(horizon_count),
        lower: Vec::with_capacity(horizon_count),
        upper: Vec::with_capacity(horizon_count),
        horizons: Vec::with_capacity(horizon_count),
        uncertainty_complete: true,
        last_posterior: None,
        horizon_posteriors: Vec::with_capacity(horizon_count),
        bootstrap_cancelled: false,
    };
    let attach_scalar_posterior = horizon_count == 1;
    let z = antecedent_stats::NORMAL_Q975;
    for (horizon_steps, entry) in temporal.horizons.iter().copied().zip(aligned.iter()) {
        require_identified(&entry.identification)?;
        let outcome_offset = i32::try_from(horizon_steps.saturating_sub(1)).unwrap_or(i32::MAX);
        let (effect, posterior) = antecedent_estimate::estimate_sequence_mechanisms(
            &antecedent_estimate::SequentialProblem {
                data,
                graph,
                indexer: &entry.indexer,
                estimand: &entry.estimand,
            },
            &antecedent_estimate::SequenceTarget { outcome, outcome_offset, overlays },
            antecedent_estimate::SequentialInference { status: entry.identification.status, assumptions: assumptions.clone(), bootstrap_replicates: // Frequentist bands: the joint tuple bootstrap below.
            0, bayesian: bayes },
            ctx,
        )
        .map_err(CausalError::from)?;
        surface.bootstrap_cancelled |= effect.uncertainty.bootstrap.cancelled;
        let (point, lo, hi) = if let Some(post) = posterior.as_ref() {
            (post.summaries.mean[0], post.summaries.q025[0], post.summaries.q975[0])
        } else {
            let se = effect.se_bootstrap().unwrap_or(effect.se_analytic());
            (effect.ate, effect.ate - z * se, effect.ate + z * se)
        };
        if let Some(post) = posterior.as_ref() {
            surface.horizon_posteriors.push(post.clone());
        }
        if attach_scalar_posterior {
            surface.last_posterior = posterior;
        }
        surface.mean.push(point);
        surface.uncertainty_complete &= lo.is_finite() && hi.is_finite();
        surface.lower.push(lo);
        surface.upper.push(hi);
        surface.horizons.push(antecedent_core::HorizonIdentification {
            horizon: horizon_steps,
            status: entry.identification.status,
            method: Arc::clone(&entry.estimand.method),
            adjustment: Arc::from(named_adjustment_keys(entry)),
        });
    }
    Ok(surface)
}

/// The support report of a Sequence overlay response: extrapolative by
/// construction, with the overlays recorded and the band disclosures attached.
fn sequence_overlay_support(
    overlays: &[antecedent_estimate::SequentialMechanismOverlay],
    temporal: &antecedent_core::TemporalResponseSpec,
    uncertainty_complete: bool,
) -> antecedent_core::SupportReport {
    let first_horizon = f64::from(temporal.horizons.first().copied().unwrap_or(1));
    let last_horizon = f64::from(temporal.horizons.last().copied().unwrap_or(1));
    let mut support = antecedent_core::SupportReport {
        // Joint longitudinal support is not established by univariate
        // ranges. Until a sequential-positivity diagnostic exists, do not
        // label an arbitrary overlay schedule empirically supported.
        status: antecedent_core::SupportStatus::Extrapolative,
        query_region: antecedent_core::SupportRegion {
            minima: Arc::from([first_horizon]),
            maxima: Arc::from([last_horizon]),
        },
        diagnostics: vec![antecedent_core::SupportDiagnostic {
            id: Arc::from("response.temporal.sequence_overlay"),
            values: Arc::from(
                overlays
                    .iter()
                    .flat_map(|overlay| {
                        [
                            f64::from(overlay.node.variable.raw()),
                            f64::from(overlay.node.offset),
                            if overlay.node.level.is_some() { 1.0 } else { 0.0 },
                            overlay.node.level.unwrap_or(0.0),
                            overlay.node.shift,
                            overlay.multiplier,
                            if overlay.bounds.is_some() { 1.0 } else { 0.0 },
                            overlay.bounds.map_or(0.0, |bounds| bounds.0),
                            overlay.bounds.map_or(0.0, |bounds| bounds.1),
                        ]
                    })
                    .collect::<Vec<_>>(),
            ),
            detail: Arc::from(
                "sequential Sequence overlays as \
                 [variable, offset, is_fixed_level, level_or_zero, shift, multiplier, \
                 has_bounds, lower_or_zero, upper_or_zero, …]; \
                 transforms apply to deterministic propagated means, not stochastic draws",
            ),
            scope: antecedent_core::DiagnosticScope::Global,
        }],
        warnings: vec![Diagnostic::new(
            "response.temporal.sequence_joint_support_unassessed",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "joint longitudinal positivity for the requested Sequence overlay is not \
             estimated; support is conservatively marked extrapolative",
        )],
        point_status: Some(Arc::from(vec![
            antecedent_core::SupportStatus::Extrapolative;
            temporal.horizons.len()
        ])),
    };
    antecedent_estimate::publish_simultaneous_band(
        &mut support,
        Err(antecedent_estimate::EstimationError::unsupported(
            "Sequence overlays publish a simultaneous band only from the Frequentist joint \
             tuple bootstrap (request bootstrap replicates); Bayesian Sequence overlays are \
             sampled one horizon at a time, so there is no joint draw surface",
        )),
        "",
    );
    if !uncertainty_complete {
        support.warnings.push(Diagnostic::new(
            "response.temporal.sequence_uncertainty_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "a complete pointwise Sequence band was unavailable; uncertainty is omitted \
             rather than represented by non-finite endpoints",
        ));
    }
    support
}

impl super::super::Study {
    #[allow(
        clippy::too_many_lines,
        reason = "one temporal response execution: identification, estimation, refutation and finish run in sequence"
    )]
    pub(in crate::analysis::execute) fn execute_temporal_response(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(data, "the temporal response")?;
        let started = Instant::now();
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal response route requires TemporalResponseSpec".into(),
            });
        };
        enforce_temporal_response_memory_budget(
            query,
            temporal,
            self.settings.bootstrap_replicates,
            ctx,
        )?;
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let observed_bayes =
            self.settings.inference.is_bayesian() && query.observation != ObservationSpec::Complete;
        let (cache, identify_cached) = self.temporal_response_identification(
            graph,
            query,
            temporal,
            (treatment, outcome),
            physical,
            ctx,
        )?;
        let aligned = identified_horizon_entries(&cache, temporal)?;
        let first = aligned[0];
        let (aggregate_status, mut aggregate_assumptions) =
            aggregate_temporal_horizon_evidence(aligned.iter().map(|entry| &entry.identification))?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();
        let run = ExecuteRun { physical, ctx, started, identify_cached };
        if super::super::response_functional_is_derivative(&query.functional) {
            let indexer = &first.indexer;
            return self.execute_temporal_derivative_response(
                DerivativeResponseInputs { data, query, temporal, indexer, treatment, outcome },
                IdentifiedClaim { identification, estimand },
                run,
            );
        }
        let identifications: Vec<_> =
            aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();

        query.require_supported_temporal_observation()?;
        let observation = self.prepare_temporal_response_observation(
            &ResponseObservationInput {
                data,
                graph,
                query,
                outcome,
                aligned: &aligned,
                observed_bayes,
            },
            &mut identification,
            &mut aggregate_assumptions,
        )?;
        let source_data = data;
        let data = observation.series.as_ref().unwrap_or(data);

        let mut temporal_budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let plan = antecedent_estimate::plan_from_response_query(query, &mut temporal_budget, ctx)
            .map_err(CausalError::from)?;
        let overlays = if observed_bayes {
            None
        } else {
            match &plan {
                Some(plan) => plan
                    .admitted_mechanism_overlays(&mut temporal_budget, ctx)
                    .map_err(CausalError::from)?,
                None => None,
            }
        };
        if let Some(overlays) = overlays {
            return self.execute_temporal_sequence_response(
                SequenceResponseInputs {
                    data,
                    source_data,
                    graph,
                    query,
                    temporal,
                    overlays: &overlays,
                    aligned: &aligned,
                    observation_adjusted: observation.adjusted.as_ref(),
                },
                IdentifiedClaim { identification, estimand },
                (treatment, outcome),
                IdentificationBasis {
                    status: aggregate_status,
                    assumptions: &aggregate_assumptions,
                },
                run,
            );
        }

        let estimated = self.estimate_temporal_response(
            &TemporalResponseEstimation {
                data,
                source_data,
                graph,
                query,
                working_query: &observation.working_query,
                temporal,
                aligned: &aligned,
                identifications: &identifications,
                status: aggregate_status,
                assumptions: &aggregate_assumptions,
                observation_adjusted: observation.adjusted.as_ref(),
                observed_bayes,
                treatment,
                outcome,
            },
            ctx,
        )?;
        Ok(self.finish_temporal_response(
            data,
            &aligned,
            IdentifiedClaim { identification, estimand },
            estimated,
            (treatment, outcome),
            run,
        ))
    }

    /// The prepared identification cache of a temporal response, computing it
    /// from the graph when the study carries none.
    ///
    /// Every mechanism-overlay estimator (the observed-data Bayesian path and
    /// the frequentist/complete-data sequential g-formula path alike) reads
    /// only each horizon's identification status and adjustment set
    /// (identifiability is level-free); neither reads the schedule contrast's
    /// active level, since both simulate the query's own per-step intervention
    /// values directly off the unfolded SEM (or `overlays`, for the sequential
    /// engine) rather than off the certificate's literal. The levels are
    /// stripped before certifying, so the certificate is the schedule's
    /// structural one at the reference literal (a per-node dose schedule is
    /// certified the same way when its levels are kept; see
    /// `resolve_schedule_active_level`).
    fn temporal_response_identification(
        &self,
        graph: &TemporalDag,
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
        (treatment, outcome): (VariableId, VariableId),
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<crate::analysis::prepared::CachedTemporalIdentification>, bool), CausalError>
    {
        let mut stripped_levels = false;
        let mut temporal_budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let schedule =
            match antecedent_estimate::plan_from_response_query(query, &mut temporal_budget, ctx)
                .map_err(CausalError::from)?
            {
                Some(plan) if plan.mechanism_overlays().is_some() => {
                    let (nodes, stripped) = plan
                        .identification_schedule_without_levels(temporal, &mut temporal_budget, ctx)
                        .map_err(CausalError::from)?;
                    stripped_levels = stripped;
                    Some(nodes)
                }
                _ => None,
            };
        // Compile retains the certificate of the schedule as the query states it;
        // this route certifies it without levels, so a retained certificate is its
        // own exactly when there were no levels to strip. A prepared cache is taken
        // as before.
        let reused = self
            .reuse_temporal_identification(physical)
            .filter(|(_, prepared)| *prepared || !stripped_levels);
        Ok(if let Some(reused) = reused {
            reused
        } else {
            report_identify_compute(ctx);
            (
                Arc::new(crate::analysis::prepared::identify_temporal_response_horizons(
                    graph,
                    EstimatorId::temporal_response_for(&self.settings.inference),
                    &crate::analysis::prepared::TemporalResponseIdentificationRequest {
                        treatment,
                        outcome,
                        temporal,
                        target_population: &query.target_population,
                        schedule: schedule.as_deref(),
                        dose: crate::analysis::prepared::single_step_dose(query)?,
                    },
                    &mut temporal_budget,
                    ctx,
                )?),
                false,
            )
        })
    }

    /// Supports the response's observation spec and, for a non-Bayesian
    /// observed series, adjusts the series and the working query; records the
    /// observation assumptions on both assumption sets.
    fn prepare_temporal_response_observation(
        &self,
        input: &ResponseObservationInput<'_>,
        identification: &mut IdentificationResult,
        aggregate_assumptions: &mut antecedent_core::AssumptionSet,
    ) -> Result<ResponseObservation, CausalError> {
        let ResponseObservationInput { data, graph, query, outcome, aligned, observed_bayes } =
            *input;
        let mut working_query = query.clone();
        let mut observation_adjusted = None;
        let series_owned = if query.observation == ObservationSpec::Complete {
            None
        } else if observed_bayes {
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not supported for temporal observation",
                });
            }
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
            append_temporal_observation_assumptions(query, aggregate_assumptions);
            None
        } else {
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not supported for temporal observation",
                });
            }
            let adjustment = contemporaneous_adjustment_variables(aligned);
            let outcome_regressors =
                temporal_observation_outcome_regressors(query, graph, outcome, aligned)?;
            let (series, adjusted) = ObservationMechanismEstimator::new(self.observation_options)
                .adjust_temporal_series(data, query, &adjustment, &outcome_regressors)
                .map_err(CausalError::from)?;
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
            append_temporal_observation_assumptions(query, aggregate_assumptions);
            working_query.observation = ObservationSpec::Complete;
            working_query.observation_assumptions = Arc::from([]);
            observation_adjusted = Some(adjusted);
            Some(series)
        };
        Ok(ResponseObservation {
            series: series_owned,
            adjusted: observation_adjusted,
            working_query,
        })
    }

    /// Estimates the temporal response surface (observed-Bayes, Bayesian or
    /// frequentist g-formula), applies the observation adjustment and its
    /// bootstrap band, and restores each horizon's own identification status.
    fn estimate_temporal_response(
        &self,
        input: &TemporalResponseEstimation<'_>,
        ctx: &ExecutionContext,
    ) -> Result<EstimatedTemporalResponse, CausalError> {
        let TemporalResponseEstimation {
            data,
            source_data,
            graph,
            query,
            working_query,
            temporal,
            aligned,
            identifications,
            status: aggregate_status,
            assumptions: aggregate_assumptions,
            observation_adjusted,
            observed_bayes,
            treatment,
            outcome,
        } = *input;
        // Surface SEs follow the Study bootstrap / replicate contract so Pulse,
        // single-step Sustained, and the dose × horizon surface report comparable
        // uncertainty. Replicates = 0 keeps the analytic OLS linear-functional SE.
        let mut estimator =
            TemporalResponseEstimator::new().with_allow_gaps(self.declares_complete_case());
        if let InferenceMode::Bayesian(cfg) = &self.settings.inference
            && let Some(gate) = cfg.white_innovation
        {
            estimator = estimator.with_white_innovation(gate);
        }
        estimator.inner.inference.bootstrap_replicates =
            if observation_adjusted.is_some() { 0 } else { self.settings.bootstrap_replicates };
        // Retain the joint replicates of the surface: the bootstrap's under Frequentist
        // inference, the block-weight posterior's draws (which replace the per-horizon
        // conjugate fits) under Bayesian inference.
        let joint_horizons = self.settings.retain_joint_horizons
            && observation_adjusted.is_none()
            && !observed_bayes;
        if joint_horizons {
            estimator = estimator.with_joint_replicates(true);
            if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
                estimator = estimator
                    .with_block_weight_posterior(true)
                    .with_bootstrap_replicates(u32::try_from(cfg.n_draws).unwrap_or(u32::MAX));
            }
        }
        let mut conflict_summary = None;
        let mut observed_posterior = None;
        let mut response = if observed_bayes {
            let InferenceMode::Bayesian(cfg) = &self.settings.inference else {
                return Err(CausalError::unsupported_inference(
                    "an observed Bayesian temporal response requires Bayesian inference",
                ));
            };
            if cfg.has_transferred_source() {
                return Err(CausalError::Unsupported {message: "observed temporal mechanisms need independent per-mechanism priors; a transferred response coefficient prior has no such mapping"});
            }
            let (response, posterior) = antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response(data, graph, antecedent_estimate::TemporalIdentification { horizons: identifications, status: aggregate_status, assumptions: aggregate_assumptions.clone() }, query, &bayesian_gcomp(cfg, ctx), ctx).map_err(CausalError::from)?;
            observed_posterior = Some(posterior);
            Ok(response)
        } else if let (InferenceMode::Bayesian(cfg), false) = (&self.settings.inference, joint_horizons) {
            let mut bayes = bayesian_gcomp(cfg, ctx);
            let (resolved, conflict) = resolve_temporal_response_prior(
                cfg, data, temporal, aligned, treatment, outcome, ctx,
            )?;
            bayes.prior = resolved;
            conflict_summary = conflict;
            estimator.estimate_bayesian(data, antecedent_estimate::TemporalIdentification { horizons: identifications, status: aggregate_status, assumptions: aggregate_assumptions.clone() }, working_query, &bayes, ctx)
        } else {
            estimator.estimate(
                data,
                identifications,
                working_query,
                aggregate_status,
                aggregate_assumptions.clone(),
                ctx,
            )
        }
        .map_err(CausalError::from)?;
        if let Some(adjusted) = observation_adjusted {
            apply_temporal_observation_result(&mut response, adjusted);
            if self.settings.bootstrap_replicates > 0
                && let Some(bootstrap) = bootstrap_observation_adjusted_temporal_response(
                    source_data,
                    query,
                    aligned,
                    ObservationBootstrap {
                        options: self.observation_options,
                        replicates: self.settings.bootstrap_replicates,
                    },
                    ctx,
                )?
            {
                apply_observation_bootstrap(
                    &mut response,
                    &bootstrap,
                    self.settings.bootstrap_replicates,
                );
            }
        }
        // The estimator's public API accepts one aggregate status for callers
        // that have a homogeneous set of horizon witnesses. This execution path
        // has the richer per-horizon records, so retain their actual statuses
        // instead of repeating the first/aggregate label across the surface.
        if let Some(per_horizon) = response.horizon_identification.as_mut() {
            for (record, entry) in Arc::make_mut(per_horizon).iter_mut().zip(aligned) {
                record.status = entry.identification.status;
            }
        }
        Ok(EstimatedTemporalResponse { response, conflict_summary, observed_posterior })
    }

    /// Assembles the temporal response result from the estimated surface.
    fn finish_temporal_response(
        &self,
        data: &TimeSeriesData,
        aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
        claim: IdentifiedClaim,
        estimated: EstimatedTemporalResponse,
        (treatment, outcome): (VariableId, VariableId),
        run: ExecuteRun<'_>,
    ) -> StudyResult {
        let IdentifiedClaim { identification, estimand } = claim;
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let EstimatedTemporalResponse { response, conflict_summary, observed_posterior } =
            estimated;
        let (scalar, standard_error) =
            super::super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(lag_aligned_analysis_rows(data, aligned));
        let mut diagnostics = temporal_response_diagnostics(
            aligned,
            &response,
            conflict_summary.as_ref(),
            identify_cached,
        );
        // An intervention level's cheap/full suite runs the per-horizon Pulse-witness
        // refuter state (unsupported); a curve keeps the skipped note.
        let (refutations, predictive_checks) = match &self.query {
            CausalQuery::Response(query)
                if !matches!(self.settings.refute, RefuteSuite::None)
                    && matches!(
                        query.functional,
                        antecedent_core::ResponseFunctional::InterventionResponse { .. }
                    ) =>
            {
                match super::super::temporal_response_refutation::refute_temporal_intervention(
                    self, data, query, aligned, ctx,
                ) {
                    Ok((reports, checks, notes)) => {
                        diagnostics.retain(|diagnostic| {
                            diagnostic.code.as_ref() != "refute.temporal_response.skipped"
                        });
                        diagnostics.extend(notes);
                        (reports, checks)
                    }
                    Err(error) => {
                        diagnostics.push(Diagnostic::new(
                            "refute.temporal_response.state_failed",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Warning,
                            format!(
                                "the temporal intervention-level refuter state failed: {error}"
                            ),
                        ));
                        (Vec::new(), Vec::new())
                    }
                }
            }
            _ => (Vec::new(), Vec::new()),
        };
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::temporal_response_for(&self.settings.inference),
                ),
                treatment,
                outcome,
            )
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                posterior: observed_posterior,
                predictive_checks,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        )
    }

    fn execute_temporal_sequence_response(
        &self,
        inputs: SequenceResponseInputs<'_>,
        claim: IdentifiedClaim,
        (treatment, outcome): (VariableId, VariableId),
        basis: IdentificationBasis<'_>,
        run: ExecuteRun<'_>,
    ) -> Result<StudyResult, CausalError> {
        let SequenceResponseInputs { temporal, overlays, observation_adjusted, .. } = inputs;
        let IdentifiedClaim { identification, estimand } = claim;
        let IdentificationBasis { status: aggregate_status, assumptions } = basis;
        let mut assumptions = assumptions.clone();
        let ExecuteRun { ctx, .. } = run;
        if self.declared.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "multi-step Sequence overlays require no discovery-estimation split",
            });
        }
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            if cfg.has_transfer() {
                return Err(CausalError::Unsupported {
                    message: "multi-step Sequence inference requires isotropic per-mechanism priors",
                });
            }
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        assumptions.push(sequential_linear_sem_assumption(SEQUENCE_LINEAR_SEM));
        if overlays.iter().any(|overlay| overlay.bounds.is_some()) {
            assumptions
                .push(antecedent_estimate::SequentialMechanismOverlay::mean_target_assumption());
        }
        let surface =
            estimate_sequence_horizons(&inputs, outcome, bayes.as_ref(), &assumptions, ctx)?;
        let support = sequence_overlay_support(overlays, temporal, surface.uncertainty_complete);
        let SequenceHorizonSurface {
            mean,
            lower,
            upper,
            horizons,
            uncertainty_complete,
            last_posterior,
            horizon_posteriors,
            bootstrap_cancelled,
        } = surface;
        let mut response = CausalResponse {
            estimand: inputs.query.functional.clone(),
            identification_status: aggregate_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(
                    temporal.horizons.iter().map(|&h| f64::from(h)).collect::<Vec<_>>(),
                ),
                dimension: 1,
                mean: Arc::from(mean),
            }),
            uncertainty: if uncertainty_complete {
                ResponseUncertainty::PointwiseBand {
                    level: 0.95,
                    lower: Arc::from(lower),
                    upper: Arc::from(upper),
                    // Posterior quantiles when a Bayesian fit ran, else Wald bounds.
                    interpretation: if bayes.is_some() {
                        antecedent_core::IntervalInterpretation::Credible
                    } else {
                        antecedent_core::IntervalInterpretation::Confidence
                    },
                    draws: None,
                }
            } else {
                ResponseUncertainty::None
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(horizons)),
            interaction_structurally_zero: false,
        };
        if let Some(adjusted) = observation_adjusted {
            apply_temporal_observation_result(&mut response, adjusted);
        }
        // Frequentist bands come from one joint tuple bootstrap of the whole Sequence
        // surface (observation nuisance refit per replicate when the query carries one).
        if bayes.is_none() && self.settings.bootstrap_replicates > 0 {
            self.apply_sequence_bootstrap_band(&inputs, outcome, &mut response, ctx)?;
        }
        Ok(self.finish_sequence_response(
            &inputs,
            IdentifiedClaim { identification, estimand },
            SequenceFinish {
                response,
                horizon_posteriors,
                last_posterior,
                bootstrap_cancelled,
                attach_scalar_posterior: temporal.horizons.len() == 1,
            },
            (treatment, outcome),
            run,
        ))
    }

    /// Replaces the sequence response's pointwise band with the joint tuple
    /// bootstrap band, refitting the observation nuisance per replicate when
    /// the query carries one.
    fn apply_sequence_bootstrap_band(
        &self,
        inputs: &SequenceResponseInputs<'_>,
        outcome: VariableId,
        response: &mut CausalResponse,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let band = bootstrap_sequence_response(
            inputs.source_data,
            inputs.query,
            SequenceMechanisms { graph: inputs.graph, overlays: inputs.overlays },
            outcome,
            inputs.aligned,
            ObservationBootstrap {
                options: self.observation_options,
                replicates: self.settings.bootstrap_replicates,
            },
            ctx,
        )?;
        apply_tuple_bootstrap_band(
            response,
            &band,
            self.settings.bootstrap_replicates,
            if inputs.observation_adjusted.is_some() {
                TupleBandSource::Observation
            } else {
                TupleBandSource::CompleteSequence
            },
        );
        Ok(())
    }

    /// Assembles the Sequence response result: the scalar summary, the
    /// overlay and per-horizon posterior disclosures, and the finish.
    fn finish_sequence_response(
        &self,
        inputs: &SequenceResponseInputs<'_>,
        claim: IdentifiedClaim,
        finish: SequenceFinish,
        (treatment, outcome): (VariableId, VariableId),
        run: ExecuteRun<'_>,
    ) -> StudyResult {
        let IdentifiedClaim { identification, estimand } = claim;
        let SequenceFinish {
            response,
            horizon_posteriors,
            last_posterior,
            bootstrap_cancelled,
            attach_scalar_posterior,
        } = finish;
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let (scalar, standard_error) =
            super::super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(lag_aligned_analysis_rows(inputs.data, inputs.aligned));
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal.sequence_overlay",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "multi-step / joint Sequence runs as overlays on unfolded sequential g-computation; \
             identifier remains temporal.backdoor.unfolded; no last-step collapse",
        )];
        diagnostics.extend(response.support.warnings.iter().cloned());
        // Every horizon's posterior notes feed the same tempering / draw-floor
        // diagnostics the scalar path derives from `extras.posterior`; a
        // multi-horizon surface must not lose a disclosure raised at a horizon
        // other than the last one just because only the last posterior is
        // attached as the scalar artifact below.
        let mut seen_diagnostic_codes: std::collections::HashSet<Arc<str>> =
            diagnostics.iter().map(|d| Arc::clone(&d.code)).collect();
        for diagnostic in posterior_note_diagnostics(&horizon_posteriors) {
            push_unique_diagnostic(&mut diagnostics, &mut seen_diagnostic_codes, diagnostic);
        }
        for diagnostic in super::super::passed_route_diagnostics(&horizon_posteriors) {
            push_unique_diagnostic(&mut diagnostics, &mut seen_diagnostic_codes, diagnostic);
        }
        if !attach_scalar_posterior && self.settings.inference.is_bayesian() {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal.sequence_posterior_not_attached",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the response retains per-horizon posterior summaries and pointwise intervals; \
                 no scalar posterior artifact is attached to a multi-horizon surface",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued temporal response",
        ));
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::temporal_response_for(&self.settings.inference),
                ),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started))
            .cancelled(bootstrap_cancelled || ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                posterior: last_posterior,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        )
    }
}
