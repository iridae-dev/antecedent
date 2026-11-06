//! Graph-posterior temporal class point execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CONDITIONAL_PRIOR_REFUSAL, CROSS_WORLD_PRIOR_REFUSAL, CausalError, CausalQuery,
    CrossWorldContrast, DERIVATIVE_PRIOR_REFUSAL, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, ExecuteRun, ExecutionContext, GraphPosterior,
    IdentificationResult, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, Instant, OverlapPolicy, PhysicalExecutionPlan, PooledAtoms,
    PosteriorAtomMasses, RefuteSuite, RouteDefaults, SlopeSource, Strategy, StudyResult,
    TimeSeriesData, VariableId, class_response_uncertainty, class_scalar_uncertainty,
    collect_class_conditional, collect_class_cross_world, cross_world_endpoints, cross_world_from,
    cross_world_note, cross_world_probe, derivative_evaluator, derivative_on_envelope,
    derivative_probe, derivative_response_finite, elapsed_ns, envelope_span,
    envelope_to_identification_result_for, identify_cached_diagnostic, isotropic_prior_scale,
    load_class_posterior_cache, mix_derivative_responses, point_uncertainty,
    resolve_posterior_status, response_center, retain_graph_dependent_derivatives,
    weighted_derivative_mean,
};

impl super::super::Study {
    /// Graph-posterior conditional, derivative, and cross-world points.
    ///
    /// Each atom runs the same producer as the explicit class route. The
    /// published scalar is the posterior-weighted mean of those
    /// completion-weighted points. A single atom publishes that atom's
    /// completion-weighted mean. The uncertainty treats every identified
    /// completion of every atom as one frozen-weight atom (posterior weight times
    /// normalized completion weight) of one shared procedure.
    pub(in crate::analysis::execute) fn execute_temporal_class_graph_posterior_family(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match &self.query {
            CausalQuery::TemporalEffect(_) => {
                self.execute_posterior_conditional(data, gp, physical, ctx)
            }
            CausalQuery::Response(_) => self.execute_posterior_derivative(data, gp, physical, ctx),
            _ => self.execute_posterior_cross_world(data, gp, physical, ctx),
        }
    }

    fn execute_posterior_conditional(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let CausalQuery::TemporalEffect(query) = &self.query else { unreachable!() };
        if query.effect_modifiers.is_empty()
            || query.is_multi_step_sustained()
            || matches!(query.policy, antecedent_core::TemporalPolicy::Dynamic { .. })
        {
            return Err(CausalError::Unsupported {
                message: "temporal conditional effect fits a modifier set on a single-step pulse or sustained policy",
            });
        }
        if self.settings.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal conditional effect is point-only; cheap/full refuters are not supported",
            });
        }
        let prior_scale =
            isotropic_prior_scale(&self.settings.inference, CONDITIONAL_PRIOR_REFUSAL)?;
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, query, ctx)?;
        let mut pairs = Vec::new();
        let mut pooled = PooledAtoms::default();
        let mut chosen = None;
        let mut masses = PosteriorAtomMasses::default();
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let fitted = collect_class_conditional(
                data,
                query,
                &atom.envelope,
                ctx.kernel_policy,
                prior_scale,
            )?;
            let point = fitted.point()?;
            let envelope = &atom.envelope.envelope;
            masses.skipped.absorb(&fitted.skipped);
            masses.add(
                weight,
                envelope.status,
                super::super::temporal_path::temporal_class_structural_mixture(
                    envelope,
                    crate::result::StructuralWeightBasis::CompletionEnumeration,
                    None,
                    &fitted.values,
                ),
            );
            if !point.is_finite() {
                continue;
            }
            pairs.push((weight, point));
            if chosen.is_none() {
                let mut diagnostics =
                    vec![super::super::temporal_path::temporal_class_envelope_diagnostic(
                        envelope,
                        self.graph.class(),
                    )];
                diagnostics.push(Diagnostic::new(
                    "estimate.temporal.conditional_marginal",
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Info,
                    "published value is the completion-weighted marginal effect over the modifier \
                     set; its interval refits every completion's interaction regression and \
                     modifier means",
                ));
                diagnostics.extend(super::super::temporal_effect_core::modifier_set_coverage_note(
                    query.effect_modifiers.len(),
                ));
                chosen = Some((
                    fitted.assumptions.clone(),
                    fitted.estimand.clone(),
                    envelope_to_identification_result_for(
                        envelope,
                        CausalQuery::TemporalEffect(query.clone()),
                    ),
                    diagnostics,
                ));
            }
            pooled.push(fitted, weight, envelope.truncated_completions > 0);
        }
        self.finish_posterior_point(
            ExecuteRun { physical, ctx, started, identify_cached },
            &cache,
            (&pairs, pooled, (None, false)),
            (chosen, masses),
            (query.treatment, query.outcome),
            (
                EstimatorId::TemporalLinearAdjustment,
                (
                    point_uncertainty::CONDITIONAL_BLOCK_STREAM,
                    point_uncertainty::CONDITIONAL_POSTERIOR_STREAM,
                ),
            ),
            &|_: usize, source: &dyn SlopeSource| {
                source.conditional_effect().map(|value| vec![value])
            },
        )
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one graph-posterior derivative execution: the atoms, the mixture and its band stay together"
    )]
    fn execute_posterior_derivative(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let CausalQuery::Response(query) = &self.query else { unreachable!() };
        query.require_supported_temporal_observation()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
        })?;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
        let probe = derivative_probe(query)?;
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, &probe, ctx)?;
        let prior_scale =
            isotropic_prior_scale(&self.settings.inference, DERIVATIVE_PRIOR_REFUSAL)?;
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset()?);
        let mut pairs = Vec::new();
        let mut responses = Vec::new();
        let mut pooled_atoms = Vec::new();
        let mut pooled_weights = Vec::new();
        let mut spans = Vec::new();
        let mut chosen = None;
        let mut masses = PosteriorAtomMasses::default();
        let mut witness_states =
            super::super::posterior_witness_state::AtomWitnessStates::default();
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let derived = derivative_on_envelope(
                data,
                query,
                &atom.envelope,
                self.graph.class(),
                (offsets, ctx.kernel_policy, prior_scale),
            )?;
            let scalar = super::super::response_path::response_scalar_summary(&derived.response).0;
            masses.skipped.absorb(&derived.skipped);
            masses.add(weight, derived.status, derived.structural);
            if !derivative_response_finite(&derived.response) {
                continue;
            }
            pairs.push((weight, scalar));
            responses.push((weight, derived.response.clone()));
            if self.settings.refute != RefuteSuite::None {
                // Each atom's derivative is refuted on its own identified completions' Pulse
                // witnesses against its own response, then the reports mix by posterior weight.
                let sites = super::super::posterior_witness_state::class_completion_sites(
                    &atom.envelope,
                    &probe,
                    temporal.horizons[0],
                )?;
                witness_states.push_derivative(
                    self,
                    data,
                    (atom.key, weight),
                    (query, &derived.response),
                    &sites,
                    ctx,
                )?;
            }
            let total: f64 = derived.weights.iter().sum();
            for (completion, completion_weight) in
                derived.atoms.into_iter().zip(derived.weights.iter())
            {
                pooled_atoms.push(completion.freeze());
                pooled_weights.push(weight * completion_weight / total);
            }
            spans.push(envelope_span(&atom.envelope));
            if chosen.is_none() {
                chosen = Some((
                    derived.identification.required_assumptions.clone(),
                    derived.estimand,
                    derived.identification,
                    derived.diagnostics,
                ));
            }
        }
        let mut response = mix_derivative_responses(&responses)?;
        let center = response_center(&response)?;
        let (uncertainty_diagnostics, replicates_ok, cancelled) = class_response_uncertainty(
            &pooled_atoms,
            &pooled_weights,
            (&mut response, &center),
            self.class_point_plan(
                spans.iter().copied().max().unwrap_or(1),
                (
                    point_uncertainty::DERIVATIVE_BLOCK_STREAM,
                    point_uncertainty::DERIVATIVE_POSTERIOR_STREAM,
                ),
            ),
            ctx,
            derivative_evaluator(&query.functional),
        )?;
        let chosen = chosen.map(|mut chosen| {
            chosen.3.extend(uncertainty_diagnostics);
            chosen
        });
        let result = self.finish_posterior_point(
            ExecuteRun { physical, ctx, started, identify_cached },
            &cache,
            (&pairs, PooledAtoms::default(), (replicates_ok, cancelled)),
            (chosen, masses),
            (treatment, outcome),
            (
                EstimatorId::temporal_response_for(&self.settings.inference),
                (
                    point_uncertainty::DERIVATIVE_BLOCK_STREAM,
                    point_uncertainty::DERIVATIVE_POSTERIOR_STREAM,
                ),
            ),
            &derivative_evaluator(&query.functional),
        )?;
        response.identification_status = result.identification.status;
        if let Some(structural) = &result.structural_response {
            retain_graph_dependent_derivatives(&mut response, structural);
        }
        let mut result = result;
        result.payload = crate::result::ResultPayload::Response(response);
        result.posterior = None;
        if let Some(structural) = result.structural_response.as_mut() {
            structural.identified_set_interval = None;
        }
        if self.settings.refute != RefuteSuite::None {
            let (reports, checks, notes) = witness_states.finish();
            result.refutations = reports;
            result.predictive_checks = checks;
            result.diagnostics.extend(notes);
        }
        result.rebind_interval(self.settings.inference.is_bayesian());
        Ok(result)
    }

    fn execute_posterior_cross_world(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.settings.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not supported",
            });
        }
        let (treatment, outcome, mediator, active, control) = cross_world_endpoints(&self.query)?;
        let contrast = CrossWorldContrast { treatment, outcome, mediator, active, control };
        let probe = cross_world_probe(treatment, outcome);
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, &probe, ctx)?;
        let prior_scale =
            isotropic_prior_scale(&self.settings.inference, CROSS_WORLD_PRIOR_REFUSAL)?;
        let offsets = (probe.try_treatment_offset()?, probe.outcome_offset()?);
        let mut pairs = Vec::new();
        let mut pooled = PooledAtoms::default();
        let mut chosen = None;
        let mut masses = PosteriorAtomMasses::default();
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        let query = &self.query;
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let fitted = collect_class_cross_world(
                data,
                query,
                &atom.envelope,
                (contrast, offsets),
                ctx.kernel_policy,
                prior_scale,
            )?;
            let point = fitted.point()?;
            let envelope = &atom.envelope.envelope;
            masses.skipped.absorb(&fitted.skipped);
            masses.add(
                weight,
                envelope.status,
                super::super::temporal_path::temporal_class_structural_mixture(
                    envelope,
                    crate::result::StructuralWeightBasis::CompletionEnumeration,
                    None,
                    &fitted.values,
                ),
            );
            if !point.is_finite() {
                continue;
            }
            if chosen.is_none() {
                let identification = super::super::temporal_path::template_scoped_cross_world(
                    fitted.primary_identification.clone(),
                    &fitted.primary_indexer,
                    self.query.clone(),
                    contrast,
                );
                let estimand = identification.estimands[0].clone();
                let diagnostics = vec![
                    super::super::temporal_path::temporal_class_envelope_diagnostic(
                        envelope,
                        self.graph.class(),
                    ),
                    cross_world_note(),
                ];
                chosen = Some((
                    identification.required_assumptions.clone(),
                    estimand,
                    identification,
                    diagnostics,
                ));
            }
            pairs.push((weight, point));
            pooled.push(fitted, weight, envelope.truncated_completions > 0);
        }
        self.finish_posterior_point(
            ExecuteRun { physical, ctx, started, identify_cached },
            &cache,
            (&pairs, pooled, (None, false)),
            (chosen, masses),
            (treatment, outcome),
            (
                EstimatorId::TemporalLinearAdjustment,
                (
                    point_uncertainty::CROSS_WORLD_BLOCK_STREAM,
                    point_uncertainty::CROSS_WORLD_POSTERIOR_STREAM,
                ),
            ),
            &|_: usize, source: &dyn SlopeSource| {
                cross_world_from(query, source, contrast).map(|value| vec![value])
            },
        )
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the run, the cache, the pooled completions, the chosen atom's claim, the pair, the estimator with its streams and the functional are distinct inputs"
    )]
    fn finish_posterior_point(
        &self,
        run: ExecuteRun<'_>,
        cache: &crate::analysis::prepared::CachedTemporalClassPosteriorIdentification,
        (pairs, pooled, accounting): (&[(f64, f64)], PooledAtoms, (Option<u32>, bool)),
        (chosen, masses): (
            Option<(
                antecedent_core::AssumptionSet,
                IdentifiedEstimand,
                IdentificationResult,
                Vec<Diagnostic>,
            )>,
            PosteriorAtomMasses,
        ),
        (treatment, outcome): (VariableId, VariableId),
        (estimator_id, streams): (EstimatorId, (u64, u64)),
        evaluate: &dyn Fn(usize, &dyn SlopeSource) -> Result<Vec<f64>, CausalError>,
    ) -> Result<StudyResult, CausalError> {
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let (assumptions, estimand, mut identification, mut diagnostics) =
            chosen.ok_or_else(|| CausalError::Compile {
                message: format!(
                    "temporal class graph-posterior point had no estimable atom (atoms={}, unidentified={})",
                    cache.class_atoms.len(),
                    cache.graphs.unidentified_mass()
                ),
            })?;
        let point = weighted_derivative_mean(pairs)?;
        diagnostics.extend(masses.skipped.diagnostic());
        let (mut structural, atom_status, summary) =
            masses.finish(cache.graphs.unidentified_mass());
        identification.status = resolve_posterior_status(
            atom_status,
            crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
            summary,
        );
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        if pairs.len() > 1 {
            diagnostics.push(Diagnostic::new(
                "estimate.graph_posterior.temporal_class_point",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "published value is the posterior-weighted mean of each atom's completion-weighted point; completion enumeration is not posterior probability",
            ));
        }
        let PooledAtoms { atoms, weights, points, truncated, spans } = pooled;
        let (estimate, posterior, replicates_ok, cancelled) = if atoms.is_empty() {
            (
                EffectEstimate::new(point, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride),
                None,
                accounting.0,
                accounting.1,
            )
        } else {
            let outcome_uncertainty = class_scalar_uncertainty(
                &atoms,
                &weights,
                &points,
                point,
                assumptions,
                truncated,
                self.class_point_plan(spans.iter().copied().max().unwrap_or(1), streams),
                ctx,
                evaluate,
            )?;
            structural.identified_set_interval = outcome_uncertainty.identified_set;
            diagnostics.extend(outcome_uncertainty.diagnostics.iter().cloned());
            (
                outcome_uncertainty.estimate,
                outcome_uncertainty.posterior,
                outcome_uncertainty.replicates_ok,
                outcome_uncertainty.cancelled,
            )
        };
        let n_draws = posterior
            .as_ref()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let result = self.finish_identified_execute(
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
            .bootstrap_replicates_ok(replicates_ok)
            .cancelled(cancelled)
            .extras(IdentifiedExecuteExtras {
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                n_draws,
                posterior,
                ..Default::default()
            }),
        );
        Ok(result)
    }
}
