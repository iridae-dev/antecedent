//! TemporalCpdag/TemporalPag conditional, derivative, and cross-world points.
//!
//! Each identified completion uses the existing temporal linear producer. The
//! published point is the completion-weighted mean. Enumeration is not a
//! posterior probability.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::planner::RouteDefaults;
use antecedent_core::SupportStatus;

const CONDITIONAL_PRIOR_REFUSAL: &str = "temporal conditional effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";
const DERIVATIVE_PRIOR_REFUSAL: &str = "temporal derivative uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";
const CROSS_WORLD_PRIOR_REFUSAL: &str = "temporal cross-world effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed";

struct WeightedPoints {
    point: f64,
    values: Vec<Option<f64>>,
    estimand: IdentifiedEstimand,
    assumptions: antecedent_core::AssumptionSet,
    /// First identified completion, used to name the cross-world functional.
    primary_identification: IdentificationResult,
    primary_indexer: TemporalIndexer,
}

fn isotropic_prior_scale(
    inference: &InferenceMode,
    message: &'static str,
) -> Result<Option<f64>, CausalError> {
    match inference {
        InferenceMode::Bayesian(cfg) => {
            if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some()
            {
                return Err(CausalError::Unsupported { message: message.into() });
            }
            Ok(Some(cfg.prior_scale))
        }
        InferenceMode::Frequentist => Ok(None),
    }
}

fn weighted_mean(values: &[(f64, f64)]) -> Result<f64, CausalError> {
    let mut weighted = 0.0;
    let mut total = 0.0;
    for &(weight, point) in values {
        if !point.is_finite() {
            return Err(CausalError::Compile {
                message: "temporal class completion produced a non-finite point".into(),
            });
        }
        weighted += weight * point;
        total += weight;
    }
    if !matches!(total.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        return Err(CausalError::Compile {
            message: "temporal class envelope had no estimable identified completions".into(),
        });
    }
    Ok(weighted / total)
}

fn weighted_derivative_mean(values: &[(f64, f64)]) -> Result<f64, CausalError> {
    if values.iter().all(|(_, point)| point.is_nan()) && !values.is_empty() {
        return Ok(f64::NAN);
    }
    weighted_mean(values)
}

fn derivative_response_finite(response: &CausalResponse) -> bool {
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => value.is_finite(),
        ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => {
            values.iter().all(|value| value.is_finite())
        }
        ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) => {
            values.iter().all(|value| value.is_finite())
        }
        _ => false,
    }
}

fn point_only_estimate(point: f64, assumptions: antecedent_core::AssumptionSet) -> EffectEstimate {
    EffectEstimate::new(point, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride)
}

fn clear_interval(mut result: StudyResult) -> StudyResult {
    result.interval = None;
    result.posterior = None;
    result
}

fn collect_weighted_points(
    bundle: &antecedent_identify::TemporalClassEnvelope,
    mut fit: impl FnMut(
        &antecedent_identify::GraphIdentificationCase<antecedent_identify::TemporalCompletionGraph>,
        &TemporalIndexer,
    ) -> Result<f64, CausalError>,
) -> Result<WeightedPoints, CausalError> {
    let envelope = &bundle.envelope;
    if matches!(envelope.status, IdentificationStatus::NotIdentified)
        || envelope.identified_weight.0 <= 0.0
    {
        return Err(CausalError::not_identified(
            envelope.status,
            envelope.truncated_completions > 0,
            "temporal class query not identified (no identified mass in envelope)",
        ));
    }
    let mut pairs = Vec::new();
    let mut values = vec![None; envelope.cases.len()];
    let mut primary = None;
    for (index, (case, indexer)) in envelope.cases.iter().zip(bundle.indexers.iter()).enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let point = fit(case, indexer)?;
        pairs.push((case.weight.0, point));
        values[index] = Some(point);
        if primary.is_none() {
            primary = Some((case.result.clone(), indexer.clone()));
        }
    }
    let (primary_identification, primary_indexer) =
        primary.ok_or_else(|| CausalError::Compile {
            message: "temporal class envelope had no estimable identified completions".into(),
        })?;
    let estimand = primary_identification.estimands.first().cloned().ok_or_else(|| {
        CausalError::Compile { message: "temporal class completion has no estimand".into() }
    })?;
    Ok(WeightedPoints {
        point: weighted_mean(&pairs)?,
        values,
        estimand,
        assumptions: primary_identification.required_assumptions.clone(),
        primary_identification,
        primary_indexer,
    })
}

pub(super) fn assemble_temporal_class_conditional(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    bundle: &antecedent_identify::TemporalClassEnvelope,
    graph_class: GraphClass,
    refute: RefuteSuite,
    inference: &InferenceMode,
    policy: &antecedent_core::KernelPolicy,
) -> Result<
    (
        EffectEstimate,
        IdentifiedEstimand,
        IdentificationResult,
        Vec<Diagnostic>,
        crate::result::StructuralResponseMixture,
    ),
    CausalError,
> {
    if query.effect_modifiers.len() != 1
        || matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
        || matches!(query.policy, antecedent_core::TemporalPolicy::Dynamic { .. })
    {
        return Err(CausalError::Unsupported {
            message: "temporal conditional effect fits one modifier on a single-step pulse or sustained policy".into(),
        });
    }
    if refute != RefuteSuite::None {
        return Err(CausalError::Unsupported {
            message:
                "temporal conditional effect is point-only; cheap/full refuters are not licensed"
                    .into(),
        });
    }
    let prior_scale = isotropic_prior_scale(inference, CONDITIONAL_PRIOR_REFUSAL)?;
    let fitted = collect_weighted_points(bundle, |case, indexer| {
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        antecedent_estimate::temporal_conditional_marginal_effect(
            data,
            &estimand,
            query,
            indexer,
            policy,
            prior_scale,
        )
        .map_err(CausalError::from)
    })?;
    let envelope = &bundle.envelope;
    let mut diagnostics =
        vec![super::temporal_path::temporal_class_envelope_diagnostic(envelope, graph_class)];
    diagnostics.push(Diagnostic::new(
        "estimate.temporal.conditional_marginal",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "published value is the completion-weighted one-modifier marginal effect; no interval is published",
    ));
    let structural = super::temporal_path::temporal_class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        None,
        &fitted.values,
    );
    let identification =
        envelope_to_identification_result_for(envelope, CausalQuery::TemporalEffect(query.clone()));
    Ok((
        point_only_estimate(fitted.point, fitted.assumptions),
        fitted.estimand,
        identification,
        diagnostics,
        structural,
    ))
}

impl super::Study {
    pub(super) fn execute_temporal_class_derivative(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.require_licensed_temporal_observation()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
        })?;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon".into(),
            });
        }
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal derivative is point-only; cheap/full refuters are not licensed"
                    .into(),
            });
        }
        let prior_scale = isotropic_prior_scale(&self.inference, DERIVATIVE_PRIOR_REFUSAL)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let probe = TemporalEffectQuery {
            treatment,
            outcome,
            policy: temporal.policy.clone(),
            control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
            active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
            horizon_steps: temporal.horizons[0],
            max_history_lag: temporal.max_history_lag,
            target_population: query.target_population.clone(),
            effect_modifiers: std::sync::Arc::from([]),
        };
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let (bundle, identify_cached) =
            if let Some(cache) = self.temporal_class_identification_cache.as_ref() {
                if let Some((_, envelope)) =
                    cache.by_horizon.iter().find(|(horizon, _)| *horizon == probe.horizon_steps)
                {
                    (Arc::clone(envelope), true)
                } else {
                    (Arc::clone(&cache.envelope), true)
                }
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, &probe)?.envelope.clone(), false)
            };
        let treatment_offset = probe.try_treatment_offset()?;
        let outcome_offset = probe.outcome_offset();
        let envelope = &bundle.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "temporal class query not identified (no identified mass in envelope)",
            ));
        }
        let mut responses = Vec::new();
        let mut values = vec![None; envelope.cases.len()];
        let mut primary_estimand = None;
        for (index, (case, indexer)) in
            envelope.cases.iter().zip(bundle.indexers.iter()).enumerate()
        {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let response = super::temporal_path::temporal_linear_derivative_response(
                data,
                query,
                &estimand,
                indexer,
                &case.result,
                treatment_offset,
                outcome_offset,
                &ctx.kernel_policy,
                prior_scale,
            )?;
            let scalar = super::response_path::response_scalar_summary(&response).0;
            if scalar.is_finite() {
                values[index] = Some(scalar);
            }
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand);
            }
            responses.push((case.weight.0, response));
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class envelope had no estimable identified completions".into(),
        })?;
        let mut response = mix_derivative_responses(&responses)?;
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "estimate.temporal.linear_derivative",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the completion-weighted slope of the linear unfolded temporal fit; no interval is published",
        ));
        diagnostics.extend(response.support.warnings.iter().cloned());
        let scalar = super::response_path::response_scalar_summary(&response).0;
        let estimate = point_only_estimate(scalar, response.assumptions.clone());
        let identification =
            envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
        let estimator_id = match &self.inference {
            InferenceMode::Bayesian(_) => EstimatorId::TemporalResponseBayesian,
            InferenceMode::Frequentist => EstimatorId::TemporalResponseGcomp,
        };
        let structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &values,
        );
        response.identification_status = identification.status;
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                identifier_id,
                estimator_id,
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: Arc::unwrap_or_clone(bundle),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                response: Some(response),
                ..Default::default()
            }),
        );
        result = clear_interval(result);
        Ok(result)
    }

    pub(super) fn execute_temporal_class_cross_world(
        &self,
        data: &TimeSeriesData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not licensed".into(),
            });
        }
        let prior_scale = isotropic_prior_scale(&self.inference, CROSS_WORLD_PRIOR_REFUSAL)?;
        let (treatment, outcome, mediator, active, control) = cross_world_endpoints(&self.query)?;
        let probe = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let (bundle, identify_cached) = {
            report_identify_compute(ctx);
            (self.identify_temporal_class(identifier_id, &probe)?.envelope.clone(), false)
        };
        let treatment_offset = probe.try_treatment_offset()?;
        let outcome_offset = probe.outcome_offset();
        let query = &self.query;
        let fitted = collect_weighted_points(&bundle, |case, indexer| {
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            if !estimand.adjustment_set.is_empty() {
                let message = match query {
                    CausalQuery::NestedCounterfactual(_) => {
                        "temporal nested effect is licensed when the back-door set is empty"
                    }
                    CausalQuery::PathSpecific(_) => {
                        "temporal path-specific effect is licensed when the back-door set is empty"
                    }
                    CausalQuery::Counterfactual(_) => {
                        "temporal counterfactual is licensed when the back-door set is empty"
                    }
                    CausalQuery::Distribution(_) => {
                        "temporal interventional mean is licensed when the back-door set is empty"
                    }
                    _ => "temporal cross-world effect is licensed when the back-door set is empty",
                };
                return Err(CausalError::Unsupported { message: message.into() });
            }
            cross_world_point(
                data,
                query,
                &estimand,
                indexer,
                treatment,
                outcome,
                mediator,
                active,
                control,
                treatment_offset,
                outcome_offset,
                &ctx.kernel_policy,
                prior_scale,
            )
        })?;
        let identification = super::temporal_path::scope_temporal_cross_world(
            fitted.primary_identification,
            &fitted.primary_indexer,
            self.query.clone(),
            treatment,
            outcome,
            mediator,
            active,
            control,
        );
        let estimand = identification.estimands[0].clone();
        let estimate =
            point_only_estimate(fitted.point, identification.required_assumptions.clone());
        let envelope = &bundle.envelope;
        let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
            envelope,
            self.graph.class(),
        )];
        diagnostics.push(Diagnostic::new(
            "estimate.temporal.cross_world",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the completion-weighted linear unfolded cross-world point; no interval is published",
        ));
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let structural = super::temporal_path::temporal_class_structural_mixture(
            envelope,
            crate::result::StructuralWeightBasis::CompletionEnumeration,
            None,
            &fitted.values,
        );
        let result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                identifier_id,
                EstimatorId::TemporalLinearAdjustment,
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: Arc::unwrap_or_clone(bundle),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        );
        Ok(clear_interval(result))
    }

    /// Graph-posterior conditional, derivative, and cross-world points.
    ///
    /// Each atom runs the same producer as the explicit class route. The
    /// published scalar is the posterior-weighted mean of those
    /// completion-weighted points. A single atom publishes that atom's
    /// completion-weighted mean.
    pub(super) fn execute_temporal_class_graph_posterior_family(
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
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, query, ctx)?;
        let mut pairs = Vec::new();
        let mut chosen = None;
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let (estimate, estimand, identification, diagnostics, structural) =
                assemble_temporal_class_conditional(
                    data,
                    query,
                    &atom.envelope,
                    self.graph.class(),
                    self.refute,
                    &self.inference,
                    &ctx.kernel_policy,
                )?;
            if !estimate.ate.is_finite() {
                continue;
            }
            pairs.push((weight, estimate.ate));
            if chosen.is_none() {
                chosen = Some((
                    estimate.assumptions.clone(),
                    estimand,
                    identification,
                    diagnostics,
                    structural,
                    atom.envelope.envelope.status,
                ));
            }
        }
        self.finish_posterior_point(
            started,
            &cache,
            pairs,
            chosen,
            physical,
            query.treatment,
            query.outcome,
            EstimatorId::TemporalLinearAdjustment,
            identify_cached,
            ctx,
        )
    }

    fn execute_posterior_derivative(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let CausalQuery::Response(query) = &self.query else { unreachable!() };
        query.require_licensed_temporal_observation()?;
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
        })?;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon".into(),
            });
        }
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal derivative is point-only; cheap/full refuters are not licensed"
                    .into(),
            });
        }
        let probe = derivative_probe(query)?;
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, &probe, ctx)?;
        let prior_scale = isotropic_prior_scale(&self.inference, DERIVATIVE_PRIOR_REFUSAL)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let treatment_offset = probe.try_treatment_offset()?;
        let outcome_offset = probe.outcome_offset();
        let mut pairs = Vec::new();
        let mut responses = Vec::new();
        let mut chosen = None;
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let (response, estimand, identification, diagnostics, structural, status) =
                derivative_on_envelope(
                    data,
                    query,
                    &atom.envelope,
                    self.graph.class(),
                    treatment_offset,
                    outcome_offset,
                    &ctx.kernel_policy,
                    prior_scale,
                )?;
            let scalar = super::response_path::response_scalar_summary(&response).0;
            if !derivative_response_finite(&response) {
                continue;
            }
            pairs.push((weight, scalar));
            responses.push((weight, response));
            if chosen.is_none() {
                chosen = Some((
                    identification.required_assumptions.clone(),
                    estimand,
                    identification,
                    diagnostics,
                    structural,
                    status,
                ));
            }
        }
        let mut response = mix_derivative_responses(&responses)?;
        let result = self.finish_posterior_point(
            started,
            &cache,
            pairs,
            chosen,
            physical,
            treatment,
            outcome,
            match &self.inference {
                InferenceMode::Bayesian(_) => EstimatorId::TemporalResponseBayesian,
                InferenceMode::Frequentist => EstimatorId::TemporalResponseGcomp,
            },
            identify_cached,
            ctx,
        )?;
        response.identification_status = result.identification.status;
        let mut result = result;
        result.payload = crate::result::ResultPayload::Response(response);
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
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not licensed".into(),
            });
        }
        let (treatment, outcome, mediator, active, control) = cross_world_endpoints(&self.query)?;
        let probe = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let (cache, identify_cached) = load_class_posterior_cache(self, gp, data, &probe, ctx)?;
        let prior_scale = isotropic_prior_scale(&self.inference, CROSS_WORLD_PRIOR_REFUSAL)?;
        let treatment_offset = probe.try_treatment_offset()?;
        let outcome_offset = probe.outcome_offset();
        let mut pairs = Vec::new();
        let mut chosen = None;
        let key_weights = IdentifiedKeyWeights::new(&cache.graphs);
        for atom in cache.class_atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let fitted = collect_weighted_points(&atom.envelope, |case, indexer| {
                let estimand =
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
                if !estimand.adjustment_set.is_empty() {
                    return Err(CausalError::Unsupported {
                        message: "temporal cross-world effect is licensed when the back-door set is empty".into(),
                    });
                }
                cross_world_point(
                    data,
                    &self.query,
                    &estimand,
                    indexer,
                    treatment,
                    outcome,
                    mediator,
                    active,
                    control,
                    treatment_offset,
                    outcome_offset,
                    &ctx.kernel_policy,
                    prior_scale,
                )
            })?;
            let identification = super::temporal_path::scope_temporal_cross_world(
                fitted.primary_identification,
                &fitted.primary_indexer,
                self.query.clone(),
                treatment,
                outcome,
                mediator,
                active,
                control,
            );
            let estimand = identification.estimands[0].clone();
            let envelope = &atom.envelope.envelope;
            let mut diagnostics = vec![super::temporal_path::temporal_class_envelope_diagnostic(
                envelope,
                self.graph.class(),
            )];
            diagnostics.push(Diagnostic::new(
                "estimate.temporal.cross_world",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                "published value is the completion-weighted linear unfolded cross-world point; no interval is published",
            ));
            let structural = super::temporal_path::temporal_class_structural_mixture(
                envelope,
                crate::result::StructuralWeightBasis::CompletionEnumeration,
                None,
                &fitted.values,
            );
            if !fitted.point.is_finite() {
                continue;
            }
            pairs.push((weight, fitted.point));
            if chosen.is_none() {
                chosen = Some((
                    identification.required_assumptions.clone(),
                    estimand,
                    identification,
                    diagnostics,
                    structural,
                    envelope.status,
                ));
            }
        }
        self.finish_posterior_point(
            started,
            &cache,
            pairs,
            chosen,
            physical,
            treatment,
            outcome,
            EstimatorId::TemporalLinearAdjustment,
            identify_cached,
            ctx,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_posterior_point(
        &self,
        started: Instant,
        cache: &crate::analysis::prepared::CachedTemporalClassPosteriorIdentification,
        pairs: Vec<(f64, f64)>,
        chosen: Option<(
            antecedent_core::AssumptionSet,
            IdentifiedEstimand,
            IdentificationResult,
            Vec<Diagnostic>,
            crate::result::StructuralResponseMixture,
            IdentificationStatus,
        )>,
        physical: &PhysicalExecutionPlan,
        treatment: VariableId,
        outcome: VariableId,
        estimator_id: EstimatorId,
        identify_cached: bool,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let _ = ctx;
        let (assumptions, estimand, mut identification, mut diagnostics, structural, atom_status) =
            chosen.ok_or_else(|| CausalError::Compile {
                message: format!(
                    "temporal class graph-posterior point had no estimable atom (atoms={}, unidentified={})",
                    cache.class_atoms.len(),
                    cache.graphs.unidentified_mass()
                ),
            })?;
        let point = weighted_derivative_mean(&pairs)?;
        let unidentified = cache.graphs.unidentified_mass();
        identification.status = resolve_posterior_status(
            atom_status,
            crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
            PosteriorMassSummary { unidentified, ..PosteriorMassSummary::default() },
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
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                point_only_estimate(point, assumptions),
                identifier_id,
                estimator_id,
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        );
        Ok(clear_interval(result))
    }
}

fn mix_derivative_responses(
    responses: &[(f64, CausalResponse)],
) -> Result<CausalResponse, CausalError> {
    let (_, first) = responses.first().ok_or_else(|| CausalError::Compile {
        message: "temporal class derivative had no completion response".into(),
    })?;
    let mut mixed = first.clone();
    let value = match &first.estimate {
        ResponseIdentification::PointIdentified(value) => mix_response_value(responses, value)?,
        _ => {
            return Err(CausalError::Compile {
                message: "temporal class derivative completion is not point identified".into(),
            });
        }
    };
    mixed.estimate = ResponseIdentification::PointIdentified(value);
    mixed.uncertainty = ResponseUncertainty::None;
    if responses.iter().any(|(_, response)| response.support.status != SupportStatus::Supported) {
        mixed.support.status = SupportStatus::OutsideEmpiricalSupport;
        for (_, response) in responses.iter().skip(1) {
            mixed.support.warnings.extend(response.support.warnings.iter().cloned());
        }
    }
    Ok(mixed)
}

fn mix_response_value(
    responses: &[(f64, CausalResponse)],
    prototype: &ResponseValue,
) -> Result<ResponseValue, CausalError> {
    match prototype {
        ResponseValue::Scalar(_) => {
            let pairs = scalar_pairs(responses)?;
            Ok(ResponseValue::Scalar(weighted_mean(&pairs)?))
        }
        ResponseValue::Vector(values) => {
            let width = values.len();
            let mut mixed = vec![0.0; width];
            let mut total = 0.0;
            for (weight, response) in responses {
                let ResponseIdentification::PointIdentified(ResponseValue::Vector(row)) =
                    &response.estimate
                else {
                    return Err(shape_error());
                };
                if row.len() != width {
                    return Err(shape_error());
                }
                for (slot, value) in mixed.iter_mut().zip(row.iter()) {
                    *slot += weight * value;
                }
                total += weight;
            }
            if total <= 0.0 {
                return Err(shape_error());
            }
            for slot in &mut mixed {
                *slot /= total;
            }
            Ok(ResponseValue::Vector(std::sync::Arc::from(mixed)))
        }
        ResponseValue::Jacobian { outcomes, treatments, values } => {
            let width = values.len();
            let mut mixed = vec![0.0; width];
            let mut total = 0.0;
            for (weight, response) in responses {
                let ResponseIdentification::PointIdentified(ResponseValue::Jacobian {
                    outcomes: rows,
                    treatments: cols,
                    values: row,
                }) = &response.estimate
                else {
                    return Err(shape_error());
                };
                if *rows != *outcomes || *cols != *treatments || row.len() != width {
                    return Err(shape_error());
                }
                for (slot, value) in mixed.iter_mut().zip(row.iter()) {
                    *slot += weight * value;
                }
                total += weight;
            }
            if total <= 0.0 {
                return Err(shape_error());
            }
            for slot in &mut mixed {
                *slot /= total;
            }
            Ok(ResponseValue::Jacobian {
                outcomes: *outcomes,
                treatments: *treatments,
                values: std::sync::Arc::from(mixed),
            })
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal linear derivative does not estimate this functional".into(),
        }),
    }
}

fn scalar_pairs(responses: &[(f64, CausalResponse)]) -> Result<Vec<(f64, f64)>, CausalError> {
    responses
        .iter()
        .map(|(weight, response)| match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(point)) => {
                Ok((*weight, *point))
            }
            _ => Err(shape_error()),
        })
        .collect()
}

fn shape_error() -> CausalError {
    CausalError::Compile {
        message: "temporal class completions disagree on the derivative shape".into(),
    }
}

fn cross_world_endpoints(
    query: &CausalQuery,
) -> Result<(VariableId, VariableId, Option<VariableId>, f64, f64), CausalError> {
    match query {
        CausalQuery::NestedCounterfactual(q) => {
            Ok((q.treatment, q.outcome, Some(q.mediator), q.active_value(), q.control_value()))
        }
        CausalQuery::PathSpecific(q) => {
            if q.path_nodes.len() != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal path-specific effect is the one-mediator edge product"
                        .into(),
                });
            }
            Ok((
                q.treatment,
                q.outcome,
                Some(q.path_nodes[0]),
                super::temporal_path::temporal_cross_world_hard_level(&q.active)?,
                super::temporal_path::temporal_cross_world_hard_level(&q.control)?,
            ))
        }
        CausalQuery::Counterfactual(q) => {
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                    || CausalError::Unsupported {
                        message: "temporal counterfactual needs one hard intervention".into(),
                    },
                )?;
            let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Unsupported {
                message: "temporal counterfactual needs one outcome".into(),
            })?;
            Ok((
                treatment,
                outcome,
                None,
                super::temporal_path::temporal_cross_world_hard_level(&q.interventions[0])?,
                super::temporal_path::temporal_cross_world_hard_level(&q.control)?,
            ))
        }
        CausalQuery::Distribution(q) => {
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                    || CausalError::Unsupported {
                        message: "temporal interventional distribution needs one hard intervention"
                            .into(),
                    },
                )?;
            let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Unsupported {
                message: "temporal interventional distribution needs one outcome".into(),
            })?;
            if !q.conditioning.is_empty() {
                return Err(CausalError::Unsupported {
                    message: "temporal interventional distribution is unconditional".into(),
                });
            }
            Ok((
                treatment,
                outcome,
                None,
                super::temporal_path::temporal_cross_world_hard_level(&q.interventions[0])?,
                0.0,
            ))
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal cross-world route does not estimate this query".into(),
        }),
    }
}

fn cross_world_point(
    data: &TimeSeriesData,
    query: &CausalQuery,
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    treatment: VariableId,
    outcome: VariableId,
    mediator: Option<VariableId>,
    active: f64,
    control: f64,
    treatment_offset: i32,
    outcome_offset: i32,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<f64, CausalError> {
    let fit = |outcome: VariableId, treatments: &[VariableId], contemporaneous: &[VariableId]| {
        antecedent_estimate::temporal_linear_slopes(
            data,
            estimand,
            indexer,
            &antecedent_estimate::TemporalSlopeSpec {
                outcome: outcome,
                treatments: treatments,
                treatment_offset: treatment_offset,
                outcome_offset: outcome_offset,
                contemporaneous: contemporaneous,
            },
            policy,
            prior_scale,
        )
        .map_err(CausalError::from)
    };
    let delta = active - control;
    match query {
        CausalQuery::NestedCounterfactual(_) => {
            let mediator = mediator.expect("nested mediator");
            let (_, slopes) = fit(outcome, &[treatment, mediator], &[mediator])?;
            Ok(slopes[0] * delta)
        }
        CausalQuery::PathSpecific(_) => {
            let mediator = mediator.expect("path mediator");
            let (_, through) = fit(mediator, &[treatment], &[])?;
            let (_, onward) = fit(outcome, &[treatment, mediator], &[mediator])?;
            Ok(through[0] * onward[1] * delta)
        }
        CausalQuery::Counterfactual(_) => {
            let (_, slopes) = fit(outcome, &[treatment], &[])?;
            Ok(slopes[0] * delta)
        }
        CausalQuery::Distribution(_) => {
            let (intercept, slopes) = fit(outcome, &[treatment], &[])?;
            Ok(intercept + slopes[0] * active)
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal cross-world route does not estimate this query".into(),
        }),
    }
}

pub(super) fn temporal_class_point_family(query: &CausalQuery) -> bool {
    match query {
        CausalQuery::TemporalEffect(effect) => effect.effect_modifiers.len() == 1,
        CausalQuery::Response(response) => {
            response.temporal.is_some()
                && matches!(
                    response.functional,
                    ResponseFunctional::AverageDerivative { .. }
                        | ResponseFunctional::PointDerivative { .. }
                        | ResponseFunctional::DirectionalDerivative { .. }
                        | ResponseFunctional::Jacobian { .. }
                )
        }
        CausalQuery::NestedCounterfactual(_)
        | CausalQuery::PathSpecific(_)
        | CausalQuery::Counterfactual(_)
        | CausalQuery::Distribution(_) => true,
        _ => false,
    }
}

pub(super) fn cross_world_plan_variables(
    query: &CausalQuery,
) -> Result<(VariableId, VariableId), CausalError> {
    let (treatment, outcome, _, _, _) = cross_world_endpoints(query)?;
    Ok((treatment, outcome))
}

fn load_class_posterior_cache(
    study: &super::Study,
    gp: &GraphPosterior,
    data: &TimeSeriesData,
    probe: &TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<
    (Arc<crate::analysis::prepared::CachedTemporalClassPosteriorIdentification>, bool),
    CausalError,
> {
    if let Some(cache) = study.temporal_class_posterior_identification_cache.as_ref() {
        return Ok((Arc::clone(cache), true));
    }
    let variables =
        data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
    let cache = crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
        gp,
        &variables,
        probe,
        study.max_completions,
        ctx,
    )?;
    Ok((Arc::new(cache), false))
}

fn derivative_probe(query: &ResponseQuery) -> Result<TemporalEffectQuery, CausalError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "class-aware temporal derivative requires TemporalResponseSpec".into(),
    })?;
    let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
    Ok(TemporalEffectQuery {
        treatment,
        outcome,
        policy: temporal.policy.clone(),
        control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
        active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
        horizon_steps: temporal.horizons[0],
        max_history_lag: temporal.max_history_lag,
        target_population: query.target_population.clone(),
        effect_modifiers: std::sync::Arc::from([]),
    })
}

fn derivative_on_envelope(
    data: &TimeSeriesData,
    query: &ResponseQuery,
    bundle: &antecedent_identify::TemporalClassEnvelope,
    graph_class: GraphClass,
    treatment_offset: i32,
    outcome_offset: i32,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<
    (
        CausalResponse,
        IdentifiedEstimand,
        IdentificationResult,
        Vec<Diagnostic>,
        crate::result::StructuralResponseMixture,
        IdentificationStatus,
    ),
    CausalError,
> {
    let envelope = &bundle.envelope;
    if matches!(envelope.status, IdentificationStatus::NotIdentified)
        || envelope.identified_weight.0 <= 0.0
    {
        return Err(CausalError::not_identified(
            envelope.status,
            envelope.truncated_completions > 0,
            "temporal class query not identified (no identified mass in envelope)",
        ));
    }
    let mut responses = Vec::new();
    let mut values = vec![None; envelope.cases.len()];
    let mut primary_estimand = None;
    for (index, (case, indexer)) in envelope.cases.iter().zip(bundle.indexers.iter()).enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let response = super::temporal_path::temporal_linear_derivative_response(
            data,
            query,
            &estimand,
            indexer,
            &case.result,
            treatment_offset,
            outcome_offset,
            policy,
            prior_scale,
        )?;
        let scalar = super::response_path::response_scalar_summary(&response).0;
        if scalar.is_finite() {
            values[index] = Some(scalar);
        }
        if primary_estimand.is_none() {
            primary_estimand = Some(estimand);
        }
        responses.push((case.weight.0, response));
    }
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "temporal class envelope had no estimable identified completions".into(),
    })?;
    let response = mix_derivative_responses(&responses)?;
    let mut diagnostics =
        vec![super::temporal_path::temporal_class_envelope_diagnostic(envelope, graph_class)];
    diagnostics.push(Diagnostic::new(
        "estimate.temporal.linear_derivative",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "published value is the completion-weighted slope of the linear unfolded temporal fit; no interval is published",
    ));
    diagnostics.extend(response.support.warnings.iter().cloned());
    let identification =
        envelope_to_identification_result_for(envelope, CausalQuery::Response(query.clone()));
    let structural = super::temporal_path::temporal_class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        None,
        &values,
    );
    Ok((response, estimand, identification, diagnostics, structural, envelope.status))
}
