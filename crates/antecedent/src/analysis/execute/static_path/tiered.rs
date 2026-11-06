//! Tiered-background average-effect execution and scenario envelopes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{CheckedUnknownTiered, Study};
use super::{
    AverageEffectQuery, CausalError, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimationWorkspace, EstimatorId, EstimatorSpec, ExecuteRun, ExecutionContext,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, Instant, OverlapPolicy, PhysicalExecutionPlan, RefutationRun,
    StaticEstimateInput, StaticEstimateWorkspaces, Strategy, StudyResult, TabularData, elapsed_ns,
    estimate_static_effect, identification_from_cache_or, scalar_identified_set,
    static_aligned_influence,
};

impl Study {
    pub(in crate::analysis::execute) fn execute_tiered_average(
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
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
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
        let data_est = crate::analysis::helpers::apply_scalar_outcome_functional(
            data,
            query.outcome,
            &query.outcome_functional,
        )?;
        let mut ws = StaticEstimateWorkspaces::default();
        let spec = self
            .declared
            .estimator_spec
            .as_deref()
            .cloned()
            .unwrap_or(EstimatorSpec::Default(estimator_id));
        let estimate = estimate_static_effect(
            &spec,
            StaticEstimateInput {
                data: &data_est,
                estimand: &estimand,
                query,
                bootstrap_replicates: self.settings.bootstrap_replicates,
                overlap_policy: self.declared.overlap_policy,
                population_registry: self.declared.population_registry.as_ref(),
                sampling_design: self.declared.sampling_design,
            },
            identification.required_assumptions.clone(),
            ctx,
            &mut ws,
        )?;
        let mut extra_diagnostics = Vec::new();
        let estimate = crate::analysis::helpers::attach_average_functional_grid(
            estimate,
            crate::analysis::helpers::AverageTarget { data, query, estimand: &estimand },
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
            suite: self.settings.refute,
            estimator: estimator_id.as_str(),
            custom: &self.declared.custom_validators,
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
        if matches!(self.settings.inference, crate::inference::InferenceMode::Bayesian(_)) {
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
        let spec = self
            .declared
            .estimator_spec
            .as_deref()
            .cloned()
            .unwrap_or(EstimatorSpec::Default(estimator_id));
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
                    bootstrap_replicates: self.settings.bootstrap_replicates,
                    overlap_policy: self.declared.overlap_policy,
                    population_registry: self.declared.population_registry.as_ref(),
                    sampling_design: self.declared.sampling_design,
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
        let data_est = crate::analysis::helpers::apply_scalar_outcome_functional(
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
            extra_diagnostics.extend(
                super::super::pag_path::static_class_identified_set_diagnostic(
                    mixture,
                    "tiered_unknown",
                ),
            );
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
        aggregation_policy: None,
    })
}
