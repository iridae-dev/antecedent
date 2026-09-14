// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl super::Study {
    pub(super) fn execute_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::CounterfactualQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let (treatment, active, control) = binary_cf_interventions(query)?;
        let outcome = query.outcomes[0];
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static_query(
                    IdentifierId::GcmParametric,
                    graph,
                    &CausalQuery::Counterfactual(query.clone()),
                )?;
                let estimand = identification.estimands[0].clone();
                Ok((identification, estimand))
            })?;
        if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() || cfg.prior.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian counterfactuals require a declared mechanism mapping; \
                              a coefficient artifact cannot be applied as an isotropic GCM prior \
                              or hydrated onto fitted GCM mechanisms",
                });
            }
        }
        let fitted = fit_gcm(graph.clone(), data)?;
        let assignments = format!("{:?}", fitted.assignments);
        let mechanism_assignments = fitted.assignments.clone();
        let base_model = fitted.model.clone();
        let ite = counterfactual_ite(fitted.model, data, treatment, outcome, active, control, ctx)?;
        let (estimate, posterior, ite) = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            let n_draws = bayesian_draw_count(&self.inference)?;
            let mut values = Vec::with_capacity(n_draws);
            let mut unit_sum = vec![0.0; ite.unit_effects.len()];
            let mut rng = ctx.rng.stream(0x0CF0);
            let n = data.row_count();
            for _ in 0..n_draws {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_ESTIMATE_POINT,
                    });
                }
                let weights: Vec<f64> = (0..n)
                    .map(|_| (-rng.next_f64().max(f64::MIN_POSITIVE).ln()).max(0.0))
                    .collect();
                let store = crate::gcm::MechanismRegistry::standard()
                    .refit_weighted(&base_model, data, &mechanism_assignments, &weights)
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let draw_model = base_model.clone().with_mechanisms(store);
                let draw_ite =
                    counterfactual_ite(draw_model, data, treatment, outcome, active, control, ctx)?;
                if draw_ite.unit_effects.len() != unit_sum.len() {
                    return Err(CausalError::Compile {
                        message: "counterfactual Bayesian draw changed the unit set".into(),
                    });
                }
                values.push(draw_ite.mean_ite);
                for (sum, unit) in unit_sum.iter_mut().zip(draw_ite.unit_effects.iter()) {
                    *sum += *unit;
                }
            }
            let posterior = counterfactual_posterior(
                values,
                identification.required_assumptions.clone(),
                identification.status,
            )?;
            let eq = posterior.effect_column().ok_or_else(|| CausalError::Compile {
                message: "counterfactual posterior missing effect column".into(),
            })?;
            let scale = n_draws.max(1) as f64;
            let mut ite = ite;
            ite.unit_effects = std::sync::Arc::from(
                unit_sum.into_iter().map(|sum| sum / scale).collect::<Vec<_>>(),
            );
            ite.mean_ite = posterior.summaries.mean[eq];
            let mut estimate = EffectEstimate::new(
                posterior.summaries.mean[eq],
                posterior.summaries.sd[eq],
                posterior.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            estimate.se_analytic = posterior.summaries.sd[eq];
            (estimate, Some(posterior), ite)
        } else {
            (
                EffectEstimate::new(
                    ite.mean_ite,
                    f64::NAN,
                    identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                None,
                ite,
            )
        };
        let observed = data.float64_values(treatment)?;
        let min = observed.iter().copied().fold(f64::INFINITY, f64::min);
        let max = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let diagnostics = vec![
            Diagnostic::new(
                "gcm.counterfactual.mechanisms",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                assignments,
            ),
            Diagnostic::new(
                "gcm.counterfactual",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "noise_inference={:?}; control={control}; active={active}",
                    ite.noise_inference
                ),
            ),
            Diagnostic::new(
                "gcm.counterfactual.support",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "observed treatment range=[{min},{max}]; extrapolative={}",
                    control < min || control > max || active < min || active > max
                ),
            ),
            if posterior.is_some() {
                Diagnostic::new(
                    "gcm.counterfactual.bayesian",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "Dirichlet row-weight posterior of fitted GCM mechanisms, conditional on the selected families and empirical support; each draw abducts–acts–predicts on the original units; published unit_effects are the posterior mean of those per-unit ITEs. The posterior effect draws, standard error, and credible interval describe mean_ite (the average contrast over the observed units) and carry mechanism-refit uncertainty only: abducted disturbances are recomputed from the observed rows, not drawn, so this is neither a predictive interval for any single unit's effect nor an interval for a population average beyond the observed units, and unit_effects carry no interval.",
                )
            } else {
                Diagnostic::new(
                    "gcm.counterfactual.uncertainty_unavailable",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "Unit effects condition on fitted mechanisms and abducted disturbances; sampling uncertainty is unavailable.",
                )
            },
        ];
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GcmParametric,
            estimator_id: EstimatorId::GcmFit,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: diagnostics,
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                gcm: Some(GcmSlot::Counterfactual(ite)),
                bootstrap_replicates_requested: Some(None),
                identify_provenance: Some(provenance_ids(
                    "identify.gcm_parametric",
                    "identify.gcm_parametric",
                )),
                estimate_provenance: Some(provenance_ids(
                    "counterfactual.aap",
                    "counterfactual.aap",
                )),
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_anomaly(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::AnomalyAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        let (_, _, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                Ok(parametric_scm_identification(
                    CausalQuery::AnomalyAttribution(query.clone()),
                    outcome,
                    outcome,
                ))
            })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let scores = anomaly_attribution(
            &fitted.model,
            data,
            query.targets.iter().copied(),
            query.max_units,
        )?;
        Ok(self.finish_gcm(
            physical,
            CausalQuery::AnomalyAttribution(query.clone()),
            outcome,
            outcome,
            nan_effect(),
            started,
            GcmSlot::Anomaly(scores),
            Vec::new(),
            identify_cached,
        ))
    }

    pub(super) fn execute_change_attribution(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::ChangeAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let (_, _, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                Ok(parametric_scm_identification(
                    CausalQuery::ChangeAttribution(query.clone()),
                    query.outcome,
                    query.outcome,
                ))
            })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_distribution_change(
            &fitted.model,
            data,
            query,
            &antecedent_attribution::DistributionChangeOptions::default(),
            ctx,
        )?;
        let estimate = EffectEstimate::new(
            result.total_change,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        Ok(self.finish_gcm(
            physical,
            CausalQuery::ChangeAttribution(query.clone()),
            query.outcome,
            query.outcome,
            estimate,
            started,
            GcmSlot::Change(result),
            Vec::new(),
            identify_cached,
        ))
    }

    pub(super) fn execute_mechanism_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MechanismChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let detections = mechanism_change_detection(
            &fitted.model,
            data,
            query,
            antecedent_attribution::MechanismChangeMethod::LikelihoodRatio,
            ctx,
        )?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        Ok(self.finish_gcm(
            physical,
            CausalQuery::MechanismChange(query.clone()),
            outcome,
            outcome,
            nan_effect(),
            started,
            GcmSlot::Mechanism(detections),
            Vec::new(),
            false,
        ))
    }

    pub(super) fn execute_unit_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::UnitChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_unit_change(&fitted.model, data, query, ctx)?;
        Ok(self.finish_gcm(
            physical,
            CausalQuery::UnitChange(query.clone()),
            query.outcome,
            query.outcome,
            nan_effect(),
            started,
            GcmSlot::Unit(result),
            Vec::new(),
            false,
        ))
    }

    fn finish_gcm(
        &self,
        physical: &PhysicalExecutionPlan,
        query: CausalQuery,
        treatment: VariableId,
        outcome: VariableId,
        estimate: EffectEstimate,
        started: Instant,
        slot: GcmSlot,
        diagnostics: Vec<Diagnostic>,
        identify_cached: bool,
    ) -> StudyResult {
        let (identification, estimand) = parametric_scm_identification(query, treatment, outcome);
        self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GcmParametric,
            estimator_id: EstimatorId::GcmFit,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                gcm: Some(slot),
                empty_provenance: true,
                ..Default::default()
            },
        })
    }
}

fn counterfactual_posterior(
    values: Vec<f64>,
    mut assumptions: antecedent_core::AssumptionSet,
    identification: antecedent_core::IdentificationStatus,
) -> Result<CausalPosterior, CausalError> {
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("counterfactual.weighted_mechanisms"),
            description: Arc::from("Dirichlet row-weight posterior of standard mechanism fits, conditional on selected mechanism families and empirical support; abduction is repeated on the original units for every draw. This is not a parametric coefficient-prior posterior. The interval is for mean_ite over the observed units and reflects mechanism-refit uncertainty only; it is not a unit-level predictive interval."),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("gcm.fit.bayesian") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let schema = antecedent_prob::PosteriorSchema {
        quantities: std::sync::Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
            name: std::sync::Arc::from("ite"),
        }]),
    };
    let n = values.len();
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n,
        std::sync::Arc::<[f64]>::from(values),
    )
    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let summaries = draws.summarize();
    Ok(CausalPosterior {
        subsampled_out_mass: 0.0,
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: antecedent_prob::InferenceDiagnostics::analytic("gcm.fit.bayesian"),
        assumptions,
        unidentified_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
    })
}
