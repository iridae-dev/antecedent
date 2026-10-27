// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalPosterior, CausalQuery, Dag, DenseNodeId, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, GcmSlot,
    IdentificationStatus, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedResultContext, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, Strategy, StudyResult, TableView, TabularData, VariableId,
    anomaly_attribution_query_with, attribute_distribution_change, attribute_unit_change,
    bayesian_draw_count, binary_cf_interventions, elapsed_ns,
    finish_identified_execute_with_context, fit_gcm, identification_from_cache_or,
    identify_static_query, map_mechanism_fit, mechanism_change_detection, nan_effect,
    parametric_scm_identification, provenance_ids, shared_identification_from_cache_or,
};
mod contrast_set;
use crate::analysis::proof_support::proof_supplies_full;
use antecedent_core::StreamDomain;

/// Inference procedure over the heterogeneity best-score counterfactual fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CounterfactualProcedure {
    Point,
    DirichletRowWeights {
        draws: usize,
    },
    /// Frequentist whole-procedure multinomial row bootstrap of the mean ITE.
    RowBootstrap {
        replicates: u32,
    },
}

impl CounterfactualProcedure {
    pub(crate) fn for_inference(inference: &InferenceMode) -> Result<Self, CausalError> {
        match inference {
            InferenceMode::Frequentist => Ok(Self::Point),
            InferenceMode::Bayesian(config) => {
                if config.has_transfer() {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian counterfactuals require a declared mechanism mapping; a coefficient artifact cannot be applied as an isotropic GCM prior or hydrated onto fitted GCM mechanisms",
                    });
                }
                let draws = bayesian_draw_count(inference)?;
                if draws < 2 {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian counterfactuals require at least two posterior draws",
                    });
                }
                Ok(Self::DirichletRowWeights { draws })
            }
        }
    }

    /// The Frequentist point procedure becomes the mechanism-refit row bootstrap when a
    /// replicate count is requested and the declared design resamples subjects.
    ///
    /// # Errors
    ///
    /// A declared design that does not allow a subject resample.
    pub(crate) fn with_bootstrap(
        self,
        replicates: u32,
        design: Option<antecedent_core::DeclaredSamplingDesign>,
    ) -> Result<Self, CausalError> {
        match self {
            Self::Point if replicates > 0 => {
                super::gcm_bootstrap::require_subject_resampling(design)?;
                Ok(Self::RowBootstrap { replicates })
            }
            other => Ok(other),
        }
    }

    fn matches_inference(&self, inference: &InferenceMode) -> bool {
        match (self, inference) {
            (Self::Point | Self::RowBootstrap { .. }, InferenceMode::Frequentist) => true,
            (Self::DirichletRowWeights { draws }, InferenceMode::Bayesian(config)) => {
                *draws == config.n_draws
            }
            _ => false,
        }
    }
}

#[derive(Clone)]
pub(crate) struct CheckedCounterfactualPlan {
    operation: crate::gcm::CheckedCounterfactualOperation,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    identify_cached: bool,
    inference: InferenceMode,
    procedure: CounterfactualProcedure,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedCounterfactualPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedCounterfactualPlan")
            .field("operation", &self.operation)
            .field("identification", &self.cache.identification)
            .field("estimand", &self.cache.estimand)
            .field("identify_cached", &self.identify_cached)
            .field("inference", &self.inference)
            .field("procedure", &self.procedure)
            .field("physical", &self.physical)
            .finish_non_exhaustive()
    }
}

impl CheckedCounterfactualPlan {
    pub(crate) fn new(
        operation: crate::gcm::CheckedCounterfactualOperation,
        cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
        identify_cached: bool,
        inference: InferenceMode,
        procedure: CounterfactualProcedure,
        physical: std::sync::Arc<PhysicalExecutionPlan>,
        result_context: IdentifiedResultContext,
    ) -> Result<Self, CausalError> {
        let (identification, estimand) = (&cache.identification, &cache.estimand);
        if !matches!(&identification.query, CausalQuery::Counterfactual(query) if query == operation.query())
            || result_context.query != CausalQuery::Counterfactual(operation.query().clone())
            || result_context.inference != inference
            || !procedure.matches_inference(&inference)
            || !matches!(
                identification.status,
                IdentificationStatus::IdentifiedUnderParametricRestrictions
            )
            || !proof_supplies_full(identification, estimand)
            || !physical.binding().names(IdentifierId::GcmParametric, EstimatorId::GcmFit)
        {
            return Err(CausalError::Compile { message: "checked counterfactual plan components do not agree on target, inference, identification, or procedure".into() });
        }
        Ok(Self {
            operation,
            cache,
            identify_cached,
            inference,
            procedure,
            physical,
            result_context,
        })
    }

    pub(crate) fn operation(&self) -> &crate::gcm::CheckedCounterfactualOperation {
        &self.operation
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        execute_checked_counterfactual_plan(self, data, ctx)
    }
}

#[cfg(test)]
mod checked_counterfactual_plan_tests {
    use super::{CounterfactualProcedure, InferenceMode};
    use crate::inference::BayesianConfig;

    #[test]
    fn refuses_procedure_inference_mismatch_and_wrong_draw_count() {
        let frequentist = InferenceMode::Frequentist;
        assert!(
            !CounterfactualProcedure::DirichletRowWeights { draws: 32 }
                .matches_inference(&frequentist)
        );

        let bayesian = InferenceMode::Bayesian(BayesianConfig::laplace());
        assert!(!CounterfactualProcedure::Point.matches_inference(&bayesian));
        let wrong_draws = CounterfactualProcedure::DirichletRowWeights { draws: 33 };
        assert!(!wrong_draws.matches_inference(&bayesian));
        let matching = CounterfactualProcedure::for_inference(&bayesian).unwrap();
        assert!(matching.matches_inference(&bayesian));
    }
}

/// Selection record for the mechanisms diagnostic. Family ids and typing labels
/// are the stable names; fitted coefficients are not part of the identity.
fn mechanism_audit(assignments: &[crate::gcm::MechanismAssignment]) -> String {
    assignments
        .iter()
        .map(|assignment| {
            format!(
                "node={};variable={};selected={};typing={}",
                assignment.node.raw(),
                assignment.variable.raw(),
                assignment.selected.id(),
                mechanism_typing_label(assignment.typing)
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn mechanism_typing_label(typing: antecedent_model::MechanismTyping) -> &'static str {
    match typing {
        antecedent_model::MechanismTyping::DeclaredDiscrete => "declared_discrete",
        antecedent_model::MechanismTyping::LowCardinality => "low_cardinality",
        antecedent_model::MechanismTyping::Continuous => "continuous",
    }
}

/// Historical diagnostic token for a noise-inference kind, not its Debug text.
fn noise_inference_label(kind: antecedent_counterfactual::NoiseInferenceKind) -> &'static str {
    match kind {
        antecedent_counterfactual::NoiseInferenceKind::Invertible => "Invertible",
        antecedent_counterfactual::NoiseInferenceKind::PosteriorNoise => "PosteriorNoise",
        antecedent_counterfactual::NoiseInferenceKind::AssumedNoise => "AssumedNoise",
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one counterfactual execution: the procedure's estimate, the homogeneity verdict and the identified finish share every fitted product"
)]
fn execute_checked_counterfactual_plan(
    plan: &CheckedCounterfactualPlan,
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let started = Instant::now();
    let identification = &plan.cache.identification;
    let estimand = plan.cache.estimand.clone();
    let identify_cached = plan.identify_cached;

    if let Some(target) = plan.operation.contrast_target() {
        return contrast_set::execute_contrast_set_plan(plan, target, data, ctx);
    }
    let query = plan.operation.query();
    let graph = plan.operation.graph();
    let (treatment, active, control) = binary_cf_interventions(query)?;
    let outcome = query.outcomes[0];
    let (fitted, ite) = plan.operation.execute(data, ctx)?;
    let assignments = mechanism_audit(&fitted.assignments);
    let mechanism_assignments = fitted.assignments.clone();
    let refit = CounterfactualRefit {
        data,
        fitted: &fitted,
        contrast: UnitContrast { treatment, outcome, active, control },
        assumptions: &identification.required_assumptions,
        status: identification.status,
    };
    let mut bootstrap_note = None;
    let (estimate, posterior, ite) = if let CounterfactualProcedure::DirichletRowWeights { draws } =
        &plan.procedure
    {
        dirichlet_counterfactual_estimate(&refit, ite, *draws, ctx)?
    } else if let CounterfactualProcedure::RowBootstrap { replicates } = &plan.procedure {
        let (estimate, note) = bootstrap_counterfactual_estimate(&refit, &ite, *replicates, ctx)?;
        bootstrap_note = Some(note);
        (estimate, None, ite)
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
    let verdict = homogeneity_verdict(graph, &mechanism_assignments, treatment, outcome);
    let mut estimate = estimate;
    estimate.heterogeneity.unit_effects_homogeneous =
        matches!(verdict, HomogeneityVerdict::Structural(_));
    let (min, max) = observed_range(&data.float64_values(treatment)?);
    let pooled_extrapolative = control < min || control > max || active < min || active > max;
    let support = unit_support(
        data,
        graph,
        &mechanism_assignments,
        &ite.exogenous,
        UnitContrast { treatment, outcome, active, control },
    )?;
    let mut ite = ite;
    ite.unit_extrapolative = Some(std::sync::Arc::from(support.extrapolation_flags()));
    let mut diagnostics = vec![
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
                "noise_inference={}; control={control}; active={active}",
                noise_inference_label(ite.noise_inference)
            ),
        ),
        support.support_diagnostic(min, max, pooled_extrapolative),
        counterfactual_uncertainty_diagnostic(posterior.is_some()),
    ];
    let bootstrapped = bootstrap_note.is_some();
    if bootstrapped {
        diagnostics.retain(|diagnostic| {
            diagnostic.code.as_ref() != "gcm.counterfactual.uncertainty_unavailable"
        });
    }
    diagnostics.extend(bootstrap_note);
    homogeneity_diagnostics(data, outcome, ite.mean_ite, &verdict, &mut diagnostics);
    Ok(finish_identified_execute_with_context(
        &plan.result_context,
        Some(data),
        IdentifiedExecuteFinish::new(
            &plan.physical,
            identification.clone(),
            estimand,
            estimate,
            Strategy::new(IdentifierId::GcmParametric, EstimatorId::GcmFit),
            treatment,
            outcome,
        )
        .identify_cached(identify_cached)
        .extra_diagnostics(diagnostics)
        .wall_time_ns(elapsed_ns(started))
        .extras(IdentifiedExecuteExtras {
            gcm: Some(GcmSlot::Counterfactual(ite)),
            bootstrap_replicates_requested: Some(match plan.procedure {
                CounterfactualProcedure::RowBootstrap { replicates } if bootstrapped => {
                    Some(replicates)
                }
                _ => None,
            }),
            identify_provenance: Some(provenance_ids(
                "identify.gcm_parametric",
                "identify.gcm_parametric",
            )),
            estimate_provenance: Some(provenance_ids("counterfactual.aap", "counterfactual.aap")),
            n_draws: posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
            posterior,
            ..Default::default()
        }),
    ))
}

/// Smallest and largest observed value of a column.
fn observed_range(values: &[f64]) -> (f64, f64) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// Inputs every Dirichlet row-weight refit of a counterfactual contrast reads.
struct CounterfactualRefit<'a> {
    data: &'a TabularData,
    fitted: &'a crate::gcm::FittedGcm,
    contrast: UnitContrast,
    assumptions: &'a antecedent_core::AssumptionSet,
    status: IdentificationStatus,
}

/// Bayesian-bootstrap refit of the mechanisms per draw, publishing the mean-ITE posterior and per-unit intervals.
fn dirichlet_counterfactual_estimate(
    refit: &CounterfactualRefit<'_>,
    ite: crate::gcm::IteResult,
    n_draws: usize,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>, crate::gcm::IteResult), CausalError> {
    let CounterfactualRefit { data, fitted, contrast, assumptions, status } = *refit;
    let UnitContrast { treatment, outcome, active, control } = contrast;
    let base_model = &fitted.model;
    let mechanism_assignments = &fitted.assignments;
    let n_units = ite.unit_effects.len();
    let mut values = Vec::with_capacity(n_draws);
    // Column-major `units × draws`: one posterior column per unit, read by
    // the same summary that publishes the mean-ITE interval.
    let mut unit_draws = vec![0.0; n_units * n_draws];
    let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0x0CF0);
    // One stream through every draw: a categorical node's rank-preserving noise is drawn once
    // per posterior draw of the mechanisms, independently across draws.
    let mut noise_rng = ctx.rng.stream_for(StreamDomain::Counterfactual, 0x0CF1);
    let n = data.row_count();
    for draw in 0..n_draws {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, n, false);
        // The same registry the point fit selected from, so every family a
        // node can have selected is one `refit_weighted` accepts.
        let store = crate::gcm::MechanismRegistry::with_heterogeneity_families()
            .refit_weighted(base_model, data, mechanism_assignments, &weights)
            .map_err(|e| match e {
                crate::gcm::ModelError::NotConverged { .. } => map_mechanism_fit(e),
                other => CausalError::rejected(other),
            })?;
        let draw_model = base_model.clone().with_mechanisms(store);
        let draw_ite = crate::gcm::counterfactual_ite_with_noise(
            draw_model,
            data,
            (treatment, outcome, active, control),
            ctx,
            &mut noise_rng,
        )?;
        if draw_ite.unit_effects.len() != n_units {
            return Err(CausalError::Compile {
                message: "counterfactual Bayesian draw changed the unit set".into(),
            });
        }
        values.push(draw_ite.mean_ite);
        for (unit, effect) in draw_ite.unit_effects.iter().enumerate() {
            unit_draws[unit * n_draws + draw] = *effect;
        }
    }
    let posterior = counterfactual_posterior(values, assumptions.clone(), status)?;
    let eq = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "counterfactual posterior missing effect column".into(),
    })?;
    // Published unit effects stay the plain draw average (sequential sum,
    // divided once), exactly as before per-unit intervals existed.
    let scale = n_draws.max(1) as f64;
    let unit_means: Vec<f64> = (0..n_units)
        .map(|unit| {
            unit_draws[unit * n_draws..(unit + 1) * n_draws].iter().fold(0.0, |s, v| s + v) / scale
        })
        .collect();
    let unit_summary = unit_effect_posterior_summary(unit_draws, n_units, n_draws)?;
    let mut ite = ite;
    ite.unit_effects = std::sync::Arc::from(unit_means);
    ite.unit_effect_intervals = Some(crate::gcm::UnitEffectIntervals {
        lower: std::sync::Arc::clone(&unit_summary.q025),
        upper: std::sync::Arc::clone(&unit_summary.q975),
        level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
        method: antecedent_core::IntervalMethod::UnitPosteriorQuantile.as_str(),
    });
    ite.mean_ite = posterior.summaries.mean[eq];
    let mut estimate = EffectEstimate::new(
        posterior.summaries.mean[eq],
        posterior.summaries.sd[eq],
        posterior.assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    estimate.uncertainty.analytic.se = posterior.summaries.sd[eq];
    Ok((estimate, Some(posterior), ite))
}

/// Frequentist whole-procedure row bootstrap of the mean ITE: each replicate refits every
/// mechanism under multinomial row counts (conditional on the selected families) and
/// abducts–acts–predicts on the original units. The point stays the original fit's mean
/// ITE; the replicates give its bootstrap standard error. Per-unit intervals are not
/// formed. Failed replicates are retained as failed mass.
fn bootstrap_counterfactual_estimate(
    refit: &CounterfactualRefit<'_>,
    ite: &crate::gcm::IteResult,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Diagnostic), CausalError> {
    let CounterfactualRefit { data, fitted, contrast, assumptions, status: _ } = *refit;
    let UnitContrast { treatment, outcome, active, control } = contrast;
    let n = data.row_count();
    let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, n)?;
    let registry = crate::gcm::MechanismRegistry::with_heterogeneity_families();
    let cancelled = || ctx.cancellation.is_cancelled();
    let mut values = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
    let mut failed = 0_u32;
    for replicate in 0..replicates {
        if cancelled() {
            return Err(CausalError::Cancelled { stage: super::super::stage::STAGE_UNCERTAINTY });
        }
        let mut rng = super::gcm_bootstrap::replicate_stream(ctx, 0x0CF0_B007, replicate);
        let counts = super::gcm_bootstrap::bootstrap_counts(&mut rng, n);
        let store = match registry.refit_frequency_weighted_polled(
            &fitted.model,
            data,
            &fitted.assignments,
            &counts,
            &cancelled,
        ) {
            Ok(store) => store,
            Err(_) if cancelled() => {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            Err(_) => {
                failed += 1;
                continue;
            }
        };
        // One noise stream per replicate: a categorical node's rank-preserving noise is
        // drawn once per replicate, keyed by the replicate.
        let mut noise_rng = ctx.rng.stream_for(
            StreamDomain::Counterfactual,
            0x0CF1_B007_u64.wrapping_add(u64::from(replicate)),
        );
        match crate::gcm::counterfactual_ite_with_noise(
            fitted.model.clone().with_mechanisms(store),
            data,
            (treatment, outcome, active, control),
            ctx,
            &mut noise_rng,
        ) {
            Ok(draw) if draw.unit_effects.len() == ite.unit_effects.len() => {
                values.push(draw.mean_ite);
            }
            Err(_) if cancelled() => {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            _ => failed += 1,
        }
    }
    let accounting = super::gcm_bootstrap::accounting(&values, failed, replicates);
    let mut estimate = EffectEstimate::new(
        ite.mean_ite,
        f64::NAN,
        assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    estimate.uncertainty.bootstrap = accounting;
    let note = super::gcm_bootstrap::bootstrap_diagnostic(
        "mean ITE over the observed units (per-unit intervals are not formed)",
        &accounting,
    );
    Ok((estimate, note))
}

/// Explains why no frequentist interval exists, or what the Bayesian interval does and does not carry.
fn counterfactual_uncertainty_diagnostic(bayesian: bool) -> Diagnostic {
    if bayesian {
        Diagnostic::new(
            "gcm.counterfactual.bayesian",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Dirichlet row-weight posterior of fitted GCM mechanisms, conditional on the selected families and empirical support; the family selection is made once on the unweighted data and is not revisited per draw. Each draw abducts–acts–predicts on the original units, with the factual and counterfactual worlds sharing that draw's abducted disturbances; published unit_effects are the posterior mean of those per-unit ITEs and unit_effect_intervals the equal-tailed posterior quantiles of each unit's ITE draws at the reported level. Disturbances of invertible mechanisms are recomputed from the observed rows under each draw's parameters, not drawn, so for them the intervals carry mechanism-refit uncertainty only. A categorical (many-to-one) node's rank-preserving disturbance is drawn once per posterior draw from one independent stream, so its noise posterior is integrated into the intervals of the units downstream of it, inseparably from the parameter uncertainty (noise_inference=PosteriorNoise in diagnostic gcm.counterfactual says when this applies). A per-unit interval is a credible interval for that observed unit's contrast under the fitted mechanism, not a predictive interval for a new unit, and the mean_ite interval is for the average over the observed units, not a population beyond them.",
        )
    } else {
        Diagnostic::new(
            "gcm.counterfactual.uncertainty_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Unit effects condition on fitted mechanisms and abducted disturbances; sampling uncertainty is unavailable. No Frequentist per-unit interval is published (counterfactual.unit_effect_intervals is None): a unit's ITE Y_i(a)-Y_i(a0) is not identified from data, only computed under the fitted SCM; its value depends jointly on every path mechanism's parameters and on disturbances abducted with those same parameters (or, downstream of a discrete mechanism, on one seeded noise draw); and each mechanism family is selected on this data, so a delta-method interval conditional on the selected family would ignore selection. The refit construction that does carry this uncertainty is the Bayesian cell's Dirichlet mechanism-refit posterior, whose per-unit unit_posterior_quantile intervals are published under inference=Bayesian. For a frequentist pointwise interval on the conditional average effect tau(x) rather than a unit ITE, see DrLearner::fit_pointwise_profiles (uncalibrated).",
        )
    }
}

/// Warns when the outcome mechanism admits no effect modification, or records the rejected heterogeneity families.
fn homogeneity_diagnostics(
    data: &TabularData,
    outcome: VariableId,
    mean_ite: f64,
    verdict: &HomogeneityVerdict,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if let HomogeneityVerdict::Structural(family) = verdict {
        let outcome_name = data
            .schema()
            .get(outcome)
            .map_or_else(|_| format!("v{}", outcome.raw()), |v| v.name.to_string());
        diagnostics.push(
            Diagnostic::new(
                "gcm.counterfactual.unit_effects_homogeneous",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "mechanism family {} for {outcome_name} admits no effect \
                         modification; unit_effects equal the mechanism slope for every unit",
                    family.id()
                ),
            )
            .with_fields([
                ("outcome", outcome_name),
                ("family", family.id().to_string()),
                ("unit_effect", mean_ite.to_string()),
            ]),
        );
    }
    if let HomogeneityVerdict::Empirical { rejected } = verdict {
        diagnostics.push(heterogeneity_rejected_diagnostic(data, rejected));
    }
}

/// The result slots of one GCM attribution finish, apart from the plan and clock.
struct GcmFinish {
    query: CausalQuery,
    treatment: VariableId,
    outcome: VariableId,
    estimate: EffectEstimate,
    slot: GcmSlot,
    diagnostics: Vec<Diagnostic>,
    identify_cached: bool,
}

impl super::Study {
    pub(super) fn execute_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::CounterfactualQuery,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let operation =
            crate::gcm::CheckedCounterfactualOperation::compile(graph.clone(), query.clone())?;
        self.execute_counterfactual_checked(data, physical, ctx, &operation)
    }

    pub(crate) fn execute_counterfactual_checked(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
        operation: &crate::gcm::CheckedCounterfactualOperation,
    ) -> Result<StudyResult, CausalError> {
        if !matches!(&self.query, CausalQuery::Counterfactual(current)
            if self.graph.as_dag().is_some_and(|graph| operation.matches(graph, current)))
        {
            return Err(CausalError::Compile {
                message:
                    "retained checked counterfactual operation no longer matches its query or graph"
                        .into(),
            });
        }
        let graph = operation.graph();
        let query = CausalQuery::Counterfactual(operation.query().clone());
        let reused = self.reuse_static_identification(physical);
        let (cache, identify_cached) = shared_identification_from_cache_or(ctx, reused, || {
            let identification = identify_static_query(IdentifierId::GcmParametric, graph, &query)?;
            let estimand =
                identification.estimands.first().cloned().ok_or_else(|| CausalError::Compile {
                    message: "counterfactual identification produced no estimand".into(),
                })?;
            Ok((identification, estimand))
        })?;
        let inference = self.settings.inference.clone();
        let procedure = CounterfactualProcedure::for_inference(&inference)?
            .with_bootstrap(self.settings.bootstrap_replicates, self.declared.sampling_design)?;
        let plan = CheckedCounterfactualPlan::new(
            operation.clone(),
            cache,
            identify_cached,
            inference,
            procedure,
            physical.clone(),
            IdentifiedResultContext::from_study(self),
        )?;
        plan.execute(data, ctx)
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
        query.validate()?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        let (_, _, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                Ok(parametric_scm_identification(
                    CausalQuery::AnomalyAttribution(query.clone()),
                    outcome,
                    outcome,
                ))
            })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        // Score all rows (preserving the prior facade behavior of ignoring `unit_rows`
        // for the published mean) while honoring the query's reference. Fixing the
        // reference is what gives the mean anomaly score a sample-independent truth.
        let scorer_query =
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference);
        let scores = anomaly_attribution_query_with(&fitted.model, data, &scorer_query, ctx)?;
        let mut result = self.finish_gcm(
            physical,
            started,
            GcmFinish {
                query: CausalQuery::AnomalyAttribution(query.clone()),
                treatment: outcome,
                outcome,
                estimate: nan_effect(),
                slot: GcmSlot::Anomaly(scores.clone()),
                diagnostics: Vec::new(),
                identify_cached,
            },
        );
        if self.settings.inference.is_bayesian() {
            let n_draws = bayesian_draw_count(&self.settings.inference)?;
            let mut columns: Vec<(String, Vec<f64>)> = scores
                .iter()
                .map(|score| {
                    (
                        format!("mean_anomaly_score[{}]", score.target.raw()),
                        Vec::with_capacity(n_draws),
                    )
                })
                .collect();
            let registry = crate::gcm::MechanismRegistry::standard();
            let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0xA110_7A7E);
            for _ in 0..n_draws {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_ESTIMATE_POINT,
                    });
                }
                let weights = dirichlet_row_weights(data.row_count(), &mut rng);
                let store = registry
                    .refit_weighted(&fitted.model, data, &fitted.assignments, &weights)
                    .map_err(CausalError::rejected)?;
                let draw_scores = anomaly_attribution_query_with(
                    &fitted.model.clone().with_mechanisms(store),
                    data,
                    &scorer_query,
                    ctx,
                )?;
                if draw_scores.len() != columns.len() {
                    return Err(CausalError::Compile {
                        message: "Bayesian anomaly draw changed target set".into(),
                    });
                }
                for (column, score) in columns.iter_mut().zip(draw_scores) {
                    // The per-draw mean anomaly score is the Bayesian bootstrap of the
                    // population mean: the shared Dirichlet(1,…,1) row weights reweight the
                    // per-row scores. This holds for both references — the factual scores
                    // depend only on the data and the reference, not the mechanism refit, so
                    // an *unweighted* mean would be identical every draw (a degenerate,
                    // zero-width posterior). With a fixed reference the target of that
                    // posterior is the fixed functional μ_A; with the empirical reference it
                    // is the observed-sample-reference mean.
                    column.1.push(dirichlet_weighted_mean(&score.scores, &score.rows, &weights));
                }
            }
            let (identification, _) = parametric_scm_identification(
                CausalQuery::AnomalyAttribution(query.clone()),
                outcome,
                outcome,
            );
            result.posterior = Some(attribution_posterior(
                columns,
                identification.required_assumptions,
                identification.status,
                super::ANOMALY_DIRICHLET_BACKEND,
                false,
            )?);
            let disclosure = match scorer_query.reference {
                antecedent_core::AnomalyReference::Empirical => {
                    "Posterior is the Bayesian bootstrap of the mean anomaly score: each draw reweights the per-row scores by one shared Dirichlet(1,…,1) row-weight vector (mechanisms are also refit under those weights for the accompanying attributions), conditional on mechanism-family selection from the original unweighted data. The anomaly marginal reference (median / 1.4826·MAD) is held at the observed data values, so the interval is a credible interval for the mean anomaly score relative to that realized sample's reference, not for a fixed population functional; inject a fixed reference for the latter."
                }
                antecedent_core::AnomalyReference::Fixed { .. } => {
                    "Posterior is the Bayesian bootstrap of the population mean anomaly score against the injected fixed reference (center, scale): each draw reweights the per-row scores by one shared Dirichlet(1,…,1) row-weight vector (mechanisms are also refit under those weights for the accompanying attributions). Because the reference is held at the injected values rather than estimated from the sample, the mean anomaly score is a fixed functional of the target's law, μ_A = E_Y[−log 2Φ(−|Y − center|/scale)], and the interval is a credible interval for that functional over the observed units."
                }
            };
            result.diagnostics.push(Diagnostic::new(
                "gcm.attribution.bayesian",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                disclosure,
            ));
        }
        Ok(result)
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
        query.validate()?;
        let (_, _, identify_cached) =
            identification_from_cache_or(ctx, self.reuse_static_identification(physical), || {
                Ok(parametric_scm_identification(
                    CausalQuery::ChangeAttribution(query.clone()),
                    query.outcome,
                    query.outcome,
                ))
            })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let point_result = attribute_distribution_change(
            &fitted.model,
            data,
            query,
            &antecedent_attribution::DistributionChangeOptions::default(),
            ctx,
        )?;
        let estimate = EffectEstimate::new(
            point_result.total_change,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut result = self.finish_gcm(
            physical,
            started,
            GcmFinish {
                query: CausalQuery::ChangeAttribution(query.clone()),
                treatment: query.outcome,
                outcome: query.outcome,
                estimate,
                slot: GcmSlot::Change(point_result),
                diagnostics: Vec::new(),
                identify_cached,
            },
        );
        if self.settings.inference.is_bayesian() {
            let posterior = self.change_attribution_posterior(data, query, &fitted, ctx)?;
            if let Some(index) = posterior.effect_column() {
                if let Some(estimate) = result.estimate.as_effect_mut() {
                    estimate.ate = posterior.summaries.mean[index];
                    estimate.uncertainty.analytic.se = posterior.summaries.sd[index];
                }
            }
            result.posterior = Some(posterior);
            result.diagnostics.push(Diagnostic::new(
                "gcm.attribution.bayesian", DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                "Posterior is conditional on mechanism-family selection from the original unweighted populations. Each population receives a Dirichlet(1,…,1) row-weight vector per draw, shared across all mechanisms in that population; intervals are modular bootstrap pushforwards of the fitted-model change attribution.",
            ));
        }
        Ok(result)
    }

    /// Dirichlet row-weight bootstrap of the change attribution: one weight vector per population per draw.
    fn change_attribution_posterior(
        &self,
        data: &TabularData,
        query: &antecedent_core::ChangeAttributionQuery,
        fitted: &crate::gcm::FittedGcm,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, CausalError> {
        let n_draws = bayesian_draw_count(&self.settings.inference)?;
        let baseline_n = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(CausalError::rejected)?
            .len();
        let comparison_n = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(CausalError::rejected)?
            .len();
        let options = antecedent_attribution::DistributionChangeOptions::default();
        let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0xC4A6_7A7E);
        let mut total = Vec::with_capacity(n_draws);
        let mut components: Option<Vec<(String, Vec<f64>)>> = None;
        for _ in 0..n_draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let baseline_weights = dirichlet_row_weights(baseline_n, &mut rng);
            let comparison_weights = dirichlet_row_weights(comparison_n, &mut rng);
            let draw = antecedent_attribution::distribution_change_with_row_weights(
                &fitted.model,
                data,
                query,
                &options,
                &baseline_weights,
                &comparison_weights,
                ctx,
            )
            .map_err(CausalError::rejected)?;
            total.push(draw.total_change);
            let columns = components.get_or_insert_with(|| {
                draw.contributions
                    .iter()
                    .map(|c| {
                        (
                            format!("component[{}]", c.component.variable().raw()),
                            Vec::with_capacity(n_draws),
                        )
                    })
                    .collect()
            });
            if draw.contributions.len() != columns.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian change attribution draw changed component set".into(),
                });
            }
            for (column, component) in columns.iter_mut().zip(draw.contributions.iter()) {
                column.1.push(component.contribution);
            }
        }
        let mut posterior_columns = vec![("total_change".to_owned(), total)];
        posterior_columns.extend(components.unwrap_or_default());
        let (identification, _) = parametric_scm_identification(
            CausalQuery::ChangeAttribution(query.clone()),
            query.outcome,
            query.outcome,
        );
        attribution_posterior(
            posterior_columns,
            identification.required_assumptions,
            identification.status,
            super::CHANGE_DIRICHLET_BACKEND,
            true,
        )
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
        query.validate()?;
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
            started,
            GcmFinish {
                query: CausalQuery::MechanismChange(query.clone()),
                treatment: outcome,
                outcome,
                estimate: nan_effect(),
                slot: GcmSlot::Mechanism(detections),
                diagnostics: Vec::new(),
                identify_cached: false,
            },
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
        query.validate()?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_unit_change(&fitted.model, data, query, ctx)?;
        Ok(self.finish_gcm(
            physical,
            started,
            GcmFinish {
                query: CausalQuery::UnitChange(query.clone()),
                treatment: query.outcome,
                outcome: query.outcome,
                estimate: nan_effect(),
                slot: GcmSlot::Unit(result),
                diagnostics: Vec::new(),
                identify_cached: false,
            },
        ))
    }

    fn finish_gcm(
        &self,
        physical: &PhysicalExecutionPlan,
        started: Instant,
        finish: GcmFinish,
    ) -> StudyResult {
        let GcmFinish { query, treatment, outcome, estimate, slot, diagnostics, identify_cached } =
            finish;
        let estimator_id = match (&query, &self.settings.inference) {
            (CausalQuery::AnomalyAttribution(_), InferenceMode::Bayesian(_)) => {
                EstimatorId::GcmFitBayesian
            }
            (CausalQuery::ChangeAttribution(_), InferenceMode::Bayesian(_)) => {
                EstimatorId::GcmAttributionBayesian
            }
            _ => EstimatorId::GcmFit,
        };
        let (identification, estimand) = parametric_scm_identification(query, treatment, outcome);
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GcmParametric, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                gcm: Some(slot),
                empty_provenance: true,
                ..Default::default()
            }),
        )
    }
}

pub(super) fn dirichlet_row_weights(n: usize, rng: &mut antecedent_core::CausalRng) -> Vec<f64> {
    antecedent_kernels::bayesian_bootstrap_weights(rng, n, false)
}

/// Bayesian-bootstrap mean of per-row `scores` under the draw's Dirichlet `weights`:
/// `Σ_k weights[rows[k]]·scores[k] / Σ_k weights[rows[k]]`. Indexing by `rows`
/// keeps the weights aligned even when only a subset of rows was scored. `NaN`
/// when no positive weight lands on a scored row.
pub(super) fn dirichlet_weighted_mean(scores: &[f64], rows: &[usize], weights: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (score, &row) in scores.iter().zip(rows.iter()) {
        let w = weights.get(row).copied().unwrap_or(0.0);
        num += w * score;
        den += w;
    }
    if den > 0.0 { num / den } else { f64::NAN }
}

pub(super) fn attribution_posterior(
    columns: Vec<(String, Vec<f64>)>,
    mut assumptions: antecedent_core::AssumptionSet,
    identification: antecedent_identify::IdentificationStatus,
    backend: &str,
    has_primary_effect: bool,
) -> Result<CausalPosterior, CausalError> {
    let n_draws = columns.first().map_or(0, |(_, draws)| draws.len());
    if n_draws < 2 || columns.iter().any(|(_, draws)| draws.len() != n_draws) {
        return Err(CausalError::Compile {
            message: "Bayesian attribution draws have inconsistent shape".into(),
        });
    }
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("gcm.attribution.shared_row_weight_posterior"),
                description: Arc::from("Dirichlet(1,…,1) Bayesian bootstrap over the empirical row law, conditional on mechanism-family selection from the original unweighted data. This posterior quantifies weighted-refit uncertainty for the fitted model and does not address model misspecification or causal identification."),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from(backend) },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let schema = antecedent_prob::PosteriorSchema {
        quantities: Arc::from(
            columns
                .iter()
                .enumerate()
                .map(|(index, (name, _))| {
                    if has_primary_effect && index == 0 {
                        antecedent_prob::PosteriorQuantityKind::Effect {
                            name: Arc::from(name.as_str()),
                        }
                    } else {
                        antecedent_prob::PosteriorQuantityKind::Scalar {
                            name: Arc::from(name.as_str()),
                        }
                    }
                })
                .collect::<Vec<_>>(),
        ),
    };
    let values: Vec<f64> = columns.into_iter().flat_map(|(_, values)| values).collect();
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n_draws,
        Arc::<[f64]>::from(values),
    )?;
    let summaries = draws.summarize();
    Ok(CausalPosterior {
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: antecedent_prob::InferenceDiagnostics::analytic(backend),
        assumptions,
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
        preparation: None,
    })
}

/// Why the published per-unit effects do, or do not, vary.
#[derive(Debug)]
enum HomogeneityVerdict {
    /// Some mechanism on a `treatment → outcome` path can modify the effect.
    Heterogeneous,
    /// Every path mechanism is additively separable, and no family that could
    /// have modified the effect was fit anywhere on those paths: the per-unit
    /// effects are equal by construction of the candidate set. Carries the
    /// outcome's selected family.
    Structural(crate::gcm::MechanismFamily),
    /// Every path mechanism is additively separable, but a heterogeneity-capable
    /// family *was* fit on some path node and lost on validation score: equal
    /// per-unit effects are a finding about the data, not a property fixed in
    /// advance.
    Empirical {
        /// `(node variable, rejected family, its score, selected family, selected score)`.
        rejected: Vec<RejectedHeterogeneity>,
    },
}

/// A heterogeneity-capable family scored on a path node and not selected.
#[derive(Debug)]
struct RejectedHeterogeneity {
    variable: VariableId,
    family: crate::gcm::MechanismFamily,
    score: f64,
    selected: crate::gcm::MechanismFamily,
    selected_score: f64,
}

/// Assignments whose mechanisms are re-evaluated in the acted world: strictly
/// downstream of the treatment and upstream of (or at) the outcome.
fn path_assignments<'a>(
    graph: &Dag,
    assignments: &'a [crate::gcm::MechanismAssignment],
    treatment: VariableId,
    outcome: VariableId,
) -> Option<(Vec<&'a crate::gcm::MechanismAssignment>, DenseNodeId, DenseNodeId)> {
    let (on_path, treatment_nodes, outcome_node) =
        path_assignments_over(graph, assignments, &[treatment], outcome)?;
    Some((on_path, treatment_nodes[0], outcome_node))
}

/// [`path_assignments`] for several simultaneously intervened treatments: the assignments
/// strictly downstream of at least one treatment and upstream of (or at) the outcome. Also
/// returns each treatment's node, in the order given.
fn path_assignments_over<'a>(
    graph: &Dag,
    assignments: &'a [crate::gcm::MechanismAssignment],
    treatments: &[VariableId],
    outcome: VariableId,
) -> Option<(Vec<&'a crate::gcm::MechanismAssignment>, Vec<DenseNodeId>, DenseNodeId)> {
    let node_of =
        |variable: VariableId| assignments.iter().find(|a| a.variable == variable).map(|a| a.node);
    let treatment_nodes: Vec<DenseNodeId> =
        treatments.iter().map(|&treatment| node_of(treatment)).collect::<Option<_>>()?;
    let outcome_node = node_of(outcome)?;
    let on_path = assignments
        .iter()
        .filter(|a| {
            !treatment_nodes.contains(&a.node)
                && treatment_nodes.iter().any(|&node| graph.reaches(node, a.node))
                && graph.reaches(a.node, outcome_node)
        })
        .collect();
    Some((on_path, treatment_nodes, outcome_node))
}

/// Classify the per-unit effects as structurally equal, empirically equal, or
/// free to vary.
///
/// A unit effect can only vary across units when some mechanism on a directed
/// `treatment → outcome` path bends with that unit's other parent values or its
/// abducted disturbance. When every such mechanism is additive and linear in its
/// parents, the two worlds differ by a fixed composition of slopes: the disturbances
/// cancel and `Y(a) − Y(a0)` is a constant.
///
/// That constant is disclosed as a property of the mechanism only when it *is*
/// one — when no family that could have represented effect modification was
/// successfully scored on any path node (none was in the registry for that node,
/// or every such family failed to fit, e.g. a single-parent node where no
/// cross-parent product exists). When such a family was scored and lost, the
/// data chose additivity, and the verdict records what was rejected instead.
fn homogeneity_verdict(
    graph: &Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    treatment: VariableId,
    outcome: VariableId,
) -> HomogeneityVerdict {
    homogeneity_verdict_over(graph, assignments, &[treatment], outcome)
}

/// [`homogeneity_verdict`] for one outcome under several simultaneous treatments: the
/// mechanisms on a directed path from *any* treatment to the outcome are the ones that can
/// modify the effect, so the verdict is structural only when all of them are additively
/// separable.
fn homogeneity_verdict_over(
    graph: &Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    treatments: &[VariableId],
    outcome: VariableId,
) -> HomogeneityVerdict {
    let Some((on_path, _, outcome_node)) =
        path_assignments_over(graph, assignments, treatments, outcome)
    else {
        return HomogeneityVerdict::Heterogeneous;
    };
    let mut outcome_family = None;
    let mut rejected = Vec::new();
    for assignment in &on_path {
        if !assignment.fitted.admits_no_effect_modification() {
            return HomogeneityVerdict::Heterogeneous;
        }
        if assignment.node == outcome_node {
            outcome_family = Some(assignment.selected);
        }
        let selected_score = assignment
            .candidates
            .iter()
            .find(|c| c.family == assignment.selected)
            .map_or(f64::NAN, |c| c.score);
        for candidate in assignment.candidates.iter() {
            // `candidates` holds only families that scored; failures live in
            // `failed_families` and never gave the data a say.
            if candidate.family != assignment.selected
                && candidate.family.can_modify_effects()
                && candidate.score.is_finite()
            {
                rejected.push(RejectedHeterogeneity {
                    variable: assignment.variable,
                    family: candidate.family,
                    score: candidate.score,
                    selected: assignment.selected,
                    selected_score,
                });
            }
        }
    }
    match outcome_family {
        None => HomogeneityVerdict::Heterogeneous,
        Some(_) if !rejected.is_empty() => HomogeneityVerdict::Empirical { rejected },
        Some(family) => HomogeneityVerdict::Structural(family),
    }
}

/// Record that equal per-unit effects were chosen by validation score over a
/// family that could have made them differ.
///
/// Info, not Warning: nothing about how the number reads is hidden — the
/// selection is auditable in `gcm.counterfactual.mechanisms`, and this names the
/// rejected alternatives and the score margin so a reader does not have to parse
/// that dump to learn that heterogeneity was tested for.
fn heterogeneity_rejected_diagnostic(
    data: &TabularData,
    rejected: &[RejectedHeterogeneity],
) -> Diagnostic {
    let name = |variable: VariableId| {
        data.schema()
            .get(variable)
            .map_or_else(|_| format!("v{}", variable.raw()), |v| v.name.to_string())
    };
    let parts: Vec<String> = rejected
        .iter()
        .map(|r| {
            format!(
                "{}: {} (score {:.6}) lost to {} (score {:.6})",
                name(r.variable),
                r.family.id(),
                r.score,
                r.selected.id(),
                r.selected_score
            )
        })
        .collect();
    let families: Vec<&str> = rejected.iter().map(|r| r.family.id()).collect();
    Diagnostic::new(
        "gcm.counterfactual.heterogeneity_rejected",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "heterogeneity-capable mechanism families were fit and lost on validation score, so \
             equal unit_effects are an empirical finding rather than a property of the mechanism: \
             {}",
            parts.join("; ")
        ),
    )
    .with_fields([("families", families.join(","))])
}

/// Per-unit support of the counterfactual prediction.
struct UnitSupport {
    /// Unit's covariate cell leaves the opposite arm's observed range.
    parent_cell: Vec<bool>,
    /// Unit's abducted disturbance leaves the opposite arm's residual range.
    disturbance: Vec<bool>,
    /// Covariates checked (names).
    covariates: Vec<String>,
    /// Additive-disturbance path nodes checked (names).
    disturbance_nodes: Vec<String>,
}

impl UnitSupport {
    fn extrapolation_flags(&self) -> Vec<bool> {
        self.parent_cell.iter().zip(&self.disturbance).map(|(a, b)| *a || *b).collect()
    }

    fn support_diagnostic(&self, min: f64, max: f64, pooled: bool) -> Diagnostic {
        let n = self.parent_cell.len();
        let cell = self.parent_cell.iter().filter(|f| **f).count();
        let disturbance = self.disturbance.iter().filter(|f| **f).count();
        let any = self.extrapolation_flags().iter().filter(|f| **f).count();
        let severity =
            if pooled || any > 0 { DiagnosticSeverity::Warning } else { DiagnosticSeverity::Info };
        Diagnostic::new(
            "gcm.counterfactual.support",
            DiagnosticKind::Support,
            severity,
            format!(
                "observed treatment range=[{min},{max}]; extrapolative={pooled}; \
                 per_unit_extrapolative={any}/{n} (covariate cell outside the opposite arm's \
                 observed range: {cell}; abducted disturbance outside the opposite arm's residual \
                 range: {disturbance}); covariates=[{}]; disturbance_nodes=[{}]",
                self.covariates.join(","),
                self.disturbance_nodes.join(",")
            ),
        )
        .with_fields([
            ("extrapolative", pooled.to_string()),
            ("per_unit_extrapolative", any.to_string()),
            ("parent_cell_extrapolative", cell.to_string()),
            ("disturbance_extrapolative", disturbance.to_string()),
            ("n_units", n.to_string()),
        ])
    }
}

/// The treatment, outcome and compared levels of a unit-level attribution contrast.
#[derive(Clone, Copy)]
struct UnitContrast {
    treatment: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
}

/// Flag, per unit, a counterfactual prediction that leaves observed support.
///
/// Every unit is predicted into both arms; the arm it was observed in is
/// factual, the other is not. A unit is assigned to the arm whose level its
/// observed treatment is nearer (exact for a binary treatment; the midpoint
/// split for a continuous one). For the *opposite* arm two things must have
/// been seen:
///
/// * **Covariate cell.** Every parent of a re-evaluated path mechanism that the
///   intervention does not move (not the treatment, not a descendant of it) is
///   held at the unit's own value in both worlds. If that value lies outside the
///   range observed among units that actually received the opposite arm, an
///   interaction or spline term is being evaluated at a `(treatment, covariate)`
///   cell the fit never saw.
/// * **Disturbance.** For each additive-disturbance path mechanism, the unit's
///   abducted residual is inside the pooled residual support by construction,
///   but not necessarily inside the residuals of the opposite arm. A unit whose
///   disturbance is unlike any seen under the arm it is predicted into is
///   carried by the additivity assumption alone.
///
/// Categorical path mechanisms have uniform abducted noise with no residual
/// scale, so only their covariate cell is checked.
fn unit_support(
    data: &TabularData,
    graph: &Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    exogenous: &crate::gcm::ExogenousPosterior,
    contrast: UnitContrast,
) -> Result<UnitSupport, CausalError> {
    let UnitContrast { treatment, outcome, active, control } = contrast;
    let n = data.row_count();
    let observed = data.float64_values(treatment)?;
    let in_active: Vec<bool> =
        observed.iter().map(|t| (t - active).abs() < (t - control).abs()).collect();
    let name = |variable: VariableId| {
        data.schema()
            .get(variable)
            .map_or_else(|_| format!("v{}", variable.raw()), |v| v.name.to_string())
    };
    let mut parent_cell = vec![false; n];
    let mut disturbance = vec![false; n];
    let mut covariates = Vec::new();
    let mut disturbance_nodes = Vec::new();
    let Some((on_path, treatment_node, _)) =
        path_assignments(graph, assignments, treatment, outcome)
    else {
        return Ok(UnitSupport { parent_cell, disturbance, covariates, disturbance_nodes });
    };
    let variable_of =
        |node: DenseNodeId| assignments.iter().find(|a| a.node == node).map(|a| a.variable);
    // Range of `values` over the rows of each arm: `[control, active]`.
    let arm_ranges = |values: &[f64]| {
        let mut ranges = [(f64::INFINITY, f64::NEG_INFINITY); 2];
        for (row, value) in values.iter().enumerate().take(n) {
            if !value.is_finite() {
                continue;
            }
            let arm = &mut ranges[usize::from(in_active[row])];
            arm.0 = arm.0.min(*value);
            arm.1 = arm.1.max(*value);
        }
        ranges
    };
    let flag = |values: &[f64], out: &mut [bool]| {
        let ranges = arm_ranges(values);
        for (row, value) in values.iter().enumerate().take(n) {
            // The opposite arm of a unit observed under `active` is `control`.
            let (lo, hi) = ranges[usize::from(!in_active[row])];
            // An arm with no observed rows has no support at all.
            if lo > hi || *value < lo || *value > hi {
                out[row] = true;
            }
        }
    };
    let mut seen = std::collections::BTreeSet::new();
    for assignment in &on_path {
        for &parent in graph.parents(assignment.node) {
            if parent == treatment_node || graph.reaches(treatment_node, parent) {
                continue;
            }
            if !seen.insert(parent.as_usize()) {
                continue;
            }
            let Some(variable) = variable_of(parent) else { continue };
            let values = data.float64_values(variable)?;
            flag(&values, &mut parent_cell);
            covariates.push(name(variable));
        }
        if assignment.selected.has_additive_disturbance()
            && exogenous.n_units == n
            && assignment.node.as_usize() < exogenous.n_nodes
        {
            let start = assignment.node.as_usize() * n;
            flag(&exogenous.noise[start..start + n], &mut disturbance);
            disturbance_nodes.push(name(assignment.variable));
        }
    }
    Ok(UnitSupport { parent_cell, disturbance, covariates, disturbance_nodes })
}

/// Posterior summary of per-unit ITE draws, through the same summary that
/// publishes the mean-ITE interval, so a unit interval and the mean interval
/// share one quantile rule.
fn unit_effect_posterior_summary(
    unit_draws: Vec<f64>,
    n_units: usize,
    n_draws: usize,
) -> Result<antecedent_prob::PosteriorSummary, CausalError> {
    let quantities: Vec<antecedent_prob::PosteriorQuantityKind> = (0..n_units)
        .map(|unit| antecedent_prob::PosteriorQuantityKind::Effect {
            name: std::sync::Arc::from(format!("ite[{unit}]")),
        })
        .collect();
    let schema = antecedent_prob::PosteriorSchema { quantities: std::sync::Arc::from(quantities) };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n_draws,
        std::sync::Arc::<[f64]>::from(unit_draws),
    )?;
    Ok(draws.summarize())
}

fn counterfactual_posterior(
    values: Vec<f64>,
    assumptions: antecedent_core::AssumptionSet,
    identification: antecedent_core::IdentificationStatus,
) -> Result<CausalPosterior, CausalError> {
    counterfactual_posterior_columns(vec![("ite".into(), values)], assumptions, identification)
}

/// [`counterfactual_posterior`] with one `Effect` column per contrast, each a named series of the
/// same number of draws. The first column is the result's primary effect.
fn counterfactual_posterior_columns(
    columns: Vec<(String, Vec<f64>)>,
    mut assumptions: antecedent_core::AssumptionSet,
    identification: antecedent_core::IdentificationStatus,
) -> Result<CausalPosterior, CausalError> {
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("counterfactual.weighted_mechanisms"),
            description: Arc::from("Dirichlet row-weight posterior of standard mechanism fits, conditional on selected mechanism families and empirical support; abduction is repeated on the original units for every draw, the factual and counterfactual worlds sharing that draw's disturbances (a categorical node's disturbance is drawn once per posterior draw). This is not a parametric coefficient-prior posterior. The interval is for mean_ite over the observed units and reflects mechanism-refit uncertainty, plus the rank-preserving noise posterior of categorical nodes where present; it is not a unit-level predictive interval."),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("gcm.fit.bayesian") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let quantities: Vec<antecedent_prob::PosteriorQuantityKind> = columns
        .iter()
        .map(|(name, _)| antecedent_prob::PosteriorQuantityKind::Effect {
            name: std::sync::Arc::from(name.as_str()),
        })
        .collect();
    let schema = antecedent_prob::PosteriorSchema { quantities: std::sync::Arc::from(quantities) };
    let n = columns.first().map_or(0, |(_, values)| values.len());
    let values: Vec<f64> = columns.into_iter().flat_map(|(_, values)| values).collect();
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n,
        std::sync::Arc::<[f64]>::from(values),
    )?;
    let summaries = draws.summarize();
    Ok(CausalPosterior {
        subsampled_out_mass: 0.0,
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: antecedent_prob::InferenceDiagnostics::analytic(
            super::COUNTERFACTUAL_DIRICHLET_BACKEND,
        ),
        assumptions,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
        preparation: None,
    })
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)] // SCM columns are named as in the graph (z, a, y).
mod tests {
    use super::*;
    use crate::gcm::{CompiledCausalModel, MechanismRegistry, SelectionPolicy};
    use antecedent_core::CausalRng;

    fn var_id(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    /// `y = a·exp(z) + z + ε` with `a ~ Bern(σ(0.5 z))`: the unit effect is
    /// `exp(z)`, monotone in `z`. Columns `z, a, y`; graph `z → a`, `{z, a} → y`.
    fn exp_modifier_table(n: usize, seed: u64) -> (TabularData, Dag) {
        let mut rng = CausalRng::from_seed(seed);
        let (mut z, mut a, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            z[i] = antecedent_kernels::standard_normal(&mut rng);
            a[i] = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-0.5 * z[i]).exp())));
            y[i] = a[i] * z[i].exp() + z[i] + antecedent_kernels::standard_normal(&mut rng);
        }
        let data = TabularData::from_f64_columns([
            ("z", z.as_slice()),
            ("a", a.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut g = Dag::with_variables(3);
        for (from, to) in [(0, 1), (0, 2), (1, 2)] {
            g.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        (data, g)
    }

    fn assignments(
        registry: &MechanismRegistry,
        data: &TabularData,
        graph: &Dag,
    ) -> Vec<crate::gcm::MechanismAssignment> {
        let compiled = CompiledCausalModel::compile(graph.clone()).unwrap();
        registry.assign_and_fit(&compiled, data, SelectionPolicy::BestScore).unwrap().1
    }

    /// With only additive-linear families available the constant contrast is a
    /// property of the candidate set, and the verdict says so.
    #[test]
    fn linear_only_registry_makes_homogeneity_structural() {
        let (data, graph) = exp_modifier_table(1500, 5);
        let fitted = assignments(&MechanismRegistry::standard(), &data, &graph);
        match homogeneity_verdict(&graph, &fitted, var_id(1), var_id(2)) {
            HomogeneityVerdict::Structural(family) => {
                assert_eq!(family, crate::gcm::MechanismFamily::LinearGaussian);
            }
            other => panic!("linear-only registry must be structural, got {other:?}"),
        }
    }

    /// The same data under the counterfactual registry selects a family that
    /// modifies the effect, so there is nothing to disclose.
    #[test]
    fn heterogeneity_registry_lets_the_exp_modifier_through() {
        let (data, graph) = exp_modifier_table(1500, 5);
        let fitted = assignments(&MechanismRegistry::with_heterogeneity_families(), &data, &graph);
        assert!(
            matches!(
                homogeneity_verdict(&graph, &fitted, var_id(1), var_id(2)),
                HomogeneityVerdict::Heterogeneous
            ),
            "{:?}",
            fitted.iter().map(|a| (a.variable, a.selected)).collect::<Vec<_>>()
        );
    }

    /// A single-parent outcome cannot fit any cross-parent product: the richer
    /// families fail, so the equal contrast is structural even under the
    /// counterfactual registry.
    #[test]
    fn failed_heterogeneity_families_leave_homogeneity_structural() {
        let n = 800;
        let mut rng = CausalRng::from_seed(9);
        let a: Vec<f64> = (0..n).map(|_| f64::from(u8::from(rng.next_f64() < 0.5))).collect();
        let y: Vec<f64> =
            a.iter().map(|ai| 0.8 * ai + antecedent_kernels::standard_normal(&mut rng)).collect();
        let data =
            TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice())]).unwrap();
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let fitted = assignments(&MechanismRegistry::with_heterogeneity_families(), &data, &graph);
        assert!(matches!(
            homogeneity_verdict(&graph, &fitted, var_id(0), var_id(1)),
            HomogeneityVerdict::Structural(_)
        ));
    }
}
