// SPDX-License-Identifier: MIT OR Apache-2.0

mod dbn_effect;
mod dbn_mediation;
mod graph_posterior;
mod tiered;

use super::class_envelope_se::SharedCircularBlockSe;
use super::sequential_validation::{
    SequentialValidation, SequentialValidationAtom, validate_sequential,
};
use super::temporal_effect_core::shared_gap_report;
use super::{
    Arc, AverageEffectQuery, BayesianGCompWorkspace, BayesianGComputationAte,
    BayesianGraphPosteriorPriorAnchor, BayesianSuiteContext, BayesianTemporalGcomp,
    BayesianValidation, BootstrapAccounting, CausalError, CausalPosterior, CausalQuery,
    ClassAtomOutcome, ClassEnvelopeRun, ConditionalLinearAdjustment, DBN_LINEAR_SEM, Dag,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EnvelopeAtomFit,
    EnvelopeOptions, EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId,
    ExecuteRun, ExecutionContext, GraphEffectDraws, GraphIdentFlag, GraphPosterior,
    GraphPosteriorAtomValue, IdentificationEnvelope, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights,
    IdentifierId, InferenceDiagnostics, InferenceMode, Instant, InteractiveSubsampleDrop,
    OverlapPolicy, PhysicalExecutionPlan, PosteriorMassSummary, PosteriorPredictiveCheck,
    PredictiveCheckReport, PreparedBayesianProblem, PriorPredictiveCheck, PriorSet, RefutationRun,
    RefuteSuite, ResponseValue, SUSTAINED_LINEAR_SEM, StaticEstimateInput,
    StaticEstimateWorkspaces, Strategy, StructuralAggregationPolicy, StudyResult, TableView,
    TabularData, TemporalAtomDesign, TemporalEffectQuery, TemporalIndexer,
    TemporalLinearAdjustment, TemporalMediationEstimate, TemporalRefutation, TimeSeriesData,
    ValidationSink, ValidationSuite, ValidatorId, VariableId, WeightedGraphSamples,
    aggregate_effect_envelope, bayesian_gcomp, bayesian_temporal_gcomp, effect_from_posterior,
    elapsed_ns, envelope_mass_diagnostic, envelope_refute_atoms, envelope_se_omission_diagnostic,
    estimate_static_effect, evaluate_bayesian_prior_sensitivity, first_key_positions,
    identification_from_cache_or, identification_status_ok_for_case, identified_envelope_keys,
    interactive_subsample_graphs_accounted, mass_fields, maybe_interactive_envelope_subsample,
    mediation_mechanism_seed, merge_posterior_notes, mix_graph_posterior_identified_atoms,
    mix_prior_sensitivity_summaries, mix_static_envelope_se, mix_weighted_analytic_se,
    mixed_static_influence, most_conservative_identification_status, nan_effect,
    overlap_diagnostic, overlap_limit_diagnostic, project_for_ate_estimate, projection_diagnostic,
    provenance_ids, push_conflict_diagnostics,
    push_graph_posterior_structural_aggregation_diagnostic, report_subsampled_out_mass,
    require_identified, resolve_bayesian_prior_with_conflict, resolve_envelope_prior_anchor,
    resolve_posterior_status, resolve_structural_aggregation, retain_envelope_assumptions,
    run_envelope_bayesian_full_validation, select_claim, select_estimand,
    sequential_linear_sem_assumption, shared_block_mixture_message,
    shared_circular_block_mixture_se, short_series_warning, static_aligned_influence,
    temporal_refuter_average, validator_not_applicable_diagnostics, with_conflict_summary,
    with_prior_sensitivity,
};
use crate::analysis::identification_cache::DbnIdentifyDemotion;
use crate::planner::RouteDefaults;

/// The Bayesian g-computation estimator id for an average or conditional effect.
fn bayesian_gcomp_estimator(conditional: bool) -> EstimatorId {
    if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp }
}

/// Borrowed inputs of the Bayesian g-computation point stage.
#[derive(Clone, Copy)]
struct BayesianGcompPointInput<'a> {
    data: &'a TabularData,
    estimand: &'a IdentifiedEstimand,
    query: &'a AverageEffectQuery,
    cfg: &'a crate::inference::BayesianConfig,
    status: IdentificationStatus,
    conditional: bool,
}

/// Fitted Bayesian g-computation point stage with the state validation reuses.
struct BayesianGcompPoint {
    est: BayesianGComputationAte,
    prep: PreparedBayesianProblem,
    ws: BayesianGCompWorkspace,
    posterior: CausalPosterior,
    estimate: EffectEstimate,
}

/// Borrowed inputs of the Bayesian g-computation validate stage.
struct BayesianGcompValidationInput<'a> {
    data: &'a TabularData,
    estimand: &'a IdentifiedEstimand,
    query: &'a AverageEffectQuery,
    estimate: &'a EffectEstimate,
    cfg: &'a crate::inference::BayesianConfig,
    est: &'a BayesianGComputationAte,
    prep: &'a PreparedBayesianProblem,
    status: IdentificationStatus,
    estimator_id: EstimatorId,
}

/// Fitted completions of a Bayesian class envelope, with the subsample accounting.
struct PagEnvelopeFits {
    full_graphs: WeightedGraphSamples,
    graphs: WeightedGraphSamples,
    subsample_notes: Vec<Diagnostic>,
    subsample_drop: InteractiveSubsampleDrop,
    outcomes: Vec<ClassAtomOutcome>,
    atoms: Vec<EnvelopeAtomFit>,
    per_graph: Vec<GraphEffectDraws>,
    primary_estimand: Option<IdentifiedEstimand>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
}

/// Borrowed inputs of the Bayesian class-envelope validate stage.
struct PagEnvelopeValidationInput<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    estimate: &'a EffectEstimate,
    cfg: &'a crate::inference::BayesianConfig,
    est: &'a BayesianGComputationAte,
    estimator_id: EstimatorId,
}

/// Identification cases of a class envelope as weighted-graph inputs, with the identified
/// atoms to fit and the first one's estimand.
fn pag_envelope_cases<G>(
    envelope: &IdentificationEnvelope<G>,
    estimator_id: EstimatorId,
) -> Result<PagEnvelopeCases, CausalError> {
    let mut weights = Vec::new();
    let mut flags = Vec::new();
    let mut keys = Vec::new();
    let mut fit_atoms = Vec::new();
    let mut primary_estimand: Option<IdentifiedEstimand> = None;
    for (i, case) in envelope.cases.iter().enumerate() {
        let key = i as u64 + 1;
        keys.push(key);
        weights.push(case.weight.0);
        if identification_status_ok_for_case(case.result.status)
            && !case.result.estimands.is_empty()
        {
            flags.push(GraphIdentFlag::Identified);
            let estimand = select_estimand(&case.result, estimator_id)?;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
            }
            fit_atoms.push((key, estimand, case.result.status));
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    Ok(PagEnvelopeCases { weights, flags, keys, fit_atoms, primary_estimand })
}

/// Scan of a class envelope's identification cases.
struct PagEnvelopeCases {
    weights: Vec<f64>,
    flags: Vec<GraphIdentFlag>,
    keys: Vec<u64>,
    fit_atoms: Vec<(u64, IdentifiedEstimand, IdentificationStatus)>,
    primary_estimand: Option<IdentifiedEstimand>,
}

/// Prepared designs, resolved priors, and the first prior conflict of a class envelope.
struct PagEnvelopePrepared {
    prepared: std::collections::HashMap<u64, PreparedBayesianProblem>,
    atom_priors: std::collections::HashMap<u64, Option<PriorSet>>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
}

/// Borrowed settings of the class-envelope posterior aggregation.
struct PagEnvelopeAggregation<'a> {
    conditional: bool,
    full_query: &'a CausalQuery,
    class_tag: &'a str,
}

/// Frozen-weight mixture posterior of the fitted class-envelope completions, coupled by
/// their sampling correlation when the influence functions align; the flag reports coupling.
fn aggregate_pag_envelope_posterior(
    data: &TabularData,
    query: &AverageEffectQuery,
    settings: &PagEnvelopeAggregation<'_>,
    fits: &mut PagEnvelopeFits,
    ctx: &ExecutionContext,
) -> Result<(CausalPosterior, bool), CausalError> {
    let PagEnvelopeAggregation { conditional, full_query, class_tag } = *settings;
    let PagEnvelopeFits { full_graphs, graphs, subsample_drop, atoms, per_graph, .. } = &*fits;
    // Completion weights here are a frozen enumeration of an equivalence
    // class, not data-updated graph probabilities, and the reported number
    // is the frozen-weight mixture functional. Publish its posterior (atoms
    // coupled by their sampling correlation on the shared rows) rather than
    // the BMA spread over completion-specific effects, which over-covers
    // the mixture functional whenever completions disagree.
    let coupling =
        static_envelope_atom_correlation(data, query, conditional, atoms, full_query, ctx);
    let mut posterior = match &coupling {
        Some(correlation) => antecedent_estimate::aggregate_mixture_functional_envelope(
            graphs,
            per_graph,
            correlation,
            InferenceDiagnostics::analytic(format!("{class_tag}_envelope")),
        ),
        None => aggregate_effect_envelope(
            graphs,
            per_graph,
            InferenceDiagnostics::analytic(format!("{class_tag}_envelope")),
            EnvelopeOptions::default(),
        ),
    }
    .map_err(CausalError::from)?;
    report_subsampled_out_mass(&mut posterior, full_graphs, subsample_drop);
    retain_envelope_assumptions(&mut posterior, atoms);
    if let Some(summary) = fits.envelope_conflict.take() {
        posterior = with_conflict_summary(posterior, summary);
    }
    Ok((posterior, coupling.is_some()))
}

/// Diagnostics of the Bayesian class envelope: identification's, the envelope's, whether
/// the completions' posteriors could be coupled, subsample notes, overlap, mass, conflict.
fn pag_envelope_diagnostics(
    identification: &IdentificationResult,
    envelope_diagnostic: Diagnostic,
    coupled: bool,
    subsample_notes: Vec<Diagnostic>,
    estimate: &EffectEstimate,
    posterior: &CausalPosterior,
    class_tag: &str,
) -> Vec<Diagnostic> {
    let mut diagnostics = identification.diagnostics.clone();
    diagnostics.push(envelope_diagnostic);
    diagnostics.push(if coupled {
        Diagnostic::new(
            "estimate.envelope.bayesian_mixture_functional",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "posterior draws describe the frozen-weight mixture functional over identified \
             completions (each completion's posterior, rank-coupled by the completions' \
             sampling correlation on shared rows); they are not a distribution over \
             completion-specific effects",
        )
    } else {
        Diagnostic::new(
            "estimate.envelope.bayesian_mixture_coupling_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "completion influence functions could not be aligned on shared rows; posterior \
             draws are the identified-completion BMA, whose spread includes between-completion \
             disagreement and is not an interval for the mixture functional",
        )
    });
    diagnostics.extend(subsample_notes);
    diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
    diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
    diagnostics.push(envelope_mass_diagnostic(format!("estimate.{class_tag}.envelope"), posterior));
    if let Some(cs) = posterior.conflict_summary.as_ref() {
        push_conflict_diagnostics(&mut diagnostics, cs);
    }
    diagnostics
}

/// Why a DBN atom was demoted before contributing draws.
#[derive(Clone, Copy)]
enum DbnAtomDemotion {
    Prepare,
    Fit,
    Draws,
}

/// A fit refused under draw controls keeps the route records the sampler measured.
///
/// R-hat, posterior effective sample size, convergence, and a computational
/// failure stay the records they are. This does not rewrite them as one
/// convergence reason.
pub(super) fn chain_refusal_under_draw_controls(
    controlled: bool,
    cfg: &crate::BayesianConfig,
    error: antecedent_estimate::EstimationError,
) -> CausalError {
    let _ = (controlled, cfg.backend);
    CausalError::from(error)
}

/// Refusal for an inference mode this route does not perform.
///
/// The record is unsupported inference, not a failed check or a computational
/// failure. The refusal reports no mass, so the mass is unreported.
pub(super) fn unsupported_inference_error(summary: &str) -> CausalError {
    CausalError::unsupported_inference(summary)
}

/// Refuse a posterior whose interval endpoints are not set by the posterior: the Monte Carlo
/// standard error of the larger endpoint must be within the accepted fraction of the width.
/// The endpoint errors are recorded with the posterior's inference diagnostics.
pub(super) fn enforce_draw_controls(posterior: &mut CausalPosterior) -> Result<(), CausalError> {
    let error = antecedent_estimate::endpoint_monte_carlo_error(posterior)?;
    if let Some(record) = error.route_record(posterior.unevaluable_mass) {
        return Err(CausalError::Estimate(antecedent_estimate::EstimationError::Prob(
            antecedent_prob::ProbError::Route {
                summary: format!(
                    "the Monte Carlo standard error of an interval endpoint ({:.3e} lower, \
                     {:.3e} upper) is {:.3} of the interval width from {:.0} effective draws \
                     of {}, above the {} tolerance; use more posterior draws or a sampler that \
                     mixes better",
                    error.lower_mcse,
                    error.upper_mcse,
                    error.width_fraction,
                    error.effective_draws,
                    error.draws,
                    antecedent_estimate::count_fitted_posterior::MAX_ENDPOINT_MCSE_FRACTION
                ),
                diagnostics: vec![record],
            },
        )));
    }
    posterior.diagnostics.notes.push(error.note());
    Ok(())
}

/// Route records of the checks a published posterior passed, with its measured unevaluable mass.
///
/// The Monte Carlo error check is recorded only when draw controls ran it and the endpoints were
/// within tolerance. Records follow the fixed check order.
pub(super) fn passed_route_records(
    diagnostics: &antecedent_prob::InferenceDiagnostics,
    unevaluable_mass: f64,
) -> Vec<antecedent_core::diagnostic::RouteNativeDiagnostic> {
    use antecedent_core::diagnostic::{RouteCheck, RouteKind, RouteMass, RouteNativeDiagnostic};
    let mass = RouteMass::new(unevaluable_mass).unwrap_or_else(|_| RouteMass::unreported());
    let mut records = diagnostics.passed_route_records(mass);
    if !records.is_empty()
        && antecedent_estimate::endpoint_mcse_from_notes(&diagnostics.notes)
            .is_some_and(|error| !error.exceeds_tolerance())
    {
        records.push(RouteNativeDiagnostic::check(
            RouteKind::MonteCarloError,
            RouteCheck::new(false, mass),
        ));
    }
    records
}

/// Append the passed route records of a published posterior to `diagnostics`.
pub(super) fn push_passed_route_diagnostics(
    diagnostics: &mut Vec<Diagnostic>,
    posterior: &CausalPosterior,
) {
    diagnostics.extend(
        passed_route_records(&posterior.diagnostics, posterior.unevaluable_mass)
            .into_iter()
            .map(antecedent_core::diagnostic::RouteNativeDiagnostic::to_diagnostic),
    );
}

impl super::Study {
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "the operation keeps its checked inputs and execution context together"
    )]
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        checked_operation: Option<&super::super::prepared::CheckedBayesianGcompOperation>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut clock = super::super::stage::StageClock::new();
        let identifier_id = physical.binding().identifier_or(RouteDefaults::STATIC_ATE)?;
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };
        let (identification, estimand, identify_cached) = self.identify_bayesian_gcomp(
            graph,
            query,
            checked_operation,
            physical,
            &mut clock,
            ctx,
        )?;

        let full_cols = data.schema().len();
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let projected_cols = data_est.schema().len();

        let inference =
            checked_operation.map_or(&self.settings.inference, |operation| &operation.inference);
        let cfg = match inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(unsupported_inference_error(
                    "execute_bayesian requires inference=Bayesian; \
                     estimator bayesian.gcomp cannot run under Frequentist",
                ));
            }
        };
        let BayesianGcompPoint { est, prep, mut ws, posterior, estimate } = self
            .fit_bayesian_gcomp_point(
                &BayesianGcompPointInput {
                    data: &data_est,
                    estimand: &estimand_est,
                    query: &query_est,
                    cfg: &cfg,
                    status: identification.status,
                    conditional,
                },
                &mut clock,
                ctx,
            )?;

        let mut extra_diagnostics = Vec::new();
        if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
            extra_diagnostics.push(d);
        }
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut extra_diagnostics, cs);
        }
        push_passed_route_diagnostics(&mut extra_diagnostics, &posterior);

        clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let (posterior, refutations, predictive_checks) = self.validate_bayesian_gcomp(
            &BayesianGcompValidationInput {
                data: &data_est,
                estimand: &estimand_est,
                query: &query_est,
                estimate: &estimate,
                cfg: &cfg,
                est: &est,
                prep: &prep,
                status: identification.status,
                estimator_id,
            },
            posterior,
            &mut ws,
            &mut extra_diagnostics,
            ctx,
        )?;
        clock.finish(super::super::stage::STAGE_VALIDATE);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Validate {
                refutations: refutations.clone(),
                predictive_checks: predictive_checks.clone(),
            },
        );

        let n_draws = u32::try_from(posterior.draws.n_draws).ok();
        let early_stopped = posterior.early_stopped;
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
            .wall_time_ns(clock.wall_time_ns())
            .cancelled(clock.cancelled())
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                stage_timings_ns: clock.timings(),
                posterior: Some(posterior),
                n_draws,
                predictive_checks,
                // Posterior draws carry the uncertainty; no replicate budget ran.
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Point and uncertainty stages of Bayesian g-computation: prepare the design, resolve the
    /// prior, fit, and record the outcome-model assumption, with the stage clock and events.
    fn fit_bayesian_gcomp_point(
        &self,
        input: &BayesianGcompPointInput<'_>,
        clock: &mut super::super::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<BayesianGcompPoint, CausalError> {
        let BayesianGcompPointInput {
            data: data_est,
            estimand: estimand_est,
            query: query_est,
            cfg,
            status,
            conditional,
        } = *input;
        if cfg.predictive_replicates.is_some() {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "retained predictive replicates are built by the sealed Bayesian g-computation \
                 of a mean average effect over the all-observed, treated or untreated \
                 population; this execution does not build them, so it refuses the declaration \
                 rather than return a result without them"
            ));
        }
        let mut est = bayesian_gcomp(cfg, ctx);
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let prep = if conditional {
            let q = antecedent_core::ConditionalEffectQuery::try_new(query_est.clone())?;
            est.prepare_conditional(data_est, estimand_est, &q)
        } else {
            est.prepare(data_est, estimand_est, query_est)
        }
        .map_err(CausalError::from)?;
        let (resolved_prior, conflict_summary) =
            resolve_bayesian_prior_with_conflict(cfg, &prep, Some(ctx))?;
        est.prior = resolved_prior;
        let mut ws = BayesianGCompWorkspace::default();
        let mut posterior = est.fit(&prep, status, &mut ws, ctx).map_err(|error| {
            chain_refusal_under_draw_controls(self.draw_controls.is_some(), cfg, error)
        })?;
        if self.draw_controls.is_some() {
            enforce_draw_controls(&mut posterior)?;
        }
        if let Some(summary) = conflict_summary {
            posterior = with_conflict_summary(posterior, summary);
        }
        let likelihood =
            if est.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian {
                antecedent_prob::BayesLikelihood::GaussianIdentity
            } else {
                est.likelihood
            };
        posterior.assumptions.push(gcomp_outcome_model_assumption(likelihood, conditional));
        if let Some(arm_effect) = gcomp_arm_effect_assumption(&query_est.target_population) {
            posterior.assumptions.push(arm_effect);
        }
        let estimate = effect_from_posterior(&posterior)?;
        clock.finish(super::super::stage::STAGE_ESTIMATE_POINT);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Point { estimate: estimate.clone() },
        );
        clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        clock.finish(super::super::stage::STAGE_UNCERTAINTY);
        super::super::stage::emit_uncertainty(self.stage_sink.as_ref(), &estimate);
        Ok(BayesianGcompPoint { est, prep, ws, posterior, estimate })
    }

    /// Identify stage of Bayesian g-computation: the prepared operation's frozen
    /// identification, or a (cached) fresh one, with the stage clock and event around it.
    fn identify_bayesian_gcomp(
        &self,
        graph: &Dag,
        query: &AverageEffectQuery,
        checked_operation: Option<&super::super::prepared::CheckedBayesianGcompOperation>,
        physical: &PhysicalExecutionPlan,
        clock: &mut super::super::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<(IdentificationResult, IdentifiedEstimand, bool), CausalError> {
        let identifier_id = physical.binding().identifier_or(RouteDefaults::STATIC_ATE)?;
        let estimator_id = if matches!(self.query, CausalQuery::ConditionalEffect(_)) {
            EstimatorId::BayesianConditional
        } else {
            EstimatorId::BayesianGcomp
        };
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query) — rd is never consulted on this path —
        // all frozen there, so reuse is exact and observable via the
        // `exec.identify.cached` diagnostic below.
        if let Some(operation) = checked_operation
            && (operation.query != *query || !operation.inference.is_bayesian())
        {
            return Err(CausalError::Compile {
                message: "prepared Bayesian g-computation binding changed after prepare".into(),
            });
        }
        let (identification, estimand, identify_cached) = if let Some(operation) = checked_operation
        {
            (operation.identification().clone(), operation.estimand().clone(), true)
        } else {
            let reused = self.reuse_static_identification_if(physical, |_| match self.query {
                CausalQuery::ConditionalEffect(_) => physical
                    .binding()
                    .identifier_or(RouteDefaults::CONDITIONAL)
                    .is_ok_and(|identifier| identifier == identifier_id),
                _ => physical.binding().resolves_to(
                    RouteDefaults::STATIC_ATE,
                    identifier_id,
                    estimator_id,
                ),
            });
            identification_from_cache_or(ctx, reused, || {
                let identification = match &self.query {
                    CausalQuery::ConditionalEffect(conditional) => {
                        crate::strategy_table::identify_conditional_static_in_context(
                            identifier_id,
                            graph,
                            conditional,
                            Some(ctx),
                        )?
                    }
                    _ => crate::strategy_table::identify_static_query_in_context(
                        identifier_id,
                        graph,
                        &CausalQuery::AverageEffect(query.clone()),
                        None,
                        Some(ctx),
                    )?,
                };
                select_claim(identification, estimator_id)
            })?
        };
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );
        Ok((identification, estimand, identify_cached))
    }

    /// Validate stage of Bayesian g-computation: refutation suite, prior / posterior
    /// predictive checks, and under the full suite prior sensitivity plus MCMC diagnostics.
    fn validate_bayesian_gcomp(
        &self,
        input: &BayesianGcompValidationInput<'_>,
        mut posterior: CausalPosterior,
        ws: &mut BayesianGCompWorkspace,
        extra_diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<
        (CausalPosterior, Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>),
        CausalError,
    > {
        let BayesianGcompValidationInput {
            data: data_est,
            estimand: estimand_est,
            query: query_est,
            estimate,
            cfg,
            est,
            prep,
            status,
            estimator_id,
        } = *input;
        let mut refute_ws = EstimationWorkspace::default();
        let (mut refutations, na_diagnostics) = match self.settings.refute {
            RefuteSuite::None => (Vec::new(), Vec::new()),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => RefutationRun {
                data: data_est,
                estimand: estimand_est,
                query: query_est,
                estimate,
                suite: self.settings.refute,
                estimator: estimator_id.as_str(),
                custom: &self.declared.custom_validators,
                temporal: None,
            }
            .reports(&mut refute_ws, None, ctx)?,
        };
        extra_diagnostics.extend(na_diagnostics);
        // Prior + posterior PPC whenever refute is enabled (full PredictiveCheckReport retained).
        let mut predictive_checks = Vec::new();
        if !matches!(self.settings.refute, RefuteSuite::None) {
            const PPC_ALPHA: f64 = 0.05;
            let sims = super::super::latency::predictive_check_sims(self.latency_mode);
            let ppc_prior = est.prior_in_force(prep.design.ncols);
            let prior_rep = PriorPredictiveCheck::for_estimator(est, ctx)
                .with_n_sims(sims)
                .check_with_prior(prep, &ppc_prior, ctx)
                .map_err(CausalError::from)?;
            refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(prior_rep);

            let post_rep = PosteriorPredictiveCheck::for_estimator(est, ctx)
                .with_n_sims(sims)
                .check(prep, &posterior)
                .map_err(CausalError::from)?;
            refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(post_rep);
        }
        // Prior sensitivity / MCMC stay behind the full suite (Shared Bayesian UX).
        // Mode-select: α-multiplier grid when an external composed prior is present;
        // isotropic scale grid otherwise (avoids clearing banked priors).
        if matches!(self.settings.refute, RefuteSuite::Full) {
            let (summary, sens) =
                evaluate_bayesian_prior_sensitivity(cfg, est, prep, status, &posterior, ws, ctx)?;
            refutations.push(sens.to_report(&summary, estimate.ate));
            posterior = with_prior_sensitivity(posterior, summary);

            let suite = ValidationSuite::new().with(ValidatorId::McmcDiagnostics);
            let mut bayes_ctx =
                BayesianSuiteContext::new(est, prep, &posterior, status, ws, estimate.ate);
            let outcomes = suite.run_bayesian(&mut bayes_ctx, ctx).map_err(CausalError::from)?;
            refutations.extend(ValidationSuite::reports_only(&outcomes));
            extra_diagnostics.extend(validator_not_applicable_diagnostics(&outcomes));
        }
        Ok((posterior, refutations, predictive_checks))
    }

    pub(super) fn execute_pag_bayesian<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        class_tag: super::pag_path::ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let class_tag = class_tag.as_str();
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic,
        } = run;
        let cfg = match &self.settings.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::unsupported_inference(
                    "execute_pag_bayesian requires inference=Bayesian",
                ));
            }
        };
        let est = bayesian_gcomp(&cfg, ctx);
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };

        let mut fits = self.fit_pag_envelope_atoms(data, query, envelope, &est, &cfg, ctx)?;
        let (mut posterior, coupled) = aggregate_pag_envelope_posterior(
            data,
            query,
            &PagEnvelopeAggregation { conditional, full_query: &self.query, class_tag },
            &mut fits,
            ctx,
        )?;
        let estimate = effect_from_posterior(&posterior)?;
        let estimand =
            fits.primary_estimand.take().or(envelope.invariant.clone()).ok_or_else(|| {
                CausalError::Compile {
                    message: format!("{class_tag} Bayesian envelope missing estimand"),
                }
            })?;

        let mut diagnostics = pag_envelope_diagnostics(
            &identification,
            envelope_diagnostic,
            coupled,
            std::mem::take(&mut fits.subsample_notes),
            &estimate,
            &posterior,
            class_tag,
        );

        let (refutations, predictive_checks) = self.validate_pag_envelope(
            &PagEnvelopeValidationInput {
                data,
                query,
                estimate: &estimate,
                cfg: &cfg,
                est: &est,
                estimator_id,
            },
            &fits.atoms,
            &mut posterior,
            &mut diagnostics,
            ctx,
        )?;
        // Completions that disagree publish the identified set over their
        // per-completion posterior means; a point-identified envelope stays a
        // point. The posterior remains the frozen-weight mixture functional's.
        let structural_response =
            super::pag_path::static_class_structural_mixture(envelope, &fits.outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(super::pag_path::static_class_identified_set_diagnostic(
                mixture, class_tag,
            ));
        }

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralizedAdjustment, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian_gcomp",
                    "estimate.aggregate_effect_envelope",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                predictive_checks,
                structural_response,
                ..Default::default()
            }),
        ))
    }

    /// Per-completion fits of a class envelope under Bayesian g-computation: identified
    /// atoms are prepared and prior-resolved, Interactive-subsampled, then fitted in parallel.
    fn fit_pag_envelope_atoms<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        envelope: &IdentificationEnvelope<G>,
        est: &BayesianGComputationAte,
        cfg: &crate::inference::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<PagEnvelopeFits, CausalError> {
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };

        let PagEnvelopeCases { weights, flags, keys, fit_atoms, primary_estimand } =
            pag_envelope_cases(envelope, estimator_id)?;
        let PagEnvelopePrepared { mut prepared, atom_priors, envelope_conflict } =
            self.prepare_pag_envelope_atoms(data, query, &fit_atoms, est, cfg, ctx)?;
        let full_graphs = WeightedGraphSamples::new(weights, flags, keys)?;
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            self.latency_mode,
            full_graphs.clone(),
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        let mut outcomes = vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()];
        let mut seen_keys = std::collections::HashSet::new();
        let mut skipped = Vec::new();
        let mut kept = Vec::new();
        for (key, estimand, status) in fit_atoms {
            let outcome_slot = usize::try_from(key).unwrap_or(usize::MAX).saturating_sub(1);
            if !keep.contains(&key) {
                skipped.push(outcome_slot);
                continue;
            }
            if !prepared.contains_key(&key) || !seen_keys.insert(key) {
                continue;
            }
            kept.push((key, estimand, status, outcome_slot));
        }
        for slot in skipped {
            outcomes[slot] = ClassAtomOutcome::SubsampledOut;
        }
        let fits = ctx.map_indexed(kept.len(), |i, inner| {
            let (key, estimand, status, outcome_slot) = &kept[i];
            let prep = prepared.get(key).ok_or_else(|| CausalError::Compile {
                message: "class atom was kept for fitting without a prepared problem".into(),
            })?;
            let mut est = est.clone();
            est.prior = atom_priors.get(key).cloned().flatten();
            let mut ws = BayesianGCompWorkspace::default();
            let posterior = est.fit(prep, *status, &mut ws, inner).map_err(CausalError::from)?;
            Ok::<_, CausalError>((
                *key,
                estimand.clone(),
                *status,
                *outcome_slot,
                posterior,
                est.prior.clone(),
            ))
        })?;
        let mut per_graph = Vec::new();
        let mut atoms = Vec::new();
        let key_weights = IdentifiedKeyWeights::new(&graphs);
        for (key, estimand, status, outcome_slot, posterior, prior) in fits {
            let Some(prep) = prepared.remove(&key) else {
                continue;
            };
            outcomes[outcome_slot] =
                ClassAtomOutcome::Evaluated(effect_from_posterior(&posterior)?.ate);
            per_graph.push(envelope_draws_from_posterior(key, &posterior)?);
            let weight = key_weights.get(key);
            atoms.push(EnvelopeAtomFit {
                key,
                prep,
                posterior,
                status,
                weight,
                estimand,
                indexer: None,
                prior,
            });
        }
        Ok(PagEnvelopeFits {
            full_graphs,
            graphs,
            subsample_notes,
            subsample_drop,
            outcomes,
            atoms,
            per_graph,
            primary_estimand,
            envelope_conflict,
        })
    }

    /// Prepare each identified completion once, before the Interactive subsample, and resolve
    /// its own prior; the first conflict summary stands for the envelope.
    fn prepare_pag_envelope_atoms(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        fit_atoms: &[(u64, IdentifiedEstimand, IdentificationStatus)],
        est: &BayesianGComputationAte,
        cfg: &crate::inference::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<PagEnvelopePrepared, CausalError> {
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        // Prepare once before Interactive subsample, stash
        // so kept atoms are not prepared a second time. PAG keys are unique
        // (1..n); still use entry() so the stash is key-safe.
        let mut prepared = std::collections::HashMap::with_capacity(fit_atoms.len());
        let mut atom_priors = std::collections::HashMap::with_capacity(fit_atoms.len());
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;
        let preps = ctx.map_indexed(fit_atoms.len(), |i, inner| {
            let (key, estimand, _) = &fit_atoms[i];
            let prep = if conditional {
                let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
                est.prepare_conditional(data, estimand, &q)
            } else {
                est.prepare(data, estimand, query)
            }
            .map_err(CausalError::from)?;
            // Per-completion filter: incompatible catalogs refuse rather than
            // sharing the first atom's prior or falling back to isotropic.
            let (resolved, conflict) =
                resolve_bayesian_prior_with_conflict(cfg, &prep, Some(inner))?;
            Ok::<_, CausalError>((*key, prep, resolved, conflict))
        })?;
        for (key, prep, resolved, conflict) in preps {
            if envelope_conflict.is_none() {
                envelope_conflict = conflict;
            }
            atom_priors.insert(key, resolved);
            prepared.entry(key).or_insert(prep);
        }
        Ok(PagEnvelopePrepared { prepared, atom_priors, envelope_conflict })
    }

    /// Validate stage of the Bayesian class envelope: refutation suite plus the
    /// envelope's prior / posterior predictive checks.
    fn validate_pag_envelope(
        &self,
        input: &PagEnvelopeValidationInput<'_>,
        atoms: &[EnvelopeAtomFit],
        posterior: &mut CausalPosterior,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>), CausalError>
    {
        let PagEnvelopeValidationInput { data, query, estimate, cfg, est, estimator_id } = *input;
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = match self.settings.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let (reports, mix_diagnostics) = EnvelopeRefutation {
                    data,
                    query,
                    suite: self.settings.refute,
                    estimator: estimator_id.as_str(),
                    custom: &self.declared.custom_validators,
                    temporal: None,
                }
                .run(&envelope_refute_atoms(atoms)?, &mut refute_ws, ctx)?;
                diagnostics.extend(mix_diagnostics);
                reports
            }
        };
        let predictive_checks = run_envelope_bayesian_full_validation(
            BayesianValidation { suite: self.settings.refute, cfg, estimator: est },
            atoms,
            posterior,
            estimate.ate,
            ctx,
            super::super::latency::predictive_check_sims(self.latency_mode),
            ValidationSink { refutations: &mut refutations, diagnostics },
        )?;
        Ok((refutations, predictive_checks))
    }
}

/// Declared outcome model behind a static `bayesian.gcomp` / conditional
/// posterior. The credible interval is for the treatment contrast of
/// that parametric regression; it is not a nonparametric interval for the
/// causal effect when the outcome mean is nonlinear in the adjustment set or
/// carries treatment × covariate interactions the design does not include
/// (`tests/static_calibration.rs::bayesian_gcomp_misspecification_probe`).
pub(super) fn gcomp_outcome_model_assumption(
    likelihood: antecedent_prob::BayesLikelihood,
    conditional: bool,
) -> antecedent_core::AssumptionRecord {
    let family = match likelihood {
        antecedent_prob::BayesLikelihood::GaussianIdentity => {
            "Gaussian identity-link outcome regression with homoskedastic residuals"
        }
        antecedent_prob::BayesLikelihood::BernoulliLogit => "Bernoulli logit outcome regression",
        antecedent_prob::BayesLikelihood::BernoulliProbit => "Bernoulli probit outcome regression",
        antecedent_prob::BayesLikelihood::PoissonLog => "Poisson log-link outcome regression",
    };
    let design = if conditional {
        "linear in the treatment, each declared modifier, each treatment × modifier product, \
         and each adjustment column; contrasts are evaluated on the outcome scale per draw"
    } else {
        "linear in the treatment and each adjustment column, with no treatment × covariate \
         interaction"
    };
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("bayesian.gcomp.linear_gaussian_outcome"),
                description: Arc::from(format!(
                    "{family}, {design}; the posterior and its credible interval are for the \
                     treatment contrast of this model. A nonlinear covariate term or an \
                     unmodelled treatment × covariate interaction biases that contrast and \
                     the interval does not cover the causal effect"
                )),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("bayesian.gcomp"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// The label of a Bayesian g-computation effect on the treated or untreated arm.
///
/// It is the contrast of the declared Gaussian outcome model averaged over that arm, with
/// treatment-by-covariate interactions centred at the arm's covariate means. It is not a
/// matching estimand: a matching contrast has no posterior, and this effect keeps the
/// model's identifying premises instead of the matcher's rule.
pub(super) fn gcomp_arm_effect_assumption(
    population: &antecedent_core::TargetPopulation,
) -> Option<antecedent_core::AssumptionRecord> {
    let arm = match population {
        antecedent_core::TargetPopulation::Treated => "treated",
        antecedent_core::TargetPopulation::Untreated => "untreated",
        _ => return None,
    };
    Some(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("bayesian.gcomp.model_based_arm_effect"),
                description: Arc::from(format!(
                    "the reported effect is the Gaussian outcome model's contrast averaged over \
                     the {arm} rows, with treatment x covariate interactions centred at the \
                     {arm} covariate means; it is a model-based effect on the {arm}, not a \
                     matching estimand and not the all-observed effect, and it is biased if the \
                     outcome mean is not linear in the adjustment columns and their \
                     interactions with treatment; the arm-effect interval is unsupported and \
                     its coverage is not_measured"
                )),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("bayesian.gcomp"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    })
}

pub(super) fn envelope_draws_from_posterior(
    key: u64,
    posterior: &CausalPosterior,
) -> Result<GraphEffectDraws, CausalError> {
    let col = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "Bayesian posterior missing effect column".into(),
    })?;
    let draws = posterior.draws.column(col)?;
    Ok(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws.to_vec()) })
}

/// Sampling correlation of static envelope atoms on their shared rows.
///
/// Each atom's Frequentist influence function (linear adjustment for the ATE,
/// the conditional linear functional for `ConditionalEffect`) is aligned to the
/// original rows; the correlation of those IFs couples the atoms' Bayesian
/// posteriors in [`antecedent_estimate::aggregate_mixture_functional_envelope`].
/// `None` when any atom lacks an aligned IF (the caller then keeps the BMA and
/// says so).
fn static_envelope_atom_correlation(
    data: &TabularData,
    query: &AverageEffectQuery,
    conditional: bool,
    atoms: &[EnvelopeAtomFit],
    full_query: &CausalQuery,
    ctx: &ExecutionContext,
) -> Option<Vec<f64>> {
    let mut ifs = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let estimate = if conditional {
            let CausalQuery::ConditionalEffect(cq) = full_query else {
                return None;
            };
            let mut mean_query = cq.clone();
            mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
            ConditionalLinearAdjustment::new()
                .estimate(data, &atom.estimand, &mean_query, ctx)
                .ok()?
        } else {
            let mut ws = StaticEstimateWorkspaces::default();
            estimate_static_effect(
                &crate::estimator_spec::EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
                StaticEstimateInput {
                    data,
                    estimand: &atom.estimand,
                    query,
                    bootstrap_replicates: 0,
                    overlap_policy: None,
                    population_registry: None,
                    sampling_design: None,
                },
                antecedent_core::AssumptionSet::default(),
                ctx,
                &mut ws,
            )
            .ok()?
        };
        let aligned = static_aligned_influence(
            data,
            query,
            &atom.estimand,
            estimate.scores.influence.as_deref()?,
        )?;
        if aligned.iter().any(|v| !v.is_finite()) {
            return None;
        }
        ifs.push(aligned);
    }
    let k = ifs.len();
    let norms: Vec<f64> = ifs.iter().map(|f| f.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    if norms.iter().any(|n| !n.is_finite() || *n <= 0.0) {
        return None;
    }
    let mut corr = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            let dot: f64 = ifs[a].iter().zip(&ifs[b]).map(|(x, y)| x * y).sum();
            corr[a * k + b] = (dot / (norms[a] * norms[b])).clamp(-1.0, 1.0);
        }
    }
    Some(corr)
}

#[cfg(test)]
mod passed_route_record_tests {
    use super::passed_route_records;
    use antecedent_prob::InferenceDiagnostics;

    fn summary(records: &[antecedent_core::diagnostic::RouteNativeDiagnostic]) -> Vec<(&str, u64)> {
        records
            .iter()
            .map(|record| {
                assert!(!record.failed());
                (record.route_name(), record.unevaluable_mass().bits().expect("a measured mass"))
            })
            .collect()
    }

    #[test]
    fn conjugate_and_laplace_posteriors_record_passed_checks_with_the_measured_mass() {
        let bits = 0.25_f64.to_bits();
        let conjugate = InferenceDiagnostics::analytic("conjugate_gaussian");
        assert_eq!(summary(&passed_route_records(&conjugate, 0.25)), [("convergence", bits)]);
        let laplace =
            InferenceDiagnostics::laplace("laplace", true, 4, 1e-6, 10.0, false, Vec::new());
        assert_eq!(summary(&passed_route_records(&laplace, 0.25)), [("convergence", bits)]);
    }

    #[test]
    fn an_unrefused_unmeasurable_mass_is_unreported_not_zero() {
        let conjugate = InferenceDiagnostics::analytic("conjugate_gaussian");
        let records = passed_route_records(&conjugate, f64::NAN);
        assert!(records.iter().all(|record| record.unevaluable_mass().is_unreported()));
    }

    #[test]
    fn a_fit_that_would_be_refused_records_no_passed_checks() {
        let unconverged =
            InferenceDiagnostics::laplace("laplace", false, 4, 1e-6, 10.0, false, Vec::new());
        assert!(passed_route_records(&unconverged, 0.0).is_empty());
    }
}

#[cfg(test)]
mod unsupported_inference_tests {
    use super::unsupported_inference_error;

    #[test]
    fn every_unified_refusal_carries_the_unsupported_inference_record() {
        for summary in [
            "execute_pag_bayesian requires inference=Bayesian",
            "execute_graph_posterior_frequentist requires inference=Frequentist",
            "bayesian.basis.gcomp requires inference=Bayesian",
            "bayesian.robust_ate requires Bayesian inference",
        ] {
            let error = crate::CausalError::unsupported_inference(summary);
            assert_eq!(error.to_string(), summary);
            let records = error.route_diagnostics();
            assert_eq!(records.len(), 1, "{summary}: {records:?}");
            assert!(records[0].is_unsupported_inference(), "{summary}");
            assert!(records[0].unevaluable_mass().is_unreported(), "{summary}");
        }
    }

    #[test]
    fn an_unsupported_inference_refusal_is_not_a_failed_check_or_a_computational_failure() {
        let error = unsupported_inference_error("inference mode is not performed by this route");
        let records = error.route_diagnostics();
        assert_eq!(records.len(), 1, "{records:?}");
        assert!(records[0].is_unsupported_inference());
        assert!(!records[0].failed());
        assert!(!records[0].is_computational_failure());
        assert!(records[0].unevaluable_mass().is_unreported());
    }
}
