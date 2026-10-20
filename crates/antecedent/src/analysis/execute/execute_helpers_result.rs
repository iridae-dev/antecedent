// SPDX-License-Identifier: MIT OR Apache-2.0

/// Frozen result metadata required to assemble an identified execution without
/// retaining or consulting its mutable `Study` builder.
#[derive(Clone)]
pub(crate) struct IdentifiedResultContext {
    pub(crate) query: CausalQuery,
    pub(crate) graph_class: GraphClass,
    pub(crate) graph_version: u32,
    graph_has_certificate: bool,
    refute_default_downgrade: Option<RefuteSuite>,
    latency_bootstrap_not_applied: Option<(u32, u32)>,
    bootstrap_replicates: u32,
    pub(crate) inference: InferenceMode,
    population_registry: Option<PopulationRegistry>,
    latency_mode: Option<LatencyMode>,
    support_status: Option<crate::support::CellStatus>,
    structure_source: crate::support::StructureSource,
    tiered: Option<antecedent_graph::TieredBackground>,
    custom_validators: bool,
    /// Declared handling of missing values, frozen with the study.
    pub(crate) missing: Option<antecedent_data::MissingDeclaration>,
}

impl IdentifiedResultContext {
    /// Freeze result metadata from a builder at preparation time.
    #[must_use]
    pub(crate) fn from_study(study: &Study) -> Self {
        Self {
            query: study.query.clone(),
            graph_class: study.graph.class(),
            graph_version: study.graph.version(),
            graph_has_certificate: study.graph_posterior.is_none()
                && (study.graph.as_dag().is_some() || study.graph.as_admg().is_some()),
            refute_default_downgrade: study.refute_default_downgrade,
            latency_bootstrap_not_applied: study.latency_bootstrap_not_applied,
            bootstrap_replicates: study.bootstrap_replicates,
            inference: study.inference.clone(),
            population_registry: study.population_registry.clone(),
            latency_mode: study.latency_mode,
            support_status: study.support_status,
            structure_source: study.structure_source,
            tiered: study.tiered.clone(),
            custom_validators: !study.custom_validators.is_empty(),
            missing: study.missing.clone(),
        }
    }
}

/// Optional finish slots that most identified paths leave at default.
#[derive(Default)]
pub(super) struct IdentifiedExecuteExtras {
    pub certificate: Option<crate::Identification>,
    pub stage_timings_ns: Vec<(Arc<str>, u64)>,
    pub identify_provenance: Option<(Arc<str>, Arc<str>)>,
    pub estimate_provenance: Option<(Arc<str>, Arc<str>)>,
    pub posterior: Option<antecedent_estimate::CausalPosterior>,
    pub mediation_grid: Option<antecedent_estimate::TemporalMediationGrid>,
    pub n_draws: Option<u32>,
    pub predictive_checks: Vec<antecedent_validate::PredictiveCheckReport>,
    /// When set, replaces the identification + overlap + cache diagnostic seed.
    pub diagnostics: Option<Vec<Diagnostic>>,
    pub response: Option<antecedent_core::CausalResponse>,
    pub structural_response: Option<crate::result::StructuralResponseMixture>,
    pub gcm: Option<GcmSlot>,
    pub empty_provenance: bool,
    /// `None` uses the study bootstrap count; `Some(v)` writes `v` (response writes `None`).
    #[allow(
        clippy::option_option,
        reason = "the outer option selects override against the study default, the inner is the written value (response writes None)"
    )]
    pub bootstrap_replicates_requested: Option<Option<u32>>,
}

pub(super) fn identification_from_cache_or(
    ctx: &ExecutionContext,
    cache: Option<&crate::analysis::prepared::CachedStaticIdentification>,
    live: impl FnOnce() -> Result<(IdentificationResult, IdentifiedEstimand), CausalError>,
) -> Result<(IdentificationResult, IdentifiedEstimand, bool), CausalError> {
    if let Some(cache) = cache {
        return Ok((cache.identification.clone(), cache.estimand.clone(), true));
    }
    report_identify_compute(ctx);
    let (identification, estimand) = live()?;
    Ok((identification, estimand, false))
}

/// [`identification_from_cache_or`] for callers that retain the product: a
/// cache hit shares the prepared allocation instead of copying it.
pub(super) fn shared_identification_from_cache_or(
    ctx: &ExecutionContext,
    cache: Option<&Arc<crate::analysis::prepared::CachedStaticIdentification>>,
    live: impl FnOnce() -> Result<(IdentificationResult, IdentifiedEstimand), CausalError>,
) -> Result<(Arc<crate::analysis::prepared::CachedStaticIdentification>, bool), CausalError> {
    if let Some(cache) = cache {
        return Ok((Arc::clone(cache), true));
    }
    report_identify_compute(ctx);
    let (identification, estimand) = live()?;
    Ok((
        Arc::new(crate::analysis::prepared::CachedStaticIdentification {
            identification,
            estimand,
        }),
        false,
    ))
}

/// Tell the progress sink that identification is being computed rather than
/// served from a prepared cache.
///
/// Every compute path (single-graph, PAG envelope, bidirected ADMG, graph and
/// DBN posterior builders, sharp RD) reports this exactly when it identifies,
/// so `exec.identify.cached` on a prepared click is verifiable, not a flag.
pub(crate) fn report_identify_compute(ctx: &ExecutionContext) {
    if let Some(progress) = &ctx.progress {
        progress.report(0.0, crate::analysis::stage::PROGRESS_IDENTIFY_COMPUTE);
    }
}

pub(super) fn nan_effect() -> EffectEstimate {
    EffectEstimate::new(
        f64::NAN,
        f64::NAN,
        antecedent_core::AssumptionSet::default(),
        OverlapPolicy::ExplicitOverride,
    )
}

/// Identified atoms the Interactive latency tier's graph budget left out of the
/// stratified subsample. They were never estimated, so their mass is neither
/// unidentified nor unevaluable; it is reported on its own.
#[derive(Clone, Debug, Default)]
pub(super) struct InteractiveSubsampleDrop {
    /// Graph keys of the dropped identified atoms, in ensemble order.
    pub(super) keys: Vec<u64>,
    /// Absolute weight of the dropped atoms (same units as the ensemble weights).
    pub(super) mass: f64,
}

/// Most dropped keys a subsample diagnostic lists before summarizing the rest.
const SUBSAMPLE_DIAGNOSTIC_KEY_LIMIT: usize = 16;

/// Stratified Interactive subsample returning the dropped identified atoms.
///
/// The returned ensemble has the dropped atoms flagged Unidentified so every
/// downstream mixture excludes them without a fit; callers must report
/// [`InteractiveSubsampleDrop::mass`] as subsampled-out mass rather than as
/// unidentified mass (for an envelope posterior, via
/// [`report_subsampled_out_mass`]). Outside the Interactive tier (or when the
/// identified atoms fit the budget) nothing is dropped.
///
/// Call this **after** resolving the shared envelope prior from the first
/// identified atom in original order ([`resolve_envelope_prior_anchor`]), and
/// **before** per-graph estimation so dropped atoms never pay a fit. Subsample
/// must not move the prior anchor.
pub(super) fn interactive_subsample_graphs_accounted(
    latency_mode: Option<LatencyMode>,
    graphs: WeightedGraphSamples,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(WeightedGraphSamples, InteractiveSubsampleDrop), CausalError> {
    if latency_mode != Some(LatencyMode::Interactive) {
        return Ok((graphs, InteractiveSubsampleDrop::default()));
    }
    let mut rng = ctx.rng.stream_for(StreamDomain::Execute, 0xE11E_u64);
    let sub = graphs.stratified_interactive_subsample(INTERACTIVE_MAX_ENVELOPE_GRAPHS, &mut rng)?;
    let keys: Vec<u64> = graphs
        .identified
        .iter()
        .zip(sub.graphs.identified.iter())
        .zip(graphs.graph_keys.iter())
        .filter(|((before, after), _)| {
            **before == GraphIdentFlag::Identified && **after != GraphIdentFlag::Identified
        })
        .map(|(_, key)| *key)
        .collect();
    let drop = InteractiveSubsampleDrop { keys, mass: sub.leftover_identified_mass };
    if sub.approximate {
        diagnostics.push(interactive_subsample_diagnostic(&drop, graphs.total_weight()));
    }
    Ok((sub.graphs, drop))
}

/// `estimate.envelope.interactive_subsample`: which identified atoms the
/// Interactive tier skipped, why, and where their mass is reported.
fn interactive_subsample_diagnostic(
    drop: &InteractiveSubsampleDrop,
    total_weight: f64,
) -> Diagnostic {
    let listed = drop
        .keys
        .iter()
        .take(SUBSAMPLE_DIAGNOSTIC_KEY_LIMIT)
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let more = drop.keys.len().saturating_sub(SUBSAMPLE_DIAGNOSTIC_KEY_LIMIT);
    let listed = if more > 0 { format!("{listed}, +{more} more") } else { listed };
    let fraction = if total_weight > 0.0 { drop.mass / total_weight } else { f64::NAN };
    Diagnostic::new(
        "estimate.envelope.interactive_subsample",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "approximate=true leftover_identified_mass={} subsampled_out_mass={fraction} \
             subsampled_out_atoms={} max_identified={}; the Interactive latency tier mixes at \
             most {} identified graph atoms, so these identified atoms were left out of the \
             mixture: [{listed}]; their mass is reported as subsampled_out_mass, not as \
             unidentified or unevaluable mass; rerun at the Standard or Report tier to include \
             every identified atom",
            drop.mass,
            drop.keys.len(),
            INTERACTIVE_MAX_ENVELOPE_GRAPHS,
            INTERACTIVE_MAX_ENVELOPE_GRAPHS,
        ),
    )
}

/// Interactive graph×effect subsample for paths that fit every atom first:
/// stratified Identified selection as in [`interactive_subsample_graphs_accounted`],
/// then `per_graph` draws filtered to keys that remain Identified.
pub(super) fn maybe_interactive_envelope_subsample(
    latency_mode: Option<LatencyMode>,
    graphs: WeightedGraphSamples,
    per_graph: Vec<GraphEffectDraws>,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(WeightedGraphSamples, Vec<GraphEffectDraws>, InteractiveSubsampleDrop), CausalError> {
    let (graphs, drop) =
        interactive_subsample_graphs_accounted(latency_mode, graphs, ctx, diagnostics)?;
    // Same trigger as the subsample's own `approximate` flag.
    if drop.mass <= 0.0 {
        return Ok((graphs, per_graph, drop));
    }
    let keep_keys = identified_envelope_keys(&graphs);
    let filtered: Vec<GraphEffectDraws> =
        per_graph.into_iter().filter(|g| keep_keys.contains(&g.graph_key)).collect();
    Ok((graphs, filtered, drop))
}

/// Split the Interactive subsample's skipped mass out of an envelope posterior.
///
/// [`antecedent_estimate::aggregate_effect_envelope`] sees the subsampled
/// ensemble, where dropped atoms carry the Unidentified flag, so its
/// `unidentified_mass` counts them. This restores `unidentified_mass` to the
/// pre-subsample ensemble's share (atoms on which identification, or on paths
/// that demote failed fits, estimation failed) and reports the dropped share as
/// `subsampled_out_mass`. The identified mixture covers the remainder, so the
/// three categories sum to one. Mass the mixture does not cover, of either
/// kind, leaves the posterior graph-dependent.
pub(super) fn report_subsampled_out_mass(
    posterior: &mut CausalPosterior,
    pre_subsample: &WeightedGraphSamples,
    drop: &InteractiveSubsampleDrop,
) {
    let total = pre_subsample.total_weight();
    if !(total.is_finite() && total > 0.0) {
        return;
    }
    posterior.unidentified_mass = pre_subsample.unidentified_mass() / total;
    posterior.subsampled_out_mass = drop.mass / total;
    debug_assert!(
        (pre_subsample.identified_mass() - drop.mass) / total
            + posterior.unidentified_mass
            + posterior.subsampled_out_mass
            <= 1.0 + 1e-9
    );
    if posterior.unidentified_mass > antecedent_core::POSTERIOR_MASS_TOLERANCE
        || posterior.subsampled_out_mass > antecedent_core::POSTERIOR_MASS_TOLERANCE
    {
        posterior.identification = IdentificationStatus::GraphDependent;
    }
}

/// Structured identification-mass fields carried beside an envelope message.
///
/// The reasoning slot reads these; `identified` is absent when the emitter
/// knows only the unidentified share.
pub(crate) fn mass_fields(identified: Option<f64>, unidentified: f64) -> Vec<(Arc<str>, Arc<str>)> {
    let mut fields: Vec<(Arc<str>, Arc<str>)> = Vec::with_capacity(2);
    if let Some(identified) = identified {
        fields.push((Arc::from("identified_mass"), Arc::from(identified.to_string())));
    }
    fields.push((Arc::from("unidentified_mass"), Arc::from(unidentified.to_string())));
    fields
}

/// Envelope mass summary for a Bayesian graph envelope: unidentified mass and,
/// when the Interactive tier skipped atoms, their subsampled-out mass as a
/// separate field (the message is unchanged when nothing was skipped).
pub(super) fn envelope_mass_diagnostic(
    code: impl Into<Arc<str>>,
    posterior: &CausalPosterior,
) -> Diagnostic {
    let mut message = format!("unidentified_mass={}", posterior.unidentified_mass);
    if posterior.unevaluable_mass > 0.0 {
        // Distinct from unidentified_mass: this mass was identified but its
        // estimation failed, which is a refusal, not a negative proof.
        message.push_str(&format!(", unevaluable_mass={}", posterior.unevaluable_mass));
    }
    if posterior.subsampled_out_mass > 0.0 {
        message.push_str(&format!(", subsampled_out_mass={}", posterior.subsampled_out_mass));
    }
    let mut fields = mass_fields(None, posterior.unidentified_mass);
    if posterior.unevaluable_mass > 0.0 {
        if let Ok(mass) = antecedent_core::diagnostic::RouteMass::new(posterior.unevaluable_mass) {
            if let Some(bits) = mass.bits() {
                fields.push((Arc::from("unevaluable_mass_bits"), Arc::from(bits.to_string())));
            }
        }
    }
    Diagnostic::new(code, DiagnosticKind::Scientific, DiagnosticSeverity::Info, message)
        .with_fields(fields)
}

/// Resolve the shared envelope prior from a prepared Bayesian problem.
///
/// Call while preparing identified atoms in **original envelope order**, before
/// Interactive stratified selection. Subsample must not change which design
/// anchors the prior, and prepare eligibility must be established before
/// selection.
pub(super) fn resolve_envelope_prior_anchor(
    cfg: &BayesianConfig,
    prep: &antecedent_estimate::PreparedBayesianProblem,
    ctx: &ExecutionContext,
) -> Result<(Option<PriorSet>, Option<antecedent_prob::ConflictSummary>), CausalError> {
    resolve_bayesian_prior_with_conflict(cfg, prep, Some(ctx))
}

pub(super) fn response_functional_is_derivative(
    functional: &antecedent_core::ResponseFunctional,
) -> bool {
    matches!(
        functional,
        antecedent_core::ResponseFunctional::AverageDerivative { .. }
            | antecedent_core::ResponseFunctional::PointDerivative { .. }
            | antecedent_core::ResponseFunctional::DirectionalDerivative { .. }
            | antecedent_core::ResponseFunctional::Jacobian { .. }
    )
}

pub(super) fn bayesian_draw_count(inference: &InferenceMode) -> Result<usize, CausalError> {
    match inference {
        InferenceMode::Bayesian(cfg) => {
            if cfg.n_draws < 2 {
                return Err(CausalError::Unsupported {
                    message: "Bayesian inference requires n_draws >= 2; refusing silent rewrite of 0 or 1",
                });
            }
            Ok(cfg.n_draws)
        }
        InferenceMode::Frequentist => Ok(0),
    }
}

/// Wording shared with `Study::estimate_functional_effect`. Checked functional
/// routes call this instead of keeping a second sentence.
const FUNCTIONAL_BAYESIAN_PRIOR_REFUSAL: &str = "functional Bayesian prior transfer requires a declared functional mapping; a backdoor coefficient artifact cannot be applied as an isotropic CPT prior";

/// `None` for frequentist inference. Bayesian routes get a draw count, or the
/// shared functional-prior / `n_draws` refusal.
pub(crate) fn functional_bayesian_draws(
    inference: &InferenceMode,
) -> Result<Option<usize>, CausalError> {
    let InferenceMode::Bayesian(config) = inference else {
        return Ok(None);
    };
    if config.has_transfer() {
        return Err(CausalError::Unsupported { message: FUNCTIONAL_BAYESIAN_PRIOR_REFUSAL });
    }
    Ok(Some(bayesian_draw_count(inference)?))
}

/// Frequentist or Bayesian functional-effect fit against an already checked program.
pub(crate) fn run_functional(
    fitter: &FunctionalEffect,
    prepared: &antecedent_estimate::PreparedFunctionalEffect,
    inference: &InferenceMode,
    status: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), CausalError> {
    if let Some(n_draws) = functional_bayesian_draws(inference)? {
        let posterior =
            fitter.estimate_bayesian(prepared, n_draws, status, ctx).map_err(CausalError::from)?;
        let estimate = effect_from_posterior(&posterior)?;
        Ok((estimate, Some(posterior)))
    } else {
        let mut workspace = FunctionalDistributionWorkspace::default();
        let estimate = fitter.estimate(prepared, &mut workspace, ctx).map_err(CausalError::from)?;
        Ok((estimate, None))
    }
}

/// Tail shared by the checked functional, path, distribution, and ADMG-response executes.
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag is an independent switch of the assembled result (cancelled, early-stopped, refutation mirroring, interval rebinding)"
)]
pub(crate) struct CheckedResultFinish<'a> {
    pub common: &'a CheckedStaticCommon,
    pub identification: &'a IdentificationResult,
    pub estimand: &'a IdentifiedEstimand,
    pub estimate: EffectEstimate,
    pub estimate_assumptions: antecedent_core::AssumptionSet,
    pub diagnostics: Vec<Diagnostic>,
    pub refutations: Vec<antecedent_validate::RefutationReport>,
    pub distribution: Option<antecedent_estimate::InterventionalDistributionEstimate>,
    pub posterior: Option<CausalPosterior>,
    pub treatment: VariableId,
    pub outcome: VariableId,
    pub wall_time_ns: u64,
    pub bootstrap_replicates_requested: Option<u32>,
    pub bootstrap_replicates_ok: Option<u32>,
    pub n_draws: Option<u32>,
    pub cancelled: bool,
    pub early_stopped: bool,
    pub certificate_query: CausalQuery,
    pub certificate_graph_class: GraphClass,
    pub response: Option<CausalResponse>,
    pub mirror_refuted_evalue: bool,
    pub rebind_bayesian_interval: bool,
}

pub(crate) fn finish_checked_result(finish: CheckedResultFinish<'_>) -> StudyResult {
    let (identify_artifact, identify_operation) =
        crate::strategy_table::identify_provenance_step(finish.common.identifier);
    let (estimate_artifact, estimate_operation) =
        crate::strategy_table::estimate_provenance_step(finish.common.estimator);
    let provenance = provenance_pair(
        (identify_artifact, identify_operation, &[], &finish.identification.required_assumptions),
        (estimate_artifact, estimate_operation, &[identify_artifact], &finish.estimate_assumptions),
    );
    let mut result = assemble_result(AssembleArgs {
        logical: &finish.common.physical.logical.record,
        physical: &finish.common.physical.record,
        identification: finish.identification.clone(),
        estimand: finish.estimand.clone(),
        estimate: finish.estimate,
        posterior: finish.posterior,
        payload: finish.distribution.map_or(
            crate::result::ResultPayload::Scalar,
            crate::result::ResultPayload::Distribution,
        ),
        refutations: finish.refutations,
        diagnostics: finish.diagnostics,
        provenance,
        treatment: finish.treatment,
        outcome: finish.outcome,
        wall_time_ns: finish.wall_time_ns,
        latency_mode: finish.common.latency_mode.map(|mode| Arc::from(mode.as_str())),
        stage_timings_ns: Vec::new(),
        bootstrap_replicates_requested: finish.bootstrap_replicates_requested,
        bootstrap_replicates_ok: finish.bootstrap_replicates_ok,
        n_draws: finish.n_draws,
        cancelled: finish.cancelled,
        early_stopped: finish.early_stopped,
        bayesian: finish.common.inference.is_bayesian(),
    });
    // The interval binding was derived above without the response; a route that
    // wants it re-derived asks for `rebind_bayesian_interval`.
    if let Some(response) = finish.response {
        debug_assert!(
            matches!(result.payload, crate::result::ResultPayload::Scalar),
            "a result carries one payload family"
        );
        result.payload = crate::result::ResultPayload::Response(response);
    }
    result.certificate = Some(crate::AnalysisIdentification {
        identification: crate::Identification::Point {
            result: finish.identification.clone(),
            temporal_indexer: None,
            strategy: finish.common.identifier,
            structure_version: finish.common.graph_version,
        },
        query: finish.certificate_query,
        graph_class: finish.certificate_graph_class,
    });
    crate::analysis::helpers::stamp_result_context(
        &mut result,
        finish.common.support_status,
        finish.common.structure_source,
        finish.common.population_registry.as_ref(),
        &finish.common.custom_validator_names,
    );
    if finish.mirror_refuted_evalue {
        crate::analysis::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
    }
    if finish.rebind_bayesian_interval {
        result.rebind_interval(finish.common.inference.is_bayesian());
    }
    result
}
