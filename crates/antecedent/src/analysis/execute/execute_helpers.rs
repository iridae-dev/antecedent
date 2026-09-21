// Free functions supporting Study execute paths.
// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) fn gcm_query_vars(query: &CausalQuery) -> Result<(VariableId, VariableId), CausalError> {
    match query {
        CausalQuery::Counterfactual(q) => {
            let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Compile {
                message: "counterfactual missing outcome".into(),
            })?;
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).unwrap_or(outcome);
            Ok((treatment, outcome))
        }
        CausalQuery::AnomalyAttribution(q) => {
            let outcome = *q.targets.first().unwrap_or(&VariableId::from_raw(0));
            Ok((outcome, outcome))
        }
        CausalQuery::ChangeAttribution(q) => Ok((q.outcome, q.outcome)),
        CausalQuery::MechanismChange(q) => {
            let outcome = *q.targets.first().unwrap_or(&VariableId::from_raw(0));
            Ok((outcome, outcome))
        }
        CausalQuery::UnitChange(q) => Ok((q.outcome, q.outcome)),
        _ => Err(CausalError::Compile { message: "gcm_query_vars: unsupported query".into() }),
    }
}

/// Named execute / compile cell. Graph-completion arms still specialize on class.
#[derive(Clone, Copy)]
pub(super) enum AnalysisRoute {
    Response,
    TemporalResponse,
    StaticAte,
    Distribution,
    PathSpecific,
    Conditional,
    TemporalMediation,
    StaticMediation,
    Counterfactual,
    Anomaly,
    ChangeAttribution,
    MechanismChange,
    UnitChange,
    TemporalEffect,
    PanelTemporalEffect,
    PanelTemporalResponse,
    MultiEnvTemporalEffect,
    Transport,
    Interference,
}

#[derive(Clone, Copy)]
pub(super) enum DataModality {
    Tabular,
    TemporalOrEvent,
    Panel,
    MultiEnv,
}

pub(super) fn data_modality(data: &DataInput) -> DataModality {
    match data {
        DataInput::Tabular(_) => DataModality::Tabular,
        DataInput::Temporal(_) | DataInput::Event(_) => DataModality::TemporalOrEvent,
        DataInput::Panel(_) => DataModality::Panel,
        DataInput::MultiEnv(_) => DataModality::MultiEnv,
    }
}

pub(super) fn classify_analysis_route(
    data: &DataInput,
    query: &CausalQuery,
) -> Option<AnalysisRoute> {
    classify_route(data_modality(data), query)
}

pub(super) fn classify_route(modality: DataModality, query: &CausalQuery) -> Option<AnalysisRoute> {
    Some(match (modality, query) {
        (DataModality::Tabular, CausalQuery::Response(q)) if q.is_temporal() => return None,
        (DataModality::Tabular, CausalQuery::Response(_)) => AnalysisRoute::Response,
        (DataModality::TemporalOrEvent, CausalQuery::Response(q)) if q.is_temporal() => {
            AnalysisRoute::TemporalResponse
        }
        (DataModality::Tabular, CausalQuery::AverageEffect(_)) => AnalysisRoute::StaticAte,
        (DataModality::Tabular, CausalQuery::Distribution(_)) => AnalysisRoute::Distribution,
        (DataModality::Tabular, CausalQuery::PathSpecific(_)) => AnalysisRoute::PathSpecific,
        (DataModality::Tabular, CausalQuery::ConditionalEffect(_)) => AnalysisRoute::Conditional,
        (DataModality::TemporalOrEvent, CausalQuery::Mediation(_)) => {
            AnalysisRoute::TemporalMediation
        }
        (DataModality::Tabular, CausalQuery::Mediation(_)) => AnalysisRoute::StaticMediation,
        (DataModality::Tabular, CausalQuery::Counterfactual(_)) => AnalysisRoute::Counterfactual,
        (DataModality::Tabular, CausalQuery::AnomalyAttribution(_)) => AnalysisRoute::Anomaly,
        (DataModality::Tabular, CausalQuery::ChangeAttribution(_)) => {
            AnalysisRoute::ChangeAttribution
        }
        (DataModality::Tabular, CausalQuery::MechanismChange(_)) => AnalysisRoute::MechanismChange,
        (DataModality::Tabular, CausalQuery::UnitChange(_)) => AnalysisRoute::UnitChange,
        (DataModality::TemporalOrEvent, CausalQuery::TemporalEffect(_)) => {
            AnalysisRoute::TemporalEffect
        }
        (DataModality::Panel, CausalQuery::TemporalEffect(_)) => AnalysisRoute::PanelTemporalEffect,
        (DataModality::Panel, CausalQuery::Response(q)) if q.is_temporal() => {
            AnalysisRoute::PanelTemporalResponse
        }
        (DataModality::MultiEnv, CausalQuery::TemporalEffect(_)) => {
            AnalysisRoute::MultiEnvTemporalEffect
        }
        (DataModality::Tabular, CausalQuery::Transport(_)) => AnalysisRoute::Transport,
        (DataModality::Tabular, CausalQuery::Interference(_)) => AnalysisRoute::Interference,
        _ => return None,
    })
}

pub(super) enum GcmSlot {
    Counterfactual(crate::gcm::IteResult),
    Anomaly(Vec<antecedent_attribution::AnomalyScores>),
    Change(antecedent_attribution::ChangeAttributionResult),
    Mechanism(Vec<antecedent_attribution::MechanismChangeDetection>),
    Unit(antecedent_attribution::UnitChangeResult),
}

pub(super) fn provenance_ids(
    artifact: impl Into<Arc<str>>,
    op: impl Into<Arc<str>>,
) -> (Arc<str>, Arc<str>) {
    (artifact.into(), op.into())
}

pub(super) fn is_gcm_route(route: AnalysisRoute) -> bool {
    matches!(
        route,
        AnalysisRoute::Counterfactual
            | AnalysisRoute::Anomaly
            | AnalysisRoute::ChangeAttribution
            | AnalysisRoute::MechanismChange
            | AnalysisRoute::UnitChange
    )
}

pub(super) fn identify_cached_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "exec.identify.cached",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "identification reused from the prepare-time cache".to_string(),
    )
}

pub(super) struct IdentifiedExecuteFinish<'a> {
    pub physical: &'a PhysicalExecutionPlan,
    pub identification: IdentificationResult,
    pub estimand: IdentifiedEstimand,
    pub estimate: EffectEstimate,
    pub identifier_id: IdentifierId,
    pub estimator_id: EstimatorId,
    pub treatment: VariableId,
    pub outcome: VariableId,
    pub identify_cached: bool,
    pub extra_diagnostics: Vec<Diagnostic>,
    pub refutations: Vec<antecedent_validate::RefutationReport>,
    pub distribution: Option<antecedent_estimate::InterventionalDistributionEstimate>,
    pub mediation: Option<antecedent_estimate::TemporalMediationEstimate>,
    pub wall_time_ns: u64,
    pub bootstrap_replicates_ok: Option<u32>,
    pub cancelled: bool,
    pub early_stopped: bool,
    pub extras: IdentifiedExecuteExtras,
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
    /// Nested option is intentional: outer selects override vs study default.
    #[allow(clippy::option_option)]
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
    let mut rng = ctx.rng.stream(0xE11E_u64);
    let sub = graphs
        .stratified_interactive_subsample(INTERACTIVE_MAX_ENVELOPE_GRAPHS, &mut rng)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
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
    if posterior.unidentified_mass > 0.0 || posterior.subsampled_out_mass > 0.0 {
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
    let message = if posterior.subsampled_out_mass > 0.0 {
        format!(
            "unidentified_mass={}, subsampled_out_mass={}",
            posterior.unidentified_mass, posterior.subsampled_out_mass
        )
    } else {
        format!("unidentified_mass={}", posterior.unidentified_mass)
    };
    Diagnostic::new(code, DiagnosticKind::Scientific, DiagnosticSeverity::Info, message)
        .with_fields(mass_fields(None, posterior.unidentified_mass))
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

impl super::Study {
    pub(super) fn estimate_functional_effect(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        identification: &IdentificationResult,
        extra: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, Option<antecedent_estimate::CausalPosterior>), CausalError> {
        let est = FunctionalEffect {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalEffect::new()
        };
        let prepared = est
            .prepare(
                data,
                estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                extra,
            )
            .map_err(CausalError::from)?;
        if let InferenceMode::Bayesian(_) = &self.inference {
            if let Some(cfg) = match &self.inference {
                InferenceMode::Bayesian(c) => Some(c),
                InferenceMode::Frequentist => None,
            } {
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
            let posterior = est
                .estimate_bayesian(
                    &prepared,
                    bayesian_draw_count(&self.inference)?,
                    identification.status,
                    ctx,
                )
                .map_err(CausalError::from)?;
            let estimate = effect_from_posterior(&posterior)?;
            Ok((estimate, Some(posterior)))
        } else {
            let mut ws = FunctionalDistributionWorkspace::default();
            let estimate = est.estimate(&prepared, &mut ws, ctx).map_err(CausalError::from)?;
            Ok((estimate, None))
        }
    }
}

pub(super) fn is_multi_step_sustained(query: &TemporalEffectQuery) -> bool {
    query.is_multi_step_sustained()
}

pub(super) fn identified_envelope_keys(
    graphs: &WeightedGraphSamples,
) -> std::collections::HashSet<u64> {
    graphs
        .graph_keys
        .iter()
        .zip(graphs.identified.iter())
        .filter(|(_, flag)| **flag == GraphIdentFlag::Identified)
        .map(|(key, _)| *key)
        .collect()
}

/// Per-atom Bayesian fit retained so envelope validation can mix across structures.
pub(super) struct EnvelopeAtomFit {
    pub key: u64,
    pub prep: PreparedBayesianProblem,
    pub posterior: CausalPosterior,
    pub status: IdentificationStatus,
    pub weight: f64,
    pub estimand: IdentifiedEstimand,
    pub indexer: Option<TemporalIndexer>,
    /// Resolved coefficient prior this atom was fitted under (`None` = isotropic).
    pub prior: Option<PriorSet>,
}

/// Estimand, mass, and the atom's own fitted effect needed to mix Frequentist or
/// Bayesian envelope refuters.
///
/// `original` is this atom's own estimate (posterior mean / SD for Bayesian
/// atoms). Refuters compare each atom's perturbation refits against it, never
/// against the pooled mixture: a stable atom whose effect differs from the
/// pooled value is not a refutation.
pub(super) struct EnvelopeRefuteAtom {
    pub key: u64,
    pub weight: f64,
    pub estimand: IdentifiedEstimand,
    pub indexer: Option<TemporalIndexer>,
    pub original: EffectEstimate,
}

impl EnvelopeRefuteAtom {
    /// Refute atom for a Bayesian envelope fit; `original` is that atom's posterior summary.
    pub(super) fn from_fit(atom: &EnvelopeAtomFit) -> Result<Self, CausalError> {
        Ok(Self {
            key: atom.key,
            weight: atom.weight,
            estimand: atom.estimand.clone(),
            indexer: atom.indexer.clone(),
            original: effect_from_posterior(&atom.posterior)?,
        })
    }
}

pub(super) fn envelope_refute_atoms(
    fits: &[EnvelopeAtomFit],
) -> Result<Vec<EnvelopeRefuteAtom>, CausalError> {
    fits.iter().map(EnvelopeRefuteAtom::from_fit).collect()
}

/// Disclose a non-finite envelope SE: with several contributing atoms the joint
/// SE was not formed ([`envelope_se_omits_between_atom_variance`]); with one
/// atom that atom's own analytic SE is unavailable.
pub(super) fn envelope_se_omission_diagnostic(
    contributing_atoms: usize,
    se: f64,
) -> Option<Diagnostic> {
    if se.is_finite() {
        return None;
    }
    Some(if contributing_atoms > 1 {
        envelope_se_omits_between_atom_variance()
    } else {
        Diagnostic::new(
            "estimate.envelope.single_atom_se_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "the single contributing atom publishes no analytic SE, so the envelope \
             carries none; request bootstrap replicates for a resampled SE",
        )
    })
}

/// The legacy diagnostic code is retained for downstream consumers.
pub(super) fn envelope_se_omits_between_atom_variance() -> Diagnostic {
    Diagnostic::new(
        "estimate.envelope.se_omits_between_atom_variance",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "multi-atom uncertainty is unavailable: per-atom SEs do not determine the \
         sampling variance of fits sharing observations, and averaged interval \
         endpoints are not mixture quantiles. Only a single contributing atom \
         retains its own uncertainty. Completion weights are modeling choices, \
         not evidence that the averaged effect is identified across graphs",
    )
}

/// Warning shared by every one-series circular-block interval when the
/// estimating score's effective rows fall below the threshold of its SE family
/// ([`antecedent_estimate::CircularBlockFamily::min_effective_rows`]).
pub(super) fn short_series_warning(
    effective_rows: f64,
    family: antecedent_estimate::CircularBlockFamily,
) -> Option<Diagnostic> {
    family.is_short_series(effective_rows).then(|| {
        Diagnostic::new(
            "estimate.temporal.circular_block_se.short_series",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "estimating-score effective rows {effective_rows:.1} < {} (the {} \
                 threshold): the series may be too short for its serial dependence, and the \
                 circular-block interval may under-cover",
                family.min_effective_rows(),
                family.label(),
            ),
        )
    })
}

pub(super) fn temporal_class_block_span<'a>(
    indexers: impl IntoIterator<Item = &'a TemporalIndexer>,
) -> usize {
    indexers
        .into_iter()
        .map(|indexer| indexer.history() as usize + indexer.horizon() as usize)
        .max()
        .unwrap_or(1)
        .max(1)
}

/// Embed complete-case influences into the original row universe before mixing.
/// Equal vector lengths do not establish common observations.
pub(super) fn static_aligned_influence(
    data: &TabularData,
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
    influence: &[f64],
) -> Option<Vec<f64>> {
    if !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved) {
        return None;
    }
    let mut ids = vec![query.treatment, query.outcome];
    ids.extend(query.effect_modifiers.iter().copied());
    ids.extend(estimand.adjustment_set.iter().copied());
    ids.extend(estimand.instruments.iter().copied());
    ids.extend(estimand.mediators.iter().copied());
    let mask = data.complete_case_mask(&ids).ok()?;
    let rows: Vec<_> = mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
    if rows.len() != influence.len() || rows.len() < 2 {
        return None;
    }
    let mean = influence.iter().sum::<f64>() / influence.len() as f64;
    let mut full = vec![0.0; mask.len()];
    for (&i, &v) in rows.iter().zip(influence) {
        full[i] = (v - mean) * mask.len() as f64 / influence.len() as f64;
    }
    Some(full)
}

/// Mix ConditionalEffect exceedance grids across envelope atoms.
///
/// Per threshold, each atom is estimated on the same indicator transform, IFs
/// are aligned to the original row universe, and `F_a(c)` / scores are mixed
/// by frozen completion weight — the same rule as class-aware mean CATEs.
pub(super) fn attach_class_conditional_functional_grid(
    estimate: EffectEstimate,
    data: &TabularData,
    query: &antecedent_core::ConditionalEffectQuery,
    atoms: &[(f64, IdentifiedEstimand)],
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let _ = ctx;
    let Some(thresholds) = super::helpers::conditional_thresholds(
        data,
        query,
        atoms.iter().flat_map(|(_, e)| e.adjustment_set.iter().copied()),
    )?
    else {
        return Ok(estimate);
    };
    if atoms.is_empty() {
        return Ok(estimate);
    }
    let est = ConditionalLinearAdjustment::new();
    let y_orig = data.float64_values(query.inner.outcome).map_err(CausalError::from)?;
    let mut mixed_cdf = Vec::with_capacity(thresholds.len() * 2);
    let mut mixed_columns = Vec::with_capacity(thresholds.len() * 2);
    let mut event_n_eff = Vec::with_capacity(thresholds.len() * 2);
    let mut threshold_supported = Vec::with_capacity(thresholds.len() * 2);
    let mut n_eff_by_arm = [0.0, 0.0];
    for &threshold in &thresholds {
        let data_c = super::helpers::apply_outcome_functional(
            data,
            query.inner.outcome,
            &antecedent_core::OutcomeFunctional::exceedance(threshold),
        )?;
        let mut arm0 = 0.0;
        let mut arm1 = 0.0;
        let mut mass = 0.0;
        let mut contributing = 0usize;
        let mut atom_if0 = Vec::new();
        let mut atom_if1 = Vec::new();
        let mut atom_weights = Vec::new();
        let mut minimum_events = [f64::INFINITY; 2];
        let mut supported_in_all = [true; 2];
        let mut minimum_arm_counts = [f64::INFINITY; 2];
        for &(w, ref estimand) in atoms {
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            let mut transformed_query = query.clone();
            transformed_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
            let (_point, scores) =
                est.estimate_with_arm_scores(&data_c, estimand, &transformed_query)?;
            arm0 += w * (1.0 - scores.means[0]);
            arm1 += w * (1.0 - scores.means[1]);
            mass += w;
            contributing += 1;
            let aligned0 =
                static_aligned_influence(data, &query.inner, estimand, &scores.influence[0]);
            let aligned1 =
                static_aligned_influence(data, &query.inner, estimand, &scores.influence[1]);
            if let (Some(if0), Some(if1)) = (aligned0, aligned1) {
                atom_if0.push(if0.into_iter().map(|v| -v).collect::<Vec<_>>());
                atom_if1.push(if1.into_iter().map(|v| -v).collect::<Vec<_>>());
                atom_weights.push(w);
            }
            for arm in 0..2 {
                let (events, supported) = super::helpers::tail_event_support(
                    &scores.treatment,
                    &scores.row_index,
                    &y_orig,
                    arm,
                    threshold,
                );
                minimum_events[arm] = minimum_events[arm].min(events);
                supported_in_all[arm] &= supported;
                let count =
                    scores.treatment.iter().filter(|&&t| (t - arm as f64).abs() <= 1e-12).count()
                        as f64;
                minimum_arm_counts[arm] = minimum_arm_counts[arm].min(count);
            }
        }
        if !matches!(mass.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "class-aware ConditionalEffect grid had no estimable atoms".into(),
            });
        }
        mixed_cdf.push(arm0 / mass);
        mixed_cdf.push(arm1 / mass);
        if atom_if0.len() != contributing || atom_if1.len() != contributing {
            return Err(CausalError::Unsupported {
                message: "class-aware ConditionalEffect grid refused: every contributing atom must supply an aligned per-arm influence function; refusing a covariance from a subset of atoms",
            });
        }
        let refs0: Vec<&[f64]> = atom_if0.iter().map(Vec::as_slice).collect();
        let refs1: Vec<&[f64]> = atom_if1.iter().map(Vec::as_slice).collect();
        mixed_columns
            .push(antecedent_estimate::frozen_weight_mixture_scores(&refs0, &atom_weights)?);
        mixed_columns
            .push(antecedent_estimate::frozen_weight_mixture_scores(&refs1, &atom_weights)?);
        n_eff_by_arm = minimum_arm_counts;
        event_n_eff.extend(minimum_events);
        threshold_supported.extend(supported_in_all);
    }
    let mut out = estimate;
    if mixed_columns.len() != thresholds.len() * 2 {
        return Err(CausalError::Unsupported {
            message: "class-aware ConditionalEffect grid refused: joint IF columns must match the declared per-arm threshold grid",
        });
    }
    let refs: Vec<&[f64]> = mixed_columns.iter().map(Vec::as_slice).collect();
    out.joint_covariance = Some(antecedent_estimate::joint_influence_covariance(&refs, None)?);
    let n_eff = n_eff_by_arm[0] + n_eff_by_arm[1];
    out.score_inference = Some(antecedent_estimate::inference_from_influence_columns(
        &mixed_cdf,
        &refs,
        &event_n_eff,
        &threshold_supported,
        antecedent_estimate::WeightedSupport {
            n_eff,
            n_eff_by_arm: n_eff_by_arm.to_vec(),
            propensity_range: None,
            overlap_ok: n_eff_by_arm
                .iter()
                .all(|n| *n >= antecedent_estimate::scores::MIN_THRESHOLD_EVENTS),
        },
    )?);
    if thresholds.len() == 1 {
        // Scalar exceedance and its CDFs must describe the same fitted functional.
        out.ate = mixed_cdf[0] - mixed_cdf[1];
        let contrast: Vec<_> =
            mixed_columns[0].iter().zip(&mixed_columns[1]).map(|(a, b)| a - b).collect();
        out.se_analytic =
            antecedent_estimate::joint_influence_covariance(&[&contrast], None)?.se(0);
        out.influence = Some(contrast.into());
        out.se_bootstrap = None;
        out.simultaneous_interval = None;
        if threshold_supported.iter().any(|&supported| !supported) {
            out.se_analytic = f64::NAN;
            out.influence = None;
        }
    }
    let raw_cdf = mixed_cdf.clone();
    let rearranged = super::helpers::project_conditional_cdf(&mut mixed_cdf)?;
    out = out.with_monotone_rearranged(rearranged);
    out.exceedance_cdf = Some(Arc::from(mixed_cdf));
    if thresholds.len() > 1 {
        out.ate = f64::NAN;
        out.se_analytic = f64::NAN;
        out.se_bootstrap = None;
        out.influence = None;
        out.simultaneous_interval = None;
    }
    if let Some(tau) = query.inner.outcome_functional.quantile_level() {
        super::helpers::attach_conditional_quantile(
            &mut out,
            &thresholds,
            &raw_cdf,
            &mixed_columns,
            &threshold_supported,
            tau,
        )?;
    }
    Ok(out)
}

/// Frozen-weight mixture SE from per-atom IFs that share rows.
///
/// Graph weights are modeling choices. Unidentified mass is not mixed in.
pub(super) fn mix_static_envelope_se(atom_ifs: &[Vec<f64>], atom_weights: &[f64]) -> f64 {
    if atom_ifs.is_empty() || atom_ifs.len() != atom_weights.len() {
        return f64::NAN;
    }
    let refs: Vec<&[f64]> = atom_ifs.iter().map(Vec::as_slice).collect();
    let Ok(mixed) = antecedent_estimate::frozen_weight_mixture_scores(&refs, atom_weights) else {
        return f64::NAN;
    };
    antecedent_estimate::joint_influence_covariance(&[&mixed], None)
        .map(|c| c.se(0))
        .unwrap_or(f64::NAN)
}

/// Frozen-weight mixture IF, or `None` when atoms do not share a row universe.
pub(super) fn mixed_static_influence(
    atom_ifs: &[Vec<f64>],
    atom_weights: &[f64],
) -> Option<std::sync::Arc<[f64]>> {
    if atom_ifs.is_empty() || atom_ifs.len() != atom_weights.len() {
        return None;
    }
    let refs: Vec<&[f64]> = atom_ifs.iter().map(Vec::as_slice).collect();
    antecedent_estimate::frozen_weight_mixture_scores(&refs, atom_weights)
        .ok()
        .map(std::sync::Arc::from)
}

/// A single contributing atom keeps its SE. Multiple fits require joint
/// sampling covariance (or an explicitly defined graph-mixture distribution).
pub(super) fn mix_weighted_analytic_se(items: impl IntoIterator<Item = (f64, f64)>) -> f64 {
    let mut single = None;
    for (w, se) in items {
        if !w.is_finite() || w < 0.0 {
            return f64::NAN;
        }
        if w == 0.0 {
            continue;
        }
        if single.is_some() || !se.is_finite() || se < 0.0 {
            return f64::NAN;
        }
        single = Some(se);
    }
    single.unwrap_or(f64::NAN)
}

fn estimands_agree(left: &IdentifiedEstimand, right: &IdentifiedEstimand) -> bool {
    left.method == right.method
        && left.adjustment_set == right.adjustment_set
        && left.instruments == right.instruments
        && left.mediators == right.mediators
        && left.rd_design == right.rd_design
}

/// Choose how identified structural atoms may be combined.
///
/// Disagreeing estimand identities never produce a scalar mixture.
/// A shared partially identified estimand publishes an identified set.
pub(super) fn resolve_structural_aggregation(
    contributing: &[&IdentifiedEstimand],
    any_partial: bool,
) -> crate::result::StructuralAggregationPolicy {
    use crate::result::StructuralAggregationPolicy;
    let Some(first) = contributing.first() else {
        return StructuralAggregationPolicy::GraphDependentAtoms;
    };
    if contributing[1..].iter().all(|estimand| estimands_agree(first, estimand)) {
        if any_partial {
            StructuralAggregationPolicy::IdentifiedSetEnvelope
        } else {
            StructuralAggregationPolicy::SameEstimandWeightedMean
        }
    } else {
        StructuralAggregationPolicy::GraphDependentAtoms
    }
}

/// One evaluated graph-posterior atom for policy-aware scalar mixing.
pub(super) struct GraphPosteriorAtomValue {
    pub key: u64,
    pub weight: f64,
    pub status: IdentificationStatus,
    pub estimand: IdentifiedEstimand,
    pub value: f64,
}

/// Policy, scalar, and structural mixture for identified graph-posterior atoms.
pub(super) struct MixedGraphPosteriorPolicy {
    pub policy: crate::result::StructuralAggregationPolicy,
    pub ate: f64,
    pub mixable_scalar: bool,
    pub mixture: crate::result::StructuralResponseMixture,
}

/// Mix identified graph-posterior atoms under [`StructuralAggregationPolicy`].
///
/// Disagreeing estimand identities withhold the scalar (`ate = NaN`) and publish
/// an identified set over atom point values. Unidentified posterior atoms are
/// appended from `graphs` when absent from `atoms`.
pub(super) fn mix_graph_posterior_identified_atoms(
    graphs: &WeightedGraphSamples,
    atoms: &[GraphPosteriorAtomValue],
    unidentified_mass: f64,
    unevaluable_mass: f64,
    subsampled_out_mass: f64,
    truncated_atoms: usize,
) -> MixedGraphPosteriorPolicy {
    use crate::result::{StructuralAggregationPolicy, StructuralResponseAtom, StructuralWeightBasis};
    let mut structural_atoms = Vec::with_capacity(atoms.len());
    let mut identified_weight = 0.0;
    let mut mixable = 0.0;
    let mut weighted = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut contributing: Vec<&IdentifiedEstimand> = Vec::new();
    for atom in atoms {
        if atom.value.is_finite() {
            identified_weight += atom.weight;
            mixable += atom.weight;
            weighted += atom.weight * atom.value;
            lo = lo.min(atom.value);
            hi = hi.max(atom.value);
            contributing.push(&atom.estimand);
        }
        structural_atoms.push(StructuralResponseAtom {
            graph_key: atom.key,
            weight: atom.weight,
            status: atom.status,
            value: Some(antecedent_core::ResponseValue::Scalar(atom.value)),
            posterior: None,
            response: None,
        });
    }
    for (key, weight, flag) in graphs
        .graph_keys
        .iter()
        .zip(graphs.weights.iter())
        .zip(graphs.identified.iter())
        .map(|((k, w), f)| (*k, *w, *f))
    {
        if flag != GraphIdentFlag::Unidentified {
            continue;
        }
        if structural_atoms.iter().any(|atom| atom.graph_key == key) {
            continue;
        }
        structural_atoms.push(StructuralResponseAtom {
            graph_key: key,
            weight,
            status: IdentificationStatus::NotIdentified,
            value: None,
            posterior: None,
            response: None,
        });
    }
    let total = graphs.total_weight();
    let identified_mass = if total > 0.0 { identified_weight / total } else { 0.0 };
    let policy = resolve_structural_aggregation(&contributing, false);
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    let mixable_scalar =
        matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean) && mixable > 0.0;
    let ate = if mixable_scalar { weighted / mixable } else { f64::NAN };
    let conditional_on_identified = mixable_scalar.then_some(antecedent_core::ResponseValue::Scalar(ate));
    let mixture = crate::result::StructuralResponseMixture {
        weight_basis: StructuralWeightBasis::PosteriorProbability,
        atoms: structural_atoms,
        identified_mass,
        unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass,
        identified_set,
        identified_set_interval: None,
        conditional_on_identified,
        full_mass_scope: true,
        truncated_atoms,
    };
    MixedGraphPosteriorPolicy { policy, ate, mixable_scalar, mixture }
}

/// Publish which [`crate::result::StructuralAggregationPolicy`] governs a
/// graph-posterior mixture.
pub(super) fn push_graph_posterior_structural_aggregation_diagnostic(
    diagnostics: &mut Vec<Diagnostic>,
    policy: crate::result::StructuralAggregationPolicy,
    identified_mass: f64,
    unidentified_mass: f64,
    unevaluable_mass: f64,
    subsampled_out_mass: f64,
) {
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.structural_aggregation",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "policy={}; weight_basis=posterior_probability; identified_mass={}; \
                 unidentified_mass={unidentified_mass}; unevaluable_mass={unevaluable_mass}; \
                 subsampled_out_mass={subsampled_out_mass}",
                policy.as_str(),
                identified_mass,
            ),
        )
        .with_fields(mass_fields(Some(identified_mass), unidentified_mass)),
    );
}

/// Fail-closed rank: larger means less identified. Never used to upgrade a status.
fn identification_closedness(status: IdentificationStatus) -> u8 {
    match status {
        IdentificationStatus::NonparametricallyIdentified => 0,
        IdentificationStatus::IdentifiedUnderParametricRestrictions => 1,
        IdentificationStatus::IdentifiedUnderPriorRestrictions => 2,
        IdentificationStatus::PartiallyIdentified => 3,
        IdentificationStatus::GraphDependent => 4,
        IdentificationStatus::NotIdentified => 5,
    }
}

/// Most conservative status among requested horizons. Empty input is unidentified.
///
/// Priors and later identified horizons do not upgrade an earlier unidentified
/// or graph-dependent slice.
pub(super) fn most_conservative_identification_status(
    statuses: impl IntoIterator<Item = IdentificationStatus>,
) -> IdentificationStatus {
    statuses
        .into_iter()
        .max_by_key(|status| identification_closedness(*status))
        .unwrap_or(IdentificationStatus::NotIdentified)
}

/// Status for a Frequentist graph-posterior mixture.
///
/// Unidentified mass is [`IdentificationStatus::GraphDependent`]. Multiple
/// identified atoms with disagreeing estimands are
/// [`IdentificationStatus::PartiallyIdentified`]. A single identified atom
/// (or agreeing atoms) keeps that atom's status.
pub(super) fn graph_posterior_mixture_status(
    unidentified_mass: f64,
    contributing: &[&IdentifiedEstimand],
    primary: IdentificationStatus,
) -> IdentificationStatus {
    if unidentified_mass > 1e-12 {
        return IdentificationStatus::GraphDependent;
    }
    let Some(first) = contributing.first() else {
        return IdentificationStatus::NotIdentified;
    };
    if contributing[1..].iter().all(|estimand| estimands_agree(first, estimand)) {
        primary
    } else {
        IdentificationStatus::PartiallyIdentified
    }
}

pub(super) fn identified_weight_for_key(graphs: &WeightedGraphSamples, key: u64) -> f64 {
    graphs
        .graph_keys
        .iter()
        .zip(graphs.weights.iter())
        .zip(graphs.identified.iter())
        .filter(|((k, _), flag)| **k == key && **flag == GraphIdentFlag::Identified)
        .map(|((_, w), _)| *w)
        .sum()
}

pub(super) fn mix_prior_sensitivity_summaries(
    items: &[(f64, &antecedent_prob::PriorSensitivitySummary)],
) -> Option<antecedent_prob::PriorSensitivitySummary> {
    let first = items.first()?.1;
    let n = first.effect_means.len();
    // Grids of different families (or different grids) are not comparable points.
    if n == 0
        || items.iter().any(|(_, s)| {
            s.effect_means.len() != n
                || s.family != first.family
                || s.prior_scales != first.prior_scales
                || s.alphas != first.alphas
                || s.variance_multipliers != first.variance_multipliers
        })
    {
        return None;
    }
    let mut w_sum = 0.0;
    let mut means = vec![0.0; n];
    let mut second = vec![0.0; n];
    for (w, s) in items {
        if *w <= 0.0 {
            continue;
        }
        w_sum += *w;
        for i in 0..n {
            let m = s.effect_means[i];
            let sd = s.effect_sds.get(i).copied().unwrap_or(0.0);
            means[i] += *w * m;
            second[i] += *w * (sd * sd + m * m);
        }
    }
    if w_sum <= 0.0 {
        return None;
    }
    let mut sds = vec![0.0; n];
    for i in 0..n {
        means[i] /= w_sum;
        sds[i] = (second[i] / w_sum - means[i] * means[i]).max(0.0).sqrt();
    }
    Some(antecedent_prob::PriorSensitivitySummary {
        family: first.family,
        prior_scales: Arc::clone(&first.prior_scales),
        alphas: Arc::clone(&first.alphas),
        variance_multipliers: Arc::clone(&first.variance_multipliers),
        effect_means: Arc::from(means),
        effect_sds: Arc::from(sds),
    })
}

/// Whether the atom's posterior check adds the serial-dependence axis.
fn temporal_predictive_check(atom: &EnvelopeAtomFit, refute: RefuteSuite) -> bool {
    matches!(refute, RefuteSuite::Full) && atom.indexer.is_some()
}

/// Whether two envelope atoms would produce identical predictive-check reports:
/// same fitted problem, prior, posterior draws and check variant (the temporal
/// axis also reads the posterior's tempering note).
fn same_predictive_check_inputs(
    a: &EnvelopeAtomFit,
    b: &EnvelopeAtomFit,
    refute: RefuteSuite,
) -> bool {
    let temporal = temporal_predictive_check(a, refute);
    temporal == temporal_predictive_check(b, refute)
        && a.prior == b.prior
        && a.posterior.draws == b.posterior.draws
        && (!temporal || a.posterior.diagnostics.notes == b.posterior.diagnostics.notes)
        && same_fitted_problem(&a.prep, &b.prep)
}

/// Run PPC (and, under `full`, prior-sensitivity) on identified envelope atoms.
///
/// Aggregation is mixture-weighted by graph-posterior mass. Records the method as a
/// diagnostic so `full` means the same thing on PAG / graph-posterior as on a single DAG.
pub(super) fn run_envelope_bayesian_full_validation(
    refute: RefuteSuite,
    cfg: &crate::inference::BayesianConfig,
    est: &BayesianGComputationAte,
    atoms: &[EnvelopeAtomFit],
    mixture_posterior: &mut CausalPosterior,
    estimate_ate: f64,
    ctx: &ExecutionContext,
    predictive_sims: u32,
    refutations: &mut Vec<antecedent_validate::RefutationReport>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<Vec<PredictiveCheckReport>, CausalError> {
    const PPC_ALPHA: f64 = 0.05;
    fn borrow(items: &[(f64, PredictiveCheckReport)]) -> Vec<(f64, &PredictiveCheckReport)> {
        items.iter().map(|(w, r)| (*w, r)).collect()
    }

    let mut predictive_checks = Vec::new();
    if matches!(refute, RefuteSuite::None) || atoms.is_empty() {
        return Ok(predictive_checks);
    }
    let mut prior_items: Vec<(f64, PredictiveCheckReport)> = Vec::with_capacity(atoms.len());
    let mut post_items: Vec<(f64, PredictiveCheckReport)> = Vec::with_capacity(atoms.len());
    for (j, atom) in atoms.iter().enumerate() {
        // Both checks are deterministic in (design, prior, posterior draws, seed),
        // so atoms that fitted the same problem under the same prior — graphs
        // sharing an adjustment set — replicate the same outcomes: run once.
        if let Some(i) = (0..j).find(|&i| same_predictive_check_inputs(&atoms[i], atom, refute)) {
            prior_items.push((atom.weight, prior_items[i].1.clone()));
            post_items.push((atom.weight, post_items[i].1.clone()));
            continue;
        }
        // Each atom's checks use the prior that atom was actually fitted under.
        let atom_est = BayesianGComputationAte { prior: atom.prior.clone(), ..est.clone() };
        let ppc_prior = atom_est.prior_in_force(atom.prep.design.ncols);
        let prior_rep = PriorPredictiveCheck::for_estimator(&atom_est, ctx)
            .with_n_sims(predictive_sims)
            .check_with_prior(&atom.prep, &ppc_prior, ctx)
            .map_err(CausalError::from)?;
        // Temporal atoms (lag indexer present) add the serial-dependence discrepancy
        // under `full`; exchangeable static rows keep the two-axis check.
        let post_check =
            PosteriorPredictiveCheck::for_estimator(&atom_est, ctx).with_n_sims(predictive_sims);
        let post_rep = if temporal_predictive_check(atom, refute) {
            post_check
                .check_temporal(&atom.prep, &atom.posterior, ctx.rng.master_seed())
                .map_err(CausalError::from)?
        } else {
            post_check.check(&atom.prep, &atom.posterior).map_err(CausalError::from)?
        };
        prior_items.push((atom.weight, prior_rep));
        post_items.push((atom.weight, post_rep));
    }
    for mixed in [
        PredictiveCheckReport::mixture_weighted(&borrow(&prior_items)),
        PredictiveCheckReport::mixture_weighted(&borrow(&post_items)),
    ]
    .into_iter()
    .flatten()
    {
        refutations.push(mixed.to_refutation_report(estimate_ate, PPC_ALPHA));
        predictive_checks.push(mixed);
    }

    // `Some(mixed)` under `full`: whether the per-completion sensitivity grids mixed.
    let mut sensitivity_mixed = None;
    if matches!(refute, RefuteSuite::Full) {
        let mut ws = BayesianGCompWorkspace::default();
        let mut owned = Vec::with_capacity(atoms.len());
        // The grid descriptor is the same for every atom (it depends only on `cfg`),
        // so keep the first one rather than paying for a second full evaluation.
        let mut grid = None;
        for atom in atoms {
            let atom_est = BayesianGComputationAte { prior: atom.prior.clone(), ..est.clone() };
            let (summary, sens) = evaluate_bayesian_prior_sensitivity(
                cfg,
                &atom_est,
                &atom.prep,
                atom.status,
                &atom.posterior,
                &mut ws,
                ctx,
            )?;
            owned.push((atom.weight, summary));
            grid.get_or_insert(sens);
        }
        let sens_items: Vec<_> = owned.iter().map(|(w, s)| (*w, s)).collect();
        let mixed = mix_prior_sensitivity_summaries(&sens_items);
        sensitivity_mixed = Some(mixed.is_some());
        if let (Some(mixed), Some(sens)) = (mixed, grid.as_ref()) {
            refutations.push(sens.to_report(&mixed, estimate_ate));
            *mixture_posterior = with_prior_sensitivity(mixture_posterior.clone(), mixed);
        }
    }

    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    diagnostics.extend(envelope_validation_diagnostics(&atom_keys, sensitivity_mixed));
    Ok(predictive_checks)
}

/// Diagnostics describing what the class-envelope Bayesian validation evaluated.
///
/// `sensitivity_mixed` is `None` without a prior-sensitivity pass (`cheap`), and
/// otherwise whether the per-completion sensitivity summaries shared one grid and
/// so produced a mixture-weighted report. Grids of different perturbed-prior
/// families (a transferred prior on some completions, the isotropic scale on
/// others) are not comparable points, so no prior-sensitivity report is published
/// and the diagnostics say so rather than claiming it was evaluated.
pub(super) fn envelope_validation_diagnostics(
    atom_keys: &str,
    sensitivity_mixed: Option<bool>,
) -> Vec<Diagnostic> {
    let summary = match sensitivity_mixed {
        Some(true) => format!(
            "PPC and prior-sensitivity evaluated per identified completion [{atom_keys}]; \
             reports are mixture-weighted by graph posterior mass"
        ),
        Some(false) => format!(
            "PPC evaluated per identified completion [{atom_keys}]; reports are \
             mixture-weighted by graph posterior mass. Prior sensitivity was computed per \
             completion but no prior-sensitivity report is published (see \
             refute.bayesian.prior_sensitivity.not_mixed)"
        ),
        None => format!(
            "PPC evaluated per identified completion [{atom_keys}]; reports are \
             mixture-weighted by graph posterior mass"
        ),
    };
    let mut out = vec![Diagnostic::new(
        "refute.bayesian.ppc.envelope",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        summary,
    )];
    if sensitivity_mixed == Some(false) {
        out.push(Diagnostic::new(
            "refute.bayesian.prior_sensitivity.not_mixed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "prior sensitivity was not reported: the identified completions [{atom_keys}] \
                 perturbed incomparable grids (different prior families, e.g. a transferred \
                 prior on some completions and the isotropic scale on others, or different grid \
                 points), so no mixture-weighted summary exists; the result carries no \
                 prior-sensitivity evidence"
            ),
        ));
    }
    out
}

/// Run cheap/full effect refuters on every contributing envelope atom and mix.
///
/// Each atom is validated on its own estimand (and lag indexer, for DBN atoms)
/// against that atom's own estimate ([`EnvelopeRefuteAtom::original`]), so a
/// stable atom whose effect differs from the pooled mixture is not reported as
/// a refutation. Reports of the same refuter id are mixed by posterior mass;
/// the mixed `original_ate` is therefore the mass-weighted mean of the per-atom
/// estimates that were actually compared (over the atoms that produced that
/// report), and the mixed check passes only if every contributing atom passes.
/// Validators that are `NotApplicable` on every contributing atom surface once;
/// a check that applies to only a subset mixes that subset.
pub(super) fn run_envelope_effect_refuters(
    data: &TabularData,
    query: &AverageEffectQuery,
    atoms: &[EnvelopeRefuteAtom],
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    suite: RefuteSuite,
    estimator: &str,
    custom: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    temporal_query: Option<&TemporalEffectQuery>,
    split: Option<&DiscoveryEstimationSplit>,
    time_index: Option<&TimeIndex>,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    if matches!(suite, RefuteSuite::None) && custom.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut order = Vec::new();
    let mut by_refuter: std::collections::HashMap<
        Arc<str>,
        Vec<(f64, antecedent_validate::RefutationReport)>,
    > = std::collections::HashMap::new();
    let mut na_weight: std::collections::HashMap<
        antecedent_validate::ValidatorId,
        (f64, Arc<str>),
    > = std::collections::HashMap::new();
    let mut contributing = 0.0;
    let mut diagnostics = Vec::new();
    for atom in atoms {
        if atom.weight <= 0.0 {
            continue;
        }
        contributing += atom.weight;
        let temporal = match (atom.indexer.as_ref(), temporal_query) {
            (Some(indexer), Some(tq)) => Some(TemporalRefitContext {
                indexer,
                temporal_query: tq,
                split,
                kernel_policy: &ctx.kernel_policy,
                time_index,
                panel: None,
            }),
            _ => None,
        };
        let outcomes = refute_outcomes(
            data,
            &atom.estimand,
            query,
            &atom.original,
            workspace,
            None,
            ctx,
            suite,
            estimator,
            custom,
            temporal,
        )?;
        for report in ValidationSuite::reports_only(&outcomes) {
            let bucket = by_refuter.entry(Arc::clone(&report.refuter)).or_insert_with(|| {
                order.push(Arc::clone(&report.refuter));
                Vec::new()
            });
            bucket.push((atom.weight, report));
        }
        for (validator, reason) in ValidationSuite::not_applicable_only(&outcomes) {
            na_weight
                .entry(validator)
                .and_modify(|(w, _)| *w += atom.weight)
                .or_insert((atom.weight, reason));
        }
    }
    let mut reports = Vec::with_capacity(order.len());
    for id in order {
        let Some(items) = by_refuter.get(&id) else {
            continue;
        };
        let borrowed: Vec<(f64, &antecedent_validate::RefutationReport)> =
            items.iter().map(|(w, r)| (*w, r)).collect();
        if let Some(mixed) = antecedent_validate::RefutationReport::mixture_weighted(&borrowed) {
            reports.push(mixed);
        }
    }
    if contributing > 0.0 {
        for (validator, (weight, reason)) in na_weight {
            if weight / contributing >= 1.0 - 1e-12 {
                diagnostics.push(validator_not_applicable_diagnostic(validator, &reason));
            }
        }
    }
    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    diagnostics.push(Diagnostic::new(
        "refute.envelope.effect_mixture",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "effect refuters evaluated each contributing graph atom [{atom_keys}] against that \
             atom's own estimate, not the pooled mixture; reports mix by envelope mass (the \
             mixed original_ate is the mass-weighted mean of the per-atom estimates compared) \
             and pass only if every contributing atom passes"
        ),
    ));
    Ok((reports, diagnostics))
}

/// Build a GCM / parametric-SCM estimand and identification result for `treatment`/`outcome`.
///
/// The estimate itself is computed elsewhere (by the fitted parametric SCM, not by evaluating
/// a backdoor-adjustment formula); this function only produces the *inspectable* estimand
/// metadata. Like `stub_accepted_graph_for` (in `analysis::builder`), the adjustment set stays
/// deliberately empty — GCM does not identify via a backdoor covariate set — but unlike the
/// placeholder this replaced, `functional` is a real expression naming the actual
/// `treatment`/`outcome` pair rather than a nil `ExprId` into an empty arena, so the estimand
/// is honest about which variables it refers to. The node is a minimal
/// `Expectation`/`Distribution` leaf (not `CausalExprArena::backdoor_ate`'s `Product`/`SumOut`
/// shape) so it is inert if a future caller ever tries to mechanically re-evaluate
/// `functional` via the arena's generic evaluator — there is no adjustment-set
/// marginalization here to (mis)compute. Licensed transport and interference
/// routes do not call this; they stamp `transport.sid` / `interference.design`.
pub(crate) fn parametric_scm_identification(
    query: CausalQuery,
    treatment: VariableId,
    outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let y = arena.intern_var_set([outcome]);
    let do_t = arena.intern_intervention_set([treatment]);
    let empty = arena.empty_var_set();
    let distribution = arena.intern_distribution(y, empty, do_t, DomainRef::Interventional);
    let functional = arena
        .intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution });
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            "gcm.parametric",
            Some(Arc::from(format!(
                "parametric SCM: treatment={treatment:?} outcome={outcome:?}; no adjustment \
                 set (GCM does not identify via backdoor covariates)"
            ))),
        ),
    );
    let estimand = IdentifiedEstimand::backdoor("gcm.parametric", Arc::from([]), functional);
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("gcm.supplied_structural_mechanisms"),
                description: Arc::from(
                    "Identification is conditional on the supplied acyclic structural mechanisms and their declared intervention semantics being an adequate model of the data-generating process.",
                ),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("gcm.parametric"),
        },
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let identification = IdentificationResult::from_parts(
        IdentificationStatus::IdentifiedUnderParametricRestrictions,
        query,
        vec![estimand.clone()],
        arena,
        DerivationTrace::default(),
        assumptions,
        Vec::new(),
        IdentificationPerformanceRecord::default(),
        None,
    );
    (identification, estimand)
}

pub(super) fn binary_cf_interventions(
    query: &antecedent_core::CounterfactualQuery,
) -> Result<(VariableId, f64, f64), CausalError> {
    if query.allow_nested || query.outcomes.len() != 1 || query.interventions.len() != 1 {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path currently supports a single hard \
                 intervention for ITE (use gcm helpers for multi-world predict)",
        });
    }
    let Intervention::Set { variable, value } = &query.interventions[0] else {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path requires a hard Set intervention",
        });
    };
    let active = value.as_f64().ok_or_else(|| CausalError::Compile {
        message: "counterfactual intervention value must be f64".into(),
    })?;
    let Intervention::Set { variable: control_var, value: control_value } = &query.control else {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path requires a hard Set control intervention",
        });
    };
    if control_var != variable {
        return Err(CausalError::Compile {
            message: format!(
                "counterfactual control targets {control_var:?}, expected {variable:?}"
            ),
        });
    }
    let control = control_value.as_f64().ok_or_else(|| CausalError::Compile {
        message: "counterfactual control value must be f64".into(),
    })?;
    if !control.is_finite() {
        return Err(CausalError::Unsupported { message: "counterfactual control must be finite" });
    }
    Ok((*variable, active, control))
}

/// The identified set of a scalar functional, in the coordinate-free shape the
/// portable format requires: no grid, one bound. The estimand already says what
/// the bound means, so no coordinate is invented for it.
///
/// Every route that publishes a scalar identified set builds it here: the
/// portable format accepts `dimension = 0` precisely so a scalar bound is not
/// dressed up with a `0.0` coordinate that nothing reads.
pub(super) fn scalar_identified_set(lower: f64, upper: f64) -> antecedent_core::ResponseEnvelope {
    antecedent_core::ResponseEnvelope {
        grid: Arc::from([]),
        dimension: 0,
        lower: Arc::from([lower]),
        upper: Arc::from([upper]),
    }
}

/// Completion-mass summary of a graph-class envelope: the one body behind every
/// `identify.*.envelope` diagnostic.
///
/// `prefix` names the class in the message (`generalized.adjustment envelope`,
/// `cpdag.mec envelope`), and each `extra` pair is appended as `, key=value`.
/// The structured half is [`mass_fields`], which the reasoning slot reads.
pub(super) fn class_envelope_diagnostic<G>(
    code: impl Into<Arc<str>>,
    prefix: &str,
    envelope: &IdentificationEnvelope<G>,
    extra: &[(&str, String)],
) -> Diagnostic {
    let mut message = format!(
        "{prefix}: identified_mass={}, unidentified_mass={}, cases={}",
        envelope.identified_weight.0,
        envelope.unidentified_weight.0,
        envelope.cases.len()
    );
    for (key, value) in extra {
        use std::fmt::Write as _;
        let _ = write!(message, ", {key}={value}");
    }
    Diagnostic::new(code, DiagnosticKind::Scientific, DiagnosticSeverity::Info, message)
        .with_fields(mass_fields(
            Some(envelope.identified_weight.0),
            envelope.unidentified_weight.0,
        ))
}

/// Per-completion outcome of a graph-class (CPDAG/PAG, static or temporal)
/// envelope arm, in `envelope.cases` order.
#[derive(Clone, Copy, Debug, Default)]
pub(super) enum ClassAtomOutcome {
    /// Identified and evaluated: the completion's own point value.
    Evaluated(f64),
    /// Identified but left unevaluated by the Interactive latency tier's graph
    /// budget. Neither unidentified nor a failed estimate.
    SubsampledOut,
    /// Not evaluated for any other reason (unidentified completions land here
    /// too; their status decides which mass they join).
    #[default]
    NotEvaluated,
}

/// What a class mixture says about itself, beside the mixture: facts a caller
/// needs to decide whether publishing it is right.
pub(super) struct ClassMixtureFacts {
    /// Total weight the masses were divided by. Not positive means the
    /// envelope carried no mass at all and the masses are all zero.
    pub total_weight: f64,
    /// All mass identified, a degenerate identified set, and an identified
    /// envelope status: the mixture restates a point, and publishing it would
    /// dress a point up as bounds.
    pub point_identified: bool,
}

/// Structural uncertainty of a graph-class envelope: every completion with its
/// weight and status, the evaluated completions' point values, and the
/// identified set `[min, max]` over them.
///
/// The one class-mixture body. `outcomes` is indexed by `envelope.cases`;
/// `key_fn` supplies each atom's `graph_key` (the case index for a static
/// class, the completion fingerprint for a temporal one); `weights` overrides
/// the enumeration weights when the caller carries its own class mass.
///
/// Mass is a fraction of the total weight and is kept apart by kind:
/// identified-and-evaluated, unidentified, identified-but-unevaluable, and
/// identified-but-subsampled-out. Which cases are identified is
/// [`identification_status_carries_identified_mass`] — the envelope's own
/// split — so `identified_mass + unidentified_mass` here agrees with
/// `envelope.identified_weight` / `unidentified_weight` in the diagnostic
/// published beside it. `unevaluable_mass` is taken as the remainder, so the
/// four masses conserve exactly under division.
pub(super) fn class_structural_mixture<G>(
    envelope: &IdentificationEnvelope<G>,
    weight_basis: crate::result::StructuralWeightBasis,
    key_fn: impl Fn(usize, &antecedent_identify::GraphIdentificationCase<G>) -> u64,
    weights: Option<&[f64]>,
    outcomes: &[ClassAtomOutcome],
) -> (crate::result::StructuralResponseMixture, ClassMixtureFacts) {
    let mut atoms = Vec::with_capacity(envelope.cases.len());
    let (mut identified_weight, mut unidentified_weight) = (0.0, 0.0);
    let (mut unevaluable_weight, mut subsampled_weight) = (0.0, 0.0);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (index, case) in envelope.cases.iter().enumerate() {
        let weight = weights.and_then(|w| w.get(index).copied()).unwrap_or(case.weight.0);
        let outcome = outcomes.get(index).copied().unwrap_or_default();
        let value = match outcome {
            ClassAtomOutcome::Evaluated(value) if value.is_finite() => Some(value),
            _ => None,
        };
        if !identification_status_carries_identified_mass(case.result.status) {
            unidentified_weight += weight;
        } else if let Some(value) = value {
            identified_weight += weight;
            lo = lo.min(value);
            hi = hi.max(value);
        } else if matches!(outcome, ClassAtomOutcome::SubsampledOut) {
            subsampled_weight += weight;
        } else {
            unevaluable_weight += weight;
        }
        atoms.push(crate::result::StructuralResponseAtom {
            graph_key: key_fn(index, case),
            weight,
            status: case.result.status,
            value: value.map(antecedent_core::ResponseValue::Scalar),
            posterior: None,
            response: None,
        });
    }
    let total = identified_weight + unidentified_weight + unevaluable_weight + subsampled_weight;
    let positive = total > 0.0;
    let (identified_mass, unidentified_mass, subsampled_out_mass) = if positive {
        (identified_weight / total, unidentified_weight / total, subsampled_weight / total)
    } else {
        (0.0, 0.0, 0.0)
    };
    // The remainder, so the four masses conserve exactly under division.
    let unevaluable_mass = if positive {
        (1.0 - identified_mass - unidentified_mass - subsampled_out_mass).max(0.0)
    } else {
        0.0
    };
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    // Exact comparisons on purpose: "every completion is identified" and "they
    // all agree" are exact facts about the mass that was summed and the values
    // that were compared, not measurements with a tolerance.
    #[allow(clippy::float_cmp)]
    let facts = ClassMixtureFacts {
        total_weight: total,
        point_identified: identified_mass == 1.0
            && lo == hi
            && matches!(
                envelope.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
                    | IdentificationStatus::IdentifiedUnderPriorRestrictions
            ),
    };
    let mixture = crate::result::StructuralResponseMixture {
        weight_basis,
        atoms,
        identified_mass,
        unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass,
        identified_set,
        identified_set_interval: None,
        conditional_on_identified: None,
        full_mass_scope: envelope.truncated_completions == 0,
        truncated_atoms: envelope.truncated_completions,
    };
    (mixture, facts)
}

/// Statuses the envelope itself counts as identified mass.
///
/// A re-export, not a second list: `antecedent-identify` owns the split —
/// [`antecedent_identify::carries_identified_mass`] is the same predicate
/// `IdentificationEnvelope::from_cases` uses to divide `identified_weight`
/// from `unidentified_weight`, so a mass this crate publishes cannot
/// contradict the envelope diagnostic beside it.
///
/// Wider than [`identification_status_ok_for_case`], which licenses
/// *estimating* a case: a completion identified only under prior restrictions
/// carries identified mass and its assumptions, but no frequentist arm
/// estimates it.
pub(crate) use antecedent_identify::carries_identified_mass as identification_status_carries_identified_mass;

/// Statuses a frequentist arm may estimate. Narrower than
/// [`identification_status_carries_identified_mass`]: a completion identified
/// only under prior restrictions is *not* estimable, but its mass is still
/// identified mass.
pub(crate) fn identification_status_ok_for_case(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

pub(super) fn envelope_to_identification_result<G>(
    envelope: &IdentificationEnvelope<G>,
    query: &AverageEffectQuery,
) -> IdentificationResult {
    envelope_to_identification_result_for(envelope, CausalQuery::AverageEffect(query.clone()))
}

pub(super) fn envelope_to_identification_result_for<G>(
    envelope: &IdentificationEnvelope<G>,
    query: CausalQuery,
) -> IdentificationResult {
    let mut estimands = Vec::new();
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut diagnostics = Vec::new();
    for case in &envelope.cases {
        if identification_status_ok_for_case(case.result.status) {
            estimands.extend(case.result.estimands.iter().cloned());
        }
        // Every case the envelope counts as identified mass carries its
        // assumptions, including one identified only under prior restrictions:
        // its mass is reported, so the conditions behind it must be too.
        if identification_status_carries_identified_mass(case.result.status) {
            assumptions.extend_unique(&case.result.required_assumptions.entries);
        }
        // Refused and unverified cases explain the missing mass too.
        diagnostics.extend(case.result.diagnostics.iter().cloned());
    }
    for feature in &envelope.critical_graph_features {
        diagnostics.push(Diagnostic::new(
            Arc::from(format!("identify.envelope.{}", feature.kind)),
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            Arc::clone(&feature.detail),
        ));
    }
    if let Some(inv) = &envelope.invariant {
        if estimands.is_empty() {
            estimands.push(inv.clone());
        }
    }
    IdentificationResult::from_parts(
        envelope.status,
        query,
        estimands,
        CausalExprArena::new(),
        DerivationTrace::default(),
        assumptions,
        diagnostics,
        IdentificationPerformanceRecord::default(),
        None,
    )
}

pub(crate) fn admg_has_bidirected(admg: &Admg) -> bool {
    admg.has_bidirected()
}

pub(super) fn admg_to_dag(admg: &Admg) -> Result<Dag, CausalError> {
    let n = u32::try_from(admg.node_count())
        .map_err(|_| CausalError::Compile { message: "ADMG too large".into() })?;
    let mut dag = Dag::with_variables(n);
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        for &to in admg.children(from) {
            dag.insert_directed(from, to)
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        }
    }
    Ok(dag)
}

/// Copy inference notes from `sources` onto `target` (deduplicated), so a mixture or
/// composed posterior still records its atoms' dependence correction / draw floor.
pub(super) fn merge_posterior_notes<'a>(
    target: &mut CausalPosterior,
    sources: impl IntoIterator<Item = &'a CausalPosterior>,
) {
    for source in sources {
        for note in &source.diagnostics.notes {
            if !target.diagnostics.notes.contains(note) {
                target.diagnostics.notes.push(Arc::clone(note));
            }
        }
    }
}

/// Result diagnostics derived from posterior inference notes.
///
/// - `estimate.bayesian.temporal.dependence_correction`: the likelihood was tempered
///   for serial dependence (R-9); lists every fitted `κ̂`.
/// - `estimate.bayesian.hmc_draw_floor`: the HMC draw floor raised the requested
///   draw count (B-5).
pub(super) fn posterior_note_diagnostics<'a>(
    posteriors: impl IntoIterator<Item = &'a CausalPosterior>,
) -> Vec<Diagnostic> {
    let mut kappas = Vec::new();
    let mut floors = Vec::new();
    let mut capped = 0usize;
    let mut inestimable = 0usize;
    let mut seen = std::collections::HashSet::new();
    for post in posteriors {
        for note in &post.diagnostics.notes {
            if !seen.insert(Arc::clone(note)) {
                continue;
            }
            let single = std::slice::from_ref(note);
            if let Some(kappa) = antecedent_estimate::tempering_kappa_from_notes(single) {
                kappas.push(kappa);
            }
            if antecedent_estimate::tempering_capped_from_notes(single) {
                capped += 1;
            }
            if antecedent_estimate::tempering_inestimable_from_notes(single) {
                inestimable += 1;
            }
            if let Some(floor) = antecedent_estimate::hmc_draw_floor_from_notes(single) {
                floors.push(floor);
            }
        }
    }
    let mut out = Vec::new();
    if !kappas.is_empty() {
        let list = kappas.iter().map(|k| format!("{k:.3}")).collect::<Vec<_>>().join(", ");
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.dependence_correction",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "generalized (power) posterior with a serial-dependence correction: each \
                 Gaussian likelihood on time-ordered rows is tempered by 1/kappa, kappa = the \
                 larger of the autoregressive-prewhitened Newey-West long-run-variance ratio of \
                 the targeted slope score (scaled by its squared fixed-b factor) and the \
                 autoregressive-residual variance ratio given the design (AR(1), plus a \
                 BIC-selected AR(q) up to order 4), floored at 1 (kappa = [{list}] over {} \
                 fit(s)); this corrects short-memory serial dependence in the outcome residual, \
                 not long memory, heteroskedasticity or a misspecified mean",
                kappas.len()
            ),
        ));
    }
    if inestimable > 0 {
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.tempering_inestimable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "long-run-variance tempering could not be estimated on {inestimable} fit(s) \
                 (n < max(8, p+2)); the published credible interval is the iid posterior and \
                 is likely too narrow"
            ),
        ));
    }
    if capped > 0 {
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.tempering_capped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "long-run-variance tempering hit the n/(p+2) cap on {capped} fit(s); kappa is \
                 known to be too small and the published credible interval is still too narrow"
            ),
        ));
    }
    if let Some(&(requested, used)) = floors.first() {
        out.push(Diagnostic::new(
            "estimate.bayesian.hmc_draw_floor",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            format!(
                "HMC draw floor raised n_draws from {requested} to {used} so the MCMC \
                 publication gate (R-hat <= 1.01, bulk/tail ESS >= 100 per chain) is reachable"
            ),
        ));
    }
    out
}

pub(super) fn bayesian_gcomp(
    cfg: &BayesianConfig,
    ctx: &ExecutionContext,
) -> BayesianGComputationAte {
    BayesianGComputationAte {
        backend: cfg.backend,
        likelihood: cfg.likelihood,
        n_draws: cfg.n_draws,
        seed: ctx.rng.master_seed(),
        overlap: OverlapPolicy::ExplicitOverride,
        prior_scale: cfg.prior_scale,
        prior: None,
    }
}

pub(super) fn apply_temporal_prior_sensitivity(
    cfg: &BayesianConfig,
    bprep: &antecedent_estimate::PreparedBayesianProblem,
    status: IdentificationStatus,
    posterior: &antecedent_estimate::CausalPosterior,
    ate: f64,
    ctx: &ExecutionContext,
    refutations: &mut Vec<antecedent_validate::RefutationReport>,
) -> Result<antecedent_estimate::CausalPosterior, CausalError> {
    let mut est = bayesian_temporal_gcomp(cfg, ctx);
    let mut ws = BayesianGCompWorkspace::default();
    // Same resolution (and conflict shrink) as the fit, so the grid perturbs the
    // prior in force rather than a fresh isotropic one.
    est.inner.prior = resolve_bayesian_prior_with_conflict(cfg, bprep, Some(ctx))?.0;
    let (summary, sens) = evaluate_bayesian_prior_sensitivity(
        cfg, &est.inner, bprep, status, posterior, &mut ws, ctx,
    )?;
    refutations.push(sens.to_report(&summary, ate));
    Ok(with_prior_sensitivity(posterior.clone(), summary))
}

pub(super) fn bayesian_temporal_gcomp(
    cfg: &BayesianConfig,
    ctx: &ExecutionContext,
) -> BayesianTemporalGcomp {
    BayesianTemporalGcomp {
        inner: BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
            prior: None,
        },
    }
}

fn push_unique_diagnostic(
    diagnostics: &mut Vec<Diagnostic>,
    seen: &mut std::collections::HashSet<Arc<str>>,
    diagnostic: Diagnostic,
) {
    if seen.insert(Arc::clone(&diagnostic.code)) {
        diagnostics.push(diagnostic);
    }
}

fn push_aipw_score_kind(
    diagnostics: &mut Vec<Diagnostic>,
    seen: &mut std::collections::HashSet<Arc<str>>,
    estimator_id: EstimatorId,
    estimate: &EffectEstimate,
) {
    if seen.contains("estimate.aipw.crossfit_scores")
        || seen.contains("estimate.aipw.full_sample_residualized")
    {
        return;
    }
    match estimator_id {
        EstimatorId::CellAipw | EstimatorId::Aipw if estimate.score_table.is_some() => {
            push_unique_diagnostic(
                diagnostics,
                seen,
                Diagnostic::new(
                    "estimate.aipw.crossfit_scores",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "cross-fitted AIPW scores φᵢ^a; retarget averages this table. A residualized full-sample AIPW fit is a different object",
                ),
            );
        }
        EstimatorId::Aipw => {
            push_unique_diagnostic(
                diagnostics,
                seen,
                Diagnostic::new(
                    "estimate.aipw.full_sample_residualized",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "this AIPW fit is full-sample residualized and has no score table; it is not the cross-fitted φ family that retarget averages. Prepare an AllObserved iid AIPW plan to retarget",
                ),
            );
        }
        _ => {}
    }
}

fn push_grid_scalar_cleared(
    diagnostics: &mut Vec<Diagnostic>,
    seen: &mut std::collections::HashSet<Arc<str>>,
    estimate: &EffectEstimate,
) {
    if let Some(inf) = estimate.score_inference.as_ref() {
        push_unique_diagnostic(
            diagnostics,
            seen,
            Diagnostic::new(
                "estimate.functional.cdf_inference",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "per-arm F_a(c) simultaneous bands describe raw CDF coordinates; rearranged exceedance_cdf values are not mixed with those intervals",
            ),
        );
        if inf.threshold_supported.iter().any(|ok| !ok) {
            push_unique_diagnostic(
                diagnostics,
                seen,
                Diagnostic::new(
                    "estimate.functional.threshold_tail.unsupported",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "at least one threshold tail lacks enough treated/control events for a tail probability; that coordinate's band is non-finite rather than an empty-cell or first-threshold SE",
                ),
            );
        }
    } else if estimate.score_table.is_none() && estimate.exceedance_cdf.is_some() {
        push_unique_diagnostic(
            diagnostics,
            seen,
            Diagnostic::new(
                "estimate.functional.cdf_inference.unavailable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "conditional CDF values have no per-arm simultaneous bands or threshold tail-support evidence; joint covariance, when present, describes raw threshold contrasts, not the projected per-arm CDF",
            ),
        );
    }
    if estimate.exceedance_cdf.as_ref().is_some_and(|cdf| cdf.len() > 2)
        && !estimate.ate.is_finite()
    {
        push_unique_diagnostic(
            diagnostics,
            seen,
            Diagnostic::new(
                "estimate.functional.grid_scalar_cleared",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
            ),
        );
    }
}

impl super::Study {
    pub(super) fn require_execute_dag(&self, message: &'static str) -> Result<&Dag, CausalError> {
        self.graph.as_dag().ok_or(CausalError::Unsupported { message })
    }

    pub(super) fn attach_certificate(
        &self,
        mut result: StudyResult,
        identification: crate::Identification,
    ) -> StudyResult {
        result.certificate = Some(crate::AnalysisIdentification {
            identification,
            query: self.query.clone(),
            graph_class: self.graph.class(),
        });
        result
    }

    pub(super) fn finish_identified_execute(
        &self,
        args: IdentifiedExecuteFinish<'_>,
    ) -> StudyResult {
        let extras = args.extras;
        let certificate = extras.certificate.or_else(|| {
            (self.graph_posterior.is_none()
                && (self.graph.as_dag().is_some() || self.graph.as_admg().is_some()))
            .then(|| crate::Identification::Point {
                result: args.identification.clone(),
                temporal_indexer: None,
                strategy: args.identifier_id,
                structure_version: self.graph.version(),
            })
        });
        let mut diagnostics = if let Some(prebuilt) = extras.diagnostics {
            prebuilt
        } else {
            let mut diagnostics = args.identification.diagnostics.clone();
            diagnostics.push(overlap_diagnostic(args.estimate.overlap));
            diagnostics.extend(args.extra_diagnostics);
            diagnostics
        };
        let mut seen: std::collections::HashSet<Arc<str>> =
            diagnostics.iter().map(|d| Arc::clone(&d.code)).collect();
        // Envelope routes supply their own diagnostic seed, but a prepared
        // envelope still must expose cache reuse just like a single-graph path.
        if args.identify_cached {
            push_unique_diagnostic(&mut diagnostics, &mut seen, identify_cached_diagnostic());
        }
        push_aipw_score_kind(&mut diagnostics, &mut seen, args.estimator_id, &args.estimate);
        push_grid_scalar_cleared(&mut diagnostics, &mut seen, &args.estimate);
        let structural_posteriors = extras
            .structural_response
            .iter()
            .flat_map(|mixture| mixture.atoms.iter().filter_map(|atom| atom.posterior.as_ref()));
        for diagnostic in
            posterior_note_diagnostics(extras.posterior.iter().chain(structural_posteriors))
        {
            push_unique_diagnostic(&mut diagnostics, &mut seen, diagnostic);
        }
        if let (Some(mode), Some(n)) = (self.latency_mode, extras.n_draws) {
            let tier = match mode {
                crate::analysis::latency::LatencyMode::Interactive => {
                    crate::analysis::latency::INTERACTIVE_N_DRAWS
                }
                crate::analysis::latency::LatencyMode::Standard => {
                    crate::analysis::latency::STANDARD_N_DRAWS
                }
                crate::analysis::latency::LatencyMode::Report => {
                    crate::analysis::latency::REPORT_N_DRAWS
                }
            };
            if usize::try_from(n).ok() != Some(tier) {
                diagnostics.push(Diagnostic::new(
                    "latency.explicit_budget_kept",
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Info,
                    format!("explicit n_draws={n} kept over {} tier default {tier}", mode.as_str()),
                ));
            }
        }
        let (id_artifact, id_op) = extras.identify_provenance.unwrap_or_else(|| {
            let (a, b) = identify_provenance_step(args.identifier_id);
            provenance_ids(a, b)
        });
        let (est_artifact, est_op) = extras.estimate_provenance.unwrap_or_else(|| {
            let (a, b) = estimate_provenance_step(args.estimator_id);
            provenance_ids(a, b)
        });
        let provenance = if extras.empty_provenance {
            ProvenanceGraph::new()
        } else {
            provenance_pair(
                (
                    id_artifact.as_ref(),
                    id_op.as_ref(),
                    &[],
                    &args.identification.required_assumptions,
                ),
                (
                    est_artifact.as_ref(),
                    est_op.as_ref(),
                    &[id_artifact.as_ref()],
                    &args.estimate.assumptions,
                ),
            )
        };
        let physical_record =
            self.apply_callback_plan_marks(args.physical.record.clone(), &mut diagnostics);
        let (counterfactual, anomaly, change_attribution, mechanism_change, unit_change) =
            match extras.gcm {
                Some(GcmSlot::Counterfactual(v)) => (Some(v), None, None, None, None),
                Some(GcmSlot::Anomaly(v)) => (None, Some(v), None, None, None),
                Some(GcmSlot::Change(v)) => (None, None, Some(v), None, None),
                Some(GcmSlot::Mechanism(v)) => (None, None, None, Some(v), None),
                Some(GcmSlot::Unit(v)) => (None, None, None, None, Some(v)),
                None => (None, None, None, None, None),
            };
        let mut result = assemble_result(AssembleArgs {
            logical: &args.physical.logical.record,
            physical: &physical_record,
            identification: args.identification,
            estimand: args.estimand,
            estimate: args.estimate,
            distribution: args.distribution,
            posterior: extras.posterior,
            mediation: args.mediation,
            mediation_grid: extras.mediation_grid,
            counterfactual,
            anomaly,
            change_attribution,
            mechanism_change,
            unit_change,
            refutations: args.refutations,
            diagnostics,
            provenance,
            treatment: args.treatment,
            outcome: args.outcome,
            wall_time_ns: args.wall_time_ns,
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: extras.stage_timings_ns,
            bootstrap_replicates_requested: extras
                .bootstrap_replicates_requested
                .unwrap_or(Some(self.bootstrap_replicates)),
            bootstrap_replicates_ok: args.bootstrap_replicates_ok,
            n_draws: extras.n_draws,
            cancelled: args.cancelled,
            early_stopped: args.early_stopped,
            bayesian: matches!(self.inference, InferenceMode::Bayesian(_)),
        });
        result.certificate = certificate.map(|identification| crate::AnalysisIdentification {
            identification,
            query: self.query.clone(),
            graph_class: self.graph.class(),
        });
        let is_quantile = match &self.query {
            CausalQuery::AverageEffect(q) => q.outcome_functional.quantile_level().is_some(),
            CausalQuery::ConditionalEffect(q) => {
                q.inner.outcome_functional.quantile_level().is_some()
            }
            CausalQuery::Response(q) => q.outcome_functional.quantile_level().is_some(),
            _ => false,
        };
        if is_quantile {
            result.estimate.evalue = None;
            result.diagnostics.push(super::helpers::quantile_scope_diagnostic());
            if matches!(self.query, CausalQuery::ConditionalEffect(_)) {
                result.diagnostics.push(Diagnostic::new("estimate.functional.conditional_quantile",
                    DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    "quantile contrast inverts arm CDFs standardized over the retained modifier distribution; this is not a pointwise conditional-quantile surface or an average of individual quantile effects"));
            }
        }
        result.predictive_checks = extras.predictive_checks;
        result.response = extras.response;
        result.structural_response = extras.structural_response;
        result.rebind_interval(matches!(self.inference, InferenceMode::Bayesian(_)));
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
        // A named predicate or custom distribution is a handle; encoding the
        // executed query (certificates, artifacts) needs its bindings.
        result.population_registry.clone_from(&self.population_registry);
        if !is_quantile {
            super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
        }
        if !is_quantile
            && self
                .tiered
                .as_ref()
                .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined)
        {
            if let DataInput::Tabular(data) = &self.data {
                super::helpers::attach_tiered_evalue(
                    &mut result.estimate,
                    data,
                    args.outcome,
                    &mut result.diagnostics,
                );
            }
        }
        if let Some(crate::support::CellStatus::Allowlisted { reason, parent }) =
            self.support_status
        {
            result.diagnostics.push(Diagnostic {
                code: Arc::from("support.allowed_unlicensed"),
                kind: DiagnosticKind::Scientific,
                severity: DiagnosticSeverity::Warning,
                message: Arc::from(
                    "this estimate executed an allowlisted cell; it is not a licensed claim",
                ),
                artifact_id: None,
                fields: Arc::from([
                    (Arc::from("reason"), Arc::from(reason)),
                    (Arc::from("parent"), Arc::from(parent)),
                ]),
            });
        }
        if let Some(requested) = self.refute_default_downgrade {
            let requested_id = requested.validation_suite_id().unwrap_or("none");
            result.diagnostics.push(Diagnostic {
                code: Arc::from("exec.refute.default_suite_unsupported"),
                kind: DiagnosticKind::Scientific,
                severity: DiagnosticSeverity::Info,
                message: Arc::from(format!(
                    "no .refute(..) was set; the default validation suite \
                     ({requested_id}) is not supported for this cell, so validation was \
                     silently downgraded to none (no refuters ran)"
                )),
                artifact_id: None,
                fields: Arc::from([
                    (Arc::from("requested_suite"), Arc::from(requested_id)),
                    (Arc::from("applied_suite"), Arc::from("none")),
                ]),
            });
        }
        result
    }
}

/// Match the estimator failure policy without counting unattempted, cancelled
/// replicates as fitting failures, and require enough successes to earn a
/// nominal 0.95 band under the usual (B+1) order-statistic floor.
///
/// Cancellation that stops after two successes must not publish a 0.95 pointwise
/// or simultaneous band: `bootstrap_has_enough_successes(2, attempted)` is false
/// whenever `attempted > 2`, and also when `completed` is below
/// [`PERCENTILE_95_BAND_MIN_SUCCESSES`]. Adaptive early-stop that actually reaches
/// that floor with a majority of successes still passes.
pub(super) fn bootstrap_has_enough_successes(completed: usize, attempted: usize) -> bool {
    completed >= PERCENTILE_95_BAND_MIN_SUCCESSES
        && completed >= attempted.saturating_sub(completed)
}

/// Fewest successful replicates that may license a facade-published nominal 0.95
/// pointwise / simultaneous band. Same (B+1)·α/2 > 1 floor as the statistical
/// transport percentile licence (`α = 0.05` ⇒ B ≥ 40).
pub(super) const PERCENTILE_95_BAND_MIN_SUCCESSES: usize = 40;

#[cfg(test)]
mod bootstrap_success_floor_tests {
    use super::{PERCENTILE_95_BAND_MIN_SUCCESSES, bootstrap_has_enough_successes};

    #[test]
    fn cancelled_two_success_bootstrap_cannot_license_nominal_band() {
        assert!(!bootstrap_has_enough_successes(2, 2));
        assert!(!bootstrap_has_enough_successes(2, 3));
        assert!(!bootstrap_has_enough_successes(2, 199));
        assert!(!bootstrap_has_enough_successes(2, PERCENTILE_95_BAND_MIN_SUCCESSES));
    }

    #[test]
    fn completed_budget_at_earned_minimum_licenses_band() {
        let min = PERCENTILE_95_BAND_MIN_SUCCESSES;
        assert!(bootstrap_has_enough_successes(min, min));
        assert!(!bootstrap_has_enough_successes(min - 1, min - 1));
        // Majority-failure policy still binds above the floor.
        assert!(!bootstrap_has_enough_successes(min, min * 2 + 1));
        assert!(bootstrap_has_enough_successes(min, min * 2));
    }
}

// Aggregation consumes only effect draws. Keep each contributing model's
// prior/estimation restrictions on the resulting mixture as well.
pub(super) fn retain_envelope_assumptions(
    posterior: &mut CausalPosterior,
    atoms: &[EnvelopeAtomFit],
) {
    for atom in atoms {
        posterior.assumptions.entries.extend(
            atom.posterior.assumptions.entries.iter().cloned().map(|mut record| {
                match &mut record.assumption {
                    antecedent_core::Assumption::PriorRestriction(prior) => {
                        prior.description =
                            Arc::from(format!("graph atom {}: {}", atom.key, prior.description));
                    }
                    antecedent_core::Assumption::ParametricRestriction(model) => {
                        model.description =
                            Arc::from(format!("graph atom {}: {}", atom.key, model.description));
                    }
                    _ => {}
                }
                record
            }),
        );
    }
}
