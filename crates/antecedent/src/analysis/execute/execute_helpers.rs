// SPDX-License-Identifier: MIT OR Apache-2.0
// Free functions supporting Study execute paths.

use antecedent_core::StreamDomain;

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
            let outcome = first_target(&q.targets, "anomaly attribution")?;
            Ok((outcome, outcome))
        }
        CausalQuery::ChangeAttribution(q) => Ok((q.outcome, q.outcome)),
        CausalQuery::MechanismChange(q) => {
            let outcome = first_target(&q.targets, "mechanism change")?;
            Ok((outcome, outcome))
        }
        CausalQuery::UnitChange(q) => Ok((q.outcome, q.outcome)),
        _ => Err(CausalError::Compile { message: "gcm_query_vars: unsupported query".into() }),
    }
}

/// Whether a graph-posterior atom is a CPDAG or PAG equivalence class (as opposed to a single
/// DAG or ADMG), which routes through the class-envelope executors.
pub(crate) const fn is_class_atom(kind: antecedent_discovery::GraphPosteriorAtomKind) -> bool {
    matches!(
        kind,
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag
    )
}

/// The first declared target of a GCM query. An empty target list is refused rather than read
/// as variable 0, which would silently pick the first schema variable.
fn first_target(
    targets: &[VariableId],
    query_kind: &'static str,
) -> Result<VariableId, CausalError> {
    targets.first().copied().ok_or_else(|| CausalError::Compile {
        message: format!("{query_kind} query declares no target variable"),
    })
}

/// Named execute / compile cell. Graph-completion arms still specialize on class.
#[derive(Clone, Copy)]
pub(crate) enum AnalysisRoute {
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
    TemporalCrossWorld,
    PanelTemporalEffect,
    PanelTemporalResponse,
    MultiEnvTemporalEffect,
    Transport,
    Interference,
    RandomizedEffect,
    PolicyValue,
    ContinuousDoseResponse,
    PanelDid,
    SyntheticControl,
    LocalPolynomialRatio,
    Survival,
    LongitudinalRegime,
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

#[allow(
    clippy::match_same_arms,
    reason = "one arm per (modality, query kind) pair so the routing table reads pair by pair"
)]
pub(super) fn classify_route(modality: DataModality, query: &CausalQuery) -> Option<AnalysisRoute> {
    Some(match (modality, query) {
        (DataModality::Tabular, CausalQuery::Response(q)) if q.is_temporal() => return None,
        (DataModality::Tabular, CausalQuery::Response(_)) => AnalysisRoute::Response,
        (DataModality::TemporalOrEvent, CausalQuery::Response(q)) if q.is_temporal() => {
            AnalysisRoute::TemporalResponse
        }
        (DataModality::Tabular, CausalQuery::AverageEffect(_)) => AnalysisRoute::StaticAte,
        (DataModality::Tabular, CausalQuery::Distribution(_)) => AnalysisRoute::Distribution,
        (DataModality::TemporalOrEvent, CausalQuery::Distribution(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::Tabular, CausalQuery::PathSpecific(_)) => AnalysisRoute::PathSpecific,
        (DataModality::TemporalOrEvent, CausalQuery::PathSpecific(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::TemporalOrEvent, CausalQuery::NestedCounterfactual(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::TemporalOrEvent, CausalQuery::Counterfactual(_)) => {
            AnalysisRoute::TemporalCrossWorld
        }
        (DataModality::Tabular, CausalQuery::ConditionalEffect(_)) => AnalysisRoute::Conditional,
        (DataModality::TemporalOrEvent, CausalQuery::Mediation(_)) => {
            AnalysisRoute::TemporalMediation
        }
        (DataModality::Tabular, CausalQuery::Mediation(_)) => AnalysisRoute::StaticMediation,
        (DataModality::Tabular, CausalQuery::NestedCounterfactual(_)) => {
            AnalysisRoute::StaticMediation
        }
        (DataModality::Tabular, CausalQuery::Counterfactual(_)) => AnalysisRoute::Counterfactual,
        (DataModality::Tabular, CausalQuery::AnomalyAttribution(_)) => AnalysisRoute::Anomaly,
        (DataModality::Tabular, CausalQuery::ChangeAttribution(_)) => {
            AnalysisRoute::ChangeAttribution
        }
        (DataModality::TemporalOrEvent, CausalQuery::AnomalyAttribution(_)) => {
            AnalysisRoute::Anomaly
        }
        (DataModality::TemporalOrEvent, CausalQuery::ChangeAttribution(_)) => {
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
        (DataModality::Tabular, CausalQuery::RandomizedEffect(_)) => {
            AnalysisRoute::RandomizedEffect
        }
        (DataModality::Tabular, CausalQuery::PolicyValue(_)) => AnalysisRoute::PolicyValue,
        (DataModality::Tabular, CausalQuery::ContinuousDoseResponse(_)) => {
            AnalysisRoute::ContinuousDoseResponse
        }
        (DataModality::Tabular, CausalQuery::PanelDid(_)) => AnalysisRoute::PanelDid,
        (DataModality::Tabular, CausalQuery::SyntheticControl(_)) => {
            AnalysisRoute::SyntheticControl
        }
        (DataModality::Tabular, CausalQuery::LocalPolynomialRatio(_)) => {
            AnalysisRoute::LocalPolynomialRatio
        }
        (DataModality::Tabular, CausalQuery::Survival(_)) => AnalysisRoute::Survival,
        (DataModality::Tabular, CausalQuery::LongitudinalRegime(_)) => {
            AnalysisRoute::LongitudinalRegime
        }
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

impl GcmSlot {
    fn into_payload(self) -> crate::result::ResultPayload {
        use crate::result::ResultPayload;
        match self {
            Self::Counterfactual(value) => ResultPayload::Counterfactual(value),
            Self::Anomaly(value) => ResultPayload::Anomaly(value),
            Self::Change(value) => ResultPayload::ChangeAttribution(value),
            Self::Mechanism(value) => ResultPayload::MechanismChange(value),
            Self::Unit(value) => ResultPayload::UnitChange(value),
        }
    }
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

/// The identifier and estimator a route executed with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Strategy {
    pub(super) identifier: IdentifierId,
    pub(super) estimator: EstimatorId,
}

impl Strategy {
    pub(crate) const fn new(identifier: IdentifierId, estimator: EstimatorId) -> Self {
        Self { identifier, estimator }
    }
}

/// The execution frame every finish of one execute path shares.
#[derive(Clone, Copy)]
pub(super) struct ExecuteRun<'a> {
    pub(super) physical: &'a PhysicalExecutionPlan,
    pub(super) ctx: &'a ExecutionContext,
    pub(super) started: Instant,
    /// Whether identification was served from the prepare-time cache.
    pub(super) identify_cached: bool,
}

/// The identified class envelope a CPDAG/PAG execute path finishes from.
pub(super) struct ClassEnvelopeRun<'a, G> {
    pub(super) run: ExecuteRun<'a>,
    pub(super) envelope: &'a IdentificationEnvelope<G>,
    pub(super) envelope_diagnostic: Diagnostic,
}

/// An identification result together with the estimand selected from it.
pub(super) struct IdentifiedClaim {
    pub(super) identification: IdentificationResult,
    pub(super) estimand: IdentifiedEstimand,
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

impl<'a> IdentifiedExecuteFinish<'a> {
    /// A finish for a freshly computed, unrefuted point result; the setters
    /// below override the slots a route fills.
    pub(super) fn new(
        physical: &'a PhysicalExecutionPlan,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        estimate: EffectEstimate,
        strategy: Strategy,
        treatment: VariableId,
        outcome: VariableId,
    ) -> Self {
        Self {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: strategy.identifier,
            estimator_id: strategy.estimator,
            treatment,
            outcome,
            identify_cached: false,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: 0,
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras::default(),
        }
    }

    pub(super) fn identify_cached(mut self, cached: bool) -> Self {
        self.identify_cached = cached;
        self
    }

    pub(super) fn extra_diagnostics(mut self, diagnostics: Vec<Diagnostic>) -> Self {
        self.extra_diagnostics = diagnostics;
        self
    }

    pub(super) fn refutations(
        mut self,
        refutations: Vec<antecedent_validate::RefutationReport>,
    ) -> Self {
        self.refutations = refutations;
        self
    }

    pub(super) fn distribution(
        mut self,
        distribution: Option<antecedent_estimate::InterventionalDistributionEstimate>,
    ) -> Self {
        self.distribution = distribution;
        self
    }

    pub(super) fn mediation(
        mut self,
        mediation: Option<antecedent_estimate::TemporalMediationEstimate>,
    ) -> Self {
        self.mediation = mediation;
        self
    }

    pub(super) fn wall_time_ns(mut self, wall_time_ns: u64) -> Self {
        self.wall_time_ns = wall_time_ns;
        self
    }

    pub(super) fn bootstrap_replicates_ok(mut self, replicates_ok: Option<u32>) -> Self {
        self.bootstrap_replicates_ok = replicates_ok;
        self
    }

    pub(super) fn cancelled(mut self, cancelled: bool) -> Self {
        self.cancelled = cancelled;
        self
    }

    pub(super) fn early_stopped(mut self, early_stopped: bool) -> Self {
        self.early_stopped = early_stopped;
        self
    }

    pub(super) fn extras(mut self, extras: IdentifiedExecuteExtras) -> Self {
        self.extras = extras;
        self
    }
}

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
    Ok((Arc::new(crate::analysis::prepared::CachedStaticIdentification { identification, estimand }), false))
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
    let sub = graphs
        .stratified_interactive_subsample(INTERACTIVE_MAX_ENVELOPE_GRAPHS, &mut rng)
        ?;
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
    if config.has_transfer()
    {
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
        payload: finish
            .distribution
            .map_or(crate::result::ResultPayload::Scalar, crate::result::ResultPayload::Distribution),
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
        run_functional(&est, &prepared, &self.inference, identification.status, ctx)
    }
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

/// The declared linear-SEM restriction every sequential g-computation fit
/// carries; `description` names the SEM family the route fitted.
pub(super) fn sequential_linear_sem_assumption(
    description: &str,
) -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("temporal.sequential.linear_sem"),
                description: Arc::from(description),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("temporal.sequential.gcomp"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// Note that a bidirected completion carries no sequential g-computation.
pub(super) fn mag_sequential_unevaluable_diagnostic(key: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal_class.mag_sequential_unevaluable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!("completion {key} has bidirected edges; sequential g-comp is unevaluable"),
    )
}

/// Frequentist mixture estimate of a frozen-weight class envelope from its
/// weighted `point`, the mixed analytic SE and the shared circular-block draw.
pub(super) fn class_mixture_estimate(
    point: f64,
    se_analytic: f64,
    block: &class_envelope_se::SharedCircularBlockSe,
    bootstrap_replicates: u32,
    assumptions: antecedent_core::AssumptionSet,
    ctx: &ExecutionContext,
) -> EffectEstimate {
    EffectEstimate::from_parts(
        point,
        se_analytic,
        BootstrapAccounting {
            se: block.se.is_finite().then_some(block.se),
            replicates_ok: (bootstrap_replicates > 0).then_some(block.completed),
            replicates_failed: (bootstrap_replicates > 0)
                .then_some(block.attempted.saturating_sub(block.completed)),
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
        },
        assumptions,
        OverlapPolicy::ExplicitOverride,
        None,
        None,
    )
    .with_block_family(antecedent_estimate::CircularBlockFamily::Mixture)
}

/// Shared circular-block mixture of the evaluable sequential atoms of a class
/// envelope, with the atoms' point estimates and mixed analytic SE.
struct SequentialAtomBlock {
    block: class_envelope_se::SharedCircularBlockSe,
    identified_set_interval: Option<antecedent_estimate::IdentifiedSetInterval>,
    se_analytic: f64,
}

fn sequential_atom_block(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    atoms: &[sequential_validation::SequentialValidationAtom],
    bootstrap_replicates: u32,
    truncated: bool,
    ctx: &ExecutionContext,
) -> Result<SequentialAtomBlock, CausalError> {
    let designs = atoms
        .iter()
        .map(|atom| {
            TemporalAtomDesign::sequential(
                data,
                &atom.graph,
                &atom.indexer,
                &atom.estimand,
                query,
                atom.status,
                ctx,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = atoms.iter().map(|atom| atom.weight).collect::<Vec<_>>();
    let block = shared_circular_block_mixture_se(
        &designs.iter().collect::<Vec<_>>(),
        &weights,
        temporal_class_block_span(atoms.iter().map(|atom| &atom.indexer)),
        bootstrap_replicates,
        0x5E0C_1A55,
        ctx,
    );
    let points = atoms.iter().map(|atom| atom.estimate.ate).collect::<Vec<_>>();
    let identified_set_interval = block
        .identified_set_interval(&points, IDENTIFIED_SET_INTERVAL_LEVEL)
        .map(|interval| interval.with_truncated(truncated));
    let se_analytic = mix_weighted_analytic_se(
        atoms.iter().map(|atom| (atom.weight, atom.estimate.se_analytic())),
    );
    Ok(SequentialAtomBlock { block, identified_set_interval, se_analytic })
}

/// Linear-adjustment atoms of a frequentist temporal class envelope, fitted in
/// case order. The totals are not checked: callers own their refusal text.
pub(super) struct ClassLinearAtoms<'a> {
    pub values: Vec<Option<f64>>,
    pub designs: Vec<TemporalAtomDesign>,
    pub weights: Vec<f64>,
    pub indexers: Vec<&'a TemporalIndexer>,
    pub points: Vec<f64>,
    pub se_items: Vec<(f64, f64)>,
    pub refute_atoms: Vec<EnvelopeRefuteAtom>,
    pub primary_estimand: Option<IdentifiedEstimand>,
    pub assumptions: antecedent_core::AssumptionSet,
    pub weighted_point: f64,
    pub total_weight: f64,
}

pub(super) fn fit_class_linear_atoms<'a>(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    envelope: &antecedent_identify::TemporalClassEnvelope,
    indexers: &'a [TemporalIndexer],
    split: Option<&antecedent_data::DiscoveryEstimationSplit>,
    ctx: &ExecutionContext,
) -> Result<ClassLinearAtoms<'a>, CausalError> {
    let mut fit = ClassLinearAtoms {
        values: vec![None; envelope.envelope.cases.len()],
        designs: Vec::new(),
        weights: Vec::new(),
        indexers: Vec::new(),
        points: Vec::new(),
        se_items: Vec::new(),
        refute_atoms: Vec::new(),
        primary_estimand: None,
        assumptions: antecedent_core::AssumptionSet::default(),
        weighted_point: 0.0,
        total_weight: 0.0,
    };
    for (index, (case, indexer)) in envelope.envelope.cases.iter().zip(indexers).enumerate() {
        if !identification_status_ok_for_case(case.result.status) || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let design = TemporalAtomDesign::linear(data, &estimand, query, indexer, split, ctx)?;
        let estimate = design.effect_estimate(case.result.required_assumptions.clone());
        let weight = case.weight.0;
        fit.weighted_point += weight * estimate.ate;
        fit.total_weight += weight;
        fit.values[index] = Some(estimate.ate);
        fit.se_items.push((weight, estimate.se_analytic()));
        if fit.primary_estimand.is_none() {
            fit.primary_estimand = Some(estimand.clone());
            fit.assumptions = estimate.provenance.assumptions.clone();
        }
        fit.points.push(estimate.ate);
        fit.refute_atoms.push(EnvelopeRefuteAtom {
            key: index as u64,
            weight,
            estimand,
            indexer: Some(indexer.clone()),
            original: estimate,
        });
        fit.designs.push(design);
        fit.weights.push(weight);
        fit.indexers.push(indexer);
    }
    Ok(fit)
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

/// Mix `ConditionalEffect` exceedance grids across envelope atoms.
///
/// Per threshold, each atom is estimated on the same indicator transform, IFs
/// are aligned to the original row universe, and `F_a(c)` / scores are mixed
/// by frozen completion weight — the same rule as class-aware mean CATEs.
#[allow(clippy::too_many_lines, reason = "for each threshold, refits every identified completion on the exceedance outcome and mixes arm CDFs, influences and supports by frozen completion weight into the same output vectors")]
pub(super) fn attach_class_conditional_functional_grid(
    estimate: EffectEstimate,
    data: &TabularData,
    query: &antecedent_core::ConditionalEffectQuery,
    atoms: &[(f64, IdentifiedEstimand)],
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
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
    let est = ConditionalLinearAdjustment::new().with_fold_seed(ctx.rng.master_seed());
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
    out.joint.covariance = Some(antecedent_estimate::joint_influence_covariance(&refs, None)?);
    let n_eff = n_eff_by_arm[0] + n_eff_by_arm[1];
    out.scores.inference = Some(antecedent_estimate::inference_from_influence_columns(
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
        out.uncertainty.analytic.se =
            antecedent_estimate::joint_influence_covariance(&[&contrast], None)?.se(0);
        out.scores.influence = Some(contrast.into());
        out.uncertainty.bootstrap.se = None;
        out.family.simultaneous_interval = None;
        if threshold_supported.iter().any(|&supported| !supported) {
            out.uncertainty.analytic.se = f64::NAN;
            out.scores.influence = None;
        }
    }
    let raw_cdf = mixed_cdf.clone();
    let rearranged = super::helpers::project_conditional_cdf(&mut mixed_cdf)?;
    out = out.with_monotone_rearranged(rearranged);
    out.joint.exceedance_cdf = Some(Arc::from(mixed_cdf));
    if thresholds.len() > 1 {
        out.ate = f64::NAN;
        out.uncertainty.analytic.se = f64::NAN;
        out.uncertainty.bootstrap.se = None;
        out.scores.influence = None;
        out.family.simultaneous_interval = None;
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
pub(super) fn mix_static_envelope_se<T: AsRef<[f64]>>(atom_ifs: &[T], atom_weights: &[f64]) -> f64 {
    if atom_ifs.is_empty() || atom_ifs.len() != atom_weights.len() {
        return f64::NAN;
    }
    let refs: Vec<&[f64]> = atom_ifs.iter().map(AsRef::as_ref).collect();
    let Ok(mixed) = antecedent_estimate::frozen_weight_mixture_scores(&refs, atom_weights) else {
        return f64::NAN;
    };
    mixture_se(&mixed)
}

/// Standard error of an already-mixed influence column.
pub(super) fn mixture_se(mixed: &[f64]) -> f64 {
    antecedent_estimate::joint_influence_covariance(&[mixed], None)
        .map(|c| c.se(0))
        .unwrap_or(f64::NAN)
}

/// Frozen-weight mixture IF, or `None` when atoms do not share a row universe.
pub(super) fn mixed_static_influence<T: AsRef<[f64]>>(
    atom_ifs: &[T],
    atom_weights: &[f64],
) -> Option<std::sync::Arc<[f64]>> {
    if atom_ifs.is_empty() || atom_ifs.len() != atom_weights.len() {
        return None;
    }
    let refs: Vec<&[f64]> = atom_ifs.iter().map(AsRef::as_ref).collect();
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
    use crate::result::{
        StructuralAggregationPolicy, StructuralResponseAtom, StructuralWeightBasis,
    };
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
    let mut present: std::collections::HashSet<u64> =
        structural_atoms.iter().map(|atom| atom.graph_key).collect();
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
        if !present.insert(key) {
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
    let conditional_on_identified =
        mixable_scalar.then_some(antecedent_core::ResponseValue::Scalar(ate));
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

/// How a functional estimate resolved variables its functional keeps free outside the
/// query (a complete ID derivation can leave one, as on the napkin graph).
pub(super) fn free_variables_averaged(
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
) -> Option<Diagnostic> {
    let bound: Vec<VariableId> = match &identification.query {
        CausalQuery::Distribution(query) => query
            .outcomes
            .iter()
            .copied()
            .chain(
                query
                    .interventions
                    .iter()
                    .filter_map(antecedent_core::Intervention::primary_variable),
            )
            .chain(query.conditioning.iter().copied())
            .collect(),
        _ => Vec::new(),
    };
    let free = antecedent_estimate::functional_free_variables(
        &identification.arena,
        estimand.functional,
        &bound,
    );
    antecedent_estimate::free_variables_diagnostic(&free)
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

#[derive(Clone, Copy)]
pub(super) struct PosteriorMassSummary {
    pub unidentified: f64,
    pub unevaluable: f64,
    pub subsampled: f64,
    pub failed: f64,
    pub full_scope: bool,
    pub truncated: bool,
}

impl Default for PosteriorMassSummary {
    fn default() -> Self {
        Self {
            unidentified: 0.0,
            unevaluable: 0.0,
            subsampled: 0.0,
            failed: 0.0,
            full_scope: true,
            truncated: false,
        }
    }
}

/// One posterior downgrade ladder.
///
/// Mass above [`antecedent_core::POSTERIOR_MASS_TOLERANCE`], a truncated
/// completion search, an incomplete scope, or disagreeing estimands is
/// [`IdentificationStatus::GraphDependent`]. An identified-set policy is
/// [`IdentificationStatus::PartiallyIdentified`]. Otherwise the atom status
/// is kept. This never promotes `base` to
/// [`IdentificationStatus::NonparametricallyIdentified`].
pub(super) fn resolve_posterior_status(
    base: IdentificationStatus,
    policy: crate::result::StructuralAggregationPolicy,
    mass: PosteriorMassSummary,
) -> IdentificationStatus {
    let exceeds = |value: f64| value > antecedent_core::POSTERIOR_MASS_TOLERANCE;
    if exceeds(mass.unidentified)
        || exceeds(mass.unevaluable)
        || exceeds(mass.subsampled)
        || exceeds(mass.failed)
        || !mass.full_scope
        || mass.truncated
        || matches!(policy, crate::result::StructuralAggregationPolicy::GraphDependentAtoms)
    {
        IdentificationStatus::GraphDependent
    } else if matches!(policy, crate::result::StructuralAggregationPolicy::IdentifiedSetEnvelope) {
        IdentificationStatus::PartiallyIdentified
    } else {
        base
    }
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
    if unidentified_mass > antecedent_core::POSTERIOR_MASS_TOLERANCE {
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

pub(super) fn temporal_refuter_average(query: &TemporalEffectQuery) -> AverageEffectQuery {
    let mut average = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
    average.active = query.active.clone();
    average.control = query.control.clone();
    average.target_population = query.target_population.clone();
    average
}

pub(super) const SUSTAINED_LINEAR_SEM: &str = "linear additive mechanisms on the identified unfolded DAG; every sustained time is intervened on; frequentist intervals use a shared circular-block row bootstrap whose replicate SD carries the circular-Bartlett fixed-b correction for the block length, Bayesian intervals share each stationary mechanism posterior across time copies, fitting its unique complete observed rows once";

pub(super) const SEQUENCE_LINEAR_SEM: &str = "linear additive mechanisms on the identified unfolded DAG; each Sequence step transforms deterministic propagated means: fixed level, mean + shift, factor * mean, or clip(mean + shift, lower, upper); clipping is not integration over stochastic realizations; no last-step collapse";

pub(super) const UNFOLDED_LINEAR_SEM: &str =
    "linear additive mechanisms on the identified unfolded DAG";

pub(super) const DBN_LINEAR_SEM: &str = "linear additive mechanisms on each identified DBN atom; fixed posterior graph weights; uncertainty uses shared outer circular-block replicates";

/// Identified mass of each graph key of a frozen posterior, summed over every
/// sample that carries the key.
pub(super) struct IdentifiedKeyWeights(std::collections::HashMap<u64, f64>);

impl IdentifiedKeyWeights {
    pub(super) fn new(graphs: &WeightedGraphSamples) -> Self {
        let empty: f64 = std::iter::empty::<f64>().sum();
        let mut weights = std::collections::HashMap::new();
        for ((key, weight), flag) in
            graphs.graph_keys.iter().zip(graphs.weights.iter()).zip(graphs.identified.iter())
        {
            if *flag == GraphIdentFlag::Identified {
                *weights.entry(*key).or_insert(empty) += *weight;
            }
        }
        Self(weights)
    }

    /// Zero (the empty sum) for a key with no identified sample.
    pub(super) fn get(&self, key: u64) -> f64 {
        self.0.get(&key).copied().unwrap_or_else(|| std::iter::empty::<f64>().sum())
    }
}

/// Index of the first occurrence of each key.
pub(super) fn first_key_positions(keys: &[u64]) -> std::collections::HashMap<u64, usize> {
    let mut positions = std::collections::HashMap::with_capacity(keys.len());
    for (index, key) in keys.iter().enumerate() {
        positions.entry(*key).or_insert(index);
    }
    positions
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

/// How the per-atom refutations of a graph-posterior class are aggregated and attributed.
#[derive(Clone, Copy)]
pub(super) struct AtomRefutationPolicy<'a> {
    pub(super) aggregation: StructuralAggregationPolicy,
    pub(super) estimator: &'a str,
}

/// The posterior and the variables its atoms are rebuilt over.
#[derive(Clone, Copy)]
pub(super) struct GraphPosteriorScope<'a> {
    pub(super) posterior: &'a GraphPosterior,
    pub(super) vars: &'a [VariableId],
}

/// The status and assumptions an aggregated identification publishes.
#[derive(Clone, Copy)]
pub(super) struct IdentificationBasis<'a> {
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: &'a antecedent_core::AssumptionSet,
}

/// How a Bayesian posterior run is validated: the suite, the inference configuration it
/// was fitted under, and the estimator each atom's checks are derived from.
#[derive(Clone, Copy)]
pub(super) struct BayesianValidation<'a> {
    pub(super) suite: RefuteSuite,
    pub(super) cfg: &'a crate::inference::BayesianConfig,
    pub(super) estimator: &'a BayesianGComputationAte,
}

/// Where a validation pass appends the reports and diagnostics it produces.
pub(super) struct ValidationSink<'a> {
    pub(super) refutations: &'a mut Vec<antecedent_validate::RefutationReport>,
    pub(super) diagnostics: &'a mut Vec<Diagnostic>,
}

/// Run PPC (and, under `full`, prior-sensitivity) on identified envelope atoms.
///
/// Aggregation is mixture-weighted by graph-posterior mass. Records the method as a
/// diagnostic so `full` means the same thing on PAG / graph-posterior as on a single DAG.
pub(super) fn run_envelope_bayesian_full_validation(
    setup: BayesianValidation<'_>,
    atoms: &[EnvelopeAtomFit],
    mixture_posterior: &mut CausalPosterior,
    estimate_ate: f64,
    ctx: &ExecutionContext,
    predictive_sims: u32,
    sink: ValidationSink<'_>,
) -> Result<Vec<PredictiveCheckReport>, CausalError> {
    const PPC_ALPHA: f64 = 0.05;
    fn borrow(items: &[(f64, PredictiveCheckReport)]) -> Vec<(f64, &PredictiveCheckReport)> {
        items.iter().map(|(w, r)| (*w, r)).collect()
    }

    let BayesianValidation { suite: refute, cfg, estimator: est } = setup;
    let ValidationSink { refutations, diagnostics } = sink;

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

/// The temporal frame refuters refit through when an envelope atom carries a lag indexer.
pub(super) struct TemporalRefutation<'a> {
    pub(super) query: &'a TemporalEffectQuery,
    pub(super) split: Option<&'a DiscoveryEstimationSplit>,
    pub(super) time_index: &'a TimeIndex,
}

/// What an envelope refuter run checks and how: the data and query every atom is validated
/// against, the suite, and the temporal frame for DBN atoms.
pub(super) struct EnvelopeRefutation<'a> {
    pub(super) data: &'a TabularData,
    pub(super) query: &'a AverageEffectQuery,
    pub(super) suite: RefuteSuite,
    pub(super) estimator: &'a str,
    pub(super) custom: &'a [Arc<dyn antecedent_validate::CustomEffectValidator>],
    pub(super) temporal: Option<TemporalRefutation<'a>>,
}

impl EnvelopeRefutation<'_> {
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
pub(super) fn run(
    self,
    atoms: &[EnvelopeRefuteAtom],
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let Self { data, query, suite, estimator, custom, temporal: temporal_frame } = self;
    if matches!(suite, RefuteSuite::None) && custom.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut per_atom: Vec<(f64, Vec<antecedent_validate::RefutationReport>)> = Vec::new();
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
        let temporal = match (atom.indexer.as_ref(), temporal_frame.as_ref()) {
            (Some(indexer), Some(frame)) => Some(TemporalRefitContext {
                indexer,
                temporal_query: frame.query,
                split: frame.split,
                kernel_policy: &ctx.kernel_policy,
                time_index: Some(frame.time_index),
                panel: None,
            }),
            _ => None,
        };
        let outcomes = RefutationRun {
            data,
            estimand: &atom.estimand,
            query,
            estimate: &atom.original,
            suite,
            estimator,
            custom,
            temporal,
        }
        .outcomes(workspace, None, ctx)?;
        per_atom.push((atom.weight, ValidationSuite::reports_only(&outcomes)));
        for (validator, reason) in ValidationSuite::not_applicable_only(&outcomes) {
            na_weight
                .entry(validator)
                .and_modify(|(w, _)| *w += atom.weight)
                .or_insert((atom.weight, reason));
        }
    }
    let (reports, coverage) = mix_atom_refutation_reports(&per_atom);
    diagnostics.extend(coverage);
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
}

/// Mix per-atom refutation reports by refuter id (weights are envelope / graph-posterior mass).
///
/// A refuter that ran on only some of the contributing mass is mixed over the atoms that reported
/// it, so its `passed` says nothing about the rest. The returned diagnostics name each such
/// refuter with the mass it covered (`refute.envelope.partial_coverage`, Warning): a mixed
/// "passed" is never left to read as covering atoms the refuter did not check.
pub(super) fn mix_atom_refutation_reports(
    per_atom: &[(f64, Vec<antecedent_validate::RefutationReport>)],
) -> (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>) {
    let mut order: Vec<Arc<str>> = Vec::new();
    let mut by_refuter: std::collections::HashMap<
        Arc<str>,
        Vec<(f64, &antecedent_validate::RefutationReport)>,
    > = std::collections::HashMap::new();
    let mut total_mass = 0.0;
    for (weight, reports) in per_atom {
        if *weight <= 0.0 {
            continue;
        }
        total_mass += *weight;
        for report in reports {
            by_refuter
                .entry(Arc::clone(&report.refuter))
                .or_insert_with(|| {
                    order.push(Arc::clone(&report.refuter));
                    Vec::new()
                })
                .push((*weight, report));
        }
    }
    let mut mixed = Vec::with_capacity(order.len());
    let mut diagnostics = Vec::new();
    for id in order {
        let Some(items) = by_refuter.get(&id) else {
            continue;
        };
        let Some(report) = antecedent_validate::RefutationReport::mixture_weighted(items) else {
            continue;
        };
        let covered: f64 = items.iter().map(|(w, _)| *w).sum();
        if total_mass > 0.0 && covered / total_mass < 1.0 - 1e-12 {
            diagnostics.push(Diagnostic::new(
                "refute.envelope.partial_coverage",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "refuter `{id}` ran on {:.6} of the contributing envelope mass; the other \
                     atoms were not applicable or produced no report, so its mixed verdict does \
                     not cover them",
                    covered / total_mass
                ),
            ));
        }
        mixed.push(report);
    }
    (mixed, diagnostics)
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
    let identification =
        IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
            status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
            query,
            estimands: vec![estimand.clone()],
            arena,
            derivation: DerivationTrace::default(),
            required_assumptions: assumptions,
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        });
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
    #[allow(
        clippy::float_cmp,
        reason = "\"every completion is identified\" and \"they all agree\" are exact facts about summed masses and compared values, not measurements with a tolerance"
    )]
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
    // Every case's estimands carry a `functional: ExprId` into *that case's own*
    // `case.result.arena`, not a shared one. Each estimand copied into the combined
    // result must have its functional re-homed into the combined `arena` via
    // `CausalExprArena::import` (which re-interns the referenced subtree and returns
    // its id here) — otherwise the copied `ExprId` indexes nothing in the empty arena
    // this used to hand back, and every consumer that renders, evaluates, or encodes
    // the functional (including the wire encoder's arena-bounds check) sees a
    // dangling id.
    let mut arena = CausalExprArena::new();
    let mut estimands = Vec::new();
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut diagnostics = Vec::new();
    let mut first_identified_case_arena = None;
    for case in &envelope.cases {
        if identification_status_ok_for_case(case.result.status) {
            if first_identified_case_arena.is_none() {
                first_identified_case_arena = Some(&case.result.arena);
            }
            for est in &case.result.estimands {
                let mut est = est.clone();
                est.functional = arena.import(&case.result.arena, est.functional);
                estimands.push(est);
            }
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
            // `IdentificationEnvelope::from_cases` sets `invariant` to a clone of
            // the first identified case's first estimand, so its functional
            // belongs to that same case's arena.
            let mut inv = inv.clone();
            if let Some(source_arena) = first_identified_case_arena {
                inv.functional = arena.import(source_arena, inv.functional);
            }
            estimands.push(inv);
        }
    }
    IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
        status: envelope.status,
        query,
        estimands,
        arena,
        derivation: DerivationTrace::default(),
        required_assumptions: assumptions,
        diagnostics,
        performance: IdentificationPerformanceRecord::default(),
        hedge: None,
    })
}

pub(crate) fn admg_has_bidirected(admg: &Admg) -> bool {
    admg.has_bidirected()
}

/// Static ADMG whose general-ID functional matches this temporal ADMG.
///
/// The projection exists only when each variable appears once. Identification
/// and estimation then reuse the static ADMG path on that graph.
pub(crate) fn temporal_admg_projection(
    graph: &antecedent_graph::TemporalAdmg,
) -> Result<Admg, CausalError> {
    graph
        .project_preserving_functional()
        .map_err(CausalError::rejected)
}

/// The directed part of an ADMG as a DAG.
///
/// Only an ADMG without bidirected edges is a DAG: dropping a bidirected edge would present a
/// latent-confounded graph as causally sufficient, so it is refused here rather than left to
/// every caller's pre-check.
pub(super) fn admg_without_latents_to_dag(admg: &Admg) -> Result<Dag, CausalError> {
    if admg.has_bidirected() {
        return Err(CausalError::Unsupported {
            message: "an ADMG with bidirected edges (latent confounding) is not a DAG; \
                      dropping them would assume causal sufficiency",
        });
    }
    let n = u32::try_from(admg.node_count())
        .map_err(|_| CausalError::Compile { message: "ADMG too large".into() })?;
    let mut dag = Dag::with_variables(n);
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        for &to in admg.children(from) {
            dag.insert_directed(from, to)
                .map_err(CausalError::rejected)?;
        }
    }
    Ok(dag)
}

#[cfg(test)]
mod admg_dag_coercion_tests {
    use super::*;

    #[test]
    fn bidirected_edge_is_refused_and_directed_only_graph_converts() {
        let mut admg = Admg::with_variables(3);
        let (a, b, c) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        admg.insert_directed(a, b).unwrap();
        admg.insert_directed(b, c).unwrap();
        let dag = admg_without_latents_to_dag(&admg).unwrap();
        assert!(dag.reaches(a, c));
        admg.insert_bidirected(a, c).unwrap();
        assert!(matches!(admg_without_latents_to_dag(&admg), Err(CausalError::Unsupported { .. })));
    }
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
/// Notes are de-duplicated by text (an envelope mixture re-carries its atoms' notes), so the same
/// text raised by several fits is one note here: the messages state that a condition occurred and
/// never a count of fits, which the de-duplication cannot recover.
///
/// - `estimate.bayesian.temporal.dependence_correction`: the likelihood was tempered
///   for serial dependence; lists every fitted `κ̂`.
/// - `estimate.bayesian.hmc_draw_floor`: the HMC draw floor raised the requested
///   draw count (B-5).
pub(super) fn posterior_note_diagnostics<'a>(
    posteriors: impl IntoIterator<Item = &'a CausalPosterior>,
) -> Vec<Diagnostic> {
    let mut kappas = Vec::new();
    let mut floors = Vec::new();
    let mut capped = false;
    let mut inestimable = false;
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
                capped = true;
            }
            if antecedent_estimate::tempering_inestimable_from_notes(single) {
                inestimable = true;
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
                 BIC-selected AR(q) up to order 4), floored at 1 (kappa = [{list}], one entry \
                 per distinct fitted tempering note); this corrects short-memory serial \
                 dependence in the outcome residual, not long memory, heteroskedasticity or a \
                 misspecified mean",
            ),
        ));
    }
    if inestimable {
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.tempering_inestimable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "long-run-variance tempering could not be estimated on at least one fit \
                 (n < max(8, p+2)); the published credible interval for it is the iid posterior \
                 and is likely too narrow"
                .to_string(),
        ));
    }
    if capped {
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.tempering_capped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "long-run-variance tempering hit the n/(p+2) cap on at least one fit; kappa is \
                 known to be too small and the published credible interval is still too narrow"
                .to_string(),
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
        EstimatorId::CellAipw
        | EstimatorId::Aipw
        | EstimatorId::Dml
        | EstimatorId::DrLearner
        | EstimatorId::CausalForest
            if estimate.scores.table.is_some() =>
        {
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

pub(crate) fn push_grid_scalar_cleared(
    diagnostics: &mut Vec<Diagnostic>,
    seen: &mut std::collections::HashSet<Arc<str>>,
    estimate: &EffectEstimate,
) {
    if let Some(inf) = estimate.scores.inference.as_ref() {
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
    } else if estimate.scores.table.is_none() && estimate.joint.exceedance_cdf.is_some() {
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
    if estimate.joint.exceedance_cdf.as_ref().is_some_and(|cdf| cdf.len() > 2)
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
        let context = IdentifiedResultContext::from_study(self);
        let tabular_data = match &self.data {
            DataInput::Tabular(data) => Some(data),
            _ => None,
        };
        finish_identified_execute_with_context(&context, tabular_data, args)
    }
}

#[allow(clippy::too_many_lines, reason = "assembles one StudyResult from the finish record: diagnostics are pushed in a fixed order against one seen-code set, then provenance, estimate and certificate are read from the same record")]
fn finish_identified_execute_with_context(
    context: &IdentifiedResultContext,
    tabular_data: Option<&TabularData>,
    args: IdentifiedExecuteFinish<'_>,
) -> StudyResult {
    let extras = args.extras;
    let certificate = extras.certificate.or_else(|| {
        context.graph_has_certificate.then(|| crate::Identification::Point {
            result: args.identification.clone(),
            temporal_indexer: None,
            strategy: args.identifier_id,
            structure_version: context.graph_version,
        })
    });
    let mut diagnostics = if let Some(prebuilt) = extras.diagnostics {
        prebuilt
    } else {
        let mut diagnostics = args.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(args.estimate.provenance.overlap));
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
    if matches!(
        args.estimator_id,
        EstimatorId::FunctionalEffect | EstimatorId::FunctionalDistribution
    ) {
        if let Some(diagnostic) = free_variables_averaged(&args.identification, &args.estimand) {
            push_unique_diagnostic(&mut diagnostics, &mut seen, diagnostic);
        }
    }
    let structural_posteriors = extras
        .structural_response
        .iter()
        .flat_map(|mixture| mixture.atoms.iter().filter_map(|atom| atom.posterior.as_ref()));
    for diagnostic in
        posterior_note_diagnostics(extras.posterior.iter().chain(structural_posteriors))
    {
        push_unique_diagnostic(&mut diagnostics, &mut seen, diagnostic);
    }
    if let (Some(mode), Some(n)) = (context.latency_mode, extras.n_draws) {
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
            (id_artifact.as_ref(), id_op.as_ref(), &[], &args.identification.required_assumptions),
            (
                est_artifact.as_ref(),
                est_op.as_ref(),
                &[id_artifact.as_ref()],
                &args.estimate.provenance.assumptions,
            ),
        )
    };
    let mut physical_record = args.physical.record.clone();
    if context.custom_validators {
        let (record, diagnostic) = mark_python_callback_plan(physical_record, "validator");
        physical_record = record;
        diagnostics.push(diagnostic);
    }
    let payload = crate::result::ResultPayload::one_of([
        args.distribution.map(crate::result::ResultPayload::Distribution),
        (args.mediation.is_some() || extras.mediation_grid.is_some()).then(|| {
            crate::result::ResultPayload::from_mediation(args.mediation, extras.mediation_grid)
        }),
        extras.gcm.map(GcmSlot::into_payload),
        extras.response.map(crate::result::ResultPayload::Response),
    ]);
    let mut result = assemble_result(AssembleArgs {
        logical: &args.physical.logical.record,
        physical: &physical_record,
        identification: args.identification,
        estimand: args.estimand,
        estimate: args.estimate,
        posterior: extras.posterior,
        payload,
        refutations: args.refutations,
        diagnostics,
        provenance,
        treatment: args.treatment,
        outcome: args.outcome,
        wall_time_ns: args.wall_time_ns,
        latency_mode: context.latency_mode.map(|m| Arc::from(m.as_str())),
        stage_timings_ns: extras.stage_timings_ns,
        bootstrap_replicates_requested: extras
            .bootstrap_replicates_requested
            .unwrap_or(Some(context.bootstrap_replicates)),
        bootstrap_replicates_ok: args.bootstrap_replicates_ok,
        n_draws: extras.n_draws,
        cancelled: args.cancelled,
        early_stopped: args.early_stopped,
        bayesian: context.inference.is_bayesian(),
    });
    result.certificate = certificate.map(|identification| crate::AnalysisIdentification {
        identification,
        query: context.query.clone(),
        graph_class: context.graph_class,
    });
    let is_quantile = match &context.query {
        CausalQuery::AverageEffect(q) => q.outcome_functional.quantile_level().is_some(),
        CausalQuery::ConditionalEffect(q) => q.inner.outcome_functional.quantile_level().is_some(),
        CausalQuery::Response(q) => q.outcome_functional.quantile_level().is_some(),
        _ => false,
    };
    if is_quantile {
        if let Some(effect) = result.estimate.as_effect_mut() {
            effect.diagnostics.evalue = None;
        }
        result.diagnostics.push(super::helpers::quantile_scope_diagnostic());
        if matches!(context.query, CausalQuery::ConditionalEffect(_)) {
            result.diagnostics.push(Diagnostic::new("estimate.functional.conditional_quantile",
                    DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    "quantile contrast inverts arm CDFs standardized over the retained modifier distribution; this is not a pointwise conditional-quantile surface or an average of individual quantile effects"));
        }
    }
    result.predictive_checks = extras.predictive_checks;
    result.structural_response = extras.structural_response;
    result.rebind_interval(context.inference.is_bayesian());
    result.support_status = context.support_status;
    result.structure_source = context.structure_source;
    // A named predicate or custom distribution is a handle; encoding the
    // executed query (certificates, artifacts) needs its bindings.
    result.population_registry.clone_from(&context.population_registry);
    if !is_quantile {
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
    }
    if let Some(data) = tabular_data {
        super::helpers::attach_tiered_evalue_for_cell(
            &mut result.estimate,
            data,
            args.outcome,
            is_quantile,
            context.tiered.as_ref(),
            &mut result.diagnostics,
        );
    }
    if let Some(crate::support::CellStatus::Allowlisted { reason, parent }) = context.support_status
    {
        result.diagnostics.push(Diagnostic {
            code: Arc::from("support.allowed_unlicensed"),
            kind: DiagnosticKind::Support,
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
    if let Some((tier, configured)) = context.latency_bootstrap_not_applied {
        result.diagnostics.push(
            crate::analysis::helpers::latency_bootstrap_not_applied_diagnostic(tier, configured),
        );
    }
    if let Some(requested) = context.refute_default_downgrade {
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
/// pointwise / simultaneous band. Alias of [`crate::result::PERCENTILE_95_MIN_REPLICATES`].
pub(super) const PERCENTILE_95_BAND_MIN_SUCCESSES: usize =
    crate::result::PERCENTILE_95_MIN_REPLICATES as usize;

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

#[cfg(test)]
mod refutation_mixture_coverage_tests {
    use super::mix_atom_refutation_reports;
    use antecedent_validate::RefutationReport;

    fn report(id: &str, passed: bool) -> RefutationReport {
        RefutationReport::new(id, 1.0, 1.0, 0.5, true, passed, None, 10)
    }

    #[test]
    fn refuter_run_on_part_of_the_mass_reports_its_coverage() {
        // Atom A (0.6) ran the placebo; atom B (0.4) had it not applicable. The mixed placebo
        // "passed" covers 0.6 of the mass and the result must say so.
        let per_atom = vec![(0.6, vec![report("placebo", true)]), (0.4, Vec::new())];
        let (reports, diagnostics) = mix_atom_refutation_reports(&per_atom);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].passed);
        let partial: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.code.as_ref() == "refute.envelope.partial_coverage")
            .collect();
        assert_eq!(partial.len(), 1, "{diagnostics:?}");
        assert!(partial[0].message.contains("`placebo`"), "{}", partial[0].message);
        assert!(partial[0].message.contains("0.600000"), "{}", partial[0].message);
    }

    #[test]
    fn fully_covering_refuter_and_zero_weight_atoms_raise_no_coverage_warning() {
        let per_atom = vec![
            (0.5, vec![report("placebo", true)]),
            (0.5, vec![report("placebo", false)]),
            (0.0, Vec::new()),
        ];
        let (reports, diagnostics) = mix_atom_refutation_reports(&per_atom);
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].passed);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
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

#[cfg(test)]
mod posterior_status_ladder_tests {
    use super::*;

    #[test]
    fn resolve_posterior_status_uses_one_tolerance() {
        let base = IdentificationStatus::IdentifiedUnderParametricRestrictions;
        let policy = crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean;
        let dust = resolve_posterior_status(
            base,
            policy,
            PosteriorMassSummary { unidentified: 1e-13, ..PosteriorMassSummary::default() },
        );
        let real = resolve_posterior_status(
            base,
            policy,
            PosteriorMassSummary { unidentified: 1e-11, ..PosteriorMassSummary::default() },
        );
        assert_eq!(dust, base);
        assert_eq!(real, IdentificationStatus::GraphDependent);
        let kept = resolve_posterior_status(base, policy, PosteriorMassSummary::default());
        assert_eq!(kept, base);
        let partial = resolve_posterior_status(
            base,
            crate::result::StructuralAggregationPolicy::IdentifiedSetEnvelope,
            PosteriorMassSummary::default(),
        );
        assert_eq!(partial, IdentificationStatus::PartiallyIdentified);
    }

    #[test]
    fn temporal_refuter_keeps_the_query_contrast() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let mut query = TemporalEffectQuery::pulse(treatment, outcome, 1.0);
        query.active = Intervention::set(treatment, antecedent_core::Value::f64(2.0));
        query.control = Intervention::set(treatment, antecedent_core::Value::f64(-1.0));
        let average = temporal_refuter_average(&query);
        let bare = AverageEffectQuery::binary_ate(treatment, outcome);
        assert_eq!(average.active, query.active);
        assert_eq!(average.control, query.control);
        assert_eq!(average.target_population, query.target_population);
        assert_ne!(average.active, bare.active);
        assert_ne!(SUSTAINED_LINEAR_SEM, UNFOLDED_LINEAR_SEM);
        assert_ne!(SUSTAINED_LINEAR_SEM, SEQUENCE_LINEAR_SEM);
        assert_ne!(UNFOLDED_LINEAR_SEM, DBN_LINEAR_SEM);
    }
}

#[cfg(test)]
mod graph_mixture_timing {
    use super::*;

    /// Timing harness for appending unidentified samples to a mixture
    /// (`cargo test -p antecedent --lib graph_mixture_timing -- --ignored --nocapture`).
    #[test]
    #[ignore = "timing harness"]
    fn unidentified_samples_scale_linearly() {
        for samples in [2_000_usize, 8_000, 32_000] {
            let keys: Vec<u64> = (0..samples as u64).collect();
            let graphs = WeightedGraphSamples::new(
                vec![1.0 / samples as f64; samples],
                vec![GraphIdentFlag::Unidentified; samples],
                keys,
            )
            .unwrap();
            let started = std::time::Instant::now();
            let mixed = mix_graph_posterior_identified_atoms(&graphs, &[], 1.0, 0.0, 0.0, 0);
            eprintln!(
                "samples={samples:>6} atoms={} append={:.1} ms",
                mixed.mixture.atoms.len(),
                started.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}
