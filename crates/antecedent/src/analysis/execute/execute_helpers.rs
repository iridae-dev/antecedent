// Free functions supporting Study execute paths.
// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn gcm_query_vars(query: &CausalQuery) -> Result<(VariableId, VariableId), CausalError> {
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
    MultiEnvTemporalEffect,
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
        (DataModality::MultiEnv, CausalQuery::TemporalEffect(_)) => {
            AnalysisRoute::MultiEnvTemporalEffect
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

/// Interactive graph×effect: stratified subsample of Identified graphs; leftover
/// identified mass is flipped to Unidentified (never silent renormalize to 1).
///
/// Call this **after** resolving the shared envelope prior from the first
/// identified atom in original order ([`resolve_envelope_prior_anchor`]), and
/// **before** per-graph estimation so dropped atoms never pay a fit. Subsample
/// must not move the prior anchor (0.6.0 semantics).
pub(super) fn maybe_interactive_subsample_graphs(
    latency_mode: Option<LatencyMode>,
    graphs: WeightedGraphSamples,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<WeightedGraphSamples, CausalError> {
    if latency_mode != Some(LatencyMode::Interactive) {
        return Ok(graphs);
    }
    let mut rng = ctx.rng.stream(0xE11E_u64);
    let sub = graphs
        .stratified_interactive_subsample(INTERACTIVE_MAX_ENVELOPE_GRAPHS, &mut rng)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if sub.approximate {
        diagnostics.push(Diagnostic::new(
            "estimate.envelope.interactive_subsample",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "approximate=true leftover_identified_mass={} max_identified={}",
                sub.leftover_identified_mass, INTERACTIVE_MAX_ENVELOPE_GRAPHS
            ),
        ));
    }
    Ok(sub.graphs)
}

/// Interactive graph×effect subsample: stratified Identified selection; leftover
/// identified mass flips to Unidentified (never silent renormalize). Filters
/// `per_graph` draws to keys that remain Identified after selection.
pub(super) fn maybe_interactive_envelope_subsample(
    latency_mode: Option<LatencyMode>,
    graphs: WeightedGraphSamples,
    per_graph: Vec<GraphEffectDraws>,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(WeightedGraphSamples, Vec<GraphEffectDraws>), CausalError> {
    if latency_mode != Some(LatencyMode::Interactive) {
        return Ok((graphs, per_graph));
    }
    let mut rng = ctx.rng.stream(0xE11E_u64);
    let sub = graphs
        .stratified_interactive_subsample(INTERACTIVE_MAX_ENVELOPE_GRAPHS, &mut rng)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if !sub.approximate {
        return Ok((sub.graphs, per_graph));
    }
    let keep_keys = identified_envelope_keys(&sub.graphs);
    let filtered: Vec<GraphEffectDraws> =
        per_graph.into_iter().filter(|g| keep_keys.contains(&g.graph_key)).collect();
    diagnostics.push(Diagnostic::new(
        "estimate.envelope.interactive_subsample",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "approximate=true leftover_identified_mass={} max_identified={}",
            sub.leftover_identified_mass, INTERACTIVE_MAX_ENVELOPE_GRAPHS
        ),
    ));
    Ok((sub.graphs, filtered))
}

/// Resolve the shared envelope prior from a prepared Bayesian problem.
///
/// Call while preparing identified atoms in **original envelope order**, before
/// Interactive stratified selection. Subsample must not change which design
/// anchors the prior, and prepare eligibility must be established before
/// selection (0.6.0 semantics).
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
    matches!(
        query.policy,
        antecedent_core::TemporalPolicy::Sustained { from, until } if from != until
    )
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

/// Emit [`envelope_se_omits_between_atom_variance`] only when something was
/// actually omitted: more than one atom contributes and no joint SE was formed.
/// A single contributing atom keeps its own SE, so there is nothing to disclose.
pub(super) fn envelope_se_omission_diagnostic(
    contributing_atoms: usize,
    se: f64,
) -> Option<Diagnostic> {
    (contributing_atoms > 1 && !se.is_finite()).then(envelope_se_omits_between_atom_variance)
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

/// Shared circular-block SE for a frozen-weight temporal class / DBN mixture.
pub(super) struct SharedCircularBlockSe {
    /// Mixture SE: replicate SD scaled by the fixed-b factor (NaN when unavailable).
    pub se: f64,
    pub completed: u32,
    pub attempted: u32,
    /// Block length in series times.
    pub block_length: usize,
    /// Series times resampled (the window every atom can evaluate).
    pub rows: usize,
    /// Effective rows of every atom's and the weighted mixture's estimating score
    /// at the block length ([`antecedent_estimate::score_effective_rows`]; NaN
    /// when unknown).
    pub effective_rows: f64,
    /// Successful replicates: every atom's refit, in atom order.
    pub atom_draws: Vec<Vec<f64>>,
    /// Fixed-b factor applied to [`Self::se`].
    pub fixed_b: f64,
}

impl SharedCircularBlockSe {
    fn empty() -> Self {
        Self {
            se: f64::NAN,
            completed: 0,
            attempted: 0,
            block_length: 0,
            rows: 0,
            effective_rows: f64::NAN,
            atom_draws: Vec::new(),
            fixed_b: 1.0,
        }
    }

    /// Imbens–Manski interval for the identified set spanned by the atoms'
    /// point estimates `points` (atom order), from the same shared replicates.
    pub(super) fn identified_set_interval(
        &self,
        points: &[f64],
        level: f64,
    ) -> Option<antecedent_estimate::IdentifiedSetInterval> {
        if !bootstrap_has_enough_successes(self.atom_draws.len(), self.attempted as usize) {
            return None;
        }
        antecedent_estimate::imbens_manski_shared_replicates(
            points,
            &self.atom_draws,
            self.fixed_b,
            self.rows,
            level,
        )
    }
}

/// One atom of a frozen-weight temporal mixture, prepared once on the original
/// series so the shared bootstrap can refit it on resampled lag-aligned rows.
pub(super) enum TemporalAtomDesign {
    /// Pulse / single-step Sustained: one lag-aligned adjustment regression.
    Linear {
        prep: Box<antecedent_estimate::PreparedEstimationProblem>,
        rows: antecedent_estimate::AlignedRows,
        fitter: antecedent_estimate::LinearAdjustmentAte,
        point: EffectEstimate,
        normal_scores: Vec<Vec<f64>>,
    },
    /// Multi-step Sustained: sequential g-computation over the unfolded window.
    Sequential {
        design: Box<antecedent_estimate::SequentialContrastDesign>,
        point: f64,
        influence: Option<Vec<f64>>,
        normal_scores: Vec<Vec<f64>>,
    },
}

impl TemporalAtomDesign {
    /// Prepare a Pulse / single-step Sustained atom.
    pub(super) fn linear(
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&antecedent_data::DiscoveryEstimationSplit>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = 0;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let (prep, rows) = estimator
            .prepare_aligned(data, estimand, query, indexer, split, &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        let point = estimator
            .inner
            .fit_point(
                &prep,
                &mut EstimationWorkspace::default(),
                antecedent_core::AssumptionSet::default(),
            )
            .map_err(CausalError::from)?;
        let normal_scores = antecedent_estimate::normal_equation_scores(
            &prep.design.matrix,
            prep.design.nrows,
            prep.design.ncols,
            &prep.design.outcome,
        )
        .unwrap_or_default();
        Ok(Self::Linear {
            prep: Box::new(prep),
            rows,
            fitter: estimator.inner,
            point,
            normal_scores,
        })
    }

    /// Prepare a multi-step Sustained atom.
    pub(super) fn sequential(
        data: &TimeSeriesData,
        graph: &TemporalDag,
        indexer: &TemporalIndexer,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let design = antecedent_estimate::SequentialContrastDesign::prepare(
            data, graph, indexer, estimand, query, status, ctx,
        )
        .map_err(CausalError::from)?;
        let point = design.estimate().map_err(CausalError::from)?;
        let influence = design.influence();
        let normal_scores = design.normal_equation_scores();
        Ok(Self::Sequential { design: Box::new(design), point, influence, normal_scores })
    }

    fn aligned_rows(&self) -> antecedent_estimate::AlignedRows {
        match self {
            Self::Linear { rows, .. } => *rows,
            Self::Sequential { design, .. } => design.aligned_rows(),
        }
    }

    /// Full-sample point and (for linear atoms) iid analytic SE, with `assumptions`.
    pub(super) fn effect_estimate(
        &self,
        assumptions: antecedent_core::AssumptionSet,
    ) -> EffectEstimate {
        match self {
            Self::Linear { point, .. } => {
                let mut estimate = point.clone();
                estimate.assumptions = assumptions;
                estimate
            }
            Self::Sequential { point, .. } => {
                EffectEstimate::new(*point, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride)
            }
        }
    }

    fn estimate_on_rows(&self, rows: &[usize], workspace: &mut EstimationWorkspace) -> Option<f64> {
        match self {
            Self::Linear { prep, fitter, .. } => {
                let mut x = vec![0.0; rows.len() * prep.design.ncols];
                let mut y = vec![0.0; rows.len()];
                fitter.ate_on_row_indices_into(prep, workspace, rows, &mut x, &mut y).ok()
            }
            Self::Sequential { design, .. } => design.estimate_on_rows(rows).ok(),
        }
    }

    /// Per-row influence of the atom's estimate on its own aligned rows.
    fn influence(&self) -> Option<&[f64]> {
        match self {
            Self::Linear { point, .. } => point.influence.as_deref(),
            Self::Sequential { influence, .. } => influence.as_deref(),
        }
    }

    /// Intercept normal-equation score (the residual series) when the design
    /// carries an intercept column first; otherwise the first fitted score.
    fn intercept_residual(&self) -> Option<&[f64]> {
        match self {
            Self::Linear { normal_scores, .. } | Self::Sequential { normal_scores, .. } => {
                normal_scores.first().map(Vec::as_slice)
            }
        }
    }
}

/// Shared circular-block SE of a frozen-weight mixture over temporal atoms.
///
/// Blocks of consecutive series times are resampled over the window where the
/// maximal lag window across all atoms is available; each atom's lag-aligned
/// design keeps its rows' lag windows from the original series, every atom is
/// refit on the same resampled times, and the replicate SD is scaled by the
/// Kiefer–Vogelsang fixed-b factor ([`antecedent_estimate::aligned_block_bootstrap`]).
/// The block length is [`antecedent_estimate::dependence_block_length`] over the
/// `m` shared times: at least `max(structural_span, ceil(m^(1/3)))`, lengthened
/// when any atom's (or the mixture's) estimating score carries slowly decaying
/// dependence. Unidentified mass is not mixed; a replicate that cannot fit every
/// atom is dropped rather than renormalized. The interval is for the reported
/// aggregate.
pub(super) fn shared_circular_block_mixture_se(
    atoms: &[&TemporalAtomDesign],
    weights: &[f64],
    structural_span: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
) -> SharedCircularBlockSe {
    let designs: Vec<_> = atoms.iter().map(|atom| atom.aligned_rows()).collect();
    let Some((start, len)) = antecedent_estimate::common_time_window(&designs)
        .filter(|_| replicates > 0)
    else {
        return SharedCircularBlockSe::empty();
    };
    let influences: Option<Vec<&[f64]>> = atoms
        .iter()
        .zip(&designs)
        .map(|(atom, design)| {
            let offset = start - design.first_time;
            atom.influence()?.get(offset..offset + len)
        })
        .collect();
    let score = influences.as_deref().and_then(|windows| mixture_score(windows, weights, len));
    // Intercept residual (the persistent score 1.9 needed) plus each atom's
    // influence and the mixture score. Other OLS columns are not PW-scanned.
    let residual_windows: Vec<&[f64]> = atoms
        .iter()
        .zip(&designs)
        .filter_map(|(atom, design)| {
            let offset = start - design.first_time;
            atom.intercept_residual()?.get(offset..offset + len)
        })
        .collect();
    let mut scores: Vec<&[f64]> = influences.iter().flatten().copied().collect();
    scores.extend(score.as_deref());
    scores.extend(residual_windows.iter().copied());
    let block_length = antecedent_estimate::dependence_block_length(structural_span, len, &scores);
    let mut workspace = EstimationWorkspace::default();
    let mut out = shared_circular_block_mixture_se_with_length(
        &designs,
        weights,
        block_length,
        replicates,
        stream_base,
        ctx,
        |atom, rows| atoms[atom].estimate_on_rows(rows, &mut workspace),
    );
    // Every atom's score, not only the weighted sum: a mixture dominated by a
    // nearly iid atom can hide another atom's slowly decaying dependence.
    out.effective_rows = if score.is_some() {
        let target: Vec<&[f64]> =
            influences.iter().flatten().copied().chain(score.as_deref()).collect();
        antecedent_estimate::score_effective_rows(&target, block_length)
    } else {
        f64::NAN
    };
    out
}

/// The weighted mixture estimating score `Σ_g w̄_g IF_g(t)` over the shared times.
fn mixture_score(influences: &[&[f64]], weights: &[f64], len: usize) -> Option<Vec<f64>> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut score = vec![0.0; len];
    for (influence, weight) in influences.iter().zip(weights) {
        for (slot, value) in score.iter_mut().zip(*influence) {
            *slot += weight / total * *value;
        }
    }
    Some(score)
}

/// Block-length rule for the shared circular block: the structural lag span
/// or `ceil(n^(1/3))`, whichever is longer, capped at `n` (the floor of
/// [`antecedent_estimate::dependence_block_length`]).
#[cfg(test)]
pub(super) fn circular_block_length(structural_span: usize, n: usize) -> usize {
    antecedent_data::circular_block_length(structural_span, n)
}

/// [`shared_circular_block_mixture_se`] over explicit aligned designs at an
/// explicit block length (the block-length sensitivity check calls this
/// directly; production callers use the rule). `fit_atom(g, rows)` refits atom
/// `g` on its design rows `rows`.
pub(super) fn shared_circular_block_mixture_se_with_length(
    designs: &[antecedent_estimate::AlignedRows],
    weights: &[f64],
    block_length: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
    mut fit_atom: impl FnMut(usize, &[usize]) -> Option<f64>,
) -> SharedCircularBlockSe {
    let total: f64 = weights.iter().sum();
    if replicates == 0 || designs.is_empty() || designs.len() != weights.len() || total <= 0.0 {
        return SharedCircularBlockSe::empty();
    }
    let Some(draws) = antecedent_estimate::aligned_block_bootstrap(
        designs,
        block_length,
        replicates,
        stream_base,
        ctx,
        |maps| {
            let mut values = Vec::with_capacity(maps.len() + 1);
            let mut mixture = 0.0;
            for (atom, (rows, weight)) in maps.iter().zip(weights).enumerate() {
                let value = fit_atom(atom, rows)?;
                mixture += weight / total * value;
                values.push(value);
            }
            values.push(mixture);
            Some(values)
        },
    ) else {
        return SharedCircularBlockSe::empty();
    };
    let k = designs.len();
    let se = draws.se_result(k).se.unwrap_or(f64::NAN);
    SharedCircularBlockSe {
        se,
        completed: u32::try_from(draws.draws.len()).unwrap_or(u32::MAX),
        attempted: draws.attempted,
        block_length: draws.block_length,
        rows: draws.rows,
        effective_rows: f64::NAN,
        fixed_b: draws.fixed_b(),
        atom_draws: draws.draws.into_iter().map(|mut draw| {
            draw.truncate(k);
            draw
        }).collect(),
    }
}

/// The shared-block provenance diagnostic for a class envelope, plus the
/// short-series warning when the mixture score is below the mixture threshold.
pub(super) fn envelope_shared_block_diagnostics(
    identified_mass: f64,
    unidentified_mass: f64,
    block: &SharedCircularBlockSe,
) -> Vec<Diagnostic> {
    let mut out = vec![Diagnostic::new(
        "estimate.temporal_class.frequentist.shared_block",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        shared_block_mixture_message(
            "frozen completion weights",
            identified_mass,
            unidentified_mass,
            block,
        ),
    )];
    out.extend(short_series_warning(
        block.effective_rows,
        antecedent_estimate::CircularBlockFamily::Mixture,
    ));
    out
}

/// One message for every frozen-weight shared circular-block mixture SE
/// (temporal class envelopes and DBN posteriors), so both carry the same
/// statement of how the interval is built and what it is for.
pub(super) fn shared_block_mixture_message(
    weight_basis: &str,
    identified_mass: f64,
    unidentified_mass: f64,
    block: &SharedCircularBlockSe,
) -> String {
    format!(
        "{weight_basis}; identified_mass={identified_mass}; \
         unidentified_mass={unidentified_mass}; shared circular-block \
         replicates={}; attempted={}; blocks of {} consecutive series times \
         (dependence-aware length, at least max(span, ceil(m^(1/3)))) over the \
         {} times where every atom's lag window is available; each atom's lag-aligned \
         rows keep their original lag windows and every atom is refit on the same \
         resampled times; replicate SD scaled by the Kiefer-Vogelsang fixed-b factor \
         {:.4}; score effective rows {:.1} (smallest over every atom's and the \
         mixture's score of the lag-1 and block-length readings); between-atom sampling \
         variance included; unidentified mass is not mixed into the SE; \
         the interval is for the reported aggregate, not a distribution \
         over graph-specific effects",
        block.completed, block.attempted, block.block_length, block.rows, block.fixed_b,
        block.effective_rows,
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
    let mut prior_items = Vec::with_capacity(atoms.len());
    let mut post_items = Vec::with_capacity(atoms.len());
    for atom in atoms {
        // Each atom's checks use the prior that atom was actually fitted under.
        let atom_est = BayesianGComputationAte { prior: atom.prior.clone(), ..est.clone() };
        let ppc_prior = atom_est.prior_in_force(atom.prep.design.ncols);
        let prior_rep = PriorPredictiveCheck {
            n_sims: 200,
            seed: ctx.rng.master_seed(),
            ..PriorPredictiveCheck::new()
        }
        .check_with_prior(&atom.prep, &ppc_prior, ctx)
        .map_err(CausalError::from)?;
        // Temporal atoms (lag indexer present) add the serial-dependence discrepancy
        // under `full`; exchangeable static rows keep the two-axis check.
        let post_rep = if matches!(refute, RefuteSuite::Full) && atom.indexer.is_some() {
            PosteriorPredictiveCheck::new()
                .check_temporal(&atom.prep, &atom.posterior, ctx.rng.master_seed())
                .map_err(CausalError::from)?
        } else {
            PosteriorPredictiveCheck::new()
                .check(&atom.prep, &atom.posterior)
                .map_err(CausalError::from)?
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
        if let (Some(mixed), Some(sens)) =
            (mix_prior_sensitivity_summaries(&sens_items), grid.as_ref())
        {
            refutations.push(sens.to_report(&mixed, estimate_ate));
            *mixture_posterior = with_prior_sensitivity(mixture_posterior.clone(), mixed);
        }
    }

    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    diagnostics.push(Diagnostic::new(
        "refute.bayesian.ppc.envelope",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        if matches!(refute, RefuteSuite::Full) {
            format!(
                "PPC and prior-sensitivity evaluated per identified completion [{atom_keys}]; \
                 reports are mixture-weighted by graph posterior mass"
            )
        } else {
            format!(
                "PPC evaluated per identified completion [{atom_keys}]; reports are \
                 mixture-weighted by graph posterior mass"
            )
        },
    ));
    Ok(predictive_checks)
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
/// marginalization here to (mis)compute.
pub(crate) fn parametric_scm_identification(
    query: CausalQuery,
    treatment: VariableId,
    outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let y = arena.intern_var_set([outcome]);
    let do_t = arena.intern_intervention_set([treatment]);
    let empty = arena.empty_var_set();
    let distribution = arena.intern(ExprNode::Distribution {
        variables: y,
        conditioned_on: empty,
        intervention: do_t,
        domain: DomainRef::Interventional,
    });
    let functional = arena
        .intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution });
    arena.set_derivation(
        functional,
        DerivationMeta {
            rule: Arc::from("gcm.parametric"),
            note: Some(Arc::from(format!(
                "parametric SCM: treatment={treatment:?} outcome={outcome:?}; no adjustment \
                 set (GCM does not identify via backdoor covariates)"
            ))),
        },
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

pub(crate) fn identification_status_ok_for_case(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
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
            for record in &case.result.required_assumptions.entries {
                if !assumptions.entries.contains(record) {
                    assumptions.push(record.clone());
                }
            }
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
                 larger of the AR(1)-prewhitened Newey-West long-run-variance ratio of the \
                 targeted slope score (scaled by its squared fixed-b factor) and the \
                 AR(1)-residual variance ratio given the design, floored at 1 \
                 (kappa = [{list}] over {} fit(s)); this corrects serial \
                 dependence in the outcome residual, not heteroskedasticity or a misspecified mean",
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

fn push_aipw_score_kind(
    diagnostics: &mut Vec<Diagnostic>,
    estimator_id: EstimatorId,
    estimate: &EffectEstimate,
) {
    if diagnostics.iter().any(|d| {
        matches!(
            d.code.as_ref(),
            "estimate.aipw.crossfit_scores" | "estimate.aipw.full_sample_residualized"
        )
    }) {
        return;
    }
    match estimator_id {
        EstimatorId::CellAipw | EstimatorId::Aipw if estimate.score_table.is_some() => {
            diagnostics.push(Diagnostic::new(
                "estimate.aipw.crossfit_scores",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "cross-fitted AIPW scores φᵢ^a; retarget averages this table. A residualized full-sample AIPW fit is a different object",
            ));
        }
        EstimatorId::Aipw => {
            diagnostics.push(Diagnostic::new(
                "estimate.aipw.full_sample_residualized",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "this AIPW fit is full-sample residualized and has no score table; it is not the cross-fitted φ family that retarget averages. Prepare an AllObserved iid AIPW plan to retarget",
            ));
        }
        _ => {}
    }
}

fn push_grid_scalar_cleared(diagnostics: &mut Vec<Diagnostic>, estimate: &EffectEstimate) {
    if let Some(inf) = estimate.score_inference.as_ref() {
        if !diagnostics.iter().any(|d| d.code.as_ref() == "estimate.functional.cdf_inference") {
            diagnostics.push(Diagnostic::new(
                "estimate.functional.cdf_inference",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "per-arm F_a(c) simultaneous bands describe raw CDF coordinates; rearranged exceedance_cdf values are not mixed with those intervals",
            ));
        }
        if inf.threshold_supported.iter().any(|ok| !ok)
            && !diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.functional.threshold_tail.unsupported")
        {
            diagnostics.push(Diagnostic::new(
                "estimate.functional.threshold_tail.unsupported",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "at least one threshold tail lacks enough treated/control events for a tail probability; that coordinate's band is non-finite rather than an empty-cell or first-threshold SE",
            ));
        }
    } else if estimate.score_table.is_none()
        && estimate.exceedance_cdf.is_some()
        && !diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.functional.cdf_inference.unavailable")
    {
        diagnostics.push(Diagnostic::new(
            "estimate.functional.cdf_inference.unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "conditional CDF values have no per-arm simultaneous bands or threshold tail-support evidence; joint covariance, when present, describes raw threshold contrasts, not the projected per-arm CDF",
        ));
    }
    if diagnostics.iter().any(|d| d.code.as_ref() == "estimate.functional.grid_scalar_cleared") {
        return;
    }
    if estimate.exceedance_cdf.as_ref().is_some_and(|cdf| cdf.len() > 2)
        && !estimate.ate.is_finite()
    {
        diagnostics.push(Diagnostic::new(
            "estimate.functional.grid_scalar_cleared",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
        ));
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
        // Envelope routes supply their own diagnostic seed, but a prepared
        // envelope still must expose cache reuse just like a single-graph path.
        if args.identify_cached
            && diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached")
        {
            diagnostics.push(identify_cached_diagnostic());
        }
        push_aipw_score_kind(&mut diagnostics, args.estimator_id, &args.estimate);
        push_grid_scalar_cleared(&mut diagnostics, &args.estimate);
        let structural_posteriors = extras
            .structural_response
            .iter()
            .flat_map(|mixture| mixture.atoms.iter().filter_map(|atom| atom.posterior.as_ref()));
        for diagnostic in
            posterior_note_diagnostics(extras.posterior.iter().chain(structural_posteriors))
        {
            if diagnostics.iter().all(|d| d.code != diagnostic.code) {
                diagnostics.push(diagnostic);
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
        result.support_status = self.support_status;
        result.structure_source = self.structure_source;
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
/// replicates as fitting failures.
pub(super) fn bootstrap_has_enough_successes(completed: usize, attempted: usize) -> bool {
    completed >= 2 && completed >= attempted.saturating_sub(completed)
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
