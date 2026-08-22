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
    pub stage_timings_ns: Vec<(Arc<str>, u64)>,
    pub identify_provenance: Option<(Arc<str>, Arc<str>)>,
    pub estimate_provenance: Option<(Arc<str>, Arc<str>)>,
    pub posterior: Option<antecedent_estimate::CausalPosterior>,
    pub n_draws: Option<u32>,
    pub predictive_checks: Vec<antecedent_validate::PredictiveCheckReport>,
    /// When set, replaces the identification + overlap + cache diagnostic seed.
    pub diagnostics: Option<Vec<Diagnostic>>,
    pub response: Option<antecedent_core::CausalResponse>,
    pub gcm: Option<GcmSlot>,
    pub empty_provenance: bool,
    /// `None` uses the study bootstrap count; `Some(v)` writes `v` (response writes `None`).
    /// Nested option is intentional: outer selects override vs study default.
    #[allow(clippy::option_option)]
    pub bootstrap_replicates_requested: Option<Option<u32>>,
}

pub(super) fn identification_from_cache_or(
    cache: Option<&crate::analysis::prepared::CachedStaticIdentification>,
    live: impl FnOnce() -> Result<(IdentificationResult, IdentifiedEstimand), CausalError>,
) -> Result<(IdentificationResult, IdentifiedEstimand, bool), CausalError> {
    if let Some(cache) = cache {
        return Ok((cache.identification.clone(), cache.estimand.clone(), true));
    }
    let (identification, estimand) = live()?;
    Ok((identification, estimand, false))
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
pub(super) fn parametric_scm_identification(
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
    let identification = IdentificationResult::from_parts(
        IdentificationStatus::IdentifiedUnderParametricRestrictions,
        query,
        vec![estimand.clone()],
        arena,
        DerivationTrace::default(),
        antecedent_core::AssumptionSet::default(),
        Vec::new(),
        IdentificationPerformanceRecord::default(),
        None,
    );
    (identification, estimand)
}

pub(super) fn binary_cf_interventions(
    query: &antecedent_core::CounterfactualQuery,
) -> Result<(VariableId, f64, f64), CausalError> {
    if query.interventions.len() != 1 {
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
    Ok((*variable, active, 0.0))
}

pub(super) fn identification_status_ok_for_case(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

pub(super) fn envelope_to_identification_result(
    envelope: &IdentificationEnvelope<Pag>,
    query: &AverageEffectQuery,
) -> IdentificationResult {
    let mut estimands = Vec::new();
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut diagnostics = Vec::new();
    for case in &envelope.cases {
        if identification_status_ok_for_case(case.result.status) {
            estimands.extend(case.result.estimands.iter().cloned());
            assumptions = case.result.required_assumptions.clone();
            diagnostics.extend(case.result.diagnostics.iter().cloned());
        }
    }
    if let Some(inv) = &envelope.invariant {
        if estimands.is_empty() {
            estimands.push(inv.clone());
        }
    }
    IdentificationResult::from_parts(
        envelope.status,
        CausalQuery::AverageEffect(query.clone()),
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
    if let Some(ext) = cfg.external_compose.as_ref() {
        est.inner.prior = Some(ext.composed.prior.clone());
    } else {
        est.inner.prior = resolve_bayesian_prior(cfg, bprep)?;
    }
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

impl super::Study {
    pub(super) fn require_execute_dag(&self, message: &'static str) -> Result<&Dag, CausalError> {
        self.graph.as_dag().ok_or(CausalError::Unsupported { message })
    }

    pub(super) fn finish_identified_execute(
        &self,
        args: IdentifiedExecuteFinish<'_>,
    ) -> StudyResult {
        let extras = args.extras;
        let mut diagnostics = if let Some(prebuilt) = extras.diagnostics {
            prebuilt
        } else {
            let mut diagnostics = args.identification.diagnostics.clone();
            diagnostics.push(overlap_diagnostic(args.estimate.overlap));
            if args.identify_cached {
                diagnostics.push(identify_cached_diagnostic());
            }
            diagnostics.extend(args.extra_diagnostics);
            diagnostics
        };
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
        result.predictive_checks = extras.predictive_checks;
        result.response = extras.response;
        result.support_status = self.support_status;
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
