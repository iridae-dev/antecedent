//! Class-aware temporal response execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianGComputationAte, BayesianResponseFit, CausalError, CausalPosterior,
    CausalResponse, ClassBandLayout, ClassBootstrapTargets, ClassObservationAtom,
    ClassObservationKind, ClassResponseAssembly, ClassResponseInput, ClassResponseParts,
    ClassResponseTail, CompletionFitMemo, Diagnostic, EffectEstimate, EstimatorId,
    ExecutionContext, IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifierId,
    InferenceMode, Instant, ObservationBootstrap, ObservationMechanismEstimator, ObservationSpec,
    OverlapPolicy, PhysicalExecutionPlan, RefuteSuite, ResponseFitKey, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, StudyResult,
    TemporalClassWeights, TemporalDag, TemporalIndexer, TemporalResponseEstimator, TimeSeriesData,
    VariableId, append_temporal_observation_assumptions, apply_class_observation_bootstrap,
    bare_class_atom, bayesian_gcomp, class_atom_adjustment, class_completion_key,
    class_horizon_witness, class_observation_band_note, class_response_identified_set_diagnostics,
    enforce_temporal_response_memory_budget, fit_class_bayesian_response,
    fit_class_frequentist_response, identification_status_ok_for_case, nan_effect,
    push_conflict_diagnostics, select_estimand, temporal_class_complete_values,
    temporal_class_envelope_diagnostic, temporal_class_response_masses,
    temporal_class_response_mean,
};

/// The series, query and context every completion of a class temporal
/// response sweep is fitted with.
#[derive(Clone, Copy)]
struct ClassResponseSweep<'a> {
    data: &'a TimeSeriesData,
    query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &'a ExecutionContext,
}

/// The completion of a class temporal response sweep being fitted: its
/// position, class weight, identification case and unfold indexer.
#[derive(Clone, Copy)]
struct ClassResponseSite<'a> {
    horizon_index: usize,
    horizon: u32,
    case_index: usize,
    weight: f64,
    case: &'a antecedent_identify::GraphIdentificationCase<
        antecedent_identify::TemporalCompletionGraph,
    >,
    indexer: &'a TemporalIndexer,
    envelope: &'a Arc<antecedent_identify::TemporalClassEnvelope>,
}

/// Accumulators of a class temporal response sweep: the assembly the atoms
/// join and the first completion's identification products.
struct ClassResponseRun {
    assembly: ClassResponseAssembly,
    primary_estimand: Option<IdentifiedEstimand>,
    primary_identification: Option<IdentificationResult>,
    primary_envelope: Option<Arc<antecedent_identify::TemporalClassEnvelope>>,
    assumptions: antecedent_core::AssumptionSet,
    diagnostics: Vec<Diagnostic>,
    observation_atoms: Vec<ClassObservationAtom>,
    /// Pulse witnesses of the evaluated completions, kept when an intervention level
    /// runs its cheap/full refuter state.
    witness_sites: Vec<super::super::temporal_response_refutation::PulseWitnessSite>,
    checked_witnesses: Vec<crate::result::TemporalResponseWitness>,
    checked_witness_leases: Vec<antecedent_core::execution::AdmissionLease>,
}

/// Completions given the same estimand, indexer, status and assumptions run
/// one fit at a horizon; every other fit input is fixed per horizon.
#[derive(Default)]
struct ClassResponseHorizonFits {
    bayesian: CompletionFitMemo<BayesianResponseFit, ResponseFitKey>,
    frequentist: CompletionFitMemo<CausalResponse, ResponseFitKey>,
}

impl ClassSequenceRun {}

/// What the observation circular-block band of a class response reads.
#[derive(Clone, Copy)]
struct ClassBandRequest<'a> {
    data: &'a TimeSeriesData,
    query: &'a ResponseQuery,
    atoms: &'a [ClassObservationAtom],
    cells_per_horizon: usize,
    n_horizons: usize,
    full_mass_scope: bool,
    unidentified_mass: f64,
    unevaluable_mass: f64,
    ctx: &'a ExecutionContext,
}

/// The series, query, overlays and Bayesian composition every directed
/// completion of a class Sequence sweep is estimated with.
#[derive(Clone, Copy)]
struct ClassSequenceSweep<'a> {
    data: &'a TimeSeriesData,
    query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    overlays: &'a antecedent_estimate::AdmittedMechanismOverlays,
    bayes: Option<&'a BayesianGComputationAte>,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &'a ExecutionContext,
}

/// Accumulators of a class Sequence sweep: per-horizon envelope bounds, the
/// structural atoms and the first completion's identification products.
pub(super) struct ClassSequenceRun {
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    pub(super) structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    pub(super) primary_estimand: Option<IdentifiedEstimand>,
    pub(super) primary_identification: Option<IdentificationResult>,
    pub(super) primary_envelope: Option<Arc<antecedent_identify::TemporalClassEnvelope>>,
    pub(super) assumptions: antecedent_core::AssumptionSet,
    pub(super) full_mass_scope: bool,
    pub(super) truncated_atoms: usize,
    pub(super) horizon_fingerprints: Vec<Vec<u64>>,
    pub(super) observation_atoms: Vec<ClassObservationAtom>,
    pub(super) diagnostics: Vec<Diagnostic>,
}

/// Admit the graph digest work before retaining a completion witness.
fn class_response_witness(
    site: &ClassResponseSite<'_>,
    atom_index: usize,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<crate::result::TemporalResponseWitness, CausalError> {
    let ClassResponseSite { horizon_index, horizon, weight, case, .. } = *site;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "class response witnesses" });
    }
    let nodes = match &case.graph {
        antecedent_identify::TemporalCompletionGraph::Dag(graph) => graph.node_count(),
        antecedent_identify::TemporalCompletionGraph::Mag(graph) => graph.node_count(),
    };
    let nodes = u64::try_from(nodes)
        .map_err(|_| CausalError::Resource { message: "completion digest size overflow".into() })?;
    // First fund and perform a bounded edge count. Every subsequent
    // successor scan visits at most all nodes and both adjacency entries
    // of each marked edge; there are at most edges + 1 scans.
    let initial = nodes
        .checked_mul(nodes)
        .and_then(|n| n.checked_add(nodes))
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| CausalError::Resource {
            message: "completion digest work overflow".into(),
        })?;
    budget
        .charge_work(initial)
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    let mut edges = 0u64;
    let mut count_edge = || -> Result<(), CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "class response witnesses" });
        }
        edges = edges.checked_add(1).ok_or_else(|| CausalError::Resource {
            message: "completion edge count overflow".into(),
        })?;
        Ok(())
    };
    match &case.graph {
        antecedent_identify::TemporalCompletionGraph::Dag(graph) => {
            for _ in graph.edges() {
                count_edge()?;
            }
        }
        antecedent_identify::TemporalCompletionGraph::Mag(graph) => {
            for _ in graph.edges_lazy() {
                count_edge()?;
            }
        }
    }
    let scan = edges
        .checked_mul(2)
        .and_then(|e| e.checked_add(nodes))
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| CausalError::Resource {
            message: "completion digest work overflow".into(),
        })?;
    let work = scan
        .checked_mul(edges.checked_add(1).ok_or_else(|| CausalError::Resource {
            message: "completion digest work overflow".into(),
        })?)
        .and_then(|w| nodes.checked_mul(nodes).and_then(|n| w.checked_add(n)))
        .ok_or_else(|| CausalError::Resource {
            message: "completion digest work overflow".into(),
        })?;
    budget
        .charge_work(work)
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    let digest = case
        .graph
        .digest_without_allocation(&|| ctx.cancellation.is_cancelled())
        .ok_or(CausalError::Cancelled { stage: "class response witnesses" })?;
    Ok(crate::result::TemporalResponseWitness {
        horizon_index,
        horizon,
        atom_index,
        weight,
        completion_digest: digest,
    })
}

impl super::super::Study {
    pub(in crate::analysis::execute) fn execute_temporal_class_response(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        self.execute_temporal_class_response_within(data, query, physical, &mut budget, ctx)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the class sweep retains horizon and completion state under one budget"
    )]
    pub(crate) fn execute_temporal_class_response_within(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "the class-aware temporal response",
        )?;
        if super::super::response_functional_is_derivative(&query.functional) {
            return self
                .execute_temporal_class_derivative_within(data, query, physical, budget, ctx);
        }
        let started = Instant::now();
        query.require_supported_temporal_observation()?;
        if matches!(
            query.observation,
            ObservationSpec::IntervalCensored { .. } | ObservationSpec::Truncated { .. }
        ) || self.observation_delayed_entry.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "class-aware temporal response refuses delayed entry, interval, and \
                          truncation observation mechanisms",
            });
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware temporal response requires TemporalResponseSpec".into(),
        })?;
        if antecedent_estimate::requires_temporal_overlays(query) {
            return self
                .execute_temporal_class_sequence_response(data, query, physical, budget, ctx);
        }
        enforce_temporal_response_memory_budget(
            query,
            temporal,
            self.settings.bootstrap_replicates,
            ctx,
        )?;
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let sweep = ClassResponseSweep { data, query, temporal, treatment, outcome, ctx };
        let mut run = ClassResponseRun {
            assembly: ClassResponseAssembly::new(query, temporal)?,
            primary_estimand: None,
            primary_identification: None,
            primary_envelope: None,
            assumptions: antecedent_core::AssumptionSet::new(),
            diagnostics: Vec::new(),
            observation_atoms: Vec::new(),
            witness_sites: Vec::new(),
            checked_witnesses: Vec::new(),
            checked_witness_leases: Vec::new(),
        };
        let mut class_weights = TemporalClassWeights::new(self.declared.class_prior.as_ref());
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            let effect_query =
                class_horizon_witness(query, temporal, (treatment, outcome), horizon);
            let bundle = self.identify_temporal_class_reusing_within(
                IdentifierId::GeneralizedAdjustment,
                &effect_query,
                physical,
                budget,
                ctx,
            )?;
            let envelope = &bundle.envelope.envelope;
            run.diagnostics.push(temporal_class_envelope_diagnostic(envelope, self.graph.class()));
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            run.assembly.begin_horizon(envelope);
            if self.retain_response_witnesses {
                // Fund the potential witness entries, growth copy, lease vector and
                // shared owner before either vector may allocate.
                let count =
                    u64::try_from(envelope.cases.len()).map_err(|_| CausalError::Resource {
                        message: "class response witness count overflow".into(),
                    })?;
                let existing = u64::try_from(run.checked_witnesses.len()).map_err(|_| {
                    CausalError::Resource {
                        message: "class response witness count overflow".into(),
                    }
                })?;
                let owned_bytes = count
                    .checked_mul(256)
                    .and_then(|n| n.checked_add(256))
                    .ok_or_else(|| CausalError::Resource {
                        message: "class response witness bytes overflow".into(),
                    })?;
                let transient_bytes = count
                    .checked_add(existing)
                    .and_then(|n| n.checked_mul(256))
                    .and_then(|n| n.checked_add(256))
                    .ok_or_else(|| CausalError::Resource {
                        message: "class response witness growth overflow".into(),
                    })?;
                if owned_bytes > budget.limits().max_bytes
                    || transient_bytes > budget.limits().max_bytes
                {
                    return Err(CausalError::Resource {
                        message: "class response witness byte limit".into(),
                    });
                }
                let admit = |bytes| {
                    ctx.acquire_admission(0, bytes).map_err(|error| match error {
                        antecedent_core::execution::AdmissionError::Cancelled => {
                            CausalError::Cancelled { stage: "class response witnesses" }
                        }
                        other => CausalError::Resource { message: other.to_string() },
                    })
                };
                let lease = admit(owned_bytes)?;
                let transient = admit(transient_bytes)?;
                run.checked_witnesses
                    .try_reserve_exact(envelope.cases.len())
                    .map_err(|error| CausalError::Resource { message: error.to_string() })?;
                run.checked_witness_leases
                    .try_reserve_exact(1)
                    .map_err(|error| CausalError::Resource { message: error.to_string() })?;
                drop(transient);
                run.checked_witness_leases.push(lease);
            }
            let mut fits = ClassResponseHorizonFits::default();
            for (case_index, (case, indexer)) in
                envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
            {
                self.class_response_completion(
                    &sweep,
                    &mut run,
                    &mut fits,
                    budget,
                    ClassResponseSite {
                        horizon_index,
                        horizon,
                        case_index,
                        weight: weights[case_index],
                        case,
                        indexer,
                        envelope: &bundle.envelope,
                    },
                )?;
            }
            run.assembly.end_horizon(horizon_index, horizon)?;
        }
        self.finish_temporal_class_response(&sweep, run, physical, started)
    }

    /// Fits one completion's temporal response at a horizon and adds it to the
    /// class assembly as an atom, or as unevaluated when it cannot be fitted.
    #[allow(
        clippy::too_many_lines,
        reason = "one completion evaluates and retains its response and validation identity"
    )]
    fn class_response_completion(
        &self,
        sweep: &ClassResponseSweep<'_>,
        run: &mut ClassResponseRun,
        fits: &mut ClassResponseHorizonFits,
        budget: &mut antecedent_core::TemporalBudget,
        site: ClassResponseSite<'_>,
    ) -> Result<(), CausalError> {
        let ClassResponseSweep { data, query, temporal, .. } = *sweep;
        let ClassResponseSite {
            horizon_index,
            horizon,
            case_index,
            weight,
            case,
            indexer,
            envelope,
        } = site;
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            run.assembly.push_unevaluated(horizon_index, case_index, weight, case.result.status);
            return Ok(());
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let mut qh = query.clone();
        let mut temporal_h = temporal.clone();
        temporal_h.horizons = Arc::from([horizon]);
        qh.temporal = Some(temporal_h);
        let observation_query = qh.clone();
        let adjustment = class_atom_adjustment(&estimand, indexer);
        let series_owned = if qh.observation == ObservationSpec::Complete
            || self.settings.inference.is_bayesian()
        {
            None
        } else {
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not supported for temporal observation",
                });
            }
            let outcome_regressors = antecedent_estimate::temporal_curve_outcome_regressors(
                &qh,
                &[(&estimand, indexer)],
            )
            .map_err(CausalError::from)?;
            let (series, _) = ObservationMechanismEstimator::new(self.observation_options)
                .adjust_temporal_series(data, &qh, &adjustment, &outcome_regressors)
                .map_err(CausalError::from)?;
            qh.observation = ObservationSpec::Complete;
            qh.observation_assumptions = Arc::from([]);
            Some(series)
        };
        let series = series_owned.as_ref().unwrap_or(data);
        let input = ClassResponseInput { series, query: &qh, estimand: &estimand, indexer };
        let Some(mut response) =
            self.class_completion_response(sweep, run, fits, &site, input, series_owned.is_some())?
        else {
            run.assembly.push_unevaluated(horizon_index, case_index, weight, case.result.status);
            return Ok(());
        };
        // The class response pools each completion's per-cell band into one
        // shared draws-agnostic band (published with `draws: None`) and rejects
        // a band that carries retained draws. A single-coordinate
        // `InterventionResponse` band now retains its posterior draws (only the
        // single-atom graph-posterior scalar path consumes them), so drop them
        // here to keep the completions poolable and the class band unchanged.
        if let ResponseUncertainty::PointwiseBand { draws, .. } = &mut response.uncertainty {
            *draws = None;
        }
        let atom_assumptions = response.assumptions.clone();
        let atom_index = run.assembly.push_atom(
            horizon_index,
            case_index,
            weight,
            case.result.status,
            horizon,
            response,
        )?;
        if self.retain_response_witnesses {
            run.checked_witnesses
                .push(class_response_witness(&site, atom_index, budget, sweep.ctx)?);
        }
        if !matches!(self.settings.refute, RefuteSuite::None)
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. })
        {
            run.witness_sites.push(super::super::temporal_response_refutation::PulseWitnessSite {
                key: class_completion_key(horizon_index, case_index),
                horizon,
                weight,
                witness: super::super::temporal_response_refutation::level_witness(query, horizon)?,
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                status: case.result.status,
                assumptions: case.result.required_assumptions.clone(),
            });
        }
        if series_owned.is_some() && self.settings.inference.is_frequentist() {
            run.observation_atoms.push(ClassObservationAtom {
                atom_index,
                horizon_index,
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                assumptions: case.result.required_assumptions.clone(),
                weight,
                adjustment,
                observation_query,
                kind: ClassObservationKind::Curve,
            });
        }
        if run.primary_estimand.is_none() {
            run.primary_estimand = Some(estimand);
            run.primary_identification = Some(case.result.clone());
            run.primary_envelope = Some(envelope.clone());
            run.assumptions = atom_assumptions;
        }
        Ok(())
    }

    /// The response of one completion under the study's inference mode, or
    /// `None` when the observed-data Bayesian path has no sequential DAG for
    /// the completion.
    fn class_completion_response(
        &self,
        sweep: &ClassResponseSweep<'_>,
        run: &mut ClassResponseRun,
        fits: &mut ClassResponseHorizonFits,
        site: &ClassResponseSite<'_>,
        input: ClassResponseInput<'_>,
        series_adjusted: bool,
    ) -> Result<Option<CausalResponse>, CausalError> {
        let ClassResponseSweep { data, query, treatment, outcome, ctx, .. } = *sweep;
        let ClassResponseSite { horizon, case, indexer, .. } = *site;
        let ClassResponseInput { series: _, query: qh, estimand, indexer: _ } = input;
        Ok(Some(match &self.settings.inference {
            InferenceMode::Bayesian(cfg) if query.observation != ObservationSpec::Complete => {
                let Some(dag) = case.graph.sequential_dag() else {
                    return Ok(None);
                };
                if cfg.has_transferred_source() {
                    return Err(CausalError::Unsupported {
                        message: "prior_artifact stays refused on observed temporal Bayes",
                    });
                }
                let bayes = bayesian_gcomp(cfg, ctx);
                let (response, _) =
                    antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response(data, &dag, antecedent_estimate::TemporalIdentification { horizons: &[(estimand, indexer)], status: case.result.status, assumptions: case.result.required_assumptions.clone() }, qh, &bayes, ctx)
                    .map_err(CausalError::from)?;
                response
            }
            InferenceMode::Bayesian(cfg) => {
                let (response, conflict) = fit_class_bayesian_response(
                    &mut fits.bayesian,
                    cfg,
                    input,
                    horizon,
                    &case.result,
                    (treatment, outcome),
                    ctx,
                )?;
                if let Some(summary) = conflict.as_ref() {
                    push_conflict_diagnostics(&mut run.diagnostics, summary);
                }
                response
            }
            InferenceMode::Frequentist => {
                // Requested replicates ride each completion atom. An
                // observation-adjusted series keeps the analytic fit: a
                // naive resample would not refit the nuisance mechanism.
                let mut estimator = TemporalResponseEstimator::new();
                estimator.inner.inference.bootstrap_replicates =
                    if series_adjusted { 0 } else { self.settings.bootstrap_replicates };
                // An observation-adjusted series is built per completion
                // from its own adjustment, so only the unadjusted series
                // shares fits.
                let memo = (!series_adjusted).then_some(&mut fits.frequentist);
                fit_class_frequentist_response(
                    memo,
                    &estimator,
                    input,
                    case.result.status,
                    &case.result.required_assumptions,
                    ctx,
                )?
            }
        }))
    }

    /// Folds the swept class atoms into the class response: the observation
    /// circular-block band when a single completion identifies, the identified
    /// set diagnostics, the structural mixture and the identified-execute
    /// finish.
    #[allow(
        clippy::too_many_lines,
        reason = "one class response finish: the band, the identified-set diagnostics and the mixture read the same swept atoms"
    )]
    fn finish_temporal_class_response(
        &self,
        sweep: &ClassResponseSweep<'_>,
        run: ClassResponseRun,
        physical: &PhysicalExecutionPlan,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let ClassResponseSweep { data, query, temporal, treatment, outcome, ctx } = *sweep;
        let ClassResponseRun {
            mut assembly,
            primary_estimand,
            primary_identification,
            primary_envelope,
            mut assumptions,
            mut diagnostics,
            observation_atoms,
            witness_sites,
            checked_witnesses,
            checked_witness_leases,
        } = run;
        // A class intervention level's cheap/full suite is the per-completion,
        // per-horizon Pulse-witness refuter state (unsupported).
        let (refutations, predictive_checks) = if witness_sites.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let (reports, checks, notes) =
                super::super::temporal_response_refutation::refute_temporal_class_intervention(
                    self,
                    data,
                    &witness_sites,
                    ctx,
                )?;
            diagnostics.retain(|diagnostic| {
                diagnostic.code.as_ref() != "refute.temporal_response.skipped"
            });
            diagnostics.extend(notes);
            (reports, checks)
        };
        let cells_per_horizon = assembly.cells_per_horizon();
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(query, &mut assumptions);
        }
        let mut parts = assembly.finish(
            query,
            temporal,
            assumptions.clone(),
            "estimate.temporal_response.class_envelope",
        )?;
        let mut structural_atoms = std::mem::take(&mut assembly.structural_atoms);
        let class_observation_band_applied = self.apply_class_observation_band(
            ClassBandRequest {
                data,
                query,
                atoms: &observation_atoms,
                cells_per_horizon,
                n_horizons: temporal.horizons.len(),
                full_mass_scope: assembly.full_mass_scope,
                unidentified_mass: parts.unidentified_mass,
                unevaluable_mass: parts.unevaluable_mass,
                ctx,
            },
            ClassBootstrapTargets {
                structural_atoms: &mut structural_atoms,
                response: &mut parts.response,
                diagnostics: &mut diagnostics,
            },
        )?;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class response missing estimand".into(),
        })?;
        let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "temporal class response missing identification".into(),
        })?;
        identification.status = parts.response.identification_status;
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
        }
        diagnostics.extend(class_response_identified_set_diagnostics(
            self.settings.inference.is_bayesian(),
            self.declared.class_prior.is_some(),
        ));
        if query.observation != ObservationSpec::Complete {
            diagnostics.push(class_observation_band_note(class_observation_band_applied));
        }
        let structural = self.class_structural_mixture(
            structural_atoms,
            &parts,
            (temporal.horizons.len(), assembly.full_mass_scope, assembly.truncated_atoms),
        );
        let mut result = self.finish_class_response_tail(ClassResponseTail {
            physical,
            identification,
            estimand,
            estimate: nan_effect(),
            identifier_id: IdentifierId::GeneralizedAdjustment,
            endpoints: (treatment, outcome),
            envelope: primary_envelope,
            response: parts.response,
            structural,
            diagnostics,
            identify_cached: self.temporal_class_cache_covers(&temporal.horizons),
            // Observation circular-block bands ride the class response when a single
            // completion identifies; otherwise requested replicates stay on the atoms and
            // the class band is withheld.
            bootstrap_replicates: class_observation_band_applied
                .then_some(self.settings.bootstrap_replicates),
            started,
            ctx,
            refutations,
            identify_provenance: None,
            estimate_provenance: None,
        });
        result.predictive_checks = predictive_checks;
        if self.retain_response_witnesses {
            result.temporal_response_witnesses =
                Some(Arc::new(crate::result::TemporalResponseWitnessSet {
                    entries: checked_witnesses,
                    _leases: checked_witness_leases,
                }));
        }
        Ok(result)
    }

    /// The observation circular-block band over the class atoms of an observed
    /// Frequentist query; reports whether it rides the class response, which
    /// needs a single identified completion over the full class mass.
    fn apply_class_observation_band(
        &self,
        request: ClassBandRequest<'_>,
        targets: ClassBootstrapTargets<'_>,
    ) -> Result<bool, CausalError> {
        let ClassBandRequest {
            data,
            query,
            atoms,
            cells_per_horizon,
            n_horizons,
            full_mass_scope,
            unidentified_mass,
            unevaluable_mass,
            ctx,
        } = request;
        if query.observation == ObservationSpec::Complete
            || !self.settings.inference.is_frequentist()
            || self.settings.bootstrap_replicates == 0
        {
            return Ok(false);
        }
        let identified_cases = targets
            .structural_atoms
            .iter()
            .filter(|atom| atom.value.is_some())
            .map(|atom| atom.graph_key & 0xFFFF_FFFF)
            .collect::<std::collections::BTreeSet<_>>();
        let apply_class_band = identified_cases.len() == 1
            && full_mass_scope
            && unidentified_mass == 0.0
            && unevaluable_mass == 0.0;
        apply_class_observation_bootstrap(
            data,
            atoms,
            targets,
            ClassBandLayout { cells_per_horizon, n_horizons, apply_class_band },
            ObservationBootstrap {
                options: self.observation_options,
                replicates: self.settings.bootstrap_replicates,
            },
            ctx,
        )
    }

    /// The structural mixture over a class sweep's atoms, weighted by the
    /// caller's class prior or the completion enumeration.
    fn class_structural_mixture(
        &self,
        atoms: Vec<crate::result::StructuralResponseAtom>,
        parts: &ClassResponseParts,
        (n_horizons, full_mass_scope, truncated_atoms): (usize, bool, usize),
    ) -> crate::result::StructuralResponseMixture {
        let conditional = self.declared.class_prior.as_ref().and_then(|_| {
            temporal_class_response_mean(
                &atoms,
                &parts.response_envelope,
                n_horizons,
                full_mass_scope,
            )
        });
        crate::result::StructuralResponseMixture {
            weight_basis: if self.declared.class_prior.is_some() {
                crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
            } else {
                crate::result::StructuralWeightBasis::CompletionEnumeration
            },
            atoms,
            identified_mass: parts.identified_mass,
            unidentified_mass: parts.unidentified_mass,
            unevaluable_mass: parts.unevaluable_mass,
            subsampled_out_mass: 0.0,
            identified_set: Some(parts.response_envelope.clone()),
            identified_set_interval: None,
            conditional_on_identified: conditional,
            full_mass_scope,
            truncated_atoms,
            aggregation_policy: None,
        }
    }

    fn execute_temporal_class_sequence_response(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.declared.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "class-aware Sequence overlays require no discovery-estimation split",
            });
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence requires TemporalResponseSpec".into(),
        })?;
        let plan = antecedent_estimate::plan_from_response_query(query, budget, ctx)
            .map_err(CausalError::from)?
            .ok_or_else(|| CausalError::Compile {
                message: "class-aware Sequence missing intervention plan".into(),
            })?;
        let overlays = plan
            .admitted_mechanism_overlays(budget, ctx)
            .map_err(CausalError::from)?
            .ok_or_else(|| CausalError::Compile {
                message: "class-aware Sequence missing mechanism overlays".into(),
            })?;
        let (treatment, outcome) =
            super::super::response_path::response_primary_pair(&query.functional)?;
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.settings.inference {
            if cfg.has_transfer() {
                return Err(CausalError::Unsupported {
                    message: "multi-step Sequence transfer stays refused on incomplete classes",
                });
            }
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        let sweep = ClassSequenceSweep {
            data,
            query,
            temporal,
            overlays: &overlays,
            bayes: bayes.as_ref(),
            treatment,
            outcome,
            ctx,
        };
        let mut run = ClassSequenceRun::new();
        let mut class_weights = TemporalClassWeights::new(self.declared.class_prior.as_ref());
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            let effect_query =
                class_horizon_witness(query, temporal, (treatment, outcome), horizon);
            let bundle = self.identify_temporal_class_reusing_within(
                IdentifierId::GeneralizedAdjustment,
                &effect_query,
                physical,
                budget,
                ctx,
            )?;
            let envelope = &bundle.envelope.envelope;
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            run.horizon_fingerprints
                .push(envelope.cases.iter().map(|case| case.graph.fingerprint()).collect());
            run.full_mass_scope &= envelope.truncated_completions == 0;
            run.truncated_atoms += envelope.truncated_completions;
            let mut horizon_values = Vec::new();
            for (case_index, (case, indexer)) in
                envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
            {
                self.class_sequence_completion(
                    &sweep,
                    &mut run,
                    &mut horizon_values,
                    ClassResponseSite {
                        horizon_index,
                        horizon,
                        case_index,
                        weight: weights[case_index],
                        case,
                        indexer,
                        envelope: &bundle.envelope,
                    },
                )?;
            }
            if horizon_values.is_empty() {
                return Err(CausalError::Compile {
                    message: format!(
                        "class-aware Sequence has no evaluable directed completion at horizon {horizon}"
                    ),
                });
            }
            run.lower.push(horizon_values.iter().copied().fold(f64::INFINITY, f64::min));
            run.upper.push(horizon_values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        }
        self.finish_class_sequence_response(&sweep, run, physical, started)
    }

    /// Estimates one directed completion's Sequence effect at a horizon and
    /// records it as a structural atom, or as unevaluated when the completion
    /// is not identified or carries no sequential DAG.
    fn class_sequence_completion(
        &self,
        sweep: &ClassSequenceSweep<'_>,
        run: &mut ClassSequenceRun,
        horizon_values: &mut Vec<f64>,
        site: ClassResponseSite<'_>,
    ) -> Result<(), CausalError> {
        let ClassSequenceSweep { query, outcome, bayes, overlays, .. } = *sweep;
        let ClassResponseSite {
            horizon_index,
            horizon,
            case_index,
            weight,
            case,
            indexer,
            envelope,
        } = site;
        let evaluable = identification_status_ok_for_case(case.result.status)
            && !case.result.estimands.is_empty();
        let Some(dag) = evaluable.then(|| case.graph.sequential_dag()).flatten() else {
            run.structural_atoms.push(bare_class_atom(
                class_completion_key(horizon_index, case_index),
                weight,
                case.result.status,
            ));
            return Ok(());
        };
        let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let mut qh = query.clone();
        qh.temporal
            .as_mut()
            .ok_or_else(|| CausalError::Compile {
                message: "class sequence response query carries no temporal spec".into(),
            })?
            .horizons = Arc::from([horizon]);
        let observation_query = qh.clone();
        let adjustment = class_atom_adjustment(&estimand, indexer);
        let (effect, posterior) =
            self.class_sequence_effect(sweep, &site, &dag, (&estimand, &qh), &adjustment)?;
        horizon_values.push(effect.ate);
        run.structural_atoms.push(crate::result::StructuralResponseAtom {
            posterior,
            value: Some(ResponseValue::Scalar(effect.ate)),
            ..bare_class_atom(
                class_completion_key(horizon_index, case_index),
                weight,
                case.result.status,
            )
        });
        if query.observation != ObservationSpec::Complete && bayes.is_none() {
            run.observation_atoms.push(ClassObservationAtom {
                atom_index: run.structural_atoms.len() - 1,
                horizon_index,
                estimand: estimand.clone(),
                indexer: indexer.clone(),
                assumptions: case.result.required_assumptions.clone(),
                weight,
                adjustment,
                observation_query,
                kind: ClassObservationKind::Sequence {
                    dag: Box::new(dag.clone()),
                    overlays: (*overlays).clone(),
                    outcome,
                    outcome_offset,
                    status: case.result.status,
                },
            });
        }
        if run.primary_estimand.is_none() {
            run.primary_estimand = Some(estimand);
            run.primary_identification = Some(case.result.clone());
            run.primary_envelope = Some(envelope.clone());
            run.assumptions = effect.provenance.assumptions;
        }
        Ok(())
    }

    /// One directed completion's Sequence effect and posterior: the observed-data
    /// Bayesian fit when the query carries an observation mechanism and the
    /// study is Bayesian, otherwise the sequential-mechanism estimate on the
    /// (observation-adjusted) series.
    fn class_sequence_effect(
        &self,
        sweep: &ClassSequenceSweep<'_>,
        site: &ClassResponseSite<'_>,
        dag: &TemporalDag,
        (estimand, qh): (&IdentifiedEstimand, &ResponseQuery),
        adjustment: &[VariableId],
    ) -> Result<(EffectEstimate, Option<CausalPosterior>), CausalError> {
        let ClassSequenceSweep { data, query, overlays, bayes, outcome, ctx, .. } = *sweep;
        let ClassResponseSite { horizon, case, indexer, .. } = *site;
        let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
        Ok(if let Some(bayes) = bayes.filter(|_| query.observation != ObservationSpec::Complete) {
            let (response, posterior) =
                antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response(
                    data,
                    dag,
                    antecedent_estimate::TemporalIdentification {
                        horizons: &[(estimand, indexer)],
                        status: case.result.status,
                        assumptions: case.result.required_assumptions.clone(),
                    },
                    qh,
                    bayes,
                    ctx,
                )
                .map_err(CausalError::from)?;
            // The observed-data posterior summarizes the sequence response as
            // its first quantity when no effect column is declared; never index
            // an absent summary.
            let column = posterior.effect_column().unwrap_or(0);
            let effect = EffectEstimate::new(
                posterior.summaries.mean.get(column).copied().unwrap_or(f64::NAN),
                posterior.summaries.sd.get(column).copied().unwrap_or(f64::NAN),
                response.assumptions,
                OverlapPolicy::ExplicitOverride,
            );
            (effect, Some(posterior))
        } else {
            let adjusted = if query.observation == ObservationSpec::Complete {
                None
            } else {
                Some(
                    ObservationMechanismEstimator::new(self.observation_options)
                        .adjust_temporal_series(
                            data,
                            qh,
                            adjustment,
                            &antecedent_estimate::temporal_sequence_outcome_regressors(
                                dag, outcome,
                            ),
                        )
                        .map_err(CausalError::from)?
                        .0,
                )
            };
            antecedent_estimate::estimate_sequence_mechanisms(
                &antecedent_estimate::SequentialProblem {
                    data: adjusted.as_ref().unwrap_or(data),
                    graph: dag,
                    indexer,
                    estimand,
                },
                &antecedent_estimate::SequenceTarget { outcome, outcome_offset, overlays },
                antecedent_estimate::SequentialInference {
                    status: case.result.status,
                    assumptions: case.result.required_assumptions.clone(),
                    bootstrap_replicates: 0,
                    bayesian: bayes,
                },
                ctx,
            )
            .map_err(CausalError::from)?
        })
    }

    /// The class Sequence response over the swept completions: the
    /// completion-envelope surface and, for an observed Frequentist query with
    /// one identified completion, its observation circular-block band.
    fn class_sequence_surface(
        &self,
        sweep: &ClassSequenceSweep<'_>,
        run: &mut ClassSequenceRun,
    ) -> Result<(ClassResponseParts, bool), CausalError> {
        let ClassSequenceSweep { data, query, temporal, ctx, .. } = *sweep;
        let response_envelope = antecedent_core::ResponseEnvelope {
            grid: Arc::from(temporal.horizons.iter().map(|&h| f64::from(h)).collect::<Vec<_>>()),
            dimension: 1,
            lower: Arc::from(std::mem::take(&mut run.lower)),
            upper: Arc::from(std::mem::take(&mut run.upper)),
        };
        let (identified_mass, unidentified_mass, unevaluable_mass) =
            temporal_class_response_masses(&run.structural_atoms);
        let incomplete = unidentified_mass > 0.0 || unevaluable_mass > 0.0 || !run.full_mass_scope;
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(query, &mut run.assumptions);
        }
        let mut response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: if incomplete {
                IdentificationStatus::GraphDependent
            } else {
                IdentificationStatus::PartiallyIdentified
            },
            estimate: if incomplete {
                ResponseIdentification::GraphDependent(temporal_class_complete_values(
                    &run.structural_atoms,
                    &run.horizon_fingerprints,
                    &response_envelope,
                    temporal.horizons.len(),
                ))
            } else {
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(
                    response_envelope.clone(),
                ))
            },
            uncertainty: ResponseUncertainty::None,
            support: antecedent_core::SupportReport {
                status: antecedent_core::SupportStatus::Extrapolative,
                query_region: antecedent_core::SupportRegion {
                    minima: Arc::from([f64::from(temporal.horizons.first().copied().unwrap_or(1))]),
                    maxima: Arc::from([f64::from(temporal.horizons.last().copied().unwrap_or(1))]),
                },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
                point_status: None,
            },
            assumptions: run.assumptions.clone(),
            provenance_id: Arc::from("estimate.temporal_response.class_sequence"),
            horizon_identification: Some(
                temporal
                    .horizons
                    .iter()
                    .map(|&horizon| antecedent_core::HorizonIdentification {
                        horizon,
                        status: if incomplete {
                            IdentificationStatus::GraphDependent
                        } else {
                            IdentificationStatus::PartiallyIdentified
                        },
                        method: Arc::from("temporal_class.completion_envelope"),
                        adjustment: Arc::from([]),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            ),
            interaction_structurally_zero: false,
        };
        let class_observation_band_applied = self.apply_class_observation_band(
            ClassBandRequest {
                data,
                query,
                atoms: &run.observation_atoms,
                cells_per_horizon: 1,
                n_horizons: temporal.horizons.len(),
                full_mass_scope: run.full_mass_scope,
                unidentified_mass,
                unevaluable_mass,
                ctx,
            },
            ClassBootstrapTargets {
                structural_atoms: &mut run.structural_atoms,
                response: &mut response,
                diagnostics: &mut run.diagnostics,
            },
        )?;
        Ok((
            ClassResponseParts {
                response,
                response_envelope,
                identified_mass,
                unidentified_mass,
                unevaluable_mass,
            },
            class_observation_band_applied,
        ))
    }

    /// Packages the class Sequence response with its structural mixture and
    /// identified-execute finish.
    fn finish_class_sequence_response(
        &self,
        sweep: &ClassSequenceSweep<'_>,
        mut run: ClassSequenceRun,
        physical: &PhysicalExecutionPlan,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let ClassSequenceSweep { query, temporal, treatment, outcome, ctx, .. } = *sweep;
        let (parts, class_observation_band_applied) =
            self.class_sequence_surface(sweep, &mut run)?;
        let ClassSequenceRun {
            structural_atoms,
            primary_estimand,
            primary_identification,
            primary_envelope,
            full_mass_scope,
            truncated_atoms,
            mut diagnostics,
            ..
        } = run;
        if query.observation != ObservationSpec::Complete {
            diagnostics.push(class_observation_band_note(class_observation_band_applied));
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence missing estimand".into(),
        })?;
        let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence missing identification".into(),
        })?;
        identification.status = parts.response.identification_status;
        if query.observation != ObservationSpec::Complete {
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
        }
        let structural = self.class_structural_mixture(
            structural_atoms,
            &parts,
            (temporal.horizons.len(), full_mass_scope, truncated_atoms),
        );
        Ok(self.finish_class_response_tail(ClassResponseTail {
            physical,
            identification,
            estimand,
            estimate: nan_effect(),
            identifier_id: IdentifierId::GeneralizedAdjustment,
            endpoints: (treatment, outcome),
            envelope: primary_envelope,
            response: parts.response,
            structural,
            diagnostics,
            identify_cached: self.temporal_class_cache_covers(&temporal.horizons),
            bootstrap_replicates: class_observation_band_applied
                .then_some(self.settings.bootstrap_replicates),
            started,
            ctx,
            refutations: Vec::new(),
            identify_provenance: None,
            estimate_provenance: None,
        }))
    }
}
