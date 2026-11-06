//! DBN-posterior temporal-effect execution.

use super::{
    Arc, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    BayesianValidation, BootstrapAccounting, CausalError, CausalPosterior, DBN_LINEAR_SEM,
    DbnAtomDemotion, DbnIdentifyDemotion, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EnvelopeAtomFit, EnvelopeOptions, EnvelopeRefutation, EnvelopeRefuteAtom,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphEffectDraws, GraphIdentFlag,
    GraphPosterior, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId,
    InferenceDiagnostics, InferenceMode, Instant, OverlapPolicy, PhysicalExecutionPlan,
    PosteriorMassSummary, PredictiveCheckReport, PriorSet, RefuteSuite, ResponseValue,
    SUSTAINED_LINEAR_SEM, SequentialValidation, SequentialValidationAtom, SharedCircularBlockSe,
    Strategy, StudyResult, TableView, TabularData, TemporalAtomDesign, TemporalEffectQuery,
    TemporalIndexer, TemporalLinearAdjustment, TemporalRefutation, TimeSeriesData, ValidationSink,
    VariableId, WeightedGraphSamples, aggregate_effect_envelope, bayesian_gcomp,
    bayesian_temporal_gcomp, effect_from_posterior, elapsed_ns, envelope_draws_from_posterior,
    envelope_mass_diagnostic, envelope_refute_atoms, first_key_positions, identified_envelope_keys,
    maybe_interactive_envelope_subsample, merge_posterior_notes, overlap_diagnostic,
    overlap_limit_diagnostic, provenance_ids, push_conflict_diagnostics,
    report_subsampled_out_mass, resolve_envelope_prior_anchor, resolve_posterior_status,
    run_envelope_bayesian_full_validation, sequential_linear_sem_assumption,
    shared_block_mixture_message, shared_circular_block_mixture_se, shared_gap_report,
    short_series_warning, temporal_refuter_average, validate_sequential, with_conflict_summary,
};

/// Per-atom designs and estimates of a DBN frequentist mixture before the shared-block SE.
struct DbnFrequentistAtoms<'a> {
    contexts: Vec<(&'a super::super::super::prepared::CachedDbnPosteriorAtomIdentification, f64)>,
    atom_estimates: Vec<EffectEstimate>,
    designs: Vec<TemporalAtomDesign>,
    point_sum: f64,
    point_mass: f64,
    primary: Option<(IdentifiedEstimand, IdentificationResult, antecedent_core::AssumptionSet)>,
    /// Each atom's windows on the original window axis, when the series has missing time points.
    windows: Vec<antecedent_estimate::GapSelection>,
}

/// Published DBN frequentist estimate with its shared-block bootstrap accounting.
struct DbnFrequentistBlockEstimate {
    estimate: EffectEstimate,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    block: SharedCircularBlockSe,
    point: f64,
}

/// Borrowed inputs of the DBN frequentist refutation phase.
#[derive(Clone, Copy)]
struct DbnFrequentistRefuteInput<'a> {
    data: &'a TimeSeriesData,
    gp: &'a GraphPosterior,
    query: &'a TemporalEffectQuery,
    vars: &'a [VariableId],
    estimate_ate: f64,
}

/// Mixture summary and short-series diagnostics of the DBN frequentist shared-block bootstrap.
fn push_dbn_frequentist_diagnostics(
    diagnostics: &mut Vec<Diagnostic>,
    block: &SharedCircularBlockSe,
    point_mass: f64,
    unidentified_mass: f64,
) {
    let se = block.se;
    diagnostics.push(
        Diagnostic::new(
            "estimate.dbn_posterior.frequentist",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "{}; bands require two successes and at most half failed attempts",
                shared_block_mixture_message(
                    "fixed graph weights",
                    point_mass,
                    unidentified_mass,
                    block,
                )
            ),
        )
        .with_fields(super::super::mass_fields(Some(point_mass), unidentified_mass)),
    );
    if se.is_finite() {
        diagnostics.extend(short_series_warning(
            block.effective_rows,
            antecedent_estimate::CircularBlockFamily::Mixture,
        ));
    }
}

/// Resolved inputs shared by the DBN Bayesian per-atom fits.
#[derive(Clone, Copy)]
struct DbnBayesianRun<'a> {
    data: &'a TimeSeriesData,
    gp: &'a GraphPosterior,
    vars: &'a [VariableId],
    query: &'a TemporalEffectQuery,
    cfg: &'a crate::inference::BayesianConfig,
    sequential_bayes: Option<&'a BayesianGComputationAte>,
    multi_step: bool,
    /// The series has missing time points: atoms fit their intact lag windows.
    gaps: bool,
}

/// One identified DBN atom handed to a per-atom fit.
struct DbnFitAtom {
    key: u64,
    estimand: IdentifiedEstimand,
    identification: IdentificationResult,
    indexer: TemporalIndexer,
}

/// Mutable state threaded through the DBN Bayesian per-atom fits.
struct DbnBayesianAtomFits {
    key_weights: IdentifiedKeyWeights,
    key_index: std::collections::HashMap<u64, usize>,
    flags: Vec<GraphIdentFlag>,
    atom_contexts: Vec<(u64, IdentifiedEstimand, IdentificationResult, TemporalIndexer)>,
    atoms: Vec<EnvelopeAtomFit>,
    envelope_prior: Option<PriorSet>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
    per_graph: Vec<GraphEffectDraws>,
    sequential_atoms: Vec<SequentialValidationAtom>,
    /// Graph key of each sequential validation atom, for the Interactive subsample.
    sequential_keys: Vec<u64>,
    bayes: BayesianTemporalGcomp,
    ws: BayesianGCompWorkspace,
    prepare_demoted: usize,
    fit_demoted: usize,
    draws_demoted: usize,
    /// Identified weight of atoms whose estimation (unfolding, preparation, fitting or
    /// draw extraction) failed: a refusal to estimate, kept apart from structural
    /// non-identification in `unidentified_mass`.
    unevaluable_weight: f64,
    /// Each fitted atom's windows on the original window axis, when the series has gaps.
    windows: Vec<antecedent_estimate::GapSelection>,
}

impl DbnBayesianAtomFits {
    /// Flag the atom unidentified for the envelope and book its weight as unevaluable.
    fn demote(&mut self, key: u64, reason: DbnAtomDemotion) {
        if let Some(idx) = self.key_index.get(&key).copied() {
            self.flags[idx] = GraphIdentFlag::Unidentified;
        }
        self.unevaluable_weight += self.key_weights.get(key);
        match reason {
            DbnAtomDemotion::Prepare => self.prepare_demoted += 1,
            DbnAtomDemotion::Fit => self.fit_demoted += 1,
            DbnAtomDemotion::Draws => self.draws_demoted += 1,
        }
    }
}

/// Aggregated Bayesian DBN envelope and the atoms it kept.
struct DbnBayesianEnvelope {
    posterior: CausalPosterior,
    estimate: EffectEstimate,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    subsample_notes: Vec<Diagnostic>,
    atoms: Vec<EnvelopeAtomFit>,
    sequential_atoms: Vec<SequentialValidationAtom>,
    bayes: BayesianTemporalGcomp,
    demoted: (usize, usize, usize),
}

/// Borrowed inputs of the Bayesian DBN validate stage.
#[derive(Clone, Copy)]
struct DbnBayesianValidationInput<'a> {
    run: &'a DbnBayesianRun<'a>,
    estimator: &'a BayesianGComputationAte,
    estimate_ate: f64,
}

/// Interactive-subsampled envelope of the fitted DBN atoms: the demotion-aware posterior,
/// the first contributing atom's identification, and the atoms and validation atoms kept.
fn aggregate_dbn_bayesian_envelope(
    latency_mode: Option<crate::LatencyMode>,
    identified_graphs: &WeightedGraphSamples,
    fits: DbnBayesianAtomFits,
    ctx: &ExecutionContext,
) -> Result<DbnBayesianEnvelope, CausalError> {
    let DbnBayesianAtomFits {
        flags,
        per_graph,
        mut atoms,
        atom_contexts,
        sequential_atoms,
        sequential_keys,
        unevaluable_weight,
        envelope_conflict,
        bayes,
        prepare_demoted,
        fit_demoted,
        draws_demoted,
        ..
    } = fits;
    let fitted_graphs = WeightedGraphSamples::new(
        Arc::clone(&identified_graphs.weights),
        flags,
        Arc::clone(&identified_graphs.graph_keys),
    )?;
    let mut subsample_notes = Vec::new();
    let (graphs, per_graph, subsample_drop) = maybe_interactive_envelope_subsample(
        latency_mode,
        fitted_graphs.clone(),
        per_graph,
        ctx,
        &mut subsample_notes,
    )?;
    let keep = identified_envelope_keys(&graphs);
    atoms.retain(|atom| keep.contains(&atom.key));
    let (_, estimand, mut identification, _) = atom_contexts
        .into_iter()
        .find(|(key, _, _, _)| keep.contains(key))
        .ok_or_else(|| CausalError::Compile {
            message: "DBN posterior envelope has no contributing context".into(),
        })?;
    let graph_key_weights = IdentifiedKeyWeights::new(&graphs);
    for atom in &mut atoms {
        atom.weight = graph_key_weights.get(atom.key);
    }
    // Validation refits only the atoms the mixture kept, at their kept weights.
    let sequential_atoms: Vec<SequentialValidationAtom> = sequential_atoms
        .into_iter()
        .zip(sequential_keys)
        .filter(|(_, key)| keep.contains(key))
        .map(|(mut atom, key)| {
            atom.weight = graph_key_weights.get(key);
            atom
        })
        .collect();
    let mut posterior = aggregate_effect_envelope(
        &graphs,
        &per_graph,
        InferenceDiagnostics::analytic("dbn_posterior_envelope"),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    report_subsampled_out_mass(&mut posterior, &fitted_graphs, &subsample_drop);
    // The estimation-failure demotions above are flagged `Unidentified`
    // (the envelope has no third state); split their mass back out so a
    // refusal to estimate is never published as non-identification.
    let total_weight = fitted_graphs.total_weight();
    let unevaluable_mass = if total_weight > 0.0 { unevaluable_weight / total_weight } else { 0.0 };
    posterior.unevaluable_mass = unevaluable_mass;
    posterior.unidentified_mass = (posterior.unidentified_mass - unevaluable_mass).max(0.0);
    // Mass the mixture does not cover, of either kind, leaves it graph-dependent.
    if posterior.identification == IdentificationStatus::GraphDependent {
        identification.status = IdentificationStatus::GraphDependent;
    }
    merge_posterior_notes(
        &mut posterior,
        atoms.iter().map(|atom| &atom.posterior).chain(
            sequential_atoms.iter().flat_map(|atom| atom.mechanisms.iter().map(|m| &m.posterior)),
        ),
    );
    if let Some(summary) = envelope_conflict {
        posterior = with_conflict_summary(posterior, summary);
    }
    let estimate = effect_from_posterior(&posterior)?;
    Ok(DbnBayesianEnvelope {
        posterior,
        estimate,
        identification,
        estimand,
        subsample_notes,
        atoms,
        sequential_atoms,
        bayes,
        demoted: (prepare_demoted, fit_demoted, draws_demoted),
    })
}

/// Diagnostics of the Bayesian DBN-posterior envelope: identification's, subsample notes,
/// overlap, mass, atom demotions, prior conflict, and the sequential-window note.
fn dbn_bayesian_diagnostics(
    identification: &IdentificationResult,
    subsample_notes: Vec<Diagnostic>,
    estimate: &EffectEstimate,
    posterior: &CausalPosterior,
    identify_demotion: &DbnIdentifyDemotion,
    demoted: (usize, usize, usize),
    multi_step: bool,
) -> Vec<Diagnostic> {
    let (prepare_demoted, fit_demoted, draws_demoted) = demoted;
    let mut diagnostics = identification.diagnostics.clone();
    diagnostics.extend(subsample_notes);
    diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
    diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
    diagnostics.push(envelope_mass_diagnostic("estimate.dbn_posterior.envelope", posterior));
    diagnostics.push(Diagnostic::new(
        "estimate.dbn_posterior.atom_demotion",
        DiagnosticKind::Scientific,
        if prepare_demoted + fit_demoted + draws_demoted > 0 {
            DiagnosticSeverity::Warning
        } else {
            DiagnosticSeverity::Info
        },
        identify_demotion.summary(prepare_demoted, fit_demoted, draws_demoted),
    ));
    if let Some(cs) = posterior.conflict_summary.as_ref() {
        push_conflict_diagnostics(&mut diagnostics, cs);
    }
    if multi_step {
        diagnostics.push(Diagnostic::new(
            "estimate.temporal.sustained_window",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the contrast propagates through all intervened times in topological order; \
             no one-node collapse; no analytic SE is asserted for the sequential fit",
        ));
    }
    diagnostics
}

impl super::super::Study {
    /// Execute a supplied DBN posterior with fixed graph weights and a shared
    /// outer circular-block bootstrap across every contributing atom.
    pub(in crate::analysis::execute) fn execute_dbn_posterior_frequentist(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "the DBN graph-posterior effect",
        )?;
        let started = Instant::now();
        if query.is_multi_step_sustained() && self.declared.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "multi-step sustained g-computation currently requires no discovery-estimation split",
            });
        }
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let (identified, identify_cached) =
            self.dbn_posterior_identification(gp, &vars, query, physical, ctx)?;
        let gaps = self.admit_posterior_gaps(data, &vars, false)?;
        let mut atoms = Self::estimate_dbn_posterior_frequentist_atoms(
            data,
            gp,
            query,
            &vars,
            (&identified, gaps),
            self.declared.split.as_ref(),
            ctx,
        )?;
        let DbnFrequentistBlockEstimate { estimate, identification, estimand, block, point } =
            self.dbn_frequentist_block_estimate(gp, &identified.graphs, &mut atoms, ctx)?;
        let (refutations, mut diagnostics) = self.refute_dbn_posterior_frequentist(
            &DbnFrequentistRefuteInput { data, gp, query, vars: &vars, estimate_ate: estimate.ate },
            &atoms,
            ctx,
        )?;
        push_dbn_frequentist_diagnostics(
            &mut diagnostics,
            &block,
            atoms.point_mass,
            identified.graphs.unidentified_mass(),
        );
        let structural_response = dbn_frequentist_structural_mixture(
            &identified.graphs,
            &atoms.contexts,
            &atoms.atom_estimates,
            point,
        );
        let missing_data =
            gaps.then(|| shared_gap_report(data, &vars, &atoms.windows)).transpose()?;
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    if query.is_multi_step_sustained() {
                        EstimatorId::TemporalSequentialGcomp
                    } else {
                        EstimatorId::TemporalLinearAdjustment
                    },
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(
                (self.settings.bootstrap_replicates > 0).then_some(block.completed),
            )
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.dbn_posterior.frequentist",
                    "estimate.temporal.linear.adjustment",
                )),
                diagnostics: Some(diagnostics),
                structural_response: Some(structural_response),
                ..Default::default()
            }),
        );
        result.extras.missing_data = missing_data;
        Ok(result)
    }

    /// The frozen DBN-posterior identification: the prepared cache, the product the
    /// plan's lowered program retained at compile, or a fresh build.
    fn dbn_posterior_identification(
        &self,
        gp: &GraphPosterior,
        vars: &[VariableId],
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<
        (Arc<super::super::super::prepared::CachedDbnPosteriorIdentification>, bool),
        CausalError,
    > {
        Ok(if let Some(cache) = self.dbn_posterior_cache() {
            (Arc::clone(cache), true)
        } else if let Some(cache) =
            crate::analysis::lowering::plan_retained_dbn_posterior(self, physical)
        {
            (cache, false)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_dbn_posterior_identification_cache(
                    gp, vars, query, ctx,
                )?),
                false,
            )
        })
    }

    /// Per-atom linear (or sequential) temporal designs and estimates for the DBN atoms with
    /// positive weight, with the primary atom's estimand, identification and assumptions.
    fn estimate_dbn_posterior_frequentist_atoms<'a>(
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        vars: &[VariableId],
        (identified, gaps): (
            &'a super::super::super::prepared::CachedDbnPosteriorIdentification,
            bool,
        ),
        split: Option<&antecedent_data::DiscoveryEstimationSplit>,
        ctx: &ExecutionContext,
    ) -> Result<DbnFrequentistAtoms<'a>, CausalError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let mut contexts = Vec::new();
        let mut atom_estimates = Vec::new();
        let mut designs = Vec::new();
        let mut windows = Vec::new();
        let mut point_sum = 0.0;
        let mut point_mass = 0.0;
        let mut primary = None;
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        for atom in identified.atoms.iter() {
            let weight = key_weights.get(atom.key);
            if weight <= 0.0 {
                continue;
            }
            let mut assumptions = atom.identification.required_assumptions.clone();
            let design = if query.is_multi_step_sustained() {
                assumptions.push(sequential_linear_sem_assumption(DBN_LINEAR_SEM));
                let index = usize::try_from(atom.key).map_err(|_| CausalError::Resource {
                    message: "DBN atom index overflow".into(),
                })?;
                let (graph, _graph_lease) =
                    crate::analysis::identification_cache::admitted_temporal_dag_atom(
                        gp,
                        index,
                        vars,
                        &mut budget,
                        ctx,
                    )?
                    .ok_or_else(|| CausalError::Compile { message: "invalid DBN atom".into() })?;
                TemporalAtomDesign::sequential_with_gaps(
                    data,
                    &graph,
                    &atom.indexer,
                    &atom.estimand,
                    query,
                    atom.identification.status,
                    gaps,
                    ctx,
                )?
            } else {
                TemporalAtomDesign::linear_with_gaps(
                    data,
                    &atom.estimand,
                    query,
                    &atom.indexer,
                    split,
                    gaps,
                    ctx,
                )?
            };
            if gaps {
                windows.push(design.window_selection());
            }
            let estimate = design.effect_estimate(assumptions);
            point_sum += weight * estimate.ate;
            point_mass += weight;
            if primary.is_none() {
                primary = Some((
                    atom.estimand.clone(),
                    atom.identification.clone(),
                    estimate.provenance.assumptions.clone(),
                ));
            }
            contexts.push((atom, weight));
            atom_estimates.push(estimate);
            designs.push(design);
        }
        Ok(DbnFrequentistAtoms {
            contexts,
            atom_estimates,
            designs,
            point_sum,
            point_mass,
            primary,
            windows,
        })
    }

    /// Shared circular-block mixture SE over the atom designs and the published estimate; the
    /// identification status turns graph-dependent when unidentified mass is left over.
    fn dbn_frequentist_block_estimate(
        &self,
        gp: &GraphPosterior,
        identified_graphs: &WeightedGraphSamples,
        atoms: &mut DbnFrequentistAtoms<'_>,
        ctx: &ExecutionContext,
    ) -> Result<DbnFrequentistBlockEstimate, CausalError> {
        let (estimand, mut identification, assumptions) =
            atoms.primary.take().ok_or_else(|| CausalError::Compile {
                message: "Frequentist DBN posterior has no estimable identified atom".into(),
            })?;
        let point = atoms.point_sum / atoms.point_mass;
        let weights: Vec<f64> = atoms.contexts.iter().map(|(_, weight)| *weight).collect();
        let block = shared_circular_block_mixture_se(
            &atoms.designs.iter().collect::<Vec<_>>(),
            &weights,
            gp.max_lag().unwrap_or(1) as usize + 1,
            self.settings.bootstrap_replicates,
            0xDBF0_0000,
            ctx,
        );
        let se = block.se;
        if resolve_posterior_status(
            identification.status,
            crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
            PosteriorMassSummary {
                unidentified: identified_graphs.unidentified_mass(),
                ..PosteriorMassSummary::default()
            },
        ) == IdentificationStatus::GraphDependent
        {
            identification.status = IdentificationStatus::GraphDependent;
        }
        let estimate = EffectEstimate::from_parts(
            point,
            f64::NAN,
            BootstrapAccounting {
                se: se.is_finite().then_some(se),
                replicates_ok: (self.settings.bootstrap_replicates > 0).then_some(block.completed),
                replicates_failed: (self.settings.bootstrap_replicates > 0)
                    .then_some(block.attempted.saturating_sub(block.completed)),
                cancelled: ctx.cancellation.is_cancelled(),
                early_stopped: false,
            },
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        )
        .with_block_family(antecedent_estimate::CircularBlockFamily::Mixture);
        Ok(DbnFrequentistBlockEstimate { estimate, identification, estimand, block, point })
    }

    /// Refutations of the DBN frequentist mixture: sequential validation for multi-step
    /// sustained queries, the envelope refutation over atom estimates otherwise.
    fn refute_dbn_posterior_frequentist(
        &self,
        input: &DbnFrequentistRefuteInput<'_>,
        atoms: &DbnFrequentistAtoms<'_>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let DbnFrequentistRefuteInput { data, gp, query, vars, estimate_ate } = *input;
        let DbnFrequentistAtoms { contexts, atom_estimates, windows, .. } = atoms;
        // The atoms were fitted on the intact windows of a series with missing time points: the
        // refuters refit through them and resample blocks of the original window axis.
        let gaps = !windows.is_empty();
        let refute_atoms = contexts
            .iter()
            .zip(atom_estimates)
            .map(|((atom, weight), original)| EnvelopeRefuteAtom {
                key: atom.key,
                weight: *weight,
                estimand: atom.estimand.clone(),
                indexer: Some(atom.indexer.clone()),
                original: original.clone(),
            })
            .collect::<Vec<_>>();
        let tabular = TabularData::new(data.storage().clone());
        let ate_query = temporal_refuter_average(query);
        let (refutations, diagnostics) = if self.settings.refute == RefuteSuite::None
            && self.declared.custom_validators.is_empty()
        {
            (Vec::new(), Vec::new())
        } else if query.is_multi_step_sustained() {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            let validation_atoms = contexts
                .iter()
                .zip(atom_estimates)
                .map(|((atom, weight), original)| {
                    let index = usize::try_from(atom.key).map_err(|_| CausalError::Resource {
                        message: "DBN atom index overflow".into(),
                    })?;
                    let (graph, lease) =
                        crate::analysis::identification_cache::admitted_temporal_dag_atom(
                            gp,
                            index,
                            vars,
                            &mut budget,
                            ctx,
                        )?
                        .ok_or_else(|| CausalError::Compile {
                            message: "invalid DBN atom".into(),
                        })?;
                    Ok(SequentialValidationAtom {
                        weight: *weight,
                        graph: Arc::new(graph),
                        _graph_lease: Some(Arc::new(lease)),
                        indexer: atom.indexer.clone(),
                        estimand: atom.estimand.clone(),
                        status: atom.identification.status,
                        estimate: original.clone(),
                        mechanisms: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>, CausalError>>()?;
            let (reports, notes, _) = validate_sequential(
                SequentialValidation {
                    data,
                    query,
                    suite: self.settings.refute,
                    custom: &self.declared.custom_validators,
                    predictive_sims: super::super::super::latency::predictive_check_sims(
                        self.latency_mode,
                    ),
                    gaps,
                },
                &validation_atoms,
                None,
                None,
                estimate_ate,
                ctx,
            )?;
            (reports, notes)
        } else {
            EnvelopeRefutation {
                data: &tabular,
                query: &ate_query,
                suite: self.settings.refute,
                estimator: "temporal.linear.adjustment",
                custom: &self.declared.custom_validators,
                temporal: Some(TemporalRefutation {
                    query,
                    split: self.declared.split.as_ref(),
                    time_index: data.time_index(),
                    gaps,
                }),
            }
            .run(&refute_atoms, &mut EstimationWorkspace::default(), ctx)?
        };
        Ok((refutations, diagnostics))
    }

    /// Execute a supplied DBN (temporal) graph posterior into a Bayesian effect envelope.
    ///
    /// Same envelope discipline as [`Self::execute_graph_posterior_bayesian`], adapted to
    /// per-atom lag masks: each atom's contemporaneous + lagged adjacency is completed into
    /// a [`antecedent_graph::TemporalDag`], identified via [`TemporalBackdoorIdentifier`],
    /// and fit with [`BayesianTemporalGcomp`] (Pulse / single-step Sustained) or
    /// sequential g-computation (multi-step Sustained; no last-step collapse).
    ///
    /// # Errors
    ///
    /// `InferenceMode::Frequentist`, a posterior missing per-atom lag masks / `max_lag`
    /// (i.e. not actually a DBN posterior), or per-graph estimation infrastructure failure.
    #[allow(
        clippy::too_many_lines,
        reason = "the per-atom fits, the gap admission and the mixed report read one frozen posterior in a single pass"
    )]
    pub(in crate::analysis::execute) fn execute_dbn_posterior_bayesian(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "the DBN graph-posterior effect",
        )?;
        let started = Instant::now();
        let cfg = match &self.settings.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::unsupported_inference(
                    "DBN graph-posterior discovery requires inference=Bayesian for effect mixture",
                ));
            }
        };
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        let multi_step = query.is_multi_step_sustained();
        if multi_step {
            self.ensure_dbn_multi_step_bayesian_admitted(&cfg)?;
        }
        let gaps = self.admit_posterior_gaps(data, &vars, true)?;

        let bayes = bayesian_temporal_gcomp(&cfg, ctx);
        let sequential_bayes = if multi_step { Some(bayesian_gcomp(&cfg, ctx)) } else { None };

        let (identified, identify_cached) =
            self.dbn_posterior_identification(gp, &vars, query, physical, ctx)?;
        let run = DbnBayesianRun {
            data,
            gp,
            vars: &vars,
            query,
            cfg: &cfg,
            sequential_bayes: sequential_bayes.as_ref(),
            multi_step,
            gaps,
        };
        let mut fits = self.fit_dbn_posterior_bayesian_atoms(&run, &identified, bayes, ctx)?;
        let windows = std::mem::take(&mut fits.windows);
        let DbnBayesianEnvelope {
            mut posterior,
            estimate,
            identification,
            estimand,
            subsample_notes,
            atoms,
            sequential_atoms,
            bayes,
            demoted,
        } = aggregate_dbn_bayesian_envelope(self.latency_mode, &identified.graphs, fits, ctx)?;
        let mut diagnostics = dbn_bayesian_diagnostics(
            &identification,
            subsample_notes,
            &estimate,
            &posterior,
            &identified.identify_demotion,
            demoted,
            multi_step,
        );
        let (refutations, predictive_checks) = self.validate_dbn_posterior_bayesian(
            &DbnBayesianValidationInput {
                run: &run,
                estimator: &bayes.inner,
                estimate_ate: estimate.ate,
            },
            &atoms,
            &sequential_atoms,
            &mut posterior,
            &mut diagnostics,
            ctx,
        )?;

        let missing_data = gaps.then(|| shared_gap_report(data, &vars, &windows)).transpose()?;
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    if multi_step {
                        EstimatorId::TemporalSequentialGcomp
                    } else {
                        EstimatorId::BayesianTemporalGcomp
                    },
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    if multi_step {
                        "estimate.temporal.sequential.gcomp"
                    } else {
                        "estimate.bayesian.temporal.gcomp"
                    },
                )),
                posterior: Some(posterior),
                predictive_checks,
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        );
        result.extras.missing_data = missing_data;
        Ok(result)
    }

    /// Multi-step sustained DBN inference needs no discovery-estimation split and isotropic
    /// per-mechanism priors.
    fn ensure_dbn_multi_step_bayesian_admitted(
        &self,
        cfg: &crate::inference::BayesianConfig,
    ) -> Result<(), CausalError> {
        if self.declared.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "multi-step sustained g-computation currently requires no discovery-estimation split",
            });
        }
        if cfg.has_transfer() {
            return Err(CausalError::Unsupported {
                message: "multi-step sustained inference requires isotropic per-mechanism priors",
            });
        }
        Ok(())
    }

    /// Soft prepare and fit of every identified DBN atom before the Interactive subsample:
    /// failures are demoted so the stratified selection only chooses among atoms that
    /// already produced draws, and the prior anchors on the first successful prepare.
    fn fit_dbn_posterior_bayesian_atoms(
        &self,
        run: &DbnBayesianRun<'_>,
        identified: &super::super::super::prepared::CachedDbnPosteriorIdentification,
        bayes: BayesianTemporalGcomp,
        ctx: &ExecutionContext,
    ) -> Result<DbnBayesianAtomFits, CausalError> {
        let keys = identified.graphs.graph_keys.to_vec();
        let mut state = DbnBayesianAtomFits {
            key_weights: IdentifiedKeyWeights::new(&identified.graphs),
            key_index: first_key_positions(&keys),
            flags: identified.graphs.identified.to_vec(),
            atom_contexts: Vec::new(),
            atoms: Vec::new(),
            envelope_prior: None,
            envelope_conflict: None,
            per_graph: Vec::new(),
            sequential_atoms: Vec::new(),
            sequential_keys: Vec::new(),
            bayes,
            ws: BayesianGCompWorkspace::default(),
            prepare_demoted: 0,
            fit_demoted: 0,
            draws_demoted: 0,
            unevaluable_weight: 0.0,
            windows: Vec::new(),
        };
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        for atom in identified.atoms.iter() {
            let atom = DbnFitAtom {
                key: atom.key,
                estimand: atom.estimand.clone(),
                identification: atom.identification.clone(),
                indexer: atom.indexer.clone(),
            };
            if run.multi_step {
                self.fit_dbn_sequential_atom(run, &mut state, atom, &mut budget, ctx)?;
            } else {
                self.fit_dbn_linear_atom(run, &mut state, atom, ctx)?;
            }
        }
        Ok(state)
    }

    /// Multi-step sustained atom: unfold its DAG, fit the sequential window, and keep it
    /// as a validation atom when its posterior yields draws; otherwise demote it.
    fn fit_dbn_sequential_atom(
        &self,
        run: &DbnBayesianRun<'_>,
        state: &mut DbnBayesianAtomFits,
        atom: DbnFitAtom,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let DbnBayesianRun { data, gp, vars, query, sequential_bayes, gaps, .. } = *run;
        let DbnFitAtom { key, estimand, identification, indexer } = atom;
        let index = usize::try_from(key)
            .map_err(|_| CausalError::Resource { message: "DBN atom index overflow".into() })?;
        let Some((graph, graph_lease)) =
            crate::analysis::identification_cache::admitted_temporal_dag_atom(
                gp, index, vars, budget, ctx,
            )?
        else {
            state.demote(key, DbnAtomDemotion::Prepare);
            return Ok(());
        };
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(sequential_linear_sem_assumption(SUSTAINED_LINEAR_SEM));
        let mut mechanisms = Vec::new();
        let result =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_gaps_if_within(
                &antecedent_estimate::SequentialProblem {
                    data,
                    graph: &graph,
                    indexer: &indexer,
                    estimand: &estimand,
                },
                query,
                antecedent_estimate::SequentialInference {
                    status: identification.status,
                    assumptions,
                    bootstrap_replicates: self.settings.bootstrap_replicates,
                    bayesian: sequential_bayes,
                },
                ctx,
                Some(&mut mechanisms),
                gaps,
                budget,
            );
        let (atom_estimate, posterior, selection) = match result {
            Ok((estimate, Some(posterior), selection)) => (estimate, posterior, selection),
            Err(error) => {
                let error = CausalError::from(error);
                if matches!(error, CausalError::Cancelled { .. } | CausalError::Resource { .. }) {
                    return Err(error);
                }
                state.demote(key, DbnAtomDemotion::Fit);
                return Ok(());
            }
            Ok(_) => {
                state.demote(key, DbnAtomDemotion::Fit);
                return Ok(());
            }
        };
        if let Ok(draws) = envelope_draws_from_posterior(key, &posterior) {
            state.windows.extend(selection);
            state.sequential_keys.push(key);
            state.sequential_atoms.push(SequentialValidationAtom {
                weight: state.key_weights.get(key),
                graph: Arc::new(graph),
                _graph_lease: Some(Arc::new(graph_lease)),
                indexer: indexer.clone(),
                estimand: estimand.clone(),
                status: identification.status,
                estimate: atom_estimate,
                mechanisms,
            });
            state.atom_contexts.push((
                key,
                estimand.clone(),
                identification.clone(),
                indexer.clone(),
            ));
            state.per_graph.push(draws);
        } else {
            state.demote(key, DbnAtomDemotion::Draws);
        }
        Ok(())
    }

    /// Pulse / single-step atom: prepare the temporal design, resolve the shared prior
    /// anchor on the first success, fit, and keep the atom when it yields draws.
    fn fit_dbn_linear_atom(
        &self,
        run: &DbnBayesianRun<'_>,
        state: &mut DbnBayesianAtomFits,
        atom: DbnFitAtom,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let DbnBayesianRun { data, query, cfg, gaps, .. } = *run;
        let DbnFitAtom { key, estimand, identification, indexer } = atom;
        let mut temporal_est = TemporalLinearAdjustment::new();
        temporal_est.inner.overlap = OverlapPolicy::ExplicitOverride;
        let prepared = if gaps {
            temporal_est
                .prepare_with_gaps(data, &estimand, query, &indexer, &ctx.kernel_policy)
                .map(|(prep, selection)| (prep, Some(selection)))
        } else {
            temporal_est
                .prepare(
                    data,
                    &estimand,
                    query,
                    &indexer,
                    self.declared.split.as_ref(),
                    &ctx.kernel_policy,
                )
                .map(|prep| (prep, None))
        };
        let Ok((prep, selection)) = prepared else {
            state.demote(key, DbnAtomDemotion::Prepare);
            return Ok(());
        };
        let Ok(names) =
            antecedent_estimate::temporal_coefficient_names(data, &estimand, query, &indexer)
        else {
            state.demote(key, DbnAtomDemotion::Prepare);
            return Ok(());
        };
        // The likelihood is tempered by the segment-wise long-run variance over the intact
        // windows when the atom lost any, so no autocovariance crosses a gap.
        let bprep = match selection.as_ref().filter(|s| s.complete.iter().any(|intact| !intact)) {
            Some(selection) => BayesianGComputationAte::from_prepared_temporal_gapped(
                &prep,
                names,
                selection.segments(),
            ),
            None => BayesianGComputationAte::from_prepared_temporal(&prep, names),
        }
        .map_err(CausalError::from)?;
        if state.envelope_prior.is_none() {
            let (resolved, conflict) = resolve_envelope_prior_anchor(cfg, &bprep, ctx)?;
            state.envelope_prior = resolved;
            state.envelope_conflict = conflict;
        }
        state.bayes.inner.prior.clone_from(&state.envelope_prior);
        let Ok(posterior) = state.bayes.fit(&bprep, identification.status, &mut state.ws, ctx)
        else {
            state.demote(key, DbnAtomDemotion::Fit);
            return Ok(());
        };
        if let Ok(draws) = envelope_draws_from_posterior(key, &posterior) {
            state.windows.extend(selection);
            state.atom_contexts.push((
                key,
                estimand.clone(),
                identification.clone(),
                indexer.clone(),
            ));
            state.per_graph.push(draws);
            state.atoms.push(EnvelopeAtomFit {
                key,
                prep: bprep,
                posterior,
                status: identification.status,
                weight: state.key_weights.get(key),
                estimand,
                indexer: Some(indexer),
                prior: state.envelope_prior.clone(),
            });
        } else {
            state.demote(key, DbnAtomDemotion::Draws);
        }
        Ok(())
    }

    /// Validate stage of the Bayesian DBN envelope: sequential validation for multi-step
    /// sustained queries, the envelope refutation plus predictive checks otherwise.
    fn validate_dbn_posterior_bayesian(
        &self,
        input: &DbnBayesianValidationInput<'_>,
        atoms: &[EnvelopeAtomFit],
        sequential_atoms: &[SequentialValidationAtom],
        posterior: &mut CausalPosterior,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>), CausalError>
    {
        let DbnBayesianValidationInput { run, estimator, estimate_ate } = *input;
        let DbnBayesianRun { data, query, cfg, sequential_bayes, multi_step, gaps, .. } = *run;
        let tabular = TabularData::new(data.storage().clone());
        let ate_query = temporal_refuter_average(query);
        let (mut refutations, notes, sequential_predictive) = if multi_step {
            validate_sequential(
                SequentialValidation {
                    data,
                    query,
                    suite: self.settings.refute,
                    custom: &self.declared.custom_validators,
                    predictive_sims: super::super::super::latency::predictive_check_sims(
                        self.latency_mode,
                    ),
                    gaps,
                },
                sequential_atoms,
                sequential_bayes,
                Some(&mut *posterior),
                estimate_ate,
                ctx,
            )?
        } else {
            let (reports, notes) = EnvelopeRefutation {
                data: &tabular,
                query: &ate_query,
                suite: self.settings.refute,
                estimator: EstimatorId::BayesianTemporalGcomp.as_str(),
                custom: &self.declared.custom_validators,
                temporal: Some(TemporalRefutation {
                    query,
                    split: self.declared.split.as_ref(),
                    time_index: data.time_index(),
                    gaps,
                }),
            }
            .run(
                &envelope_refute_atoms(atoms)?,
                &mut EstimationWorkspace::default(),
                ctx,
            )?;
            (reports, notes, Vec::new())
        };
        diagnostics.extend(notes);
        let predictive_checks = if multi_step {
            sequential_predictive
        } else {
            run_envelope_bayesian_full_validation(
                BayesianValidation { suite: self.settings.refute, cfg, estimator },
                atoms,
                posterior,
                estimate_ate,
                ctx,
                super::super::super::latency::predictive_check_sims(self.latency_mode),
                ValidationSink { refutations: &mut refutations, diagnostics },
            )?
        };
        Ok((refutations, predictive_checks))
    }
}

/// Structural mass accounting for the Frequentist DBN-posterior Pulse /
/// Sustained mixture: every posterior atom with its weight and status, the
/// evaluated atoms' point effects, and the range over them. Masses are
/// fractions of the posterior's total weight; every identified atom with
/// positive weight is evaluated (an estimation failure aborts the run), so
/// unevaluable mass is zero.
fn dbn_frequentist_structural_mixture(
    graphs: &WeightedGraphSamples,
    evaluated: &[(&crate::analysis::prepared::CachedDbnPosteriorAtomIdentification, f64)],
    estimates: &[EffectEstimate],
    point: f64,
) -> crate::result::StructuralResponseMixture {
    let values: std::collections::HashMap<u64, (IdentificationStatus, f64)> = evaluated
        .iter()
        .zip(estimates)
        .map(|((atom, _), estimate)| (atom.key, (atom.identification.status, estimate.ate)))
        .collect();
    let atoms: Vec<crate::result::StructuralResponseAtom> = graphs
        .graph_keys
        .iter()
        .zip(graphs.weights.iter())
        .zip(graphs.identified.iter())
        .map(|((&graph_key, &weight), flag)| {
            let fit =
                (*flag == GraphIdentFlag::Identified).then(|| values.get(&graph_key)).flatten();
            crate::result::StructuralResponseAtom {
                graph_key,
                weight,
                status: match (flag, fit) {
                    (GraphIdentFlag::Identified, Some((status, _))) => *status,
                    (GraphIdentFlag::Identified, None) => {
                        IdentificationStatus::NonparametricallyIdentified
                    }
                    (GraphIdentFlag::Unidentified, _) => IdentificationStatus::NotIdentified,
                },
                value: fit.map(|(_, value)| ResponseValue::Scalar(*value)),
                posterior: None,
                response: None,
            }
        })
        .collect();
    let total = graphs.total_weight();
    let identified_mass =
        atoms.iter().filter(|atom| atom.value.is_some()).map(|atom| atom.weight).sum::<f64>();
    let unidentified_mass = graphs.unidentified_mass();
    let (lo, hi) = estimates
        .iter()
        .map(|estimate| estimate.ate)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
    crate::result::StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass: identified_mass / total,
        unidentified_mass: unidentified_mass / total,
        unevaluable_mass: 0.0,
        subsampled_out_mass: 0.0,
        identified_set: (lo.is_finite() && hi.is_finite()).then(|| {
            antecedent_core::ResponseEnvelope {
                grid: Arc::from([0.0]),
                dimension: 1,
                lower: Arc::from([lo]),
                upper: Arc::from([hi]),
            }
        }),
        identified_set_interval: None,
        conditional_on_identified: Some(ResponseValue::Scalar(point)),
        full_mass_scope: true,
        truncated_atoms: 0,
        aggregation_policy: Some(
            crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
        ),
    }
}
