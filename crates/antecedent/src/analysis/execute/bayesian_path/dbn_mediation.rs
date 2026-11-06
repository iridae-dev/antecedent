//! DBN posterior mediation fitting, aggregation and validation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::TableView;

use super::{
    Arc, BayesianGCompWorkspace, BayesianGComputationAte, BayesianValidation, CausalError,
    CausalPosterior, DbnAtomDemotion, DbnIdentifyDemotion, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EnvelopeOptions, EstimatorId, ExecutionContext,
    GraphEffectDraws, GraphIdentFlag, GraphPosterior, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights,
    IdentifierId, InferenceDiagnostics, InferenceMode, Instant, PhysicalExecutionPlan,
    PosteriorMassSummary, PosteriorPredictiveCheck, PredictiveCheckReport, PreparedBayesianProblem,
    PriorPredictiveCheck, PriorSet, RefuteSuite, Strategy, StudyResult, TemporalIndexer,
    TemporalMediationEstimate, TimeSeriesData, ValidationSink, WeightedGraphSamples,
    aggregate_effect_envelope, bayesian_gcomp, effect_from_posterior, elapsed_ns,
    envelope_draws_from_posterior, envelope_mass_diagnostic, first_key_positions,
    identified_envelope_keys, maybe_interactive_envelope_subsample, mediation_mechanism_seed,
    mix_prior_sensitivity_summaries, most_conservative_identification_status, nan_effect,
    overlap_diagnostic, overlap_limit_diagnostic, provenance_ids,
    push_graph_posterior_structural_aggregation_diagnostic, report_subsampled_out_mass,
    require_identified, resolve_posterior_status, resolve_structural_aggregation,
    with_prior_sensitivity,
};

impl super::super::Study {
    /// Mix a DBN posterior into a Bayesian temporal-mediation envelope.
    ///
    /// Each atom uses that atom's per-horizon mediation set `S(h)` (its `I(h)`
    /// plus mediator/outcome confounders). Unidentified atoms
    /// keep their mass. Priors do not upgrade identification.
    pub(in crate::analysis::execute) fn execute_dbn_posterior_mediation(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "DBN graph-posterior mediation",
        )?;
        if query.horizons.len() > 1 && self.settings.inference.is_frequentist() {
            return Err(CausalError::Unsupported {
                message: "Frequentist DBN-posterior mediation is supported for one horizon; \
                          multi-horizon grids need their own joint uncertainty contract",
            });
        }
        if query.horizons.len() > 1 {
            return self
                .execute_dbn_posterior_mediation_multi_horizon(data, gp, query, physical, ctx);
        }
        if self.settings.inference.is_frequentist() {
            return self
                .execute_dbn_posterior_mediation_frequentist(data, gp, query, physical, ctx);
        }
        self.execute_dbn_posterior_mediation_bayesian(data, gp, query, physical, ctx)
    }

    /// Multi-horizon DBN-posterior mediation: identify once, project that atom cache per
    /// horizon, run each horizon as its own study, and gather the pointwise slices.
    fn execute_dbn_posterior_mediation_multi_horizon(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        // Identify once for the complete request, then project the same
        // atom cache independently for each horizon (fresh and prepared).
        let (cache, _) =
            self.dbn_posterior_mediation_identification(data, gp, query, physical, ctx)?;
        let mut horizon_study = self.clone();
        horizon_study.install_identification(crate::analysis::IdentificationCache::DbnPosterior(
            Arc::clone(&cache),
        ));

        let mut slices = Vec::with_capacity(query.horizons.len());
        let mut first = None;
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.dbn_posterior.mediation_multi_horizon",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each horizon is independently identified and graph-mixed; summaries are \
             pointwise and do not define a joint horizon posterior",
        )];
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        for &horizon in query.horizons.iter() {
            let mut horizon_query = query.clone();
            horizon_query.horizons = Arc::from([horizon]);
            let horizon_cache = cache.mediation_horizon(horizon)?;
            if horizon_cache.atoms.is_empty() {
                let (slice, notes) =
                    unidentified_mediation_horizon_slice(horizon, &horizon_cache.identify_demotion);
                slices.push(slice);
                diagnostics.extend(notes);
                continue;
            }
            let result = horizon_study.execute_dbn_posterior_mediation(
                data,
                gp,
                &horizon_query,
                physical,
                ctx,
            )?;
            let slice = result
                .mediation_grid()
                .and_then(|grid| grid.slices.first())
                .ok_or_else(|| CausalError::Compile {
                    message: "DBN mediation horizon omitted its grid slice".into(),
                })?
                .clone();
            slices.push(slice);
            for mut report in result.view.refutations {
                report.refuter = Arc::from(format!("horizon.{horizon}.{}", report.refuter));
                refutations.push(report);
            }
            predictive_checks.extend(result.view.predictive_checks);
            diagnostics.extend(result.view.diagnostics);
            if first.is_none() {
                first = Some((result.view.identification, result.view.estimand));
            }
        }
        let (mut identification, estimand) = first.ok_or_else(|| CausalError::Compile {
            message: "DBN mediation requires at least one horizon".into(),
        })?;
        identification.status = most_conservative_identification_status(
            slices.iter().map(|slice| slice.identification_status),
        );
        diagnostics.push(Diagnostic::new(
            "identify.temporal_mediation.multi_horizon_status",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "parent identification.status is the most conservative requested-horizon \
             status; mediation_grid is authoritative per horizon",
        ));
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                nan_effect(),
                Strategy::new(IdentifierId::Frontdoor, EstimatorId::BayesianTemporalMediation),
                query.treatment,
                query.outcome,
            )
            .identify_cached(self.dbn_posterior_cache().is_some())
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                mediation_grid: Some(antecedent_estimate::TemporalMediationGrid {
                    slices: Arc::from(slices),
                    joint_posterior: false,
                }),
                predictive_checks,
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }

    /// The frozen DBN-posterior mediation identification: the prepared cache, the product
    /// the plan's lowered program retained at compile, or a fresh build.
    fn dbn_posterior_mediation_identification(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
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
            let variables = data.schema().variables().iter().map(|v| v.id).collect::<Vec<_>>();
            (
                Arc::new(
                    crate::analysis::prepared::build_dbn_posterior_mediation_identification_cache(
                        gp, &variables, query, ctx,
                    )?,
                ),
                false,
            )
        })
    }

    /// Single-horizon Bayesian DBN-posterior mediation: every atom's per-horizon mechanisms are
    /// fitted and composed, mixed into one envelope, and published with its pointwise grid slice.
    fn execute_dbn_posterior_mediation_bayesian(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use antecedent_estimate::bayesian_mediation::require_gaussian_mediation;

        let started = Instant::now();
        let cfg = match &self.settings.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::unsupported_inference(
                    "Bayesian DBN-posterior mediation requires inference=Bayesian",
                ));
            }
        };
        if cfg.has_transfer() {
            return Err(CausalError::Unsupported {
                message: "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
            });
        }
        let estimator = bayesian_gcomp(&cfg, ctx);
        require_gaussian_mediation(&estimator).map_err(CausalError::from)?;

        let (identified, identify_cached) =
            self.dbn_posterior_mediation_identification(data, gp, query, physical, ctx)?;
        let horizon = query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
            message: "DBN mediation requires at least one horizon".into(),
        })?;
        let identified = identified.mediation_horizon(horizon)?;
        let run = DbnMediationRun { data, query, cfg: &cfg, estimator: &estimator };
        let fits = Self::fit_dbn_mediation_atoms(&run, &identified, ctx);
        let mut envelope =
            aggregate_dbn_mediation_envelope(self.latency_mode, &identified.graphs, fits, ctx)?;
        let subsample_notes = std::mem::take(&mut envelope.subsample_notes);
        let mut diagnostics = dbn_mediation_diagnostics(&envelope, subsample_notes, &identified);
        let (refutations, predictive_checks) = self.validate_dbn_mediation(
            &run,
            &envelope.refute_atoms,
            &mut envelope.posterior,
            envelope.estimate.ate,
            &mut diagnostics,
            ctx,
        )?;
        let mediation_grid = dbn_mediation_grid(query.horizons[0], &mut envelope, &diagnostics)?;
        let DbnMediationEnvelope {
            posterior,
            estimate,
            identification,
            estimand,
            indexer,
            mediation,
            ..
        } = envelope;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification.clone(),
                estimand,
                estimate,
                Strategy::new(IdentifierId::Frontdoor, EstimatorId::BayesianTemporalMediation),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .mediation(Some(mediation))
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    "estimate.bayesian_temporal_mediation",
                )),
                posterior: Some(posterior),
                mediation_grid: Some(mediation_grid),
                predictive_checks,
                diagnostics: Some(diagnostics),
                certificate: Some(crate::Identification::Point {
                    result: identification,
                    temporal_indexer: Some(indexer),
                    strategy: IdentifierId::Frontdoor,
                    structure_version: self.graph.version(),
                }),
                ..Default::default()
            }),
        ))
    }

    /// Per-atom Bayesian mediation fits over the horizon-projected DBN cache: every atom is
    /// prepared and fitted, and failures are demoted before the Interactive subsample.
    fn fit_dbn_mediation_atoms(
        run: &DbnMediationRun<'_>,
        identified: &super::super::super::prepared::CachedDbnPosteriorIdentification,
        ctx: &ExecutionContext,
    ) -> DbnMediationFits {
        let keys = identified.graphs.graph_keys.to_vec();
        let mut state = DbnMediationFits {
            key_weights: IdentifiedKeyWeights::new(&identified.graphs),
            key_index: first_key_positions(&keys),
            flags: identified.graphs.identified.to_vec(),
            atom_contexts: Vec::new(),
            per_graph: Vec::new(),
            refute_atoms: Vec::new(),
            prepare_demoted: 0,
            fit_demoted: 0,
            draws_demoted: 0,
            distinct_z: false,
            first_z: None,
            horizon_dependent: false,
            unevaluable_weight: 0.0,
            first_estimation_error: None,
        };
        for atom in identified.atoms.iter() {
            let Some(published) = Self::prepare_dbn_mediation_atom(run, &mut state, atom, ctx)
            else {
                continue;
            };
            Self::fit_dbn_mediation_atom(run, &mut state, atom.key, published, ctx);
        }
        state
    }

    /// Resolve the atom's per-horizon adjustment and prepare its two mechanism designs; an
    /// atom that cannot be prepared is demoted and yields `None`.
    fn prepare_dbn_mediation_atom(
        run: &DbnMediationRun<'_>,
        state: &mut DbnMediationFits,
        atom: &super::super::super::prepared::CachedDbnPosteriorAtomIdentification,
        ctx: &ExecutionContext,
    ) -> Option<DbnPublishedClick> {
        use antecedent_estimate::bayesian_mediation::prepare_temporal_mediation_adjusted;

        let DbnMediationRun { data, query, .. } = *run;
        let Some(horizons) = atom.horizons.as_ref() else {
            state.demote(atom.key, DbnAtomDemotion::Prepare);
            return None;
        };
        let Ok(clicks) =
            super::super::temporal_path::mediation_horizon_clicks(horizons, query, |horizon| {
                horizons
                    .get(horizon)
                    .map(super::super::temporal_path::lagged_adjustment_from_entry)
                    .ok_or_else(|| CausalError::Compile {
                        message: format!(
                            "DBN mediation atom missing I({horizon}) for key {}",
                            atom.key
                        ),
                    })
            })
        else {
            state.demote(atom.key, DbnAtomDemotion::Prepare);
            return None;
        };
        if clicks.is_empty() {
            state.demote(atom.key, DbnAtomDemotion::Prepare);
            return None;
        }
        state.horizon_dependent |=
            super::super::temporal_path::mediation_horizon_z_differs(&clicks);
        let mut published = None;
        let mut click_ok = true;
        let mut estimation_failed = false;
        for click in &clicks {
            if require_identified(&click.identification).is_err() {
                click_ok = false;
                break;
            }
            let mut qh = query.clone();
            qh.horizons = Arc::from([click.horizon]);
            match prepare_temporal_mediation_adjusted(
                data,
                &click.estimand,
                &qh,
                &click.adjustment,
                ctx,
            ) {
                Ok(preparations) => {
                    if published.is_none() {
                        published = Some((click, qh, preparations));
                    }
                }
                Err(error) => {
                    click_ok = false;
                    estimation_failed = true;
                    state.first_estimation_error.get_or_insert_with(|| error.to_string());
                    break;
                }
            }
        }
        if !click_ok {
            if estimation_failed {
                state.unevaluable_weight += state.key_weights.get(atom.key);
            }
            state.demote(atom.key, DbnAtomDemotion::Prepare);
            return None;
        }
        let Some((click, qh, preparations)) = published else {
            state.demote(atom.key, DbnAtomDemotion::Prepare);
            return None;
        };
        match state.first_z.as_ref() {
            Some(z) if z.as_ref() != click.adjustment.as_ref() => state.distinct_z = true,
            None => state.first_z = Some(Arc::clone(&click.adjustment)),
            _ => {}
        }
        Some(DbnPublishedClick { click: click.clone(), qh, preparations })
    }

    /// Fit both mechanisms of one prepared atom, compose the mediation posterior, and keep
    /// the atom when it yields draws; otherwise demote it.
    fn fit_dbn_mediation_atom(
        run: &DbnMediationRun<'_>,
        state: &mut DbnMediationFits,
        key: u64,
        published: DbnPublishedClick,
        ctx: &ExecutionContext,
    ) {
        use antecedent_estimate::bayesian_mediation::compose_temporal_mediation_mechanisms;

        let DbnMediationRun { cfg, estimator, .. } = *run;
        let DbnPublishedClick { click, qh, preparations } = published;
        let fit = |scale: f64| -> Result<Vec<CausalPosterior>, CausalError> {
            let fitted = preparations
                .iter()
                .enumerate()
                .map(|(i, prep)| {
                    let mut est = estimator.clone();
                    est.prior_scale = scale;
                    est.seed = mediation_mechanism_seed(est.seed, click.horizon, i);
                    est.fit(
                        prep,
                        click.identification.status,
                        &mut BayesianGCompWorkspace::default(),
                        ctx,
                    )
                    .map_err(CausalError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(antecedent_estimate::bayesian_mediation::couple_mediation_mechanisms(
                &preparations,
                fitted,
                click.horizon as usize,
                mediation_mechanism_seed(estimator.seed, click.horizon, preparations.len()),
                ctx,
            ))
        };
        let mechanisms = match fit(cfg.prior_scale) {
            Ok(mechanisms) => mechanisms,
            Err(error) => {
                state.demote_unevaluable(key, DbnAtomDemotion::Fit, &error);
                return;
            }
        };
        let composed = match compose_temporal_mediation_mechanisms(
            &mechanisms,
            &qh,
            click.identification.status,
        ) {
            Ok(composed) => composed,
            Err(error) => {
                state.demote_unevaluable(key, DbnAtomDemotion::Fit, &error);
                return;
            }
        };
        match envelope_draws_from_posterior(key, &composed) {
            Ok(draws) => {
                state.atom_contexts.push((
                    key,
                    click.estimand.clone(),
                    click.identification.clone(),
                    click.indexer.clone(),
                ));
                state.per_graph.push(draws);
                let weight = state.key_weights.get(key);
                state.refute_atoms.push(DbnMediationAtom {
                    key,
                    weight,
                    estimand: click.estimand.clone(),
                    adjustment: Arc::clone(&click.adjustment),
                    query: qh,
                    preparations,
                    mechanisms,
                    composed,
                });
            }
            Err(error) => state.demote_unevaluable(key, DbnAtomDemotion::Draws, &error),
        }
    }

    /// Validate stage of the Bayesian DBN mediation envelope: the per-atom mediation
    /// refuters and the envelope's predictive checks.
    fn validate_dbn_mediation(
        &self,
        run: &DbnMediationRun<'_>,
        refute_atoms: &[DbnMediationAtom],
        posterior: &mut CausalPosterior,
        estimate_ate: f64,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>), CausalError>
    {
        let DbnMediationRun { data, cfg, estimator, .. } = *run;
        let mut refutations = Vec::new();
        if self.settings.refute != RefuteSuite::None {
            let (reports, notes) = mix_dbn_mediation_refuters(
                data,
                refute_atoms,
                (self.settings.refute == RefuteSuite::Full, self.declares_complete_case()),
                ctx,
            )?;
            refutations.extend(reports);
            diagnostics.extend(notes);
        }
        let predictive_checks = run_dbn_mediation_bayesian_validation(
            BayesianValidation { suite: self.settings.refute, cfg, estimator },
            refute_atoms,
            posterior,
            estimate_ate,
            ctx,
            ValidationSink { refutations: &mut refutations, diagnostics },
        )?;
        Ok((refutations, predictive_checks))
    }
}

/// Resolved inputs shared by the DBN Bayesian mediation phases.
#[derive(Clone, Copy)]
struct DbnMediationRun<'a> {
    data: &'a TimeSeriesData,
    query: &'a antecedent_core::MediationQuery,
    cfg: &'a crate::inference::BayesianConfig,
    estimator: &'a BayesianGComputationAte,
}

/// The first prepared horizon click of an atom, with its single-horizon query and designs.
struct DbnPublishedClick {
    click: super::super::temporal_path::MediationHorizonClick,
    qh: antecedent_core::MediationQuery,
    preparations: Vec<PreparedBayesianProblem>,
}

/// Mutable state threaded through the DBN Bayesian mediation per-atom fits.
struct DbnMediationFits {
    key_weights: IdentifiedKeyWeights,
    key_index: std::collections::HashMap<u64, usize>,
    flags: Vec<GraphIdentFlag>,
    atom_contexts: Vec<(u64, IdentifiedEstimand, IdentificationResult, TemporalIndexer)>,
    per_graph: Vec<GraphEffectDraws>,
    refute_atoms: Vec<DbnMediationAtom>,
    prepare_demoted: usize,
    fit_demoted: usize,
    draws_demoted: usize,
    distinct_z: bool,
    first_z: Option<Arc<[antecedent_data::LaggedColumn]>>,
    horizon_dependent: bool,
    /// Mass of atoms that WERE identified but whose estimation (design preparation,
    /// fitting, composition or draw extraction) failed. This is a refusal to estimate, not
    /// a proof of non-identification, so it is kept apart from `unidentified_mass` (the
    /// `require_identified` failure, and an atom with no per-horizon identification at
    /// all, stay genuine non-identification).
    unevaluable_weight: f64,
    first_estimation_error: Option<String>,
}

impl DbnMediationFits {
    /// Mark the atom's graph key unidentified for the envelope.
    fn flag_unidentified(&mut self, key: u64) {
        if let Some(idx) = self.key_index.get(&key).copied() {
            self.flags[idx] = GraphIdentFlag::Unidentified;
        }
    }

    /// Flag the atom unidentified and count the stage that dropped it.
    fn demote(&mut self, key: u64, reason: DbnAtomDemotion) {
        self.flag_unidentified(key);
        match reason {
            DbnAtomDemotion::Prepare => self.prepare_demoted += 1,
            DbnAtomDemotion::Fit => self.fit_demoted += 1,
            DbnAtomDemotion::Draws => self.draws_demoted += 1,
        }
    }

    /// [`Self::demote`] for an atom whose estimation failed: its identified weight is booked
    /// as unevaluable and the first failure message is kept for the disclosure.
    fn demote_unevaluable(
        &mut self,
        key: u64,
        reason: DbnAtomDemotion,
        error: &impl std::fmt::Display,
    ) {
        self.unevaluable_weight += self.key_weights.get(key);
        self.first_estimation_error.get_or_insert_with(|| error.to_string());
        self.demote(key, reason);
    }
}

/// Aggregated Bayesian DBN mediation envelope and the atoms it kept.
struct DbnMediationEnvelope {
    posterior: CausalPosterior,
    estimate: EffectEstimate,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    mediation: TemporalMediationEstimate,
    graphs: WeightedGraphSamples,
    subsample_notes: Vec<Diagnostic>,
    refute_atoms: Vec<DbnMediationAtom>,
    prepare_demoted: usize,
    fit_demoted: usize,
    draws_demoted: usize,
    distinct_z: bool,
    horizon_dependent: bool,
    first_estimation_error: Option<String>,
}

/// Unidentified-horizon slice and its notes, for a horizon whose projected cache has no atoms.
fn unidentified_mediation_horizon_slice(
    horizon: u32,
    identify_demotion: &DbnIdentifyDemotion,
) -> (antecedent_estimate::TemporalMediationSlice, Vec<Diagnostic>) {
    let notes = vec![
        Diagnostic::new(
            "estimate.dbn_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "unidentified_mass=1",
        )
        .with_fields(super::super::mass_fields(Some(0.0), 1.0)),
        Diagnostic::new(
            "estimate.dbn_posterior.atom_demotion",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            identify_demotion.summary(0, 0, 0),
        ),
    ];
    let slice = antecedent_estimate::TemporalMediationSlice {
        horizon,
        identification_status: IdentificationStatus::NotIdentified,
        method: Arc::from("temporal_mediation.unidentified"),
        adjustment: Arc::from([]),
        estimate: TemporalMediationEstimate {
            effect: nan_effect(),
            total: None,
            direct: None,
            mediated: None,
        },
        uncertainty: antecedent_estimate::TemporalMediationUncertainty::Unavailable,
        identified_set: None,
        diagnostics: notes.clone(),
    };
    (slice, notes)
}

/// Interactive-subsampled envelope of the fitted DBN mediation atoms: the posterior with
/// estimation failures split out of the unidentified mass, and the contributing atom's identity.
fn aggregate_dbn_mediation_envelope(
    latency_mode: Option<crate::LatencyMode>,
    identified_graphs: &WeightedGraphSamples,
    fits: DbnMediationFits,
    ctx: &ExecutionContext,
) -> Result<DbnMediationEnvelope, CausalError> {
    let DbnMediationFits {
        flags,
        per_graph,
        mut refute_atoms,
        atom_contexts,
        prepare_demoted,
        fit_demoted,
        draws_demoted,
        distinct_z,
        horizon_dependent,
        unevaluable_weight,
        first_estimation_error,
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
    refute_atoms.retain(|atom| keep.contains(&atom.key));
    let (_, estimand, mut identification, indexer) = atom_contexts
        .into_iter()
        .find(|(key, _, _, _)| keep.contains(key))
        .ok_or_else(|| CausalError::Compile {
            message: "DBN posterior envelope has no contributing context".into(),
        })?;
    let graph_key_weights = IdentifiedKeyWeights::new(&graphs);
    for atom in &mut refute_atoms {
        atom.weight = graph_key_weights.get(atom.key);
    }
    let mut posterior = aggregate_effect_envelope(
        &graphs,
        &per_graph,
        InferenceDiagnostics::analytic("dbn_posterior_envelope"),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    report_subsampled_out_mass(&mut posterior, &fitted_graphs, &subsample_drop);
    // `report_subsampled_out_mass` reads `unidentified_mass` off the
    // pre-subsample flags, which still carry every estimation-failure
    // demotion above as `Unidentified` (the envelope aggregation has no
    // third state to exclude an atom by). Split that mass back out so a
    // refusal to estimate is never published as a negative identification
    // finding.
    let total_weight = fitted_graphs.total_weight();
    let unevaluable_mass = if total_weight > 0.0 { unevaluable_weight / total_weight } else { 0.0 };
    posterior.unevaluable_mass = unevaluable_mass;
    posterior.unidentified_mass = (posterior.unidentified_mass - unevaluable_mass).max(0.0);
    // The envelope must preserve structural restrictions and disclose this
    // horizon's unidentified / unevaluable graph mass, independently of
    // other horizons.
    identification.status = resolve_posterior_status(
        identification.status,
        crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean,
        PosteriorMassSummary {
            unidentified: posterior.unidentified_mass,
            unevaluable: posterior.unevaluable_mass,
            subsampled: posterior.subsampled_out_mass,
            ..PosteriorMassSummary::default()
        },
    );
    posterior.identification = identification.status;
    let estimate = effect_from_posterior(&posterior)?;
    let mediation = mix_dbn_mediation_estimate(&estimate, &refute_atoms);
    Ok(DbnMediationEnvelope {
        posterior,
        estimate,
        identification,
        estimand,
        indexer,
        mediation,
        graphs,
        subsample_notes,
        refute_atoms,
        prepare_demoted,
        fit_demoted,
        draws_demoted,
        distinct_z,
        horizon_dependent,
        first_estimation_error,
    })
}

/// Diagnostics of the Bayesian DBN mediation envelope: identification's, subsample notes,
/// overlap, mass, atom demotions, structural aggregation, and the per-horizon identification notes.
fn dbn_mediation_diagnostics(
    envelope: &DbnMediationEnvelope,
    subsample_notes: Vec<Diagnostic>,
    identified: &super::super::super::prepared::CachedDbnPosteriorIdentification,
) -> Vec<Diagnostic> {
    let DbnMediationEnvelope {
        posterior,
        estimate,
        identification,
        refute_atoms,
        prepare_demoted,
        fit_demoted,
        draws_demoted,
        distinct_z,
        horizon_dependent,
        first_estimation_error,
        ..
    } = envelope;
    let (prepare_demoted, fit_demoted, draws_demoted) =
        (*prepare_demoted, *fit_demoted, *draws_demoted);
    let (distinct_z, horizon_dependent) = (*distinct_z, *horizon_dependent);
    let mut diagnostics = identification.diagnostics.clone();
    diagnostics.extend(subsample_notes);
    diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
    diagnostics.extend(overlap_limit_diagnostic(estimate.diagnostics.overlap_report.as_ref()));
    diagnostics.push(envelope_mass_diagnostic("estimate.dbn_posterior.envelope", posterior));
    let estimate_demoted = prepare_demoted + fit_demoted + draws_demoted;
    let mut demotion_summary =
        identified.identify_demotion.summary(prepare_demoted, fit_demoted, draws_demoted);
    if let Some(error) = first_estimation_error.as_ref() {
        demotion_summary.push_str(&format!("; first_estimation_error={error:?}"));
    }
    diagnostics.push(Diagnostic::new(
        "estimate.dbn_posterior.atom_demotion",
        DiagnosticKind::Scientific,
        if estimate_demoted > 0 { DiagnosticSeverity::Warning } else { DiagnosticSeverity::Info },
        demotion_summary,
    ));
    let contributing: Vec<&IdentifiedEstimand> =
        refute_atoms.iter().map(|atom| &atom.estimand).collect();
    let any_partial = refute_atoms
        .iter()
        .any(|atom| atom.composed.identification == IdentificationStatus::PartiallyIdentified);
    let contributing_identified_mass =
        refute_atoms.iter().map(|atom| atom.weight).sum::<f64>() / identified.graphs.total_weight();
    push_graph_posterior_structural_aggregation_diagnostic(
        &mut diagnostics,
        resolve_structural_aggregation(&contributing, any_partial),
        contributing_identified_mass,
        posterior.unidentified_mass,
        posterior.unevaluable_mass,
        posterior.subsampled_out_mass,
    );
    diagnostics.push(Diagnostic::new(
        "identify.dbn_posterior.per_atom_horizon",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "each graph atom uses that atom's I(h); adjustment sets are not unioned across atoms",
    ));
    if horizon_dependent {
        diagnostics.push(Diagnostic::new(
            "identify.temporal_mediation.horizon_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "adjustment sets differ across requested horizons; each contrast uses S(h) = I(h) \
             plus mediator-outcome confounders identified for that horizon, not a shared \
             max-horizon set",
        ));
    }
    if distinct_z {
        diagnostics.push(Diagnostic::new(
            "identify.dbn_posterior.atom_horizon_sets_differ",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "contributing DBN atoms have distinct I(h) adjustment sets at the published horizon",
        ));
    }
    diagnostics.push(Diagnostic::new(
        "estimate.mediation.bayesian",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "independent Gaussian mediator and outcome mechanisms; total = direct + mediated for every posterior draw; natural effects use the linear no-interaction alias",
    ));
    diagnostics
}

/// Pointwise grid slice of the single-horizon DBN mediation envelope; also fills the
/// total / direct / mediated components of the envelope's mediation estimate.
fn dbn_mediation_grid(
    horizon: u32,
    envelope: &mut DbnMediationEnvelope,
    diagnostics: &[Diagnostic],
) -> Result<antecedent_estimate::TemporalMediationGrid, CausalError> {
    let requested_summary = antecedent_estimate::MediationPosteriorSummary {
        mean: envelope.posterior.summaries.mean[0],
        standard_deviation: envelope.posterior.summaries.sd[0],
        q025: envelope.posterior.summaries.q025[0],
        q975: envelope.posterior.summaries.q975[0],
    };
    let total_summary =
        dbn_mediation_component_summary(&envelope.graphs, &envelope.refute_atoms, 1)?;
    let direct_summary =
        dbn_mediation_component_summary(&envelope.graphs, &envelope.refute_atoms, 2)?;
    let mediated_summary =
        dbn_mediation_component_summary(&envelope.graphs, &envelope.refute_atoms, 3)?;
    envelope.mediation.total = Some(total_summary.mean);
    envelope.mediation.direct = Some(direct_summary.mean);
    envelope.mediation.mediated = Some(mediated_summary.mean);
    let mut adjustment = envelope
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|id| envelope.indexer.key_of(id.raw()).ok())
        .collect::<Vec<_>>();
    adjustment.sort();
    let mediation_grid = antecedent_estimate::TemporalMediationGrid {
        slices: Arc::from([antecedent_estimate::TemporalMediationSlice {
            horizon,
            identification_status: envelope.identification.status,
            method: Arc::clone(&envelope.estimand.method),
            adjustment: Arc::from(adjustment),
            estimate: envelope.mediation.clone(),
            uncertainty: antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
                requested: requested_summary,
                total: total_summary,
                direct: direct_summary,
                mediated: mediated_summary,
                n_draws: envelope.posterior.draws.n_draws,
                backend: Arc::clone(&envelope.posterior.diagnostics.backend_id),
            },
            identified_set: None,
            diagnostics: diagnostics.to_vec(),
        }]),
        joint_posterior: false,
    };
    Ok(mediation_grid)
}

struct DbnMediationAtom {
    key: u64,
    weight: f64,
    estimand: IdentifiedEstimand,
    adjustment: Arc<[antecedent_data::LaggedColumn]>,
    query: antecedent_core::MediationQuery,
    preparations: Vec<PreparedBayesianProblem>,
    mechanisms: Vec<CausalPosterior>,
    composed: CausalPosterior,
}

fn dbn_mediation_component_summary(
    graphs: &WeightedGraphSamples,
    atoms: &[DbnMediationAtom],
    quantity: usize,
) -> Result<antecedent_estimate::MediationPosteriorSummary, CausalError> {
    let per_graph = atoms
        .iter()
        .map(|atom| {
            let draws = atom.composed.draws.column(quantity)?;
            Ok(GraphEffectDraws { graph_key: atom.key, effect_draws: Arc::from(draws.to_vec()) })
        })
        .collect::<Result<Vec<_>, CausalError>>()?;
    let posterior = aggregate_effect_envelope(
        graphs,
        &per_graph,
        InferenceDiagnostics::analytic("dbn_mediation_component"),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    Ok(antecedent_estimate::MediationPosteriorSummary {
        mean: posterior.summaries.mean[0],
        standard_deviation: posterior.summaries.sd[0],
        q025: posterior.summaries.q025[0],
        q975: posterior.summaries.q975[0],
    })
}

fn mix_dbn_mediation_estimate(
    estimate: &EffectEstimate,
    atoms: &[DbnMediationAtom],
) -> TemporalMediationEstimate {
    let mut total = 0.0;
    let mut direct = 0.0;
    let mut mediated = 0.0;
    let mut w = 0.0;
    for atom in atoms {
        if atom.weight <= 0.0 || atom.composed.summaries.mean.len() < 4 {
            continue;
        }
        w += atom.weight;
        total += atom.weight * atom.composed.summaries.mean[1];
        direct += atom.weight * atom.composed.summaries.mean[2];
        mediated += atom.weight * atom.composed.summaries.mean[3];
    }
    TemporalMediationEstimate {
        effect: estimate.clone(),
        total: (w > 0.0).then_some(total / w),
        direct: (w > 0.0).then_some(direct / w),
        mediated: (w > 0.0).then_some(mediated / w),
    }
}

/// This atom's own composed mediation summary: the refuters' reference value.
fn dbn_mediation_atom_estimate(
    atom: &DbnMediationAtom,
) -> Result<TemporalMediationEstimate, CausalError> {
    let means = &atom.composed.summaries.mean;
    let component = |index: usize| (means.len() >= 4).then(|| means[index]);
    Ok(TemporalMediationEstimate {
        effect: effect_from_posterior(&atom.composed)?,
        total: component(1),
        direct: component(2),
        mediated: component(3),
    })
}

/// Mediation refuters per contributing atom, each against that atom's own
/// composed estimate (not the pooled mixture), mixed by envelope mass.
fn mix_dbn_mediation_refuters(
    data: &TimeSeriesData,
    atoms: &[DbnMediationAtom],
    (full, gaps): (bool, bool),
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let mut per_atom = Vec::new();
    for atom in atoms {
        if atom.weight <= 0.0 {
            continue;
        }
        let original = dbn_mediation_atom_estimate(atom)?;
        let reports = antecedent_validate::mediation::refute_temporal_mediation_with_gaps(
            data,
            &atom.estimand,
            &atom.query,
            &original,
            full,
            &atom.adjustment,
            gaps,
            ctx,
        )
        .map_err(CausalError::from)?;
        per_atom.push((atom.weight, reports));
    }
    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    Ok(super::super::atom_mix::finish_atom_refutations(
        per_atom,
        Vec::new(),
        true,
        super::super::atom_mix::ScalarRefuterMix::Effect,
        Some(Diagnostic::new(
            "refute.envelope.effect_mixture",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "mediation refuters evaluated each contributing graph atom [{atom_keys}] against \
                 that atom's own composed estimate using that atom's I(h), not the pooled \
                 mixture; reports mix by envelope mass and pass only if every contributing atom \
                 passes"
            ),
        )),
        true,
    ))
}

/// Isotropic prior-scale sensitivity of every mediation atom: its composed effect mean and
/// SD across the standard scale grid, weighted by the atom's mass.
fn dbn_mediation_prior_sensitivity(
    atoms: &[DbnMediationAtom],
    estimator: &BayesianGComputationAte,
    ctx: &ExecutionContext,
) -> Result<
    (antecedent_validate::PriorSensitivity, Vec<(f64, antecedent_prob::PriorSensitivitySummary)>),
    CausalError,
> {
    let sensitivity = antecedent_validate::PriorSensitivity::standard_grid();
    let mut owned = Vec::new();
    for atom in atoms {
        let mut means = Vec::new();
        let mut sds = Vec::new();
        for &scale in sensitivity.scales.iter() {
            let horizon = atom.query.horizons.first().copied().unwrap_or(0);
            let posts: Result<Vec<_>, _> = atom
                .preparations
                .iter()
                .enumerate()
                .map(|(i, prep)| {
                    let mut est = estimator.clone();
                    est.prior_scale = scale;
                    est.seed = mediation_mechanism_seed(est.seed, horizon, i);
                    est.fit(
                        prep,
                        atom.composed.identification,
                        &mut BayesianGCompWorkspace::default(),
                        ctx,
                    )
                    .map_err(CausalError::from)
                })
                .collect();
            let posts = antecedent_estimate::bayesian_mediation::couple_mediation_mechanisms(
                &atom.preparations,
                posts?,
                horizon as usize,
                mediation_mechanism_seed(estimator.seed, horizon, atom.preparations.len()),
                ctx,
            );
            let post = compose_temporal_mediation_for_atom(atom, &posts)?;
            means.push(post.summaries.mean[0]);
            sds.push(post.summaries.sd[0]);
        }
        owned.push((
            atom.weight,
            antecedent_prob::PriorSensitivitySummary {
                family: antecedent_prob::PriorSensitivityFamily::IsotropicScale,
                prior_scales: sensitivity.scales.clone(),
                alphas: Arc::from([]),
                variance_multipliers: Arc::from([]),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            },
        ));
    }
    Ok((sensitivity, owned))
}

fn run_dbn_mediation_bayesian_validation(
    setup: BayesianValidation<'_>,
    atoms: &[DbnMediationAtom],
    mixture_posterior: &mut CausalPosterior,
    estimate_ate: f64,
    ctx: &ExecutionContext,
    sink: ValidationSink<'_>,
) -> Result<Vec<PredictiveCheckReport>, CausalError> {
    const PPC_ALPHA: f64 = 0.05;
    let BayesianValidation { suite: refute, cfg, estimator } = setup;
    let ValidationSink { refutations, diagnostics } = sink;
    if matches!(refute, RefuteSuite::None) || atoms.is_empty() {
        return Ok(Vec::new());
    }
    let mut predictive_checks = Vec::new();
    let mut prior_items = Vec::new();
    let mut post_items = Vec::new();
    for atom in atoms {
        for (i, (prep, post)) in atom.preparations.iter().zip(&atom.mechanisms).enumerate() {
            let prior = estimator.prior.clone().unwrap_or_else(|| {
                let mut prior = PriorSet::weakly_informative(prep.design.ncols);
                prior.specs = vec![antecedent_prob::PriorSpec::GaussianCoefficients(
                    antecedent_prob::GaussianCoefficientPrior::isotropic(
                        prep.design.ncols,
                        cfg.prior_scale,
                    ),
                )];
                prior
            });
            let prior_rep = PriorPredictiveCheck::for_estimator(estimator, ctx)
                .check_with_prior(prep, &prior, ctx)
                .map_err(CausalError::from)?;
            let post_rep = PosteriorPredictiveCheck::for_estimator(estimator, ctx)
                .check(prep, post)
                .map_err(CausalError::from)?;
            prior_items.push((atom.weight, i, prior_rep));
            post_items.push((atom.weight, i, post_rep));
        }
    }
    for (label, items) in [("prior", &prior_items), ("posterior", &post_items)] {
        let _ = label;
        let borrowed: Vec<(f64, &PredictiveCheckReport)> =
            items.iter().map(|(w, _, r)| (*w, r)).collect();
        if let Some(mixed) = PredictiveCheckReport::mixture_weighted(&borrowed) {
            refutations.push(mixed.to_refutation_report(estimate_ate, PPC_ALPHA));
            predictive_checks.push(mixed);
        }
    }
    if matches!(refute, RefuteSuite::Full) {
        let (sensitivity, owned) = dbn_mediation_prior_sensitivity(atoms, estimator, ctx)?;
        let items: Vec<_> = owned.iter().map(|(w, s)| (*w, s)).collect();
        if let Some(mixed) = mix_prior_sensitivity_summaries(&items) {
            refutations.push(sensitivity.to_report(&mixed, estimate_ate));
            *mixture_posterior = with_prior_sensitivity(mixture_posterior.clone(), mixed);
        }
    }
    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    diagnostics.push(Diagnostic::new(
        "refute.bayesian.ppc.envelope",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "mechanism PPC evaluated per identified mediation atom [{atom_keys}]; reports mix \
             by graph posterior mass"
        ),
    ));
    Ok(predictive_checks)
}

fn compose_temporal_mediation_for_atom(
    atom: &DbnMediationAtom,
    posts: &[CausalPosterior],
) -> Result<CausalPosterior, CausalError> {
    use antecedent_estimate::bayesian_mediation::compose_temporal_mediation_mechanisms;
    compose_temporal_mediation_mechanisms(posts, &atom.query, atom.composed.identification)
        .map_err(CausalError::from)
}
