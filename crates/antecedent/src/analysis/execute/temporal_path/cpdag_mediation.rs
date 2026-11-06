//! CPDAG temporal mediation execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianConfig, BayesianMediationFit, CausalError, ClassMediationInput, CompletionFitMemo,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    IdentificationResult, IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant, MediationFitKey, MediationRoute,
    PhysicalExecutionPlan, RefuteSuite, ResponseValue, Strategy, StudyResult, TemporalClassWeights,
    TemporalEffectQuery, TemporalIndexer, TemporalMediationEstimate, TemporalMediationEstimator,
    TimeSeriesData, bare_class_atom, class_completion_key, effect_from_posterior, elapsed_ns,
    fit_class_bayesian_mediation, fit_class_frequentist_mediation,
    identification_status_ok_for_case, lagged_column_relative_to_outcome, mediation_block_stream,
    nan_effect, push_conflict_diagnostics, refute_class_mediation, select_estimand,
    temporal_class_response_mean, temporal_dependence_se_diagnostics,
};

/// The series, query and estimator every completion of a CPDAG mediation
/// sweep is fitted with.
#[derive(Clone, Copy)]
struct CpdagMediationInputs<'a> {
    data: &'a TimeSeriesData,
    query: &'a antecedent_core::MediationQuery,
    estimator: &'a TemporalMediationEstimator,
    ctx: &'a ExecutionContext,
}

/// The completion a CPDAG mediation fit is run for: its horizon, position,
/// class weight, identification and unfold indexer.
#[derive(Clone, Copy)]
struct CpdagCompletionSite<'a> {
    horizon: u32,
    horizon_index: usize,
    completion_idx: usize,
    weight: f64,
    identification: &'a IdentificationResult,
    indexer: &'a TemporalIndexer,
    envelope: &'a Arc<antecedent_identify::TemporalClassEnvelope>,
}

/// The completion atom the sweep pushed for the completion being estimated.
fn last_mediation_atom(
    run: &mut CpdagMediationRun,
) -> Result<&mut crate::result::StructuralResponseAtom, CausalError> {
    run.structural_atoms.last_mut().ok_or_else(|| CausalError::Compile {
        message: "CPDAG mediation sweep estimated a completion before recording its atom".into(),
    })
}

/// Accumulators the CPDAG mediation sweep threads across horizons and
/// completions and folds into the result.
struct CpdagMediationRun {
    slices: Vec<antecedent_estimate::TemporalMediationSlice>,
    primary: Option<(IdentificationResult, IdentifiedEstimand)>,
    primary_envelope: Option<Arc<antecedent_identify::TemporalClassEnvelope>>,
    refutations: Vec<antecedent_validate::RefutationReport>,
    aggregate_graph_dependent: bool,
    structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    full_mass_scope: bool,
    truncated_atoms: usize,
    block_replicates_ok: Option<u32>,
    /// The contrast the completions share, when the shared-design branch
    /// fires: at one horizon it is the result's scalar estimate. The Bayesian
    /// arm also keeps the composed posterior that contrast was taken from, so
    /// the reported interval is its quantile.
    shared_contrast: Option<TemporalMediationEstimate>,
    shared_posterior: Option<antecedent_estimate::CausalPosterior>,
    shared_horizons: usize,
    block_cancelled: bool,
    diagnostics: Vec<Diagnostic>,
}

impl CpdagMediationRun {
    fn new(horizon_count: usize) -> Self {
        Self {
            slices: Vec::with_capacity(horizon_count),
            primary: None,
            primary_envelope: None,
            refutations: Vec::new(),
            aggregate_graph_dependent: false,
            structural_atoms: Vec::new(),
            full_mass_scope: true,
            truncated_atoms: 0,
            block_replicates_ok: None,
            shared_contrast: None,
            shared_posterior: None,
            shared_horizons: 0,
            block_cancelled: false,
            diagnostics: vec![Diagnostic::new(
                "estimate.temporal_mediation.class_identified_set",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completion-specific mediation effects are returned as per-horizon identified \
                 sets; completion enumeration is not averaged",
            )],
        }
    }

    /// One horizon whose completions all fit the same mediation design: the
    /// identified set is that single contrast, so it is the result's scalar
    /// estimate. Frequentist reports the design's circular-block SE; Bayesian
    /// reports the composed posterior's quantile. The identified set stays
    /// beside it; each states its own calibration.
    fn scalar_contrast(&mut self) -> (EffectEstimate, Option<TemporalMediationEstimate>) {
        if let Some(contrast) = self.shared_contrast.take() {
            let interval = if self.shared_posterior.is_some() {
                "its posterior quantile is the reported interval"
            } else {
                "its circular-block SE is the reported interval"
            };
            self.diagnostics.push(Diagnostic::new(
                "estimate.temporal_mediation.class_shared_design_scalar",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "every completion is identified and fits the same mediation design, so the \
                     completion identified set is one contrast; that contrast is the scalar \
                     estimate and {interval}, beside the identified set"
                ),
            ));
            (
                contrast
                    .effect
                    .clone()
                    .with_block_family(antecedent_estimate::CircularBlockFamily::Mediation),
                Some(contrast),
            )
        } else {
            if self.shared_horizons > 0 {
                self.diagnostics.push(Diagnostic::new(
                    "estimate.temporal_mediation.multi_horizon",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "all requested horizons are retained in mediation_grid; the scalar estimate \
                     and scalar mediation fields are intentionally absent because no horizon is \
                     authoritative",
                ));
            }
            (nan_effect(), None)
        }
    }
}

/// One horizon's completion fit memos and shared-design bookkeeping.
struct CpdagHorizonState {
    /// Completions with one method, mediator, adjustment and status run one
    /// fit at this horizon; every other input of a fit is fixed per horizon.
    bayesian_fits: CompletionFitMemo<BayesianMediationFit, MediationFitKey>,
    point_fits: CompletionFitMemo<TemporalMediationEstimate, MediationFitKey>,
    refute_fits: CompletionFitMemo<Vec<antecedent_validate::RefutationReport>, MediationFitKey>,
    /// Completions (frequentist or Bayesian) that all fit the same mediation
    /// design (same estimand, same S(h)) estimate one contrast; that design is
    /// kept for the class interval.
    shared_design: Option<(IdentifiedEstimand, Arc<[antecedent_data::LaggedColumn]>)>,
    shared_status: Option<IdentificationStatus>,
    designs_agree: bool,
    effects: Vec<f64>,
    hit_cap: bool,
    completion_count: usize,
}

impl CpdagHorizonState {
    fn new(hit_cap: bool, completion_count: usize) -> Self {
        Self {
            bayesian_fits: CompletionFitMemo::default(),
            point_fits: CompletionFitMemo::default(),
            refute_fits: CompletionFitMemo::default(),
            shared_design: None,
            shared_status: None,
            designs_agree: true,
            effects: Vec::new(),
            hit_cap,
            completion_count,
        }
    }

    fn bounds(&self) -> (f64, f64) {
        let lower = self.effects.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = self.effects.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lower, upper)
    }

    fn every_completion_identified(&self) -> bool {
        self.effects.len() == self.completion_count && !self.hit_cap
    }
}

impl super::super::Study {
    pub(in crate::analysis::execute) fn execute_temporal_cpdag_mediation(
        &self,
        data: &TimeSeriesData,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        super::super::series_weights::refuse_unread_series_weights(
            data,
            "temporal CPDAG mediation",
        )?;
        let started = Instant::now();
        let estimator = TemporalMediationEstimator::new()
            .with_allow_natural_controlled_alias(true)
            .with_allow_gaps(self.declares_complete_case());
        let inputs = CpdagMediationInputs { data, query, estimator: &estimator, ctx };
        let mut run = CpdagMediationRun::new(query.horizons.len());
        let mut class_weights = TemporalClassWeights::new(self.declared.class_prior.as_ref());
        let mut temporal_budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        for (horizon_index, &horizon) in query.horizons.iter().enumerate() {
            let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
            witness.horizon_steps = horizon;
            let bundle = self.identify_temporal_class_reusing_within(
                IdentifierId::GeneralizedAdjustment,
                &witness,
                physical,
                &mut temporal_budget,
                ctx,
            )?;
            let envelope = &bundle.envelope.envelope;
            let hit_cap = envelope.truncated_completions > 0;
            run.full_mass_scope &= !hit_cap;
            run.truncated_atoms += envelope.truncated_completions;
            run.aggregate_graph_dependent |= hit_cap;
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            let mut horizon_state = CpdagHorizonState::new(hit_cap, envelope.cases.len());
            for (completion_idx, (case, indexer)) in
                envelope.cases.iter().zip(&bundle.envelope.indexers).enumerate()
            {
                self.cpdag_mediation_completion(
                    &inputs,
                    &mut run,
                    &mut horizon_state,
                    CpdagCompletionSite {
                        horizon,
                        horizon_index,
                        completion_idx,
                        weight: weights[completion_idx],
                        identification: &case.result,
                        indexer,
                        envelope: &bundle.envelope,
                    },
                )?;
            }
            self.cpdag_mediation_horizon_slice(&inputs, &mut run, horizon_state, horizon)?;
        }
        self.finish_cpdag_mediation(physical, query, run, started)
    }

    /// Fits, refutes and records one CPDAG completion's mediation effect at a
    /// horizon, folding it into the horizon's shared-design bookkeeping.
    fn cpdag_mediation_completion(
        &self,
        inputs: &CpdagMediationInputs<'_>,
        run: &mut CpdagMediationRun,
        horizon_state: &mut CpdagHorizonState,
        site: CpdagCompletionSite<'_>,
    ) -> Result<(), CausalError> {
        let CpdagMediationInputs { data, query, .. } = *inputs;
        let CpdagCompletionSite {
            horizon,
            horizon_index,
            completion_idx,
            weight,
            identification,
            indexer,
            envelope,
        } = site;
        run.structural_atoms.push(bare_class_atom(
            class_completion_key(horizon_index, completion_idx),
            weight,
            identification.status,
        ));
        if !identification_status_ok_for_case(identification.status) {
            run.aggregate_graph_dependent = true;
            return Ok(());
        }
        let estimand = select_estimand(identification, EstimatorId::TemporalMediation)?;
        let mut horizon_query = query.clone();
        horizon_query.horizons = Arc::from([horizon]);
        let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
        let adjustment: Arc<[antecedent_data::LaggedColumn]> = estimand
            .adjustment_set
            .iter()
            .filter_map(|dense| {
                let key = indexer.key_of(dense.raw()).ok()?;
                lagged_column_relative_to_outcome(key, outcome_offset)
            })
            .collect::<Vec<_>>()
            .into();
        // Tracked for both inference modes: whether every completion at this
        // horizon fits the same mediation design (same estimand, same S(h)).
        // When it does, the class's identified set collapses to one contrast,
        // which the post-loop branch below recomputes canonically instead of
        // reusing whichever completion happened to run last.
        match horizon_state.shared_design.as_ref() {
            None => {
                horizon_state.shared_design = Some((estimand.clone(), adjustment.clone()));
                horizon_state.shared_status = Some(identification.status);
            }
            Some((first, first_adjustment)) => {
                horizon_state.designs_agree &= first.adjustment_set == estimand.adjustment_set
                    && first_adjustment.as_ref() == adjustment.as_ref();
            }
        }
        let input = ClassMediationInput {
            data,
            estimand: &estimand,
            query: &horizon_query,
            adjustment: &adjustment,
            status: identification.status,
        };
        let estimate =
            self.cpdag_completion_estimate(inputs, run, horizon_state, input, horizon)?;
        self.cpdag_completion_refutations(
            inputs,
            run,
            horizon_state,
            (input, &estimate),
            (horizon, completion_idx),
        )?;
        last_mediation_atom(run)?.value =
            horizon_state.effects.last().copied().map(ResponseValue::Scalar);
        if run.primary.is_none() {
            run.primary = Some((identification.clone(), estimand));
            run.primary_envelope = Some(envelope.clone());
        }
        Ok(())
    }

    /// One CPDAG completion's mediation estimate under the study's inference
    /// mode; records its effect (and, Bayesian, its posterior) on the sweep.
    fn cpdag_completion_estimate(
        &self,
        inputs: &CpdagMediationInputs<'_>,
        run: &mut CpdagMediationRun,
        horizon_state: &mut CpdagHorizonState,
        input: ClassMediationInput<'_>,
        horizon: u32,
    ) -> Result<TemporalMediationEstimate, CausalError> {
        match &self.settings.inference {
            InferenceMode::Bayesian(cfg) => {
                let (composed, conflicts) = fit_class_bayesian_mediation(
                    &mut horizon_state.bayesian_fits,
                    cfg,
                    input,
                    horizon,
                    inputs.ctx,
                )?;
                for summary in &conflicts {
                    push_conflict_diagnostics(&mut run.diagnostics, summary);
                }
                let mean = composed.summaries.mean.first().copied().unwrap_or(f64::NAN);
                horizon_state.effects.push(mean);
                last_mediation_atom(run)?.posterior = Some(composed.clone());
                Ok(antecedent_estimate::TemporalMediationEstimate {
                    effect: effect_from_posterior(&composed)?,
                    total: None,
                    direct: None,
                    mediated: None,
                })
            }
            InferenceMode::Frequentist => {
                let estimate = fit_class_frequentist_mediation(
                    &mut horizon_state.point_fits,
                    inputs.estimator,
                    input,
                    inputs.ctx,
                )?;
                horizon_state.effects.push(estimate.effect.ate);
                Ok(estimate)
            }
        }
    }

    /// The refutation reports of one CPDAG completion's estimate, labelled by
    /// horizon and completion, when the study refutes.
    fn cpdag_completion_refutations(
        &self,
        inputs: &CpdagMediationInputs<'_>,
        run: &mut CpdagMediationRun,
        horizon_state: &mut CpdagHorizonState,
        (input, estimate): (ClassMediationInput<'_>, &TemporalMediationEstimate),
        (horizon, completion_idx): (u32, usize),
    ) -> Result<(), CausalError> {
        if self.settings.refute == RefuteSuite::None {
            return Ok(());
        }
        let mut reports = refute_class_mediation(
            &mut horizon_state.refute_fits,
            input,
            estimate,
            (self.settings.refute == RefuteSuite::Full, self.declares_complete_case()),
            inputs.ctx,
        )?;
        for report in &mut reports {
            report.refuter = Arc::from(format!(
                "horizon.{horizon}.completion.{completion_idx}.{}",
                report.refuter
            ));
        }
        run.refutations.extend(reports);
        Ok(())
    }

    /// Closes one CPDAG mediation horizon: the identified-set bounds, the
    /// shared-design class contrast when it applies, and the horizon's slice.
    fn cpdag_mediation_horizon_slice(
        &self,
        inputs: &CpdagMediationInputs<'_>,
        run: &mut CpdagMediationRun,
        mut horizon_state: CpdagHorizonState,
        horizon: u32,
    ) -> Result<(), CausalError> {
        if horizon_state.effects.is_empty() {
            return Err(CausalError::Compile {
                message: format!(
                    "TemporalCpdag mediation has no identified completion at horizon {horizon}"
                ),
            });
        }
        let (lower, upper) = horizon_state.bounds();
        let mut local_diagnostics = vec![Diagnostic::new(
            "identify.temporal_mediation.cpdag_envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "horizon={horizon}, identified_completions={}, examined_completions={}, \
                 capped={}",
                horizon_state.effects.len(),
                horizon_state.completion_count,
                horizon_state.hit_cap
            ),
        )];
        let every_completion_identified = horizon_state.every_completion_identified();
        let (slice_estimate, slice_uncertainty) = self.cpdag_class_slice_estimate(
            inputs,
            run,
            &mut horizon_state,
            horizon,
            &mut local_diagnostics,
        )?;
        run.slices.push(antecedent_estimate::TemporalMediationSlice {
            horizon,
            identification_status: if every_completion_identified {
                IdentificationStatus::PartiallyIdentified
            } else {
                IdentificationStatus::GraphDependent
            },
            method: Arc::from("temporal_mediation.cpdag_completion_envelope"),
            adjustment: Arc::from([]),
            estimate: slice_estimate,
            uncertainty: slice_uncertainty,
            identified_set: Some(antecedent_estimate::TemporalMediationIdentifiedSet {
                lower,
                upper,
            }),
            diagnostics: local_diagnostics,
        });
        Ok(())
    }

    /// Every completion identified and fitting the one shared design: the
    /// identified set is a single contrast, so the design's own circular-block
    /// SE (the `TemporalDag` route's construction, same stream) is the class
    /// interval. Otherwise the set stays unbanded.
    fn cpdag_class_slice_estimate(
        &self,
        inputs: &CpdagMediationInputs<'_>,
        run: &mut CpdagMediationRun,
        horizon_state: &mut CpdagHorizonState,
        horizon: u32,
        local_diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (TemporalMediationEstimate, antecedent_estimate::TemporalMediationUncertainty),
        CausalError,
    > {
        let (lower, upper) = horizon_state.bounds();
        let every_completion_identified = horizon_state.every_completion_identified();
        let shared_design = horizon_state.shared_design.take();
        match shared_design.as_ref() {
            Some((shared_estimand, shared_adjustment))
                if horizon_state.designs_agree
                    && every_completion_identified
                    // One shared design fits every completion, so their
                    // contrasts are the same number, bit for bit.
                    && lower.total_cmp(&upper) == std::cmp::Ordering::Equal =>
            {
                let shared = (shared_estimand, shared_adjustment);
                match &self.settings.inference {
                    InferenceMode::Frequentist => self.cpdag_shared_design_frequentist(
                        inputs,
                        run,
                        shared,
                        horizon,
                        local_diagnostics,
                    ),
                    InferenceMode::Bayesian(cfg) => Self::cpdag_shared_design_bayesian(
                        inputs,
                        run,
                        horizon_state,
                        cfg,
                        shared,
                        horizon,
                    ),
                }
            }
            _ => Ok((
                TemporalMediationEstimate {
                    effect: nan_effect(),
                    total: None,
                    direct: None,
                    mediated: None,
                },
                antecedent_estimate::TemporalMediationUncertainty::Unavailable,
            )),
        }
    }

    /// The shared mediation design's class contrast under the circular-block
    /// bootstrap, recorded as the sweep's shared contrast at a single horizon.
    fn cpdag_shared_design_frequentist(
        &self,
        inputs: &CpdagMediationInputs<'_>,
        run: &mut CpdagMediationRun,
        (shared_estimand, shared_adjustment): (
            &IdentifiedEstimand,
            &Arc<[antecedent_data::LaggedColumn]>,
        ),
        horizon: u32,
        local_diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (TemporalMediationEstimate, antecedent_estimate::TemporalMediationUncertainty),
        CausalError,
    > {
        let mut qh = inputs.query.clone();
        qh.horizons = Arc::from([horizon]);
        let (mediation, block) = inputs
            .estimator
            .estimate_with_block_bootstrap(
                inputs.data,
                shared_estimand,
                &qh,
                shared_adjustment,
                antecedent_estimate::BlockBootstrap {
                    replicates: self.settings.bootstrap_replicates,
                    stream_base: mediation_block_stream(MediationRoute::Temporal, horizon),
                },
                inputs.ctx,
            )
            .map_err(CausalError::from)?;
        run.block_replicates_ok = Some(
            run.block_replicates_ok
                .map_or(block.replicates_ok, |ok: u32| ok.min(block.replicates_ok)),
        );
        run.block_cancelled |= mediation.effect.uncertainty.bootstrap.cancelled;
        run.shared_horizons += 1;
        if inputs.query.horizons.len() == 1 {
            run.shared_contrast = Some(mediation.clone());
        }
        local_diagnostics.extend(temporal_dependence_se_diagnostics(
            antecedent_estimate::CircularBlockFamily::Mediation,
            block.block_length,
            block.rows,
            block.kernel_bias,
            block.effective_rows,
            block.replicates_attempted > 0,
            &format!(
                "horizon {horizon}: every completion fits the same mediation \
                 design, so the completion identified set is one contrast; \
                 one shared circular-block replicate of consecutive \
                 lag-aligned rows refits all three mechanism regressions, \
                 {}/{} replicates",
                block.replicates_ok, block.replicates_attempted,
            ),
        ));
        let uncertainty = if block.replicates_attempted > 0 {
            antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                requested: mediation.effect.se_bootstrap(),
                block,
            }
        } else {
            antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
                standard_error: None,
            }
        };
        Ok((mediation, uncertainty))
    }

    /// Every completion fits the same design, so the class's identified set is
    /// one contrast: recompute it canonically from the shared design (mirrors
    /// the frequentist branch) rather than reusing whichever completion's
    /// composed posterior happened to run last.
    fn cpdag_shared_design_bayesian(
        inputs: &CpdagMediationInputs<'_>,
        run: &mut CpdagMediationRun,
        horizon_state: &mut CpdagHorizonState,
        cfg: &BayesianConfig,
        (shared_estimand, shared_adjustment): (
            &IdentifiedEstimand,
            &Arc<[antecedent_data::LaggedColumn]>,
        ),
        horizon: u32,
    ) -> Result<
        (TemporalMediationEstimate, antecedent_estimate::TemporalMediationUncertainty),
        CausalError,
    > {
        let mut qh = inputs.query.clone();
        qh.horizons = Arc::from([horizon]);
        let status = horizon_state
            .shared_status
            .unwrap_or(IdentificationStatus::NonparametricallyIdentified);
        let (composed, conflicts) = fit_class_bayesian_mediation(
            &mut horizon_state.bayesian_fits,
            cfg,
            ClassMediationInput {
                data: inputs.data,
                estimand: shared_estimand,
                query: &qh,
                adjustment: shared_adjustment,
                status,
            },
            horizon,
            inputs.ctx,
        )?;
        for summary in &conflicts {
            push_conflict_diagnostics(&mut run.diagnostics, summary);
        }
        let mediation = TemporalMediationEstimate {
            effect: effect_from_posterior(&composed)?,
            total: None,
            direct: None,
            mediated: None,
        };
        run.shared_horizons += 1;
        if inputs.query.horizons.len() == 1 {
            run.shared_contrast = Some(mediation.clone());
            run.shared_posterior = Some(composed.clone());
        }
        let summary = |index: usize| antecedent_estimate::MediationPosteriorSummary {
            mean: composed.summaries.mean[index],
            standard_deviation: composed.summaries.sd[index],
            q025: composed.summaries.q025[index],
            q975: composed.summaries.q975[index],
        };
        let uncertainty = antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
            requested: summary(0),
            total: summary(1),
            direct: summary(2),
            mediated: summary(3),
            n_draws: composed.draws.n_draws,
            backend: Arc::clone(&composed.diagnostics.backend_id),
        };
        Ok((mediation, uncertainty))
    }

    /// The CPDAG mediation sweep's result: scalar contrast, response envelope,
    /// structural mixture and the identified-execute finish.
    fn finish_cpdag_mediation(
        &self,
        physical: &PhysicalExecutionPlan,
        query: &antecedent_core::MediationQuery,
        mut run: CpdagMediationRun,
        started: Instant,
    ) -> Result<StudyResult, CausalError> {
        let (scalar_estimate, scalar_mediation) = run.scalar_contrast();
        let (mut identification, estimand) =
            run.primary.take().ok_or_else(|| CausalError::Compile {
                message: "TemporalCpdag mediation missing primary completion".into(),
            })?;
        identification.status = if run.aggregate_graph_dependent {
            IdentificationStatus::GraphDependent
        } else {
            IdentificationStatus::PartiallyIdentified
        };
        let grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from(std::mem::take(&mut run.slices)),
            joint_posterior: false,
        };
        let structural = self.cpdag_structural_mixture(&mut run, &grid, query)?;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                scalar_estimate,
                Strategy::new(
                    IdentifierId::GeneralizedAdjustment,
                    if self.settings.inference.is_bayesian() {
                        EstimatorId::BayesianTemporalMediation
                    } else {
                        EstimatorId::TemporalMediation
                    },
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(self.temporal_class_cache_covers(&query.horizons))
            .extra_diagnostics(run.diagnostics)
            .refutations(run.refutations)
            .mediation(scalar_mediation)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(run.block_replicates_ok)
            .cancelled(run.block_cancelled)
            .extras(IdentifiedExecuteExtras {
                certificate: run.primary_envelope.map(|envelope| {
                    crate::Identification::TemporalEnvelope {
                        envelope: Arc::unwrap_or_clone(envelope),
                        strategy: IdentifierId::GeneralizedAdjustment,
                        structure_version: self.graph.version(),
                    }
                }),
                mediation_grid: Some(grid),
                structural_response: Some(structural),
                n_draws: run
                    .shared_posterior
                    .as_ref()
                    .and_then(|posterior| u32::try_from(posterior.draws.n_draws).ok()),
                posterior: run.shared_posterior,
                ..Default::default()
            }),
        ))
    }

    /// The structural mixture over the sweep's completion atoms, with the
    /// per-horizon identified-set bounds as its response envelope.
    fn cpdag_structural_mixture(
        &self,
        run: &mut CpdagMediationRun,
        grid: &antecedent_estimate::TemporalMediationGrid,
        query: &antecedent_core::MediationQuery,
    ) -> Result<crate::result::StructuralResponseMixture, CausalError> {
        let bounds = grid
            .slices
            .iter()
            .map(|slice| slice.identified_set.as_ref().map(|set| (set.lower, set.upper)))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| CausalError::Compile {
                message: "CPDAG mediation grid is missing a per-horizon identified set".into(),
            })?;
        let response_envelope = antecedent_core::ResponseEnvelope {
            grid: query.horizons.iter().map(|h| f64::from(*h)).collect::<Vec<_>>().into(),
            dimension: 1,
            lower: bounds.iter().map(|&(lower, _)| lower).collect::<Vec<_>>().into(),
            upper: bounds.iter().map(|&(_, upper)| upper).collect::<Vec<_>>().into(),
        };
        let conditional = self.declared.class_prior.as_ref().and_then(|_| {
            temporal_class_response_mean(
                &run.structural_atoms,
                &response_envelope,
                query.horizons.len(),
                run.full_mass_scope,
            )
        });
        let identified_mass = run
            .structural_atoms
            .iter()
            .filter(|atom| atom.value.is_some())
            .map(|atom| atom.weight)
            .sum::<f64>()
            / query.horizons.len() as f64;
        Ok(crate::result::StructuralResponseMixture {
            weight_basis: if self.declared.class_prior.is_some() {
                crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
            } else {
                crate::result::StructuralWeightBasis::CompletionEnumeration
            },
            atoms: std::mem::take(&mut run.structural_atoms),
            identified_mass,
            unidentified_mass: (1.0 - identified_mass).max(0.0),
            unevaluable_mass: 0.0,
            subsampled_out_mass: 0.0,
            identified_set: Some(response_envelope),
            identified_set_interval: None,
            conditional_on_identified: conditional,
            full_mass_scope: run.full_mass_scope,
            truncated_atoms: run.truncated_atoms,
            aggregation_policy: None,
        })
    }
}
