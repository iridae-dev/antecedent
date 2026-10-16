// SPDX-License-Identifier: MIT OR Apache-2.0

use super::completion_fit_memo::{CompletionFitMemo, MediationFitKey, ResponseFitKey};
use super::temporal_effect_core::FitFinishMeta;
use super::{
    Arc, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte, BayesianValidation,
    CausalError, CausalExprArena, CausalPosterior, CausalQuery, CausalResponse, ClassAtomOutcome,
    DerivationMeta, Diagnostic, DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate,
    EnvelopeAtomFit, EnvelopeOptions, EnvelopeRefutation, EnvelopeRefuteAtom, EstimandMethod,
    EstimationWorkspace, EstimatorId, ExecuteRun, ExecutionContext, ExprNode, GraphClass,
    GraphEffectDraws, GraphIdentFlag, IdentificationBasis, IdentificationEnvelope,
    IdentificationResult, IdentificationStatus, IdentifiedClaim, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedResultContext, IdentifierId,
    InferenceDiagnostics, InferenceMode, Instant, Intervention, MediationRoute,
    ObservationMechanismEstimator, ObservationSpec, OutcomeExprId, OverlapPolicy,
    PhysicalExecutionPlan, PriorSet, RefuteSuite, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SEQUENCE_LINEAR_SEM, Strategy, StudyResult,
    TableView, TabularData, TemporalBackdoorIdentifier, TemporalDag, TemporalEffectQuery,
    TemporalIndexer, TemporalLinearAdjustment, TemporalMediationEstimate,
    TemporalMediationEstimator, TemporalMediationIdentifier, TemporalRefutation,
    TemporalResponseEstimator, TimeSeriesData, TupleObservationTarget, TupleReplicates,
    TupleSurface, ValidationSink, VariableId, WeightedGraphSamples, aggregate_effect_envelope,
    bayesian_gcomp, bayesian_temporal_gcomp, completion_fit_seed, effect_from_posterior,
    elapsed_ns, envelope_to_identification_result_for, identification_status_ok_for_case,
    identified_set_interval_diagnostics, identify_cached_diagnostic,
    identify_temporal_cpdag_configured, identify_temporal_pag_configured, mediation_block_stream,
    mediation_mechanism_seed, merge_posterior_notes, nan_effect, posterior_identified_set_interval,
    posterior_note_diagnostics, provenance_ids, push_conflict_diagnostics, push_unique_diagnostic,
    report_identify_compute, require_identified, resolve_bayesian_prior_with_conflict,
    resolve_envelope_prior_anchor, response_functional_is_derivative,
    run_envelope_bayesian_full_validation, same_fit_inputs, same_fitted_problem, select_estimand,
    sequential_linear_sem_assumption, short_series_warning, temporal_refuter_average,
    tuple_block_observation_replicates, with_conflict_summary,
};
use crate::planner::RouteDefaults;
#[path = "temporal_response_execution.rs"]
pub(super) mod checked_temporal_response;
// The identification crate owns the observation-assumption id and text table;
// the temporal path records the same assumptions from the same body.
use antecedent_identify::response::append_observation_assumptions as append_temporal_observation_assumptions;

impl super::Study {
    pub(super) fn execute_temporal_cpdag_mediation(
        &self,
        data: &TimeSeriesData,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let estimator = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
        let inputs = CpdagMediationInputs { data, query, estimator: &estimator, ctx };
        let mut run = CpdagMediationRun::new(query.horizons.len());
        let mut class_weights = TemporalClassWeights::new(self.class_prior.as_ref());
        for (horizon_index, &horizon) in query.horizons.iter().enumerate() {
            let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
            witness.horizon_steps = horizon;
            let bundle =
                self.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &witness)?;
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
        match &self.inference {
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
        if self.refute == RefuteSuite::None {
            return Ok(());
        }
        let mut reports = refute_class_mediation(
            &mut horizon_state.refute_fits,
            input,
            estimate,
            self.refute == RefuteSuite::Full,
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
                match &self.inference {
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
                    replicates: self.bootstrap_replicates,
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
                    if self.inference.is_bayesian() {
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
        let conditional = self.class_prior.as_ref().and_then(|_| {
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
            weight_basis: if self.class_prior.is_some() {
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
        })
    }

    pub(crate) fn mediation_adjustment(
        &self,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        horizon: u32,
    ) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError> {
        if let Some(entry) =
            self.temporal_identification_cache.as_deref().and_then(|cache| cache.get(horizon))
        {
            return Ok(lagged_adjustment_from_entry(entry));
        }
        let ider = TemporalMediationIdentifier {
            allow_natural_controlled_alias: true,
            ..TemporalMediationIdentifier::new()
        };
        let (identification, temporal) =
            ider.identify_with_horizon(graph, query, horizon).map_err(CausalError::from)?;
        // The mediation estimand carries the shared set S(h) = I(h) plus the
        // mediator/outcome confounders, not the bare T→Y back-door set I(h).
        let estimand = identification.estimands.first().ok_or_else(|| CausalError::Compile {
            message: format!("temporal mediation identified no estimand at horizon {horizon}"),
        })?;
        let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(0);
        Ok(estimand
            .adjustment_set
            .iter()
            .filter_map(|&dense| {
                let key = temporal.indexer.key_of(dense.raw()).ok()?;
                lagged_column_relative_to_outcome(key, outcome_offset)
            })
            .collect::<Vec<_>>()
            .into())
    }

    pub(crate) fn execute_temporal(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        checked_bayesian: Option<&crate::analysis::CheckedBayesianTemporalEffectOperation>,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let (identification, estimand, indexer, identify_cached) =
            if let Some(operation) = checked_bayesian {
                let target = operation.target();
                if !target.matches_graph(graph)
                    || !target.matches_query(&CausalQuery::TemporalEffect(query.clone()))
                {
                    return Err(CausalError::Compile {
                    message:
                        "checked Bayesian temporal target differs from the prepared graph or query"
                            .into(),
                });
                }
                (
                    target.identification().clone(),
                    target.estimand().clone(),
                    target.indexer().clone(),
                    true,
                )
            } else if let Some(cache) = self.temporal_identification_cache.as_deref() {
                let entry = cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                    message: format!(
                        "prepared temporal identification missing horizon {}",
                        query.horizon_steps
                    ),
                })?;
                (entry.identification.clone(), entry.estimand.clone(), entry.indexer.clone(), true)
            } else {
                let id_res = TemporalBackdoorIdentifier::new()
                    .with_parent_adjustment_fallback()
                    .identify_temporal(graph, query)
                    .map_err(CausalError::from)?;
                let estimand = select_estimand(
                    &id_res.result,
                    if query.is_multi_step_sustained() {
                        EstimatorId::TemporalSequentialGcomp
                    } else {
                        EstimatorId::TemporalLinearAdjustment
                    },
                )?;
                (id_res.result, estimand, id_res.indexer, false)
            };
        let fit = super::temporal_effect_core::fit_temporal_effect(
            &super::temporal_effect_core::TemporalEffectInputs {
                data,
                graph,
                query,
                identification: &identification,
                estimand: &estimand,
                indexer: &indexer,
                inference: &self.inference,
                checked_bayesian,
                bootstrap_replicates: self.bootstrap_replicates,
                refute: self.refute,
                custom_validators: &self.custom_validators,
                split: self.split.as_ref(),
                latency_mode: self.latency_mode,
                completed_to_dag: super::temporal_effect_core::plan_completed_to_dag(physical),
                missing: self.missing.as_ref(),
            },
            ctx,
        )?;
        let point_only = fit.point_only;
        let missing_data = fit.missing_data.clone();
        let args = fit.finish_args(
            physical,
            identification,
            estimand,
            &indexer,
            query,
            FitFinishMeta {
                identify_cached,
                structure_version: self.graph.version(),
                wall_time_ns: elapsed_ns(started),
            },
        );
        let mut result = super::temporal_effect_core::seal_point_only(
            point_only,
            self.finish_identified_execute(args),
        );
        result.missing_data = missing_data;
        Ok(result)
    }

    /// Temporal natural mediation on a resolved `TemporalDag`: identify per horizon, then fit
    /// the mechanism product (frequentist) or compose per-mechanism posteriors (Bayesian,
    /// isotropic mechanism priors only).
    pub(super) fn execute_temporal_mediation(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use super::temporal_mediation_core::{
            MediationInference, TemporalMediationInputs, fit_temporal_mediation,
            require_isotropic_mediation_prior,
        };
        let started = Instant::now();
        let inference = match &self.inference {
            InferenceMode::Frequentist => {
                MediationInference::Frequentist { bootstrap_replicates: self.bootstrap_replicates }
            }
            InferenceMode::Bayesian(config) => {
                require_isotropic_mediation_prior(config)?;
                MediationInference::Bayesian(config)
            }
        };
        let (cache, identify_cached) =
            if let Some(cache) = self.temporal_identification_cache.clone() {
                (cache, true)
            } else {
                report_identify_compute(ctx);
                (
                    Arc::new(crate::analysis::prepared::identify_temporal_mediation_horizons(
                        graph,
                        query,
                        inference.estimator_id(),
                    )?),
                    false,
                )
            };
        let clicks = mediation_horizon_clicks(&cache, query, |horizon| {
            self.mediation_adjustment(graph, query, horizon)
        })?;
        let fit = fit_temporal_mediation(
            &TemporalMediationInputs {
                data,
                query,
                horizons: &clicks,
                inference,
                validation: self.refute,
                missing: self.missing.as_ref(),
            },
            ctx,
        )?;
        let missing_data = fit.missing_data.clone();
        let args = fit.finish_args(
            physical,
            query,
            identify_cached,
            self.graph.version(),
            elapsed_ns(started),
        );
        let mut result = self.finish_identified_execute(args);
        result.missing_data = missing_data;
        Ok(result)
    }

    pub(super) fn execute_temporal_response(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let Some(temporal) = query.temporal.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal response route requires TemporalResponseSpec".into(),
            });
        };
        enforce_temporal_response_memory_budget(query, temporal, self.bootstrap_replicates, ctx)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let observed_bayes =
            self.inference.is_bayesian() && query.observation != ObservationSpec::Complete;
        let (cache, identify_cached) =
            self.temporal_response_identification(graph, query, temporal, (treatment, outcome))?;
        let aligned = identified_horizon_entries(&cache, temporal)?;
        let first = aligned[0];
        let (aggregate_status, mut aggregate_assumptions) =
            aggregate_temporal_horizon_evidence(aligned.iter().map(|entry| &entry.identification))?;
        let mut identification = first.identification.clone();
        identification.status = aggregate_status;
        identification.required_assumptions = aggregate_assumptions.clone();
        let estimand = first.estimand.clone();
        let run = ExecuteRun { physical, ctx, started, identify_cached };
        if super::response_functional_is_derivative(&query.functional) {
            let indexer = &first.indexer;
            return self.execute_temporal_derivative_response(
                DerivativeResponseInputs { data, query, temporal, indexer, treatment, outcome },
                IdentifiedClaim { identification, estimand },
                run,
            );
        }
        let identifications: Vec<_> =
            aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();

        query.require_licensed_temporal_observation()?;
        let observation = self.prepare_temporal_response_observation(
            &ResponseObservationInput {
                data,
                graph,
                query,
                outcome,
                aligned: &aligned,
                observed_bayes,
            },
            &mut identification,
            &mut aggregate_assumptions,
        )?;
        let source_data = data;
        let data = observation.series.as_ref().unwrap_or(data);

        if let Some(overlays) = antecedent_estimate::plan_from_response_query(query)
            .map_err(CausalError::from)?
            .and_then(|plan| plan.mechanism_overlays())
            .filter(|_| !observed_bayes)
        {
            return self.execute_temporal_sequence_response(
                SequenceResponseInputs {
                    data,
                    source_data,
                    graph,
                    query,
                    temporal,
                    overlays: &overlays,
                    aligned: &aligned,
                    observation_adjusted: observation.adjusted.as_ref(),
                },
                IdentifiedClaim { identification, estimand },
                (treatment, outcome),
                IdentificationBasis {
                    status: aggregate_status,
                    assumptions: &aggregate_assumptions,
                },
                run,
            );
        }

        let estimated = self.estimate_temporal_response(
            &TemporalResponseEstimation {
                data,
                source_data,
                graph,
                query,
                working_query: &observation.working_query,
                temporal,
                aligned: &aligned,
                identifications: &identifications,
                status: aggregate_status,
                assumptions: &aggregate_assumptions,
                observation_adjusted: observation.adjusted.as_ref(),
                observed_bayes,
                treatment,
                outcome,
            },
            ctx,
        )?;
        Ok(self.finish_temporal_response(
            data,
            &aligned,
            IdentifiedClaim { identification, estimand },
            estimated,
            (treatment, outcome),
            run,
        ))
    }

    /// The prepared identification cache of a temporal response, computing it
    /// from the graph when the study carries none.
    ///
    /// Every mechanism-overlay estimator (the observed-data Bayesian path and
    /// the frequentist/complete-data sequential g-formula path alike) reads
    /// only each horizon's identification status and adjustment set
    /// (identifiability is level-free); neither reads the schedule contrast's
    /// active level, since both simulate the query's own per-step intervention
    /// values directly off the unfolded SEM (or `overlays`, for the sequential
    /// engine) rather than off the certificate's literal. A per-node
    /// varying-dose schedule that the general-ID single-literal contrast
    /// cannot express (see `resolve_schedule_active_level`) is therefore still
    /// identifiable here: strip the levels before certifying so a genuine
    /// per-node dose schedule is not refused for a contrast neither
    /// consumption path reads.
    fn temporal_response_identification(
        &self,
        graph: &TemporalDag,
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
        (treatment, outcome): (VariableId, VariableId),
    ) -> Result<(Arc<crate::analysis::prepared::CachedTemporalIdentification>, bool), CausalError>
    {
        let schedule = match antecedent_estimate::plan_from_response_query(query) {
            Ok(Some(plan)) if plan.mechanism_overlays().is_some() => {
                let mut nodes = plan.identification_schedule(temporal);
                for node in &mut nodes {
                    node.2 = None;
                }
                Some(nodes)
            }
            Ok(_) => None,
            Err(error) => return Err(CausalError::from(error)),
        };
        Ok(if let Some(cache) = self.temporal_identification_cache.clone() {
            (cache, true)
        } else {
            (
                Arc::new(crate::analysis::prepared::identify_temporal_response_horizons(
                    graph,
                    EstimatorId::temporal_response_for(&self.inference),
                    &crate::analysis::prepared::TemporalResponseIdentificationRequest {
                        treatment,
                        outcome,
                        temporal,
                        target_population: &query.target_population,
                        schedule: schedule.as_deref(),
                        dose: crate::analysis::prepared::single_step_dose(query)?,
                    },
                )?),
                false,
            )
        })
    }

    /// A temporal derivative response: the local slope of the linear unfolded
    /// fit at the one requested horizon, published without an interval.
    fn execute_temporal_derivative_response(
        &self,
        inputs: DerivativeResponseInputs<'_>,
        claim: IdentifiedClaim,
        run: ExecuteRun<'_>,
    ) -> Result<StudyResult, CausalError> {
        let DerivativeResponseInputs { data, query, temporal, indexer, treatment, outcome } =
            inputs;
        let IdentifiedClaim { identification, estimand } = claim;
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        if temporal.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "a temporal derivative is the local slope at one horizon",
            });
        }
        let prior_scale = match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                if cfg.has_transfer() {
                    return Err(CausalError::Unsupported {
                        message: "temporal derivative uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed",
                    });
                }
                Some(cfg.prior_scale)
            }
            InferenceMode::Frequentist => None,
        };
        let probe = unit_contrast_query(
            temporal,
            (treatment, outcome),
            temporal.horizons[0],
            query.target_population.clone(),
        );
        let response = temporal_linear_derivative_response(
            data,
            query,
            &estimand,
            indexer,
            &identification,
            DerivativeSpec {
                treatment_offset: probe.try_treatment_offset()?,
                outcome_offset: probe.outcome_offset(),
                policy: &ctx.kernel_policy,
                prior_scale,
            },
        )?;
        let scalar = super::response_path::response_scalar_summary(&response).0;
        let estimator_id = EstimatorId::temporal_response_for(&self.inference);
        let estimate = EffectEstimate::new(
            scalar,
            f64::NAN,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::TemporalBackdoorUnfolded, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .wall_time_ns(elapsed_ns(started)),
        );
        result.payload = crate::result::ResultPayload::Response(response);
        result.interval = None;
        result.posterior = None;
        result.diagnostics.push(Diagnostic::new(
            "estimate.temporal.linear_derivative",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the slope of the linear unfolded temporal fit; no interval is published",
        ));
        Ok(result)
    }

    /// Licenses the response's observation spec and, for a non-Bayesian
    /// observed series, adjusts the series and the working query; records the
    /// observation assumptions on both assumption sets.
    fn prepare_temporal_response_observation(
        &self,
        input: &ResponseObservationInput<'_>,
        identification: &mut IdentificationResult,
        aggregate_assumptions: &mut antecedent_core::AssumptionSet,
    ) -> Result<ResponseObservation, CausalError> {
        let ResponseObservationInput { data, graph, query, outcome, aligned, observed_bayes } =
            *input;
        let mut working_query = query.clone();
        let mut observation_adjusted = None;
        let series_owned = if query.observation == ObservationSpec::Complete {
            None
        } else if observed_bayes {
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not licensed for temporal observation",
                });
            }
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
            append_temporal_observation_assumptions(query, aggregate_assumptions);
            None
        } else {
            if self.observation_delayed_entry.is_some() {
                return Err(CausalError::Unsupported {
                    message: "delayed entry is not licensed for temporal observation",
                });
            }
            let adjustment = contemporaneous_adjustment_variables(aligned);
            let outcome_regressors =
                temporal_observation_outcome_regressors(query, graph, outcome, aligned)?;
            let (series, adjusted) = ObservationMechanismEstimator::new(self.observation_options)
                .adjust_temporal_series(data, query, &adjustment, &outcome_regressors)
                .map_err(CausalError::from)?;
            append_temporal_observation_assumptions(
                query,
                &mut identification.required_assumptions,
            );
            append_temporal_observation_assumptions(query, aggregate_assumptions);
            working_query.observation = ObservationSpec::Complete;
            working_query.observation_assumptions = Arc::from([]);
            observation_adjusted = Some(adjusted);
            Some(series)
        };
        Ok(ResponseObservation {
            series: series_owned,
            adjusted: observation_adjusted,
            working_query,
        })
    }

    /// Estimates the temporal response surface (observed-Bayes, Bayesian or
    /// frequentist g-formula), applies the observation adjustment and its
    /// bootstrap band, and restores each horizon's own identification status.
    fn estimate_temporal_response(
        &self,
        input: &TemporalResponseEstimation<'_>,
        ctx: &ExecutionContext,
    ) -> Result<EstimatedTemporalResponse, CausalError> {
        let TemporalResponseEstimation {
            data,
            source_data,
            graph,
            query,
            working_query,
            temporal,
            aligned,
            identifications,
            status: aggregate_status,
            assumptions: aggregate_assumptions,
            observation_adjusted,
            observed_bayes,
            treatment,
            outcome,
        } = *input;
        // Surface SEs follow the Study bootstrap / replicate contract so Pulse,
        // single-step Sustained, and the dose × horizon surface report comparable
        // uncertainty. Replicates = 0 keeps the analytic OLS linear-functional SE.
        let mut estimator = TemporalResponseEstimator::new();
        estimator.inner.bootstrap_replicates =
            if observation_adjusted.is_some() { 0 } else { self.bootstrap_replicates };
        let mut conflict_summary = None;
        let mut observed_posterior = None;
        let mut response = if observed_bayes {
            let InferenceMode::Bayesian(cfg) = &self.inference else {
                return Err(CausalError::Unsupported {
                    message: "an observed Bayesian temporal response requires Bayesian inference",
                });
            };
            if cfg.has_transferred_source() {
                return Err(CausalError::Unsupported {message: "observed temporal mechanisms need independent per-mechanism priors; a transferred response coefficient prior has no such mapping"});
            }
            let (response, posterior) = antecedent_estimate::temporal_observed_bayes::estimate_observed_temporal_response(data, graph, antecedent_estimate::TemporalIdentification { horizons: identifications, status: aggregate_status, assumptions: aggregate_assumptions.clone() }, query, &bayesian_gcomp(cfg, ctx), ctx).map_err(CausalError::from)?;
            observed_posterior = Some(posterior);
            Ok(response)
        } else if let InferenceMode::Bayesian(cfg) = &self.inference {
            let mut bayes = bayesian_gcomp(cfg, ctx);
            let (resolved, conflict) = resolve_temporal_response_prior(
                cfg, data, temporal, aligned, treatment, outcome, ctx,
            )?;
            bayes.prior = resolved;
            conflict_summary = conflict;
            estimator.estimate_bayesian(data, antecedent_estimate::TemporalIdentification { horizons: identifications, status: aggregate_status, assumptions: aggregate_assumptions.clone() }, working_query, &bayes, ctx)
        } else {
            estimator.estimate(
                data,
                identifications,
                working_query,
                aggregate_status,
                aggregate_assumptions.clone(),
                ctx,
            )
        }
        .map_err(CausalError::from)?;
        if let Some(adjusted) = observation_adjusted {
            apply_temporal_observation_result(&mut response, adjusted);
            if self.bootstrap_replicates > 0 {
                if let Some(bootstrap) = bootstrap_observation_adjusted_temporal_response(
                    source_data,
                    query,
                    aligned,
                    ObservationBootstrap {
                        options: self.observation_options,
                        replicates: self.bootstrap_replicates,
                    },
                    ctx,
                )? {
                    apply_observation_bootstrap(
                        &mut response,
                        &bootstrap,
                        self.bootstrap_replicates,
                    );
                }
            }
        }
        // The estimator's public API accepts one aggregate status for callers
        // that have a homogeneous set of horizon witnesses. This execution path
        // has the richer per-horizon records, so retain their actual statuses
        // instead of repeating the first/aggregate label across the surface.
        if let Some(per_horizon) = response.horizon_identification.as_mut() {
            for (record, entry) in Arc::make_mut(per_horizon).iter_mut().zip(aligned) {
                record.status = entry.identification.status;
            }
        }
        Ok(EstimatedTemporalResponse { response, conflict_summary, observed_posterior })
    }

    /// Assembles the temporal response result from the estimated surface.
    fn finish_temporal_response(
        &self,
        data: &TimeSeriesData,
        aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
        claim: IdentifiedClaim,
        estimated: EstimatedTemporalResponse,
        (treatment, outcome): (VariableId, VariableId),
        run: ExecuteRun<'_>,
    ) -> StudyResult {
        let IdentifiedClaim { identification, estimand } = claim;
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let EstimatedTemporalResponse { response, conflict_summary, observed_posterior } =
            estimated;
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(lag_aligned_analysis_rows(data, aligned));
        let diagnostics = temporal_response_diagnostics(
            aligned,
            &response,
            conflict_summary.as_ref(),
            identify_cached,
        );
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::temporal_response_for(&self.inference),
                ),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                posterior: observed_posterior,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        )
    }

    fn execute_temporal_sequence_response(
        &self,
        inputs: SequenceResponseInputs<'_>,
        claim: IdentifiedClaim,
        (treatment, outcome): (VariableId, VariableId),
        basis: IdentificationBasis<'_>,
        run: ExecuteRun<'_>,
    ) -> Result<StudyResult, CausalError> {
        let SequenceResponseInputs { temporal, overlays, observation_adjusted, .. } = inputs;
        let IdentifiedClaim { identification, estimand } = claim;
        let IdentificationBasis { status: aggregate_status, assumptions } = basis;
        let mut assumptions = assumptions.clone();
        let ExecuteRun { ctx, .. } = run;
        if self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "multi-step Sequence overlays require no discovery-estimation split",
            });
        }
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.has_transfer() {
                return Err(CausalError::Unsupported {
                    message: "multi-step Sequence inference requires isotropic per-mechanism priors",
                });
            }
            Some(bayesian_gcomp(cfg, ctx))
        } else {
            None
        };
        assumptions.push(sequential_linear_sem_assumption(SEQUENCE_LINEAR_SEM));
        if overlays.iter().any(|overlay| overlay.bounds.is_some()) {
            assumptions
                .push(antecedent_estimate::SequentialMechanismOverlay::mean_target_assumption());
        }
        let surface =
            estimate_sequence_horizons(&inputs, outcome, bayes.as_ref(), &assumptions, ctx)?;
        let support = sequence_overlay_support(overlays, temporal, surface.uncertainty_complete);
        let SequenceHorizonSurface {
            mean,
            lower,
            upper,
            horizons,
            uncertainty_complete,
            last_posterior,
            horizon_posteriors,
            bootstrap_cancelled,
        } = surface;
        let mut response = CausalResponse {
            estimand: inputs.query.functional.clone(),
            identification_status: aggregate_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(
                    temporal.horizons.iter().map(|&h| f64::from(h)).collect::<Vec<_>>(),
                ),
                dimension: 1,
                mean: Arc::from(mean),
            }),
            uncertainty: if uncertainty_complete {
                ResponseUncertainty::PointwiseBand {
                    level: 0.95,
                    lower: Arc::from(lower),
                    upper: Arc::from(upper),
                    // Posterior quantiles when a Bayesian fit ran, else Wald bounds.
                    interpretation: if bayes.is_some() {
                        antecedent_core::IntervalInterpretation::Credible
                    } else {
                        antecedent_core::IntervalInterpretation::Confidence
                    },
                    draws: None,
                }
            } else {
                ResponseUncertainty::None
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.temporal_response.intervention_gcomp"),
            horizon_identification: Some(Arc::from(horizons)),
            interaction_structurally_zero: false,
        };
        if let Some(adjusted) = observation_adjusted {
            apply_temporal_observation_result(&mut response, adjusted);
        }
        // Frequentist bands come from one joint tuple bootstrap of the whole Sequence
        // surface (observation nuisance refit per replicate when the query carries one).
        if bayes.is_none() && self.bootstrap_replicates > 0 {
            self.apply_sequence_bootstrap_band(&inputs, outcome, &mut response, ctx)?;
        }
        Ok(self.finish_sequence_response(
            &inputs,
            IdentifiedClaim { identification, estimand },
            SequenceFinish {
                response,
                horizon_posteriors,
                last_posterior,
                bootstrap_cancelled,
                attach_scalar_posterior: temporal.horizons.len() == 1,
            },
            (treatment, outcome),
            run,
        ))
    }

    /// Replaces the sequence response's pointwise band with the joint tuple
    /// bootstrap band, refitting the observation nuisance per replicate when
    /// the query carries one.
    fn apply_sequence_bootstrap_band(
        &self,
        inputs: &SequenceResponseInputs<'_>,
        outcome: VariableId,
        response: &mut CausalResponse,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let band = bootstrap_sequence_response(
            inputs.source_data,
            inputs.query,
            SequenceMechanisms { graph: inputs.graph, overlays: inputs.overlays },
            outcome,
            inputs.aligned,
            ObservationBootstrap {
                options: self.observation_options,
                replicates: self.bootstrap_replicates,
            },
            ctx,
        )?;
        apply_tuple_bootstrap_band(
            response,
            &band,
            self.bootstrap_replicates,
            if inputs.observation_adjusted.is_some() {
                TupleBandSource::Observation
            } else {
                TupleBandSource::CompleteSequence
            },
        );
        Ok(())
    }

    /// Assembles the Sequence response result: the scalar summary, the
    /// overlay and per-horizon posterior disclosures, and the finish.
    fn finish_sequence_response(
        &self,
        inputs: &SequenceResponseInputs<'_>,
        claim: IdentifiedClaim,
        finish: SequenceFinish,
        (treatment, outcome): (VariableId, VariableId),
        run: ExecuteRun<'_>,
    ) -> StudyResult {
        let IdentifiedClaim { identification, estimand } = claim;
        let SequenceFinish {
            response,
            horizon_posteriors,
            last_posterior,
            bootstrap_cancelled,
            attach_scalar_posterior,
        } = finish;
        let ExecuteRun { physical, ctx, started, identify_cached } = run;
        let (scalar, standard_error) = super::response_path::response_scalar_summary(&response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(lag_aligned_analysis_rows(inputs.data, inputs.aligned));
        let mut diagnostics = vec![Diagnostic::new(
            "estimate.temporal.sequence_overlay",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "multi-step / joint Sequence runs as overlays on unfolded sequential g-computation; \
             identifier remains temporal.backdoor.unfolded; no last-step collapse",
        )];
        diagnostics.extend(response.support.warnings.iter().cloned());
        // Every horizon's posterior notes feed the same tempering / draw-floor
        // diagnostics the scalar path derives from `extras.posterior`; a
        // multi-horizon surface must not lose a disclosure raised at a horizon
        // other than the last one just because only the last posterior is
        // attached as the scalar artifact below.
        let mut seen_diagnostic_codes: std::collections::HashSet<Arc<str>> =
            diagnostics.iter().map(|d| Arc::clone(&d.code)).collect();
        for diagnostic in posterior_note_diagnostics(&horizon_posteriors) {
            push_unique_diagnostic(&mut diagnostics, &mut seen_diagnostic_codes, diagnostic);
        }
        if !attach_scalar_posterior && self.inference.is_bayesian() {
            diagnostics.push(Diagnostic::new(
                "estimate.temporal.sequence_posterior_not_attached",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the response retains per-horizon posterior summaries and pointwise intervals; \
                 no scalar posterior artifact is attached to a multi-horizon surface",
            ));
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "refute.temporal_response.skipped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar ATE refuters are not applicable to a function-valued temporal response",
        ));
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::temporal_response_for(&self.inference),
                ),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started))
            .cancelled(bootstrap_cancelled || ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(
                    Arc::clone(&response.provenance_id),
                    Arc::clone(&response.provenance_id),
                )),
                diagnostics: Some(diagnostics),
                response: Some(response),
                posterior: last_posterior,
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        )
    }

    pub(super) fn execute_temporal_class_response(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if super::response_functional_is_derivative(&query.functional) {
            return self.execute_temporal_class_derivative(data, query, physical, ctx);
        }
        let started = Instant::now();
        query.require_licensed_temporal_observation()?;
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
        if matches!(
            antecedent_estimate::plan_from_response_query(query),
            Ok(Some(
                antecedent_estimate::TemporalInterventionPlan::Sequential { .. }
                    | antecedent_estimate::TemporalInterventionPlan::Mechanisms { .. }
            ))
        ) {
            return self.execute_temporal_class_sequence_response(data, query, physical, ctx);
        }
        enforce_temporal_response_memory_budget(query, temporal, self.bootstrap_replicates, ctx)?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let sweep = ClassResponseSweep { data, query, temporal, treatment, outcome, ctx };
        let mut run = ClassResponseRun {
            assembly: ClassResponseAssembly::new(query, temporal)?,
            primary_estimand: None,
            primary_identification: None,
            primary_envelope: None,
            assumptions: antecedent_core::AssumptionSet::new(),
            diagnostics: Vec::new(),
            observation_atoms: Vec::new(),
        };
        let mut class_weights = TemporalClassWeights::new(self.class_prior.as_ref());
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            let effect_query =
                class_horizon_witness(query, temporal, (treatment, outcome), horizon);
            let bundle =
                self.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &effect_query)?;
            let envelope = &bundle.envelope.envelope;
            run.diagnostics.push(temporal_class_envelope_diagnostic(envelope, self.graph.class()));
            let weights = class_weights.for_envelope(&bundle.envelope)?;
            run.assembly.begin_horizon(envelope);
            let mut fits = ClassResponseHorizonFits::default();
            for (case_index, (case, indexer)) in
                envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
            {
                self.class_response_completion(
                    &sweep,
                    &mut run,
                    &mut fits,
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
    fn class_response_completion(
        &self,
        sweep: &ClassResponseSweep<'_>,
        run: &mut ClassResponseRun,
        fits: &mut ClassResponseHorizonFits,
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
        let series_owned =
            if qh.observation == ObservationSpec::Complete || self.inference.is_bayesian() {
                None
            } else {
                if self.observation_delayed_entry.is_some() {
                    return Err(CausalError::Unsupported {
                        message: "delayed entry is not licensed for temporal observation",
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
        if series_owned.is_some() && self.inference.is_frequentist() {
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
        Ok(Some(match &self.inference {
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
                estimator.inner.bootstrap_replicates =
                    if series_adjusted { 0 } else { self.bootstrap_replicates };
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
        } = run;
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
            self.inference.is_bayesian(),
            self.class_prior.is_some(),
        ));
        if query.observation != ObservationSpec::Complete {
            diagnostics.push(class_observation_band_note(class_observation_band_applied));
        }
        let structural = self.class_structural_mixture(
            structural_atoms,
            &parts,
            (temporal.horizons.len(), assembly.full_mass_scope, assembly.truncated_atoms),
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
            // Observation circular-block bands ride the class response when a single
            // completion identifies; otherwise requested replicates stay on the atoms and
            // the class band is withheld.
            bootstrap_replicates: class_observation_band_applied
                .then_some(self.bootstrap_replicates),
            started,
            ctx,
            refutations: Vec::new(),
            identify_provenance: None,
            estimate_provenance: None,
        }))
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
            || !self.inference.is_frequentist()
            || self.bootstrap_replicates == 0
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
                replicates: self.bootstrap_replicates,
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
        let conditional = self.class_prior.as_ref().and_then(|_| {
            temporal_class_response_mean(
                &atoms,
                &parts.response_envelope,
                n_horizons,
                full_mass_scope,
            )
        });
        crate::result::StructuralResponseMixture {
            weight_basis: if self.class_prior.is_some() {
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
        }
    }

    fn execute_temporal_class_sequence_response(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "class-aware Sequence overlays require no discovery-estimation split",
            });
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence requires TemporalResponseSpec".into(),
        })?;
        let plan = antecedent_estimate::plan_from_response_query(query)
            .map_err(CausalError::from)?
            .ok_or_else(|| CausalError::Compile {
                message: "class-aware Sequence missing intervention plan".into(),
            })?;
        let overlays = plan.mechanism_overlays().ok_or_else(|| CausalError::Compile {
            message: "class-aware Sequence missing mechanism overlays".into(),
        })?;
        let (treatment, outcome) = super::response_path::response_primary_pair(&query.functional)?;
        let bayes = if let InferenceMode::Bayesian(cfg) = &self.inference {
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
        let mut class_weights = TemporalClassWeights::new(self.class_prior.as_ref());
        for (horizon_index, &horizon) in temporal.horizons.iter().enumerate() {
            let effect_query =
                class_horizon_witness(query, temporal, (treatment, outcome), horizon);
            let bundle =
                self.identify_temporal_class(IdentifierId::GeneralizedAdjustment, &effect_query)?;
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
                    overlays: overlays.to_vec(),
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
                .then_some(self.bootstrap_replicates),
            started,
            ctx,
            refutations: Vec::new(),
            identify_provenance: None,
            estimate_provenance: None,
        }))
    }

    pub(super) fn execute_temporal_class(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.inference.is_bayesian()
            && query.effect_modifiers.is_empty()
            && !query.is_multi_step_sustained()
        {
            return self.execute_temporal_class_bayesian(data, query, physical, ctx);
        }
        self.execute_temporal_class_frozen_weights(data, query, physical, ctx)
    }

    /// The class routes whose fit and assembly the sealed operation shares:
    /// conditional point, linear adjustment and sequential g-computation.
    pub(super) fn execute_temporal_class_frozen_weights(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use super::temporal_class_effect_core::{
            TemporalClassEffectInputs, fit_temporal_class_effect,
        };
        let started = Instant::now();
        let (identifier_id, bundle, identify_cached) =
            self.temporal_class_proof(physical, query, ctx)?;
        let fit = fit_temporal_class_effect(
            &TemporalClassEffectInputs {
                data,
                query,
                bundle: &bundle,
                graph_class: self.graph.class(),
                inference: &self.inference,
                class_prior: self.class_prior.as_ref(),
                bootstrap_replicates: self.bootstrap_replicates,
                split: self.split.as_ref(),
                refute: self.refute,
                custom_validators: &self.custom_validators,
                latency_mode: self.latency_mode,
            },
            ctx,
        )?;
        let point_only = fit.point_only;
        let mut identification = envelope_to_identification_result_for(
            &bundle.envelope.envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        identification.status = bundle.envelope.envelope.status;
        let args = fit.finish_args(
            physical,
            identification,
            &bundle,
            identifier_id,
            query,
            FitFinishMeta {
                identify_cached,
                structure_version: self.graph.version(),
                wall_time_ns: elapsed_ns(started),
            },
        );
        Ok(super::temporal_effect_core::seal_point_only(
            point_only,
            self.finish_identified_execute(args),
        ))
    }

    /// The identifier and class proof of a one-shot class effect: the prepared
    /// cache when present, else a fresh identification.
    fn temporal_class_proof(
        &self,
        physical: &PhysicalExecutionPlan,
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<
        (IdentifierId, Arc<crate::analysis::prepared::CachedTemporalClassIdentification>, bool),
        CausalError,
    > {
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        if let Some(cache) = self.temporal_class_identification_cache.as_ref() {
            Ok((identifier_id, Arc::clone(cache), true))
        } else {
            report_identify_compute(ctx);
            Ok((identifier_id, self.identify_temporal_class(identifier_id, query)?, false))
        }
    }

    pub(super) fn execute_temporal_class_bayesian(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let InferenceMode::Bayesian(cfg) = &self.inference else {
            return Err(CausalError::Compile {
                message: "temporal class Bayesian execute requires Bayesian inference".into(),
            });
        };
        let identifier_id = physical.binding().identifier_or(RouteDefaults::PAG)?;
        let (bundle, identify_cached) =
            if let Some(cache) = self.temporal_class_identification_cache.as_ref() {
                (Arc::clone(cache), true)
            } else {
                report_identify_compute(ctx);
                (self.identify_temporal_class(identifier_id, query)?, false)
            };
        let envelope = &bundle.envelope.envelope;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            return Err(CausalError::not_identified(
                envelope.status,
                envelope.truncated_completions > 0,
                "temporal class-aware effect not identified (no identified mass in envelope)",
            ));
        }
        let class_masses = self
            .class_prior
            .as_ref()
            .map(|prior| prior.masses_for_envelope(&bundle.envelope))
            .transpose()?;
        let mut diagnostics =
            vec![temporal_class_envelope_diagnostic(envelope, self.graph.class())];
        if class_masses.is_none() {
            diagnostics.push(enumeration_not_probability_diagnostic());
        }
        let inputs = ClassBayesianInputs { data, query, cfg, ctx };
        let ClassBayesianPlan { weights, flags, keys, fit_atoms, primary_estimand } =
            plan_class_bayesian_atoms(&bundle.envelope, class_masses.as_deref())?;
        let mut fits = self.fit_class_bayesian_atoms(&inputs, &fit_atoms, &mut diagnostics)?;
        if fits.atoms.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal class-aware envelope had no estimable identified cases".into(),
            });
        }
        let identification = envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "temporal class Bayesian envelope missing estimand".into(),
        })?;
        let (estimate, posterior, mut structural_response) = Self::mix_class_bayesian_atoms(
            &inputs,
            (&bundle.envelope, class_masses.as_deref()),
            (weights, flags, keys),
            &mut fits,
            &mut diagnostics,
        )?;
        let (refutations, predictive_checks) = self.refute_class_bayesian_atoms(
            &inputs,
            &fits,
            &mut structural_response,
            &mut diagnostics,
        )?;
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, EstimatorId::BayesianTemporalGcomp),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                certificate: Some(crate::Identification::TemporalEnvelope {
                    envelope: (*bundle.envelope).clone(),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                posterior,
                structural_response: Some(structural_response),
                diagnostics: Some(diagnostics),
                predictive_checks,
                estimate_provenance: Some(
                    super::temporal_class_effect_core::class_envelope_provenance(
                        EstimatorId::BayesianTemporalGcomp,
                    ),
                ),
                ..Default::default()
            }),
        ))
    }

    /// Fits each identified completion's Bayesian temporal g-computation,
    /// sharing a posterior (and seed) between completions that fit the same
    /// problem under the same prior and status.
    fn fit_class_bayesian_atoms(
        &self,
        inputs: &ClassBayesianInputs<'_>,
        fit_atoms: &[ClassBayesianFitAtom],
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<ClassBayesianFits, CausalError> {
        let ClassBayesianInputs { data, query, cfg, ctx } = *inputs;
        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = 0;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
        let mut fits = ClassBayesianFits {
            atoms: Vec::new(),
            atom_seeds: Vec::new(),
            per_graph: Vec::new(),
            atom_means: Vec::new(),
            envelope_conflict: None,
            estimator: bayes.inner.clone(),
        };
        let mut ws = BayesianGCompWorkspace::default();
        for (key, estimand, status, indexer, weight) in fit_atoms {
            let prep = estimator
                .prepare(data, estimand, query, indexer, self.split.as_ref(), &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            let names =
                antecedent_estimate::temporal_coefficient_names(data, estimand, query, indexer)
                    .map_err(CausalError::from)?;
            let bprep = BayesianGComputationAte::from_prepared_temporal(&prep, names)
                .map_err(CausalError::from)?;
            let (resolved, conflict) = resolve_envelope_prior_anchor(cfg, &bprep, ctx)?;
            // Each completion draws from its own seed stream; completions fitting the
            // same problem under the same prior share a seed, so one model carries
            // one posterior.
            let seed = fits
                .atoms
                .iter()
                .zip(&fits.atom_seeds)
                .find(|(fit, _): &(&EnvelopeAtomFit, _)| {
                    fit.prior == resolved && same_fitted_problem(&fit.prep, &bprep)
                })
                .map_or_else(|| completion_fit_seed(ctx, *key), |(_, seed)| *seed);
            fits.atom_seeds.push(seed);
            bayes.inner.prior.clone_from(&resolved);
            bayes.inner.seed = seed;
            // An earlier completion that fitted every input of this fit (the problem,
            // the resolved prior, the seed they share, the identification status)
            // already holds its posterior, conflict summary included.
            let earlier = fits.atoms.iter().zip(&fits.atom_seeds).find(
                |(fit, fitted_seed): &(&EnvelopeAtomFit, _)| {
                    **fitted_seed == seed
                        && fit.status == *status
                        && fit.prior == resolved
                        && same_fit_inputs(&fit.prep, &bprep)
                },
            );
            let posterior = if let Some((fit, _)) = earlier {
                if let Some(summary) = conflict.as_ref() {
                    push_conflict_diagnostics(diagnostics, summary);
                }
                fit.posterior.clone()
            } else {
                let mut posterior =
                    bayes.fit(&bprep, *status, &mut ws, ctx).map_err(CausalError::from)?;
                if let Some(summary) = conflict.as_ref() {
                    // A transfer conflict is a per-completion diagnostic. It rides the
                    // atom posterior and the result diagnostics whether or not a class
                    // prior licenses a mixture, and it never changes identification.
                    push_conflict_diagnostics(diagnostics, summary);
                    posterior = with_conflict_summary(posterior, summary.clone());
                }
                posterior
            };
            fits.envelope_conflict = fits.envelope_conflict.take().or(conflict);
            let mean = posterior.summaries.mean.first().copied().unwrap_or_else(|| {
                posterior
                    .effect_column()
                    .and_then(|col| {
                        posterior.draws.column(col).ok().map(|draws| {
                            draws.iter().copied().sum::<f64>() / draws.len().max(1) as f64
                        })
                    })
                    .unwrap_or(f64::NAN)
            });
            fits.atom_means.push(mean);
            let draws = posterior
                .effect_column()
                .and_then(|col| posterior.draws.column(col).ok().map(|d| Arc::from(d.to_vec())));
            if let Some(effect_draws) = draws {
                fits.per_graph.push(GraphEffectDraws { graph_key: *key, effect_draws });
            }
            fits.atoms.push(EnvelopeAtomFit {
                key: *key,
                prep: bprep,
                posterior,
                status: *status,
                weight: *weight,
                estimand: estimand.clone(),
                indexer: Some(indexer.clone()),
                prior: resolved,
            });
        }
        fits.estimator = bayes.inner.clone();
        Ok(fits)
    }

    /// Mixes the atom posteriors into the class effect and builds the
    /// structural mixture with its posterior identified-set interval.
    fn mix_class_bayesian_atoms(
        inputs: &ClassBayesianInputs<'_>,
        (bundle_envelope, class_masses): (
            &antecedent_identify::TemporalClassEnvelope,
            Option<&[f64]>,
        ),
        (weights, flags, keys): (Vec<f64>, Vec<GraphIdentFlag>, Vec<u64>),
        fits: &mut ClassBayesianFits,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (EffectEstimate, Option<CausalPosterior>, crate::result::StructuralResponseMixture),
        CausalError,
    > {
        let ClassBayesianInputs { data, .. } = *inputs;
        let envelope = &bundle_envelope.envelope;
        let mut case_means: Vec<Option<f64>> = vec![None; envelope.cases.len()];
        let mut mean_idx = 0usize;
        for (i, case) in envelope.cases.iter().enumerate() {
            if identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty()
            {
                case_means[i] = fits.atom_means.get(mean_idx).copied();
                mean_idx += 1;
            }
        }
        let (estimate, posterior, weight_basis) = mix_class_effect_posteriors(
            ClassPosteriorMix {
                class_prior_supplied: class_masses.is_some(),
                truncated_completions: envelope.truncated_completions,
                evaluable_mass_positive: fits.atoms.iter().any(|atom| atom.weight > 0.0),
                weights,
                flags,
                keys,
                per_graph: &fits.per_graph,
                conflict: fits.envelope_conflict.take(),
                label: "temporal_class_envelope",
            },
            fits.atoms.iter().map(|atom| &atom.posterior),
            diagnostics,
        )?;
        let mut structural_response =
            temporal_class_structural_mixture(envelope, weight_basis, class_masses, &case_means);
        for atom in &mut structural_response.atoms {
            atom.posterior = fits
                .atoms
                .iter()
                .find(|fit| fit.key == atom.graph_key)
                .map(|fit| fit.posterior.clone());
        }
        structural_response.identified_set_interval = posterior_identified_set_interval(
            fits.atoms.iter().map(|fit| &fit.posterior),
            data.row_count(),
            envelope.truncated_completions > 0,
        );
        diagnostics.extend(
            structural_response
                .identified_set_interval
                .as_ref()
                .into_iter()
                .flat_map(identified_set_interval_diagnostics),
        );
        Ok((estimate, posterior, structural_response))
    }

    /// Runs the refutation suite and full Bayesian validation on each fitted
    /// completion, labelling reports by completion and attaching each
    /// validated posterior to its structural atom.
    fn refute_class_bayesian_atoms(
        &self,
        inputs: &ClassBayesianInputs<'_>,
        fits: &ClassBayesianFits,
        structural_response: &mut crate::result::StructuralResponseMixture,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> Result<
        (
            Vec<antecedent_validate::RefutationReport>,
            Vec<antecedent_validate::PredictiveCheckReport>,
        ),
        CausalError,
    > {
        let ClassBayesianInputs { data, query, cfg, ctx } = *inputs;
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = temporal_refuter_average(query);
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = Vec::new();
        let mut predictive_checks = Vec::new();
        for (atom, &seed) in fits.atoms.iter().zip(&fits.atom_seeds) {
            let refute_atom = EnvelopeRefuteAtom::from_fit(atom)?;
            let atom_ate = refute_atom.original.ate;
            let start = refutations.len();
            let (reports, na_diagnostics) = EnvelopeRefutation {
                data: &tabular,
                query: &ate_q,
                suite: self.refute,
                estimator: EstimatorId::BayesianTemporalGcomp.as_str(),
                custom: &self.custom_validators,
                temporal: Some(TemporalRefutation {
                    query,
                    split: self.split.as_ref(),
                    time_index: data.time_index(),
                }),
            }
            .run(std::slice::from_ref(&refute_atom), &mut refute_ws, ctx)?;
            refutations.extend(reports);
            diagnostics.extend(na_diagnostics);
            let mut atom_posterior = atom.posterior.clone();
            let atom_estimator = BayesianGComputationAte {
                prior: atom.prior.clone(),
                seed,
                ..fits.estimator.clone()
            };
            predictive_checks.extend(run_envelope_bayesian_full_validation(
                BayesianValidation { suite: self.refute, cfg, estimator: &atom_estimator },
                std::slice::from_ref(atom),
                &mut atom_posterior,
                atom_ate,
                ctx,
                super::super::latency::predictive_check_sims(self.latency_mode),
                ValidationSink { refutations: &mut refutations, diagnostics },
            )?);
            for report in &mut refutations[start..] {
                report.refuter = Arc::from(format!("completion.{}.{}", atom.key, report.refuter));
            }
            if let Some(structural) =
                structural_response.atoms.iter_mut().find(|value| value.graph_key == atom.key)
            {
                structural.posterior = Some(atom_posterior);
            }
        }
        Ok((refutations, predictive_checks))
    }

    /// Whether the prepared class cache holds a certificate for every horizon.
    pub(crate) fn temporal_class_cache_covers(&self, horizons: &[u32]) -> bool {
        self.temporal_class_identification_cache.as_deref().is_some_and(|cache| {
            horizons
                .iter()
                .all(|horizon| cache.by_horizon.iter().any(|(cached, _)| cached == horizon))
        })
    }

    pub(crate) fn identify_temporal_class(
        &self,
        identifier_id: IdentifierId,
        query: &TemporalEffectQuery,
    ) -> Result<Arc<crate::analysis::prepared::CachedTemporalClassIdentification>, CausalError>
    {
        if let Some(cache) = self.temporal_class_identification_cache.as_ref() {
            if let Some((_, envelope)) =
                cache.by_horizon.iter().find(|(horizon, _)| *horizon == query.horizon_steps)
            {
                return Ok(Arc::new(
                    crate::analysis::prepared::CachedTemporalClassIdentification {
                        envelope: Arc::clone(envelope),
                        by_horizon: Vec::new(),
                    },
                ));
            }
        }
        let mut config = antecedent_identify::GeneralizedAdjustmentConfig::default();
        if let Some(max) = self.max_completions {
            config.max_completions = max;
        }
        let mut envelope = match self.graph.class() {
            GraphClass::TemporalCpdag => {
                let cpdag = self.graph.as_temporal_cpdag().ok_or_else(|| CausalError::Compile {
                    message: "TemporalCpdag execute missing supplied graph".into(),
                })?;
                identify_temporal_cpdag_configured(identifier_id, cpdag, query, config)?
            }
            GraphClass::TemporalPag => {
                let pag = self.graph.as_temporal_pag().ok_or_else(|| CausalError::Compile {
                    message: "TemporalPag execute missing supplied graph".into(),
                })?;
                identify_temporal_pag_configured(identifier_id, pag, query, config)?
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "class-aware temporal execute requires TemporalCpdag or TemporalPag",
                });
            }
        };
        crate::identify_api::refine_temporal_class_identification(
            &mut envelope,
            &self.query,
            query.horizon_steps,
        )?;
        Ok(Arc::new(crate::analysis::prepared::CachedTemporalClassIdentification {
            envelope: Arc::new(envelope),
            by_horizon: Vec::new(),
        }))
    }
}

pub(super) fn enforce_temporal_response_memory_budget(
    query: &ResponseQuery,
    temporal: &antecedent_core::TemporalResponseSpec,
    bootstrap_replicates: u32,
    ctx: &ExecutionContext,
) -> Result<(), CausalError> {
    const MIN_BYTES_PER_CELL: u64 = 40;

    let doses = match &query.functional {
        antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } => match &treatment.grid {
            antecedent_core::GridSpec::Values(values) => values.len(),
            antecedent_core::GridSpec::Linspace { points, .. } => *points,
        },
        antecedent_core::ResponseFunctional::InterventionResponse { .. } => 1,
        _ => return Ok(()),
    };
    let cells = doses.checked_mul(temporal.horizons.len()).ok_or_else(|| {
        CausalError::Resource { message: "temporal response output cell count overflow".into() }
    })?;
    // Observation bands retain one surface per outer replicate. Account for
    // that buffer as well as the result before any models or draws are fitted.
    let bytes_per_cell = if query.observation == ObservationSpec::Complete {
        MIN_BYTES_PER_CELL
    } else {
        MIN_BYTES_PER_CELL + u64::from(bootstrap_replicates) * 8
    };
    let minimum_bytes = u64::try_from(cells)
        .ok()
        .and_then(|cells| cells.checked_mul(bytes_per_cell))
        .ok_or_else(|| CausalError::Resource {
            message: "temporal response output byte estimate overflow".into(),
        })?;
    let limit =
        [ctx.memory.soft_limit_bytes, ctx.memory.hard_limit_bytes].into_iter().flatten().min();
    if let Some(limit) = limit {
        if minimum_bytes > limit {
            return Err(CausalError::Resource {
                message: format!(
                    "temporal response needs at least {minimum_bytes} output bytes for {cells} \
                     cells; memory limit is {limit}; coarsen the dose grid or request fewer \
                     horizons"
                ),
            });
        }
    }
    Ok(())
}

/// Per-horizon completion weights for a class-aware analysis.
///
/// Caller-supplied class mass binds once, at the first horizon, and is then
/// read by completion fingerprint. Without a prior the enumeration weights are
/// normalized per horizon; they are not probabilities.
pub(super) struct TemporalClassWeights<'a> {
    prior: Option<&'a crate::ClassPrior>,
    binding: Option<crate::class_prior::ClassPriorBinding>,
}

impl<'a> TemporalClassWeights<'a> {
    pub(super) const fn new(prior: Option<&'a crate::ClassPrior>) -> Self {
        Self { prior, binding: None }
    }

    pub(super) fn for_envelope(
        &mut self,
        envelope: &antecedent_identify::TemporalClassEnvelope,
    ) -> Result<Vec<f64>, CausalError> {
        if let Some(prior) = self.prior {
            if let Some(binding) = &self.binding {
                return binding.masses_for_envelope(envelope);
            }
            let binding = crate::class_prior::ClassPriorBinding::bind(prior, envelope)?;
            let masses = binding.masses_for_envelope(envelope);
            self.binding = Some(binding);
            return masses;
        }
        let raw: Vec<f64> = envelope.envelope.cases.iter().map(|case| case.weight.0).collect();
        let total: f64 = raw.iter().sum();
        Ok(if total > 0.0 { raw.iter().map(|weight| weight / total).collect() } else { raw })
    }
}

/// Completion curves stitched across horizons by completion fingerprint.
///
/// Each horizon identifies its own envelope, and a PAG window can retain a
/// different completion set or order per horizon, so the positional case index
/// is not a stable identity. Keys are completion fingerprints, the same keys
/// [`crate::ClassPrior::from_pairs`] accepts.
fn lag_aligned_analysis_rows(
    data: &TimeSeriesData,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> u64 {
    let span = aligned
        .iter()
        .map(|entry| entry.indexer.history().saturating_add(entry.indexer.horizon()))
        .max()
        .unwrap_or(1);
    u64::try_from(
        data.row_count().saturating_sub(usize::try_from(span.saturating_sub(1)).unwrap_or(0)),
    )
    .unwrap_or(u64::MAX)
}

fn temporal_class_complete_values(
    atoms: &[crate::result::StructuralResponseAtom],
    horizon_fingerprints: &[Vec<u64>],
    envelope: &antecedent_core::ResponseEnvelope,
    n_horizons: usize,
) -> Vec<(u64, ResponseValue)> {
    let mut curves = std::collections::BTreeMap::<u64, Vec<Option<f64>>>::new();
    for atom in atoms {
        let values: &[f64] = match atom.value.as_ref() {
            Some(ResponseValue::Scalar(value)) => std::slice::from_ref(value),
            Some(ResponseValue::Surface { mean, .. }) => mean,
            _ => continue,
        };
        let horizon = (atom.graph_key >> 32) as usize;
        let case_index = (atom.graph_key & 0xffff_ffff) as usize;
        let Some(fingerprint) =
            horizon_fingerprints.get(horizon).and_then(|keys| keys.get(case_index)).copied()
        else {
            continue;
        };
        let curve = curves.entry(fingerprint).or_insert_with(|| vec![None; envelope.lower.len()]);
        for (cell, value) in values.iter().enumerate() {
            if let Some(target) = curve.get_mut(cell * n_horizons + horizon) {
                *target = Some(*value);
            }
        }
    }
    curves
        .into_iter()
        .filter_map(|(key, values)| {
            let values = values.into_iter().collect::<Option<Vec<_>>>()?;
            Some((
                key,
                ResponseValue::Surface {
                    grid: envelope.grid.clone(),
                    dimension: envelope.dimension,
                    mean: values.into(),
                },
            ))
        })
        .collect()
}

fn temporal_class_response_masses(
    atoms: &[crate::result::StructuralResponseAtom],
) -> (f64, f64, f64) {
    let total: f64 = atoms.iter().map(|atom| atom.weight).sum();
    let mut identified = 0.0;
    let mut unidentified = 0.0;
    let mut unevaluable = 0.0;
    for atom in atoms {
        if !identification_status_ok_for_case(atom.status) {
            unidentified += atom.weight;
        } else if atom.value.is_some() {
            identified += atom.weight;
        } else {
            unevaluable += atom.weight;
        }
    }
    if total > 0.0 {
        (identified / total, unidentified / total, unevaluable / total)
    } else {
        (0.0, 0.0, 0.0)
    }
}

pub(super) fn temporal_class_response_mean(
    atoms: &[crate::result::StructuralResponseAtom],
    envelope: &antecedent_core::ResponseEnvelope,
    n_horizons: usize,
    full_mass_scope: bool,
) -> Option<ResponseValue> {
    if !full_mass_scope || n_horizons == 0 {
        return None;
    }
    // Licensed contract: `conditional_on_identified` is published only when
    // every identified atom is aligned *and* unidentified/unevaluable mass is
    // zero. Renormalizing over identified atoms would hide retained class mass.
    let (_, unidentified, unevaluable) = temporal_class_response_masses(atoms);
    if unidentified > 0.0 || unevaluable > 0.0 {
        return None;
    }
    let n_cells = envelope.lower.len() / n_horizons;
    let mut mean = vec![0.0; envelope.lower.len()];
    let mut totals = vec![0.0; n_horizons];
    for atom in atoms {
        if atom.weight == 0.0 {
            continue;
        }
        if !identification_status_ok_for_case(atom.status) {
            continue;
        }
        let values: &[f64] = match atom.value.as_ref()? {
            ResponseValue::Scalar(value) => std::slice::from_ref(value),
            ResponseValue::Surface { mean, .. } => mean,
            _ => return None,
        };
        let horizon = usize::try_from(atom.graph_key >> 32).ok()?;
        if horizon >= n_horizons || values.len() != n_cells {
            return None;
        }
        totals[horizon] += atom.weight;
        for (cell, value) in values.iter().enumerate() {
            mean[cell * n_horizons + horizon] += atom.weight * value;
        }
    }
    for (index, value) in mean.iter_mut().enumerate() {
        let total = totals[index % n_horizons];
        if total <= 0.0 {
            return None;
        }
        *value /= total;
    }
    Some(ResponseValue::Surface {
        grid: envelope.grid.clone(),
        dimension: envelope.dimension,
        mean: Arc::from(mean),
    })
}

/// Envelope assembly of a class-aware temporal response, shared by the series and
/// panel owners.
///
/// One completion at one horizon is an *atom*: the owner fits it (a series surface, or
/// the panel's equal-weight average of unit surfaces) and hands the surface here. The
/// assembly keeps the pointwise minimum and maximum over the completions of each
/// (cell, horizon), every atom's structural record and support, and publishes the
/// identified set as a `PartiallyIdentified` envelope over the full surfaces — or, when
/// mass is unidentified, unevaluable or capped, the completions' surfaces as
/// `GraphDependent` atoms. The grid comes from the query, so it is the same whatever
/// the horizon count.
pub(super) struct ClassResponseAssembly {
    /// Cells of one horizon: the dose grid, or one intervention-response cell.
    cells_per_horizon: usize,
    n_horizons: usize,
    lower: Vec<f64>,
    upper: Vec<f64>,
    pub(super) structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    supports: Vec<antecedent_core::SupportReport>,
    horizon_supports: Vec<antecedent_core::SupportReport>,
    horizon_fingerprints: Vec<Vec<u64>>,
    /// No horizon capped its completion enumeration.
    pub(super) full_mass_scope: bool,
    pub(super) truncated_atoms: usize,
    horizon_values: Vec<Vec<f64>>,
    support_start: usize,
}

/// What [`ClassResponseAssembly::finish`] publishes.
pub(super) struct ClassResponseParts {
    pub response: CausalResponse,
    pub response_envelope: antecedent_core::ResponseEnvelope,
    pub identified_mass: f64,
    pub unidentified_mass: f64,
    pub unevaluable_mass: f64,
}

/// The identified-execute tail every class response executor (temporal Pulse/Sustained,
/// Sequence and panel) ends with: the strategy, certificate and extras are the same, only the
/// fitted pieces differ.
pub(super) struct ClassResponseTail<'a> {
    pub physical: &'a PhysicalExecutionPlan,
    pub identification: IdentificationResult,
    pub estimand: IdentifiedEstimand,
    /// Scalar summary of the response, or a NaN effect when the response carries the result.
    pub estimate: EffectEstimate,
    pub identifier_id: IdentifierId,
    pub endpoints: (VariableId, VariableId),
    /// The completion envelope the certificate retains.
    pub envelope: Option<Arc<antecedent_identify::TemporalClassEnvelope>>,
    pub response: CausalResponse,
    pub structural: crate::result::StructuralResponseMixture,
    pub diagnostics: Vec<Diagnostic>,
    pub identify_cached: bool,
    /// The bootstrap replicate count the result records (`None`: the class band is withheld).
    pub bootstrap_replicates: Option<u32>,
    pub started: Instant,
    pub ctx: &'a ExecutionContext,
    /// Refutation reports. Class routes pass an empty list; the graph-posterior
    /// route passes the reports it already computed.
    pub refutations: Vec<antecedent_validate::RefutationReport>,
    /// Overrides the identifier provenance step. `None` keeps the strategy default.
    pub identify_provenance: Option<(Arc<str>, Arc<str>)>,
    /// Overrides the estimator provenance step. `None` keeps the strategy default.
    pub estimate_provenance: Option<(Arc<str>, Arc<str>)>,
}

impl super::Study {
    /// Publish a class response through the identified-execute finish.
    pub(super) fn finish_class_response_tail(&self, tail: ClassResponseTail<'_>) -> StudyResult {
        let ClassResponseTail {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            endpoints: (treatment, outcome),
            envelope,
            response,
            structural,
            diagnostics,
            identify_cached,
            bootstrap_replicates,
            started,
            ctx,
            refutations,
            identify_provenance,
            estimate_provenance,
        } = tail;
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, EstimatorId::temporal_response_for(&self.inference)),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                certificate: envelope.map(|envelope| crate::Identification::TemporalEnvelope {
                    envelope: Arc::unwrap_or_clone(envelope),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                response: Some(response),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                bootstrap_replicates_requested: Some(bootstrap_replicates),
                identify_provenance,
                estimate_provenance,
                ..Default::default()
            }),
        )
    }
}

impl ClassResponseAssembly {
    /// Start an assembly for `query`'s functional over `temporal.horizons`.
    ///
    /// # Errors
    ///
    /// A functional that is neither a mean curve nor an intervention response, or an
    /// unreadable dose grid.
    pub(super) fn new(
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
    ) -> Result<Self, CausalError> {
        let cells_per_horizon = match &query.functional {
            ResponseFunctional::MeanCurve { treatment, .. } => treatment.grid.values()?.len(),
            ResponseFunctional::InterventionResponse { .. } => 1,
            _ => {
                return Err(CausalError::Unsupported {
                    message: "class-aware temporal response supports curves and intervention responses",
                });
            }
        };
        let n_horizons = temporal.horizons.len();
        Ok(Self {
            cells_per_horizon,
            n_horizons,
            lower: vec![f64::NAN; cells_per_horizon * n_horizons],
            upper: vec![f64::NAN; cells_per_horizon * n_horizons],
            structural_atoms: Vec::new(),
            supports: Vec::new(),
            horizon_supports: Vec::new(),
            horizon_fingerprints: Vec::new(),
            full_mass_scope: true,
            truncated_atoms: 0,
            horizon_values: Vec::new(),
            support_start: 0,
        })
    }

    pub(super) const fn cells_per_horizon(&self) -> usize {
        self.cells_per_horizon
    }

    /// Open a horizon's completion envelope.
    pub(super) fn begin_horizon(
        &mut self,
        envelope: &IdentificationEnvelope<antecedent_identify::TemporalCompletionGraph>,
    ) {
        self.horizon_fingerprints
            .push(envelope.cases.iter().map(|case| case.graph.fingerprint()).collect());
        self.full_mass_scope &= envelope.truncated_completions == 0;
        self.truncated_atoms += envelope.truncated_completions;
        self.horizon_values = Vec::new();
        self.support_start = self.supports.len();
    }

    /// Record a completion the owner did not evaluate at this horizon.
    pub(super) fn push_unevaluated(
        &mut self,
        horizon_index: usize,
        case_index: usize,
        weight: f64,
        status: IdentificationStatus,
    ) {
        self.structural_atoms.push(bare_class_atom(
            class_completion_key(horizon_index, case_index),
            weight,
            status,
        ));
    }

    /// Record one fitted completion surface at this horizon; returns its atom index.
    ///
    /// # Errors
    ///
    /// An atom that is not a point surface, or whose cell count is not the horizon's.
    pub(super) fn push_atom(
        &mut self,
        horizon_index: usize,
        case_index: usize,
        weight: f64,
        status: IdentificationStatus,
        horizon: u32,
        mut response: CausalResponse,
    ) -> Result<usize, CausalError> {
        if let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        {
            response.estimate = ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from([f64::from(horizon)]),
                dimension: 1,
                mean: Arc::from([value]),
            });
        }
        let values = match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
                mean.to_vec()
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "temporal class response atom did not produce a surface".into(),
                });
            }
        };
        if values.len() != self.cells_per_horizon {
            return Err(CausalError::Compile {
                message: "temporal class response atom shape mismatch".into(),
            });
        }
        self.horizon_values.push(values);
        self.structural_atoms.push(crate::result::StructuralResponseAtom {
            value: match &response.estimate {
                ResponseIdentification::PointIdentified(value)
                | ResponseIdentification::PartiallyIdentified(value) => Some(value.clone()),
                _ => None,
            },
            ..bare_class_atom(class_completion_key(horizon_index, case_index), weight, status)
        });
        let index = self.structural_atoms.len() - 1;
        self.supports.push(response.support.clone());
        self.structural_atoms[index].response = Some(response);
        Ok(index)
    }

    /// Close a horizon: mix its atoms' supports and take the pointwise envelope.
    ///
    /// # Errors
    ///
    /// No evaluable atom at this horizon.
    pub(super) fn end_horizon(
        &mut self,
        horizon_index: usize,
        horizon: u32,
    ) -> Result<(), CausalError> {
        if self.horizon_values.is_empty() {
            return Err(CausalError::Compile {
                message: format!(
                    "temporal class response has no evaluable atom at horizon {horizon}"
                ),
            });
        }
        self.horizon_supports.push(super::response_path::mix_support_reports(
            &self.supports[self.support_start..].iter().collect::<Vec<_>>(),
        ));
        for cell in 0..self.cells_per_horizon {
            let destination = cell * self.n_horizons + horizon_index;
            self.lower[destination] =
                self.horizon_values.iter().map(|values| values[cell]).fold(f64::INFINITY, f64::min);
            self.upper[destination] = self
                .horizon_values
                .iter()
                .map(|values| values[cell])
                .fold(f64::NEG_INFINITY, f64::max);
        }
        Ok(())
    }

    /// Assemble the published response over every horizon's atoms.
    ///
    /// # Errors
    ///
    /// An unreadable dose grid.
    pub(super) fn finish(
        &self,
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
        assumptions: antecedent_core::AssumptionSet,
        provenance_id: &'static str,
    ) -> Result<ClassResponseParts, CausalError> {
        let (grid, dimension) = match &query.functional {
            ResponseFunctional::MeanCurve { treatment, .. } => {
                let doses = treatment.grid.values()?;
                let grid: Vec<f64> = doses
                    .iter()
                    .flat_map(|dose| {
                        temporal.horizons.iter().flat_map(|horizon| [*dose, f64::from(*horizon)])
                    })
                    .collect();
                (grid, 2)
            }
            ResponseFunctional::InterventionResponse { .. } => {
                (temporal.horizons.iter().map(|horizon| f64::from(*horizon)).collect(), 1)
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "class-aware temporal response supports curves and intervention responses",
                });
            }
        };
        let response_envelope = antecedent_core::ResponseEnvelope {
            grid: Arc::from(grid),
            dimension,
            lower: Arc::from(self.lower.clone()),
            upper: Arc::from(self.upper.clone()),
        };
        let graph_values = temporal_class_complete_values(
            &self.structural_atoms,
            &self.horizon_fingerprints,
            &response_envelope,
            self.n_horizons,
        );
        let (identified_mass, unidentified_mass, unevaluable_mass) =
            temporal_class_response_masses(&self.structural_atoms);
        let incomplete = unidentified_mass > 0.0 || unevaluable_mass > 0.0 || !self.full_mass_scope;
        let mut support =
            super::response_path::mix_support_reports(&self.supports.iter().collect::<Vec<_>>());
        support.point_status = (0..self.cells_per_horizon)
            .flat_map(|cell| (0..self.n_horizons).map(move |horizon| (cell, horizon)))
            .map(|(cell, horizon)| {
                self.horizon_supports[horizon].point_status.as_ref()?.get(cell).copied()
            })
            .collect::<Option<Vec<_>>>()
            .map(Arc::from);
        let status = if incomplete {
            IdentificationStatus::GraphDependent
        } else {
            IdentificationStatus::PartiallyIdentified
        };
        let response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: status,
            estimate: if incomplete {
                ResponseIdentification::GraphDependent(graph_values)
            } else {
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(
                    response_envelope.clone(),
                ))
            },
            uncertainty: if incomplete {
                ResponseUncertainty::None
            } else {
                shared_temporal_class_uncertainty(
                    &self.structural_atoms,
                    self.n_horizons,
                    self.cells_per_horizon,
                )
                .unwrap_or(ResponseUncertainty::None)
            },
            support,
            assumptions,
            provenance_id: Arc::from(provenance_id),
            horizon_identification: Some(
                temporal
                    .horizons
                    .iter()
                    .map(|&horizon| antecedent_core::HorizonIdentification {
                        horizon,
                        status,
                        method: Arc::from("temporal_class.completion_envelope"),
                        adjustment: Arc::from([]),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            ),
            interaction_structurally_zero: false,
        };
        Ok(ClassResponseParts {
            response,
            response_envelope,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
        })
    }
}

/// The band every completion shares, laid out over the class surface, when each
/// horizon's evaluated completions publish the same estimate and the same
/// pointwise band. A horizon whose completions disagree, or that publishes no
/// band, leaves the class unbanded.
#[expect(
    clippy::float_cmp,
    reason = "class members must agree on the exact declared confidence level"
)]
fn shared_temporal_class_uncertainty(
    atoms: &[crate::result::StructuralResponseAtom],
    n_horizons: usize,
    cells_per_horizon: usize,
) -> Option<ResponseUncertainty> {
    if n_horizons == 0 || cells_per_horizon == 0 {
        return None;
    }
    let mut exemplars: Vec<Option<&CausalResponse>> = vec![None; n_horizons];
    for atom in atoms {
        if atom.weight <= 0.0 {
            continue;
        }
        let response = atom.response.as_ref()?;
        let horizon = usize::try_from(atom.graph_key >> 32).ok()?;
        let slot = exemplars.get_mut(horizon)?;
        if matches!(response.uncertainty, ResponseUncertainty::None) {
            return None;
        }
        if let Some(first) = slot {
            if response.estimate != first.estimate || response.uncertainty != first.uncertainty {
                return None;
            }
        } else {
            *slot = Some(response);
        }
    }
    let exemplars = exemplars.into_iter().collect::<Option<Vec<_>>>()?;
    let mut lower = vec![0.0; cells_per_horizon * n_horizons];
    let mut upper = vec![0.0; cells_per_horizon * n_horizons];
    let (mut level, mut interpretation) = (None, None);
    for (horizon, response) in exemplars.iter().enumerate() {
        let ResponseUncertainty::PointwiseBand {
            level: horizon_level,
            lower: horizon_lower,
            upper: horizon_upper,
            interpretation: horizon_interpretation,
            draws: None,
        } = &response.uncertainty
        else {
            return None;
        };
        if horizon_lower.len() != cells_per_horizon || horizon_upper.len() != cells_per_horizon {
            return None;
        }
        match (level, interpretation) {
            (None, None) => {
                level = Some(*horizon_level);
                interpretation = Some(*horizon_interpretation);
            }
            (Some(shared_level), Some(shared_interpretation))
                if shared_level == *horizon_level
                    && shared_interpretation == *horizon_interpretation => {}
            _ => return None,
        }
        for cell in 0..cells_per_horizon {
            let index = cell * n_horizons + horizon;
            lower[index] = horizon_lower[cell];
            upper[index] = horizon_upper[cell];
        }
    }
    Some(ResponseUncertainty::PointwiseBand {
        level: level?,
        lower: Arc::from(lower),
        upper: Arc::from(upper),
        interpretation: interpretation?,
        draws: None,
    })
}

/// The completion-envelope diagnostic every class response publishes, plus the
/// Bayesian note when no class prior licenses a posterior mixture.
pub(super) fn class_response_identified_set_diagnostics(
    bayesian: bool,
    class_prior: bool,
) -> Vec<Diagnostic> {
    let mut out = vec![Diagnostic::new(
        "estimate.temporal_response.class_identified_set",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "bounds are pointwise ranges over completion-specific point surfaces; completion \
         weights are enumeration weights, not posterior probabilities or sampling uncertainty",
    )];
    if bayesian && !class_prior {
        out.push(Diagnostic::new(
            "estimate.envelope.response_posterior_not_mixed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Bayesian completion posteriors are retained as atoms; enumeration weights \
             are not mixed into a posterior",
        ));
    }
    out
}

/// Completion-enumeration weights are not a class prior.
pub(super) fn enumeration_not_probability_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal_class.enumeration_not_probability",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "completion-enumeration weights are not a class prior; no blended posterior \
         is published",
    )
}

/// Inputs of a Bayesian class-effect mixture over completion posteriors.
pub(super) struct ClassPosteriorMix<'a> {
    /// A caller class prior supplied the completion masses.
    pub class_prior_supplied: bool,
    pub truncated_completions: usize,
    /// Some evaluated completion carries positive mass.
    pub evaluable_mass_positive: bool,
    pub weights: Vec<f64>,
    pub flags: Vec<GraphIdentFlag>,
    pub keys: Vec<u64>,
    pub per_graph: &'a [GraphEffectDraws],
    pub conflict: Option<antecedent_prob::ConflictSummary>,
    /// Inference label of the mixed posterior.
    pub label: &'static str,
}

/// The class-prior contract of a Bayesian class effect, shared by the series and
/// panel owners: with a caller class prior over an uncapped class and positive
/// evaluable mass, the completion posteriors' draws are mixed by class mass
/// (within- plus between-completion variance); otherwise the effect is NaN and the
/// completion posteriors stay atoms of the structural response.
pub(super) fn mix_class_effect_posteriors<'a>(
    mix: ClassPosteriorMix<'_>,
    atom_posteriors: impl IntoIterator<Item = &'a CausalPosterior>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<
    (EffectEstimate, Option<CausalPosterior>, crate::result::StructuralWeightBasis),
    CausalError,
> {
    let basis = if mix.class_prior_supplied {
        crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
    } else {
        crate::result::StructuralWeightBasis::CompletionEnumeration
    };
    if !(mix.class_prior_supplied && mix.truncated_completions == 0 && mix.evaluable_mass_positive)
    {
        diagnostics.push(Diagnostic::new(
            "estimate.envelope.response_posterior_not_mixed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "completion posteriors are retained as atoms; mixing requires caller-supplied \
             class mass, positive evaluable mass, and uncapped class scope",
        ));
        return Ok((nan_effect(), None, basis));
    }
    let graphs = WeightedGraphSamples::new(mix.weights, mix.flags, mix.keys)?;
    let mut mixed = aggregate_effect_envelope(
        &graphs,
        mix.per_graph,
        InferenceDiagnostics::analytic(mix.label),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    merge_posterior_notes(&mut mixed, atom_posteriors);
    if let Some(summary) = mix.conflict {
        mixed = with_conflict_summary(mixed, summary);
    }
    let estimate = effect_from_posterior(&mixed)?;
    diagnostics.push(
        Diagnostic::new(
            "estimate.temporal_class.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", mixed.unidentified_mass),
        )
        .with_fields(super::mass_fields(None, mixed.unidentified_mass)),
    );
    Ok((estimate, Some(mixed), basis))
}

/// Structural uncertainty of a temporal class envelope, keyed by completion
/// fingerprint.
///
/// The mixture itself is [`super::class_structural_mixture`],
/// shared with the static class routes, so the mass split here is the
/// envelope's own ([`antecedent_identify::carries_identified_mass`]) and
/// agrees with the `identify.temporal_*.envelope` diagnostic published on the
/// same result. Unlike the static arm, a temporal class result always carries
/// its mixture: the completion atoms are part of the answer even when no
/// completion produced a finite value.
pub(crate) fn temporal_class_structural_mixture(
    envelope: &IdentificationEnvelope<antecedent_identify::TemporalCompletionGraph>,
    weight_basis: crate::result::StructuralWeightBasis,
    masses: Option<&[f64]>,
    values: &[Option<f64>],
) -> crate::result::StructuralResponseMixture {
    let outcomes: Vec<_> = values
        .iter()
        .map(|value| value.map_or(ClassAtomOutcome::NotEvaluated, ClassAtomOutcome::Evaluated))
        .collect();
    super::class_structural_mixture(
        envelope,
        weight_basis,
        |_, case| case.graph.fingerprint(),
        masses,
        &outcomes,
    )
    .0
}

pub(crate) fn temporal_class_envelope_diagnostic<G>(
    envelope: &IdentificationEnvelope<G>,
    class: GraphClass,
) -> Diagnostic {
    let code = if class == GraphClass::TemporalCpdag {
        "identify.temporal_cpdag.envelope"
    } else {
        "identify.temporal_pag.envelope"
    };
    super::class_envelope_diagnostic(
        code,
        "generalized.adjustment envelope",
        envelope,
        &[("limitations", format!("{:?}", envelope.critical_graph_features))],
    )
}

pub(crate) fn aggregate_temporal_horizon_evidence<'a>(
    identifications: impl IntoIterator<Item = &'a IdentificationResult>,
) -> Result<(IdentificationStatus, antecedent_core::AssumptionSet), CausalError> {
    let mut aggregate_status = IdentificationStatus::NonparametricallyIdentified;
    let mut assumptions = antecedent_core::AssumptionSet::new();
    let mut saw_any = false;
    for identification in identifications {
        saw_any = true;
        match identification.status {
            IdentificationStatus::NonparametricallyIdentified => {}
            IdentificationStatus::IdentifiedUnderParametricRestrictions => {
                aggregate_status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
            }
            status => {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal response requires point identification at every horizon; got {status:?}"
                    ),
                });
            }
        }
        assumptions.extend_unique(&identification.required_assumptions.entries);
    }
    if !saw_any {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon identification".into(),
        });
    }
    Ok((aggregate_status, assumptions))
}

#[derive(Clone)]
pub(super) struct MediationHorizonClick {
    pub(super) horizon: u32,
    pub(super) identification: IdentificationResult,
    pub(super) estimand: IdentifiedEstimand,
    pub(super) indexer: TemporalIndexer,
    pub(super) adjustment: Arc<[antecedent_data::LaggedColumn]>,
}

pub(super) fn mediation_horizon_clicks(
    cache: &crate::analysis::prepared::CachedTemporalIdentification,
    query: &antecedent_core::MediationQuery,
    mut adjustment_for: impl FnMut(u32) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError>,
) -> Result<Vec<MediationHorizonClick>, CausalError> {
    let mut clicks = Vec::with_capacity(query.horizons.len());
    for &horizon in query.horizons.iter() {
        let entry = cache.get(horizon).ok_or_else(|| CausalError::Compile {
            message: format!(
                "prepared temporal mediation identification missing horizon {horizon}"
            ),
        })?;
        clicks.push(MediationHorizonClick {
            horizon,
            identification: entry.identification.clone(),
            estimand: entry.estimand.clone(),
            indexer: entry.indexer.clone(),
            adjustment: adjustment_for(horizon)?,
        });
    }
    if clicks.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        });
    }
    Ok(clicks)
}

pub(super) fn lagged_adjustment_from_entry(
    entry: &crate::analysis::prepared::CachedTemporalHorizonIdentification,
) -> Arc<[antecedent_data::LaggedColumn]> {
    let outcome_offset = i32::try_from(entry.horizon.saturating_sub(1)).unwrap_or(0);
    entry
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| {
            let key = entry.indexer.key_of(dense.raw()).ok()?;
            lagged_column_relative_to_outcome(key, outcome_offset)
        })
        .collect::<Vec<_>>()
        .into()
}

pub(super) fn lagged_column_relative_to_outcome(
    key: antecedent_core::TemporalNodeKey,
    outcome_offset: i32,
) -> Option<antecedent_data::LaggedColumn> {
    let lag = outcome_offset.checked_sub(key.offset)?;
    if lag < 0 {
        return None;
    }
    Some(antecedent_data::LaggedColumn {
        variable: key.variable,
        lag: antecedent_core::Lag::from_raw(u32::try_from(lag).ok()?),
    })
}

pub(super) fn mediation_horizon_z_differs(clicks: &[MediationHorizonClick]) -> bool {
    let Some(first) = clicks.first() else {
        return false;
    };
    clicks.iter().skip(1).any(|click| click.adjustment.as_ref() != first.adjustment.as_ref())
}

pub(super) fn mediation_click_adjustment_keys(
    click: &MediationHorizonClick,
) -> Arc<[antecedent_core::TemporalNodeKey]> {
    let mut keys = click
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|variable| click.indexer.key_of(variable.raw()).ok())
        .collect::<Vec<_>>();
    keys.sort();
    Arc::from(keys)
}

pub(super) fn horizon_adjustment_sets_differ(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> bool {
    let Some(first) = entries.first() else {
        return false;
    };
    let first_z = named_adjustment_keys(first);
    entries.iter().skip(1).any(|entry| named_adjustment_keys(entry) != first_z)
}

/// Downstream design columns of an observation-adjusted temporal response (offsets
/// relative to the outcome time): the outcome mechanism's parents for a Sequence overlay,
/// otherwise every horizon's lag-aligned treatment and adjustment columns.
fn temporal_observation_outcome_regressors(
    query: &ResponseQuery,
    graph: &TemporalDag,
    outcome: VariableId,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> Result<Vec<antecedent_core::TemporalNodeKey>, CausalError> {
    let sequence = antecedent_estimate::plan_from_response_query(query)
        .map_err(CausalError::from)?
        .and_then(|plan| plan.mechanism_overlays())
        .is_some();
    if sequence {
        return Ok(antecedent_estimate::temporal_sequence_outcome_regressors(graph, outcome));
    }
    let identifications: Vec<_> =
        aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect();
    antecedent_estimate::temporal_curve_outcome_regressors(query, &identifications)
        .map_err(CausalError::from)
}

fn contemporaneous_adjustment_variables(
    entries: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
) -> Vec<VariableId> {
    let mut adjustment = Vec::new();
    for entry in entries {
        for &dense in entry.estimand.adjustment_set.iter() {
            let Ok(key) = entry.indexer.key_of(dense.raw()) else {
                continue;
            };
            if !adjustment.contains(&key.variable) {
                adjustment.push(key.variable);
            }
        }
    }
    adjustment
}

fn named_adjustment_keys(
    entry: &crate::analysis::prepared::CachedTemporalHorizonIdentification,
) -> Vec<antecedent_core::TemporalNodeKey> {
    let mut keys: Vec<_> = entry
        .estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| entry.indexer.key_of(dense.raw()).ok())
        .collect();
    keys.sort();
    keys
}

fn apply_temporal_observation_result(
    response: &mut CausalResponse,
    adjusted: &antecedent_estimate::ObservationAdjustedOutcome,
) {
    let (minimum_weight, maximum_weight, effective_sample_size) = {
        let minimum = adjusted
            .weights
            .iter()
            .copied()
            .filter(|weight| *weight > 0.0)
            .fold(f64::INFINITY, f64::min);
        let maximum = adjusted.weights.iter().copied().fold(0.0_f64, f64::max);
        let sum = adjusted.weights.iter().sum::<f64>();
        let sum_squares = adjusted.weights.iter().map(|weight| weight * weight).sum::<f64>();
        let ess = if sum_squares > 0.0 { sum * sum / sum_squares } else { 0.0 };
        (if minimum.is_finite() { minimum } else { 0.0 }, maximum, ess)
    };
    response.uncertainty = ResponseUncertainty::None;
    // Complete-data bands (pointwise or simultaneous) never describe the corrected
    // surface; only the outer observation bootstrap below may publish one.
    antecedent_estimate::clear_simultaneous_band(&mut response.support);
    response.provenance_id = Arc::from("estimate.temporal_response.observation_adjusted");
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from("response.observation_adjustment_weights"),
        values: Arc::from([minimum_weight, maximum_weight, effective_sample_size]),
        detail: Arc::from(
            "minimum positive weight, maximum weight, and Kish effective sample size; weights are diagnostic only and were already incorporated into the pseudo-outcome",
        ),
        scope: antecedent_core::DiagnosticScope::Global,
    });
    response.support.warnings.push(Diagnostic::new(
        "response.observation_joint_uncertainty_unavailable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "point estimate includes observation correction; uncertainty is omitted because complete-data curve bands do not account for the estimated observation mechanism",
    ));
    response.support.warnings.push(Diagnostic::new(
        "response.observation_adjustment_method",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        adjusted.method.clone(),
    ));
}

/// How a class-observation response reports its band: from the outer circular-block
/// bootstrap when a single completion identifies, otherwise withheld.
fn class_observation_band_note(class_band_applied: bool) -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal_class.observation_no_complete_band",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        if class_band_applied {
            "observation is adjusted per completion; complete-data bands are not reused; the \
             class band is the outer circular-block observation bootstrap of the single \
             identified completion"
        } else {
            "observation is adjusted per completion; complete-data bands are not reused and \
             joint observation/curve bands stay unavailable at the class level"
        },
    )
}

struct ObservationBootstrapBand {
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Joint replicate surfaces (`draws[r]` in the mean layout) behind the band, kept for
    /// the simultaneous band; empty when the pointwise band is withheld.
    draws: Vec<Vec<f64>>,
    completed: u32,
    attempted: u32,
    cancelled: bool,
    /// Block length, replicate dispersion factor and per-cell dispersion readings of the
    /// resample, disclosed with a published band.
    block: Option<(
        antecedent_estimate::ResponseBlockLength,
        f64,
        antecedent_estimate::CellDispersion,
    )>,
}

impl ObservationBootstrapBand {
    /// Attach the block length, dispersion factor and per-cell readings of target
    /// `target` of the tuple resample behind the band.
    fn with_block(mut self, replicates: &TupleReplicates, target: usize) -> Self {
        let dispersion = replicates.dispersion.get(target).cloned().unwrap_or_default();
        self.block = Some((replicates.block, replicates.inflation, dispersion));
        self
    }

    /// Attach the block length, dispersion factor and the one-cell summary of every
    /// target's readings (a band mixed over several targets).
    fn with_block_summary(mut self, replicates: &TupleReplicates) -> Self {
        self.block =
            Some((replicates.block, replicates.inflation, replicates.summary_dispersion()));
        self
    }
}

/// Construction text for the observation-adjusted outer bootstrap simultaneous band.
const OBSERVATION_SIMULTANEOUS_CONSTRUCTION: &str = "max-studentized deviation of the outer \
     circular-block bootstrap replicates of the whole observation-adjusted surface around the \
     full-sample estimate; every replicate resamples blocks of lag-aligned outcome-time tuples \
     (block length: support diagnostic response.temporal.block_length), refits the observation \
     nuisance on them and refits every horizon (curve / Set-Shift) or every unfolded sequential \
     mechanism (Sequence); replicate deviations carry the circular-Bartlett fixed-b factor, the \
     HC1 factor and each cell's kernel-bias factor (response.temporal.kernel_bias_factor)";

/// Pointwise band `center ± z·SD` of the (already fixed-b scaled) joint replicates.
///
/// Same success floor as the inner estimator's own joint bootstrap
/// (`temporal_response::bootstrap_surface`): at least two surviving replicates and at
/// most half of the attempted ones failed, or a cancelled run (even at the success
/// floor) cannot justify a reported band. Cancellation is not adaptive early-stop. This
/// is deliberately looser than [`super::PERCENTILE_95_BAND_MIN_SUCCESSES`], which gates
/// only the *simultaneous* band (see [`antecedent_estimate::TEMPORAL_RESPONSE_FEW_REPLICATES`]):
/// a pointwise band below that floor still publishes, flagged as resting on a noisy SD.
fn summarize_observation_bootstrap(
    draws: &[Vec<f64>],
    center: &[f64],
    attempted: u32,
    cancelled: bool,
) -> ObservationBootstrapBand {
    let completed = u32::try_from(draws.len()).unwrap_or(u32::MAX);
    let mut band = ObservationBootstrapBand {
        lower: Vec::new(),
        upper: Vec::new(),
        draws: Vec::new(),
        completed,
        attempted,
        cancelled,
        block: None,
    };
    let enough_successes = completed >= 2 && completed >= attempted.saturating_sub(completed);
    if cancelled
        || !enough_successes
        || draws.iter().any(|draw| draw.len() != center.len())
        || center.iter().any(|value| !value.is_finite())
    {
        return band;
    }
    let z = crate::result::reported_se_interval_z();
    let mut values = Vec::with_capacity(draws.len());
    for (cell, &mid) in center.iter().enumerate() {
        values.clear();
        values.extend(draws.iter().map(|draw| draw[cell]));
        let sd = antecedent_stats::sample_std(&values);
        band.lower.push(mid - z * sd);
        band.upper.push(mid + z * sd);
    }
    band.draws = draws.to_vec();
    band
}

/// Full-sample point surface of a response, in the mean layout.
fn response_point_values(response: &CausalResponse) -> Option<Vec<f64>> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => Some(vec![*value]),
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            Some(mean.to_vec())
        }
        // A single identified completion: the class set collapses to one surface.
        ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(envelope))
            if envelope.lower == envelope.upper =>
        {
            Some(envelope.lower.to_vec())
        }
        _ => None,
    }
}

#[derive(Clone)]
enum ClassObservationKind {
    Curve,
    /// Sequence overlay atom: refit through the unfolded sequential engine.
    Sequence {
        dag: Box<TemporalDag>,
        overlays: Vec<antecedent_estimate::SequentialMechanismOverlay>,
        outcome: VariableId,
        outcome_offset: i32,
        status: IdentificationStatus,
    },
}

struct ClassObservationAtom {
    atom_index: usize,
    horizon_index: usize,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    assumptions: antecedent_core::AssumptionSet,
    weight: f64,
    adjustment: Vec<VariableId>,
    observation_query: ResponseQuery,
    kind: ClassObservationKind,
}

fn class_atom_adjustment(
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
) -> Vec<VariableId> {
    estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| indexer.key_of(dense.raw()).ok().map(|key| key.variable))
        .collect()
}

/// The series, graph and per-horizon retained results a Sequence temporal response runs over.
#[derive(Clone, Copy)]
struct SequenceResponseInputs<'a> {
    data: &'a TimeSeriesData,
    source_data: &'a TimeSeriesData,
    graph: &'a TemporalDag,
    query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    overlays: &'a [antecedent_estimate::SequentialMechanismOverlay],
    aligned: &'a [&'a crate::analysis::prepared::CachedTemporalHorizonIdentification],
    observation_adjusted: Option<&'a antecedent_estimate::ObservationAdjustedOutcome>,
}

/// The observation-adjustment bootstrap budget of one temporal response: the
/// observation estimator options and the replicate count.
#[derive(Clone, Copy)]
struct ObservationBootstrap {
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
}

/// The shared mechanism graph and per-horizon overlays of a sequence response.
#[derive(Clone, Copy)]
struct SequenceMechanisms<'a> {
    graph: &'a TemporalDag,
    overlays: &'a [antecedent_estimate::SequentialMechanismOverlay],
}

/// The class-level slots a class observation bootstrap writes into.
struct ClassBootstrapTargets<'a> {
    structural_atoms: &'a mut [crate::result::StructuralResponseAtom],
    response: &'a mut CausalResponse,
    diagnostics: &'a mut Vec<Diagnostic>,
}

/// How a class response's cells are laid out over horizons, and whether the class band applies.
#[derive(Clone, Copy)]
struct ClassBandLayout {
    cells_per_horizon: usize,
    n_horizons: usize,
    apply_class_band: bool,
}

/// The joint circular-block tuple replicates the class atoms' observation
/// bands are summarized from.
fn class_tuple_replicates(
    source: &TimeSeriesData,
    atoms: &[ClassObservationAtom],
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<TupleReplicates, CausalError> {
    let structural_span = atoms
        .iter()
        .map(|atom| atom.indexer.history() as usize + atom.indexer.horizon() as usize)
        .max()
        .unwrap_or(1);
    // Every atom (curve or Sequence) resamples the same lag-aligned outcome-time tuples.
    let identifications: Vec<[(&IdentifiedEstimand, &TemporalIndexer); 1]> =
        atoms.iter().map(|atom| [(&atom.estimand, &atom.indexer)]).collect();
    let targets: Vec<TupleObservationTarget<'_>> = atoms
        .iter()
        .zip(&identifications)
        .map(|(atom, identification)| -> Result<TupleObservationTarget<'_>, CausalError> {
            Ok(TupleObservationTarget {
                query: &atom.observation_query,
                adjustment: &atom.adjustment,
                outcome_regressors: match &atom.kind {
                    ClassObservationKind::Curve => {
                        antecedent_estimate::temporal_curve_outcome_regressors(
                            &atom.observation_query,
                            identification,
                        )
                        .map_err(CausalError::from)?
                    }
                    ClassObservationKind::Sequence { dag, outcome, .. } => {
                        antecedent_estimate::temporal_sequence_outcome_regressors(dag, *outcome)
                    }
                },
                surface: match &atom.kind {
                    ClassObservationKind::Curve => {
                        TupleSurface::Curve { identifications: identification }
                    }
                    ClassObservationKind::Sequence {
                        dag,
                        overlays,
                        outcome,
                        outcome_offset,
                        status,
                    } => TupleSurface::Sequence {
                        graph: dag,
                        overlays,
                        outcome: *outcome,
                        horizons: vec![(&atom.estimand, &atom.indexer, *outcome_offset, *status)],
                    },
                },
            })
        })
        .collect::<Result<_, _>>()?;
    tuple_block_observation_replicates(
        source,
        &targets,
        structural_span,
        options,
        replicates,
        0x0C1A_5500,
        ctx,
    )
}

/// Withholds the class-level observation band on a multi-completion
/// identified set, recording the counts and why the atoms keep their own bands.
fn withhold_class_observation_band(
    response: &mut CausalResponse,
    diagnostics: &mut Vec<Diagnostic>,
    (replicates, attempted): (u32, u32),
    cancelled: bool,
) {
    response.support.warnings.retain(|warning| {
        warning.code.as_ref() != antecedent_estimate::TEMPORAL_RESPONSE_BAND_WITHHELD
    });
    diagnostics.push(Diagnostic::new(
        "estimate.temporal_class.observation_class_band_withheld",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "class-level observation band is withheld on a multi-completion identified set; \
         each completion atom retains its outer circular-block band; complete-data bands \
         are not reused",
    ));
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from("response.observation_block_bootstrap"),
        values: Arc::from([
            f64::from(replicates),
            0.0,
            f64::from(u32::from(cancelled)),
            f64::from(attempted),
        ]),
        detail: Arc::from(
            "requested, completed class-band, cancellation, and attempted counts; the \
             class band is withheld; atoms retain per-completion circular-block bands",
        ),
        scope: antecedent_core::DiagnosticScope::Global,
    });
}

/// The tuple replicates, per-atom draws and the (attempted, cancelled) counts
/// the atom bands are summarized from.
type AtomBandSource<'a> = (&'a TupleReplicates, &'a [Vec<Vec<f64>>], (u32, bool));

/// Writes each completion atom's observation circular-block band onto its
/// structural atom, building a point-only response for atoms without one.
fn apply_atom_observation_bands(
    atoms: &[ClassObservationAtom],
    structural_atoms: &mut [crate::result::StructuralResponseAtom],
    response: &CausalResponse,
    (tuple, atom_draws, (attempted, cancelled)): AtomBandSource<'_>,
    replicates: u32,
) {
    let points = &tuple.points;
    for (atom_i, atom) in atoms.iter().enumerate() {
        let band = summarize_observation_bootstrap(
            &atom_draws[atom_i],
            &points[atom_i],
            attempted,
            cancelled,
        )
        .with_block(tuple, atom_i);
        if let Some(atom_response) =
            structural_atoms.get_mut(atom.atom_index).and_then(|item| item.response.as_mut())
        {
            apply_observation_bootstrap(atom_response, &band, replicates);
        } else if let Some(structural) = structural_atoms.get_mut(atom.atom_index) {
            if let Some(value) = structural.value.clone() {
                let mut stub = CausalResponse {
                    estimand: response.estimand.clone(),
                    identification_status: structural.status,
                    estimate: ResponseIdentification::PointIdentified(value),
                    uncertainty: ResponseUncertainty::None,
                    support: response.support.clone(),
                    assumptions: atom.assumptions.clone(),
                    provenance_id: Arc::from("estimate.temporal_response.class_atom"),
                    horizon_identification: None,
                    interaction_structurally_zero: false,
                };
                apply_observation_bootstrap(&mut stub, &band, replicates);
                structural.response = Some(stub);
            }
        }
    }
}

fn apply_class_observation_bootstrap(
    source: &TimeSeriesData,
    atoms: &[ClassObservationAtom],
    targets: ClassBootstrapTargets<'_>,
    layout: ClassBandLayout,
    bootstrap: ObservationBootstrap,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    let ClassBootstrapTargets { structural_atoms, response, diagnostics } = targets;
    let ClassBandLayout { cells_per_horizon, n_horizons, apply_class_band } = layout;
    let ObservationBootstrap { options, replicates } = bootstrap;
    if replicates == 0 || atoms.is_empty() {
        return Ok(false);
    }
    let tuple = class_tuple_replicates(source, atoms, options, replicates, ctx)?;
    let mut atom_draws: Vec<Vec<Vec<f64>>> = vec![Vec::new(); atoms.len()];
    let mut class_draws: Vec<Vec<f64>> = Vec::new();
    let class_len = cells_per_horizon.saturating_mul(n_horizons);
    // Frozen-weight mix of per-atom surfaces into the class layout.
    let mix = |per_atom: &[Option<&Vec<f64>>]| -> Option<Vec<f64>> {
        let mut mixed = vec![0.0; class_len];
        let mut totals = vec![0.0; n_horizons];
        for (atom, values) in atoms.iter().zip(per_atom) {
            let values = (*values)?;
            if atom.horizon_index >= n_horizons || values.len() != cells_per_horizon {
                return None;
            }
            totals[atom.horizon_index] += atom.weight;
            for (cell, value) in values.iter().enumerate() {
                mixed[cell * n_horizons + atom.horizon_index] += atom.weight * value;
            }
        }
        if !totals.iter().all(|&total| total > 0.0) {
            return None;
        }
        for (index, value) in mixed.iter_mut().enumerate() {
            *value /= totals[index % n_horizons];
        }
        Some(mixed)
    };

    let (attempted, points) = (tuple.attempted, &tuple.points);
    for replicate_atom_values in &tuple.values {
        for (atom_i, values) in replicate_atom_values.iter().enumerate() {
            if let Some(values) = values {
                if atom_draws[atom_i].first().is_none_or(|first| first.len() == values.len()) {
                    atom_draws[atom_i].push(values.clone());
                }
            }
        }
        if apply_class_band && class_len > 0 {
            let per_atom: Vec<Option<&Vec<f64>>> =
                replicate_atom_values.iter().map(Option::as_ref).collect();
            class_draws.extend(mix(&per_atom));
        }
    }

    let cancelled = ctx.cancellation.is_cancelled();
    apply_atom_observation_bands(
        atoms,
        structural_atoms,
        response,
        (&tuple, &atom_draws, (attempted, cancelled)),
        replicates,
    );

    if apply_class_band {
        let center = mix(&points.iter().map(Some).collect::<Vec<_>>()).unwrap_or_default();
        let band = summarize_observation_bootstrap(&class_draws, &center, attempted, cancelled)
            .with_block_summary(&tuple);
        apply_observation_bootstrap(response, &band, replicates);
        Ok(!band.lower.is_empty())
    } else {
        withhold_class_observation_band(response, diagnostics, (replicates, attempted), cancelled);
        Ok(false)
    }
}

/// Which tuple bootstrap produced an [`ObservationBootstrapBand`], for its diagnostics.
#[derive(Clone, Copy)]
enum TupleBandSource {
    /// Outer bootstrap refitting the observation nuisance.
    Observation,
    /// Complete-data Sequence overlays (no observation nuisance).
    CompleteSequence,
}

impl TupleBandSource {
    const fn counts_id(self) -> &'static str {
        match self {
            Self::Observation => "response.observation_block_bootstrap",
            Self::CompleteSequence => "response.sequence_block_bootstrap",
        }
    }

    const fn counts_detail(self) -> &'static str {
        match self {
            Self::Observation => {
                "requested, completed, cancellation, and attempted counts for the outer \
                 circular-block bootstrap; each attempt refits observation nuisances and all \
                 horizon models; bands require two successes and at most half failed attempts"
            }
            Self::CompleteSequence => {
                "requested, completed, cancellation, and attempted counts for the joint \
                 circular-block tuple bootstrap; each attempt refits every unfolded sequential \
                 mechanism of every horizon; bands require two successes and at most half \
                 failed attempts"
            }
        }
    }

    /// What one replicate refits, for the block-bootstrap assumption record.
    const fn refit(self) -> &'static str {
        match self {
            Self::Observation => {
                "one replicate refits the observation nuisance and every horizon model (or every \
                 unfolded sequential mechanism) on the same resampled outcome-time tuples"
            }
            Self::CompleteSequence => {
                "one replicate refits every unfolded sequential mechanism of every horizon on the \
                 same resampled outcome-time tuples and recomputes the root-node means"
            }
        }
    }

    const fn construction(self) -> &'static str {
        match self {
            Self::Observation => OBSERVATION_SIMULTANEOUS_CONSTRUCTION,
            Self::CompleteSequence => SEQUENCE_SIMULTANEOUS_CONSTRUCTION,
        }
    }
}

/// Construction text for the complete-data Sequence joint tuple bootstrap band.
const SEQUENCE_SIMULTANEOUS_CONSTRUCTION: &str = "max-studentized deviation of the joint \
     circular-block bootstrap replicates of the whole Sequence surface around the full-sample \
     estimate; every replicate resamples blocks of lag-aligned outcome-time tuples (block length: \
     support diagnostic response.temporal.block_length), refits every unfolded sequential \
     mechanism of every horizon and recomputes the root-node means; replicate deviations carry \
     the circular-Bartlett fixed-b factor, the HC1 factor and each level's kernel-bias factor \
     (response.temporal.kernel_bias_factor, the largest over that level's estimating scores)";

fn apply_observation_bootstrap(
    response: &mut CausalResponse,
    bootstrap: &ObservationBootstrapBand,
    requested: u32,
) {
    apply_tuple_bootstrap_band(response, bootstrap, requested, TupleBandSource::Observation);
}

fn apply_tuple_bootstrap_band(
    response: &mut CausalResponse,
    bootstrap: &ObservationBootstrapBand,
    requested: u32,
    source: TupleBandSource,
) {
    // A tuple bootstrap ran, so the estimator's zero-replicate withholding note (the
    // inner fit runs without replicates on these paths) no longer describes the band.
    response.support.warnings.retain(|warning| {
        warning.code.as_ref() != antecedent_estimate::TEMPORAL_RESPONSE_BAND_WITHHELD
    });
    if bootstrap.lower.is_empty() {
        antecedent_estimate::publish_simultaneous_band(
            &mut response.support,
            Err(antecedent_estimate::EstimationError::unsupported(
                "the tuple bootstrap produced too few successful replicates",
            )),
            "",
        );
        response.support.warnings.push(Diagnostic::new(
            match source {
                TupleBandSource::Observation => "response.observation_bootstrap_insufficient",
                TupleBandSource::CompleteSequence => "response.sequence_bootstrap_insufficient",
            },
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "bootstrap band withheld: {} of {} attempted replicates succeeded; at least two \
                 successes and at most half failed attempts are required",
                bootstrap.completed, bootstrap.attempted
            ),
        ));
    } else {
        response.uncertainty = ResponseUncertainty::PointwiseBand {
            level: 0.95,
            lower: Arc::from(bootstrap.lower.clone()),
            upper: Arc::from(bootstrap.upper.clone()),
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        };
        response.support.warnings.retain(|warning| {
            !matches!(
                warning.code.as_ref(),
                "response.observation_joint_uncertainty_unavailable"
                    | "response.temporal.sequence_uncertainty_unavailable"
            )
        });
        if let Some((block, inflation, dispersion)) = &bootstrap.block {
            antecedent_estimate::disclose_response_block_bootstrap(
                &mut response.support,
                &mut response.assumptions,
                *block,
                *inflation,
                dispersion,
                bootstrap.draws.len(),
                source.refit(),
            );
        }
        let simultaneous = response_point_values(response)
            .ok_or_else(|| {
                antecedent_estimate::EstimationError::unsupported(
                    "the response has no single point surface",
                )
            })
            .and_then(|center| {
                antecedent_estimate::max_deviation_band(&center, &bootstrap.draws, 0.95)
            });
        antecedent_estimate::publish_simultaneous_band(
            &mut response.support,
            simultaneous,
            source.construction(),
        );
    }
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from(source.counts_id()),
        values: Arc::from([
            f64::from(requested),
            f64::from(bootstrap.completed),
            f64::from(bootstrap.cancelled),
            f64::from(bootstrap.attempted),
        ]),
        detail: Arc::from(source.counts_detail()),
        scope: antecedent_core::DiagnosticScope::Global,
    });
}

/// Outer circular-block bootstrap of an observation-adjusted curve / Set-Shift surface.
///
/// Resampling is tuple-level: each replicate draws circular blocks of outcome-time
/// anchors, refits the observation nuisance on those anchors' lag-aligned observation
/// rows, and refits every horizon on the same anchors' lag-aligned design rows with
/// the replicate pseudo-outcome. Reordering the raw series instead (and rebuilding lags
/// on the reordered rows) pairs values across block junctions and biases the replicate
/// slope toward zero.
fn bootstrap_observation_adjusted_temporal_response(
    source: &TimeSeriesData,
    query: &ResponseQuery,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    bootstrap: ObservationBootstrap,
    ctx: &ExecutionContext,
) -> Result<Option<ObservationBootstrapBand>, CausalError> {
    let ObservationBootstrap { options, replicates } = bootstrap;
    if replicates == 0 {
        return Ok(None);
    }
    let adjustment = contemporaneous_adjustment_variables(aligned);
    let identifications =
        aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect::<Vec<_>>();
    let target = TupleObservationTarget {
        query,
        adjustment: &adjustment,
        outcome_regressors: antecedent_estimate::temporal_curve_outcome_regressors(
            query,
            &identifications,
        )
        .map_err(CausalError::from)?,
        surface: TupleSurface::Curve { identifications: &identifications },
    };
    single_target_observation_band(source, target, aligned, options, replicates, 0x0B5E_0000, ctx)
        .map(Some)
}

/// Joint circular-block tuple bootstrap of a Sequence overlay surface.
///
/// Each replicate resamples blocks of lag-aligned outcome-time tuples and refits every
/// unfolded sequential mechanism of every horizon on them through
/// [`antecedent_estimate::PreparedSequenceLevel`], recomputing root-node means. For an
/// observation-adjusted query, every replicate also refits the observation nuisance on
/// the resampled tuples and replaces the outcome-time column with the replicate
/// pseudo-outcomes (a design that reads the outcome at a nonzero lag is refused); a
/// complete-data query keeps its recorded outcomes.
fn bootstrap_sequence_response(
    source: &TimeSeriesData,
    query: &ResponseQuery,
    mechanisms: SequenceMechanisms<'_>,
    outcome: VariableId,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    bootstrap: ObservationBootstrap,
    ctx: &ExecutionContext,
) -> Result<ObservationBootstrapBand, CausalError> {
    let SequenceMechanisms { graph, overlays } = mechanisms;
    let ObservationBootstrap { options, replicates } = bootstrap;
    let adjustment = contemporaneous_adjustment_variables(aligned);
    let horizons = aligned
        .iter()
        .map(|entry| {
            (
                &entry.estimand,
                &entry.indexer,
                i32::try_from(entry.horizon.saturating_sub(1)).unwrap_or(i32::MAX),
                entry.identification.status,
            )
        })
        .collect();
    let target = TupleObservationTarget {
        query,
        adjustment: &adjustment,
        outcome_regressors: antecedent_estimate::temporal_sequence_outcome_regressors(
            graph, outcome,
        ),
        surface: TupleSurface::Sequence { graph, overlays, outcome, horizons },
    };
    single_target_observation_band(source, target, aligned, options, replicates, 0x0B5E_5E00, ctx)
}

fn single_target_observation_band(
    source: &TimeSeriesData,
    target: TupleObservationTarget<'_>,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    stream: u64,
    ctx: &ExecutionContext,
) -> Result<ObservationBootstrapBand, CausalError> {
    let structural_span = aligned
        .iter()
        .map(|entry| entry.indexer.history() as usize + entry.indexer.horizon() as usize)
        .max()
        .unwrap_or(1);
    let mut tuple = tuple_block_observation_replicates(
        source,
        &[target],
        structural_span,
        options,
        replicates,
        stream,
        ctx,
    )?;
    let draws: Vec<Vec<f64>> = std::mem::take(&mut tuple.values)
        .into_iter()
        .filter_map(|mut values| values.pop().flatten())
        .collect();
    let center = tuple.points.pop().unwrap_or_default();
    Ok(summarize_observation_bootstrap(
        &draws,
        &center,
        tuple.attempted,
        ctx.cancellation.is_cancelled(),
    )
    .with_block(&tuple, 0))
}

/// Hydrate a shared coefficient prior into the unfolded design at every horizon.
///
/// Same-design and mapped transfer use the existing
/// [`resolve_bayesian_prior_with_conflict`] filter. Horizons whose coefficient
/// semantics differ fail closed even when their column counts happen to match.
/// Conflict-sensitive composition is also horizon-specific and therefore
/// refuses a multi-horizon surface until the response estimator accepts one
/// resolved prior per horizon.
pub(super) fn resolve_temporal_response_prior(
    cfg: &BayesianConfig,
    data: &TimeSeriesData,
    temporal: &antecedent_core::TemporalResponseSpec,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    treatment: VariableId,
    outcome: VariableId,
    ctx: &ExecutionContext,
) -> Result<(Option<PriorSet>, Option<antecedent_prob::ConflictSummary>), CausalError> {
    if cfg.prior.is_none() && cfg.prior_artifact.is_none() && cfg.external_compose.is_none() {
        return Ok((None, None));
    }
    if aligned.len() > 1
        && cfg.external_compose.as_ref().is_some_and(|compose| compose.conflict_policy.is_some())
    {
        return Err(CausalError::Unsupported {
            message: "conflict-sensitive Bayesian temporal response prior transfer requires \
                      horizon-specific conflict evaluation",
        });
    }
    let mut resolved = None;
    let mut conflict = None;
    let mut expected_design = None;
    for entry in aligned {
        let pulse = unit_contrast_query(
            temporal,
            (treatment, outcome),
            entry.horizon,
            antecedent_core::TargetPopulation::AllObserved,
        );
        let prep = TemporalLinearAdjustment::new()
            .prepare(data, &entry.estimand, &pulse, &entry.indexer, None, &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
        let adjustment = named_adjustment_keys(entry);
        match &expected_design {
            None => {
                expected_design = Some((bprep.design.ncols, adjustment));
                let (prior, summary) =
                    resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
                resolved = prior;
                conflict = summary;
            }
            Some((ncols, expected_adjustment))
                if temporal_prior_designs_match(
                    *ncols,
                    expected_adjustment,
                    bprep.design.ncols,
                    &adjustment,
                ) => {}
            Some(_) => {
                return Err(CausalError::Unsupported {
                    message: "Bayesian temporal response prior transfer requires a \
                              horizon-specific mapping",
                });
            }
        }
    }
    Ok((resolved, conflict))
}

fn temporal_prior_designs_match(
    expected_ncols: usize,
    expected_adjustment: &[antecedent_core::TemporalNodeKey],
    ncols: usize,
    adjustment: &[antecedent_core::TemporalNodeKey],
) -> bool {
    expected_ncols == ncols && expected_adjustment == adjustment
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus,
    };

    use super::super::most_conservative_identification_status;
    use super::*;
    use antecedent_core::AverageEffectQuery;
    use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord};

    fn horizon_result(
        status: IdentificationStatus,
        evidence: &'static str,
    ) -> IdentificationResult {
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(evidence),
                description: Arc::from("horizon-specific evidence"),
            },
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("test.temporal_horizon"),
            },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
            status,
            query: CausalQuery::AverageEffect(query),
            estimands: Vec::new(),
            arena: CausalExprArena::new(),
            derivation: DerivationTrace::default(),
            required_assumptions: assumptions,
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        })
    }

    #[test]
    fn multi_horizon_parent_status_is_the_most_conservative_slice() {
        assert_eq!(
            most_conservative_identification_status([
                IdentificationStatus::NonparametricallyIdentified,
                IdentificationStatus::GraphDependent,
            ]),
            IdentificationStatus::GraphDependent
        );
        assert_eq!(
            most_conservative_identification_status([
                IdentificationStatus::NonparametricallyIdentified,
                IdentificationStatus::NotIdentified,
            ]),
            IdentificationStatus::NotIdentified
        );
        assert_eq!(
            most_conservative_identification_status([
                IdentificationStatus::IdentifiedUnderParametricRestrictions,
                IdentificationStatus::NonparametricallyIdentified,
            ]),
            IdentificationStatus::IdentifiedUnderParametricRestrictions
        );
        assert_eq!(
            most_conservative_identification_status([] as [IdentificationStatus; 0]),
            IdentificationStatus::NotIdentified
        );
    }

    #[test]
    fn temporal_horizon_evidence_is_unioned_and_never_upgraded() {
        let first =
            horizon_result(IdentificationStatus::NonparametricallyIdentified, "horizon.one");
        let second = horizon_result(
            IdentificationStatus::IdentifiedUnderParametricRestrictions,
            "horizon.two",
        );
        let (status, assumptions) = aggregate_temporal_horizon_evidence([&first, &second]).unwrap();
        assert_eq!(status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
        for evidence in ["horizon.one", "horizon.two"] {
            assert!(assumptions.entries.iter().any(|record| {
                matches!(&record.assumption, Assumption::Custom { id, .. } if id.as_ref() == evidence)
            }));
        }

        let partial = horizon_result(IdentificationStatus::PartiallyIdentified, "horizon.partial");
        let error = aggregate_temporal_horizon_evidence([&first, &partial]).unwrap_err();
        assert!(error.to_string().contains("at every horizon"));
    }

    #[test]
    fn equal_width_temporal_prior_designs_still_compare_adjustment_semantics() {
        let z = antecedent_core::TemporalNodeKey { variable: VariableId::from_raw(2), offset: -1 };
        let w = antecedent_core::TemporalNodeKey { variable: VariableId::from_raw(3), offset: -1 };
        assert!(temporal_prior_designs_match(3, &[z], 3, &[z]));
        assert!(!temporal_prior_designs_match(3, &[z], 3, &[w]));
    }

    /// A minimal posterior carrying one horizon's inference notes, standing in
    /// for what `estimate_sequence_mechanisms` returns per horizon.
    fn posterior_with_notes(notes: Vec<Arc<str>>) -> CausalPosterior {
        let schema = antecedent_prob::PosteriorSchema {
            quantities: Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
                name: Arc::from("test"),
            }]),
        };
        let draws =
            antecedent_prob::PosteriorDraws::from_column_major(schema, 2, Arc::from([0.0, 0.0]))
                .unwrap();
        let summaries = draws.summarize();
        let mut diagnostics = InferenceDiagnostics::analytic("test.posterior_with_notes");
        diagnostics.notes = notes;
        CausalPosterior {
            draws,
            summaries,
            identification: IdentificationStatus::NonparametricallyIdentified,
            prior_sensitivity: None,
            conflict_summary: None,
            diagnostics,
            assumptions: antecedent_core::AssumptionSet::new(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            unevaluable_mass: 0.0,
            early_stopped: false,
            treatment_contrast: None,
        }
    }

    #[test]
    fn multi_horizon_posterior_notes_all_reach_result_diagnostics() {
        // A multi-horizon Sequence-overlay response used to keep only the LAST
        // horizon's posterior (`attach_scalar_posterior = horizons.len() == 1`),
        // so a disclosure raised at an earlier horizon never reached the
        // result. This pins the fix: every horizon's notes are fed through
        // `posterior_note_diagnostics`, regardless of which horizon carries
        // the disclosure and regardless of the scalar-attachment decision.
        let inestimable_note: Arc<str> = Arc::from(
            "serial_dependence.long_run_tempering kappa=1.000000 raw_ratio=1.000000 \
             ar_ratio=1.000000 df_loss=2.000000 kappa_log_sd=0.000000 hac_ratio=1.000000 \
             fixed_b=1.000000 score_ar_order=0 residual_ar_order=0 n=6 n_eff=6.000 bandwidth=1 \
             scope=treatment bounded=false capped=false inestimable=true",
        );
        let first_horizon = posterior_with_notes(vec![inestimable_note]);
        let second_horizon = posterior_with_notes(Vec::new());
        let horizon_posteriors = vec![first_horizon, second_horizon];
        let diagnostics = posterior_note_diagnostics(&horizon_posteriors);
        assert!(
            diagnostics
                .iter()
                .any(|d| d.code.as_ref() == "estimate.bayesian.temporal.tempering_inestimable"),
            "the first horizon's inestimable disclosure must reach the diagnostics even though \
             it is not the last horizon: {:?}",
            diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
        );
        // The old code fed only `extras.posterior` (the last horizon's posterior,
        // attached only when there is a single horizon) into this helper; with
        // two horizons and the disclosure on the first, that path saw nothing.
        let old_behavior = posterior_note_diagnostics(std::iter::empty::<&CausalPosterior>());
        assert!(
            old_behavior.is_empty(),
            "sanity check: the pre-fix input (no posterior attached) discloses nothing"
        );
    }

    #[test]
    fn repeated_inestimable_notes_do_not_read_as_one_fit() {
        // Five fits raise the identical inestimable note. De-duplicating the notes by text must not
        // leave a message that claims the condition hit a single fit.
        let note: Arc<str> = Arc::from(
            "serial_dependence.long_run_tempering kappa=1.000000 raw_ratio=1.000000 \
             ar_ratio=1.000000 df_loss=2.000000 kappa_log_sd=0.000000 hac_ratio=1.000000 \
             fixed_b=1.000000 score_ar_order=0 residual_ar_order=0 n=6 n_eff=6.000 bandwidth=1 \
             scope=treatment bounded=false capped=false inestimable=true",
        );
        let posteriors: Vec<CausalPosterior> =
            (0..5).map(|_| posterior_with_notes(vec![Arc::clone(&note)])).collect();
        let diagnostics = posterior_note_diagnostics(&posteriors);
        let inestimable = diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "estimate.bayesian.temporal.tempering_inestimable")
            .expect("inestimable disclosure");
        assert!(!inestimable.message.contains("1 fit"), "{}", inestimable.message);
        assert!(inestimable.message.contains("at least one fit"), "{}", inestimable.message);
    }
}

#[cfg(test)]
mod observation_bootstrap_tests {
    use super::*;

    #[test]
    fn excessive_failures_withhold_band_but_retain_counts() {
        let band = summarize_observation_bootstrap(&[vec![1.0], vec![3.0]], &[2.0], 5, false);
        assert!(band.lower.is_empty());
        assert!(band.upper.is_empty());
        assert_eq!(band.completed, 2);
        assert_eq!(band.attempted, 5);
    }

    #[test]
    fn cancelled_bootstrap_keeps_attempted_denominator_and_status() {
        let band = summarize_observation_bootstrap(&[vec![1.0], vec![3.0]], &[2.5], 2, true);
        assert!(band.cancelled);
        assert_eq!(band.completed, 2);
        assert_eq!(band.attempted, 2);
        // Two successes cannot earn a nominal 0.95 band under the (B+1) floor.
        assert!(band.lower.is_empty());
        assert!(band.upper.is_empty());
        let empty = summarize_observation_bootstrap(&[], &[], 0, true);
        assert!(empty.cancelled && empty.lower.is_empty());
    }

    #[test]
    fn earned_minimum_successes_publish_pointwise_band() {
        let draws: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
        let band = summarize_observation_bootstrap(&draws, &[19.5], 40, false);
        assert!(!band.lower.is_empty());
        assert!(!band.upper.is_empty());
        assert_eq!(band.completed, 40);
        assert_eq!(band.attempted, 40);
    }

    #[test]
    fn cancelled_at_earned_minimum_withholds_band() {
        let draws: Vec<Vec<f64>> = (0..40).map(|i| vec![f64::from(i)]).collect();
        let band = summarize_observation_bootstrap(&draws, &[19.5], 40, true);
        assert!(band.cancelled);
        assert_eq!(band.completed, 40);
        assert_eq!(band.attempted, 40);
        assert!(band.lower.is_empty());
        assert!(band.upper.is_empty());
    }
}

/// [`temporal_dependence_se_diagnostics`] for a Frequentist multi-step Sustained
/// fit on one `TemporalDag`, from the block length, aligned rows and contrast
/// influence the sequential estimator returned on `estimate` (the quantities its
/// circular-block bootstrap used; the rule length when `replicates == 0`, since
/// the score scan only runs for a bootstrap). An estimate without them reports
/// NaN effective rows, which warns.
pub(super) fn sequential_dependence_se_diagnostics(
    estimate: &EffectEstimate,
    replicates: u32,
) -> Vec<Diagnostic> {
    let (block_length, rows, kernel_bias) =
        estimate.uncertainty.block.resampling.map_or((0, 0, 1.0), |geometry| {
            (geometry.block_length, geometry.rows, geometry.kernel_bias)
        });
    let scores: Vec<&[f64]> = estimate.scores.influence.as_deref().into_iter().collect();
    let effective_rows = if estimate.uncertainty.block.resampling.is_some() {
        antecedent_estimate::score_effective_rows(&scores, block_length)
    } else {
        f64::NAN
    };
    temporal_dependence_se_diagnostics(
        antecedent_estimate::CircularBlockFamily::Sequential,
        block_length,
        rows,
        kernel_bias,
        effective_rows,
        replicates > 0,
        &format!(
            "se_bootstrap refits every mechanism of the sequential g-computation on the \
             same circular blocks of consecutive lag-aligned rows ({}/{replicates} \
             replicates){}",
            estimate.uncertainty.bootstrap.replicates_ok.unwrap_or(0),
            if replicates == 0 { "; request bootstrap_replicates > 0 for an interval" } else { "" },
        ),
    )
}

/// [`temporal_dependence_se_diagnostics`] for a Frequentist single-step fit on
/// one series, from the dependence-honest fit's own block geometry.
pub(super) fn single_window_dependence_se_diagnostics(
    estimate: &EffectEstimate,
    dependence: &antecedent_estimate::TemporalDependenceSe,
) -> Vec<Diagnostic> {
    temporal_dependence_se_diagnostics(
        antecedent_estimate::CircularBlockFamily::SingleWindow,
        dependence.block_length,
        dependence.rows,
        dependence.kernel_bias,
        dependence.effective_rows,
        dependence.replicates_attempted > 0,
        &format!(
            "se_bootstrap refits the lag-aligned design on circular blocks of \
             consecutive rows ({}/{} replicates); se_analytic is NaN: the iid OLS SE \
             ignores serial dependence and no HAC SE is calibrated for this cell{}",
            estimate.uncertainty.bootstrap.replicates_ok.unwrap_or(0),
            dependence.replicates_attempted,
            if dependence.replicates_attempted == 0 {
                "; request bootstrap_replicates > 0 for an interval"
            } else {
                ""
            },
        ),
    )
}

/// Provenance for a one-series circular-block interval, plus a warning when the
/// estimating score is too dependent for its SE family's threshold.
///
/// `dependence_aware` says whether `block_length` came from the score scan
/// ([`antecedent_estimate::dependence_block_length`]); without replicates the
/// scan is skipped and the length is the plain rule, which the text says.
pub(super) fn temporal_dependence_se_diagnostics(
    family: antecedent_estimate::CircularBlockFamily,
    block_length: usize,
    rows: usize,
    kernel_bias: f64,
    effective_rows: f64,
    dependence_aware: bool,
    detail: &str,
) -> Vec<Diagnostic> {
    let scale = antecedent_estimate::circular_fixed_b_scale(block_length, rows);
    let length = if dependence_aware {
        "at least max(structural span, ceil(n^(1/3))), capped at n, and lengthened for a \
         persistently dependent estimating score"
    } else {
        "the rule max(structural span, ceil(n^(1/3))) capped at n; the dependence-aware \
         lengthening scans the estimating scores only when replicates are drawn"
    };
    let mut out = vec![Diagnostic::new(
        "estimate.temporal.circular_block_se",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "dependence-honest SE: circular-block length {block_length} ({length}), \
             n={rows} lag-aligned rows; \
             replicate SD scaled by the circular-Bartlett fixed-b factor {scale:.4} and \
             the Bartlett kernel-bias factor {kernel_bias:.4} of the estimating score; \
             score effective rows {effective_rows:.1} (the smaller of the lag-1 and \
             block-length readings); {detail}"
        ),
    )];
    out.extend(short_series_warning(effective_rows, family));
    out
}

impl super::Study {
    /// The isotropic conjugate prior scale a temporal cross-world fit uses
    /// under Bayesian inference; a mapped coefficient prior is refused.
    fn cross_world_prior_scale(&self) -> Result<Option<f64>, CausalError> {
        match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                if cfg.has_transfer() {
                    return Err(CausalError::Unsupported {
                        message: "temporal cross-world effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed",
                    });
                }
                Ok(Some(cfg.prior_scale))
            }
            InferenceMode::Frequentist => Ok(None),
        }
    }

    pub(super) fn execute_temporal_cross_world(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world effects are point-only; cheap/full refuters are not licensed",
            });
        }
        let prior_scale = self.cross_world_prior_scale()?;
        let CrossWorldContrast { treatment, outcome, mediator, active, control } =
            cross_world_contrast(&self.query)?;
        let probe = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let found = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, &probe)
            .map_err(CausalError::from)?;
        let identification = found.result;
        require_identified(&identification)?;
        let estimand = identification.estimands.first().cloned().ok_or_else(|| {
            CausalError::Compile { message: "temporal cross-world proof has no estimand".into() }
        })?;
        let mut direct_estimand = estimand.clone();
        if matches!(
            &self.query,
            CausalQuery::NestedCounterfactual(_) | CausalQuery::PathSpecific(_)
        ) {
            direct_estimand.adjustment_set = Arc::from([]);
        }
        let fit =
            |outcome: VariableId, treatments: &[VariableId], contemporaneous: &[VariableId]| {
                let used = match &self.query {
                    CausalQuery::NestedCounterfactual(_) | CausalQuery::PathSpecific(_) => {
                        &direct_estimand
                    }
                    _ => &estimand,
                };
                antecedent_estimate::temporal_linear_slopes(
                    data,
                    used,
                    &found.indexer,
                    &antecedent_estimate::TemporalSlopeSpec {
                        outcome,
                        treatments,
                        treatment_offset: probe.try_treatment_offset()?,
                        outcome_offset: probe.outcome_offset(),
                        contemporaneous,
                    },
                    &ctx.kernel_policy,
                    prior_scale,
                )
                .map_err(CausalError::from)
            };
        let point = cross_world_point(
            &self.query,
            &fit,
            CrossWorldContrast { treatment, outcome, mediator, active, control },
        )?;
        let identification = template_scoped_cross_world(
            identification,
            &found.indexer,
            self.query.clone(),
            CrossWorldContrast { treatment, outcome, mediator, active, control },
        );
        let estimand = identification.estimands[0].clone();
        let estimate = EffectEstimate::new(
            point,
            f64::NAN,
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    EstimatorId::TemporalLinearAdjustment,
                ),
                treatment,
                outcome,
            )
            .wall_time_ns(elapsed_ns(started)),
        );
        result.interval = None;
        result.posterior = None;
        result.diagnostics.push(Diagnostic::new(
            "estimate.temporal.cross_world",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the linear unfolded cross-world point; no interval is published",
        ));
        Ok(result)
    }
}

/// The treatment, outcome, mediator and levels of a cross-world query, or the
/// refusal when the query is not a one-mediator or hard-intervention contrast.
fn cross_world_contrast(query: &CausalQuery) -> Result<CrossWorldContrast, CausalError> {
    let (treatment, outcome, mediator, active, control) = match query {
        CausalQuery::NestedCounterfactual(q) => {
            (q.treatment, q.outcome, Some(q.mediator), q.active_value(), q.control_value())
        }
        CausalQuery::PathSpecific(q) => {
            if q.path_nodes.len() != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal path-specific effect is the one-mediator edge product",
                });
            }
            let active = hard_level(&q.active)?;
            let control = hard_level(&q.control)?;
            (q.treatment, q.outcome, Some(q.path_nodes[0]), active, control)
        }
        CausalQuery::Counterfactual(q) => {
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).ok_or(
                    CausalError::Unsupported {
                        message: "temporal counterfactual needs one hard intervention",
                    },
                )?;
            let outcome = *q.outcomes.first().ok_or(CausalError::Unsupported {
                message: "temporal counterfactual needs one outcome",
            })?;
            (treatment, outcome, None, hard_level(&q.interventions[0])?, hard_level(&q.control)?)
        }
        CausalQuery::Distribution(q) => {
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).ok_or(
                    CausalError::Unsupported {
                        message: "temporal interventional distribution needs one hard intervention",
                    },
                )?;
            let outcome = *q.outcomes.first().ok_or(CausalError::Unsupported {
                message: "temporal interventional distribution needs one outcome",
            })?;
            if !q.conditioning.is_empty() {
                return Err(CausalError::Unsupported {
                    message: "temporal interventional distribution is unconditional",
                });
            }
            (treatment, outcome, None, hard_level(&q.interventions[0])?, 0.0)
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "temporal cross-world route does not estimate this query",
            });
        }
    };
    Ok(CrossWorldContrast { treatment, outcome, mediator, active, control })
}

/// The linear unfolded cross-world point of the query's contrast, from the
/// fitted slopes of `fit`.
fn cross_world_point(
    query: &CausalQuery,
    fit: &impl Fn(VariableId, &[VariableId], &[VariableId]) -> Result<(f64, Vec<f64>), CausalError>,
    contrast: CrossWorldContrast,
) -> Result<f64, CausalError> {
    let CrossWorldContrast { treatment, outcome, mediator, active, control } = contrast;
    let delta = active - control;
    Ok(match query {
        CausalQuery::NestedCounterfactual(_) => {
            let mediator = mediator.ok_or_else(missing_cross_world_mediator)?;
            let (_, slopes) = fit(outcome, &[treatment, mediator], &[mediator])?;
            slopes[0] * delta
        }
        CausalQuery::PathSpecific(_) => {
            let mediator = mediator.ok_or_else(missing_cross_world_mediator)?;
            let (_, through) = fit(mediator, &[treatment], &[])?;
            let (_, onward) = fit(outcome, &[treatment, mediator], &[mediator])?;
            through[0] * onward[1] * delta
        }
        CausalQuery::Counterfactual(_) => {
            let (_, slopes) = fit(outcome, &[treatment], &[])?;
            slopes[0] * delta
        }
        CausalQuery::Distribution(_) => {
            let (intercept, slopes) = fit(outcome, &[treatment], &[])?;
            intercept + slopes[0] * active
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal cross-world route does not estimate this query",
        })?,
    })
}

/// A mediated cross-world contrast whose contrast record carries no mediator.
pub(super) fn missing_cross_world_mediator() -> CausalError {
    CausalError::Compile { message: "cross-world mediated contrast needs a mediator".into() }
}

/// The variables and levels of one cross-world (mediation) contrast.
#[derive(Clone, Copy)]
pub(super) struct CrossWorldContrast {
    pub(super) treatment: VariableId,
    pub(super) outcome: VariableId,
    pub(super) mediator: Option<VariableId>,
    pub(super) active: f64,
    pub(super) control: f64,
}

/// How a temporal derivative response differentiates: the lag offsets of the
/// derivative's treatment and outcome, the kernel policy and the prior scale.
#[derive(Clone, Copy)]
pub(super) struct DerivativeSpec<'a> {
    pub(super) treatment_offset: i32,
    pub(super) outcome_offset: i32,
    pub(super) policy: &'a antecedent_core::KernelPolicy,
    pub(super) prior_scale: Option<f64>,
}

pub(super) fn template_scoped_cross_world(
    mut identification: IdentificationResult,
    indexer: &antecedent_core::TemporalIndexer,
    query: CausalQuery,
    contrast: CrossWorldContrast,
) -> IdentificationResult {
    let CrossWorldContrast { treatment, outcome, mediator, active, control } = contrast;
    let map = |id: VariableId| indexer.key_of(id.raw()).map(|key| key.variable).unwrap_or(id);
    let adjustment: Vec<VariableId> = identification
        .estimands
        .first()
        .map(|estimand| {
            let mut mapped: Vec<VariableId> =
                estimand.adjustment_set.iter().copied().map(map).collect();
            mapped.sort_by_key(|id| id.raw());
            mapped.dedup();
            mapped
        })
        .unwrap_or_default();
    let mut arena = CausalExprArena::new();
    let backdoor_method = identification.estimands.first().map_or_else(
        || Arc::from("temporal.backdoor.unfolded"),
        |estimand| estimand.method.clone(),
    );
    let active_value = antecedent_core::Value::f64(active);
    let control_value = antecedent_core::Value::f64(control);
    // Nested and path points are the direct slope and the mediator-edge product
    // that were fit. A counterfactual and an interventional mean stay the
    // back-door formula of the adjustment set that entered that fit.
    let (method, adjustment, mediators, functional) = match (&query, mediator) {
        (CausalQuery::NestedCounterfactual(_), Some(mediator)) => (
            Arc::from(EstimandMethod::TemporalMediationDirect.as_str()),
            Arc::from([]),
            Arc::from([mediator]),
            direct_slope_functional(&mut arena, treatment, mediator, outcome, active, control),
        ),
        (CausalQuery::PathSpecific(_), Some(mediator)) => (
            Arc::from(EstimandMethod::PathSpecificNatural.as_str()),
            Arc::from([]),
            Arc::from([mediator]),
            mediator_edge_product_functional(
                &mut arena, treatment, mediator, outcome, active, control,
            ),
        ),
        (CausalQuery::Distribution(_), _) => {
            let functional = arena.backdoor_mean(treatment, outcome, &adjustment, active_value);
            (backdoor_method, Arc::from(adjustment), Arc::from([]), functional)
        }
        _ => {
            let functional =
                arena.backdoor_ate(treatment, outcome, &adjustment, active_value, control_value);
            (backdoor_method, Arc::from(adjustment), Arc::from([]), functional)
        }
    };
    identification.query = query;
    identification.arena = arena;
    identification.estimands = vec![IdentifiedEstimand::new(
        method,
        adjustment,
        Arc::from([]),
        mediators,
        functional,
        None,
    )];
    identification.hedge = None;
    identification
}

/// Linear natural direct effect: the treatment slope in the outcome regression
/// that holds the contemporaneous mediator, times the level gap.
fn direct_slope_functional(
    arena: &mut CausalExprArena,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
) -> antecedent_expr::ExprId {
    let left = held_mediator_mean(arena, treatment, mediator, outcome, active);
    let right = held_mediator_mean(arena, treatment, mediator, outcome, control);
    let contrast = arena.intern(ExprNode::Contrast {
        left,
        right,
        op: antecedent_expr::ContrastOp::Difference,
    });
    arena.set_derivation(
        contrast,
        DerivationMeta::rule(
            "temporal.cross_world.direct_slope",
            Some(Arc::from(
                "outcome-regression treatment slope with the contemporaneous mediator held, times the level gap",
            )),
        ),
    );
    contrast
}

fn held_mediator_mean(
    arena: &mut CausalExprArena,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    level: f64,
) -> antecedent_expr::ExprId {
    let variables = arena.intern_var_set([outcome]);
    let conditioned_on = arena.intern_var_set([treatment, mediator]);
    let intervention =
        arena.intern_intervention_assignments([antecedent_expr::InterventionAssignment::concrete(
            treatment,
            antecedent_core::Value::f64(level),
        )]);
    let distribution = arena.intern_distribution(
        variables,
        conditioned_on,
        intervention,
        DomainRef::Interventional,
    );
    arena.intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution })
}

/// Product of the treatment-to-mediator edge over the level gap and the
/// mediator-to-outcome partial slope.
fn mediator_edge_product_functional(
    arena: &mut CausalExprArena,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
) -> antecedent_expr::ExprId {
    let through = arena.backdoor_ate(
        treatment,
        mediator,
        &[],
        antecedent_core::Value::f64(active),
        antecedent_core::Value::f64(control),
    );
    let onward_high = held_treatment_mean(arena, treatment, mediator, outcome, control, 1.0);
    let onward_low = held_treatment_mean(arena, treatment, mediator, outcome, control, 0.0);
    let onward = arena.intern(ExprNode::Contrast {
        left: onward_high,
        right: onward_low,
        op: antecedent_expr::ContrastOp::Difference,
    });
    let factors = arena.intern_list([through, onward]);
    let product = arena.intern(ExprNode::Product(factors));
    arena.set_derivation(
        product,
        DerivationMeta::rule(
            "temporal.cross_world.mediator_edge_product",
            Some(Arc::from(
                "product of the treatment-to-mediator slope and the mediator-to-outcome slope, times the level gap",
            )),
        ),
    );
    product
}

fn held_treatment_mean(
    arena: &mut CausalExprArena,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    treatment_level: f64,
    mediator_level: f64,
) -> antecedent_expr::ExprId {
    let variables = arena.intern_var_set([outcome]);
    let conditioned_on = arena.intern_var_set([treatment, mediator]);
    let intervention = arena.intern_intervention_assignments([
        antecedent_expr::InterventionAssignment::concrete(
            treatment,
            antecedent_core::Value::f64(treatment_level),
        ),
        antecedent_expr::InterventionAssignment::concrete(
            mediator,
            antecedent_core::Value::f64(mediator_level),
        ),
    ]);
    let distribution = arena.intern_distribution(
        variables,
        conditioned_on,
        intervention,
        DomainRef::Interventional,
    );
    arena.intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution })
}

fn hard_level(intervention: &Intervention) -> Result<f64, CausalError> {
    match intervention {
        Intervention::Set { value: antecedent_core::Value::Float64(level), .. }
            if level.is_finite() =>
        {
            Ok(*level)
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal cross-world interventions are finite hard sets",
        }),
    }
}

/// The fitted slope of a point derivative on the requested scale, evaluated at
/// the treatment point `at`.
fn scaled_point_slope(
    scale: antecedent_core::DerivativeScale,
    at: f64,
    (intercept, slope): (f64, f64),
) -> Result<f64, CausalError> {
    use antecedent_core::DerivativeScale;
    let mean = intercept + slope * at;
    match scale {
        DerivativeScale::Identity => Ok(slope),
        DerivativeScale::LogTreatment if at > 0.0 => Ok(at * slope),
        DerivativeScale::LogOutcome if mean.is_finite() && mean != 0.0 => Ok(slope / mean),
        DerivativeScale::LogLog if at > 0.0 && mean.is_finite() && mean != 0.0 => {
            Ok(at * slope / mean)
        }
        _ => Err(CausalError::Unsupported {
            message: "temporal derivative scale needs a positive treatment point and a non-zero fitted mean",
        }),
    }
}

/// The directional derivative of each outcome along `direction`, from the
/// fitted treatment slopes.
fn directional_derivative_value(
    fit: &impl Fn(VariableId, &[VariableId]) -> Result<(f64, Vec<f64>), CausalError>,
    outcomes: &[VariableId],
    treatments: &[VariableId],
    direction: &[f64],
) -> Result<ResponseValue, CausalError> {
    let mut values = Vec::with_capacity(outcomes.len());
    for &outcome in outcomes {
        let (_, slopes) = fit(outcome, treatments)?;
        values.push(slopes.iter().zip(direction.iter()).map(|(slope, step)| slope * step).sum());
    }
    Ok(ResponseValue::Vector(std::sync::Arc::from(values)))
}

/// The Jacobian of the outcomes with respect to the treatments at `at`, with
/// the requested derivative scale applied to each fitted slope.
fn jacobian_derivative_value(
    fit: &impl Fn(VariableId, &[VariableId]) -> Result<(f64, Vec<f64>), CausalError>,
    (outcomes, treatments): (&[VariableId], &[VariableId]),
    at: &[f64],
    scale: antecedent_core::DerivativeScale,
) -> Result<ResponseValue, CausalError> {
    use antecedent_core::DerivativeScale;
    let mut values = Vec::with_capacity(outcomes.len() * treatments.len());
    for &outcome in outcomes {
        let (intercept, slopes) = fit(outcome, treatments)?;
        let mean = intercept
            + slopes.iter().zip(at.iter()).map(|(slope, point)| slope * point).sum::<f64>();
        for (&slope, &point) in slopes.iter().zip(at.iter()) {
            let scaled = match scale {
                DerivativeScale::Identity => slope,
                DerivativeScale::LogTreatment if point > 0.0 => point * slope,
                DerivativeScale::LogOutcome if mean.is_finite() && mean != 0.0 => slope / mean,
                DerivativeScale::LogLog if point > 0.0 && mean.is_finite() && mean != 0.0 => {
                    point * slope / mean
                }
                _ => {
                    return Err(CausalError::Unsupported {
                        message: "temporal Jacobian scale needs a positive treatment point and a non-zero fitted mean",
                    });
                }
            };
            values.push(scaled);
        }
    }
    Ok(ResponseValue::Jacobian {
        outcomes: outcomes.len(),
        treatments: treatments.len(),
        values: std::sync::Arc::from(values),
    })
}

pub(super) fn temporal_linear_derivative_response(
    data: &TimeSeriesData,
    query: &ResponseQuery,
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    identification: &IdentificationResult,
    spec: DerivativeSpec<'_>,
) -> Result<CausalResponse, CausalError> {
    use antecedent_core::{DerivativeWeighting, ResponseFunctional as F};
    let DerivativeSpec { treatment_offset, outcome_offset, policy, prior_scale } = spec;
    if query.target_population != antecedent_core::TargetPopulation::AllObserved {
        return Err(CausalError::Unsupported {
            message: "temporal derivative requires the all-observed population",
        });
    }
    let fit = |outcome, treatments: &[VariableId]| {
        antecedent_estimate::temporal_linear_slopes(
            data,
            estimand,
            indexer,
            &antecedent_estimate::TemporalSlopeSpec {
                outcome,
                treatments,
                treatment_offset,
                outcome_offset,
                contemporaneous: &[],
            },
            policy,
            prior_scale,
        )
        .map_err(CausalError::from)
    };
    let mut support_at = vec![0.0_f64];
    let support_treatments;
    let mut average_over_observed = false;
    let value = match &query.functional {
        F::AverageDerivative { outcome, treatment, weighting } => {
            if !matches!(weighting, DerivativeWeighting::Observed) {
                return Err(CausalError::Unsupported {
                    message: "temporal average derivative is licensed for the observed weighting",
                });
            }
            support_treatments = vec![*treatment];
            average_over_observed = true;
            let (_, slopes) = fit(*outcome, &[*treatment])?;
            ResponseValue::Scalar(slopes[0])
        }
        F::PointDerivative { outcome, treatment, at, order, scale } => {
            if *order != 1 {
                return Err(CausalError::Unsupported {
                    message: "temporal point derivative is the first derivative of the linear fit",
                });
            }
            let (intercept, slopes) = fit(*outcome, &[*treatment])?;
            support_treatments = vec![*treatment];
            support_at = vec![*at];
            ResponseValue::Scalar(scaled_point_slope(*scale, *at, (intercept, slopes[0]))?)
        }
        F::DirectionalDerivative { outcomes, treatments, at, direction } => {
            if at.len() != treatments.len() || direction.len() != treatments.len() {
                return Err(CausalError::Unsupported {
                    message: "temporal directional derivative point and direction follow the treatments",
                });
            }
            support_treatments = treatments.to_vec();
            support_at = at.to_vec();
            directional_derivative_value(&fit, outcomes, treatments, direction)?
        }
        F::Jacobian { outcomes, treatments, at, scale } => {
            if at.len() != treatments.len() {
                return Err(CausalError::Unsupported {
                    message: "temporal Jacobian is evaluated at one point per treatment",
                });
            }
            support_treatments = treatments.to_vec();
            support_at = at.to_vec();
            jacobian_derivative_value(&fit, (outcomes, treatments), at, *scale)?
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "temporal linear derivative does not estimate this functional",
            });
        }
    };
    let support =
        derivative_support_report(data, &support_treatments, &support_at, average_over_observed);
    Ok(CausalResponse {
        estimand: query.functional.clone(),
        identification_status: identification.status,
        estimate: ResponseIdentification::PointIdentified(value),
        uncertainty: ResponseUncertainty::None,
        support,
        assumptions: identification.required_assumptions.clone(),
        provenance_id: std::sync::Arc::from("estimate.temporal.linear_derivative"),
        horizon_identification: None,
        interaction_structurally_zero: false,
    })
}

/// Average derivatives are supported on the observed treatment range. A requested
/// evaluation point is outside the data when it leaves that range.
fn derivative_support_report(
    data: &TimeSeriesData,
    treatments: &[VariableId],
    requested: &[f64],
    average_over_observed: bool,
) -> antecedent_core::SupportReport {
    use antecedent_core::{SupportRegion, SupportReport, SupportStatus};
    let range = |variable: VariableId| -> Option<(f64, f64)> {
        let values = data.float64_values(variable).ok()?;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for value in values {
            if value.is_finite() {
                min = min.min(value);
                max = max.max(value);
            }
        }
        (min.is_finite() && max.is_finite()).then_some((min, max))
    };
    if average_over_observed {
        let (minima, maxima, status) = match treatments.first().and_then(|id| range(*id)) {
            Some((min, max)) => (vec![min], vec![max], SupportStatus::Supported),
            None => (vec![f64::NAN], vec![f64::NAN], SupportStatus::OutsideEmpiricalSupport),
        };
        let mut warnings = Vec::new();
        if status != SupportStatus::Supported {
            warnings.push(Diagnostic::new(
                "estimate.temporal.linear_derivative.support",
                DiagnosticKind::Support,
                DiagnosticSeverity::Warning,
                "the average derivative has no finite observed treatment values",
            ));
        }
        return SupportReport {
            status,
            query_region: SupportRegion {
                minima: std::sync::Arc::from(minima),
                maxima: std::sync::Arc::from(maxima),
            },
            diagnostics: Vec::new(),
            warnings,
            point_status: None,
        };
    }
    let inside = treatments.iter().zip(requested).all(|(variable, point)| {
        range(*variable)
            .is_some_and(|(min, max)| point.is_finite() && *point >= min && *point <= max)
    });
    let status = if inside && treatments.len() == requested.len() && !requested.is_empty() {
        SupportStatus::Supported
    } else {
        SupportStatus::OutsideEmpiricalSupport
    };
    let mut warnings = Vec::new();
    if status != SupportStatus::Supported {
        warnings.push(Diagnostic::new(
            "estimate.temporal.linear_derivative.support",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "the requested evaluation point is outside the observed treatment series",
        ));
    }
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: std::sync::Arc::from(requested),
            maxima: std::sync::Arc::from(requested),
        },
        diagnostics: Vec::new(),
        warnings,
        point_status: None,
    }
}

pub(super) fn temporal_cross_world_hard_level(
    intervention: &Intervention,
) -> Result<f64, CausalError> {
    hard_level(intervention)
}

type BayesianMediationFit =
    (antecedent_estimate::CausalPosterior, Vec<antecedent_prob::ConflictSummary>);

/// The inputs a class completion's temporal mediation fit or refutation is keyed
/// and computed from.
#[derive(Clone, Copy)]
struct ClassMediationInput<'a> {
    data: &'a TimeSeriesData,
    estimand: &'a IdentifiedEstimand,
    query: &'a antecedent_core::MediationQuery,
    adjustment: &'a Arc<[antecedent_data::LaggedColumn]>,
    status: IdentificationStatus,
}

/// The series, query, estimand and lag indexer a class completion's temporal
/// response is fitted from.
#[derive(Clone, Copy)]
struct ClassResponseInput<'a> {
    series: &'a TimeSeriesData,
    query: &'a ResponseQuery,
    estimand: &'a IdentifiedEstimand,
    indexer: &'a TemporalIndexer,
}

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

/// The unit-contrast effect query a class response identifies its completion
/// envelope with at one horizon.
fn class_horizon_witness(
    query: &ResponseQuery,
    temporal: &antecedent_core::TemporalResponseSpec,
    (treatment, outcome): (VariableId, VariableId),
    horizon: u32,
) -> TemporalEffectQuery {
    unit_contrast_query(temporal, (treatment, outcome), horizon, query.target_population.clone())
}

/// Graph key of the completion at `case_index` of the envelope identified at `horizon_index`.
pub(super) const fn class_completion_key(horizon_index: usize, case_index: usize) -> u64 {
    ((horizon_index as u64) << 32) | case_index as u64
}

/// A completion atom that carries only its mass and identification status.
pub(super) fn bare_class_atom(
    graph_key: u64,
    weight: f64,
    status: IdentificationStatus,
) -> crate::result::StructuralResponseAtom {
    crate::result::StructuralResponseAtom {
        posterior: None,
        response: None,
        graph_key,
        weight,
        status,
        value: None,
    }
}

/// The unit-contrast (`do(x = 0)` against `do(x = 1)`) effect query a class response
/// identifies its completion envelopes with at one horizon, under the response's temporal
/// policy and history bound and with no effect modifiers.
pub(super) fn unit_contrast_query(
    temporal: &antecedent_core::TemporalResponseSpec,
    (treatment, outcome): (VariableId, VariableId),
    horizon: u32,
    target_population: antecedent_core::TargetPopulation,
) -> TemporalEffectQuery {
    TemporalEffectQuery {
        treatment,
        outcome,
        policy: temporal.policy.clone(),
        control: Intervention::set(treatment, antecedent_core::Value::f64(0.0)),
        active: Intervention::set(treatment, antecedent_core::Value::f64(1.0)),
        horizon_steps: horizon,
        max_history_lag: temporal.max_history_lag,
        target_population,
        effect_modifiers: std::sync::Arc::from([]),
    }
}

/// The cached identification of every requested horizon, each required to be
/// identified, in horizon order.
fn identified_horizon_entries<'a>(
    cache: &'a crate::analysis::prepared::CachedTemporalIdentification,
    temporal: &antecedent_core::TemporalResponseSpec,
) -> Result<Vec<&'a crate::analysis::prepared::CachedTemporalHorizonIdentification>, CausalError> {
    let mut aligned = Vec::with_capacity(temporal.horizons.len());
    for &horizon in temporal.horizons.iter() {
        let entry = cache.get(horizon).ok_or_else(|| CausalError::Compile {
            message: format!("temporal identification missing horizon {horizon}"),
        })?;
        require_identified(&entry.identification)?;
        aligned.push(entry);
    }
    Ok(aligned)
}

/// The series, query, specification and unfold indexer a temporal derivative
/// response is computed from.
#[derive(Clone, Copy)]
struct DerivativeResponseInputs<'a> {
    data: &'a TimeSeriesData,
    query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    indexer: &'a TemporalIndexer,
    treatment: VariableId,
    outcome: VariableId,
}

/// What the observation preparation of a temporal response reads.
#[derive(Clone, Copy)]
struct ResponseObservationInput<'a> {
    data: &'a TimeSeriesData,
    graph: &'a TemporalDag,
    query: &'a ResponseQuery,
    outcome: VariableId,
    aligned: &'a [&'a crate::analysis::prepared::CachedTemporalHorizonIdentification],
    observed_bayes: bool,
}

/// The observation-adjusted series and working query of a temporal response;
/// the series and adjustment are absent when the data is complete or the
/// observation is modelled inside the Bayesian fit.
struct ResponseObservation {
    series: Option<TimeSeriesData>,
    adjusted: Option<antecedent_estimate::ObservationAdjustedOutcome>,
    working_query: ResponseQuery,
}

/// Everything the temporal response surface estimation reads.
#[derive(Clone, Copy)]
struct TemporalResponseEstimation<'a> {
    data: &'a TimeSeriesData,
    source_data: &'a TimeSeriesData,
    graph: &'a TemporalDag,
    query: &'a ResponseQuery,
    working_query: &'a ResponseQuery,
    temporal: &'a antecedent_core::TemporalResponseSpec,
    aligned: &'a [&'a crate::analysis::prepared::CachedTemporalHorizonIdentification],
    identifications: &'a [(&'a IdentifiedEstimand, &'a TemporalIndexer)],
    status: IdentificationStatus,
    assumptions: &'a antecedent_core::AssumptionSet,
    observation_adjusted: Option<&'a antecedent_estimate::ObservationAdjustedOutcome>,
    observed_bayes: bool,
    treatment: VariableId,
    outcome: VariableId,
}

/// A temporal response surface with the prior-conflict summary and observed
/// posterior its estimation produced.
struct EstimatedTemporalResponse {
    response: CausalResponse,
    conflict_summary: Option<antecedent_prob::ConflictSummary>,
    observed_posterior: Option<antecedent_estimate::CausalPosterior>,
}

/// The diagnostics a temporal response result carries, in publication order.
fn temporal_response_diagnostics(
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    response: &CausalResponse,
    conflict_summary: Option<&antecedent_prob::ConflictSummary>,
    identify_cached: bool,
) -> Vec<Diagnostic> {
    let scalar = super::response_path::response_scalar_summary(response).0;
    let mut diagnostics = Vec::new();
    for entry in aligned {
        diagnostics.extend(entry.identification.diagnostics.iter().cloned());
    }
    if horizon_adjustment_sets_differ(aligned) {
        diagnostics.push(Diagnostic::new(
            "identify.temporal_response.horizon_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "adjustment sets differ across requested horizons; each cell uses I(h) \
             identified for that horizon, not a shared max-horizon set",
        ));
    }
    if identify_cached {
        diagnostics.push(identify_cached_diagnostic());
    }
    if let Some(summary) = conflict_summary {
        push_conflict_diagnostics(&mut diagnostics, summary);
    }
    diagnostics.push(Diagnostic::new(
        "refute.temporal_response.skipped",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "scalar ATE refuters are not applicable to a function-valued temporal response",
    ));
    if !scalar.is_finite() {
        diagnostics.push(Diagnostic::new(
            "estimate.response.no_scalar_summary",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "this response is function-valued; the scalar effect summary is not applicable \
             and the result is carried by the response payload",
        ));
    }
    for warning in &response.support.warnings {
        diagnostics.push(warning.clone());
    }
    diagnostics
}

/// The per-horizon surface of a Sequence response: pointwise means and bands,
/// each horizon's identification record, and the posteriors kept for
/// disclosure.
struct SequenceHorizonSurface {
    mean: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    horizons: Vec<antecedent_core::HorizonIdentification>,
    uncertainty_complete: bool,
    last_posterior: Option<CausalPosterior>,
    /// Every horizon's posterior notes are kept (not just the scalar artifact's),
    /// so a `tempering_capped` / `tempering_inestimable` disclosure raised at an
    /// earlier horizon still reaches the result even when only the last horizon's
    /// posterior is attached as the scalar summary.
    horizon_posteriors: Vec<CausalPosterior>,
    bootstrap_cancelled: bool,
}

/// What a Sequence response is finished from once its band is final.
struct SequenceFinish {
    response: CausalResponse,
    horizon_posteriors: Vec<CausalPosterior>,
    last_posterior: Option<CausalPosterior>,
    bootstrap_cancelled: bool,
    attach_scalar_posterior: bool,
}

/// Fits the sequential mechanisms at every requested horizon and collects the
/// pointwise surface (Wald bounds, or posterior quantiles under Bayesian
/// inference).
fn estimate_sequence_horizons(
    inputs: &SequenceResponseInputs<'_>,
    outcome: VariableId,
    bayes: Option<&BayesianGComputationAte>,
    assumptions: &antecedent_core::AssumptionSet,
    ctx: &ExecutionContext,
) -> Result<SequenceHorizonSurface, CausalError> {
    let SequenceResponseInputs { data, graph, temporal, overlays, aligned, .. } = *inputs;
    let horizon_count = temporal.horizons.len();
    let mut surface = SequenceHorizonSurface {
        mean: Vec::with_capacity(horizon_count),
        lower: Vec::with_capacity(horizon_count),
        upper: Vec::with_capacity(horizon_count),
        horizons: Vec::with_capacity(horizon_count),
        uncertainty_complete: true,
        last_posterior: None,
        horizon_posteriors: Vec::with_capacity(horizon_count),
        bootstrap_cancelled: false,
    };
    let attach_scalar_posterior = horizon_count == 1;
    let z = antecedent_stats::NORMAL_Q975;
    for (horizon_steps, entry) in temporal.horizons.iter().copied().zip(aligned.iter()) {
        require_identified(&entry.identification)?;
        let outcome_offset = i32::try_from(horizon_steps.saturating_sub(1)).unwrap_or(i32::MAX);
        let (effect, posterior) = antecedent_estimate::estimate_sequence_mechanisms(
            &antecedent_estimate::SequentialProblem {
                data,
                graph,
                indexer: &entry.indexer,
                estimand: &entry.estimand,
            },
            &antecedent_estimate::SequenceTarget { outcome, outcome_offset, overlays },
            antecedent_estimate::SequentialInference { status: entry.identification.status, assumptions: assumptions.clone(), bootstrap_replicates: // Frequentist bands: the joint tuple bootstrap below.
            0, bayesian: bayes },
            ctx,
        )
        .map_err(CausalError::from)?;
        surface.bootstrap_cancelled |= effect.uncertainty.bootstrap.cancelled;
        let (point, lo, hi) = if let Some(post) = posterior.as_ref() {
            (post.summaries.mean[0], post.summaries.q025[0], post.summaries.q975[0])
        } else {
            let se = effect.se_bootstrap().unwrap_or(effect.se_analytic());
            (effect.ate, effect.ate - z * se, effect.ate + z * se)
        };
        if let Some(post) = posterior.as_ref() {
            surface.horizon_posteriors.push(post.clone());
        }
        if attach_scalar_posterior {
            surface.last_posterior = posterior;
        }
        surface.mean.push(point);
        surface.uncertainty_complete &= lo.is_finite() && hi.is_finite();
        surface.lower.push(lo);
        surface.upper.push(hi);
        surface.horizons.push(antecedent_core::HorizonIdentification {
            horizon: horizon_steps,
            status: entry.identification.status,
            method: Arc::clone(&entry.estimand.method),
            adjustment: Arc::from(named_adjustment_keys(entry)),
        });
    }
    Ok(surface)
}

/// The support report of a Sequence overlay response: extrapolative by
/// construction, with the overlays recorded and the band disclosures attached.
fn sequence_overlay_support(
    overlays: &[antecedent_estimate::SequentialMechanismOverlay],
    temporal: &antecedent_core::TemporalResponseSpec,
    uncertainty_complete: bool,
) -> antecedent_core::SupportReport {
    let first_horizon = f64::from(temporal.horizons.first().copied().unwrap_or(1));
    let last_horizon = f64::from(temporal.horizons.last().copied().unwrap_or(1));
    let mut support = antecedent_core::SupportReport {
        // Joint longitudinal support is not established by univariate
        // ranges. Until a sequential-positivity diagnostic exists, do not
        // label an arbitrary overlay schedule empirically supported.
        status: antecedent_core::SupportStatus::Extrapolative,
        query_region: antecedent_core::SupportRegion {
            minima: Arc::from([first_horizon]),
            maxima: Arc::from([last_horizon]),
        },
        diagnostics: vec![antecedent_core::SupportDiagnostic {
            id: Arc::from("response.temporal.sequence_overlay"),
            values: Arc::from(
                overlays
                    .iter()
                    .flat_map(|overlay| {
                        [
                            f64::from(overlay.node.variable.raw()),
                            f64::from(overlay.node.offset),
                            if overlay.node.level.is_some() { 1.0 } else { 0.0 },
                            overlay.node.level.unwrap_or(0.0),
                            overlay.node.shift,
                            overlay.multiplier,
                            if overlay.bounds.is_some() { 1.0 } else { 0.0 },
                            overlay.bounds.map_or(0.0, |bounds| bounds.0),
                            overlay.bounds.map_or(0.0, |bounds| bounds.1),
                        ]
                    })
                    .collect::<Vec<_>>(),
            ),
            detail: Arc::from(
                "sequential Sequence overlays as \
                 [variable, offset, is_fixed_level, level_or_zero, shift, multiplier, \
                 has_bounds, lower_or_zero, upper_or_zero, …]; \
                 transforms apply to deterministic propagated means, not stochastic draws",
            ),
            scope: antecedent_core::DiagnosticScope::Global,
        }],
        warnings: vec![Diagnostic::new(
            "response.temporal.sequence_joint_support_unassessed",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "joint longitudinal positivity for the requested Sequence overlay is not \
             estimated; support is conservatively marked extrapolative",
        )],
        point_status: Some(Arc::from(vec![
            antecedent_core::SupportStatus::Extrapolative;
            temporal.horizons.len()
        ])),
    };
    antecedent_estimate::publish_simultaneous_band(
        &mut support,
        Err(antecedent_estimate::EstimationError::unsupported(
            "Sequence overlays publish a simultaneous band only from the Frequentist joint \
             tuple bootstrap (request bootstrap replicates); Bayesian Sequence overlays are \
             sampled one horizon at a time, so there is no joint draw surface",
        )),
        "",
    );
    if !uncertainty_complete {
        support.warnings.push(Diagnostic::new(
            "response.temporal.sequence_uncertainty_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "a complete pointwise Sequence band was unavailable; uncertainty is omitted \
             rather than represented by non-finite endpoints",
        ));
    }
    support
}

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
}

/// Completions given the same estimand, indexer, status and assumptions run
/// one fit at a horizon; every other fit input is fixed per horizon.
#[derive(Default)]
struct ClassResponseHorizonFits {
    bayesian: CompletionFitMemo<BayesianResponseFit, ResponseFitKey>,
    frequentist: CompletionFitMemo<CausalResponse, ResponseFitKey>,
}

impl ClassSequenceRun {
    fn new() -> Self {
        Self {
            lower: Vec::new(),
            upper: Vec::new(),
            structural_atoms: Vec::new(),
            primary_estimand: None,
            primary_identification: None,
            primary_envelope: None,
            assumptions: antecedent_core::AssumptionSet::new(),
            full_mass_scope: true,
            truncated_atoms: 0,
            horizon_fingerprints: Vec::new(),
            observation_atoms: Vec::new(),
            diagnostics: vec![Diagnostic::new(
                "estimate.temporal.sequence_overlay",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "class-aware Sequence runs per directed completion; a joint-coordinate miss \
                 unidentifies that completion, not the class; no last-step collapse",
            )],
        }
    }
}

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
    overlays: &'a [antecedent_estimate::SequentialMechanismOverlay],
    bayes: Option<&'a BayesianGComputationAte>,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &'a ExecutionContext,
}

/// Accumulators of a class Sequence sweep: per-horizon envelope bounds, the
/// structural atoms and the first completion's identification products.
struct ClassSequenceRun {
    lower: Vec<f64>,
    upper: Vec<f64>,
    structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    primary_estimand: Option<IdentifiedEstimand>,
    primary_identification: Option<IdentificationResult>,
    primary_envelope: Option<Arc<antecedent_identify::TemporalClassEnvelope>>,
    assumptions: antecedent_core::AssumptionSet,
    full_mass_scope: bool,
    truncated_atoms: usize,
    horizon_fingerprints: Vec<Vec<u64>>,
    observation_atoms: Vec<ClassObservationAtom>,
    diagnostics: Vec<Diagnostic>,
}

/// The series, query, Bayesian configuration and context a class Bayesian
/// temporal effect fits and refutes with.
#[derive(Clone, Copy)]
struct ClassBayesianInputs<'a> {
    data: &'a TimeSeriesData,
    query: &'a TemporalEffectQuery,
    cfg: &'a BayesianConfig,
    ctx: &'a ExecutionContext,
}

/// One identified completion to fit: its fingerprint, estimand, status,
/// unfold indexer and class weight.
type ClassBayesianFitAtom = (u64, IdentifiedEstimand, IdentificationStatus, TemporalIndexer, f64);

/// Per-completion weights, identification flags and fingerprints of a class
/// envelope, with the completions to fit and the first estimand.
struct ClassBayesianPlan {
    weights: Vec<f64>,
    flags: Vec<GraphIdentFlag>,
    keys: Vec<u64>,
    fit_atoms: Vec<ClassBayesianFitAtom>,
    primary_estimand: Option<IdentifiedEstimand>,
}

/// The fitted completion atoms of a class Bayesian temporal effect with the
/// seeds, effect draws and means they were fitted under.
struct ClassBayesianFits {
    atoms: Vec<EnvelopeAtomFit>,
    atom_seeds: Vec<u64>,
    per_graph: Vec<GraphEffectDraws>,
    atom_means: Vec<f64>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
    /// The Bayesian g-computation estimator the atoms were fitted with; each
    /// refutation replaces its prior and seed.
    estimator: BayesianGComputationAte,
}

/// Classifies each completion of a class envelope: its weight (the class
/// prior's mass when supplied), whether it is identified, and which to fit.
fn plan_class_bayesian_atoms(
    bundle_envelope: &antecedent_identify::TemporalClassEnvelope,
    class_masses: Option<&[f64]>,
) -> Result<ClassBayesianPlan, CausalError> {
    let mut plan = ClassBayesianPlan {
        weights: Vec::new(),
        flags: Vec::new(),
        keys: Vec::new(),
        fit_atoms: Vec::new(),
        primary_estimand: None,
    };
    for (i, (case, indexer)) in
        bundle_envelope.envelope.cases.iter().zip(bundle_envelope.indexers.iter()).enumerate()
    {
        let key = case.graph.fingerprint();
        plan.keys.push(key);
        let weight = class_masses.map_or(case.weight.0, |masses| masses[i]);
        plan.weights.push(weight);
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            plan.flags.push(GraphIdentFlag::Unidentified);
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        plan.primary_estimand.get_or_insert_with(|| estimand.clone());
        plan.flags.push(GraphIdentFlag::Identified);
        plan.fit_atoms.push((key, estimand, case.result.status, indexer.clone(), weight));
    }
    Ok(plan)
}

/// The composed Bayesian mediation posterior of one class completion, with the
/// prior-conflict summaries its mechanism fits resolved, served from `fits`
/// when an earlier completion (or the shared-design recompute) of the same
/// horizon fitted the same method, mediator, adjustment and status.
#[inline(never)]
fn fit_class_bayesian_mediation(
    fits: &mut CompletionFitMemo<BayesianMediationFit, MediationFitKey>,
    cfg: &BayesianConfig,
    input: ClassMediationInput<'_>,
    horizon: u32,
    ctx: &ExecutionContext,
) -> Result<BayesianMediationFit, CausalError> {
    let ClassMediationInput { data, estimand, query, adjustment, status } = input;
    let key = MediationFitKey::new(estimand, adjustment, status);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let bayes = bayesian_gcomp(cfg, ctx);
    let preparations =
        antecedent_estimate::bayesian_mediation::prepare_temporal_mediation_adjusted(
            data, estimand, query, adjustment, ctx,
        )
        .map_err(CausalError::from)?;
    let mut conflicts = Vec::new();
    let mut mechanisms = Vec::with_capacity(preparations.len());
    for (index, prep) in preparations.iter().enumerate() {
        let mut mechanism = bayes.clone();
        mechanism.seed = mediation_mechanism_seed(mechanism.seed, horizon, index);
        let (prior, conflict) = resolve_envelope_prior_anchor(cfg, prep, ctx)?;
        conflicts.extend(conflict);
        mechanism.prior = prior;
        mechanisms.push(
            mechanism
                .fit(prep, status, &mut BayesianGCompWorkspace::default(), ctx)
                .map_err(CausalError::from)?,
        );
    }
    let composed = antecedent_estimate::bayesian_mediation::compose_temporal_mediation(
        &mechanisms[0],
        &mechanisms[1],
        query,
        status,
    )
    .map_err(CausalError::from)?;
    let fitted = (composed, conflicts);
    fits.insert(key, fitted.clone());
    Ok(fitted)
}

/// The point mediation estimate of one class completion, served from `fits`
/// when an earlier completion of the horizon fitted the same method, mediator
/// and adjustment.
#[inline(never)]
fn fit_class_frequentist_mediation(
    fits: &mut CompletionFitMemo<antecedent_estimate::TemporalMediationEstimate, MediationFitKey>,
    estimator: &TemporalMediationEstimator,
    input: ClassMediationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<antecedent_estimate::TemporalMediationEstimate, CausalError> {
    let ClassMediationInput { data, estimand, query, adjustment, status } = input;
    let key = MediationFitKey::new(estimand, adjustment, status);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let estimate = estimator
        .estimate_with_adjustment(data, estimand, query, adjustment, &[], ctx)
        .map_err(CausalError::from)?;
    fits.insert(key, estimate.clone());
    Ok(estimate)
}

/// The refutation reports of one class completion's mediation estimate, served
/// from `fits` under the key of the estimate they check. The reports read the
/// estimand only through its mediator and the estimator's method check.
#[inline(never)]
fn refute_class_mediation(
    fits: &mut CompletionFitMemo<Vec<antecedent_validate::RefutationReport>, MediationFitKey>,
    input: ClassMediationInput<'_>,
    estimate: &antecedent_estimate::TemporalMediationEstimate,
    full: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<antecedent_validate::RefutationReport>, CausalError> {
    let ClassMediationInput { data, estimand, query, adjustment, status } = input;
    let key = MediationFitKey::new(estimand, adjustment, status);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let reports = antecedent_validate::mediation::refute_temporal_mediation_adjusted(
        data, estimand, query, estimate, full, adjustment, ctx,
    )
    .map_err(CausalError::from)?;
    fits.insert(key, reports.clone());
    Ok(reports)
}

type BayesianResponseFit = (CausalResponse, Option<antecedent_prob::ConflictSummary>);

/// One class completion's Bayesian temporal response and the prior-conflict
/// summary its prior resolution produced, served from `fits` when an earlier
/// completion of the horizon was given the same fit inputs.
#[inline(never)]
fn fit_class_bayesian_response(
    fits: &mut CompletionFitMemo<BayesianResponseFit, ResponseFitKey>,
    cfg: &BayesianConfig,
    input: ClassResponseInput<'_>,
    horizon: u32,
    identification: &IdentificationResult,
    (treatment, outcome): (VariableId, VariableId),
    ctx: &ExecutionContext,
) -> Result<BayesianResponseFit, CausalError> {
    let ClassResponseInput { series, query, estimand, indexer } = input;
    let status = identification.status;
    let assumptions = &identification.required_assumptions;
    let key = ResponseFitKey::new(estimand, indexer, status, assumptions);
    if let Some(hit) = fits.get(&key) {
        return Ok(hit.clone());
    }
    let mut bayes = bayesian_gcomp(cfg, ctx);
    let entry = crate::analysis::prepared::CachedTemporalHorizonIdentification {
        horizon,
        identification: identification.clone(),
        estimand: estimand.clone(),
        indexer: indexer.clone(),
    };
    let (prior, conflict) = resolve_temporal_response_prior(
        cfg,
        series,
        query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "class temporal response query carries no temporal spec".into(),
        })?,
        &[&entry],
        treatment,
        outcome,
        ctx,
    )?;
    bayes.prior = prior;
    let response = TemporalResponseEstimator::new()
        .estimate_bayesian(
            series,
            antecedent_estimate::TemporalIdentification {
                horizons: &[(estimand, indexer)],
                status,
                assumptions: assumptions.clone(),
            },
            query,
            &bayes,
            ctx,
        )
        .map_err(CausalError::from)?;
    let fitted = (response, conflict);
    fits.insert(key, fitted.clone());
    Ok(fitted)
}

/// One class completion's frequentist temporal response, served from `fits`
/// (when the caller supplies a memo) under the same condition.
#[inline(never)]
fn fit_class_frequentist_response(
    fits: Option<&mut CompletionFitMemo<CausalResponse, ResponseFitKey>>,
    estimator: &TemporalResponseEstimator,
    input: ClassResponseInput<'_>,
    status: IdentificationStatus,
    assumptions: &antecedent_core::AssumptionSet,
    ctx: &ExecutionContext,
) -> Result<CausalResponse, CausalError> {
    let ClassResponseInput { series, query, estimand, indexer } = input;
    let key = ResponseFitKey::new(estimand, indexer, status, assumptions);
    if let Some(hit) = fits.as_deref().and_then(|memo| memo.get(&key)) {
        return Ok(hit.clone());
    }
    let response = estimator
        .estimate(series, &[(estimand, indexer)], query, status, assumptions.clone(), ctx)
        .map_err(CausalError::from)?;
    if let Some(memo) = fits {
        memo.insert(key, response.clone());
    }
    Ok(response)
}
