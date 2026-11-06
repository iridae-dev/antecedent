// SPDX-License-Identifier: MIT OR Apache-2.0

use super::completion_fit_memo::{CompletionFitMemo, MediationFitKey, ResponseFitKey};
use super::temporal_effect_core::FitFinishMeta;
use super::temporal_point_uncertainty::{self as point_uncertainty, SlopeSource};
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

mod class_effect;
mod class_response;
mod cpdag_mediation;
mod cross_world;
mod derivative;
mod response;

impl super::Study {
    pub(crate) fn mediation_adjustment(
        graph: &TemporalDag,
        query: &antecedent_core::MediationQuery,
        horizon: u32,
        reused: Option<&crate::analysis::prepared::CachedTemporalIdentification>,
    ) -> Result<Arc<[antecedent_data::LaggedColumn]>, CausalError> {
        if let Some(entry) = reused.and_then(|cache| cache.get(horizon)) {
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
        if matches!(
            self.declared.estimator,
            Some(
                EstimatorId::TemporalLaggedInstrument
                    | EstimatorId::BayesianTemporalLaggedInstrument
            )
        ) {
            return self.execute_temporal_iv(data, graph, query, physical, ctx);
        }
        let started = Instant::now();
        let (identification, estimand, indexer, identify_cached) = if let Some(operation) =
            checked_bayesian
        {
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
        } else if let Some((cache, prepared)) = self.reuse_temporal_identification(physical) {
            let entry = cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                message: format!(
                    "prepared temporal identification missing horizon {}",
                    query.horizon_steps
                ),
            })?;
            (entry.identification.clone(), entry.estimand.clone(), entry.indexer.clone(), prepared)
        } else {
            report_identify_compute(ctx);
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
        let pulse_support = super::pulse_support::pulse_support(data, query, ctx)?;
        let mut fit = super::temporal_effect_core::fit_temporal_effect(
            &super::temporal_effect_core::TemporalEffectInputs {
                data,
                graph,
                query,
                identification: &identification,
                estimand: &estimand,
                indexer: &indexer,
                inference: &self.settings.inference,
                checked_bayesian,
                bootstrap_replicates: self.settings.bootstrap_replicates,
                refute: self.settings.refute,
                custom_validators: &self.declared.custom_validators,
                split: self.declared.split.as_ref(),
                latency_mode: self.latency_mode,
                completed_to_dag: super::temporal_effect_core::plan_completed_to_dag(physical),
                missing: self.declared.missing.as_ref(),
            },
            ctx,
        )?;
        fit.diagnostics.extend(pulse_support);
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
        result.extras.missing_data = missing_data;
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
        let inference = match &self.settings.inference {
            InferenceMode::Frequentist => MediationInference::Frequentist {
                bootstrap_replicates: self.settings.bootstrap_replicates,
            },
            InferenceMode::Bayesian(config) => {
                require_isotropic_mediation_prior(config)?;
                MediationInference::Bayesian(config)
            }
        };
        let reused = self.reuse_temporal_identification(physical);
        let (cache, identify_cached) = if let Some((cache, prepared)) = reused.clone() {
            (cache, prepared)
        } else {
            report_identify_compute(ctx);
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            (
                Arc::new(crate::analysis::prepared::identify_temporal_mediation_horizons_within(
                    graph,
                    query,
                    inference.estimator_id(),
                    &mut budget,
                    ctx,
                )?),
                false,
            )
        };
        let reused = reused.as_ref().map(|(cache, _)| &**cache);
        let clicks = mediation_horizon_clicks(&cache, query, |horizon| {
            Self::mediation_adjustment(graph, query, horizon, reused)
        })?;
        let fit = fit_temporal_mediation(
            &TemporalMediationInputs {
                data,
                query,
                horizons: &clicks,
                inference,
                validation: self.settings.refute,
                missing: self.declared.missing.as_ref(),
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
        result.extras.missing_data = missing_data;
        Ok(result)
    }

    /// Whether the prepared class cache holds a certificate for every horizon.
    pub(crate) fn temporal_class_cache_covers(&self, horizons: &[u32]) -> bool {
        self.temporal_class_cache().map(|cache| &**cache).is_some_and(|cache| {
            horizons
                .iter()
                .all(|horizon| cache.by_horizon.iter().any(|(cached, _)| cached == horizon))
        })
    }

    /// The class envelope the plan's lowered program retained for `horizon`, when
    /// this study carries no prepared class cache.
    pub(crate) fn retained_temporal_class_horizon(
        &self,
        physical: &PhysicalExecutionPlan,
        horizon: u32,
    ) -> Option<Arc<crate::analysis::prepared::CachedTemporalClassIdentification>> {
        if self.temporal_class_cache().is_some() {
            return None;
        }
        let retained = crate::analysis::lowering::plan_retained_temporal_class(self, physical)?;
        let (_, envelope) = retained.by_horizon.iter().find(|(cached, _)| *cached == horizon)?;
        Some(Arc::new(crate::analysis::prepared::CachedTemporalClassIdentification {
            envelope: Arc::clone(envelope),
            by_horizon: Vec::new(),
        }))
    }

    /// [`Self::identify_temporal_class`] in the executor order: the prepared
    /// certificate for the query's horizon, then the one the plan's lowered program
    /// retained, then a fresh identification.
    pub(crate) fn identify_temporal_class_reusing_within(
        &self,
        identifier_id: IdentifierId,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Arc<crate::analysis::prepared::CachedTemporalClassIdentification>, CausalError>
    {
        match self.retained_temporal_class_horizon(physical, query.horizon_steps) {
            Some(bundle) => Ok(bundle),
            None => self.identify_temporal_class_within(identifier_id, query, budget, ctx),
        }
    }

    pub(crate) fn identify_temporal_class(
        &self,
        identifier_id: IdentifierId,
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<Arc<crate::analysis::prepared::CachedTemporalClassIdentification>, CausalError>
    {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        self.identify_temporal_class_within(identifier_id, query, &mut budget, ctx)
    }

    pub(crate) fn identify_temporal_class_within(
        &self,
        identifier_id: IdentifierId,
        query: &TemporalEffectQuery,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Arc<crate::analysis::prepared::CachedTemporalClassIdentification>, CausalError>
    {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        if let Some(cache) = self.temporal_class_cache() {
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
        let mut config = antecedent_identify::GeneralizedAdjustmentConfig {
            context: Some(ctx.clone()),
            temporal_budget: Some(budget.clone()),
            ..Default::default()
        };
        if let Some(max) = self.declared.max_completions {
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
        crate::identify_api::refine_temporal_class_identification_within(
            &mut envelope,
            &self.query,
            query.horizon_steps,
            budget,
            ctx,
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
    let limit = ctx.memory_limit_bytes();
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
    // Supported contract: `conditional_on_identified` is published only when
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

mod observation_bootstrap;

#[cfg(test)]
use observation_bootstrap::summarize_observation_bootstrap;
use observation_bootstrap::{
    apply_class_observation_bootstrap, apply_observation_bootstrap, apply_tuple_bootstrap_band,
    bootstrap_observation_adjusted_temporal_response, bootstrap_sequence_response,
    class_atom_adjustment, class_observation_band_note,
};

mod class_assembly;

pub(super) use class_assembly::{ClassResponseAssembly, ClassResponseParts, ClassResponseTail};

mod class_mixture;

use class_mixture::shared_temporal_class_uncertainty;
pub(super) use class_mixture::{
    ClassPosteriorMix, class_response_identified_set_diagnostics,
    enumeration_not_probability_diagnostic, mix_class_effect_posteriors,
};
pub(crate) use class_mixture::{
    aggregate_temporal_horizon_evidence, temporal_class_envelope_diagnostic,
    temporal_class_structural_mixture,
};

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
    let sequence = antecedent_estimate::requires_temporal_overlays(query);
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
    response.support.warnings.extend(adjusted.censoring_diagnostics());
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

#[derive(Clone)]
enum ClassObservationKind {
    Curve,
    /// Sequence overlay atom: refit through the unfolded sequential engine.
    Sequence {
        dag: Box<TemporalDag>,
        overlays: antecedent_estimate::AdmittedMechanismOverlays,
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

/// The tuple replicates, per-atom draws and the (attempted, cancelled) counts
/// the atom bands are summarized from.
type AtomBandSource<'a> = (&'a TupleReplicates, &'a [Vec<Vec<f64>>], (u32, bool));

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
            preparation: None,
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

mod cross_world_math;

use cross_world_math::cross_world_contrast;
pub(super) use cross_world_math::{
    CrossWorldContrast, cross_world_point, require_path_chain, require_path_nodes,
    template_scoped_cross_world, temporal_cross_world_hard_level,
};

mod derivative_math;

pub(super) use derivative_math::{
    DagDerivativeFit, combine_horizon_derivatives, temporal_dag_derivative_response,
    temporal_linear_derivative_response, temporal_linear_derivative_value,
};

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

/// Names the horizon whose refuter state produced the neighbouring reports.
fn horizon_note(horizon: u32, state: &str) -> Diagnostic {
    Diagnostic::new(
        "refute.temporal_response.horizon_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "horizon {horizon}: the {state} ran on this horizon's own window and estimand; \
             horizons are not mixed"
        ),
    )
    .with_fields([("horizon", horizon.to_string())])
}

mod class_fits;

use class_fits::{
    BayesianResponseFit, fit_class_bayesian_mediation, fit_class_bayesian_response,
    fit_class_frequentist_mediation, fit_class_frequentist_response, plan_class_bayesian_atoms,
    refute_class_mediation,
};
