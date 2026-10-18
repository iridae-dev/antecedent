//! TemporalCpdag/Pag graph-posterior mediation: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::atom_mix::{
    AtomBoundRule, AtomEvalView, AtomMixInput, AtomRefuterBatch, ScalarRefuterMix,
    drive_posterior_refuters, mix_atom_evals,
};
use super::{
    Arc, BayesianConfig, BayesianGCompWorkspace, CausalError, CausalPosterior, CausalQuery,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    GraphPosterior, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights, IdentifierId,
    InferenceMode, Instant, MediationRoute, OverlapPolicy, PhysicalExecutionPlan,
    PosteriorMassSummary, QueryRefutationPlan, RefuteSuite, Strategy, Study, StudyResult,
    TableView, TemporalEffectQuery, TemporalMediationEstimate, TemporalMediationEstimator,
    TimeSeriesData, bayesian_gcomp, effect_from_posterior, elapsed_ns,
    identification_status_ok_for_case, identify_cached_diagnostic, mediation_block_stream,
    mediation_mechanism_seed, provenance_ids,
    push_graph_posterior_structural_aggregation_diagnostic, resolve_posterior_status,
    select_estimand, short_series_warning,
};
use crate::analysis::prepared::{
    CachedTemporalClassPosteriorAtomIdentification, CachedTemporalClassPosteriorIdentification,
};
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;
use crate::result::{StructuralAggregationPolicy, StructuralResponseMixture};

struct TemporalClassMediationAtomEval {
    key: u64,
    status: IdentificationStatus,
    estimand: Option<IdentifiedEstimand>,
    mediation: Option<TemporalMediationEstimate>,
    prepared: Option<antecedent_estimate::PreparedTemporalMediation>,
    posterior: Option<CausalPosterior>,
    identified_set: Option<(f64, f64)>,
    partial: bool,
    refute_atom: Option<(
        IdentifiedEstimand,
        Arc<[antecedent_data::LaggedColumn]>,
        TemporalMediationEstimate,
    )>,
}

impl super::Study {
    /// The class posterior identification for a mediation horizon, built from
    /// the posterior at the horizon's pulse witness when the study carries no
    /// cache.
    fn class_mediation_posterior_identification(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        horizon: u32,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<CachedTemporalClassPosteriorIdentification>, bool), CausalError> {
        let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
        witness.horizon_steps = horizon;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        Ok(if let Some(cache) = self.temporal_class_posterior_identification_cache.as_ref() {
            (Arc::clone(cache), true)
        } else {
            (
                Arc::new(
                    crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
                        gp,
                        &vars,
                        &witness,
                        self.max_completions,
                        ctx,
                    )?,
                ),
                false,
            )
        })
    }

    /// Mix TemporalCpdag/Pag posterior atoms through the class mediation envelope.
    pub(super) fn execute_temporal_class_graph_posterior_mediation(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if query.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "Frequentist TemporalCpdag/Pag graph-posterior mediation is licensed \
                          for one horizon; multi-horizon grids need their own joint uncertainty \
                          contract",
            });
        }
        if self.inference.is_bayesian() {
            return self.execute_temporal_class_graph_posterior_mediation_bayesian(
                data, gp, query, physical, ctx,
            );
        }
        let started = Instant::now();
        let horizon = single_mediation_horizon(query)?;
        let (identified, identify_cached) =
            self.class_mediation_posterior_identification(data, gp, query, horizon, ctx)?;
        let evals = self.evaluate_class_mediation_atoms_frequentist(
            (data, query),
            horizon,
            &identified,
            ctx,
        )?;
        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_temporal_class_mediation_posterior(
            &identified,
            &evals,
            unidentified_mass,
            query,
            horizon,
        )?;
        let block = self.class_mediation_shared_block((&identified, &evals), query, horizon, ctx);
        let estimate = block.stamp(mixed.estimate);
        let mediation = class_mediation_with_effect(&estimate, mixed.mediation);
        let mut diagnostics = self.class_mediation_frequentist_diagnostics(
            mixed.identification.diagnostics.clone(),
            (mixed.policy, &mixed.mixture, mixed.posterior.is_none()),
            &block,
        );
        let (refutations, refute_diagnostics) = refute_temporal_class_mediation_posterior(
            self,
            data,
            query,
            &identified,
            &evals,
            mixed.policy,
            ctx,
        )?;
        diagnostics.extend(refute_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let uncertainty = block.uncertainty();
        let mediation_grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from([antecedent_estimate::TemporalMediationSlice {
                horizon,
                identification_status: mixed.identification.status,
                method: Arc::clone(&mixed.estimand.method),
                adjustment: mixed.adjustment,
                estimate: mediation.clone().unwrap_or_else(|| TemporalMediationEstimate {
                    effect: estimate.clone(),
                    total: None,
                    direct: None,
                    mediated: None,
                }),
                uncertainty,
                identified_set: mixed.slice_identified_set,
                diagnostics: Vec::new(),
            }]),
            joint_posterior: false,
        };
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                mixed.identification,
                mixed.estimand,
                estimate,
                Strategy::new(IdentifierId::Frontdoor, EstimatorId::TemporalMediation),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .mediation(mediation)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(block.replicates_requested.then_some(block.replicates_ok))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_class_graph_posterior_mediation",
                    mixed.policy.as_str(),
                )),
                structural_response: Some(mixed.mixture),
                mediation_grid: Some(mediation_grid),
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }

    /// Prepares each class atom's mediation designs on the context's workers.
    fn evaluate_class_mediation_atoms_frequentist(
        &self,
        (data, query): (&TimeSeriesData, &antecedent_core::MediationQuery),
        horizon: u32,
        identified: &CachedTemporalClassPosteriorIdentification,
        ctx: &ExecutionContext,
    ) -> Result<Vec<TemporalClassMediationAtomEval>, CausalError> {
        let estimator = TemporalMediationEstimator::new()
            .with_allow_natural_controlled_alias(true)
            .with_allow_gaps(self.declares_complete_case());
        ctx.map_indexed(identified.class_atoms.len(), |i, inner| {
            evaluate_temporal_class_mediation_atom_frequentist(
                self,
                data,
                query,
                horizon,
                &identified.class_atoms[i],
                &estimator,
                inner,
            )
        })
    }

    /// The shared circular-block bootstrap over the evaluated atoms' prepared
    /// mediation designs, frozen-weighted by posterior graph weight.
    fn class_mediation_shared_block(
        &self,
        (identified, evals): (
            &CachedTemporalClassPosteriorIdentification,
            &[TemporalClassMediationAtomEval],
        ),
        query: &antecedent_core::MediationQuery,
        horizon: u32,
        ctx: &ExecutionContext,
    ) -> ClassMediationBlock {
        let mut designs = Vec::new();
        let mut weights = Vec::new();
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        for eval in evals {
            let graph_w = key_weights.get(eval.key);
            if graph_w <= 0.0 {
                continue;
            }
            if let Some(prepared) = eval.prepared.as_ref() {
                designs.push(prepared);
                weights.push(graph_w);
            }
        }
        let refs: Vec<&antecedent_estimate::PreparedTemporalMediation> = designs.clone();
        let shared = antecedent_estimate::shared_mediation_block_bootstrap(
            &refs,
            &weights,
            query.contrast,
            self.bootstrap_replicates,
            mediation_block_stream(MediationRoute::TemporalClass, horizon),
            ctx,
        );
        let requested_se = shared.as_ref().and_then(|s| s.requested).filter(|s| s.is_finite());
        let replicates_requested = self.bootstrap_replicates > 0;
        let (replicates_ok, replicates_attempted) = shared
            .as_ref()
            .map_or((0, 0), |s| (s.block.replicates_ok, s.block.replicates_attempted));
        let family = shared.as_ref().map_or_else(
            || antecedent_estimate::CircularBlockFamily::for_mediation_atoms(designs.len()),
            |s| s.family,
        );
        ClassMediationBlock {
            shared,
            requested_se,
            replicates_requested,
            replicates_ok,
            replicates_attempted,
            family,
        }
    }

    /// The diagnostics of a frequentist class graph-posterior mediation before
    /// refutation: the withheld posterior, the evaluation and aggregation
    /// notes, and the shared-block or withheld-uncertainty note.
    fn class_mediation_frequentist_diagnostics(
        &self,
        mut diagnostics: Vec<Diagnostic>,
        (policy, mixture, posterior_withheld): (
            StructuralAggregationPolicy,
            &StructuralResponseMixture,
            bool,
        ),
        block: &ClassMediationBlock,
    ) -> Vec<Diagnostic> {
        if self.inference.is_bayesian() && posterior_withheld {
            diagnostics.push(graph_posterior_withheld_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.temporal_class_mediation_envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each TemporalCpdag/Pag posterior atom was evaluated with the existing temporal class \
             mediation envelope; outer weights are posterior probabilities; completion enumeration \
             is not posterior probability",
        ));
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            policy,
            mixture.identified_mass,
            mixture.unidentified_mass,
            mixture.unevaluable_mass,
            0.0,
        );
        match block.shared.as_ref() {
            Some(s) if block.replicates_requested && block.requested_se.is_some() => {
                diagnostics.push(Diagnostic::new(
                    "estimate.temporal_class.mediation.shared_block",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "shared circular-block replicates={}; attempted={}; each posterior atom \
                         keeps its own S(h); between-atom covariance included in the aggregate SE",
                        s.block.replicates_ok, s.block.replicates_attempted,
                    ),
                ));
                diagnostics.extend(short_series_warning(s.block.effective_rows, block.family));
            }
            _ => diagnostics.push(Diagnostic::new(
                "estimate.temporal_class.mediation.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if block.replicates_requested {
                    "the shared circular-block bootstrap did not produce a usable SE; the \
                     aggregate SE is withheld and iid analytic SEs are not substituted"
                } else {
                    "no bootstrap replicates were requested; lagged rows are serially dependent, \
                     so no iid analytic SE is published for the aggregate"
                },
            )),
        }
        diagnostics
    }

    fn execute_temporal_class_graph_posterior_mediation_bayesian(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use antecedent_estimate::bayesian_mediation::require_gaussian_mediation;
        let InferenceMode::Bayesian(cfg) = &self.inference else {
            return Err(CausalError::Unsupported {
                message: "Bayesian TemporalCpdag/Pag graph-posterior mediation requires Bayesian inference",
            });
        };
        let cfg = cfg.clone();
        let estimator = bayesian_gcomp(&cfg, ctx);
        require_gaussian_mediation(&estimator).map_err(CausalError::from)?;
        let started = Instant::now();
        let horizon = single_mediation_horizon(query)?;
        let (identified, identify_cached) =
            self.class_mediation_posterior_identification(data, gp, query, horizon, ctx)?;
        let evals = ctx.map_indexed(identified.class_atoms.len(), |i, inner| {
            evaluate_temporal_class_mediation_atom_bayesian(
                self,
                data,
                query,
                horizon,
                &identified.class_atoms[i],
                &cfg,
                inner,
            )
        })?;
        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_temporal_class_mediation_posterior(
            &identified,
            &evals,
            unidentified_mass,
            query,
            horizon,
        )?;
        let mut diagnostics = mixed.identification.diagnostics.clone();
        if self.inference.is_bayesian() && mixed.posterior.is_none() {
            diagnostics.push(graph_posterior_withheld_diagnostic());
        }
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.temporal_class_mediation_envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "each TemporalCpdag/Pag posterior atom was evaluated with the existing temporal class \
             Bayesian mediation envelope; outer weights are posterior probabilities",
        ));
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            mixed.policy,
            mixed.mixture.identified_mass,
            mixed.mixture.unidentified_mass,
            mixed.mixture.unevaluable_mass,
            0.0,
        );
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let (refutations, refute_diagnostics) = refute_temporal_class_mediation_posterior(
            self,
            data,
            query,
            &identified,
            &evals,
            mixed.policy,
            ctx,
        )?;
        diagnostics.extend(refute_diagnostics);
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                mixed.identification,
                mixed.estimand,
                mixed.estimate,
                Strategy::new(IdentifierId::Frontdoor, EstimatorId::BayesianTemporalMediation),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .mediation(mixed.mediation)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_class_graph_posterior_mediation",
                    mixed.policy.as_str(),
                )),
                posterior: mixed.posterior,
                structural_response: Some(mixed.mixture),
                mediation_grid: mixed.mediation_grid,
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }
}

/// The shared circular-block bootstrap of a class graph-posterior mediation
/// and the replicate accounting its estimate is stamped with.
struct ClassMediationBlock {
    shared: Option<antecedent_estimate::SharedMediationBlockSe>,
    requested_se: Option<f64>,
    replicates_requested: bool,
    replicates_ok: u32,
    replicates_attempted: u32,
    family: antecedent_estimate::CircularBlockFamily,
}

/// The mediation estimate carrying the stamped effect with the mixed Total,
/// Direct and Mediated parts.
fn class_mediation_with_effect(
    effect: &EffectEstimate,
    mixed: Option<TemporalMediationEstimate>,
) -> Option<TemporalMediationEstimate> {
    mixed.map(|m| TemporalMediationEstimate {
        effect: effect.clone(),
        total: m.total,
        direct: m.direct,
        mediated: m.mediated,
    })
}

impl ClassMediationBlock {
    /// Stamps the mixed estimate with the block family and bootstrap
    /// accounting; the iid analytic SE is not published.
    fn stamp(&self, estimate: EffectEstimate) -> EffectEstimate {
        let mut estimate = estimate.with_block_family(self.family);
        estimate.uncertainty.analytic.se = f64::NAN;
        estimate.uncertainty.bootstrap.se = self.requested_se;
        estimate.uncertainty.bootstrap.replicates_ok =
            self.replicates_requested.then_some(self.replicates_ok);
        estimate.uncertainty.bootstrap.replicates_failed = self
            .replicates_requested
            .then_some(self.replicates_attempted.saturating_sub(self.replicates_ok));
        estimate
    }

    /// The slice uncertainty: the shared block bootstrap when requested.
    fn uncertainty(&self) -> antecedent_estimate::TemporalMediationUncertainty {
        match self.shared.clone() {
            Some(s) if self.replicates_requested => {
                antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                    requested: s.requested,
                    block: s.block,
                }
            }
            _ => antecedent_estimate::TemporalMediationUncertainty::Unavailable,
        }
    }
}

struct MixedTemporalClassMediationPosterior {
    policy: StructuralAggregationPolicy,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    mediation: Option<TemporalMediationEstimate>,
    posterior: Option<CausalPosterior>,
    mixture: StructuralResponseMixture,
    adjustment: Arc<[antecedent_core::TemporalNodeKey]>,
    slice_identified_set: Option<antecedent_estimate::TemporalMediationIdentifiedSet>,
    mediation_grid: Option<antecedent_estimate::TemporalMediationGrid>,
}

/// The lagged adjustment columns of a completion's mediation estimand, read
/// relative to the outcome time.
fn class_case_lagged_adjustment(
    estimand: &IdentifiedEstimand,
    indexer: &antecedent_core::TemporalIndexer,
    outcome_offset: i32,
) -> Arc<[antecedent_data::LaggedColumn]> {
    estimand
        .adjustment_set
        .iter()
        .filter_map(|dense| {
            let key = indexer.key_of(dense.raw()).ok()?;
            super::temporal_path::lagged_column_relative_to_outcome(key, outcome_offset)
        })
        .collect::<Vec<_>>()
        .into()
}

/// One completion's composed Bayesian mediation posterior: each mechanism is
/// fitted on its own seed stream and the pair composed.
fn fit_class_mediation_case_posterior(
    bayes: &antecedent_estimate::BayesianGComputationAte,
    (data, horizon_query, horizon): (&TimeSeriesData, &antecedent_core::MediationQuery, u32),
    (estimand, adjustment, status): (
        &IdentifiedEstimand,
        &Arc<[antecedent_data::LaggedColumn]>,
        IdentificationStatus,
    ),
    ctx: &ExecutionContext,
) -> Result<CausalPosterior, CausalError> {
    use antecedent_estimate::bayesian_mediation::{
        compose_temporal_mediation, prepare_temporal_mediation_adjusted,
    };
    let preparations =
        prepare_temporal_mediation_adjusted(data, estimand, horizon_query, adjustment, ctx)
            .map_err(CausalError::from)?;
    let mechanisms = preparations
        .iter()
        .enumerate()
        .map(|(index, prep)| {
            let mut mechanism = bayes.clone();
            mechanism.seed = mediation_mechanism_seed(mechanism.seed, horizon, index);
            mechanism
                .fit(prep, status, &mut BayesianGCompWorkspace::default(), ctx)
                .map_err(CausalError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [natural_direct, natural_indirect] = mechanisms.as_slice() else {
        return Err(CausalError::Compile {
            message: "temporal mediation must fit exactly two mechanisms".into(),
        });
    };
    compose_temporal_mediation(natural_direct, natural_indirect, horizon_query, status)
        .map_err(CausalError::from)
}

/// The one requested horizon of a class graph-posterior mediation.
fn single_mediation_horizon(query: &antecedent_core::MediationQuery) -> Result<u32, CausalError> {
    match query.horizons.as_ref() {
        [horizon] => Ok(*horizon),
        _ => Err(CausalError::Unsupported {
            message: "TemporalCpdag/Pag graph-posterior mediation is licensed for one horizon",
        }),
    }
}

fn evaluate_temporal_class_mediation_atom_frequentist(
    _study: &Study,
    data: &TimeSeriesData,
    query: &antecedent_core::MediationQuery,
    horizon: u32,
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    estimator: &TemporalMediationEstimator,
    ctx: &ExecutionContext,
) -> Result<TemporalClassMediationAtomEval, CausalError> {
    let mut class_envelope = antecedent_identify::TemporalClassEnvelope::clone(&atom.envelope);
    crate::identify_api::refine_temporal_class_identification(
        &mut class_envelope,
        &CausalQuery::Mediation(query.clone()),
        horizon,
    )?;
    let envelope = &class_envelope.envelope;
    let mut horizon_query = query.clone();
    horizon_query.horizons = Arc::from([horizon]);
    let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
    let mut effects = Vec::new();
    let mut shared_design: Option<(IdentifiedEstimand, Arc<[antecedent_data::LaggedColumn]>)> =
        None;
    let mut designs_agree = true;
    let mut prepared = None;
    let mut refute_atom = None;
    let mut primary_estimand = None;
    for (case, indexer) in envelope.cases.iter().zip(class_envelope.indexers.iter()) {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalMediation)?;
        let adjustment = class_case_lagged_adjustment(&estimand, indexer, outcome_offset);
        let prep = estimator
            .prepare_shared(data, &estimand, &horizon_query, &adjustment, ctx)
            .map_err(CausalError::from)?;
        if !prep.estimate().effect.ate.is_finite() {
            continue;
        }
        effects.push(prep.estimate().effect.ate);
        match shared_design.as_ref() {
            None => {
                shared_design = Some((estimand.clone(), adjustment.clone()));
                let estimate = prep.estimate().clone();
                prepared = Some(prep);
                primary_estimand = Some(estimand.clone());
                refute_atom = Some((estimand, adjustment, estimate));
            }
            Some((first, first_adj)) => {
                designs_agree &= first.adjustment_set == estimand.adjustment_set
                    && first_adj.as_ref() == adjustment.as_ref();
            }
        }
    }
    let lo = effects.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = effects.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scalar = if designs_agree
        && effects.len()
            == envelope
                .cases
                .iter()
                .filter(|c| {
                    identification_status_ok_for_case(c.result.status)
                        && !c.result.estimands.is_empty()
                })
                .count()
        && lo.is_finite()
        && hi.is_finite()
        && lo.total_cmp(&hi) == std::cmp::Ordering::Equal
    {
        Some(lo)
    } else {
        None
    };
    let mediation = scalar.zip(prepared.as_ref()).map(|(ate, prep)| TemporalMediationEstimate {
        effect: EffectEstimate::new(
            ate,
            f64::NAN,
            prep.estimate().effect.provenance.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        ),
        total: prep.estimate().total,
        direct: prep.estimate().direct,
        mediated: prep.estimate().mediated,
    });
    Ok(TemporalClassMediationAtomEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or(primary_estimand),
        mediation,
        prepared: if scalar.is_some() { prepared } else { None },
        posterior: None,
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atom,
    })
}

fn evaluate_temporal_class_mediation_atom_bayesian(
    _study: &Study,
    data: &TimeSeriesData,
    query: &antecedent_core::MediationQuery,
    horizon: u32,
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    cfg: &BayesianConfig,
    ctx: &ExecutionContext,
) -> Result<TemporalClassMediationAtomEval, CausalError> {
    let bayes = bayesian_gcomp(cfg, ctx);
    let mut class_envelope = antecedent_identify::TemporalClassEnvelope::clone(&atom.envelope);
    crate::identify_api::refine_temporal_class_identification(
        &mut class_envelope,
        &CausalQuery::Mediation(query.clone()),
        horizon,
    )?;
    let envelope = &class_envelope.envelope;
    let mut horizon_query = query.clone();
    horizon_query.horizons = Arc::from([horizon]);
    let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
    let mut weighted = 0.0;
    let mut total = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut primary_estimand = None;
    let mut atom_posterior = None;
    let mut refute_atom = None;
    for (case, indexer) in envelope.cases.iter().zip(atom.envelope.indexers.iter()) {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalMediation)?;
        let adjustment = class_case_lagged_adjustment(&estimand, indexer, outcome_offset);
        let composed = fit_class_mediation_case_posterior(
            &bayes,
            (data, &horizon_query, horizon),
            (&estimand, &adjustment, case.result.status),
            ctx,
        )?;
        let mean = composed.summaries.mean.first().copied().unwrap_or(f64::NAN);
        if mean.is_finite() {
            weighted += case.weight.0 * mean;
            total += case.weight.0;
            lo = lo.min(mean);
            hi = hi.max(mean);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                atom_posterior = Some(composed.clone());
                refute_atom = Some((
                    estimand,
                    adjustment,
                    TemporalMediationEstimate {
                        effect: EffectEstimate::new(
                            mean,
                            f64::NAN,
                            effect_from_posterior(&composed)?.provenance.assumptions,
                            OverlapPolicy::ExplicitOverride,
                        ),
                        total: None,
                        direct: None,
                        mediated: None,
                    },
                ));
            }
        }
    }
    let mediation = match atom_posterior.as_ref().filter(|_| total > 0.0) {
        Some(posterior) => Some(TemporalMediationEstimate {
            effect: EffectEstimate::new(
                weighted / total,
                f64::NAN,
                effect_from_posterior(posterior)?.provenance.assumptions,
                OverlapPolicy::ExplicitOverride,
            ),
            total: None,
            direct: None,
            mediated: None,
        }),
        None => None,
    };
    Ok(TemporalClassMediationAtomEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or(primary_estimand),
        mediation,
        prepared: None,
        posterior: if atom.envelope.envelope.cases.len() == 1 { atom_posterior } else { None },
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atom,
    })
}

impl AtomEvalView for TemporalClassMediationAtomEval {
    fn key(&self) -> u64 {
        self.key
    }

    fn status(&self) -> IdentificationStatus {
        self.status
    }

    fn estimand(&self) -> Option<&IdentifiedEstimand> {
        self.estimand.as_ref()
    }

    fn scalar(&self) -> Option<f64> {
        self.mediation
            .as_ref()
            .and_then(|mediation| mediation.effect.ate.is_finite().then_some(mediation.effect.ate))
    }

    fn posterior(&self) -> Option<&CausalPosterior> {
        self.posterior.as_ref()
    }

    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
        self.mediation.as_ref().map(|mediation| &mediation.effect.provenance.assumptions)
    }

    fn identified_bounds(&self) -> Option<(f64, f64)> {
        self.identified_set
    }

    fn partial(&self) -> bool {
        self.partial
    }

    fn analytic_se(&self) -> Option<f64> {
        None
    }

    fn influence(&self) -> Option<&[f64]> {
        None
    }
}

fn mix_temporal_class_mediation_posterior(
    identified: &CachedTemporalClassPosteriorIdentification,
    evals: &[TemporalClassMediationAtomEval],
    unidentified_mass: f64,
    _query: &antecedent_core::MediationQuery,
    _horizon: u32,
) -> Result<MixedTemporalClassMediationPosterior, CausalError> {
    let mixed = mix_atom_evals(
        evals,
        AtomMixInput {
            graphs: &identified.graphs,
            proofs: &identified.class_atoms,
            unidentified_mass,
            full_mass_scope: identified
                .class_atoms
                .iter()
                .all(|atom| atom.truncated_completions == 0),
            truncated_atoms: identified
                .class_atoms
                .iter()
                .map(|atom| atom.truncated_completions)
                .sum::<usize>(),
            bounds: AtomBoundRule::SetAlways,
            estimand_fallback: false,
        },
        |_n, _mixable| {
            "temporal class graph-posterior mediation: no evaluable atom values".to_string()
        },
        || "temporal class graph-posterior mediation: missing estimand".to_string(),
    )?;
    let mut identification = mixed.identification;
    identification.status = resolve_posterior_status(
        identification.status,
        mixed.policy,
        PosteriorMassSummary {
            unidentified: unidentified_mass,
            unevaluable: mixed.mixture.unevaluable_mass,
            full_scope: mixed.mixture.full_mass_scope,
            ..PosteriorMassSummary::default()
        },
    );
    let slice_identified_set = (mixed.bound_lo.is_finite() && mixed.bound_hi.is_finite())
        .then_some(antecedent_estimate::TemporalMediationIdentifiedSet {
            lower: mixed.bound_lo,
            upper: mixed.bound_hi,
        });
    let mediation = mixed.mixable_scalar.then(|| TemporalMediationEstimate {
        effect: mixed.estimate.clone(),
        total: None,
        direct: None,
        mediated: None,
    });
    Ok(MixedTemporalClassMediationPosterior {
        policy: mixed.policy,
        identification,
        estimand: mixed.estimand,
        estimate: mixed.estimate,
        mediation,
        posterior: mixed.posterior,
        mixture: mixed.mixture,
        adjustment: Arc::<[antecedent_core::TemporalNodeKey]>::from([]),
        slice_identified_set,
        mediation_grid: None,
    })
}

fn refute_temporal_class_mediation_posterior(
    study: &Study,
    data: &TimeSeriesData,
    query: &antecedent_core::MediationQuery,
    identified: &CachedTemporalClassPosteriorIdentification,
    evals: &[TemporalClassMediationAtomEval],
    policy: StructuralAggregationPolicy,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    if study.refute == RefuteSuite::None {
        return Ok((Vec::new(), Vec::new()));
    }
    let mix_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean);
    let plan = QueryRefutationPlan::temporal_mediation(study.refute == RefuteSuite::Full);
    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    drive_posterior_refuters(
        evals,
        mix_scalar,
        ScalarRefuterMix::Mediation,
        |eval| eval.refute_atom.is_some(),
        |eval| eval.key,
        |eval| {
            let weight = key_weights.get(eval.key);
            if weight <= 0.0 {
                return Ok(Vec::new());
            }
            let Some((estimand, adjustment, estimate)) = eval.refute_atom.as_ref() else {
                return Ok(Vec::new());
            };
            let mut horizon_query = query.clone();
            horizon_query.horizons = Arc::from([single_mediation_horizon(query)?]);
            let reports = plan
                .refute_temporal_atom(data, estimand, &horizon_query, estimate, adjustment, ctx)
                .map_err(CausalError::from)?;
            Ok(vec![AtomRefuterBatch { weight, reports, notes: Vec::new() }])
        },
        |inner_keys, mix_scalar| {
            Diagnostic::new(
                "refute.envelope.temporal_class_mediation_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                if mix_scalar {
                    format!(
                        "mediation refuters evaluated each posterior atom [{inner_keys}] against that \
                         atom's own contrast; outer reports mix by frozen posterior graph weight"
                    )
                } else {
                    format!(
                        "mediation refuters evaluated each posterior atom [{inner_keys}] against that \
                         atom's own contrast; outer scalar mix skipped because StructuralAggregationPolicy \
                         is {} — per-atom reports are retained",
                        policy.as_str()
                    )
                },
            )
        },
    )
}
