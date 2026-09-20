//! TemporalCpdag/Pag graph-posterior mediation: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::prepared::{
    CachedTemporalClassPosteriorAtomIdentification, CachedTemporalClassPosteriorIdentification,
};
use crate::result::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
};

const TEMPORAL_CLASS_MEDIATION_BLOCK_STREAM: u64 = 0x7C60_0001_0000;

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
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            return self.execute_temporal_class_graph_posterior_mediation_bayesian(
                data, gp, query, physical, ctx,
            );
        }
        let started = Instant::now();
        let horizon = query.horizons[0];
        let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
        witness.horizon_steps = horizon;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let (identified, identify_cached) =
            if let Some(cache) = self.temporal_class_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
                        gp,
                        &vars,
                        &witness,
                        self.max_completions,
                        ctx,
                    )?,
                    false,
                )
            };
        let estimator = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
        let evals = ctx.map_indexed(identified.class_atoms.len(), |i, inner| {
            evaluate_temporal_class_mediation_atom_frequentist(
                self,
                data,
                query,
                horizon,
                &identified.class_atoms[i],
                &estimator,
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
        let mut designs = Vec::new();
        let mut weights = Vec::new();
        for eval in &evals {
            let graph_w = identified_weight_for_key(&identified.graphs, eval.key);
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
            TEMPORAL_CLASS_MEDIATION_BLOCK_STREAM.wrapping_add(u64::from(horizon) << 32),
            ctx,
        );
        let requested_se = shared.as_ref().and_then(|s| s.requested).filter(|s| s.is_finite());
        let replicates_requested = self.bootstrap_replicates > 0;
        let (replicates_ok, replicates_attempted) = shared
            .as_ref()
            .map_or((0, 0), |s| (s.block.replicates_ok, s.block.replicates_attempted));
        let mut estimate = mixed.estimate;
        estimate.se_analytic = f64::NAN;
        estimate.se_bootstrap = requested_se;
        estimate.bootstrap_replicates_ok = replicates_requested.then_some(replicates_ok);
        estimate.bootstrap_replicates_failed =
            replicates_requested.then_some(replicates_attempted.saturating_sub(replicates_ok));
        let mediation = mixed.mediation.map(|m| TemporalMediationEstimate {
            effect: estimate.clone(),
            total: m.total,
            direct: m.direct,
            mediated: m.mediated,
        });
        let mut diagnostics = mixed.identification.diagnostics.clone();
        if matches!(self.inference, InferenceMode::Bayesian(_)) && mixed.posterior.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.graph_posterior.posterior_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "aggregate posterior withheld: a graph- or completion-conditional posterior \
                 does not describe the full weighted target; retained atom posteriors are conditional",
            ));
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
            mixed.policy,
            mixed.mixture.identified_mass,
            mixed.mixture.unidentified_mass,
            mixed.mixture.unevaluable_mass,
            0.0,
        );
        let family = if designs.len() == 1 {
            antecedent_estimate::CircularBlockFamily::Mediation
        } else {
            antecedent_estimate::CircularBlockFamily::Mixture
        };
        match shared.as_ref() {
            Some(s) if replicates_requested && requested_se.is_some() => {
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
                diagnostics.extend(short_series_warning(s.block.effective_rows, family));
            }
            _ => diagnostics.push(Diagnostic::new(
                "estimate.temporal_class.mediation.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if replicates_requested {
                    "the shared circular-block bootstrap did not produce a usable SE; the \
                     aggregate SE is withheld and iid analytic SEs are not substituted"
                } else {
                    "no bootstrap replicates were requested; lagged rows are serially dependent, \
                     so no iid analytic SE is published for the aggregate"
                },
            )),
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
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let uncertainty = match shared {
            Some(s) if replicates_requested => {
                antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                    requested: s.requested,
                    block: s.block,
                }
            }
            _ => antecedent_estimate::TemporalMediationUncertainty::Unavailable,
        };
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
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification: mixed.identification,
            estimand: mixed.estimand,
            estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::TemporalMediation,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: replicates_requested.then_some(replicates_ok),
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_class_graph_posterior_mediation",
                    mixed.policy.as_str(),
                )),
                structural_response: Some(mixed.mixture),
                mediation_grid: Some(mediation_grid),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
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
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => unreachable!(),
        };
        let estimator = bayesian_gcomp(&cfg, ctx);
        require_gaussian_mediation(&estimator).map_err(CausalError::from)?;
        let started = Instant::now();
        let horizon = query.horizons[0];
        let mut witness = TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
        witness.horizon_steps = horizon;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let (identified, identify_cached) =
            if let Some(cache) = self.temporal_class_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
                        gp,
                        &vars,
                        &witness,
                        self.max_completions,
                        ctx,
                    )?,
                    false,
                )
            };
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
        if matches!(self.inference, InferenceMode::Bayesian(_)) && mixed.posterior.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.graph_posterior.posterior_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "aggregate posterior withheld: a graph- or completion-conditional posterior \
                 does not describe the full weighted target; retained atom posteriors are conditional",
            ));
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
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification: mixed.identification,
            estimand: mixed.estimand,
            estimate: mixed.estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::BayesianTemporalMediation,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: mixed.mediation,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
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
            },
        }))
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

fn evaluate_temporal_class_mediation_atom_frequentist(
    _study: &Study,
    data: &TimeSeriesData,
    query: &antecedent_core::MediationQuery,
    horizon: u32,
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    estimator: &TemporalMediationEstimator,
    ctx: &ExecutionContext,
) -> Result<TemporalClassMediationAtomEval, CausalError> {
    let mut class_envelope = atom.envelope.clone();
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
        let adjustment: Arc<[antecedent_data::LaggedColumn]> = estimand
            .adjustment_set
            .iter()
            .filter_map(|dense| {
                let key = indexer.key_of(dense.raw()).ok()?;
                super::temporal_path::lagged_column_relative_to_outcome(key, outcome_offset)
            })
            .collect::<Vec<_>>()
            .into();
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
    let mediation = scalar.map(|ate| {
        let prep = prepared.as_ref().expect("scalar mediation");
        TemporalMediationEstimate {
            effect: EffectEstimate::new(
                ate,
                f64::NAN,
                prep.estimate().effect.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            ),
            total: prep.estimate().total,
            direct: prep.estimate().direct,
            mediated: prep.estimate().mediated,
        }
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
    use antecedent_estimate::bayesian_mediation::{
        compose_temporal_mediation, prepare_temporal_mediation_adjusted,
    };
    let bayes = bayesian_gcomp(cfg, ctx);
    let mut class_envelope = atom.envelope.clone();
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
        let adjustment: Arc<[antecedent_data::LaggedColumn]> = estimand
            .adjustment_set
            .iter()
            .filter_map(|dense| {
                let key = indexer.key_of(dense.raw()).ok()?;
                super::temporal_path::lagged_column_relative_to_outcome(key, outcome_offset)
            })
            .collect::<Vec<_>>()
            .into();
        let preparations =
            prepare_temporal_mediation_adjusted(data, &estimand, &horizon_query, &adjustment, ctx)
                .map_err(CausalError::from)?;
        let mechanisms = preparations
            .iter()
            .enumerate()
            .map(|(index, prep)| {
                let mut mechanism = bayes.clone();
                mechanism.seed = mechanism.seed.wrapping_add(if index == 0 { 0 } else { 0xBA71 });
                mechanism
                    .fit(prep, case.result.status, &mut BayesianGCompWorkspace::default(), ctx)
                    .map_err(CausalError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let composed = compose_temporal_mediation(
            &mechanisms[0],
            &mechanisms[1],
            &horizon_query,
            case.result.status,
        )
        .map_err(CausalError::from)?;
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
                            effect_from_posterior(&composed)?.assumptions,
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
    let mediation = if total > 0.0 {
        let posterior = atom_posterior.as_ref().expect("bayesian atom posterior");
        Some(TemporalMediationEstimate {
            effect: EffectEstimate::new(
                weighted / total,
                f64::NAN,
                effect_from_posterior(posterior)?.assumptions,
                OverlapPolicy::ExplicitOverride,
            ),
            total: None,
            direct: None,
            mediated: None,
        })
    } else {
        None
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

fn mix_temporal_class_mediation_posterior(
    identified: &CachedTemporalClassPosteriorIdentification,
    evals: &[TemporalClassMediationAtomEval],
    unidentified_mass: f64,
    _query: &antecedent_core::MediationQuery,
    _horizon: u32,
) -> Result<MixedTemporalClassMediationPosterior, CausalError> {
    let mut atoms = Vec::new();
    let mut identified_weight = 0.0;
    let mut unevaluable_weight = 0.0;
    let mut mixable = 0.0;
    let mut weighted = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut contributing = Vec::new();
    let mut any_partial = false;
    let mut primary_identification = None;
    let mut primary_estimand = None;
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut atom_posteriors: Vec<(f64, CausalPosterior)> = Vec::new();
    let primary_adjustment: Arc<[antecedent_core::TemporalNodeKey]> = Arc::from([]);

    for eval in evals {
        let w = identified_weight_for_key(&identified.graphs, eval.key);
        let value =
            eval.mediation.as_ref().and_then(|m| m.effect.ate.is_finite().then_some(m.effect.ate));
        if let Some((a, b)) = eval.identified_set {
            lo = lo.min(a);
            hi = hi.max(b);
        }
        if value.is_some() {
            identified_weight += w;
            if let Some(estimand) = eval.estimand.as_ref() {
                contributing.push(estimand.clone());
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                    primary_identification = Some(eval_identification_mediation(eval, identified));
                    if let Some(m) = eval.mediation.as_ref() {
                        assumptions = m.effect.assumptions.clone();
                    }
                }
            }
            any_partial |= eval.partial;
            if let Some(p) = eval.posterior.clone() {
                atom_posteriors.push((w, p));
            }
            if let Some(v) = value {
                mixable += w;
                weighted += w * v;
            }
        } else if w > 0.0 {
            unevaluable_weight += w;
        }
        atoms.push(StructuralResponseAtom {
            graph_key: eval.key,
            weight: w,
            status: eval.status,
            value: value.map(antecedent_core::ResponseValue::Scalar),
            posterior: eval.posterior.clone(),
            response: None,
        });
    }

    for (key, weight, flag) in identified
        .graphs
        .graph_keys
        .iter()
        .zip(identified.graphs.weights.iter())
        .zip(identified.graphs.identified.iter())
        .map(|((k, w), f)| (*k, *w, *f))
    {
        if flag != GraphIdentFlag::Unidentified {
            continue;
        }
        if atoms.iter().any(|atom| atom.graph_key == key) {
            continue;
        }
        atoms.push(StructuralResponseAtom {
            graph_key: key,
            weight,
            status: IdentificationStatus::NotIdentified,
            value: None,
            posterior: None,
            response: None,
        });
    }

    let total = identified.graphs.total_weight();
    let identified_mass = if total > 0.0 { identified_weight / total } else { 0.0 };
    let unevaluable_mass = if total > 0.0 { unevaluable_weight / total } else { 0.0 };
    let refs: Vec<&IdentifiedEstimand> = contributing.iter().collect();
    let policy = if refs.is_empty() && any_partial {
        StructuralAggregationPolicy::IdentifiedSetEnvelope
    } else {
        resolve_structural_aggregation(&refs, any_partial)
    };
    let mixable_scalar =
        matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean) && mixable > 0.0;
    let ate = if mixable_scalar { weighted / mixable } else { f64::NAN };
    let conditional_on_identified =
        mixable_scalar.then_some(antecedent_core::ResponseValue::Scalar(ate));
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    let mixture = StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass,
        unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass: 0.0,
        identified_set,
        identified_set_interval: None,
        conditional_on_identified,
        full_mass_scope: identified.class_atoms.iter().all(|a| a.truncated_completions == 0),
        truncated_atoms: identified
            .class_atoms
            .iter()
            .map(|a| a.truncated_completions)
            .sum::<usize>(),
    };
    let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: "temporal class graph-posterior mediation: no evaluable atom values".into(),
    })?;
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "temporal class graph-posterior mediation: missing estimand".into(),
    })?;
    let estimate = EffectEstimate::new(ate, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride);
    if unidentified_mass > 0.0
        || unevaluable_mass > 0.0
        || !mixture.full_mass_scope
        || matches!(policy, StructuralAggregationPolicy::GraphDependentAtoms)
    {
        identification.status = IdentificationStatus::GraphDependent;
    }
    let slice_identified_set = if lo.is_finite() && hi.is_finite() {
        Some(antecedent_estimate::TemporalMediationIdentifiedSet { lower: lo, upper: hi })
    } else {
        None
    };
    // A component posterior is not the posterior of a weighted aggregate.
    // Retain it at the top level only when it covers the entire target.
    let posterior = if mixable_scalar
        && contributing.len() == 1
        && atom_posteriors.len() == 1
        && unidentified_mass == 0.0
        && unevaluable_mass == 0.0
        && mixture.full_mass_scope
    {
        atom_posteriors.into_iter().next().map(|(_, p)| p)
    } else {
        None
    };
    let mediation = mixable_scalar.then(|| TemporalMediationEstimate {
        effect: estimate.clone(),
        total: None,
        direct: None,
        mediated: None,
    });
    Ok(MixedTemporalClassMediationPosterior {
        policy,
        identification,
        estimand,
        estimate,
        mediation,
        posterior,
        mixture,
        adjustment: primary_adjustment,
        slice_identified_set,
        mediation_grid: None,
    })
}

fn eval_identification_mediation(
    eval: &TemporalClassMediationAtomEval,
    identified: &CachedTemporalClassPosteriorIdentification,
) -> IdentificationResult {
    identified.class_atoms.iter().find(|atom| atom.key == eval.key).map_or_else(
        || identified.class_atoms[0].identification.clone(),
        |atom| atom.identification.clone(),
    )
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
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    for eval in evals {
        let weight = identified_weight_for_key(&identified.graphs, eval.key);
        if weight <= 0.0 {
            continue;
        }
        let Some((estimand, adjustment, estimate)) = eval.refute_atom.as_ref() else {
            continue;
        };
        let mut horizon_query = query.clone();
        horizon_query.horizons = Arc::from([query.horizons[0]]);
        let reports = plan
            .refute_temporal_atom(data, estimand, &horizon_query, estimate, adjustment, ctx)
            .map_err(CausalError::from)?;
        per_atom.push((weight, reports));
    }
    let inner_keys: String = evals
        .iter()
        .filter(|eval| eval.refute_atom.is_some())
        .map(|eval| format!("{:x}", eval.key))
        .collect::<Vec<_>>()
        .join(",");
    diagnostics.push(Diagnostic::new(
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
    ));
    let reports = if mix_scalar {
        QueryRefutationPlan::mix_weighted(per_atom)
    } else {
        per_atom.into_iter().flat_map(|(_, reports)| reports).collect()
    };
    Ok((reports, diagnostics))
}
