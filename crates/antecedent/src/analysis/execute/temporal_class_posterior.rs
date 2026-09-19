//! TemporalCpdag/Pag graph-posterior Pulse/Sustained: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::prepared::{
    CachedTemporalClassPosteriorAtomIdentification, CachedTemporalClassPosteriorIdentification,
};
use crate::result::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
};

struct TemporalClassFittedDesign {
    completion_weight: f64,
    design: TemporalAtomDesign,
    indexer: TemporalIndexer,
}

struct TemporalClassAtomEval {
    key: u64,
    status: IdentificationStatus,
    estimand: Option<IdentifiedEstimand>,
    estimate: Option<EffectEstimate>,
    posterior: Option<CausalPosterior>,
    identified_set: Option<(f64, f64)>,
    partial: bool,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
    sequential_atoms: Vec<super::sequential_validation::SequentialValidationAtom>,
    fitted_designs: Vec<TemporalClassFittedDesign>,
}

impl super::Study {
    /// Mix TemporalCpdag/Pag posterior atoms through the existing class envelope.
    pub(super) fn execute_temporal_class_graph_posterior(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if is_multi_step_sustained(query) && self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "class-aware multi-step sustained requires no discovery-estimation split",
            });
        }
        let class_tag = match gp.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => "temporal_cpdag",
            antecedent_discovery::GraphPosteriorAtomKind::Pag => "temporal_pag",
            _ => {
                return Err(CausalError::Compile {
                    message: "temporal class graph-posterior requires Cpdag or Pag atom_kind"
                        .into(),
                });
            }
        };
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let (identified, identify_cached) =
            if let Some(cache) = self.temporal_class_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
                        gp,
                        &vars,
                        query,
                        self.max_completions,
                        ctx,
                    )?,
                    false,
                )
            };
        if identified.class_atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "temporal class graph-posterior envelope: no identified class atoms"
                    .into(),
            });
        }

        let mut evals = Vec::new();
        match &self.inference {
            InferenceMode::Frequentist => {
                for atom in identified.class_atoms.iter() {
                    evals.push(evaluate_temporal_class_atom_frequentist(
                        self, data, query, atom, ctx,
                    )?);
                }
            }
            InferenceMode::Bayesian(_) => {
                for atom in identified.class_atoms.iter() {
                    evals
                        .push(evaluate_temporal_class_atom_bayesian(self, data, query, atom, ctx)?);
                }
            }
        }

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_temporal_class_posterior_evals(&identified, &evals, unidentified_mass)?;
        let mut estimate = mixed.estimate;
        let mut shared_block = None;
        let mut shared_block_designs = 0usize;
        if matches!(self.inference, InferenceMode::Frequentist)
            && matches!(mixed.policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && self.bootstrap_replicates > 0
        {
            let mut designs = Vec::new();
            let mut weights = Vec::new();
            let mut indexers = Vec::new();
            for eval in &evals {
                let graph_w = identified_weight_for_key(&identified.graphs, eval.key);
                if graph_w <= 0.0 {
                    continue;
                }
                for fitted in &eval.fitted_designs {
                    designs.push(&fitted.design);
                    weights.push(graph_w * fitted.completion_weight);
                    indexers.push(&fitted.indexer);
                }
            }
            shared_block_designs = designs.len();
            if !designs.is_empty() {
                let block = shared_circular_block_mixture_se(
                    &designs,
                    &weights,
                    temporal_class_block_span(indexers.iter().copied()),
                    self.bootstrap_replicates,
                    0x7C60_C1A5,
                    ctx,
                );
                let se_bootstrap = block.se.is_finite().then_some(block.se);
                let influence = estimate.influence.clone();
                estimate = EffectEstimate::from_parts(
                    estimate.ate,
                    estimate.se_analytic,
                    se_bootstrap,
                    (self.bootstrap_replicates > 0).then_some(block.completed),
                    (self.bootstrap_replicates > 0)
                        .then_some(block.attempted.saturating_sub(block.completed)),
                    ctx.cancellation.is_cancelled(),
                    false,
                    estimate.assumptions.clone(),
                    estimate.overlap,
                    estimate.overlap_report.clone(),
                    estimate.retained_memory_bytes,
                )
                .with_block_family(antecedent_estimate::CircularBlockFamily::Mixture);
                estimate.influence = influence;
                if se_bootstrap.is_some() {
                    shared_block = Some(block);
                }
            }
        }
        let mut identification = mixed.identification;
        identification.status = if unidentified_mass > 0.0
            || mixed.mixture.unevaluable_mass > 0.0
            || !mixed.mixture.full_mass_scope
            || matches!(mixed.policy, StructuralAggregationPolicy::GraphDependentAtoms)
        {
            IdentificationStatus::GraphDependent
        } else if matches!(mixed.policy, StructuralAggregationPolicy::IdentifiedSetEnvelope) {
            IdentificationStatus::PartiallyIdentified
        } else {
            identification.status
        };

        let mut diagnostics = identification.diagnostics.clone();
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
            "estimate.graph_posterior.temporal_class_envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "each {class_tag} posterior atom was evaluated with the existing temporal class \
                 envelope; outer weights are posterior probabilities"
            ),
        ));
        diagnostics.push(
            Diagnostic::new(
                "estimate.graph_posterior.structural_aggregation",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "policy={}; weight_basis=posterior_probability; completion enumeration is \
                     not posterior probability; identified_mass={}; unidentified_mass={}",
                    mixed.policy.as_str(),
                    mixed.mixture.identified_mass,
                    mixed.mixture.unidentified_mass
                ),
            )
            .with_fields(mass_fields(
                Some(mixed.mixture.identified_mass),
                mixed.mixture.unidentified_mass,
            )),
        );
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if let Some(block) = shared_block {
            diagnostics.push(
                Diagnostic::new(
                    "estimate.temporal_class.frequentist.shared_block",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    shared_block_mixture_message(
                        "frozen posterior graph weights",
                        mixed.mixture.identified_mass,
                        mixed.mixture.unidentified_mass,
                        &block,
                    ),
                )
                .with_fields(mass_fields(
                    Some(mixed.mixture.identified_mass),
                    mixed.mixture.unidentified_mass,
                )),
            );
            diagnostics.extend(short_series_warning(
                block.effective_rows,
                antecedent_estimate::CircularBlockFamily::Mixture,
            ));
        } else if shared_block_designs > 1
            && matches!(mixed.policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && matches!(self.inference, InferenceMode::Frequentist)
            && self.bootstrap_replicates > 0
        {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        let n_contributing = evals
            .iter()
            .filter(|eval| eval.estimate.as_ref().is_some_and(|estimate| estimate.ate.is_finite()))
            .count();
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic));
        if n_contributing > 1
            && estimate.se_analytic.is_finite()
            && estimate.influence.as_ref().is_some_and(|inf| inf.iter().all(|v| v.is_finite()))
        {
            diagnostics.push(Diagnostic::new(
                "estimate.graph_posterior.joint_if_se",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "SE of E[τ | identified] from the joint influence-function covariance of \
                     {n_contributing} temporal class-posterior atoms fit on shared rows, with \
                     frozen graph weights; graph-weight uncertainty and unidentified mass are not \
                     in the SE; the interval is for the reported aggregate, not a distribution \
                     over graph-specific effects"
                ),
            ));
        }

        let estimator_id = match (&self.inference, is_multi_step_sustained(query)) {
            (InferenceMode::Frequentist, false) => EstimatorId::TemporalLinearAdjustment,
            (InferenceMode::Frequentist, true) => EstimatorId::TemporalSequentialGcomp,
            (InferenceMode::Bayesian(_), true) => EstimatorId::TemporalSequentialGcomp,
            (InferenceMode::Bayesian(_), false) => EstimatorId::BayesianTemporalGcomp,
        };
        let (refutations, refute_diagnostics) = refute_temporal_class_graph_posterior(
            self,
            data,
            query,
            &identified,
            &evals,
            mixed.policy,
            estimator_id.as_str(),
            ctx,
        )?;
        diagnostics.extend(refute_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let bootstrap_replicates_ok = estimate.bootstrap_replicates_ok;
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand: mixed.estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_class_graph_posterior",
                    mixed.policy.as_str(),
                )),
                posterior: mixed.posterior,
                diagnostics: Some(diagnostics),
                structural_response: Some(mixed.mixture),
                ..Default::default()
            },
        }))
    }
}

struct MixedTemporalClassPosterior {
    policy: StructuralAggregationPolicy,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
    mixture: StructuralResponseMixture,
}

fn evaluate_temporal_class_atom_frequentist(
    study: &Study,
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    ctx: &ExecutionContext,
) -> Result<TemporalClassAtomEval, CausalError> {
    let envelope = &atom.envelope.envelope;
    let mut weighted = 0.0;
    let mut total = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut primary = None;
    let mut se_items = Vec::new();
    let mut refute_atoms = Vec::new();
    let mut sequential_atoms = Vec::new();
    let mut fitted_designs = Vec::new();
    for (i, (case, indexer)) in envelope.cases.iter().zip(atom.envelope.indexers.iter()).enumerate()
    {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let (estimand, estimate, design, sequential) = if is_multi_step_sustained(query) {
            let Some(dag) = case.graph.sequential_dag() else {
                continue;
            };
            let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
                .or_else(|_| {
                    select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)
                })?;
            let mut assumptions = case.result.required_assumptions.clone();
            assumptions.push(antecedent_core::AssumptionRecord {
                assumption: antecedent_core::Assumption::ParametricRestriction(
                    antecedent_core::ParametricAssumption {
                        id: Arc::from("temporal.sequential.linear_sem"),
                        description: Arc::from(
                            "linear additive mechanisms on the identified unfolded DAG",
                        ),
                    },
                ),
                source: antecedent_core::AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("temporal.sequential.gcomp"),
                },
                scope: antecedent_core::AssumptionScope::Estimation,
                status: antecedent_core::AssumptionStatus::Declared,
            });
            let design = TemporalAtomDesign::sequential(
                data,
                &dag,
                indexer,
                &estimand,
                query,
                case.result.status,
                ctx,
            )?;
            let estimate = design.effect_estimate(assumptions);
            let sequential = super::sequential_validation::SequentialValidationAtom {
                weight: case.weight.0,
                graph: dag,
                indexer: indexer.clone(),
                estimand: estimand.clone(),
                status: case.result.status,
                estimate: estimate.clone(),
                mechanisms: Vec::new(),
            };
            (estimand, estimate, design, Some(sequential))
        } else {
            let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
            let design = TemporalAtomDesign::linear(
                data,
                &estimand,
                query,
                indexer,
                study.split.as_ref(),
                ctx,
            )?;
            (
                estimand,
                design.effect_estimate(case.result.required_assumptions.clone()),
                design,
                None,
            )
        };
        let w = case.weight.0;
        if estimate.ate.is_finite() {
            weighted += w * estimate.ate;
            total += w;
            lo = lo.min(estimate.ate);
            hi = hi.max(estimate.ate);
            se_items.push((w, estimate.se_analytic));
            fitted_designs.push(TemporalClassFittedDesign {
                completion_weight: w,
                design,
                indexer: indexer.clone(),
            });
            if primary.is_none() {
                primary = Some((estimand.clone(), estimate.assumptions.clone()));
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: w,
                estimand,
                indexer: Some(indexer.clone()),
                original: estimate,
            });
            if let Some(sequential) = sequential {
                sequential_atoms.push(sequential);
            }
        }
    }
    let estimate = if total > 0.0 {
        let (_, assumptions) = primary.clone().ok_or_else(|| CausalError::Compile {
            message: "temporal class posterior atom missing estimand".into(),
        })?;
        Some(EffectEstimate::new(
            weighted / total,
            mix_weighted_analytic_se(se_items),
            assumptions,
            OverlapPolicy::ExplicitOverride,
        ))
    } else {
        None
    };
    Ok(TemporalClassAtomEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or_else(|| primary.map(|(e, _)| e)),
        estimate,
        posterior: None,
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atoms,
        sequential_atoms,
        fitted_designs,
    })
}

fn evaluate_temporal_class_atom_bayesian(
    study: &Study,
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    ctx: &ExecutionContext,
) -> Result<TemporalClassAtomEval, CausalError> {
    let cfg = match &study.inference {
        InferenceMode::Bayesian(c) => c.clone(),
        InferenceMode::Frequentist => {
            return Err(CausalError::Unsupported {
                message: "evaluate_temporal_class_atom_bayesian requires inference=Bayesian",
            });
        }
    };
    if is_multi_step_sustained(query) {
        return evaluate_temporal_class_atom_sequential_bayesian(study, data, query, atom, ctx);
    }
    let envelope = &atom.envelope.envelope;
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    let bayes = bayesian_temporal_gcomp(&cfg, ctx);
    let mut ws = BayesianGCompWorkspace::default();
    let mut weighted = 0.0;
    let mut total = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut primary = None;
    let mut atom_posterior = None;
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut refute_atoms = Vec::new();
    for (i, (case, indexer)) in envelope.cases.iter().zip(atom.envelope.indexers.iter()).enumerate()
    {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let prep = estimator
            .prepare(data, &estimand, query, indexer, study.split.as_ref(), &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        let names =
            antecedent_estimate::temporal_coefficient_names(data, &estimand, query, indexer)
                .map_err(CausalError::from)?;
        let bprep = BayesianGComputationAte::from_prepared_temporal(&prep, names)
            .map_err(CausalError::from)?;
        let posterior =
            bayes.fit(&bprep, case.result.status, &mut ws, ctx).map_err(CausalError::from)?;
        let summary = effect_from_posterior(&posterior)?;
        if summary.ate.is_finite() {
            weighted += case.weight.0 * summary.ate;
            total += case.weight.0;
            lo = lo.min(summary.ate);
            hi = hi.max(summary.ate);
            if primary.is_none() {
                primary = Some(estimand.clone());
                assumptions = summary.assumptions.clone();
                atom_posterior = Some(posterior);
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: i as u64,
                weight: case.weight.0,
                estimand,
                indexer: Some(indexer.clone()),
                original: summary,
            });
        }
    }
    let estimate = (total > 0.0).then(|| {
        EffectEstimate::new(
            weighted / total,
            f64::NAN,
            assumptions,
            OverlapPolicy::ExplicitOverride,
        )
    });
    Ok(TemporalClassAtomEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or(primary),
        estimate,
        posterior: if atom.envelope.envelope.cases.len() == 1 { atom_posterior } else { None },
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atoms,
        sequential_atoms: Vec::new(),
        fitted_designs: Vec::new(),
    })
}

fn evaluate_temporal_class_atom_sequential_bayesian(
    study: &Study,
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    ctx: &ExecutionContext,
) -> Result<TemporalClassAtomEval, CausalError> {
    let cfg = match &study.inference {
        InferenceMode::Bayesian(c) => c,
        InferenceMode::Frequentist => {
            return Err(CausalError::Unsupported {
                message: "evaluate_temporal_class_atom_sequential_bayesian requires Bayesian",
            });
        }
    };
    let bayes = bayesian_gcomp(cfg, ctx);
    let envelope = &atom.envelope.envelope;
    let mut weighted = 0.0;
    let mut total = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut primary = None;
    let mut atom_posterior = None;
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut sequential_atoms = Vec::new();
    for (i, (case, indexer)) in envelope.cases.iter().zip(atom.envelope.indexers.iter()).enumerate()
    {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let Some(dag) = case.graph.sequential_dag() else {
            continue;
        };
        let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
            .or_else(|_| select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment))?;
        let mut case_assumptions = case.result.required_assumptions.clone();
        case_assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: Arc::from("temporal.sequential.linear_sem"),
                    description: Arc::from(
                        "linear additive mechanisms on the identified unfolded DAG",
                    ),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("temporal.sequential.gcomp"),
            },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        let seed = completion_fit_seed(ctx, atom.key.wrapping_add(i as u64));
        let completion_bayes = BayesianGComputationAte { seed, ..bayes.clone() };
        let (estimate, posterior) =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
                data,
                &dag,
                indexer,
                &estimand,
                query,
                case.result.status,
                case_assumptions,
                0,
                Some(&completion_bayes),
                ctx,
                None,
            )
            .map_err(CausalError::from)?;
        if estimate.ate.is_finite() {
            weighted += case.weight.0 * estimate.ate;
            total += case.weight.0;
            lo = lo.min(estimate.ate);
            hi = hi.max(estimate.ate);
            if primary.is_none() {
                primary = Some(estimand.clone());
                assumptions = estimate.assumptions.clone();
                atom_posterior = posterior;
            }
            sequential_atoms.push(super::sequential_validation::SequentialValidationAtom {
                weight: case.weight.0,
                graph: dag,
                indexer: indexer.clone(),
                estimand,
                status: case.result.status,
                estimate,
                mechanisms: Vec::new(),
            });
        }
    }
    let estimate = (total > 0.0).then(|| {
        EffectEstimate::new(
            weighted / total,
            f64::NAN,
            assumptions,
            OverlapPolicy::ExplicitOverride,
        )
    });
    Ok(TemporalClassAtomEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or(primary),
        estimate,
        posterior: if atom.envelope.envelope.cases.len() == 1 { atom_posterior } else { None },
        identified_set: (lo.is_finite() && hi.is_finite()).then_some((lo, hi)),
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        refute_atoms: Vec::new(),
        sequential_atoms,
        fitted_designs: Vec::new(),
    })
}

fn mix_temporal_class_posterior_evals(
    identified: &CachedTemporalClassPosteriorIdentification,
    evals: &[TemporalClassAtomEval],
    unidentified_mass: f64,
) -> Result<MixedTemporalClassPosterior, CausalError> {
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
    let mut atom_posteriors = Vec::new();
    let mut outer_se_items = Vec::new();
    let mut outer_atom_ifs = Vec::new();
    let mut outer_atom_weights = Vec::new();

    for eval in evals {
        let w = identified_weight_for_key(&identified.graphs, eval.key);
        let value = eval.estimate.as_ref().and_then(|e| e.ate.is_finite().then_some(e.ate));
        if value.is_some() {
            identified_weight += w;
            if let Some((a, b)) = eval.identified_set {
                lo = lo.min(a);
                hi = hi.max(b);
            } else if let Some(v) = value {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if let Some(estimand) = eval.estimand.as_ref() {
                contributing.push(estimand.clone());
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                    primary_identification = Some(eval_identification(eval, identified));
                    if let Some(estimate) = eval.estimate.as_ref() {
                        assumptions = estimate.assumptions.clone();
                    }
                }
            }
            any_partial |= eval.partial;
            if let Some(p) = eval.posterior.clone() {
                atom_posteriors.push((w, p));
            }
            if let Some(estimate) = eval.estimate.as_ref() {
                outer_se_items.push((w, estimate.se_analytic));
                if let Some(inf) = estimate.influence.as_ref() {
                    outer_atom_ifs.push(inf.to_vec());
                    outer_atom_weights.push(w);
                }
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
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    let mixable_scalar =
        matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean) && mixable > 0.0;
    let ate = if mixable_scalar { weighted / mixable } else { f64::NAN };
    let conditional_on_identified =
        mixable_scalar.then(|| antecedent_core::ResponseValue::Scalar(ate));
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
    let identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: format!(
            "temporal class graph-posterior mix: no evaluable atom values (evals={}, mixable={mixable})",
            evals.len()
        ),
    })?;
    let estimand = primary_estimand
        .or_else(|| identification.estimands.first().cloned())
        .ok_or_else(|| CausalError::Compile {
            message: "temporal class graph-posterior envelope: missing estimand".into(),
        })?;
    let n_contributing = outer_se_items.len();
    let joint = mixable_scalar && n_contributing > 1 && outer_atom_ifs.len() == n_contributing;
    let se = if mixable_scalar {
        if joint {
            mix_static_envelope_se(&outer_atom_ifs, &outer_atom_weights)
        } else {
            mix_weighted_analytic_se(outer_se_items)
        }
    } else {
        f64::NAN
    };
    let mut estimate = EffectEstimate::new(ate, se, assumptions, OverlapPolicy::ExplicitOverride);
    if joint {
        if let Some(inf) = mixed_static_influence(&outer_atom_ifs, &outer_atom_weights) {
            estimate.influence = Some(inf);
        }
    }
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
    Ok(MixedTemporalClassPosterior {
        policy,
        identification,
        estimand,
        estimate,
        posterior,
        mixture,
    })
}

fn eval_identification(
    eval: &TemporalClassAtomEval,
    identified: &CachedTemporalClassPosteriorIdentification,
) -> IdentificationResult {
    identified
        .class_atoms
        .iter()
        .find(|atom| atom.key == eval.key)
        .map(|atom| atom.identification.clone())
        .unwrap_or_else(|| identified.class_atoms[0].identification.clone())
}

/// Inner: TemporalDag Pulse/Sustained refuters per identified completion.
/// Outer: mix those atom reports by frozen posterior weight.
fn refute_temporal_class_graph_posterior(
    study: &Study,
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    identified: &CachedTemporalClassPosteriorIdentification,
    evals: &[TemporalClassAtomEval],
    policy: StructuralAggregationPolicy,
    estimator: &str,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    if matches!(study.refute, RefuteSuite::None) && study.custom_validators.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mix_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean);
    let tabular = TabularData::new(data.storage().clone());
    let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    for eval in evals {
        let weight = identified_weight_for_key(&identified.graphs, eval.key);
        if weight <= 0.0 {
            continue;
        }
        let (reports, notes) = if is_multi_step_sustained(query) {
            if eval.sequential_atoms.is_empty() {
                continue;
            }
            let original = eval.estimate.as_ref().map_or(f64::NAN, |e| e.ate);
            let (reports, notes, _) = super::sequential_validation::validate_sequential(
                data,
                query,
                &eval.sequential_atoms,
                study.refute,
                &study.custom_validators,
                None,
                None,
                original,
                ctx,
                super::super::latency::predictive_check_sims(study.latency_mode),
            )?;
            (reports, notes)
        } else {
            if eval.refute_atoms.is_empty() {
                continue;
            }
            run_envelope_effect_refuters(
                &tabular,
                &ate_q,
                &eval.refute_atoms,
                &mut EstimationWorkspace::default(),
                ctx,
                study.refute,
                estimator,
                &study.custom_validators,
                Some(query),
                study.split.as_ref(),
                Some(data.time_index()),
            )?
        };
        diagnostics.extend(
            notes.into_iter().filter(|d| d.code.as_ref() != "refute.envelope.effect_mixture"),
        );
        per_atom.push((weight, reports));
    }
    let inner_keys: String = evals
        .iter()
        .filter(|eval| !eval.refute_atoms.is_empty() || !eval.sequential_atoms.is_empty())
        .map(|eval| format!("{:x}", eval.key))
        .collect::<Vec<_>>()
        .join(",");
    diagnostics.push(Diagnostic::new(
        "refute.envelope.temporal_class_posterior",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        if mix_scalar {
            format!(
                "inner: TemporalDag Pulse/Sustained refuters evaluated each identified completion \
                 within class-posterior atoms [{inner_keys}] against that completion's own \
                 estimate; outer: those atom reports mix by frozen posterior graph weight and \
                 pass only if every contributing atom passes"
            )
        } else {
            format!(
                "inner: TemporalDag Pulse/Sustained refuters evaluated each identified completion \
                 within class-posterior atoms [{inner_keys}] against that completion's own \
                 estimate; outer scalar mix skipped because StructuralAggregationPolicy is {} — \
                 per-atom reports are retained and are not compared against a pooled mixture",
                policy.as_str()
            )
        },
    ));
    let reports = if mix_scalar {
        mix_temporal_class_posterior_refutation_reports(&per_atom)
    } else {
        per_atom.into_iter().flat_map(|(_, reports)| reports).collect()
    };
    Ok((reports, diagnostics))
}

fn mix_temporal_class_posterior_refutation_reports(
    per_atom: &[(f64, Vec<antecedent_validate::RefutationReport>)],
) -> Vec<antecedent_validate::RefutationReport> {
    let mut order = Vec::new();
    let mut by_refuter: std::collections::HashMap<
        Arc<str>,
        Vec<(f64, antecedent_validate::RefutationReport)>,
    > = std::collections::HashMap::new();
    for (weight, reports) in per_atom {
        for report in reports {
            let bucket = by_refuter.entry(Arc::clone(&report.refuter)).or_insert_with(|| {
                order.push(Arc::clone(&report.refuter));
                Vec::new()
            });
            bucket.push((*weight, report.clone()));
        }
    }
    let mut out = Vec::with_capacity(order.len());
    for id in order {
        let Some(items) = by_refuter.get(&id) else {
            continue;
        };
        let borrowed: Vec<(f64, &antecedent_validate::RefutationReport)> =
            items.iter().map(|(w, report)| (*w, report)).collect();
        if let Some(mixed) = antecedent_validate::RefutationReport::mixture_weighted(&borrowed) {
            out.push(mixed);
        }
    }
    out
}
