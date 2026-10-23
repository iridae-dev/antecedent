//! TemporalCpdag/Pag graph-posterior Pulse/Sustained: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::atom_mix::{
    AtomBoundRule, AtomEvalView, AtomMixInput, AtomRefuterBatch, ScalarRefuterMix,
    drive_posterior_refuters, mix_atom_evals,
};
use super::sequential_validation::SequentialValidation;
use super::temporal_effect_core::shared_gap_report;
use super::{
    Arc, AtomRefutationPolicy, BayesianGCompWorkspace, BayesianGComputationAte,
    BootstrapAccounting, CausalError, CausalPosterior, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EnvelopeRefutation, EnvelopeRefuteAtom,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphPosterior, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, PosteriorMassSummary, RefuteSuite, Strategy, Study, StudyResult,
    TableView, TabularData, TemporalAtomDesign, TemporalEffectQuery, TemporalIndexer,
    TemporalLinearAdjustment, TemporalRefutation, TimeSeriesData, UNFOLDED_LINEAR_SEM,
    bayesian_gcomp, bayesian_temporal_gcomp, completion_fit_seed, effect_from_posterior,
    elapsed_ns, envelope_se_omission_diagnostic, envelope_se_omits_between_atom_variance,
    identification_status_ok_for_case, identify_cached_diagnostic, mass_fields,
    mix_weighted_analytic_se, overlap_diagnostic, provenance_ids, resolve_posterior_status,
    select_estimand, sequential_linear_sem_assumption, shared_block_mixture_message,
    shared_circular_block_mixture_se, short_series_warning, temporal_class_block_span,
    temporal_refuter_average,
};
use crate::analysis::prepared::{
    CachedTemporalClassPosteriorAtomIdentification, CachedTemporalClassPosteriorIdentification,
};
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;
use crate::result::{StructuralAggregationPolicy, StructuralResponseMixture};

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
    /// The windows the atom's completions were fitted on, on the original window axis; empty
    /// unless the series has missing time points.
    windows: Vec<antecedent_estimate::GapSelection>,
}

impl super::Study {
    /// Replaces the mixed estimate's bootstrap accounting with the shared
    /// circular-block SE over the evaluated completions' designs, when the
    /// frequentist weighted mean asks for one; returns the block (when its SE
    /// is finite) and the number of designs resampled.
    fn class_posterior_shared_block(
        &self,
        (identified, evals): (
            &CachedTemporalClassPosteriorIdentification,
            &[TemporalClassAtomEval],
        ),
        mut estimate: EffectEstimate,
        policy: StructuralAggregationPolicy,
        ctx: &ExecutionContext,
    ) -> (EffectEstimate, Option<super::class_envelope_se::SharedCircularBlockSe>, usize) {
        let mut shared_block = None;
        let mut shared_block_designs = 0usize;
        let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
        if self.inference.is_frequentist()
            && matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && self.bootstrap_replicates > 0
        {
            let mut designs = Vec::new();
            let mut weights = Vec::new();
            let mut indexers = Vec::new();
            for eval in evals {
                let graph_w = key_weights.get(eval.key);
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
                let influence = estimate.scores.influence.clone();
                estimate = EffectEstimate::from_parts(
                    estimate.ate,
                    estimate.se_analytic(),
                    BootstrapAccounting {
                        se: se_bootstrap,
                        replicates_ok: (self.bootstrap_replicates > 0).then_some(block.completed),
                        replicates_failed: (self.bootstrap_replicates > 0)
                            .then_some(block.attempted.saturating_sub(block.completed)),
                        cancelled: ctx.cancellation.is_cancelled(),
                        early_stopped: false,
                    },
                    estimate.provenance.assumptions.clone(),
                    estimate.provenance.overlap,
                    estimate.diagnostics.overlap_report.clone(),
                    estimate.provenance.retained_memory_bytes,
                )
                .with_block_family(antecedent_estimate::CircularBlockFamily::Mixture);
                estimate.scores.influence = influence;
                if se_bootstrap.is_some() {
                    shared_block = Some(block);
                }
            }
        }
        (estimate, shared_block, shared_block_designs)
    }

    /// The pre-refutation diagnostics of a class graph-posterior effect: the
    /// withheld posterior, the evaluation and aggregation notes, overlap, the
    /// shared-block or omitted between-atom SE, and the joint influence SE.
    fn class_posterior_diagnostics(
        &self,
        class_tag: &str,
        mut diagnostics: Vec<Diagnostic>,
        (policy, mixture, posterior_withheld): (
            StructuralAggregationPolicy,
            &StructuralResponseMixture,
            bool,
        ),
        (estimate, shared_block, shared_block_designs, evals): (
            &EffectEstimate,
            Option<&super::class_envelope_se::SharedCircularBlockSe>,
            usize,
            &[TemporalClassAtomEval],
        ),
    ) -> Vec<Diagnostic> {
        if self.inference.is_bayesian() && posterior_withheld {
            diagnostics.push(graph_posterior_withheld_diagnostic());
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
                    policy.as_str(),
                    mixture.identified_mass,
                    mixture.unidentified_mass
                ),
            )
            .with_fields(mass_fields(Some(mixture.identified_mass), mixture.unidentified_mass)),
        );
        diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
        if let Some(block) = shared_block {
            diagnostics.push(
                Diagnostic::new(
                    "estimate.temporal_class.frequentist.shared_block",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    shared_block_mixture_message(
                        "frozen posterior graph weights",
                        mixture.identified_mass,
                        mixture.unidentified_mass,
                        block,
                    ),
                )
                .with_fields(mass_fields(Some(mixture.identified_mass), mixture.unidentified_mass)),
            );
            diagnostics.extend(short_series_warning(
                block.effective_rows,
                antecedent_estimate::CircularBlockFamily::Mixture,
            ));
        } else if shared_block_designs > 1
            && matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && self.inference.is_frequentist()
            && self.bootstrap_replicates > 0
        {
            diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        let n_contributing = evals
            .iter()
            .filter(|eval| eval.estimate.as_ref().is_some_and(|estimate| estimate.ate.is_finite()))
            .count();
        diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic()));
        if n_contributing > 1
            && estimate.se_analytic().is_finite()
            && estimate
                .scores
                .influence
                .as_ref()
                .is_some_and(|inf| inf.iter().all(|v| v.is_finite()))
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
        diagnostics
    }

    /// The prepared class posterior identification, the product the plan's lowered
    /// program retained at compile, or a build from the posterior when the study
    /// carries neither; refuses a posterior with no identified class atoms.
    fn temporal_class_posterior_identification(
        &self,
        gp: &GraphPosterior,
        vars: &[antecedent_core::VariableId],
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<CachedTemporalClassPosteriorIdentification>, bool), CausalError> {
        let (identified, identify_cached) =
            if let Some(cache) = self.temporal_class_posterior_identification_cache.as_ref() {
                (Arc::clone(cache), true)
            } else if let Some(cache) =
                crate::analysis::lowering::plan_retained_temporal_class_posterior(self, physical)
            {
                (cache, false)
            } else {
                (
                Arc::new(
                    crate::analysis::prepared::build_temporal_class_posterior_identification_cache(
                        gp,
                        vars,
                        query,
                        self.max_completions,
                        ctx,
                    )?,
                ),
                false,
            )
            };
        if identified.class_atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "temporal class graph-posterior envelope: no identified class atoms"
                    .into(),
            });
        }
        Ok((identified, identify_cached))
    }

    /// Evaluates each class atom's envelope under the study's inference mode
    /// on the context's workers.
    fn evaluate_temporal_class_atoms(
        &self,
        (data, query, gaps): (&TimeSeriesData, &TemporalEffectQuery, bool),
        identified: &CachedTemporalClassPosteriorIdentification,
        ctx: &ExecutionContext,
    ) -> Result<Vec<TemporalClassAtomEval>, CausalError> {
        ctx.map_indexed(identified.class_atoms.len(), |i, inner| {
            let atom = &identified.class_atoms[i];
            match &self.inference {
                InferenceMode::Frequentist => {
                    evaluate_temporal_class_atom_frequentist(self, (data, query, gaps), atom, inner)
                }
                InferenceMode::Bayesian(_) => {
                    evaluate_temporal_class_atom_bayesian(self, (data, query, gaps), atom, inner)
                }
            }
        })
    }

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
        if query.is_multi_step_sustained() && self.split.is_some() {
            return Err(CausalError::Unsupported {
                message: "class-aware multi-step sustained requires no discovery-estimation split",
            });
        }
        let class_tag = temporal_class_posterior_tag(gp.atom_kind)?;
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let (identified, identify_cached) =
            self.temporal_class_posterior_identification(gp, &vars, query, physical, ctx)?;
        let gaps = self.admit_posterior_gaps(data, &vars, false)?;
        let evals = self.evaluate_temporal_class_atoms((data, query, gaps), &identified, ctx)?;

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_temporal_class_posterior_evals(&identified, &evals, unidentified_mass)?;
        let (estimate, shared_block, shared_block_designs) = self.class_posterior_shared_block(
            (&identified, &evals),
            mixed.estimate,
            mixed.policy,
            ctx,
        );
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

        let mut diagnostics = self.class_posterior_diagnostics(
            class_tag,
            identification.diagnostics.clone(),
            (mixed.policy, &mixed.mixture, mixed.posterior.is_none()),
            (&estimate, shared_block.as_ref(), shared_block_designs, &evals),
        );
        let estimator_id = match (&self.inference, query.is_multi_step_sustained()) {
            (InferenceMode::Frequentist, false) => EstimatorId::TemporalLinearAdjustment,
            (InferenceMode::Frequentist | InferenceMode::Bayesian(_), true) => {
                EstimatorId::TemporalSequentialGcomp
            }
            (InferenceMode::Bayesian(_), false) => EstimatorId::BayesianTemporalGcomp,
        };
        let (refutations, refute_diagnostics) = refute_temporal_class_graph_posterior(
            self,
            data,
            query,
            &identified,
            &evals,
            AtomRefutationPolicy { aggregation: mixed.policy, estimator: estimator_id.as_str() },
            ctx,
        )?;
        diagnostics.extend(refute_diagnostics);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let bootstrap_replicates_ok = estimate.uncertainty.bootstrap.replicates_ok;
        let missing_data = if gaps {
            let windows: Vec<_> =
                evals.iter().flat_map(|eval| eval.windows.iter().cloned()).collect();
            Some(shared_gap_report(data, &vars, &windows)?)
        } else {
            None
        };
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                mixed.estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralizedAdjustment, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_replicates_ok)
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.temporal_class_graph_posterior",
                    mixed.policy.as_str(),
                )),
                posterior: mixed.posterior,
                diagnostics: Some(diagnostics),
                structural_response: Some(mixed.mixture),
                ..Default::default()
            }),
        );
        result.missing_data = missing_data;
        Ok(result)
    }

    /// Whether a graph posterior fits the intact lag windows of a series with missing time
    /// points: the series has a gap in the posterior's variables and complete-case handling
    /// was declared. Without the declaration the fit refuses an incomplete series as it always
    /// has. A discovery-estimation split fits aligned rows of a complete series only, and a
    /// Bayesian fit with `bayesian_checks` reads consecutive rows in its predictive checks.
    pub(super) fn admit_posterior_gaps(
        &self,
        data: &TimeSeriesData,
        variables: &[antecedent_core::VariableId],
        bayesian_checks: bool,
    ) -> Result<bool, CausalError> {
        use crate::analysis::missing_data::MissingRoute;
        if !self.declares_complete_case()
            || antecedent_data::first_missing_cell(data, variables)?.is_none()
        {
            return Ok(false);
        }
        let route = MissingRoute::Temporal;
        let refuse = |reason: &str| {
            CausalError::Estimate(antecedent_estimate::missing_policy::policy_not_applicable(
                antecedent_data::MissingPolicy::CompleteCase,
                route.label(),
                reason,
                &route.supported()[..1],
            ))
        };
        if self.split.is_some() {
            return Err(refuse("gap-aware windows are executed on an unsplit series"));
        }
        if bayesian_checks && self.refute != RefuteSuite::None {
            return Err(refuse(
                "the Bayesian predictive checks and prior sensitivity read consecutive rows; request no refuters on a series with gaps",
            ));
        }
        Ok(true)
    }
}

/// The tag a class graph-posterior names its atoms by, or the refusal when
/// the atoms are not CPDAG or PAG completions.
fn temporal_class_posterior_tag(
    kind: antecedent_discovery::GraphPosteriorAtomKind,
) -> Result<&'static str, CausalError> {
    match kind {
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag => Ok("temporal_cpdag"),
        antecedent_discovery::GraphPosteriorAtomKind::Pag => Ok("temporal_pag"),
        _ => Err(CausalError::Compile {
            message: "temporal class graph-posterior requires Cpdag or Pag atom_kind".into(),
        }),
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

/// A completion's estimand, estimate, design and (for a multi-step sustained
/// query) sequential validation atom.
type FittedClassCase = (
    IdentifiedEstimand,
    EffectEstimate,
    TemporalAtomDesign,
    Option<super::sequential_validation::SequentialValidationAtom>,
);

/// One completion's fitted design and estimate: the sequential g-computation
/// for a multi-step sustained query (absent when the completion has no
/// sequential DAG), otherwise the linear adjustment.
fn fit_temporal_class_case(
    study: &Study,
    (data, query, gaps): (&TimeSeriesData, &TemporalEffectQuery, bool),
    (case, indexer): (
        &antecedent_identify::GraphIdentificationCase<antecedent_identify::TemporalCompletionGraph>,
        &TemporalIndexer,
    ),
    ctx: &ExecutionContext,
) -> Result<Option<FittedClassCase>, CausalError> {
    if query.is_multi_step_sustained() {
        let Some(dag) = case.graph.sequential_dag() else {
            return Ok(None);
        };
        let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
            .or_else(|_| select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment))?;
        let mut assumptions = case.result.required_assumptions.clone();
        assumptions.push(sequential_linear_sem_assumption(UNFOLDED_LINEAR_SEM));
        let design = TemporalAtomDesign::sequential_with_gaps(
            data,
            &dag,
            indexer,
            &estimand,
            query,
            case.result.status,
            gaps,
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
        Ok(Some((estimand, estimate, design, Some(sequential))))
    } else {
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let design = TemporalAtomDesign::linear_with_gaps(
            data,
            &estimand,
            query,
            indexer,
            study.split.as_ref(),
            gaps,
            ctx,
        )?;
        let estimate = design.effect_estimate(case.result.required_assumptions.clone());
        Ok(Some((estimand, estimate, design, None)))
    }
}

fn evaluate_temporal_class_atom_frequentist(
    study: &Study,
    (data, query, gaps): (&TimeSeriesData, &TemporalEffectQuery, bool),
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
    let mut windows = Vec::new();
    for (i, (case, indexer)) in envelope.cases.iter().zip(atom.envelope.indexers.iter()).enumerate()
    {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let Some((estimand, estimate, design, sequential)) =
            fit_temporal_class_case(study, (data, query, gaps), (case, indexer), ctx)?
        else {
            continue;
        };
        let w = case.weight.0;
        if estimate.ate.is_finite() {
            weighted += w * estimate.ate;
            total += w;
            lo = lo.min(estimate.ate);
            hi = hi.max(estimate.ate);
            se_items.push((w, estimate.se_analytic()));
            if gaps {
                windows.push(design.window_selection());
            }
            fitted_designs.push(TemporalClassFittedDesign {
                completion_weight: w,
                design,
                indexer: indexer.clone(),
            });
            if primary.is_none() {
                primary = Some((estimand.clone(), estimate.provenance.assumptions.clone()));
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
        windows,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "the atom's gap admission, prepared design and segment-wise tempering read one completion"
)]
fn evaluate_temporal_class_atom_bayesian(
    study: &Study,
    (data, query, gaps): (&TimeSeriesData, &TemporalEffectQuery, bool),
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    ctx: &ExecutionContext,
) -> Result<TemporalClassAtomEval, CausalError> {
    let cfg = match &study.inference {
        InferenceMode::Bayesian(c) => c.clone(),
        InferenceMode::Frequentist => {
            return Err(CausalError::unsupported_inference(
                "evaluate_temporal_class_atom_bayesian requires inference=Bayesian",
            ));
        }
    };
    if query.is_multi_step_sustained() {
        return evaluate_temporal_class_atom_sequential_bayesian(
            study,
            (data, query, gaps),
            atom,
            ctx,
        );
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
    let mut windows = Vec::new();
    for (i, (case, indexer)) in envelope.cases.iter().zip(atom.envelope.indexers.iter()).enumerate()
    {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let (prep, selection) = if gaps {
            let (prep, selection) = estimator
                .prepare_with_gaps(data, &estimand, query, indexer, &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            (prep, Some(selection))
        } else {
            let prep = estimator
                .prepare(data, &estimand, query, indexer, study.split.as_ref(), &ctx.kernel_policy)
                .map_err(CausalError::from)?;
            (prep, None)
        };
        let names =
            antecedent_estimate::temporal_coefficient_names(data, &estimand, query, indexer)
                .map_err(CausalError::from)?;
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
        windows.extend(selection);
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
                assumptions = summary.provenance.assumptions.clone();
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
        windows,
    })
}

fn evaluate_temporal_class_atom_sequential_bayesian(
    study: &Study,
    (data, query, gaps): (&TimeSeriesData, &TemporalEffectQuery, bool),
    atom: &CachedTemporalClassPosteriorAtomIdentification,
    ctx: &ExecutionContext,
) -> Result<TemporalClassAtomEval, CausalError> {
    let cfg = match &study.inference {
        InferenceMode::Bayesian(c) => c,
        InferenceMode::Frequentist => {
            return Err(CausalError::unsupported_inference(
                "evaluate_temporal_class_atom_sequential_bayesian requires Bayesian",
            ));
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
    let mut windows = Vec::new();
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
        case_assumptions.push(sequential_linear_sem_assumption(UNFOLDED_LINEAR_SEM));
        let seed = completion_fit_seed(ctx, atom.key.wrapping_add(i as u64));
        let completion_bayes = BayesianGComputationAte { seed, ..bayes.clone() };
        let (estimate, posterior, selection) =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_gaps_if(
                &antecedent_estimate::SequentialProblem {
                    data,
                    graph: &dag,
                    indexer,
                    estimand: &estimand,
                },
                query,
                antecedent_estimate::SequentialInference {
                    status: case.result.status,
                    assumptions: case_assumptions,
                    bootstrap_replicates: 0,
                    bayesian: Some(&completion_bayes),
                },
                ctx,
                None,
                gaps,
            )
            .map_err(CausalError::from)?;
        windows.extend(selection);
        if estimate.ate.is_finite() {
            weighted += case.weight.0 * estimate.ate;
            total += case.weight.0;
            lo = lo.min(estimate.ate);
            hi = hi.max(estimate.ate);
            if primary.is_none() {
                primary = Some(estimand.clone());
                assumptions = estimate.provenance.assumptions.clone();
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
        windows,
    })
}

impl AtomEvalView for TemporalClassAtomEval {
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
        self.estimate.as_ref().and_then(|estimate| estimate.ate.is_finite().then_some(estimate.ate))
    }

    fn posterior(&self) -> Option<&CausalPosterior> {
        self.posterior.as_ref()
    }

    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
        self.estimate.as_ref().map(|estimate| &estimate.provenance.assumptions)
    }

    fn identified_bounds(&self) -> Option<(f64, f64)> {
        self.identified_set
    }

    fn partial(&self) -> bool {
        self.partial
    }

    fn analytic_se(&self) -> Option<f64> {
        self.estimate.as_ref().map(antecedent_estimate::EffectEstimate::se_analytic)
    }

    fn influence(&self) -> Option<&[f64]> {
        self.estimate.as_ref().and_then(|estimate| estimate.scores.influence.as_deref())
    }
}

fn mix_temporal_class_posterior_evals(
    identified: &CachedTemporalClassPosteriorIdentification,
    evals: &[TemporalClassAtomEval],
    unidentified_mass: f64,
) -> Result<MixedTemporalClassPosterior, CausalError> {
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
            bounds: AtomBoundRule::SetOrPoint,
            estimand_fallback: true,
        },
        |n, mixable| {
            format!(
                "temporal class graph-posterior mix: no evaluable atom values (evals={n}, mixable={mixable})"
            )
        },
        || "temporal class graph-posterior envelope: missing estimand".to_string(),
    )?;
    Ok(MixedTemporalClassPosterior {
        policy: mixed.policy,
        identification: mixed.identification,
        estimand: mixed.estimand,
        estimate: mixed.estimate,
        posterior: mixed.posterior,
        mixture: mixed.mixture,
    })
}

/// Inner: `TemporalDag` Pulse/Sustained refuters per identified completion.
/// Outer: mix those atom reports by frozen posterior weight.
#[allow(
    clippy::too_many_lines,
    reason = "per-completion refuters and their posterior-weight mixing share the gap context of one frozen posterior"
)]
fn refute_temporal_class_graph_posterior(
    study: &Study,
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    identified: &CachedTemporalClassPosteriorIdentification,
    evals: &[TemporalClassAtomEval],
    refutation: AtomRefutationPolicy<'_>,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let AtomRefutationPolicy { aggregation: policy, estimator } = refutation;
    if matches!(study.refute, RefuteSuite::None) && study.custom_validators.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mix_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean);
    // The atoms were fitted on the intact windows of a series with missing time points: the
    // refuters refit through them and resample blocks of the original window axis.
    let gaps = evals.iter().any(|eval| !eval.windows.is_empty());
    let tabular = TabularData::new(data.storage().clone());
    let ate_q = temporal_refuter_average(query);
    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    drive_posterior_refuters(
        evals,
        mix_scalar,
        ScalarRefuterMix::Effect,
        |eval| !eval.refute_atoms.is_empty() || !eval.sequential_atoms.is_empty(),
        |eval| eval.key,
        |eval| {
            let weight = key_weights.get(eval.key);
            if weight <= 0.0 {
                return Ok(Vec::new());
            }
            let (reports, notes) = if query.is_multi_step_sustained() {
                if eval.sequential_atoms.is_empty() {
                    return Ok(Vec::new());
                }
                let original = eval.estimate.as_ref().map_or(f64::NAN, |estimate| estimate.ate);
                let (reports, notes, _) = super::sequential_validation::validate_sequential(
                    SequentialValidation {
                        data,
                        query,
                        suite: study.refute,
                        custom: &study.custom_validators,
                        predictive_sims: super::super::latency::predictive_check_sims(
                            study.latency_mode,
                        ),
                        gaps,
                    },
                    &eval.sequential_atoms,
                    None,
                    None,
                    original,
                    ctx,
                )?;
                (reports, notes)
            } else {
                if eval.refute_atoms.is_empty() {
                    return Ok(Vec::new());
                }
                EnvelopeRefutation {
                    data: &tabular,
                    query: &ate_q,
                    suite: study.refute,
                    estimator,
                    custom: &study.custom_validators,
                    temporal: Some(TemporalRefutation {
                        query,
                        split: study.split.as_ref(),
                        time_index: data.time_index(),
                        gaps,
                    }),
                }
                .run(
                    &eval.refute_atoms,
                    &mut EstimationWorkspace::default(),
                    ctx,
                )?
            };
            Ok(vec![AtomRefuterBatch {
                weight,
                reports,
                notes: notes
                    .into_iter()
                    .filter(|diagnostic| {
                        diagnostic.code.as_ref() != "refute.envelope.effect_mixture"
                    })
                    .collect(),
            }])
        },
        |inner_keys, mix_scalar| {
            Diagnostic::new(
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
            )
        },
    )
}
