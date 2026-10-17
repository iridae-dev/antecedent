//! The one fit-and-assemble implementation of an incomplete-class temporal
//! effect (linear pulse or single-step sustained, and multi-step sustained).
//!
//! The sealed operation ([`super::CheckedTemporalClassEffectExecution`]), the
//! one-shot route and the sealed Bayesian operation all resolve their proof,
//! gate their own licence, and call these functions. Callers own what they
//! admit (the sealed frequentist operation takes no class prior and no
//! Bayesian fit; the one-shot route does), never how an admitted fit is
//! computed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::sequential_validation::{SequentialValidation, SequentialValidationAtom};
use super::temporal_effect_core::FitFinishMeta;
use super::temporal_path::{temporal_class_envelope_diagnostic, temporal_class_structural_mixture};
use super::{
    Arc, BayesianGComputationAte, CausalError, CausalPosterior, ClassLinearAtoms, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EnvelopeOptions, EnvelopeRefutation,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphClass, GraphEffectDraws,
    GraphIdentFlag, IDENTIFIED_SET_INTERVAL_LEVEL, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceDiagnostics, InferenceMode, PhysicalExecutionPlan, PredictiveCheckReport, RefuteSuite,
    SequentialAtomBlock, Strategy, TableView, TabularData, TemporalDag, TemporalEffectQuery,
    TemporalRefutation, TimeSeriesData, UNFOLDED_LINEAR_SEM, WeightedGraphSamples,
    aggregate_effect_envelope, bayesian_gcomp, class_mixture_estimate, completion_fit_seed,
    effect_from_posterior, envelope_se_omits_between_atom_variance,
    envelope_shared_block_diagnostics, fit_class_linear_atoms, identification_status_ok_for_case,
    identified_set_interval_diagnostics, mag_sequential_unevaluable_diagnostic,
    mix_weighted_analytic_se, nan_effect, posterior_identified_set_interval, provenance_ids,
    same_fitted_mechanisms, select_estimand, sequential_atom_block,
    sequential_linear_sem_assumption, shared_circular_block_mixture_se, temporal_class_block_span,
    temporal_refuter_average,
};
use crate::analysis::prepared::CachedTemporalClassIdentification;
use antecedent_estimate::temporal_sequential::SequentialBayesianMechanism;

/// Everything one class-effect fit reads, resolved by the caller.
pub(super) struct TemporalClassEffectInputs<'a> {
    pub data: &'a TimeSeriesData,
    pub query: &'a TemporalEffectQuery,
    pub bundle: &'a CachedTemporalClassIdentification,
    pub graph_class: GraphClass,
    pub inference: &'a InferenceMode,
    pub class_prior: Option<&'a crate::ClassPrior>,
    pub bootstrap_replicates: u32,
    pub split: Option<&'a antecedent_data::DiscoveryEstimationSplit>,
    pub refute: RefuteSuite,
    pub custom_validators: &'a [Arc<dyn antecedent_validate::CustomEffectValidator>],
    pub latency_mode: Option<crate::analysis::latency::LatencyMode>,
    /// Complete-case handling of missing time points was declared: atoms fit the intact lag
    /// windows and the shared bootstrap resamples the original window axis.
    pub gaps: bool,
}

/// An assembled class-effect fit, before the result envelope.
pub(super) struct TemporalClassEffectFit {
    pub estimand: IdentifiedEstimand,
    pub estimate: EffectEstimate,
    pub estimator_id: EstimatorId,
    pub refutations: Vec<antecedent_validate::RefutationReport>,
    pub diagnostics: Vec<Diagnostic>,
    pub posterior: Option<CausalPosterior>,
    pub structural: crate::result::StructuralResponseMixture,
    pub predictive_checks: Vec<PredictiveCheckReport>,
    pub bootstrap_replicates_ok: Option<u32>,
    pub cancelled: bool,
    /// A point value with no published interval or posterior.
    pub point_only: bool,
}

impl TemporalClassEffectFit {
    /// The result-assembly arguments of this fit on the proof it ran against.
    pub(super) fn finish_args<'a>(
        self,
        physical: &'a PhysicalExecutionPlan,
        identification: IdentificationResult,
        bundle: &CachedTemporalClassIdentification,
        identifier_id: IdentifierId,
        query: &TemporalEffectQuery,
        meta: FitFinishMeta,
    ) -> IdentifiedExecuteFinish<'a> {
        let FitFinishMeta { identify_cached, structure_version, wall_time_ns } = meta;
        IdentifiedExecuteFinish::new(
            physical,
            identification,
            self.estimand,
            self.estimate,
            Strategy::new(identifier_id, self.estimator_id),
            query.treatment,
            query.outcome,
        )
        .identify_cached(identify_cached)
        .refutations(self.refutations)
        .wall_time_ns(wall_time_ns)
        .bootstrap_replicates_ok(self.bootstrap_replicates_ok)
        .cancelled(self.cancelled)
        .extras(IdentifiedExecuteExtras {
            certificate: Some(crate::Identification::TemporalEnvelope {
                envelope: (*bundle.envelope).clone(),
                strategy: identifier_id,
                structure_version,
            }),
            posterior: self.posterior,
            structural_response: Some(self.structural),
            diagnostics: Some(self.diagnostics),
            predictive_checks: self.predictive_checks,
            estimate_provenance: Some(class_envelope_provenance(self.estimator_id)),
            ..Default::default()
        })
    }
}

/// Estimate provenance of a class-envelope fit: the envelope artifact, produced
/// by the estimator's operation.
pub(super) fn class_envelope_provenance(estimator_id: EstimatorId) -> (Arc<str>, Arc<str>) {
    provenance_ids("estimate.temporal_class.envelope", estimator_id.as_str())
}

/// Refuse an envelope with no identified mass.
pub(super) fn require_identified_class_mass(
    envelope: &antecedent_identify::IdentificationEnvelope<
        antecedent_identify::TemporalCompletionGraph,
    >,
) -> Result<(), CausalError> {
    if matches!(envelope.status, IdentificationStatus::NotIdentified)
        || envelope.identified_weight.0 <= 0.0
    {
        return Err(CausalError::not_identified(
            envelope.status,
            envelope.truncated_completions > 0,
            "temporal class-aware effect not identified (no identified mass in envelope)",
        ));
    }
    Ok(())
}

/// The fit of a class envelope the frozen-weight routes serve: a conditional
/// point, a multi-step sustained g-computation (frequentist, or Bayesian with
/// a class prior) or the frequentist linear adjustment. The Bayesian linear
/// mixture is a separate operation.
pub(super) fn fit_temporal_class_effect(
    inputs: &TemporalClassEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalClassEffectFit, CausalError> {
    require_identified_class_mass(&inputs.bundle.envelope.envelope)?;
    if !inputs.query.effect_modifiers.is_empty() {
        fit_class_conditional(inputs, ctx)
    } else if inputs.query.is_multi_step_sustained() {
        fit_class_sequential(inputs, ctx)
    } else {
        fit_class_linear(inputs, ctx)
    }
}

fn fit_class_conditional(
    inputs: &TemporalClassEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalClassEffectFit, CausalError> {
    let (estimate, estimand, _, diagnostics, structural) =
        super::temporal_class_point::assemble_temporal_class_conditional(
            inputs.data,
            inputs.query,
            &inputs.bundle.envelope,
            inputs.graph_class,
            inputs.refute,
            inputs.inference,
            ctx.kernel_policy,
        )?;
    Ok(TemporalClassEffectFit {
        estimand,
        estimate,
        estimator_id: EstimatorId::TemporalLinearAdjustment,
        refutations: Vec::new(),
        diagnostics,
        posterior: None,
        structural,
        predictive_checks: Vec::new(),
        bootstrap_replicates_ok: None,
        cancelled: false,
        point_only: true,
    })
}

/// Frequentist linear fit of a pulse or single-step sustained class envelope.
///
/// The envelope is a frozen-weight enumeration of completions, so there is no
/// class prior; the Bayesian counterpart mixes posteriors and lives with its
/// own operation.
pub(super) fn fit_class_linear(
    inputs: &TemporalClassEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalClassEffectFit, CausalError> {
    let TemporalClassEffectInputs {
        data, query, bundle, bootstrap_replicates, split, gaps, ..
    } = *inputs;
    let envelope = &bundle.envelope.envelope;
    let mut diagnostics = vec![temporal_class_envelope_diagnostic(envelope, inputs.graph_class)];
    let ClassLinearAtoms {
        values,
        designs,
        weights,
        indexers,
        points,
        se_items,
        refute_atoms,
        primary_estimand,
        assumptions,
        weighted_point,
        total_weight,
    } = fit_class_linear_atoms(
        data,
        query,
        &bundle.envelope,
        &bundle.envelope.indexers,
        (split, gaps),
        ctx,
    )?;
    if total_weight <= 0.0 {
        return Err(CausalError::Compile {
            message: "temporal class envelope has no estimable identified cases".into(),
        });
    }
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "temporal class effect has no primary estimand".into(),
    })?;
    let design_refs = designs.iter().collect::<Vec<_>>();
    let block = shared_circular_block_mixture_se(
        &design_refs,
        &weights,
        temporal_class_block_span(indexers.iter().copied()),
        bootstrap_replicates,
        0xC1A5_5E00,
        ctx,
    );
    let identified_set_interval = block
        .identified_set_interval(&points, IDENTIFIED_SET_INTERVAL_LEVEL)
        .map(|interval| interval.with_truncated(envelope.truncated_completions > 0));
    let estimate = class_mixture_estimate(
        weighted_point / total_weight,
        mix_weighted_analytic_se(se_items),
        &block,
        bootstrap_replicates,
        assumptions,
        ctx,
    );
    let tabular = TabularData::new(data.storage().clone());
    let ate_query = temporal_refuter_average(query);
    let (refutations, refute_diagnostics) = EnvelopeRefutation {
        data: &tabular,
        query: &ate_query,
        suite: inputs.refute,
        estimator: EstimatorId::TemporalLinearAdjustment.as_str(),
        custom: inputs.custom_validators,
        temporal: Some(TemporalRefutation { query, split, time_index: data.time_index() }),
    }
    .run(&refute_atoms, &mut EstimationWorkspace::default(), ctx)?;
    diagnostics.extend(refute_diagnostics);
    if block.se.is_finite() {
        diagnostics.extend(envelope_shared_block_diagnostics(
            total_weight,
            envelope.unidentified_weight.0,
            &block,
        ));
    } else if designs.len() > 1 {
        diagnostics.push(envelope_se_omits_between_atom_variance());
    }
    diagnostics.extend(
        identified_set_interval.as_ref().into_iter().flat_map(identified_set_interval_diagnostics),
    );
    let mut structural = temporal_class_structural_mixture(
        envelope,
        crate::result::StructuralWeightBasis::CompletionEnumeration,
        None,
        &values,
    );
    structural.identified_set_interval = identified_set_interval;
    Ok(TemporalClassEffectFit {
        estimand,
        estimator_id: EstimatorId::TemporalLinearAdjustment,
        bootstrap_replicates_ok: (bootstrap_replicates > 0).then_some(block.completed),
        cancelled: ctx.cancellation.is_cancelled(),
        estimate,
        refutations,
        diagnostics,
        posterior: None,
        structural,
        predictive_checks: Vec::new(),
        point_only: false,
    })
}

/// The completion sites a sequential class fit visits.
#[derive(Clone, Copy)]
struct SequentialClassSite<'a> {
    index: usize,
    case: &'a antecedent_identify::GraphIdentificationCase<
        antecedent_identify::TemporalCompletionGraph,
    >,
    indexer: &'a antecedent_core::TemporalIndexer,
    weight: f64,
}

/// Accumulators of a sequential class fit across its completions.
struct SequentialClassRun {
    weights: Vec<f64>,
    flags: Vec<GraphIdentFlag>,
    keys: Vec<u64>,
    atom_values: Vec<Option<f64>>,
    per_graph: Vec<GraphEffectDraws>,
    seq_atoms: Vec<SequentialValidationAtom>,
    seq_atom_keys: Vec<u64>,
    seq_atom_seeds: Vec<u64>,
    atom_posteriors: std::collections::HashMap<u64, CausalPosterior>,
    primary_estimand: Option<IdentifiedEstimand>,
    unevaluable_weight: f64,
    weighted_ate: f64,
    total_w: f64,
    diagnostics: Vec<Diagnostic>,
}

impl SequentialClassRun {
    fn new(case_count: usize, diagnostics: Vec<Diagnostic>) -> Self {
        Self {
            weights: Vec::new(),
            flags: Vec::new(),
            keys: Vec::new(),
            atom_values: vec![None; case_count],
            per_graph: Vec::new(),
            seq_atoms: Vec::new(),
            seq_atom_keys: Vec::new(),
            seq_atom_seeds: Vec::new(),
            atom_posteriors: std::collections::HashMap::new(),
            primary_estimand: None,
            unevaluable_weight: 0.0,
            weighted_ate: 0.0,
            total_w: 0.0,
            diagnostics,
        }
    }
}

/// Multi-step sustained fit of a class envelope: a sequential g-computation per
/// directed completion, mixed by the completion enumeration (frequentist) or by
/// the caller's class prior (Bayesian).
pub(super) fn fit_class_sequential(
    inputs: &TemporalClassEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalClassEffectFit, CausalError> {
    let TemporalClassEffectInputs { bundle, .. } = *inputs;
    if inputs.split.is_some() {
        return Err(CausalError::Unsupported {
            message: "class-aware multi-step sustained requires no discovery-estimation split",
        });
    }
    let envelope = &bundle.envelope.envelope;
    let class_masses =
        inputs.class_prior.map(|prior| prior.masses_for_envelope(&bundle.envelope)).transpose()?;
    let bayes = if let InferenceMode::Bayesian(cfg) = inputs.inference {
        if cfg.has_transfer() {
            return Err(CausalError::Unsupported {
                message: "multi-step Sequence transfer stays refused on incomplete classes",
            });
        }
        Some(bayesian_gcomp(cfg, ctx))
    } else {
        None
    };
    let mut diagnostics = vec![temporal_class_envelope_diagnostic(envelope, inputs.graph_class)];
    diagnostics.push(Diagnostic::new(
        "estimate.temporal.sustained_window",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "the contrast propagates through all intervened times on each directed completion; \
         no last-step collapse; bidirected MAG completions stay unevaluable",
    ));
    let mut run = SequentialClassRun::new(envelope.cases.len(), diagnostics);
    for (i, (case, indexer)) in
        envelope.cases.iter().zip(bundle.envelope.indexers.iter()).enumerate()
    {
        let weight = class_masses.as_ref().map_or(case.weight.0, |masses| masses[i]);
        fit_sequential_completion(
            inputs,
            bayes.as_ref(),
            &mut run,
            SequentialClassSite { index: i, case, indexer, weight },
            ctx,
        )?;
    }
    if run.seq_atoms.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal class sequential envelope has no evaluable directed completions"
                .into(),
        });
    }
    let estimand = run.primary_estimand.take().ok_or_else(|| CausalError::Compile {
        message: "temporal class sequential envelope has no primary estimand".into(),
    })?;
    let weight_basis = if class_masses.is_some() {
        crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
    } else {
        crate::result::StructuralWeightBasis::CompletionEnumeration
    };
    let (estimate, mut posterior, identified_set_interval) =
        mix_sequential_class_atoms(inputs, bayes.as_ref(), class_masses.is_some(), &mut run, ctx)?;
    let (refutations, extra_diagnostics, predictive_checks) = validate_sequential_class_atoms(
        inputs,
        bayes.as_ref(),
        &estimate,
        &mut run,
        &mut posterior,
        ctx,
    )?;
    run.diagnostics.extend(extra_diagnostics);
    let mut structural = temporal_class_structural_mixture(
        envelope,
        weight_basis,
        class_masses.as_deref(),
        &run.atom_values,
    );
    structural.identified_set_interval = identified_set_interval;
    for atom in &mut structural.atoms {
        atom.posterior = run.atom_posteriors.remove(&atom.graph_key);
    }
    let total: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
    structural.unevaluable_mass = if total > 0.0 { run.unevaluable_weight / total } else { 0.0 };
    structural.unidentified_mass =
        (structural.unidentified_mass - structural.unevaluable_mass).max(0.0);
    Ok(TemporalClassEffectFit {
        estimand,
        estimator_id: EstimatorId::TemporalSequentialGcomp,
        bootstrap_replicates_ok: estimate.uncertainty.bootstrap.replicates_ok,
        cancelled: ctx.cancellation.is_cancelled(),
        estimate,
        refutations,
        diagnostics: run.diagnostics,
        posterior,
        structural,
        predictive_checks,
        point_only: false,
    })
}

/// Fits one directed completion's sequential g-computation and records its
/// value, posterior draws and validation atom; a completion that is not
/// identified or has no sequential DAG is flagged and skipped.
fn fit_sequential_completion(
    inputs: &TemporalClassEffectInputs<'_>,
    bayes: Option<&BayesianGComputationAte>,
    run: &mut SequentialClassRun,
    site: SequentialClassSite<'_>,
    ctx: &ExecutionContext,
) -> Result<(), CausalError> {
    let SequentialClassSite { index: i, case, indexer, weight } = site;
    let key = case.graph.fingerprint();
    run.keys.push(key);
    run.weights.push(weight);
    if !identification_status_ok_for_case(case.result.status) || case.result.estimands.is_empty() {
        run.flags.push(GraphIdentFlag::Unidentified);
        return Ok(());
    }
    let Some(dag) = case.graph.sequential_dag() else {
        run.flags.push(GraphIdentFlag::Unidentified);
        run.unevaluable_weight += weight;
        run.diagnostics.push(mag_sequential_unevaluable_diagnostic(key));
        return Ok(());
    };
    let estimand = select_estimand(&case.result, EstimatorId::TemporalSequentialGcomp)
        .or_else(|_| select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment))?;
    if run.primary_estimand.is_none() {
        run.primary_estimand = Some(estimand.clone());
    }
    run.flags.push(GraphIdentFlag::Identified);
    let mut assumptions = case.result.required_assumptions.clone();
    assumptions.push(sequential_linear_sem_assumption(UNFOLDED_LINEAR_SEM));
    let (estimate, posterior, mechanisms, seed) = fit_completion_mechanisms(
        inputs,
        bayes,
        (&run.seq_atoms, &run.seq_atom_seeds),
        (&dag, indexer, &estimand, case.result.status),
        (assumptions, key),
        ctx,
    )?;
    run.seq_atom_seeds.push(seed);
    run.atom_values[i] = Some(estimate.ate);
    run.weighted_ate += weight * estimate.ate;
    run.total_w += weight;
    if let Some(posterior) = posterior.as_ref() {
        run.atom_posteriors.insert(key, posterior.clone());
        if let Some(col) = posterior.effect_column() {
            if let Ok(draws) = posterior.draws.column(col) {
                run.per_graph.push(GraphEffectDraws {
                    graph_key: key,
                    effect_draws: Arc::from(draws.to_vec()),
                });
            }
        }
    }
    run.seq_atom_keys.push(key);
    run.seq_atoms.push(SequentialValidationAtom {
        weight,
        graph: dag,
        indexer: indexer.clone(),
        estimand,
        status: case.result.status,
        estimate,
        mechanisms,
    });
    Ok(())
}

/// A completion's sequential DAG, unfold indexer, estimand and status.
type CompletionDesign<'a> = (
    &'a TemporalDag,
    &'a antecedent_core::TemporalIndexer,
    &'a IdentifiedEstimand,
    IdentificationStatus,
);

/// A completion's estimate, posterior, fitted mechanisms and the seed it drew on.
type CompletionMechanismFit =
    (EffectEstimate, Option<CausalPosterior>, Vec<SequentialBayesianMechanism>, u64);

/// One completion's sustained-window fit and posterior. Each completion's
/// posterior draws from its own seed stream; a completion whose fitted
/// mechanisms coincide with an earlier one's is refit on that completion's
/// seed, so one model carries one posterior.
fn fit_completion_mechanisms(
    inputs: &TemporalClassEffectInputs<'_>,
    bayes: Option<&BayesianGComputationAte>,
    (seq_atoms, seq_atom_seeds): (&[SequentialValidationAtom], &[u64]),
    (dag, indexer, estimand, status): CompletionDesign<'_>,
    (assumptions, key): (antecedent_core::AssumptionSet, u64),
    ctx: &ExecutionContext,
) -> Result<CompletionMechanismFit, CausalError> {
    let TemporalClassEffectInputs { data, query, bootstrap_replicates, gaps, .. } = *inputs;
    let mut seed = completion_fit_seed(ctx, key);
    loop {
        let mut mechanisms = Vec::new();
        let completion_bayes = bayes.map(|est| BayesianGComputationAte { seed, ..est.clone() });
        let (estimate, posterior, _) =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_gaps_if(
                &antecedent_estimate::SequentialProblem { data, graph: dag, indexer, estimand },
                query,
                antecedent_estimate::SequentialInference {
                    status,
                    assumptions: assumptions.clone(),
                    // The published frequentist interval bootstraps the atoms jointly
                    // below. An atom's own replicates only fix the dependence-aware
                    // block length its refuters resample on, so they are drawn only
                    // when a refuter will read it.
                    bootstrap_replicates: if bayes.is_some() || inputs.refute != RefuteSuite::None {
                        bootstrap_replicates
                    } else {
                        0
                    },
                    bayesian: completion_bayes.as_ref(),
                },
                ctx,
                Some(&mut mechanisms),
                gaps,
            )
            .map_err(CausalError::from)?;
        let shared_seed = seq_atoms
            .iter()
            .zip(seq_atom_seeds)
            .find(|(atom, atom_seed)| {
                bayes.is_some()
                    && **atom_seed != seed
                    && same_fitted_mechanisms(&atom.mechanisms, &mechanisms)
            })
            .map(|(_, atom_seed)| *atom_seed);
        match shared_seed {
            Some(shared) => seed = shared,
            None => return Ok((estimate, posterior, mechanisms, seed)),
        }
    }
}

/// The class estimate of the evaluated completions: the class-prior posterior
/// mixture (Bayesian), or the frozen-weight mean with the joint block
/// bootstrap (frequentist), with the identified-set interval it carries.
fn mix_sequential_class_atoms(
    inputs: &TemporalClassEffectInputs<'_>,
    bayes: Option<&BayesianGComputationAte>,
    class_prior_supplied: bool,
    run: &mut SequentialClassRun,
    ctx: &ExecutionContext,
) -> Result<
    (EffectEstimate, Option<CausalPosterior>, Option<antecedent_estimate::IdentifiedSetInterval>),
    CausalError,
> {
    let TemporalClassEffectInputs { data, query, bundle, bootstrap_replicates, gaps, .. } = *inputs;
    let envelope = &bundle.envelope.envelope;
    let mut identified_set_interval = None;
    if bayes.is_some() {
        identified_set_interval = posterior_identified_set_interval(
            run.seq_atom_keys.iter().filter_map(|key| run.atom_posteriors.get(key)),
            data.row_count(),
            envelope.truncated_completions > 0,
        );
    }
    // Unevaluable (bidirected) completions are flagged unidentified for the
    // mixture, which never draws from them; their share is reported as
    // `unevaluable_mass`, not as unidentified mass.
    let unevaluable_share = run.unevaluable_weight / run.weights.iter().sum::<f64>();
    let (estimate, posterior) = if bayes.is_some() {
        if class_prior_supplied
            && envelope.truncated_completions == 0
            && run.seq_atoms.iter().any(|atom| atom.weight > 0.0)
        {
            let graphs = WeightedGraphSamples::new(
                std::mem::take(&mut run.weights),
                std::mem::take(&mut run.flags),
                std::mem::take(&mut run.keys),
            )?;
            let mut mixed = aggregate_effect_envelope(
                &graphs,
                &run.per_graph,
                InferenceDiagnostics::analytic("temporal_class_sequential"),
                EnvelopeOptions::default(),
            )
            .map_err(CausalError::from)?;
            mixed.unidentified_mass = (mixed.unidentified_mass - unevaluable_share).max(0.0);
            (effect_from_posterior(&mixed)?, Some(mixed))
        } else {
            run.diagnostics.push(Diagnostic::new(
                "estimate.envelope.response_posterior_not_mixed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "completion posteriors are retained as atoms; mixing requires caller-supplied \
                 class mass, positive evaluable mass, and uncapped class scope",
            ));
            (nan_effect(), None)
        }
    } else if run.total_w > 0.0 {
        let SequentialAtomBlock { block, identified_set_interval: interval, se_analytic } =
            sequential_atom_block(
                data,
                query,
                &run.seq_atoms,
                bootstrap_replicates,
                envelope.truncated_completions > 0,
                gaps,
                ctx,
            )?;
        identified_set_interval = interval;
        if block.se.is_finite() {
            run.diagnostics.extend(envelope_shared_block_diagnostics(
                run.total_w,
                envelope.unidentified_weight.0,
                &block,
            ));
        } else if run.seq_atoms.len() > 1 {
            run.diagnostics.push(envelope_se_omits_between_atom_variance());
        }
        (
            class_mixture_estimate(
                run.weighted_ate / run.total_w,
                se_analytic,
                &block,
                bootstrap_replicates,
                run.seq_atoms[0].estimate.provenance.assumptions.clone(),
                ctx,
            ),
            None,
        )
    } else {
        (nan_effect(), None)
    };
    run.diagnostics.extend(
        identified_set_interval.as_ref().into_iter().flat_map(identified_set_interval_diagnostics),
    );
    Ok((estimate, posterior, identified_set_interval))
}

/// Refutation reports, diagnostics and predictive checks of a sequential class fit.
type SequentialClassValidation =
    (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>, Vec<PredictiveCheckReport>);

/// The refutation and predictive-check results of the evaluated completions:
/// on the class posterior when the estimate is finite, otherwise on each
/// completion with its report labelled by completion.
fn validate_sequential_class_atoms(
    inputs: &TemporalClassEffectInputs<'_>,
    bayes: Option<&BayesianGComputationAte>,
    estimate: &EffectEstimate,
    run: &mut SequentialClassRun,
    posterior: &mut Option<CausalPosterior>,
    ctx: &ExecutionContext,
) -> Result<SequentialClassValidation, CausalError> {
    let TemporalClassEffectInputs { data, query, .. } = *inputs;
    let sims = crate::analysis::latency::predictive_check_sims(inputs.latency_mode);
    if estimate.ate.is_finite() {
        return super::sequential_validation::validate_sequential(
            SequentialValidation {
                data,
                query,
                suite: inputs.refute,
                custom: inputs.custom_validators,
                predictive_sims: sims,
                gaps: inputs.gaps,
            },
            &run.seq_atoms,
            bayes,
            posterior.as_mut(),
            estimate.ate,
            ctx,
        );
    }
    let mut reports = Vec::new();
    let mut extra = Vec::new();
    let mut checks = Vec::new();
    for (atom, key) in run.seq_atoms.iter().zip(&run.seq_atom_keys) {
        let (mut local_reports, local_extra, local_checks) =
            super::sequential_validation::validate_sequential(
                SequentialValidation {
                    data,
                    query,
                    suite: inputs.refute,
                    custom: inputs.custom_validators,
                    predictive_sims: sims,
                    gaps: inputs.gaps,
                },
                std::slice::from_ref(atom),
                bayes,
                run.atom_posteriors.get_mut(key),
                atom.estimate.ate,
                ctx,
            )?;
        for report in &mut local_reports {
            report.refuter = Arc::from(format!("completion.{key}.{}", report.refuter));
        }
        reports.extend(local_reports);
        extra.extend(local_extra);
        checks.extend(local_checks);
    }
    Ok((reports, extra, checks))
}
