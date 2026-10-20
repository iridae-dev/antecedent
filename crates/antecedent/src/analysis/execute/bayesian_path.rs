// SPDX-License-Identifier: MIT OR Apache-2.0

use super::class_envelope_se::SharedCircularBlockSe;
use super::sequential_validation::{
    SequentialValidation, SequentialValidationAtom, validate_sequential,
};
use super::temporal_effect_core::shared_gap_report;
use super::{
    Arc, AverageEffectQuery, BayesianGCompWorkspace, BayesianGComputationAte,
    BayesianGraphPosteriorPriorAnchor, BayesianSuiteContext, BayesianTemporalGcomp,
    BayesianValidation, BootstrapAccounting, CausalError, CausalPosterior, CausalQuery,
    ClassAtomOutcome, ClassEnvelopeRun, ConditionalLinearAdjustment, DBN_LINEAR_SEM, Dag,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EnvelopeAtomFit,
    EnvelopeOptions, EnvelopeRefutation, EnvelopeRefuteAtom, EstimationWorkspace, EstimatorId,
    ExecuteRun, ExecutionContext, GraphEffectDraws, GraphIdentFlag, GraphPosterior,
    GraphPosteriorAtomValue, IdentificationEnvelope, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifiedKeyWeights,
    IdentifierId, InferenceDiagnostics, InferenceMode, Instant, InteractiveSubsampleDrop,
    OverlapPolicy, PhysicalExecutionPlan, PosteriorMassSummary, PosteriorPredictiveCheck,
    PredictiveCheckReport, PreparedBayesianProblem, PriorPredictiveCheck, PriorSet, RefutationRun,
    RefuteSuite, ResponseValue, SUSTAINED_LINEAR_SEM, StaticEstimateInput,
    StaticEstimateWorkspaces, Strategy, StructuralAggregationPolicy, StudyResult, TableView,
    TabularData, TemporalAtomDesign, TemporalEffectQuery, TemporalIndexer,
    TemporalLinearAdjustment, TemporalMediationEstimate, TemporalRefutation, TimeSeriesData,
    ValidationSink, ValidationSuite, ValidatorId, VariableId, WeightedGraphSamples,
    aggregate_effect_envelope, bayesian_gcomp, bayesian_temporal_gcomp, effect_from_posterior,
    elapsed_ns, envelope_mass_diagnostic, envelope_refute_atoms, envelope_se_omission_diagnostic,
    estimate_static_effect, evaluate_bayesian_prior_sensitivity, first_key_positions,
    identification_from_cache_or, identification_status_ok_for_case, identified_envelope_keys,
    identify_static, interactive_subsample_graphs_accounted, mass_fields,
    maybe_interactive_envelope_subsample, mediation_mechanism_seed, merge_posterior_notes,
    mix_graph_posterior_identified_atoms, mix_prior_sensitivity_summaries, mix_static_envelope_se,
    mix_weighted_analytic_se, mixed_static_influence, most_conservative_identification_status,
    nan_effect, overlap_diagnostic, project_for_ate_estimate, projection_diagnostic,
    provenance_ids, push_conflict_diagnostics,
    push_graph_posterior_structural_aggregation_diagnostic, report_subsampled_out_mass,
    require_identified, resolve_bayesian_prior_with_conflict, resolve_envelope_prior_anchor,
    resolve_posterior_status, resolve_structural_aggregation, retain_envelope_assumptions,
    run_envelope_bayesian_full_validation, select_claim, select_estimand,
    sequential_linear_sem_assumption, shared_block_mixture_message,
    shared_circular_block_mixture_se, short_series_warning, static_aligned_influence,
    temporal_refuter_average, validator_not_applicable_diagnostics, with_conflict_summary,
    with_prior_sensitivity,
};
use crate::analysis::identification_cache::DbnIdentifyDemotion;
use crate::planner::RouteDefaults;

/// The Bayesian g-computation estimator id for an average or conditional effect.
fn bayesian_gcomp_estimator(conditional: bool) -> EstimatorId {
    if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp }
}

/// Borrowed inputs of the Bayesian g-computation point stage.
#[derive(Clone, Copy)]
struct BayesianGcompPointInput<'a> {
    data: &'a TabularData,
    estimand: &'a IdentifiedEstimand,
    query: &'a AverageEffectQuery,
    cfg: &'a crate::inference::BayesianConfig,
    status: IdentificationStatus,
    conditional: bool,
}

/// Fitted Bayesian g-computation point stage with the state validation reuses.
struct BayesianGcompPoint {
    est: BayesianGComputationAte,
    prep: PreparedBayesianProblem,
    ws: BayesianGCompWorkspace,
    posterior: CausalPosterior,
    estimate: EffectEstimate,
}

/// Borrowed inputs of the Bayesian g-computation validate stage.
struct BayesianGcompValidationInput<'a> {
    data: &'a TabularData,
    estimand: &'a IdentifiedEstimand,
    query: &'a AverageEffectQuery,
    estimate: &'a EffectEstimate,
    cfg: &'a crate::inference::BayesianConfig,
    est: &'a BayesianGComputationAte,
    prep: &'a PreparedBayesianProblem,
    status: IdentificationStatus,
    estimator_id: EstimatorId,
}

/// Fitted completions of a Bayesian class envelope, with the subsample accounting.
struct PagEnvelopeFits {
    full_graphs: WeightedGraphSamples,
    graphs: WeightedGraphSamples,
    subsample_notes: Vec<Diagnostic>,
    subsample_drop: InteractiveSubsampleDrop,
    outcomes: Vec<ClassAtomOutcome>,
    atoms: Vec<EnvelopeAtomFit>,
    per_graph: Vec<GraphEffectDraws>,
    primary_estimand: Option<IdentifiedEstimand>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
}

/// Borrowed inputs of the Bayesian class-envelope validate stage.
struct PagEnvelopeValidationInput<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    estimate: &'a EffectEstimate,
    cfg: &'a crate::inference::BayesianConfig,
    est: &'a BayesianGComputationAte,
    estimator_id: EstimatorId,
}

/// Identification cases of a class envelope as weighted-graph inputs, with the identified
/// atoms to fit and the first one's estimand.
fn pag_envelope_cases<G>(
    envelope: &IdentificationEnvelope<G>,
    estimator_id: EstimatorId,
) -> Result<PagEnvelopeCases, CausalError> {
    let mut weights = Vec::new();
    let mut flags = Vec::new();
    let mut keys = Vec::new();
    let mut fit_atoms = Vec::new();
    let mut primary_estimand: Option<IdentifiedEstimand> = None;
    for (i, case) in envelope.cases.iter().enumerate() {
        let key = i as u64 + 1;
        keys.push(key);
        weights.push(case.weight.0);
        if identification_status_ok_for_case(case.result.status)
            && !case.result.estimands.is_empty()
        {
            flags.push(GraphIdentFlag::Identified);
            let estimand = select_estimand(&case.result, estimator_id)?;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
            }
            fit_atoms.push((key, estimand, case.result.status));
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    Ok(PagEnvelopeCases { weights, flags, keys, fit_atoms, primary_estimand })
}

/// Scan of a class envelope's identification cases.
struct PagEnvelopeCases {
    weights: Vec<f64>,
    flags: Vec<GraphIdentFlag>,
    keys: Vec<u64>,
    fit_atoms: Vec<(u64, IdentifiedEstimand, IdentificationStatus)>,
    primary_estimand: Option<IdentifiedEstimand>,
}

/// Prepared designs, resolved priors, and the first prior conflict of a class envelope.
struct PagEnvelopePrepared {
    prepared: std::collections::HashMap<u64, PreparedBayesianProblem>,
    atom_priors: std::collections::HashMap<u64, Option<PriorSet>>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
}

/// Borrowed settings of the class-envelope posterior aggregation.
struct PagEnvelopeAggregation<'a> {
    conditional: bool,
    full_query: &'a CausalQuery,
    class_tag: &'a str,
}

/// Frozen-weight mixture posterior of the fitted class-envelope completions, coupled by
/// their sampling correlation when the influence functions align; the flag reports coupling.
fn aggregate_pag_envelope_posterior(
    data: &TabularData,
    query: &AverageEffectQuery,
    settings: &PagEnvelopeAggregation<'_>,
    fits: &mut PagEnvelopeFits,
    ctx: &ExecutionContext,
) -> Result<(CausalPosterior, bool), CausalError> {
    let PagEnvelopeAggregation { conditional, full_query, class_tag } = *settings;
    let PagEnvelopeFits { full_graphs, graphs, subsample_drop, atoms, per_graph, .. } = &*fits;
    // Completion weights here are a frozen enumeration of an equivalence
    // class, not data-updated graph probabilities, and the reported number
    // is the frozen-weight mixture functional. Publish its posterior (atoms
    // coupled by their sampling correlation on the shared rows) rather than
    // the BMA spread over completion-specific effects, which over-covers
    // the mixture functional whenever completions disagree.
    let coupling =
        static_envelope_atom_correlation(data, query, conditional, atoms, full_query, ctx);
    let mut posterior = match &coupling {
        Some(correlation) => antecedent_estimate::aggregate_mixture_functional_envelope(
            graphs,
            per_graph,
            correlation,
            InferenceDiagnostics::analytic(format!("{class_tag}_envelope")),
        ),
        None => aggregate_effect_envelope(
            graphs,
            per_graph,
            InferenceDiagnostics::analytic(format!("{class_tag}_envelope")),
            EnvelopeOptions::default(),
        ),
    }
    .map_err(CausalError::from)?;
    report_subsampled_out_mass(&mut posterior, full_graphs, subsample_drop);
    retain_envelope_assumptions(&mut posterior, atoms);
    if let Some(summary) = fits.envelope_conflict.take() {
        posterior = with_conflict_summary(posterior, summary);
    }
    Ok((posterior, coupling.is_some()))
}

/// Diagnostics of the Bayesian class envelope: identification's, the envelope's, whether
/// the completions' posteriors could be coupled, subsample notes, overlap, mass, conflict.
fn pag_envelope_diagnostics(
    identification: &IdentificationResult,
    envelope_diagnostic: Diagnostic,
    coupled: bool,
    subsample_notes: Vec<Diagnostic>,
    estimate: &EffectEstimate,
    posterior: &CausalPosterior,
    class_tag: &str,
) -> Vec<Diagnostic> {
    let mut diagnostics = identification.diagnostics.clone();
    diagnostics.push(envelope_diagnostic);
    diagnostics.push(if coupled {
        Diagnostic::new(
            "estimate.envelope.bayesian_mixture_functional",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "posterior draws describe the frozen-weight mixture functional over identified \
             completions (each completion's posterior, rank-coupled by the completions' \
             sampling correlation on shared rows); they are not a distribution over \
             completion-specific effects",
        )
    } else {
        Diagnostic::new(
            "estimate.envelope.bayesian_mixture_coupling_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "completion influence functions could not be aligned on shared rows; posterior \
             draws are the identified-completion BMA, whose spread includes between-completion \
             disagreement and is not an interval for the mixture functional",
        )
    });
    diagnostics.extend(subsample_notes);
    diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
    diagnostics.push(envelope_mass_diagnostic(format!("estimate.{class_tag}.envelope"), posterior));
    if let Some(cs) = posterior.conflict_summary.as_ref() {
        push_conflict_diagnostics(&mut diagnostics, cs);
    }
    diagnostics
}

/// Atoms of a Bayesian graph-posterior envelope after preparation and Interactive subsampling.
struct GraphPosteriorBayesianPrepared {
    graphs: WeightedGraphSamples,
    subsample_notes: Vec<Diagnostic>,
    subsample_drop: InteractiveSubsampleDrop,
    kept: Vec<(u64, IdentifiedEstimand, IdentificationResult)>,
    prepared: std::collections::HashMap<u64, PreparedBayesianProblem>,
    anchor: BayesianGraphPosteriorPriorAnchor,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
}

/// Fitted atoms of a Bayesian graph-posterior envelope and its primary estimand.
struct GraphPosteriorBayesianFits {
    per_graph: Vec<GraphEffectDraws>,
    atoms: Vec<EnvelopeAtomFit>,
    primary_estimand: Option<IdentifiedEstimand>,
    primary_identification: Option<IdentificationResult>,
}

/// Borrowed inputs of the Bayesian graph-posterior envelope mix.
#[derive(Clone, Copy)]
struct GraphPosteriorBayesianMixInput<'a> {
    graphs: &'a WeightedGraphSamples,
    identified_graphs: &'a WeightedGraphSamples,
    subsample_drop: &'a InteractiveSubsampleDrop,
}

/// Mixed Bayesian graph-posterior envelope: posterior, scalar estimate, and the mix policy.
struct GraphPosteriorBayesianMix {
    posterior: CausalPosterior,
    estimate: EffectEstimate,
    mixed: super::MixedGraphPosteriorPolicy,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    unidentified_mass: f64,
}

/// Borrowed inputs of the Bayesian graph-posterior validate stage.
struct GraphPosteriorBayesianValidationInput<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    cfg: &'a crate::inference::BayesianConfig,
    est: &'a BayesianGComputationAte,
    estimate_ate: f64,
    graph_dependent: bool,
}

/// Envelope aggregation of the fitted atoms plus the identified-atom scalar mix.
fn mix_graph_posterior_bayesian_envelope(
    input: &GraphPosteriorBayesianMixInput<'_>,
    envelope_conflict: Option<antecedent_prob::ConflictSummary>,
    fitted: &mut GraphPosteriorBayesianFits,
) -> Result<GraphPosteriorBayesianMix, CausalError> {
    let GraphPosteriorBayesianMixInput { graphs, identified_graphs, subsample_drop } = *input;
    let GraphPosteriorBayesianFits { per_graph, atoms, primary_estimand, primary_identification } =
        fitted;
    let (primary_estimand, primary_identification) =
        (primary_estimand.take(), primary_identification.take());
    let mut posterior = aggregate_effect_envelope(
        graphs,
        per_graph,
        InferenceDiagnostics::analytic("graph_posterior_envelope"),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    report_subsampled_out_mass(&mut posterior, identified_graphs, subsample_drop);
    retain_envelope_assumptions(&mut posterior, atoms);
    if let Some(summary) = envelope_conflict {
        posterior = with_conflict_summary(posterior, summary);
    }
    let mut estimate = effect_from_posterior(&posterior)?;
    let atom_values: Vec<GraphPosteriorAtomValue> = atoms
        .iter()
        .filter_map(|atom| {
            effect_from_posterior(&atom.posterior).ok().map(|summary| GraphPosteriorAtomValue {
                key: atom.key,
                weight: atom.weight,
                status: atom.status,
                estimand: atom.estimand.clone(),
                value: summary.ate,
            })
        })
        .collect();
    let unidentified_mass = identified_graphs.unidentified_mass();
    let mixed = mix_graph_posterior_identified_atoms(
        identified_graphs,
        &atom_values,
        unidentified_mass,
        0.0,
        subsample_drop.mass,
        0,
    );
    if mixed.mixable_scalar {
        estimate.ate = mixed.ate;
    } else {
        estimate.ate = f64::NAN;
        estimate.uncertainty.analytic.se = f64::NAN;
    }
    let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: no identified graph atoms".into(),
    })?;
    identification.status = resolve_posterior_status(
        identification.status,
        mixed.policy,
        PosteriorMassSummary {
            unidentified: unidentified_mass.max(posterior.unidentified_mass),
            unevaluable: posterior.unevaluable_mass,
            subsampled: subsample_drop.mass.max(posterior.subsampled_out_mass),
            ..PosteriorMassSummary::default()
        },
    );
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: missing estimand".into(),
    })?;
    Ok(GraphPosteriorBayesianMix {
        posterior,
        estimate,
        mixed,
        identification,
        estimand,
        unidentified_mass,
    })
}

/// Diagnostics of the Bayesian graph-posterior envelope: identification's, subsample notes,
/// structural aggregation, overlap, mass, and prior conflict.
fn graph_posterior_bayesian_diagnostics(
    mix: &GraphPosteriorBayesianMix,
    subsample_notes: Vec<Diagnostic>,
    subsample_mass: f64,
) -> Vec<Diagnostic> {
    let unidentified_mass = mix.unidentified_mass;
    let mut diagnostics = mix.identification.diagnostics.clone();
    diagnostics.extend(subsample_notes);
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.structural_aggregation",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "policy={}; weight_basis=posterior_probability; identified_mass={}; \
                 unidentified_mass={unidentified_mass}; subsampled_out_mass={}",
                mix.mixed.policy.as_str(),
                mix.mixed.mixture.identified_mass,
                subsample_mass
            ),
        )
        .with_fields(mass_fields(Some(mix.mixed.mixture.identified_mass), mix.unidentified_mass)),
    );
    diagnostics.push(overlap_diagnostic(mix.estimate.provenance.overlap));
    diagnostics.push(envelope_mass_diagnostic("estimate.graph_posterior.envelope", &mix.posterior));
    if let Some(cs) = mix.posterior.conflict_summary.as_ref() {
        push_conflict_diagnostics(&mut diagnostics, cs);
    }
    diagnostics
}

/// The run-level settings a Frequentist graph-posterior effect reads: the study's own when the
/// study runs it, the sealed operation's when a checked operation does. Resolved once, so the
/// estimate and refute phases never re-decide which side to read.
struct EffectRunConfig<'a> {
    latency_mode: Option<super::LatencyMode>,
    refute: RefuteSuite,
    bootstrap_replicates: u32,
    overlap_policy: Option<OverlapPolicy>,
    population_registry: Option<&'a super::PopulationRegistry>,
    /// A sealed operation carries no custom validators.
    custom_validators: &'a [std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>],
    /// The estimator procedure every atom is fitted with.
    spec: crate::estimator_spec::EstimatorSpec,
}

/// Per-scenario posteriors of an unidentified within-tier order: each scenario keeps its own
/// posterior and interval, the scalar stays withheld, and no scenario weight is invented.
fn tiered_unknown_scenarios(
    fits: Vec<(CausalPosterior, EffectEstimate)>,
    identification: &IdentificationResult,
    rows: usize,
) -> Result<(EffectEstimate, crate::result::StructuralResponseMixture), CausalError> {
    let mut values = Vec::with_capacity(fits.len());
    let mut intervals = Vec::with_capacity(fits.len());
    let mut atoms = Vec::with_capacity(fits.len());
    let mut assumptions = identification.required_assumptions.clone();
    for (index, (posterior, estimate)) in fits.into_iter().enumerate() {
        let column = posterior.effect_column().ok_or_else(|| CausalError::Compile {
            message: "tiered Bayesian scenario posterior has no effect column".into(),
        })?;
        let mut draws = posterior
            .draws
            .column(column)
            .map_err(|error| CausalError::Compile { message: error.to_string() })?
            .to_vec();
        draws.sort_by(f64::total_cmp);
        let at = |q: f64| {
            let last = draws.len().saturating_sub(1);
            let position = (last as f64 * q).round();
            // The rounded position of a probability lies in [0, last].
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let position = position as usize;
            draws.get(position.min(last)).copied().unwrap_or(f64::NAN)
        };
        intervals.push((at(0.025), at(0.975)));
        values.push(estimate.ate);
        if index == 0 {
            assumptions = estimate.provenance.assumptions.clone();
        }
        atoms.push(crate::result::StructuralResponseAtom {
            graph_key: u64::try_from(index).unwrap_or(u64::MAX),
            weight: 1.0,
            status: identification.status,
            value: Some(antecedent_core::ResponseValue::Scalar(estimate.ate)),
            posterior: Some(posterior),
            response: None,
        });
    }
    let (lo, hi) = values
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let identified_set_interval = super::posterior_identified_set_interval(
        atoms.iter().filter_map(|atom| atom.posterior.as_ref()),
        rows,
        false,
    );
    let mixture = crate::result::StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::CompletionEnumeration,
        atoms,
        identified_mass: 1.0,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        subsampled_out_mass: 0.0,
        identified_set: Some(super::scalar_identified_set(lo, hi)),
        identified_set_interval,
        conditional_on_identified: None,
        full_mass_scope: true,
        truncated_atoms: 0,
    };
    let mut estimate =
        EffectEstimate::new(f64::NAN, f64::NAN, assumptions, OverlapPolicy::ExplicitOverride);
    estimate.scenarios.effects = Some(values.into());
    estimate.scenarios.intervals = Some(intervals.into());
    Ok((estimate, mixture))
}

impl<'a> EffectRunConfig<'a> {
    fn resolve(
        study: &'a super::Study,
        checked: Option<&'a super::super::CheckedGraphPosteriorEffect>,
        estimator_id: EstimatorId,
    ) -> Self {
        match checked {
            Some(operation) => Self {
                latency_mode: operation.latency_mode(),
                refute: operation.validation(),
                bootstrap_replicates: operation.bootstrap_replicates(),
                overlap_policy: Some(operation.overlap()),
                population_registry: operation
                    .population_registry()
                    .or(study.population_registry.as_ref()),
                custom_validators: &[],
                spec: operation.procedure().clone(),
            },
            None => Self {
                latency_mode: study.latency_mode,
                refute: study.refute,
                bootstrap_replicates: study.bootstrap_replicates,
                overlap_policy: study.overlap_policy,
                population_registry: study.population_registry.as_ref(),
                custom_validators: &study.custom_validators,
                spec: study
                    .estimator_spec
                    .as_deref()
                    .cloned()
                    .unwrap_or(crate::estimator_spec::EstimatorSpec::Default(estimator_id)),
            },
        }
    }
}

/// Resolved inputs shared by the Frequentist graph-posterior estimate and refute phases.
#[derive(Clone, Copy)]
struct GraphPosteriorFrequentistRoute<'a> {
    data: &'a TabularData,
    query: &'a AverageEffectQuery,
    run: &'a EffectRunConfig<'a>,
    estimator_id: EstimatorId,
    conditional: bool,
}

/// Per-atom estimates of a Frequentist graph-posterior mixture before mixing.
struct GraphPosteriorFrequentistAtoms {
    se_items: Vec<(f64, f64)>,
    atom_ifs: Vec<Vec<f64>>,
    atom_weights: Vec<f64>,
    primary_estimand: Option<IdentifiedEstimand>,
    primary_identification: Option<IdentificationResult>,
    assumptions: antecedent_core::AssumptionSet,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
}

/// Mixed Frequentist graph-posterior estimate with the mass and SE bookkeeping it reports.
struct GraphPosteriorFrequentistMix {
    mixed: super::MixedGraphPosteriorPolicy,
    estimate: EffectEstimate,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    unidentified_mass: f64,
    subsampled_out_mass: f64,
    n_contributing: usize,
    joint: bool,
    refute_atoms: Vec<EnvelopeRefuteAtom>,
}

/// Frozen-weight mix of the per-atom estimates: scalar, SE (joint influence-function
/// covariance when every contributing atom aligned), and the resolved identification status.
fn mix_graph_posterior_frequentist_atoms(
    identified_graphs: &WeightedGraphSamples,
    subsampled_out_mass: f64,
    atoms: GraphPosteriorFrequentistAtoms,
) -> Result<GraphPosteriorFrequentistMix, CausalError> {
    let GraphPosteriorFrequentistAtoms {
        se_items,
        atom_ifs,
        atom_weights,
        primary_estimand,
        primary_identification,
        assumptions,
        refute_atoms,
    } = atoms;
    if !matches!(
        refute_atoms.iter().map(|a| a.weight).sum::<f64>().partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return Err(CausalError::Compile {
            message: "graph-posterior envelope: no identified graph atoms".into(),
        });
    }
    let mut identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: no identified graph atoms".into(),
    })?;
    let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
        message: "graph-posterior envelope: missing estimand".into(),
    })?;
    let atom_values: Vec<GraphPosteriorAtomValue> = refute_atoms
        .iter()
        .map(|atom| GraphPosteriorAtomValue {
            key: atom.key,
            weight: atom.weight,
            status: IdentificationStatus::NonparametricallyIdentified,
            estimand: atom.estimand.clone(),
            value: atom.original.ate,
        })
        .collect();
    let unidentified_mass = identified_graphs.unidentified_mass();
    let mixed = mix_graph_posterior_identified_atoms(
        identified_graphs,
        &atom_values,
        unidentified_mass,
        0.0,
        subsampled_out_mass,
        0,
    );
    let n_contributing = se_items.len();
    let joint = mixed.mixable_scalar && n_contributing > 1 && atom_ifs.len() == n_contributing;
    let se = if mixed.mixable_scalar {
        if joint {
            mix_static_envelope_se(&atom_ifs, &atom_weights)
        } else {
            mix_weighted_analytic_se(se_items)
        }
    } else {
        f64::NAN
    };
    let mut estimate =
        EffectEstimate::new(mixed.ate, se, assumptions, OverlapPolicy::ExplicitOverride);
    if joint {
        if let Some(inf) = mixed_static_influence(&atom_ifs, &atom_weights) {
            estimate.scores.influence = Some(inf);
        }
    }
    identification.status = resolve_posterior_status(
        identification.status,
        mixed.policy,
        PosteriorMassSummary {
            unidentified: unidentified_mass,
            subsampled: subsampled_out_mass,
            ..PosteriorMassSummary::default()
        },
    );
    Ok(GraphPosteriorFrequentistMix {
        mixed,
        estimate,
        identification,
        estimand,
        unidentified_mass,
        subsampled_out_mass,
        n_contributing,
        joint,
        refute_atoms,
    })
}

/// Diagnostics of the Frequentist graph-posterior mixture: identification's, subsample
/// notes, structural aggregation, overlap, SE provenance, and the envelope summary.
fn graph_posterior_frequentist_diagnostics(
    mix: &GraphPosteriorFrequentistMix,
    subsample_notes: Vec<Diagnostic>,
) -> Vec<Diagnostic> {
    let GraphPosteriorFrequentistMix {
        mixed,
        estimate,
        identification,
        unidentified_mass,
        subsampled_out_mass,
        n_contributing,
        joint,
        refute_atoms,
        ..
    } = mix;
    let (unidentified_mass, subsampled_out_mass, n_contributing, joint) =
        (*unidentified_mass, *subsampled_out_mass, *n_contributing, *joint);
    let mut diagnostics = identification.diagnostics.clone();
    diagnostics.extend(subsample_notes);
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.structural_aggregation",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "policy={}; weight_basis=posterior_probability; identified_mass={}; \
                 unidentified_mass={unidentified_mass}; subsampled_out_mass={subsampled_out_mass}",
                mixed.policy.as_str(),
                mixed.mixture.identified_mass,
            ),
        )
        .with_fields(mass_fields(Some(mixed.mixture.identified_mass), unidentified_mass)),
    );
    diagnostics.push(overlap_diagnostic(estimate.provenance.overlap));
    diagnostics.extend(envelope_se_omission_diagnostic(n_contributing, estimate.se_analytic()));
    if joint && estimate.se_analytic().is_finite() {
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.joint_if_se",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "SE of E[τ | identified] from the joint influence-function covariance of \
                 {n_contributing} atoms fit on shared rows, with frozen graph weights; \
                 graph-weight uncertainty and unidentified mass are not in the SE; the \
                 interval is for the reported aggregate, not a distribution over \
                 graph-specific effects"
            ),
        ));
    }
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            if mixed.mixable_scalar {
                format!(
                    "published effect is E[τ | identified]; identified_mass={}, \
                     unidentified_mass={unidentified_mass}, subsampled_out_mass={subsampled_out_mass}, \
                     atoms={}",
                    mixed.mixture.identified_mass,
                    refute_atoms.len()
                )
            } else {
                format!(
                    "scalar withheld: contributing atoms disagree on estimand identity; \
                     identified_mass={}, unidentified_mass={unidentified_mass}, \
                     subsampled_out_mass={subsampled_out_mass}, atoms={}",
                    mixed.mixture.identified_mass,
                    refute_atoms.len()
                )
            },
        )
        .with_fields(mass_fields(
            Some(mixed.mixture.identified_mass),
            unidentified_mass,
        )),
    );
    diagnostics
}

/// Per-atom designs and estimates of a DBN frequentist mixture before the shared-block SE.
struct DbnFrequentistAtoms<'a> {
    contexts: Vec<(&'a super::super::prepared::CachedDbnPosteriorAtomIdentification, f64)>,
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
        .with_fields(super::mass_fields(Some(point_mass), unidentified_mass)),
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

/// Why a DBN atom was demoted before contributing draws.
#[derive(Clone, Copy)]
enum DbnAtomDemotion {
    Prepare,
    Fit,
    Draws,
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

/// A fit refused under draw controls keeps the route records the sampler measured.
///
/// R-hat, posterior effective sample size, convergence, and a computational
/// failure stay the records they are. This does not rewrite them as one
/// convergence reason.
pub(super) fn chain_refusal_under_draw_controls(
    controlled: bool,
    cfg: &crate::BayesianConfig,
    error: antecedent_estimate::EstimationError,
) -> CausalError {
    let _ = (controlled, cfg.backend);
    CausalError::from(error)
}

/// Refusal for an inference mode this route does not perform.
///
/// The record is unsupported inference, not a failed check or a computational
/// failure. The refusal reports no mass, so the mass is unreported.
pub(super) fn unsupported_inference_error(summary: &str) -> CausalError {
    CausalError::unsupported_inference(summary)
}

/// Refuse a posterior whose interval endpoints are not set by the posterior: the Monte Carlo
/// standard error of the larger endpoint must be within the accepted fraction of the width.
/// The endpoint errors are recorded with the posterior's inference diagnostics.
pub(super) fn enforce_draw_controls(posterior: &mut CausalPosterior) -> Result<(), CausalError> {
    let error = antecedent_estimate::endpoint_monte_carlo_error(posterior)?;
    if let Some(record) = error.route_record(posterior.unevaluable_mass) {
        return Err(CausalError::Estimate(antecedent_estimate::EstimationError::Prob(
            antecedent_prob::ProbError::Route {
                summary: format!(
                    "the Monte Carlo standard error of an interval endpoint ({:.3e} lower, \
                     {:.3e} upper) is {:.3} of the interval width from {:.0} effective draws \
                     of {}, above the {} tolerance; use more posterior draws or a sampler that \
                     mixes better",
                    error.lower_mcse,
                    error.upper_mcse,
                    error.width_fraction,
                    error.effective_draws,
                    error.draws,
                    antecedent_estimate::count_fitted_posterior::MAX_ENDPOINT_MCSE_FRACTION
                ),
                diagnostics: vec![record],
            },
        )));
    }
    posterior.diagnostics.notes.push(error.note());
    Ok(())
}

/// Route records of the checks a published posterior passed, with its measured unevaluable mass.
///
/// The Monte Carlo error check is recorded only when draw controls ran it and the endpoints were
/// within tolerance. Records follow the fixed check order.
pub(super) fn passed_route_records(
    diagnostics: &antecedent_prob::InferenceDiagnostics,
    unevaluable_mass: f64,
) -> Vec<antecedent_core::diagnostic::RouteNativeDiagnostic> {
    use antecedent_core::diagnostic::{RouteCheck, RouteKind, RouteMass, RouteNativeDiagnostic};
    let mass = RouteMass::new(unevaluable_mass).unwrap_or_else(|_| RouteMass::unreported());
    let mut records = diagnostics.passed_route_records(mass);
    if !records.is_empty()
        && antecedent_estimate::endpoint_mcse_from_notes(&diagnostics.notes)
            .is_some_and(|error| !error.exceeds_tolerance())
    {
        records.push(RouteNativeDiagnostic::check(
            RouteKind::MonteCarloError,
            RouteCheck::new(false, mass),
        ));
    }
    records
}

/// Append the passed route records of a published posterior to `diagnostics`.
pub(super) fn push_passed_route_diagnostics(
    diagnostics: &mut Vec<Diagnostic>,
    posterior: &CausalPosterior,
) {
    diagnostics.extend(
        passed_route_records(&posterior.diagnostics, posterior.unevaluable_mass)
            .into_iter()
            .map(antecedent_core::diagnostic::RouteNativeDiagnostic::to_diagnostic),
    );
}

impl super::Study {
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        checked_operation: Option<&super::super::prepared::CheckedBayesianGcompOperation>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut clock = super::super::stage::StageClock::new();
        let identifier_id = physical.binding().identifier_or(RouteDefaults::STATIC_ATE)?;
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };
        let (identification, estimand, identify_cached) = self.identify_bayesian_gcomp(
            graph,
            query,
            checked_operation,
            identifier_id,
            &mut clock,
            ctx,
        )?;

        let full_cols = data.schema().len();
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let projected_cols = data_est.schema().len();

        let inference = checked_operation.map_or(&self.inference, |operation| &operation.inference);
        let cfg = match inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(unsupported_inference_error(
                    "execute_bayesian requires inference=Bayesian; \
                     estimator bayesian.gcomp cannot run under Frequentist",
                ));
            }
        };
        let BayesianGcompPoint { est, prep, mut ws, posterior, estimate } = self
            .fit_bayesian_gcomp_point(
                &BayesianGcompPointInput {
                    data: &data_est,
                    estimand: &estimand_est,
                    query: &query_est,
                    cfg: &cfg,
                    status: identification.status,
                    conditional,
                },
                &mut clock,
                ctx,
            )?;

        let mut extra_diagnostics = Vec::new();
        if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
            extra_diagnostics.push(d);
        }
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut extra_diagnostics, cs);
        }
        push_passed_route_diagnostics(&mut extra_diagnostics, &posterior);

        clock.begin(ctx, super::super::stage::STAGE_VALIDATE, 0.8)?;
        let (posterior, refutations, predictive_checks) = self.validate_bayesian_gcomp(
            &BayesianGcompValidationInput {
                data: &data_est,
                estimand: &estimand_est,
                query: &query_est,
                estimate: &estimate,
                cfg: &cfg,
                est: &est,
                prep: &prep,
                status: identification.status,
                estimator_id,
            },
            posterior,
            &mut ws,
            &mut extra_diagnostics,
            ctx,
        )?;
        clock.finish(super::super::stage::STAGE_VALIDATE);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Validate {
                refutations: refutations.clone(),
                predictive_checks: predictive_checks.clone(),
            },
        );

        let n_draws = u32::try_from(posterior.draws.n_draws).ok();
        let early_stopped = posterior.early_stopped;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(extra_diagnostics)
            .refutations(refutations)
            .wall_time_ns(clock.wall_time_ns())
            .cancelled(clock.cancelled())
            .early_stopped(early_stopped)
            .extras(IdentifiedExecuteExtras {
                stage_timings_ns: clock.timings(),
                posterior: Some(posterior),
                n_draws,
                predictive_checks,
                // Posterior draws carry the uncertainty; no replicate budget ran.
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Bayesian average effect under a declared tier background.
    ///
    /// `CoDetermined` identifies one adjustment set (the tier closure), so the posterior is the
    /// ordinary Bayesian g-computation posterior of that identified functional. `Unknown`
    /// identifies two canonical orientation scenarios that are neither exhaustive nor
    /// weighted: each scenario keeps its own posterior of its own functional, none is mixed
    /// into another, no scenario weight is invented, and the scalar stays withheld. A prior
    /// alone never identifies the within-tier order.
    pub(crate) fn execute_tiered_bayesian_average(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        background: &antecedent_graph::TieredBackground,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let InferenceMode::Bayesian(cfg) = &self.inference else {
            return Err(CausalError::unsupported_inference(
                "tiered Bayesian average requires inference=Bayesian",
            ));
        };
        self.refuse_unsupported_tiered_average(query, cfg)?;
        let identifier_id = IdentifierId::GeneralizedAdjustment;
        let estimator_id = EstimatorId::BayesianGcomp;
        let (identification, _, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = antecedent_identify::identify_tiered(background, query)?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "tiered identification returned no estimand".into(),
                    }
                })?;
                Ok((identification, estimand))
            })?;
        let unknown = background.within_tier == antecedent_graph::WithinTier::Unknown;
        let mut clock = super::super::stage::StageClock::new();
        let mut fits =
            self.fit_tiered_scenarios(data, query, cfg, &identification, &mut clock, ctx)?;
        let Some(first_estimand) = fits.first().map(|(estimand, _, _)| estimand.clone()) else {
            return Err(CausalError::Compile {
                message: "tiered identification returned no estimand".into(),
            });
        };
        if !unknown {
            let Some((estimand, posterior, estimate)) = fits.pop() else {
                return Err(CausalError::Compile { message: "tiered fit missing".into() });
            };
            let n_draws = u32::try_from(posterior.draws.n_draws).ok();
            return Ok(self.finish_identified_execute(
                IdentifiedExecuteFinish::new(
                    physical,
                    identification,
                    estimand,
                    estimate,
                    Strategy::new(identifier_id, estimator_id),
                    query.treatment,
                    query.outcome,
                )
                .identify_cached(identify_cached)
                .wall_time_ns(elapsed_ns(started))
                .extras(IdentifiedExecuteExtras {
                    posterior: Some(posterior),
                    n_draws,
                    bootstrap_replicates_requested: Some(None),
                    ..Default::default()
                }),
            ));
        }
        let (estimate, mixture) = tiered_unknown_scenarios(
            fits.into_iter().map(|(_, posterior, estimate)| (posterior, estimate)).collect(),
            &identification,
            data.row_count(),
        )?;
        let diagnostic = Diagnostic::new(
            "tiered.unknown.bayesian_scenarios",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "each canonical scenario keeps its own posterior of its own functional (credible \
             intervals per scenario); the scenarios are not exhaustive completions and carry no \
             posterior weight, so none is mixed into another and the scalar stays withheld",
        );
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                first_estimand,
                estimate,
                Strategy::new(identifier_id, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(vec![diagnostic])
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                structural_response: Some(mixture),
                bootstrap_replicates_requested: Some(None),
                ..Default::default()
            }),
        ))
    }

    /// Fits the Bayesian g-computation posterior of every identified tiered estimand.
    fn fit_tiered_scenarios(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        cfg: &crate::inference::BayesianConfig,
        identification: &IdentificationResult,
        clock: &mut super::super::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(IdentifiedEstimand, CausalPosterior, EffectEstimate)>, CausalError> {
        let mut fits = Vec::new();
        for estimand in &identification.estimands {
            let (data_est, query_est, estimand_est) =
                project_for_ate_estimate(data, query, estimand)?;
            let point = self.fit_bayesian_gcomp_point(
                &BayesianGcompPointInput {
                    data: &data_est,
                    estimand: &estimand_est,
                    query: &query_est,
                    cfg,
                    status: identification.status,
                    conditional: false,
                },
                clock,
                ctx,
            )?;
            fits.push((estimand.clone(), point.posterior, point.estimate));
        }
        Ok(fits)
    }

    /// Refuses a tiered Bayesian average outside the `AllObserved` mean at validation none.
    fn refuse_unsupported_tiered_average(
        &self,
        query: &AverageEffectQuery,
        cfg: &crate::inference::BayesianConfig,
    ) -> Result<(), CausalError> {
        if self.refute != RefuteSuite::None
            || !self.custom_validators.is_empty()
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
        {
            return Err(CausalError::Unsupported {
                message: "a tiered Bayesian average is the AllObserved mean at validation none",
            });
        }
        if cfg.has_transfer() {
            return Err(CausalError::Unsupported {
                message: "a tiered Bayesian average uses the isotropic coefficient prior",
            });
        }
        Ok(())
    }

    /// Point and uncertainty stages of Bayesian g-computation: prepare the design, resolve the
    /// prior, fit, and record the outcome-model assumption, with the stage clock and events.
    fn fit_bayesian_gcomp_point(
        &self,
        input: &BayesianGcompPointInput<'_>,
        clock: &mut super::super::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<BayesianGcompPoint, CausalError> {
        let BayesianGcompPointInput {
            data: data_est,
            estimand: estimand_est,
            query: query_est,
            cfg,
            status,
            conditional,
        } = *input;
        let mut est = bayesian_gcomp(cfg, ctx);
        clock.begin(ctx, super::super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let prep = if conditional {
            let q = antecedent_core::ConditionalEffectQuery::try_new(query_est.clone())?;
            est.prepare_conditional(data_est, estimand_est, &q)
        } else {
            est.prepare(data_est, estimand_est, query_est)
        }
        .map_err(CausalError::from)?;
        let (resolved_prior, conflict_summary) =
            resolve_bayesian_prior_with_conflict(cfg, &prep, Some(ctx))?;
        est.prior = resolved_prior;
        let mut ws = BayesianGCompWorkspace::default();
        let mut posterior = est.fit(&prep, status, &mut ws, ctx).map_err(|error| {
            chain_refusal_under_draw_controls(self.draw_controls.is_some(), cfg, error)
        })?;
        if self.draw_controls.is_some() {
            enforce_draw_controls(&mut posterior)?;
        }
        if let Some(summary) = conflict_summary {
            posterior = with_conflict_summary(posterior, summary);
        }
        let likelihood =
            if est.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian {
                antecedent_prob::BayesLikelihood::GaussianIdentity
            } else {
                est.likelihood
            };
        posterior.assumptions.push(gcomp_outcome_model_assumption(likelihood, conditional));
        let estimate = effect_from_posterior(&posterior)?;
        clock.finish(super::super::stage::STAGE_ESTIMATE_POINT);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Point { estimate: estimate.clone() },
        );
        clock.begin(ctx, super::super::stage::STAGE_UNCERTAINTY, 0.55)?;
        clock.finish(super::super::stage::STAGE_UNCERTAINTY);
        super::super::stage::emit_uncertainty(self.stage_sink.as_ref(), &estimate);
        Ok(BayesianGcompPoint { est, prep, ws, posterior, estimate })
    }

    /// Identify stage of Bayesian g-computation: the prepared operation's frozen
    /// identification, or a (cached) fresh one, with the stage clock and event around it.
    fn identify_bayesian_gcomp(
        &self,
        graph: &Dag,
        query: &AverageEffectQuery,
        checked_operation: Option<&super::super::prepared::CheckedBayesianGcompOperation>,
        identifier_id: IdentifierId,
        clock: &mut super::super::stage::StageClock,
        ctx: &ExecutionContext,
    ) -> Result<(IdentificationResult, IdentifiedEstimand, bool), CausalError> {
        let estimator_id = if matches!(self.query, CausalQuery::ConditionalEffect(_)) {
            EstimatorId::BayesianConditional
        } else {
            EstimatorId::BayesianGcomp
        };
        clock.begin(ctx, super::super::stage::STAGE_IDENTIFY, 0.05)?;
        // Prepared handles identify once at prepare time; identification reads
        // only (identifier, graph, query) — rd is never consulted on this path —
        // all frozen there, so reuse is exact and observable via the
        // `exec.identify.cached` diagnostic below.
        if let Some(operation) = checked_operation {
            if operation.query != *query || !operation.inference.is_bayesian() {
                return Err(CausalError::Compile {
                    message: "prepared Bayesian g-computation binding changed after prepare".into(),
                });
            }
        }
        let (identification, estimand, identify_cached) = if let Some(operation) = checked_operation
        {
            (operation.identification().clone(), operation.estimand().clone(), true)
        } else {
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static(identifier_id, graph, query)?;
                select_claim(identification, estimator_id)
            })?
        };
        clock.finish(super::super::stage::STAGE_IDENTIFY);
        super::super::stage::emit_stage(
            self.stage_sink.as_ref(),
            &super::super::stage::StageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );
        Ok((identification, estimand, identify_cached))
    }

    /// Validate stage of Bayesian g-computation: refutation suite, prior / posterior
    /// predictive checks, and under the full suite prior sensitivity plus MCMC diagnostics.
    fn validate_bayesian_gcomp(
        &self,
        input: &BayesianGcompValidationInput<'_>,
        mut posterior: CausalPosterior,
        ws: &mut BayesianGCompWorkspace,
        extra_diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<
        (CausalPosterior, Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>),
        CausalError,
    > {
        let BayesianGcompValidationInput {
            data: data_est,
            estimand: estimand_est,
            query: query_est,
            estimate,
            cfg,
            est,
            prep,
            status,
            estimator_id,
        } = *input;
        let mut refute_ws = EstimationWorkspace::default();
        let (mut refutations, na_diagnostics) = match self.refute {
            RefuteSuite::None => (Vec::new(), Vec::new()),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => RefutationRun {
                data: data_est,
                estimand: estimand_est,
                query: query_est,
                estimate,
                suite: self.refute,
                estimator: estimator_id.as_str(),
                custom: &self.custom_validators,
                temporal: None,
            }
            .reports(&mut refute_ws, None, ctx)?,
        };
        extra_diagnostics.extend(na_diagnostics);
        // Prior + posterior PPC whenever refute is enabled (full PredictiveCheckReport retained).
        let mut predictive_checks = Vec::new();
        if !matches!(self.refute, RefuteSuite::None) {
            const PPC_ALPHA: f64 = 0.05;
            let sims = super::super::latency::predictive_check_sims(self.latency_mode);
            let ppc_prior = est.prior_in_force(prep.design.ncols);
            let prior_rep = PriorPredictiveCheck::for_estimator(est, ctx)
                .with_n_sims(sims)
                .check_with_prior(prep, &ppc_prior, ctx)
                .map_err(CausalError::from)?;
            refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(prior_rep);

            let post_rep = PosteriorPredictiveCheck::for_estimator(est, ctx)
                .with_n_sims(sims)
                .check(prep, &posterior)
                .map_err(CausalError::from)?;
            refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(post_rep);
        }
        // Prior sensitivity / MCMC stay behind the full suite (Shared Bayesian UX).
        // Mode-select: α-multiplier grid when an external composed prior is present;
        // isotropic scale grid otherwise (avoids clearing banked priors).
        if matches!(self.refute, RefuteSuite::Full) {
            let (summary, sens) =
                evaluate_bayesian_prior_sensitivity(cfg, est, prep, status, &posterior, ws, ctx)?;
            refutations.push(sens.to_report(&summary, estimate.ate));
            posterior = with_prior_sensitivity(posterior, summary);

            let suite = ValidationSuite::new().with(ValidatorId::McmcDiagnostics);
            let mut bayes_ctx =
                BayesianSuiteContext::new(est, prep, &posterior, status, ws, estimate.ate);
            let outcomes = suite.run_bayesian(&mut bayes_ctx, ctx).map_err(CausalError::from)?;
            refutations.extend(ValidationSuite::reports_only(&outcomes));
            extra_diagnostics.extend(validator_not_applicable_diagnostics(&outcomes));
        }
        Ok((posterior, refutations, predictive_checks))
    }

    pub(super) fn execute_pag_bayesian<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        run: ClassEnvelopeRun<'_, G>,
        identification: IdentificationResult,
        class_tag: super::pag_path::ClassTag,
    ) -> Result<StudyResult, CausalError> {
        let class_tag = class_tag.as_str();
        let ClassEnvelopeRun {
            run: ExecuteRun { physical, ctx, started, identify_cached },
            envelope,
            envelope_diagnostic,
        } = run;
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::unsupported_inference(
                    "execute_pag_bayesian requires inference=Bayesian",
                ));
            }
        };
        let est = bayesian_gcomp(&cfg, ctx);
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };

        let mut fits = self.fit_pag_envelope_atoms(data, query, envelope, &est, &cfg, ctx)?;
        let (mut posterior, coupled) = aggregate_pag_envelope_posterior(
            data,
            query,
            &PagEnvelopeAggregation { conditional, full_query: &self.query, class_tag },
            &mut fits,
            ctx,
        )?;
        let estimate = effect_from_posterior(&posterior)?;
        let estimand =
            fits.primary_estimand.take().or(envelope.invariant.clone()).ok_or_else(|| {
                CausalError::Compile {
                    message: format!("{class_tag} Bayesian envelope missing estimand"),
                }
            })?;

        let mut diagnostics = pag_envelope_diagnostics(
            &identification,
            envelope_diagnostic,
            coupled,
            std::mem::take(&mut fits.subsample_notes),
            &estimate,
            &posterior,
            class_tag,
        );

        let (refutations, predictive_checks) = self.validate_pag_envelope(
            &PagEnvelopeValidationInput {
                data,
                query,
                estimate: &estimate,
                cfg: &cfg,
                est: &est,
                estimator_id,
            },
            &fits.atoms,
            &mut posterior,
            &mut diagnostics,
            ctx,
        )?;
        // Completions that disagree publish the identified set over their
        // per-completion posterior means; a point-identified envelope stays a
        // point. The posterior remains the frozen-weight mixture functional's.
        let structural_response =
            super::pag_path::static_class_structural_mixture(envelope, &fits.outcomes);
        if let Some(mixture) = structural_response.as_ref() {
            diagnostics.extend(super::pag_path::static_class_identified_set_diagnostic(
                mixture, class_tag,
            ));
        }

        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralizedAdjustment, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                estimate_provenance: Some(provenance_ids(
                    "estimate.bayesian_gcomp",
                    "estimate.aggregate_effect_envelope",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                predictive_checks,
                structural_response,
                ..Default::default()
            }),
        ))
    }

    /// Per-completion fits of a class envelope under Bayesian g-computation: identified
    /// atoms are prepared and prior-resolved, Interactive-subsampled, then fitted in parallel.
    fn fit_pag_envelope_atoms<G>(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        envelope: &IdentificationEnvelope<G>,
        est: &BayesianGComputationAte,
        cfg: &crate::inference::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<PagEnvelopeFits, CausalError> {
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let estimator_id =
            if conditional { EstimatorId::BayesianConditional } else { EstimatorId::BayesianGcomp };

        let PagEnvelopeCases { weights, flags, keys, fit_atoms, primary_estimand } =
            pag_envelope_cases(envelope, estimator_id)?;
        let PagEnvelopePrepared { mut prepared, atom_priors, envelope_conflict } =
            self.prepare_pag_envelope_atoms(data, query, &fit_atoms, est, cfg, ctx)?;
        let full_graphs = WeightedGraphSamples::new(weights, flags, keys)?;
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            self.latency_mode,
            full_graphs.clone(),
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        let mut outcomes = vec![ClassAtomOutcome::NotEvaluated; envelope.cases.len()];
        let mut seen_keys = std::collections::HashSet::new();
        let mut skipped = Vec::new();
        let mut kept = Vec::new();
        for (key, estimand, status) in fit_atoms {
            let outcome_slot = usize::try_from(key).unwrap_or(usize::MAX).saturating_sub(1);
            if !keep.contains(&key) {
                skipped.push(outcome_slot);
                continue;
            }
            if !prepared.contains_key(&key) || !seen_keys.insert(key) {
                continue;
            }
            kept.push((key, estimand, status, outcome_slot));
        }
        for slot in skipped {
            outcomes[slot] = ClassAtomOutcome::SubsampledOut;
        }
        let fits = ctx.map_indexed(kept.len(), |i, inner| {
            let (key, estimand, status, outcome_slot) = &kept[i];
            let prep = prepared.get(key).ok_or_else(|| CausalError::Compile {
                message: "class atom was kept for fitting without a prepared problem".into(),
            })?;
            let mut est = est.clone();
            est.prior = atom_priors.get(key).cloned().flatten();
            let mut ws = BayesianGCompWorkspace::default();
            let posterior = est.fit(prep, *status, &mut ws, inner).map_err(CausalError::from)?;
            Ok::<_, CausalError>((
                *key,
                estimand.clone(),
                *status,
                *outcome_slot,
                posterior,
                est.prior.clone(),
            ))
        })?;
        let mut per_graph = Vec::new();
        let mut atoms = Vec::new();
        let key_weights = IdentifiedKeyWeights::new(&graphs);
        for (key, estimand, status, outcome_slot, posterior, prior) in fits {
            let Some(prep) = prepared.remove(&key) else {
                continue;
            };
            outcomes[outcome_slot] =
                ClassAtomOutcome::Evaluated(effect_from_posterior(&posterior)?.ate);
            per_graph.push(envelope_draws_from_posterior(key, &posterior)?);
            let weight = key_weights.get(key);
            atoms.push(EnvelopeAtomFit {
                key,
                prep,
                posterior,
                status,
                weight,
                estimand,
                indexer: None,
                prior,
            });
        }
        Ok(PagEnvelopeFits {
            full_graphs,
            graphs,
            subsample_notes,
            subsample_drop,
            outcomes,
            atoms,
            per_graph,
            primary_estimand,
            envelope_conflict,
        })
    }

    /// Prepare each identified completion once, before the Interactive subsample, and resolve
    /// its own prior; the first conflict summary stands for the envelope.
    fn prepare_pag_envelope_atoms(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        fit_atoms: &[(u64, IdentifiedEstimand, IdentificationStatus)],
        est: &BayesianGComputationAte,
        cfg: &crate::inference::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<PagEnvelopePrepared, CausalError> {
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        // Prepare once before Interactive subsample, stash
        // so kept atoms are not prepared a second time. PAG keys are unique
        // (1..n); still use entry() so the stash is key-safe.
        let mut prepared = std::collections::HashMap::with_capacity(fit_atoms.len());
        let mut atom_priors = std::collections::HashMap::with_capacity(fit_atoms.len());
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;
        let preps = ctx.map_indexed(fit_atoms.len(), |i, inner| {
            let (key, estimand, _) = &fit_atoms[i];
            let prep = if conditional {
                let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
                est.prepare_conditional(data, estimand, &q)
            } else {
                est.prepare(data, estimand, query)
            }
            .map_err(CausalError::from)?;
            // Per-completion filter: incompatible catalogs refuse rather than
            // sharing the first atom's prior or falling back to isotropic.
            let (resolved, conflict) =
                resolve_bayesian_prior_with_conflict(cfg, &prep, Some(inner))?;
            Ok::<_, CausalError>((*key, prep, resolved, conflict))
        })?;
        for (key, prep, resolved, conflict) in preps {
            if envelope_conflict.is_none() {
                envelope_conflict = conflict;
            }
            atom_priors.insert(key, resolved);
            prepared.entry(key).or_insert(prep);
        }
        Ok(PagEnvelopePrepared { prepared, atom_priors, envelope_conflict })
    }

    /// Validate stage of the Bayesian class envelope: refutation suite plus the
    /// envelope's prior / posterior predictive checks.
    fn validate_pag_envelope(
        &self,
        input: &PagEnvelopeValidationInput<'_>,
        atoms: &[EnvelopeAtomFit],
        posterior: &mut CausalPosterior,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>), CausalError>
    {
        let PagEnvelopeValidationInput { data, query, estimate, cfg, est, estimator_id } = *input;
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let (reports, mix_diagnostics) = EnvelopeRefutation {
                    data,
                    query,
                    suite: self.refute,
                    estimator: estimator_id.as_str(),
                    custom: &self.custom_validators,
                    temporal: None,
                }
                .run(&envelope_refute_atoms(atoms)?, &mut refute_ws, ctx)?;
                diagnostics.extend(mix_diagnostics);
                reports
            }
        };
        let predictive_checks = run_envelope_bayesian_full_validation(
            BayesianValidation { suite: self.refute, cfg, estimator: est },
            atoms,
            posterior,
            estimate.ate,
            ctx,
            super::super::latency::predictive_check_sims(self.latency_mode),
            ValidationSink { refutations: &mut refutations, diagnostics },
        )?;
        Ok((refutations, predictive_checks))
    }

    /// Execute a supplied static graph posterior into a Bayesian effect envelope.
    ///
    /// Identifies and fits each posterior atom independently; atoms that fail to
    /// identify (or fail to build a valid DAG from their adjacency mask) are marked
    /// [`GraphIdentFlag::Unidentified`] and contribute their posterior weight to
    /// [`antecedent_estimate::CausalPosterior::unidentified_mass`] rather than being
    /// dropped or having their mass redistributed onto the identified atoms.
    /// Identified atoms the Interactive tier leaves out of its subsample are
    /// reported as [`antecedent_estimate::CausalPosterior::subsampled_out_mass`]
    /// instead. Either kind of uncovered mass makes the result graph-dependent.
    ///
    /// # Errors
    ///
    /// `InferenceMode::Frequentist`, or any per-graph identification/estimation
    /// infrastructure failure (envelope aggregation, missing effect column, …).
    pub(in crate::analysis) fn execute_graph_posterior_bayesian(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::unsupported_inference(
                    "graph-posterior discovery requires inference=Bayesian for effect mixture",
                ));
            }
        };
        let est = bayesian_gcomp(&cfg, ctx);
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));

        let (identified, identify_cached) =
            self.graph_posterior_identification_for_bayesian(gp, query, ctx)?;
        let GraphPosteriorBayesianPrepared {
            graphs,
            subsample_notes,
            subsample_drop,
            kept,
            mut prepared,
            anchor,
            envelope_conflict,
        } =
            self.prepare_graph_posterior_bayesian_atoms(data, query, &identified, &est, &cfg, ctx)?;
        let mut fitted = Self::fit_graph_posterior_bayesian_atoms(
            &est,
            &graphs,
            &kept,
            &mut prepared,
            anchor,
            ctx,
        )?;
        let mix = mix_graph_posterior_bayesian_envelope(
            &GraphPosteriorBayesianMixInput {
                graphs: &graphs,
                identified_graphs: &identified.graphs,
                subsample_drop: &subsample_drop,
            },
            envelope_conflict,
            &mut fitted,
        )?;
        let mut diagnostics =
            graph_posterior_bayesian_diagnostics(&mix, subsample_notes, subsample_drop.mass);
        let GraphPosteriorBayesianMix {
            mut posterior,
            estimate,
            mixed,
            identification,
            estimand,
            ..
        } = mix;

        let (refutations, predictive_checks) = self.validate_graph_posterior_bayesian(
            &GraphPosteriorBayesianValidationInput {
                data,
                query,
                cfg: &cfg,
                est: &est,
                estimate_ate: estimate.ate,
                graph_dependent: matches!(
                    mixed.policy,
                    StructuralAggregationPolicy::GraphDependentAtoms
                ),
            },
            &fitted.atoms,
            &mut posterior,
            &mut diagnostics,
            ctx,
        )?;

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::BackdoorAdjustment,
                    bayesian_gcomp_estimator(conditional),
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.aggregate_effect_envelope",
                    "estimate.bayesian_gcomp",
                )),
                posterior: Some(posterior),
                diagnostics: Some(diagnostics),
                predictive_checks,
                structural_response: Some(mixed.mixture),
                ..Default::default()
            }),
        ))
    }

    /// The frozen graph-posterior identification: the prepared cache, or a fresh build.
    fn graph_posterior_identification_for_bayesian(
        &self,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<super::super::prepared::CachedGraphPosteriorIdentification>, bool), CausalError>
    {
        Ok(if let Some(cache) = self.graph_posterior_identification_cache.as_ref() {
            (Arc::clone(cache), true)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    gp, query, ctx,
                )?),
                false,
            )
        })
    }

    /// Prepare every identified atom's design once, resolve the shared prior anchor on the
    /// first atom, then Interactive-subsample and keep the atoms that still contribute draws.
    fn prepare_graph_posterior_bayesian_atoms(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        identified: &super::super::prepared::CachedGraphPosteriorIdentification,
        est: &BayesianGComputationAte,
        cfg: &crate::inference::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosteriorBayesianPrepared, CausalError> {
        let conditional = matches!(self.query, CausalQuery::ConditionalEffect(_));
        let graphs = identified.graphs.clone();
        let fit_atoms: Vec<_> = identified
            .atoms
            .iter()
            .map(|atom| (atom.key, atom.estimand.clone(), atom.identification.clone()))
            .collect();
        let original_anchor_key = fit_atoms.first().map(|(key, _, _)| *key);
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<antecedent_prob::ConflictSummary> = None;

        // Prepare once before Interactive subsample, stash
        // so kept atoms are not prepared a second time. Keys may collide when
        // several atoms share an adjacency mask — keep the first prep per key.
        let mut prepared = std::collections::HashMap::with_capacity(fit_atoms.len());
        let preps = ctx.map_indexed(fit_atoms.len(), |i, _inner| {
            let (key, estimand, _) = &fit_atoms[i];
            let prep = if conditional {
                let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
                est.prepare_conditional(data, estimand, &q)
            } else {
                est.prepare(data, estimand, query)
            }
            .map_err(CausalError::from)?;
            Ok::<_, CausalError>((*key, prep))
        })?;
        for (i, (key, prep)) in preps.into_iter().enumerate() {
            if i == 0 {
                let anchor =
                    super::BayesianGraphPosteriorPriorAnchor::resolve(key, cfg, &prep, ctx)?;
                envelope_prior = anchor.prior;
                envelope_conflict = anchor.conflict;
            }
            prepared.entry(key).or_insert(prep);
        }
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            self.latency_mode,
            graphs,
            ctx,
            &mut subsample_notes,
        )?;
        let keep = identified_envelope_keys(&graphs);
        let mut seen_keys = std::collections::HashSet::new();
        let kept: Vec<_> = fit_atoms
            .into_iter()
            .filter(|(key, _, _)| {
                keep.contains(key) && prepared.contains_key(key) && seen_keys.insert(*key)
            })
            .collect();
        let Some(anchor_key) = original_anchor_key else {
            return Err(CausalError::Compile {
                message:
                    "Bayesian graph-posterior envelope has no identified atom for prior anchoring"
                        .into(),
            });
        };
        let anchor = super::BayesianGraphPosteriorPriorAnchor {
            key: anchor_key,
            prior: envelope_prior.clone(),
            conflict: envelope_conflict.clone(),
        };
        Ok(GraphPosteriorBayesianPrepared {
            graphs,
            subsample_notes,
            subsample_drop,
            kept,
            prepared,
            anchor,
            envelope_conflict,
        })
    }

    /// Fit the kept atoms under the original prior anchor; the public estimand and
    /// identification come from the first atom that actually contributes draws.
    fn fit_graph_posterior_bayesian_atoms(
        est: &BayesianGComputationAte,
        graphs: &WeightedGraphSamples,
        kept: &[(u64, IdentifiedEstimand, IdentificationResult)],
        prepared: &mut std::collections::HashMap<u64, PreparedBayesianProblem>,
        anchor: BayesianGraphPosteriorPriorAnchor,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosteriorBayesianFits, CausalError> {
        let anchor_key = anchor.key;
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let key_weights = IdentifiedKeyWeights::new(graphs);
        let inputs = kept
            .iter()
            .map(|(key, estimand, identification)| {
                let prep = prepared.remove(key).ok_or_else(|| CausalError::Compile {
                    message: "kept Bayesian graph-posterior key has no prepared design".into(),
                })?;
                Ok(super::BayesianGraphPosteriorAtomInput {
                    key: *key,
                    weight: key_weights.get(*key),
                    status: identification.status,
                    estimand: estimand.clone(),
                    prepared: prep,
                })
            })
            .collect::<Result<Vec<_>, CausalError>>()?;
        let fitted = super::BayesianGraphPosteriorAtomFits::fit(est.clone(), anchor, inputs, ctx)?;
        if fitted.prior_anchor.key != anchor_key {
            return Err(CausalError::Compile {
                message: "Bayesian graph-posterior fit changed its original prior anchor".into(),
            });
        }
        let mut per_graph = Vec::new();
        let mut atoms = Vec::new();
        per_graph.extend(fitted.draw_columns.iter().cloned());
        for atom in fitted.atoms {
            let key = atom.key;
            let Some((_, estimand, identification)) =
                kept.iter().find(|(found, _, _)| *found == key)
            else {
                return Err(CausalError::Compile {
                    message: "Bayesian graph-posterior fit returned an unknown atom key".into(),
                });
            };
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                primary_identification = Some(identification.clone());
            }
            atoms.push(EnvelopeAtomFit {
                key,
                prep: atom.prepared,
                posterior: atom.posterior,
                status: atom.status,
                weight: atom.weight,
                estimand: atom.estimand,
                indexer: None,
                prior: atom.prior,
            });
        }
        Ok(GraphPosteriorBayesianFits {
            per_graph,
            atoms,
            primary_estimand,
            primary_identification,
        })
    }

    /// Validate stage of the Bayesian graph-posterior envelope: the refutation suite per atom
    /// group (graph-dependent atoms are refuted one by one) and the predictive checks.
    fn validate_graph_posterior_bayesian(
        &self,
        input: &GraphPosteriorBayesianValidationInput<'_>,
        atoms: &[EnvelopeAtomFit],
        posterior: &mut CausalPosterior,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<PredictiveCheckReport>), CausalError>
    {
        let GraphPosteriorBayesianValidationInput {
            data,
            query,
            cfg,
            est,
            estimate_ate,
            graph_dependent,
        } = *input;
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
                let refute_atoms = envelope_refute_atoms(atoms)?;
                let groups: Vec<&[EnvelopeRefuteAtom]> = if graph_dependent {
                    refute_atoms.chunks(1).collect()
                } else {
                    vec![&refute_atoms]
                };
                let mut reports = Vec::new();
                for group in groups {
                    let (atom_reports, mut notes) = EnvelopeRefutation {
                        data,
                        query,
                        suite: self.refute,
                        estimator: "bayesian.gcomp",
                        custom: &self.custom_validators,
                        temporal: None,
                    }
                    .run(group, &mut refute_ws, ctx)?;
                    if graph_dependent {
                        notes.retain(|d| d.code.as_ref() != "refute.envelope.effect_mixture");
                    }
                    diagnostics.extend(notes);
                    reports.extend(atom_reports);
                }
                if graph_dependent {
                    diagnostics.push(Diagnostic::new(
                        "refute.envelope.graph_dependent_atoms",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "outer scalar mix skipped under graph_dependent_atoms; Bayesian atom refutations remain conditional on their own estimands",
                    ));
                }
                reports
            }
        };
        let predictive_checks = if graph_dependent {
            let mut checks = Vec::new();
            for atom in atoms {
                let mut atom_posterior = atom.posterior.clone();
                let atom_effect = effect_from_posterior(&atom_posterior)?.ate;
                checks.extend(run_envelope_bayesian_full_validation(
                    BayesianValidation { suite: self.refute, cfg, estimator: est },
                    std::slice::from_ref(atom),
                    &mut atom_posterior,
                    atom_effect,
                    ctx,
                    super::super::latency::predictive_check_sims(self.latency_mode),
                    ValidationSink { refutations: &mut refutations, diagnostics },
                )?);
            }
            checks
        } else {
            run_envelope_bayesian_full_validation(
                BayesianValidation { suite: self.refute, cfg, estimator: est },
                atoms,
                posterior,
                estimate_ate,
                ctx,
                super::super::latency::predictive_check_sims(self.latency_mode),
                ValidationSink { refutations: &mut refutations, diagnostics },
            )?
        };
        Ok((refutations, predictive_checks))
    }

    /// Execute a supplied static graph posterior into a Frequentist effect mixture.
    ///
    /// Same atoms and unidentified-mass rule as
    /// [`Self::execute_graph_posterior_bayesian`]: identified atoms contribute
    /// `linear.adjustment.ate` point estimates, mixed as
    /// `E[τ | identified] = Σ w_i τ_i / identified_mass`. Unidentified mass is
    /// retained in the envelope diagnostic and is not redistributed.
    ///
    /// # Errors
    ///
    /// No identified atoms, or any per-graph identification/estimation failure.
    pub(super) fn execute_graph_posterior_frequentist(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_graph_posterior_frequentist_inner(data, gp, query, physical, None, ctx)
    }

    pub(in crate::analysis) fn execute_checked_graph_posterior_frequentist(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        operation: &super::super::CheckedGraphPosteriorEffect,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_graph_posterior_frequentist_inner(
            data,
            operation.posterior(),
            operation.query(),
            physical,
            Some(operation),
            ctx,
        )
    }

    fn execute_graph_posterior_frequentist_inner(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        checked: Option<&super::super::CheckedGraphPosteriorEffect>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if checked.is_none() && !self.inference.is_frequentist() {
            return Err(CausalError::unsupported_inference(
                "execute_graph_posterior_frequentist requires inference=Frequentist",
            ));
        }
        let conditional = checked.map_or_else(
            || matches!(self.query, CausalQuery::ConditionalEffect(_)),
            |operation| operation.target().is_conditional(),
        );
        let estimator_id = if let Some(operation) = checked {
            operation.estimator()
        } else if conditional {
            EstimatorId::ConditionalLinearAdjustment
        } else {
            EstimatorId::LinearAdjustmentAte
        };
        let run = EffectRunConfig::resolve(self, checked, estimator_id);
        let (identified, identify_cached) = if let Some(operation) = checked {
            (operation.identification().clone(), true)
        } else if let Some(cache) = self.graph_posterior_identification_cache.as_ref() {
            (Arc::clone(cache), true)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    gp, query, ctx,
                )?),
                false,
            )
        };
        let mut subsample_notes = Vec::new();
        let (graphs, subsample_drop) = interactive_subsample_graphs_accounted(
            run.latency_mode,
            identified.graphs.clone(),
            ctx,
            &mut subsample_notes,
        )?;
        let route =
            GraphPosteriorFrequentistRoute { data, query, run: &run, estimator_id, conditional };
        let atoms =
            Self::estimate_graph_posterior_frequentist_atoms(&route, &identified, &graphs, ctx)?;
        let mix =
            mix_graph_posterior_frequentist_atoms(&identified.graphs, subsample_drop.mass, atoms)?;
        let mut diagnostics = graph_posterior_frequentist_diagnostics(&mix, subsample_notes);
        let (refutations, na_diagnostics) = Self::refute_graph_posterior_frequentist(
            &route,
            matches!(mix.mixed.policy, StructuralAggregationPolicy::GraphDependentAtoms),
            &mix.refute_atoms,
            ctx,
        )?;
        diagnostics.extend(na_diagnostics);
        let GraphPosteriorFrequentistMix { mixed, estimate, identification, estimand, .. } = mix;

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::BackdoorAdjustment, estimator_id),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.linear_adjustment",
                    "estimate.linear_adjustment_ate",
                )),
                diagnostics: Some(diagnostics),
                structural_response: Some(mixed.mixture),
                ..Default::default()
            }),
        ))
    }

    /// Per-atom linear-adjustment estimates for the atoms the Interactive subsample kept,
    /// with each atom's weighted SE item and row-aligned influence function.
    fn estimate_graph_posterior_frequentist_atoms(
        route: &GraphPosteriorFrequentistRoute<'_>,
        identified: &super::super::prepared::CachedGraphPosteriorIdentification,
        graphs: &WeightedGraphSamples,
        ctx: &ExecutionContext,
    ) -> Result<GraphPosteriorFrequentistAtoms, CausalError> {
        let GraphPosteriorFrequentistRoute { data, query, run, conditional, .. } = *route;
        let keep = identified_envelope_keys(graphs);

        let mut se_items = Vec::new();
        let mut atom_ifs = Vec::new();
        let mut atom_weights = Vec::new();
        let mut primary_estimand = None;
        let mut primary_identification = None;
        let mut assumptions = antecedent_core::AssumptionSet::default();
        let mut refute_atoms = Vec::new();
        let kept: Vec<_> =
            identified.atoms.iter().filter(|atom| keep.contains(&atom.key)).collect();
        let estimates = ctx.map_indexed(kept.len(), |i, inner| {
            let atom = kept[i];
            let mut case_ws = StaticEstimateWorkspaces::default();
            let estimate = if conditional {
                let q = antecedent_core::ConditionalEffectQuery::try_new(query.clone())?;
                ConditionalLinearAdjustment::new()
                    .estimate(data, &atom.estimand, &q, inner)
                    .map_err(CausalError::from)
            } else {
                estimate_static_effect(
                    &run.spec,
                    StaticEstimateInput {
                        data,
                        estimand: &atom.estimand,
                        query,
                        bootstrap_replicates: run.bootstrap_replicates,
                        overlap_policy: run.overlap_policy,
                        population_registry: run.population_registry,
                    },
                    atom.identification.required_assumptions.clone(),
                    inner,
                    &mut case_ws,
                )
            }?;
            Ok::<_, CausalError>((atom, estimate))
        })?;
        let key_weights = IdentifiedKeyWeights::new(graphs);
        for (atom, estimate) in estimates {
            let w = key_weights.get(atom.key);
            se_items.push((w, estimate.se_analytic()));
            // Every atom is fit on the same rows; embed its IF in the original
            // row universe so the mixture SE carries cross-atom covariance.
            if let Some(inf) = estimate
                .scores
                .influence
                .as_deref()
                .and_then(|inf| static_aligned_influence(data, query, &atom.estimand, inf))
            {
                atom_ifs.push(inf);
                atom_weights.push(w);
            }
            if primary_estimand.is_none() {
                primary_estimand = Some(atom.estimand.clone());
                primary_identification = Some(atom.identification.clone());
                assumptions = estimate.provenance.assumptions.clone();
            }
            refute_atoms.push(EnvelopeRefuteAtom {
                key: atom.key,
                weight: w,
                estimand: atom.estimand.clone(),
                indexer: None,
                original: estimate,
            });
        }
        Ok(GraphPosteriorFrequentistAtoms {
            se_items,
            atom_ifs,
            atom_weights,
            primary_estimand,
            primary_identification,
            assumptions,
            refute_atoms,
        })
    }

    /// Refutation suite over the Frequentist graph-posterior atoms: per atom when the mix is
    /// graph-dependent, the pooled envelope refutation otherwise.
    fn refute_graph_posterior_frequentist(
        route: &GraphPosteriorFrequentistRoute<'_>,
        graph_dependent: bool,
        refute_atoms: &[EnvelopeRefuteAtom],
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let GraphPosteriorFrequentistRoute { data, query, run, estimator_id, .. } = *route;
        let estimator = estimator_id.as_str();
        let mut refute_ws = EstimationWorkspace::default();
        let (refutations, na_diagnostics) = if graph_dependent {
            let mut reports = Vec::new();
            let mut notes = Vec::new();
            for atom in refute_atoms {
                if atom.weight <= 0.0 {
                    continue;
                }
                let (mut per_atom, mut per_notes) = EnvelopeRefutation {
                    data,
                    query,
                    suite: run.refute,
                    estimator,
                    custom: run.custom_validators,
                    temporal: None,
                }
                .run(std::slice::from_ref(atom), &mut refute_ws, ctx)?;
                reports.append(&mut per_atom);
                notes.append(&mut per_notes);
            }
            notes.retain(|d| d.code.as_ref() != "refute.envelope.effect_mixture");
            notes.push(Diagnostic::new(
                "refute.envelope.class_posterior",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "outer scalar mix skipped because StructuralAggregationPolicy is \
                 graph_dependent_atoms — per-atom reports are retained and are not compared \
                 against a pooled mixture ATE",
            ));
            (reports, notes)
        } else {
            EnvelopeRefutation {
                data,
                query,
                suite: run.refute,
                estimator,
                custom: run.custom_validators,
                temporal: None,
            }
            .run(refute_atoms, &mut refute_ws, ctx)?
        };
        Ok((refutations, na_diagnostics))
    }

    /// Execute a supplied DBN posterior with fixed graph weights and a shared
    /// outer circular-block bootstrap across every contributing atom.
    pub(super) fn execute_dbn_posterior_frequentist(
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
                message: "multi-step sustained g-computation currently requires no discovery-estimation split",
            });
        }
        let vars = data.schema().variables().iter().map(|variable| variable.id).collect::<Vec<_>>();
        let (identified, identify_cached) =
            self.dbn_posterior_identification(gp, &vars, query, ctx)?;
        let gaps = self.admit_posterior_gaps(data, &vars, false)?;
        let mut atoms = Self::estimate_dbn_posterior_frequentist_atoms(
            data,
            gp,
            query,
            &vars,
            (&identified, gaps),
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
            .bootstrap_replicates_ok((self.bootstrap_replicates > 0).then_some(block.completed))
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
        result.missing_data = missing_data;
        Ok(result)
    }

    /// The frozen DBN-posterior identification: the prepared cache, or a fresh build.
    fn dbn_posterior_identification(
        &self,
        gp: &GraphPosterior,
        vars: &[VariableId],
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<super::super::prepared::CachedDbnPosteriorIdentification>, bool), CausalError>
    {
        Ok(if let Some(cache) = self.dbn_posterior_identification_cache.as_ref() {
            (Arc::clone(cache), true)
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
        (identified, gaps): (&'a super::super::prepared::CachedDbnPosteriorIdentification, bool),
        ctx: &ExecutionContext,
    ) -> Result<DbnFrequentistAtoms<'a>, CausalError> {
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
                TemporalAtomDesign::sequential_with_gaps(
                    data,
                    &crate::analysis::prepared::temporal_dag_from_dbn_atom(gp, atom.key, vars)?,
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
                    None,
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
            gp.max_lag.unwrap_or(1) as usize + 1,
            self.bootstrap_replicates,
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
                replicates_ok: (self.bootstrap_replicates > 0).then_some(block.completed),
                replicates_failed: (self.bootstrap_replicates > 0)
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
        let (refutations, diagnostics) =
            if self.refute == RefuteSuite::None && self.custom_validators.is_empty() {
                (Vec::new(), Vec::new())
            } else if query.is_multi_step_sustained() {
                let validation_atoms = contexts
                    .iter()
                    .zip(atom_estimates)
                    .map(|((atom, weight), original)| {
                        Ok(SequentialValidationAtom {
                            weight: *weight,
                            graph: crate::analysis::prepared::temporal_dag_from_dbn_atom(
                                gp, atom.key, vars,
                            )?,
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
                        suite: self.refute,
                        custom: &self.custom_validators,
                        predictive_sims: super::super::latency::predictive_check_sims(
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
                    suite: self.refute,
                    estimator: "temporal.linear.adjustment",
                    custom: &self.custom_validators,
                    temporal: Some(TemporalRefutation {
                        query,
                        split: self.split.as_ref(),
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
    pub(super) fn execute_dbn_posterior_bayesian(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let cfg = match &self.inference {
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
            self.dbn_posterior_identification(gp, &vars, query, ctx)?;
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
        result.missing_data = missing_data;
        Ok(result)
    }

    /// Multi-step sustained DBN inference needs no discovery-estimation split and isotropic
    /// per-mechanism priors.
    fn ensure_dbn_multi_step_bayesian_admitted(
        &self,
        cfg: &crate::inference::BayesianConfig,
    ) -> Result<(), CausalError> {
        if self.split.is_some() {
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
        identified: &super::super::prepared::CachedDbnPosteriorIdentification,
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
        for atom in identified.atoms.iter() {
            let atom = DbnFitAtom {
                key: atom.key,
                estimand: atom.estimand.clone(),
                identification: atom.identification.clone(),
                indexer: atom.indexer.clone(),
            };
            if run.multi_step {
                self.fit_dbn_sequential_atom(run, &mut state, atom, ctx);
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
        ctx: &ExecutionContext,
    ) {
        let DbnBayesianRun { data, gp, vars, query, sequential_bayes, gaps, .. } = *run;
        let DbnFitAtom { key, estimand, identification, indexer } = atom;
        let Ok(graph) = crate::analysis::prepared::temporal_dag_from_dbn_atom(gp, key, vars) else {
            state.demote(key, DbnAtomDemotion::Prepare);
            return;
        };
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(sequential_linear_sem_assumption(SUSTAINED_LINEAR_SEM));
        let mut mechanisms = Vec::new();
        let Ok((atom_estimate, Some(posterior), selection)) =
            antecedent_estimate::temporal_sequential::estimate_sustained_window_gaps_if(
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
                    bootstrap_replicates: self.bootstrap_replicates,
                    bayesian: sequential_bayes,
                },
                ctx,
                Some(&mut mechanisms),
                gaps,
            )
        else {
            state.demote(key, DbnAtomDemotion::Fit);
            return;
        };
        if let Ok(draws) = envelope_draws_from_posterior(key, &posterior) {
            state.windows.extend(selection);
            state.sequential_keys.push(key);
            state.sequential_atoms.push(SequentialValidationAtom {
                weight: state.key_weights.get(key),
                graph,
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
                .prepare(data, &estimand, query, &indexer, self.split.as_ref(), &ctx.kernel_policy)
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
                    suite: self.refute,
                    custom: &self.custom_validators,
                    predictive_sims: super::super::latency::predictive_check_sims(
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
                suite: self.refute,
                estimator: EstimatorId::BayesianTemporalGcomp.as_str(),
                custom: &self.custom_validators,
                temporal: Some(TemporalRefutation {
                    query,
                    split: self.split.as_ref(),
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
                BayesianValidation { suite: self.refute, cfg, estimator },
                atoms,
                posterior,
                estimate_ate,
                ctx,
                super::super::latency::predictive_check_sims(self.latency_mode),
                ValidationSink { refutations: &mut refutations, diagnostics },
            )?
        };
        Ok((refutations, predictive_checks))
    }

    /// Mix a DBN posterior into a Bayesian temporal-mediation envelope.
    ///
    /// Each atom uses that atom's per-horizon mediation set `S(h)` (its `I(h)`
    /// plus mediator/outcome confounders). Unidentified atoms
    /// keep their mass. Priors do not upgrade identification.
    pub(super) fn execute_dbn_posterior_mediation(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if query.horizons.len() > 1 && self.inference.is_frequentist() {
            return Err(CausalError::Unsupported {
                message: "Frequentist DBN-posterior mediation is licensed for one horizon; \
                          multi-horizon grids need their own joint uncertainty contract",
            });
        }
        if query.horizons.len() > 1 {
            return self
                .execute_dbn_posterior_mediation_multi_horizon(data, gp, query, physical, ctx);
        }
        if self.inference.is_frequentist() {
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
        let (cache, _) = self.dbn_posterior_mediation_identification(data, gp, query, ctx)?;
        let mut horizon_study = self.clone();
        horizon_study.dbn_posterior_identification_cache = Some(Arc::clone(&cache));

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
            for mut report in result.refutations {
                report.refuter = Arc::from(format!("horizon.{horizon}.{}", report.refuter));
                refutations.push(report);
            }
            predictive_checks.extend(result.predictive_checks);
            diagnostics.extend(result.diagnostics);
            if first.is_none() {
                first = Some((result.identification, result.estimand));
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
            .identify_cached(self.dbn_posterior_identification_cache.is_some())
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

    /// The frozen DBN-posterior mediation identification: the prepared cache, or a fresh build.
    fn dbn_posterior_mediation_identification(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<super::super::prepared::CachedDbnPosteriorIdentification>, bool), CausalError>
    {
        Ok(if let Some(cache) = self.dbn_posterior_identification_cache.as_ref() {
            (Arc::clone(cache), true)
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
        let cfg = match &self.inference {
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
            self.dbn_posterior_mediation_identification(data, gp, query, ctx)?;
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
        identified: &super::super::prepared::CachedDbnPosteriorIdentification,
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
        atom: &super::super::prepared::CachedDbnPosteriorAtomIdentification,
        ctx: &ExecutionContext,
    ) -> Option<DbnPublishedClick> {
        use antecedent_estimate::bayesian_mediation::prepare_temporal_mediation_adjusted;

        let DbnMediationRun { data, query, .. } = *run;
        let Some(horizons) = atom.horizons.as_ref() else {
            state.demote(atom.key, DbnAtomDemotion::Prepare);
            return None;
        };
        let Ok(clicks) =
            super::temporal_path::mediation_horizon_clicks(horizons, query, |horizon| {
                horizons
                    .get(horizon)
                    .map(super::temporal_path::lagged_adjustment_from_entry)
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
        state.horizon_dependent |= super::temporal_path::mediation_horizon_z_differs(&clicks);
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
        use antecedent_estimate::bayesian_mediation::compose_temporal_mediation;

        let DbnMediationRun { cfg, estimator, .. } = *run;
        let DbnPublishedClick { click, qh, preparations } = published;
        let fit = |scale: f64| -> Result<Vec<CausalPosterior>, CausalError> {
            preparations
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
                .collect()
        };
        let mechanisms = match fit(cfg.prior_scale) {
            Ok(mechanisms) => mechanisms,
            Err(error) => {
                state.demote_unevaluable(key, DbnAtomDemotion::Fit, &error);
                return;
            }
        };
        let composed = match compose_temporal_mediation(
            &mechanisms[0],
            &mechanisms[1],
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
        if self.refute != RefuteSuite::None {
            let (reports, notes) = mix_dbn_mediation_refuters(
                data,
                refute_atoms,
                self.refute == RefuteSuite::Full,
                ctx,
            )?;
            refutations.extend(reports);
            diagnostics.extend(notes);
        }
        let predictive_checks = run_dbn_mediation_bayesian_validation(
            BayesianValidation { suite: self.refute, cfg, estimator },
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
    click: super::temporal_path::MediationHorizonClick,
    qh: antecedent_core::MediationQuery,
    preparations: [PreparedBayesianProblem; 2],
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
        .with_fields(super::mass_fields(Some(0.0), 1.0)),
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
    identified: &super::super::prepared::CachedDbnPosteriorIdentification,
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
    preparations: [PreparedBayesianProblem; 2],
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
    full: bool,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let mut per_atom = Vec::new();
    for atom in atoms {
        if atom.weight <= 0.0 {
            continue;
        }
        let original = dbn_mediation_atom_estimate(atom)?;
        let reports = antecedent_validate::mediation::refute_temporal_mediation_adjusted(
            data,
            &atom.estimand,
            &atom.query,
            &original,
            full,
            &atom.adjustment,
            ctx,
        )
        .map_err(CausalError::from)?;
        per_atom.push((atom.weight, reports));
    }
    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    Ok(super::atom_mix::finish_atom_refutations(
        per_atom,
        Vec::new(),
        true,
        super::atom_mix::ScalarRefuterMix::Effect,
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
            let posts: Result<Vec<_>, _> = atom
                .preparations
                .iter()
                .enumerate()
                .map(|(i, prep)| {
                    let mut est = estimator.clone();
                    est.prior_scale = scale;
                    est.seed = mediation_mechanism_seed(
                        est.seed,
                        atom.query.horizons.first().copied().unwrap_or(0),
                        i,
                    );
                    est.fit(
                        prep,
                        atom.composed.identification,
                        &mut BayesianGCompWorkspace::default(),
                        ctx,
                    )
                    .map_err(CausalError::from)
                })
                .collect();
            let posts = posts?;
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
    use antecedent_estimate::bayesian_mediation::compose_temporal_mediation;
    compose_temporal_mediation(&posts[0], &posts[1], &atom.query, atom.composed.identification)
        .map_err(CausalError::from)
}

/// Declared outcome model behind a static `bayesian.gcomp` / conditional
/// posterior. The credible interval is for the treatment contrast of
/// that parametric regression; it is not a nonparametric interval for the
/// causal effect when the outcome mean is nonlinear in the adjustment set or
/// carries treatment × covariate interactions the design does not include
/// (`tests/static_calibration.rs::bayesian_gcomp_misspecification_probe`).
pub(super) fn gcomp_outcome_model_assumption(
    likelihood: antecedent_prob::BayesLikelihood,
    conditional: bool,
) -> antecedent_core::AssumptionRecord {
    let family = match likelihood {
        antecedent_prob::BayesLikelihood::GaussianIdentity => {
            "Gaussian identity-link outcome regression with homoskedastic residuals"
        }
        antecedent_prob::BayesLikelihood::BernoulliLogit => "Bernoulli logit outcome regression",
        antecedent_prob::BayesLikelihood::BernoulliProbit => "Bernoulli probit outcome regression",
        antecedent_prob::BayesLikelihood::PoissonLog => "Poisson log-link outcome regression",
    };
    let design = if conditional {
        "linear in the treatment, each declared modifier, each treatment × modifier product, \
         and each adjustment column; contrasts are evaluated on the outcome scale per draw"
    } else {
        "linear in the treatment and each adjustment column, with no treatment × covariate \
         interaction"
    };
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("bayesian.gcomp.linear_gaussian_outcome"),
                description: Arc::from(format!(
                    "{family}, {design}; the posterior and its credible interval are for the \
                     treatment contrast of this model. A nonlinear covariate term or an \
                     unmodelled treatment × covariate interaction biases that contrast and \
                     the interval does not cover the causal effect"
                )),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("bayesian.gcomp"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

pub(super) fn envelope_draws_from_posterior(
    key: u64,
    posterior: &CausalPosterior,
) -> Result<GraphEffectDraws, CausalError> {
    let col = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "Bayesian posterior missing effect column".into(),
    })?;
    let draws = posterior.draws.column(col)?;
    Ok(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws.to_vec()) })
}

/// Sampling correlation of static envelope atoms on their shared rows.
///
/// Each atom's Frequentist influence function (linear adjustment for the ATE,
/// the conditional linear functional for `ConditionalEffect`) is aligned to the
/// original rows; the correlation of those IFs couples the atoms' Bayesian
/// posteriors in [`antecedent_estimate::aggregate_mixture_functional_envelope`].
/// `None` when any atom lacks an aligned IF (the caller then keeps the BMA and
/// says so).
fn static_envelope_atom_correlation(
    data: &TabularData,
    query: &AverageEffectQuery,
    conditional: bool,
    atoms: &[EnvelopeAtomFit],
    full_query: &CausalQuery,
    ctx: &ExecutionContext,
) -> Option<Vec<f64>> {
    let mut ifs = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let estimate = if conditional {
            let CausalQuery::ConditionalEffect(cq) = full_query else {
                return None;
            };
            let mut mean_query = cq.clone();
            mean_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
            ConditionalLinearAdjustment::new()
                .estimate(data, &atom.estimand, &mean_query, ctx)
                .ok()?
        } else {
            let mut ws = StaticEstimateWorkspaces::default();
            estimate_static_effect(
                &crate::estimator_spec::EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
                StaticEstimateInput {
                    data,
                    estimand: &atom.estimand,
                    query,
                    bootstrap_replicates: 0,
                    overlap_policy: None,
                    population_registry: None,
                },
                antecedent_core::AssumptionSet::default(),
                ctx,
                &mut ws,
            )
            .ok()?
        };
        let aligned = static_aligned_influence(
            data,
            query,
            &atom.estimand,
            estimate.scores.influence.as_deref()?,
        )?;
        if aligned.iter().any(|v| !v.is_finite()) {
            return None;
        }
        ifs.push(aligned);
    }
    let k = ifs.len();
    let norms: Vec<f64> = ifs.iter().map(|f| f.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    if norms.iter().any(|n| !n.is_finite() || *n <= 0.0) {
        return None;
    }
    let mut corr = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            let dot: f64 = ifs[a].iter().zip(&ifs[b]).map(|(x, y)| x * y).sum();
            corr[a * k + b] = (dot / (norms[a] * norms[b])).clamp(-1.0, 1.0);
        }
    }
    Some(corr)
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
    }
}

#[cfg(test)]
mod passed_route_record_tests {
    use super::passed_route_records;
    use antecedent_prob::InferenceDiagnostics;

    fn summary(records: &[antecedent_core::diagnostic::RouteNativeDiagnostic]) -> Vec<(&str, u64)> {
        records
            .iter()
            .map(|record| {
                assert!(!record.failed());
                (record.route_name(), record.unevaluable_mass().bits().expect("a measured mass"))
            })
            .collect()
    }

    #[test]
    fn conjugate_and_laplace_posteriors_record_passed_checks_with_the_measured_mass() {
        let bits = 0.25_f64.to_bits();
        let conjugate = InferenceDiagnostics::analytic("conjugate_gaussian");
        assert_eq!(summary(&passed_route_records(&conjugate, 0.25)), [("convergence", bits)]);
        let laplace =
            InferenceDiagnostics::laplace("laplace", true, 4, 1e-6, 10.0, false, Vec::new());
        assert_eq!(summary(&passed_route_records(&laplace, 0.25)), [("convergence", bits)]);
    }

    #[test]
    fn an_unrefused_unmeasurable_mass_is_unreported_not_zero() {
        let conjugate = InferenceDiagnostics::analytic("conjugate_gaussian");
        let records = passed_route_records(&conjugate, f64::NAN);
        assert!(records.iter().all(|record| record.unevaluable_mass().is_unreported()));
    }

    #[test]
    fn a_fit_that_would_be_refused_records_no_passed_checks() {
        let unconverged =
            InferenceDiagnostics::laplace("laplace", false, 4, 1e-6, 10.0, false, Vec::new());
        assert!(passed_route_records(&unconverged, 0.0).is_empty());
    }
}

#[cfg(test)]
mod unsupported_inference_tests {
    use super::unsupported_inference_error;

    #[test]
    fn every_unified_refusal_carries_the_unsupported_inference_record() {
        for summary in [
            "execute_pag_bayesian requires inference=Bayesian",
            "execute_graph_posterior_frequentist requires inference=Frequentist",
            "bayesian.basis.gcomp requires inference=Bayesian",
            "bayesian.robust_ate requires Bayesian inference",
        ] {
            let error = crate::CausalError::unsupported_inference(summary);
            assert_eq!(error.to_string(), summary);
            let records = error.route_diagnostics();
            assert_eq!(records.len(), 1, "{summary}: {records:?}");
            assert!(records[0].is_unsupported_inference(), "{summary}");
            assert!(records[0].unevaluable_mass().is_unreported(), "{summary}");
        }
    }

    #[test]
    fn an_unsupported_inference_refusal_is_not_a_failed_check_or_a_computational_failure() {
        let error = unsupported_inference_error("inference mode is not performed by this route");
        let records = error.route_diagnostics();
        assert_eq!(records.len(), 1, "{records:?}");
        assert!(records[0].is_unsupported_inference());
        assert!(!records[0].failed());
        assert!(!records[0].is_computational_failure());
        assert!(records[0].unevaluable_mass().is_unreported());
    }
}
