//! CPDAG/PAG graph-posterior Response: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::atom_mix::{ScalarRefuterMix, append_unidentified_atoms, finish_atom_refutations};
use super::completion_fit_memo::{CompletionFitKey, CompletionFitMemo};
use super::response_path::{
    graph_posterior_response_supported, mix_response_values, mix_support_reports,
    response_envelope_from_weighted, response_identified_value, response_primary_pair,
    response_scalar_summary,
};
use super::{
    Arc, AtomRefutationPolicy, AverageEffectQuery, CausalError, CausalResponse,
    ContinuousResponseEstimator, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphPosterior, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedKeyWeights, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, PosteriorMassSummary, RefutationRun, RefuteSuite, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, Strategy, Study,
    StudyResult, TableView, TabularData, VariableId, bayesian_gcomp, elapsed_ns,
    identification_status_ok_for_case, identify_cached_diagnostic, provenance_ids,
    push_graph_posterior_structural_aggregation_diagnostic, resolve_posterior_status,
    resolve_structural_aggregation, response_path,
};
use crate::analysis::prepared::{
    CachedClassPosteriorAtomIdentification, CachedGraphPosteriorIdentification,
};
use crate::result::{
    StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
};

struct ClassAtomResponseEval {
    key: u64,
    status: IdentificationStatus,
    estimand: Option<IdentifiedEstimand>,
    response: Option<CausalResponse>,
    partial: bool,
    atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)>,
    plugin_levels: Vec<(f64, IdentifiedEstimand, f64, antecedent_core::AssumptionSet)>,
}

impl super::Study {
    /// Mix CPDAG/PAG posterior atoms through the existing class response envelope.
    pub(super) fn execute_class_graph_posterior_response(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        query: &ResponseQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        graph_posterior_response_supported(query)?;
        let class_tag = gp.atom_kind.as_str().to_ascii_lowercase();
        let witness = response_path::response_witness_ate(query)?;
        let (treatment, outcome) = response_primary_pair(&query.functional)?;
        let (identified, identify_cached) =
            self.class_graph_posterior_response_identification(gp, &witness, ctx)?;
        let options = self.response_options.clone().unwrap_or_default();
        let evals = self.evaluate_class_posterior_response_atoms(
            data,
            query,
            outcome,
            &identified,
            &options,
            ctx,
        )?;

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mut mixed = mix_class_posterior_responses(
            &identified,
            &evals,
            unidentified_mass,
            query,
            data.row_count(),
            options.confidence_level,
        )?;
        mixed.identification.status = resolve_posterior_status(
            mixed.identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                unevaluable: mixed.mixture.unevaluable_mass,
                failed: mixed.failed_mass,
                full_scope: mixed.mixture.full_mass_scope,
                ..PosteriorMassSummary::default()
            },
        );

        let (scalar, standard_error) = response_scalar_summary(&mixed.response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            mixed.response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let estimator_id = if self.inference.is_bayesian() {
            EstimatorId::ResponseBayesian
        } else {
            EstimatorId::default_for_response(&query.functional)
        };

        let mut diagnostics = self.class_graph_posterior_response_diagnostics(&mixed, &class_tag);
        let (refutations, refute_diags) = self.refute_class_graph_posterior_response(
            ClassResponseRefuteInput {
                data,
                query,
                identified: &identified,
                evals: &evals,
                mixed: &mixed,
                scalar,
                standard_error,
                estimator_id,
                treatment,
                outcome,
            },
            &mut diagnostics,
            ctx,
        )?;
        diagnostics.extend(refute_diags);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        diagnostics.extend(mixed.response.support.warnings.iter().cloned());

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let MixedClassPosteriorResponse {
            policy, identification, estimand, response, mixture, ..
        } = mixed;
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralizedAdjustment, estimator_id),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.class_graph_posterior_response",
                    policy.as_str(),
                )),
                response: Some(response),
                structural_response: Some(mixture),
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }

    /// The prepared class posterior identification cache, or a live build for the witness query.
    fn class_graph_posterior_response_identification(
        &self,
        gp: &GraphPosterior,
        witness: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<CachedGraphPosteriorIdentification>, bool), CausalError> {
        let (identified, identify_cached) = if let Some(cache) =
            self.graph_posterior_cache()
        {
            (Arc::clone(cache), true)
        } else {
            (
                Arc::new(crate::analysis::prepared::build_graph_posterior_identification_cache(
                    gp, witness, ctx,
                )?),
                false,
            )
        };
        if identified.class_atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "class graph-posterior response: no identified class atoms".into(),
            });
        }
        Ok((identified, identify_cached))
    }

    /// Evaluates every class posterior atom's response on the worker pool.
    fn evaluate_class_posterior_response_atoms(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        outcome: VariableId,
        identified: &CachedGraphPosteriorIdentification,
        options: &antecedent_estimate::ContinuousResponseOptions,
        ctx: &ExecutionContext,
    ) -> Result<Vec<ClassAtomResponseEval>, CausalError> {
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;
        let data_est = &data_est;
        ctx.map_indexed(identified.class_atoms.len(), |i, inner| {
            let atom = &identified.class_atoms[i];
            match &self.inference {
                InferenceMode::Frequentist => evaluate_class_atom_response_frequentist(
                    self, data_est, query, atom, options, inner,
                ),
                InferenceMode::Bayesian(_) => evaluate_class_atom_response_bayesian(
                    self, data_est, query, atom, options, inner,
                ),
            }
        })
    }

    /// Class-envelope, aggregation-policy and withheld-uncertainty disclosures of the response.
    fn class_graph_posterior_response_diagnostics(
        &self,
        mixed: &MixedClassPosteriorResponse,
        class_tag: &str,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = mixed.identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "estimate.response.class_graph_posterior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "each {class_tag} posterior atom was evaluated with the existing class response \
                 envelope; outer weights are posterior probabilities"
            ),
        ));
        push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            mixed.policy,
            mixed.mixture.identified_mass,
            mixed.mixture.unidentified_mass,
            mixed.mixture.unevaluable_mass,
            mixed.mixture.subsampled_out_mass,
        );
        if mixed.weighted.len() > 1 && mixed.mixed_if_se.is_none() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if self.inference.is_bayesian() {
                    "multi-atom Bayesian class-posterior response withholds an aggregate \
                     credible interval: graph-specific dispersion and a frozen-weight aggregate \
                     are different objects"
                } else if !matches!(mixed.conditional, Some(ResponseValue::Scalar(_))) {
                    "multi-atom Frequentist class-posterior curve withholds an aggregate band: \
                     the joint-IF SE is published for scalar aggregates only"
                } else if !matches!(
                    mixed.policy,
                    StructuralAggregationPolicy::SameEstimandWeightedMean
                ) {
                    "aggregate uncertainty withheld under GraphDependentAtoms: per-atom values \
                     and identified set are retained"
                } else {
                    "multi-atom Frequentist class-posterior response withholds an aggregate \
                     interval because aligned atom influences were unavailable"
                },
            ));
        } else if mixed.mixed_if_se.is_some() {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.joint_if_se",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "SE of the frozen-weight aggregate from the joint influence-function covariance \
                 of identified class-posterior atoms on the shared sample; simultaneous bands \
                 are not claimed",
            ));
        }
        diagnostics
    }

    /// Plugin-level refutation of a class graph-posterior response, per aggregate or per atom.
    fn refute_class_graph_posterior_response(
        &self,
        input: ClassResponseRefuteInput<'_>,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let ClassResponseRefuteInput {
            data,
            query,
            identified,
            evals,
            mixed,
            scalar,
            standard_error,
            estimator_id,
            treatment,
            outcome,
        } = input;
        let plugin_scalar = match &mixed.conditional {
            Some(ResponseValue::Scalar(value)) => *value,
            _ => scalar,
        };
        let scalar_intervention = plugin_scalar.is_finite()
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        if matches!(query.functional, ResponseFunctional::InterventionResponse { .. })
            && (!matches!(self.refute, RefuteSuite::None) || !self.custom_validators.is_empty())
        {
            if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; \
                     cheap runs overlap only and full runs overlap plus sampling-stability of the \
                     g-comp level",
                ));
            }
            if scalar_intervention
                && matches!(mixed.policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            {
                let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
                let mut refute_ws = EstimationWorkspace::default();
                let plugin_estimate = EffectEstimate::new(
                    plugin_scalar,
                    standard_error,
                    mixed.response.assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                );
                RefutationRun {
                    data,
                    estimand: &mixed.estimand,
                    query: &ate_query,
                    estimate: &plugin_estimate,
                    suite: self.refute,
                    estimator: estimator_id.as_str(),
                    custom: &self.custom_validators,
                    temporal: None,
                }
                .plugin_level(&mut refute_ws, ctx)
            } else {
                plugin_level_on_class_response_atoms(
                    self,
                    data,
                    identified,
                    evals,
                    (treatment, outcome),
                    ctx,
                    AtomRefutationPolicy {
                        aggregation: mixed.policy,
                        estimator: estimator_id.as_str(),
                    },
                )
            }
        } else {
            if !matches!(self.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                    "refute.response.skipped",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    if matches!(mixed.policy, StructuralAggregationPolicy::GraphDependentAtoms) {
                        "query-native validation does not compare refuters against a withheld \
                         aggregate under GraphDependentAtoms"
                    } else {
                        "scalar ATE refuters are not applicable to a function-valued \
                         class-posterior response"
                    },
                ));
            }
            Ok((Vec::new(), Vec::new()))
        }
    }
}

/// Borrowed inputs of the plugin-level refutation of a class graph-posterior response.
#[derive(Clone, Copy)]
struct ClassResponseRefuteInput<'a> {
    data: &'a TabularData,
    query: &'a ResponseQuery,
    identified: &'a CachedGraphPosteriorIdentification,
    evals: &'a [ClassAtomResponseEval],
    mixed: &'a MixedClassPosteriorResponse,
    scalar: f64,
    standard_error: f64,
    estimator_id: EstimatorId,
    treatment: VariableId,
    outcome: VariableId,
}

struct MixedClassPosteriorResponse {
    policy: StructuralAggregationPolicy,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    response: CausalResponse,
    mixture: StructuralResponseMixture,
    conditional: Option<ResponseValue>,
    weighted: Vec<(u64, f64, CausalResponse)>,
    failed_mass: f64,
    mixed_if_se: Option<f64>,
}

fn evaluate_class_atom_response_frequentist(
    _study: &Study,
    data: &TabularData,
    query: &ResponseQuery,
    atom: &CachedClassPosteriorAtomIdentification,
    options: &antecedent_estimate::ContinuousResponseOptions,
    _ctx: &ExecutionContext,
) -> Result<ClassAtomResponseEval, CausalError> {
    let mut weighted = Vec::new();
    let mut atom_scores = Vec::new();
    let mut primary_estimand = None;
    let mut plugin_levels = Vec::new();
    let mut completion_fits: CompletionFitMemo<(
        CausalResponse,
        Option<antecedent_estimate::ResponseInfluence>,
    )> = CompletionFitMemo::default();
    for (case_index, case) in atom.cases.iter().enumerate() {
        let Some(estimand) = case.estimand.as_ref() else {
            continue;
        };
        if !identification_status_ok_for_case(case.status) {
            continue;
        }
        // An atom's completions share its required assumptions, and many share
        // an adjustment set and status: those run one fit.
        let fit_key = CompletionFitKey::new(
            &estimand.adjustment_set,
            case.status,
            &atom.identification.required_assumptions,
        );
        let (response, scores) = if let Some((response, scores)) = completion_fits.get(&fit_key) {
            (response.clone(), scores.clone())
        } else {
            let mut response_estimator =
                ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
            response_estimator.options = options.clone();
            let fitted = response_estimator
                .estimate_identified_scored(
                    data,
                    query,
                    case.status,
                    atom.identification.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            completion_fits.insert(fit_key, fitted.clone());
            fitted
        };
        if let Some(scores) = scores.filter(|s| !s.columns.is_empty()) {
            atom_scores.push((case.weight, scores));
        }
        if primary_estimand.is_none() {
            primary_estimand = Some(estimand.clone());
        }
        if let Some(scalar) = atom_plugin_scalar(&response) {
            plugin_levels.push((
                case.weight,
                estimand.clone(),
                scalar,
                response.assumptions.clone(),
            ));
        }
        weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), case.weight, response));
    }
    let response = if weighted.is_empty() {
        None
    } else {
        Some(response_path::mix_class_responses(&weighted, atom.identification.status)?)
    };
    Ok(ClassAtomResponseEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or(primary_estimand),
        response,
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        atom_scores,
        plugin_levels,
    })
}

fn evaluate_class_atom_response_bayesian(
    study: &Study,
    data: &TabularData,
    query: &ResponseQuery,
    atom: &CachedClassPosteriorAtomIdentification,
    options: &antecedent_estimate::ContinuousResponseOptions,
    ctx: &ExecutionContext,
) -> Result<ClassAtomResponseEval, CausalError> {
    let cfg = match &study.inference {
        InferenceMode::Bayesian(c) => c.clone(),
        InferenceMode::Frequentist => {
            return Err(CausalError::unsupported_inference(
                "evaluate_class_atom_response_bayesian requires inference=Bayesian",
            ));
        }
    };
    let mut weighted = Vec::new();
    let mut primary_estimand = None;
    let mut plugin_levels = Vec::new();
    let mut completion_fits: CompletionFitMemo<CausalResponse> = CompletionFitMemo::default();
    for (case_index, case) in atom.cases.iter().enumerate() {
        let Some(estimand) = case.estimand.as_ref() else {
            continue;
        };
        if !identification_status_ok_for_case(case.status) {
            continue;
        }
        let fit_key = CompletionFitKey::new(
            &estimand.adjustment_set,
            case.status,
            &atom.identification.required_assumptions,
        );
        let response = if let Some(response) = completion_fits.get(&fit_key) {
            response.clone()
        } else {
            let mut response_estimator =
                ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
            response_estimator.options = options.clone();
            let mut bayes = bayesian_gcomp(&cfg, ctx);
            bayes.prior.clone_from(&cfg.prior);
            let fitted = response_estimator
                .estimate_bayesian(
                    data,
                    query,
                    case.status,
                    atom.identification.required_assumptions.clone(),
                    &bayes,
                    ctx,
                )
                .map_err(CausalError::from)?;
            completion_fits.insert(fit_key, fitted.clone());
            fitted
        };
        if primary_estimand.is_none() {
            primary_estimand = Some(estimand.clone());
        }
        if let Some(scalar) = atom_plugin_scalar(&response) {
            // Refuters use the posterior mean intervention level as their reference
            // point. Posterior uncertainty remains attached to the response itself.
            plugin_levels.push((
                case.weight,
                estimand.clone(),
                scalar,
                response.assumptions.clone(),
            ));
        }
        weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), case.weight, response));
    }
    let response = if weighted.is_empty() {
        None
    } else {
        Some(response_path::mix_class_responses(&weighted, atom.identification.status)?)
    };
    Ok(ClassAtomResponseEval {
        key: atom.key,
        status: atom.identification.status,
        estimand: atom.invariant.clone().or(primary_estimand),
        response,
        partial: atom.invariant.is_none()
            || matches!(atom.identification.status, IdentificationStatus::PartiallyIdentified),
        atom_scores: Vec::new(),
        plugin_levels,
    })
}

/// Mass shares a class posterior response resolves its identification status from.
#[derive(Clone, Copy)]
struct ClassMass {
    unidentified: f64,
    unevaluable: f64,
    failed: f64,
}

impl ClassMass {
    fn summary(self, identified: &CachedGraphPosteriorIdentification) -> PosteriorMassSummary {
        PosteriorMassSummary {
            unidentified: self.unidentified,
            unevaluable: self.unevaluable,
            failed: self.failed,
            truncated: identified.class_atoms.iter().any(|atom| atom.truncated_completions > 0),
            ..PosteriorMassSummary::default()
        }
    }
}

/// Borrowed inputs of the mixed class posterior response.
#[derive(Clone, Copy)]
struct ClassPayloadInputs<'a> {
    identified: &'a CachedGraphPosteriorIdentification,
    query: &'a ResponseQuery,
    weighted: &'a [(u64, f64, CausalResponse)],
    atom_scores: &'a [(f64, antecedent_estimate::ResponseInfluence)],
    first: &'a CausalResponse,
    policy: StructuralAggregationPolicy,
    conditional: &'a Option<ResponseValue>,
    mixable_scalar: bool,
    mass: ClassMass,
    row_count: usize,
    confidence_level: f64,
}

/// Evaluated atoms of a class posterior response before mixing.
struct ClassResponseCollected {
    atoms: Vec<StructuralResponseAtom>,
    weighted: Vec<(u64, f64, CausalResponse)>,
    identified_weight: f64,
    unevaluable_weight: f64,
    failed_mass: f64,
    contributing: Vec<IdentifiedEstimand>,
    any_partial: bool,
    primary_identification: Option<IdentificationResult>,
    primary_estimand: Option<IdentifiedEstimand>,
    atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)>,
}

/// Weights, atoms and primary claim of the evaluated class posterior atoms.
fn collect_class_response_atoms(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomResponseEval],
) -> ClassResponseCollected {
    let mut atoms = Vec::new();
    let mut weighted = Vec::new();
    let mut identified_weight = 0.0;
    let mut unevaluable_weight = 0.0;
    let mut failed_mass = 0.0;
    let mut contributing = Vec::new();
    let mut any_partial = false;
    let mut primary_identification = None;
    let mut primary_estimand = None;
    let mut atom_scores: Vec<(f64, antecedent_estimate::ResponseInfluence)> = Vec::new();
    let proofs = super::atom_mix::AtomProofIndex::build(&identified.class_atoms);

    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    for eval in evals {
        let w = key_weights.get(eval.key);
        for (score_w, scores) in &eval.atom_scores {
            atom_scores.push((w * score_w, scores.clone()));
        }
        let value = eval.response.as_ref().and_then(response_identified_value);
        if let Some(response) = eval.response.clone() {
            if value.is_some() {
                identified_weight += w;
                if let Some(estimand) = eval.estimand.as_ref() {
                    contributing.push(estimand.clone());
                    if primary_estimand.is_none() {
                        primary_estimand = Some(estimand.clone());
                        primary_identification = Some(proofs.identification_for(eval.key));
                    }
                }
                any_partial |= eval.partial;
                weighted.push((eval.key, w, response));
            } else {
                unevaluable_weight += w;
            }
        } else if w > 0.0 {
            failed_mass += w;
            unevaluable_weight += w;
        }
        atoms.push(StructuralResponseAtom {
            graph_key: eval.key,
            weight: w,
            status: eval.status,
            value: value.clone(),
            posterior: None,
            response: eval.response.clone(),
        });
    }
    ClassResponseCollected {
        atoms,
        weighted,
        identified_weight,
        unevaluable_weight,
        failed_mass,
        contributing,
        any_partial,
        primary_identification,
        primary_estimand,
        atom_scores,
    }
}

/// Mixed response of the evaluated class atoms, with the joint-IF SE when one is licensed.
fn class_posterior_response_payload(
    inputs: &ClassPayloadInputs<'_>,
) -> (CausalResponse, Option<f64>) {
    let ClassPayloadInputs {
        identified,
        query,
        weighted,
        atom_scores,
        first,
        policy,
        conditional,
        mixable_scalar,
        mass,
        row_count,
        confidence_level,
    } = *inputs;
    let graph_dependent = resolve_posterior_status(
        IdentificationStatus::NonparametricallyIdentified,
        policy,
        mass.summary(identified),
    ) == IdentificationStatus::GraphDependent;
    let mixed_if_se = (!graph_dependent
        && weighted.len() > 1
        && matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
        && matches!(conditional, Some(ResponseValue::Scalar(_))))
    .then(|| response_path::mixed_response_scalar_se(atom_scores, row_count))
    .flatten();
    let uncertainty = if graph_dependent {
        ResponseUncertainty::None
    } else if weighted.len() == 1 {
        first.uncertainty.clone()
    } else if let (Some(se), Some(ResponseValue::Scalar(value))) = (mixed_if_se, &conditional) {
        let level = confidence_level;
        let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
        ResponseUncertainty::Scalar {
            standard_error: se,
            level,
            lower: *value - z * se,
            upper: *value + z * se,
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        }
    } else {
        ResponseUncertainty::None
    };

    let mut assumptions = first.assumptions.clone();
    for (_, _, response) in weighted.iter().skip(1) {
        assumptions.extend_unique(&response.assumptions.entries);
    }
    let estimate_kind = match conditional {
        Some(value) if !graph_dependent && mixable_scalar => {
            ResponseIdentification::PointIdentified(value.clone())
        }
        _ => ResponseIdentification::GraphDependent(
            weighted
                .iter()
                .filter_map(|(key, _, response)| {
                    response_identified_value(response).map(|value| (*key, value))
                })
                .collect(),
        ),
    };
    let response = CausalResponse {
        estimand: query.functional.clone(),
        identification_status: resolve_posterior_status(
            first.identification_status,
            policy,
            mass.summary(identified),
        ),
        estimate: estimate_kind,
        uncertainty,
        support: mix_support_reports(
            &weighted.iter().map(|(_, _, response)| &response.support).collect::<Vec<_>>(),
        ),
        assumptions,
        provenance_id: Arc::from("estimate.response.class_graph_posterior"),
        horizon_identification: None,
        interaction_structurally_zero: first.interaction_structurally_zero,
    };
    (response, mixed_if_se)
}

fn mix_class_posterior_responses(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomResponseEval],
    unidentified_mass: f64,
    query: &ResponseQuery,
    row_count: usize,
    confidence_level: f64,
) -> Result<MixedClassPosteriorResponse, CausalError> {
    let ClassResponseCollected {
        mut atoms,
        weighted,
        identified_weight,
        unevaluable_weight,
        failed_mass,
        contributing,
        any_partial,
        primary_identification,
        primary_estimand,
        atom_scores,
    } = collect_class_response_atoms(identified, evals);
    append_unidentified_atoms(&mut atoms, &identified.graphs);

    let total = identified.graphs.total_weight();
    let identified_mass = if total > 0.0 { identified_weight / total } else { 0.0 };
    let unevaluable_mass = if total > 0.0 { unevaluable_weight / total } else { 0.0 };
    let refs: Vec<&IdentifiedEstimand> = contributing.iter().collect();
    let policy = if refs.is_empty() && any_partial {
        StructuralAggregationPolicy::IdentifiedSetEnvelope
    } else {
        resolve_structural_aggregation(&refs, any_partial)
    };

    let conditional_values = weighted
        .iter()
        .filter_map(|(_, weight, response)| {
            response_identified_value(response).map(|value| (*weight, value))
        })
        .collect::<Vec<_>>();
    let conditional_refs =
        conditional_values.iter().map(|(weight, value)| (*weight, value)).collect::<Vec<_>>();
    let mixable_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
        && !conditional_refs.is_empty();
    let conditional =
        if mixable_scalar { Some(mix_response_values(&conditional_refs)?) } else { None };
    let structural_set = response_envelope_from_weighted(&weighted);
    let first =
        weighted.first().map(|(_, _, response)| response).ok_or_else(|| CausalError::Compile {
            message: "class graph-posterior response mix: no evaluable atom values".into(),
        })?;

    let (response, mixed_if_se) = class_posterior_response_payload(&ClassPayloadInputs {
        identified,
        query,
        weighted: &weighted,
        atom_scores: &atom_scores,
        first,
        policy,
        conditional: &conditional,
        mixable_scalar,
        mass: ClassMass {
            unidentified: unidentified_mass,
            unevaluable: unevaluable_mass,
            failed: failed_mass,
        },
        row_count,
        confidence_level,
    });

    let mixture = StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass,
        unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass: 0.0,
        identified_set: structural_set,
        identified_set_interval: None,
        conditional_on_identified: conditional.clone(),
        full_mass_scope: identified.class_atoms.iter().all(|a| a.truncated_completions == 0),
        truncated_atoms: evals.iter().map(|_| 0).sum::<usize>()
            + identified.class_atoms.iter().map(|a| a.truncated_completions).sum::<usize>(),
    };
    let identification = primary_identification.ok_or_else(|| CausalError::Compile {
        message: "class graph-posterior response mix: no evaluable atom values".into(),
    })?;
    let estimand = primary_estimand
        .or_else(|| identification.estimands.first().cloned())
        .ok_or_else(|| CausalError::Compile {
            message: "class graph-posterior response envelope: missing estimand".into(),
        })?;
    Ok(MixedClassPosteriorResponse {
        policy,
        identification,
        estimand,
        response,
        mixture,
        conditional,
        weighted,
        failed_mass: if total > 0.0 { failed_mass / total } else { 0.0 },
        mixed_if_se,
    })
}

fn atom_plugin_scalar(response: &CausalResponse) -> Option<f64> {
    match response_identified_value(response)? {
        ResponseValue::Scalar(value) if value.is_finite() => Some(value),
        ResponseValue::Surface { mean, .. } if mean.len() == 1 => {
            mean.first().copied().filter(|value| value.is_finite())
        }
        _ => None,
    }
}

fn plugin_level_on_class_response_atoms(
    study: &Study,
    data: &TabularData,
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomResponseEval],
    (treatment, outcome): (VariableId, VariableId),
    ctx: &ExecutionContext,
    refutation: AtomRefutationPolicy<'_>,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let AtomRefutationPolicy { aggregation: policy, estimator } = refutation;
    let suite = study.refute;
    let custom = &study.custom_validators[..];
    let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
    let mut refute_ws = EstimationWorkspace::default();
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    for eval in evals {
        let atom_weight = key_weights.get(eval.key);
        if atom_weight <= 0.0 {
            continue;
        }
        for (case_weight, estimand, scalar, assumptions) in &eval.plugin_levels {
            let weight = atom_weight * *case_weight;
            if weight <= 0.0 {
                continue;
            }
            let estimate = EffectEstimate::new(
                *scalar,
                f64::NAN,
                assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            let (reports, notes) = RefutationRun {
                data,
                estimand,
                query: &ate_query,
                estimate: &estimate,
                suite,
                estimator,
                custom,
                temporal: None,
            }
            .plugin_level(&mut refute_ws, ctx)?;
            diagnostics.extend(notes);
            per_atom.push((weight, reports));
        }
    }
    let mix_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean);
    Ok(finish_atom_refutations(
        per_atom,
        diagnostics,
        mix_scalar,
        ScalarRefuterMix::Effect,
        None,
        false,
    ))
}
