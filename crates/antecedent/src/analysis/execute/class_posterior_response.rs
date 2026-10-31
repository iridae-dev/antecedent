//! CPDAG/PAG graph-posterior Response: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::atom_mix::append_unidentified_atoms;
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
    PhysicalExecutionPlan, PosteriorMassSummary, RefuteSuite, ResponseFunctional,
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
    point_weights: Vec<f64>,
    scores_expected: bool,
    plugin_levels: Vec<(
        f64,
        [u8; 32],
        IdentificationStatus,
        IdentifiedEstimand,
        f64,
        antecedent_core::AssumptionSet,
    )>,
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
        let class_tag = gp.atom_kind().as_str().to_ascii_lowercase();
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
        let estimator_id = if self.settings.inference.is_bayesian() {
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
                source: physical
                    .lowered()
                    .and_then(|p| p.semantic_identity())
                    .map(|id| *id.as_bytes()),
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
        let (identified, identify_cached) = if let Some(cache) = self.graph_posterior_cache() {
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
            match &self.settings.inference {
                InferenceMode::Frequentist => evaluate_class_atom_response_frequentist(
                    self, data_est, query, atom, options, inner,
                ),
                InferenceMode::Bayesian(_) => evaluate_class_atom_response_bayesian(
                    self, data_est, query, atom, options, inner,
                ),
            }
        })
        .map_err(Into::into)
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
        if (mixed.weighted.iter().filter(|(_, w, _)| *w > 0.0).count() > 1
            || mixed.single_class_needs_joint_scores)
            && matches!(mixed.response.uncertainty, ResponseUncertainty::None)
        {
            diagnostics.push(Diagnostic::new(
                "estimate.response.graph_posterior.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if self.settings.inference.is_bayesian() {
                    "multi-atom Bayesian class-posterior response withholds an aggregate \
                     credible interval: graph-specific dispersion and a frozen-weight aggregate \
                     are different objects"
                } else if !matches!(mixed.conditional, Some(ResponseValue::Scalar(_))) {
                    "Frequentist class-posterior response withholds an aggregate band: \
                     complete aligned pointwise influences are unavailable or incompatible"
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
        } else if mixed.mixed_if_se.is_some()
            || (mixed.single_class_needs_joint_scores
                && matches!(
                    mixed.response.uncertainty,
                    ResponseUncertainty::PointwiseBand {
                        interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        ..
                    }
                ))
        {
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
            source,
            estimator_id,
            treatment,
            outcome,
        } = input;
        if matches!(query.functional, ResponseFunctional::InterventionResponse { .. })
            && (!matches!(self.settings.refute, RefuteSuite::None)
                || !self.declared.custom_validators.is_empty())
        {
            if matches!(self.settings.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
                diagnostics.push(Diagnostic::new(
                    "refute.evalue.not_a_contrast",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "contrast-shaped refuters are not licensed for a plugin intervention level; \
                     cheap runs overlap only and full runs overlap plus sampling-stability of the \
                     g-comp level",
                ));
            }
            plugin_level_on_class_response_atoms(
                self,
                data,
                identified,
                evals,
                (treatment, outcome),
                query,
                source,
                ctx,
                AtomRefutationPolicy {
                    aggregation: mixed.policy,
                    estimator: estimator_id.as_str(),
                },
            )
        } else {
            if !matches!(self.settings.refute, RefuteSuite::None) {
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
    source: Option<[u8; 32]>,
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
    single_class_needs_joint_scores: bool,
}

fn evaluate_class_atom_response_frequentist(
    _study: &Study,
    data: &TabularData,
    query: &ResponseQuery,
    atom: &CachedClassPosteriorAtomIdentification,
    options: &antecedent_estimate::ContinuousResponseOptions,
    ctx: &ExecutionContext,
) -> Result<ClassAtomResponseEval, CausalError> {
    // The response estimator reads column validity, not the table analysis mask.
    // Materialize that mask for the point as well as every subsequent refit.
    let (selected, _selection_lease) =
        level_refit_masked_data(data, ctx).map_err(CausalError::from)?;
    let data = selected.as_ref().unwrap_or(data);
    let mut weighted = Vec::new();
    let mut atom_scores = Vec::new();
    let mut primary_estimand = None;
    let mut plugin_levels = Vec::new();
    let mut completion_fits: CompletionFitMemo<(
        CausalResponse,
        Option<antecedent_estimate::ResponseInfluence>,
    )> = CompletionFitMemo::default();
    for (case_index, case) in atom.cases.iter().enumerate() {
        if case.weight == 0.0 {
            continue;
        }
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
                case.completion_digest,
                case.status,
                estimand.clone(),
                scalar,
                response.assumptions.clone(),
            ));
        }
        weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), case.weight, response));
    }
    let point_weights = weighted.iter().map(|(_, weight, _)| *weight).collect();
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
        point_weights,
        scores_expected: true,
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
    let cfg = match &study.settings.inference {
        InferenceMode::Bayesian(c) => c.clone(),
        InferenceMode::Frequentist => {
            return Err(CausalError::unsupported_inference(
                "evaluate_class_atom_response_bayesian requires inference=Bayesian",
            ));
        }
    };
    let (selected, _selection_lease) =
        level_refit_masked_data(data, ctx).map_err(CausalError::from)?;
    let data = selected.as_ref().unwrap_or(data);
    let mut weighted = Vec::new();
    let mut primary_estimand = None;
    let mut plugin_levels = Vec::new();
    let mut completion_fits: CompletionFitMemo<CausalResponse> = CompletionFitMemo::default();
    for (case_index, case) in atom.cases.iter().enumerate() {
        if case.weight == 0.0 {
            continue;
        }
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
                case.completion_digest,
                case.status,
                estimand.clone(),
                scalar,
                response.assumptions.clone(),
            ));
        }
        weighted.push((u64::try_from(case_index).unwrap_or(u64::MAX), case.weight, response));
    }
    let point_weights = weighted.iter().map(|(_, weight, _)| *weight).collect();
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
        point_weights,
        scores_expected: false,
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
    single_class_needs_joint_scores: bool,
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
    single_class_needs_joint_scores: bool,
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
    let mut scores_complete = true;
    let mut single_class_needs_joint_scores = false;

    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    for eval in evals {
        let w = key_weights.get(eval.key);
        let value = eval.response.as_ref().and_then(response_identified_value);
        if w > 0.0 && value.is_some() {
            let expected: Vec<f64> =
                eval.point_weights.iter().copied().filter(|w| *w > 0.0).collect();
            let scores: Vec<_> =
                eval.atom_scores.iter().filter(|(weight, _)| *weight > 0.0).collect();
            let score_weights: Vec<f64> =
                eval.atom_scores.iter().map(|(weight, _)| *weight).collect();
            let nested =
                super::atom_mix::nested_complete_weights(w, &eval.point_weights, &score_weights);
            // Equal row-aligned scores and the class's existing identical-interval
            // check license retaining its uncertainty, including surface bands.
            let identical_scores = nested.is_some()
                && eval.response.as_ref().is_some_and(|response| {
                    !matches!(response.uncertainty, ResponseUncertainty::None)
                })
                && scores.windows(2).all(|pair| pair[0].1 == pair[1].1);
            single_class_needs_joint_scores |=
                eval.scores_expected && expected.len() > 1 && !identical_scores;
            if let Some(nested) = nested {
                for ((_, scores), weight) in scores.into_iter().zip(nested) {
                    atom_scores.push((weight, scores.clone()));
                }
            } else {
                // Any point contributor without scores invalidates the joint IF.
                scores_complete = false;
            }
        }
        if let Some(response) = eval.response.clone() {
            if value.is_some() && w > 0.0 {
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
            } else if w > 0.0 {
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
    if !scores_complete {
        atom_scores.clear();
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
        single_class_needs_joint_scores,
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
        single_class_needs_joint_scores,
    } = *inputs;
    let graph_dependent = resolve_posterior_status(
        IdentificationStatus::NonparametricallyIdentified,
        policy,
        mass.summary(identified),
    ) == IdentificationStatus::GraphDependent;
    let positive_graphs = weighted.iter().filter(|(_, weight, _)| *weight > 0.0).count();
    let mixed_if_se = (!graph_dependent
        && (positive_graphs > 1 || single_class_needs_joint_scores)
        && matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
        && matches!(conditional, Some(ResponseValue::Scalar(_))))
    .then(|| response_path::mixed_response_scalar_se(atom_scores, row_count))
    .flatten();
    let uncertainty = if graph_dependent {
        ResponseUncertainty::None
    } else if positive_graphs == 1 && !single_class_needs_joint_scores {
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
    } else if matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean) {
        joint_surface_band(atom_scores, row_count, conditional.as_ref(), confidence_level)
            .unwrap_or(ResponseUncertainty::None)
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

/// Pointwise intervals from the same complete, row-aligned joint influence as the mean.
fn joint_surface_band(
    scores: &[(f64, antecedent_estimate::ResponseInfluence)],
    rows: usize,
    value: Option<&ResponseValue>,
    level: f64,
) -> Option<ResponseUncertainty> {
    let mean = match value? {
        ResponseValue::Surface { mean, .. }
        | ResponseValue::Vector(mean)
        | ResponseValue::Jacobian { values: mean, .. } => mean,
        _ => return None,
    };
    let mixed = response_path::mix_response_influences(scores, rows)?;
    if mixed.columns.len() != mean.len() {
        return None;
    }
    let cols: Vec<&[f64]> = mixed.columns.iter().map(Vec::as_slice).collect();
    let cov = antecedent_estimate::joint_influence_covariance(&cols, None).ok()?;
    let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
    if !z.is_finite() || (0..mean.len()).any(|i| !mean[i].is_finite() || !cov.se(i).is_finite()) {
        return None;
    }
    let lower: Arc<[f64]> = mean.iter().enumerate().map(|(i, v)| v - z * cov.se(i)).collect();
    let upper: Arc<[f64]> = mean.iter().enumerate().map(|(i, v)| v + z * cov.se(i)).collect();
    if lower.iter().chain(upper.iter()).any(|bound| !bound.is_finite()) {
        return None;
    }
    Some(ResponseUncertainty::PointwiseBand {
        level,
        lower,
        upper,
        interpretation: antecedent_core::IntervalInterpretation::Confidence,
        draws: None,
    })
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
        single_class_needs_joint_scores,
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
    let first = weighted
        .iter()
        .find(|(_, weight, _)| *weight > 0.0)
        .or_else(|| weighted.first())
        .map(|(_, _, response)| response)
        .ok_or_else(|| CausalError::Compile {
            message: "class graph-posterior response mix: no evaluable atom values".into(),
        })?;

    let (response, mixed_if_se) = class_posterior_response_payload(&ClassPayloadInputs {
        identified,
        query,
        weighted: &weighted,
        atom_scores: &atom_scores,
        single_class_needs_joint_scores,
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
        aggregation_policy: Some(policy),
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
        single_class_needs_joint_scores,
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

/// Refit the same intervention *level* and response procedure on perturbed rows.
#[derive(Clone, Copy)]
struct StaticResponseLevelRefit<'a> {
    study: &'a Study,
    query: &'a ResponseQuery,
    estimand: &'a IdentifiedEstimand,
    status: IdentificationStatus,
    assumptions: &'a antecedent_core::AssumptionSet,
    operational: Option<&'a std::cell::RefCell<Option<antecedent_validate::ValidationError>>>,
}

impl std::fmt::Debug for StaticResponseLevelRefit<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticResponseLevelRefit").finish_non_exhaustive()
    }
}

/// Convert an analysis mask into column validity for the response estimator,
/// which reads column validity but does not consume the table analysis mask.
/// The admission covers copied float buffers, validity maps and storage metadata.
fn level_refit_masked_data(
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Result<
    (Option<TabularData>, Option<antecedent_core::execution::AdmissionLease>),
    antecedent_validate::ValidationError,
> {
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
    use antecedent_validate::ValidationError;
    let Some(mask) = data.storage().analysis_mask() else {
        return Ok((None, None));
    };
    if ctx.cancellation.is_cancelled() {
        return Err(ValidationError::Cancelled);
    }
    let rows = u64::try_from(data.row_count()).map_err(|_| ValidationError::Resource {
        message: "level-refit row count overflow".into(),
    })?;
    let cols = u64::try_from(data.storage().columns().len()).map_err(|_| {
        ValidationError::Resource { message: "level-refit column count overflow".into() }
    })?;
    let bytes = rows
        .checked_mul(cols)
        .and_then(|v| v.checked_mul(16))
        .and_then(|v| rows.checked_mul(8).and_then(|r| v.checked_add(r)))
        .and_then(|v| cols.checked_mul(512).and_then(|c| v.checked_add(c)))
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| ValidationError::Resource {
            message: "level-refit mask admission overflow".into(),
        })?;
    let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ValidationError::Cancelled,
        other => ValidationError::Resource { message: other.to_string() },
    })?;
    let mut columns = Vec::with_capacity(data.storage().columns().len());
    for column in data.storage().columns() {
        if ctx.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let original = column.as_view().validity();
        let mut keep = Vec::with_capacity(data.row_count());
        for row in 0..data.row_count() {
            if row % 256 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            keep.push(mask.is_valid(row) && original.is_valid(row));
        }
        let validity = ValidityBitmap::from_bools(&keep);
        let selected = match column.clone() {
            OwnedColumn::Float64(column) => {
                OwnedColumn::Float64(Float64Column::new(column.id, column.values, validity)?)
            }
            OwnedColumn::Int64(mut column) => {
                column.validity = validity;
                OwnedColumn::Int64(column)
            }
            OwnedColumn::Boolean(mut column) => {
                column.validity = validity;
                OwnedColumn::Boolean(column)
            }
            // The response estimator cannot coerce these types; keep their
            // payloads and metadata intact rather than rewriting them.
            OwnedColumn::Categorical(column) => OwnedColumn::Categorical(column),
            OwnedColumn::Timestamp(column) => OwnedColumn::Timestamp(column),
            OwnedColumn::FixedVector(column) => OwnedColumn::FixedVector(column),
        };
        columns.push(selected);
    }
    let storage = OwnedColumnarStorage::try_new(
        data.storage().schema().clone(),
        columns,
        None,
        data.storage().weights().map(Arc::from),
    )?;
    Ok((Some(TabularData::new(storage)), Some(lease)))
}

impl StaticResponseLevelRefit<'_> {
    fn refit_inner(
        &self,
        data: &TabularData,
        extras: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, antecedent_validate::ValidationError> {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_validate::ValidationError::Cancelled);
        }
        if !extras.is_empty() {
            return Err(antecedent_validate::ValidationError::NotApplicable {
                message: "additional nuisance variables are not licensed for this intervention-level refit",
            });
        }
        let (selected, _selection_lease) = level_refit_masked_data(data, ctx)?;
        let data = selected.as_ref().unwrap_or(data);
        let mut estimator =
            ContinuousResponseEstimator::new(Arc::clone(&self.estimand.adjustment_set));
        estimator.options = self.study.response_options.clone().unwrap_or_default();
        let response = match &self.study.settings.inference {
            InferenceMode::Frequentist => estimator.estimate_identified(
                data,
                self.query,
                self.status,
                self.assumptions.clone(),
            ),
            InferenceMode::Bayesian(config) => {
                let mut bayes = bayesian_gcomp(config, ctx);
                bayes.prior.clone_from(&config.prior);
                estimator.estimate_bayesian(
                    data,
                    self.query,
                    self.status,
                    self.assumptions.clone(),
                    &bayes,
                    ctx,
                )
            }
        }
        .map_err(antecedent_validate::ValidationError::from)?;
        let value = atom_plugin_scalar(&response).ok_or_else(|| {
            antecedent_validate::ValidationError::estimation_msg(
                "intervention-level refit did not produce a finite scalar",
            )
        })?;
        Ok(EffectEstimate::new(
            value,
            f64::NAN,
            self.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        ))
    }
}

impl antecedent_validate::common::EffectRefit for StaticResponseLevelRefit<'_> {
    fn static_bootstrap_uses_witness_columns(&self) -> bool {
        true
    }

    fn refit(
        &self,
        data: &TabularData,
        extras: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, antecedent_validate::ValidationError> {
        let result = self.refit_inner(data, extras, ctx);
        if let (Err(error), Some(latch)) = (&result, self.operational) {
            let classified = CausalError::from(error.clone());
            if matches!(classified, CausalError::Cancelled { .. } | CausalError::Resource { .. }) {
                *latch.borrow_mut() = Some(error.clone());
            }
        }
        result
    }
}

/// Drop each adjustment column while refitting the same level on one fixed
/// complete-case population. The generic GraphRefuter cannot do this through
/// EffectRefit, whose trait does not receive the reduced estimand.
fn level_adjustment_sensitivity(
    refitter: &StaticResponseLevelRefit<'_>,
    data: &TabularData,
    original: &EffectEstimate,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &ExecutionContext,
) -> Result<antecedent_validate::RefutationReport, antecedent_validate::ValidationError> {
    use antecedent_validate::common::EffectRefit;
    use antecedent_validate::{RefutationReport, ValidationError};
    let adjustments = &refitter.estimand.adjustment_set;
    if !refitter.estimand.is_adjustment_shaped() {
        return Err(ValidationError::NotApplicable {
            message: "adjustment drop requires an adjustment-shaped estimand",
        });
    }
    if adjustments.is_empty() {
        return Ok(RefutationReport::new(
            "adjustment.drop_covariate",
            original.ate,
            original.ate,
            0.0,
            false,
            true,
            None,
            0,
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(ValidationError::Cancelled);
    }
    let n = u64::try_from(data.row_count()).map_err(|_| ValidationError::Resource {
        message: "level adjustment row count overflow".into(),
    })?;
    let columns = u64::try_from(data.storage().columns().len()).map_err(|_| {
        ValidationError::Resource { message: "level adjustment column count overflow".into() }
    })?;
    let adjusted = u64::try_from(adjustments.len()).map_err(|_| ValidationError::Resource {
        message: "level adjustment count overflow".into(),
    })?;
    let bytes = n
        .checked_mul(40)
        .and_then(|v| columns.checked_mul(512).and_then(|c| v.checked_add(c)))
        .and_then(|v| adjusted.checked_mul(64).and_then(|a| v.checked_add(a)))
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| ValidationError::Resource {
            message: "level adjustment scratch admission overflow".into(),
        })?;
    let _scratch = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ValidationError::Cancelled,
        other => ValidationError::Resource { message: other.to_string() },
    })?;
    let mut ids = vec![treatment, outcome];
    for &id in adjustments.iter() {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let mut mask = vec![true; data.row_count()];
    for id in ids {
        let column = data.column(id)?;
        for (row, keep) in mask.iter_mut().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            let finite = match column {
                antecedent_data::ColumnView::Float64(values) => values.values[row].is_finite(),
                antecedent_data::ColumnView::Int64(_) | antecedent_data::ColumnView::Boolean(_) => {
                    true
                }
                _ => {
                    return Err(ValidationError::NotApplicable {
                        message: "adjustment drop requires numeric fitting columns",
                    });
                }
            };
            *keep &= column.validity().is_valid(row) && finite;
        }
    }
    if let Some(existing) = data.storage().analysis_mask() {
        for (row, keep) in mask.iter_mut().enumerate() {
            if row % 256 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            *keep &= existing.is_valid(row);
        }
    }
    let numeric = |id| -> Result<Vec<f64>, ValidationError> {
        let column = data.column(id)?;
        let mut values = Vec::with_capacity(data.row_count());
        for (row, &keep) in mask.iter().enumerate() {
            if row % 256 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            if !keep {
                continue;
            }
            let value = match column {
                antecedent_data::ColumnView::Float64(c) => c.values[row],
                antecedent_data::ColumnView::Int64(c) => c.values[row] as f64,
                antecedent_data::ColumnView::Boolean(c) => f64::from(c.values[row]),
                _ => {
                    return Err(ValidationError::NotApplicable {
                        message: "adjustment drop requires numeric treatment and outcome",
                    });
                }
            };
            values.push(value);
        }
        Ok(values)
    };
    let sd_t = antecedent_stats::sample_std(&numeric(treatment)?);
    let sd_y = antecedent_stats::sample_std(&numeric(outcome)?);
    if !(sd_t.is_finite() && sd_t > 0.0 && sd_y.is_finite() && sd_y > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "adjustment drop requires non-degenerate treatment and outcome",
        });
    }
    let fixed = data.with_analysis_mask(antecedent_data::ValidityBitmap::from_bools(&mask))?;
    let centre = refitter.refit(&fixed, &[], ctx)?.ate;
    let floor = 1e-3 * (sd_y / sd_t);
    let mut worst = 0.0_f64;
    let mut worst_value = centre;
    let mut worst_dropped = adjustments[0];
    for (index, &dropped) in adjustments.iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let remaining: Vec<_> = adjustments
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (i != index).then_some(v))
            .collect();
        let reduced = IdentifiedEstimand::new(
            Arc::clone(&refitter.estimand.method),
            remaining.into(),
            Arc::clone(&refitter.estimand.instruments),
            Arc::clone(&refitter.estimand.mediators),
            refitter.estimand.functional,
            refitter.estimand.rd_design,
        );
        let altered = StaticResponseLevelRefit { estimand: &reduced, ..*refitter };
        let value = altered.refit(&fixed, &[], ctx)?.ate;
        let delta = (value - centre).abs() / centre.abs().max(floor);
        if delta >= worst {
            worst = delta;
            worst_value = value;
            worst_dropped = dropped;
        }
    }
    let passed = worst < 0.5;
    Ok(RefutationReport::new("adjustment.drop_covariate", original.ate, worst_value, worst,
        false, passed, (!passed).then(|| Arc::from(format!("relative intervention-level change {worst} exceeded 0.5 after dropping {worst_dropped:?}"))),
        u32::try_from(adjustments.len()).unwrap_or(u32::MAX)))
}

fn plugin_level_on_class_response_atoms(
    study: &Study,
    data: &TabularData,
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomResponseEval],
    (treatment, outcome): (VariableId, VariableId),
    query: &ResponseQuery,
    source: Option<[u8; 32]>,
    ctx: &ExecutionContext,
    refutation: AtomRefutationPolicy<'_>,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let AtomRefutationPolicy { aggregation: policy, estimator } = refutation;
    let suite = study.settings.refute;
    let custom = &study.declared.custom_validators[..];
    let source = source.ok_or_else(|| CausalError::Compile {
        message: "static completion refutation requires its lowered program identity".into(),
    })?;
    let mut ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
    if let ResponseFunctional::InterventionResponse { interventions, .. } = &query.functional {
        if let [action] = interventions.as_ref() {
            ate_query.active = action.clone();
        }
    }
    ate_query.target_population = query.target_population.clone();
    let mut refute_ws = EstimationWorkspace::default();
    let mut scoped_reports = Vec::new();
    let mut diagnostics = Vec::new();
    let key_weights = IdentifiedKeyWeights::new(&identified.graphs);
    for eval in evals {
        let atom_weight = key_weights.get(eval.key);
        if atom_weight <= 0.0 {
            continue;
        }
        if matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && eval.response.as_ref().and_then(response_identified_value).is_none()
        {
            continue;
        }
        let plugins: Vec<_> =
            eval.plugin_levels.iter().filter(|(weight, _, _, _, _, _)| *weight > 0.0).collect();
        let available: Vec<f64> =
            eval.plugin_levels.iter().map(|(weight, _, _, _, _, _)| *weight).collect();
        // Incomplete coverage cannot certify the whole mixture; the available
        // completion reports remain useful, explicitly scoped evidence.
        if matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
            && super::atom_mix::nested_complete_weights(
                atom_weight,
                &eval.point_weights,
                &available,
            )
            .is_none()
        {
            diagnostics.push(Diagnostic::new(
                "refute.class_response.incomplete_completions", DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "completion reports cover only available refuter inputs and do not certify the aggregate",
            ));
        }
        let inner_scale = eval.point_weights.iter().copied().fold(0.0_f64, f64::max);
        let inner_total = eval.point_weights.iter().map(|w| w / inner_scale).sum::<f64>();
        for (completion_weight, completion_digest, status, estimand, scalar, assumptions) in plugins
        {
            let estimate = EffectEstimate::new(
                *scalar,
                f64::NAN,
                assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            let operational = std::cell::RefCell::new(None);
            let refitter = StaticResponseLevelRefit {
                study,
                query,
                estimand,
                status: *status,
                assumptions,
                operational: Some(&operational),
            };
            let problem = antecedent_validate::RefutationProblem::new(
                data,
                estimand,
                &ate_query,
                &estimate,
                Some(estimator),
                None,
            )
            .with_effect_refit(&refitter);
            let mut validation = match suite {
                RefuteSuite::None => antecedent_validate::ValidationSuite::new(),
                RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc => {
                    antecedent_validate::ValidationSuite::overlap_only()
                }
                RefuteSuite::Full => antecedent_validate::ValidationSuite::new()
                    .with(antecedent_validate::ValidatorId::Overlap)
                    .with(antecedent_validate::ValidatorId::OverlapRule)
                    .with(antecedent_validate::ValidatorId::Bootstrap)
                    .with(antecedent_validate::ValidatorId::DataSubset),
            };
            for validator in custom {
                validation = validation.with_custom(Arc::clone(validator));
            }
            let outcome_result = validation.run(&problem, &mut refute_ws, ctx);
            if let Some(error) = operational.borrow_mut().take() {
                return Err(CausalError::from(error));
            }
            let outcomes = outcome_result.map_err(CausalError::from)?;
            diagnostics.extend(super::validator_not_applicable_diagnostics(&outcomes));
            let label = format!(
                "scope:v1:static_class_completion:{}:{}|level",
                eval.key,
                antecedent_core::identity::to_hex(completion_digest)
            );
            let target = antecedent_validate::RefutationTarget::new(
                antecedent_validate::TargetFamily::WorldValued,
                antecedent_validate::TargetQuadrant {
                    temporal: false,
                    bayesian: study.settings.inference.is_bayesian(),
                },
                [label],
                source,
            )
            .map_err(CausalError::from)?;
            let scope = antecedent_validate::RefutationScope::StaticClassCompletion {
                outer_atom: eval.key,
                completion_digest: *completion_digest,
                target,
            };
            let mut reports = antecedent_validate::ValidationSuite::reports_only(&outcomes);
            if matches!(suite, RefuteSuite::Full) {
                match level_adjustment_sensitivity(
                    &refitter, data, &estimate, treatment, outcome, ctx,
                ) {
                    Ok(report) => reports.push(report),
                    Err(antecedent_validate::ValidationError::NotApplicable { message }) => {
                        let mut diagnostic = Diagnostic::new(
                            "refute.validator.not_applicable",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Info,
                            message,
                        );
                        diagnostic.fields =
                            Arc::from([(Arc::from("validator"), Arc::from("graph"))]);
                        diagnostics.push(diagnostic);
                    }
                    Err(error) => {
                        let reason = error.to_string();
                        let causal = CausalError::from(error);
                        if matches!(
                            causal,
                            CausalError::Cancelled { .. } | CausalError::Resource { .. }
                        ) {
                            return Err(causal);
                        }
                        let mut diagnostic = Diagnostic::new(
                            "refute.validator.failed",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Warning,
                            format!("validator graph could not compute a verdict: {reason}"),
                        );
                        diagnostic.fields = Arc::from([
                            (Arc::from("validator"), Arc::from("graph")),
                            (Arc::from("reason"), Arc::from(reason)),
                        ]);
                        diagnostics.push(diagnostic);
                    }
                }
            }
            for report in &mut reports {
                if report.scope.as_ref().is_some_and(|existing| existing != &scope) {
                    return Err(CausalError::Compile {
                        message:
                            "validator report scope disagrees with its actual class completion"
                                .into(),
                    });
                }
                report.scope = Some(scope.clone());
            }
            if !reports.is_empty() {
                let mut provenance = Diagnostic::new(
                    "refute.class_response.completion_weight",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "point-producer completion weights are descriptive and do not certify an aggregate",
                );
                let conditional_weight = if inner_scale.is_finite()
                    && inner_scale > 0.0
                    && inner_total.is_finite()
                    && inner_total > 0.0
                    && eval.point_weights.iter().all(|weight| weight.is_finite() && *weight >= 0.0)
                {
                    ((*completion_weight / inner_scale) / inner_total).to_string()
                } else {
                    "unavailable".to_owned()
                };
                provenance.fields = Arc::from([
                    (Arc::from("outer_atom"), Arc::from(eval.key.to_string())),
                    (
                        Arc::from("completion_digest"),
                        Arc::from(antecedent_core::identity::to_hex(completion_digest)),
                    ),
                    (Arc::from("outer_graph_weight"), Arc::from(atom_weight.to_string())),
                    (Arc::from("completion_weight"), Arc::from(completion_weight.to_string())),
                    (Arc::from("within_graph_weight"), Arc::from(conditional_weight)),
                ]);
                diagnostics.push(provenance);
            }
            scoped_reports.extend(reports);
        }
    }
    // Each report is a completion-level result. Do not reduce it into an
    // apparently authoritative whole-target pass.
    Ok((scoped_reports, diagnostics))
}

#[cfg(test)]
mod nested_weight_tests {
    use super::*;
    use crate::strategy_table::{DEFAULT_IDENTIFIER_ID, identify_static, select_estimand};
    use antecedent_core::{Intervention, SupportRegion, SupportReport, SupportStatus, Value};
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

    fn fixture()
    -> (CachedGraphPosteriorIdentification, Vec<ClassAtomResponseEval>, ResponseQuery, Dag) {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let identification =
            identify_static(DEFAULT_IDENTIFIER_ID, &dag, &AverageEffectQuery::binary_ate(t, y))
                .unwrap();
        let estimand = select_estimand(&identification, EstimatorId::LinearAdjustmentAte).unwrap();
        let functional = ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::set(t, Value::f64(1.0))]),
        };
        let query = ResponseQuery::new(functional.clone());
        let response = |value| CausalResponse {
            estimand: functional.clone(),
            identification_status: IdentificationStatus::NonparametricallyIdentified,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)),
            uncertainty: ResponseUncertainty::None,
            support: SupportReport {
                status: SupportStatus::Supported,
                query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([1.0]) },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
                point_status: None,
            },
            assumptions: antecedent_core::AssumptionSet::default(),
            provenance_id: Arc::from("test"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        let scores = |column| antecedent_estimate::ResponseInfluence {
            columns: vec![column],
            row_index: Arc::from([
                antecedent_data::GlobalId::new(0),
                antecedent_data::GlobalId::new(1),
                antecedent_data::GlobalId::new(2),
                antecedent_data::GlobalId::new(3),
            ]),
        };
        let make = |key, value, weights: Vec<f64>, columns: Vec<Vec<f64>>, partial| {
            ClassAtomResponseEval {
                key,
                status: IdentificationStatus::NonparametricallyIdentified,
                estimand: Some(estimand.clone()),
                response: Some(response(value)),
                partial,
                atom_scores: weights.iter().copied().zip(columns.into_iter().map(scores)).collect(),
                point_weights: weights,
                scores_expected: true,
                plugin_levels: Vec::new(),
            }
        };
        let evals = vec![
            make(9, 99.0, vec![1.0], vec![vec![20.0, -20.0, 20.0, -20.0]], true),
            make(3, 0.0, vec![1.0], vec![vec![1.0, -1.0, 1.0, -1.0]], false),
            make(5, 6.0, vec![1.0, 1.0, 1.0], vec![vec![0.0; 4]; 3], false),
        ];
        let class_atoms = [3_u64, 5, 9]
            .into_iter()
            .map(|key| crate::analysis::prepared::CachedClassPosteriorAtomIdentification {
                key,
                identification: identification.clone(),
                invariant: Some(estimand.clone()),
                cases: Arc::from([]),
                identified_weight: 1.0,
                truncated_completions: 0,
            })
            .collect::<Vec<_>>();
        let cache = CachedGraphPosteriorIdentification {
            graphs: WeightedGraphSamples::new(
                vec![0.0, 0.5, 0.5],
                vec![GraphIdentFlag::Identified; 3],
                vec![9, 3, 5],
            )
            .unwrap(),
            atoms: Arc::from([]),
            class_atoms: class_atoms.into(),
        };
        (cache, evals, query, dag)
    }

    #[test]
    fn collector_uses_nested_weights_and_ignores_zero_mass_partial_atom() {
        let (cache, mut evals, query, _) = fixture();
        let collected = collect_class_response_atoms(&cache, &evals);
        assert_eq!(collected.weighted.len(), 2);
        assert_eq!(collected.contributing.len(), 2);
        assert!(!collected.any_partial);
        assert_eq!(collected.atom_scores.len(), 4);
        assert!((collected.atom_scores[0].0 - 0.5).abs() < 1e-12);
        assert!(collected.atom_scores[1..].iter().all(|(w, _)| (*w - 1.0 / 6.0).abs() < 1e-12));
        let mixed = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 4, 0.95).unwrap();
        assert!(
            matches!(mixed.conditional, Some(ResponseValue::Scalar(v)) if (v - 3.0).abs() < 1e-12)
        );
        assert!(mixed.mixed_if_se.is_some());
        evals[2].atom_scores.pop();
        let incomplete = collect_class_response_atoms(&cache, &evals);
        assert!(incomplete.atom_scores.is_empty());
        let mixed = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 4, 0.95).unwrap();
        assert!(mixed.mixed_if_se.is_none());
    }

    #[test]
    fn surface_band_uses_joint_scores_and_refuses_missing_columns() {
        let rows: Arc<[antecedent_data::GlobalId]> =
            (0..4).map(antecedent_data::GlobalId::new).collect::<Vec<_>>().into();
        let scores = vec![
            (
                0.25,
                antecedent_estimate::ResponseInfluence {
                    columns: vec![vec![1.0, -1.0, 1.0, -1.0], vec![2.0, -2.0, 2.0, -2.0]],
                    row_index: rows.clone(),
                },
            ),
            (
                0.75,
                antecedent_estimate::ResponseInfluence {
                    columns: vec![vec![0.0; 4], vec![0.0; 4]],
                    row_index: rows,
                },
            ),
        ];
        let value = ResponseValue::Surface {
            grid: Arc::from([0.0, 1.0]),
            dimension: 1,
            mean: Arc::from([1.0, 2.0]),
        };
        let Some(ResponseUncertainty::PointwiseBand { lower, upper, .. }) =
            joint_surface_band(&scores, 4, Some(&value), 0.95)
        else {
            panic!("joint pointwise band");
        };
        assert!(lower[0] < 1.0 && upper[0] > 1.0 && lower[1] < 2.0 && upper[1] > 2.0);
        assert!(((upper[1] - lower[1]) / (upper[0] - lower[0]) - 2.0).abs() < 1e-12);
        let missing =
            ResponseValue::Surface { grid: Arc::from([0.0]), dimension: 1, mean: Arc::from([1.0]) };
        assert!(joint_surface_band(&scores, 4, Some(&missing), 0.95).is_none());
    }

    #[test]
    fn single_class_surface_retains_joint_pointwise_band_for_distinct_completions() {
        let (mut cache, mut evals, query, _) = fixture();
        cache.graphs = WeightedGraphSamples::new(
            vec![0.0, 1.0, 0.0],
            vec![GraphIdentFlag::Identified; 3],
            vec![9, 3, 5],
        )
        .unwrap();
        evals[1].response.as_mut().unwrap().estimate =
            ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from([0.0, 1.0]),
                dimension: 1,
                mean: Arc::from([1.0, 2.0]),
            });
        evals[1].point_weights = vec![1.0, 1.0];
        let rows = evals[1].atom_scores[0].1.row_index.clone();
        evals[1].atom_scores = vec![
            (
                1.0,
                antecedent_estimate::ResponseInfluence {
                    columns: vec![vec![1.0, -1.0, 1.0, -1.0], vec![2.0, -2.0, 2.0, -2.0]],
                    row_index: rows.clone(),
                },
            ),
            (
                1.0,
                antecedent_estimate::ResponseInfluence {
                    columns: vec![vec![0.0; 4], vec![0.0; 4]],
                    row_index: rows,
                },
            ),
        ];
        let mixed = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 4, 0.95).unwrap();
        assert!(matches!(mixed.response.uncertainty, ResponseUncertainty::PointwiseBand { .. }));
        let same = evals[1].atom_scores[0].1.clone();
        evals[1].atom_scores[1].1 = same;
        let existing = ResponseUncertainty::PointwiseBand {
            level: 0.95,
            lower: Arc::from([0.5, 1.5]),
            upper: Arc::from([1.5, 2.5]),
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        };
        evals[1].response.as_mut().unwrap().uncertainty = existing.clone();
        let identical =
            mix_class_posterior_responses(&cache, &evals, 0.0, &query, 4, 0.95).unwrap();
        assert_eq!(identical.response.uncertainty, existing);
        evals[1].atom_scores.pop();
        let missing = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 4, 0.95).unwrap();
        assert!(matches!(missing.response.uncertainty, ResponseUncertainty::None));
    }

    #[test]
    fn intervention_level_refit_uses_requested_set_and_shift() {
        use antecedent_validate::common::EffectRefit;
        let (_, evals, _, dag) = fixture();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        // Distinct doses identify the unpenalized cubic target spline. Its basis
        // evaluates outside the observed domain at the endpoint, so the shift
        // oracle uses the endpoint-clamped dose, not linear extrapolation.
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 25.0).collect();
        let max_x = x.last().copied().unwrap();
        let shift_expected = 10.0 + 2.0 * x.iter().map(|v| (v + 1.0).min(max_x)).sum::<f64>() / x.len() as f64;
        let y: Vec<f64> = x.iter().map(|v| 10.0 + 2.0 * v).collect();
        let data =
            TabularData::from_f64_columns([("t", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let assumptions = antecedent_core::AssumptionSet::default();
        let ctx = ExecutionContext::for_tests(1);
        for (action, expected) in [
            (Intervention::set(treatment, Value::f64(2.0)), 14.0),
            (Intervention::shift(treatment, Value::f64(1.0)), shift_expected),
        ] {
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome,
                interventions: Arc::from([action]),
            });
            let study = Study::tabular(data.clone())
                .graph(dag.clone())
                .query(query.clone())
                .refute(RefuteSuite::None)
                .build()
                .unwrap();
            let refit = StaticResponseLevelRefit {
                study: &study,
                query: &query,
                estimand: evals[1].estimand.as_ref().unwrap(),
                status: IdentificationStatus::NonparametricallyIdentified,
                assumptions: &assumptions,
                operational: None,
            };
            let estimate = refit.refit(&data, &[], &ctx).unwrap();
            assert!((estimate.ate - expected).abs() < 1e-5, "{} != {expected}", estimate.ate);
            assert!((estimate.ate - 2.0).abs() > 10.0, "not the unit contrast");
        }
    }

    #[test]
    fn level_refit_observes_analysis_mask_and_releases_its_admission() {
        use antecedent_validate::common::EffectRefit;
        let (_, evals, _, dag) = fixture();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let x: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| if i < 50 { 10.0 + 2.0 * v } else { 100.0 + 2.0 * v })
            .collect();
        let data =
            TabularData::from_f64_columns([("t", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let flags: Vec<bool> = (0..100).map(|i| i < 50).collect();
        let masked =
            data.with_analysis_mask(antecedent_data::ValidityBitmap::from_bools(&flags)).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome,
            interventions: Arc::from([Intervention::set(treatment, Value::f64(2.0))]),
        });
        let study = Study::tabular(data.clone())
            .graph(dag)
            .query(query.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let assumptions = antecedent_core::AssumptionSet::default();
        let refitter = StaticResponseLevelRefit {
            study: &study,
            query: &query,
            estimand: evals[1].estimand.as_ref().unwrap(),
            status: IdentificationStatus::NonparametricallyIdentified,
            assumptions: &assumptions,
            operational: None,
        };
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let before = pool.available().bytes();
        let result = refitter.refit(&masked, &[], &ctx).unwrap();
        assert!((result.ate - 14.0).abs() < 1e-5);
        let (cache, _, _, _) = fixture();
        let mut atom = cache.class_atoms.iter().find(|atom| atom.key == 3).unwrap().clone();
        atom.cases = Arc::from([crate::analysis::prepared::CachedClassPosteriorCase {
            completion_digest: [4; 32],
            weight: 1.0,
            status: IdentificationStatus::NonparametricallyIdentified,
            identification: Arc::new(atom.identification.clone()),
            estimand: Some((*refitter.estimand).clone()),
        }]);
        let point = evaluate_class_atom_response_frequentist(
            &study,
            &masked,
            &query,
            &atom,
            &antecedent_estimate::ContinuousResponseOptions::default(),
            &ctx,
        )
        .unwrap();
        let point_value = atom_plugin_scalar(point.response.as_ref().unwrap()).unwrap();
        assert!(
            (point_value - result.ate).abs() < 1e-8,
            "masked point and refit must use identical observations"
        );
        assert_eq!(pool.available().bytes(), before);
        let latch = std::cell::RefCell::new(None);
        let limited_refitter = StaticResponseLevelRefit { operational: Some(&latch), ..refitter };
        let limited = ExecutionContext::for_tests(1)
            .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 0));
        assert!(matches!(
            limited_refitter.refit(&masked, &[], &limited),
            Err(antecedent_validate::ValidationError::Resource { .. })
        ));
        assert!(matches!(
            latch.borrow().as_ref(),
            Some(antecedent_validate::ValidationError::Resource { .. })
        ));
        ctx.cancellation.cancel();
        assert!(matches!(
            refitter.refit(&masked, &[], &ctx),
            Err(antecedent_validate::ValidationError::Cancelled)
        ));
    }

    #[test]
    fn full_suite_bootstrap_refits_sampled_levels_on_selected_rows() {
        use antecedent_validate::common::EffectRefit;
        let (_, evals, _, dag) = fixture();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let x: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();
        let y: Vec<f64> = x.iter().enumerate().map(|(i, v)| {
            10.0 + 2.0 * v + (i % 7) as f64 * 0.1 + if i < 50 { 0.0 } else { 1000.0 }
        }).collect();
        let data = TabularData::from_f64_columns([("t", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let flags: Vec<bool> = (0..100).map(|i| i < 50).collect();
        let masked = data.with_analysis_mask(antecedent_data::ValidityBitmap::from_bools(&flags)).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome, interventions: Arc::from([Intervention::set(treatment, Value::f64(2.0))]),
        });
        let study = Study::tabular(data).graph(dag).query(query.clone()).refute(RefuteSuite::None).build().unwrap();
        let assumptions = antecedent_core::AssumptionSet::default();
        let refitter = StaticResponseLevelRefit {
            study: &study, query: &query, estimand: evals[1].estimand.as_ref().unwrap(),
            status: IdentificationStatus::NonparametricallyIdentified, assumptions: &assumptions, operational: None,
        };
        let ctx = ExecutionContext::for_tests(1);
        let original = refitter.refit(&masked, &[], &ctx).unwrap();
        let witness = antecedent_core::AverageEffectQuery::binary_ate(treatment, outcome);
        let problem = antecedent_validate::RefutationProblem::new(&masked, refitter.estimand, &witness, &original, Some("linear.adjustment.ate"), None).with_effect_refit(&refitter);
        let mut workspace = antecedent_estimate::EstimationWorkspace::default();
        let outcomes = antecedent_validate::ValidationSuite::plugin_level_full().run(&problem, &mut workspace, &ctx).unwrap();
        let report = outcomes.iter().find_map(|outcome| match outcome {
            antecedent_validate::ValidationOutcome::Report(report) if report.refuter.as_ref() == "bootstrap.ci_coverage" => Some(report),
            _ => None,
        }).expect("static level bootstrap report");
        assert_eq!(report.replicates, 200);
        assert!(report.comparison > 0.0, "sampled levels must vary");
        assert!((13.0..16.0).contains(&report.refuted_ate), "masked-out outcomes must not enter draws: {report:?}");
    }

    #[test]
    fn adjustment_drop_refits_the_level_with_the_reduced_adjustment() {
        let (_, evals, _, _) = fixture();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let z_id = VariableId::from_raw(2);
        let z: Vec<f64> = (0..100).map(|i| (i % 2) as f64).collect();
        let x: Vec<f64> = (0..100).map(|i| 0.5 * z[i] + 0.2 * ((i % 5) as f64 - 2.0)).collect();
        let y: Vec<f64> = (0..100).map(|i| 10.0 + 2.0 * x[i] + 5.0 * z[i]).collect();
        let data = TabularData::from_f64_columns([
            ("t", x.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome,
            interventions: Arc::from([Intervention::set(treatment, Value::f64(0.5))]),
        });
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let study = Study::tabular(data.clone())
            .graph(dag)
            .query(query.clone())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let base = evals[1].estimand.as_ref().unwrap();
        let estimand = IdentifiedEstimand::new(
            Arc::clone(&base.method),
            Arc::from([z_id]),
            Arc::clone(&base.instruments),
            Arc::clone(&base.mediators),
            base.functional,
            base.rd_design,
        );
        let assumptions = antecedent_core::AssumptionSet::default();
        let refitter = StaticResponseLevelRefit {
            study: &study,
            query: &query,
            estimand: &estimand,
            status: IdentificationStatus::NonparametricallyIdentified,
            assumptions: &assumptions,
            operational: None,
        };
        let original = antecedent_validate::common::EffectRefit::refit(
            &refitter,
            &data,
            &[],
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let report = level_adjustment_sensitivity(
            &refitter,
            &data,
            &original,
            treatment,
            outcome,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!(
            (report.refuted_ate - original.ate).abs() > 0.1,
            "dropping z must actually change the level"
        );
        assert!(report.comparison > 0.0);
    }

    struct LevelRefitValidator;
    impl antecedent_validate::CustomEffectValidator for LevelRefitValidator {
        fn name(&self) -> &str {
            "test.level_refit"
        }
        fn validate(
            &self,
            problem: &antecedent_validate::RefutationProblem<'_>,
            ctx: &ExecutionContext,
        ) -> Result<antecedent_validate::RefutationReport, antecedent_validate::ValidationError>
        {
            let refit = problem.effect_refit.ok_or(
                antecedent_validate::ValidationError::NotApplicable {
                    message: "missing level refit",
                },
            )?;
            let value = refit.refit(problem.data, &[], ctx)?.ate;
            Ok(antecedent_validate::RefutationReport::new(
                self.name(),
                problem.original.ate,
                value,
                1.0,
                true,
                true,
                None,
                0,
            ))
        }
    }

    #[test]
    fn completion_reports_refit_the_level_and_keep_distinct_scopes() {
        let (cache, mut evals, _, dag) = fixture();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let x: Vec<f64> = (0..100).map(|i| (i % 5) as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 10.0 + 2.0 * v).collect();
        let data =
            TabularData::from_f64_columns([("t", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome,
            interventions: Arc::from([Intervention::set(treatment, Value::f64(2.0))]),
        });
        for eval in &mut evals {
            eval.response.as_mut().unwrap().estimate =
                ResponseIdentification::PointIdentified(ResponseValue::Scalar(14.0));
            eval.plugin_levels = eval
                .point_weights
                .iter()
                .enumerate()
                .map(|(index, weight)| {
                    (
                        *weight,
                        [eval.key as u8 + index as u8; 32],
                        eval.status,
                        eval.estimand.as_ref().unwrap().clone(),
                        14.0,
                        antecedent_core::AssumptionSet::default(),
                    )
                })
                .collect();
        }
        let study = Study::tabular(data.clone())
            .graph(dag)
            .query(query.clone())
            .refute(RefuteSuite::None)
            .custom_validators(vec![Arc::new(LevelRefitValidator)])
            .build()
            .unwrap();
        let mixed = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 100, 0.95).unwrap();
        let (reports, notes) = study
            .refute_class_graph_posterior_response(
                ClassResponseRefuteInput {
                    data: &data,
                    query: &query,
                    identified: &cache,
                    evals: &evals,
                    mixed: &mixed,
                    source: Some([3; 32]),
                    estimator_id: EstimatorId::LinearAdjustmentAte,
                    treatment,
                    outcome,
                },
                &mut Vec::new(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(
            notes
                .iter()
                .filter(|d| d.code.as_ref() == "refute.class_response.completion_weight")
                .count(),
            4
        );
        assert_eq!(reports.len(), 4); // one + three completions; zero-weight graph is excluded.
        let mut seen = std::collections::HashSet::new();
        for report in &reports {
            assert!((report.original_ate - 14.0).abs() < 1e-12);
            assert!((report.refuted_ate - 14.0).abs() < 1e-5);
            let Some(antecedent_validate::RefutationScope::StaticClassCompletion {
                outer_atom,
                completion_digest,
                target,
            }) = &report.scope
            else {
                panic!("completion scope")
            };
            assert!(seen.insert((*outer_atom, *completion_digest)));
            assert_eq!(target.family(), antecedent_validate::TargetFamily::WorldValued);
            assert_eq!(target.source(), &[3; 32]);
            let prefix = format!(
                "scope:v1:static_class_completion:{}:{}|",
                outer_atom,
                antecedent_core::identity::to_hex(completion_digest)
            );
            assert!(target.labels().iter().all(|label| label.starts_with(&prefix)));
        }
        // Missing one completion does not promote the available subset to a
        // whole-target verdict or erase its scoped reports.
        evals[2].plugin_levels.pop();
        let mixed = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 100, 0.95).unwrap();
        let (partial, notes) = study
            .refute_class_graph_posterior_response(
                ClassResponseRefuteInput {
                    data: &data,
                    query: &query,
                    identified: &cache,
                    evals: &evals,
                    mixed: &mixed,
                    source: Some([3; 32]),
                    estimator_id: EstimatorId::LinearAdjustmentAte,
                    treatment,
                    outcome,
                },
                &mut Vec::new(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_eq!(partial.len(), 3);
        assert!(partial.iter().all(|r| matches!(
            &r.scope,
            Some(antecedent_validate::RefutationScope::StaticClassCompletion { .. })
        )));
        assert!(
            notes.iter().any(|d| d.code.as_ref() == "refute.class_response.incomplete_completions")
        );
    }

    #[test]
    fn scalar_refuter_dispatch_runs_complete_nested_inputs() {
        let (cache, mut evals, query, dag) = fixture();
        for eval in &mut evals {
            eval.plugin_levels = eval
                .point_weights
                .iter()
                .map(|weight| {
                    (
                        *weight,
                        [eval.key as u8; 32],
                        eval.status,
                        eval.estimand.as_ref().unwrap().clone(),
                        match &eval.response.as_ref().unwrap().estimate {
                            ResponseIdentification::PointIdentified(ResponseValue::Scalar(
                                value,
                            )) => *value,
                            _ => unreachable!(),
                        },
                        antecedent_core::AssumptionSet::default(),
                    )
                })
                .collect();
        }
        let data = TabularData::from_f64_columns([
            ("t", &[0.0, 1.0, 0.0, 1.0][..]),
            ("y", &[1.0, 2.0, 1.0, 2.0][..]),
        ])
        .unwrap();
        let study = Study::tabular(data.clone())
            .graph(dag)
            .query(query.clone())
            .refute(RefuteSuite::Cheap)
            .build()
            .unwrap();
        let mixed = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 4, 0.95).unwrap();
        let mut diagnostics = Vec::new();
        let (reports, notes) = study
            .refute_class_graph_posterior_response(
                ClassResponseRefuteInput {
                    data: &data,
                    query: &query,
                    identified: &cache,
                    evals: &evals,
                    mixed: &mixed,
                    source: Some([3; 32]),
                    estimator_id: EstimatorId::LinearAdjustmentAte,
                    treatment: VariableId::from_raw(0),
                    outcome: VariableId::from_raw(1),
                },
                &mut diagnostics,
                &ExecutionContext::for_tests(0),
            )
            .unwrap();
        assert!(
            !notes
                .iter()
                .any(|d| d.code.as_ref() == "refute.class_response.incomplete_completions")
        );
        assert!(!reports.is_empty(), "the complete route must produce scoped reports: {notes:?}");
        for report in &reports {
            assert!(matches!(report.original_ate, 0.0 | 6.0));
            assert!(matches!(
                &report.scope,
                Some(antecedent_validate::RefutationScope::StaticClassCompletion { .. })
            ));
        }
    }

    #[test]
    fn scalar_refuter_dispatch_withholds_incomplete_completion_inputs() {
        let (cache, evals, query, dag) = fixture();
        let data = TabularData::from_f64_columns([
            ("t", &[0.0, 1.0, 0.0, 1.0][..]),
            ("y", &[1.0, 2.0, 1.0, 2.0][..]),
        ])
        .unwrap();
        let study = Study::tabular(data.clone())
            .graph(dag)
            .query(query.clone())
            .refute(RefuteSuite::Cheap)
            .build()
            .unwrap();
        let mixed = mix_class_posterior_responses(&cache, &evals, 0.0, &query, 4, 0.95).unwrap();
        let mut diagnostics = Vec::new();
        let (reports, notes) = study
            .refute_class_graph_posterior_response(
                ClassResponseRefuteInput {
                    data: &data,
                    query: &query,
                    identified: &cache,
                    evals: &evals,
                    mixed: &mixed,
                    source: Some([3; 32]),
                    estimator_id: EstimatorId::LinearAdjustmentAte,
                    treatment: VariableId::from_raw(0),
                    outcome: VariableId::from_raw(1),
                },
                &mut diagnostics,
                &ExecutionContext::for_tests(0),
            )
            .unwrap();
        assert!(reports.is_empty());
        assert!(
            notes.iter().any(|d| d.code.as_ref() == "refute.class_response.incomplete_completions")
        );
    }
}
