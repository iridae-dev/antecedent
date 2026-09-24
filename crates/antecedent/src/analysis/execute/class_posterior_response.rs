//! CPDAG/PAG graph-posterior Response: class envelope per atom, then policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::{
    graph_posterior_response_supported, mix_response_values, mix_support_reports,
    response_envelope_from_weighted, response_identified_value, response_primary_pair,
    response_scalar_summary,
};
use super::*;
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
            if let Some(cache) = self.graph_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_graph_posterior_identification_cache(
                        gp, &witness, ctx,
                    )?,
                    false,
                )
            };
        if identified.class_atoms.is_empty() && identified.graphs.identified_mass() <= 0.0 {
            return Err(CausalError::Compile {
                message: "class graph-posterior response: no identified class atoms".into(),
            });
        }

        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            data,
            outcome,
            &query.outcome_functional,
        )?;
        let options = self.response_options.clone().unwrap_or_default();
        let evals = ctx.map_indexed(identified.class_atoms.len(), |i, inner| {
            let atom = &identified.class_atoms[i];
            match &self.inference {
                InferenceMode::Frequentist => evaluate_class_atom_response_frequentist(
                    self, &data_est, query, atom, &options, inner,
                ),
                InferenceMode::Bayesian(_) => evaluate_class_atom_response_bayesian(
                    self, &data_est, query, atom, &options, inner,
                ),
            }
        })?;

        let unidentified_mass = identified.graphs.unidentified_mass();
        let mixed = mix_class_posterior_responses(
            &identified,
            &evals,
            unidentified_mass,
            query,
            data.row_count(),
            options.confidence_level,
        )?;
        let mut identification = mixed.identification;
        identification.status = if unidentified_mass > 0.0
            || mixed.mixture.unevaluable_mass > 0.0
            || !mixed.mixture.full_mass_scope
            || mixed.failed_mass > 0.0
            || matches!(mixed.policy, StructuralAggregationPolicy::GraphDependentAtoms)
        {
            IdentificationStatus::GraphDependent
        } else if matches!(mixed.policy, StructuralAggregationPolicy::IdentifiedSetEnvelope) {
            IdentificationStatus::PartiallyIdentified
        } else {
            identification.status
        };

        let (scalar, standard_error) = response_scalar_summary(&mixed.response);
        let estimate = EffectEstimate::new(
            scalar,
            standard_error,
            mixed.response.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::ResponseBayesian
        } else {
            EstimatorId::default_for_response(&query.functional)
        };

        let mut diagnostics = identification.diagnostics.clone();
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
                if matches!(self.inference, InferenceMode::Bayesian(_)) {
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

        let plugin_scalar = match &mixed.conditional {
            Some(ResponseValue::Scalar(value)) => *value,
            _ => scalar,
        };
        let scalar_intervention = plugin_scalar.is_finite()
            && matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        let (refutations, refute_diags) =
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
                    run_plugin_level_refuters(
                        data,
                        &mixed.estimand,
                        &ate_query,
                        &plugin_estimate,
                        &mut refute_ws,
                        ctx,
                        self.refute,
                        estimator_id.as_str(),
                        &self.custom_validators,
                    )?
                } else {
                    plugin_level_on_class_response_atoms(
                        data,
                        &identified,
                        &evals,
                        treatment,
                        outcome,
                        ctx,
                        self.refute,
                        estimator_id.as_str(),
                        &self.custom_validators,
                        mixed.policy,
                    )?
                }
            } else if !matches!(self.refute, RefuteSuite::None) {
                diagnostics.push(Diagnostic::new(
                "refute.response.skipped",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                if matches!(mixed.policy, StructuralAggregationPolicy::GraphDependentAtoms) {
                    "query-native validation does not compare refuters against a withheld \
                     aggregate under GraphDependentAtoms"
                } else {
                    "scalar ATE refuters are not applicable to a function-valued class-posterior \
                     response"
                },
            ));
                (Vec::new(), Vec::new())
            } else {
                (Vec::new(), Vec::new())
            };
        diagnostics.extend(refute_diags);
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        for warning in &mixed.response.support.warnings {
            diagnostics.push(warning.clone());
        }

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand: mixed.estimand,
            estimate,
            identifier_id: IdentifierId::GeneralizedAdjustment,
            estimator_id,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations,
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
                estimate_provenance: Some(provenance_ids(
                    "estimate.class_graph_posterior_response",
                    mixed.policy.as_str(),
                )),
                response: Some(mixed.response),
                structural_response: Some(mixed.mixture),
                diagnostics: Some(diagnostics),
                ..Default::default()
            },
        }))
    }
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
    for (case_index, case) in atom.cases.iter().enumerate() {
        let Some(estimand) = case.estimand.as_ref() else {
            continue;
        };
        if !identification_status_ok_for_case(case.status) {
            continue;
        }
        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        response_estimator.options = options.clone();
        let (response, scores) = response_estimator
            .estimate_identified_scored(
                data,
                query,
                case.status,
                atom.identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)?;
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
            return Err(CausalError::Unsupported {
                message: "evaluate_class_atom_response_bayesian requires inference=Bayesian",
            });
        }
    };
    let mut weighted = Vec::new();
    let mut primary_estimand = None;
    let mut plugin_levels = Vec::new();
    for (case_index, case) in atom.cases.iter().enumerate() {
        let Some(estimand) = case.estimand.as_ref() else {
            continue;
        };
        if !identification_status_ok_for_case(case.status) {
            continue;
        }
        let mut response_estimator =
            ContinuousResponseEstimator::new(Arc::clone(&estimand.adjustment_set));
        response_estimator.options = options.clone();
        let mut bayes = bayesian_gcomp(&cfg, ctx);
        bayes.prior.clone_from(&cfg.prior);
        let response = response_estimator
            .estimate_bayesian(
                data,
                query,
                case.status,
                atom.identification.required_assumptions.clone(),
                &bayes,
                ctx,
            )
            .map_err(CausalError::from)?;
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

fn mix_class_posterior_responses(
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomResponseEval],
    unidentified_mass: f64,
    query: &ResponseQuery,
    row_count: usize,
    confidence_level: f64,
) -> Result<MixedClassPosteriorResponse, CausalError> {
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

    for eval in evals {
        let w = identified_weight_for_key(&identified.graphs, eval.key);
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
                        primary_identification =
                            Some(eval_identification_response(eval, identified));
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

    let graph_dependent = unidentified_mass > 0.0
        || identified.class_atoms.iter().any(|a| a.truncated_completions > 0)
        || failed_mass > 0.0
        || unevaluable_mass > 0.0
        || matches!(policy, StructuralAggregationPolicy::GraphDependentAtoms);
    let mixed_if_se = (!graph_dependent
        && weighted.len() > 1
        && matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean)
        && matches!(conditional, Some(ResponseValue::Scalar(_))))
    .then(|| response_path::mixed_response_scalar_se(&atom_scores, row_count))
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
    let estimate_kind = if graph_dependent || !mixable_scalar {
        ResponseIdentification::GraphDependent(
            weighted
                .iter()
                .filter_map(|(key, _, response)| {
                    response_identified_value(response).map(|value| (*key, value))
                })
                .collect(),
        )
    } else {
        ResponseIdentification::PointIdentified(conditional.clone().expect("mixable scalar"))
    };
    let response = CausalResponse {
        estimand: query.functional.clone(),
        identification_status: if graph_dependent {
            IdentificationStatus::GraphDependent
        } else if matches!(policy, StructuralAggregationPolicy::IdentifiedSetEnvelope) {
            IdentificationStatus::PartiallyIdentified
        } else {
            first.identification_status
        },
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

fn eval_identification_response(
    eval: &ClassAtomResponseEval,
    identified: &CachedGraphPosteriorIdentification,
) -> IdentificationResult {
    identified.class_atoms.iter().find(|atom| atom.key == eval.key).map_or_else(
        || identified.class_atoms[0].identification.clone(),
        |atom| atom.identification.clone(),
    )
}

fn atom_plugin_scalar(response: &CausalResponse) -> Option<f64> {
    match response_identified_value(response)? {
        ResponseValue::Scalar(value) if value.is_finite() => Some(value),
        ResponseValue::Surface { mean, .. } if mean.len() == 1 && mean[0].is_finite() => {
            Some(mean[0])
        }
        _ => None,
    }
}

fn plugin_level_on_class_response_atoms(
    data: &TabularData,
    identified: &CachedGraphPosteriorIdentification,
    evals: &[ClassAtomResponseEval],
    treatment: VariableId,
    outcome: VariableId,
    ctx: &ExecutionContext,
    suite: RefuteSuite,
    estimator: &str,
    custom: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    policy: StructuralAggregationPolicy,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
    let mut refute_ws = EstimationWorkspace::default();
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    for eval in evals {
        let atom_weight = identified_weight_for_key(&identified.graphs, eval.key);
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
            let (reports, notes) = run_plugin_level_refuters(
                data,
                estimand,
                &ate_query,
                &estimate,
                &mut refute_ws,
                ctx,
                suite,
                estimator,
                custom,
            )?;
            diagnostics.extend(notes);
            per_atom.push((weight, reports));
        }
    }
    let mix_scalar = matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean);
    let reports = if mix_scalar {
        let (mixed, coverage) = mix_atom_refutation_reports(&per_atom);
        diagnostics.extend(coverage);
        mixed
    } else {
        per_atom.into_iter().flat_map(|(_, reports)| reports).collect()
    };
    Ok((reports, diagnostics))
}
