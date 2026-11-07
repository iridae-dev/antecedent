//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

mod batch;
mod structure;

pub(crate) use batch::{PyCompletedBatch, PyPreparedBatch};

use crate::graphs;
use crate::*;
use antecedent::AcceptedGraph;
use antecedent::prelude::VariableId;
use antecedent::query::{AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ResponseQuery};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub(crate) fn parse_rd_config<F>(
    estimator: Option<&str>,
    running_variable: Option<&str>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    resolve_var: F,
) -> PyResult<Option<(VariableId, f64, f64)>>
where
    F: FnOnce(&str) -> PyResult<VariableId>,
{
    let wants_rd = estimator.is_some_and(|e| e.eq_ignore_ascii_case("rd.sharp"))
        || running_variable.is_some()
        || cutoff.is_some()
        || bandwidth.is_some();
    if !wants_rd {
        return Ok(None);
    }
    let (Some(rv), Some(cut), Some(bw)) = (running_variable, cutoff, bandwidth) else {
        return Err(crate::with_reason_code(
            crate::value_err(
                "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth",
            ),
            antecedent::reason::reason_code!("required_option_missing"),
        ));
    };
    Ok(Some((resolve_var(rv)?, cut, bw)))
}

/// `rd.sharp` design from the resolved triple plus the optional configured SE kind
/// (`None` keeps [`antecedent::RdConfig`]'s HC1 default).
pub(crate) fn rd_design(
    running_variable: VariableId,
    cutoff: f64,
    bandwidth: f64,
    se_kind: Option<antecedent::estimate::AnalyticSeKind>,
) -> antecedent::RdConfig {
    let config = antecedent::RdConfig::new(running_variable, cutoff, bandwidth);
    match se_kind {
        Some(kind) => config.with_se_kind(kind),
        None => config,
    }
}

struct PpcFields {
    prior_ppc_p_value: Option<f64>,
    prior_ppc_observed: Option<f64>,
    prior_ppc_predictive_mean: Option<f64>,
    prior_ppc_predictive_sd: Option<f64>,
    prior_ppc_n_sims: Option<u32>,
    prior_ppc_dispersion_p_value: Option<f64>,
    prior_ppc_noise: Option<String>,
    posterior_ppc_p_value: Option<f64>,
    posterior_ppc_observed: Option<f64>,
    posterior_ppc_predictive_mean: Option<f64>,
    posterior_ppc_predictive_sd: Option<f64>,
    posterior_ppc_n_sims: Option<u32>,
    posterior_ppc_dispersion_p_value: Option<f64>,
    posterior_ppc_noise: Option<String>,
}

/// Stable wire name of how a predictive check's replicates carry observation noise.
fn predictive_noise_name(noise: antecedent::validate::PredictiveNoise) -> &'static str {
    match noise {
        antecedent::validate::PredictiveNoise::Likelihood => "likelihood",
        antecedent::validate::PredictiveNoise::MeanOnlyImproperScale => "mean_only_improper_scale",
    }
}

fn ppc_fields_from_checks(checks: &[antecedent::validate::PredictiveCheckReport]) -> PpcFields {
    let mut fields = PpcFields {
        prior_ppc_p_value: None,
        prior_ppc_observed: None,
        prior_ppc_predictive_mean: None,
        prior_ppc_predictive_sd: None,
        prior_ppc_n_sims: None,
        prior_ppc_dispersion_p_value: None,
        prior_ppc_noise: None,
        posterior_ppc_p_value: None,
        posterior_ppc_observed: None,
        posterior_ppc_predictive_mean: None,
        posterior_ppc_predictive_sd: None,
        posterior_ppc_n_sims: None,
        posterior_ppc_dispersion_p_value: None,
        posterior_ppc_noise: None,
    };
    for pc in checks {
        match pc.kind {
            PredictiveCheckKind::Prior => {
                fields.prior_ppc_p_value = Some(pc.p_value);
                fields.prior_ppc_observed = Some(pc.observed);
                fields.prior_ppc_predictive_mean = Some(pc.predictive_mean);
                fields.prior_ppc_predictive_sd = Some(pc.predictive_sd);
                fields.prior_ppc_n_sims = Some(pc.n_sims);
                fields.prior_ppc_dispersion_p_value = Some(pc.dispersion_p_value);
                fields.prior_ppc_noise = Some(predictive_noise_name(pc.noise).to_owned());
            }
            PredictiveCheckKind::Posterior => {
                fields.posterior_ppc_p_value = Some(pc.p_value);
                fields.posterior_ppc_observed = Some(pc.observed);
                fields.posterior_ppc_predictive_mean = Some(pc.predictive_mean);
                fields.posterior_ppc_predictive_sd = Some(pc.predictive_sd);
                fields.posterior_ppc_n_sims = Some(pc.n_sims);
                fields.posterior_ppc_dispersion_p_value = Some(pc.dispersion_p_value);
                fields.posterior_ppc_noise = Some(predictive_noise_name(pc.noise).to_owned());
            }
        }
    }
    fields
}

/// The encoded posterior artifact of a Bayesian study result, when one was produced.
fn posterior_artifact_from_result(result: &antecedent::StudyResult) -> PyResult<Option<Vec<u8>>> {
    result
        .posterior
        .as_ref()
        .map(|post| encode_causal_posterior_bytes(post, "ate-analysis").map_err(py_err))
        .transpose()
}

/// Prior-sensitivity sweep of a posterior, as the fields of the Python result.
#[derive(Default)]
struct PriorSensitivityFields {
    scales: Option<Vec<f64>>,
    alphas: Option<Vec<f64>>,
    means: Option<Vec<f64>>,
    sds: Option<Vec<f64>>,
    family: Option<String>,
    variance_multipliers: Option<Vec<f64>>,
}

fn prior_sensitivity_from_result(result: &antecedent::StudyResult) -> PriorSensitivityFields {
    let Some(sens) = result.posterior.as_ref().and_then(|post| post.prior_sensitivity.as_ref())
    else {
        return PriorSensitivityFields::default();
    };
    let non_empty = |values: Vec<f64>| if values.is_empty() { None } else { Some(values) };
    PriorSensitivityFields {
        scales: non_empty(sens.prior_scales.iter().copied().collect()),
        alphas: non_empty(sens.alphas.iter().copied().collect()),
        means: Some(sens.effect_means.iter().copied().collect()),
        sds: Some(sens.effect_sds.iter().copied().collect()),
        family: Some(sens.family.as_str().to_string()),
        variance_multipliers: non_empty(sens.variance_multipliers.iter().copied().collect()),
    }
}

/// Source-conflict summary of a posterior, as the fields of the Python result.
#[derive(Default)]
struct ConflictSummaryFields {
    source_ids: Option<Vec<String>>,
    alphas_requested: Option<Vec<f64>>,
    alphas_applied: Option<Vec<f64>>,
}

fn conflict_summary_from_result(result: &antecedent::StudyResult) -> ConflictSummaryFields {
    let Some(cs) = result.posterior.as_ref().and_then(|post| post.conflict_summary.as_ref()) else {
        return ConflictSummaryFields::default();
    };
    ConflictSummaryFields {
        source_ids: Some(cs.source_ids.iter().map(std::string::ToString::to_string).collect()),
        alphas_requested: Some(cs.alphas_requested.iter().copied().collect()),
        alphas_applied: Some(cs.alphas_applied.iter().copied().collect()),
    }
}

// Python batch query: treatment, outcome, control, active, functional specification,
// target population specification.
type PyBatchQuery<'py> =
    (String, String, f64, f64, Option<Bound<'py, PyDict>>, Option<Bound<'py, PyDict>>);

/// Run static ATE: identify → estimate → optional refute .
///
/// Parse optional `target_population` dict from Python (`kind` + fields).
pub(crate) fn parse_target_population(
    spec: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<TargetPopulation>> {
    let Some(d) = spec else {
        return Ok(None);
    };
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| crate::value_err("target_population requires 'kind'"))?
        .extract()?;
    let kind = kind.to_ascii_lowercase();
    Ok(Some(match kind.as_str() {
        "all" | "all_observed" => TargetPopulation::AllObserved,
        "treated" => TargetPopulation::Treated,
        "untreated" => TargetPopulation::Untreated,
        "named" => {
            let name: String = d
                .get_item("name")?
                .ok_or_else(|| crate::value_err("named target_population requires 'name'"))?
                .extract()?;
            TargetPopulation::Predicate(PredicateExpr::named(name))
        }
        "rows" => {
            let rows: Vec<usize> = d
                .get_item("rows")?
                .ok_or_else(|| crate::value_err("rows target_population requires 'rows'"))?
                .extract()?;
            TargetPopulation::Predicate(PredicateExpr::rows(rows))
        }
        "custom_distribution" | "custom" => {
            let id: u32 = d
                .get_item("id")?
                .ok_or_else(|| {
                    crate::value_err("custom_distribution target_population requires 'id'")
                })?
                .extract()?;
            TargetPopulation::CustomDistribution(DistributionRef::from_raw(id))
        }
        other => {
            return Err(crate::value_err(format!("unknown target_population kind {other:?}")));
        }
    }))
}

pub(crate) fn parse_outcome_functional(
    spec: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<antecedent::query::OutcomeFunctional>> {
    let Some(d) = spec else {
        return Ok(None);
    };
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| crate::value_err("outcome_functional requires 'kind'"))?
        .extract()?;
    Ok(Some(match kind.to_ascii_lowercase().as_str() {
        "mean" => antecedent::query::OutcomeFunctional::Mean,
        "exceedance" => {
            let threshold: f64 = d
                .get_item("threshold")?
                .ok_or_else(|| crate::value_err("exceedance requires 'threshold'"))?
                .extract()?;
            antecedent::query::OutcomeFunctional::exceedance(threshold)
        }
        "exceedance_grid" | "grid" => {
            let thresholds: Vec<f64> = d
                .get_item("thresholds")?
                .ok_or_else(|| crate::value_err("exceedance_grid requires 'thresholds'"))?
                .extract()?;
            antecedent::query::OutcomeFunctional::exceedance_grid(thresholds)
        }
        "quantile" => {
            let tau: f64 = d
                .get_item("tau")?
                .ok_or_else(|| crate::value_err("quantile requires 'tau'"))?
                .extract()?;
            antecedent::query::OutcomeFunctional::quantile(tau)
        }
        other => {
            return Err(crate::value_err(format!("unknown outcome_functional kind {other:?}")));
        }
    }))
}

/// Build a [`PopulationRegistry`] from optional predicate/distribution dicts.
pub(crate) fn parse_population_registry(
    predicates: Option<&Bound<'_, PyDict>>,
    distributions: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<PopulationRegistry>> {
    if predicates.is_none() && distributions.is_none() {
        return Ok(None);
    }
    let mut reg = PopulationRegistry::new();
    if let Some(preds) = predicates {
        for (k, v) in preds.iter() {
            let name: String = k.extract()?;
            let rows: Vec<usize> = v.extract()?;
            reg.insert_predicate(name, rows);
        }
    }
    if let Some(dists) = distributions {
        for (k, v) in dists.iter() {
            let id: u32 = k.extract()?;
            if let Ok(spec) = v.cast::<PyDict>() {
                let weights: Vec<f64> = spec
                    .get_item("weights")?
                    .ok_or_else(|| crate::value_err("missing weights"))?
                    .extract()?;
                let depends: Vec<u32> = spec
                    .get_item("depends_on")?
                    .ok_or_else(|| crate::value_err("missing depends_on"))?
                    .extract()?;
                reg.insert_distribution_with_dependence(
                    DistributionRef::from_raw(id),
                    weights,
                    depends
                        .into_iter()
                        .map(antecedent::prelude::VariableId::from_raw)
                        .collect::<Vec<_>>(),
                );
            } else {
                let weights: Vec<f64> = v.extract()?;
                reg.insert_distribution(DistributionRef::from_raw(id), weights);
            }
        }
    }
    Ok(Some(reg))
}

fn paired_bounds(lower: &[f64], upper: &[f64]) -> PyResult<Vec<(f64, f64)>> {
    if lower.len() != upper.len() {
        return Err(crate::value_err(format!(
            "interval bounds disagree in length ({} lower, {} upper)",
            lower.len(),
            upper.len()
        )));
    }
    Ok(lower.iter().copied().zip(upper.iter().copied()).collect())
}

pub(crate) fn ate_result_from_analysis(
    names: &[String],
    result: &antecedent::StudyResult,
    include_posterior_artifact: bool,
    ctx: &antecedent::prelude::ExecutionContext,
) -> PyResult<AteAnalysisResult> {
    let effect_authority =
        result.identification.average_effect().zip(result.estimate().as_effect()).map(
            |(query, effect)| crate::scalar_decision_api::NativeEffectAuthority {
                names: names.to_vec(),
                query: query.clone(),
                status: result.identification.status,
                ate: effect.ate,
                has_structural_response: result.structural_response.is_some(),
                support_status: result.support_status(),
                executed_contract: result.executed_contract().copied(),
            },
        );
    let effect = result.estimate().as_effect();
    let certificate_json = crate::identification_details::analysis_to_json(result, names, ctx)?;
    // A single-outcome counterfactual answers with its own unit result; one with several
    // outcomes or a joint baseline answers with contrasts, whose first outcome fills the
    // same fields so the single-outcome surface reads identically.
    let counterfactual_view = match (result.counterfactual(), result.counterfactual_contrasts()) {
        (Some(cf), _) => Some((
            &cf.unit_effects,
            cf.unit_effect_intervals.as_ref(),
            cf.unit_extrapolative.as_ref(),
        )),
        (None, Some(set)) => {
            let primary = set.primary();
            Some((
                &primary.unit_effects,
                primary.unit_effect_intervals.as_ref(),
                primary.unit_extrapolative.as_ref(),
            ))
        }
        (None, None) => None,
    };
    let estimator_id = result.logical_plan.estimator.as_deref().unwrap_or("").to_string();
    let posterior_artifact =
        if include_posterior_artifact { posterior_artifact_from_result(result)? } else { None };
    let sections = crate::shared_study_sections(names, result, estimator_id)?
        .with_posterior_artifact(posterior_artifact);
    let ppc = ppc_fields_from_checks(&result.predictive_checks);
    let PriorSensitivityFields {
        scales: prior_sensitivity_scales,
        alphas: prior_sensitivity_alphas,
        means: prior_sensitivity_means,
        sds: prior_sensitivity_sds,
        family: prior_sensitivity_family,
        variance_multipliers: prior_sensitivity_variance_multipliers,
    } = prior_sensitivity_from_result(result);
    let ConflictSummaryFields {
        source_ids: conflict_source_ids,
        alphas_requested: conflict_alphas_requested,
        alphas_applied: conflict_alphas_applied,
    } = conflict_summary_from_result(result);
    let SharedStudySections {
        identification,
        estimate,
        posterior,
        validation,
        performance,
        identified_set,
        identification_status,
        method,
        adjustment_set,
        estimator_id,
        plan_id,
        modality,
        refutations,
        evidence_status,
        structural_weight_basis,
        structural_identified_mass,
        structural_unidentified_mass,
        structural_unevaluable_mass,
        structural_aggregation_policy,
    } = sections;
    let mediation_slices: &[antecedent::estimate::TemporalMediationSlice] =
        result.mediation_grid().map_or(&[], |grid| grid.slices.as_ref());
    let mediation_uncertainty: Vec<_> =
        mediation_slices.iter().map(crate::mediation_uncertainty_projection).collect();

    let graphless_support_status =
        result.support_status().map(antecedent::support::CellStatus::as_str).map(str::to_string);

    Ok(AteAnalysisResult {
        structural_weight_basis,
        structural_identified_mass,
        structural_unidentified_mass,
        structural_unevaluable_mass,
        structural_aggregation_policy,
        structural_identified_set: identified_set.set,
        structural_identified_set_interval: identified_set.interval,
        structural_identified_set_interval_level: identified_set.level,
        structural_identified_set_interval_method: identified_set.method,
        structural_identified_set_interval_truncated: identified_set.truncated,
        certificate_json,
        missing_data_json: crate::missing_report::report_json(result.missing_data().as_ref()),
        interval_bindings_json: crate::interval_binding_api::bindings_json(result),
        ate: effect.map(|estimate| estimate.ate).filter(|ate| ate.is_finite()),
        se_analytic: effect.map_or(f64::NAN, antecedent::estimate::EffectEstimate::se_analytic),
        se_bootstrap: effect.and_then(antecedent::estimate::EffectEstimate::se_bootstrap),
        bootstrap_replicates_failed: effect
            .and_then(|estimate| estimate.uncertainty.bootstrap.replicates_failed),
        adjustment_set,
        identification_status,
        refutation_passed: validation.passed,
        refutation_ran: validation.ran,
        refutation_count: validation.count,
        refutations,
        policy_value: result.policy_value().map(|value| {
            let mut section = crate::policy_api::PolicyValueSection::from(value);
            section.graphless_support_status.clone_from(&graphless_support_status);
            section
        }),
        continuous_dose_response: result.continuous_dose_response().map(|value| {
            let mut section = crate::policy_api::ContinuousDoseResponseSection::from(value);
            section.graphless_support_status.clone_from(&graphless_support_status);
            section
        }),
        assumption_count: result.identification.required_assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
        method,
        estimator_id,
        overlap_ess: estimate.overlap_ess,
        overlap_propensity_min: estimate.overlap_propensity_min,
        posterior_effect_mean: posterior.effect_mean,
        posterior_effect_sd: posterior.effect_sd,
        posterior_q025: posterior.q025,
        posterior_q975: posterior.q975,
        posterior_n_draws: posterior.n_draws,
        posterior_p_below_zero: posterior.p_below_zero,
        posterior_backend: posterior.backend.clone(),
        posterior_artifact: posterior.artifact.clone(),
        diagnostics: result
            .diagnostics
            .iter()
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect(),
        route_diagnostics: result
            .diagnostics
            .iter()
            .filter_map(antecedent::schema::diagnostic::RouteNativeDiagnostic::from_diagnostic)
            .map(|record| {
                (
                    record.route_name().to_owned(),
                    record.outcome_name().to_owned(),
                    record.failed(),
                    record.unevaluable_mass().get(),
                )
            })
            .collect(),
        provenance_node_count: result.provenance.len(),
        plan_id,
        modality,
        discovery_algorithm: result
            .logical_plan
            .discovery_algorithm
            .as_ref()
            .map(std::string::ToString::to_string),
        structure_source: result.structure_source.as_str().to_string(),
        graph_review_required: result.logical_plan.graph_review_required,
        plan_identifier: result
            .logical_plan
            .identifier
            .as_ref()
            .map(std::string::ToString::to_string),
        plan_estimator: result
            .logical_plan
            .estimator
            .as_ref()
            .map(std::string::ToString::to_string),
        validation_suite: result
            .logical_plan
            .validation_suite
            .as_ref()
            .map(std::string::ToString::to_string),
        peak_memory_bytes: result.physical_plan.estimated_peak_memory_bytes,
        worker_threads: result.physical_plan.worker_threads,
        expected_python_crossings: result.physical_plan.expected_python_crossings,
        prior_ppc_p_value: ppc.prior_ppc_p_value,
        prior_ppc_observed: ppc.prior_ppc_observed,
        prior_ppc_predictive_mean: ppc.prior_ppc_predictive_mean,
        prior_ppc_predictive_sd: ppc.prior_ppc_predictive_sd,
        prior_ppc_n_sims: ppc.prior_ppc_n_sims,
        prior_ppc_dispersion_p_value: ppc.prior_ppc_dispersion_p_value,
        prior_ppc_noise: ppc.prior_ppc_noise,
        posterior_ppc_p_value: ppc.posterior_ppc_p_value,
        posterior_ppc_observed: ppc.posterior_ppc_observed,
        posterior_ppc_predictive_mean: ppc.posterior_ppc_predictive_mean,
        posterior_ppc_predictive_sd: ppc.posterior_ppc_predictive_sd,
        posterior_ppc_n_sims: ppc.posterior_ppc_n_sims,
        posterior_ppc_dispersion_p_value: ppc.posterior_ppc_dispersion_p_value,
        posterior_ppc_noise: ppc.posterior_ppc_noise,
        predictive_replicates: result
            .predictive_replicates()
            .map(crate::predictive_assessment_api::PredictiveReplicateLaws::from),
        prior_sensitivity_scales,
        prior_sensitivity_alphas,
        prior_sensitivity_means,
        prior_sensitivity_sds,
        prior_sensitivity_family,
        prior_sensitivity_variance_multipliers,
        conflict_source_ids,
        conflict_alphas_requested,
        conflict_alphas_applied,
        posterior_unidentified_mass: posterior.unidentified_mass,
        posterior_subsampled_out_mass: posterior.subsampled_out_mass,
        latency_mode: performance.latency_mode.clone(),
        wall_time_ns: result.performance.wall_time_ns,
        bootstrap_replicates_requested: result.performance.bootstrap_replicates_requested,
        bootstrap_replicates_ok: performance.bootstrap_replicates_ok,
        n_draws_effort: result.performance.n_draws,
        cancelled: performance.cancelled,
        early_stopped: result.performance.early_stopped,
        stage_timings: performance.stage_timings.clone(),
        identification,
        estimate,
        posterior,
        validation,
        performance,
        assumptions: effect
            .map_or(&result.identification.required_assumptions, |estimate| {
                &estimate.provenance.assumptions
            })
            .entries
            .iter()
            .map(|r| format!("{:?}", r.assumption))
            .collect(),
        support_diagnostics: result.support_diagnostics().map(|d| d.message.to_string()).collect(),
        unit_effects: counterfactual_view.as_ref().map(|view| view.0.to_vec()),
        unit_effect_intervals: counterfactual_view
            .as_ref()
            .and_then(|view| view.1)
            .map(|i| paired_bounds(&i.lower, &i.upper))
            .transpose()?,
        unit_effect_intervals_level: counterfactual_view
            .as_ref()
            .and_then(|view| view.1)
            .map(|i| i.level),
        unit_effect_intervals_method: counterfactual_view
            .as_ref()
            .and_then(|view| view.1)
            .map(|i| i.method.to_owned()),
        unit_extrapolative: counterfactual_view
            .as_ref()
            .and_then(|view| view.2)
            .map(|flags| flags.to_vec()),
        counterfactual_contrasts: result
            .counterfactual_contrasts()
            .map(|set| {
                set.contrasts
                    .iter()
                    .map(|contrast| {
                        Ok((
                            names
                                .get(contrast.outcome.as_usize())
                                .cloned()
                                .unwrap_or_else(|| contrast.outcome.as_usize().to_string()),
                            contrast.mean_ite,
                            contrast.standard_error,
                            contrast.unit_effects.to_vec(),
                            contrast
                                .unit_effect_intervals
                                .as_ref()
                                .map(|i| paired_bounds(&i.lower, &i.upper))
                                .transpose()?,
                            contrast.unit_extrapolative.as_ref().map(|flags| flags.to_vec()),
                            contrast.unit_effects_homogeneous,
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?,
        mediation_total: result.mediation().and_then(|m| m.total),
        mediation_direct: result.mediation().and_then(|m| m.direct),
        mediation_mediated: result.mediation().and_then(|m| m.mediated),
        mediation_horizons: mediation_slices.iter().map(|slice| slice.horizon).collect(),
        mediation_effects: mediation_slices.iter().map(|slice| slice.estimate.effect.ate).collect(),
        mediation_totals: mediation_slices.iter().map(|slice| slice.estimate.total).collect(),
        mediation_directs: mediation_slices.iter().map(|slice| slice.estimate.direct).collect(),
        mediation_mediated_effects: mediation_slices
            .iter()
            .map(|slice| slice.estimate.mediated)
            .collect(),
        mediation_identification_statuses: mediation_slices
            .iter()
            .map(|slice| format!("{:?}", slice.identification_status))
            .collect(),
        mediation_methods: mediation_slices.iter().map(|slice| slice.method.to_string()).collect(),
        mediation_adjustments: mediation_slices
            .iter()
            .map(|slice| {
                slice.adjustment.iter().map(|key| (key.variable.raw(), key.offset)).collect()
            })
            .collect(),
        mediation_uncertainty_kinds: mediation_uncertainty
            .iter()
            .map(|row| row.0.clone())
            .collect(),
        mediation_standard_deviations: mediation_uncertainty.iter().map(|row| row.1).collect(),
        mediation_q025: mediation_uncertainty.iter().map(|row| row.2).collect(),
        mediation_q975: mediation_uncertainty.iter().map(|row| row.3).collect(),
        mediation_identified_lower: mediation_slices
            .iter()
            .map(|slice| slice.identified_set.map(|set| set.lower))
            .collect(),
        mediation_identified_upper: mediation_slices
            .iter()
            .map(|slice| slice.identified_set.map(|set| set.upper))
            .collect(),
        mediation_joint_posterior: result.mediation_grid().map(|grid| grid.joint_posterior),
        transport: result
            .transport()
            .map(crate::transport_interference_api::TransportSection::from_estimate),
        interference: result.interference().map(|estimate| {
            crate::transport_interference_api::InterferenceSection::from_estimate(
                estimate,
                &result.diagnostics,
                result.interference_inference(),
            )
        }),
        randomized_effect: result.randomized_effect().map(|value| {
            let mut section = crate::experiment_api::RandomizedEffectSection::from(value);
            section.graphless_support_status.clone_from(&graphless_support_status);
            section
        }),
        panel_did: result.panel_did().map(|value| {
            let mut section = crate::quasi_api::PanelDidSection::from(value);
            section.graphless_support_status.clone_from(&graphless_support_status);
            section
        }),
        synthetic_control: result.synthetic_control().map(Into::into),
        synthetic_did: result.synthetic_did().map(Into::into),
        local_polynomial_ratio: result.local_polynomial_ratio().map(|value| {
            let mut section = crate::quasi_api::LocalPolynomialRatioSection::from(value);
            section.graphless_support_status.clone_from(&graphless_support_status);
            section
        }),
        survival: result.survival().map(Into::into),
        posterior_row: result.posterior_row().map(Into::into),
        longitudinal_regime: result.longitudinal_regime().map(Into::into),
        sequential_contrast: result.sequential_contrast().map(Into::into),
        history_match: result.history_match().map(Into::into),
        temporal_design: result.temporal_design().map(Into::into),
        iv_bounds: result.iv_bounds().map(Into::into),
        association: result.association().map(Into::into),
        quasi_rollout: result.quasi_rollout().map(Into::into),
        anomaly: result
            .anomaly()
            .map(|scores| crate::gcm_api::anomaly_scores_from_rust(scores.iter().cloned(), names)),
        change_attribution: result.change_attribution().map(|change| {
            crate::gcm_api::change_result_from_rust(change.clone(), names)
                .with_compiled_study_origin()
        }),
        evidence_status,
        would_refuse: result
            .would_refuse()
            .iter()
            .map(|entry| (entry.code.to_owned(), entry.detail.clone(), entry.missing.clone()))
            .collect(),
        effect_authority,
    })
}

/// One marked edge from an oriented temporal CPDAG/PAG.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct GraphEdge {
    #[pyo3(get)]
    pub(crate) source: String,
    #[pyo3(get)]
    pub(crate) source_lag: u32,
    #[pyo3(get)]
    pub(crate) target: String,
    #[pyo3(get)]
    pub(crate) target_lag: u32,
    /// Endpoint mark at `source`: `tail` | `arrow` | `circle` | `conflict`.
    #[pyo3(get)]
    pub(crate) at_source: String,
    /// Endpoint mark at `target`: `tail` | `arrow` | `circle` | `conflict`.
    #[pyo3(get)]
    pub(crate) at_target: String,
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    batch::register(m)?;
    structure::register(m)?;
    Ok(())
}
