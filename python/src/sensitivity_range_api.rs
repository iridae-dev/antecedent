//! Python binding of the compiled sensitivity ranges of a prepared study's own result.
//!
//! [`antecedent::PreparedStudy::sensitivity_range`] computes every number; Python receives JSON
//! in the shape of the direct sensitivity functions (`antecedent.validation`) and builds a
//! `SensitivityResult` from it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::msm_sensitivity::MsmSensitivitySpec;
use antecedent::analysis::sensitivity_range::{
    MsmRefitRequest, SensitivityRangeOutcome, SensitivityRangeRequest, SensitivityVectorOutcome,
    SensitivityVectorRequest, sensitivity_vector_range,
};
use antecedent::prelude::VariableId;
use antecedent::validate::{LinearAdjustmentSummary, MediationComponent, MediationPath};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::prepared_api::PyPreparedAnalysis;
use crate::target_validation_api::{components_json, range_json};
use crate::{detach_catch, py_err, py_execution_context, resolve_user_threads, value_err};

fn finite(value: f64) -> Value {
    if value.is_finite() { json!(value) } else { Value::Null }
}

/// The keys of the direct posterior E-value function, from a posterior E-value summary.
macro_rules! posterior_evalue_json {
    ($summary:expr, $level:expr) => {{
        let summary = $summary;
        json!({
            "scale": "standardized_mean_difference",
            "level": $level,
            "median_risk_ratio": summary.median_risk_ratio.iter().copied().map(finite).collect::<Vec<_>>(),
            "credible_interval": summary.credible_interval.iter()
                .map(|(low, high)| json!([finite(*low), finite(*high)])).collect::<Vec<_>>(),
            "evalue": summary.bounds.iter().map(|b| finite(b.point)).collect::<Vec<_>>(),
            "evalue_limit": summary.bounds.iter().map(|b| b.limit.map(finite)).collect::<Vec<_>>(),
            "retention": summary.retention.iter()
                .map(|(strength, probability)| json!({"strength": strength, "probability": probability}))
                .collect::<Vec<_>>(),
        })
    }};
}

/// The keys of the direct vector E-value function, from a slice of E-value bounds.
macro_rules! evalue_bounds_json {
    ($bounds:expr) => {{
        let bounds = $bounds;
        json!({
            "scale": "standardized_mean_difference",
            "risk_ratio": bounds.iter().map(|b| finite(b.risk_ratio)).collect::<Vec<_>>(),
            "risk_ratio_limit": bounds.iter().map(|b| b.risk_ratio_limit.map(finite)).collect::<Vec<_>>(),
            "evalue": bounds.iter().map(|b| finite(b.point)).collect::<Vec<_>>(),
            "evalue_limit": bounds.iter().map(|b| b.limit.map(finite)).collect::<Vec<_>>(),
        })
    }};
}

fn component_of(name: &str) -> PyResult<MediationComponent> {
    match name {
        "indirect" => Ok(MediationComponent::Indirect),
        "direct" => Ok(MediationComponent::Direct),
        other => Err(value_err(format!(
            "unknown mediation component {other:?}; expected indirect or direct"
        ))),
    }
}

fn path_of(name: &str) -> PyResult<MediationPath> {
    match name {
        "treatment_mediator" => Ok(MediationPath::TreatmentMediator),
        "mediator_outcome" => Ok(MediationPath::MediatorOutcome),
        "treatment_outcome" => Ok(MediationPath::TreatmentOutcome),
        "indirect" => Ok(MediationPath::Indirect),
        other => Err(value_err(format!(
            "unknown path {other:?}; expected treatment_mediator, mediator_outcome, \
             treatment_outcome or indirect"
        ))),
    }
}

/// The JSON of one compiled outcome, keyed as the direct path of the same analysis keys it.
fn outcome_json(
    outcome: &SensitivityRangeOutcome,
    names: &dyn Fn(VariableId) -> String,
    settings: (f64, f64, f64),
) -> Value {
    let (q, alpha, level) = settings;
    match outcome {
        SensitivityRangeOutcome::EValue {
            standardized_effect,
            standardized_standard_error,
            residual_sd,
            bounds,
            range,
        } => json!({
            "scale": "standardized_mean_difference",
            "standardized_effect": finite(*standardized_effect),
            "standardized_standard_error": standardized_standard_error.map(finite),
            "residual_sd": finite(*residual_sd),
            "risk_ratio": finite(bounds.risk_ratio),
            "risk_ratio_limit": bounds.risk_ratio_limit.map(finite),
            "evalue": finite(bounds.point),
            "evalue_limit": bounds.limit.map(finite),
            "range": range_json(range),
        }),
        SensitivityRangeOutcome::OmittedVariableBias { fit, covariates, benchmarks, range } => {
            let summary = &fit.summary;
            json!({
                "estimate": summary.estimate,
                "standard_error": summary.standard_error,
                "dof": summary.dof,
                "partial_r2_treatment": summary.partial_r2_treatment(),
                "robustness_value": summary.robustness_value(q, None),
                "robustness_value_alpha": summary.robustness_value(q, Some(alpha)),
                "q": q,
                "alpha": alpha,
                "benchmarks": covariates.iter().zip(benchmarks).map(|(id, bound)| json!({
                    "covariate": names(*id),
                    "kd": bound.kd,
                    "ky": bound.ky,
                    "r2dz": bound.r2dz,
                    "r2yz": bound.r2yz,
                    "adjusted_estimate": finite(summary.adjusted_estimate(bound.r2dz, bound.r2yz)),
                    "adjusted_standard_error":
                        finite(summary.adjusted_standard_error(bound.r2dz, bound.r2yz)),
                })).collect::<Vec<_>>(),
                "range": range_json(range),
            })
        }
        SensitivityRangeOutcome::Mediation { summary, range } => json!({
            "beta_mediator": summary.beta_mediator,
            "sigma_total": summary.sigma_total,
            "sigma_mediator": summary.sigma_mediator,
            "rho_tilde": summary.rho_tilde,
            "indirect_at_zero": summary.acme(0.0).ok().map(finite),
            "range": range_json(range),
        }),
        SensitivityRangeOutcome::MediationComponent { components, range } => {
            let component = if range.target.labels().first().is_some_and(|l| &**l == "direct") {
                MediationComponent::Direct
            } else {
                MediationComponent::Indirect
            };
            json!({
                "coordinates": [components_json(components, component)],
                "range": range_json(range),
            })
        }
        SensitivityRangeOutcome::MediationPath { paths, range } => {
            let summary = |s: &antecedent::validate::LinearAdjustmentSummary| {
                json!({
                    "estimate": s.estimate,
                    "standard_error": s.standard_error,
                    "dof": s.dof,
                    "robustness_value": s.robustness_value(q, None),
                    "robustness_value_alpha": s.robustness_value(q, Some(alpha)),
                })
            };
            json!({
                "coordinates": [{
                    "treatment_mediator": summary(&paths.treatment_mediator),
                    "mediator_outcome": summary(&paths.mediator_outcome),
                    "treatment_outcome": summary(&paths.treatment_outcome),
                }],
                "q": q,
                "alpha": alpha,
                "range": range_json(range),
            })
        }
        SensitivityRangeOutcome::MarginalSensitivity { strata, result } => json!({
            "identified": finite(result.identified),
            "lambda_max": result.lambda_max,
            "grid": result.grid.iter().map(|p| json!({
                "lambda": p.lambda,
                "lower": finite(p.lower),
                "upper": finite(p.upper),
            })).collect::<Vec<_>>(),
            "decision_threshold": result.decision_threshold.map(finite),
            "tipping": result.tipping.map(|t| json!({
                "threshold": t.threshold,
                "direction": t.direction.name(),
                "bracketed": t.bracketed,
                "lower": t.bracket.map(|b| b.lower),
                "upper": t.bracket.map(|b| b.upper),
            })),
            "strata": strata.iter().map(|s| json!({
                "mass": s.mass,
                "propensity": s.propensity,
                "treated": {"values": &s.treated.values, "probabilities": &s.treated.probabilities},
                "control": {"values": &s.control.values, "probabilities": &s.control.probabilities},
            })).collect::<Vec<_>>(),
            "method": result.method,
            "interpretation": result.interpretation,
            "inference_claim": result.inference_claim,
            "sampling_interval": result.uncertainty.sampling_interval,
            "calibration": "not_measured",
        }),
        SensitivityRangeOutcome::EValuePosterior { draws, summary, range } => {
            let mut wire = posterior_evalue_json!(summary, level);
            wire["draws"] = json!(draws);
            wire["range"] = range_json(range);
            wire
        }
        SensitivityRangeOutcome::OmittedVariableBiasPosterior {
            coefficients,
            bias_scale,
            range,
        } => json!({
            "level": level,
            "coefficients": coefficients,
            "bias_scale": bias_scale,
            "range": range_json(range),
        }),
        SensitivityRangeOutcome::MediationPosterior { posterior, evalue, range } => {
            let mut wire = match evalue {
                Some(summary) => posterior_evalue_json!(summary, level),
                None => json!({ "level": level }),
            };
            wire["beta_mediator"] = json!(posterior.beta_mediator);
            wire["sigma_total"] = json!(posterior.sigma_total);
            wire["sigma_mediator"] = json!(posterior.sigma_mediator);
            wire["rho_tilde"] = json!(posterior.rho_tilde);
            wire["total_effect"] = json!(posterior.total_effect);
            wire["mediator_outcome"] = json!(posterior.mediator_outcome);
            wire["sigma_outcome"] = json!(posterior.sigma_outcome);
            wire["mediator_scale"] = json!(posterior.mediator_scale);
            wire["range"] = range_json(range);
            wire
        }
        SensitivityRangeOutcome::MsmParameters { summaries, weights, range } => {
            let mut wire = summaries_json(summaries, q, alpha);
            wire["weights"] = json!(weights);
            wire["range"] = range_json(range);
            wire
        }
        SensitivityRangeOutcome::MsmParameterEValue { summaries, residual_sd, bounds, range } => {
            let mut wire = evalue_bounds_json!(bounds);
            wire["estimate"] = json!(summaries.iter().map(|s| s.estimate).collect::<Vec<_>>());
            wire["standard_error"] =
                json!(summaries.iter().map(|s| s.standard_error).collect::<Vec<_>>());
            wire["residual_sd"] = json!(residual_sd);
            wire["range"] = range_json(range);
            wire
        }
        SensitivityRangeOutcome::MsmParametersRefittingWeights { fit, range } => json!({
            "estimate": fit.refit.iter().map(|s| s.estimate).collect::<Vec<_>>(),
            "standard_error": fit.refit.iter().map(|s| s.standard_error).collect::<Vec<_>>(),
            "fixed_weight_standard_error":
                fit.fixed_weights.iter().map(|s| s.standard_error).collect::<Vec<_>>(),
            "dof": fit.refit.iter().map(|s| s.dof).collect::<Vec<_>>(),
            "replicates": [fit.replicates.0, fit.replicates.1],
            "robustness_value": fit.refit.iter().map(|s| s.robustness_value(q, None)).collect::<Vec<_>>(),
            "q": q,
            "alpha": alpha,
            "range": range_json(range),
        }),
        SensitivityRangeOutcome::MsmParameterPosterior { posterior, evalue, range } => {
            let mut wire = match evalue {
                Some(summary) => posterior_evalue_json!(summary, level),
                None => json!({ "level": level }),
            };
            wire["coefficients"] = json!(posterior.coefficients);
            wire["sigma"] = json!(posterior.sigma);
            wire["estimate"] = json!(posterior.estimates);
            wire["exposure_scale"] = json!(posterior.exposure_scale);
            wire["dof"] = json!(posterior.dof);
            wire["weights"] = json!(posterior.weights);
            wire["range"] = range_json(range);
            wire
        }
        _ => json!({}),
    }
}

/// The keys of the direct posterior E-value function.
/// The keys of the direct vector and MSM partial-R² functions.
fn summaries_json(summaries: &[LinearAdjustmentSummary], q: f64, alpha: f64) -> Value {
    json!({
        "estimate": summaries.iter().map(|s| s.estimate).collect::<Vec<_>>(),
        "standard_error": summaries.iter().map(|s| s.standard_error).collect::<Vec<_>>(),
        "dof": summaries.iter().map(|s| s.dof).collect::<Vec<_>>(),
        "partial_r2_treatment":
            summaries.iter().map(LinearAdjustmentSummary::partial_r2_treatment).collect::<Vec<_>>(),
        "robustness_value": summaries.iter().map(|s| s.robustness_value(q, None)).collect::<Vec<_>>(),
        "robustness_value_alpha":
            summaries.iter().map(|s| s.robustness_value(q, Some(alpha))).collect::<Vec<_>>(),
        "q": q,
        "alpha": alpha,
    })
}

/// The JSON of one compiled vector outcome.
fn vector_json(
    outcome: &SensitivityVectorOutcome,
    q: f64,
    alpha: f64,
    component: MediationComponent,
) -> Value {
    let mut wire = match outcome {
        SensitivityVectorOutcome::EValue { standardized, bounds, .. } => {
            let mut wire = evalue_bounds_json!(bounds);
            wire["standardized_effect"] =
                json!(standardized.iter().map(|s| s.0).collect::<Vec<_>>());
            wire["standardized_standard_error"] =
                json!(standardized.iter().map(|s| s.1).collect::<Vec<_>>());
            wire["residual_sd"] = json!(standardized.iter().map(|s| s.2).collect::<Vec<_>>());
            wire
        }
        SensitivityVectorOutcome::OmittedVariableBias { summaries, .. } => {
            summaries_json(summaries, q, alpha)
        }
        SensitivityVectorOutcome::MediationComponent { components, .. } => {
            json!({
                "coordinates": components.iter().map(|c| components_json(c, component)).collect::<Vec<_>>(),
            })
        }
        SensitivityVectorOutcome::MediationPath { paths, .. } => {
            let summary = |s: &LinearAdjustmentSummary| {
                json!({
                    "estimate": s.estimate,
                    "standard_error": s.standard_error,
                    "dof": s.dof,
                    "robustness_value": s.robustness_value(q, None),
                    "robustness_value_alpha": s.robustness_value(q, Some(alpha)),
                })
            };
            json!({
                "coordinates": paths.iter().map(|c| json!({
                    "treatment_mediator": summary(&c.treatment_mediator),
                    "mediator_outcome": summary(&c.mediator_outcome),
                    "treatment_outcome": summary(&c.treatment_outcome),
                })).collect::<Vec<_>>(),
                "q": q,
                "alpha": alpha,
            })
        }
        _ => json!({}),
    };
    wire["range"] = range_json(outcome.range());
    wire
}

/// Settings of one native sensitivity call that are not the analysis itself.
struct Settings {
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
    kd: f64,
    ky: Option<f64>,
    level: f64,
    draws: usize,
    replicates: usize,
}

/// The per-time treatment and confounder sets a marginal structural model refit reads.
type RefitSets = (Vec<Vec<VariableId>>, Vec<Vec<VariableId>>);

/// The scalar request of `analysis`.
#[allow(clippy::too_many_arguments)]
fn scalar_request(
    analysis: &str,
    settings: &Settings,
    component: &str,
    path: &str,
    spec: MsmSensitivitySpec,
    refit: Option<RefitSets>,
    seed: u64,
) -> PyResult<SensitivityRangeRequest> {
    let Settings { q, alpha, r2yz_extreme, kd, ky, level, draws, replicates } = *settings;
    Ok(match analysis {
        "evalue" => SensitivityRangeRequest::EValue,
        "omitted_variable_bias" => {
            SensitivityRangeRequest::OmittedVariableBias { q, alpha, r2yz_extreme, kd, ky }
        }
        "mediation" => SensitivityRangeRequest::Mediation,
        "mediation_component" => {
            SensitivityRangeRequest::MediationComponent(component_of(component)?)
        }
        "mediation_path" => {
            SensitivityRangeRequest::MediationPath { path: path_of(path)?, q, alpha, r2yz_extreme }
        }
        "marginal_sensitivity" => SensitivityRangeRequest::MarginalSensitivity(spec),
        "evalue_posterior" => SensitivityRangeRequest::EValuePosterior { level },
        "omitted_variable_bias_posterior" => {
            SensitivityRangeRequest::OmittedVariableBiasPosterior { level, r2yz_extreme }
        }
        "mediation_posterior" => SensitivityRangeRequest::MediationPosterior {
            component: component_of(component)?,
            level,
            draws,
        },
        "mediation_component_evalue_posterior" => {
            SensitivityRangeRequest::MediationComponentEValuePosterior {
                component: component_of(component)?,
                level,
                draws,
            }
        }
        "mediation_component_ovb_posterior" => {
            SensitivityRangeRequest::MediationComponentOvbPosterior { level, r2yz_extreme, draws }
        }
        "msm_parameters" => SensitivityRangeRequest::MsmParameters { q, alpha, r2yz_extreme },
        "msm_parameter_evalue" => SensitivityRangeRequest::MsmParameterEValue,
        "msm_parameter_evalue_posterior" => {
            SensitivityRangeRequest::MsmParameterEValuePosterior { level, draws }
        }
        "msm_parameters_ovb_posterior" => {
            SensitivityRangeRequest::MsmParametersOvbPosterior { level, r2yz_extreme, draws }
        }
        "msm_parameters_refit" => {
            let (numerator, denominator) = refit.ok_or_else(|| {
                value_err("msm_parameters_refit declares the numerators and denominators")
            })?;
            let mut request = MsmRefitRequest::new(numerator, denominator, seed);
            request.replicates = replicates;
            request.q = q;
            request.alpha = alpha;
            request.r2yz_extreme = r2yz_extreme;
            SensitivityRangeRequest::MsmParametersRefittingWeights(request)
        }
        other => {
            return Err(value_err(format!(
                "unknown sensitivity analysis {other:?}; expected evalue, omitted_variable_bias, \
                 mediation, mediation_component, mediation_path, marginal_sensitivity, \
                 evalue_posterior, omitted_variable_bias_posterior, mediation_posterior, \
                 mediation_component_evalue_posterior, mediation_component_ovb_posterior, \
                 msm_parameters, msm_parameter_evalue, msm_parameters_refit, \
                 msm_parameter_evalue_posterior or msm_parameters_ovb_posterior"
            )));
        }
    })
}

/// The vector request of `analysis`.
fn vector_request(
    analysis: &str,
    settings: &Settings,
    component: &str,
    path: &str,
) -> PyResult<SensitivityVectorRequest> {
    let Settings { q, alpha, r2yz_extreme, .. } = *settings;
    Ok(match analysis {
        "evalue" => SensitivityVectorRequest::EValue,
        "omitted_variable_bias" => {
            SensitivityVectorRequest::OmittedVariableBias { q, alpha, r2yz_extreme }
        }
        "mediation_component" => {
            SensitivityVectorRequest::MediationComponent(component_of(component)?)
        }
        "mediation_path" => {
            SensitivityVectorRequest::MediationPath { path: path_of(path)?, q, alpha, r2yz_extreme }
        }
        "mediation_component_evalue" => SensitivityVectorRequest::MediationComponentEValue,
        "mediation_component_ovb" => {
            SensitivityVectorRequest::MediationComponentOvb { q, alpha, r2yz_extreme }
        }
        other => {
            return Err(value_err(format!(
                "unknown vector sensitivity analysis {other:?}; expected evalue, \
                 omitted_variable_bias, mediation_component, mediation_path, \
                 mediation_component_evalue or mediation_component_ovb"
            )));
        }
    })
}

/// Column ids of `names` per period.
fn period_columns(
    schema: &antecedent::prelude::CausalSchema,
    names: Option<Vec<Vec<String>>>,
) -> PyResult<Option<Vec<Vec<VariableId>>>> {
    names
        .map(|periods| {
            periods
                .iter()
                .map(|period| {
                    period
                        .iter()
                        .map(|name| crate::graph_build::schema_var_id(schema, name))
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Run one sensitivity analysis on this preparation's own retained result; the outcome as
    /// JSON in the shape of the direct sensitivity function of the same analysis.
    #[pyo3(signature = (analysis, *, q=1.0, alpha=0.05, r2yz_extreme=1.0, kd=1.0, ky=None,
        component="indirect", path="indirect", lambda_max=2.0, grid_points=17,
        decision_threshold=None, tolerance=1e-10, level=0.95, draws=1000, numerators=None,
        denominators=None, replicates=200, require_completed=false, seed=1, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn sensitivity_range_bound(
        &self,
        py: Python<'_>,
        analysis: &str,
        q: f64,
        alpha: f64,
        r2yz_extreme: f64,
        kd: f64,
        ky: Option<f64>,
        component: &str,
        path: &str,
        lambda_max: f64,
        grid_points: usize,
        decision_threshold: Option<f64>,
        tolerance: f64,
        level: f64,
        draws: usize,
        numerators: Option<Vec<Vec<String>>>,
        denominators: Option<Vec<Vec<String>>>,
        replicates: usize,
        require_completed: bool,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        crate::bridge::check_count("replicates", replicates as u64, crate::bridge::MAX_RESAMPLES)?;
        let (source, retained) = self.target_refutation_source();
        let mut spec = MsmSensitivitySpec::new(lambda_max);
        spec.grid_points = grid_points;
        spec.decision_threshold = decision_threshold;
        spec.tolerance = tolerance;
        let refit = match (
            period_columns(source.schema(), numerators)?,
            period_columns(source.schema(), denominators)?,
        ) {
            (Some(numerator), Some(denominator)) => Some((numerator, denominator)),
            (None, None) => None,
            _ => return Err(value_err("declare both numerators and denominators, or neither")),
        };
        let settings = Settings { q, alpha, r2yz_extreme, kd, ky, level, draws, replicates };
        let request = scalar_request(analysis, &settings, component, path, spec, refit, seed)?;
        if require_completed && retained.is_none() {
            return Err(crate::refusal(
                "route_not_supported",
                "target_refutation.live_result_required: a sensitivity range of an outcome \
                 requires the exact retained native producer result",
            ));
        }
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let study = source.materialize(&ctx).map_err(py_err)?;
            let study = crate::interrupt::bind_study_validators(&study)?;
            let mut original_executions = 0_u32;
            let result = if let Some(result) = retained {
                result
            } else {
                let result = study.estimate_retained(&ctx).map_err(py_err)?;
                original_executions += 1;
                Arc::new(result)
            };
            let outcome = study.sensitivity_range(&result, &request, &ctx).map_err(py_err)?;
            let schema = study.schema();
            let names = |id: VariableId| {
                schema
                    .variables()
                    .iter()
                    .find(|variable| variable.id == id)
                    .map_or_else(|| format!("v{}", id.raw()), |variable| variable.name.to_string())
            };
            let mut wire = outcome_json(&outcome, &names, (q, alpha, level));
            if let Value::Object(map) = &mut wire {
                map.insert("original_executions".into(), json!(original_executions));
                map.insert(
                    "program".into(),
                    json!(
                        result
                            .lowered_program_identity()
                            .map(|id| crate::target_validation_api::hex(id.as_bytes()))
                    ),
                );
            }
            Ok(wire.to_string())
        })
    }

    /// Run one sensitivity analysis on the retained results of several preparations, one
    /// coordinate each; the outcome as JSON in the shape of the direct vector function.
    #[staticmethod]
    #[pyo3(signature = (analyses, analysis, *, q=1.0, alpha=0.05, r2yz_extreme=1.0,
        component="indirect", path="indirect", seed=1, threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn sensitivity_vector_range_bound(
        py: Python<'_>,
        analyses: Vec<PyRef<'_, PyPreparedAnalysis>>,
        analysis: &str,
        q: f64,
        alpha: f64,
        r2yz_extreme: f64,
        component: &str,
        path: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let settings = Settings {
            q,
            alpha,
            r2yz_extreme,
            kd: 1.0,
            ky: None,
            level: 0.95,
            draws: 0,
            replicates: 0,
        };
        let request = vector_request(analysis, &settings, component, path)?;
        let mediation_component = component_of(component)?;
        let sources: Vec<_> =
            analyses.iter().map(|analysis| analysis.target_refutation_source()).collect();
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            let studies = sources
                .iter()
                .map(|(source, _)| source.materialize(&ctx).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            let studies = studies
                .iter()
                .map(crate::interrupt::bind_study_validators)
                .collect::<PyResult<Vec<_>>>()?;
            let mut results = Vec::with_capacity(studies.len());
            for (study, (_, retained)) in studies.iter().zip(&sources) {
                results.push(match retained {
                    Some(result) => Arc::clone(result),
                    None => Arc::new(study.estimate_retained(&ctx).map_err(py_err)?),
                });
            }
            let coordinates: Vec<(&antecedent::PreparedStudy, &antecedent::StudyResult)> =
                studies.iter().map(|study| &**study).zip(results.iter().map(|r| &**r)).collect();
            let outcome = sensitivity_vector_range(&coordinates, &request, &ctx).map_err(py_err)?;
            Ok(vector_json(&outcome, q, alpha, mediation_component).to_string())
        })
    }
}
