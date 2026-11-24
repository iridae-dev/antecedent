//! Python binding of the compiled sensitivity ranges of a prepared study's own result.
//!
//! [`antecedent::PreparedStudy::sensitivity_range`] computes every number; Python receives JSON
//! in the shape of the direct sensitivity functions (`antecedent.validation`) and builds a
//! `SensitivityResult` from it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::msm_sensitivity::MsmSensitivitySpec;
use antecedent::analysis::sensitivity_range::{SensitivityRangeOutcome, SensitivityRangeRequest};
use antecedent::prelude::VariableId;
use antecedent::validate::{MediationComponent, MediationPath};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::prepared_api::PyPreparedAnalysis;
use crate::target_validation_api::{components_json, range_json};
use crate::{detach_catch, py_err, py_execution_context, resolve_user_threads, value_err};

fn finite(value: f64) -> Value {
    if value.is_finite() { json!(value) } else { Value::Null }
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
    settings: (f64, f64),
) -> Value {
    let (q, alpha) = settings;
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
        _ => json!({}),
    }
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Run one sensitivity analysis on this preparation's own retained result; the outcome as
    /// JSON in the shape of the direct sensitivity function of the same analysis.
    #[pyo3(signature = (analysis, *, q=1.0, alpha=0.05, r2yz_extreme=1.0, kd=1.0, ky=None,
        component="indirect", path="indirect", lambda_max=2.0, grid_points=17,
        decision_threshold=None, tolerance=1e-10, require_completed=false, seed=1, threads=None))]
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
        require_completed: bool,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let request = match analysis {
            "evalue" => SensitivityRangeRequest::EValue,
            "omitted_variable_bias" => {
                SensitivityRangeRequest::OmittedVariableBias { q, alpha, r2yz_extreme, kd, ky }
            }
            "mediation" => SensitivityRangeRequest::Mediation,
            "mediation_component" => {
                SensitivityRangeRequest::MediationComponent(component_of(component)?)
            }
            "mediation_path" => SensitivityRangeRequest::MediationPath {
                path: path_of(path)?,
                q,
                alpha,
                r2yz_extreme,
            },
            "marginal_sensitivity" => {
                let mut spec = MsmSensitivitySpec::new(lambda_max);
                spec.grid_points = grid_points;
                spec.decision_threshold = decision_threshold;
                spec.tolerance = tolerance;
                SensitivityRangeRequest::MarginalSensitivity(spec)
            }
            other => {
                return Err(value_err(format!(
                    "unknown sensitivity analysis {other:?}; expected evalue, \
                     omitted_variable_bias, mediation, mediation_component, mediation_path or \
                     marginal_sensitivity"
                )));
            }
        };
        let (source, retained) = self.target_refutation_source();
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
            let result = match retained {
                Some(result) => result,
                None => {
                    let result = study.estimate_retained(&ctx).map_err(py_err)?;
                    original_executions += 1;
                    Arc::new(result)
                }
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
            let mut wire = outcome_json(&outcome, &names, (q, alpha));
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
}
