//! Python bridge of the missing-not-at-random (MNAR) sensitivity analysis: the declared
//! arm-mean contrast re-run across a bounded grid of a delta-adjusted pattern-mixture shift or
//! an exponential-tilting selection model, with the tipping points and the declared decision
//! changes.
//!
//! Python declares the data, the mechanism, the parameter range and the decisions; the effect
//! as a function of the parameter, the sampling interval at every point, the tipping points,
//! the assumption records and every refusal are Rust's. A refusal comes back as structured
//! JSON (`code`, `detail`, `message`) for the Python layer to raise as its own typed
//! exception. Every range is an assumption range, never a confidence interval.
use crate::bridge::RefusalJson;
use crate::detach_catch;
use antecedent::analysis::mnar_sensitivity::{
    mnar_sensitivity, ArmLoading, MnarData, MnarDecision, MnarEstimator, MnarMechanism, MnarModel,
    MnarPoint, MnarSensitivityError, MnarSensitivityResult, MnarSensitivitySpec, MnarTipping,
    MnarTippingSide,
};
use antecedent_core::reason_code;
use pyo3::prelude::*;
use serde_json::{json, Value};

fn refusal_json(code: &str, detail: &str, message: &str) -> String {
    RefusalJson::new(code, "mnar_sensitivity", detail, message).json()
}

fn mnar_refusal(error: &MnarSensitivityError) -> String {
    refusal_json(error.code, error.detail, &error.message)
}

fn point_json(point: &MnarPoint) -> Value {
    json!({
        "parameter": point.parameter,
        "treated_mean": point.treated_mean,
        "control_mean": point.control_mean,
        "effect": point.effect,
        "sampling": {
            "standard_error": point.sampling.standard_error,
            "lower": point.sampling.lower,
            "upper": point.sampling.upper,
        },
        "acts": point.acts,
    })
}

fn side_json(side: Option<MnarTippingSide>) -> Value {
    side.map_or(Value::Null, |side| {
        json!({
            "parameter": side.parameter,
            "bracket_lower": side.bracket_lower,
            "iterations": side.iterations,
            "changes_to": side.changes_to,
        })
    })
}

fn tipping_json(tipping: &MnarTipping) -> Value {
    json!({
        "kind": tipping.kind.name(),
        "label": tipping.label,
        "threshold": tipping.threshold,
        "at_mar": tipping.at_mar,
        "negative": side_json(tipping.negative),
        "positive": side_json(tipping.positive),
    })
}

fn result_json(result: &MnarSensitivityResult) -> Value {
    let range = &result.assumption_range;
    json!({
        "family": result.family,
        "target": result.target,
        "model": result.model.name(),
        "loading": {"treated": result.loading.treated, "control": result.loading.control},
        "estimator": result.estimator.name(),
        "mar": point_json(&result.mar),
        "observed_case_effect": result.observed_case_effect,
        "rows": result.rows,
        "treated_rows": result.treated_rows,
        "treated_missing": result.treated_missing,
        "control_missing": result.control_missing,
        "grid": result.grid.iter().map(point_json).collect::<Vec<_>>(),
        "assumption_range": {
            "parameter_lower": range.parameter_lower,
            "parameter_upper": range.parameter_upper,
            "effect_lower": range.effect_lower,
            "effect_upper": range.effect_upper,
            "is_confidence_interval": range.is_confidence_interval,
            "empirically_established": range.empirically_established,
        },
        "tipping": result.tipping.iter().map(tipping_json).collect::<Vec<_>>(),
        "decisions": result
            .decisions
            .iter()
            .map(|d| json!({
                "name": d.name,
                "threshold": d.threshold,
                "acts_at_mar": d.acts_at_mar,
                "changes_within_range": d.changes_within_range,
                "tipping": tipping_json(&d.tipping),
            }))
            .collect::<Vec<_>>(),
        "level": result.level,
        "tolerance": result.tolerance,
        "assumptions": result
            .assumptions
            .iter()
            .map(|a| json!({"id": a.id, "statement": a.statement, "status": a.status.as_str()}))
            .collect::<Vec<_>>(),
        "method": result.method,
        "interpretation": result.interpretation,
        "inference_claim": result.inference_claim,
        "empirical_status": result.empirical_status,
        "sampling_basis": result.sampling_basis,
    })
}

/// The MNAR sensitivity of a two-arm mean contrast over a bounded parameter grid; returns the
/// result JSON or a structured refusal.
#[pyfunction]
#[pyo3(signature = (
    treatment, outcome, model, loading_treated, loading_control, estimator, lower, upper,
    grid_points, level, decisions, tolerance, covariates=Vec::new(), bootstrap_replicates=200,
    seed=0, sampling_composition=None
))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn mnar_sensitivity_run(
    py: Python<'_>,
    treatment: Vec<f64>,
    outcome: Vec<Option<f64>>,
    model: Option<String>,
    loading_treated: f64,
    loading_control: f64,
    estimator: String,
    lower: f64,
    upper: f64,
    grid_points: usize,
    level: f64,
    decisions: Vec<(String, f64)>,
    tolerance: f64,
    covariates: Vec<Vec<f64>>,
    bootstrap_replicates: u32,
    seed: u64,
    sampling_composition: Option<String>,
) -> PyResult<(Option<String>, Option<String>)> {
    detach_catch(py, move || {
        let estimator = match MnarEstimator::from_name(&estimator) {
            Ok(estimator) => estimator,
            Err(error) => return Ok((None, Some(mnar_refusal(&error)))),
        };
        let mechanism = match model.as_deref() {
            None => None,
            Some(name) => match MnarModel::from_name(name) {
                Some(model) => Some(MnarMechanism {
                    model,
                    loading: ArmLoading { treated: loading_treated, control: loading_control },
                }),
                None => {
                    return Ok((
                        None,
                        Some(refusal_json(
                            reason_code!("route_not_supported"),
                            "mnar_sensitivity.unsupported_mechanism",
                            &format!(
                                "`{name}` is not a supported departure model; the supported models are {:?}",
                                MnarModel::ALL.map(MnarModel::name)
                            ),
                        )),
                    ));
                }
            },
        };
        let spec = MnarSensitivitySpec {
            mechanism,
            estimator,
            lower,
            upper,
            grid_points,
            level,
            decisions: decisions
                .into_iter()
                .map(|(name, threshold)| MnarDecision { name, threshold })
                .collect(),
            tolerance,
            sampling_composition,
            bootstrap_replicates,
            seed,
        };
        let views: Vec<&[f64]> = covariates.iter().map(Vec::as_slice).collect();
        Ok(
            match mnar_sensitivity(
                &MnarData::new(&treatment, &outcome).with_covariates(&views),
                &spec,
            ) {
                Ok(result) => (Some(result_json(&result).to_string()), None),
                Err(error) => (None, Some(mnar_refusal(&error))),
            },
        )
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mnar_sensitivity_run, module)?)?;
    Ok(())
}
