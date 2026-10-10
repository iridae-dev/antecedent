//! Python bridge of the 2.3 B3 marginal sensitivity model (MSM): sharp assumption-range
//! bounds of a stratified ATE, a tipping point and the F17 sensitivity artifact.
//!
//! Python declares the strata, the analysis settings and (for the artifact) the effect
//! coordinate and the compared actions; the bounds, the tipping bracket, the artifact and
//! every refusal are Rust's. A refusal comes back as structured JSON (`code`, `detail`,
//! `message`) for the Python layer to raise as its own typed exception. Every number is an
//! assumption range or an identified value; the sampling interval is withheld.
use antecedent::analysis::msm_sensitivity::{
    ActionUtility, MsmOutcomeLaw, MsmSensitivityError, MsmSensitivityResult, MsmSensitivitySpec,
    MsmStratum, msm_sensitivity, msm_sensitivity_artifact_bytes,
};
use antecedent_core::{ScientificQuantity, reason_code};
use antecedent_io::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value, json};

/// One stratum as `(mass, propensity, treated values, treated probabilities, control
/// values, control probabilities)`.
type StratumTuple = (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn refusal_json(code: &str, detail: &str, message: &str) -> String {
    json!({"code": code, "detail": detail, "message": message}).to_string()
}

fn msm_refusal(error: &MsmSensitivityError) -> String {
    refusal_json(error.code, error.detail, &error.message)
}

fn declaration_refusal(message: &str) -> String {
    refusal_json(
        reason_code!("invalid_argument"),
        "sensitivity_decision_composition.invalid_surface",
        message,
    )
}

fn io_refusal(error: &IoError) -> String {
    match error {
        IoError::Refused { code, message } => {
            let (detail, text) = message
                .split_once(": ")
                .unwrap_or(("sensitivity_decision_composition.wrong_contract", message.as_str()));
            refusal_json(code, detail, text)
        }
        other => refusal_json(
            reason_code!("invalid_argument"),
            "sensitivity_decision_composition.invalid_artifact",
            &other.to_string(),
        ),
    }
}

fn strata_of(strata: Vec<StratumTuple>) -> Vec<MsmStratum> {
    strata
        .into_iter()
        .map(|(mass, propensity, tv, tp, cv, cp)| MsmStratum {
            mass,
            propensity,
            treated: MsmOutcomeLaw { values: tv, probabilities: tp },
            control: MsmOutcomeLaw { values: cv, probabilities: cp },
        })
        .collect()
}

fn spec_of(
    lambda_max: f64,
    grid_points: usize,
    decision_threshold: Option<f64>,
    tolerance: f64,
    sampling_composition: Option<String>,
) -> MsmSensitivitySpec {
    MsmSensitivitySpec {
        lambda_max,
        grid_points,
        decision_threshold,
        tolerance,
        sampling_composition,
    }
}

fn result_json(result: &MsmSensitivityResult) -> Value {
    let tipping = result.tipping.as_ref().map(|tipping| {
        json!({
            "threshold": tipping.threshold,
            "direction": tipping.direction.name(),
            "status": tipping.status.name(),
            "bracketed": tipping.bracketed,
            "bracket": tipping.bracket.map(|b| json!({
                "lower": b.lower,
                "upper": b.upper,
                "iterations": b.iterations,
            })),
            "tolerance": tipping.tolerance,
        })
    });
    json!({
        "family": result.family,
        "perturbation_scale": result.perturbation_scale,
        "normalization": result.normalization,
        "target": result.target,
        "identified": result.identified,
        "lambda_max": result.lambda_max,
        "grid": result
            .grid
            .iter()
            .map(|p| json!({"lambda": p.lambda, "lower": p.lower, "upper": p.upper}))
            .collect::<Vec<_>>(),
        "decision_threshold": result.decision_threshold,
        "tipping": tipping,
        "tolerance": result.tolerance,
        "strata": result.strata,
        "method": result.method,
        "interpretation": result.interpretation,
        "inference_claim": result.inference_claim,
        "uncertainty": {
            "sampling_interval": result.uncertainty.sampling_interval,
            "reason_code": result.uncertainty.reason_code,
            "detail": result.uncertainty.detail,
        },
    })
}

/// Sharp MSM bounds of the stratified ATE over a `Lambda` grid; returns the result JSON or
/// a structured refusal.
#[pyfunction]
#[pyo3(signature = (
    strata, lambda_max, grid_points, decision_threshold, tolerance, sampling_composition=None
))]
#[doc(hidden)]
fn msm_sensitivity_run(
    strata: Vec<StratumTuple>,
    lambda_max: f64,
    grid_points: usize,
    decision_threshold: Option<f64>,
    tolerance: f64,
    sampling_composition: Option<String>,
) -> (Option<String>, Option<String>) {
    let spec =
        spec_of(lambda_max, grid_points, decision_threshold, tolerance, sampling_composition);
    match msm_sensitivity(&strata_of(strata), &spec) {
        Ok(result) => (Some(result_json(&result).to_string()), None),
        Err(error) => (None, Some(msm_refusal(&error))),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PointQuantityIn {
    quantity: ScientificQuantityWire,
    values: Vec<f64>,
}

fn quantity_of(wire: ScientificQuantityWire) -> Result<ScientificQuantity, String> {
    ScientificQuantity::try_from(wire).map_err(String::from)
}

/// The F17 sensitivity artifact of an MSM analysis as container bytes; Python consumes the
/// bytes into a `SensitivityArtifact` (recomputing its identity and decision outcome).
#[pyfunction]
#[pyo3(signature = (
    strata, lambda_max, grid_points, decision_threshold, tolerance, effect_json,
    point_quantities_json, actions_json, causal_contract_id, artifact_id
))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn msm_sensitivity_artifact_run<'py>(
    py: Python<'py>,
    strata: Vec<StratumTuple>,
    lambda_max: f64,
    grid_points: usize,
    decision_threshold: Option<f64>,
    tolerance: f64,
    effect_json: &str,
    point_quantities_json: &str,
    actions_json: &str,
    causal_contract_id: &str,
    artifact_id: &str,
) -> (Option<Bound<'py, PyBytes>>, Option<String>) {
    let effect = match serde_json::from_str::<ScientificQuantityWire>(effect_json)
        .map_err(|e| e.to_string())
        .and_then(quantity_of)
    {
        Ok(effect) => effect,
        Err(message) => return (None, Some(declaration_refusal(&message))),
    };
    let wires: Vec<PointQuantityIn> = match serde_json::from_str(point_quantities_json) {
        Ok(wires) => wires,
        Err(error) => return (None, Some(declaration_refusal(&error.to_string()))),
    };
    let mut points = Vec::with_capacity(wires.len());
    for wire in wires {
        match quantity_of(wire.quantity) {
            Ok(quantity) => points.push((quantity, wire.values)),
            Err(message) => return (None, Some(declaration_refusal(&message))),
        }
    }
    let actions: Vec<ActionUtility> = match serde_json::from_str(actions_json) {
        Ok(actions) => actions,
        Err(error) => return (None, Some(declaration_refusal(&error.to_string()))),
    };
    let spec = spec_of(lambda_max, grid_points, decision_threshold, tolerance, None);
    let result = match msm_sensitivity(&strata_of(strata), &spec) {
        Ok(result) => result,
        Err(error) => return (None, Some(msm_refusal(&error))),
    };
    match msm_sensitivity_artifact_bytes(
        &result,
        &effect,
        &points,
        actions,
        causal_contract_id,
        artifact_id,
    ) {
        Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
        Err(error) => (None, Some(io_refusal(&error))),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(msm_sensitivity_run, module)?)?;
    module.add_function(wrap_pyfunction!(msm_sensitivity_artifact_run, module)?)?;
    Ok(())
}
