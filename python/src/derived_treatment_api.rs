//! Python bindings for derived treatments and factorized joint cells (2.2 E5).
//!
//! Results cross the boundary as JSON text and the Python wrapper parses them into frozen
//! dataclasses. A refusal keeps its registered reason code and structured fields; an interval
//! request and a machine-learning nuisance are refused before any data is read.

use antecedent::{
    CausalError, ConstituentRole, DerivedTreatmentDeclaration, check_derived_treatment,
    estimate_derived_joint_cells,
};
use antecedent_estimate::{
    CellStatus, DEFAULT_RIDGE_GRID, EstimationError, FactorizedJointConfig, FactorizedJointFit,
    JointCellReport, JointContrast, RidgeTuning, declared_joint_nuisance, orderings_for,
    refuse_joint_inference, refuse_joint_learner_inference,
};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::{
    detach_catch, py_err, py_execution_context_cancel, resolve_user_threads,
    tabular_from_py_columns, value_err,
};

fn parse_declaration(text: &str) -> PyResult<DerivedTreatmentDeclaration> {
    serde_json::from_str(text).map_err(|e| value_err(format!("derived-treatment declaration: {e}")))
}

fn json_text(value: &Value) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| value_err(e.to_string()))
}

fn estimation_err(error: EstimationError) -> PyErr {
    py_err(CausalError::from(error))
}

/// Check a derived-treatment declaration against a table and an adjustment set; the accepted
/// plan (components, adjustment after declared exclusions, observed levels, preflight report)
/// as JSON. Refusals name the implicated columns on `refusal_fields`.
#[pyfunction]
#[pyo3(signature = (names, columns, declaration, outcome, adjustment, *, seed=1, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn derived_treatment_check_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    declaration: &str,
    outcome: &str,
    adjustment: Vec<String>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let declaration = parse_declaration(declaration)?;
    let (data, _) = tabular_from_py_columns(py, names, columns)?;
    let ctx = py_execution_context_cancel(seed, resolve_user_threads(threads), cancel);
    detach_catch(py, move || {
        let plan = check_derived_treatment(&data, &declaration, outcome, &adjustment, &ctx)
            .map_err(py_err)?;
        json_text(&serde_json::to_value(&plan).map_err(|e| value_err(e.to_string()))?)
    })
}

fn parse_contrast(text: &str) -> PyResult<JointContrast> {
    if text == "interaction" {
        return Ok(JointContrast::Interaction);
    }
    text.strip_prefix("cell_minus_control:")
        .and_then(|cell| cell.parse::<u32>().ok())
        .map(JointContrast::CellMinusControl)
        .ok_or_else(|| {
            value_err(format!(
                "unknown contrast {text:?}; use \"interaction\" or \"cell_minus_control:<cell>\""
            ))
        })
}

fn cell_value(cell: &JointCellReport) -> Value {
    match &cell.status {
        CellStatus::Supported(e) => json!({
            "cell": cell.cell, "levels": cell.levels, "rows": cell.rows, "status": "supported",
            "estimate": e.estimate, "ess": e.ess, "propensity_min": e.propensity_min,
            "propensity_max": e.propensity_max, "clipped_share": e.clipped_share,
        }),
        CellStatus::Unsupported(r) => json!({
            "cell": cell.cell, "levels": cell.levels, "rows": cell.rows, "status": "unsupported",
            "code": r.code, "detail": r.detail, "message": r.message,
        }),
    }
}

fn fit_value(fit: &FactorizedJointFit, contrasts: Vec<Value>) -> Value {
    let sensitivity = &fit.sensitivity;
    json!({
        "n_rows": fit.n_rows,
        "folds": fit.folds,
        "cells": fit.cells.iter().map(cell_value).collect::<Vec<_>>(),
        "normalization": fit.normalization.iter().map(|c| json!({
            "ordering": c.ordering, "cells_enumerated": c.cells_enumerated,
            "max_abs_error": c.max_abs_error, "max_row_sum": c.max_row_sum,
        })).collect::<Vec<_>>(),
        "sensitivity": {
            "orderings": sensitivity.orderings,
            "tolerance": sensitivity.tolerance,
            "disagreement": sensitivity.disagreement,
            "cells": sensitivity.cells.iter().map(|c| json!({
                "cell": c.cell, "estimates": c.estimates, "spread": c.spread,
                "flagged": c.flagged,
            })).collect::<Vec<_>>(),
        },
        "degenerate_conditionals": fit.degenerate_conditionals,
        "contrasts": contrasts,
        "provenance": &*fit.scores.nuisance_provenance,
    })
}

/// Estimate the joint cells of a declared derived treatment with factorized, ridge-penalized
/// conditional propensities, as JSON (point estimates only).
///
/// `ordering` lists the construction columns in the declared factorization order (`None`: the
/// declared source order); `all_orderings` re-estimates under every permutation. `nuisance` is
/// `ridge_logistic` (default), `random_forest` or `gradient_boosted_trees` (a cross-fitted
/// learner for every conditional and outcome model; point only). An interval and any
/// undeclared nuisance are refused before any data is read.
#[pyfunction]
#[pyo3(signature = (
    names, columns, declaration, outcome, adjustment, *, ordering=None, all_orderings=true,
    nuisance="ridge_logistic", penalties=None, inner_folds=5, folds=5, seed=1, clip=0.01,
    min_cell_ess=10.0, normalization_tolerance=1e-9, ordering_tolerance_sd=0.05,
    contrasts=None, interval=false, threads=None, cancel=None
))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn factorized_joint_cells_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    declaration: &str,
    outcome: &str,
    adjustment: Vec<String>,
    ordering: Option<Vec<String>>,
    all_orderings: bool,
    nuisance: &str,
    penalties: Option<Vec<f64>>,
    inner_folds: usize,
    folds: usize,
    seed: u64,
    clip: f64,
    min_cell_ess: f64,
    normalization_tolerance: f64,
    ordering_tolerance_sd: f64,
    contrasts: Option<Vec<String>>,
    interval: bool,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let declaration = parse_declaration(declaration)?;
    let learner = declared_joint_nuisance(nuisance).map_err(estimation_err)?;
    if interval {
        return Err(estimation_err(if learner.is_some() {
            refuse_joint_learner_inference()
        } else {
            refuse_joint_inference()
        }));
    }
    let grid = penalties.unwrap_or_else(|| DEFAULT_RIDGE_GRID.to_vec());
    let tuning = RidgeTuning::new(&grid, inner_folds).map_err(estimation_err)?;
    let requested = contrasts
        .unwrap_or_default()
        .iter()
        .map(|text| parse_contrast(text).map(|c| (text.clone(), c)))
        .collect::<PyResult<Vec<_>>>()?;
    let components: Vec<&str> = declaration
        .sources
        .iter()
        .filter(|s| s.role == ConstituentRole::TreatmentConstruction)
        .map(|s| s.name.as_str())
        .collect();
    let declared: Vec<usize> = match &ordering {
        None => (0..components.len()).collect(),
        Some(listed) => listed
            .iter()
            .map(|name| {
                components.iter().position(|c| c == name).ok_or_else(|| {
                    value_err(format!(
                        "ordering names {name:?}, which is not a construction column"
                    ))
                })
            })
            .collect::<PyResult<_>>()?,
    };
    let orderings =
        orderings_for(components.len(), &declared, all_orderings).map_err(estimation_err)?;
    let mut config = FactorizedJointConfig::new(tuning);
    config.learner = learner;
    config.folds = folds;
    config.seed = seed;
    config.clip = clip;
    config.min_cell_ess = min_cell_ess;
    config.normalization_tolerance = normalization_tolerance;
    config.ordering_tolerance_sd = ordering_tolerance_sd;
    let (data, _) = tabular_from_py_columns(py, names, columns)?;
    let ctx = py_execution_context_cancel(seed, resolve_user_threads(threads), cancel);
    detach_catch(py, move || {
        let (plan, fit) = estimate_derived_joint_cells(
            &data,
            &declaration,
            outcome,
            &adjustment,
            &orderings,
            &config,
            &ctx,
        )
        .map_err(py_err)?;
        let mut contrast_values = Vec::with_capacity(requested.len());
        for (name, contrast) in &requested {
            contrast_values.push(match fit.contrast_point(*contrast) {
                Ok(value) => json!({"name": name, "value": value}),
                Err(EstimationError::Refused { code, message }) => {
                    json!({"name": name, "value": null, "code": code, "message": message})
                }
                Err(other) => return Err(estimation_err(other)),
            });
        }
        let mut value = fit_value(&fit, contrast_values);
        value["plan"] = serde_json::to_value(&plan).map_err(|e| value_err(e.to_string()))?;
        value["treatments"] = json!(plan.treatment_columns);
        json_text(&value)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(derived_treatment_check_json, module)?)?;
    module.add_function(wrap_pyfunction!(factorized_joint_cells_json, module)?)?;
    Ok(())
}
