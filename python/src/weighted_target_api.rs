//! Weighted target populations: the declaration read from Python and the report returned to it.
//!
//! The semantics (weights from the table, the overlap gate, the report) are the facade's
//! ([`antecedent::WeightedTargetDeclaration`]); this module only builds the declaration from a
//! mapping of column indices and numbers, and reads the report back out.

use std::sync::Arc;

use antecedent::inference::{KernelFamily, TransportKernel};
use antecedent::prelude::VariableId;
use antecedent::{WeightedTargetDeclaration, WeightedTargetReport};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn field<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .filter(|value| !value.is_none())
        .ok_or_else(|| crate::value_err(format!("weighted_target is missing {key:?}")))
}

fn optional<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
}

/// Build the declaration from `{kind, covariates, ...}` where covariates are column indices.
pub(crate) fn parse(dict: &Bound<'_, PyDict>) -> PyResult<WeightedTargetDeclaration> {
    let kind: String = field(dict, "kind")?.extract()?;
    let covariates: Vec<u32> = field(dict, "covariates")?.extract()?;
    let ids: Vec<VariableId> = covariates.into_iter().map(VariableId::from_raw).collect();
    let declaration = match kind.as_str() {
        "row_weights" => {
            let weights: Vec<f64> = field(dict, "weights")?.extract()?;
            WeightedTargetDeclaration::row_weights(weights, ids)
        }
        "kernel" => {
            let family: String = field(dict, "family")?.extract()?;
            let bandwidth: f64 = field(dict, "bandwidth")?.extract()?;
            let center: Vec<f64> = field(dict, "center")?.extract()?;
            let scale: Option<Vec<f64>> =
                optional(dict, "scale")?.map(|value| value.extract()).transpose()?;
            let family = KernelFamily::parse(&family).map_err(crate::value_err_display)?;
            let kernel =
                TransportKernel::new(family, bandwidth).map_err(crate::value_err_display)?;
            WeightedTargetDeclaration::kernel(kernel, ids, center, scale.map(Arc::from))
        }
        other => {
            return Err(crate::value_err(format!(
                "unknown weighted_target kind {other:?}; use kernel or row_weights"
            )));
        }
    }
    .map_err(crate::value_err_display)?;
    match optional(dict, "min_ess_fraction")? {
        Some(value) => {
            declaration.with_min_ess_fraction(value.extract()?).map_err(crate::value_err_display)
        }
        None => Ok(declaration),
    }
}

/// The report as plain data.
pub(crate) fn report_to_dict<'py>(
    py: Python<'py>,
    report: &WeightedTargetReport,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let overlap = PyDict::new(py);
    let o = report.overlap();
    overlap.set_item("n_rows", o.n_rows)?;
    overlap.set_item("n_support", o.n_support)?;
    overlap.set_item("support_fraction", o.support_fraction)?;
    overlap.set_item("kish_ess", o.kish_ess)?;
    overlap.set_item("ess_fraction", o.ess_fraction)?;
    overlap.set_item("max_weight_share", o.max_weight_share)?;
    out.set_item("overlap", overlap)?;
    if let Some(t) = report.trial_overlap() {
        let trial = PyDict::new(py);
        trial.set_item("n_rows", t.n_rows)?;
        trial.set_item("n_support", t.n_support)?;
        trial.set_item("support_fraction", t.support_fraction)?;
        trial.set_item("kish_ess", t.kish_ess)?;
        trial.set_item("ess_fraction", t.ess_fraction)?;
        trial.set_item("max_weight_share", t.max_weight_share)?;
        out.set_item("trial_overlap", trial)?;
    }
    out.set_item("kind", if report.kernel().is_some() { "kernel" } else { "row_weights" })?;
    out.set_item("family", report.kernel().map(|k| k.family().id()))?;
    out.set_item("bandwidth", report.kernel().map(|k| k.bandwidth()))?;
    out.set_item("covariates", report.covariates().iter().map(|v| v.raw()).collect::<Vec<u32>>())?;
    out.set_item("center", report.center().map(<[f64]>::to_vec))?;
    out.set_item("scale", report.scale().map(<[f64]>::to_vec))?;
    out.set_item("min_ess_fraction", report.min_ess_fraction())?;
    out.set_item("identity", report.identity())?;
    out.set_item("distribution_id", report.distribution().raw())?;
    out.set_item("assumption", report.assumption())?;
    Ok(out)
}
