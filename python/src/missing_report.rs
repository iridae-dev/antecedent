//! JSON view of a result's missing-data report, and parsing of the declared policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::result::MissingDataReport;
use antecedent_data::{MissingDeclaration, MissingPolicy, RowAccounting};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// The row accounting as a JSON value.
pub(crate) fn accounting_json(accounting: &RowAccounting) -> serde_json::Value {
    serde_json::json!({
        "policy": accounting.policy.name(),
        "rows_total": accounting.rows_total,
        "rows_kept": accounting.rows_kept,
        "rows_dropped": accounting.rows_dropped(),
        "rows_in_denominator": accounting.rows_in_denominator,
        "missing_by_variable": accounting
            .missing_by_variable
            .iter()
            .map(|(name, rows)| serde_json::json!({"variable": name.as_ref(), "rows": rows}))
            .collect::<Vec<_>>(),
        "reasons": accounting
            .reasons
            .iter()
            .map(|(reason, rows)| serde_json::json!({"reason": reason.name(), "rows": rows}))
            .collect::<Vec<_>>(),
        "dropped_rows": accounting.dropped_rows.as_ref(),
        "dropped_digest": accounting.dropped_digest.as_ref(),
        "groups": accounting
            .groups
            .iter()
            .map(|g| serde_json::json!({
                "label": g.label.as_ref(),
                "total": g.total,
                "kept": g.kept,
                "observed": g.observed,
            }))
            .collect::<Vec<_>>(),
        "units": accounting.units.map(|u| serde_json::json!({"total": u.total, "dropped": u.dropped})),
    })
}

/// The report of a result as a JSON string, `None` when the route reports none.
pub(crate) fn report_json(report: Option<&MissingDataReport>) -> Option<String> {
    let report = report?;
    Some(
        serde_json::json!({
            "declared": report.declared,
            "assumption": report.assumption.as_ref(),
            "covariates": report.covariates.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
            "accounting": accounting_json(&report.accounting),
        })
        .to_string(),
    )
}

/// A policy by name; the refusal lists the names.
pub(crate) fn parse_policy(name: &str) -> PyResult<MissingPolicy> {
    MissingPolicy::from_name(name).ok_or_else(|| {
        crate::value_err(format!(
            "unknown missing-data policy {name:?}; use one of {:?}",
            MissingPolicy::NAMES
        ))
    })
}

/// The `missing` prepare option: `{"policy": name, "covariates": [column positions]}`.
pub(crate) fn parse_declaration(value: &Bound<'_, PyAny>) -> PyResult<MissingDeclaration> {
    let dict = value.cast::<PyDict>()?;
    let policy = dict
        .get_item("policy")?
        .ok_or_else(|| crate::value_err("missing option needs a policy"))?
        .extract::<String>()?;
    let covariates: Vec<u32> = match dict.get_item("covariates")? {
        Some(list) if !list.is_none() => list.extract()?,
        _ => Vec::new(),
    };
    Ok(MissingDeclaration::new(parse_policy(&policy)?).with_covariates(
        covariates.into_iter().map(antecedent_core::VariableId::from_raw).collect::<Vec<_>>(),
    ))
}
