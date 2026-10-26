//! JSON view of a result's missing-data report, and parsing of the declared policy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::result::MissingDataReport;
use antecedent_data::{
    ImputationModel, ImputationSpec, MissingDeclaration, MissingPolicy, MissingnessAssumption,
    RowAccounting,
};
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
            "imputation": report.imputation.as_ref().map(imputation_json),
        })
        .to_string(),
    )
}

/// The impute-fit-pool record of a multiple-imputation result.
fn imputation_json(report: &antecedent::result::ImputationReport) -> serde_json::Value {
    serde_json::json!({
        "model": report.spec.model.name(),
        "imputations": report.spec.imputations,
        "iterations": report.spec.iterations,
        "seed": report.spec.seed,
        "assumption": report.spec.assumption.name(),
        "unit": report.unit.as_ref(),
        "units": report.units,
        "imputed_cells": report
            .imputed_cells
            .iter()
            .map(|(name, cells)| serde_json::json!({"variable": name.as_ref(), "cells": cells}))
            .collect::<Vec<_>>(),
        "estimator": report.estimator.as_ref(),
        "pooling": report.pooling.as_ref(),
        "points": report.points.as_ref(),
        "within": report.within,
        "between": report.between,
        "total": report.total,
        "df": report.df,
        "df_complete": report.df_complete,
        "fraction_missing_information": report.fraction_missing_information,
        "relative_increase": report.relative_increase,
        "assumptions_digest": report.assumptions_digest.as_ref(),
        "lags": report.lags,
        "vector": report.vector.as_ref().map(vector_json),
    })
}

/// The joint pooling of an answer with several coordinates. A non-finite number (an infinite
/// denominator degrees of freedom) is `null`.
fn vector_json(vector: &antecedent::result::ImputationVectorReport) -> serde_json::Value {
    serde_json::json!({
        "labels": vector.labels.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        "covariance_basis": vector.covariance_basis.as_ref(),
        "points": vector.points.iter().map(AsRef::as_ref).collect::<Vec<&[f64]>>(),
        "point": vector.point.as_ref(),
        "within": vector.within.as_ref(),
        "between": vector.between.as_ref(),
        "total": vector.total.as_ref(),
        "coordinate_df": vector.coordinate_df.as_ref(),
        "coordinate_fraction_missing_information":
            vector.coordinate_fraction_missing_information.as_ref(),
        "average_relative_increase": vector.average_relative_increase,
        "wald": vector.wald.as_ref().map(|wald| serde_json::json!({
            "null": wald.null.as_ref(),
            "statistic": wald.statistic,
            "df_numerator": wald.df_numerator,
            "df_denominator": wald.df_denominator,
        })),
        "extras": vector
            .extras
            .iter()
            .map(|extra| serde_json::json!({
                "label": extra.label.as_ref(),
                "points": extra.points.as_ref(),
                "point": extra.point,
                "between": extra.between,
            }))
            .collect::<Vec<_>>(),
    })
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
    let mut policy = parse_policy(&policy)?;
    if let Some(spec) = dict.get_item("imputation")?.filter(|value| !value.is_none()) {
        if policy.imputation().is_none() {
            return Err(crate::value_err(
                "an imputation specification is read only by policy 'multiple_imputation'",
            ));
        }
        policy = MissingPolicy::MultipleImputation(parse_imputation(&spec)?);
    }
    Ok(MissingDeclaration::new(policy).with_covariates(
        covariates.into_iter().map(antecedent_core::VariableId::from_raw).collect::<Vec<_>>(),
    ))
}

/// The `imputation` member of the `missing` option:
/// `{"model", "imputations", "iterations", "seed", "assumption"}`.
fn parse_imputation(value: &Bound<'_, PyAny>) -> PyResult<ImputationSpec> {
    let dict = value.cast::<PyDict>()?;
    let mut spec = ImputationSpec::new();
    if let Some(model) = dict.get_item("model")? {
        let name = model.extract::<String>()?;
        spec.model = ImputationModel::from_name(&name).ok_or_else(|| {
            crate::value_err(format!(
                "unknown imputation model {name:?}; use one of {:?}",
                ImputationModel::ALL.map(ImputationModel::name)
            ))
        })?;
    }
    if let Some(imputations) = dict.get_item("imputations")? {
        spec.imputations = imputations.extract()?;
    }
    if let Some(iterations) = dict.get_item("iterations")? {
        spec.iterations = iterations.extract()?;
    }
    if let Some(seed) = dict.get_item("seed")? {
        spec.seed = seed.extract()?;
    }
    if let Some(assumption) = dict.get_item("assumption")? {
        let name = assumption.extract::<String>()?;
        spec.assumption = MissingnessAssumption::from_name(&name).ok_or_else(|| {
            crate::value_err(format!(
                "unknown missingness assumption {name:?}; use one of {:?}",
                MissingnessAssumption::ALL.map(MissingnessAssumption::name)
            ))
        })?;
    }
    Ok(spec)
}
