//! Multiple imputation of a joint panel's missing outcomes and confounders.
//!
//! The panel is declared exactly as for `prepare_joint_panel`; the imputation declaration
//! adds the imputation specification. Rust imputes, fits every completed panel and pools the
//! response vectors; this module only translates the declaration and the pooled result.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{Declaration, admit_declaration, compile, context, facade, panel_error};
use antecedent::data::{ImputationDomain, ImputationModel, ImputationSpec, MissingnessAssumption};
use antecedent::estimate::multiple_imputation::PooledVectorEstimate;
use pyo3::prelude::*;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImputationDeclaration {
    model: String,
    imputations: u32,
    iterations: u32,
    seed: u64,
    assumption: String,
    bootstrap_replicates: u32,
    domains: Vec<(String, String)>,
}

/// A lowered imputation: its spec and each named domain.
type NativeImputation = (ImputationSpec, Vec<(Arc<str>, ImputationDomain)>);

impl ImputationDeclaration {
    fn native(self) -> PyResult<NativeImputation> {
        let mut spec = ImputationSpec::new();
        spec.model = ImputationModel::from_name(&self.model).ok_or_else(|| {
            crate::value_err(format!(
                "unknown imputation model {:?}; use one of {:?}",
                self.model,
                ImputationModel::ALL.map(ImputationModel::name)
            ))
        })?;
        spec.assumption = MissingnessAssumption::from_name(&self.assumption).ok_or_else(|| {
            crate::value_err(format!(
                "unknown missingness assumption {:?}; use one of {:?}",
                self.assumption,
                MissingnessAssumption::ALL.map(MissingnessAssumption::name)
            ))
        })?;
        spec.imputations = self.imputations;
        spec.iterations = self.iterations;
        spec.seed = self.seed;
        spec.bootstrap_replicates = self.bootstrap_replicates;
        let domains = self
            .domains
            .into_iter()
            .map(|(name, kind)| {
                let domain = match kind.as_str() {
                    "binary" => ImputationDomain::Binary,
                    "continuous" => ImputationDomain::Continuous,
                    _ => {
                        return Err(crate::value_err(format!(
                            "unknown imputation domain {kind:?}; use binary or continuous"
                        )));
                    }
                };
                Ok((Arc::<str>::from(name), domain))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((spec, domains))
    }
}

#[derive(Deserialize)]
struct Request {
    panel: Declaration,
    imputation: ImputationDeclaration,
}

#[pyclass(name = "_PreparedJointPanelImputation", frozen)]
pub(crate) struct PyPreparedJointPanelImputation {
    inner: facade::PreparedJointPanelImputation,
}

#[pymethods]
impl PyPreparedJointPanelImputation {
    /// Impute, fit every completed panel and pool; the result is a JSON record.
    #[pyo3(signature=(*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn estimate(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let result = crate::detach_catch(py, || self.inner.estimate(&ctx).map_err(panel_error))?;
        Ok(result_json(&result).to_string())
    }
}

/// The pooled vector with how it was pooled, per coordinate and jointly, the per-imputation
/// fits and the full imputation assumptions record. A non-finite number is `null`.
fn result_json(result: &facade::JointPanelImputedResult) -> serde_json::Value {
    let assumptions = &result.imputation_assumptions;
    let (method, imputations, coordinates) = match &result.pooled {
        PooledVectorEstimate::Rubin(pooled) => (
            "rubin",
            pooled.imputations,
            pooled
                .coordinates
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "point": c.point, "within": c.within, "between": c.between,
                        "total": c.total, "fraction_missing_information":
                            c.fraction_missing_information, "df": c.df,
                    })
                })
                .collect::<Vec<_>>(),
        ),
        PooledVectorEstimate::Stacked(stacked) => (
            "stacked",
            stacked.quantities.first().map_or(0, |quantity| quantity.imputations),
            stacked
                .quantities
                .iter()
                .map(|q| {
                    serde_json::json!({
                        "point": q.mean, "within": q.within, "between": q.between,
                        "total": q.variance,
                    })
                })
                .collect::<Vec<_>>(),
        ),
        _ => ("unknown", 0, Vec::new()),
    };
    let point = result.pooled.point();
    serde_json::json!({
        "programs": result.programs,
        "horizons": result.horizons,
        "method": method,
        "imputations": imputations,
        "dimension": point.len(),
        "point": point,
        "total_covariance": result.pooled.total_covariance(),
        "coordinates": coordinates,
        "fits": result.fits,
        "imputed_cells": result.imputed_cells.iter()
            .map(|(name, count)| serde_json::json!([name.as_ref(), count]))
            .collect::<Vec<_>>(),
        "assumptions": {
            "model": assumptions.spec.model.name(),
            "imputations": assumptions.spec.imputations,
            "iterations": assumptions.spec.iterations,
            "seed": assumptions.spec.seed,
            "assumption": assumptions.spec.assumption.name(),
            "unit": assumptions.unit,
            "units": assumptions.units,
            "rows": assumptions.rows,
            "lags": assumptions.lags,
            "domains": assumptions.domains.iter()
                .map(|(name, domain)| serde_json::json!([name.as_ref(), domain.name()]))
                .collect::<Vec<_>>(),
            "binary_boundary_variables": assumptions.binary_boundary_variables.iter()
                .map(|(name, class)| serde_json::json!([name.as_ref(), class]))
                .collect::<Vec<_>>(),
            "valid_under_declared_assumption": assumptions.valid_under_declared_assumption(),
            "digest": assumptions.digest(),
        },
    })
}

/// Prepare a multiple-imputation analysis of a panel with missing outcomes or confounders.
#[pyfunction]
#[pyo3(signature=(request_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn prepare_joint_panel_imputed(
    py: Python<'_>,
    request_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedJointPanelImputation> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let _input = admit_declaration(&ctx, request_json)?;
    let request: Request =
        serde_json::from_str(request_json).map_err(|error| crate::value_err(error.to_string()))?;
    crate::detach_catch(py, || {
        let (spec, domains) = request.imputation.native()?;
        let compiled = compile(request.panel)?;
        let inner = antecedent::Study::prepare_joint_panel_imputed_with_domains(
            compiled.data,
            compiled.request,
            spec,
            &domains,
            &ctx,
        )
        .map_err(panel_error)?;
        Ok(PyPreparedJointPanelImputation { inner })
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPreparedJointPanelImputation>()?;
    module.add_function(wrap_pyfunction!(prepare_joint_panel_imputed, module)?)?;
    Ok(())
}
