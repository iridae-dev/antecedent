//! Python bridge for the logistic g-computation run through admitted recipes.
//!
//! Python declares the outcome-model provider (its id and name; the capabilities and the
//! answered log density are the recipe pairing's own) and hands over numeric columns. The
//! provider capability check, the score-gradient check, the Newton solve, the sandwich
//! covariance and the standardized means are Rust's.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recipe_glm::{logistic_provider, logistic_standardized_effect};
use antecedent_core::{ProviderCapability, ProviderDeclaration};
use antecedent_data::TabularData;
use antecedent_data::TableView;
use pyo3::prelude::*;

use crate::transport_common::execution_context;
use crate::{PyCancellationToken, detach_catch, py_err, value_err};

/// The provider declaration the logistic fit requires, with its log density answered.
#[pyclass(name = "LogisticProvider", frozen)]
pub struct PyLogisticProvider {
    declaration: ProviderDeclaration,
}

#[pymethods]
impl PyLogisticProvider {
    /// Declare provider `id` named `name`.
    #[new]
    fn new(id: u64, name: &str) -> PyResult<Self> {
        Ok(Self { declaration: logistic_provider(id, name).map_err(py_err)? })
    }

    #[getter]
    fn id(&self) -> u64 {
        self.declaration.id().raw()
    }

    #[getter]
    fn name(&self) -> String {
        self.declaration.name().to_owned()
    }

    /// Capabilities the provider declares, by registered name.
    #[getter]
    fn capabilities(&self) -> Vec<String> {
        [
            (ProviderCapability::DifferentiableLogDensity, "differentiable_log_density"),
            (ProviderCapability::EstimatingEquations, "estimating_equations"),
            (ProviderCapability::Prediction, "prediction"),
            (ProviderCapability::Density, "density"),
            (ProviderCapability::Sampling, "sampling"),
        ]
        .into_iter()
        .filter(|(capability, _)| self.declaration.supports(*capability))
        .map(|(_, name)| name.to_owned())
        .collect()
    }
}

/// Effect, treated mean, control mean, coefficients, covariance, iterations and rows.
type RecipeAteParts = (f64, f64, f64, Vec<f64>, Vec<f64>, usize, u64);

/// Fit the logistic outcome model with the generalized-linear recipe and average its
/// predictions at `T = 1` and `T = 0` with the g-computation recipe.
#[pyfunction]
#[pyo3(signature = (columns, treatment, outcome, adjustment, provider, cancel=None))]
fn recipe_logistic_standardized_effect(
    py: Python<'_>,
    columns: Vec<(String, Vec<f64>)>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    provider: PyRef<'_, PyLogisticProvider>,
    cancel: Option<PyCancellationToken>,
) -> PyResult<RecipeAteParts> {
    if columns.is_empty() {
        return Err(value_err("the logistic recipe fit needs at least one column"));
    }
    let declaration = provider.declaration.clone();
    let (treatment, outcome) = (treatment.to_owned(), outcome.to_owned());
    detach_catch(py, move || {
        let pairs: Vec<(&str, &[f64])> =
            columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
        let data = TabularData::from_f64_columns(pairs).map_err(py_err)?;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let mut z_ids = Vec::with_capacity(adjustment.len());
        for name in &adjustment {
            z_ids.push(data.schema().id_of(name).map_err(py_err)?);
        }
        let ctx = execution_context(0, None, cancel);
        let fit = logistic_standardized_effect(&data, t_id, y_id, &z_ids, &declaration, &ctx)
            .map_err(py_err)?;
        Ok((
            fit.effect,
            fit.treated_mean,
            fit.control_mean,
            fit.coefficients,
            fit.covariance,
            fit.iterations,
            fit.rows,
        ))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLogisticProvider>()?;
    module.add_function(wrap_pyfunction!(recipe_logistic_standardized_effect, module)?)?;
    Ok(())
}
