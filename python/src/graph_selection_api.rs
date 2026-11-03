//! Python bridge of multiple imputation with the graph selection re-run on every completed
//! table.
//!
//! Python states the study declaratively: the data columns (missing cells as NaN), the
//! treatment and outcome columns, the Bayesian g-computation settings and the imputation
//! specification. The structure search is the exact DAG posterior over the table's columns, in
//! column order. Rust builds the effect study on each completed table and its own graph
//! posterior, runs it, and stacks the effect posteriors; the adjustment set is whatever each
//! sampled graph identifies, never a Python declaration. A refusal comes back as structured
//! JSON for the Python layer to raise as its own typed exception.
use crate::bridge::{RefusalJson, split_detail};
use crate::detach_catch;
use antecedent::discovery::BayesianDiscoverParams;
use antecedent::{
    BayesianConfig, CausalError, GraphSelectionPerImputation, InferenceMode, RefuteSuite, Study,
    estimate_with_graph_selection_per_imputation,
};
use antecedent_core::{AverageEffectQuery, VariableId, reason_code};
use antecedent_data::{ImputationDomain, ImputationModel, ImputationSpec, TabularData};
use pyo3::prelude::*;
use serde_json::json;

fn refusal(code: &str, detail: &str, message: &str) -> String {
    RefusalJson::new(code, "graph_selection_imputation", detail, message).json()
}

fn error_refusal(error: &CausalError) -> String {
    let text = error.to_string();
    let message = antecedent_core::reason_code::split_prefix(&text)
        .map_or(text.as_str(), |(_, message)| message);
    let (detail, message) = split_detail(message);
    let code = error.reason_code().unwrap_or(reason_code!("invalid_argument"));
    refusal(code, detail, message)
}

#[pyfunction]
#[doc(hidden)]
fn graph_selection_imputation_interval() -> String {
    match GraphSelectionPerImputation::interval() {
        Err(error) => error_refusal(&error),
        Ok(never) => match never {},
    }
}

#[pyfunction]
#[pyo3(signature = (
    names, columns, treatment, outcome, *, model="chained_equations".to_owned(), imputations=20,
    iterations=10, seed=1, n_draws=1000, prior_scale=10.0, domains=Vec::new()
))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn graph_selection_imputation_run(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    treatment: String,
    outcome: String,
    model: String,
    imputations: u32,
    iterations: u32,
    seed: u64,
    n_draws: usize,
    prior_scale: f64,
    domains: Vec<(String, String)>,
) -> PyResult<(Option<String>, Option<String>)> {
    detach_catch(py, move || {
        let invalid = |detail: &str, message: String| {
            Ok((None, Some(refusal(reason_code!("invalid_argument"), detail, &message))))
        };
        if names.len() != columns.len() || names.is_empty() {
            return invalid(
                "graph_selection_imputation.columns",
                "names and columns must be non-empty and of equal length".to_owned(),
            );
        }
        let position = |name: &str| names.iter().position(|n| n == name);
        let (Some(t), Some(y)) = (position(&treatment), position(&outcome)) else {
            return invalid(
                "graph_selection_imputation.variables",
                format!("treatment `{treatment}` and outcome `{outcome}` must be data columns"),
            );
        };
        let Some(model) = ImputationModel::from_name(&model) else {
            return invalid(
                "graph_selection_imputation.model",
                format!("unknown imputation model `{model}`"),
            );
        };
        let table = match TabularData::from_f64_columns(
            names.iter().zip(&columns).map(|(name, column)| (name.as_str(), column.as_slice())),
        ) {
            Ok(table) => table,
            Err(error) => return invalid("graph_selection_imputation.data", error.to_string()),
        };
        let raw = |i: usize| VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        let variables: Vec<VariableId> = (0..names.len()).map(raw).collect();
        let query = AverageEffectQuery::binary_ate(raw(t), raw(y));
        let spec = ImputationSpec::new()
            .with_model(model)
            .with_imputations(imputations)
            .with_iterations(iterations)
            .with_seed(seed);
        let mut parsed_domains = Vec::with_capacity(domains.len());
        for (name, domain) in domains {
            let domain = match domain.as_str() {
                "binary" => ImputationDomain::Binary,
                "continuous" => ImputationDomain::Continuous,
                _ => {
                    return invalid(
                        "graph_selection_imputation.domain",
                        format!("unknown domain {domain:?}"),
                    );
                }
            };
            parsed_domains.push((std::sync::Arc::<str>::from(name), domain));
        }
        let selection =
            GraphSelectionPerImputation::new(BayesianDiscoverParams::default(), variables)
                .with_domains(parsed_domains);
        let ctx = crate::execution_api::context(seed);
        let result = estimate_with_graph_selection_per_imputation(
            &table,
            spec,
            &selection,
            &ctx,
            |completed, posterior| {
                Study::tabular(completed.clone())
                    .graph_posterior(posterior)
                    .query(query.clone())
                    .inference(InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale),
                    ))
                    .refute(RefuteSuite::None)
            },
        );
        let result = match result {
            Ok(result) => result,
            Err(error) => return Ok((None, Some(error_refusal(&error)))),
        };
        let Some(effect) = result.estimate().as_effect() else {
            return invalid(
                "graph_selection_imputation.result",
                "the pooled answer is not a scalar effect".to_owned(),
            );
        };
        let Some(report) = result.missing_data().as_ref().and_then(|m| m.imputation.as_ref())
        else {
            return invalid(
                "graph_selection_imputation.result",
                "the pooled result carries no imputation report".to_owned(),
            );
        };
        let mut payload = json!({
            "effect": effect.ate,
            "posterior_sd": effect.uncertainty.analytic.se,
            "imputations": report.spec.imputations,
            "points": report.points.as_ref(),
            "within": report.within,
            "between": report.between,
            "total": report.total,
            "pooling": report.pooling.as_ref(),
            "estimator": report.estimator.as_ref(),
            "imputed_cells": report
                .imputed_cells
                .iter()
                .map(|(name, cells)| json!({"variable": name.as_ref(), "cells": cells}))
                .collect::<Vec<_>>(),
            "assumptions_digest": report.assumptions_digest.as_ref(),
            "assumption": report.spec.assumption.name(),
            "structure": "graph_selection_rerun_per_imputation",
        });
        crate::missing_report::add_imputation_domains(&mut payload, report);
        Ok((Some(payload.to_string()), None))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(graph_selection_imputation_run, module)?)?;
    module.add_function(wrap_pyfunction!(graph_selection_imputation_interval, module)?)?;
    module.add_class::<PyCompiledGraphImputation>()?;
    module.add_class::<PyConsumedGraphImputation>()?;
    module.add_function(wrap_pyfunction!(consume_compiled_graph_imputation, module)?)?;
    Ok(())
}

// This handle owns the retained native compiler output; Python supplies no scientific callback.
#[pyclass(name = "CompiledGraphImputationHandle", unsendable)]
struct PyCompiledGraphImputation {
    plan: antecedent::analysis::graph_imputation::CompiledGraphImputation,
    point: Option<antecedent::analysis::graph_imputation::GraphImputationPoint>,
    ctx: antecedent_core::ExecutionContext,
}

fn compiled_py_error(message: impl std::fmt::Display) -> PyErr {
    crate::errors::value_error_of(&refusal(
        reason_code!("invalid_argument"),
        "graph_imputation.compiled",
        &message.to_string(),
    ))
}

fn compiled_runtime_error(
    error: antecedent::analysis::graph_imputation::GraphImputationError,
) -> PyErr {
    use antecedent::analysis::graph_imputation::GraphImputationError;
    use antecedent_core::search::SearchStop;
    let (code, detail, message) = match error {
        GraphImputationError::PointUnavailable | GraphImputationError::IntervalUnavailable => {
            (reason_code!("cell_not_licensed"), "graph_imputation.unlicensed", error.to_string())
        }
        GraphImputationError::SearchStopped(receipt) => {
            let code = if receipt.stop == SearchStop::Cancelled {
                reason_code!("cancelled_no_claim")
            } else {
                reason_code!("execution_resource_limit")
            };
            (code, "graph_imputation.search_stopped", receipt.summary())
        }
        GraphImputationError::Engine(CausalError::Cancelled { stage }) => {
            (reason_code!("cancelled_no_claim"), "graph_imputation.cancelled", stage.to_owned())
        }
        GraphImputationError::Engine(CausalError::Resource { message }) => {
            (reason_code!("execution_resource_limit"), "graph_imputation.resource_limit", message)
        }
        GraphImputationError::Engine(error) => {
            return crate::errors::value_error_of(&error_refusal(&error));
        }
        other => (
            reason_code!("invalid_argument"),
            "graph_imputation.execution_refused",
            other.to_string(),
        ),
    };
    crate::errors::value_error_of(&refusal(code, detail, &message))
}

type CompiledArtifactParts = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

#[pymethods]
impl PyCompiledGraphImputation {
    #[new]
    #[pyo3(signature = (names, columns, treatment, outcome, graph_support, domains, imputations, iterations, n_draws, prior_scale, seed))]
    fn new(
        names: Vec<String>,
        columns: Vec<Vec<f64>>,
        treatment: String,
        outcome: String,
        graph_support: String,
        domains: Vec<String>,
        imputations: u32,
        iterations: u32,
        n_draws: u32,
        prior_scale: f64,
        seed: u64,
    ) -> PyResult<Self> {
        use antecedent::analysis::graph_imputation::{
            CompiledGraphImputation, GraphImputationSpec, GraphImputationSupport,
        };
        if !(2..=antecedent_core::graph_imputation_contract::MAX_VARIABLES).contains(&names.len())
            || names.len() != columns.len()
            || names.len() != domains.len()
            || columns.iter().any(|column| !(4..=10_000).contains(&column.len()))
        {
            return Err(compiled_py_error("column shape is outside the bounded candidate"));
        }
        let position = |name: &str| {
            names
                .iter()
                .position(|n| n == name)
                .and_then(|index| u32::try_from(index).ok())
                .map(VariableId::from_raw)
        };
        let treatment =
            position(&treatment).ok_or_else(|| compiled_py_error("unknown treatment"))?;
        let outcome = position(&outcome).ok_or_else(|| compiled_py_error("unknown outcome"))?;
        let support = GraphImputationSupport::from_label(&graph_support).ok_or_else(|| {
            compiled_py_error(
                "unknown graph support; declare `treatment_root_direct_effect` or \
                 `treatment_direct_effect`",
            )
        })?;
        let domains = domains
            .iter()
            .map(|domain| match domain.as_str() {
                "continuous" => Ok(0),
                "binary" => Ok(1),
                _ => Err(compiled_py_error("unknown variable domain")),
            })
            .collect::<PyResult<Vec<u8>>>()?;
        let source = TabularData::from_f64_columns(
            names.iter().zip(&columns).map(|(name, values)| (name.as_str(), values.as_slice())),
        )
        .map_err(compiled_py_error)?;
        let spec = GraphImputationSpec {
            treatment,
            outcome,
            support,
            imputation: ImputationSpec::new()
                .with_imputations(imputations)
                .with_iterations(iterations)
                .with_seed(seed),
            domains: std::sync::Arc::from(domains),
            posterior_draws: n_draws,
            prior_scale,
            execution_seed: seed,
        };
        let ctx = crate::execution_api::context(seed);
        let plan =
            CompiledGraphImputation::compile(source, &spec, &ctx).map_err(compiled_runtime_error)?;
        Ok(Self { plan, point: None, ctx })
    }

    fn execute(&mut self) -> PyResult<String> {
        let point = self.plan.execute(&self.ctx).map_err(compiled_runtime_error)?;
        let result = json!({"mean": point.mean, "diagnostic_sd": point.diagnostic_sd,
            "receipt": point.receipt.to_vec(), "unidentified_by_table": &point.unidentified_by_table});
        self.point = Some(point);
        Ok(result.to_string())
    }

    fn export(&self) -> PyResult<CompiledArtifactParts> {
        let point =
            self.point.as_ref().ok_or_else(|| compiled_py_error("execute before exporting"))?;
        let artifact = self
            .plan
            .export(point, &self.ctx)
            .map_err(|error| compiled_runtime_error(error.into()))?;
        let expected = artifact.expectation();
        Ok((
            artifact.bytes().to_vec(),
            expected.program.to_vec(),
            expected.source.to_vec(),
            expected.receipt.to_vec(),
        ))
    }

    fn measured_interval(&self) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
        let point = self
            .point
            .as_ref()
            .ok_or_else(|| compiled_py_error("execute before requesting interval"))?;
        measured_of(point)
    }

    fn interval(&self) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
        self.measured_interval()
    }
}

fn measured_of(
    point: &antecedent::analysis::graph_imputation::GraphImputationPoint,
) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
    let measured = point.measured().map_err(compiled_runtime_error)?;
    crate::measured_inference_api::NativeMeasuredInference::from_io(measured)
}

// The replayed point of a consumed artifact; it retains the original source bytes, so the
// measured route is available exactly as on the handle that executed.
#[pyclass(name = "ConsumedGraphImputationHandle", unsendable)]
struct PyConsumedGraphImputation {
    point: antecedent::analysis::graph_imputation::GraphImputationPoint,
}

#[pymethods]
impl PyConsumedGraphImputation {
    fn measured_interval(&self) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
        measured_of(&self.point)
    }

    fn interval(&self) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
        self.measured_interval()
    }
}

#[pyfunction]
fn consume_compiled_graph_imputation(
    bytes: Vec<u8>,
    program: Vec<u8>,
    source: Vec<u8>,
    receipt: Vec<u8>,
    seed: u64,
) -> PyResult<(String, PyConsumedGraphImputation)> {
    use antecedent::analysis::graph_imputation::{GraphImputationExpectation, consume};
    let digest = |bytes: Vec<u8>| {
        <[u8; 32]>::try_from(bytes)
            .map_err(|_| compiled_py_error("expected identity must contain 32 bytes"))
    };
    let expected = GraphImputationExpectation {
        program: digest(program)?,
        source: digest(source)?,
        receipt: digest(receipt)?,
    };
    let point = consume(&bytes, expected, &crate::execution_api::context(seed))
        .map_err(compiled_runtime_error)?;
    let text = json!({"mean": point.mean, "diagnostic_sd": point.diagnostic_sd,
        "receipt": point.receipt.to_vec(), "unidentified_by_table": &point.unidentified_by_table})
    .to_string();
    Ok((text, PyConsumedGraphImputation { point }))
}
