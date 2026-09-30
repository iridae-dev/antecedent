//! Python bindings for the smoothed dose-response transport grid (2.2B cell X4).
use crate::transport_common::{
    error, execution_context, frame_named_artifact, serialization_error, unframe_named_artifact,
};
use antecedent_core::{SmoothedDoseTransportQuery, VariableId};
use antecedent_estimate::{SmoothedDoseInput, SmoothedDoseOptions};
use antecedent_io::smoothed_dose_artifact::SmoothedDoseConsumeLimits;
use pyo3::{prelude::*, types::PyBytes};
use std::sync::Arc;

/// Magic prefix of a portable smoothed-dose artifact: the io crate's versioned CBOR
/// wire, framed with the variable names it was built under.
const PREFIX: &[u8] = b"ANTECEDENT-SMOOTHED-DOSE\x01";

/// Fixed-bandwidth smoothed dose response at the declared grid, point only.
const SCOPE: &str = "smoothed_dose_response_transport_point_only";

fn value_error(e: impl std::fmt::Display) -> PyErr {
    crate::value_err(e.to_string())
}

/// Parse the rows; the sampling design is checked first so clustered or undeclared
/// designs refuse by name (`dose_response.non_iid_design`).
fn parse_input(data: &Bound<'_, PyAny>, names: &[String]) -> PyResult<SmoothedDoseInput> {
    let wire: String = data.call_method1("_json", (names.to_vec(),))?.extract()?;
    let mut value: serde_json::Value = serde_json::from_str(&wire).map_err(serialization_error)?;
    let sampling =
        value.get("sampling").and_then(serde_json::Value::as_str).unwrap_or_default().to_owned();
    antecedent_estimate::smoothed_dose::parse_smoothed_dose_sampling(&sampling).map_err(error)?;
    value["sampling"] = sampling.into();
    serde_json::from_value(value).map_err(value_error)
}

fn resolve(graph: &crate::graphs::Admg, name: &str) -> PyResult<VariableId> {
    graph
        .names
        .iter()
        .position(|n| n == name)
        .and_then(|i| u32::try_from(i).ok())
        .map(VariableId::from_raw)
        .ok_or_else(|| value_error(format!("unknown variable {name}")))
}

/// The query's frozen shape as Python passes it.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QuerySpec {
    source: String,
    target: String,
    dose: String,
    outcome: String,
    grid: Vec<f64>,
    bandwidth: f64,
    dose_support: (f64, f64),
    kernel: String,
    density_provenance: String,
}

fn build(
    graph: &crate::graphs::Admg,
    selections: &[String],
    spec: &str,
) -> PyResult<(antecedent_graph::SelectionDiagram, SmoothedDoseTransportQuery)> {
    let spec: QuerySpec = serde_json::from_str(spec).map_err(value_error)?;
    let kernel =
        antecedent_estimate::smoothed_dose::parse_smoothing_kernel(&spec.kernel).map_err(error)?;
    let query = SmoothedDoseTransportQuery {
        outcome: resolve(graph, &spec.outcome)?,
        dose: resolve(graph, &spec.dose)?,
        source_population: Arc::from(spec.source.as_str()),
        target_population: Arc::from(spec.target.as_str()),
        grid: Arc::from(spec.grid.as_slice()),
        bandwidth: spec.bandwidth,
        kernel,
        dose_support: spec.dose_support,
        density_provenance: Arc::from(spec.density_provenance.as_str()),
    };
    let diagram = antecedent_graph::SelectionDiagram::try_new(
        graph.admg.clone(),
        selections.iter().map(|s| resolve(graph, s)).collect::<PyResult<Vec<_>>>()?,
    )
    .map_err(error)?;
    Ok((diagram, query))
}

fn estimate_json(result: &antecedent::SmoothedDoseResult, names: &[String]) -> String {
    let wire = result.wire();
    serde_json::json!({
        "status": "available",
        "scope": SCOPE,
        "grid": wire.result.grid,
        "uncertainty": wire.result.uncertainty,
        "overlap": wire.result.overlap,
        "diagnostics": wire.result.diagnostics,
        "provenance": wire.result.provenance,
        "folds": {"count": wire.result.folds.count, "scheme": wire.result.folds.scheme},
        "options": wire.options,
        "sampling": wire.input.sampling,
        "certificate": wire.certificate,
        "query": wire.query,
        "seed": wire.seed,
        "premises_digest": wire.premises_digest,
        "data_digest": wire.data_digest,
        "evidence_digest": wire.evidence_digest,
        "execution_id": result.identity(),
        "variable_names": names,
    })
    .to_string()
}

#[pyclass(skip_from_py_object)]
#[doc(hidden)]
#[derive(Clone)]
struct PreparedSmoothedDoseNative {
    inner: antecedent::PreparedSmoothedDose,
    names: Vec<String>,
    last: Option<antecedent::SmoothedDoseResult>,
    seed: u64,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl PreparedSmoothedDoseNative {
    /// Cross-fit the nuisances on the retained rows and report the grid. The
    /// certificate derived at preparation is reused.
    #[pyo3(signature=(cancel=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = execution_context(self.seed, self.memory_bytes, cancel);
        let inner = self.inner.clone();
        let result = crate::detach_catch(py, move || inner.estimate(&ctx).map_err(error))?;
        let payload = estimate_json(&result, &self.names);
        self.last = Some(result);
        Ok(payload)
    }

    /// The closed interval route: always refuses with `cell_not_licensed`.
    #[pyo3(signature=(cancel=None))]
    fn interval(&self, py: Python<'_>, cancel: Option<crate::PyCancellationToken>) -> PyResult<()> {
        let ctx = execution_context(self.seed, self.memory_bytes, cancel);
        let inner = self.inner.clone();
        crate::detach_catch(py, move || inner.interval(&ctx).map(|_| ()).map_err(error))
    }

    /// Replace the rows with a compatible snapshot; clears the last claim.
    #[pyo3(signature=(data,cancel=None))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let input = parse_input(data, &self.names)?;
        let ctx = execution_context(self.seed, self.memory_bytes, cancel);
        let inner = self.inner.clone();
        self.inner = crate::detach_catch(py, move || inner.refresh(input, &ctx).map_err(error))?;
        self.last = None;
        Ok(())
    }

    /// The last execution as a framed, independently consumable artifact.
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: estimate before exporting a smoothed dose artifact",
            )
        })?;
        let framed = frame_named_artifact(PREFIX, &self.names, result.export().map_err(error)?)?;
        Ok(PyBytes::new(py, &framed).unbind())
    }

    /// The estimator menu for the prepared graph, query and options.
    fn estimator_menu(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.estimator_menu()).map_err(serialization_error)
    }

    #[doc(hidden)]
    fn last_result(&self) -> PyResult<String> {
        let result = self.last.as_ref().ok_or_else(|| {
            crate::refusal(
                antecedent_core::reason_code!("not_executed"),
                "transport.no_execution_claim: estimate before reading a result",
            )
        })?;
        Ok(estimate_json(result, &self.names))
    }
}

/// Prepare the certified smoothed dose-response transport grid.
#[pyfunction]
#[pyo3(signature=(graph, selections, query, data, options, target_name, *, seed=1, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn prepare_smoothed_dose(
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    query: &str,
    data: &Bound<'_, PyAny>,
    options: &str,
    target_name: &str,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedSmoothedDoseNative> {
    antecedent_estimate::smoothed_dose::parse_smoothed_dose_target(target_name).map_err(error)?;
    let (diagram, query) = build(graph, &selections, query)?;
    let input = parse_input(data, &graph.names)?;
    let options: SmoothedDoseOptions = serde_json::from_str(options).map_err(value_error)?;
    let ctx = execution_context(seed, memory_bytes, cancel);
    let inner = antecedent::StudyBuilder::smoothed_dose_transport(
        diagram,
        query,
        input,
        options,
        graph.names.clone(),
        &ctx,
    )
    .map_err(error)?;
    Ok(PreparedSmoothedDoseNative {
        inner,
        names: graph.names.clone(),
        last: None,
        seed,
        memory_bytes,
    })
}

/// The estimator menu for a graph, smoothed-dose query and optional options.
#[pyfunction]
#[doc(hidden)]
#[pyo3(signature=(graph, selections, query, options=None))]
fn smoothed_dose_estimator_menu(
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    query: &str,
    options: Option<&str>,
) -> PyResult<String> {
    let (diagram, query) = build(graph, &selections, query)?;
    let options: Option<SmoothedDoseOptions> =
        options.map(serde_json::from_str).transpose().map_err(value_error)?;
    let menu = antecedent::StudyBuilder::smoothed_dose_menu(&diagram, &query, options.as_ref())
        .map_err(error)?;
    serde_json::to_string(&menu).map_err(serialization_error)
}

/// Independently recheck a framed artifact: re-derive the certificate, recompute the
/// folds, re-predict from the stored fold models and replay the grid.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_rows=None, max_features=None))]
fn consume_smoothed_dose(
    py: Python<'_>,
    artifact: &[u8],
    max_rows: Option<usize>,
    max_features: Option<usize>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(PREFIX, artifact, "smoothed dose")?;
    let defaults = SmoothedDoseConsumeLimits::default();
    let limits = SmoothedDoseConsumeLimits {
        max_rows: max_rows.unwrap_or(defaults.max_rows),
        max_features: max_features.unwrap_or(defaults.max_features),
    };
    crate::detach_catch(py, move || {
        let result = antecedent::consume_smoothed_dose_artifact(&bytes, limits).map_err(error)?;
        result.wire().check_variable_names(&names).map_err(error)?;
        Ok(estimate_json(&result, &names))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedSmoothedDoseNative>()?;
    module.add_function(wrap_pyfunction!(prepare_smoothed_dose, module)?)?;
    module.add_function(wrap_pyfunction!(smoothed_dose_estimator_menu, module)?)?;
    module.add_function(wrap_pyfunction!(consume_smoothed_dose, module)?)?;
    Ok(())
}
