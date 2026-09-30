//! Python bindings for learned continuous-outcome trial transport (2.2A cell X4).
use crate::transport_common::{
    error, execution_context, frame_named_artifact, serialization_error, unframe_named_artifact,
};
use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_estimate::{LearnedContinuousOptions, LearnerSpec, TrialAipwInput};
use antecedent_io::learned_continuous_artifact::LearnedContinuousConsumeLimits;
use pyo3::{prelude::*, types::PyBytes};
use std::sync::Arc;

/// Magic prefix of a portable learned-continuous artifact: the io crate's versioned CBOR
/// wire, framed with the variable names it was built under.
const PREFIX: &[u8] = b"ANTECEDENT-LEARNED-CONTINUOUS\x01";

/// Population mean contrast of the certified binary treatment on a continuous outcome.
const SCOPE: &str = "learned_continuous_trial_transport_point_only";

fn value_error(e: impl std::fmt::Display) -> PyErr {
    crate::value_err(e.to_string())
}

/// Parse the rows; the sampling design is checked first so clustered or undeclared
/// designs refuse by name (`learned_transport.non_iid_design`).
fn parse_input(data: &Bound<'_, PyAny>, names: &[String]) -> PyResult<TrialAipwInput> {
    let wire: String = data.call_method1("_json", (names.to_vec(),))?.extract()?;
    let mut value: serde_json::Value = serde_json::from_str(&wire).map_err(serialization_error)?;
    let sampling =
        value.get("sampling").and_then(serde_json::Value::as_str).unwrap_or_default().to_owned();
    antecedent_estimate::parse_learned_continuous_sampling(&sampling).map_err(error)?;
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

fn build(
    graph: &crate::graphs::Admg,
    selections: &[String],
    source: String,
    target: String,
    treatment: &str,
    outcome: &str,
) -> PyResult<(antecedent_graph::SelectionDiagram, TransportQuery)> {
    let treatment = resolve(graph, treatment)?;
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: resolve(graph, outcome)?,
        treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    let query = TransportQuery::new(response, source, target, [treatment]);
    let diagram = antecedent_graph::SelectionDiagram::try_new(
        graph.admg.clone(),
        selections.iter().map(|s| resolve(graph, s)).collect::<PyResult<Vec<_>>>()?,
    )
    .map_err(error)?;
    Ok((diagram, query))
}

fn estimate_json(result: &antecedent::LearnedContinuousResult, names: &[String]) -> String {
    let wire = result.wire();
    serde_json::json!({
        "status": "available",
        "scope": SCOPE,
        "estimate": wire.result.estimate,
        "uncertainty": wire.result.uncertainty,
        "overlap": wire.result.overlap,
        "diagnostics": wire.result.diagnostics,
        "provenance": wire.result.provenance,
        "folds": {"count": wire.result.folds.count, "scheme": wire.result.folds.scheme},
        "options": wire.options,
        "sampling": wire.input.sampling,
        "certificate": wire.certificate,
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
#[derive(Clone)]
struct PreparedLearnedContinuousNative {
    inner: antecedent::PreparedLearnedContinuous,
    names: Vec<String>,
    last: Option<antecedent::LearnedContinuousResult>,
    seed: u64,
    memory_bytes: Option<u64>,
}

#[pymethods]
impl PreparedLearnedContinuousNative {
    /// Cross-fit every nuisance on the retained rows and report the point. The
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
                "transport.no_execution_claim: estimate before exporting a learned continuous artifact",
            )
        })?;
        let framed = frame_named_artifact(PREFIX, &self.names, result.export().map_err(error)?)?;
        Ok(PyBytes::new(py, &framed).unbind())
    }

    /// The estimator menu for the prepared graph, query and learners.
    fn estimator_menu(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.estimator_menu()).map_err(serialization_error)
    }

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

/// Prepare the certified, learner-backed continuous-outcome mean contrast.
#[pyfunction]
#[pyo3(signature=(graph, selections, source, target, treatment, outcome, data, options, target_name, *, seed=1, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn prepare_learned_continuous(
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    source: String,
    target: String,
    treatment: String,
    outcome: String,
    data: &Bound<'_, PyAny>,
    options: String,
    target_name: String,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PreparedLearnedContinuousNative> {
    antecedent_estimate::parse_learned_continuous_target(&target_name).map_err(error)?;
    let (diagram, query) = build(graph, &selections, source, target, &treatment, &outcome)?;
    let input = parse_input(data, &graph.names)?;
    let options: LearnedContinuousOptions = serde_json::from_str(&options).map_err(value_error)?;
    let ctx = execution_context(seed, memory_bytes, cancel);
    let inner = antecedent::StudyBuilder::learned_continuous_transport(
        diagram,
        query,
        input,
        options,
        graph.names.clone(),
        &ctx,
    )
    .map_err(error)?;
    Ok(PreparedLearnedContinuousNative {
        inner,
        names: graph.names.clone(),
        last: None,
        seed,
        memory_bytes,
    })
}

/// The estimator menu for a graph, query and optional learners: inspection only.
#[pyfunction]
#[pyo3(signature=(graph, selections, source, target, treatment, outcome, learners=None))]
fn learned_continuous_estimator_menu(
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    source: String,
    target: String,
    treatment: String,
    outcome: String,
    learners: Option<String>,
) -> PyResult<String> {
    let (diagram, query) = build(graph, &selections, source, target, &treatment, &outcome)?;
    let learners = learners
        .map(|text| {
            #[derive(serde::Deserialize)]
            struct Learners {
                outcome: LearnerSpec,
                membership: LearnerSpec,
            }
            serde_json::from_str::<Learners>(&text)
                .map(|l| (l.outcome, l.membership))
                .map_err(value_error)
        })
        .transpose()?;
    let menu = antecedent::StudyBuilder::learned_continuous_menu(&diagram, &query, learners)
        .map_err(error)?;
    serde_json::to_string(&menu).map_err(serialization_error)
}

/// Independently recheck a framed artifact: re-derive the certificate, recompute the
/// folds, replay the point and diagnostics, and re-check the interval status.
#[pyfunction]
#[pyo3(signature=(artifact, *, max_rows=None, max_features=None))]
fn consume_learned_continuous(
    py: Python<'_>,
    artifact: &[u8],
    max_rows: Option<usize>,
    max_features: Option<usize>,
) -> PyResult<String> {
    let (names, bytes) = unframe_named_artifact(PREFIX, artifact, "learned continuous")?;
    let defaults = LearnedContinuousConsumeLimits::default();
    let limits = LearnedContinuousConsumeLimits {
        max_rows: max_rows.unwrap_or(defaults.max_rows),
        max_features: max_features.unwrap_or(defaults.max_features),
    };
    crate::detach_catch(py, move || {
        let result =
            antecedent::consume_learned_continuous_artifact(&bytes, limits).map_err(error)?;
        // The frame's names must be the mapping the verified identity binds.
        result.wire().check_variable_names(&names).map_err(error)?;
        Ok(estimate_json(&result, &names))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedLearnedContinuousNative>()?;
    module.add_function(wrap_pyfunction!(prepare_learned_continuous, module)?)?;
    module.add_function(wrap_pyfunction!(learned_continuous_estimator_menu, module)?)?;
    module.add_function(wrap_pyfunction!(consume_learned_continuous, module)?)?;
    Ok(())
}
