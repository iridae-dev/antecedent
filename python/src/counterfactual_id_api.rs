//! Python bindings for counterfactual identification of the effect of
//! treatment on the treated on a bounded ADMG (2.2B X8).
//!
//! The Python layer owns names and argument shapes; identification, evaluation
//! and the artifact consumer are the Rust route's, so a Python answer is the
//! Rust answer bit for bit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::counterfactual_id::{
    CounterfactualIdConsumeLimits, CounterfactualIdEffect, CounterfactualIdOptions,
    PreparedCounterfactualId, consume_counterfactual_id_artifact, prepare_counterfactual_id,
};
use antecedent_core::{CounterfactualEventQuery, RegimeId, SearchLimits, Value, VariableId};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_io::counterfactual_id_artifact::CounterfactualIdArtifactError;
use pyo3::prelude::*;

use crate::{detach_catch, py_err, py_execution_context};

/// A reason-coded facade error as its Python class: `invalid_argument` is the
/// value class, every other registered refusal the unsupported class, each with
/// its `reason_code`.
fn refused(error: antecedent::CausalError) -> PyErr {
    let text = error.to_string();
    let Some((code, rest)) = antecedent_core::reason_code::split_prefix(&text) else {
        return py_err(error);
    };
    if !antecedent_core::reason_code::is_runtime_refusal(code) {
        return py_err(error);
    }
    if code == "invalid_argument" {
        return crate::with_reason_code(crate::value_err(rest), code);
    }
    crate::refusal(code, rest)
}

/// A malformed call (unknown names, a law of the wrong shape) is the
/// `counterfactual_id.invalid_query` detail under `invalid_argument`.
fn invalid_query(message: impl std::fmt::Display) -> PyErr {
    crate::with_reason_code(
        crate::value_err(format!("counterfactual_id.invalid_query: {message}")),
        "invalid_argument",
    )
}

/// A consumer refusal with its registered `(reason code, detail)` pair.
fn artifact_refused(error: &CounterfactualIdArtifactError) -> PyErr {
    let (code, detail) = error.reason();
    let message = format!("{detail}: {error}");
    if code == "invalid_argument" {
        crate::with_reason_code(crate::value_err(message), code)
    } else {
        crate::refusal(code, message)
    }
}

/// The result as JSON and the exported artifact bytes.
fn payload(effect: &CounterfactualIdEffect) -> PyResult<(String, Vec<u8>)> {
    let artifact = effect.export_artifact().map_err(py_err)?;
    let derivation = effect.derivation();
    let json = serde_json::json!({
        "probability": effect.probability,
        "probability_bits": format!("{:016x}", effect.probability.to_bits()),
        "conditioning_probability": effect.conditioning_probability,
        "outcome_distribution": effect.outcome_distribution,
        "counterfactual_mean": effect.counterfactual_mean,
        "observed_mean": effect.observed_mean,
        "effect": effect.effect,
        "data_digest": effect.data_digest,
        "independently_verified": effect.independently_verified,
        "derivation": derivation.canonical_text(),
        "counterfactual_graph": derivation.counterfactual_graph,
        "search": derivation.search,
        "query_text": derivation.query_text,
        "names": effect.names(),
    });
    Ok((json.to_string(), artifact))
}

/// A decided query: evaluate it on any law of its variables and levels
/// without identifying again.
#[pyclass(skip_from_py_object, frozen)]
struct NativePreparedCounterfactualId {
    prepared: PreparedCounterfactualId,
    levels: Vec<Vec<f64>>,
}

#[pymethods]
impl NativePreparedCounterfactualId {
    /// Evaluate on a law given as `probabilities` (a supplied exact law) or
    /// `counts` (an empirical count table), row-major over the variables in
    /// name order, last fastest: `(json, artifact bytes)`.
    #[pyo3(signature = (*, probabilities=None, counts=None, seed=0))]
    fn evaluate(
        &self,
        py: Python<'_>,
        probabilities: Option<Vec<f64>>,
        counts: Option<Vec<u64>>,
        seed: u64,
    ) -> PyResult<(String, Vec<u8>)> {
        let law = law_of(&self.levels, probabilities, counts)?;
        let prepared = self.prepared.clone();
        detach_catch(py, move || {
            let effect =
                prepared.evaluate(&law, &py_execution_context(seed, 1)).map_err(refused)?;
            payload(&effect)
        })
    }

    /// Canonical text of the derivation.
    #[getter]
    fn derivation(&self) -> String {
        self.prepared.derivation().canonical_text()
    }

    /// The search accounting of the decision, as JSON.
    #[getter]
    fn search(&self) -> String {
        serde_json::to_string(&self.prepared.derivation().search).unwrap_or_default()
    }
}

fn law_of(
    levels: &[Vec<f64>],
    probabilities: Option<Vec<f64>>,
    counts: Option<Vec<u64>>,
) -> PyResult<ExactDiscreteLaw> {
    let axes: Vec<DiscreteAxis> = levels
        .iter()
        .enumerate()
        .map(|(i, l)| DiscreteAxis {
            variable: VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX)),
            values: l.iter().map(|x| Value::f64(*x)).collect::<Vec<_>>().into(),
        })
        .collect();
    let tolerance = LawTolerance::default();
    match (probabilities, counts) {
        (Some(probabilities), None) => ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            axes,
            probabilities,
            "python",
            tolerance,
        )
        .map_err(invalid_query),
        (None, Some(counts)) => {
            let total: u64 = counts.iter().sum();
            if total == 0 {
                return Err(invalid_query("the count table is empty"));
            }
            #[allow(clippy::cast_precision_loss, reason = "counts are small cell totals")]
            let probabilities: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
            ExactDiscreteLaw::try_empirical(
                "target",
                RegimeId::from_raw(0),
                [],
                axes,
                probabilities,
                "python",
                tolerance,
            )
            .and_then(|law| law.with_empirical_counts(counts))
            .map_err(invalid_query)
        }
        _ => Err(invalid_query("give exactly one of probabilities or counts")),
    }
}

/// Decide the effect of treatment on the treated once.
#[pyfunction]
#[pyo3(signature = (names, directed, bidirected, levels, treatment, active, observed, outcome, outcome_level, *, operations=20_000, depth=48, interval_requested=false, seed=0, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one native entry mirrors the Python signature")]
fn prepare_counterfactual_id_native(
    py: Python<'_>,
    names: Vec<String>,
    directed: Vec<(String, String)>,
    bidirected: Vec<(String, String)>,
    levels: Vec<Vec<f64>>,
    treatment: String,
    active: f64,
    observed: f64,
    outcome: String,
    outcome_level: f64,
    operations: usize,
    depth: usize,
    interval_requested: bool,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<NativePreparedCounterfactualId> {
    let id = |name: &str| -> PyResult<u32> {
        names
            .iter()
            .position(|n| n == name)
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| invalid_query(format!("unknown variable {name:?}")))
    };
    let n = u32::try_from(names.len()).map_err(invalid_query)?;
    let mut graph = Admg::with_variables(n);
    for (a, b) in &directed {
        graph
            .insert_directed(DenseNodeId::from_raw(id(a)?), DenseNodeId::from_raw(id(b)?))
            .map_err(invalid_query)?;
    }
    for (a, b) in &bidirected {
        graph
            .insert_bidirected(DenseNodeId::from_raw(id(a)?), DenseNodeId::from_raw(id(b)?))
            .map_err(invalid_query)?;
    }
    let query = CounterfactualEventQuery::effect_on_treated(
        VariableId::from_raw(id(&treatment)?),
        active,
        observed,
        VariableId::from_raw(id(&outcome)?),
        outcome_level,
    )
    .map_err(invalid_query)?;
    let options = CounterfactualIdOptions {
        limits: SearchLimits { operations, depth },
        interval_requested,
        ..CounterfactualIdOptions::default()
    };
    let kept_levels = levels.clone();
    let prepared = detach_catch(py, move || {
        prepare_counterfactual_id(
            graph,
            names,
            levels,
            &query,
            options,
            &crate::py_execution_context_cancel(seed, 1, cancel),
        )
        .map_err(refused)
    })?;
    Ok(NativePreparedCounterfactualId { prepared, levels: kept_levels })
}

/// Consume a counterfactual identification artifact by re-deriving and
/// recomputing it: `(json, artifact bytes)`.
#[pyfunction]
#[pyo3(signature = (artifact, *, seed=0))]
fn consume_counterfactual_id_artifact_native(
    py: Python<'_>,
    artifact: Vec<u8>,
    seed: u64,
) -> PyResult<(String, Vec<u8>)> {
    detach_catch(py, move || {
        let effect = consume_counterfactual_id_artifact(
            &artifact,
            CounterfactualIdConsumeLimits::default(),
            &py_execution_context(seed, 1),
        )
        .map_err(|e| artifact_refused(&e))?;
        payload(&effect)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePreparedCounterfactualId>()?;
    module.add_function(wrap_pyfunction!(prepare_counterfactual_id_native, module)?)?;
    module.add_function(wrap_pyfunction!(consume_counterfactual_id_artifact_native, module)?)?;
    Ok(())
}
