//! Joint Gaussian-linear worlds lowered from an identified, prepared static response.
//!
//! Python declares worlds (assigned levels per intervention coordinate) and an optional
//! assembled contrast; compilation, the single shared fit, the joint covariance or aligned
//! posterior, the retained identification and the replayable artifact are Rust's.
// SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::BTreeMap;

use crate::prepared_api::PyPreparedAnalysis;
use antecedent::{JointResponseResult, PreparedJointResponse};
use antecedent_core::{
    AssumptionSet, CausalSchema, Conditioning, ExecutionContext, GraphFamily, InferenceSelection,
    Intervention, InterventionContrast, InterventionWorld, ModelAssumptions, ObservationUnit,
    OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId, TargetForm,
    TargetPopulation, TimeFrame, Value, WorldAction, WorldBinding, WorldCollection,
};
use pyo3::{prelude::*, types::PyBytes};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorldWire {
    id: u32,
    reference: bool,
    levels: BTreeMap<String, f64>,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ContrastWire {
    Linear { terms: Vec<(u32, f64)>, reference: u32 },
    Difference { plus: u32, minus: u32, reference: u32 },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    outcome: String,
    units: String,
    worlds: Vec<WorldWire>,
    contrast: Option<ContrastWire>,
}

fn context(
    seed: u64,
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ExecutionContext> {
    let threads = threads.unwrap_or(1);
    if threads == 0 {
        return Err(crate::value_err("threads must be positive"));
    }
    let mut ctx = crate::py_execution_context_cancel(seed, threads, cancel);
    ctx.set_memory_limit(max_bytes);
    Ok(ctx)
}
fn invalid(error: impl std::fmt::Display) -> PyErr {
    crate::value_err(error.to_string())
}
fn target(schema: &CausalSchema, text: &str) -> PyResult<ScientificTarget> {
    crate::bridge::check_declared_size(
        text,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "joint response declaration",
    )?;
    let d: Declaration = serde_json::from_str(text).map_err(invalid)?;
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("row").map_err(invalid)?,
        OutcomeQuantity::mean(
            schema.id_of(&d.outcome).map_err(invalid)?,
            QuantityUnits::new(d.units).map_err(invalid)?,
        )
        .map_err(invalid)?,
        Conditioning::unconditional(),
    )
    .map_err(invalid)?;
    let worlds = d
        .worlds
        .into_iter()
        .map(|world| {
            let interventions = world
                .levels
                .iter()
                .map(|(name, level)| {
                    Ok(Intervention::set(schema.id_of(name).map_err(invalid)?, Value::f64(*level)))
                })
                .collect::<PyResult<Vec<_>>>()?;
            InterventionWorld::new(
                ScientificWorldId::from_raw(world.id),
                WorldAction::at_decision(interventions).map_err(invalid)?,
                world.reference,
                binding.clone(),
                TimeFrame::static_decision(),
            )
            .map_err(invalid)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let collection = WorldCollection::new(worlds).map_err(invalid)?;
    let id = ScientificWorldId::from_raw;
    let form = match d.contrast {
        None => TargetForm::Worlds(collection),
        Some(ContrastWire::Linear { terms, reference }) => TargetForm::Contrast(
            InterventionContrast::linear(
                collection,
                terms.into_iter().map(|(world, coefficient)| (id(world), coefficient)),
                id(reference),
            )
            .map_err(invalid)?,
        ),
        Some(ContrastWire::Difference { plus, minus, reference }) => TargetForm::Contrast(
            InterventionContrast::difference(collection, id(plus), id(minus), id(reference))
                .map_err(invalid)?,
        ),
    };
    Ok(ScientificTarget::new(
        form,
        ModelAssumptions::new(GraphFamily::Unspecified, AssumptionSet::new()),
        InferenceSelection::Unspecified,
    ))
}

/// An immutable compiled joint response: one shared design and fit, many world projections.
#[pyclass(name = "_PreparedJointResponse", frozen)]
pub(crate) struct PyPreparedJointResponse {
    inner: PreparedJointResponse,
    schema: CausalSchema,
}
/// One joint numerical result with the exact compiled program that produced it.
#[pyclass(name = "_JointResponseResult", frozen)]
pub(crate) struct PyJointResponseResult {
    inner: JointResponseResult,
}

#[pymethods]
impl PyPreparedJointResponse {
    fn inspect(&self) -> PyResult<String> {
        let program = &self.inner;
        Ok(serde_json::json!({
            "world_ids": program.world_ids().iter().map(|w| w.raw()).collect::<Vec<_>>(),
            "contrast_coefficients": program.contrast_coefficients(),
            "source_digest": antecedent_core::to_hex(&program.source_digest()),
            "target": program.target().identity().to_hex(),
            "identified_claim": program.identification().is_identified_claim(),
            "identification": program.identification().identity().to_hex(),
            "fitted": program.numerical_program().has_fit(),
            "calibration": "unmeasured",
        })
        .to_string())
    }
    #[pyo3(signature=(*,seed=0,threads=None,max_bytes=None,cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyJointResponseResult> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || self.inner.execute(&ctx).map_err(crate::py_err))?;
        Ok(PyJointResponseResult { inner })
    }
    #[pyo3(signature=(declaration_json,*,seed=0,threads=None,max_bytes=None,cancel=None))]
    fn retarget(
        &self,
        py: Python<'_>,
        declaration_json: &str,
        seed: u64,
        threads: Option<u32>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Self> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let wanted = target(&self.schema, declaration_json)?;
        let inner =
            crate::detach_catch(py, || self.inner.retarget(&wanted, &ctx).map_err(crate::py_err))?;
        Ok(Self { inner, schema: self.schema.clone() })
    }
    #[pyo3(signature=(artifact,*,seed=0,threads=None,max_bytes=None,cancel=None))]
    fn consume_current(
        &self,
        py: Python<'_>,
        artifact: &[u8],
        seed: u64,
        threads: Option<u32>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyJointResponseResult> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || {
            self.inner
                .consume_current(artifact, &ctx)
                .map_err(|error| crate::value_err(error.to_string()))
        })?;
        Ok(PyJointResponseResult { inner })
    }
}
#[pymethods]
impl PyJointResponseResult {
    fn report(&self) -> PyResult<String> {
        let evaluation = self.inner.evaluation();
        let dim = evaluation.covariance.dim;
        let covariance = (0..dim)
            .map(|row| (0..dim).map(|col| evaluation.covariance.get(row, col)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let failed = evaluation.failed_draws.iter().filter(|failed| **failed).count();
        Ok(serde_json::json!({
            "world_ids": self.inner.world_ids().iter().map(|w| w.raw()).collect::<Vec<_>>(),
            "means": evaluation.means.to_vec(),
            "covariance": covariance,
            "contrast": evaluation.contrast.as_ref().map(|c| serde_json::json!({"value": c.value, "se": c.se})),
            "uncertainty": if evaluation.posterior_draws.is_some() { "aligned_posterior" } else { "sampling_covariance" },
            "draw_count": evaluation.draw_count,
            "failed_draws": failed,
            "identified_claim": self.inner.identification().is_identified_claim(),
            "identification": self.inner.identification().identity().to_hex(),
            "target": self.inner.target().identity().to_hex(),
            "source_digest": antecedent_core::to_hex(&self.inner.source_digest()),
            "calibration": self.inner.calibration(),
        })
        .to_string())
    }
    #[pyo3(signature=(*,seed=0,threads=None,max_bytes=None,cancel=None))]
    fn export(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Py<PyBytes>> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let bytes = crate::detach_catch(py, || {
            self.inner.export(&ctx).map_err(|error| crate::serialization_err(error.to_string()))
        })?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

/// Lower worlds (and an optional contrast) from an identified prepared static response.
#[pyfunction]
#[pyo3(signature=(prepared,declaration_json,*,seed=0,threads=None,max_bytes=None,cancel=None))]
fn compile_joint_response(
    py: Python<'_>,
    prepared: PyRef<'_, PyPreparedAnalysis>,
    declaration_json: &str,
    seed: u64,
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedJointResponse> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let schema = prepared.inner.schema().clone();
    let wanted = target(&schema, declaration_json)?;
    let study = prepared.inner.clone();
    let inner = crate::detach_catch(py, || {
        study.compile_joint_response(&wanted, &ctx).map_err(crate::py_err)
    })?;
    Ok(PyPreparedJointResponse { inner, schema })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPreparedJointResponse>()?;
    module.add_class::<PyJointResponseResult>()?;
    module.add_function(wrap_pyfunction!(compile_joint_response, module)?)
}
