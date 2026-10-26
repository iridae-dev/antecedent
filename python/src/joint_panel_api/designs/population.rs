//! Population reference-world transport and independently replayable artifacts.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointDesignResult, bounded_json, context, panel_error};
use antecedent::analysis::joint_panel::{
    ChangeComponent, PopulationReferenceBindings, PopulationReferenceDecomposition,
    ReferenceComponentBindings,
};
use antecedent_design::scientific_source::ScientificConsumptionScope;
use antecedent_graph::SelectionDiagram;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Diagram {
    node_count: u32,
    directed: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
    selection_targets: Vec<u32>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    baseline: u32,
    strata: Vec<f64>,
    stratum_worlds: Vec<[u32; 16]>,
    before_weights: Vec<f64>,
    after_weights: Vec<f64>,
    exposure: Vec<u32>,
    saturation: Vec<u32>,
    mechanism: Vec<u32>,
    order: [ChangeComponent; 4],
    diagram: Diagram,
    #[serde(default = "identified_default")]
    require_identified: bool,
}
fn identified_default() -> bool {
    true
}
fn parse(
    text: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<(PopulationReferenceBindings, SelectionDiagram, ScientificConsumptionScope)> {
    bounded_json(text, "population reference decomposition")?;
    let declaration: Declaration =
        serde_json::from_str(text).map_err(|error| crate::value_err(error.to_string()))?;
    let worlds = declaration
        .stratum_worlds
        .iter()
        .map(|row| row.map(antecedent_core::ScientificWorldId::from_raw))
        .collect::<Vec<_>>();
    let hybrids = worlds
        .first()
        .copied()
        .ok_or_else(|| crate::value_err("population reference requires a stratum world basis"))?;
    let variables =
        |values: Vec<u32>| values.into_iter().map(antecedent_core::VariableId::from_raw).collect();
    let bindings = PopulationReferenceBindings {
        baseline: antecedent_core::VariableId::from_raw(declaration.baseline),
        strata: declaration.strata,
        stratum_worlds: worlds,
        before_weights: declaration.before_weights,
        after_weights: declaration.after_weights,
        components: ReferenceComponentBindings {
            exposure: variables(declaration.exposure),
            saturation: variables(declaration.saturation),
            mechanism: variables(declaration.mechanism),
            hybrids,
            order: declaration.order,
        },
    };
    let graph_bytes = u64::from(declaration.diagram.node_count)
        .checked_mul(512)
        .and_then(|n| {
            n.checked_add(
                (declaration.diagram.directed.len() + declaration.diagram.bidirected.len()) as u64
                    * 256,
            )
        })
        .ok_or_else(|| crate::value_err("population graph memory quote overflow"))?;
    let _graph_lease = ctx.acquire_admission(0, graph_bytes).map_err(|error| {
        panel_error(match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                antecedent_estimate::joint_panel::PanelError::Cancelled.into()
            }
            error => {
                antecedent_estimate::joint_panel::PanelError::Resource(error.to_string()).into()
            }
        })
    })?;
    let graph = antecedent_io::admg_from_wire(&antecedent_io_core::wire::AdmgWire {
        node_count: declaration.diagram.node_count,
        directed: declaration.diagram.directed,
        bidirected: declaration.diagram.bidirected,
    })
    .map_err(|error| crate::value_err(error.to_string()))?;
    let diagram = SelectionDiagram::try_new(
        graph,
        declaration
            .diagram
            .selection_targets
            .into_iter()
            .map(antecedent_core::VariableId::from_raw)
            .collect::<Vec<_>>(),
    )
    .map_err(|error| crate::value_err(error.to_string()))?;
    let scope = if declaration.require_identified {
        ScientificConsumptionScope::IdentifiedCausal
    } else {
        ScientificConsumptionScope::ModelConditional
    };
    Ok((bindings, diagram, scope))
}
fn report(result: &PopulationReferenceDecomposition) -> PyResult<String> {
    let graph = antecedent_io::admg_to_wire(result.transport.diagram().causal_graph())
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    let uncertainty = match result.source.as_numerical_source() {
        antecedent_design::scientific_source::ScientificSourcePayload::Law(law) => {
            serde_json::json!({"law":law.metadata()})
        }
        antecedent_design::scientific_source::ScientificSourcePayload::Means(means) => {
            serde_json::json!({"estimate":format!("{:?}",means.estimate)})
        }
    };
    serde_json::to_string(&serde_json::json!({"decomposition":{"point":result.decomposition.point,"draws":result.decomposition.draws,"weights":result.decomposition.weights},"covariance":result.covariance,"joint_uncertainty":result.decomposition.joint_uncertainty(),"procedure_failures":result.procedure_failures,"full_procedure_bootstrap":!result.procedure_failures.is_empty(),"transport":{"identity":result.transport.identity().to_hex(),"functional":result.transport.functional(),"graph":graph,"selection_targets":result.transport.diagram().selection_targets().iter().map(|id|id.raw()).collect::<Vec<_>>(),"baseline":result.bindings.baseline.raw(),"strata":result.bindings.strata,"before_weights":result.bindings.before_weights,"after_weights":result.bindings.after_weights},"source_question":result.source.question_identity().to_hex(),"identification":result.source.identification().identity().to_hex(),"dependency":result.source.dependency_identity().to_hex(),"scope":if result.scope==ScientificConsumptionScope::IdentifiedCausal{"identified_population_reference"}else{"model_conditional_population_reference"},"uncertainty":uncertainty,"calibration":"unmeasured","grants_unique_causal_attribution":false})).map_err(|error|crate::serialization_err(error.to_string()))
}
pub(super) fn execute(
    result: &JointDesignResult,
    text: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<String> {
    let (bindings, diagram, scope) = parse(text, ctx)?;
    report(
        &result
            .decompose_population_reference(&bindings, &diagram, scope, ctx)
            .map_err(panel_error)?,
    )
}
pub(super) fn export(
    result: &JointDesignResult,
    text: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<Vec<u8>> {
    let (bindings, diagram, scope) = parse(text, ctx)?;
    result.export_population_reference(&bindings, &diagram, scope, ctx).map_err(panel_error)
}
pub(super) fn consume(
    result: &JointDesignResult,
    artifact: &[u8],
    text: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<String> {
    let (bindings, diagram, scope) = parse(text, ctx)?;
    report(
        &result
            .consume_population_reference_current(artifact, &bindings, &diagram, scope, ctx)
            .map_err(panel_error)?,
    )
}
#[pyfunction]
#[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn replay_population_reference(
    py: Python<'_>,
    artifact: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let result = crate::detach_catch(py, || {
        antecedent::analysis::joint_panel::replay_population_reference(artifact, &ctx)
            .map_err(panel_error)
    })?;
    serde_json::to_string(&serde_json::json!({"decomposition":{"point":result.decomposition.point,"draws":result.decomposition.draws,"weights":result.decomposition.weights},"covariance":result.covariance,"procedure_failures":result.procedure_failures,"transport_identity":result.transport_identity.to_hex(),"current_authorization":false})).map_err(|error|crate::serialization_err(error.to_string()))
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(replay_reference_decomposition, module)?)?;
    module.add_function(wrap_pyfunction!(replay_population_reference, module)?)?;
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Components {
    exposure: Vec<u32>,
    saturation: Vec<u32>,
    mechanism: Vec<u32>,
    hybrids: [u32; 16],
    order: [ChangeComponent; 4],
    #[serde(default = "identified_default")]
    require_identified: bool,
}
fn components(text: &str) -> PyResult<(ReferenceComponentBindings, ScientificConsumptionScope)> {
    bounded_json(text, "scientific component reference artifact")?;
    let declaration: Components =
        serde_json::from_str(text).map_err(|error| crate::value_err(error.to_string()))?;
    let variables =
        |values: Vec<u32>| values.into_iter().map(antecedent_core::VariableId::from_raw).collect();
    Ok((
        ReferenceComponentBindings {
            exposure: variables(declaration.exposure),
            saturation: variables(declaration.saturation),
            mechanism: variables(declaration.mechanism),
            hybrids: declaration.hybrids.map(antecedent_core::ScientificWorldId::from_raw),
            order: declaration.order,
        },
        if declaration.require_identified {
            ScientificConsumptionScope::IdentifiedCausal
        } else {
            ScientificConsumptionScope::ModelConditional
        },
    ))
}
pub(super) fn export_components(
    result: &JointDesignResult,
    text: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<Vec<u8>> {
    let (bindings, scope) = components(text)?;
    result.export_reference_decomposition(&bindings, scope, ctx).map_err(panel_error)
}
pub(super) fn consume_components(
    result: &JointDesignResult,
    artifact: &[u8],
    text: &str,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<String> {
    let (bindings, scope) = components(text)?;
    let result = result
        .consume_reference_decomposition_current(artifact, &bindings, scope, ctx)
        .map_err(panel_error)?;
    serde_json::to_string(&serde_json::json!({"decomposition":{"point":result.decomposition.point,"draws":result.decomposition.draws,"weights":result.decomposition.weights},"covariance":result.covariance,"joint_uncertainty":result.decomposition.joint_uncertainty(),"procedure_failures":result.procedure_failures,"full_procedure_bootstrap":!result.procedure_failures.is_empty(),"question_identity":result.question_identity.to_hex(),"scope":if scope==ScientificConsumptionScope::IdentifiedCausal{"identified"}else{"model_conditional"}})).map_err(|error|crate::serialization_err(error.to_string()))
}
#[pyfunction]
#[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn replay_reference_decomposition(
    py: Python<'_>,
    artifact: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    let result = crate::detach_catch(py, || {
        antecedent::analysis::joint_panel::replay_reference_decomposition(artifact, &ctx)
            .map_err(panel_error)
    })?;
    serde_json::to_string(&serde_json::json!({"decomposition":{"point":result.decomposition.point,"draws":result.decomposition.draws,"weights":result.decomposition.weights},"covariance":result.covariance,"procedure_failures":result.procedure_failures,"question_identity":result.question_identity.to_hex(),"current_authorization":false})).map_err(|error|crate::serialization_err(error.to_string()))
}
