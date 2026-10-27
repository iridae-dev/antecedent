//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::*;
use antecedent_graph::Dag;
use pyo3::prelude::*;

fn dag_from_node_edges(node_count: u32, edges: Vec<(u32, u32)>) -> PyResult<Dag> {
    let wire = antecedent_io::DagWire { node_count, edges };
    antecedent_io::dag_from_wire(&wire).map_err(py_err)
}

/// Encode a minimal SCM model bundle (schema names + edges + mechanism slots).
///
/// `mechanisms` entries are `(kind, constant|intercept, coeffs|None, sigma|None)`
/// with `kind` in `{vacant, constant, linear_gaussian}`.
#[pyfunction]
fn encode_model_bundle(
    variable_names: Vec<String>,
    edges: Vec<(u32, u32)>,
    mechanisms: Vec<MechanismWireEntry>,
) -> PyResult<Vec<u8>> {
    catch_ffi(|| {
        use antecedent::gcm::{CompiledMechanismStore, MechanismSlot};
        use antecedent_core::{CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType};
        use antecedent_io::{ModelBundleEncode, ModelBundleHeaderWire, ModelKindWire};
        use std::sync::Arc;

        let mut b = CausalSchemaBuilder::new();
        for name in &variable_names {
            b.add_variable(
                name.as_str(),
                ValueType::Continuous,
                SmallRoleSet::empty(),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(|e| py_err(IoError::Convert(e.to_string())))?;
        }
        let schema = b.build().map_err(|e| py_err(IoError::Convert(e.to_string())))?;
        let dag = dag_from_node_edges(u32::try_from(variable_names.len()).unwrap_or(0), edges)?;
        let slots: Vec<MechanismSlot> = mechanisms
            .into_iter()
            .map(|(kind, constant, coeffs, sigma)| match kind.as_str() {
                "constant" => MechanismSlot::Constant { value: constant.unwrap_or(0.0) },
                "linear_gaussian" => MechanismSlot::LinearGaussian {
                    intercept: constant.unwrap_or(0.0),
                    coeffs: Arc::from(coeffs.unwrap_or_default()),
                    sigma: sigma.unwrap_or(1.0),
                },
                _ => MechanismSlot::Vacant,
            })
            .collect();
        let store = CompiledMechanismStore { slots: slots.into() };
        antecedent::io::encode_model_bundle_bytes(&ModelBundleEncode {
            header: ModelBundleHeaderWire { model_kind: ModelKindWire::Scm, label: None },
            schema: &schema,
            dag: &dag,
            mechanisms: &store,
            artifact_id: "py-model-bundle",
            contrast: None,
            query: None,
            analysis_trace: None,
            identification: None,
            estimate: None,
            refutations: None,
            logical_plan: None,
            physical_plan: None,
            performance: None,
            diagnostics: None,
            provenance: None,
            posterior: None,
            discovery: None,
        })
        .map_err(py_err)
    })
}

/// Decode a model bundle; return `(variable_names, edges, n_mechanisms)`.
#[pyfunction]
fn decode_model_bundle(bytes: &[u8]) -> PyResult<ModelBundleSummary> {
    catch_ffi(|| {
        let bundle = antecedent::io::decode_model_bundle_bytes(bytes).map_err(py_err)?;
        let names = bundle.schema.variables().iter().map(|v| v.name.to_string()).collect();
        let wire = antecedent_io::dag_to_wire(&bundle.dag).map_err(py_err)?;
        Ok((names, wire.edges, bundle.mechanisms.slots.len()))
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode_model_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(decode_model_bundle, m)?)?;
    Ok(())
}
