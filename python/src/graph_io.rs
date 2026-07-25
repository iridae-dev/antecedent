//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::*;
use pyo3::prelude::*;

#[pyfunction]
fn dag_from_dot(dot: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = facade_dag_from_dot(dot).map_err(py_err)?;
        let wire = antecedent_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit DOT for a numeric DAG given `node_count` and `edges`.
#[pyfunction]
fn dag_to_dot(node_count: u32, edges: Vec<(u32, u32)>) -> PyResult<String> {
    catch_ffi(|| {
        let wire = antecedent_io::DagWire { node_count, edges };
        let dag = antecedent_io::dag_from_wire(&wire).map_err(py_err)?;
        facade_dag_to_dot(&dag, None).map_err(py_err)
    })
}

/// Parsed JSON DAG: `(node_count, edges, variable_names)`.
type ParsedDagJson = (usize, Vec<(u32, u32)>, Option<Vec<String>>);

/// Parse JSON DAG document; return `(node_count, edges, variable_names|None)`.
#[pyfunction]
fn dag_from_json(json: &str) -> PyResult<ParsedDagJson> {
    catch_ffi(|| {
        let doc = antecedent_io::dag_json_from_str(json).map_err(py_err)?;
        let dag = antecedent_io::dag_from_wire(&doc.to_wire()).map_err(py_err)?;
        let _ = dag;
        Ok((doc.node_count as usize, doc.edges, doc.variable_names))
    })
}

/// Emit JSON for a numeric DAG.
#[pyfunction]
fn dag_to_json(
    node_count: u32,
    edges: Vec<(u32, u32)>,
    variable_names: Option<Vec<String>>,
) -> PyResult<String> {
    catch_ffi(|| {
        let wire = antecedent_io::DagWire { node_count, edges };
        let dag = antecedent_io::dag_from_wire(&wire).map_err(py_err)?;
        facade_dag_to_json(&dag, variable_names.as_deref()).map_err(py_err)
    })
}

/// Parse GML digraph text; return `(node_count, edges)`.
#[pyfunction]
fn dag_from_gml(gml: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = antecedent::io::dag_from_gml(gml).map_err(py_err)?;
        let wire = antecedent_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit GML for a numeric DAG.
#[pyfunction]
fn dag_to_gml(node_count: u32, edges: Vec<(u32, u32)>) -> PyResult<String> {
    catch_ffi(|| {
        let wire = antecedent_io::DagWire { node_count, edges };
        let dag = antecedent_io::dag_from_wire(&wire).map_err(py_err)?;
        antecedent::io::dag_to_gml(&dag, None).map_err(py_err)
    })
}

/// Parse NetworkX node-link JSON; return `(node_count, edges)`.
#[pyfunction]
fn dag_from_networkx_node_link(json: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = antecedent::io::dag_from_networkx_node_link(json).map_err(py_err)?;
        let wire = antecedent_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit NetworkX node-link JSON for a numeric DAG.
#[pyfunction]
fn dag_to_networkx_node_link(node_count: u32, edges: Vec<(u32, u32)>) -> PyResult<String> {
    catch_ffi(|| {
        let wire = antecedent_io::DagWire { node_count, edges };
        let dag = antecedent_io::dag_from_wire(&wire).map_err(py_err)?;
        antecedent::io::dag_to_networkx_node_link(&dag, None).map_err(py_err)
    })
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
        use antecedent_io::{
            ModelBundleEncode, ModelBundleHeaderWire, ModelKindWire, encode_model_bundle as enc,
        };
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
        let wire = antecedent_io::DagWire {
            node_count: u32::try_from(variable_names.len()).unwrap_or(0),
            edges,
        };
        let dag = antecedent_io::dag_from_wire(&wire).map_err(py_err)?;
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
        let art = enc(&ModelBundleEncode {
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
        .map_err(py_err)?;
        let mut buf = Vec::new();
        art.write_to(&mut buf).map_err(py_err)?;
        Ok(buf)
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

#[pyfunction]
fn dag_from_networkx_adjacency(json: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = facade_dag_from_networkx_adjacency(json).map_err(py_err)?;
        let wire = antecedent_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit NetworkX adjacency JSON for a numeric DAG.
#[pyfunction]
fn dag_to_networkx_adjacency(
    node_count: u32,
    edges: Vec<(u32, u32)>,
    variable_names: Option<Vec<String>>,
) -> PyResult<String> {
    catch_ffi(|| {
        let wire = antecedent_io::DagWire { node_count, edges };
        let dag = antecedent_io::dag_from_wire(&wire).map_err(py_err)?;
        facade_dag_to_networkx_adjacency(&dag, variable_names.as_deref()).map_err(py_err)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dag_from_dot, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_dot, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_gml, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_gml, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_networkx_node_link, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_networkx_node_link, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_networkx_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_networkx_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(encode_model_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(decode_model_bundle, m)?)?;
    Ok(())
}
