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
        let node_count = u32::try_from(variable_names.len())
            .map_err(|_| crate::value_err("too many variables for a model bundle"))?;
        if mechanisms.len() != variable_names.len() {
            return Err(crate::value_err(format!(
                "{} mechanisms for {} variables; give one slot per variable",
                mechanisms.len(),
                variable_names.len()
            )));
        }
        let mut parent_counts = vec![0_usize; variable_names.len()];
        for (_, child) in &edges {
            if let Some(count) = parent_counts.get_mut(*child as usize) {
                *count += 1;
            }
        }
        let dag = dag_from_node_edges(node_count, edges)?;
        let finite = |value: f64, what: &str, node: usize| {
            if value.is_finite() {
                Ok(value)
            } else {
                Err(crate::value_err(format!("mechanism {node}: {what} must be finite")))
            }
        };
        let slots = mechanisms
            .into_iter()
            .enumerate()
            .map(|(node, (kind, constant, coeffs, sigma))| match kind.as_str() {
                "vacant" => Ok(MechanismSlot::Vacant),
                "constant" => {
                    let value = constant.ok_or_else(|| {
                        crate::value_err(format!("mechanism {node}: a constant needs its value"))
                    })?;
                    Ok(MechanismSlot::Constant { value: finite(value, "the value", node)? })
                }
                "linear_gaussian" => {
                    let intercept = constant.ok_or_else(|| {
                        crate::value_err(format!("mechanism {node}: an intercept is required"))
                    })?;
                    let coeffs = coeffs.unwrap_or_default();
                    if coeffs.len() != parent_counts[node] {
                        return Err(crate::value_err(format!(
                            "mechanism {node}: {} coefficients for {} parents",
                            coeffs.len(),
                            parent_counts[node]
                        )));
                    }
                    for coefficient in &coeffs {
                        finite(*coefficient, "every coefficient", node)?;
                    }
                    let sigma = sigma.ok_or_else(|| {
                        crate::value_err(format!("mechanism {node}: sigma is required"))
                    })?;
                    if !(sigma.is_finite() && sigma > 0.0) {
                        return Err(crate::value_err(format!(
                            "mechanism {node}: sigma must be finite and positive"
                        )));
                    }
                    Ok(MechanismSlot::LinearGaussian {
                        intercept: finite(intercept, "the intercept", node)?,
                        coeffs: Arc::from(coeffs),
                        sigma,
                    })
                }
                other => Err(crate::value_err(format!(
                    "mechanism {node}: unknown kind {other:?}; use vacant|constant|linear_gaussian"
                ))),
            })
            .collect::<PyResult<Vec<MechanismSlot>>>()?;
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
