//! Python bridge for result-level method provenance.
//!
//! The lookup, the record table and the bibliography de-duplication are Rust's
//! (`antecedent::io::method_provenance`); Python receives JSON and builds its typed views.

use antecedent::io::method_provenance::{
    MethodProvenanceWire, bibliography, capability_provenance, method_provenance,
};
use pyo3::prelude::*;

fn to_json<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| crate::value_err(error.to_string()))
}

/// Provenance records behind the route with these identifier and estimator ids, as JSON.
#[pyfunction]
#[pyo3(signature = (identifier=None, estimator=None))]
fn method_provenance_json(identifier: Option<&str>, estimator: Option<&str>) -> PyResult<String> {
    to_json(&method_provenance(identifier, estimator))
}

/// Provenance records behind a capability-specific route, as JSON.
#[pyfunction]
fn capability_provenance_json(capability: &str) -> PyResult<String> {
    to_json(&capability_provenance(capability))
}

/// De-duplicated citations across provenance JSON documents, as JSON.
#[pyfunction]
fn bibliography_json(provenances: Vec<String>) -> PyResult<String> {
    let parsed = provenances
        .iter()
        .map(|text| serde_json::from_str::<MethodProvenanceWire>(text))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| crate::value_err(format!("malformed method provenance: {error}")))?;
    let citations = bibliography(&parsed);
    let rendered: Vec<serde_json::Value> = citations
        .iter()
        .map(|citation| {
            let mut value = serde_json::to_value(citation).unwrap_or(serde_json::Value::Null);
            if let serde_json::Value::Object(map) = &mut value {
                map.insert("reference".into(), citation.reference().into());
            }
            value
        })
        .collect();
    to_json(&rendered)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(method_provenance_json, m)?)?;
    m.add_function(wrap_pyfunction!(capability_provenance_json, m)?)?;
    m.add_function(wrap_pyfunction!(bibliography_json, m)?)
}
