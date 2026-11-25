//! Python bridge for a format 0.6 compiled study.

use antecedent::io::compiled_study::{self, StudyBody, StudyParts};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::json;

fn same_length(label: &str, got: usize, expected: usize) -> PyResult<()> {
    if got == expected {
        Ok(())
    } else {
        Err(crate::value_err(format!("{label} has {got} entries; {expected} roles were given")))
    }
}

/// Write a compiled study. Returns `(bytes, expected_identity)`.
///
/// An empty string in `external_uris` stores that role inline.
#[pyfunction]
#[pyo3(signature = (invariant, roles, bodies, external_uris, required, summary_only, names, claim=None))]
fn write_compiled_study(
    py: Python<'_>,
    invariant: &[u8],
    roles: Vec<String>,
    bodies: Vec<Vec<u8>>,
    external_uris: Vec<String>,
    required: Vec<bool>,
    summary_only: Vec<bool>,
    names: Vec<Vec<String>>,
    claim: Option<&[u8]>,
) -> PyResult<(Py<PyBytes>, String)> {
    let count = roles.len();
    same_length("bodies", bodies.len(), count)?;
    same_length("external_uris", external_uris.len(), count)?;
    same_length("required", required.len(), count)?;
    same_length("summary_only", summary_only.len(), count)?;
    same_length("names", names.len(), count)?;
    let stored: Vec<StudyBody> = roles
        .into_iter()
        .zip(bodies)
        .zip(external_uris)
        .zip(required)
        .zip(summary_only)
        .zip(names)
        .map(|(((((role, bytes), uri), required), summary_only), names)| StudyBody {
            role,
            bytes,
            external_uri: if uri.is_empty() { None } else { Some(uri) },
            required,
            summary_only,
            names,
        })
        .collect();
    let written = compiled_study::write(StudyParts {
        invariant,
        bodies: &stored,
        claim,
    })
    .map_err(|error| crate::value_err_display(error))?;
    Ok((PyBytes::new(py, &written.bytes).unbind(), written.expected_identity))
}

/// Open a compiled study.
///
/// `resolved_uris` and `resolved_bodies` are the resolver's answer for external
/// placements. A missing URI leaves that required body unresolved. Returns
/// `(status_json, invariant, roles, bodies)`.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, resolved_uris=None, resolved_bodies=None, caller_supplied=false))]
fn open_compiled_study(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    resolved_uris: Option<Vec<String>>,
    resolved_bodies: Option<Vec<Vec<u8>>>,
    caller_supplied: bool,
) -> PyResult<(String, Py<PyBytes>, Vec<String>, Vec<Py<PyBytes>>)> {
    let resolved = match (resolved_uris, resolved_bodies) {
        (None, None) => Vec::new(),
        (Some(uris), Some(bodies)) if uris.len() == bodies.len() => {
            uris.into_iter().zip(bodies).collect::<Vec<_>>()
        }
        _ => {
            return Err(crate::value_err(
                "resolved_uris and resolved_bodies must be the same length",
            ));
        }
    };
    let opened = compiled_study::open(
        data,
        expected_identity,
        &|uri| {
            resolved
                .iter()
                .find(|(candidate, _)| candidate == uri)
                .map(|(_, body)| body.clone())
        },
        caller_supplied,
    )
    .map_err(|error| crate::value_err_display(error))?;
    let status = json!({
        "resumable": opened.resumable,
        "unresolved": opened.unresolved.iter().map(|item| json!({
            "role": item.role,
            "uri": item.uri,
        })).collect::<Vec<_>>(),
        "layer5_bytes": opened.layer5_bytes,
        "estimate": opened.estimate,
        "claim_intact": opened.claim.as_ref().map(|claim| claim.is_intact()),
        "claim_estimate": opened.claim.as_ref().and_then(|claim| claim.body.estimate),
        "refit_refused": opened.refit_refused,
        "callback_requires_caller": opened.callback_requires_caller,
        "callback_live": opened.callback_live,
    });
    let mut roles = Vec::new();
    let mut bodies = Vec::new();
    for (role, bytes) in &opened.bodies {
        roles.push(role.clone());
        bodies.push(PyBytes::new(py, bytes).unbind());
    }
    let invariant = PyBytes::new(py, &opened.invariant).unbind();
    Ok((status.to_string(), invariant, roles, bodies))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_compiled_study, m)?)?;
    m.add_function(wrap_pyfunction!(open_compiled_study, m)?)
}
