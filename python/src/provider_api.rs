//! Fixed native host boundary for separately installed Python providers.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io::provider_envelope::{
    ProviderEnvelope, ProviderEnvelopeHeader, open_provider_envelope,
    open_verified_provider_envelope, seal_provider_envelope,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn value_error(message: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(message.to_string())
}

#[pyfunction]
fn seal_provider_result<'py>(
    py: Python<'py>,
    header_json: &str,
    external_artifact: Option<&[u8]>,
) -> PyResult<Bound<'py, PyBytes>> {
    let header: ProviderEnvelopeHeader = serde_json::from_str(header_json).map_err(value_error)?;
    let envelope =
        ProviderEnvelope { header, external_artifact: external_artifact.map(<[u8]>::to_vec) };
    let bytes = seal_provider_envelope(envelope).map_err(value_error)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
#[pyo3(signature = (bytes, *, verified_spec_digest=None, verified_request_digest=None, verified_evidence_digest=None))]
fn open_provider_result<'py>(
    py: Python<'py>,
    bytes: &[u8],
    verified_spec_digest: Option<&str>,
    verified_request_digest: Option<&str>,
    verified_evidence_digest: Option<&str>,
) -> PyResult<(String, Option<Bound<'py, PyBytes>>)> {
    let envelope =
        match (verified_spec_digest, verified_request_digest, verified_evidence_digest) {
            (None, None, None) => open_provider_envelope(bytes),
            (Some(spec), Some(request), Some(evidence)) => {
                open_verified_provider_envelope(bytes, spec, request, evidence)
            }
            _ => {
                return Err(value_error(
                    "verified provider opening requires all three host digests",
                ));
            }
        }
        .map_err(value_error)?;
    let header_json = serde_json::to_string(&envelope.header).map_err(value_error)?;
    Ok((
        header_json,
        envelope.external_artifact.as_deref().map(|artifact| PyBytes::new(py, artifact)),
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(seal_provider_result, m)?)?;
    m.add_function(wrap_pyfunction!(open_provider_result, m)?)?;
    Ok(())
}
