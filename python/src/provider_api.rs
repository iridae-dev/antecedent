//! Fixed native host boundary for separately installed Python providers.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ProviderVerificationReceipt;
use antecedent_io::canonical_json::canonical_json_digest;
use antecedent_io::provider_envelope::{
    ProviderEnvelope, ProviderEnvelopeHeader, open_provider_envelope,
    open_verified_provider_envelope, seal_provider_envelope,
};
use antecedent_io::provider_verification_artifact::{
    ExecutionVerificationWire, MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES,
    ProviderVerificationReceiptWire, receipt_from_bytes, receipt_to_bytes,
    verify_execution_requests,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::bridge::{check_declared_len, check_declared_size};

fn value_error(message: impl std::fmt::Display) -> PyErr {
    crate::value_err(message.to_string())
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

/// A passing provider verification receipt, issued only by Rust.
///
/// Python states probe *values*; whether they pass, which requests they cover
/// and what the receipt digests to are decided here. No constructor accepts a
/// pre-made receipt.
#[pyclass(name = "ProviderVerificationReceipt", skip_from_py_object)]
pub(crate) struct PyProviderVerificationReceipt {
    pub(crate) inner: ProviderVerificationReceipt,
}

#[pymethods]
impl PyProviderVerificationReceipt {
    /// Verify executable-provider requests; returns the receipt or a structured refusal as JSON.
    #[staticmethod]
    fn verify(requests_json: &str) -> PyResult<(Option<Self>, Option<String>)> {
        check_declared_size(requests_json, 4 * 1024 * 1024, "provider verification requests")?;
        let wire: ExecutionVerificationWire =
            serde_json::from_str(requests_json).map_err(value_error)?;
        match verify_execution_requests(&wire) {
            Ok(inner) => Ok((Some(Self { inner }), None)),
            Err(refusal) => Ok((None, Some(serde_json::to_string(&refusal).map_err(value_error)?))),
        }
    }

    /// Load an exported receipt under the digest the consumer retained independently.
    #[staticmethod]
    fn load(data: &[u8], expected_digest: &str) -> PyResult<Self> {
        check_declared_len(
            data.len(),
            MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES,
            "provider verification receipt",
        )?;
        let inner = receipt_from_bytes(data, expected_digest).map_err(value_error)?;
        Ok(Self { inner })
    }

    /// The stored wire: origin, verified requests with their probes, and the digest.
    #[getter]
    fn wire_json(&self) -> PyResult<String> {
        serde_json::to_string(&ProviderVerificationReceiptWire::from(&self.inner))
            .map_err(value_error)
    }

    #[getter]
    fn digest(&self) -> String {
        self.inner.digest()
    }

    fn export<'py>(&self, py: Python<'py>, artifact_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = receipt_to_bytes(&self.inner, artifact_id).map_err(value_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Canonical BLAKE3 digest of a JSON document under a domain tag.
#[pyfunction]
fn canonical_json_digest_hex(domain: &str, json: &str) -> PyResult<String> {
    canonical_json_digest(domain, json).map_err(value_error)
}

/// Unframed BLAKE3 digest of raw bytes, as 64 lowercase hex characters.
#[pyfunction]
fn blake3_digest_hex(data: &[u8]) -> String {
    antecedent_core::blake3_hex(data)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProviderVerificationReceipt>()?;
    m.add_function(wrap_pyfunction!(canonical_json_digest_hex, m)?)?;
    m.add_function(wrap_pyfunction!(blake3_digest_hex, m)?)?;
    m.add_function(wrap_pyfunction!(seal_provider_result, m)?)?;
    m.add_function(wrap_pyfunction!(open_provider_result, m)?)?;
    Ok(())
}
