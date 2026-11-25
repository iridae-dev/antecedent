//! Fixed native host boundary for separately installed Python providers.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::contract::ProviderVerificationReceipt;
use antecedent::io::canonical_json::canonical_json_digest;
use antecedent::io::provider_envelope::{
    ProviderEnvelope, ProviderEnvelopeHeader, seal_provider_envelope,
};
use antecedent::io::provider_verification_artifact::{
    ExecutionVerificationWire, MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES,
    ProviderVerificationReceiptWire, receipt_to_bytes, verify_execution_requests,
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
    crate::catch_ffi(|| {
        let header: ProviderEnvelopeHeader =
            serde_json::from_str(header_json).map_err(value_error)?;
        let envelope =
            ProviderEnvelope { header, external_artifact: external_artifact.map(<[u8]>::to_vec) };
        let bytes = seal_provider_envelope(envelope).map_err(value_error)?;
        Ok(PyBytes::new(py, &bytes))
    })
}

#[pyfunction]
#[pyo3(signature = (bytes, *, verified_spec_digest=None, verified_request_digest=None, verified_evidence_digest=None, retained_identity=None))]
fn open_provider_result<'py>(
    py: Python<'py>,
    bytes: &[u8],
    verified_spec_digest: Option<&str>,
    verified_request_digest: Option<&str>,
    verified_evidence_digest: Option<&str>,
    retained_identity: Option<&str>,
) -> PyResult<(String, Option<Bound<'py, PyBytes>>)> {
    let retained = crate::retained_artifact_api::retained(retained_identity)?;
    crate::catch_ffi(|| {
        let envelope = match (
            verified_spec_digest,
            verified_request_digest,
            verified_evidence_digest,
        ) {
            (None, None, None) => match &retained {
                Some(retained) => {
                    antecedent::io::provider_envelope::open_provider_envelope_expecting(
                        bytes, retained,
                    )
                }
                #[allow(
                    deprecated,
                    reason = "the unbound Python loader warns and stays reachable until 4.0"
                )]
                None => antecedent::io::provider_envelope::open_provider_envelope(bytes),
            },
            (Some(spec), Some(request), Some(evidence)) => match &retained {
                Some(retained) => {
                    antecedent::io::provider_envelope::open_verified_provider_envelope_expecting(
                        bytes, spec, request, evidence, retained,
                    )
                }
                #[allow(
                    deprecated,
                    reason = "the unbound Python loader warns and stays reachable until 4.0"
                )]
                None => antecedent::io::provider_envelope::open_verified_provider_envelope(
                    bytes, spec, request, evidence,
                ),
            },
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
    })
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
        crate::catch_ffi(|| {
            check_declared_size(requests_json, 4 * 1024 * 1024, "provider verification requests")?;
            let wire: ExecutionVerificationWire =
                serde_json::from_str(requests_json).map_err(value_error)?;
            match verify_execution_requests(&wire) {
                Ok(inner) => Ok((Some(Self { inner }), None)),
                Err(refusal) => {
                    Ok((None, Some(serde_json::to_string(&refusal).map_err(value_error)?)))
                }
            }
        })
    }

    /// Load an exported receipt under the digest the consumer retained independently.
    #[staticmethod]
    #[pyo3(signature = (data, expected_digest, retained_identity=None))]
    fn load(data: &[u8], expected_digest: &str, retained_identity: Option<&str>) -> PyResult<Self> {
        let retained = crate::retained_artifact_api::retained(retained_identity)?;
        crate::catch_ffi(|| {
            check_declared_len(
                data.len(),
                MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES,
                "provider verification receipt",
            )?;
            // The receipt is already bound to `expected_digest`, a digest the consumer
            // retained; the retained artifact identity binds the bytes as well.
            let inner = match &retained {
                Some(retained) => {
                    antecedent::io::provider_verification_artifact::receipt_from_bytes_expecting(
                        data,
                        expected_digest,
                        retained,
                    )
                }
                #[allow(
                    deprecated,
                    reason = "the unbound Python loader warns and stays reachable until 4.0"
                )]
                None => antecedent::io::provider_verification_artifact::receipt_from_bytes(
                    data,
                    expected_digest,
                ),
            }
            .map_err(value_error)?;
            Ok(Self { inner })
        })
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
    antecedent::identity::blake3_hex(data)
}

/// Hash the canonical row-major order of a native-endian float64 view without repacking it.
#[pyfunction]
fn _blake3_float64_digest_hex(data: numpy::PyReadonlyArrayDyn<'_, f64>) -> PyResult<String> {
    let mut hasher = blake3::Hasher::new();
    let mut scratch = [0_u8; 4096];
    let mut used = 0;
    for &value in &data.as_array() {
        if !value.is_finite() {
            return Err(crate::value_err("Out of range float values are not JSON compliant"));
        }
        let end = used + size_of::<f64>();
        scratch[used..end].copy_from_slice(&value.to_le_bytes());
        used = end;
        if used == scratch.len() {
            hasher.update(&scratch);
            used = 0;
        }
    }
    hasher.update(&scratch[..used]);
    Ok(hasher.finalize().to_hex().to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProviderVerificationReceipt>()?;
    m.add_function(wrap_pyfunction!(canonical_json_digest_hex, m)?)?;
    m.add_function(wrap_pyfunction!(blake3_digest_hex, m)?)?;
    m.add_function(wrap_pyfunction!(_blake3_float64_digest_hex, m)?)?;
    m.add_function(wrap_pyfunction!(seal_provider_result, m)?)?;
    m.add_function(wrap_pyfunction!(open_provider_result, m)?)?;
    Ok(())
}
