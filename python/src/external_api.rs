//! Bounded Python bridge for binding typed external responses.
//!
//! Python builds the declarations; every identity, check and refusal rule is
//! Rust's. A refusal comes back as structured JSON for the Python layer to
//! raise as its own exception type.

use antecedent_io::external_binding_wire::{
    ContractWire, MAX_BINDING_WIRE_BYTES, RefusalWire, ResponseWire, bind_response,
};
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES,
};
use numpy::PyArray1;
use numpy::ndarray::Array1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::CausalSerializationError;

fn artifact_error(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

#[pyclass(name = "ExternalClaimArtifact", skip_from_py_object)]
pub(crate) struct PyExternalClaimArtifact {
    // Shared with `program_claims_api`, which binds the same artifact under a program.
    pub(crate) artifact: ExternalClaimArtifact,
}

#[pymethods]
impl PyExternalClaimArtifact {
    #[getter]
    fn source_evidence(&self) -> PyResult<crate::source_evidence_api::PySourceEvidence> {
        crate::source_evidence_api::PySourceEvidence::from_external(self)
    }
    #[staticmethod]
    fn load(data: &[u8], expected_identity_json: &str) -> PyResult<Self> {
        if data.len() > MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES
            || expected_identity_json.len() > MAX_BINDING_WIRE_BYTES
        {
            return Err(PyValueError::new_err("external claim artifact or identity is too large"));
        }
        let expected: ExternalClaimIdentity =
            serde_json::from_str(expected_identity_json).map_err(artifact_error)?;
        let artifact =
            ExternalClaimArtifact::from_bytes(data, &expected).map_err(artifact_error)?;
        Ok(Self { artifact })
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(self.artifact.metadata()).map_err(artifact_error)
    }

    #[getter]
    fn provenance_label(&self) -> String {
        self.artifact.provenance_label()
    }

    /// One bounded copy into NumPy-owned storage.
    fn values_copy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_owned_array(py, Array1::from(self.artifact.values().to_vec()))
    }

    fn export<'py>(&self, py: Python<'py>, artifact_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.artifact.to_bytes(artifact_id).map_err(artifact_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Bind a response; returns the claim or a structured refusal as JSON.
#[pyfunction]
fn bind_external_response(
    contract_json: &str,
    response_json: &str,
    causal_contract_id: &str,
) -> PyResult<(Option<PyExternalClaimArtifact>, Option<String>)> {
    if contract_json.len() > MAX_BINDING_WIRE_BYTES || response_json.len() > MAX_BINDING_WIRE_BYTES
    {
        return Err(PyValueError::new_err("external binding declaration is too large"));
    }
    let contract: ContractWire = serde_json::from_str(contract_json).map_err(artifact_error)?;
    let response: ResponseWire = serde_json::from_str(response_json).map_err(artifact_error)?;
    match bind_response(&contract, &response, causal_contract_id) {
        Ok(artifact) => Ok((Some(PyExternalClaimArtifact { artifact }), None)),
        Err(refusal) => {
            let refusal: RefusalWire = refusal;
            Ok((None, Some(serde_json::to_string(&refusal).map_err(artifact_error)?)))
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExternalClaimArtifact>()?;
    m.add_function(wrap_pyfunction!(bind_external_response, m)?)
}
