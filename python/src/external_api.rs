//! Bounded Python bridge for binding typed external responses.
//!
//! Python builds the declarations; every identity, check and refusal rule is
//! Rust's. A refusal comes back as structured JSON for the Python layer to
//! raise as its own exception type.

use crate::bridge::{check_declared_len, check_declared_size};
use antecedent_io::external_binding_wire::{
    ColumnResponseWire, ContractWire, MAX_BINDING_WIRE_BYTES, RefusalWire, ResponseWire,
    bind_column, bind_response,
};
use antecedent_io::external_claim_artifact::{
    ExternalClaimArtifact, ExternalClaimIdentity, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES,
};
use antecedent_io::external_column_claim_artifact::{
    ExternalColumnClaimArtifact, ExternalColumnIdentity, MAX_EXTERNAL_COLUMN_CLAIM_ARTIFACT_BYTES,
    MAX_EXTERNAL_COLUMN_ROWS,
};
use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn artifact_error(error: impl std::fmt::Display) -> PyErr {
    crate::serialization_err(error.to_string())
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
        check_declared_len(
            data.len(),
            MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES,
            "external claim artifact",
        )?;
        check_declared_size(
            expected_identity_json,
            MAX_BINDING_WIRE_BYTES,
            "external claim identity",
        )?;
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

/// A bound row-level column: identity and digest, never the rows.
#[pyclass(name = "ExternalColumnClaimArtifact", skip_from_py_object)]
pub(crate) struct PyExternalColumnClaimArtifact {
    artifact: ExternalColumnClaimArtifact,
}

#[pymethods]
impl PyExternalColumnClaimArtifact {
    #[staticmethod]
    fn load(data: &[u8], expected_identity_json: &str) -> PyResult<Self> {
        check_declared_len(
            data.len(),
            MAX_EXTERNAL_COLUMN_CLAIM_ARTIFACT_BYTES,
            "external column claim artifact",
        )?;
        check_declared_size(
            expected_identity_json,
            MAX_BINDING_WIRE_BYTES,
            "external column claim identity",
        )?;
        let expected: ExternalColumnIdentity =
            serde_json::from_str(expected_identity_json).map_err(artifact_error)?;
        let artifact =
            ExternalColumnClaimArtifact::from_bytes(data, &expected).map_err(artifact_error)?;
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

    /// Whether `values` is exactly the column this claim was bound to.
    fn covers(&self, values: PyReadonlyArray1<'_, f64>) -> bool {
        let view = values.as_array();
        match view.as_slice() {
            Some(slice) => self.artifact.covers(slice),
            None => self.artifact.covers(&view.to_vec()),
        }
    }

    fn export<'py>(&self, py: Python<'py>, artifact_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.artifact.to_bytes(artifact_id).map_err(artifact_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Bind a row-level column; returns the claim or a structured refusal as JSON.
#[pyfunction]
fn bind_external_column(
    contract_json: &str,
    column_json: &str,
    values: PyReadonlyArray1<'_, f64>,
    causal_contract_id: &str,
) -> PyResult<(Option<PyExternalColumnClaimArtifact>, Option<String>)> {
    check_declared_size(contract_json, MAX_BINDING_WIRE_BYTES, "external binding contract")?;
    check_declared_size(column_json, MAX_BINDING_WIRE_BYTES, "external binding column")?;
    check_declared_len(
        values.len()?,
        usize::try_from(MAX_EXTERNAL_COLUMN_ROWS).unwrap_or(usize::MAX),
        "external column rows",
    )?;
    let contract: ContractWire = serde_json::from_str(contract_json).map_err(artifact_error)?;
    let column: ColumnResponseWire = serde_json::from_str(column_json).map_err(artifact_error)?;
    match bind_column(&contract, &column, values.as_array().to_vec(), causal_contract_id) {
        Ok(artifact) => Ok((Some(PyExternalColumnClaimArtifact { artifact }), None)),
        Err(refusal) => {
            let refusal: RefusalWire = refusal;
            Ok((None, Some(serde_json::to_string(&refusal).map_err(artifact_error)?)))
        }
    }
}

/// Bind a response; returns the claim or a structured refusal as JSON.
#[pyfunction]
fn bind_external_response(
    contract_json: &str,
    response_json: &str,
    causal_contract_id: &str,
) -> PyResult<(Option<PyExternalClaimArtifact>, Option<String>)> {
    check_declared_size(contract_json, MAX_BINDING_WIRE_BYTES, "external binding contract")?;
    check_declared_size(response_json, MAX_BINDING_WIRE_BYTES, "external binding response")?;
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
    m.add_class::<PyExternalColumnClaimArtifact>()?;
    m.add_function(wrap_pyfunction!(bind_external_column, m)?)?;
    m.add_function(wrap_pyfunction!(bind_external_response, m)?)
}
