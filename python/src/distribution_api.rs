//! Bounded Python bridge for the 2.3 aligned distribution artifact.

use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionIdentity, DistributionMetadata, DistributionTrust,
    MAX_DISTRIBUTION_ARTIFACT_BYTES, MAX_DISTRIBUTION_COORDINATES, MAX_DISTRIBUTION_DRAWS,
};
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::CausalSerializationError;

fn artifact_error(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

#[pyclass(name = "JointDistributionArtifact", skip_from_py_object)]
pub(crate) struct PyJointDistributionArtifact {
    pub(crate) artifact: DistributionArtifact,
    pub(crate) source_claim: Option<crate::program_claims_api::PyNativeResponseClaim>,
}

impl PyJointDistributionArtifact {
    /// The validated artifact, for bridges that consume its aligned draws.
    pub(crate) fn inner(&self) -> &DistributionArtifact {
        &self.artifact
    }
}

#[pymethods]
impl PyJointDistributionArtifact {
    #[new]
    fn new(metadata_json: &str, draws: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        if metadata_json.len() > MAX_DISTRIBUTION_ARTIFACT_BYTES {
            return Err(PyValueError::new_err("distribution metadata is too large"));
        }
        let metadata: DistributionMetadata =
            serde_json::from_str(metadata_json).map_err(artifact_error)?;
        if metadata.trust == DistributionTrust::NativeLicensed {
            return Err(crate::recalc_api::invalid(
                "native_distribution.authority_required",
                "native trust requires retained producer-issued execution state",
            ));
        }
        let shape = draws.shape();
        if shape[0] == 0
            || shape[0] > MAX_DISTRIBUTION_DRAWS
            || shape[1] == 0
            || shape[1] > MAX_DISTRIBUTION_COORDINATES
            || shape[0]
                .checked_mul(shape[1])
                .and_then(|n| n.checked_mul(8))
                .is_none_or(|bytes| bytes > MAX_DISTRIBUTION_ARTIFACT_BYTES)
        {
            return Err(PyValueError::new_err("distribution draw array exceeds its bounds"));
        }
        let values = draws.as_array().iter().copied().collect();
        let artifact = DistributionArtifact::new(metadata, values).map_err(artifact_error)?;
        Ok(Self { artifact, source_claim: None })
    }

    #[staticmethod]
    fn load(data: &[u8], expected_identity_json: &str) -> PyResult<Self> {
        if data.len() > MAX_DISTRIBUTION_ARTIFACT_BYTES
            || expected_identity_json.len() > MAX_DISTRIBUTION_ARTIFACT_BYTES
        {
            return Err(PyValueError::new_err("distribution artifact or identity is too large"));
        }
        let expected: DistributionIdentity =
            serde_json::from_str(expected_identity_json).map_err(artifact_error)?;
        let artifact = DistributionArtifact::from_bytes(data, &expected).map_err(artifact_error)?;
        Ok(Self { artifact, source_claim: None })
    }

    #[getter]
    fn semantic(&self) -> PyResult<String> {
        let value = serde_json::to_value(self.artifact.semantic()).map_err(artifact_error)?;
        Ok(value.as_str().expect("distribution meaning serializes as a string").into())
    }

    #[getter]
    fn source_evidence(&self) -> PyResult<Option<crate::source_evidence_api::PySourceEvidence>> {
        self.source_claim
            .as_ref()
            .map(crate::source_evidence_api::PySourceEvidence::from_claim)
            .transpose()
    }

    #[getter]
    fn axes(&self) -> Vec<String> {
        self.artifact.axes().iter().map(|axis| (*axis).to_owned()).collect()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        let shape = self.artifact.shape();
        (shape[0], shape[1])
    }

    #[getter]
    fn n_draws(&self) -> usize {
        self.artifact.n_draws()
    }

    #[getter]
    fn metadata_json(&self) -> PyResult<String> {
        serde_json::to_string(self.artifact.metadata()).map_err(artifact_error)
    }

    /// One bounded copy into NumPy-owned, row-major f64 storage.
    fn draws_copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let [rows, columns] = self.artifact.shape();
        let array = Array2::from_shape_vec((rows, columns), self.artifact.draws().to_vec())
            .map_err(artifact_error)?;
        Ok(PyArray2::from_owned_array(py, array))
    }

    fn mean(&self, coordinate: usize) -> PyResult<f64> {
        self.artifact.mean(coordinate).map_err(artifact_error)
    }

    fn covariance(&self, left: usize, right: usize) -> PyResult<f64> {
        self.artifact.covariance(left, right).map_err(artifact_error)
    }

    fn joint_product_expectation(&self, left: usize, right: usize) -> PyResult<f64> {
        self.artifact.joint_expectation(left, right, |x, y| x * y).map_err(artifact_error)
    }

    fn export<'py>(&self, py: Python<'py>, artifact_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.artifact.to_bytes(artifact_id).map_err(artifact_error)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJointDistributionArtifact>()
}
