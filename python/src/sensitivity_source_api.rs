//! Explicit original joint-sensitivity source materialization and complete replay.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::sensitivity_decision_api::PySensitivityArtifact;
use antecedent_core::reason_code;
use antecedent_io::IoError;
use antecedent_validate::JointDeviationSpec;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
fn serialization(error: impl std::fmt::Display) -> PyErr {
    crate::CausalSerializationError::new_err(error.to_string())
}
#[derive(Clone)]
pub(crate) struct SourceRecipe {
    pub(crate) prepared: antecedent::PreparedZTransport,
    pub(crate) spec: JointDeviationSpec,
    pub(crate) context: antecedent_core::ExecutionContext,
}
impl PySensitivityArtifact {
    pub(crate) fn checked_source(
        &self,
        py: Python<'_>,
    ) -> PyResult<
        Option<std::sync::Arc<antecedent::analysis::sensitivity_source::SourceBackedSensitivity>>,
    > {
        if let Some(source) = self.source.lock().expect("source cache lock").as_ref() {
            return Ok(Some(source.clone()));
        }
        let Some(recipe) = self.recipe.clone() else {
            return Ok(None);
        };
        let artifact = self.artifact.clone();
        let source = crate::detach_catch(py, move || {
            let baseline = recipe
                .prepared
                .estimate(&recipe.context)
                .map_err(source_error)?
                .export(&recipe.prepared)
                .map_err(source_error)?;
            let original = antecedent::analysis::JointSensitivityArtifactWire::checked(
                baseline,
                &recipe.spec,
                &recipe.context,
            )
            .map_err(source_error)?
            .export()
            .map_err(source_error)?;
            antecedent::analysis::sensitivity_source::SourceBackedSensitivity::checked(
                &original,
                &artifact,
                &recipe.context,
            )
            .map_err(source_error)
        })?;
        let source = std::sync::Arc::new(source);
        *self.source.lock().expect("source cache lock") = Some(source.clone());
        Ok(Some(source))
    }
}
fn source_error(error: IoError) -> PyErr {
    if let IoError::Refused { code, message } = error {
        crate::with_reason_code(crate::value_err(message), code)
    } else {
        serialization(error)
    }
}

/// Original source standing and ancestry; absent for a supplied or historical surface.
#[pyfunction]
fn sensitivity_source_summary(
    artifact: &PySensitivityArtifact,
    py: Python<'_>,
) -> PyResult<Option<String>> {
    artifact
        .checked_source(py)?
        .as_ref()
        .map(|source| source.summary().map(|value| value.to_string()).map_err(serialization))
        .transpose()
}
/// Original complete v3 source, never a metadata-only numerical authority.
#[pyfunction]
fn original_sensitivity_source<'py>(
    artifact: &PySensitivityArtifact,
    py: Python<'py>,
) -> PyResult<Option<Bound<'py, PyBytes>>> {
    Ok(artifact
        .checked_source(py)?
        .as_ref()
        .map(|source| PyBytes::new(py, source.original_bytes())))
}
/// Export complete original source and derived surface for independent replay.
#[pyfunction]
fn export_sensitivity_source<'py>(
    artifact: &PySensitivityArtifact,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyBytes>> {
    let source = artifact.checked_source(py)?.ok_or_else(|| {
        crate::with_reason_code(
            crate::value_err("sensitivity_source.source_unavailable"),
            reason_code!("invalid_argument"),
        )
    })?;
    Ok(PyBytes::new(py, &source.export().map_err(serialization)?))
}
/// Independently execute the original source consumer and original surface adapter.
#[pyfunction]
fn consume_sensitivity_source(py: Python<'_>, data: &[u8]) -> PyResult<PySensitivityArtifact> {
    if data.len() > antecedent::analysis::sensitivity_source::MAX_BYTES {
        return Err(crate::with_reason_code(
            crate::value_err("sensitivity_source.limits_exceeded"),
            reason_code!("invalid_argument"),
        ));
    }
    let bytes = data.to_vec();
    let source = crate::detach_catch(py, move || {
        antecedent::analysis::sensitivity_source::SourceBackedSensitivity::consume(
            &bytes,
            &antecedent_core::ExecutionContext::production_default(0),
        )
        .map_err(source_error)
    })?;
    Ok(PySensitivityArtifact {
        artifact: source.surface().clone(),
        source: std::sync::Mutex::new(Some(std::sync::Arc::new(source))),
        recipe: None,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sensitivity_source_summary, m)?)?;
    m.add_function(wrap_pyfunction!(original_sensitivity_source, m)?)?;
    m.add_function(wrap_pyfunction!(export_sensitivity_source, m)?)?;
    m.add_function(wrap_pyfunction!(consume_sensitivity_source, m)?)
}
