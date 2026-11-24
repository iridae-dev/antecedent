//! The retained-artifact check every Python loader goes through.
//!
//! A Python loader that receives `retained_identity` calls [`bind_retained_artifact`] before it
//! reads the bytes, so the comparison is the same Rust `bind` the Rust `*_expecting` loaders
//! use (`antecedent::io::bind_retained_artifact`), never a Python string comparison.

use antecedent::io::{RetainedArtifactIdentity, bind_retained_artifact as bind_retained};
use pyo3::prelude::*;

/// The identity to retain for `data`: lowercase hex of its artifact digest.
#[pyfunction]
fn retained_artifact_identity(data: &[u8]) -> String {
    RetainedArtifactIdentity::of_bytes(data).to_hex()
}

/// Admit `data` only as the artifact whose identity the caller retained.
///
/// Raises the serialization error carrying `reason=external_binding_mismatch` when the bytes
/// are not that artifact, and `reason=invalid_argument` when `retained_identity` is not 64 hex
/// characters.
#[pyfunction]
fn bind_retained_artifact(data: &[u8], retained_identity: &str) -> PyResult<()> {
    let retained = RetainedArtifactIdentity::from_hex(retained_identity)
        .map_err(crate::bridge::serialization)?;
    bind_retained(data, &retained).map_err(crate::bridge::serialization)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(retained_artifact_identity, module)?)?;
    module.add_function(wrap_pyfunction!(bind_retained_artifact, module)?)?;
    Ok(())
}
