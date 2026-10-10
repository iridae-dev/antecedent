//! Original-source semantic projections and actual checked native operation resolution.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{detach_catch, value_err, with_reason_code};
use antecedent::analysis::composition::BundleLimits;
use antecedent::analysis::source_evidence::SourceResolution;
use antecedent::analysis::source_projection::source_resolved_consumer;
use antecedent_design::source_projection_artifact::{
    MAX_SOURCE_PROJECTION_BYTES, ProjectionSource, SourceProjection, SourceProjectionArtifact,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn error(error: antecedent_io::IoError) -> PyErr {
    match error {
        antecedent_io::IoError::Refused { code, message } => {
            with_reason_code(value_err(message), code)
        }
        other => invalid("source_projection.original_consumer_refused", other.to_string()),
    }
}
fn invalid(detail: &'static str, message: impl std::fmt::Display) -> PyErr {
    with_reason_code(
        value_err(format!("{detail}: {message}")),
        antecedent_core::reason_code!("invalid_argument"),
    )
}
fn bounded(bytes: &[u8]) -> PyResult<()> {
    if bytes.len() > MAX_SOURCE_PROJECTION_BYTES {
        return Err(invalid(
            "source_projection.limits_exceeded",
            "source projection exceeds byte bound",
        ));
    }
    Ok(())
}
fn summary(artifact: &SourceProjectionArtifact) -> String {
    serde_json::json!({"identity":artifact.identity(),"kind":artifact.node_kind().as_str(),"report":artifact.report()}).to_string()
}
#[pyfunction]
#[pyo3(signature=(source, native, projection_json))]
fn produce_source_projection(
    source: &[u8],
    native: bool,
    projection_json: &str,
) -> PyResult<(String, Vec<u8>)> {
    bounded(source)?;
    if projection_json.len() > 1024 * 1024 {
        return Err(invalid(
            "source_projection.limits_exceeded",
            "projection declaration exceeds byte bound",
        ));
    }
    let projection: SourceProjection = serde_json::from_str(projection_json)
        .map_err(|e| invalid("source_projection.invalid_request", e))?;
    let source = if native {
        ProjectionSource::NativeResponse(source.to_vec())
    } else {
        ProjectionSource::ExternalClaim(source.to_vec())
    };
    let artifact = SourceProjectionArtifact::produce(source, projection).map_err(error)?;
    Ok((summary(&artifact), artifact.export().map_err(error)?))
}
#[pyfunction]
fn consume_source_projection(data: &[u8], expected_identity: &str) -> PyResult<String> {
    bounded(data)?;
    let artifact = SourceProjectionArtifact::consume(data, expected_identity).map_err(error)?;
    Ok(summary(&artifact))
}
/// Resolve each distinct original source by actual original prepared-study execution.
#[pyfunction]
fn consume_source_projection_bundle(
    py: Python<'_>,
    data: &[u8],
    expected_identity: &str,
    supplied_json: Option<&str>,
    actual_claims: &Bound<'_, PyDict>,
) -> PyResult<(Option<String>, Option<String>)> {
    let limits = BundleLimits::default();
    if data.len() > limits.max_bundle_bytes || actual_claims.len() > 64 {
        return Err(invalid(
            "source_projection.limits_exceeded",
            "bundle or source resolver count exceeds bound",
        ));
    }
    let supplied = crate::composition_bundle_api::supplied_from(supplied_json)?;
    let mut sources = Vec::with_capacity(actual_claims.len());
    for (digest, value) in actual_claims.iter() {
        let digest: String = digest.extract()?;
        let actual: PyRef<'_, crate::program_claims_api::PyNativeResponseClaim> =
            value.extract()?;
        let evidence = crate::source_evidence_api::PySourceEvidence::from_claim(&actual)?;
        if evidence.inner.summary()["source_artifact_digest"].as_str() != Some(digest.as_str()) {
            return Err(invalid(
                "source_projection.source_binding_mismatch",
                "issued native execution differs from the expected original source",
            ));
        }
        sources.push((
            evidence.inner,
            std::sync::Arc::clone(&actual.authority.prepared),
            actual.authority.context.clone(),
        ));
    }
    let data = data.to_vec();
    let expected_identity = expected_identity.to_owned();
    detach_catch(py, move || {
        let receipts = sources
            .iter()
            .map(|(evidence, prepared, context)| {
                SourceResolution::execute(evidence, prepared, context)
                    .map_err(|e| invalid("source_projection.source_binding_mismatch", e))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let consumer = source_resolved_consumer(receipts).map_err(error)?;
        Ok(match consumer.consume(&data, &limits, &expected_identity, &supplied) {
            Ok(result) => (Some(crate::composition_bundle_api::consumed_json(&result)), None),
            Err(error) => (None, Some(crate::composition_bundle_api::error_refusal(&error))),
        })
    })
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(produce_source_projection, m)?)?;
    m.add_function(wrap_pyfunction!(consume_source_projection, m)?)?;
    m.add_function(wrap_pyfunction!(consume_source_projection_bundle, m)?)
}
