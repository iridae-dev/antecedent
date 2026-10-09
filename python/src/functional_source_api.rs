//! Original finite-law functional proof binding; historical replay issues no authority.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_design::decision_artifact::contract_from_json_refusal;
use pyo3::prelude::*;

/// Retain the original law and full functional request for independent original-engine replay.
#[pyfunction]
fn export_law_functional_source(
    contract_json: &str,
    action_id: &str,
    functional_json: &str,
    input: &crate::composition_api::PyDecisionInput,
) -> PyResult<(String, Vec<u8>)> {
    if contract_json.len() > 1024 * 1024 {
        return Err(crate::with_reason_code(
            crate::value_err("functional_source.invalid_contract"),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    }
    let functional = crate::composition_api::parsed_functional(functional_json)?;
    let contract = contract_from_json_refusal(contract_json)
        .map_err(|cause| crate::with_reason_code(crate::value_err(cause.detail), cause.code))?;
    let antecedent_design::composition_boundary::InputSource::JointLaw(source) =
        input.inner().source()
    else {
        return Err(crate::with_reason_code(
            crate::value_err("functional_source.original_law_required"),
            antecedent_core::reason_code!("route_not_supported"),
        ));
    };
    let mut artifact = antecedent_design::functional_source::LawFunctionalArtifact::produce(
        source.clone(),
        contract,
        action_id,
        functional,
    )
    .map_err(functional_source_error)?;
    if let Some(claim) = &input.source_claim {
        artifact = artifact
            .with_source_evidence(
                &crate::source_evidence_api::PySourceEvidence::from_claim(claim)?.inner,
            )
            .map_err(functional_source_error)?;
    }
    Ok((
        artifact.summary().map_err(functional_source_error)?.to_string(),
        artifact.export().map_err(functional_source_error)?,
    ))
}
fn functional_source_error(error: antecedent_io::IoError) -> PyErr {
    if let antecedent_io::IoError::Refused { code, message } = error {
        crate::with_reason_code(crate::value_err(message), code)
    } else {
        crate::CausalSerializationError::new_err(error.to_string())
    }
}
/// Recompute the original finite-law functional from retained rows, semantics and request.
#[pyfunction]
#[pyo3(signature = (bytes, expected_identity=None))]
fn consume_law_functional_source(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> PyResult<String> {
    antecedent_design::functional_source::LawFunctionalArtifact::consume_expected(
        bytes,
        expected_identity,
    )
    .and_then(|artifact| artifact.summary())
    .map(|value| value.to_string())
    .map_err(functional_source_error)
}

/// Explicit actual producer replay validates retained original rows without importing authority.
#[pyfunction]
fn resolve_law_functional_source(
    bytes: &[u8],
    actual: &crate::program_claims_api::PyNativeResponseClaim,
) -> PyResult<String> {
    use antecedent::analysis::native_claims::NativeDecisionSource;
    use antecedent_design::decision_contract::{SourceRepresentation, SourceRequirement};
    let artifact = antecedent_design::functional_source::LawFunctionalArtifact::consume(bytes)
        .map_err(functional_source_error)?;
    let expected = artifact.source_evidence().ok_or_else(|| {
        crate::with_reason_code(
            crate::value_err("functional_source.native_source_unavailable"),
            antecedent_core::reason_code!("invalid_argument"),
        )
    })?;
    let fresh = crate::source_evidence_api::PySourceEvidence::from_claim(actual)?;
    expected.require_same_source(&fresh.inner).map_err(|_| {
        crate::with_reason_code(
            crate::value_err("functional_source.source_mismatch"),
            antecedent_core::reason_code!("invalid_argument"),
        )
    })?;
    let original = actual
        .claim
        .decision_source(&SourceRequirement {
            any_of: vec![SourceRepresentation::JointDraws],
            sampled_needs_error_receipt: false,
        })
        .map_err(|cause| crate::with_reason_code(crate::value_err(cause.detail), cause.code))?;
    let NativeDecisionSource::JointLaw(law) = original.source else {
        return Err(crate::with_reason_code(
            crate::value_err("functional_source.native_source_unavailable"),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    };
    if artifact
        .original_law()
        .to_bytes("native-law-source-check")
        .map_err(functional_source_error)?
        != law.to_bytes("native-law-source-check").map_err(functional_source_error)?
    {
        return Err(crate::with_reason_code(
            crate::value_err("functional_source.source_mismatch"),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    }
    let mut report = artifact.summary().map_err(functional_source_error)?;
    report["native_row_origin_verified"] = serde_json::json!(true);
    report["source_resolution"] = serde_json::json!({"kind":"actual_original_native_law_comparison","native_authority_issued":false,"calibration_license_issued":false});
    Ok(report.to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(export_law_functional_source, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_law_functional_source, m)?)?;
    m.add_function(wrap_pyfunction!(consume_law_functional_source, m)?)
}
