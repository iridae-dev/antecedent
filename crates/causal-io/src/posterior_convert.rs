//! Convert [`CausalPosterior`] (internal) ↔ versioned wire artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::redundant_closure_for_method_calls)]

use causal_core::VERSION;
use causal_estimate::CausalPosterior;
use causal_identify::IdentificationStatus;
use causal_prob::PosteriorQuantityKind;

use crate::container::EncodedArtifact;
use crate::error::IoError;
use crate::posterior::{
    CausalPosteriorWire, PosteriorQuantityWire, decode_posterior_artifact,
    encode_posterior_artifact,
};

/// Which posterior payload to serialize across the FFI / artifact boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PosteriorPayload {
    /// Meta summaries only (mean/sd/quantiles); no draw bytes. Enough for prior hydrate.
    Summary,
    /// Meta + full column-major draws.
    FullDraws,
}

/// Encode a [`CausalPosterior`] to container bytes (Python / tooling).
///
/// Defaults to [`PosteriorPayload::FullDraws`] for Rust tooling parity.
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior_bytes(
    posterior: &CausalPosterior,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    encode_causal_posterior_bytes_with_payload(posterior, artifact_id, PosteriorPayload::FullDraws)
}

/// Encode a [`CausalPosterior`] with an explicit payload mode.
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior_bytes_with_payload(
    posterior: &CausalPosterior,
    artifact_id: &str,
    payload: PosteriorPayload,
) -> Result<Vec<u8>, IoError> {
    let art = encode_causal_posterior_with_payload(posterior, artifact_id, payload)?;
    let mut buf = Vec::new();
    art.write_to(&mut buf)?;
    Ok(buf)
}

/// Encode a [`CausalPosterior`] to a durable artifact (full draws).
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior(
    posterior: &CausalPosterior,
    artifact_id: &str,
) -> Result<EncodedArtifact, IoError> {
    encode_causal_posterior_with_payload(posterior, artifact_id, PosteriorPayload::FullDraws)
}

/// Encode a [`CausalPosterior`] with an explicit payload mode.
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior_with_payload(
    posterior: &CausalPosterior,
    artifact_id: &str,
    payload: PosteriorPayload,
) -> Result<EncodedArtifact, IoError> {
    let quantities: Vec<PosteriorQuantityWire> = posterior
        .draws
        .schema
        .quantities
        .iter()
        .map(|q| match q {
            PosteriorQuantityKind::Coefficient { index, name } => {
                PosteriorQuantityWire::Coefficient {
                    index: *index as u32,
                    name: name.as_ref().map(|s| s.to_string()),
                }
            }
            PosteriorQuantityKind::ResidualVariance => PosteriorQuantityWire::ResidualVariance,
            PosteriorQuantityKind::Effect { name } => {
                PosteriorQuantityWire::Effect { name: name.to_string() }
            }
            PosteriorQuantityKind::Scalar { name } => {
                PosteriorQuantityWire::Scalar { name: name.to_string() }
            }
        })
        .collect();
    let (draws_encoding, draws): (&str, &[f64]) = match payload {
        PosteriorPayload::FullDraws => ("f64_le_colmajor", posterior.draws.values.as_ref()),
        PosteriorPayload::Summary => ("none", &[]),
    };
    let meta = CausalPosteriorWire {
        quantities,
        n_draws: posterior.draws.n_draws as u32,
        mean: posterior.summaries.mean.to_vec(),
        sd: posterior.summaries.sd.to_vec(),
        q025: posterior.summaries.q025.to_vec(),
        q975: posterior.summaries.q975.to_vec(),
        identification: match posterior.identification {
            IdentificationStatus::NonparametricallyIdentified => {
                "NonparametricallyIdentified".into()
            }
            IdentificationStatus::IdentifiedUnderParametricRestrictions => {
                "IdentifiedUnderParametricRestrictions".into()
            }
            IdentificationStatus::IdentifiedUnderPriorRestrictions => {
                "IdentifiedUnderPriorRestrictions".into()
            }
            IdentificationStatus::PartiallyIdentified => "PartiallyIdentified".into(),
            IdentificationStatus::GraphDependent => "GraphDependent".into(),
            IdentificationStatus::NotIdentified => "NotIdentified".into(),
        },
        unidentified_mass: posterior.unidentified_mass,
        backend_id: posterior.diagnostics.backend_id.to_string(),
        converged: posterior.diagnostics.converged,
        hessian_condition: posterior.diagnostics.hessian_condition,
        draws_encoding: draws_encoding.into(),
    };
    encode_posterior_artifact(&meta, draws, artifact_id, VERSION)
}

/// Decode posterior wire metadata + draws (Python / tooling consumers).
///
/// Summary artifacts return an empty draws vector.
///
/// # Errors
///
/// IO failures.
pub fn decode_causal_posterior_bytes(
    bytes: &[u8],
) -> Result<(CausalPosteriorWire, Vec<f64>), IoError> {
    let artifact = crate::migrate::read_and_migrate(bytes)?;
    decode_posterior_artifact(&artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::{CausalPosteriorWire, PosteriorQuantityWire};

    #[test]
    fn summary_payload_encoding_omits_draw_bytes() {
        let meta = CausalPosteriorWire {
            quantities: vec![
                PosteriorQuantityWire::Coefficient { index: 0, name: Some("intercept".into()) },
                PosteriorQuantityWire::Effect { name: "ate".into() },
            ],
            n_draws: 8192,
            mean: vec![0.1, 2.0],
            sd: vec![0.05, 0.2],
            q025: vec![0.0, 1.6],
            q975: vec![0.2, 2.4],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "none".into(),
        };
        let summary = encode_posterior_artifact(&meta, &[], "summary", VERSION).unwrap();
        let mut summary_bytes = Vec::new();
        summary.write_to(&mut summary_bytes).unwrap();

        let mut full_meta = meta.clone();
        full_meta.draws_encoding = "f64_le_colmajor".into();
        let draws = vec![0.0_f64; 8192 * 2];
        let full = encode_posterior_artifact(&full_meta, &draws, "full", VERSION).unwrap();
        let mut full_bytes = Vec::new();
        full.write_to(&mut full_bytes).unwrap();
        assert!(summary_bytes.len() < full_bytes.len());
        // Draw payload is omitted under summary encoding (container may compress).
        assert_eq!(
            summary.sections.iter().find(|s| s.id == "posterior.draws").map(|s| s.data.len()),
            Some(0)
        );

        let (wire, decoded_draws) = decode_causal_posterior_bytes(&summary_bytes).unwrap();
        assert!(decoded_draws.is_empty());
        assert_eq!(wire.draws_encoding, "none");
        assert_eq!(wire.n_draws, 8192);
        assert_eq!(wire.mean, vec![0.1, 2.0]);
        assert_eq!(wire.sd, vec![0.05, 0.2]);
    }
}
