//! Convert [`CausalPosterior`] (internal) ↔ versioned wire artifacts (DESIGN.md §24.2).
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

/// Encode a [`CausalPosterior`] to container bytes (Python / tooling).
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior_bytes(
    posterior: &CausalPosterior,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    let art = encode_causal_posterior(posterior, artifact_id)?;
    let mut buf = Vec::new();
    art.write_to(&mut buf)?;
    Ok(buf)
}

/// Encode a [`CausalPosterior`] to a durable artifact.
///
/// # Errors
///
/// IO failures.
pub fn encode_causal_posterior(
    posterior: &CausalPosterior,
    artifact_id: &str,
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
            IdentificationStatus::PartiallyIdentified => "PartiallyIdentified".into(),
            IdentificationStatus::GraphDependent => "GraphDependent".into(),
            IdentificationStatus::NotIdentified => "NotIdentified".into(),
        },
        unidentified_mass: posterior.unidentified_mass,
        backend_id: posterior.diagnostics.backend_id.to_string(),
        converged: posterior.diagnostics.converged,
        hessian_condition: posterior.diagnostics.hessian_condition,
        draws_encoding: "f64_le_colmajor".into(),
    };
    encode_posterior_artifact(&meta, &posterior.draws.values, artifact_id, VERSION)
}

/// Decode posterior wire metadata + draws (Python / tooling consumers).
///
/// # Errors
///
/// IO failures.
pub fn decode_causal_posterior_bytes(
    bytes: &[u8],
) -> Result<(CausalPosteriorWire, Vec<f64>), IoError> {
    let artifact = EncodedArtifact::read_from(bytes)?;
    decode_posterior_artifact(&artifact)
}
