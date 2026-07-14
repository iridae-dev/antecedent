//! Wire types and encode/decode for columnar posterior artifacts (Phase 6).
//!
//! Draws live in an Arrow-IPC (or raw f64 LE) numerical section; metadata is CBOR.
//! Internal Rust structs are never serialized directly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor,
};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::wire::{
    ArtifactKind, FormatVersion, ProvenanceWire, SemanticVersion,
};

/// Quantity kind on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PosteriorQuantityWire {
    /// Coefficient index.
    Coefficient {
        /// Index.
        index: u32,
        /// Optional name.
        name: Option<String>,
    },
    /// Residual variance.
    ResidualVariance,
    /// Named effect.
    Effect {
        /// Name.
        name: String,
    },
    /// Named scalar.
    Scalar {
        /// Name.
        name: String,
    },
}

/// CBOR metadata for a posterior artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CausalPosteriorWire {
    /// Schema quantities in column order.
    pub quantities: Vec<PosteriorQuantityWire>,
    /// Number of draws.
    pub n_draws: u32,
    /// Per-quantity mean.
    pub mean: Vec<f64>,
    /// Per-quantity SD.
    pub sd: Vec<f64>,
    /// 2.5% quantile.
    pub q025: Vec<f64>,
    /// 97.5% quantile.
    pub q975: Vec<f64>,
    /// Identification status tag.
    pub identification: String,
    /// Unidentified graph mass retained.
    pub unidentified_mass: f64,
    /// Backend id.
    pub backend_id: String,
    /// Whether Laplace/conjugate reported convergence.
    pub converged: bool,
    /// Hessian condition (NaN if analytic).
    pub hessian_condition: f64,
    /// Draw encoding: `f64_le_colmajor` in section `posterior.draws`.
    pub draws_encoding: String,
}

/// Encode a posterior artifact (CBOR meta + little-endian f64 column-major draws).
///
/// # Errors
///
/// CBOR / IO failures.
pub fn encode_posterior_artifact(
    meta: &CausalPosteriorWire,
    draws_colmajor: &[f64],
    artifact_id: &str,
    library_version: &str,
) -> Result<EncodedArtifact, IoError> {
    let expected = meta.n_draws as usize * meta.quantities.len();
    if draws_colmajor.len() != expected {
        return Err(IoError::Convert(format!(
            "draws length {} != n_draws*n_quantities {}",
            draws_colmajor.len(),
            expected
        )));
    }
    let meta_bytes = to_cbor(meta)?;
    let mut draw_bytes = Vec::with_capacity(draws_colmajor.len() * 8);
    for &v in draws_colmajor {
        draw_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let meta_desc = section_descriptor("posterior.meta", "application/cbor", &meta_bytes);
    let draw_desc =
        section_descriptor("posterior.draws", "application/octet-stream", &draw_bytes);
    Ok(EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: FormatVersion { major: 0, minor: 1 },
            minimum_reader_version: FormatVersion { major: 0, minor: 1 },
            artifact_kind: ArtifactKind::CausalPosterior,
            library_version: SemanticVersion::from_crate_version(library_version),
            artifact_id: artifact_id.into(),
            sections: vec![meta_desc, draw_desc],
            provenance: ProvenanceWire { note: "causal_posterior".into() },
        },
        sections: vec![
            SectionBytes { id: "posterior.meta".into(), data: meta_bytes },
            SectionBytes { id: "posterior.draws".into(), data: draw_bytes },
        ],
    })
}

/// Decode a posterior artifact into metadata + column-major draws.
///
/// # Errors
///
/// Missing sections or format errors.
pub fn decode_posterior_artifact(
    artifact: &EncodedArtifact,
) -> Result<(CausalPosteriorWire, Vec<f64>), IoError> {
    if artifact.manifest.artifact_kind != ArtifactKind::CausalPosterior {
        return Err(IoError::Convert(format!(
            "expected CausalPosterior, got {:?}",
            artifact.manifest.artifact_kind
        )));
    }
    let meta_sec = artifact
        .sections
        .iter()
        .find(|s| s.id == "posterior.meta")
        .ok_or_else(|| IoError::Convert("missing posterior.meta".into()))?;
    let draw_sec = artifact
        .sections
        .iter()
        .find(|s| s.id == "posterior.draws")
        .ok_or_else(|| IoError::Convert("missing posterior.draws".into()))?;
    let meta: CausalPosteriorWire = from_cbor(&meta_sec.data)?;
    if draw_sec.data.len() % 8 != 0 {
        return Err(IoError::Convert("posterior.draws not multiple of 8".into()));
    }
    let mut draws = Vec::with_capacity(draw_sec.data.len() / 8);
    for chunk in draw_sec.data.chunks_exact(8) {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        draws.push(f64::from_le_bytes(buf));
    }
    let expected = meta.n_draws as usize * meta.quantities.len();
    if draws.len() != expected {
        return Err(IoError::Convert(format!(
            "decoded draws {} != expected {}",
            draws.len(),
            expected
        )));
    }
    Ok((meta, draws))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterior_artifact_round_trip() {
        let meta = CausalPosteriorWire {
            quantities: vec![PosteriorQuantityWire::Effect { name: "ate".into() }],
            n_draws: 3,
            mean: vec![1.0],
            sd: vec![0.1],
            q025: vec![0.8],
            q975: vec![1.2],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 10.0,
            draws_encoding: "f64_le_colmajor".into(),
        };
        let draws = vec![0.9, 1.0, 1.1];
        let art = encode_posterior_artifact(&meta, &draws, "test-post", "0.1.0").unwrap();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        let (meta2, draws2) = decode_posterior_artifact(&decoded).unwrap();
        assert_eq!(meta2.n_draws, 3);
        assert_eq!(draws2, draws);
        assert_eq!(meta2.backend_id, "laplace");
    }
}
