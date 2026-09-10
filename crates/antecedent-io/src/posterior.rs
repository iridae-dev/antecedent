//! Wire types and encode/decode for columnar posterior artifacts .
//!
//! Draws live in an Arrow-IPC (or raw f64 LE) numerical section; metadata is CBOR.
//! Internal Rust structs are never serialized directly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

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

fn validate_posterior_meta(
    meta: &CausalPosteriorWire,
    draws: Option<&[f64]>,
) -> Result<(), IoError> {
    let quantities = meta.quantities.len();
    if quantities == 0
        || meta.mean.len() != quantities
        || meta.sd.len() != quantities
        || meta.q025.len() != quantities
        || meta.q975.len() != quantities
    {
        return Err(IoError::Convert(
            "posterior summaries must match a non-empty quantity schema".into(),
        ));
    }
    if meta
        .mean
        .iter()
        .chain(&meta.sd)
        .chain(&meta.q025)
        .chain(&meta.q975)
        .any(|value| !value.is_finite())
        || meta.sd.iter().any(|value| *value < 0.0)
        || meta.q025.iter().zip(&meta.q975).any(|(lower, upper)| lower > upper)
    {
        return Err(IoError::Convert(
            "posterior summaries must be finite with non-negative SD and ordered quantiles".into(),
        ));
    }
    if !meta.unidentified_mass.is_finite() || !(0.0..=1.0).contains(&meta.unidentified_mass) {
        return Err(IoError::Convert("posterior unidentified mass must lie in [0,1]".into()));
    }
    if meta.backend_id.trim().is_empty() || meta.n_draws == 0 {
        return Err(IoError::Convert(
            "posterior backend id must be non-blank and draw count must be positive".into(),
        ));
    }
    if !matches!(
        meta.identification.as_str(),
        "NonparametricallyIdentified"
            | "IdentifiedUnderParametricRestrictions"
            | "IdentifiedUnderPriorRestrictions"
            | "PartiallyIdentified"
            | "GraphDependent"
            | "NotIdentified"
    ) {
        return Err(IoError::Convert(format!(
            "unknown posterior identification status `{}`",
            meta.identification
        )));
    }
    if !matches!(meta.draws_encoding.as_str(), "none" | "f64_le_colmajor") {
        return Err(IoError::Convert(format!(
            "unknown posterior draws encoding `{}`",
            meta.draws_encoding
        )));
    }
    if let Some(draws) = draws {
        if draws.iter().any(|value| !value.is_finite()) {
            return Err(IoError::Convert("posterior draws must be finite".into()));
        }
        let expected = if meta.draws_encoding == "none" {
            0
        } else {
            usize::try_from(meta.n_draws)
                .ok()
                .and_then(|count| count.checked_mul(quantities))
                .ok_or(IoError::TooLarge)?
        };
        if draws.len() != expected {
            return Err(IoError::Convert(format!(
                "posterior draws length {} != expected {expected}",
                draws.len()
            )));
        }
    }
    Ok(())
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
    validate_posterior_meta(meta, Some(draws_colmajor))?;
    let summary_only = meta.draws_encoding == "none";
    if summary_only {
        if !draws_colmajor.is_empty() {
            return Err(IoError::Convert(
                "summary posterior encoding expects empty draws payload".into(),
            ));
        }
    } else {
        let expected = meta.n_draws as usize * meta.quantities.len();
        if draws_colmajor.len() != expected {
            return Err(IoError::Convert(format!(
                "draws length {} != n_draws*n_quantities {}",
                draws_colmajor.len(),
                expected
            )));
        }
    }
    let meta_bytes = to_cbor(meta)?;
    let mut draw_bytes = Vec::with_capacity(draws_colmajor.len() * 8);
    for &v in draws_colmajor {
        draw_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let meta_desc = section_descriptor("posterior.meta", "application/cbor", &meta_bytes);
    let draw_desc = section_descriptor("posterior.draws", "application/octet-stream", &draw_bytes);
    Ok(EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::CausalPosterior,
            library_version: SemanticVersion::from_crate_version(library_version)?,
            artifact_id: artifact_id.into(),
            sections: vec![meta_desc, draw_desc],
            provenance: ProvenanceWire { note: "causal_posterior".into() },
        },
        sections: vec![
            SectionBytes::new("posterior.meta", meta_bytes),
            SectionBytes::new("posterior.draws", draw_bytes),
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
    validate_posterior_meta(&meta, Some(&draws))?;
    Ok((meta, draws))
}

/// Decode only posterior metadata from a seekable artifact (draws stay unread).
///
/// # Errors
///
/// Missing `posterior.meta`, wrong kind, or IO/framing failures.
pub fn decode_posterior_meta_from_seek<R: std::io::Read + std::io::Seek>(
    r: R,
) -> Result<CausalPosteriorWire, IoError> {
    let mut reader = crate::reader::ArtifactReader::open_seek(r)?;
    if reader.manifest().artifact_kind != ArtifactKind::CausalPosterior {
        return Err(IoError::Convert(format!(
            "expected CausalPosterior, got {:?}",
            reader.manifest().artifact_kind
        )));
    }
    let access = reader.load_section("posterior.meta")?;
    let meta = from_cbor(access.as_bytes())?;
    validate_posterior_meta(&meta, None)?;
    Ok(meta)
}

/// Decode only posterior metadata from a memory-mapped artifact path.
///
/// # Errors
///
/// Same as [`decode_posterior_meta_from_seek`].
pub fn decode_posterior_meta_from_path(
    path: impl AsRef<std::path::Path>,
) -> Result<CausalPosteriorWire, IoError> {
    let mut reader = crate::reader::MappedArtifactReader::open_path(path)?;
    if reader.manifest().artifact_kind != ArtifactKind::CausalPosterior {
        return Err(IoError::Convert(format!(
            "expected CausalPosterior, got {:?}",
            reader.manifest().artifact_kind
        )));
    }
    let access = reader.load_section("posterior.meta")?;
    let meta = from_cbor(access.as_bytes())?;
    validate_posterior_meta(&meta, None)?;
    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_meta() -> CausalPosteriorWire {
        CausalPosteriorWire {
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
        }
    }

    #[test]
    fn posterior_artifact_round_trip() {
        let meta = valid_meta();
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

    #[test]
    fn posterior_rejects_invalid_mass_and_summary_shape() {
        let draws = vec![0.9, 1.0, 1.1];
        let mut invalid_mass = valid_meta();
        invalid_mass.unidentified_mass = 1.1;
        assert!(encode_posterior_artifact(&invalid_mass, &draws, "bad", "0.9.0").is_err());

        let mut invalid_summary = valid_meta();
        invalid_summary.q975.clear();
        assert!(encode_posterior_artifact(&invalid_summary, &draws, "bad", "0.9.0").is_err());

        let mut invalid_encoding = valid_meta();
        invalid_encoding.draws_encoding = "opaque".into();
        assert!(encode_posterior_artifact(&invalid_encoding, &draws, "bad", "0.9.0").is_err());
    }

    #[test]
    fn posterior_meta_only_skips_draws() {
        let meta = CausalPosteriorWire {
            quantities: vec![PosteriorQuantityWire::Effect { name: "ate".into() }],
            n_draws: 8192,
            mean: vec![1.0],
            sd: vec![0.1],
            q025: vec![0.9],
            q975: vec![1.1],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "f64_le_colmajor".into(),
        };
        let draws = vec![0.5f64; 8192];
        let art = encode_posterior_artifact(&meta, &draws, "meta-only", "0.1.0").unwrap();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let got = decode_posterior_meta_from_seek(std::io::Cursor::new(buf.clone())).unwrap();
        assert_eq!(got.n_draws, 8192);
        assert_eq!(got.backend_id, "laplace");

        // Zero-copy accounting (DESIGN rule 22): a meta-only read must not pay
        // for the 64 KiB draws section — bytes_loaded stays below the draws
        // payload and the draws section remains skipped.
        let mut reader =
            crate::reader::ArtifactReader::open_seek(std::io::Cursor::new(buf)).unwrap();
        let _ = reader.load_section("posterior.meta").unwrap();
        let stats = reader.stats();
        let draws_bytes = (8192 * std::mem::size_of::<f64>()) as u64;
        assert_eq!(stats.sections_loaded, 1);
        assert!(
            stats.bytes_loaded < draws_bytes,
            "meta-only load read {} bytes (draws section is {draws_bytes})",
            stats.bytes_loaded
        );
        assert!(stats.sections_skipped >= 1);
    }
}
