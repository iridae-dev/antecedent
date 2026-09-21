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
    /// Identified graph mass a latency tier left out of the envelope subsample
    /// (never evaluated; not unidentified). Omitted when zero; artifacts written
    /// before this field decode as zero.
    #[serde(default, skip_serializing_if = "is_zero_mass")]
    pub subsampled_out_mass: f64,
    /// Backend id.
    pub backend_id: String,
    /// Whether Laplace/conjugate reported convergence.
    pub converged: bool,
    /// Hessian condition (NaN if analytic).
    pub hessian_condition: f64,
    /// Draw encoding: `f64_le_colmajor` in section `posterior.draws`.
    pub draws_encoding: String,
    /// Source treatment contrast `active − control`, when the posterior is an
    /// identity-link effect. Used by effect-functional hydrate (`ATE / Δ`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment_contrast: Option<f64>,
}

#[allow(clippy::trivially_copy_pass_by_ref)] // serde's skip_serializing_if passes `&T`.
fn is_zero_mass(mass: &f64) -> bool {
    *mass == 0.0
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
    if !meta.subsampled_out_mass.is_finite()
        || !(0.0..=1.0).contains(&meta.subsampled_out_mass)
        || meta.unidentified_mass + meta.subsampled_out_mass > 1.0 + 1e-9
    {
        return Err(IoError::Convert(
            "posterior subsampled-out mass must lie in [0,1] and not exceed the mass left \
             after unidentified mass"
                .into(),
        ));
    }
    if meta.backend_id.trim().is_empty() || meta.n_draws == 0 {
        return Err(IoError::Convert(
            "posterior backend id must be non-blank and draw count must be positive".into(),
        ));
    }
    if crate::analysis_wire::identification_status_from_any(&meta.identification).is_none() {
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
        if !draws.is_empty() {
            validate_summaries_against_draws(meta, draws)?;
        }
    }
    Ok(())
}

/// Stored summaries must describe the embedded draws, so a summary-reading and a
/// draw-reading consumer see one posterior.
///
/// The quantiles are a deterministic function of the draws and are recomputed with the
/// producer's own routine (`PosteriorDraws::summarize`), agreeing to a relative 1e-9. The
/// mean and SD may legitimately be exact moments of a mixture whose draws are a Monte-Carlo
/// sample (the structural-envelope posterior), so they are held to the draws' own sampling
/// error rather than bit agreement: eight standard errors of the draw mean / draw SD, which
/// still rejects any summary that belongs to a different distribution.
fn validate_summaries_against_draws(
    meta: &CausalPosteriorWire,
    draws: &[f64],
) -> Result<(), IoError> {
    use antecedent_prob::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};

    const RELATIVE: f64 = 1e-9;
    const STANDARD_ERRORS: f64 = 8.0;
    let n_draws = meta.n_draws as usize;
    let schema = PosteriorSchema {
        quantities: (0..meta.quantities.len())
            .map(|_| PosteriorQuantityKind::Scalar { name: "q".into() })
            .collect(),
    };
    let recomputed = PosteriorDraws::from_column_major(schema, n_draws, draws.to_vec())
        .map_err(|err| IoError::Convert(err.to_string()))?
        .summarize();
    let close = |stored: f64, derived: f64, slack: f64| {
        (stored - derived).abs() <= RELATIVE * derived.abs().max(stored.abs()).max(1.0) + slack
    };
    let n = n_draws as f64;
    for q in 0..meta.quantities.len() {
        let sd = recomputed.sd[q];
        let mean_se = sd / n.sqrt();
        let sd_se = if n_draws > 1 { sd / (2.0 * (n - 1.0)).sqrt() } else { 0.0 };
        let agrees = close(meta.mean[q], recomputed.mean[q], STANDARD_ERRORS * mean_se)
            && close(meta.sd[q], sd, STANDARD_ERRORS * sd_se)
            && close(meta.q025[q], recomputed.q025[q], 0.0)
            && close(meta.q975[q], recomputed.q975[q], 0.0);
        if !agrees {
            return Err(IoError::Convert(format!(
                "posterior summaries for quantity {q} disagree with the embedded draws"
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
            q025: vec![0.9],
            q975: vec![1.1],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 10.0,
            draws_encoding: "f64_le_colmajor".into(),
            treatment_contrast: None,
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
    fn subsampled_out_mass_is_optional_on_the_wire_and_round_trips() {
        let draws = vec![0.9, 1.0, 1.1];
        let round_trip = |meta: &CausalPosteriorWire| {
            let art = encode_posterior_artifact(meta, &draws, "subsampled", "0.1.0").unwrap();
            let mut buf = Vec::new();
            art.write_to(&mut buf).unwrap();
            decode_posterior_artifact(&EncodedArtifact::read_from(buf.as_slice()).unwrap())
                .unwrap()
                .0
        };
        // A zero share stays off the wire, so older readers and artifacts agree.
        let zero = valid_meta();
        let bytes = to_cbor(&zero).unwrap();
        let value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
        let keys: Vec<String> = value
            .as_map()
            .unwrap()
            .iter()
            .filter_map(|(key, _)| key.as_text().map(str::to_owned))
            .collect();
        assert!(keys.iter().any(|key| key == "unidentified_mass"));
        assert!(keys.iter().all(|key| key != "subsampled_out_mass"));
        assert!(round_trip(&zero).subsampled_out_mass.abs() < f64::EPSILON);

        let mut split = valid_meta();
        split.identification = "GraphDependent".into();
        split.unidentified_mass = 0.1;
        split.subsampled_out_mass = 0.4;
        let decoded = round_trip(&split);
        assert!((decoded.subsampled_out_mass - 0.4).abs() < f64::EPSILON);
        assert!((decoded.unidentified_mass - 0.1).abs() < f64::EPSILON);

        for bad in [-0.1, 1.5, f64::NAN, 0.95] {
            let mut invalid = split.clone();
            invalid.subsampled_out_mass = bad;
            assert!(encode_posterior_artifact(&invalid, &draws, "bad", "0.1.0").is_err(), "{bad}");
        }
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
    fn posterior_summaries_must_describe_the_embedded_draws() {
        // Draws centred at 0 (mean 0, sample SD 1, nearest-rank quantiles -1 / 1).
        let draws = vec![-1.0, 0.0, 1.0];
        let mut consistent = valid_meta();
        consistent.mean = vec![0.0];
        consistent.sd = vec![1.0];
        consistent.q025 = vec![-1.0];
        consistent.q975 = vec![1.0];
        encode_posterior_artifact(&consistent, &draws, "ok", "0.1.0").unwrap();

        // Summary reader would see 5 +- 0.5 while a draw reader sees 0 +- 1.
        let mut shifted = consistent.clone();
        shifted.mean = vec![5.0];
        shifted.q025 = vec![4.5];
        shifted.q975 = vec![5.5];
        let error =
            encode_posterior_artifact(&shifted, &draws, "bad", "0.1.0").unwrap_err().to_string();
        assert!(error.contains("disagree with the embedded draws"), "{error}");

        // Quantiles are exact functions of the draws.
        let mut narrowed = consistent.clone();
        narrowed.q975 = vec![0.5];
        assert!(encode_posterior_artifact(&narrowed, &draws, "bad", "0.1.0").is_err());

        // Mean / SD tolerate Monte-Carlo scale (exact mixture moments) but not a
        // different distribution.
        let mut mc = consistent.clone();
        mc.mean = vec![0.1];
        mc.sd = vec![1.2];
        encode_posterior_artifact(&mc, &draws, "mc", "0.1.0").unwrap();
        let mut wide = consistent;
        wide.sd = vec![40.0];
        assert!(encode_posterior_artifact(&wide, &draws, "bad", "0.1.0").is_err());

        // Summary-only artifacts have no draws to compare against.
        let mut summary = valid_meta();
        summary.draws_encoding = "none".into();
        summary.mean = vec![5.0];
        encode_posterior_artifact(&summary, &[], "summary", "0.1.0").unwrap();
    }

    #[test]
    fn posterior_meta_only_skips_draws() {
        let meta = CausalPosteriorWire {
            quantities: vec![PosteriorQuantityWire::Effect { name: "ate".into() }],
            n_draws: 8192,
            mean: vec![0.5],
            sd: vec![0.0],
            q025: vec![0.5],
            q975: vec![0.5],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "f64_le_colmajor".into(),
            treatment_contrast: None,
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
