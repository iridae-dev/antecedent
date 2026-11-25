//! Wire types and encode/decode for columnar posterior artifacts .
//!
//! Draws live in an Arrow-IPC (or raw f64 LE) numerical section; metadata is CBOR.
//! Internal Rust structs are never serialized directly.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};
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
    /// How the columns the fit read were prepared, when the estimator recorded a receipt.
    /// A physical fact about the run, so it takes no part in any identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation: Option<PosteriorPreparationWire>,
}

/// Preparation receipt summary recorded on a posterior.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PosteriorPreparationWire {
    /// Stable name of the preparation backend.
    pub backend: String,
    /// Stable name of the end-to-end route bound.
    pub route_bound: String,
    /// Rows the estimator read; absent when it was handed a scan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_admitted: Option<u64>,
    /// Lower-case hex digest of the preparation plan.
    pub receipt_digest: String,
    /// Payload bytes the preparation admitted; absent when the writer recorded none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admitted_bytes: Option<u64>,
    /// Byte budget the payload was compared with; absent when none was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_bytes: Option<u64>,
    /// What the estimator was given (`finite`, `materialized_from_scan`, `scan`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
}

#[allow(clippy::trivially_copy_pass_by_ref, reason = "serde's skip_serializing_if passes `&T`")]
fn is_zero_mass(mass: &f64) -> bool {
    *mass == 0.0
}

/// Validate a posterior's summary metadata and, when supplied, its stored draws.
///
/// # Errors
/// Returns an error when the metadata or draws violate the posterior wire contract.
pub fn validate_posterior_meta(
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
        validate_draw_count(meta, draws.len())?;
        if !draws.is_empty() {
            let mut scratch = Vec::with_capacity(meta.n_draws as usize);
            for (quantity, column) in draws.chunks_exact(meta.n_draws as usize).enumerate() {
                scratch.clear();
                scratch.extend_from_slice(column);
                validate_draw_column(meta, quantity, &mut scratch)?;
            }
        }
    }
    Ok(())
}

fn validate_draw_count(meta: &CausalPosteriorWire, count: usize) -> Result<(), IoError> {
    let expected = if meta.draws_encoding == "none" {
        0
    } else {
        usize::try_from(meta.n_draws)
            .ok()
            .and_then(|count| count.checked_mul(meta.quantities.len()))
            .ok_or(IoError::TooLarge)?
    };
    if count != expected {
        return Err(IoError::Convert(format!(
            "posterior draws length {count} != expected {expected}"
        )));
    }
    Ok(())
}

/// Verify one draw column with the summary kernels used by the producer, with a relative
/// tolerance of 1e-9. The column is scratch space for sorting the quantiles.
fn validate_draw_column(
    meta: &CausalPosteriorWire,
    quantity: usize,
    column: &mut [f64],
) -> Result<(), IoError> {
    use antecedent_kernels::{PosteriorReduceOp, quantile_type7_sorted, reduce_posterior_draws};

    if column.iter().any(|value| !value.is_finite()) {
        return Err(IoError::Convert("posterior draws must be finite".into()));
    }
    const SUMMARY: ReplayTolerance =
        ReplayTolerance::new(1e-9, ReplayScale::LargerOrOne, ReplayNonFinite::Refuse);
    let policy = antecedent_core::KernelPolicy::default_policy();
    let mean = reduce_posterior_draws(column, PosteriorReduceOp::Mean, &policy).unwrap_or(f64::NAN);
    let sd = reduce_posterior_draws(column, PosteriorReduceOp::Std, &policy).unwrap_or(f64::NAN);
    column.sort_by(f64::total_cmp);
    let agrees = |lower, upper| {
        SUMMARY.close(meta.q025[quantity], lower) && SUMMARY.close(meta.q975[quantity], upper)
    };
    // A Laplace posterior can publish the exchangeable-rank interval below its measured
    // draw count. Both accepted intervals are computed from the same stored draws.
    let interval =
        agrees(quantile_type7_sorted(column, 0.025), quantile_type7_sorted(column, 0.975)) || {
            let (lower, upper) = antecedent_stats::equal_tail_interval_sorted(
                column,
                0.95,
                antecedent_stats::QuantileRule::ExchangeableRank,
            );
            agrees(lower, upper)
        };
    if !interval
        || !SUMMARY.close(meta.mean[quantity], mean)
        || !SUMMARY.close(meta.sd[quantity], sd)
    {
        return Err(IoError::Convert(format!(
            "posterior summaries for quantity {quantity} disagree with the embedded draws"
        )));
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
    let (meta, draw_bytes) = posterior_sections(artifact)?;
    let draws = crate::artifact_codec::f64s_from_le_bytes(draw_bytes);
    validate_posterior_meta(&meta, Some(&draws))?;
    Ok((meta, draws))
}

fn posterior_sections(artifact: &EncodedArtifact) -> Result<(CausalPosteriorWire, &[u8]), IoError> {
    require_posterior_kind(&artifact.manifest.artifact_kind)?;
    let section = |id| {
        artifact
            .sections
            .iter()
            .find(|section| section.id == id)
            .ok_or_else(|| IoError::Convert(format!("missing {id}")))
    };
    let meta = decode_posterior_meta(&section("posterior.meta")?.data)?;
    let draws = section("posterior.draws")?.data.as_ref();
    if draws.len() % size_of::<f64>() != 0 {
        return Err(IoError::Convert("posterior.draws not multiple of 8".into()));
    }
    Ok((meta, draws))
}

/// Validate the body and summaries without materializing a float64 draw matrix.
pub(crate) fn validate_causal_posterior_bytes(bytes: &[u8]) -> Result<(), IoError> {
    let artifact = crate::migrate::read_and_migrate(bytes)?;
    let (meta, draws) = posterior_sections(&artifact)?;
    validate_draw_count(&meta, draws.len() / size_of::<f64>())?;
    if !draws.is_empty() {
        let rows = meta.n_draws as usize;
        let mut scratch = Vec::with_capacity(rows);
        for (quantity, column) in draws.chunks_exact(rows * size_of::<f64>()).enumerate() {
            scratch.clear();
            scratch.extend(column.as_chunks::<8>().0.iter().copied().map(f64::from_le_bytes));
            validate_draw_column(&meta, quantity, &mut scratch)?;
        }
    }
    Ok(())
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
    load_posterior_meta(&mut reader)
}

fn load_posterior_meta<R: std::io::Read + std::io::Seek>(
    reader: &mut crate::reader::ArtifactReader<R>,
) -> Result<CausalPosteriorWire, IoError> {
    require_posterior_kind(&reader.manifest().artifact_kind)?;
    let access = reader.load_section("posterior.meta")?;
    decode_posterior_meta(access.as_bytes())
}

fn check_section_extent(
    index: &[crate::reader::SectionIndexEntry],
    len: u64,
) -> Result<(), IoError> {
    if let Some(section) = index.last()
        && section
            .file_offset
            .checked_add(u64::from(section.on_wire_len))
            .is_none_or(|end| end > len)
    {
        return Err(IoError::io("artifact ends inside a declared section"));
    }
    Ok(())
}

fn require_posterior_kind(kind: &ArtifactKind) -> Result<(), IoError> {
    if *kind != ArtifactKind::CausalPosterior {
        return Err(IoError::Convert(format!("expected CausalPosterior, got {kind:?}")));
    }
    Ok(())
}

fn decode_posterior_meta(bytes: &[u8]) -> Result<CausalPosteriorWire, IoError> {
    let meta = from_cbor(bytes)?;
    validate_posterior_meta(&meta, None)?;
    Ok(meta)
}

/// Decode only posterior metadata from a file path, seeking past the draws.
///
/// # Errors
///
/// Same as [`decode_posterior_meta_from_seek`].
pub fn decode_posterior_meta_from_path(
    path: impl AsRef<std::path::Path>,
) -> Result<CausalPosteriorWire, IoError> {
    let file = std::fs::File::open(path).map_err(IoError::from)?;
    let len = file.metadata().map_err(IoError::from)?.len();
    let mut reader = crate::reader::ArtifactReader::open_seek(file)?;
    check_section_extent(reader.index(), len)?;
    load_posterior_meta(&mut reader)
}

/// As [`decode_posterior_meta_from_seek`], admitting only the artifact the consumer retained.
///
/// The bytes from the current position of `r` are hashed as they are read. Only the metadata
/// is materialized, and it is returned after the digest matches `retained`.
///
/// # Errors
///
/// As [`decode_posterior_meta_from_seek`], and `external_binding_mismatch` when the bytes are
/// not the retained artifact.
pub fn decode_posterior_meta_from_seek_expecting<R: std::io::Read + std::io::Seek>(
    r: R,
    retained: &crate::retained_artifact::RetainedArtifactIdentity,
) -> Result<CausalPosteriorWire, IoError> {
    use std::io::Read;

    use crate::container::ContainerLimits;
    use crate::retained_artifact::{ARTIFACT_IDENTITY_DOMAIN, bind_retained_digest};

    struct HashingReader<R> {
        inner: R,
        digest: blake3::Hasher,
    }

    impl<R: Read> Read for HashingReader<R> {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buffer)?;
            self.digest.update(&buffer[..n]);
            Ok(n)
        }
    }

    let limits = ContainerLimits::default();
    let cap = (limits.max_manifest_bytes as u64)
        .saturating_add(limits.max_total_uncompressed_bytes)
        .saturating_add(1 << 20);
    let mut reader = HashingReader {
        inner: r.take(cap.saturating_add(1)),
        digest: blake3::Hasher::new_derive_key(ARTIFACT_IDENTITY_DOMAIN),
    };
    let metadata = read_selective_posterior_meta(&mut reader);
    // Bind the entire presented input, including any bytes after the last declared section.
    std::io::copy(&mut reader, &mut std::io::sink()).map_err(IoError::from)?;
    if reader.inner.limit() == 0 {
        return Err(IoError::TooLarge);
    }
    bind_retained_digest(
        antecedent_core::SemanticDigest::from_bytes(*reader.digest.finalize().as_bytes()),
        retained,
    )?;
    metadata
}

/// As [`decode_posterior_meta_from_path`], admitting only the artifact the consumer retained.
///
/// The whole file is hashed to bind it to `retained`, with only its metadata payload retained.
///
/// # Errors
///
/// As [`decode_posterior_meta_from_path`], and `external_binding_mismatch` when the bytes are
/// not the retained artifact.
pub fn decode_posterior_meta_from_path_expecting(
    path: impl AsRef<std::path::Path>,
    retained: &crate::retained_artifact::RetainedArtifactIdentity,
) -> Result<CausalPosteriorWire, IoError> {
    let file = std::fs::File::open(path).map_err(IoError::from)?;
    decode_posterior_meta_from_seek_expecting(file, retained)
}

/// Read and validate posterior metadata from retained bytes without materializing the draws.
///
/// # Errors
/// Invalid container, unsupported format, invalid metadata checksum, or invalid metadata.
pub fn decode_posterior_meta_from_bytes(bytes: &[u8]) -> Result<CausalPosteriorWire, IoError> {
    let mut reader = crate::reader::ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    check_section_extent(reader.index(), bytes.len() as u64)?;
    let manifest = reader.manifest();
    let version = manifest.format_version;
    let minimum = manifest.minimum_reader_version;
    if !crate::migrate::is_supported_source(version) {
        return Err(IoError::UnsupportedFormat { major: version.major, minor: version.minor });
    }
    let stable = crate::migrate::STABLE_FORMAT;
    if (minimum.major, minimum.minor) > (stable.major, stable.minor) {
        return Err(IoError::UnsupportedFormat { major: minimum.major, minor: minimum.minor });
    }
    require_posterior_kind(&manifest.artifact_kind)?;
    let draws = manifest
        .sections
        .iter()
        .find(|section| section.id == "posterior.draws")
        .ok_or_else(|| IoError::Convert("missing posterior.draws".into()))?
        .uncompressed_size;
    let meta = decode_posterior_meta(reader.load_section("posterior.meta")?.as_bytes())?;
    let draws = usize::try_from(draws).map_err(|_| IoError::TooLarge)?;
    if draws % size_of::<f64>() != 0 {
        return Err(IoError::Convert("posterior.draws not multiple of 8".into()));
    }
    validate_draw_count(&meta, draws / size_of::<f64>())?;
    Ok(meta)
}

fn read_selective_posterior_meta(r: impl std::io::Read) -> Result<CausalPosteriorWire, IoError> {
    let mut artifact =
        EncodedArtifact::read_selective(r, &std::collections::HashSet::from(["posterior.meta"]))?;
    // Migration only needs the section we materialized. Posterior metadata has no v0.1
    // schema section to upgrade.
    artifact.manifest.sections.retain(|section| section.id == "posterior.meta");
    let artifact = crate::migrate::migrate_artifact(artifact)?;
    require_posterior_kind(&artifact.manifest.artifact_kind)?;
    let section = artifact
        .sections
        .iter()
        .find(|section| section.id == "posterior.meta")
        .ok_or_else(|| IoError::Convert("missing posterior.meta".into()))?;
    decode_posterior_meta(&section.data)
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

/// As [`decode_causal_posterior_bytes`], admitting only the artifact the consumer retained.
///
/// The bytes bind to `retained`
/// ([`crate::retained_artifact::bind_retained_artifact`]) before they are read.
///
/// # Errors
///
/// As [`decode_causal_posterior_bytes`], and `external_binding_mismatch` when the bytes are not the
/// retained artifact.
pub fn decode_causal_posterior_bytes_expecting(
    bytes: &[u8],
    retained: &crate::retained_artifact::RetainedArtifactIdentity,
) -> Result<(CausalPosteriorWire, Vec<f64>), IoError> {
    crate::retained_artifact::load_expecting(bytes, retained, decode_causal_posterior_bytes)
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
            // Type-7 quantiles of the draws (0.9, 1.0, 1.1): 0.9 + 0.05 * 0.1 and 1.0 + 0.95 * 0.1.
            q025: vec![0.905],
            q975: vec![1.095],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            subsampled_out_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 10.0,
            draws_encoding: "f64_le_colmajor".into(),
            treatment_contrast: None,
            preparation: None,
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
        // Draws (-1, 0, 1): mean 0, sample SD 1, type-7 quantiles -0.95 / 0.95.
        let draws = vec![-1.0, 0.0, 1.0];
        let mut consistent = valid_meta();
        consistent.mean = vec![0.0];
        consistent.sd = vec![1.0];
        consistent.q025 = vec![-0.95];
        consistent.q975 = vec![0.95];
        encode_posterior_artifact(&consistent, &draws, "ok", "0.1.0").unwrap();

        // Summary reader would see 5 +- 0.5 while a draw reader sees 0 +- 1.
        let mut shifted = consistent.clone();
        shifted.mean = vec![5.0];
        shifted.q025 = vec![4.5];
        shifted.q975 = vec![5.5];
        let error =
            encode_posterior_artifact(&shifted, &draws, "bad", "0.1.0").unwrap_err().to_string();
        assert!(error.contains("disagree with the embedded draws"), "{error}");

        // Quantiles, mean and SD are all exact functions of the draws: no sampling slack.
        let mut narrowed = consistent.clone();
        narrowed.q975 = vec![0.5];
        assert!(encode_posterior_artifact(&narrowed, &draws, "bad", "0.1.0").is_err());
        let mut near_mean = consistent.clone();
        near_mean.mean = vec![0.1];
        assert!(encode_posterior_artifact(&near_mean, &draws, "bad", "0.1.0").is_err());
        let mut near_sd = consistent;
        near_sd.sd = vec![1.2];
        assert!(encode_posterior_artifact(&near_sd, &draws, "bad", "0.1.0").is_err());

        // Summary-only artifacts have no draws to compare against.
        let mut summary = valid_meta();
        summary.draws_encoding = "none".into();
        summary.mean = vec![5.0];
        encode_posterior_artifact(&summary, &[], "summary", "0.1.0").unwrap();
    }

    #[test]
    fn posterior_meta_expecting_binds_the_retained_bytes() {
        use crate::retained_artifact::{RETAINED_ARTIFACT_REFUSAL, RetainedArtifactIdentity};
        let art =
            encode_posterior_artifact(&valid_meta(), &[0.9, 1.0, 1.1], "retained-meta", "0.1.0")
                .unwrap();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let retained = RetainedArtifactIdentity::of_bytes(&buf);
        let other = RetainedArtifactIdentity::of_bytes(b"another artifact");
        let plain = decode_posterior_meta_from_seek(std::io::Cursor::new(buf.clone())).unwrap();
        let bound =
            decode_posterior_meta_from_seek_expecting(std::io::Cursor::new(buf.clone()), &retained)
                .unwrap();
        assert_eq!(bound, plain);
        let refused =
            decode_posterior_meta_from_seek_expecting(std::io::Cursor::new(buf.clone()), &other)
                .unwrap_err();
        assert!(refused.to_string().contains(RETAINED_ARTIFACT_REFUSAL), "{refused}");

        let path = std::env::temp_dir()
            .join(format!("antecedent-retained-posterior-{}.bin", std::process::id()));
        std::fs::write(&path, &buf).unwrap();
        assert_eq!(decode_posterior_meta_from_path_expecting(&path, &retained).unwrap(), plain);
        let refused = decode_posterior_meta_from_path_expecting(&path, &other).unwrap_err();
        assert!(refused.to_string().contains(RETAINED_ARTIFACT_REFUSAL), "{refused}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn posterior_meta_only_skips_draws() {
        use std::io::{Cursor, Read, Seek, SeekFrom};

        struct NoDrawReads<'a> {
            inner: Cursor<&'a [u8]>,
            draws_start: u64,
        }
        impl Read for NoDrawReads<'_> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                assert!(self.inner.position() + buf.len() as u64 <= self.draws_start);
                self.inner.read(buf)
            }
        }
        impl Seek for NoDrawReads<'_> {
            fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
                self.inner.seek(pos)
            }
        }

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
            preparation: None,
        };
        let draws = vec![0.5f64; 8192];
        let art = encode_posterior_artifact(&meta, &draws, "meta-only", "0.1.0").unwrap();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let mut reader = crate::reader::ArtifactReader::open_seek(Cursor::new(&buf)).unwrap();
        let draws_start = reader.index()[1].file_offset;
        let guarded = NoDrawReads { inner: Cursor::new(&buf), draws_start };
        let got = decode_posterior_meta_from_seek(guarded).unwrap();
        assert_eq!(got.n_draws, 8192);
        assert_eq!(got.backend_id, "laplace");
        assert_eq!(decode_posterior_meta_from_bytes(&buf).unwrap(), got);
        let (full_meta, full_draws) = decode_causal_posterior_bytes(&buf).unwrap();
        assert_eq!(full_meta, got);
        assert_eq!(draws, full_draws);
        validate_causal_posterior_bytes(&buf).unwrap();
        let path = std::env::temp_dir()
            .join(format!("antecedent-posterior-meta-only-{}.bin", std::process::id()));
        std::fs::write(&path, &buf).unwrap();
        assert_eq!(decode_posterior_meta_from_path(&path).unwrap(), got);
        std::fs::write(&path, &buf[..buf.len() - 1]).unwrap();
        assert!(decode_posterior_meta_from_path(&path).is_err());
        std::fs::remove_file(path).unwrap();

        // Zero-copy accounting (DESIGN rule 22): a meta-only read must not pay
        // for the 64 KiB draws section — bytes_loaded stays below the draws
        // payload and the draws section remains skipped.
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

#[cfg(test)]
mod retained_tests {
    use super::*;
    use crate::retained_artifact::{RETAINED_ARTIFACT_REFUSAL, RetainedArtifactIdentity};

    #[test]
    fn posterior_expecting_binds_the_retained_bytes() {
        let bytes: &[u8] = b"not an artifact";
        let retained = RetainedArtifactIdentity::of_bytes(bytes);
        let other = RetainedArtifactIdentity::of_bytes(b"another artifact");
        // decode_causal_posterior_bytes_expecting
        assert_eq!(
            format!("{:?}", decode_causal_posterior_bytes_expecting(bytes, &retained).err()),
            format!("{:?}", decode_causal_posterior_bytes(bytes).err()),
        );
        let refused = format!("{:?}", decode_causal_posterior_bytes_expecting(bytes, &other).err());
        assert!(refused.contains(RETAINED_ARTIFACT_REFUSAL), "{refused}");
    }
}
