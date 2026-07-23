//! Versioned sectioned artifact container.
//!
//! Layout:
//! ```text
//! magic (8) | container_version (u32 LE) | manifest_len (u32 LE)
//! | canonical CBOR manifest | section payloads...
//! ```
//!
//! Section payloads may be Zstandard-compressed. [`SectionBytes::data`] always
//! holds **logical** (decompressed) bytes after read and before write.
//! BLAKE3 checksums cover the **on-wire** (possibly compressed) bytes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::io::{Read, Write};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::IoError;
use crate::wire::{
    ArtifactKind, FormatVersion, ProvenanceWire, SectionDescriptor, SemanticVersion,
};

/// Magic bytes: `CAUSAL\0\0`.
pub const MAGIC: &[u8; 8] = b"CAUSAL\0\0";

/// Current container format version.
pub const CONTAINER_VERSION: u32 = 1;

/// Stable compression algorithm name on the wire.
pub const COMPRESSION_ZSTD: &str = "zstd";

/// Default Zstd compression level (deterministic for a given payload).
const ZSTD_LEVEL: i32 = 3;

/// Minimum logical size before Auto compression is attempted.
pub const AUTO_COMPRESS_MIN_BYTES: usize = 4096;

/// Keep compressed form only when `compressed_len < logical_len * ratio`.
pub const AUTO_COMPRESS_MAX_RATIO: f64 = 0.95;

/// Scratch size when streaming-skipping unread sections on a pure [`Read`].
const SKIP_SCRATCH: usize = 64 * 1024;

const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_SECTION_BYTES: usize = 512 * 1024 * 1024;

/// Artifact manifest (canonical CBOR).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactManifest {
    /// Format version of this artifact.
    pub format_version: FormatVersion,
    /// Minimum reader version.
    pub minimum_reader_version: FormatVersion,
    /// Artifact kind.
    pub artifact_kind: ArtifactKind,
    /// Producing library version.
    pub library_version: SemanticVersion,
    /// Artifact id.
    pub artifact_id: String,
    /// Section table.
    pub sections: Vec<SectionDescriptor>,
    /// Provenance summary.
    pub provenance: ProvenanceWire,
}

/// One section's payload bytes with checksum verification on read.
///
/// Logical bytes are reference-counted so writers can share Arrow / draw buffers
/// without cloning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionBytes {
    /// Section id matching the manifest.
    pub id: String,
    /// Logical (decompressed) payload.
    pub data: Arc<[u8]>,
}

impl SectionBytes {
    /// Owned section from a byte vector.
    #[must_use]
    pub fn new(id: impl Into<String>, data: impl Into<Arc<[u8]>>) -> Self {
        Self { id: id.into(), data: data.into() }
    }
}

/// Encoded artifact ready for storage/transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedArtifact {
    /// Manifest.
    pub manifest: ArtifactManifest,
    /// Section payloads in manifest order (logical bytes).
    pub sections: Vec<SectionBytes>,
}

/// Section compression policy for [`section_descriptor`] / [`pack_section`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompressPolicy {
    /// Never compress.
    Never,
    /// Compress when large enough and ratio improves.
    Auto,
    /// Always attempt zstd (still store uncompressed if encode fails).
    Always,
}

impl Default for CompressPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

impl EncodedArtifact {
    /// Serialize to the sectioned container format.
    ///
    /// Uncompressed sections are written from the shared logical slice without
    /// an intermediate clone. Compressed sections allocate an on-wire buffer.
    ///
    /// # Errors
    ///
    /// CBOR or IO failures; section/manifest inconsistency.
    pub fn write_to<W: Write>(&self, mut w: W) -> Result<(), IoError> {
        if self.manifest.sections.len() != self.sections.len() {
            return Err(IoError::ManifestMismatch {
                message: "section count != manifest section count",
            });
        }
        // Precompute on-wire for compressed sections; uncompressed use logical borrow.
        let mut compressed_bufs: Vec<Option<Vec<u8>>> = Vec::with_capacity(self.sections.len());
        for (desc, sec) in self.manifest.sections.iter().zip(self.sections.iter()) {
            if desc.id != sec.id {
                return Err(IoError::ManifestMismatch { message: "section id mismatch" });
            }
            let expected_uncomp =
                usize::try_from(desc.uncompressed_size).map_err(|_| IoError::TooLarge)?;
            if sec.data.len() != expected_uncomp {
                return Err(IoError::ManifestMismatch { message: "section logical size mismatch" });
            }
            let on_wire_owned =
                encode_on_wire_owned(sec.data.as_ref(), desc.compression.as_deref())?;
            let on_wire: &[u8] = match &on_wire_owned {
                Some(v) => v.as_slice(),
                None => sec.data.as_ref(),
            };
            let expected_comp =
                usize::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
            if on_wire.len() != expected_comp {
                return Err(IoError::ManifestMismatch {
                    message: "section compressed size mismatch",
                });
            }
            let hash = blake3::hash(on_wire);
            if hash.as_bytes() != &desc.blake3 {
                return Err(IoError::ChecksumMismatch { section: sec.id.clone() });
            }
            compressed_bufs.push(on_wire_owned);
        }

        w.write_all(MAGIC)?;
        w.write_all(&CONTAINER_VERSION.to_le_bytes())?;
        let mut manifest_buf = Vec::new();
        ciborium::into_writer(&self.manifest, &mut manifest_buf)
            .map_err(|e| IoError::Cbor(e.to_string()))?;
        let manifest_len = u32::try_from(manifest_buf.len()).map_err(|_| IoError::TooLarge)?;
        w.write_all(&manifest_len.to_le_bytes())?;
        w.write_all(&manifest_buf)?;
        for (desc, sec, owned) in self
            .manifest
            .sections
            .iter()
            .zip(self.sections.iter())
            .zip(compressed_bufs.iter())
            .map(|((d, s), o)| (d, s, o))
        {
            let on_wire: &[u8] = match owned {
                Some(v) => v.as_slice(),
                None => sec.data.as_ref(),
            };
            let len = u32::try_from(on_wire.len()).map_err(|_| IoError::TooLarge)?;
            debug_assert_eq!(u64::from(len), desc.compressed_size);
            w.write_all(&len.to_le_bytes())?;
            w.write_all(on_wire)?;
        }
        Ok(())
    }

    /// Decode a container, materializing every section into owned logical bytes.
    ///
    /// # Errors
    ///
    /// Bad magic, version, CBOR, length, or checksum.
    pub fn read_from<R: Read>(mut r: R) -> Result<Self, IoError> {
        let (manifest, mut r) = read_header_and_manifest(&mut r)?;
        let mut sections = Vec::with_capacity(manifest.sections.len());
        for desc in &manifest.sections {
            let logical = read_section_logical(&mut r, desc)?;
            sections.push(SectionBytes { id: desc.id.clone(), data: Arc::from(logical) });
        }
        Ok(Self { manifest, sections })
    }

    /// Decode a container, materializing only sections whose ids are in `want`.
    ///
    /// Unselected sections are stream-hashed for BLAKE3 integrity and discarded
    /// (no retained payload). Selected sections appear in manifest order among
    /// those requested; missing wanted ids are omitted from `sections` (callers
    /// that require them must check).
    ///
    /// # Errors
    ///
    /// Bad magic, version, CBOR, length, or checksum.
    pub fn read_selective<R: Read>(mut r: R, want: &HashSet<&str>) -> Result<Self, IoError> {
        let (manifest, mut r) = read_header_and_manifest(&mut r)?;
        let mut sections = Vec::new();
        for desc in &manifest.sections {
            if want.contains(desc.id.as_str()) {
                let logical = read_section_logical(&mut r, desc)?;
                sections.push(SectionBytes { id: desc.id.clone(), data: Arc::from(logical) });
            } else {
                skip_section_verified(&mut r, desc)?;
            }
        }
        Ok(Self { manifest, sections })
    }
}

/// Read magic, container version, and CBOR manifest; leave `r` at the first section.
pub(crate) fn read_header_and_manifest<R: Read>(
    r: &mut R,
) -> Result<(ArtifactManifest, &mut R), IoError> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(IoError::BadMagic);
    }
    let mut ver_buf = [0u8; 4];
    r.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);
    if version != CONTAINER_VERSION {
        return Err(IoError::UnsupportedVersion { version });
    }
    r.read_exact(&mut ver_buf)?;
    let manifest_len =
        usize::try_from(u32::from_le_bytes(ver_buf)).map_err(|_| IoError::TooLarge)?;
    if manifest_len > MAX_MANIFEST_BYTES {
        return Err(IoError::TooLarge);
    }
    let mut manifest_buf = vec![0u8; manifest_len];
    r.read_exact(&mut manifest_buf)?;
    let manifest: ArtifactManifest =
        ciborium::from_reader(manifest_buf.as_slice()).map_err(|e| IoError::Cbor(e.to_string()))?;
    Ok((manifest, r))
}

fn read_section_logical<R: Read>(r: &mut R, desc: &SectionDescriptor) -> Result<Vec<u8>, IoError> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = usize::try_from(u32::from_le_bytes(len_buf)).map_err(|_| IoError::TooLarge)?;
    let expected_comp = usize::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
    if len != expected_comp {
        return Err(IoError::ManifestMismatch { message: "section size mismatch" });
    }
    if len > MAX_SECTION_BYTES {
        return Err(IoError::TooLarge);
    }
    let mut on_wire = vec![0u8; len];
    r.read_exact(&mut on_wire)?;
    let hash = blake3::hash(&on_wire);
    if hash.as_bytes() != &desc.blake3 {
        return Err(IoError::ChecksumMismatch { section: desc.id.clone() });
    }
    let logical = decode_on_wire(&on_wire, desc.compression.as_deref(), &desc.id)?;
    let expected_uncomp = usize::try_from(desc.uncompressed_size).map_err(|_| IoError::TooLarge)?;
    if logical.len() != expected_uncomp {
        return Err(IoError::Decompress {
            section: desc.id.clone(),
            message: format!(
                "logical size {} != uncompressed_size {expected_uncomp}",
                logical.len()
            ),
        });
    }
    Ok(logical)
}

/// Consume an on-wire section, verifying BLAKE3, without retaining the payload.
fn skip_section_verified<R: Read>(r: &mut R, desc: &SectionDescriptor) -> Result<(), IoError> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = usize::try_from(u32::from_le_bytes(len_buf)).map_err(|_| IoError::TooLarge)?;
    let expected_comp = usize::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
    if len != expected_comp {
        return Err(IoError::ManifestMismatch { message: "section size mismatch" });
    }
    if len > MAX_SECTION_BYTES {
        return Err(IoError::TooLarge);
    }
    let mut hasher = blake3::Hasher::new();
    let mut remaining = len;
    let mut scratch = vec![0u8; SKIP_SCRATCH];
    while remaining > 0 {
        let n = remaining.min(SKIP_SCRATCH);
        r.read_exact(&mut scratch[..n])?;
        hasher.update(&scratch[..n]);
        remaining -= n;
    }
    if hasher.finalize().as_bytes() != &desc.blake3 {
        return Err(IoError::ChecksumMismatch { section: desc.id.clone() });
    }
    Ok(())
}

/// Build a section descriptor with [`CompressPolicy::Auto`].
#[must_use]
pub fn section_descriptor(
    id: impl Into<String>,
    content_type: impl Into<String>,
    data: &[u8],
) -> SectionDescriptor {
    section_descriptor_with_policy(id, content_type, data, CompressPolicy::Auto)
}

/// Build a section descriptor under an explicit compression policy.
#[must_use]
pub fn section_descriptor_with_policy(
    id: impl Into<String>,
    content_type: impl Into<String>,
    logical: &[u8],
    policy: CompressPolicy,
) -> SectionDescriptor {
    let (compression, on_wire_owned) = choose_on_wire(logical, policy);
    let on_wire: &[u8] = on_wire_owned.as_deref().unwrap_or(logical);
    let hash = blake3::hash(on_wire);
    let mut blake3_bytes = [0u8; 32];
    blake3_bytes.copy_from_slice(hash.as_bytes());
    SectionDescriptor {
        id: id.into(),
        content_type: content_type.into(),
        encoding_version: 1,
        required: true,
        compression,
        compressed_size: u64::try_from(on_wire.len()).unwrap_or(u64::MAX),
        uncompressed_size: u64::try_from(logical.len()).unwrap_or(u64::MAX),
        blake3: blake3_bytes,
        logical_schema: "cbor.v1".into(),
    }
}

/// Pack logical bytes into a descriptor + logical [`SectionBytes`].
#[must_use]
pub fn pack_section(
    id: impl Into<String>,
    content_type: impl Into<String>,
    logical: Vec<u8>,
    policy: CompressPolicy,
) -> (SectionDescriptor, SectionBytes) {
    pack_section_shared(id, content_type, Arc::from(logical), policy)
}

/// Pack a shared logical buffer without cloning the payload bytes.
#[must_use]
pub fn pack_section_shared(
    id: impl Into<String>,
    content_type: impl Into<String>,
    logical: Arc<[u8]>,
    policy: CompressPolicy,
) -> (SectionDescriptor, SectionBytes) {
    let id = id.into();
    let desc = section_descriptor_with_policy(id.clone(), content_type, logical.as_ref(), policy);
    (desc, SectionBytes { id, data: logical })
}

/// Returns `(compression_tag, Some(owned_on_wire))` when compressed, or
/// `(None, None)` when the logical slice is the on-wire form (Never / Auto miss).
fn choose_on_wire(logical: &[u8], policy: CompressPolicy) -> (Option<String>, Option<Vec<u8>>) {
    match policy {
        CompressPolicy::Never => (None, None),
        CompressPolicy::Always => match try_zstd(logical) {
            Some(c) => (Some(COMPRESSION_ZSTD.into()), Some(c)),
            None => (None, None),
        },
        CompressPolicy::Auto => {
            if logical.len() < AUTO_COMPRESS_MIN_BYTES {
                return (None, None);
            }
            match try_zstd(logical) {
                Some(c) if (c.len() as u128) * 100 < (logical.len() as u128) * 95 => {
                    (Some(COMPRESSION_ZSTD.into()), Some(c))
                }
                _ => (None, None),
            }
        }
    }
}

fn try_zstd(logical: &[u8]) -> Option<Vec<u8>> {
    zstd::encode_all(logical, ZSTD_LEVEL).ok()
}

/// `Ok(None)` means write the logical slice as-is; `Ok(Some(v))` is compressed.
fn encode_on_wire_owned(
    logical: &[u8],
    compression: Option<&str>,
) -> Result<Option<Vec<u8>>, IoError> {
    match compression {
        None => Ok(None),
        Some(COMPRESSION_ZSTD) => zstd::encode_all(logical, ZSTD_LEVEL)
            .map(Some)
            .map_err(|e| IoError::Io(format!("zstd encode: {e}"))),
        Some(other) => Err(IoError::UnsupportedCompression { algo: other.into() }),
    }
}

pub(crate) fn decode_on_wire(
    on_wire: &[u8],
    compression: Option<&str>,
    section: &str,
) -> Result<Vec<u8>, IoError> {
    match compression {
        None => Ok(on_wire.to_vec()),
        Some(COMPRESSION_ZSTD) => zstd::decode_all(on_wire)
            .map_err(|e| IoError::Decompress { section: section.into(), message: e.to_string() }),
        Some(other) => Err(IoError::UnsupportedCompression { algo: other.into() }),
    }
}

/// Decode without copying when already uncompressed — returns owned only when needed.
pub(crate) fn decode_on_wire_arc(
    on_wire: &[u8],
    compression: Option<&str>,
    section: &str,
) -> Result<(Arc<[u8]>, bool), IoError> {
    match compression {
        None => Ok((Arc::from(on_wire.to_vec()), false)),
        Some(COMPRESSION_ZSTD) => {
            let v = zstd::decode_all(on_wire).map_err(|e| IoError::Decompress {
                section: section.into(),
                message: e.to_string(),
            })?;
            Ok((Arc::from(v), true))
        }
        Some(other) => Err(IoError::UnsupportedCompression { algo: other.into() }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::VERSION;

    use super::*;
    use crate::convert::to_cbor;

    fn tiny_artifact(sections: Vec<(SectionDescriptor, SectionBytes)>) -> EncodedArtifact {
        let (descs, secs): (Vec<_>, Vec<_>) = sections.into_iter().unzip();
        EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: FormatVersion { major: 0, minor: 1 },
                minimum_reader_version: FormatVersion { major: 0, minor: 1 },
                artifact_kind: ArtifactKind::Other("test".into()),
                library_version: SemanticVersion::from_crate_version(VERSION)
                    .expect("CARGO_PKG_VERSION"),
                artifact_id: "compress-test".into(),
                sections: descs,
                provenance: ProvenanceWire { note: "test".into() },
            },
            sections: secs,
        }
    }

    #[test]
    fn uncompressed_round_trip() {
        let payload = b"hello".to_vec();
        let (desc, sec) = pack_section(
            "note",
            "application/octet-stream",
            payload.clone(),
            CompressPolicy::Never,
        );
        assert!(desc.compression.is_none());
        let art = tiny_artifact(vec![(desc, sec)]);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        assert_eq!(&*decoded.sections[0].data, payload.as_slice());
    }

    #[test]
    fn zstd_round_trip_large_payload() {
        let payload = vec![0xABu8; 16 * 1024];
        let (desc, sec) = pack_section(
            "blob",
            "application/octet-stream",
            payload.clone(),
            CompressPolicy::Always,
        );
        assert_eq!(desc.compression.as_deref(), Some(COMPRESSION_ZSTD));
        assert!(desc.compressed_size < desc.uncompressed_size);
        let art = tiny_artifact(vec![(desc, sec)]);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let decoded = EncodedArtifact::read_from(buf.as_slice()).unwrap();
        assert_eq!(&*decoded.sections[0].data, payload.as_slice());
        assert_eq!(decoded.manifest.sections[0].compression.as_deref(), Some(COMPRESSION_ZSTD));
    }

    #[test]
    fn auto_keeps_tiny_uncompressed() {
        let payload = b"tiny".to_vec();
        let desc = section_descriptor_with_policy(
            "t",
            "application/octet-stream",
            &payload,
            CompressPolicy::Auto,
        );
        assert!(desc.compression.is_none());
    }

    #[test]
    fn auto_compresses_highly_redundant() {
        let payload = vec![0u8; 16 * 1024];
        let desc = section_descriptor_with_policy(
            "t",
            "application/octet-stream",
            &payload,
            CompressPolicy::Auto,
        );
        assert_eq!(desc.compression.as_deref(), Some(COMPRESSION_ZSTD));
    }

    #[test]
    fn rejects_bad_checksum_on_compressed() {
        let payload = vec![1u8; 8 * 1024];
        let (mut desc, sec) =
            pack_section("blob", "application/octet-stream", payload, CompressPolicy::Always);
        desc.blake3[0] ^= 0xff;
        let art = tiny_artifact(vec![(desc, sec)]);
        let err = art.write_to(Vec::new()).unwrap_err();
        assert!(matches!(err, IoError::ChecksumMismatch { .. }));
    }

    #[test]
    fn rejects_unknown_compression_on_read() {
        let payload = to_cbor(&"x").unwrap();
        let mut desc = section_descriptor_with_policy(
            "note",
            "application/cbor",
            &payload,
            CompressPolicy::Never,
        );
        desc.compression = Some("lz4".into());
        let art = tiny_artifact(vec![(desc, SectionBytes::new("note", payload))]);
        let err = art.write_to(Vec::new()).unwrap_err();
        assert!(matches!(err, IoError::UnsupportedCompression { .. }));
    }

    #[test]
    fn shared_arc_write_does_not_clone_logical() {
        let shared: Arc<[u8]> = Arc::from(vec![7u8; 1024]);
        let before = Arc::strong_count(&shared);
        let (desc, sec) = pack_section_shared(
            "a",
            "application/octet-stream",
            Arc::clone(&shared),
            CompressPolicy::Never,
        );
        let (desc2, sec2) = pack_section_shared(
            "b",
            "application/octet-stream",
            Arc::clone(&shared),
            CompressPolicy::Never,
        );
        assert!(Arc::strong_count(&shared) >= before + 2);
        let art = tiny_artifact(vec![(desc, sec), (desc2, sec2)]);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        // Logical buffers still shared after write (Never path does not uniquify).
        assert!(Arc::ptr_eq(&art.sections[0].data, &art.sections[1].data));
        assert!(Arc::ptr_eq(&art.sections[0].data, &shared));
    }

    #[test]
    fn read_selective_skips_unread_payload() {
        let meta = b"meta".to_vec();
        let blob = vec![0xCDu8; 32 * 1024];
        let (d0, s0) =
            pack_section("meta", "application/octet-stream", meta.clone(), CompressPolicy::Never);
        let (d1, s1) =
            pack_section("blob", "application/octet-stream", blob, CompressPolicy::Never);
        let art = tiny_artifact(vec![(d0, s0), (d1, s1)]);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let mut want = HashSet::new();
        want.insert("meta");
        let partial = EncodedArtifact::read_selective(buf.as_slice(), &want).unwrap();
        assert_eq!(partial.sections.len(), 1);
        assert_eq!(partial.sections[0].id, "meta");
        assert_eq!(&*partial.sections[0].data, meta.as_slice());
        assert_eq!(partial.manifest.sections.len(), 2);
    }
}
