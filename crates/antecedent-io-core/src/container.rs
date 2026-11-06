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
//! The streamed operations ([`describe_streaming`], [`write_streaming`],
//! [`read_streaming`]) never hold a section's on-wire or logical bytes whole. They
//! lease a fixed working set from the context's admission pool before the first
//! byte moves, pass chunks through zstd, a BLAKE3 hasher and the caller's sink,
//! and check cancellation between chunks. A section's descriptor is a pass over
//! the source that keeps only the on-wire length and hash; the write pass
//! compresses again into the output and fails if it does not reproduce them.
//! A seekable read hashes a section before it decodes it, so a sink sees only
//! bytes whose on-wire checksum has verified.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod streaming;

pub use streaming::{
    DEFAULT_STREAM_CHUNK_BYTES, DomainDigest, LogicalSource, STREAM_DECODER_WINDOW_LOG_MAX,
    StreamLease, StreamReceipt, StreamWork, describe_streaming, read_streaming,
    stream_decoder_workspace_bytes, stream_encoder_workspace_bytes, write_streaming,
    zstd_compress_chunked, zstd_decompress_chunked,
};

use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;
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
/// Largest single section (on-wire or declared uncompressed) a reader accepts.
pub(crate) const MAX_SECTION_BYTES: usize = 512 * 1024 * 1024;
/// Largest sum of declared `uncompressed_size` over a manifest's sections a
/// reader accepts (four maximal sections). Checked before any payload is
/// read, so a small file cannot force a huge total allocation.
pub(crate) const MAX_TOTAL_UNCOMPRESSED_BYTES: u64 = 4 * MAX_SECTION_BYTES as u64;

/// Bounds a reader applies to an untrusted container before it allocates for it. The
/// defaults are the crate's conservative limits; a consumer that legitimately handles larger
/// artifacts (or wants tighter ones) passes its own to the `*_with_limits` readers. They stay
/// refusals ([`IoError::TooLarge`]), never truncations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerLimits {
    /// Largest manifest, in bytes.
    pub max_manifest_bytes: usize,
    /// Largest single section, on the wire or declared uncompressed, in bytes.
    pub max_section_bytes: usize,
    /// Largest sum of declared uncompressed section sizes, checked before any payload is
    /// read so a small file cannot force a huge total allocation.
    pub max_total_uncompressed_bytes: u64,
}

impl Default for ContainerLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: MAX_MANIFEST_BYTES,
            max_section_bytes: MAX_SECTION_BYTES,
            max_total_uncompressed_bytes: MAX_TOTAL_UNCOMPRESSED_BYTES,
        }
    }
}

impl ContainerLimits {
    /// Limits with sections of at most `max_section_bytes` and a total of four maximal
    /// sections (the default ratio), keeping the default manifest bound.
    #[must_use]
    pub const fn with_max_section_bytes(max_section_bytes: usize) -> Self {
        Self {
            max_manifest_bytes: MAX_MANIFEST_BYTES,
            max_section_bytes,
            max_total_uncompressed_bytes: 4 * max_section_bytes as u64,
        }
    }
}

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

/// First line of a development stamp in [`ProvenanceWire::note`].
const DEVELOPMENT_BUILD_LINE: &str = "build=development";
/// Prefix of each recorded would-refuse line of a development stamp.
const DEVELOPMENT_ENTRY_PREFIX: &str = "would_refuse=";

impl ArtifactManifest {
    /// Whether this manifest was sealed by a development build.
    #[must_use]
    pub fn is_development_build(&self) -> bool {
        self.provenance.note.lines().any(|line| line == DEVELOPMENT_BUILD_LINE)
    }

    /// The would-refuse entries a development build sealed into this manifest, as text lines.
    #[must_use]
    pub fn development_entries(&self) -> Vec<&str> {
        self.provenance
            .note
            .lines()
            .filter_map(|line| line.strip_prefix(DEVELOPMENT_ENTRY_PREFIX))
            .collect()
    }

    /// Stamp this manifest as sealed by a development build, with what that build ran
    /// although a release build would have refused it. The stamp lives in the provenance note,
    /// so it is part of the manifest every reader verifies. A release build leaves the
    /// manifest unchanged.
    pub fn stamp_development(&mut self, entries: &[antecedent_core::WouldRefuse]) {
        #[cfg(feature = "development")]
        {
            if !self.is_development_build() {
                self.provenance.note.push('\n');
                self.provenance.note.push_str(DEVELOPMENT_BUILD_LINE);
            }
            for entry in entries {
                let line = format!("{DEVELOPMENT_ENTRY_PREFIX}{entry}").replace('\n', " ");
                if !self.provenance.note.lines().any(|existing| existing == line) {
                    self.provenance.note.push('\n');
                    self.provenance.note.push_str(&line);
                }
            }
        }
        #[cfg(not(feature = "development"))]
        {
            let _ = entries;
        }
    }

    /// Refuse a manifest a development build sealed, in a release build.
    ///
    /// # Errors
    ///
    /// [`IoError::Refused`] with the reason code `development_artifact` in a release build
    /// when the manifest carries a development stamp.
    pub fn refuse_development(&self) -> Result<(), IoError> {
        if antecedent_core::BUILD_FLAVOR == "release" && self.is_development_build() {
            return Err(IoError::Refused {
                code: antecedent_core::reason_code!("development_artifact"),
                message: "this artifact was sealed by a development build and carries results a \
                          release build would have refused; a release build does not consume it"
                    .to_owned(),
            });
        }
        Ok(())
    }

    /// This manifest as the container writes it: a development build stamps it, a release
    /// build refuses to write a manifest that already carries a development stamp.
    fn sealed(&self) -> Result<std::borrow::Cow<'_, Self>, IoError> {
        self.refuse_development()?;
        if cfg!(feature = "development") && !self.is_development_build() {
            let mut stamped = self.clone();
            stamped.stamp_development(&[]);
            return Ok(std::borrow::Cow::Owned(stamped));
        }
        Ok(std::borrow::Cow::Borrowed(self))
    }
}

/// One section's payload bytes with checksum verification on read.
///
/// Logical bytes are reference-counted so writers can share Arrow / draw buffers
/// without cloning.
#[derive(Clone, Debug)]
pub struct SectionBytes {
    /// Section id matching the manifest.
    pub id: String,
    /// Logical (decompressed) payload.
    pub data: Arc<[u8]>,
    /// Pack-time zstd on-wire bytes (the exact buffer the descriptor hashed).
    /// [`EncodedArtifact::write_to`] streams this instead of re-encoding;
    /// `None` for uncompressed sections or sections built without packing.
    packed_on_wire: Option<Arc<Vec<u8>>>,
}

// Manual impl: the on-wire cache is derived from `data` and must not affect
// section equality.
impl PartialEq for SectionBytes {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.data == other.data
    }
}

impl Eq for SectionBytes {}

impl SectionBytes {
    /// Owned section from a byte vector.
    #[must_use]
    pub fn new(id: impl Into<String>, data: impl Into<Arc<[u8]>>) -> Self {
        Self { id: id.into(), data: data.into(), packed_on_wire: None }
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

/// One section validated for writing: the logical slice and, when compressed,
/// the on-wire buffer that replaces it.
struct PreparedSection<'a> {
    logical: &'a [u8],
    encoded: Option<Arc<Vec<u8>>>,
    compressed_size: u64,
}

impl PreparedSection<'_> {
    fn on_wire(&self) -> &[u8] {
        match &self.encoded {
            Some(v) => v.as_slice(),
            None => self.logical,
        }
    }
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
        let prepared = self.prepare_on_wire()?;
        w.write_all(MAGIC)?;
        w.write_all(&CONTAINER_VERSION.to_le_bytes())?;
        let mut manifest_buf = Vec::new();
        ciborium::into_writer(&*self.manifest.sealed()?, &mut manifest_buf)
            .map_err(IoError::cbor_encode)?;
        let manifest_len = u32::try_from(manifest_buf.len()).map_err(|_| IoError::TooLarge)?;
        w.write_all(&manifest_len.to_le_bytes())?;
        w.write_all(&manifest_buf)?;
        for section in &prepared {
            let on_wire = section.on_wire();
            let len = u32::try_from(on_wire.len()).map_err(|_| IoError::TooLarge)?;
            debug_assert_eq!(u64::from(len), section.compressed_size);
            w.write_all(&len.to_le_bytes())?;
            w.write_all(on_wire)?;
        }
        Ok(())
    }

    /// Validate every section against its descriptor and resolve the buffer that
    /// goes on the wire.
    ///
    /// Uncompressed sections borrow the shared logical slice; sections packed via
    /// `pack_section` / `pack_section_shared` carry the pack-time zstd buffer, so
    /// no section is zstd-encoded twice. The size and BLAKE3 guards validate
    /// whichever buffer is streamed.
    fn prepare_on_wire(&self) -> Result<Vec<PreparedSection<'_>>, IoError> {
        if self.manifest.sections.len() != self.sections.len() {
            return Err(IoError::ManifestMismatch {
                message: "section count != manifest section count",
            });
        }
        let mut prepared = Vec::with_capacity(self.sections.len());
        for (desc, sec) in self.manifest.sections.iter().zip(self.sections.iter()) {
            if desc.id != sec.id {
                return Err(IoError::ManifestMismatch { message: "section id mismatch" });
            }
            let expected_uncomp =
                usize::try_from(desc.uncompressed_size).map_err(|_| IoError::TooLarge)?;
            if sec.data.len() != expected_uncomp {
                return Err(IoError::ManifestMismatch { message: "section logical size mismatch" });
            }
            let encoded: Option<Arc<Vec<u8>>> =
                match (desc.compression.as_deref(), &sec.packed_on_wire) {
                    (Some(COMPRESSION_ZSTD), Some(cached)) => Some(Arc::clone(cached)),
                    (compression, _) => {
                        encode_on_wire_owned(sec.data.as_ref(), compression)?.map(Arc::new)
                    }
                };
            let section = PreparedSection {
                logical: sec.data.as_ref(),
                encoded,
                compressed_size: desc.compressed_size,
            };
            let expected_comp =
                usize::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
            let on_wire = section.on_wire();
            if on_wire.len() != expected_comp {
                return Err(IoError::ManifestMismatch {
                    message: "section compressed size mismatch",
                });
            }
            if blake3::hash(on_wire).as_bytes() != &desc.blake3 {
                return Err(IoError::ChecksumMismatch { section: sec.id.clone() });
            }
            prepared.push(section);
        }
        Ok(prepared)
    }

    /// Decode a container, materializing every section into owned logical bytes.
    ///
    /// # Errors
    ///
    /// Bad magic, version, CBOR, length, or checksum.
    pub fn read_from<R: Read>(r: R) -> Result<Self, IoError> {
        Self::read_from_with_limits(r, &ContainerLimits::default())
    }

    /// [`Self::read_from`] under the caller's container bounds.
    ///
    /// # Errors
    ///
    /// As [`Self::read_from`]; a bound of `limits` exceeded is [`IoError::TooLarge`].
    pub fn read_from_with_limits<R: Read>(
        mut r: R,
        limits: &ContainerLimits,
    ) -> Result<Self, IoError> {
        let (manifest, mut r) = read_header_and_manifest_with(&mut r, limits)?;
        let mut sections = Vec::with_capacity(manifest.sections.len());
        for desc in &manifest.sections {
            let logical = read_section_logical(&mut r, desc, limits)?;
            sections.push(SectionBytes::new(desc.id.clone(), logical));
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
    pub fn read_selective<R: Read>(r: R, want: &HashSet<&str>) -> Result<Self, IoError> {
        Self::read_selective_with_limits(r, want, &ContainerLimits::default())
    }

    /// [`Self::read_selective`] under the caller's container bounds.
    ///
    /// # Errors
    ///
    /// As [`Self::read_selective`]; a bound of `limits` exceeded is [`IoError::TooLarge`].
    pub fn read_selective_with_limits<R: Read>(
        mut r: R,
        want: &HashSet<&str>,
        limits: &ContainerLimits,
    ) -> Result<Self, IoError> {
        let (manifest, mut r) = read_header_and_manifest_with(&mut r, limits)?;
        let mut sections = Vec::new();
        let mut scratch = vec![0u8; SKIP_SCRATCH];
        for desc in &manifest.sections {
            if want.contains(desc.id.as_str()) {
                let logical = read_section_logical(&mut r, desc, limits)?;
                sections.push(SectionBytes::new(desc.id.clone(), logical));
            } else {
                skip_section_verified(&mut r, desc, &mut scratch, limits)?;
            }
        }
        Ok(Self { manifest, sections })
    }
}

/// Read magic, container version, and CBOR manifest; leave `r` at the first section.
pub(crate) fn read_header_and_manifest<R: Read>(
    r: &mut R,
) -> Result<(ArtifactManifest, &mut R), IoError> {
    read_header_and_manifest_with(r, &ContainerLimits::default())
}

/// [`read_header_and_manifest`] under the caller's container bounds.
pub(crate) fn read_header_and_manifest_with<'r, R: Read>(
    r: &'r mut R,
    limits: &ContainerLimits,
) -> Result<(ArtifactManifest, &'r mut R), IoError> {
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
    if manifest_len > limits.max_manifest_bytes {
        return Err(IoError::TooLarge);
    }
    let manifest_buf = read_exact_len(r, manifest_len)?;
    // The manifest is one CBOR item; trailing bytes are refused like in section payloads.
    let manifest: ArtifactManifest = crate::convert::from_cbor(&manifest_buf)?;
    // A release build never consumes what a development build sealed.
    manifest.refuse_development()?;
    let total = manifest
        .sections
        .iter()
        .try_fold(0u64, |acc, desc| acc.checked_add(desc.uncompressed_size))
        .ok_or(IoError::TooLarge)?;
    if total > limits.max_total_uncompressed_bytes {
        return Err(IoError::TooLarge);
    }
    // Lookups are by id, so a repeated id would let one payload be read while another
    // is hashed, listed or skipped.
    let mut ids = HashSet::with_capacity(manifest.sections.len());
    if !manifest.sections.iter().all(|desc| ids.insert(desc.id.as_str())) {
        return Err(IoError::ManifestMismatch { message: "duplicate section id" });
    }
    Ok((manifest, r))
}

fn read_section_logical<R: Read>(
    r: &mut R,
    desc: &SectionDescriptor,
    limits: &ContainerLimits,
) -> Result<Vec<u8>, IoError> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = usize::try_from(u32::from_le_bytes(len_buf)).map_err(|_| IoError::TooLarge)?;
    let expected_comp = usize::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
    if len != expected_comp {
        return Err(IoError::ManifestMismatch { message: "section size mismatch" });
    }
    if len > limits.max_section_bytes {
        return Err(IoError::TooLarge);
    }
    let on_wire = read_exact_len(r, len)?;
    let hash = blake3::hash(&on_wire);
    if hash.as_bytes() != &desc.blake3 {
        return Err(IoError::ChecksumMismatch { section: desc.id.clone() });
    }
    // An uncompressed section is already the logical payload: move it instead of copying.
    if desc.compression.is_none() {
        return Ok(on_wire);
    }
    decode_on_wire_with(
        &on_wire,
        desc.compression.as_deref(),
        &desc.id,
        desc.uncompressed_size,
        limits.max_section_bytes,
    )
}

/// Read exactly `len` bytes without reserving `len` up front.
///
/// A declared length is untrusted. `read_to_end` on a limited reader grows with
/// the bytes that actually arrive, then the length is checked. A short stream
/// therefore cannot force a multi-hundred-megabyte zeroed buffer.
pub(crate) fn read_exact_len<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>, IoError> {
    let mut limited = reader.take(crate::error::lossless_u64(len));
    let mut buf = Vec::new();
    limited.read_to_end(&mut buf)?;
    if buf.len() != len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "failed to fill whole buffer",
        )
        .into());
    }
    Ok(buf)
}

/// Consume an on-wire section, verifying BLAKE3, without retaining the payload.
///
/// `scratch` is a caller-owned skip buffer reused across sections (grown to
/// [`SKIP_SCRATCH`] on first use).
fn skip_section_verified<R: Read>(
    r: &mut R,
    desc: &SectionDescriptor,
    scratch: &mut Vec<u8>,
    limits: &ContainerLimits,
) -> Result<(), IoError> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = usize::try_from(u32::from_le_bytes(len_buf)).map_err(|_| IoError::TooLarge)?;
    let expected_comp = usize::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
    if len != expected_comp {
        return Err(IoError::ManifestMismatch { message: "section size mismatch" });
    }
    if len > limits.max_section_bytes {
        return Err(IoError::TooLarge);
    }
    if scratch.len() < SKIP_SCRATCH {
        scratch.resize(SKIP_SCRATCH, 0);
    }
    let mut hasher = blake3::Hasher::new();
    let mut remaining = len;
    while remaining > 0 {
        let n = remaining.min(scratch.len());
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
    descriptor_and_on_wire(id, content_type, logical, policy).0
}

/// Descriptor plus the compressed on-wire buffer it hashed (when compressed),
/// so packers can retain the buffer for [`EncodedArtifact::write_to`].
fn descriptor_and_on_wire(
    id: impl Into<String>,
    content_type: impl Into<String>,
    logical: &[u8],
    policy: CompressPolicy,
) -> (SectionDescriptor, Option<Vec<u8>>) {
    let (compression, on_wire_owned) = choose_on_wire(logical, policy);
    let on_wire: &[u8] = on_wire_owned.as_deref().unwrap_or(logical);
    let hash = blake3::hash(on_wire);
    let mut blake3_bytes = [0u8; 32];
    blake3_bytes.copy_from_slice(hash.as_bytes());
    let desc = SectionDescriptor {
        id: id.into(),
        content_type: content_type.into(),
        encoding_version: 1,
        required: true,
        compression,
        compressed_size: crate::error::lossless_u64(on_wire.len()),
        uncompressed_size: crate::error::lossless_u64(logical.len()),
        blake3: blake3_bytes,
        logical_schema: "cbor.v1".into(),
    };
    (desc, on_wire_owned)
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
    let (desc, on_wire) =
        descriptor_and_on_wire(id.clone(), content_type, logical.as_ref(), policy);
    (desc, SectionBytes { id, data: logical, packed_on_wire: on_wire.map(Arc::new) })
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
            .map_err(|e| IoError::io(format!("zstd encode: {e}"))),
        Some(other) => Err(IoError::UnsupportedCompression { algo: other.into() }),
    }
}

/// Decode on-wire bytes to logical bytes.
///
/// For zstd, `uncompressed_size` is a hard cap applied *during* streaming
/// decompression (not after). Missing / unrepresentable sizes and frames that
/// expand past the declared size are refused without allocating the expansion.
#[cfg(test)]
fn decode_on_wire(
    on_wire: &[u8],
    compression: Option<&str>,
    section: &str,
    uncompressed_size: u64,
) -> Result<Vec<u8>, IoError> {
    decode_on_wire_with(on_wire, compression, section, uncompressed_size, MAX_SECTION_BYTES)
}

/// Decode on-wire bytes with the caller's largest section.
pub(crate) fn decode_on_wire_with(
    on_wire: &[u8],
    compression: Option<&str>,
    section: &str,
    uncompressed_size: u64,
    max_section_bytes: usize,
) -> Result<Vec<u8>, IoError> {
    match compression {
        None => Ok(on_wire.to_vec()),
        Some(COMPRESSION_ZSTD) => {
            let max = decode_size_cap(uncompressed_size, max_section_bytes)?;
            let logical = decode_zstd_bounded(on_wire, max, section)?;
            if logical.len() != max {
                return Err(IoError::Decompress {
                    section: section.into(),
                    message: format!("logical size {} != uncompressed_size {max}", logical.len()),
                });
            }
            Ok(logical)
        }
        Some(other) => Err(IoError::UnsupportedCompression { algo: other.into() }),
    }
}

/// Decode borrowed on-wire bytes (mmap path; the uncompressed case there is
/// served by `SectionAccess::Mapped` and never reaches this copy).
pub(crate) fn decode_on_wire_arc(
    on_wire: &[u8],
    compression: Option<&str>,
    section: &str,
    uncompressed_size: u64,
    max_section_bytes: usize,
) -> Result<(Arc<[u8]>, bool), IoError> {
    match compression {
        None => Ok((Arc::<[u8]>::from(on_wire), false)),
        Some(COMPRESSION_ZSTD) => {
            let v = decode_on_wire_with(
                on_wire,
                compression,
                section,
                uncompressed_size,
                max_section_bytes,
            )?;
            Ok((Arc::from(v), true))
        }
        Some(other) => Err(IoError::UnsupportedCompression { algo: other.into() }),
    }
}

/// Decode owned on-wire bytes: the uncompressed case moves the buffer into the
/// `Arc` with no copy (the seek reader previously paid a full extra copy of
/// every uncompressed section here).
pub(crate) fn decode_on_wire_arc_owned(
    on_wire: Vec<u8>,
    compression: Option<&str>,
    section: &str,
    uncompressed_size: u64,
    max_section_bytes: usize,
) -> Result<(Arc<[u8]>, bool), IoError> {
    match compression {
        None => Ok((Arc::from(on_wire), false)),
        _ => {
            decode_on_wire_arc(&on_wire, compression, section, uncompressed_size, max_section_bytes)
        }
    }
}

fn decode_size_cap(uncompressed_size: u64, max_section_bytes: usize) -> Result<usize, IoError> {
    let max = usize::try_from(uncompressed_size).map_err(|_| IoError::TooLarge)?;
    if max > max_section_bytes {
        return Err(IoError::TooLarge);
    }
    Ok(max)
}

/// Expected expansion ratio used to size the first output allocation.
const DECODE_INITIAL_EXPANSION: usize = 4;
/// Largest first output allocation of a bounded decode.
const DECODE_INITIAL_CAPACITY_MAX: usize = 16 * 1024 * 1024;

/// Stream-decompress zstd, never retaining more than `max_uncompressed + 1` bytes
/// so a hostile frame cannot expand past the declared size.
fn decode_zstd_bounded(
    on_wire: &[u8],
    max_uncompressed: usize,
    section: &str,
) -> Result<Vec<u8>, IoError> {
    let decoder = zstd::Decoder::new(on_wire)
        .map_err(|e| IoError::Decompress { section: section.into(), message: e.to_string() })?;
    let read_cap = u64::try_from(max_uncompressed)
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or(IoError::TooLarge)?;
    let mut limited = decoder.take(read_cap);
    // The declared size is a limit, not a trusted allocation request: start from
    // a small multiple of the compressed length and let the vector grow.
    let mut out = Vec::with_capacity(
        max_uncompressed
            .min(on_wire.len().saturating_mul(DECODE_INITIAL_EXPANSION))
            .min(DECODE_INITIAL_CAPACITY_MAX),
    );
    limited
        .read_to_end(&mut out)
        .map_err(|e| IoError::Decompress { section: section.into(), message: e.to_string() })?;
    if out.len() > max_uncompressed {
        return Err(IoError::Decompress {
            section: section.into(),
            message: format!(
                "decompressed size exceeds declared uncompressed_size {max_uncompressed}"
            ),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::VERSION;

    use super::*;
    use crate::convert::to_cbor;

    struct ImmediateEof {
        largest: usize,
    }

    impl Read for ImmediateEof {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.largest = self.largest.max(buf.len());
            Ok(0)
        }
    }

    #[test]
    fn declared_length_is_not_reserved_before_bytes_arrive() {
        let mut reader = ImmediateEof { largest: 0 };
        let err = read_exact_len(&mut reader, 1_000_000).unwrap_err();
        assert!(matches!(err, IoError::Io(_)), "{err}");
        assert!(
            reader.largest < 1_000_000,
            "read requested {} bytes for a stream that ended immediately",
            reader.largest
        );
        let bytes = b"section-bytes";
        assert_eq!(read_exact_len(&mut &bytes[..], bytes.len()).unwrap(), bytes);
        assert!(read_exact_len(&mut &bytes[..], bytes.len() + 1).is_err());
    }

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

    /// The pack-time on-wire cache must produce byte-identical files: writing
    /// with the cached zstd buffer and writing with the cache stripped (forcing
    /// a fresh encode) must emit the same bytes, and both must round-trip.
    #[test]
    fn cached_on_wire_writes_identical_bytes_to_reencode() {
        let payload: Vec<u8> = (0u32..32 * 1024).map(|i| u8::try_from(i % 251).unwrap()).collect();
        let (desc, sec) = pack_section(
            "blob",
            "application/octet-stream",
            payload.clone(),
            CompressPolicy::Always,
        );
        assert!(sec.packed_on_wire.is_some(), "Always-compressed pack must cache on-wire bytes");
        let stripped = SectionBytes::new(sec.id.clone(), Arc::clone(&sec.data));
        assert!(stripped.packed_on_wire.is_none());

        let cached_art = tiny_artifact(vec![(desc.clone(), sec)]);
        let stripped_art = tiny_artifact(vec![(desc, stripped)]);
        let mut cached_buf = Vec::new();
        cached_art.write_to(&mut cached_buf).unwrap();
        let mut stripped_buf = Vec::new();
        stripped_art.write_to(&mut stripped_buf).unwrap();
        assert_eq!(cached_buf, stripped_buf, "cached write must match re-encoded write");

        let decoded = EncodedArtifact::read_from(cached_buf.as_slice()).unwrap();
        assert_eq!(&*decoded.sections[0].data, payload.as_slice());
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
    fn duplicate_section_ids_are_rejected_on_read() {
        let (first, first_bytes) = pack_section(
            "note",
            "application/octet-stream",
            b"one".to_vec(),
            CompressPolicy::Never,
        );
        let (second, second_bytes) = pack_section(
            "note",
            "application/octet-stream",
            b"two".to_vec(),
            CompressPolicy::Never,
        );
        let art = tiny_artifact(vec![(first, first_bytes), (second, second_bytes)]);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let err = EncodedArtifact::read_from(buf.as_slice()).unwrap_err();
        assert_eq!(err, IoError::ManifestMismatch { message: "duplicate section id" });
        assert!(EncodedArtifact::read_selective(buf.as_slice(), &HashSet::from(["note"])).is_err());
        assert!(crate::reader::ArtifactReader::open_seek(std::io::Cursor::new(buf)).is_err());
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

    /// Container header (magic, version, manifest) with `extra` appended to the manifest bytes.
    fn header_bytes(manifest: &ArtifactManifest, extra: &[u8]) -> Vec<u8> {
        let mut cbor = Vec::new();
        ciborium::into_writer(manifest, &mut cbor).unwrap();
        cbor.extend_from_slice(extra);
        let mut buf = MAGIC.to_vec();
        buf.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
        buf.extend_from_slice(&u32::try_from(cbor.len()).unwrap().to_le_bytes());
        buf.extend_from_slice(&cbor);
        buf
    }

    fn declared_total_header(sizes: &[u64]) -> Vec<u8> {
        let sections = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let (mut desc, sec) = pack_section(
                    format!("s{i}"),
                    "application/octet-stream",
                    b"x".to_vec(),
                    CompressPolicy::Never,
                );
                desc.uncompressed_size = size;
                (desc, sec)
            })
            .collect();
        header_bytes(&tiny_artifact(sections).manifest, &[])
    }

    #[test]
    fn declared_uncompressed_total_is_capped_before_reading_sections() {
        let per = MAX_SECTION_BYTES as u64;
        let ok = declared_total_header(&[per, per, per, per]);
        assert!(read_header_and_manifest(&mut ok.as_slice()).is_ok());
        for sizes in [vec![per; 5], vec![u64::MAX, 1]] {
            let buf = declared_total_header(&sizes);
            let err = read_header_and_manifest(&mut buf.as_slice()).unwrap_err();
            assert_eq!(err, IoError::TooLarge, "{sizes:?}");
            let err = EncodedArtifact::read_from(buf.as_slice()).unwrap_err();
            assert_eq!(err, IoError::TooLarge, "{sizes:?}");
        }
    }

    #[test]
    fn manifest_with_trailing_bytes_is_refused() {
        let buf = header_bytes(&tiny_artifact(Vec::new()).manifest, &[0x00]);
        let err = EncodedArtifact::read_from(buf.as_slice()).unwrap_err();
        assert!(matches!(&err, IoError::Cbor(m) if m.contains("trailing")), "{err:?}");
        let clean = header_bytes(&tiny_artifact(Vec::new()).manifest, &[]);
        assert!(EncodedArtifact::read_from(clean.as_slice()).is_ok());
    }

    /// Hostile zstd frame that expands far past a lying `uncompressed_size`
    /// must fail during the capped stream decode (io-5), not after allocating.
    #[test]
    fn zstd_decode_refuses_oversize_or_lying_declared_size() {
        let logical = vec![0xABu8; 64 * 1024];
        let on_wire = zstd::encode_all(logical.as_slice(), ZSTD_LEVEL).unwrap();
        let err = decode_on_wire(&on_wire, Some(COMPRESSION_ZSTD), "blob", 128).unwrap_err();
        assert!(
            matches!(err, IoError::Decompress { .. }),
            "lying uncompressed_size must be Err, got {err:?}"
        );
        let err_huge =
            decode_on_wire(&on_wire, Some(COMPRESSION_ZSTD), "blob", MAX_SECTION_BYTES as u64 + 1)
                .unwrap_err();
        assert!(
            matches!(err_huge, IoError::TooLarge),
            "missing usable size bound must be Err, got {err_huge:?}"
        );
        let ok = decode_on_wire(
            &on_wire,
            Some(COMPRESSION_ZSTD),
            "blob",
            u64::try_from(logical.len()).unwrap(),
        )
        .unwrap();
        assert_eq!(ok, logical);
    }

    mod streamed {
        use std::io::Cursor;
        use std::sync::atomic::{AtomicU64, Ordering};

        use antecedent_core::execution::AdmissionPool;
        use antecedent_core::{CanonicalWriter, ExecutionContext};

        use super::super::*;
        use super::tiny_artifact;

        /// Bytes produced on demand, so a test source is never resident.
        struct Generated {
            len: u64,
            random: bool,
        }

        struct GeneratedReader {
            left: u64,
            state: u64,
            at: u64,
            random: bool,
        }

        impl Read for GeneratedReader {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the value is bounded by its validated numerical or fixture domain"
            )]
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let n = usize::try_from(self.left).unwrap_or(usize::MAX).min(buf.len());
                for byte in &mut buf[..n] {
                    *byte = if self.random {
                        self.state ^= self.state << 13;
                        self.state ^= self.state >> 7;
                        self.state ^= self.state << 17;
                        (self.state >> 24) as u8
                    } else {
                        self.at += 1;
                        b'a' + ((self.at / 64) % 7) as u8
                    };
                }
                self.left -= n as u64;
                Ok(n)
            }
        }

        impl LogicalSource for Generated {
            fn logical_len(&self) -> u64 {
                self.len
            }

            fn open(&self) -> std::io::Result<Box<dyn Read + '_>> {
                Ok(Box::new(GeneratedReader {
                    left: self.len,
                    state: 0x9E37_79B9_7F4A_7C15,
                    at: 0,
                    random: self.random,
                }))
            }
        }

        fn hash_of(source: &dyn LogicalSource) -> [u8; 32] {
            let mut reader = source.open().unwrap();
            let mut hasher = blake3::Hasher::new();
            let mut buf = vec![0u8; 8192];
            loop {
                let n = reader.read(&mut buf).unwrap();
                if n == 0 {
                    return *hasher.finalize().as_bytes();
                }
                hasher.update(&buf[..n]);
            }
        }

        fn study(pool: &AdmissionPool) -> ExecutionContext {
            ExecutionContext::for_tests(1).with_admission(pool.clone())
        }

        fn manifest_of(descs: Vec<SectionDescriptor>) -> ArtifactManifest {
            tiny_artifact(Vec::new()).manifest.clone_with(descs)
        }

        trait CloneWith {
            fn clone_with(&self, sections: Vec<SectionDescriptor>) -> ArtifactManifest;
        }

        impl CloneWith for ArtifactManifest {
            fn clone_with(&self, sections: Vec<SectionDescriptor>) -> ArtifactManifest {
                ArtifactManifest { sections, ..self.clone() }
            }
        }

        fn scratch_file(label: &str) -> std::path::PathBuf {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            std::env::temp_dir().join(format!(
                "antecedent-streamed-{}-{label}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::SeqCst)
            ))
        }

        #[test]
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the value is bounded by its validated numerical or fixture domain"
        )]
        fn a_streamed_write_equals_the_buffered_container_and_reads_back() {
            let text: Vec<u8> = (0..300_000u32).map(|i| b'a' + ((i / 50) % 9) as u8).collect();
            let noise: Vec<u8> =
                (0..100_000u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
            let (d1, s1) = pack_section("text", "t", text.clone(), CompressPolicy::Always);
            let (d2, s2) = pack_section("noise", "n", noise.clone(), CompressPolicy::Auto);
            let (d3, s3) = pack_section("tiny", "x", b"abc".to_vec(), CompressPolicy::Never);
            assert!(d1.compression.is_some());
            let artifact = tiny_artifact(vec![(d1, s1), (d2, s2), (d3, s3)]);
            let mut buffered = Vec::new();
            artifact.write_to(&mut buffered).unwrap();

            let pool = AdmissionPool::fixed(1, 64 << 20);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(4096);
            let mut streamed = Vec::new();
            let receipt = artifact.write_streaming(&mut streamed, &work).unwrap();
            assert_eq!(streamed, buffered, "streaming reproduces the buffered bytes");
            assert!(receipt.largest_read <= 4096);
            assert_eq!(pool.available().bytes(), 64 << 20, "the working set is returned");

            let mut seen: Vec<(String, Vec<u8>)> = Vec::new();
            let (manifest, read) = read_streaming(
                Cursor::new(&streamed),
                &work,
                |_| true,
                |desc, chunk| {
                    assert!(chunk.len() <= 4096);
                    match seen.last_mut() {
                        Some((id, bytes)) if *id == desc.id => bytes.extend_from_slice(chunk),
                        _ => seen.push((desc.id.clone(), chunk.to_vec())),
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(manifest, artifact.manifest);
            assert_eq!(read.logical_bytes, (text.len() + noise.len() + 3) as u64);
            assert_eq!(seen[0], ("text".to_owned(), text));
            assert_eq!(seen[1], ("noise".to_owned(), noise));
            assert_eq!(seen[2], ("tiny".to_owned(), b"abc".to_vec()));

            let mut only = 0usize;
            read_streaming(
                Cursor::new(&streamed),
                &work,
                |desc| desc.id == "tiny",
                |desc, chunk| {
                    assert_eq!(desc.id, "tiny");
                    only += chunk.len();
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(only, 3, "unwanted sections are verified and skipped");
        }

        #[test]
        fn sections_larger_than_the_budget_stream_through_a_chunk_sized_working_set() {
            const BUDGET: u64 = 24 << 20;
            let pool = AdmissionPool::fixed(1, BUDGET);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(32 * 1024);
            let sources = [
                Generated { len: 20 << 20, random: true },
                Generated { len: 96 << 20, random: false },
            ];
            let mut descs = Vec::new();
            for (index, source) in sources.iter().enumerate() {
                let (desc, receipt) = describe_streaming(
                    format!("s{index}"),
                    "bulk",
                    source,
                    CompressPolicy::Auto,
                    &work,
                )
                .unwrap();
                assert!(receipt.leased_bytes <= BUDGET);
                assert!(receipt.largest_read <= 32 * 1024);
                descs.push(desc);
            }
            assert!(descs[0].compression.is_none(), "noise does not earn compression");
            assert!(descs[1].compression.is_some());
            assert!(descs[1].compressed_size < descs[1].uncompressed_size / 20);
            assert_eq!(pool.available().bytes(), BUDGET);

            let manifest = manifest_of(descs);
            let path = scratch_file("bulk");
            let file = std::fs::File::create(&path).unwrap();
            let refs: Vec<&dyn LogicalSource> = vec![&sources[0], &sources[1]];
            let written = write_streaming(file, &manifest, &refs, &work).unwrap();
            assert!(written.leased_bytes <= BUDGET);
            assert!(written.largest_read <= 32 * 1024);
            assert!(written.largest_write <= 1 << 20, "{}", written.largest_write);
            assert!(
                written.leased_bytes < manifest.sections[0].uncompressed_size,
                "the working set is smaller than one section"
            );
            assert_eq!(pool.available().bytes(), BUDGET);

            let mut hashers = [blake3::Hasher::new(), blake3::Hasher::new()];
            let mut largest = 0usize;
            let (read_manifest, read) = read_streaming(
                std::fs::File::open(&path).unwrap(),
                &work,
                |_| true,
                |desc, chunk| {
                    largest = largest.max(chunk.len());
                    hashers[usize::from(desc.id == "s1")].update(chunk);
                    Ok(())
                },
            )
            .unwrap();
            let _ = std::fs::remove_file(&path);
            assert_eq!(read_manifest, manifest);
            assert!(largest <= 32 * 1024, "{largest}");
            assert!(read.leased_bytes <= BUDGET);
            assert_eq!(hashers[0].finalize().as_bytes(), &hash_of(&sources[0]));
            assert_eq!(hashers[1].finalize().as_bytes(), &hash_of(&sources[1]));
            assert_eq!(pool.available().bytes(), BUDGET);
        }

        #[test]
        fn a_pool_short_of_the_working_set_refuses_before_any_byte_moves() {
            let source = Generated { len: 100_000, random: false };
            let chunk = 4096usize;
            let need = 2 * chunk as u64 + stream_encoder_workspace_bytes();
            let pool = AdmissionPool::fixed(1, need - 1);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(chunk);
            let refused =
                describe_streaming("s", "t", &source, CompressPolicy::Always, &work).unwrap_err();
            assert!(matches!(&refused, IoError::StreamBudgetRefused { .. }), "{refused}");
            assert_eq!(pool.available().bytes(), need - 1);

            let exact = AdmissionPool::fixed(1, need);
            let ctx = study(&exact);
            let work = StreamWork::new(&ctx).with_chunk_bytes(chunk);
            let (desc, _) =
                describe_streaming("s", "t", &source, CompressPolicy::Always, &work).unwrap();
            let manifest = manifest_of(vec![desc]);
            let short = AdmissionPool::fixed(1, need - 1);
            let ctx = study(&short);
            let work = StreamWork::new(&ctx).with_chunk_bytes(chunk);
            let mut out = Vec::new();
            let refused = write_streaming(&mut out, &manifest, &[&source], &work).unwrap_err();
            assert!(matches!(&refused, IoError::StreamBudgetRefused { .. }));
            assert!(out.is_empty(), "a refused lease writes no header");
        }

        struct CancelAfter<'a> {
            seen: u64,
            limit: u64,
            token: &'a antecedent_core::CancellationToken,
        }

        impl Write for CancelAfter<'_> {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.seen += buf.len() as u64;
                if self.seen >= self.limit {
                    self.token.cancel();
                }
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        #[test]
        fn cancellation_stops_a_streamed_write_and_read_between_chunks() {
            let source = Generated { len: 16 << 20, random: true };
            let pool = AdmissionPool::fixed(1, 64 << 20);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(16 * 1024);
            let (desc, _) =
                describe_streaming("s", "t", &source, CompressPolicy::Never, &work).unwrap();
            let manifest = manifest_of(vec![desc]);

            let token = ctx.cancellation.clone();
            let mut sink = CancelAfter { seen: 0, limit: 1 << 20, token: &token };
            let refused = write_streaming(&mut sink, &manifest, &[&source], &work).unwrap_err();
            assert!(matches!(&refused, IoError::StreamCancelled), "{refused}");
            assert!(sink.seen < 4 << 20, "the write stopped soon after the cancel: {}", sink.seen);
            assert_eq!(pool.available().bytes(), 64 << 20, "the lease came back");

            let fresh = study(&pool);
            let work = StreamWork::new(&fresh).with_chunk_bytes(16 * 1024);
            let mut bytes = Vec::new();
            write_streaming(&mut bytes, &manifest, &[&source], &work).unwrap();
            let token = fresh.cancellation.clone();
            let mut delivered = 0u64;
            let refused = read_streaming(
                Cursor::new(&bytes),
                &work,
                |_| true,
                |_, chunk| {
                    delivered += chunk.len() as u64;
                    if delivered >= 1 << 20 {
                        token.cancel();
                    }
                    Ok(())
                },
            )
            .unwrap_err();
            assert!(matches!(&refused, IoError::StreamCancelled), "{refused}");
            assert!(delivered < 4 << 20, "{delivered}");
            assert_eq!(pool.available().bytes(), 64 << 20);
        }

        #[test]
        fn a_corrupt_section_reaches_no_sink_and_an_expanding_frame_is_cut_at_its_declared_size() {
            let pool = AdmissionPool::fixed(1, 64 << 20);
            let ctx = study(&pool);
            let work = StreamWork::new(&ctx).with_chunk_bytes(4096);
            let source = Generated { len: 200_000, random: true };
            let (desc, _) =
                describe_streaming("s", "t", &source, CompressPolicy::Never, &work).unwrap();
            let manifest = manifest_of(vec![desc]);
            let mut bytes = Vec::new();
            write_streaming(&mut bytes, &manifest, &[&source], &work).unwrap();
            let middle = bytes.len() - 100_000;
            bytes[middle] ^= 0x40;
            let mut calls = 0;
            let refused = read_streaming(
                Cursor::new(&bytes),
                &work,
                |_| true,
                |_, _| {
                    calls += 1;
                    Ok(())
                },
            )
            .unwrap_err();
            assert!(matches!(refused, IoError::ChecksumMismatch { .. }), "{refused}");
            assert_eq!(calls, 0, "verification precedes delivery");

            // A frame of one mebibyte of zeros whose descriptor declares 1000 bytes.
            let zeros = vec![0u8; 1 << 20];
            let frame = zstd::encode_all(&zeros[..], 3).unwrap();
            let mut descriptor =
                section_descriptor_with_policy("z", "t", &zeros, CompressPolicy::Always);
            descriptor.uncompressed_size = 1000;
            assert_eq!(descriptor.compressed_size, frame.len() as u64);
            let manifest = manifest_of(vec![descriptor]);
            let mut container = Vec::new();
            container.extend_from_slice(MAGIC);
            container.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
            let mut cbor = Vec::new();
            ciborium::into_writer(&manifest, &mut cbor).unwrap();
            container.extend_from_slice(&u32::try_from(cbor.len()).unwrap().to_le_bytes());
            container.extend_from_slice(&cbor);
            container.extend_from_slice(&u32::try_from(frame.len()).unwrap().to_le_bytes());
            container.extend_from_slice(&frame);
            let mut delivered = 0usize;
            let refused = read_streaming(
                Cursor::new(&container),
                &work,
                |_| true,
                |_, chunk| {
                    delivered += chunk.len();
                    Ok(())
                },
            )
            .unwrap_err();
            assert!(matches!(refused, IoError::Decompress { .. }), "{refused}");
            assert!(delivered <= 1000, "output past the declared size never reaches the sink");
        }

        #[test]
        fn the_piecewise_domain_digest_equals_the_one_shot_canonical_digest() {
            let bytes: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
            let tag = "antecedent.test.domain";
            let mut digest = DomainDigest::new(tag, bytes.len() as u64);
            for piece in bytes.chunks(777) {
                digest.update(piece);
            }
            assert_eq!(digest.finish(), Some(CanonicalWriter::digest_bytes(tag, &bytes)));
            let mut short = DomainDigest::new(tag, bytes.len() as u64);
            short.update(&bytes[..bytes.len() - 1]);
            assert_eq!(short.finish(), None);
        }
    }
}
