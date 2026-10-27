//! Selective artifact reader with skip / mmap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::container::{
    ArtifactManifest, ContainerLimits, EncodedArtifact, SectionBytes, decode_on_wire_arc,
    decode_on_wire_arc_owned, read_header_and_manifest_with,
};
use crate::error::IoError;
use crate::mmap_file::map_file_readonly;
use crate::wire::SectionDescriptor;

/// Index entry for one section payload inside a seekable/mmap artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectionIndexEntry {
    /// Section id.
    pub id: String,
    /// Absolute file offset of on-wire bytes (after the `u32` length prefix).
    pub file_offset: u64,
    /// On-wire length in bytes.
    pub on_wire_len: u32,
}

/// Counters for selective load / mmap (DESIGN rule 22).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SectionLoadStats {
    /// Sections whose payloads were never materialized.
    pub sections_skipped: u64,
    /// Sections loaded into owned logical bytes.
    pub sections_loaded: u64,
    /// On-wire bytes skipped (seek or stream discard).
    pub bytes_skipped: u64,
    /// On-wire bytes read into memory for load.
    pub bytes_loaded: u64,
    /// Mapped (zero-copy) logical views issued.
    pub mmap_views: u64,
    /// Sections that required decompression.
    pub decompressions: u64,
}

/// Immutable artifact bytes: an owned snapshot, or a file-backed map.
#[derive(Clone, Debug)]
enum ArtifactBacking {
    Owned(Arc<[u8]>),
    /// A file snapshot read straight into its final allocation.
    OwnedVec(Arc<Vec<u8>>),
    Mapped(Arc<Mmap>),
}

impl ArtifactBacking {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes.as_ref(),
            Self::OwnedVec(bytes) => bytes.as_slice(),
            Self::Mapped(mmap) => mmap.as_ref(),
        }
    }
}

/// Zero-copy view into an uncompressed section.
#[derive(Clone, Debug)]
pub struct MappedSection {
    backing: ArtifactBacking,
    start: usize,
    len: usize,
    /// Section id.
    pub id: String,
}

impl MappedSection {
    /// Borrow the on-wire (== logical, uncompressed) bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.backing.as_slice()[self.start..self.start + self.len]
    }
}

impl AsRef<[u8]> for MappedSection {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl PartialEq for MappedSection {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.as_bytes() == other.as_bytes()
    }
}

impl Eq for MappedSection {}

/// Access to a section's logical payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SectionAccess {
    /// Owned / shared logical bytes (possibly decompressed).
    Shared(Arc<[u8]>),
    /// Mmap view of an uncompressed on-wire section.
    Mapped(MappedSection),
}

impl SectionAccess {
    /// Borrow logical bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Shared(a) => a.as_ref(),
            Self::Mapped(m) => m.as_bytes(),
        }
    }

    /// Clone into a shared owned buffer (copies mapped views).
    #[must_use]
    pub fn into_shared(self) -> Arc<[u8]> {
        match self {
            Self::Shared(a) => a,
            Self::Mapped(m) => Arc::from(m.as_bytes().to_vec()),
        }
    }
}

/// Seekable selective reader: indexes sections without materializing payloads.
pub struct ArtifactReader<R> {
    manifest: ArtifactManifest,
    index: Vec<SectionIndexEntry>,
    by_id: HashMap<String, usize>,
    inner: R,
    stats: SectionLoadStats,
    loaded: Vec<bool>,
    /// Decompressed payloads, kept so a repeated load of a compressed section
    /// neither re-reads nor re-decompresses it.
    decoded: Vec<Option<Arc<[u8]>>>,
    limits: ContainerLimits,
}

impl<R: Read + Seek> ArtifactReader<R> {
    /// Open a seekable source: parse manifest and record section offsets without loading payloads.
    ///
    /// # Errors
    ///
    /// IO / framing / CBOR failures.
    pub fn open_seek(inner: R) -> Result<Self, IoError> {
        Self::open_seek_with_limits(inner, ContainerLimits::default())
    }

    /// [`Self::open_seek`] under the caller's container bounds, which also bound every later
    /// section load.
    ///
    /// # Errors
    ///
    /// As [`Self::open_seek`]; a bound of `limits` exceeded is [`IoError::TooLarge`].
    pub fn open_seek_with_limits(mut inner: R, limits: ContainerLimits) -> Result<Self, IoError> {
        let (manifest, index, mut stats) = index_seekable(&mut inner, &limits)?;
        stats.bytes_skipped = index.iter().map(|e| u64::from(e.on_wire_len)).sum();
        stats.sections_skipped = index.len() as u64;
        let by_id = index_by_id(&index);
        let loaded = vec![false; index.len()];
        let decoded = vec![None; index.len()];
        Ok(Self { manifest, index, by_id, inner, stats, loaded, decoded, limits })
    }

    /// Manifest (always available after open).
    #[must_use]
    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Section index (offsets into the source).
    #[must_use]
    pub fn index(&self) -> &[SectionIndexEntry] {
        &self.index
    }

    /// Load / skip statistics.
    #[must_use]
    pub fn stats(&self) -> SectionLoadStats {
        self.stats
    }

    /// Load one section's logical bytes (verify + decompress as needed).
    ///
    /// # Errors
    ///
    /// Unknown section, IO, checksum, or decompress failures.
    pub fn load_section(&mut self, id: &str) -> Result<SectionAccess, IoError> {
        let pos = self.lookup_pos(id)?;
        if let Some(cached) = &self.decoded[pos] {
            self.stats.sections_loaded += 1;
            return Ok(SectionAccess::Shared(Arc::clone(cached)));
        }
        let SectionIndexEntry { file_offset, on_wire_len, .. } = self.index[pos];
        self.inner.seek(SeekFrom::Start(file_offset)).map_err(IoError::from)?;
        let len = usize::try_from(on_wire_len).map_err(|_| IoError::TooLarge)?;
        if len > self.limits.max_section_bytes {
            return Err(IoError::TooLarge);
        }
        let on_wire = crate::container::read_exact_len(&mut self.inner, len)?;
        let desc = &self.manifest.sections[pos];
        let hash = blake3::hash(&on_wire);
        if hash.as_bytes() != &desc.blake3 {
            return Err(IoError::ChecksumMismatch { section: id.into() });
        }
        let (logical, decompressed) = decode_on_wire_arc_owned(
            on_wire,
            desc.compression.as_deref(),
            &desc.id,
            desc.uncompressed_size,
            self.limits.max_section_bytes,
        )?;
        if decompressed {
            self.decoded[pos] = Some(Arc::clone(&logical));
        }
        note_loaded(&mut self.stats, &mut self.loaded[pos], on_wire_len, decompressed);
        Ok(SectionAccess::Shared(logical))
    }

    /// Load several sections by id; others remain skipped.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::load_section`] errors.
    pub fn load_sections(&mut self, ids: &[&str]) -> Result<Vec<(String, SectionAccess)>, IoError> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let access = self.load_section(id)?;
            out.push(((*id).to_string(), access));
        }
        Ok(out)
    }

    /// Materialize every section into an [`EncodedArtifact`].
    ///
    /// # Errors
    ///
    /// Load failures for any section.
    pub fn into_encoded_artifact(mut self) -> Result<EncodedArtifact, IoError> {
        let ids: Vec<String> = self.index.iter().map(|e| e.id.clone()).collect();
        let mut sections = Vec::with_capacity(ids.len());
        for id in ids {
            let access = self.load_section(&id)?;
            sections.push(SectionBytes::new(id, access.into_shared()));
        }
        Ok(EncodedArtifact { manifest: self.manifest, sections })
    }

    fn lookup_pos(&self, id: &str) -> Result<usize, IoError> {
        self.by_id
            .get(id)
            .copied()
            .ok_or_else(|| IoError::Convert(format!("unknown section `{id}`")))
    }
}

/// Artifact reader over an owned snapshot or a file-backed map.
pub struct MappedArtifactReader {
    backing: ArtifactBacking,
    manifest: ArtifactManifest,
    index: Vec<SectionIndexEntry>,
    by_id: HashMap<String, usize>,
    stats: SectionLoadStats,
    /// Per-section BLAKE3-verification memo. Sound for an owned snapshot
    /// (immutable bytes) and for a mapped file only while the caller upholds
    /// the immutability contract of [`Self::open_path_mapped`].
    verified: Vec<bool>,
    loaded: Vec<bool>,
    /// Decompressed payloads, kept so a repeated load of a compressed section
    /// is not decompressed again.
    decoded: Vec<Option<Arc<[u8]>>>,
    limits: ContainerLimits,
}

impl MappedArtifactReader {
    /// Read `path` into an owned snapshot and index sections.
    ///
    /// The snapshot is independent of later writes to the path.
    ///
    /// # Errors
    ///
    /// IO, bad magic, or manifest errors.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, IoError> {
        Self::open_path_with_limits(path, ContainerLimits::default())
    }

    /// [`Self::open_path`] under the caller's container bounds.
    ///
    /// # Errors
    ///
    /// As [`Self::open_path`]; a bound of `limits` exceeded is [`IoError::TooLarge`].
    pub fn open_path_with_limits(
        path: impl AsRef<Path>,
        limits: ContainerLimits,
    ) -> Result<Self, IoError> {
        // `Arc<Vec<u8>>` adopts the file buffer as is; `Arc<[u8]>::from(Vec)` would
        // allocate and copy the whole artifact a second time.
        let bytes = std::fs::read(path.as_ref()).map_err(IoError::from)?;
        Self::from_backing(ArtifactBacking::OwnedVec(Arc::new(bytes)), limits)
    }

    /// Memory-map `path` without copying payloads.
    ///
    /// # Safety
    ///
    /// The file at `path` must not be modified or truncated by any process for
    /// the entire lifetime of the reader and every [`MappedSection`] derived
    /// from it. Opening one descriptor read-only does not establish that.
    ///
    /// # Errors
    ///
    /// IO, bad magic, or manifest errors.
    #[allow(
        unsafe_code,
        reason = "memory map of a caller-owned file; the caller guarantees it is not mutated while mapped"
    )]
    pub unsafe fn open_path_mapped(path: impl AsRef<Path>) -> Result<Self, IoError> {
        // SAFETY: the caller upholds the file-immutability contract of this function.
        unsafe { Self::open_path_mapped_with_limits(path, ContainerLimits::default()) }
    }

    /// [`Self::open_path_mapped`] under the caller's container bounds.
    ///
    /// # Safety
    ///
    /// As [`Self::open_path_mapped`].
    ///
    /// # Errors
    ///
    /// As [`Self::open_path_mapped`]; a bound of `limits` exceeded is [`IoError::TooLarge`].
    #[allow(
        unsafe_code,
        reason = "memory map of a caller-owned file; the caller guarantees it is not mutated while mapped"
    )]
    pub unsafe fn open_path_mapped_with_limits(
        path: impl AsRef<Path>,
        limits: ContainerLimits,
    ) -> Result<Self, IoError> {
        let file = File::open(path.as_ref()).map_err(IoError::from)?;
        // SAFETY: caller upholds file immutability for the map lifetime.
        let mmap = Arc::new(unsafe { map_file_readonly(&file)? });
        // SAFETY: `from_mmap_with_limits` carries the same file-immutability contract as
        // `map_file_readonly` above; the caller of this function upholds it for the map.
        unsafe { Self::from_mmap_with_limits(mmap, limits) }
    }

    /// Index an owned byte snapshot.
    ///
    /// # Errors
    ///
    /// Bad magic, version, CBOR, or section framing.
    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, IoError> {
        Self::from_bytes_with_limits(bytes, ContainerLimits::default())
    }

    /// [`Self::from_bytes`] under the caller's container bounds.
    ///
    /// # Errors
    ///
    /// As [`Self::from_bytes`]; a bound of `limits` exceeded is [`IoError::TooLarge`].
    pub fn from_bytes_with_limits(
        bytes: Arc<[u8]>,
        limits: ContainerLimits,
    ) -> Result<Self, IoError> {
        Self::from_backing(ArtifactBacking::Owned(bytes), limits)
    }

    /// Index an already-mapped buffer.
    ///
    /// # Safety
    ///
    /// The mapped file must remain unmodified for the lifetime of the reader
    /// and every view derived from it.
    ///
    /// # Errors
    ///
    /// Bad magic, version, CBOR, or section framing.
    #[allow(
        unsafe_code,
        reason = "memory map of a caller-owned file; the caller guarantees it is not mutated while mapped"
    )]
    pub unsafe fn from_mmap(mmap: Arc<Mmap>) -> Result<Self, IoError> {
        // SAFETY: forwarded; the caller upholds the same contract.
        unsafe { Self::from_mmap_with_limits(mmap, ContainerLimits::default()) }
    }

    /// [`Self::from_mmap`] under the caller's container bounds.
    ///
    /// # Safety
    ///
    /// As [`Self::from_mmap`].
    ///
    /// # Errors
    ///
    /// As [`Self::from_mmap`]; a bound of `limits` exceeded is [`IoError::TooLarge`].
    #[allow(
        unsafe_code,
        reason = "memory map of a caller-owned file; the caller guarantees it is not mutated while mapped"
    )]
    pub unsafe fn from_mmap_with_limits(
        mmap: Arc<Mmap>,
        limits: ContainerLimits,
    ) -> Result<Self, IoError> {
        Self::from_backing(ArtifactBacking::Mapped(mmap), limits)
    }

    fn from_backing(backing: ArtifactBacking, limits: ContainerLimits) -> Result<Self, IoError> {
        let mut cursor = Cursor::new(backing.as_slice());
        let (manifest, index, mut stats) = index_seekable(&mut cursor, &limits)?;
        stats.bytes_skipped = index.iter().map(|e| u64::from(e.on_wire_len)).sum();
        stats.sections_skipped = index.len() as u64;
        let by_id = index_by_id(&index);
        let verified = vec![false; index.len()];
        let loaded = vec![false; index.len()];
        let decoded = vec![None; index.len()];
        Ok(Self { backing, manifest, index, by_id, stats, verified, loaded, decoded, limits })
    }

    /// Manifest.
    #[must_use]
    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Section index.
    #[must_use]
    pub fn index(&self) -> &[SectionIndexEntry] {
        &self.index
    }

    /// Load statistics.
    #[must_use]
    pub fn stats(&self) -> SectionLoadStats {
        self.stats
    }

    /// Zero-copy view of an **uncompressed** section.
    ///
    /// # Errors
    ///
    /// Unknown id, compressed section, or checksum mismatch.
    pub fn load_section_mapped(&mut self, id: &str) -> Result<MappedSection, IoError> {
        let pos = self.lookup_pos(id)?;
        if self.manifest.sections[pos].compression.is_some() {
            return Err(IoError::MappedCompressed { section: id.into() });
        }
        let (start, len) = self.locate_section(pos, id)?;
        let expected = usize::try_from(self.manifest.sections[pos].uncompressed_size)
            .map_err(|_| IoError::TooLarge)?;
        if len != expected {
            return Err(IoError::ManifestMismatch { message: "uncompressed mmap size mismatch" });
        }
        self.stats.mmap_views += 1;
        let on_wire_len = self.index[pos].on_wire_len;
        note_loaded(&mut self.stats, &mut self.loaded[pos], on_wire_len, false);
        Ok(MappedSection { backing: self.backing.clone(), start, len, id: id.into() })
    }

    /// Load logical bytes (decompress when needed). Copies on-wire into a heap buffer.
    ///
    /// # Errors
    ///
    /// Unknown section, checksum, or decompress failures.
    pub fn load_section(&mut self, id: &str) -> Result<SectionAccess, IoError> {
        let pos = self.lookup_pos(id)?;
        if self.manifest.sections[pos].compression.is_none() {
            return Ok(SectionAccess::Mapped(self.load_section_mapped(id)?));
        }
        if let Some(cached) = &self.decoded[pos] {
            self.stats.sections_loaded += 1;
            return Ok(SectionAccess::Shared(Arc::clone(cached)));
        }
        let (start, len) = self.locate_section(pos, id)?;
        let on_wire = &self.backing.as_slice()[start..start + len];
        let desc = &self.manifest.sections[pos];
        let (logical, decompressed) = decode_on_wire_arc(
            on_wire,
            desc.compression.as_deref(),
            &desc.id,
            desc.uncompressed_size,
            self.limits.max_section_bytes,
        )?;
        if decompressed {
            self.decoded[pos] = Some(Arc::clone(&logical));
        }
        let on_wire_len = self.index[pos].on_wire_len;
        note_loaded(&mut self.stats, &mut self.loaded[pos], on_wire_len, decompressed);
        Ok(SectionAccess::Shared(logical))
    }

    /// On-wire byte range of section `pos`, verified once.
    fn locate_section(&mut self, pos: usize, id: &str) -> Result<(usize, usize), IoError> {
        let entry = &self.index[pos];
        let start = usize::try_from(entry.file_offset).map_err(|_| IoError::TooLarge)?;
        let len = usize::try_from(entry.on_wire_len).map_err(|_| IoError::TooLarge)?;
        let end = start
            .checked_add(len)
            .filter(|end| *end <= self.backing.as_slice().len())
            .ok_or_else(|| IoError::io("mmap section range out of bounds"))?;
        if !self.verified[pos] {
            let hash = blake3::hash(&self.backing.as_slice()[start..end]);
            if hash.as_bytes() != &self.manifest.sections[pos].blake3 {
                return Err(IoError::ChecksumMismatch { section: id.into() });
            }
            self.verified[pos] = true;
        }
        Ok((start, len))
    }

    fn lookup_pos(&self, id: &str) -> Result<usize, IoError> {
        self.by_id
            .get(id)
            .copied()
            .ok_or_else(|| IoError::Convert(format!("unknown section `{id}`")))
    }
}

/// Count load work on every call, but remove a section from the skipped set only once.
fn note_loaded(stats: &mut SectionLoadStats, loaded: &mut bool, bytes: u32, decompressed: bool) {
    stats.sections_loaded += 1;
    stats.bytes_loaded += u64::from(bytes);
    if decompressed {
        stats.decompressions += 1;
    }
    if !*loaded {
        *loaded = true;
        stats.sections_skipped -= 1;
        stats.bytes_skipped -= u64::from(bytes);
    }
}

fn index_by_id(index: &[SectionIndexEntry]) -> HashMap<String, usize> {
    let mut by_id = HashMap::with_capacity(index.len());
    for (i, entry) in index.iter().enumerate() {
        by_id.entry(entry.id.clone()).or_insert(i);
    }
    by_id
}

fn index_seekable<R: Read + Seek>(
    r: &mut R,
    limits: &ContainerLimits,
) -> Result<(ArtifactManifest, Vec<SectionIndexEntry>, SectionLoadStats), IoError> {
    let (manifest, r) = read_header_and_manifest_with(r, limits)?;
    let mut index = Vec::with_capacity(manifest.sections.len());
    let stats = SectionLoadStats::default();
    for desc in &manifest.sections {
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf).map_err(IoError::from)?;
        let on_wire_len = u32::from_le_bytes(len_buf);
        let expected = u32::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
        if on_wire_len != expected {
            return Err(IoError::ManifestMismatch { message: "section size mismatch" });
        }
        let file_offset = r.stream_position().map_err(IoError::from)?;
        r.seek(SeekFrom::Current(i64::from(on_wire_len))).map_err(IoError::from)?;
        index.push(SectionIndexEntry { id: desc.id.clone(), file_offset, on_wire_len });
    }
    Ok((manifest, index, stats))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::{Cursor, Write};
    use std::sync::Arc;

    use antecedent_core::VERSION;

    use super::*;
    use crate::container::{ArtifactManifest, CompressPolicy, EncodedArtifact, pack_section};
    use crate::wire::{ArtifactKind, FormatVersion, ProvenanceWire, SemanticVersion};

    fn artifact_with_blob() -> EncodedArtifact {
        let meta = b"meta".to_vec();
        let blob = vec![0xEFu8; 48 * 1024];
        let (d0, s0) =
            pack_section("meta", "application/octet-stream", meta, CompressPolicy::Never);
        let (d1, s1) =
            pack_section("blob", "application/octet-stream", blob, CompressPolicy::Never);
        EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: FormatVersion { major: 0, minor: 2 },
                minimum_reader_version: FormatVersion { major: 0, minor: 2 },
                artifact_kind: ArtifactKind::Other("test".into()),
                library_version: SemanticVersion::from_crate_version(VERSION).unwrap(),
                artifact_id: "reader-test".into(),
                sections: vec![d0, d1],
                provenance: ProvenanceWire { note: "t".into() },
            },
            sections: vec![s0, s1],
        }
    }

    #[test]
    fn open_seek_loads_meta_only() {
        let art = artifact_with_blob();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let mut reader = ArtifactReader::open_seek(Cursor::new(buf)).unwrap();
        assert_eq!(reader.index().len(), 2);
        assert_eq!(reader.stats().sections_skipped, 2);
        let meta = reader.load_section("meta").unwrap();
        assert_eq!(meta.as_bytes(), b"meta");
        assert_eq!(reader.stats().sections_loaded, 1);
        assert!(reader.stats().bytes_loaded < 48 * 1024);
    }

    #[test]
    fn repeated_loads_keep_unvisited_sections_skipped() {
        let mut bytes = Vec::new();
        artifact_with_blob().write_to(&mut bytes).unwrap();
        let mut seek = ArtifactReader::open_seek(Cursor::new(&bytes)).unwrap();
        let mut mapped = MappedArtifactReader::from_bytes(Arc::from(bytes.clone())).unwrap();
        for _ in 0..3 {
            seek.load_section("meta").unwrap();
            mapped.load_section("meta").unwrap();
            for stats in [seek.stats(), mapped.stats()] {
                assert_eq!(stats.sections_skipped, 1);
                assert_eq!(stats.bytes_skipped, 48 * 1024);
            }
        }
        assert_eq!(seek.stats().sections_loaded, 3);
        assert_eq!(mapped.stats().mmap_views, 3);
        seek.load_section("blob").unwrap();
        mapped.load_section("blob").unwrap();
        for stats in [seek.stats(), mapped.stats()] {
            assert_eq!(stats.sections_skipped, 0);
            assert_eq!(stats.bytes_skipped, 0);
            assert_eq!(stats.sections_loaded, 4);
        }
    }

    #[test]
    fn mmap_uncompressed_section() {
        let art = artifact_with_blob();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("causal-mmap-{}.bin", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            art.write_to(&mut f).unwrap();
        }
        let mut reader = MappedArtifactReader::open_path(&path).unwrap();
        let mapped = reader.load_section_mapped("blob").unwrap();
        assert_eq!(mapped.as_bytes().len(), 48 * 1024);
        assert_eq!(mapped.as_bytes()[0], 0xEF);
        // Zero-copy accounting: exactly one mapped view, no owned-load or
        // decompression counters disturbed (DESIGN rule 22).
        assert_eq!(reader.stats().mmap_views, 1);
        assert_eq!(reader.stats().decompressions, 0);
        // A second view of the same section reuses the memoized verification
        // (no re-hash) and still counts as a view with identical bytes.
        let again = reader.load_section_mapped("blob").unwrap();
        assert_eq!(again.as_bytes(), mapped.as_bytes());
        assert_eq!(reader.stats().mmap_views, 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_path_snapshot_survives_file_replacement() {
        let art = artifact_with_blob();
        let dir = std::env::temp_dir();
        let path = dir.join(format!("causal-owned-{}.bin", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            art.write_to(&mut f).unwrap();
        }
        let mut reader = MappedArtifactReader::open_path(&path).unwrap();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"not-an-artifact").unwrap();
        }
        let mapped = reader.load_section_mapped("blob").unwrap();
        assert_eq!(mapped.as_bytes().len(), 48 * 1024);
        assert_eq!(mapped.as_bytes()[0], 0xEF);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mapped_rejects_compressed() {
        let payload = vec![0u8; 16 * 1024];
        let (d0, s0) =
            pack_section("blob", "application/octet-stream", payload, CompressPolicy::Always);
        let art = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: FormatVersion { major: 0, minor: 2 },
                minimum_reader_version: FormatVersion { major: 0, minor: 2 },
                artifact_kind: ArtifactKind::Other("test".into()),
                library_version: SemanticVersion::from_crate_version(VERSION).unwrap(),
                artifact_id: "zstd".into(),
                sections: vec![d0],
                provenance: ProvenanceWire { note: "t".into() },
            },
            sections: vec![s0],
        };
        let dir = std::env::temp_dir();
        let path = dir.join(format!("causal-mmap-zstd-{}.bin", std::process::id()));
        {
            let mut f = File::create(&path).unwrap();
            art.write_to(&mut f).unwrap();
        }
        let mut reader = MappedArtifactReader::open_path(&path).unwrap();
        let err = reader.load_section_mapped("blob").unwrap_err();
        assert!(matches!(err, IoError::MappedCompressed { .. }));
        let shared = reader.load_section("blob").unwrap();
        assert_eq!(shared.as_bytes().len(), 16 * 1024);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn selective_read_matches_seek_meta() {
        let art = artifact_with_blob();
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let mut want = HashSet::new();
        want.insert("meta");
        let partial = EncodedArtifact::read_selective(buf.as_slice(), &want).unwrap();
        assert_eq!(partial.sections.len(), 1);
        assert_eq!(partial.sections[0].id, "meta");
    }

    /// Truncated on-wire payload for a compressed mapped section must return
    /// `Err` from `load_section` (io-6), not panic on the slice.
    #[test]
    fn mapped_load_section_truncated_compressed_returns_err() {
        let payload = vec![0u8; 16 * 1024];
        let (d0, s0) =
            pack_section("blob", "application/octet-stream", payload, CompressPolicy::Always);
        let art = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: FormatVersion { major: 0, minor: 2 },
                minimum_reader_version: FormatVersion { major: 0, minor: 2 },
                artifact_kind: ArtifactKind::Other("test".into()),
                library_version: SemanticVersion::from_crate_version(VERSION).unwrap(),
                artifact_id: "zstd-trunc".into(),
                sections: vec![d0],
                provenance: ProvenanceWire { note: "t".into() },
            },
            sections: vec![s0],
        };
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        // Leave the section length prefix intact so indexing succeeds, but shorten
        // the on-wire payload so `start + len` exceeds the backing slice.
        assert!(buf.len() > 16, "artifact must be large enough to truncate");
        buf.truncate(buf.len() - 8);
        let mut reader = MappedArtifactReader::from_bytes(Arc::from(buf)).unwrap();
        let err = reader.load_section("blob").unwrap_err();
        assert!(
            matches!(err, IoError::Io(_)),
            "truncated mapped compressed section must be Err, got {err:?}"
        );
    }
}
