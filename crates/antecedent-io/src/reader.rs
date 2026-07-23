//! Selective artifact reader with skip / mmap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::container::{
    ArtifactManifest, EncodedArtifact, SectionBytes, decode_on_wire_arc, read_header_and_manifest,
};
use crate::error::IoError;
use crate::mmap_file::map_file_readonly;
use crate::wire::SectionDescriptor;

const MAX_SECTION_BYTES: usize = 512 * 1024 * 1024;

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

/// Zero-copy view into an mmap'd uncompressed section.
#[derive(Clone, Debug)]
pub struct MappedSection {
    mmap: Arc<Mmap>,
    start: usize,
    len: usize,
    /// Section id.
    pub id: String,
}

impl MappedSection {
    /// Borrow the on-wire (== logical, uncompressed) bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[self.start..self.start + self.len]
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
    inner: R,
    stats: SectionLoadStats,
}

impl<R: Read + Seek> ArtifactReader<R> {
    /// Open a seekable source: parse manifest and record section offsets without loading payloads.
    ///
    /// # Errors
    ///
    /// IO / framing / CBOR failures.
    pub fn open_seek(mut inner: R) -> Result<Self, IoError> {
        let (manifest, index, mut stats) = index_seekable(&mut inner)?;
        stats.bytes_skipped = index.iter().map(|e| u64::from(e.on_wire_len)).sum();
        stats.sections_skipped = index.len() as u64;
        Ok(Self { manifest, index, inner, stats })
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
        let (entry, desc) = {
            let (e, d) = self.lookup(id)?;
            (e.clone(), d.clone())
        };
        self.inner
            .seek(SeekFrom::Start(entry.file_offset))
            .map_err(|e| IoError::Io(e.to_string()))?;
        let len = usize::try_from(entry.on_wire_len).map_err(|_| IoError::TooLarge)?;
        if len > MAX_SECTION_BYTES {
            return Err(IoError::TooLarge);
        }
        let mut on_wire = vec![0u8; len];
        self.inner.read_exact(&mut on_wire).map_err(|e| IoError::Io(e.to_string()))?;
        let hash = blake3::hash(&on_wire);
        if hash.as_bytes() != &desc.blake3 {
            return Err(IoError::ChecksumMismatch { section: id.into() });
        }
        let (logical, decompressed) =
            decode_on_wire_arc(&on_wire, desc.compression.as_deref(), &desc.id)?;
        let expected = usize::try_from(desc.uncompressed_size).map_err(|_| IoError::TooLarge)?;
        if logical.len() != expected {
            return Err(IoError::Decompress {
                section: desc.id.clone(),
                message: format!("logical size {} != uncompressed_size {expected}", logical.len()),
            });
        }
        self.note_loaded(entry.on_wire_len, decompressed);
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
            sections.push(SectionBytes { id, data: access.into_shared() });
        }
        Ok(EncodedArtifact { manifest: self.manifest, sections })
    }

    fn lookup(&self, id: &str) -> Result<(&SectionIndexEntry, &SectionDescriptor), IoError> {
        let pos = self
            .index
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| IoError::Convert(format!("unknown section `{id}`")))?;
        Ok((&self.index[pos], &self.manifest.sections[pos]))
    }

    fn note_loaded(&mut self, on_wire_len: u32, decompressed: bool) {
        self.stats.sections_loaded += 1;
        self.stats.bytes_loaded += u64::from(on_wire_len);
        if decompressed {
            self.stats.decompressions += 1;
        }
        if self.stats.sections_skipped > 0 {
            self.stats.sections_skipped = self.stats.sections_skipped.saturating_sub(1);
            self.stats.bytes_skipped =
                self.stats.bytes_skipped.saturating_sub(u64::from(on_wire_len));
        }
    }
}

/// Memory-mapped artifact reader (zero-copy uncompressed sections).
pub struct MappedArtifactReader {
    mmap: Arc<Mmap>,
    manifest: ArtifactManifest,
    index: Vec<SectionIndexEntry>,
    stats: SectionLoadStats,
}

impl MappedArtifactReader {
    /// Memory-map `path` and index sections without copying payloads.
    ///
    /// # Errors
    ///
    /// IO, bad magic, or manifest errors.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let file = File::open(path.as_ref()).map_err(|e| IoError::Io(e.to_string()))?;
        let mmap = Arc::new(map_file_readonly(&file)?);
        Self::from_mmap(mmap)
    }

    /// Index an already-mapped buffer.
    ///
    /// # Errors
    ///
    /// Bad magic, version, CBOR, or section framing.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self, IoError> {
        let mut cursor = Cursor::new(mmap.as_ref());
        let (manifest, index, mut stats) = index_seekable(&mut cursor)?;
        stats.bytes_skipped = index.iter().map(|e| u64::from(e.on_wire_len)).sum();
        stats.sections_skipped = index.len() as u64;
        Ok(Self { mmap, manifest, index, stats })
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
        let (entry, desc) = {
            let (e, d) = self.lookup(id)?;
            (e.clone(), d.clone())
        };
        if desc.compression.is_some() {
            return Err(IoError::MappedCompressed { section: id.into() });
        }
        let start = usize::try_from(entry.file_offset).map_err(|_| IoError::TooLarge)?;
        let len = usize::try_from(entry.on_wire_len).map_err(|_| IoError::TooLarge)?;
        if start.checked_add(len).is_none_or(|end| end > self.mmap.len()) {
            return Err(IoError::Io("mmap section range out of bounds".into()));
        }
        let on_wire = &self.mmap[start..start + len];
        let hash = blake3::hash(on_wire);
        if hash.as_bytes() != &desc.blake3 {
            return Err(IoError::ChecksumMismatch { section: id.into() });
        }
        let expected = usize::try_from(desc.uncompressed_size).map_err(|_| IoError::TooLarge)?;
        if on_wire.len() != expected {
            return Err(IoError::ManifestMismatch { message: "uncompressed mmap size mismatch" });
        }
        self.stats.mmap_views += 1;
        self.stats.sections_loaded += 1;
        self.stats.bytes_loaded += u64::from(entry.on_wire_len);
        if self.stats.sections_skipped > 0 {
            self.stats.sections_skipped = self.stats.sections_skipped.saturating_sub(1);
            self.stats.bytes_skipped =
                self.stats.bytes_skipped.saturating_sub(u64::from(entry.on_wire_len));
        }
        Ok(MappedSection { mmap: Arc::clone(&self.mmap), start, len, id: id.into() })
    }

    /// Load logical bytes (decompress when needed). Copies on-wire into a heap buffer.
    ///
    /// # Errors
    ///
    /// Unknown section, checksum, or decompress failures.
    pub fn load_section(&mut self, id: &str) -> Result<SectionAccess, IoError> {
        let (entry, desc) = {
            let (e, d) = self.lookup(id)?;
            (e.clone(), d.clone())
        };
        if desc.compression.is_none() {
            return Ok(SectionAccess::Mapped(self.load_section_mapped(id)?));
        }
        let start = usize::try_from(entry.file_offset).map_err(|_| IoError::TooLarge)?;
        let len = usize::try_from(entry.on_wire_len).map_err(|_| IoError::TooLarge)?;
        let on_wire = &self.mmap[start..start + len];
        let hash = blake3::hash(on_wire);
        if hash.as_bytes() != &desc.blake3 {
            return Err(IoError::ChecksumMismatch { section: id.into() });
        }
        let (logical, decompressed) =
            decode_on_wire_arc(on_wire, desc.compression.as_deref(), &desc.id)?;
        let expected = usize::try_from(desc.uncompressed_size).map_err(|_| IoError::TooLarge)?;
        if logical.len() != expected {
            return Err(IoError::Decompress {
                section: desc.id.clone(),
                message: format!("logical size {} != uncompressed_size {expected}", logical.len()),
            });
        }
        self.stats.sections_loaded += 1;
        self.stats.bytes_loaded += u64::from(entry.on_wire_len);
        if decompressed {
            self.stats.decompressions += 1;
        }
        if self.stats.sections_skipped > 0 {
            self.stats.sections_skipped = self.stats.sections_skipped.saturating_sub(1);
            self.stats.bytes_skipped =
                self.stats.bytes_skipped.saturating_sub(u64::from(entry.on_wire_len));
        }
        Ok(SectionAccess::Shared(logical))
    }

    fn lookup(&self, id: &str) -> Result<(&SectionIndexEntry, &SectionDescriptor), IoError> {
        let pos = self
            .index
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| IoError::Convert(format!("unknown section `{id}`")))?;
        Ok((&self.index[pos], &self.manifest.sections[pos]))
    }
}

fn index_seekable<R: Read + Seek>(
    r: &mut R,
) -> Result<(ArtifactManifest, Vec<SectionIndexEntry>, SectionLoadStats), IoError> {
    let (manifest, r) = read_header_and_manifest(r)?;
    let mut index = Vec::with_capacity(manifest.sections.len());
    let stats = SectionLoadStats::default();
    for desc in &manifest.sections {
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf).map_err(|e| IoError::Io(e.to_string()))?;
        let on_wire_len = u32::from_le_bytes(len_buf);
        let expected = u32::try_from(desc.compressed_size).map_err(|_| IoError::TooLarge)?;
        if on_wire_len != expected {
            return Err(IoError::ManifestMismatch { message: "section size mismatch" });
        }
        let file_offset = r.stream_position().map_err(|e| IoError::Io(e.to_string()))?;
        r.seek(SeekFrom::Current(i64::from(on_wire_len)))
            .map_err(|e| IoError::Io(e.to_string()))?;
        index.push(SectionIndexEntry { id: desc.id.clone(), file_offset, on_wire_len });
    }
    Ok((manifest, index, stats))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::Cursor;

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
}
