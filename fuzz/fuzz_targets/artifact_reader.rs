//! Fuzz the byte-slice artifact reader.
//!
//! `mmap_file::map_file_readonly` only maps a filesystem `File`. This target
//! uses `MappedArtifactReader::from_bytes`, which indexes the same container
//! framing from an owned snapshot and does not touch the filesystem.
//!
//! A truncated snapshot is not an index. Dropping one reader does not drop the
//! snapshot another reader still holds.
#![no_main]

use std::sync::Arc;

use antecedent_core::VERSION;
use antecedent_io::{
    ArtifactKind, ArtifactManifest, CompressPolicy, EncodedArtifact, IoError, MAGIC,
    MappedArtifactReader, ProvenanceWire, STABLE_FORMAT, SemanticVersion, pack_section,
};
use libfuzzer_sys::fuzz_target;

fn uncompressed_note() -> Vec<u8> {
    let (desc, section) =
        pack_section("note", "application/octet-stream", b"note".to_vec(), CompressPolicy::Never);
    let artifact = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: STABLE_FORMAT,
            minimum_reader_version: STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other("note".into()),
            library_version: SemanticVersion::from_crate_version(VERSION).expect("crate version"),
            artifact_id: "fuzz-note".into(),
            sections: vec![desc],
            provenance: ProvenanceWire { note: "fuzz".into() },
        },
        sections: vec![section],
    };
    let mut buf = Vec::new();
    artifact.write_to(&mut buf).expect("tiny container");
    buf
}

/// The index agrees with the manifest, and the section view outlives the first reader.
fn snapshot_outlives_the_reader() {
    let bytes = uncompressed_note();
    let snapshot = Arc::<[u8]>::from(bytes);
    let mut reader = MappedArtifactReader::from_bytes(Arc::clone(&snapshot)).expect("index");
    assert_eq!(reader.index().len(), reader.manifest().sections.len());
    let section = reader.load_section_mapped("note").expect("section");
    assert_eq!(section.as_bytes(), b"note");
    drop(section);
    drop(reader);
    let again = MappedArtifactReader::from_bytes(snapshot).expect("snapshot still indexed");
    assert_eq!(again.index()[0].id, "note");

    let mut truncated = uncompressed_note();
    truncated.pop();
    assert!(MappedArtifactReader::from_bytes(Arc::from(truncated)).is_err());
    assert!(matches!(
        MappedArtifactReader::from_bytes(Arc::from(*MAGIC)),
        Err(IoError::BadMagic) | Err(IoError::Io(_))
    ));
}

fuzz_target!(|data: &[u8]| {
    snapshot_outlives_the_reader();
    if let Ok(reader) = MappedArtifactReader::from_bytes(Arc::<[u8]>::from(data)) {
        assert_eq!(reader.index().len(), reader.manifest().sections.len());
        for (entry, descriptor) in reader.index().iter().zip(&reader.manifest().sections) {
            assert_eq!(entry.id, descriptor.id);
        }
    }
});
