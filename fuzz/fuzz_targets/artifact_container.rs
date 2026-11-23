//! Fuzz artifact container deserialization (length-cap / no panic).
//!
//! A declared manifest length is not allocated, and a truncated container is
//! not a successful read.
#![no_main]

use antecedent_core::VERSION;
use antecedent_io::{
    ArtifactKind, ArtifactManifest, CONTAINER_VERSION, CompressPolicy, EncodedArtifact, IoError,
    MAGIC, ProvenanceWire, STABLE_FORMAT, SemanticVersion, pack_section,
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

/// Bad magic, a manifest length that does not fit, and a truncated payload refuse.
fn container_framing_refuses_a_short_declaration() {
    assert!(matches!(EncodedArtifact::read_from(b"notmagic".as_slice()), Err(IoError::BadMagic)));
    let mut huge = Vec::from(*MAGIC);
    huge.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
    huge.extend_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(EncodedArtifact::read_from(huge.as_slice()), Err(IoError::TooLarge)));

    let bytes = uncompressed_note();
    let decoded = EncodedArtifact::read_from(bytes.as_slice()).expect("round trip");
    assert_eq!(decoded.sections.len(), decoded.manifest.sections.len());
    assert_eq!(decoded.sections[0].id, "note");
    assert_eq!(&decoded.sections[0].data[..], b"note");
    assert_eq!(decoded.manifest.format_version, STABLE_FORMAT);

    let mut truncated = bytes;
    truncated.pop();
    assert!(EncodedArtifact::read_from(truncated.as_slice()).is_err());
}

fuzz_target!(|data: &[u8]| {
    container_framing_refuses_a_short_declaration();
    if let Ok(decoded) = EncodedArtifact::read_from(data) {
        assert_eq!(decoded.sections.len(), decoded.manifest.sections.len());
        for (section, descriptor) in decoded.sections.iter().zip(&decoded.manifest.sections) {
            assert_eq!(section.id, descriptor.id);
        }
    }
});
