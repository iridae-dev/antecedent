//! Artifact format migration registry (DESIGN.md §24.3 / Phase 12).
//!
//! Supported durable format is `0.1`. Identity migration is the only path today;
//! unknown versions fail explicitly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::container::EncodedArtifact;
use crate::error::IoError;
use crate::wire::FormatVersion;

/// Frozen stable format for 1.0 preparation (library stays at package 0.1.0).
pub const STABLE_FORMAT: FormatVersion = FormatVersion { major: 0, minor: 1 };

/// Formats this reader can migrate *from* into [`STABLE_FORMAT`].
pub const SUPPORTED_SOURCE_FORMATS: &[FormatVersion] = &[STABLE_FORMAT];

/// True when `v` is a known source format.
#[must_use]
pub fn is_supported_source(v: FormatVersion) -> bool {
    SUPPORTED_SOURCE_FORMATS.iter().any(|s| *s == v)
}

/// Migrate an encoded artifact's format version toward [`STABLE_FORMAT`].
///
/// Today only identity `0.1 → 0.1` is defined. Payload bytes are preserved.
///
/// # Errors
///
/// Unsupported source format versions.
pub fn migrate_artifact(mut artifact: EncodedArtifact) -> Result<EncodedArtifact, IoError> {
    let from = artifact.manifest.format_version;
    if !is_supported_source(from) {
        return Err(IoError::UnsupportedFormat { major: from.major, minor: from.minor });
    }
    // Identity for 0.1; future minors chain here.
    artifact.manifest.format_version = STABLE_FORMAT;
    if artifact.manifest.minimum_reader_version.major > STABLE_FORMAT.major
        || (artifact.manifest.minimum_reader_version.major == STABLE_FORMAT.major
            && artifact.manifest.minimum_reader_version.minor > STABLE_FORMAT.minor)
    {
        return Err(IoError::UnsupportedFormat {
            major: artifact.manifest.minimum_reader_version.major,
            minor: artifact.manifest.minimum_reader_version.minor,
        });
    }
    Ok(artifact)
}

/// Read a container and migrate to [`STABLE_FORMAT`].
///
/// # Errors
///
/// IO, container, or migration failures.
pub fn read_and_migrate<R: std::io::Read>(r: R) -> Result<EncodedArtifact, IoError> {
    let artifact = EncodedArtifact::read_from(r)?;
    migrate_artifact(artifact)
}

#[cfg(test)]
mod tests {
    use causal_core::VERSION;

    use super::*;
    use crate::container::{ArtifactManifest, SectionBytes, section_descriptor};
    use crate::convert::to_cbor;
    use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

    fn tiny_artifact(fmt: FormatVersion) -> EncodedArtifact {
        let payload = to_cbor(&"phase12").unwrap();
        let desc = section_descriptor("note", "application/cbor", &payload);
        EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: fmt,
                minimum_reader_version: FormatVersion { major: 0, minor: 1 },
                artifact_kind: ArtifactKind::Other("note".into()),
                library_version: SemanticVersion::from_crate_version(VERSION),
                artifact_id: "migrate-test".into(),
                sections: vec![desc],
                provenance: ProvenanceWire { note: "migrate".into() },
            },
            sections: vec![SectionBytes { id: "note".into(), data: payload }],
        }
    }

    #[test]
    fn identity_migrate_0_1() {
        let art = tiny_artifact(STABLE_FORMAT);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let migrated = read_and_migrate(buf.as_slice()).unwrap();
        assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
        assert_eq!(migrated.sections[0].data, art.sections[0].data);
    }

    #[test]
    fn rejects_unknown_format() {
        let art = tiny_artifact(FormatVersion { major: 9, minor: 9 });
        let err = migrate_artifact(art).unwrap_err();
        assert!(matches!(err, IoError::UnsupportedFormat { major: 9, minor: 9 }));
    }
}
