//! Artifact format migration registry (DESIGN.md §24.3).
//!
//! Supported durable formats: `0.1` (migrate-from) and `0.2` (stable).
//! Unknown versions fail explicitly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::container::{
    CompressPolicy, EncodedArtifact, SectionBytes, section_descriptor_with_policy,
};
use crate::convert::{from_cbor, schema_wire_from_v01, to_cbor};
use crate::error::IoError;
use crate::wire::{FormatVersion, SchemaWire, SchemaWireV01};

/// Frozen stable format for durable artifacts.
pub const STABLE_FORMAT: FormatVersion = FormatVersion { major: 0, minor: 2 };

/// Formats this reader can migrate *from* into [`STABLE_FORMAT`].
pub const SUPPORTED_SOURCE_FORMATS: &[FormatVersion] = &[
    FormatVersion { major: 0, minor: 1 },
    FormatVersion { major: 0, minor: 2 },
];

/// True when `v` is a known source format.
#[must_use]
pub fn is_supported_source(v: FormatVersion) -> bool {
    SUPPORTED_SOURCE_FORMATS.iter().any(|s| *s == v)
}

/// Migrate an encoded artifact's format version toward [`STABLE_FORMAT`].
///
/// # Errors
///
/// Unsupported source format versions.
pub fn migrate_artifact(mut artifact: EncodedArtifact) -> Result<EncodedArtifact, IoError> {
    let from = artifact.manifest.format_version;
    if !is_supported_source(from) {
        return Err(IoError::UnsupportedFormat { major: from.major, minor: from.minor });
    }
    if artifact.manifest.minimum_reader_version.major > STABLE_FORMAT.major
        || (artifact.manifest.minimum_reader_version.major == STABLE_FORMAT.major
            && artifact.manifest.minimum_reader_version.minor > STABLE_FORMAT.minor)
    {
        return Err(IoError::UnsupportedFormat {
            major: artifact.manifest.minimum_reader_version.major,
            minor: artifact.manifest.minimum_reader_version.minor,
        });
    }
    if from == STABLE_FORMAT {
        return Ok(artifact);
    }
    if from == (FormatVersion { major: 0, minor: 1 }) {
        artifact = migrate_0_1_to_0_2(artifact)?;
    }
    artifact.manifest.format_version = STABLE_FORMAT;
    artifact.manifest.minimum_reader_version = STABLE_FORMAT;
    Ok(artifact)
}

fn migrate_0_1_to_0_2(mut artifact: EncodedArtifact) -> Result<EncodedArtifact, IoError> {
    for (desc, sec) in artifact.manifest.sections.iter_mut().zip(artifact.sections.iter_mut()) {
        if desc.id == "schema" {
            // Prefer already-v2 decode (`variables`); else upgrade skinny v01 (`variable_names`).
            // The two wire shapes require different fields, so they do not cross-decode.
            let upgraded = match from_cbor::<SchemaWire>(&sec.data) {
                Ok(w) => w,
                Err(_) => {
                    let v01: SchemaWireV01 = from_cbor(&sec.data)?;
                    schema_wire_from_v01(&v01)
                }
            };
            let bytes = to_cbor(&upgraded)?;
            *desc = section_descriptor_with_policy(
                desc.id.clone(),
                desc.content_type.clone(),
                &bytes,
                CompressPolicy::Auto,
            );
            desc.logical_schema = "schema.v2".into();
            *sec = SectionBytes { id: sec.id.clone(), data: bytes };
        }
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
        let payload = to_cbor(&"release").unwrap();
        let desc = section_descriptor("note", "application/cbor", &payload);
        EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: fmt,
                minimum_reader_version: FormatVersion { major: 0, minor: 1 },
                artifact_kind: ArtifactKind::Other("note".into()),
                library_version: SemanticVersion::from_crate_version(VERSION)
                    .expect("CARGO_PKG_VERSION"),
                artifact_id: "migrate-test".into(),
                sections: vec![desc],
                provenance: ProvenanceWire { note: "migrate".into() },
            },
            sections: vec![SectionBytes { id: "note".into(), data: payload }],
        }
    }

    #[test]
    fn identity_migrate_0_2() {
        let art = tiny_artifact(STABLE_FORMAT);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let migrated = read_and_migrate(buf.as_slice()).unwrap();
        assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
        assert_eq!(migrated.sections[0].data, art.sections[0].data);
    }

    #[test]
    fn migrate_0_1_schema_to_0_2() {
        let v01 = SchemaWireV01 { variable_names: vec!["x".into(), "y".into()] };
        let payload = to_cbor(&v01).unwrap();
        let desc = section_descriptor("schema", "application/cbor", &payload);
        let art = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: FormatVersion { major: 0, minor: 1 },
                minimum_reader_version: FormatVersion { major: 0, minor: 1 },
                artifact_kind: ArtifactKind::SchemaGraph,
                library_version: SemanticVersion::from_crate_version(VERSION).unwrap(),
                artifact_id: "schema-migrate".into(),
                sections: vec![desc],
                provenance: ProvenanceWire { note: "t".into() },
            },
            sections: vec![SectionBytes { id: "schema".into(), data: payload }],
        };
        let migrated = migrate_artifact(art).unwrap();
        assert_eq!(migrated.manifest.format_version, STABLE_FORMAT);
        let schema: SchemaWire = from_cbor(&migrated.sections[0].data).unwrap();
        assert_eq!(schema.variable_names(), vec!["x".to_string(), "y".to_string()]);
        assert!(matches!(
            schema.variables[0].value_type,
            crate::wire::ValueTypeWire::Continuous
        ));
    }

    #[test]
    fn migrate_preserves_empty_schema_v2() {
        // Empty `variables` is valid v2 and must not be mistaken for skinny v01.
        let payload = to_cbor(&SchemaWire { variables: vec![] }).unwrap();
        let desc = section_descriptor("schema", "application/cbor", &payload);
        let art = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: FormatVersion { major: 0, minor: 1 },
                minimum_reader_version: FormatVersion { major: 0, minor: 1 },
                artifact_kind: ArtifactKind::SchemaGraph,
                library_version: SemanticVersion::from_crate_version(VERSION).unwrap(),
                artifact_id: "empty-schema".into(),
                sections: vec![desc],
                provenance: ProvenanceWire { note: "t".into() },
            },
            sections: vec![SectionBytes { id: "schema".into(), data: payload.clone() }],
        };
        let migrated = migrate_artifact(art).unwrap();
        let schema: SchemaWire = from_cbor(&migrated.sections[0].data).unwrap();
        assert!(schema.variables.is_empty());
    }

    #[test]
    fn rejects_unknown_format() {
        let art = tiny_artifact(FormatVersion { major: 9, minor: 9 });
        let err = migrate_artifact(art).unwrap_err();
        assert!(matches!(err, IoError::UnsupportedFormat { major: 9, minor: 9 }));
    }
}
