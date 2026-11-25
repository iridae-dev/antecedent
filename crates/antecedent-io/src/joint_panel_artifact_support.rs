//! Common admission and encoding helpers for frozen panel artifacts.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;

use crate::joint_panel_artifact::{MAX_JOINT_PANEL_ARTIFACT_BYTES, PanelArtifactError};
use serde::Serialize;

pub(crate) fn invalid(error: impl std::fmt::Display) -> PanelArtifactError {
    PanelArtifactError::Invalid(error.to_string())
}
pub(crate) fn cancelled(ctx: &ExecutionContext) -> Result<(), PanelArtifactError> {
    if ctx.cancellation.is_cancelled() { Err(PanelArtifactError::Cancelled) } else { Ok(()) }
}
pub(crate) fn admit(
    ctx: &ExecutionContext,
    bytes: u64,
) -> Result<AdmissionLease, PanelArtifactError> {
    crate::artifact_budget::admit_bytes(ctx, bytes).map_err(PanelArtifactError::Resource)
}
pub(crate) struct Counter(pub(crate) usize);
impl std::io::Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("byte count overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) const PROGRAM_SECTION: &str = "panel.program";
pub(super) const DRAWS_SECTION: &str = "posterior.draws";

pub(super) fn write_sections<'a>(
    header: &crate::ArtifactContainer<'_>,
    metadata_section: &str,
    metadata: &impl Serialize,
    program: &impl Serialize,
    mut draws: impl Iterator<Item = &'a f64>,
    draw_count: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<u8>, PanelArtifactError> {
    let draw_bytes = draw_count
        .checked_mul(size_of::<f64>())
        .ok_or_else(|| invalid("panel draw byte count overflow"))?;
    let mut metadata_size = Counter(0);
    serde_json::to_writer(&mut metadata_size, metadata).map_err(invalid)?;
    let mut program_size = Counter(0);
    ciborium::into_writer(program, &mut program_size).map_err(invalid)?;
    let logical = metadata_size
        .0
        .checked_add(program_size.0)
        .and_then(|bytes| bytes.checked_add(draw_bytes))
        .ok_or_else(|| invalid("panel serialization quote overflow"))?;
    if logical > header.max_bytes {
        return Err(invalid("panel payload exceeds byte ceiling"));
    }
    let _lease = admit(
        ctx,
        (logical as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("panel serialization quote overflow"))?,
    )?;
    let mut head = Vec::with_capacity(metadata_size.0);
    serde_json::to_writer(&mut head, metadata).map_err(invalid)?;
    let mut body = Vec::with_capacity(program_size.0);
    ciborium::into_writer(program, &mut body).map_err(invalid)?;
    let mut draws_le = Vec::with_capacity(draw_bytes);
    for draw in draws.by_ref().take(draw_count) {
        draws_le.extend_from_slice(&draw.to_le_bytes());
    }
    if draws_le.len() != draw_bytes || draws.next().is_some() {
        return Err(invalid("panel draw count changed while writing"));
    }
    cancelled(ctx)?;
    crate::encode_container(
        header,
        &[
            crate::OutSection {
                id: metadata_section,
                content_type: crate::CONTENT_BYTES,
                bytes: &head,
            },
            crate::OutSection {
                id: PROGRAM_SECTION,
                content_type: crate::CONTENT_CBOR,
                bytes: &body,
            },
            crate::OutSection {
                id: DRAWS_SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: &draws_le,
            },
        ],
    )
    .map_err(|error| PanelArtifactError::Io(error.into_io("panel encoding")))
}

/// Hash the unchanged length-prefixed canonical JSON without materializing a JSON buffer.
pub(super) fn canonical_json_digest(
    tag: &str,
    value: &impl Serialize,
) -> Result<([u8; 32], usize), PanelArtifactError> {
    let mut count = Counter(0);
    serde_json::to_writer(&mut count, value).map_err(invalid)?;
    // CanonicalWriter::digest_bytes frames the domain and then the serialized byte string.
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(tag.len() as u64).to_le_bytes());
    hasher.update(tag.as_bytes());
    hasher.update(&(count.0 as u64).to_le_bytes());
    let mut writer = std::io::BufWriter::with_capacity(4096, hasher);
    serde_json::to_writer(&mut writer, value).map_err(invalid)?;
    let hasher = writer.into_inner().map_err(invalid)?;
    Ok((*hasher.finalize().as_bytes(), count.0))
}

pub(super) fn read_sections(
    kind: &str,
    legacy_section: &str,
    metadata_section: &str,
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<(AdmissionLease, Vec<Vec<u8>>), PanelArtifactError> {
    cancelled(ctx)?;
    if bytes.len() > MAX_JOINT_PANEL_ARTIFACT_BYTES {
        return Err(invalid("panel input exceeds ceiling"));
    }
    let reader = crate::ArtifactReader::open_seek(std::io::Cursor::new(bytes))?;
    let sections = &reader.manifest().sections;
    let logical = sections.iter().try_fold(0u64, |sum, section| {
        sum.checked_add(section.uncompressed_size)
            .ok_or_else(|| invalid("panel decode quote overflow"))
    })?;
    if logical > MAX_JOINT_PANEL_ARTIFACT_BYTES as u64 {
        return Err(invalid("panel decoded payload exceeds ceiling"));
    }
    let lease =
        admit(ctx, logical.checked_mul(8).ok_or_else(|| invalid("panel decode quote overflow"))?)?;
    let section_ids: &[&str] =
        if sections.first().is_some_and(|section| section.id == legacy_section) {
            &[legacy_section]
        } else {
            &[metadata_section, PROGRAM_SECTION, DRAWS_SECTION]
        };
    let sections =
        crate::decode_container(kind, MAX_JOINT_PANEL_ARTIFACT_BYTES, section_ids, bytes)
            .map_err(|error| PanelArtifactError::Io(error.into_io("panel decoding")))?;
    Ok((lease, sections))
}

pub(super) fn matches_draw_bytes<'a>(bytes: &[u8], draws: impl Iterator<Item = &'a f64>) -> bool {
    let (words, tail) = bytes.as_chunks::<8>();
    tail.is_empty()
        && words.iter().copied().map(u64::from_le_bytes).eq(draws.map(|draw| draw.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;

    #[test]
    fn panel_admission_holds_and_releases_its_quote() {
        let pool = AdmissionPool::fixed(1, 32);
        let ctx = ExecutionContext::for_tests(0).with_admission(pool.clone());
        let lease = admit(&ctx, 24).unwrap();
        assert_eq!(pool.available().bytes(), 8);
        assert!(matches!(admit(&ctx, 9), Err(PanelArtifactError::Resource(_))));
        drop(lease);
        assert_eq!(pool.available().bytes(), 32);
        ctx.cancellation.cancel();
        assert!(matches!(cancelled(&ctx), Err(PanelArtifactError::Cancelled)));
    }

    #[test]
    fn panel_artifact_sections_preserve_ieee_draw_words() {
        let ctx = ExecutionContext::for_tests(1);
        let metadata = serde_json::json!({"present": [true, false, true]});
        let draws = [-0.0, 1.5, 7.25];
        let bytes = write_sections(
            &crate::ArtifactContainer {
                kind: "panel_test",
                note: "panel_section_test",
                max_bytes: MAX_JOINT_PANEL_ARTIFACT_BYTES,
                artifact_id: "panel_test",
                library_version: env!("CARGO_PKG_VERSION"),
            },
            "panel_test_v2",
            &metadata,
            &["frozen_program"],
            draws.iter(),
            draws.len(),
            &ctx,
        )
        .unwrap();
        let (_lease, sections) =
            read_sections("panel_test", "panel_test_v1", "panel_test_v2", &bytes, &ctx).unwrap();
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&sections[0]).unwrap(), metadata);
        assert_eq!(
            antecedent_io_core::from_cbor::<[String; 1]>(&sections[1]).unwrap(),
            ["frozen_program"]
        );
        assert_eq!(sections[2].len(), std::mem::size_of_val(&draws));
        assert!(matches_draw_bytes(&sections[2], draws.iter()));
        assert!(!matches_draw_bytes(&sections[2], [0.0, 1.5, 7.25].iter()));
    }

    #[test]
    fn streamed_panel_checksum_keeps_the_legacy_json_frame() {
        let tag = "antecedent.joint_panel.frozen_program.v1";
        let value = serde_json::json!({"rows": [8, -0.0, 3.1], "label": "滞後"});
        let serialized = serde_json::to_vec(&value).unwrap();
        let (digest, size) = canonical_json_digest(tag, &value).unwrap();
        assert_eq!(size, serialized.len());
        assert_eq!(digest, antecedent_core::CanonicalWriter::digest_bytes(tag, &serialized));
    }
}
