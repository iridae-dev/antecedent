//! Arrow IPC section helpers for the artifact container .
//!
//! Large numerical payloads use Arrow IPC bytes rather than CBOR arrays
//!. This module records the section content type and wraps
//! opaque IPC payloads; decoding is performed by Arrow adapters at the boundary.
//!
//! Arrow IPC sections use [`CompressPolicy::Never`] so they remain mmap-eligible
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::container::{CompressPolicy, SectionBytes, pack_section, pack_section_shared};
use crate::wire::SectionDescriptor;

/// Content type for Arrow IPC file payloads inside an artifact section.
pub const ARROW_IPC_CONTENT_TYPE: &str = "application/vnd.apache.arrow.file";

/// Wrap opaque Arrow IPC bytes as a named artifact section (uncompressed).
#[must_use]
pub fn arrow_ipc_section(
    id: impl Into<String>,
    ipc_bytes: Vec<u8>,
) -> (SectionDescriptor, SectionBytes) {
    let (mut desc, sec) =
        pack_section(id, ARROW_IPC_CONTENT_TYPE, ipc_bytes, CompressPolicy::Never);
    desc.logical_schema = "arrow.ipc.v1".into();
    (desc, sec)
}

/// Wrap a shared Arrow IPC buffer without cloning payload bytes.
#[must_use]
pub fn arrow_ipc_section_shared(
    id: impl Into<String>,
    ipc_bytes: Arc<[u8]>,
) -> (SectionDescriptor, SectionBytes) {
    let (mut desc, sec) =
        pack_section_shared(id, ARROW_IPC_CONTENT_TYPE, ipc_bytes, CompressPolicy::Never);
    desc.logical_schema = "arrow.ipc.v1".into();
    (desc, sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_section_uses_ipc_content_type_uncompressed() {
        let payload = b"ARROW1\0\0".to_vec();
        let (desc, sec) = arrow_ipc_section("tabular", payload.clone());
        assert_eq!(desc.content_type, ARROW_IPC_CONTENT_TYPE);
        assert_eq!(&*sec.data, payload.as_slice());
        assert_eq!(desc.id, "tabular");
        assert!(desc.compression.is_none());
    }

    #[test]
    fn arrow_section_shared_reuses_arc() {
        let payload: Arc<[u8]> = Arc::from(b"ARROW1\0\0".as_slice());
        let (_, sec) = arrow_ipc_section_shared("tabular", Arc::clone(&payload));
        assert!(std::sync::Arc::ptr_eq(&sec.data, &payload));
    }
}
