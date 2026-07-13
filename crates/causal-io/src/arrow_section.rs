//! Arrow IPC section helpers for the artifact container (Phase 0 skeleton).
//!
//! Large numerical payloads use Arrow IPC bytes rather than CBOR arrays
//! (DESIGN.md §24). This module records the section content type and wraps
//! opaque IPC payloads; decoding is performed by Arrow adapters at the boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::container::{SectionBytes, section_descriptor};
use crate::wire::SectionDescriptor;

/// Content type for Arrow IPC file payloads inside an artifact section.
pub const ARROW_IPC_CONTENT_TYPE: &str = "application/vnd.apache.arrow.file";

/// Wrap opaque Arrow IPC bytes as a named artifact section.
#[must_use]
pub fn arrow_ipc_section(
    id: impl Into<String>,
    ipc_bytes: Vec<u8>,
) -> (SectionDescriptor, SectionBytes) {
    let id = id.into();
    let desc = section_descriptor(id.clone(), ARROW_IPC_CONTENT_TYPE, &ipc_bytes);
    (desc, SectionBytes { id, data: ipc_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_section_uses_ipc_content_type() {
        let payload = b"ARROW1\0\0".to_vec();
        let (desc, sec) = arrow_ipc_section("tabular", payload.clone());
        assert_eq!(desc.content_type, ARROW_IPC_CONTENT_TYPE);
        assert_eq!(sec.data, payload);
        assert_eq!(desc.id, "tabular");
    }
}
