//! Thin mmap wrapper — sole `unsafe` boundary in antecedent-io.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs::File;

use memmap2::Mmap;

use crate::error::IoError;

/// Map a read-only file into memory.
///
/// # Errors
///
/// OS mmap failures.
pub(crate) fn map_file_readonly(file: &File) -> Result<Mmap, IoError> {
    // SAFETY: `file` is opened read-only for the lifetime of the map. Callers
    // must not write to the underlying path while the `Mmap` is live. This is
    // the documented contract of `memmap2::Mmap::map`.
    unsafe { Mmap::map(file) }.map_err(|e| IoError::Io(e.to_string()))
}
