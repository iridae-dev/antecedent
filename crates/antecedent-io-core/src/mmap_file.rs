//! `Mmap::map` for a read-only file.
//!
//! [`crate::reader::MappedArtifactReader::open_path_mapped`] and
//! [`crate::reader::MappedArtifactReader::from_mmap`] are the other `unsafe`
//! entry points; they call this wrapper and keep the same immutability contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs::File;

use memmap2::Mmap;

use crate::error::IoError;

/// Map a read-only file into memory.
///
/// # Safety
///
/// The file must not be modified or truncated by any process for the entire
/// lifetime of the returned map. Opening a descriptor read-only does not
/// establish that obligation.
///
/// # Errors
///
/// OS mmap failures.
pub(crate) unsafe fn map_file_readonly(file: &File) -> Result<Mmap, IoError> {
    // SAFETY: caller upholds the file-immutability contract of `Mmap::map`.
    unsafe { Mmap::map(file) }.map_err(IoError::from)
}
