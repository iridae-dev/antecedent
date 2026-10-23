//! Full-content identity of a file-backed table, computed while it streams.
//!
//! Each projected column has its own BLAKE3 hasher. Every row contributes a
//! presence byte and, when present, the value as the library will hold it:
//! the IEEE-754 bits of a float, a little-endian `i64` for integers and
//! nanosecond timestamps, one byte for a boolean, and the length-prefixed
//! label of a category. The byte stream does not depend on where chunks,
//! record batches, or row groups begin, so the identity is the same for every
//! chunk size, and any change to a value changes it. Container metadata (file
//! name, modification time, compression, row-group layout) is not hashed: two
//! files with the same rows and schema have one identity, and one file whose
//! content changed has another.
//!
//! The table identity binds the row count, the column names and kinds, each
//! category domain in order, and each column digest. It is issued only by a
//! reader that reached the end of a complete first scan; a partial or cancelled
//! scan has no identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write as _;
use std::sync::Arc;

use crate::global_index::GlobalId;

use super::FileColumnKind;

/// Version of the table identity encoding.
pub const FILE_IDENTITY_VERSION: u16 = 1;

const COLUMN_DOMAIN: &str = "antecedent.data.file_table.column.v1";
const TABLE_DOMAIN: &str = "antecedent.data.file_table.v1";

/// Bytes staged before one hasher update.
const STAGE: usize = 64 * 1024;

/// Streaming digest of one column's rows.
pub(crate) struct ColumnHasher {
    hasher: blake3::Hasher,
    staged: Vec<u8>,
}

impl ColumnHasher {
    pub(crate) fn new() -> Self {
        Self { hasher: blake3::Hasher::new_derive_key(COLUMN_DOMAIN), staged: Vec::new() }
    }

    fn flush(&mut self) {
        if !self.staged.is_empty() {
            self.hasher.update(&self.staged);
            self.staged.clear();
        }
    }

    fn put(&mut self, bytes: &[u8]) {
        if self.staged.len() + bytes.len() > STAGE {
            self.flush();
        }
        if bytes.len() >= STAGE {
            self.hasher.update(bytes);
        } else {
            if self.staged.capacity() == 0 {
                self.staged.reserve(STAGE);
            }
            self.staged.extend_from_slice(bytes);
        }
    }

    /// A missing cell.
    pub(crate) fn null(&mut self) {
        self.put(&[0]);
    }

    /// A present fixed-width value.
    pub(crate) fn value(&mut self, bytes: &[u8]) {
        self.put(&[1]);
        self.put(bytes);
    }

    /// A present category label.
    pub(crate) fn label(&mut self, label: &str) {
        self.put(&[1]);
        self.put(&(label.len() as u64).to_le_bytes());
        self.put(label.as_bytes());
    }

    /// Digest of every row seen.
    pub(crate) fn finish(mut self) -> [u8; 32] {
        self.flush();
        *self.hasher.finalize().as_bytes()
    }
}

/// One column's part of the table identity.
pub(crate) struct ColumnDigest<'a> {
    pub name: &'a str,
    pub kind: &'a FileColumnKind,
    /// Category labels in domain order. Empty for other kinds.
    pub levels: &'a [Arc<str>],
    pub digest: [u8; 32],
}

/// Full-content identity of a file-backed table.
///
/// Issued only after a complete scan. Equal identities mean equal row counts,
/// column names, kinds, category domains, and values, whatever the container.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FileContentIdentity {
    version: u16,
    rows: GlobalId,
    digest: [u8; 32],
}

impl FileContentIdentity {
    pub(crate) fn seal(rows: GlobalId, columns: &[ColumnDigest<'_>]) -> Self {
        let mut hasher = blake3::Hasher::new_derive_key(TABLE_DOMAIN);
        hasher.update(&FILE_IDENTITY_VERSION.to_le_bytes());
        hasher.update(&rows.get().to_le_bytes());
        hasher.update(&(columns.len() as u64).to_le_bytes());
        for column in columns {
            text(&mut hasher, column.name);
            match column.kind {
                FileColumnKind::Float64 => {
                    hasher.update(&[1]);
                }
                FileColumnKind::Int64 => {
                    hasher.update(&[2]);
                }
                FileColumnKind::Boolean => {
                    hasher.update(&[3]);
                }
                FileColumnKind::Timestamp { timezone } => {
                    hasher.update(&[4]);
                    match timezone {
                        None => {
                            hasher.update(&[0]);
                        }
                        Some(zone) => {
                            hasher.update(&[1]);
                            text(&mut hasher, zone);
                        }
                    }
                }
                FileColumnKind::Categorical { ordered } => {
                    hasher.update(&[5, u8::from(*ordered)]);
                    hasher.update(&(column.levels.len() as u64).to_le_bytes());
                    for level in column.levels {
                        text(&mut hasher, level);
                    }
                }
                FileColumnKind::Unsupported { arrow_type } => {
                    hasher.update(&[255]);
                    text(&mut hasher, arrow_type);
                }
            }
            hasher.update(&column.digest);
        }
        Self { version: FILE_IDENTITY_VERSION, rows, digest: *hasher.finalize().as_bytes() }
    }

    /// Encoding version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Rows the scan read.
    #[must_use]
    pub const fn rows(&self) -> GlobalId {
        self.rows
    }

    /// Digest bytes.
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Lower-case hexadecimal digest.
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.digest {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

fn text(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}
