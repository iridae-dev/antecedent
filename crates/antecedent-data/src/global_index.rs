// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked global row and sample identities.
//!
//! A [`GlobalId`] is a `u64` row or sample identity. A [`ChunkIndex`] is a local
//! offset inside one bounded chunk and must fit in `u32`. Converting a
//! [`GlobalId`] into a [`ChunkIndex`] fails when the value does not fit in `u32`
//! instead of truncating it.

use core::fmt;

/// Next attempt after a global identity does not fit in one chunk.
const NEXT_ATTEMPT: &str =
    "address the row by GlobalId and keep ChunkIndex inside one bounded chunk";

/// A `u32` backend switch does not replace a bounded chunk.
const NOT_A_FEATURE_FLAG: &str = "A u32 backend feature flag is not that attempt";

/// A row or sample identity.
///
/// The value is the full `u64`. It is not a `u32` row index and it is not
/// narrowed by a cast.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct GlobalId(u64);

/// A local offset inside one bounded chunk.
///
/// Every [`ChunkIndex`] fits in `u32`. A global identity is admitted only
/// through [`TryFrom<GlobalId>`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ChunkIndex(u32);

/// A [`GlobalId`] does not fit in a [`ChunkIndex`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalIndexError {
    id: GlobalId,
}

impl GlobalId {
    /// Identity `raw`.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The full identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Narrow this identity to a local chunk offset.
    ///
    /// # Errors
    ///
    /// [`GlobalIndexError`] when `self` does not fit in `u32`. The error names
    /// the next attempt: address the row by [`GlobalId`] and keep [`ChunkIndex`]
    /// inside one bounded chunk. A `u32` backend feature flag is not that attempt.
    #[must_use = "a global id past u32::MAX must stay a GlobalId inside one bounded chunk"]
    pub fn to_chunk_index(self) -> Result<ChunkIndex, GlobalIndexError> {
        u32::try_from(self.0).map(ChunkIndex).map_err(|_| GlobalIndexError { id: self })
    }
}

impl ChunkIndex {
    /// Local offset `local`, which already fits in `u32`.
    #[must_use]
    pub const fn new(local: u32) -> Self {
        Self(local)
    }

    /// The local offset.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl GlobalIndexError {
    /// The identity that does not fit in one chunk.
    #[must_use]
    pub const fn id(self) -> GlobalId {
        self.id
    }

    /// What to do next: address the row by [`GlobalId`] and keep [`ChunkIndex`]
    /// inside one bounded chunk.
    #[must_use]
    pub const fn next_attempt(self) -> &'static str {
        NEXT_ATTEMPT
    }
}

impl TryFrom<GlobalId> for ChunkIndex {
    type Error = GlobalIndexError;

    fn try_from(id: GlobalId) -> Result<Self, Self::Error> {
        id.to_chunk_index()
    }
}

impl From<u32> for GlobalId {
    fn from(local: u32) -> Self {
        Self(u64::from(local))
    }
}

impl From<u32> for ChunkIndex {
    fn from(local: u32) -> Self {
        Self(local)
    }
}

impl From<GlobalId> for u64 {
    fn from(id: GlobalId) -> Self {
        id.0
    }
}

impl From<ChunkIndex> for u32 {
    fn from(index: ChunkIndex) -> Self {
        index.0
    }
}

impl fmt::Display for GlobalIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let id = self.id.get();
        write!(f, "global id {id} does not fit in a chunk index; {NEXT_ATTEMPT}. ")?;
        f.write_str(NOT_A_FEATURE_FLAG)
    }
}

impl std::error::Error for GlobalIndexError {}

#[cfg(test)]
mod tests {
    use super::{ChunkIndex, GlobalId, GlobalIndexError};

    #[test]
    fn u32_max_fits_a_chunk() {
        let id = GlobalId::from(u32::MAX);
        let index = ChunkIndex::try_from(id).expect("u32::MAX fits one chunk");
        assert_eq!(index.get(), u32::MAX);
        assert_eq!(id.get(), u64::from(u32::MAX));
        assert_eq!(u64::from(u32::from(index)), id.get());
    }

    #[test]
    fn one_past_u32_max_does_not_fit_a_chunk() {
        let id = GlobalId::new(u64::from(u32::MAX) + 1);
        let err: GlobalIndexError =
            ChunkIndex::try_from(id).expect_err("one past u32::MAX is outside a chunk");
        assert_eq!(err.id(), id);
        let message = err.to_string();
        assert!(message.contains(err.next_attempt()));
        assert!(message.contains("A u32 backend feature flag is not that attempt"));
    }

    #[test]
    fn global_id_is_not_silently_truncated() {
        let raw = u64::from(u32::MAX) + 1;
        let id = GlobalId::new(raw);
        assert_eq!(id.get(), raw);
        assert!(u32::try_from(id.get()).is_err());
        let err = id.to_chunk_index().expect_err("conversion must refuse");
        assert_eq!(err.id().get(), raw);
        assert_ne!(err.id().get(), u64::from(u32::MAX));
    }
}
