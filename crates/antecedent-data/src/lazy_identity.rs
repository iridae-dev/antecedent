//! Versioned identity for a lazy or chunked view of an owned source.
//!
//! Full content identity is version 1 of [`crate::storage::OwnedColumnarStorage::content_digest`]:
//! the digest of columns, masks, and weights, not a second hash of those bytes. A chunk or
//! lazy view uses its own domain and records the parent digest, a digest of the parent
//! contract (modality, validity policy, cohort order, unit, splits, row count, schema
//! names, and time index), the row range, and a version. Byte-identical storage under a
//! different column name, modality, cohort, split, or time index is therefore a different
//! parent and a different view. A projected column set has no full-source identity. An
//! incremental digest is accepted only after a base scan has been checked, and it is bound
//! to that same contract; skipping the scan fails closed.
//!
//! This is not another owned snapshot. Callers keep [`OwnedSource`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Constructors and readers are the identity contract. Later preparation call
// sites use them; until then only the unit tests do, which CI does not count.
#![allow(dead_code)]

use std::ops::Range;

use crate::error::DataError;
use crate::source::{
    CohortOrder, OwnedSource, RowChunk, SnapshotContract, SourceIdentity, SourceModality,
    ValidityPolicy,
};
use crate::temporal::SamplingRegularity;

/// Version of the full-content binding.
///
/// Version 1 is the bytes of [`crate::storage::OwnedColumnarStorage::content_digest`].
pub(crate) const FULL_CONTENT_VERSION: u16 = 1;

/// Version recorded on a chunk or lazy view.
pub(crate) const LAZY_VIEW_VERSION: u16 = 1;

/// Version of an incremental continuation of a verified base scan.
pub(crate) const INCREMENTAL_VERSION: u16 = 1;

/// BLAKE3 domain for a chunk or lazy view. Not the storage-digest domain.
const LAZY_DOMAIN: &str = "antecedent.data.lazy.v1";

/// BLAKE3 domain for the digest of a snapshot contract's scientific fields.
const CONTRACT_DOMAIN: &str = "antecedent.data.lazy.contract.v1";

/// BLAKE3 domain for a continuation checked against a verified base scan.
const INCREMENT_DOMAIN: &str = "antecedent.data.lazy.increment.v1";

/// Version 1 full-source digest.
///
/// The digest is the storage digest itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FullContentIdentity {
    version: u16,
    digest: [u8; 32],
    /// Digest of the contract the storage digest was recorded under.
    binding: [u8; 32],
}

/// Chunk or lazy view of a full source.
///
/// `digest` is domain-separated from [`FullContentIdentity`]. Equality with the parent
/// is not defined on this type; compare [`VersionedIdentity`] when both are in hand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LazyViewIdentity {
    version: u16,
    parent: [u8; 32],
    binding: [u8; 32],
    start: usize,
    end: usize,
    digest: [u8; 32],
}

/// Full source, or a lazy view of one. The variants are not equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VersionedIdentity {
    /// Columns, mask, and weights.
    Full(FullContentIdentity),
    /// Row range in the lazy domain.
    Lazy(LazyViewIdentity),
}

/// Incremental continuation bound to a verified parent digest and its contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IncrementalDigest {
    version: u16,
    parent: [u8; 32],
    digest: [u8; 32],
}

/// Whether the version-1 storage digest has been checked.
///
/// [`Self::unverified`] takes a claimed parent and does not scan. [`Self::verify`]
/// is the required first scan.
#[derive(Clone, Copy, Debug)]
pub(crate) enum BaseScan {
    /// No `content_digest` check has run.
    Unverified,
    /// Parent digest checked against the owned source.
    Verified(FullContentIdentity),
}

impl FullContentIdentity {
    /// Stamp version 1 onto a contract that already stores the storage digest.
    ///
    /// A projection is refused. This does not hash the columns again.
    ///
    /// # Errors
    ///
    /// The contract is [`SourceIdentity::Projected`].
    pub(crate) fn try_from_contract(contract: &SnapshotContract) -> Result<Self, DataError> {
        match contract.identity {
            SourceIdentity::Full(digest) => Ok(Self {
                version: FULL_CONTENT_VERSION,
                digest,
                binding: contract_binding(contract),
            }),
            SourceIdentity::Projected { .. } => Err(projection_refused()),
        }
    }

    /// Version 1 after checking the contract against `content_digest`.
    ///
    /// # Errors
    ///
    /// The source is a projection, or its recorded digest is not the storage digest.
    pub(crate) fn try_from_source(source: &OwnedSource) -> Result<Self, DataError> {
        let SourceIdentity::Full(recorded) = source.contract().identity else {
            return Err(projection_refused());
        };
        let scanned = source.borrow().storage().content_digest();
        if recorded != scanned {
            return Err(DataError::InvalidArgument {
                message: "snapshot digest is not version 1 of the storage content digest".into(),
            });
        }
        Ok(Self {
            version: FULL_CONTENT_VERSION,
            digest: scanned,
            binding: contract_binding(source.contract()),
        })
    }

    /// Format version. Version 1 is the storage digest.
    #[must_use]
    pub(crate) const fn version(&self) -> u16 {
        self.version
    }

    /// Storage digest bytes.
    #[must_use]
    pub(crate) const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Digest of the contract fields the storage digest does not cover.
    #[must_use]
    pub(crate) const fn binding(&self) -> [u8; 32] {
        self.binding
    }
}

impl LazyViewIdentity {
    /// Identity of `chunk` under the lazy domain.
    ///
    /// # Errors
    ///
    /// `full` is not version 1, or `chunk` names a different parent digest.
    pub(crate) fn from_chunk(
        full: &FullContentIdentity,
        chunk: &RowChunk,
    ) -> Result<Self, DataError> {
        if full.version != FULL_CONTENT_VERSION {
            return Err(DataError::InvalidArgument {
                message: "full content identity is not version 1".into(),
            });
        }
        if chunk.parent() != full.digest {
            return Err(DataError::InvalidArgument {
                message: "chunk parent is not the full content digest".into(),
            });
        }
        let rows = chunk.rows();
        Self::seal(full.digest, full.binding, rows.start, rows.end)
    }

    fn seal(
        parent: [u8; 32],
        binding: [u8; 32],
        start: usize,
        end: usize,
    ) -> Result<Self, DataError> {
        Ok(Self {
            version: LAZY_VIEW_VERSION,
            parent,
            binding,
            start,
            end,
            digest: lazy_digest(&parent, &binding, start, end)?,
        })
    }

    /// Digest of the parent contract this view was taken under.
    #[must_use]
    pub(crate) const fn binding(&self) -> [u8; 32] {
        self.binding
    }

    /// Lazy-view version.
    #[must_use]
    pub(crate) const fn version(&self) -> u16 {
        self.version
    }

    /// Parent full-content digest.
    #[must_use]
    pub(crate) const fn parent(&self) -> [u8; 32] {
        self.parent
    }

    /// Half-open row range.
    #[must_use]
    pub(crate) const fn rows(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Domain-separated digest. Not the parent storage digest.
    #[must_use]
    pub(crate) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

impl IncrementalDigest {
    /// Stream `parts` under the increment domain, bound to `parent` and its contract.
    fn stream(parent: &[u8; 32], binding: &[u8; 32], parts: &[&[u8]]) -> [u8; 32] {
        let mut hasher = incremental_hasher(parent, binding);
        for part in parts {
            hasher.update(part);
        }
        finish(&hasher)
    }

    /// One-shot digest of `bytes` under the same domain, parent, and contract as [`Self::stream`].
    fn recompute(parent: &[u8; 32], binding: &[u8; 32], bytes: &[u8]) -> [u8; 32] {
        let mut hasher = incremental_hasher(parent, binding);
        hasher.update(bytes);
        finish(&hasher)
    }

    /// Increment version.
    #[must_use]
    pub(crate) const fn version(&self) -> u16 {
        self.version
    }

    /// Verified parent digest.
    #[must_use]
    pub(crate) const fn parent(&self) -> [u8; 32] {
        self.parent
    }

    /// Accepted continuation digest.
    #[must_use]
    pub(crate) const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

impl BaseScan {
    /// A claimed parent that has not been checked against a base scan.
    ///
    /// `claimed` is not recorded. Passing the real storage digest still refuses an increment.
    #[must_use]
    pub(crate) const fn unverified(_claimed: [u8; 32]) -> Self {
        Self::Unverified
    }

    /// Required first scan. Records version 1 of `content_digest` when it matches.
    ///
    /// # Errors
    ///
    /// The source is a projection, or its recorded digest is not the storage digest.
    pub(crate) fn verify(source: &OwnedSource) -> Result<Self, DataError> {
        Ok(Self::Verified(FullContentIdentity::try_from_source(source)?))
    }

    /// Accept streamed `parts` when they are the same bytes as `bytes`.
    ///
    /// The streamed digest is checked against [`IncrementalDigest::recompute`].
    /// An unverified scan is refused before that comparison.
    ///
    /// # Errors
    ///
    /// The base scan was skipped, or the two digests differ.
    pub(crate) fn accept(
        &self,
        parts: &[&[u8]],
        bytes: &[u8],
    ) -> Result<IncrementalDigest, DataError> {
        let Self::Verified(full) = self else {
            return Err(scan_required());
        };
        let streamed = IncrementalDigest::stream(&full.digest, &full.binding, parts);
        let recomputed = IncrementalDigest::recompute(&full.digest, &full.binding, bytes);
        if streamed != recomputed {
            return Err(DataError::InvalidArgument {
                message: "incremental digest does not match a recomputation of the same bytes"
                    .into(),
            });
        }
        Ok(IncrementalDigest {
            version: INCREMENTAL_VERSION,
            parent: full.digest,
            digest: recomputed,
        })
    }
}

fn lazy_digest(
    parent: &[u8; 32],
    binding: &[u8; 32],
    start: usize,
    end: usize,
) -> Result<[u8; 32], DataError> {
    let start = u64::try_from(start).map_err(|_| DataError::InvalidArgument {
        message: "chunk start does not fit in u64".into(),
    })?;
    let end = u64::try_from(end).map_err(|_| DataError::InvalidArgument {
        message: "chunk end does not fit in u64".into(),
    })?;
    let mut hasher = blake3::Hasher::new_derive_key(LAZY_DOMAIN);
    hasher.update(&LAZY_VIEW_VERSION.to_le_bytes());
    hasher.update(parent);
    hasher.update(binding);
    hasher.update(&start.to_le_bytes());
    hasher.update(&end.to_le_bytes());
    Ok(finish(&hasher))
}

/// Digest of every scientific field of a contract except the content identity, which the
/// storage digest already carries.
fn contract_binding(contract: &SnapshotContract) -> [u8; 32] {
    fn text(hasher: &mut blake3::Hasher, value: &str) {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    fn word(hasher: &mut blake3::Hasher, value: usize) {
        hasher.update(&(value as u64).to_le_bytes());
    }
    let mut hasher = blake3::Hasher::new_derive_key(CONTRACT_DOMAIN);
    hasher.update(&[match contract.modality {
        SourceModality::Tabular => 1,
        SourceModality::Panel => 2,
        SourceModality::MultiEnvironment => 3,
        SourceModality::TimeSeries => 4,
    }]);
    hasher.update(&[match contract.validity {
        ValidityPolicy::F64InvalidIffNan => 1,
        ValidityPolicy::NullDistinctFromNan => 2,
    }]);
    match &contract.cohort {
        CohortOrder::Storage => {
            hasher.update(&[1]);
        }
        CohortOrder::Labeled(label) => {
            hasher.update(&[2]);
            text(&mut hasher, label);
        }
    }
    match &contract.unit {
        None => {
            hasher.update(&[0]);
        }
        Some(unit) => {
            hasher.update(&[1]);
            text(&mut hasher, unit);
        }
    }
    word(&mut hasher, contract.splits.len());
    for split in &contract.splits {
        text(&mut hasher, &split.label);
        word(&mut hasher, split.rows.start);
        word(&mut hasher, split.rows.end);
    }
    word(&mut hasher, contract.rows);
    word(&mut hasher, contract.schema_names.len());
    for name in contract.schema_names.iter() {
        text(&mut hasher, name);
    }
    match &contract.time_index {
        None => {
            hasher.update(&[0]);
        }
        Some(index) => {
            hasher.update(&[1]);
            match index.regularity {
                SamplingRegularity::Irregular => {
                    hasher.update(&[0]);
                }
                SamplingRegularity::Regular { interval_ns } => {
                    hasher.update(&[1]);
                    hasher.update(&interval_ns.to_le_bytes());
                }
            }
            word(&mut hasher, index.length);
        }
    }
    finish(&hasher)
}

fn incremental_hasher(parent: &[u8; 32], binding: &[u8; 32]) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new_derive_key(INCREMENT_DOMAIN);
    hasher.update(&INCREMENTAL_VERSION.to_le_bytes());
    hasher.update(parent);
    hasher.update(binding);
    hasher
}

fn finish(hasher: &blake3::Hasher) -> [u8; 32] {
    *hasher.finalize().as_bytes()
}

fn projection_refused() -> DataError {
    DataError::InvalidArgument {
        message: "a projected column set is not the full-source identity".into(),
    }
}

fn scan_required() -> DataError {
    DataError::InvalidArgument {
        message: "incremental digest requires a verified base scan".into(),
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::TabularData;

    use super::*;

    fn source(columns: &[(&str, &[f64])]) -> OwnedSource {
        let storage =
            TabularData::from_f64_columns(columns.iter().copied()).unwrap().storage().clone();
        OwnedSource::from_storage(storage, SourceModality::Tabular)
    }

    #[test]
    fn full_content_identity_is_storage_digest_version_1() {
        let owned = source(&[("x", &[1.0, 2.0, 3.0])]);
        let full = FullContentIdentity::try_from_source(&owned).unwrap();
        let stamped = FullContentIdentity::try_from_contract(owned.contract()).unwrap();
        assert_eq!(full.version(), FULL_CONTENT_VERSION);
        assert_eq!(full.digest(), owned.borrow().storage().content_digest());
        assert_eq!(stamped, full);

        let missing = source(&[("x", &[1.0, f64::NAN])]);
        assert_ne!(full.digest(), FullContentIdentity::try_from_source(&missing).unwrap().digest());
    }

    #[test]
    fn lazy_view_is_not_the_parent_full_identity() {
        let owned = source(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
        let full = FullContentIdentity::try_from_source(&owned).unwrap();
        let tail = LazyViewIdentity::from_chunk(&full, &owned.chunk(2, 4).unwrap()).unwrap();
        let empty = LazyViewIdentity::from_chunk(&full, &owned.chunk(1, 1).unwrap()).unwrap();

        assert_eq!(tail.version(), LAZY_VIEW_VERSION);
        assert_eq!(tail.parent(), full.digest());
        assert_eq!(tail.rows(), 2..4);
        assert_eq!(empty.rows(), 1..1);
        assert_ne!(tail.digest(), full.digest());
        assert_ne!(VersionedIdentity::Full(full), VersionedIdentity::Lazy(tail));
        assert_ne!(empty, tail);

        let other = FullContentIdentity::try_from_source(&source(&[("x", &[9.0])])).unwrap();
        assert!(LazyViewIdentity::from_chunk(&other, &owned.chunk(0, 1).unwrap()).is_err());
    }

    #[test]
    fn same_bytes_under_a_different_contract_is_a_different_view() {
        let base = source(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
        let renamed = source(&[("z", &[1.0, 2.0, 3.0, 4.0])]);
        let panel =
            OwnedSource::from_storage(base.borrow().storage().clone(), SourceModality::Panel);
        let cohort = base.clone().with_cohort(CohortOrder::Labeled("arrival".into()));
        let split = base
            .clone()
            .with_splits(vec![crate::source::SplitSpan { label: "fit".into(), rows: 0..2 }])
            .unwrap();
        let view = |owned: &OwnedSource| {
            let full = FullContentIdentity::try_from_source(owned).unwrap();
            LazyViewIdentity::from_chunk(&full, &owned.chunk(1, 3).unwrap()).unwrap()
        };
        let reference = view(&base);
        // Same storage digest, same row range: only the contract differs.
        assert_eq!(
            base.borrow().storage().content_digest(),
            panel.borrow().storage().content_digest()
        );
        assert_eq!(reference, view(&base.clone()));
        for other in [&renamed, &panel, &cohort, &split] {
            let other = view(other);
            assert_eq!(other.rows(), reference.rows());
            assert_ne!(other.digest(), reference.digest());
            assert_ne!(other.binding(), reference.binding());
        }
    }

    #[test]
    fn projected_columns_are_not_the_full_source_identity() {
        let owned = source(&[("x", &[1.0, 2.0]), ("y", &[3.0, 4.0])]);
        let full = FullContentIdentity::try_from_source(&owned).unwrap();
        let one = owned.project(&[0]).unwrap();
        let both = owned.project(&[0, 1]).unwrap();

        assert!(FullContentIdentity::try_from_contract(&one).is_err());
        assert!(FullContentIdentity::try_from_contract(&both).is_err());
        assert!(one.full_digest().is_none());
        assert!(both.full_digest().is_none());
        match both.identity {
            SourceIdentity::Projected { parent, .. } => assert_eq!(parent, full.digest()),
            SourceIdentity::Full(_) => panic!("projection reused the full identity"),
        }
    }

    #[test]
    fn unverified_increment_is_refused_and_verified_matches_recompute() {
        let owned = source(&[("x", &[1.0, 2.0, 3.0])]);
        let parent = owned.borrow().storage().content_digest();
        let parts: &[&[u8]] = &[b"scan-", b"continuation"];
        let bytes = b"scan-continuation";

        assert!(BaseScan::unverified(parent).accept(parts, bytes).is_err());

        let scan = BaseScan::verify(&owned).unwrap();
        let accepted = scan.accept(parts, bytes).unwrap();
        let binding = FullContentIdentity::try_from_source(&owned).unwrap().binding();
        assert_eq!(accepted.version(), INCREMENTAL_VERSION);
        assert_eq!(accepted.parent(), parent);
        assert_eq!(accepted.digest(), IncrementalDigest::recompute(&parent, &binding, bytes));
        assert_eq!(accepted.digest(), IncrementalDigest::stream(&parent, &binding, parts));
        assert_ne!(accepted.digest(), parent);
        assert!(scan.accept(parts, b"scan-other").is_err());
    }

    #[test]
    fn time_index_changes_the_lazy_view_and_the_incremental_digest() {
        use crate::temporal::{SamplingRegularity, TimeIndex};

        let owned = source(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
        let timed = owned
            .clone()
            .with_time_index(TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 5 },
                length: 4,
            })
            .unwrap();
        let view = |source: &OwnedSource| {
            let full = FullContentIdentity::try_from_source(source).unwrap();
            let lazy = LazyViewIdentity::from_chunk(&full, &source.chunk(0, 2).unwrap()).unwrap();
            (full, lazy)
        };
        let (base_full, base_view) = view(&owned);
        let (timed_full, timed_view) = view(&timed);
        assert_eq!(base_full.digest(), timed_full.digest());
        assert_ne!(base_full.binding(), timed_full.binding());
        assert_ne!(base_view.digest(), timed_view.digest());
        assert_ne!(base_view.binding(), timed_view.binding());

        let parts: &[&[u8]] = &[b"scan"];
        let bytes = b"scan";
        let base_inc = BaseScan::verify(&owned).unwrap().accept(parts, bytes).unwrap();
        let timed_inc = BaseScan::verify(&timed).unwrap().accept(parts, bytes).unwrap();
        assert_eq!(base_inc.parent(), timed_inc.parent());
        assert_ne!(base_inc.digest(), timed_inc.digest());
    }
}
