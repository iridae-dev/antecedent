//! Resident-byte account for cache, provider state, draws, scores, and exports.
//!
//! One budget covers every resident byte. A shared [`Arc`] is one charge, a
//! separate allocation is another, and eviction drops the entry it names.
//! Admission that does not fit returns an error and stores nothing: there is
//! no side cache that can make the check succeed. Summary-only retention
//! replaces each payload with a digest and then serves only reuse of that
//! summary. Disk-backed output writes each payload to its own digest-checked
//! file in a caller-given directory, keeps no bytes, and serves only a verified
//! read of that file.
//!
//! A decompression reservation is a [`Vec`] created with `with_capacity` at
//! the declared size. That size has to be positive and at most
//! [`DECOMPRESSION_CAP`], which is the container section cap. This module does
//! not change that cap and does not decode a frame.
//!
//! Limits. Payloads are opaque bytes. Provider envelopes, draws, and scores
//! are not interpreted. Disk-backed files are not charged against the resident
//! budget and are not removed by the account: the caller owns the directory.
//! Allocator slack above a requested decompression buffer is not a second
//! budget. A digest is not the payload it replaced.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use antecedent_core::CanonicalWriter;
use antecedent_core::execution_facts::OutputRetention;

mod spill_file;
mod weighted_gaussian_capacity;
pub use spill_file::SpillFault;
pub use weighted_gaussian_capacity::{
    CapacityControl, CapacityOptions, CeilingCharges, DesignShape, FileWeightedRows,
    GaussianCrossProducts, LeverageSummary, ResidentWeightedRows, ResidualCheck, RowFileFault,
    SpillHandle, SpillOptions, SpilledCapacity, WeightFault, WeightedCapacityError,
    WeightedCapacityOutput, WeightedCapacityResult, WeightedRowSource, encode_weighted_row_file,
    read_spilled_summary, run_weighted_capacity,
};

/// Default largest decompression buffer an account will reserve.
///
/// This is the container's default maximum section size;
/// [`CapacityAccount::with_decompression_cap`] sets another.
pub const DECOMPRESSION_CAP: usize = crate::container::MAX_SECTION_BYTES;

const SUMMARY_TAG: &str = "antecedent.capacity_account.summary";
const DISK_TAG: &str = "antecedent.capacity_account.disk";

/// What a resident or declared payload is.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AccountedKind {
    /// Reusable cache entry.
    Cache,
    /// Provider state bytes.
    ProviderState,
    /// Draw bytes.
    Draws,
    /// Score bytes.
    Scores,
    /// Exported payload bytes.
    ExportedPayload,
}

impl AccountedKind {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::ProviderState => "provider state",
            Self::Draws => "draws",
            Self::Scores => "scores",
            Self::ExportedPayload => "exported payload",
        }
    }
}

impl std::fmt::Display for AccountedKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// An operation the retention mode may serve.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AccountedOperation {
    /// Reconstruct rows from what was kept.
    RowReconstruction,
    /// Fit again from retained rows.
    RefitFromRetainedRows,
    /// Drop one retained entry and keep an exact result for what remains.
    ExactRowEviction,
    /// Insert a column and keep an exact result.
    InPlaceColumnInsert,
    /// Change a weight or a mask and keep an exact result.
    InPlaceWeightOrMaskUpdate,
    /// Reuse the full in-memory state.
    ReuseFullState,
    /// Reuse the retained summary.
    ReuseSummary,
    /// Read a disk artifact back and verify its length and digest.
    ReadDiskArtifact,
    /// Admit a cache entry.
    AdmitCache,
    /// Admit provider state.
    AdmitProviderState,
    /// Admit draws.
    AdmitDraws,
    /// Admit scores.
    AdmitScores,
    /// Admit an exported payload.
    AdmitExportedPayload,
    /// Reserve a decompression buffer.
    ReserveDecompression,
}

impl AccountedOperation {
    /// Every operation, in the order a reduced mode reports losses.
    pub const ALL: [Self; 14] = [
        Self::RowReconstruction,
        Self::RefitFromRetainedRows,
        Self::ExactRowEviction,
        Self::InPlaceColumnInsert,
        Self::InPlaceWeightOrMaskUpdate,
        Self::ReuseFullState,
        Self::ReuseSummary,
        Self::ReadDiskArtifact,
        Self::AdmitCache,
        Self::AdmitProviderState,
        Self::AdmitDraws,
        Self::AdmitScores,
        Self::AdmitExportedPayload,
        Self::ReserveDecompression,
    ];

    /// Stable name used in refusals.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowReconstruction => "row reconstruction",
            Self::RefitFromRetainedRows => "refit from retained rows",
            Self::ExactRowEviction => "exact row eviction",
            Self::InPlaceColumnInsert => "in-place column insert",
            Self::InPlaceWeightOrMaskUpdate => "in-place weight or mask update",
            Self::ReuseFullState => "reuse of full state",
            Self::ReuseSummary => "reuse of the summary",
            Self::ReadDiskArtifact => "read of a disk artifact",
            Self::AdmitCache => "admission of a cache entry",
            Self::AdmitProviderState => "admission of provider state",
            Self::AdmitDraws => "admission of draws",
            Self::AdmitScores => "admission of scores",
            Self::AdmitExportedPayload => "admission of an exported payload",
            Self::ReserveDecompression => "reservation of a decompression buffer",
        }
    }

    const fn for_kind(kind: AccountedKind) -> Self {
        match kind {
            AccountedKind::Cache => Self::AdmitCache,
            AccountedKind::ProviderState => Self::AdmitProviderState,
            AccountedKind::Draws => Self::AdmitDraws,
            AccountedKind::Scores => Self::AdmitScores,
            AccountedKind::ExportedPayload => Self::AdmitExportedPayload,
        }
    }
}

impl std::fmt::Display for AccountedOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why an account mutation was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CapacityError {
    /// The retention mode cannot serve the operation.
    #[error("{retention} cannot serve {operation}")]
    Retention {
        /// Mode that was asked.
        retention: OutputRetention,
        /// Operation that mode cannot serve.
        operation: AccountedOperation,
    },
    /// `resident + additional` is over the budget. Nothing was stored.
    #[error("resident {resident} bytes plus {additional} bytes exceeds the budget of {budget}")]
    OverBudget {
        /// Bytes already charged.
        resident: u64,
        /// Bytes the refused operation would have added.
        additional: u64,
        /// Declared budget.
        budget: u64,
    },
    /// Summary digests would not fit, so the payloads stay as they were.
    #[error("summary digests need {needed} bytes, over the budget of {budget}")]
    SummaryExceedsBudget {
        /// Bytes the digests would charge.
        needed: u64,
        /// Declared budget.
        budget: u64,
    },
    /// The declared decompression size is zero or above [`DECOMPRESSION_CAP`].
    #[error("decompression of {declared} bytes exceeds the cap of {cap} bytes")]
    DecompressionCap {
        /// Requested size.
        declared: u64,
        /// The account's decompression cap.
        cap: usize,
    },
    /// A payload has to contain at least one byte.
    #[error("accounted payload must be non-empty")]
    EmptyPayload,
    /// No entry has this id.
    #[error("unknown account entry {id}")]
    UnknownEntry {
        /// Requested id.
        id: u64,
    },
    /// A decompression reservation is still charged.
    #[error("a decompression reservation is still charged")]
    ReservationOutstanding,
    /// A count of bytes did not fit in `u64`.
    #[error("capacity account overflowed")]
    Overflow,
    /// A spill file failed its length, digest, or layout check.
    #[error("spill file: {0}")]
    Spill(SpillFault),
    /// A spill file could not be created, written, read, or removed.
    #[error("spill file could not {context}: {kind}")]
    SpillIo {
        /// The step that failed.
        context: &'static str,
        /// The operating-system error kind.
        kind: io::ErrorKind,
    },
    /// The allocator refused a spill buffer.
    #[error("could not allocate {bytes} bytes for a spill buffer")]
    SpillAllocation {
        /// Bytes requested.
        bytes: u64,
    },
    /// A write was longer than the reserved decompression size.
    #[error("decompression write of {got} bytes exceeds the reserved {requested}")]
    WritePastReservation {
        /// Bytes offered.
        got: usize,
        /// Bytes reserved.
        requested: usize,
    },
}

/// A buffer requested at a declared size and charged until it is released.
#[derive(Debug)]
pub struct DecompressionBuffer {
    id: u64,
    requested: usize,
    buffer: Vec<u8>,
}

impl DecompressionBuffer {
    /// Account id of the reservation.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Size passed to `Vec::with_capacity`. At most [`DECOMPRESSION_CAP`].
    #[must_use]
    pub const fn requested(&self) -> usize {
        self.requested
    }

    /// Allocator capacity. This may be larger than [`Self::requested`].
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Bytes written so far.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// Replace the contents with `bytes`.
    ///
    /// # Errors
    ///
    /// `bytes` is longer than the reserved size. The previous contents stay.
    #[must_use = "a write past the reservation is refused"]
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), CapacityError> {
        if bytes.len() > self.requested {
            return Err(CapacityError::WritePastReservation {
                got: bytes.len(),
                requested: self.requested,
            });
        }
        self.buffer.clear();
        self.buffer.extend_from_slice(bytes);
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum EntryClass {
    Accounted(AccountedKind),
    Decompression,
}

#[derive(Clone, Debug)]
enum Body {
    Shared(Arc<[u8]>),
    Summary(Arc<[u8]>),
    Disk(DiskRecord),
    Reserved { bytes: u64 },
}

/// Where a payload was written and what it has to hash to when read back.
#[derive(Clone, Debug)]
struct DiskRecord {
    path: PathBuf,
    payload_bytes: u64,
    frame_bytes: u64,
    digest: [u8; 32],
}

impl From<spill_file::IoFail> for CapacityError {
    fn from(fail: spill_file::IoFail) -> Self {
        Self::SpillIo { context: fail.context, kind: fail.source.kind() }
    }
}

impl From<SpillFault> for CapacityError {
    fn from(fault: SpillFault) -> Self {
        Self::Spill(fault)
    }
}

impl From<spill_file::AllocFail> for CapacityError {
    fn from(fail: spill_file::AllocFail) -> Self {
        Self::SpillAllocation { bytes: fail.bytes }
    }
}

#[derive(Clone, Debug)]
struct Entry {
    class: EntryClass,
    body: Body,
}

/// Budgeted store. The map is the only place payloads live.
#[derive(Clone, Debug)]
pub struct CapacityAccount {
    budget: u64,
    retention: OutputRetention,
    entries: BTreeMap<u64, Entry>,
    next_id: u64,
    decompression_cap: usize,
}

impl CapacityAccount {
    /// Empty account. `budget` is the most resident bytes a successful mutation may charge.
    #[must_use]
    pub const fn new(budget: u64) -> Self {
        Self {
            budget,
            retention: OutputRetention::FullReusable,
            entries: BTreeMap::new(),
            next_id: 1,
            decompression_cap: DECOMPRESSION_CAP,
        }
    }

    /// The same account accepting decompression reservations of at most `cap` bytes
    /// instead of [`DECOMPRESSION_CAP`]. The budget still bounds what is resident.
    #[must_use]
    pub fn with_decompression_cap(self, cap: usize) -> Self {
        Self { decompression_cap: cap, ..self }
    }

    /// Largest decompression buffer this account will reserve.
    #[must_use]
    pub const fn decompression_cap(&self) -> usize {
        self.decompression_cap
    }

    /// Declared budget.
    #[must_use]
    pub const fn budget(&self) -> u64 {
        self.budget
    }

    /// Retention mode.
    #[must_use]
    pub const fn retention(&self) -> OutputRetention {
        self.retention
    }

    /// Number of entries, including reservations and disk entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the account has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes charged right now. Disk entries are not included.
    ///
    /// # Errors
    ///
    /// The sum does not fit in `u64`.
    #[must_use = "an overflowed resident total is not a budget check"]
    pub fn resident_bytes(&self) -> Result<u64, CapacityError> {
        resident_of(&self.entries)
    }

    /// Sum of payload bytes written to disk. These bytes are not resident.
    ///
    /// # Errors
    ///
    /// The sum does not fit in `u64`.
    #[must_use = "an overflowed declaration is not a resident charge"]
    pub fn declared_disk_bytes(&self) -> Result<u64, CapacityError> {
        let mut total = 0u64;
        for entry in self.entries.values() {
            if let Body::Disk(record) = &entry.body {
                total = total.checked_add(record.payload_bytes).ok_or(CapacityError::Overflow)?;
            }
        }
        Ok(total)
    }

    /// Whether `additional` resident bytes would still fit.
    ///
    /// The check reads the entries and the budget. It does not look at any other store.
    ///
    /// # Errors
    ///
    /// The resident total overflowed.
    #[must_use = "a refused fit stored nothing"]
    pub fn would_fit(&self, additional: u64) -> Result<bool, CapacityError> {
        match self.ensure_fits(additional) {
            Ok(()) => Ok(true),
            Err(CapacityError::OverBudget { .. }) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Accept `operation` under the current retention mode.
    ///
    /// # Errors
    ///
    /// This mode cannot serve `operation`.
    #[must_use = "an operation this mode cannot serve is refused"]
    pub const fn require(&self, operation: AccountedOperation) -> Result<(), CapacityError> {
        if serves(self.retention, operation) {
            Ok(())
        } else {
            Err(CapacityError::Retention { retention: self.retention, operation })
        }
    }

    /// Kind of an accounted entry.
    #[must_use]
    pub fn kind(&self, id: u64) -> Option<AccountedKind> {
        match self.entries.get(&id)?.class {
            EntryClass::Accounted(kind) => Some(kind),
            EntryClass::Decompression => None,
        }
    }

    /// Shared or summary bytes. Disk entries and reservations have none.
    #[must_use]
    pub fn stored_bytes(&self, id: u64) -> Option<Arc<[u8]>> {
        match &self.entries.get(&id)?.body {
            Body::Shared(bytes) | Body::Summary(bytes) => Some(Arc::clone(bytes)),
            Body::Disk(_) | Body::Reserved { .. } => None,
        }
    }

    /// Charge `bytes` under `kind`.
    ///
    /// A clone of an [`Arc`] the account already holds adds no bytes.
    ///
    /// # Errors
    ///
    /// Retention refuses admission, the payload is empty, or the bytes do not fit.
    /// On error the payload is not stored.
    #[must_use = "a refused admission is not in the account"]
    pub fn admit(&mut self, kind: AccountedKind, bytes: Arc<[u8]>) -> Result<u64, CapacityError> {
        self.require(AccountedOperation::for_kind(kind))?;
        if bytes.is_empty() {
            return Err(CapacityError::EmptyPayload);
        }
        let additional = if self.charged(&bytes) { 0 } else { byte_len(&bytes)? };
        self.ensure_fits(additional)?;
        let id = self.alloc_id()?;
        self.entries
            .insert(id, Entry { class: EntryClass::Accounted(kind), body: Body::Shared(bytes) });
        Ok(id)
    }

    /// Drop an entry.
    ///
    /// The payload is not moved anywhere else. Summary-only and disk-backed
    /// output refuse this, because they cannot serve exact eviction.
    ///
    /// # Errors
    ///
    /// Retention refuses eviction, or the id is unknown.
    #[must_use = "a refused eviction leaves the entry in place"]
    pub fn evict(&mut self, id: u64) -> Result<(), CapacityError> {
        self.require(AccountedOperation::ExactRowEviction)?;
        if self.entries.remove(&id).is_none() {
            return Err(CapacityError::UnknownEntry { id });
        }
        Ok(())
    }

    /// Replace an entry with `bytes` when the result still fits.
    ///
    /// The arc is cloned only into a speculative map. If that map is over
    /// budget, it is dropped and the account is unchanged.
    ///
    /// # Errors
    ///
    /// Unknown id, retention refuses reuse of full state, an empty payload,
    /// or a result that does not fit.
    #[must_use = "a refused materialization stores nothing"]
    pub fn materialize(&mut self, id: u64, bytes: &Arc<[u8]>) -> Result<(), CapacityError> {
        let Some(current) = self.entries.get(&id) else {
            return Err(CapacityError::UnknownEntry { id });
        };
        self.require(AccountedOperation::ReuseFullState)?;
        if bytes.is_empty() {
            return Err(CapacityError::EmptyPayload);
        }
        let class = current.class.clone();
        let mut speculative = self.entries.clone();
        speculative.insert(id, Entry { class, body: Body::Shared(Arc::clone(bytes)) });
        let needed = resident_of(&speculative)?;
        if needed > self.budget {
            let resident = self.resident_bytes()?;
            return Err(CapacityError::OverBudget {
                resident,
                additional: needed.saturating_sub(resident),
                budget: self.budget,
            });
        }
        self.entries = speculative;
        Ok(())
    }

    /// Replace resident payloads with digests and switch to summary-only.
    ///
    /// If the digests would exceed the budget, the payloads stay and the mode
    /// stays. An outstanding decompression reservation refuses the switch.
    ///
    /// # Errors
    ///
    /// A reservation is outstanding, or the digests do not fit.
    #[must_use = "a refused summary keeps the payloads"]
    pub fn retain_summary_only(&mut self) -> Result<(), CapacityError> {
        if self.retention == OutputRetention::SummaryOnly {
            return Ok(());
        }
        self.require_no_reservation()?;
        let mut next = BTreeMap::new();
        for (id, entry) in &self.entries {
            let body = match &entry.body {
                Body::Shared(bytes) | Body::Summary(bytes) => Body::Summary(digest_payload(bytes)),
                Body::Disk(record) => Body::Summary(digest_declared(record.payload_bytes)),
                Body::Reserved { .. } => return Err(CapacityError::ReservationOutstanding),
            };
            next.insert(*id, Entry { class: entry.class.clone(), body });
        }
        let needed = resident_of(&next)?;
        if needed > self.budget {
            return Err(CapacityError::SummaryExceedsBudget { needed, budget: self.budget });
        }
        self.entries = next;
        self.retention = OutputRetention::SummaryOnly;
        Ok(())
    }

    /// Write each resident payload to its own file in `directory`, drop the
    /// resident bytes, and switch to disk-backed output.
    ///
    /// Entry `n` is written to `entry-n.spill` as a digest-checked frame, by a
    /// temporary name that is published without replacing existing paths and
    /// read back. Publication requires same-directory hard links. The account is
    /// unchanged on failure; rollback only removes files this call created.
    /// `directory` has to exist. Reservations must be released first.
    ///
    /// # Errors
    ///
    /// A decompression reservation is still charged, or a file could not be
    /// written, synced, or verified.
    #[must_use = "a refused disk write leaves the payloads resident"]
    pub fn retain_disk_backed(&mut self, directory: &Path) -> Result<(), CapacityError> {
        if self.retention == OutputRetention::DiskBacked {
            return Ok(());
        }
        self.require_no_reservation()?;
        let mut written = SpillDirectory::default();
        let mut next = BTreeMap::new();
        for (id, entry) in &self.entries {
            let record = match &entry.body {
                Body::Shared(bytes) | Body::Summary(bytes) => {
                    written.write(directory, *id, bytes)?
                }
                Body::Disk(record) => record.clone(),
                Body::Reserved { .. } => return Err(CapacityError::ReservationOutstanding),
            };
            next.insert(*id, Entry { class: entry.class.clone(), body: Body::Disk(record) });
        }
        written.keep();
        self.entries = next;
        self.retention = OutputRetention::DiskBacked;
        Ok(())
    }

    /// Read a disk entry back, checking its length and digest.
    ///
    /// The returned bytes belong to the caller and are not charged to the
    /// budget.
    ///
    /// # Errors
    ///
    /// Retention does not serve a disk read, the id is unknown, or the file is
    /// missing, truncated, or changed.
    #[must_use = "a refused read returns no bytes"]
    pub fn read_disk_artifact(&self, id: u64) -> Result<Vec<u8>, CapacityError> {
        self.require(AccountedOperation::ReadDiskArtifact)?;
        let Some(Entry { body: Body::Disk(record), .. }) = self.entries.get(&id) else {
            return Err(CapacityError::UnknownEntry { id });
        };
        let declared =
            spill_file::Declared { frame_bytes: record.frame_bytes, digest: record.digest };
        let mut frame = spill_file::read_verified::<CapacityError>(&record.path, declared)?;
        frame.drain(..HEADER_BYTES);
        Ok(frame)
    }

    /// Reserve a decompression buffer of `declared` bytes.
    ///
    /// The buffer is allocated only after the cap and the budget both accept
    /// `declared`. The charge is `declared`, not the allocator's slack.
    ///
    /// # Errors
    ///
    /// Retention refuses the reservation, `declared` is outside
    /// `1..=DECOMPRESSION_CAP`, or the budget cannot hold it.
    #[must_use = "a refused reservation allocates no buffer"]
    pub fn reserve_decompression(
        &mut self,
        declared: u64,
    ) -> Result<DecompressionBuffer, CapacityError> {
        self.require(AccountedOperation::ReserveDecompression)?;
        let size = usize::try_from(declared).unwrap_or(usize::MAX);
        if declared == 0 || size > self.decompression_cap {
            return Err(CapacityError::DecompressionCap { declared, cap: self.decompression_cap });
        }
        self.ensure_fits(declared)?;
        let id = self.alloc_id()?;
        let buffer = Vec::with_capacity(size);
        self.entries.insert(
            id,
            Entry { class: EntryClass::Decompression, body: Body::Reserved { bytes: declared } },
        );
        Ok(DecompressionBuffer { id, requested: size, buffer })
    }

    /// Drop the reservation and its charge.
    ///
    /// # Errors
    ///
    /// `buffer` is not a live reservation. The buffer is still consumed.
    #[must_use = "a missing reservation did not change the account"]
    pub fn release_decompression(
        &mut self,
        buffer: DecompressionBuffer,
    ) -> Result<(), CapacityError> {
        let DecompressionBuffer { id, buffer: reserved, .. } = buffer;
        drop(reserved);
        match self.entries.remove(&id) {
            Some(Entry { body: Body::Reserved { .. }, .. }) => Ok(()),
            Some(entry) => {
                self.entries.insert(id, entry);
                Err(CapacityError::UnknownEntry { id })
            }
            None => Err(CapacityError::UnknownEntry { id }),
        }
    }

    fn require_no_reservation(&self) -> Result<(), CapacityError> {
        if self.entries.values().any(|entry| matches!(entry.body, Body::Reserved { .. })) {
            Err(CapacityError::ReservationOutstanding)
        } else {
            Ok(())
        }
    }

    fn charged(&self, bytes: &Arc<[u8]>) -> bool {
        let key = allocation_key(bytes);
        self.entries.values().any(|entry| match &entry.body {
            Body::Shared(existing) | Body::Summary(existing) => allocation_key(existing) == key,
            Body::Disk(_) | Body::Reserved { .. } => false,
        })
    }

    fn ensure_fits(&self, additional: u64) -> Result<(), CapacityError> {
        let resident = resident_of(&self.entries)?;
        let sum = resident.checked_add(additional).ok_or(CapacityError::Overflow)?;
        if sum > self.budget {
            Err(CapacityError::OverBudget { resident, additional, budget: self.budget })
        } else {
            Ok(())
        }
    }

    fn alloc_id(&mut self) -> Result<u64, CapacityError> {
        let id = self.next_id;
        self.next_id = id.checked_add(1).ok_or(CapacityError::Overflow)?;
        Ok(id)
    }
}

const fn serves(retention: OutputRetention, operation: AccountedOperation) -> bool {
    match retention {
        OutputRetention::Unmeasured => false,
        OutputRetention::FullReusable => !matches!(operation, AccountedOperation::ReadDiskArtifact),
        OutputRetention::SummaryOnly => matches!(operation, AccountedOperation::ReuseSummary),
        OutputRetention::DiskBacked => matches!(operation, AccountedOperation::ReadDiskArtifact),
    }
}

fn resident_of(entries: &BTreeMap<u64, Entry>) -> Result<u64, CapacityError> {
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for entry in entries.values() {
        let add = match &entry.body {
            Body::Shared(bytes) | Body::Summary(bytes) => {
                if !seen.insert(allocation_key(bytes)) {
                    continue;
                }
                byte_len(bytes)?
            }
            Body::Disk(_) => 0,
            Body::Reserved { bytes } => *bytes,
        };
        total = total.checked_add(add).ok_or(CapacityError::Overflow)?;
    }
    Ok(total)
}

fn allocation_key(bytes: &Arc<[u8]>) -> usize {
    Arc::as_ptr(bytes).addr()
}

fn byte_len(bytes: &[u8]) -> Result<u64, CapacityError> {
    u64::try_from(bytes.len()).map_err(|_| CapacityError::Overflow)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the frame header is the fixed 48-byte format constant on every target"
)]
const HEADER_BYTES: usize = spill_file::FRAME_HEADER_BYTES as usize;

/// Files written by one `retain_disk_backed` call, removed unless kept.
#[derive(Default)]
struct SpillDirectory(Vec<PathBuf>);

impl SpillDirectory {
    fn write(
        &mut self,
        directory: &Path,
        id: u64,
        payload: &[u8],
    ) -> Result<DiskRecord, CapacityError> {
        let payload_bytes = byte_len(payload)?;
        let mut frame = spill_file::begin_frame(payload_bytes)?;
        frame.extend_from_slice(payload);
        spill_file::seal_frame(&mut frame);
        let path = directory.join(format!("entry-{id}.spill"));
        let receipt = spill_file::commit_frame::<CapacityError>(
            &path,
            spill_file::Publication::CreateNew,
            frame,
            spill_file::write_whole,
            || Ok(()),
        )?;
        self.0.push(path.clone());
        Ok(DiskRecord {
            path,
            payload_bytes,
            frame_bytes: receipt.frame_bytes,
            digest: receipt.digest,
        })
    }

    fn keep(&mut self) {
        self.0.clear();
    }
}

impl Drop for SpillDirectory {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = spill_file::remove_file(path);
        }
    }
}

fn digest_payload(bytes: &[u8]) -> Arc<[u8]> {
    Arc::from(CanonicalWriter::digest_bytes(SUMMARY_TAG, bytes))
}

fn digest_declared(declared_bytes: u64) -> Arc<[u8]> {
    let mut writer = CanonicalWriter::new(DISK_TAG);
    writer.word(declared_bytes);
    Arc::from(writer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(bytes: &[u8]) -> Arc<[u8]> {
        Arc::from(bytes.to_vec())
    }

    #[test]
    fn budget_counts_cache_provider_draws_scores_and_exports() {
        let mut account = CapacityAccount::new(15);
        let cache = account.admit(AccountedKind::Cache, payload(&[1])).expect("cache");
        let provider =
            account.admit(AccountedKind::ProviderState, payload(&[2, 2])).expect("provider");
        let draws = account.admit(AccountedKind::Draws, payload(&[3, 3, 3])).expect("draws");
        let scores = account.admit(AccountedKind::Scores, payload(&[4, 4, 4, 4])).expect("scores");
        let exported = account
            .admit(AccountedKind::ExportedPayload, payload(&[5, 5, 5, 5, 5]))
            .expect("export");
        assert_eq!(account.resident_bytes().expect("resident"), 15);
        assert_eq!(account.kind(cache), Some(AccountedKind::Cache));
        assert_eq!(account.kind(provider), Some(AccountedKind::ProviderState));
        assert_eq!(account.kind(draws), Some(AccountedKind::Draws));
        assert_eq!(account.kind(scores), Some(AccountedKind::Scores));
        assert_eq!(account.kind(exported), Some(AccountedKind::ExportedPayload));
        assert_eq!(account.retention(), OutputRetention::FullReusable);
        let extra = payload(&[9]);
        let before = Arc::strong_count(&extra);
        let refusal = account.admit(AccountedKind::Cache, Arc::clone(&extra)).expect_err("over");
        assert!(matches!(refusal, CapacityError::OverBudget { resident: 15, additional: 1, .. }));
        assert_eq!(Arc::strong_count(&extra), before);
        assert!(account.stored_bytes(u64::MAX).is_none());
    }

    #[test]
    fn shared_ownership_charges_one_allocation() {
        let shared = payload(&[1, 2, 3, 4]);
        let mut account = CapacityAccount::new(4);
        let cache = account.admit(AccountedKind::Cache, Arc::clone(&shared)).expect("cache");
        let scores = account.admit(AccountedKind::Scores, Arc::clone(&shared)).expect("scores");
        assert_eq!(account.resident_bytes().expect("resident"), 4);
        assert_eq!(Arc::strong_count(&shared), 3);
        assert!(Arc::ptr_eq(&account.stored_bytes(cache).expect("cache"), &shared));
        account.evict(cache).expect("evict cache");
        assert_eq!(account.resident_bytes().expect("still held"), 4);
        assert_eq!(Arc::strong_count(&shared), 2);
        account.evict(scores).expect("evict scores");
        assert_eq!(account.resident_bytes().expect("empty"), 0);
        assert_eq!(Arc::strong_count(&shared), 1);

        let mut copies = CapacityAccount::new(8);
        let first = payload(&[1, 2, 3, 4]);
        let second = payload(&[1, 2, 3, 4]);
        copies.admit(AccountedKind::Draws, Arc::clone(&first)).expect("first");
        copies.admit(AccountedKind::Scores, Arc::clone(&second)).expect("second");
        assert_eq!(copies.resident_bytes().expect("two allocations"), 8);
        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn eviction_releases_the_charge_and_refused_payloads_are_not_kept() {
        let mut account = CapacityAccount::new(4);
        let held = payload(&[1, 2, 3, 4]);
        let id = account.admit(AccountedKind::ProviderState, Arc::clone(&held)).expect("admit");
        let refused = payload(&[9]);
        let before = Arc::strong_count(&refused);
        assert!(account.admit(AccountedKind::Cache, Arc::clone(&refused)).is_err());
        assert_eq!(Arc::strong_count(&refused), before);
        assert_eq!(account.len(), 1);
        account.evict(id).expect("evict");
        assert!(account.stored_bytes(id).is_none());
        assert_eq!(account.resident_bytes().expect("cleared"), 0);
        account.admit(AccountedKind::Cache, refused).expect("fits after eviction");
        assert_eq!(account.resident_bytes().expect("one byte"), 1);
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn overflow_is_not_rescued_by_a_hidden_cache() {
        let mut account = CapacityAccount::new(4);
        let big = payload(&[0; 8]);
        let before = Arc::strong_count(&big);
        let refusal =
            account.admit(AccountedKind::ExportedPayload, Arc::clone(&big)).expect_err("over");
        assert!(matches!(
            refusal,
            CapacityError::OverBudget { resident: 0, additional: 8, budget: 4 }
        ));
        assert_eq!(Arc::strong_count(&big), before);
        assert!(account.is_empty());
        assert_eq!(account.resident_bytes().expect("empty"), 0);
        assert!(!account.would_fit(8).expect("fit"));
        assert!(account.would_fit(4).expect("fit"));
        let replacement = payload(&[9, 9, 9, 9, 9]);
        let id = account.admit(AccountedKind::Cache, payload(&[1, 2, 3, 4])).expect("full");
        let before_replacement = Arc::strong_count(&replacement);
        assert!(account.materialize(id, &replacement).is_err());
        assert_eq!(Arc::strong_count(&replacement), before_replacement);
        assert_eq!(account.stored_bytes(id).expect("original").as_ref(), &[1, 2, 3, 4]);
        assert_eq!(account.resident_bytes().expect("unchanged"), 4);
    }

    #[test]
    fn summary_only_refuses_row_operations_and_drops_payloads() {
        let mut account = CapacityAccount::new(64);
        let rows = payload(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let id = account.admit(AccountedKind::Draws, Arc::clone(&rows)).expect("draws");
        account.retain_summary_only().expect("summary");
        assert_eq!(account.retention(), OutputRetention::SummaryOnly);
        assert_eq!(Arc::strong_count(&rows), 1);
        assert_eq!(account.resident_bytes().expect("digest"), 32);
        let stored = account.stored_bytes(id).expect("digest");
        assert_ne!(stored.as_ref(), rows.as_ref());
        assert_eq!(stored.len(), 32);
        assert_eq!(account.kind(id), Some(AccountedKind::Draws));
        for operation in AccountedOperation::ALL {
            let served = matches!(operation, AccountedOperation::ReuseSummary);
            assert_eq!(account.require(operation).is_ok(), served, "{operation}");
        }
        let refusal = account.require(AccountedOperation::RowReconstruction).expect_err("rows");
        assert_eq!(refusal.to_string(), "summary-only cannot serve row reconstruction");
        assert!(account.evict(id).is_err());
        assert!(account.stored_bytes(id).is_some());
        assert!(account.admit(AccountedKind::Cache, payload(&[1])).is_err());
        assert!(account.reserve_decompression(8).is_err());
    }

    #[test]
    fn summary_retention_refuses_when_digests_exceed_the_budget() {
        let shared = payload(&[7; 40]);
        let mut account = CapacityAccount::new(40);
        account.admit(AccountedKind::Cache, Arc::clone(&shared)).expect("cache");
        account.admit(AccountedKind::Scores, Arc::clone(&shared)).expect("scores");
        assert_eq!(account.resident_bytes().expect("shared"), 40);
        assert_eq!(Arc::strong_count(&shared), 3);
        let refusal = account.retain_summary_only().expect_err("two digests are 64 bytes");
        assert!(matches!(refusal, CapacityError::SummaryExceedsBudget { needed: 64, budget: 40 }));
        assert_eq!(account.retention(), OutputRetention::FullReusable);
        assert_eq!(Arc::strong_count(&shared), 3);
        assert_eq!(account.resident_bytes().expect("payloads kept"), 40);
        assert!(Arc::ptr_eq(&account.stored_bytes(1).expect("first"), &shared));
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("capacity-account-{}-{name}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }

        fn entries(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .expect("read test directory")
                .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn disk_backed_state_is_not_resident_and_refuses_materialize() {
        let dir = TempDir::new("disk-backed");
        let rows = payload(&[1, 2, 3, 4]);
        let mut account = CapacityAccount::new(4);
        let id = account.admit(AccountedKind::ProviderState, Arc::clone(&rows)).expect("provider");
        account.retain_disk_backed(&dir.0).expect("disk");
        assert_eq!(account.retention(), OutputRetention::DiskBacked);
        assert_eq!(account.resident_bytes().expect("nothing resident"), 0);
        assert_eq!(account.declared_disk_bytes().expect("declared"), 4);
        assert_eq!(Arc::strong_count(&rows), 1);
        assert!(account.stored_bytes(id).is_none());
        assert_eq!(dir.entries(), vec![format!("entry-{id}.spill")]);
        assert_eq!(account.read_disk_artifact(id).expect("read back"), vec![1, 2, 3, 4]);
        for operation in AccountedOperation::ALL {
            let served = matches!(operation, AccountedOperation::ReadDiskArtifact);
            assert_eq!(account.require(operation).is_ok(), served, "{operation}");
        }
        let refusal = account.require(AccountedOperation::ReuseSummary).expect_err("summary");
        assert_eq!(refusal.to_string(), "disk-backed output cannot serve reuse of the summary");
        let before = Arc::strong_count(&rows);
        let materialize = account.materialize(id, &rows).expect_err("not resident");
        assert!(matches!(
            materialize,
            CapacityError::Retention { retention: OutputRetention::DiskBacked, .. }
        ));
        assert_eq!(Arc::strong_count(&rows), before);
        assert_eq!(account.resident_bytes().expect("still empty"), 0);
        assert_eq!(account.declared_disk_bytes().expect("still declared"), 4);
        assert!(matches!(
            account.read_disk_artifact(id + 1),
            Err(CapacityError::UnknownEntry { .. })
        ));
    }

    #[test]
    fn a_changed_or_truncated_disk_entry_is_refused_on_read() {
        let dir = TempDir::new("tamper");
        let mut account = CapacityAccount::new(64);
        let id = account.admit(AccountedKind::Draws, payload(&[9; 16])).expect("draws");
        account.retain_disk_backed(&dir.0).expect("disk");
        let file = dir.0.join(format!("entry-{id}.spill"));
        let original = std::fs::read(&file).expect("read frame");
        let mut flipped = original.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        std::fs::write(&file, &flipped).expect("flip a byte");
        assert!(matches!(
            account.read_disk_artifact(id),
            Err(CapacityError::Spill(SpillFault::DigestMismatch))
        ));
        std::fs::write(&file, &original[..original.len() - 1]).expect("truncate");
        assert!(matches!(
            account.read_disk_artifact(id),
            Err(CapacityError::Spill(SpillFault::LengthMismatch { .. }))
        ));
        std::fs::write(&file, &original).expect("restore");
        assert_eq!(account.read_disk_artifact(id).expect("restored"), vec![9; 16]);
    }

    #[test]
    fn a_failed_disk_spill_preserves_existing_caller_files() {
        let dir = TempDir::new("existing-file-rollback");
        let mut account = CapacityAccount::new(64);
        let first = account.admit(AccountedKind::Cache, payload(&[1, 2])).unwrap();
        let second = account.admit(AccountedKind::Scores, payload(&[3, 4])).unwrap();
        let existing = dir.0.join(format!("entry-{first}.spill"));
        std::fs::write(&existing, b"caller-owned data").unwrap();
        std::fs::create_dir(dir.0.join(format!("entry-{second}.spill"))).unwrap();
        let before = dir.entries();
        assert!(account.retain_disk_backed(&dir.0).is_err());
        assert_eq!(std::fs::read(&existing).expect("preserve caller's file"), b"caller-owned data");
        assert_eq!(dir.entries(), before);
        assert_eq!(account.retention(), OutputRetention::FullReusable);
        assert_eq!(account.resident_bytes().unwrap(), 4);
        assert_eq!(&*account.stored_bytes(first).unwrap(), &[1, 2]);
    }

    #[cfg(unix)]
    #[test]
    fn disk_spill_permissions_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        // Set the umask only in a child process; parallel tests keep their own
        // process-wide creation policy. No unsafe umask manipulation is needed.
        if std::env::var_os("ANTECEDENT_SPILL_PRIVATE_MODE_CHILD").is_none() {
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg("umask 000; exec \"$@\"")
                .arg("spill-permissions-test")
                .arg(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "capacity_account::tests::disk_spill_permissions_are_owner_only",
                    "--test-threads=1",
                ])
                .env("ANTECEDENT_SPILL_PRIVATE_MODE_CHILD", "1")
                .output()
                .expect("start isolated permission test");
            assert!(
                output.status.success(),
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let dir = TempDir::new("private-mode");
        std::fs::set_permissions(&dir.0, std::fs::Permissions::from_mode(0o700)).unwrap();
        let mut frame = spill_file::begin_frame(2).unwrap();
        frame.extend_from_slice(&[1, 2]);
        spill_file::seal_frame(&mut frame);
        spill_file::commit_frame::<CapacityError>(
            &dir.0.join("creation-probe.spill"),
            spill_file::Publication::CreateNew,
            frame,
            |file, bytes| {
                // The file is private before any payload is written, including
                // while it still has its temporary name.
                assert_eq!(file.metadata()?.permissions().mode() & 0o777, 0o600);
                spill_file::write_whole(file, bytes)
            },
            || Ok(()),
        )
        .unwrap();
        let mut account = CapacityAccount::new(64);
        let id = account.admit(AccountedKind::Cache, payload(&[1, 2])).unwrap();
        account.retain_disk_backed(&dir.0).unwrap();
        let mode = std::fs::metadata(dir.0.join(format!("entry-{id}.spill")))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn a_failed_disk_write_leaves_the_account_and_the_directory_unchanged() {
        let dir = TempDir::new("failed-write");
        let mut account = CapacityAccount::new(64);
        let first = account.admit(AccountedKind::Cache, payload(&[1, 2])).expect("first");
        let second = account.admit(AccountedKind::Scores, payload(&[3, 4])).expect("second");
        // The second entry's target is an existing directory, so its rename fails
        // after the first entry was already written.
        std::fs::create_dir(dir.0.join(format!("entry-{second}.spill"))).expect("block target");
        let refusal = account.retain_disk_backed(&dir.0).expect_err("rename onto a directory");
        assert!(matches!(refusal, CapacityError::SpillIo { .. }));
        assert_eq!(account.retention(), OutputRetention::FullReusable);
        assert_eq!(account.resident_bytes().expect("payloads kept"), 4);
        assert!(account.stored_bytes(first).is_some());
        assert_eq!(dir.entries(), vec![format!("entry-{second}.spill")]);
    }

    #[test]
    fn decompression_buffer_stays_within_the_existing_cap() {
        assert_eq!(DECOMPRESSION_CAP, crate::container::MAX_SECTION_BYTES);
        assert_eq!(DECOMPRESSION_CAP, 512 * 1024 * 1024);
        let mut account = CapacityAccount::new(128);
        let mut buffer = account.reserve_decompression(64).expect("reserve");
        assert_eq!(buffer.requested(), 64);
        assert!(buffer.capacity() >= 64);
        assert!(buffer.requested() <= DECOMPRESSION_CAP);
        assert_eq!(buffer.as_slice().len(), 0);
        assert_eq!(account.resident_bytes().expect("charged"), 64);
        buffer.write(&[0; 64]).expect("fill");
        assert_eq!(buffer.as_slice().len(), 64);
        assert!(buffer.write(&[0; 65]).is_err());
        assert_eq!(buffer.as_slice().len(), 64);
        let over_cap = u64::try_from(DECOMPRESSION_CAP).expect("cap") + 1;
        let cap_refusal = account.reserve_decompression(over_cap).expect_err("cap");
        assert!(matches!(
            cap_refusal,
            CapacityError::DecompressionCap { cap: DECOMPRESSION_CAP, .. }
        ));
        assert_eq!(account.resident_bytes().expect("only the first reservation"), 64);
        let mut tight = CapacityAccount::new(16);
        let budget_refusal =
            tight.reserve_decompression(u64::try_from(DECOMPRESSION_CAP).expect("cap"));
        assert!(matches!(budget_refusal, Err(CapacityError::OverBudget { .. })));
        assert!(tight.is_empty());
        assert_eq!(tight.resident_bytes().expect("nothing allocated"), 0);
        account.release_decompression(buffer).expect("release");
        assert_eq!(account.resident_bytes().expect("released"), 0);
        assert!(account.require(AccountedOperation::ReadDiskArtifact).is_err());
    }
}
