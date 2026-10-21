//! Measured facts for one physical run.
//!
//! [`crate::execution::ExecutionReceipt`] names the host request and its
//! lifecycle. This module records what that run actually did: scans, copies,
//! fits, tests, cache hits, output retention, resources, spill, the backend,
//! and cancellation or failure. Counts are `u64`. A sum that does not fit is
//! refused. Nothing saturates and nothing wraps.
//!
//! [`PhysicalExecutionIdentity`] is derived only from those measurements.
//! [`ExecutionFacts::scientific_identity`] is an optional digest stored beside
//! them. [`ExecutionFacts::rebuild`] replaces the measurements and leaves that
//! digest as it was. [`ExecutionFacts::scientific_contract_changed`] is true
//! only when the caller passes or sets the numerical-policy flag. Kernel
//! policy is recorded with the backend; it is not read as that declaration.
//! [`ExecutionFactStatus::Cancelled`] and [`ExecutionFactStatus::Failed`] do
//! not publish a claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use crate::execution::KernelPolicy;
use crate::identity::{CanonicalWriter, SemanticDigest};

/// How output was retained after the run.
///
/// Local to this crate: core does not depend on `antecedent-state`. Each
/// variant is a mode. [`Self::DiskBacked`] stores no path. Spill volume is
/// [`ExecutionResources::spill_bytes`], a byte count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum OutputRetention {
    /// No output retention boundary was measured. Grants no reuse capability.
    Unmeasured,
    /// Full reusable in-memory state.
    #[default]
    FullReusable,
    /// Summary only.
    SummaryOnly,
    /// Disk-backed output, with no path stored here.
    DiskBacked,
}

impl OutputRetention {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unmeasured => "unmeasured",
            Self::FullReusable => "full reusable state",
            Self::SummaryOnly => "summary-only",
            Self::DiskBacked => "disk-backed output",
        }
    }
}

impl fmt::Display for OutputRetention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How the measured run ended.
///
/// Cancellation and failure are statuses. Neither publishes a claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ExecutionFactStatus {
    /// Finished. A claim may be published.
    Completed,
    /// Cancelled. A claim must not be published.
    Cancelled,
    /// Failed. A claim must not be published.
    Failed,
}

impl ExecutionFactStatus {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }

    /// Whether this status may publish a claim.
    #[must_use]
    pub const fn publishes_claim(self) -> bool {
        matches!(self, Self::Completed)
    }
}

impl fmt::Display for ExecutionFactStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Scans, copies, fits, tests, and cache hits actually performed.
///
/// Zero means none reported. Addition that does not fit in `u64` is refused.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ExecutionCounts {
    scans: u64,
    copies: u64,
    fits: u64,
    tests: u64,
    cache_hits: u64,
}

impl ExecutionCounts {
    /// Record the counts that were passed.
    #[must_use]
    pub const fn new(scans: u64, copies: u64, fits: u64, tests: u64, cache_hits: u64) -> Self {
        Self { scans, copies, fits, tests, cache_hits }
    }

    /// Scans actually performed.
    #[must_use]
    pub const fn scans(self) -> u64 {
        self.scans
    }

    /// Copies actually performed.
    #[must_use]
    pub const fn copies(self) -> u64 {
        self.copies
    }

    /// Fits actually performed.
    #[must_use]
    pub const fn fits(self) -> u64 {
        self.fits
    }

    /// Tests actually performed.
    #[must_use]
    pub const fn tests(self) -> u64 {
        self.tests
    }

    /// Cache hits actually served.
    #[must_use]
    pub const fn cache_hits(self) -> u64 {
        self.cache_hits
    }

    /// Add `extra`.
    ///
    /// # Errors
    ///
    /// [`ExecutionFactError::Overflow`] when any sum does not fit in `u64`.
    /// The sum is not wrapped and not saturated. `self` is left unchanged
    /// because this method consumes a copy.
    #[must_use = "an overflowing count is refused and was not wrapped"]
    pub fn try_add(self, extra: Self) -> Result<Self, ExecutionFactError> {
        Ok(Self {
            scans: checked_add(self.scans, extra.scans)?,
            copies: checked_add(self.copies, extra.copies)?,
            fits: checked_add(self.fits, extra.fits)?,
            tests: checked_add(self.tests, extra.tests)?,
            cache_hits: checked_add(self.cache_hits, extra.cache_hits)?,
        })
    }
}

/// Threads, memory, and spill actually used.
///
/// [`Self::spill_bytes`] is a byte count. It is not a path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ExecutionResources {
    threads: u64,
    memory_bytes: u64,
    spill_bytes: u64,
}

impl ExecutionResources {
    /// Record the resources that were passed.
    #[must_use]
    pub const fn new(threads: u64, memory_bytes: u64, spill_bytes: u64) -> Self {
        Self { threads, memory_bytes, spill_bytes }
    }

    /// Worker threads used.
    #[must_use]
    pub const fn threads(self) -> u64 {
        self.threads
    }

    /// Memory bytes used.
    #[must_use]
    pub const fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }

    /// Bytes spilled. A count, not a path.
    #[must_use]
    pub const fn spill_bytes(self) -> u64 {
        self.spill_bytes
    }

    /// Add `extra`.
    ///
    /// # Errors
    ///
    /// [`ExecutionFactError::Overflow`] when any sum does not fit in `u64`.
    /// The sum is not wrapped and not saturated. `self` is left unchanged
    /// because this method consumes a copy.
    #[must_use = "an overflowing resource count is refused and was not wrapped"]
    pub fn try_add(self, extra: Self) -> Result<Self, ExecutionFactError> {
        Ok(Self {
            threads: checked_add(self.threads, extra.threads)?,
            memory_bytes: checked_add(self.memory_bytes, extra.memory_bytes)?,
            spill_bytes: checked_add(self.spill_bytes, extra.spill_bytes)?,
        })
    }
}

/// Backend that ran: a caller-supplied name plus the kernel policy in force.
///
/// The name is stored as supplied. The policy is the numerical choice that was
/// used. Neither one declares that the scientific contract changed.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ExecutionBackend {
    name: String,
    kernel: KernelPolicy,
}

impl ExecutionBackend {
    /// Record `name` and the kernel policy the caller passed.
    #[must_use]
    pub const fn new(name: String, kernel: KernelPolicy) -> Self {
        Self { name, kernel }
    }

    /// Caller-supplied backend name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Kernel policy in force for this run.
    #[must_use]
    pub const fn kernel(&self) -> KernelPolicy {
        self.kernel
    }
}

/// Measurements, without a scientific digest.
///
/// Replacing this value is a rebuild of the facts. It is not a new scientific
/// identity.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MeasuredExecution {
    counts: ExecutionCounts,
    retention: OutputRetention,
    resources: ExecutionResources,
    backend: ExecutionBackend,
    status: ExecutionFactStatus,
    copied_bytes: Option<u64>,
    retained_bytes: Option<u64>,
}

impl MeasuredExecution {
    /// Record the measurements that were passed.
    #[must_use]
    pub const fn new(
        counts: ExecutionCounts,
        retention: OutputRetention,
        resources: ExecutionResources,
        backend: ExecutionBackend,
        status: ExecutionFactStatus,
    ) -> Self {
        Self {
            counts,
            retention,
            resources,
            backend,
            status,
            copied_bytes: None,
            retained_bytes: None,
        }
    }

    /// Store measured numeric buffer volumes. `None` means the boundary was unmeasured.
    ///
    /// Copied bytes describe numeric buffer transfers; retained bytes describe admitted
    /// output reservations at handoff, never allocator usage or process RSS.
    #[must_use]
    pub const fn with_observed_buffer_bytes(
        mut self,
        copied_bytes: Option<u64>,
        retained_bytes: Option<u64>,
    ) -> Self {
        self.copied_bytes = copied_bytes;
        self.retained_bytes = retained_bytes;
        self
    }

    /// Counts actually performed.
    #[must_use]
    pub const fn counts(&self) -> ExecutionCounts {
        self.counts
    }

    /// Output retention.
    #[must_use]
    pub const fn retention(&self) -> OutputRetention {
        self.retention
    }

    /// Resources and spill.
    #[must_use]
    pub const fn resources(&self) -> ExecutionResources {
        self.resources
    }

    /// Backend name and kernel policy.
    #[must_use]
    pub const fn backend(&self) -> &ExecutionBackend {
        &self.backend
    }

    /// Cancellation or failure status, or completion.
    #[must_use]
    pub const fn status(&self) -> ExecutionFactStatus {
        self.status
    }

    /// Add counts.
    ///
    /// # Errors
    ///
    /// [`ExecutionFactError::Overflow`] when any sum does not fit in `u64`.
    /// This value is unchanged on that refusal.
    pub fn try_add_counts(&mut self, extra: ExecutionCounts) -> Result<(), ExecutionFactError> {
        self.counts = self.counts.try_add(extra)?;
        Ok(())
    }

    /// Add resources, including spill bytes.
    ///
    /// # Errors
    ///
    /// [`ExecutionFactError::Overflow`] when any sum does not fit in `u64`.
    /// This value is unchanged on that refusal.
    pub fn try_add_resources(
        &mut self,
        extra: ExecutionResources,
    ) -> Result<(), ExecutionFactError> {
        self.resources = self.resources.try_add(extra)?;
        Ok(())
    }
}

/// A count sum did not fit in `u64`.
///
/// The sum was not wrapped and not saturated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutionFactError {
    /// A scan, copy, fit, test, cache-hit, thread, memory, or spill sum overflowed.
    #[error("execution fact count overflowed u64 and was refused")]
    Overflow,
}

/// Identity of the physical run.
///
/// This is not a scientific digest. It is produced from the measurements
/// alone: scans, copies, fits, tests, cache hits, retention, threads, memory
/// bytes, spill bytes, backend name, kernel policy, and status. The stored
/// scientific digest and the numerical-policy contract flag are not inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PhysicalExecutionIdentity {
    digest: SemanticDigest,
}

impl PhysicalExecutionIdentity {
    /// Digest bytes of the physical run.
    #[must_use]
    pub const fn digest(self) -> SemanticDigest {
        self.digest
    }
}

/// Measured facts with a scientific digest stored beside them.
///
/// The digest is whatever the caller stored. Rebuilding the measurements does
/// not recompute it. The numerical-policy flag is whatever the caller set.
/// Kernel policy, counts, retention, resources, spill, and status do not set it.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ExecutionFacts {
    native_coverage: Option<u64>,
    native_origin: bool,
    measured: MeasuredExecution,
    scientific_identity: Option<SemanticDigest>,
    numerical_policy_changes_scientific_contract: bool,
}

impl ExecutionFacts {
    /// Record measurements, an optional scientific digest, and the contract flag.
    ///
    /// `numerical_policy_changes_scientific_contract` is stored as passed.
    /// Nothing in `measured` is consulted to fill it in.
    #[must_use]
    pub const fn new(
        measured: MeasuredExecution,
        scientific_identity: Option<SemanticDigest>,
        numerical_policy_changes_scientific_contract: bool,
    ) -> Self {
        Self {
            native_coverage: None,
            native_origin: false,
            measured,
            scientific_identity,
            numerical_policy_changes_scientific_contract,
        }
    }

    /// Facts emitted by native execution; memory is peak admitted bytes, not RSS.
    pub(crate) fn from_native(measured: MeasuredExecution, coverage: u64) -> Self {
        Self {
            native_coverage: Some(coverage),
            native_origin: true,
            measured,
            scientific_identity: None,
            numerical_policy_changes_scientific_contract: false,
        }
    }
    /// Import non-authoritative observations recorded in an artifact.
    /// The imported record does not become a new native execution.
    #[must_use]
    pub fn from_artifact(measured: MeasuredExecution, coverage: u64) -> Self {
        Self {
            native_coverage: Some(coverage),
            native_origin: false,
            measured,
            scientific_identity: None,
            numerical_policy_changes_scientific_contract: false,
        }
    }
    /// Whether the engine emitted this record instead of accepting caller declarations.
    #[must_use]
    pub fn is_native(&self) -> bool {
        self.native_origin
    }
    /// Instrumented physical-event bitset; absent fields remain unknown.
    #[must_use]
    pub fn native_coverage(&self) -> Option<u64> {
        self.native_coverage
    }

    /// Scans observed by instrumented source boundaries; absent means unmeasured.
    #[must_use]
    pub fn observed_scans(&self) -> Option<u64> {
        self.covered(0).then_some(self.counts().scans())
    }
    /// Copies observed at instrumented buffer boundaries.
    #[must_use]
    pub fn observed_copies(&self) -> Option<u64> {
        self.covered(1).then_some(self.counts().copies())
    }
    /// Bytes copied at instrumented numeric buffer boundaries, or unknown.
    #[must_use]
    pub fn observed_copied_bytes(&self) -> Option<u64> {
        self.covered(2).then_some(self.measured.copied_bytes).flatten()
    }
    /// Admitted output reservation bytes at instrumented handoffs, or unknown.
    /// This cumulative volume can conservatively cover owned allocations. It is
    /// neither exact heap allocation nor peak live memory nor process RSS.
    #[must_use]
    pub fn observed_retained_bytes(&self) -> Option<u64> {
        self.covered(4).then_some(self.measured.retained_bytes).flatten()
    }
    /// Completed enclosing model fits at instrumented engine boundaries.
    #[must_use]
    pub fn observed_model_fits(&self) -> Option<u64> {
        self.covered(9).then_some(self.counts().fits())
    }
    /// Peak admitted reservations, never process RSS.
    #[must_use]
    pub fn peak_admitted_bytes(&self) -> Option<u64> {
        self.covered(6).then_some(self.resources().memory_bytes())
    }
    /// Peak admitted CPU slots, never an OS thread census.
    #[must_use]
    pub fn peak_admitted_threads(&self) -> Option<u64> {
        self.covered(7).then_some(self.resources().threads())
    }
    /// Actual spill bytes when the spill boundary was observed.
    #[must_use]
    pub fn observed_spill_bytes(&self) -> Option<u64> {
        self.covered(5).then_some(self.resources().spill_bytes())
    }
    fn covered(&self, bit: u32) -> bool {
        self.native_coverage.is_some_and(|mask| mask & (1 << bit) != 0)
    }

    /// Replace the measurements.
    ///
    /// The stored scientific digest and the numerical-policy flag stay as they
    /// are. The flag is not inferred from `measured`.
    #[must_use]
    pub fn rebuild(mut self, measured: MeasuredExecution) -> Self {
        self.measured = measured;
        self.native_coverage = None;
        self.native_origin = false;
        self
    }

    /// Measurements.
    #[must_use]
    pub const fn measured(&self) -> &MeasuredExecution {
        &self.measured
    }

    /// Counts actually performed.
    #[must_use]
    pub const fn counts(&self) -> ExecutionCounts {
        self.measured.counts
    }

    /// Output retention.
    #[must_use]
    pub const fn retention(&self) -> OutputRetention {
        self.measured.retention
    }

    /// Resources and spill.
    #[must_use]
    pub const fn resources(&self) -> ExecutionResources {
        self.measured.resources
    }

    /// Backend name and kernel policy.
    #[must_use]
    pub const fn backend(&self) -> &ExecutionBackend {
        &self.measured.backend
    }

    /// Cancellation, failure, or completion.
    #[must_use]
    pub const fn status(&self) -> ExecutionFactStatus {
        self.measured.status
    }

    /// Scientific digest stored beside the facts, if the caller supplied one.
    #[must_use]
    pub const fn scientific_identity(&self) -> Option<SemanticDigest> {
        self.scientific_identity
    }

    /// Store `scientific_identity` beside the current facts.
    ///
    /// The measurements are not an input to this digest.
    #[must_use]
    pub const fn with_scientific_identity(
        mut self,
        scientific_identity: Option<SemanticDigest>,
    ) -> Self {
        self.scientific_identity = scientific_identity;
        self
    }

    /// Whether the caller said numerical policy changes the scientific contract.
    ///
    /// This is the stored flag. It is not derived from kernel policy or from
    /// any other measurement.
    #[must_use]
    pub const fn scientific_contract_changed(&self) -> bool {
        self.numerical_policy_changes_scientific_contract
    }

    /// Set the numerical-policy flag to `changes`.
    ///
    /// This is the only write of that flag after construction. The scientific
    /// digest is left as stored.
    #[must_use]
    pub const fn declare_numerical_policy_changes_scientific_contract(
        mut self,
        changes: bool,
    ) -> Self {
        self.numerical_policy_changes_scientific_contract = changes;
        self
    }

    /// Whether the status may publish a claim.
    ///
    /// Cancellation and failure do not. The numerical-policy flag does not
    /// publish one either.
    #[must_use]
    pub const fn publishes_claim(&self) -> bool {
        self.measured.status.publishes_claim()
    }

    /// Physical identity of the measurements.
    ///
    /// The scientific digest and the numerical-policy flag are not inputs.
    #[must_use]
    pub fn physical_identity(&self) -> PhysicalExecutionIdentity {
        physical_digest(&self.measured)
    }

    /// Add counts.
    ///
    /// # Errors
    ///
    /// [`ExecutionFactError::Overflow`] when any sum does not fit in `u64`.
    /// The facts, including the scientific digest, are unchanged on that refusal.
    pub fn try_add_counts(&mut self, extra: ExecutionCounts) -> Result<(), ExecutionFactError> {
        self.measured.try_add_counts(extra)?;
        self.native_origin = false;
        self.native_coverage = None;
        Ok(())
    }

    /// Add resources, including spill bytes.
    ///
    /// # Errors
    ///
    /// [`ExecutionFactError::Overflow`] when any sum does not fit in `u64`.
    /// The facts, including the scientific digest, are unchanged on that refusal.
    pub fn try_add_resources(
        &mut self,
        extra: ExecutionResources,
    ) -> Result<(), ExecutionFactError> {
        self.measured.try_add_resources(extra)?;
        self.native_origin = false;
        self.native_coverage = None;
        Ok(())
    }
}

fn checked_add(left: u64, right: u64) -> Result<u64, ExecutionFactError> {
    left.checked_add(right).ok_or(ExecutionFactError::Overflow)
}

fn physical_digest(measured: &MeasuredExecution) -> PhysicalExecutionIdentity {
    let mut writer = CanonicalWriter::new("antecedent.execution_facts.physical.v1");
    let counts = measured.counts;
    writer.word(counts.scans);
    writer.word(counts.copies);
    writer.word(counts.fits);
    writer.word(counts.tests);
    writer.word(counts.cache_hits);
    writer.word(retention_code(measured.retention));
    let resources = measured.resources;
    writer.word(resources.threads);
    writer.word(resources.memory_bytes);
    writer.word(resources.spill_bytes);
    writer.text(&measured.backend.name);
    let kernel = measured.backend.kernel;
    writer.word(u64::from(kernel.allow_portable_optimized));
    writer.word(u64::from(kernel.allow_arch_simd));
    writer.word(u64::from(kernel.force_scalar));
    writer.word(status_code(measured.status));
    // Preserve v1 physical identities when optional buffer boundaries were absent.
    if measured.copied_bytes.is_some() || measured.retained_bytes.is_some() {
        writer.text("buffer_volumes.v2");
        writer.word(u64::from(measured.copied_bytes.is_some()));
        writer.word(measured.copied_bytes.unwrap_or(0));
        writer.word(u64::from(measured.retained_bytes.is_some()));
        writer.word(measured.retained_bytes.unwrap_or(0));
    }
    PhysicalExecutionIdentity { digest: SemanticDigest::from_bytes(writer.finish()) }
}

const fn retention_code(retention: OutputRetention) -> u64 {
    match retention {
        OutputRetention::Unmeasured => 0,
        OutputRetention::FullReusable => 1,
        OutputRetention::SummaryOnly => 2,
        OutputRetention::DiskBacked => 3,
    }
}

const fn status_code(status: ExecutionFactStatus) -> u64 {
    match status {
        ExecutionFactStatus::Completed => 1,
        ExecutionFactStatus::Cancelled => 2,
        ExecutionFactStatus::Failed => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExecutionBackend, ExecutionCounts, ExecutionFactError, ExecutionFactStatus, ExecutionFacts,
        ExecutionResources, MeasuredExecution, OutputRetention,
    };
    use crate::execution::KernelPolicy;
    use crate::identity::SemanticDigest;

    const fn counts() -> ExecutionCounts {
        ExecutionCounts::new(2, 3, 5, 7, 11)
    }

    const fn resources() -> ExecutionResources {
        ExecutionResources::new(4, 1024, 64)
    }

    fn measured(
        counts: ExecutionCounts,
        retention: OutputRetention,
        resources: ExecutionResources,
        kernel: KernelPolicy,
        status: ExecutionFactStatus,
    ) -> MeasuredExecution {
        MeasuredExecution::new(
            counts,
            retention,
            resources,
            ExecutionBackend::new("cpu".to_owned(), kernel),
            status,
        )
    }

    const fn facts(
        run: MeasuredExecution,
        scientific: Option<SemanticDigest>,
        contract_changed: bool,
    ) -> ExecutionFacts {
        ExecutionFacts::new(run, scientific, contract_changed)
    }

    #[test]
    fn counts_record_what_was_passed() {
        let run = MeasuredExecution::new(
            counts(),
            OutputRetention::DiskBacked,
            resources(),
            ExecutionBackend::new("portable".to_owned(), KernelPolicy::scalar_only()),
            ExecutionFactStatus::Completed,
        );
        let record = facts(run, None, false);
        assert_eq!(record.counts().scans(), 2);
        assert_eq!(record.counts().copies(), 3);
        assert_eq!(record.counts().fits(), 5);
        assert_eq!(record.counts().tests(), 7);
        assert_eq!(record.counts().cache_hits(), 11);
        assert_eq!(record.resources().threads(), 4);
        assert_eq!(record.resources().memory_bytes(), 1024);
        assert_eq!(record.resources().spill_bytes(), 64);
        assert_eq!(record.retention(), OutputRetention::DiskBacked);
        assert_eq!(record.backend().name(), "portable");
        assert_eq!(record.backend().kernel(), KernelPolicy::scalar_only());
        assert_eq!(record.status(), ExecutionFactStatus::Completed);
    }

    #[test]
    fn overflow_refuses() {
        let one = ExecutionCounts::new(1, 1, 1, 1, 1);
        let at_limit = [
            ExecutionCounts::new(u64::MAX, 0, 0, 0, 0),
            ExecutionCounts::new(0, u64::MAX, 0, 0, 0),
            ExecutionCounts::new(0, 0, u64::MAX, 0, 0),
            ExecutionCounts::new(0, 0, 0, u64::MAX, 0),
            ExecutionCounts::new(0, 0, 0, 0, u64::MAX),
        ];
        for counts in at_limit {
            assert_eq!(counts.try_add(one), Err(ExecutionFactError::Overflow));
        }
        let unchanged = ExecutionCounts::new(u64::MAX, 1, 2, 3, 4);
        assert_eq!(unchanged.try_add(ExecutionCounts::new(0, 0, 0, 0, 0)), Ok(unchanged));
        let spill = ExecutionResources::new(1, 2, u64::MAX);
        assert_eq!(
            spill.try_add(ExecutionResources::new(0, 0, 1)),
            Err(ExecutionFactError::Overflow)
        );
        assert_eq!(spill.spill_bytes(), u64::MAX);

        let digest = SemanticDigest::from_bytes([3; 32]);
        let mut record = facts(
            measured(
                ExecutionCounts::new(u64::MAX, 0, 0, 0, 0),
                OutputRetention::FullReusable,
                spill,
                KernelPolicy::default_policy(),
                ExecutionFactStatus::Completed,
            ),
            Some(digest),
            false,
        );
        let refused = record.try_add_counts(ExecutionCounts::new(1, 0, 0, 0, 0));
        assert_eq!(refused, Err(ExecutionFactError::Overflow));
        assert_eq!(record.counts().scans(), u64::MAX);
        assert_eq!(record.scientific_identity(), Some(digest));
        let refused_spill = record.try_add_resources(ExecutionResources::new(0, 0, 1));
        assert_eq!(refused_spill, Err(ExecutionFactError::Overflow));
        assert_eq!(record.resources().spill_bytes(), u64::MAX);
        assert_eq!(record.scientific_identity(), Some(digest));
    }

    #[test]
    fn cancelled_status_does_not_publish() {
        assert!(!ExecutionFactStatus::Cancelled.publishes_claim());
        assert!(!ExecutionFactStatus::Failed.publishes_claim());
        assert!(ExecutionFactStatus::Completed.publishes_claim());
        let digest = SemanticDigest::from_bytes([5; 32]);
        let completed = facts(
            measured(
                counts(),
                OutputRetention::FullReusable,
                resources(),
                KernelPolicy::default_policy(),
                ExecutionFactStatus::Completed,
            ),
            Some(digest),
            false,
        );
        assert!(completed.publishes_claim());
        let cancelled = completed.rebuild(measured(
            counts(),
            OutputRetention::FullReusable,
            resources(),
            KernelPolicy::default_policy(),
            ExecutionFactStatus::Cancelled,
        ));
        assert_eq!(cancelled.status(), ExecutionFactStatus::Cancelled);
        assert!(!cancelled.publishes_claim());
        assert_eq!(cancelled.counts(), counts());
        assert_eq!(cancelled.scientific_identity(), Some(digest));
        let flagged = cancelled.declare_numerical_policy_changes_scientific_contract(true);
        assert!(flagged.scientific_contract_changed());
        assert!(!flagged.publishes_claim());
        let failed = flagged.rebuild(measured(
            counts(),
            OutputRetention::SummaryOnly,
            resources(),
            KernelPolicy::default_policy(),
            ExecutionFactStatus::Failed,
        ));
        assert!(!failed.publishes_claim());
        assert_eq!(failed.status().as_str(), "failed");
    }

    #[test]
    fn scan_count_change_leaves_scientific_digest_unchanged() {
        let digest = SemanticDigest::from_bytes([7; 32]);
        let original = facts(
            measured(
                counts(),
                OutputRetention::FullReusable,
                resources(),
                KernelPolicy::default_policy(),
                ExecutionFactStatus::Completed,
            ),
            Some(digest),
            false,
        );
        let rebuilt = original.clone().rebuild(measured(
            ExecutionCounts::new(9, 3, 5, 7, 11),
            OutputRetention::FullReusable,
            resources(),
            KernelPolicy::default_policy(),
            ExecutionFactStatus::Completed,
        ));
        assert_eq!(rebuilt.counts().scans(), 9);
        assert_eq!(rebuilt.scientific_identity(), Some(digest));
        assert_eq!(original.scientific_identity(), rebuilt.scientific_identity());
        assert_ne!(rebuilt.physical_identity(), original.physical_identity());
        let other_digest = SemanticDigest::from_bytes([8; 32]);
        let original_physical = original.physical_identity();
        let original_scans = original.counts().scans();
        let replaced = original.with_scientific_identity(Some(other_digest));
        assert_eq!(replaced.scientific_identity(), Some(other_digest));
        assert_eq!(replaced.physical_identity(), original_physical);
        assert_eq!(replaced.counts().scans(), original_scans);
    }

    #[test]
    fn numerical_policy_flag_is_the_only_scientific_contract_change() {
        let digest = SemanticDigest::from_bytes([9; 32]);
        let original = facts(
            measured(
                counts(),
                OutputRetention::FullReusable,
                resources(),
                KernelPolicy::default_policy(),
                ExecutionFactStatus::Completed,
            ),
            Some(digest),
            false,
        );
        assert!(!original.scientific_contract_changed());
        let rebuilt = original.clone().rebuild(MeasuredExecution::new(
            ExecutionCounts::new(40, 1, 1, 1, 1),
            OutputRetention::SummaryOnly,
            ExecutionResources::new(8, 2048, 128),
            ExecutionBackend::new("other".to_owned(), KernelPolicy::scalar_only()),
            ExecutionFactStatus::Cancelled,
        ));
        assert_ne!(rebuilt.backend().kernel(), original.backend().kernel());
        assert_ne!(rebuilt.counts().scans(), original.counts().scans());
        assert_ne!(rebuilt.retention(), original.retention());
        assert_ne!(rebuilt.resources().spill_bytes(), original.resources().spill_bytes());
        assert!(!rebuilt.publishes_claim());
        assert!(!rebuilt.scientific_contract_changed());
        assert_eq!(rebuilt.scientific_identity(), Some(digest));
        assert_ne!(rebuilt.physical_identity(), original.physical_identity());
        let rebuilt_physical = rebuilt.physical_identity();
        let flagged = rebuilt.declare_numerical_policy_changes_scientific_contract(true);
        assert!(flagged.scientific_contract_changed());
        assert_eq!(flagged.scientific_identity(), Some(digest));
        assert_eq!(flagged.physical_identity(), rebuilt_physical);
        let cleared = flagged.declare_numerical_policy_changes_scientific_contract(false);
        assert!(!cleared.scientific_contract_changed());
        assert_eq!(cleared.scientific_identity(), Some(digest));
        let explicit = ExecutionFacts::new(original.measured().clone(), Some(digest), true);
        assert!(explicit.scientific_contract_changed());
        assert!(!original.scientific_contract_changed());
        assert_eq!(explicit.physical_identity(), original.physical_identity());
        assert_eq!(explicit.scientific_identity(), original.scientific_identity());
    }
    #[test]
    fn measured_buffer_volumes_affect_physical_identity_without_becoming_caller_authority() {
        let values = measured(
            counts(),
            OutputRetention::Unmeasured,
            resources(),
            KernelPolicy::default_policy(),
            ExecutionFactStatus::Completed,
        );
        let plain = ExecutionFacts::new(values.clone(), None, false);
        let measured = ExecutionFacts::from_artifact(
            values.with_observed_buffer_bytes(Some(0), Some(128)),
            (1 << 2) | (1 << 4),
        );
        assert_eq!(plain.observed_copied_bytes(), None);
        assert_eq!(measured.observed_copied_bytes(), Some(0));
        assert_eq!(measured.observed_retained_bytes(), Some(128));
        assert_ne!(plain.physical_identity(), measured.physical_identity());
        assert!(!measured.is_native());
        let changed = measured.rebuild(plain.measured().clone());
        assert_eq!(changed.observed_copied_bytes(), None);
        assert_eq!(changed.observed_retained_bytes(), None);
    }
}
