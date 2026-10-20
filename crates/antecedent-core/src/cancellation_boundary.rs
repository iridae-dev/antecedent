//! Cancellation boundaries for scans, solvers, resampling, search, chains,
//! providers, and device work.
//!
//! A [`Boundary`] names the site, the maximum call size in bytes, the latency
//! bound, whether a partial result may be retained, and whether that partial
//! may be published as a claim. Publishing a partial result as a claim is
//! refused. The latency bound is stored. A recorded elapsed [`std::time::Duration`]
//! is compared with that bound. This module does not sleep and does not read a
//! clock.
//!
//! Cancellation is reported when the bounded call returns to the boundary.
//! A provider that cannot be interrupted is admitted as
//! [`ProviderDisposition::Isolated`] or [`ProviderDisposition::Refused`].
//!
//! [`BoundarySession::stop`] reports [`BoundaryReport::Stopped`] only after
//! lease recovery and spill cleanup are both recorded. The lease id is a
//! caller-supplied `u64`; recovery sets the flag on the receipt.
//! [`BoundarySession::succeed`] is refused when a spill was opened and its
//! cleanup was not recorded. An unrecorded call size or elapsed time is an
//! absent measurement, not zero.
//!
//! Spill open and cleanup are records of a temporary spill. This module does
//! not create or delete files. [`CancelSite::Device`] is the boundary for
//! device work. Device reservation is a separate account.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::time::Duration;

use thiserror::Error;

/// Site at which a bounded call returns to its cancellation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CancelSite {
    /// Table or temporal scan.
    Scan,
    /// Numerical solve.
    Solver,
    /// Resampling, including bootstrap replicates.
    Resample,
    /// Bounded search.
    Search,
    /// Markov chain.
    Chain,
    /// External provider call.
    Provider,
    /// Device work. Reservation of the device is a separate account.
    Device,
}

impl CancelSite {
    /// Sites in declaration order.
    pub const ALL: [Self; 7] = [
        Self::Scan,
        Self::Solver,
        Self::Resample,
        Self::Search,
        Self::Chain,
        Self::Provider,
        Self::Device,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scan => "scan",
            Self::Solver => "solver",
            Self::Resample => "resample",
            Self::Search => "search",
            Self::Chain => "chain",
            Self::Provider => "provider",
            Self::Device => "device",
        }
    }
}

impl fmt::Display for CancelSite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Admission chosen when a provider cannot be interrupted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UninterruptibleAdmission {
    /// Keep the provider on its own bounded call until that call returns.
    Isolate,
    /// Leave the provider unentered.
    Refuse,
}

impl UninterruptibleAdmission {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Isolate => "isolate",
            Self::Refuse => "refuse",
        }
    }
}

impl fmt::Display for UninterruptibleAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How a provider call sits relative to the boundary.
///
/// `interruptible == false` is [`Self::Isolated`] or [`Self::Refused`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ProviderDisposition {
    /// The provider returns between calls. Cancellation waits for that return.
    Interruptible,
    /// The provider cannot be interrupted. Its call runs until it returns.
    Isolated,
    /// The provider cannot be interrupted and is not entered.
    Refused,
}

impl ProviderDisposition {
    /// Admit a provider.
    ///
    /// `uninterruptible` is used when `interruptible` is false. A true
    /// `interruptible` flag admits [`Self::Interruptible`].
    #[must_use]
    pub const fn admit(interruptible: bool, uninterruptible: UninterruptibleAdmission) -> Self {
        if interruptible {
            Self::Interruptible
        } else {
            match uninterruptible {
                UninterruptibleAdmission::Isolate => Self::Isolated,
                UninterruptibleAdmission::Refuse => Self::Refused,
            }
        }
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Interruptible => "interruptible",
            Self::Isolated => "isolated",
            Self::Refused => "refused",
        }
    }
}

impl fmt::Display for ProviderDisposition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Partial work retained at a boundary.
///
/// Retention is the declared partial. The published-claim flag stays clear.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PartialResult {
    site: CancelSite,
    bytes: u64,
    published_claim: bool,
}

impl PartialResult {
    /// Site that retained this partial.
    #[must_use]
    pub const fn site(self) -> CancelSite {
        self.site
    }

    /// Retained size in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Whether this partial was published as a claim.
    ///
    /// The flag is clear. [`BoundarySession::publish_claim`] refuses the publication.
    #[must_use]
    pub const fn is_published_claim(self) -> bool {
        self.published_claim
    }
}

/// Declared cancellation boundary for one site.
///
/// The maximum call size and the latency bound are inclusive: a recorded value
/// equal to the bound is within it. A larger recorded value is refused. Zero is
/// a declared bound, not an absent one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Boundary {
    site: CancelSite,
    max_call_bytes: u64,
    latency_bound: Duration,
    retain_partial: bool,
    publish_partial_claim: bool,
}

impl Boundary {
    /// Boundary for `site`.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::PartialClaimRefused`] when `publish_partial_claim` is
    /// true. A partial result is not published as a claim.
    #[must_use = "publishing a partial result as a claim is refused"]
    pub const fn new(
        site: CancelSite,
        max_call_bytes: u64,
        latency_bound: Duration,
        retain_partial: bool,
        publish_partial_claim: bool,
    ) -> Result<Self, BoundaryError> {
        if publish_partial_claim {
            return Err(BoundaryError::PartialClaimRefused { site });
        }
        Ok(Self {
            site,
            max_call_bytes,
            latency_bound,
            retain_partial,
            publish_partial_claim: false,
        })
    }

    /// Site this boundary watches.
    #[must_use]
    pub const fn site(self) -> CancelSite {
        self.site
    }

    /// Maximum call size in bytes.
    #[must_use]
    pub const fn max_call_bytes(self) -> u64 {
        self.max_call_bytes
    }

    /// Stored latency bound.
    ///
    /// The bound is the value to compare with a recorded elapsed time.
    #[must_use]
    pub const fn latency_bound(self) -> Duration {
        self.latency_bound
    }

    /// Whether a partial result may be retained.
    #[must_use]
    pub const fn retain_partial(self) -> bool {
        self.retain_partial
    }

    /// Whether a partial result may be published as a claim.
    ///
    /// A constructed boundary returns false. The publication is refused.
    #[must_use]
    pub const fn publish_partial_claim(self) -> bool {
        self.publish_partial_claim
    }

    /// Open the boundary. Lease recovery is recorded later from a
    /// [`DeviceReservationLease`](crate::device_reservation::DeviceReservationLease).
    #[must_use]
    pub const fn open(self) -> BoundarySession {
        BoundarySession {
            boundary: self,
            call_bytes: None,
            elapsed: None,
            spill: SpillAccount::Idle,
            lease_recovered: false,
            partial: None,
            provider: None,
        }
    }
}

/// Refusal from a cancellation boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum BoundaryError {
    /// Recorded call size is larger than the declared maximum.
    #[error("call of {bytes} bytes exceeds the {max_call_bytes}-byte bound at {site}")]
    CallTooLarge {
        /// Site of the bounded call.
        site: CancelSite,
        /// Recorded call size in bytes.
        bytes: u64,
        /// Declared maximum in bytes.
        max_call_bytes: u64,
    },
    /// Call size was not recorded. Absence is not a zero-byte call.
    #[error("call size was not recorded at {site}")]
    CallSizeNotRecorded {
        /// Site of the bounded call.
        site: CancelSite,
    },
    /// Recorded elapsed time is longer than the stored latency bound.
    #[error("recorded elapsed {elapsed:?} exceeds the latency bound {bound:?} at {site}")]
    LatencyExceeded {
        /// Site of the bounded call.
        site: CancelSite,
        /// Elapsed time the caller recorded.
        elapsed: Duration,
        /// Stored latency bound.
        bound: Duration,
    },
    /// Elapsed time was not recorded. Absence is not a zero duration.
    #[error("elapsed time was not recorded at {site}")]
    ElapsedNotRecorded {
        /// Site of the bounded call.
        site: CancelSite,
    },
    /// A partial result was offered for publication as a claim.
    #[error("publishing a partial result as a claim is refused at {site}")]
    PartialClaimRefused {
        /// Site that retained, or was asked to publish, the partial.
        site: CancelSite,
    },
    /// The boundary does not retain partial results.
    #[error("partial results are not retained at {site}")]
    PartialRetentionRefused {
        /// Site whose declaration leaves partials unretained.
        site: CancelSite,
    },
    /// A provider that cannot be interrupted was refused and not entered.
    #[error("non-interruptible provider is refused")]
    UninterruptibleProvider,
    /// Provider admission was asked for a site other than [`CancelSite::Provider`].
    #[error("provider admission requires the provider site, found {site}")]
    ProviderSiteRequired {
        /// Site of the session that was asked to admit a provider.
        site: CancelSite,
    },
    /// Lease recovery was not derived from a live device reservation.
    #[error("device reservation lease was not recovered")]
    LeaseNotRecovered,
    /// Spill cleanup was not recorded.
    #[error("spill cleanup was not recorded at {site}")]
    MissingSpillCleanup {
        /// Site whose spill account is still short of cleanup.
        site: CancelSite,
    },
    /// A spill is already open and cleanup has not been recorded.
    #[error("spill is already open at {site}")]
    SpillAlreadyOpen {
        /// Site with an open spill record.
        site: CancelSite,
    },
}

/// Recorded state of a temporary spill.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum SpillAccount {
    /// No spill is open and cleanup has not been recorded.
    Idle,
    /// A spill was opened and cleanup has not been recorded.
    Open,
    /// Cleanup was recorded and no spill had been opened.
    CleanedWithoutSpill,
    /// Cleanup was recorded after a spill had been opened.
    CleanedAfterSpill,
}

impl SpillAccount {
    const fn cleanup_recorded(self) -> bool {
        matches!(self, Self::CleanedWithoutSpill | Self::CleanedAfterSpill)
    }
}

/// Open boundary: recorded call size, latency, lease recovery, and spill cleanup.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BoundarySession {
    boundary: Boundary,
    call_bytes: Option<u64>,
    elapsed: Option<Duration>,
    spill: SpillAccount,
    lease_recovered: bool,
    partial: Option<PartialResult>,
    provider: Option<ProviderDisposition>,
}

impl BoundarySession {
    /// Declaration this session is enforcing.
    #[must_use]
    pub const fn boundary(self) -> Boundary {
        self.boundary
    }

    /// Recorded call size, when one has been recorded.
    #[must_use]
    pub const fn recorded_call_bytes(self) -> Option<u64> {
        self.call_bytes
    }

    /// Recorded elapsed time, when one has been recorded.
    #[must_use]
    pub const fn recorded_elapsed(self) -> Option<Duration> {
        self.elapsed
    }

    /// Whether [`Self::recover_lease`] released a live non-zero reservation.
    #[must_use]
    pub const fn lease_recovered(self) -> bool {
        self.lease_recovered
    }

    /// Whether spill cleanup has been recorded.
    #[must_use]
    pub const fn spill_cleanup_recorded(self) -> bool {
        self.spill.cleanup_recorded()
    }

    /// Provider admission recorded for this session, when one has been.
    #[must_use]
    pub const fn provider(self) -> Option<ProviderDisposition> {
        self.provider
    }

    /// Record the call size in bytes.
    ///
    /// The size is stored before the bound is applied. A size equal to the
    /// maximum is within the bound.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::CallTooLarge`] when `bytes` is greater than the maximum.
    pub const fn record_call_bytes(&mut self, bytes: u64) -> Result<(), BoundaryError> {
        self.call_bytes = Some(bytes);
        if bytes > self.boundary.max_call_bytes {
            return Err(BoundaryError::CallTooLarge {
                site: self.boundary.site,
                bytes,
                max_call_bytes: self.boundary.max_call_bytes,
            });
        }
        Ok(())
    }

    /// Record elapsed time for comparison with the stored latency bound.
    ///
    /// The duration is stored. Nothing here sleeps or reads a clock. A duration
    /// equal to the bound is within it.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::LatencyExceeded`] when `elapsed` is greater than the bound.
    pub fn record_elapsed(&mut self, elapsed: Duration) -> Result<(), BoundaryError> {
        self.elapsed = Some(elapsed);
        if elapsed > self.boundary.latency_bound {
            return Err(BoundaryError::LatencyExceeded {
                site: self.boundary.site,
                elapsed,
                bound: self.boundary.latency_bound,
            });
        }
        Ok(())
    }

    /// Record that a temporary spill was opened.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::SpillAlreadyOpen`] when a spill is already open.
    pub const fn open_spill(&mut self) -> Result<(), BoundaryError> {
        if matches!(self.spill, SpillAccount::Open) {
            return Err(BoundaryError::SpillAlreadyOpen { site: self.boundary.site });
        }
        self.spill = SpillAccount::Open;
        Ok(())
    }

    /// Record that temporary-spill cleanup has been done.
    ///
    /// The record is the cleanup step. It is required before [`Self::stop`],
    /// including when no spill was opened.
    pub const fn record_spill_cleanup(&mut self) {
        self.spill = match self.spill {
            SpillAccount::Open | SpillAccount::CleanedAfterSpill => SpillAccount::CleanedAfterSpill,
            SpillAccount::Idle | SpillAccount::CleanedWithoutSpill => {
                SpillAccount::CleanedWithoutSpill
            }
        };
    }

    /// Release `lease` and record recovery only when that release returns the
    /// same non-zero remainder `lease` still held.
    ///
    /// An already-released lease reports a zero remainder, so it does not count.
    /// A live zero-byte reservation is indistinguishable from that case through
    /// the public lease API and does not count either.
    pub fn recover_lease(
        &mut self,
        lease: crate::device_reservation::DeviceReservationLease,
    ) -> bool {
        let held = lease.remainder();
        let released = lease.release();
        let recovered = held.device_bytes() > 0
            && released.device_bytes() == held.device_bytes()
            && released.spill_bytes() == held.spill_bytes();
        self.lease_recovered = recovered;
        recovered
    }

    /// Retain a partial result.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::PartialRetentionRefused`] when the boundary does not
    /// retain partials. [`BoundaryError::CallTooLarge`] when `bytes` exceeds
    /// the maximum call size.
    pub const fn retain_partial_result(
        &mut self,
        bytes: u64,
    ) -> Result<PartialResult, BoundaryError> {
        if !self.boundary.retain_partial {
            return Err(BoundaryError::PartialRetentionRefused { site: self.boundary.site });
        }
        if bytes > self.boundary.max_call_bytes {
            return Err(BoundaryError::CallTooLarge {
                site: self.boundary.site,
                bytes,
                max_call_bytes: self.boundary.max_call_bytes,
            });
        }
        let partial = PartialResult { site: self.boundary.site, bytes, published_claim: false };
        self.partial = Some(partial);
        Ok(partial)
    }

    /// Publish the partial result as a claim.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::PartialClaimRefused`] always. A partial result is not a
    /// published claim.
    pub const fn publish_claim(&self) -> Result<(), BoundaryError> {
        let site = match self.partial {
            Some(partial) => partial.site(),
            None => self.boundary.site,
        };
        Err(BoundaryError::PartialClaimRefused { site })
    }

    /// Admit a provider at [`CancelSite::Provider`].
    ///
    /// A provider with `interruptible` false is isolated or refused.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::ProviderSiteRequired`] when the site is not the provider
    /// site. [`BoundaryError::UninterruptibleProvider`] when the admission is
    /// [`ProviderDisposition::Refused`].
    pub const fn enter_provider(
        &mut self,
        interruptible: bool,
        uninterruptible: UninterruptibleAdmission,
    ) -> Result<ProviderDisposition, BoundaryError> {
        if !matches!(self.boundary.site, CancelSite::Provider) {
            return Err(BoundaryError::ProviderSiteRequired { site: self.boundary.site });
        }
        let disposition = ProviderDisposition::admit(interruptible, uninterruptible);
        self.provider = Some(disposition);
        match disposition {
            ProviderDisposition::Refused => Err(BoundaryError::UninterruptibleProvider),
            ProviderDisposition::Interruptible | ProviderDisposition::Isolated => Ok(disposition),
        }
    }

    /// Report [`BoundaryReport::Stopped`].
    ///
    /// Lease recovery and spill cleanup must both already be recorded.
    ///
    /// # Errors
    ///
    /// [`BoundaryError::UninterruptibleProvider`] when a refused provider was
    /// recorded. [`BoundaryError::CallSizeNotRecorded`] or
    /// [`BoundaryError::CallTooLarge`] for the call size.
    /// [`BoundaryError::ElapsedNotRecorded`] or [`BoundaryError::LatencyExceeded`]
    /// for the elapsed time. [`BoundaryError::LeaseNotRecovered`] when recovery
    /// was not recorded. [`BoundaryError::MissingSpillCleanup`] when spill
    /// cleanup was not recorded.
    pub fn stop(self) -> Result<StopReceipt, BoundaryError> {
        let assessed = self.assess()?;
        if !self.lease_recovered {
            return Err(BoundaryError::LeaseNotRecovered);
        }
        if !self.spill.cleanup_recorded() {
            return Err(BoundaryError::MissingSpillCleanup { site: self.boundary.site });
        }
        Ok(StopReceipt {
            report: BoundaryReport::Stopped,
            site: self.boundary.site,
            lease_recovered: self.lease_recovered,
            spill_cleaned: self.spill.cleanup_recorded(),
            partial: self.partial,
            published_claim: published_claim_of(self.partial),
            call_bytes: assessed.call_bytes,
            elapsed: assessed.elapsed,
            provider: self.provider,
        })
    }

    /// Finish the bounded call.
    ///
    /// # Errors
    ///
    /// The call-size, latency, and refused-provider errors of [`Self::stop`].
    /// [`BoundaryError::MissingSpillCleanup`] when a spill was opened and
    /// cleanup was not recorded.
    pub fn succeed(self) -> Result<SuccessReceipt, BoundaryError> {
        let assessed = self.assess()?;
        if matches!(self.spill, SpillAccount::Open) {
            return Err(BoundaryError::MissingSpillCleanup { site: self.boundary.site });
        }
        Ok(SuccessReceipt {
            site: self.boundary.site,
            lease_recovered: self.lease_recovered,
            spill_cleaned: self.spill.cleanup_recorded(),
            call_bytes: assessed.call_bytes,
            elapsed: assessed.elapsed,
            provider: self.provider,
        })
    }

    fn assess(&self) -> Result<Assessed, BoundaryError> {
        if self.provider == Some(ProviderDisposition::Refused) {
            return Err(BoundaryError::UninterruptibleProvider);
        }
        let Some(call_bytes) = self.call_bytes else {
            return Err(BoundaryError::CallSizeNotRecorded { site: self.boundary.site });
        };
        if call_bytes > self.boundary.max_call_bytes {
            return Err(BoundaryError::CallTooLarge {
                site: self.boundary.site,
                bytes: call_bytes,
                max_call_bytes: self.boundary.max_call_bytes,
            });
        }
        let Some(elapsed) = self.elapsed else {
            return Err(BoundaryError::ElapsedNotRecorded { site: self.boundary.site });
        };
        if elapsed > self.boundary.latency_bound {
            return Err(BoundaryError::LatencyExceeded {
                site: self.boundary.site,
                elapsed,
                bound: self.boundary.latency_bound,
            });
        }
        Ok(Assessed { call_bytes, elapsed })
    }
}

const fn published_claim_of(partial: Option<PartialResult>) -> bool {
    match partial {
        Some(partial) => partial.published_claim,
        None => false,
    }
}

struct Assessed {
    call_bytes: u64,
    elapsed: Duration,
}

/// Status [`BoundarySession::stop`] reports after recovery and cleanup.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BoundaryReport {
    /// Lease recovery and spill cleanup were both recorded.
    Stopped,
}

impl BoundaryReport {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
        }
    }
}

impl fmt::Display for BoundaryReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Receipt for a boundary that reported [`BoundaryReport::Stopped`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StopReceipt {
    report: BoundaryReport,
    site: CancelSite,
    lease_recovered: bool,
    spill_cleaned: bool,
    partial: Option<PartialResult>,
    published_claim: bool,
    call_bytes: u64,
    elapsed: Duration,
    provider: Option<ProviderDisposition>,
}

impl StopReceipt {
    /// [`BoundaryReport::Stopped`].
    #[must_use]
    pub const fn report(self) -> BoundaryReport {
        self.report
    }

    /// Site that stopped.
    #[must_use]
    pub const fn site(self) -> CancelSite {
        self.site
    }

    /// Whether [`BoundarySession::recover_lease`] released a live reservation.
    #[must_use]
    pub const fn lease_recovered(self) -> bool {
        self.lease_recovered
    }

    /// Whether spill cleanup was recorded.
    #[must_use]
    pub const fn spill_cleaned(self) -> bool {
        self.spill_cleaned
    }

    /// Retained partial, when the boundary kept one.
    #[must_use]
    pub const fn partial(self) -> Option<PartialResult> {
        self.partial
    }

    /// Whether a partial result was published as a claim.
    ///
    /// A retained partial leaves this clear.
    #[must_use]
    pub const fn published_claim(self) -> bool {
        self.published_claim
    }

    /// Recorded call size in bytes.
    #[must_use]
    pub const fn call_bytes(self) -> u64 {
        self.call_bytes
    }

    /// Recorded elapsed time.
    #[must_use]
    pub const fn elapsed(self) -> Duration {
        self.elapsed
    }

    /// Provider admission, when the session recorded one.
    #[must_use]
    pub const fn provider(self) -> Option<ProviderDisposition> {
        self.provider
    }
}

/// Receipt for a bounded call that finished inside the boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SuccessReceipt {
    site: CancelSite,
    lease_recovered: bool,
    spill_cleaned: bool,
    call_bytes: u64,
    elapsed: Duration,
    provider: Option<ProviderDisposition>,
}

impl SuccessReceipt {
    /// Site that finished.
    #[must_use]
    pub const fn site(self) -> CancelSite {
        self.site
    }

    /// Whether [`BoundarySession::recover_lease`] released a live reservation.
    #[must_use]
    pub const fn lease_recovered(self) -> bool {
        self.lease_recovered
    }

    /// Whether spill cleanup was recorded.
    #[must_use]
    pub const fn spill_cleaned(self) -> bool {
        self.spill_cleaned
    }

    /// Recorded call size in bytes.
    #[must_use]
    pub const fn call_bytes(self) -> u64 {
        self.call_bytes
    }

    /// Recorded elapsed time.
    #[must_use]
    pub const fn elapsed(self) -> Duration {
        self.elapsed
    }

    /// Provider admission, when the session recorded one.
    #[must_use]
    pub const fn provider(self) -> Option<ProviderDisposition> {
        self.provider
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::device_reservation::{DeviceReservationLease, DeviceReservationPool};

    use super::{
        Boundary, BoundaryError, BoundaryReport, BoundarySession, CancelSite, ProviderDisposition,
        UninterruptibleAdmission,
    };

    fn live_lease() -> DeviceReservationLease {
        DeviceReservationPool::fixed(64, 8).try_grant(16, 4, false).expect("pool funds the lease")
    }

    fn open_bounded(site: CancelSite, retain_partial: bool) -> BoundarySession {
        let boundary =
            Boundary::new(site, 64, Duration::from_millis(10), retain_partial, false).unwrap();
        boundary.open()
    }

    fn record_within_bound(session: &mut BoundarySession) {
        session.record_call_bytes(64).unwrap();
        session.record_elapsed(Duration::from_millis(10)).unwrap();
    }

    #[test]
    fn each_site_stops_with_lease_recovered_and_spill_cleaned() {
        for site in CancelSite::ALL {
            let mut session = open_bounded(site, false);
            record_within_bound(&mut session);
            session.open_spill().unwrap();
            assert!(session.recover_lease(live_lease()), "{site}");
            session.record_spill_cleanup();
            let receipt = session.stop().unwrap();
            assert_eq!(receipt.report(), BoundaryReport::Stopped, "{site}");
            assert_eq!(receipt.site(), site, "{site}");
            assert!(receipt.lease_recovered(), "{site}");
            assert!(receipt.spill_cleaned(), "{site}");
            assert!(receipt.partial().is_none(), "{site}");
            assert!(!receipt.published_claim(), "{site}");
        }
    }

    #[test]
    fn partial_result_is_not_a_published_claim() {
        let refused = Boundary::new(CancelSite::Resample, 32, Duration::from_millis(5), true, true);
        assert_eq!(
            refused.unwrap_err(),
            BoundaryError::PartialClaimRefused { site: CancelSite::Resample },
        );

        let boundary =
            Boundary::new(CancelSite::Resample, 32, Duration::from_millis(5), true, false).unwrap();
        assert!(!boundary.publish_partial_claim());
        let mut session = boundary.open();
        session.record_call_bytes(16).unwrap();
        session.record_elapsed(Duration::from_millis(1)).unwrap();
        let partial = session.retain_partial_result(16).unwrap();
        assert_eq!(partial.site(), CancelSite::Resample);
        assert_eq!(partial.bytes(), 16);
        assert!(!partial.is_published_claim());
        assert_eq!(
            session.publish_claim().unwrap_err(),
            BoundaryError::PartialClaimRefused { site: CancelSite::Resample },
        );
        assert!(session.recover_lease(live_lease()));
        session.record_spill_cleanup();
        let receipt = session.stop().unwrap();
        assert_eq!(receipt.partial(), Some(partial));
        assert!(!receipt.published_claim());

        let mut closed = open_bounded(CancelSite::Resample, false);
        assert_eq!(
            closed.retain_partial_result(1).unwrap_err(),
            BoundaryError::PartialRetentionRefused { site: CancelSite::Resample },
        );
    }

    #[test]
    fn non_interruptible_provider_is_refused_rather_than_preempted() {
        assert_eq!(
            ProviderDisposition::admit(false, UninterruptibleAdmission::Refuse),
            ProviderDisposition::Refused,
        );
        assert_eq!(
            ProviderDisposition::admit(false, UninterruptibleAdmission::Isolate),
            ProviderDisposition::Isolated,
        );
        assert_eq!(
            ProviderDisposition::admit(true, UninterruptibleAdmission::Refuse),
            ProviderDisposition::Interruptible,
        );
        assert_eq!(ProviderDisposition::Refused.as_str(), "refused");
        assert_eq!(ProviderDisposition::Isolated.as_str(), "isolated");
        assert_eq!(ProviderDisposition::Interruptible.as_str(), "interruptible");

        let mut refused = open_bounded(CancelSite::Provider, false);
        let err = refused.enter_provider(false, UninterruptibleAdmission::Refuse).unwrap_err();
        assert_eq!(err, BoundaryError::UninterruptibleProvider);
        assert_eq!(refused.provider(), Some(ProviderDisposition::Refused));
        assert_eq!(refused.stop().unwrap_err(), BoundaryError::UninterruptibleProvider);

        let mut isolated = open_bounded(CancelSite::Provider, false);
        let admission = isolated.enter_provider(false, UninterruptibleAdmission::Isolate).unwrap();
        assert_eq!(admission, ProviderDisposition::Isolated);
        record_within_bound(&mut isolated);
        assert!(isolated.recover_lease(live_lease()));
        isolated.record_spill_cleanup();
        let receipt = isolated.stop().unwrap();
        assert_eq!(receipt.report(), BoundaryReport::Stopped);
        assert_eq!(receipt.provider(), Some(ProviderDisposition::Isolated));
        assert_eq!(receipt.provider().unwrap().as_str(), "isolated");
    }

    #[test]
    fn elapsed_time_past_the_bound_is_refused() {
        let bound = Duration::from_millis(5);
        let boundary = Boundary::new(CancelSite::Solver, 100, bound, false, false).unwrap();
        assert_eq!(boundary.latency_bound(), bound);
        let mut work = boundary.open();
        work.record_call_bytes(1).unwrap();
        let over = Duration::from_millis(6);
        let err = work.record_elapsed(over).unwrap_err();
        assert_eq!(
            err,
            BoundaryError::LatencyExceeded { site: CancelSite::Solver, elapsed: over, bound },
        );
        assert_eq!(work.recorded_elapsed(), Some(over));
        assert_eq!(work.stop().unwrap_err(), err);

        let mut at_bound =
            Boundary::new(CancelSite::Solver, 100, bound, false, false).unwrap().open();
        at_bound.record_call_bytes(0).unwrap();
        at_bound.record_elapsed(bound).unwrap();
        assert!(at_bound.recover_lease(live_lease()));
        at_bound.record_spill_cleanup();
        assert_eq!(at_bound.stop().unwrap().elapsed(), bound);
    }

    #[test]
    fn missing_spill_cleanup_is_refused() {
        let mut unrecovered = open_bounded(CancelSite::Search, false);
        record_within_bound(&mut unrecovered);
        unrecovered.record_spill_cleanup();
        let spent = live_lease();
        let _returned = spent.release_on_cancel(true);
        assert!(!unrecovered.recover_lease(spent));
        assert_eq!(unrecovered.stop().unwrap_err(), BoundaryError::LeaseNotRecovered);

        let mut opened = open_bounded(CancelSite::Chain, false);
        record_within_bound(&mut opened);
        assert!(opened.recover_lease(live_lease()));
        opened.open_spill().unwrap();
        assert_eq!(
            opened.stop().unwrap_err(),
            BoundaryError::MissingSpillCleanup { site: CancelSite::Chain },
        );

        let mut succeed_open = open_bounded(CancelSite::Chain, false);
        record_within_bound(&mut succeed_open);
        succeed_open.open_spill().unwrap();
        assert_eq!(
            succeed_open.succeed().unwrap_err(),
            BoundaryError::MissingSpillCleanup { site: CancelSite::Chain },
        );

        let mut idle = open_bounded(CancelSite::Device, false);
        record_within_bound(&mut idle);
        assert!(idle.recover_lease(live_lease()));
        assert_eq!(
            idle.stop().unwrap_err(),
            BoundaryError::MissingSpillCleanup { site: CancelSite::Device },
        );

        let mut cleaned = open_bounded(CancelSite::Chain, false);
        record_within_bound(&mut cleaned);
        cleaned.open_spill().unwrap();
        cleaned.record_spill_cleanup();
        let success = cleaned.succeed().unwrap();
        assert!(!success.lease_recovered());
        assert!(success.spill_cleaned());
    }

    #[test]
    fn call_larger_than_the_bound_is_refused() {
        let mut work = open_bounded(CancelSite::Scan, false);
        let err = work.record_call_bytes(65).unwrap_err();
        assert_eq!(
            err,
            BoundaryError::CallTooLarge { site: CancelSite::Scan, bytes: 65, max_call_bytes: 64 },
        );
        assert_eq!(work.recorded_call_bytes(), Some(65));
        work.record_elapsed(Duration::from_millis(1)).unwrap();
        assert_eq!(work.stop().unwrap_err(), err);
    }
}
