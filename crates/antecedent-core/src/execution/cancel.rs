//! Execution context: cancellation tokens and progress reporting.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::admission::{SharedAdmission, lock_mutex};
#[cfg(doc)]
use super::context::ExecutionContext;
#[cfg(any(test, feature = "test-support"))]
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::{AtomicBool, Ordering};
use std::fmt;
use std::sync::{Arc, Mutex, Weak};

/// Shared wait state. A production token is this and nothing else.
pub(super) struct CancelWait {
    pub(super) cancelled: Arc<AtomicBool>,
    pub(super) waiters: Mutex<Vec<Weak<SharedAdmission>>>,
}

impl CancelWait {
    pub(super) fn register_pool(&self, shared: &Arc<SharedAdmission>) {
        let mut waiters = lock_mutex(&self.waiters);
        waiters.retain(|waiter| waiter.strong_count() > 0);
        let incoming = Arc::downgrade(shared);
        if !waiters.iter().any(|waiter| Weak::ptr_eq(waiter, &incoming)) {
            waiters.push(incoming);
        }
    }

    // Registration and publication are serialized: a late registration cannot observe
    // the old flag after it has missed the notification snapshot.
    fn publish_and_snapshot(&self) -> Vec<Weak<SharedAdmission>> {
        let mut waiters = lock_mutex(&self.waiters);
        self.cancelled.store(true, Ordering::SeqCst);
        waiters.retain(|waiter| waiter.strong_count() > 0);
        waiters.clone()
    }

    /// Store the cancelled flag and notify each pool while holding that pool's state lock.
    ///
    /// The flag is stored again under the pool lock, before `notify_all`. A waiter holds
    /// that same lock while it reads the flag and only then waits, so it cannot pass the
    /// read, miss the notify, and sleep until its bound.
    pub(super) fn wake(&self) {
        let waiters = self.publish_and_snapshot();
        for waiter in waiters {
            let Some(shared) = waiter.upgrade() else {
                continue;
            };
            let _state = lock_mutex(&shared.state);
            self.cancelled.store(true, Ordering::SeqCst);
            shared.released.notify_all();
        }
    }
}

/// Test-only observation counter. It is not a field of [`CancelWait`], and it is compiled
/// only for this crate's tests or under the `test-support` feature.
#[cfg(any(test, feature = "test-support"))]
pub(super) struct AfterChecks {
    pub(super) wait: CancelWait,
    pub(super) left: AtomicUsize,
}

pub(super) enum CancelFlag {
    Production(CancelWait),
    #[cfg(any(test, feature = "test-support"))]
    AfterChecks(AfterChecks),
}

impl Default for CancelFlag {
    fn default() -> Self {
        Self::Production(CancelWait {
            cancelled: Arc::new(AtomicBool::new(false)),
            waiters: Mutex::new(Vec::new()),
        })
    }
}

impl CancelFlag {
    pub(super) fn wait(&self) -> &CancelWait {
        match self {
            Self::Production(wait) => wait,
            #[cfg(any(test, feature = "test-support"))]
            Self::AfterChecks(checks) => &checks.wait,
        }
    }

    pub(super) fn register_pool(&self, shared: &Arc<SharedAdmission>) {
        self.wait().register_pool(shared);
    }

    pub(super) fn wake(&self) {
        self.wait().wake();
    }

    pub(super) fn is_requested(&self) -> bool {
        self.wait().cancelled.load(Ordering::SeqCst)
    }

    /// Production reads the flag. [`CancellationToken::cancel_after_checks`] spends one
    /// observation first. That counter lives only on [`Self::AfterChecks`].
    pub(super) fn is_cancelled(&self) -> bool {
        #[cfg(any(test, feature = "test-support"))]
        {
            if let Self::AfterChecks(checks) = self {
                let expired = checks
                    .left
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                    .is_err();
                if expired {
                    checks.wait.wake();
                }
            }
        }
        self.is_requested()
    }
}

/// Cooperative cancellation token.
///
/// [`Self::cancel`] wakes admission waits that registered on this token. The admission pool
/// lives on [`ExecutionContext`], so installing a pool does not retarget other clones of this
/// token.
#[derive(Clone, Default)]
pub struct CancellationToken {
    pub(super) flag: Arc<CancelFlag>,
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancellationToken")
            .field("cancelled", &self.flag.is_requested())
            .finish()
    }
}

impl CancellationToken {
    /// Create a fresh token. It does not own an admission pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A token that cancels itself on its `checks + 1`-th observation, so a
    /// test can stop a search deterministically in the middle (every budget
    /// charge observes the token once). The counter is a separate flag body.
    /// [`Self::new`] does not carry it.
    // Available to other crates' tests through the `test-support` feature only.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    #[must_use]
    pub fn cancel_after_checks(checks: usize) -> Self {
        Self {
            flag: Arc::new(CancelFlag::AfterChecks(AfterChecks {
                wait: CancelWait {
                    cancelled: Arc::new(AtomicBool::new(false)),
                    waiters: Mutex::new(Vec::new()),
                },
                left: AtomicUsize::new(checks),
            })),
        }
    }

    /// A non-owning handle that does not keep this token alive.
    #[must_use]
    pub fn downgrade(&self) -> WeakCancellationToken {
        WeakCancellationToken { flag: Arc::downgrade(&self.flag) }
    }

    /// Request cancellation and wake admission waits bound to this token.
    ///
    /// The cancelled flag is stored while the pool state lock is held, and the
    /// condvar is notified before that lock is released.
    pub fn cancel(&self) {
        self.flag.wake();
    }

    /// Whether cancellation was requested, read without spending one of the checks a
    /// [`Self::cancel_after_checks`] token counts.
    ///
    /// A hold that releases its lease on cancellation observes the request through this, so the
    /// number of checks a cell makes stays the cell's own planned count.
    #[must_use]
    pub fn is_requested(&self) -> bool {
        self.flag.is_requested()
    }

    /// Whether cancellation was requested.
    ///
    /// A production token only reads the flag. A token from [`Self::cancel_after_checks`] spends
    /// one observation first.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.is_cancelled()
    }

    pub(super) fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.flag.wait().cancelled)
    }
}

/// A non-owning handle to a [`CancellationToken`].
///
/// It keeps the token's cancellation flag reachable without keeping the token
/// alive: once every owner of the token is gone, [`Self::upgrade`] returns
/// `None`. A queue of contexts waiting for a run can hold these without
/// leaking the contexts that were built and dropped.
#[derive(Clone, Debug)]
pub struct WeakCancellationToken {
    pub(super) flag: Weak<CancelFlag>,
}

impl WeakCancellationToken {
    /// The token, if any owner of it is still alive.
    #[must_use]
    pub fn upgrade(&self) -> Option<CancellationToken> {
        Some(CancellationToken { flag: self.flag.upgrade()? })
    }

    /// Whether an owner of the token is still alive.
    #[must_use]
    pub fn is_alive(&self) -> bool {
        self.flag.strong_count() > 0
    }
}

/// Optional progress reporting sink.
pub trait ProgressSink: Send + Sync {
    /// Report progress in `[0.0, 1.0]` with an optional stage label.
    fn report(&self, fraction: f64, stage: &str);
}

#[cfg(test)]
mod cancellation_race_tests {
    use super::*;
    use crate::execution::{AdmissionError, AdmissionPool, ExecutionContext};
    use std::time::Duration;

    #[test]
    fn late_registration_after_snapshot_observes_cancellation() {
        let token = CancellationToken::new();
        // This is the race point: a second pool registers after the first wake snapshot.
        let snapshot = token.flag.wait().publish_and_snapshot();
        assert!(snapshot.is_empty());
        let pool = AdmissionPool::fixed(1, 1);
        let owner = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let _hold = owner.acquire_admission(1, 1).unwrap();
        let mut waiter = ExecutionContext::for_tests(2).with_admission(pool);
        waiter.cancellation = token;
        assert!(matches!(
            waiter.acquire_admission_bounded(1, 1, Duration::from_secs(1)),
            Err(AdmissionError::Cancelled)
        ));
    }
}
