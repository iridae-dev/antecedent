//! Observed work units recorded at the boundaries the algorithms actually cross.
//!
//! * CI tests are counted where a test is handed to the test implementation, not where a
//!   caller asks for one: a retained statistic answers a request without any work, and a
//!   speculative batch that the search later discards is still work that was done.
//! * Orientation work is counted inside the shared fixed-point driver every orientation
//!   scheduler runs through, per round and per applied orientation.
//! * Search passes are the per-depth iteration records the search loops themselves emit.
//!
//! A wrapper call is never a work unit: `algorithm_executions` is reported separately and
//! named for what it is.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;

/// Orientation work observed on the calling thread.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OrientationWork {
    /// Fixed-point drives that ran to completion.
    pub fixed_point_runs: u64,
    /// Rule rounds executed across all drives (including the confirming full-scan round).
    pub rounds: u64,
    /// Edge orientations applied.
    pub orientations_applied: u64,
    /// Orientation conflicts reported by the rules.
    pub conflicts: u64,
}

thread_local! {
    static OBSERVER: RefCell<Option<OrientationWork>> = const { RefCell::new(None) };
}

/// Record one orientation round (called by the fixed-point driver).
pub(crate) fn record_round(edges_changed: u32, conflicts: u32) {
    OBSERVER.with(|cell| {
        if let Some(work) = cell.borrow_mut().as_mut() {
            work.rounds += 1;
            work.orientations_applied += u64::from(edges_changed);
            work.conflicts += u64::from(conflicts);
        }
    });
}

/// Record that one fixed-point drive reached its fixed point.
pub(crate) fn record_fixed_point() {
    OBSERVER.with(|cell| {
        if let Some(work) = cell.borrow_mut().as_mut() {
            work.fixed_point_runs += 1;
        }
    });
}

/// Run `f` and return its value with the orientation work done on this thread meanwhile.
///
/// Nested observation restores the outer observer, which then does not see the inner work.
/// Orientation is serial in every algorithm of this crate, so the calling thread is where it
/// runs.
pub fn observe_orientation<T>(f: impl FnOnce() -> T) -> (T, OrientationWork) {
    let previous = OBSERVER.with(|cell| cell.borrow_mut().replace(OrientationWork::default()));
    let value = f();
    let work = OBSERVER
        .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), previous))
        .unwrap_or_default();
    (value, work)
}

/// Work units of one selective run, named by what happened.
///
/// Every field counts completed or attempted work observed at an algorithm boundary; cache
/// hits are excluded from the executed counts and reported on their own line.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkUnits {
    /// CI tests handed to the test implementation.
    pub ci_tests_attempted: u64,
    /// CI tests the test implementation returned a result for (attempted minus failures).
    pub ci_tests_completed: u64,
    /// CI requests answered from a retained statistic at an identical input coordinate.
    pub ci_tests_reused: u64,
    /// CI tests the algorithm consumed (its own count; speculative batch work that was
    /// discarded is in `completed` but not here).
    pub ci_tests_consumed: u64,
    /// Search passes (one per conditioning-set size or parent-selection level).
    pub search_passes: u64,
    /// Orientation fixed-point drives that completed.
    pub orientation_runs: u64,
    /// Orientation rule rounds.
    pub orientation_rounds: u64,
    /// Edge orientations applied.
    pub orientations_applied: u64,
    /// Score evaluations. Always zero: the delivered routes are constraint based and
    /// score-based search is a closed route (see the module documentation of
    /// [`crate::selective`]).
    pub score_evaluations: u64,
    /// Times the discovery algorithm was entered. A wrapper-level count, not a work unit.
    pub algorithm_executions: u64,
}

impl WorkUnits {
    /// Whether no statistic was computed and no search ran: everything was served from
    /// retained identities.
    #[must_use]
    pub const fn is_pure_reuse(&self) -> bool {
        self.ci_tests_attempted == 0 && self.algorithm_executions == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_counts_only_inside_its_scope_and_nests() {
        record_round(3, 1);
        let ((), outer) = observe_orientation(|| {
            record_round(2, 0);
            let ((), inner) = observe_orientation(|| {
                record_round(5, 2);
                record_fixed_point();
            });
            assert_eq!(inner.orientations_applied, 5);
            assert_eq!(inner.conflicts, 2);
            assert_eq!(inner.fixed_point_runs, 1);
            record_round(1, 0);
        });
        assert_eq!(outer.rounds, 2);
        assert_eq!(outer.orientations_applied, 3);
        assert_eq!(outer.fixed_point_runs, 0);
        let ((), none) = observe_orientation(|| ());
        assert_eq!(none, OrientationWork::default());
    }
}
