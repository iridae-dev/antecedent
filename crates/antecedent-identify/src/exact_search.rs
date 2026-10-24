//! Storage quotes and cancellation for exact adjustment-set search.
//!
//! The work of an exact subset search is the number of separation tests it charges against
//! the caller's examination budget; there is no worst-case count above which it refuses to
//! start. What an exact search allocates before it tests anything (the mutilated graph, the
//! enumeration cursor and the retained result sets) is quoted from the candidate pool and
//! leased from the caller's [`ExecutionContext`] before it is allocated. A refused lease
//! hands the decision back to the certified ancestral set, so a pool the context cannot fund
//! is a resource outcome, never a scientific one. The context's cancellation token is
//! observed at every test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;

use antecedent_core::execution::AdmissionLease;
use antecedent_core::{ExecutionContext, SearchBudget, SearchLimits, SearchStop};
use antecedent_graph::DenseNodeId;

use crate::error::IdentificationError;

/// The caller's context, or a serial production context when none was supplied.
pub(crate) fn context_or_default(context: Option<&ExecutionContext>) -> Cow<'_, ExecutionContext> {
    context.map_or_else(|| Cow::Owned(ExecutionContext::production(0, 1)), Cow::Borrowed)
}

/// Bytes of the graph copies one exact search holds for `nodes` nodes: the mutilated graph's
/// parent and child bit rows.
pub(crate) fn graph_bytes(nodes: usize) -> u64 {
    let nodes = nodes as u64;
    let row = nodes.div_ceil(64).saturating_mul(8).saturating_add(24);
    nodes.saturating_mul(row).saturating_mul(2)
}

/// Bytes the subset enumeration over `candidates` holds: the combination cursor, the current
/// subset, and the retained result sets (at most `max_results`, and never more than the
/// `2^candidates` subsets that exist).
pub(crate) fn enumeration_bytes(candidates: usize, max_results: usize) -> u64 {
    let m = candidates as u64;
    let id = std::mem::size_of::<DenseNodeId>() as u64;
    let word = std::mem::size_of::<usize>() as u64;
    let cursor = m.saturating_mul(word.saturating_add(id));
    let sets = u32::try_from(candidates)
        .ok()
        .and_then(|bits| 1u64.checked_shl(bits))
        .map_or(max_results as u64, |subsets| subsets.min(max_results as u64));
    let per_set =
        m.saturating_mul(id).saturating_add(std::mem::size_of::<Vec<DenseNodeId>>() as u64);
    cursor.saturating_add(sets.saturating_mul(per_set))
}

/// One exact search's hold on the caller's context: its leases, kept until the search ends,
/// and its cancellation observer.
pub(crate) struct ExactSearch<'c> {
    budget: SearchBudget<'c>,
    held: Vec<AdmissionLease>,
}

impl<'c> ExactSearch<'c> {
    /// Open a search on `ctx`.
    ///
    /// # Errors
    /// [`IdentificationError::Cancelled`] when the context is already cancelled.
    pub(crate) fn open(ctx: &'c ExecutionContext) -> Result<Self, IdentificationError> {
        // The operation account is the caller's examination budget, kept by the search itself;
        // this budget supplies the memory cap, the leases and the cancellation observer.
        SearchBudget::new(SearchLimits { operations: usize::MAX, depth: 1 }, ctx)
            .map(|budget| Self { budget, held: Vec::new() })
            .map_err(|_| IdentificationError::Cancelled)
    }

    /// Observe cancellation before one more test.
    ///
    /// # Errors
    /// [`IdentificationError::Cancelled`].
    pub(crate) fn observe(&mut self) -> Result<(), IdentificationError> {
        self.budget.charge(1, 0).map_err(|_| IdentificationError::Cancelled)
    }

    /// Lease `bytes` and keep the lease until the search ends. `Ok(false)` is a refusal for
    /// memory: nothing was leased and the caller allocates nothing.
    ///
    /// # Errors
    /// [`IdentificationError::Cancelled`].
    pub(crate) fn hold(&mut self, bytes: u64) -> Result<bool, IdentificationError> {
        match self.budget.lease_storage(bytes) {
            Ok(lease) => {
                self.held.push(lease);
                Ok(true)
            }
            Err(SearchStop::Cancelled) => Err(IdentificationError::Cancelled),
            Err(_) => Ok(false),
        }
    }

    /// Whether an exact enumeration over `candidates` is funded: within the caller's declared
    /// `ceiling` and with its working storage leased.
    ///
    /// # Errors
    /// [`IdentificationError::Cancelled`].
    pub(crate) fn fund_enumeration(
        &mut self,
        candidates: usize,
        ceiling: usize,
        max_results: usize,
    ) -> Result<bool, IdentificationError> {
        if candidates > ceiling {
            return Ok(false);
        }
        self.hold(enumeration_bytes(candidates, max_results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;

    #[test]
    fn enumeration_storage_grows_with_the_pool_and_is_bounded_by_the_subsets_that_exist() {
        assert!(enumeration_bytes(40, 64) > enumeration_bytes(10, 64));
        // Three candidates have eight subsets, so retaining 64 sets is never quoted.
        assert!(enumeration_bytes(3, 64) < enumeration_bytes(3, 8) + 1);
        assert_eq!(enumeration_bytes(3, 8), enumeration_bytes(3, 1_000_000));
        assert!(graph_bytes(1_000) > graph_bytes(100));
    }

    #[test]
    fn a_pool_that_cannot_fund_the_enumeration_is_refused_without_a_lease() {
        let pool = AdmissionPool::fixed(1, 1_000);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let mut search = ExactSearch::open(&ctx).unwrap();
        assert!(!search.fund_enumeration(60, usize::MAX, 64).unwrap());
        assert_eq!(pool.available().bytes(), 1_000);
        assert!(search.fund_enumeration(2, usize::MAX, 4).unwrap());
        assert!(pool.available().bytes() < 1_000);
        drop(search);
        assert_eq!(pool.available().bytes(), 1_000);
    }

    #[test]
    fn a_declared_ceiling_refuses_before_any_lease() {
        let ctx = ExecutionContext::for_tests(1);
        let mut search = ExactSearch::open(&ctx).unwrap();
        assert!(!search.fund_enumeration(5, 4, 64).unwrap());
    }

    #[test]
    fn cancellation_is_observed_by_the_search() {
        let ctx = ExecutionContext::for_tests(1);
        let mut search = ExactSearch::open(&ctx).unwrap();
        search.observe().unwrap();
        ctx.cancellation.cancel();
        assert!(matches!(search.observe(), Err(IdentificationError::Cancelled)));
        assert!(matches!(search.hold(1), Err(IdentificationError::Cancelled)));
    }
}
