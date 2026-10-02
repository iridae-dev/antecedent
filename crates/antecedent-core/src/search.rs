//! The bounded-search contract every new 2.2 search runs under.
//!
//! A search cannot be built without operation and depth limits, and every
//! charge also observes the [`ExecutionContext`]'s cancellation token and a
//! memory cap that is never absent: the effective cap is the smaller of the
//! context's hard limit (when set) and the budget's own cap, which is
//! [`DEFAULT_SEARCH_MEMORY_BYTES`] unless [`SearchBudget::with_memory`] sets
//! another. None of the four bounds is optional. A stopped search
//! returns a [`SearchReceipt`] recording the limits in force, what was
//! consumed, and the explored and unevaluated regions; running out of budget is
//! a resource outcome, never an impossibility or non-identification claim.
//! Wire encoding of a receipt belongs to the artifact that carries it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::execution::ExecutionContext;

/// Default memory cap of a search budget: 512 MiB of estimated live state.
pub const DEFAULT_SEARCH_MEMORY_BYTES: u64 = 512 * 1024 * 1024;

/// Operation and depth limits. Cancellation and the hard memory limit come from
/// the [`ExecutionContext`] the search is charged against; the memory cap
/// itself defaults to [`DEFAULT_SEARCH_MEMORY_BYTES`] (see
/// [`SearchBudget::with_memory`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SearchLimits {
    /// Maximum operations (rule applications, states or subproblems) charged.
    pub operations: usize,
    /// Maximum search or recursion depth.
    pub depth: usize,
}

/// Which bound stopped a search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SearchStop {
    /// The operation limit was exhausted.
    Operations,
    /// The depth limit was reached.
    Depth,
    /// The context's hard memory limit would be exceeded.
    Memory,
    /// Cooperative cancellation was observed.
    Cancelled,
}

impl SearchStop {
    /// Stable detail code, `search.<stop>`.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Operations => "search.operations",
            Self::Depth => "search.depth",
            Self::Memory => "search.memory",
            Self::Cancelled => "search.cancelled",
        }
    }
}

/// What a stopped search consumed against the limits it ran under.
///
/// `operations_consumed` and `depth_reached` are `None` when the search stopped
/// before it was entered (a zero limit, or cancellation observed at
/// construction); no accounting is fabricated. `depth_reached` is the deepest
/// level any charge was attempted at, including the one a depth stop refused,
/// so a depth stop reports the level that exceeded `depth_limit`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchReceipt {
    /// Which bound stopped the search.
    pub stop: SearchStop,
    /// Operation limit in force.
    pub operations_limit: usize,
    /// Depth limit in force.
    pub depth_limit: usize,
    /// Effective memory cap in force: always `Some`, the smaller of the
    /// context's hard limit and the budget's cap.
    pub memory_limit_bytes: Option<u64>,
    /// Operations charged when the search stopped, if measured.
    pub operations_consumed: Option<usize>,
    /// Deepest level reached when the search stopped, if measured.
    pub depth_reached: Option<usize>,
    /// Regions (rules, states, scenarios) explored before the stop, in order.
    pub explored: Vec<String>,
    /// Regions known to exist but left unevaluated by the stop.
    pub unevaluated: Vec<String>,
}

impl SearchReceipt {
    /// One-line account of the stop for error text a binding surfaces: the
    /// stop code, the operations charged against the limit, and every explored
    /// and unevaluated region, so a stopped search never reads as a list of
    /// its successes alone.
    #[must_use]
    pub fn summary(&self) -> String {
        let consumed = self.operations_consumed.map_or_else(
            || "not entered".to_owned(),
            |n| format!("{n} of {}", self.operations_limit),
        );
        format!(
            "receipt: stop {}; operations {consumed}; explored [{}]; unevaluated [{}]",
            self.stop.code(),
            self.explored.join(", "),
            self.unevaluated.join(", ")
        )
    }
}

/// A live budget: the only way a 2.2 search charges work.
#[derive(Debug)]
pub struct SearchBudget<'a> {
    limits: SearchLimits,
    ctx: &'a ExecutionContext,
    memory_limit_bytes: u64,
    operations: usize,
    depth_reached: usize,
    entered: bool,
}

impl<'a> SearchBudget<'a> {
    /// Start a budget under the default memory cap
    /// ([`DEFAULT_SEARCH_MEMORY_BYTES`], further limited by the context's hard
    /// limit). A zero limit or an already cancelled context stops before the
    /// search is entered.
    ///
    /// # Errors
    ///
    /// Returns the receipt of that pre-search stop.
    pub fn new(limits: SearchLimits, ctx: &'a ExecutionContext) -> Result<Self, SearchReceipt> {
        Self::with_memory(limits, DEFAULT_SEARCH_MEMORY_BYTES, ctx)
    }

    /// [`Self::new`] under an explicit memory cap. The effective cap is the
    /// smaller of `memory_limit_bytes` and the context's hard limit.
    ///
    /// # Errors
    ///
    /// Returns the receipt of a pre-search stop.
    pub fn with_memory(
        limits: SearchLimits,
        memory_limit_bytes: u64,
        ctx: &'a ExecutionContext,
    ) -> Result<Self, SearchReceipt> {
        let memory_limit_bytes =
            ctx.memory.hard_limit_bytes.map_or(memory_limit_bytes, |h| h.min(memory_limit_bytes));
        let budget = Self {
            limits,
            ctx,
            memory_limit_bytes,
            operations: 0,
            depth_reached: 0,
            entered: false,
        };
        let stop = if ctx.cancellation.is_cancelled() {
            Some(SearchStop::Cancelled)
        } else if limits.operations == 0 {
            Some(SearchStop::Operations)
        } else if limits.depth == 0 {
            Some(SearchStop::Depth)
        } else {
            None
        };
        match stop {
            Some(stop) => Err(budget.receipt(stop, Vec::new(), Vec::new())),
            None => Ok(budget),
        }
    }

    /// Charge one operation at `depth` whose live state needs `bytes`.
    /// Cancellation is checked first, then memory, operations and depth.
    ///
    /// # Errors
    ///
    /// Returns the bound that stopped the search; build its receipt with
    /// [`Self::receipt`].
    pub fn charge(&mut self, depth: usize, bytes: u64) -> Result<(), SearchStop> {
        self.entered = true;
        self.depth_reached = self.depth_reached.max(depth);
        if self.ctx.cancellation.is_cancelled() {
            return Err(SearchStop::Cancelled);
        }
        if bytes > self.memory_limit_bytes {
            return Err(SearchStop::Memory);
        }
        if self.operations >= self.limits.operations {
            return Err(SearchStop::Operations);
        }
        if depth > self.limits.depth {
            return Err(SearchStop::Depth);
        }
        self.operations += 1;
        Ok(())
    }

    /// Limits in force.
    #[must_use]
    pub const fn limits(&self) -> SearchLimits {
        self.limits
    }

    /// Operations charged so far.
    #[must_use]
    pub const fn operations(&self) -> usize {
        self.operations
    }

    /// Deepest level any charge was attempted at so far (a refused charge counts).
    #[must_use]
    pub const fn depth_reached(&self) -> usize {
        self.depth_reached
    }

    /// Effective memory cap in force.
    #[must_use]
    pub const fn memory_limit_bytes(&self) -> u64 {
        self.memory_limit_bytes
    }

    /// Record why and where the search stopped.
    #[must_use]
    pub fn receipt(
        &self,
        stop: SearchStop,
        explored: Vec<String>,
        unevaluated: Vec<String>,
    ) -> SearchReceipt {
        SearchReceipt {
            stop,
            operations_limit: self.limits.operations,
            depth_limit: self.limits.depth,
            memory_limit_bytes: Some(self.memory_limit_bytes),
            operations_consumed: self.entered.then_some(self.operations),
            depth_reached: self.entered.then_some(self.depth_reached),
            explored,
            unevaluated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::MemoryBudget;

    const LIMITS: SearchLimits = SearchLimits { operations: 3, depth: 2 };

    #[test]
    fn a_tripping_token_cancels_a_search_in_the_middle() {
        let mut ctx = ExecutionContext::for_tests(0);
        ctx.cancellation = crate::CancellationToken::cancel_after_checks(2);
        let limits = SearchLimits { operations: 10, depth: 2 };
        // One observation at entry, one at the first charge; the second charge stops.
        let mut budget = SearchBudget::new(limits, &ctx).unwrap();
        assert_eq!(budget.charge(1, 0), Ok(()));
        assert_eq!(budget.charge(1, 0), Err(SearchStop::Cancelled));
        assert_eq!(budget.operations(), 1);
        assert!(ctx.cancellation.is_cancelled());
    }

    #[test]
    fn the_summary_names_the_stop_and_every_region() {
        let ctx = ExecutionContext::for_tests(0);
        let mut budget = SearchBudget::new(LIMITS, &ctx).unwrap();
        while budget.charge(1, 0).is_ok() {}
        let receipt =
            budget.receipt(SearchStop::Operations, vec!["stage:a".into()], vec!["stage:b".into()]);
        assert_eq!(
            receipt.summary(),
            "receipt: stop search.operations; operations 3 of 3; explored [stage:a]; \
             unevaluated [stage:b]"
        );
        let mut entered = receipt;
        entered.operations_consumed = None;
        assert!(entered.summary().contains("operations not entered"));
    }

    #[test]
    fn zero_limits_and_cancellation_stop_before_entry_without_accounting() {
        let ctx = ExecutionContext::for_tests(1);
        let zero_ops =
            SearchBudget::new(SearchLimits { operations: 0, depth: 2 }, &ctx).unwrap_err();
        assert_eq!(zero_ops.stop, SearchStop::Operations);
        assert_eq!(zero_ops.operations_consumed, None);
        let zero_depth =
            SearchBudget::new(SearchLimits { operations: 3, depth: 0 }, &ctx).unwrap_err();
        assert_eq!(zero_depth.stop, SearchStop::Depth);
        ctx.cancellation.cancel();
        let cancelled = SearchBudget::new(LIMITS, &ctx).unwrap_err();
        assert_eq!(cancelled.stop, SearchStop::Cancelled);
        assert_eq!(cancelled.depth_reached, None);
    }

    #[test]
    fn operations_and_depth_limits_stop_with_measured_receipts() {
        let ctx = ExecutionContext::for_tests(1);
        let mut budget = SearchBudget::new(LIMITS, &ctx).unwrap();
        for depth in 0..3 {
            budget.charge(depth.min(2), 0).unwrap();
        }
        let stop = budget.charge(1, 0).unwrap_err();
        assert_eq!(stop, SearchStop::Operations);
        let receipt = budget.receipt(stop, vec!["rule.a".into()], vec!["state.b".into()]);
        assert_eq!(receipt.operations_consumed, Some(3));
        assert_eq!(receipt.depth_reached, Some(2));
        assert_eq!(receipt.unevaluated, ["state.b"]);

        let mut deep = SearchBudget::new(LIMITS, &ctx).unwrap();
        assert_eq!(deep.charge(3, 0), Err(SearchStop::Depth));
        assert_eq!(deep.operations(), 0);
    }

    #[test]
    fn memory_and_mid_search_cancellation_are_observed_on_every_charge() {
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(1024) };
        let mut budget = SearchBudget::new(LIMITS, &ctx).unwrap();
        budget.charge(0, 1024).unwrap();
        assert_eq!(budget.charge(0, 1025), Err(SearchStop::Memory));
        assert_eq!(
            budget.receipt(SearchStop::Memory, Vec::new(), Vec::new()).memory_limit_bytes,
            Some(1024)
        );
        ctx.cancellation.cancel();
        assert_eq!(budget.charge(0, 0), Err(SearchStop::Cancelled));
    }

    #[test]
    fn the_memory_bound_is_never_absent_and_is_the_smaller_of_cap_and_context() {
        // No context limit: the default cap applies and is reported.
        let ctx = ExecutionContext::for_tests(1);
        assert_eq!(ctx.memory.hard_limit_bytes, None);
        let mut budget = SearchBudget::new(LIMITS, &ctx).unwrap();
        assert_eq!(budget.memory_limit_bytes(), DEFAULT_SEARCH_MEMORY_BYTES);
        budget.charge(0, DEFAULT_SEARCH_MEMORY_BYTES).unwrap();
        assert_eq!(budget.charge(0, DEFAULT_SEARCH_MEMORY_BYTES + 1), Err(SearchStop::Memory));
        let receipt = budget.receipt(SearchStop::Memory, Vec::new(), Vec::new());
        assert_eq!(receipt.memory_limit_bytes, Some(DEFAULT_SEARCH_MEMORY_BYTES));
        // An explicit cap and the context limit: the smaller wins, either way round.
        let mut small = SearchBudget::with_memory(LIMITS, 100, &ctx).unwrap();
        assert_eq!(small.charge(0, 101), Err(SearchStop::Memory));
        let mut limited = ExecutionContext::for_tests(1);
        limited.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(50) };
        let capped = SearchBudget::with_memory(LIMITS, 100, &limited).unwrap();
        assert_eq!(capped.memory_limit_bytes(), 50);
        let wider = SearchBudget::with_memory(LIMITS, 10, &limited).unwrap();
        assert_eq!(wider.memory_limit_bytes(), 10);
        // Pre-entry stops report the bound too.
        let zero = SearchBudget::new(SearchLimits { operations: 0, depth: 1 }, &ctx).unwrap_err();
        assert_eq!(zero.memory_limit_bytes, Some(DEFAULT_SEARCH_MEMORY_BYTES));
    }
}
