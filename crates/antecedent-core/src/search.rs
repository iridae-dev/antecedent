//! The bounded-search contract every new 2.2 search runs under.
//!
//! A search cannot be built without operation and depth limits, and every
//! charge also observes the [`ExecutionContext`]'s cancellation token and hard
//! memory limit, so none of the four bounds is optional. A stopped search
//! returns a [`SearchReceipt`] recording the limits in force, what was
//! consumed, and the explored and unevaluated regions; running out of budget is
//! a resource outcome, never an impossibility or non-identification claim.
//! Wire encoding of a receipt belongs to the artifact that carries it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::execution::ExecutionContext;

/// Operation and depth limits. Memory and cancellation come from the
/// [`ExecutionContext`] the search is charged against.
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
/// before it was entered (a zero limit, or cancellation or memory observed at
/// construction); no accounting is fabricated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchReceipt {
    /// Which bound stopped the search.
    pub stop: SearchStop,
    /// Operation limit in force.
    pub operations_limit: usize,
    /// Depth limit in force.
    pub depth_limit: usize,
    /// Hard memory limit in force, if the context set one.
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

/// A live budget: the only way a 2.2 search charges work.
#[derive(Debug)]
pub struct SearchBudget<'a> {
    limits: SearchLimits,
    ctx: &'a ExecutionContext,
    operations: usize,
    depth_reached: usize,
    entered: bool,
}

impl<'a> SearchBudget<'a> {
    /// Start a budget. A zero limit or an already cancelled context stops
    /// before the search is entered.
    ///
    /// # Errors
    ///
    /// Returns the receipt of that pre-search stop.
    pub fn new(limits: SearchLimits, ctx: &'a ExecutionContext) -> Result<Self, SearchReceipt> {
        let budget = Self { limits, ctx, operations: 0, depth_reached: 0, entered: false };
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
        if self.ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes > limit) {
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
            memory_limit_bytes: self.ctx.memory.hard_limit_bytes,
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
}
