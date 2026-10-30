//! Shared mechanics for the probability and cost checks of query contracts.
//!
//! Each tolerance below belongs to one contract and is named for it. They are
//! deliberately different numbers: a row of declared action probabilities is
//! typed by a person or serialized at limited precision, so its total is held
//! to a looser bound than two declarations of the *same* probability, which
//! must agree to near machine precision. Strict positivity (assignment
//! probabilities) and closed positivity (partitions allowing zero mass) are
//! separate predicates and are not merged here.
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Absolute bound on `|sum(row) - 1|` for a declared row of mutually exclusive
/// assignment probabilities. The total is accumulated with compensated
/// summation, so this bound is the declaration's precision, not rounding error.
pub(crate) const ASSIGNMENT_ROW_TOTAL_TOLERANCE: f64 = 1e-8;

/// Absolute bound on the difference between two declarations of the same
/// assignment probability (a row's chosen-arm entry against the unit-level
/// scalar, a common probability across units, a count ratio against a stated
/// probability).
pub(crate) const ASSIGNMENT_AGREEMENT_TOLERANCE: f64 = 1e-12;

/// Absolute floor of the slack allowed on a spend total against its budget.
pub(crate) const COST_ABSOLUTE_TOLERANCE: f64 = 1e-12;

/// Relative part of the slack allowed on a spend total against its budget,
/// scaled by the larger magnitude of the total and the limit so the check is
/// invariant to the caller's cost unit. It covers the rounding of a sum of up
/// to a few thousand terms.
pub(crate) const COST_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Absolute bound for reading a stored treatment value as exactly 0 or 1.
pub(crate) const BINARY_CODING_TOLERANCE: f64 = 1e-12;

/// Neumaier-compensated sum of `values`.
pub(crate) fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = sum + value;
        compensation +=
            if sum.abs() >= value.abs() { (sum - next) + value } else { (value - next) + sum };
        sum = next;
    }
    sum + compensation
}

/// Whether `row` totals one within [`ASSIGNMENT_ROW_TOTAL_TOLERANCE`].
pub(crate) fn row_totals_one(row: &[f64]) -> bool {
    (compensated_sum(row.iter().copied()) - 1.0).abs() <= ASSIGNMENT_ROW_TOTAL_TOLERANCE
}

/// Whether two declarations of one probability agree.
pub(crate) fn probabilities_agree(a: f64, b: f64) -> bool {
    (a - b).abs() <= ASSIGNMENT_AGREEMENT_TOLERANCE
}

/// Whether `total` spend fits within `limit`, allowing the absolute plus
/// relative slack above. A non-finite input never fits.
pub(crate) fn within_budget(total: f64, limit: f64) -> bool {
    if !total.is_finite() || !limit.is_finite() {
        return false;
    }
    total
        <= limit + COST_ABSOLUTE_TOLERANCE + COST_RELATIVE_TOLERANCE * total.abs().max(limit.abs())
}

/// Whether a finite nonnegative assignment cost is within a finite nonnegative budget,
/// using the same rounding allowance as policy query validation.
#[must_use]
pub fn policy_assignment_cost_within_budget(total: f64, budget: f64) -> bool {
    total >= 0.0 && budget >= 0.0 && within_budget(total, budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sum_recovers_cancelled_small_terms() {
        let values = [1.0, 1e-16, 1e-16, -1.0];
        assert!((compensated_sum(values) - 2e-16).abs() < 1e-30);
    }

    #[test]
    fn row_total_uses_the_declaration_tolerance() {
        assert!(row_totals_one(&[0.25, 0.25, 0.5]));
        assert!(row_totals_one(&[0.25, 0.25, 0.5 + 5e-9]));
        assert!(!row_totals_one(&[0.25, 0.25, 0.5 + 5e-8]));
    }

    #[test]
    fn agreement_is_tighter_than_the_row_total() {
        assert!(probabilities_agree(0.5, 0.5 + 1e-13));
        assert!(!probabilities_agree(0.5, 0.5 + 1e-9));
    }

    #[test]
    fn public_policy_budget_check_matches_contract_and_rejects_nonfinite() {
        assert!(policy_assignment_cost_within_budget(0.1 + 0.1 + 0.1, 0.3));
        assert!(!policy_assignment_cost_within_budget(0.3, 0.2999));
        assert!(!policy_assignment_cost_within_budget(f64::INFINITY, f64::INFINITY));
        assert!(!policy_assignment_cost_within_budget(-1.0, 1.0));
    }

    #[test]
    fn budget_slack_scales_with_the_cost_unit() {
        assert!(within_budget(10.0, 10.0));
        assert!(!within_budget(10.1, 10.0));
        // Large-unit costs tolerate rounding that would exceed a fixed 1e-12.
        let limit = 3.0e9;
        assert!(within_budget(limit * (1.0 + 5e-13), limit));
        assert!(!within_budget(limit * (1.0 + 1e-9), limit));
        assert!(!within_budget(f64::NAN, 1.0));
    }
}
