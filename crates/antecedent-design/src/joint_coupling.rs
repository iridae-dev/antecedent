//! Counterexample: equal marginals on `{0, 2}` do not decide `XY - 1`.
//!
//! `X` and `Y` each take `0` or `2` with equal probability. Acting yields
//! `XY - 1` and not acting yields `0`. Rows `(0, 0)` and `(2, 2)` have
//! expected acting utility `+1`. Rows `(0, 2)` and `(2, 0)` have expected
//! acting utility `-1`. Those joints have the same marginals, so the
//! marginals do not identify the sign.
//!
//! [`joint_sensitive_decision`] refuses when no coupling is supplied and
//! returns the bound attained by those two joints. It does not fill in a
//! coupling. Independence would give expected utility `0` and is not a value
//! of [`JointCoupling`]. When a coupling is supplied, the action is the
//! comparison of that coupling's expected acting utility with `0`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Atom `0` of the shared support.
pub const SUPPORT_ZERO: i64 = 0;
/// Atom `2` of the shared support.
pub const SUPPORT_TWO: i64 = 2;
/// Denominator of every cell and marginal probability in this counterexample.
pub const PROBABILITY_DENOMINATOR: i64 = 2;
/// Utility of not acting. Acting is chosen only when it is strictly better.
pub const NOT_ACTING_UTILITY: i64 = 0;

/// Utility of acting at one atom, `x * y - 1`.
#[must_use]
pub const fn acting_utility(x: i64, y: i64) -> i64 {
    x * y - 1
}

/// Marginal law on `{0, 2}`, numerators over [`PROBABILITY_DENOMINATOR`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupportMarginal {
    /// Numerator of the probability of `0`.
    pub mass_zero: i64,
    /// Numerator of the probability of `2`.
    pub mass_two: i64,
}

impl SupportMarginal {
    /// Both atoms have probability one half (`1` over [`PROBABILITY_DENOMINATOR`]).
    #[must_use]
    pub const fn is_equal_halves(self) -> bool {
        self.mass_zero == 1 && self.mass_two == 1
    }
}

/// A joint law of the equal marginals, with cell numerators over
/// [`PROBABILITY_DENOMINATOR`].
///
/// Only the two extremal couplings are values of this type. There is no
/// constructor for an independence coupling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointCoupling {
    /// Numerator of `P(X = 0, Y = 0)`.
    mass_0_0: i64,
    /// Numerator of `P(X = 0, Y = 2)`.
    mass_0_2: i64,
    /// Numerator of `P(X = 2, Y = 0)`.
    mass_2_0: i64,
    /// Numerator of `P(X = 2, Y = 2)`.
    mass_2_2: i64,
}

impl JointCoupling {
    /// Rows `(0, 0)` and `(2, 2)`, each with probability one half.
    ///
    /// This is the comonotone coupling. `E[XY - 1] = 1`.
    pub const ROWS_0_0_AND_2_2: Self = Self { mass_0_0: 1, mass_0_2: 0, mass_2_0: 0, mass_2_2: 1 };

    /// Rows `(0, 2)` and `(2, 0)`, each with probability one half.
    ///
    /// This is the countermonotone coupling. `E[XY - 1] = -1`.
    pub const ROWS_0_2_AND_2_0: Self = Self { mass_0_0: 0, mass_0_2: 1, mass_2_0: 1, mass_2_2: 0 };

    /// Numerator of `P(X = x, Y = y)`, or `None` when `(x, y)` is off the support.
    #[must_use = "an off-support pair is not a cell of this joint"]
    pub const fn mass_numerator(self, x: i64, y: i64) -> Option<i64> {
        match (x, y) {
            (SUPPORT_ZERO, SUPPORT_ZERO) => Some(self.mass_0_0),
            (SUPPORT_ZERO, SUPPORT_TWO) => Some(self.mass_0_2),
            (SUPPORT_TWO, SUPPORT_ZERO) => Some(self.mass_2_0),
            (SUPPORT_TWO, SUPPORT_TWO) => Some(self.mass_2_2),
            _ => None,
        }
    }

    /// Marginal law of `X`.
    #[must_use]
    pub const fn marginal_x(self) -> SupportMarginal {
        SupportMarginal {
            mass_zero: self.mass_0_0 + self.mass_0_2,
            mass_two: self.mass_2_0 + self.mass_2_2,
        }
    }

    /// Marginal law of `Y`.
    #[must_use]
    pub const fn marginal_y(self) -> SupportMarginal {
        SupportMarginal {
            mass_zero: self.mass_0_0 + self.mass_2_0,
            mass_two: self.mass_0_2 + self.mass_2_2,
        }
    }

    /// Numerator of `E[XY - 1]` over [`PROBABILITY_DENOMINATOR`].
    #[must_use]
    pub const fn acting_utility_numerator(self) -> i64 {
        self.mass_0_0 * acting_utility(SUPPORT_ZERO, SUPPORT_ZERO)
            + self.mass_0_2 * acting_utility(SUPPORT_ZERO, SUPPORT_TWO)
            + self.mass_2_0 * acting_utility(SUPPORT_TWO, SUPPORT_ZERO)
            + self.mass_2_2 * acting_utility(SUPPORT_TWO, SUPPORT_TWO)
    }

    /// `E[XY - 1]` as an exact integer.
    ///
    /// `XY` is `4` at `(2, 2)` and `0` on the other three atoms, so
    /// `E[XY - 1] = 4 * P(X = 2, Y = 2) - 1`. The cell probability is
    /// `mass_2_2 / 2`, and `4 / 2 = 2`, so the expectation is the integer
    /// `2 * mass_2_2 - 1`. That equals [`Self::acting_utility_numerator`]
    /// divided by [`PROBABILITY_DENOMINATOR`].
    #[must_use]
    pub const fn expected_acting_utility(self) -> i64 {
        2 * self.mass_2_2 - 1
    }
}

/// Whether the decision acts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Acting, whose utility is `XY - 1`.
    Act,
    /// Not acting, whose utility is [`NOT_ACTING_UTILITY`].
    NotAct,
}

/// Decision under one supplied coupling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChosenAction {
    /// The action with the higher utility. Not acting when the two are equal.
    pub action: Action,
    /// `E[XY - 1]` under the supplied coupling.
    pub expected_acting_utility: i64,
}

impl ChosenAction {
    const fn from_coupling(coupling: JointCoupling) -> Self {
        let expected_acting_utility = coupling.expected_acting_utility();
        let action = if expected_acting_utility > NOT_ACTING_UTILITY {
            Action::Act
        } else {
            Action::NotAct
        };
        Self { action, expected_acting_utility }
    }
}

/// Inclusive range of `E[XY - 1]` over every coupling of the marginals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActingUtilityBound {
    /// Lowest value. Attained by [`JointCoupling::ROWS_0_2_AND_2_0`].
    pub lower: i64,
    /// Highest value. Attained by [`JointCoupling::ROWS_0_0_AND_2_2`].
    pub upper: i64,
}

impl ActingUtilityBound {
    /// Bound preserved when the coupling is absent.
    ///
    /// The endpoints are the expectations of the two extremal joints. The
    /// bound records that range and does not select either joint.
    #[must_use]
    pub const fn preserved() -> Self {
        let concordant = JointCoupling::ROWS_0_0_AND_2_2.expected_acting_utility();
        let discordant = JointCoupling::ROWS_0_2_AND_2_0.expected_acting_utility();
        if concordant < discordant {
            Self { lower: concordant, upper: discordant }
        } else {
            Self { lower: discordant, upper: concordant }
        }
    }

    /// Whether `value` lies between the endpoints, inclusive.
    #[must_use]
    pub const fn contains(self, value: i64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// What the decision is allowed to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CouplingSource {
    /// The equal marginals on `{0, 2}` only. No pairing of `X` with `Y`.
    MarginalsOnly,
    /// A joint law the caller supplied.
    Supplied(JointCoupling),
}

/// Joint-sensitive decision refused because no coupling was supplied.
///
/// [`Self::bound`] is every value `E[XY - 1]` can take. The refusal does not
/// choose an endpoint or any interior coupling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("joint-sensitive decision refused: E[XY - 1] lies in [{lower}, {upper}]")]
pub struct CouplingAbsent {
    lower: i64,
    upper: i64,
}

impl CouplingAbsent {
    const fn from_bound(bound: ActingUtilityBound) -> Self {
        Self { lower: bound.lower, upper: bound.upper }
    }

    /// The preserved range of acting utilities.
    #[must_use]
    pub const fn bound(self) -> ActingUtilityBound {
        ActingUtilityBound { lower: self.lower, upper: self.upper }
    }
}

/// Act or refuse, according to whether a coupling was supplied.
///
/// Marginals alone produce [`CouplingAbsent`] and the Fréchet-Hoeffding bound
/// of `E[XY - 1]`. A supplied coupling is scored by that coupling's
/// expectation: act when it is strictly above [`NOT_ACTING_UTILITY`], and do
/// not act otherwise.
#[must_use = "a missing coupling is a refusal that still carries the acting-utility bound"]
pub const fn joint_sensitive_decision(
    source: CouplingSource,
) -> Result<ChosenAction, CouplingAbsent> {
    match source {
        CouplingSource::MarginalsOnly => Err(
            CouplingAbsent::from_bound(ActingUtilityBound::preserved()),
        ),
        CouplingSource::Supplied(coupling) => Ok(ChosenAction::from_coupling(coupling)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell_numerator(coupling: JointCoupling) -> i64 {
        let mut numerator = 0_i64;
        for x in [SUPPORT_ZERO, SUPPORT_TWO] {
            for y in [SUPPORT_ZERO, SUPPORT_TWO] {
                let mass = coupling.mass_numerator(x, y).expect("atom of the support");
                numerator += mass * acting_utility(x, y);
            }
        }
        numerator
    }

    #[test]
    fn acting_utility_is_xy_minus_one_and_not_acting_is_zero() {
        assert_eq!(acting_utility(SUPPORT_ZERO, SUPPORT_ZERO), -1);
        assert_eq!(acting_utility(SUPPORT_TWO, SUPPORT_TWO), 3);
        assert_eq!(acting_utility(SUPPORT_ZERO, SUPPORT_TWO), -1);
        assert_eq!(acting_utility(SUPPORT_TWO, SUPPORT_ZERO), -1);
        assert_eq!(NOT_ACTING_UTILITY, 0);
    }

    #[test]
    fn extremal_joints_expect_plus_one_and_minus_one() {
        let concordant = JointCoupling::ROWS_0_0_AND_2_2;
        let discordant = JointCoupling::ROWS_0_2_AND_2_0;

        assert_eq!(concordant.mass_numerator(SUPPORT_ZERO, SUPPORT_ZERO), Some(1));
        assert_eq!(concordant.mass_numerator(SUPPORT_TWO, SUPPORT_TWO), Some(1));
        assert_eq!(concordant.mass_numerator(SUPPORT_ZERO, SUPPORT_TWO), Some(0));
        assert_eq!(concordant.mass_numerator(SUPPORT_TWO, SUPPORT_ZERO), Some(0));
        assert_eq!(discordant.mass_numerator(SUPPORT_ZERO, SUPPORT_TWO), Some(1));
        assert_eq!(discordant.mass_numerator(SUPPORT_TWO, SUPPORT_ZERO), Some(1));
        assert_eq!(discordant.mass_numerator(SUPPORT_ZERO, SUPPORT_ZERO), Some(0));
        assert_eq!(discordant.mass_numerator(SUPPORT_TWO, SUPPORT_TWO), Some(0));
        assert_eq!(concordant.mass_numerator(1, 1), None);

        assert_eq!(cell_numerator(concordant), concordant.acting_utility_numerator());
        assert_eq!(cell_numerator(discordant), discordant.acting_utility_numerator());
        assert_eq!(
            concordant.acting_utility_numerator(),
            concordant.expected_acting_utility() * PROBABILITY_DENOMINATOR,
        );
        assert_eq!(
            discordant.acting_utility_numerator(),
            discordant.expected_acting_utility() * PROBABILITY_DENOMINATOR,
        );
        assert_eq!(concordant.expected_acting_utility(), 1);
        assert_eq!(discordant.expected_acting_utility(), -1);
    }

    #[test]
    fn extremal_joints_share_equal_marginals_on_zero_and_two() {
        let concordant = JointCoupling::ROWS_0_0_AND_2_2;
        let discordant = JointCoupling::ROWS_0_2_AND_2_0;
        assert_eq!(concordant.marginal_x(), concordant.marginal_y());
        assert_eq!(discordant.marginal_x(), discordant.marginal_y());
        assert_eq!(concordant.marginal_x(), discordant.marginal_x());
        assert_eq!(concordant.marginal_y(), discordant.marginal_y());
        assert!(concordant.marginal_x().is_equal_halves());
        assert!(discordant.marginal_y().is_equal_halves());
        assert_eq!(concordant.marginal_x().mass_zero, 1);
        assert_eq!(concordant.marginal_x().mass_two, 1);
    }

    #[test]
    fn marginals_only_refuse_the_decision_and_preserve_the_bound() {
        let refusal = joint_sensitive_decision(CouplingSource::MarginalsOnly)
            .expect_err("marginals do not identify the joint");
        let bound = refusal.bound();
        assert_eq!(bound, ActingUtilityBound::preserved());
        assert_eq!(bound.lower, -1);
        assert_eq!(bound.upper, 1);
        assert_eq!(bound.lower, JointCoupling::ROWS_0_2_AND_2_0.expected_acting_utility());
        assert_eq!(bound.upper, JointCoupling::ROWS_0_0_AND_2_2.expected_acting_utility());
        // Independence would give expected acting utility 0. That value sits
        // in the bound. The refusal does not adopt it or either endpoint.
        assert!(bound.contains(NOT_ACTING_UTILITY));
        assert!(bound.lower < NOT_ACTING_UTILITY);
        assert!(bound.upper > NOT_ACTING_UTILITY);
        assert!(joint_sensitive_decision(CouplingSource::MarginalsOnly).is_err());
    }

    #[test]
    fn supplied_coupling_follows_that_coupling() {
        let concordant = joint_sensitive_decision(CouplingSource::Supplied(
            JointCoupling::ROWS_0_0_AND_2_2,
        ))
        .expect("concordant coupling");
        let discordant = joint_sensitive_decision(CouplingSource::Supplied(
            JointCoupling::ROWS_0_2_AND_2_0,
        ))
        .expect("discordant coupling");

        assert_eq!(
            concordant,
            ChosenAction { action: Action::Act, expected_acting_utility: 1 }
        );
        assert_eq!(
            discordant,
            ChosenAction { action: Action::NotAct, expected_acting_utility: -1 }
        );
        assert_eq!(
            concordant.expected_acting_utility,
            JointCoupling::ROWS_0_0_AND_2_2.expected_acting_utility()
        );
        assert_eq!(
            discordant.expected_acting_utility,
            JointCoupling::ROWS_0_2_AND_2_0.expected_acting_utility()
        );
        assert!(concordant.expected_acting_utility > NOT_ACTING_UTILITY);
        assert!(discordant.expected_acting_utility < NOT_ACTING_UTILITY);
        assert_ne!(concordant.action, discordant.action);
    }

    #[test]
    fn expectations_are_exact_integers_with_exact_f64_images() {
        let positive = JointCoupling::ROWS_0_0_AND_2_2.expected_acting_utility();
        let negative = JointCoupling::ROWS_0_2_AND_2_0.expected_acting_utility();
        assert_eq!(positive, 1);
        assert_eq!(negative, -1);
        assert!(positive.unsigned_abs() <= (1_u64 << 53));
        assert!(negative.unsigned_abs() <= (1_u64 << 53));
        assert_eq!((positive as f64).to_bits(), 1.0_f64.to_bits());
        assert_eq!((negative as f64).to_bits(), (-1.0_f64).to_bits());
    }
}
