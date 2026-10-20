// SPDX-License-Identifier: MIT OR Apache-2.0

//! Information availability for prospective features, policies, and held-out rows.
//!
//! A decision time and an outcome time are [`CadenceTick`] values: `u32`
//! indexes on one [`DeclaredCadence`]. The history operators expose their
//! cadence as a `TimeIndex` (`OperatorFrame::cadence()`), which is a different
//! type from [`DeclaredCadence`]; a caller declares the tick grid here and is
//! responsible for it matching the operator's. Index order is time order.
//!
//! A prospective feature or a held-out row may use an outcome only when
//! `outcome <= decision`. An outcome at the decision tick counts as available:
//! this module assumes the outcome of a tick is observed before the decision of
//! that same tick is made, and a deployment in which the same-tick outcome is
//! realized after the decision must declare the outcome one tick later.
//! A training nuisance may use a later outcome only when
//! [`RetrospectivePermission::Granted`] and the deployment is
//! [`ForwardOnly::NotEnforced`]. [`ForwardOnly::Enforced`] refuses that later
//! outcome. A refusal is [`InformationRefusal`]. The row is not dropped.
//!
//! Nothing here fits a model.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use thiserror::Error;

/// Index of one tick on a [`DeclaredCadence`].
///
/// `0` is the origin. [`admit_outcome`] refuses an index outside the cadence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CadenceTick(u32);

impl CadenceTick {
    /// Tick index. The cadence check is [`admit_outcome`].
    #[must_use]
    pub const fn new(tick: u32) -> Self {
        Self(tick)
    }

    /// Tick index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Regular cadence that decision and outcome ticks share.
///
/// Valid indexes are `0..tick_count`. The span is `tick_count` times
/// `interval_ns` and must fit in `u64`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DeclaredCadence {
    interval_ns: u64,
    tick_count: u32,
}

impl DeclaredCadence {
    /// Declare a positive interval and at least one tick.
    ///
    /// # Errors
    ///
    /// [`InformationRefusal::ZeroInterval`] when `interval_ns` is `0`.
    /// [`InformationRefusal::EmptyCadence`] when `tick_count` is `0`.
    /// [`InformationRefusal::CadenceSpanOverflow`] when `tick_count` times
    /// `interval_ns` does not fit in `u64`.
    #[must_use = "a refused cadence is not a declared tick grid"]
    pub const fn try_new(interval_ns: u64, tick_count: u32) -> Result<Self, InformationRefusal> {
        if interval_ns == 0 {
            return Err(InformationRefusal::ZeroInterval);
        }
        if tick_count == 0 {
            return Err(InformationRefusal::EmptyCadence);
        }
        if (tick_count as u64).checked_mul(interval_ns).is_none() {
            return Err(InformationRefusal::CadenceSpanOverflow { interval_ns, tick_count });
        }
        Ok(Self { interval_ns, tick_count })
    }

    /// Positive regular interval, in nanoseconds.
    #[must_use]
    pub const fn interval_ns(self) -> u64 {
        self.interval_ns
    }

    /// Number of ticks. Valid indexes are `0..tick_count`.
    #[must_use]
    pub const fn tick_count(self) -> u32 {
        self.tick_count
    }
}

/// How the outcome would be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum InformationUse {
    /// Policy or other prospective feature at the decision time.
    Prospective,
    /// Held-out evaluation row.
    HeldOut,
    /// Nuisance fit on a training subject.
    TrainingNuisance,
}

/// Whether the method may fit a training nuisance on a later outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RetrospectivePermission {
    /// Later outcomes in training subjects are permitted.
    Granted,
    /// The method does not permit a later outcome.
    Denied,
}

/// Deployment flag for forward-only fitting.
///
/// [`Self::Enforced`] refuses an outcome after the decision even when a
/// training nuisance has [`RetrospectivePermission::Granted`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ForwardOnly {
    /// Later outcomes are refused for every information use.
    Enforced,
    /// A granted retrospective permission may use a later training outcome.
    NotEnforced,
}

/// An outcome the requested use may see.
///
/// Holding this value does not remove the row. A refused use is
/// [`InformationRefusal`], not a missing row.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AvailableOutcome {
    decision: CadenceTick,
    outcome: CadenceTick,
    information_use: InformationUse,
}

impl AvailableOutcome {
    /// Decision tick that was admitted.
    #[must_use]
    pub const fn decision(self) -> CadenceTick {
        self.decision
    }

    /// Outcome tick that was admitted.
    #[must_use]
    pub const fn outcome(self) -> CadenceTick {
        self.outcome
    }

    /// Use that was admitted.
    #[must_use]
    pub const fn information_use(self) -> InformationUse {
        self.information_use
    }
}

/// Refusal to use an outcome. The row stays.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum InformationRefusal {
    /// The cadence interval was zero.
    #[error("cadence interval is 0 ns; next attempt: declare a positive interval")]
    ZeroInterval,
    /// The cadence declared no ticks.
    #[error("cadence declares no ticks; next attempt: declare at least one")]
    EmptyCadence,
    /// Tick count times the interval does not fit in `u64`.
    #[error(
        "cadence of {tick_count} ticks at {interval_ns} ns overflows; next attempt: shorten it"
    )]
    CadenceSpanOverflow {
        /// Regular interval, in nanoseconds.
        interval_ns: u64,
        /// Declared tick count.
        tick_count: u32,
    },
    /// The decision tick is outside `0..tick_count`.
    #[error("decision tick {tick} is outside 0..{tick_count}; next attempt: use a declared tick")]
    DecisionOffCadence {
        /// Requested decision index.
        tick: u32,
        /// Declared tick count.
        tick_count: u32,
    },
    /// The outcome tick is outside `0..tick_count`.
    #[error("outcome tick {tick} is outside 0..{tick_count}; next attempt: use a declared tick")]
    OutcomeOffCadence {
        /// Requested outcome index.
        tick: u32,
        /// Declared tick count.
        tick_count: u32,
    },
    /// A prospective feature read an outcome after the decision.
    #[error(
        "prospective tick {outcome} follows decision {decision}; next attempt: use an earlier tick"
    )]
    ProspectiveFuture {
        /// Decision index.
        decision: u32,
        /// Outcome index.
        outcome: u32,
    },
    /// A held-out row read an outcome after the decision.
    #[error(
        "held-out tick {outcome} follows decision {decision}; next attempt: use an earlier tick"
    )]
    HeldOutFuture {
        /// Decision index.
        decision: u32,
        /// Outcome index.
        outcome: u32,
    },
    /// A training nuisance read a later outcome without permission.
    ///
    /// Reported when permission is denied, including when the deployment is
    /// also [`ForwardOnly::Enforced`].
    #[error("nuisance tick {outcome} after {decision} lacks permission; next attempt: grant it")]
    RetrospectiveDenied {
        /// Decision index.
        decision: u32,
        /// Outcome index.
        outcome: u32,
    },
    /// A forward-only deployment refused a later training outcome.
    ///
    /// Reported when permission is granted and [`ForwardOnly::Enforced`].
    #[error(
        "forward-only refuses tick {outcome} after decision {decision}; next attempt: fit forward"
    )]
    ForwardOnlyDeployment {
        /// Decision index.
        decision: u32,
        /// Outcome index.
        outcome: u32,
    },
}

/// Admit `outcome` at `decision` for `information_use`.
///
/// An outcome at or before the decision is available for every use, with the
/// same-tick outcome treated as already observed (see the module note). A later
/// outcome is available only for [`InformationUse::TrainingNuisance`] with
/// [`RetrospectivePermission::Granted`] and [`ForwardOnly::NotEnforced`].
///
/// # Errors
///
/// [`InformationRefusal::DecisionOffCadence`] or
/// [`InformationRefusal::OutcomeOffCadence`] when a tick is outside `cadence`.
/// [`InformationRefusal::ProspectiveFuture`] or
/// [`InformationRefusal::HeldOutFuture`] when that use reads a later outcome.
/// [`InformationRefusal::RetrospectiveDenied`] when a training nuisance lacks
/// permission. [`InformationRefusal::ForwardOnlyDeployment`] when permission
/// is granted and the deployment is forward-only.
#[must_use = "a refused outcome stays on the row; it is not dropped"]
pub const fn admit_outcome(
    cadence: DeclaredCadence,
    decision: CadenceTick,
    outcome: CadenceTick,
    information_use: InformationUse,
    permission: RetrospectivePermission,
    forward_only: ForwardOnly,
) -> Result<AvailableOutcome, InformationRefusal> {
    let tick_count = cadence.tick_count();
    if decision.get() >= tick_count {
        return Err(InformationRefusal::DecisionOffCadence { tick: decision.get(), tick_count });
    }
    if outcome.get() >= tick_count {
        return Err(InformationRefusal::OutcomeOffCadence { tick: outcome.get(), tick_count });
    }
    if outcome.get() > decision.get() {
        match refuse_future(
            information_use,
            permission,
            forward_only,
            decision.get(),
            outcome.get(),
        ) {
            Ok(()) => {}
            Err(refusal) => return Err(refusal),
        }
    }
    Ok(AvailableOutcome { decision, outcome, information_use })
}

const fn refuse_future(
    information_use: InformationUse,
    permission: RetrospectivePermission,
    forward_only: ForwardOnly,
    decision: u32,
    outcome: u32,
) -> Result<(), InformationRefusal> {
    match information_use {
        InformationUse::Prospective => {
            Err(InformationRefusal::ProspectiveFuture { decision, outcome })
        }
        InformationUse::HeldOut => Err(InformationRefusal::HeldOutFuture { decision, outcome }),
        InformationUse::TrainingNuisance => match (permission, forward_only) {
            (RetrospectivePermission::Granted, ForwardOnly::NotEnforced) => Ok(()),
            (RetrospectivePermission::Denied, ForwardOnly::Enforced | ForwardOnly::NotEnforced) => {
                Err(InformationRefusal::RetrospectiveDenied { decision, outcome })
            }
            (RetrospectivePermission::Granted, ForwardOnly::Enforced) => {
                Err(InformationRefusal::ForwardOnlyDeployment { decision, outcome })
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> DeclaredCadence {
        DeclaredCadence::try_new(10, 8).unwrap()
    }

    fn at_decision(
        outcome: u32,
        information_use: InformationUse,
        permission: RetrospectivePermission,
        forward_only: ForwardOnly,
    ) -> Result<AvailableOutcome, InformationRefusal> {
        admit_outcome(
            grid(),
            CadenceTick::new(3),
            CadenceTick::new(outcome),
            information_use,
            permission,
            forward_only,
        )
    }

    #[test]
    fn prospective_future_outcome_is_refused() {
        let refusal = at_decision(
            5,
            InformationUse::Prospective,
            RetrospectivePermission::Granted,
            ForwardOnly::NotEnforced,
        )
        .unwrap_err();
        assert_eq!(refusal, InformationRefusal::ProspectiveFuture { decision: 3, outcome: 5 });
    }

    #[test]
    fn held_out_future_outcome_is_refused() {
        let refusal = at_decision(
            5,
            InformationUse::HeldOut,
            RetrospectivePermission::Granted,
            ForwardOnly::NotEnforced,
        )
        .unwrap_err();
        assert_eq!(refusal, InformationRefusal::HeldOutFuture { decision: 3, outcome: 5 });
    }

    #[test]
    fn training_nuisance_with_permission_accepts_a_later_outcome() {
        let available = at_decision(
            5,
            InformationUse::TrainingNuisance,
            RetrospectivePermission::Granted,
            ForwardOnly::NotEnforced,
        )
        .unwrap();
        assert_eq!(available.decision().get(), 3);
        assert_eq!(available.outcome().get(), 5);
        assert_eq!(available.information_use(), InformationUse::TrainingNuisance);
    }

    #[test]
    fn forward_only_refuses_that_later_outcome() {
        let refusal = at_decision(
            5,
            InformationUse::TrainingNuisance,
            RetrospectivePermission::Granted,
            ForwardOnly::Enforced,
        )
        .unwrap_err();
        assert_eq!(refusal, InformationRefusal::ForwardOnlyDeployment { decision: 3, outcome: 5 });
    }

    #[test]
    fn outcome_at_the_decision_time_is_available() {
        for information_use in
            [InformationUse::Prospective, InformationUse::HeldOut, InformationUse::TrainingNuisance]
        {
            let available = at_decision(
                3,
                information_use,
                RetrospectivePermission::Denied,
                ForwardOnly::Enforced,
            )
            .unwrap();
            assert_eq!(available.decision().get(), 3);
            assert_eq!(available.outcome().get(), 3);
            assert_eq!(available.information_use(), information_use);
        }
    }

    #[test]
    fn training_nuisance_without_permission_refuses_a_later_outcome() {
        let refusal = at_decision(
            5,
            InformationUse::TrainingNuisance,
            RetrospectivePermission::Denied,
            ForwardOnly::Enforced,
        )
        .unwrap_err();
        assert_eq!(refusal, InformationRefusal::RetrospectiveDenied { decision: 3, outcome: 5 });
    }

    #[test]
    fn a_tick_outside_the_cadence_is_refused() {
        let cadence = grid();
        let decision = admit_outcome(
            cadence,
            CadenceTick::new(8),
            CadenceTick::new(1),
            InformationUse::Prospective,
            RetrospectivePermission::Granted,
            ForwardOnly::NotEnforced,
        )
        .unwrap_err();
        assert_eq!(decision, InformationRefusal::DecisionOffCadence { tick: 8, tick_count: 8 });
        let outcome = admit_outcome(
            cadence,
            CadenceTick::new(1),
            CadenceTick::new(8),
            InformationUse::HeldOut,
            RetrospectivePermission::Granted,
            ForwardOnly::NotEnforced,
        )
        .unwrap_err();
        assert_eq!(outcome, InformationRefusal::OutcomeOffCadence { tick: 8, tick_count: 8 });
    }

    #[test]
    fn an_unusable_cadence_is_refused() {
        assert_eq!(DeclaredCadence::try_new(0, 4).unwrap_err(), InformationRefusal::ZeroInterval);
        assert_eq!(DeclaredCadence::try_new(10, 0).unwrap_err(), InformationRefusal::EmptyCadence);
        let interval_ns = (1_u64 << 32) + 2;
        let tick_count = u32::MAX;
        assert_eq!(
            DeclaredCadence::try_new(interval_ns, tick_count).unwrap_err(),
            InformationRefusal::CadenceSpanOverflow { interval_ns, tick_count },
        );
    }
}
