//! Randomized survival and competing-risk query contract.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ObservationAssumption, QueryError};
use crate::ids::VariableId;

/// Requested functional of a right-censored event-time distribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SurvivalFunctional {
    /// Arm survival curves and restricted mean survival time through `tau`.
    SurvivalAndRmst,
    /// Arm cumulative-incidence curves for a positive event-cause code.
    CumulativeIncidence {
        /// Target event code; zero is reserved for censoring.
        target_cause: i64,
    },
}

/// A randomized two-arm event-time query with an explicit observation claim.
///
/// The event column is zero for censoring and one for a binary event, or a
/// positive cause code for competing risks. `IndependentGiven([])` declares
/// marginally independent censoring. Conditional assumptions require a
/// censoring-adjusted estimator and are refused by this point estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct SurvivalQuery {
    /// Observed follow-up duration.
    pub duration: VariableId,
    /// Event indicator or coded event cause.
    pub event: VariableId,
    /// Observed binary randomized treatment assignment.
    pub treatment: VariableId,
    /// Restriction horizon shared by both arms.
    pub tau: f64,
    /// Left-truncation entry time, when observed.
    pub delayed_entry: Option<VariableId>,
    /// Explicit independent-censoring and entry assumption.
    pub observation_assumption: ObservationAssumption,
    /// Requested curve or competing-risk functional.
    pub functional: SurvivalFunctional,
}

impl SurvivalQuery {
    /// Validate identifiers, horizon, functional, and observation contract.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.duration == self.event
            || self.duration == self.treatment
            || self.event == self.treatment
            || self.delayed_entry.is_some_and(|entry| {
                entry == self.duration || entry == self.event || entry == self.treatment
            })
        {
            return Err(QueryError::InvalidSurvival(
                "duration, event, treatment, and entry must name distinct columns".into(),
            ));
        }
        if !self.tau.is_finite() || self.tau <= 0.0 {
            return Err(QueryError::InvalidSurvival("tau must be finite and positive".into()));
        }
        if matches!(self.functional, SurvivalFunctional::CumulativeIncidence { target_cause } if target_cause <= 0)
        {
            return Err(QueryError::InvalidSurvival(
                "target cause must be a positive event code".into(),
            ));
        }
        if !matches!(&self.observation_assumption, ObservationAssumption::IndependentGiven(vars) if vars.is_empty())
        {
            return Err(QueryError::InvalidSurvival(
                "the unadjusted randomized survival route requires explicit marginal IndependentGiven([]) censoring/entry assumptions".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn query() -> SurvivalQuery {
        SurvivalQuery {
            duration: VariableId::from_raw(0),
            event: VariableId::from_raw(1),
            treatment: VariableId::from_raw(2),
            tau: 2.0,
            delayed_entry: None,
            observation_assumption: ObservationAssumption::IndependentGiven(Arc::from([])),
            functional: SurvivalFunctional::SurvivalAndRmst,
        }
    }

    #[test]
    fn accepts_explicit_marginal_observation_contract() {
        assert!(query().validate().is_ok());
    }

    #[test]
    fn refuses_conditional_censoring_without_adjustment() {
        let mut q = query();
        q.observation_assumption =
            ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(3)]));
        assert!(q.validate().is_err());
    }
}
