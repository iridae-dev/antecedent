//! Randomized survival and competing-risk query contract.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ObservationAssumption, QueryError};
use crate::ids::VariableId;
use std::sync::Arc;

/// Caller-supplied censoring survival on a common time grid.
///
/// Each column contains the probability of remaining uncensored through its
/// corresponding time for every observation. The probabilities are treated as
/// known; this query does not fit or validate a censoring model.
#[derive(Clone, Debug, PartialEq)]
pub struct KnownCensoringSurvival {
    /// Common time grid beginning at zero and ending at the query horizon.
    pub times: Arc<[f64]>,
    /// Data columns holding each subject's censoring survival at those times.
    pub columns: Arc<[VariableId]>,
    /// Strict positivity floor applied to every supplied probability.
    pub minimum_probability: f64,
}

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
/// marginally independent censoring and, when present, entry. Conditional
/// censoring requires caller-supplied known censoring survival. Combining
/// known censoring survival with delayed entry retains the marginal empty-set
/// observation claim; conditional entry is not adjusted.
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
    /// Optional known conditional censoring survival, aligned by data row.
    pub known_censoring: Option<KnownCensoringSurvival>,
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
        if let Some(known) = &self.known_censoring {
            if self.delayed_entry.is_some()
                && !matches!(&self.observation_assumption, ObservationAssumption::IndependentGiven(vars) if vars.is_empty())
            {
                return Err(QueryError::InvalidSurvival(
                    "combined delayed entry and known censoring requires marginal IndependentGiven(())".into(),
                ));
            }
            if known.times.len() < 2
                || known.columns.len() != known.times.len()
                || known.times[0] != 0.0
                || (known.times[known.times.len() - 1] - self.tau).abs() > 1e-10
                || known.times.iter().any(|time| !time.is_finite())
                || known.times.windows(2).any(|pair| pair[0] >= pair[1])
                || !known.minimum_probability.is_finite()
                || !(0.0..=1.0).contains(&known.minimum_probability)
                || known.minimum_probability == 0.0
                || known.columns.iter().any(|id| *id == self.duration || *id == self.event || *id == self.treatment)
                || known.columns.iter().enumerate().any(|(index, id)| known.columns[..index].contains(id))
            {
                return Err(QueryError::InvalidSurvival(
                    "known censoring requires aligned columns, increasing times from zero through tau, and a positive probability floor".into(),
                ));
            }
        }
        if !matches!(&self.observation_assumption, ObservationAssumption::IndependentGiven(vars) if self.known_censoring.is_some() || vars.is_empty()) {
            return Err(QueryError::InvalidSurvival(
                "randomized survival requires IndependentGiven; conditional censoring also requires known censoring survival".into(),
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
            known_censoring: None,
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

    #[test]
    fn combined_entry_and_known_censoring_requires_marginal_observation_claim() {
        let mut q = query();
        q.observation_assumption =
            ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(3)]));
        q.known_censoring = Some(KnownCensoringSurvival {
            times: Arc::from([0.0, 1.0, 2.0]),
            columns: Arc::from([
                VariableId::from_raw(4), VariableId::from_raw(5), VariableId::from_raw(6),
            ]),
            minimum_probability: 0.01,
        });
        assert!(q.validate().is_ok());
        q.delayed_entry = Some(VariableId::from_raw(7));
        assert!(q.validate().is_err());
        q.observation_assumption = ObservationAssumption::IndependentGiven(Arc::from([]));
        assert!(q.validate().is_ok());
    }
}
