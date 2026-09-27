//! Query contract for held-out doubly robust policy value.
//!
//! The nuisance predictions and policy actions are frozen on the evaluation
//! rows; prediction ownership is retained as caller-declared metadata.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Binary policy value query evaluated on held-out or declared cross-fitted rows.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValueQuery {
    /// Observed outcome column.
    pub outcome: VariableId,
    /// Realized randomized assignment in row order.
    pub assignment: Arc<[bool]>,
    /// Known assignment propensity, scalar or row-aligned.
    pub propensity: Arc<[f64]>,
    /// Binary action recommended by the policy.
    pub actions: Arc<[bool]>,
    /// Reference policy (all control when omitted).
    pub reference: Arc<[bool]>,
    /// Frozen control predictions; both prediction vectors may be empty for IPW.
    pub mu0: Arc<[f64]>,
    /// Frozen treated conditional outcome predictions.
    pub mu1: Arc<[f64]>,
    /// Policy and reference costs, scalar or row-aligned.
    pub costs: Arc<[f64]>,
    /// Reference-policy costs, scalar or row-aligned.
    pub reference_costs: Arc<[f64]>,
    /// Evaluation subject IDs used to bind prediction ownership.
    pub evaluation_subject_ids: Arc<[Arc<str>]>,
    /// Whether training subject IDs were declared disjoint.
    pub disjoint_training_subjects: bool,
    /// Whether fold IDs match the prediction-excluded fold IDs.
    pub crossfit_fold_ownership_valid: bool,
}

impl PolicyValueQuery {
    /// Validate frozen row-aligned policy value inputs and positivity.
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.assignment.len();
        let ipw = self.mu0.is_empty() && self.mu1.is_empty();
        if n < 2
            || self.actions.len() != n
            || self.reference.len() != n
            || (!ipw && (self.mu0.len() != n || self.mu1.len() != n))
            || self.evaluation_subject_ids.len() != n
            || !(self.propensity.len() == 1 || self.propensity.len() == n)
            || !(self.costs.len() == 1 || self.costs.len() == n)
            || !(self.reference_costs.len() == 1 || self.reference_costs.len() == n)
        {
            return Err(QueryError::InvalidPolicyValue(
                "policy, outcome predictions, ownership IDs, and assignment must align on at least two rows".into(),
            ));
        }
        if self.propensity.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) {
            return Err(QueryError::InvalidPolicyValue(
                "randomized propensities must be finite and strictly between zero and one".into(),
            ));
        }
        if self.mu0.iter().chain(self.mu1.iter()).any(|x| !x.is_finite())
            || self
                .costs
                .iter()
                .chain(self.reference_costs.iter())
                .any(|x| !x.is_finite() || *x < 0.0)
            || self.evaluation_subject_ids.iter().any(|id| id.trim().is_empty())
            || self.evaluation_subject_ids.iter().collect::<std::collections::HashSet<_>>().len()
                != n
        {
            return Err(QueryError::InvalidPolicyValue(
                "predictions, costs, and unique non-empty evaluation subject IDs must be valid"
                    .into(),
            ));
        }
        if ipw && (self.disjoint_training_subjects || self.crossfit_fold_ownership_valid) {
            return Err(QueryError::InvalidPolicyValue(
                "IPW has no nuisance prediction ownership metadata".into(),
            ));
        }
        if !ipw && !(self.disjoint_training_subjects || self.crossfit_fold_ownership_valid) {
            return Err(QueryError::InvalidPolicyValue(
                "prediction ownership requires disjoint training IDs or matching excluded-fold metadata".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> PolicyValueQuery {
        PolicyValueQuery {
            outcome: VariableId::from_raw(0),
            assignment: Arc::from([false, true]),
            propensity: Arc::from([0.5]),
            actions: Arc::from([false, true]),
            reference: Arc::from([false, false]),
            mu0: Arc::from([1.0, 1.0]),
            mu1: Arc::from([2.0, 2.0]),
            costs: Arc::from([0.0]),
            reference_costs: Arc::from([0.0]),
            evaluation_subject_ids: Arc::from([Arc::<str>::from("a"), Arc::<str>::from("b")]),
            disjoint_training_subjects: true,
            crossfit_fold_ownership_valid: false,
        }
    }

    #[test]
    fn accepts_aligned_held_out_contract() {
        assert!(valid().validate().is_ok());
    }

    #[test]
    fn refuses_bad_ownership_and_zero_propensity() {
        let mut query = valid();
        query.disjoint_training_subjects = false;
        assert!(query.validate().is_err());
        query = valid();
        query.propensity = Arc::from([0.0]);
        assert!(query.validate().is_err());
    }

    #[test]
    fn ipw_requires_no_nuisance_ownership_but_refuses_partial_predictions() {
        let mut query = valid();
        query.mu0 = Arc::from([]);
        query.mu1 = Arc::from([]);
        query.disjoint_training_subjects = false;
        assert!(query.validate().is_ok());
        query.mu0 = Arc::from([1.0, 1.0]);
        assert!(query.validate().is_err());
    }
}
