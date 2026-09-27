//! Query contract for randomized binary and multi-action policy value.
//!
//! The nuisance predictions and policy actions are frozen on the evaluation
//! rows; prediction ownership is retained as caller-declared metadata.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Frozen randomized multi-action policy inputs in row-major action order.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiActionPolicyInputs {
    /// Ordered action labels; the first is control.
    pub action_labels: Arc<[Arc<str>]>,
    /// Realized action index for each evaluation row.
    pub assignment: Arc<[usize]>,
    /// Fixed policy action index for each row.
    pub actions: Arc<[usize]>,
    /// Fixed reference action index for each row.
    pub reference: Arc<[usize]>,
    /// Positive row-major randomized action probabilities.
    pub propensities: Arc<[f64]>,
    /// Row-major action availability mask.
    pub available: Arc<[bool]>,
    /// Per-action policy cost.
    pub costs: Arc<[f64]>,
    /// Per-action reference cost.
    pub reference_costs: Arc<[f64]>,
    /// Per-action policy capacity.
    pub capacities: Arc<[usize]>,
    /// Per-action reference capacity.
    pub reference_capacities: Arc<[usize]>,
    /// Maximum total policy spend.
    pub budget: Option<f64>,
    /// Maximum total reference spend.
    pub reference_budget: Option<f64>,
    /// Frozen pre-treatment stratum for action-specific conditional effects.
    pub cate_groups: Arc<[Arc<str>]>,
}

impl MultiActionPolicyInputs {
    /// Whether capacities or budgets can couple recommendations across rows.
    /// An unconstraining declared limit is harmless to row-score inference.
    pub fn global_constraints_couple_rows(&self) -> bool {
        let n = self.assignment.len();
        [(&self.capacities, &self.costs, self.budget),
         (&self.reference_capacities, &self.reference_costs, self.reference_budget)]
            .into_iter().any(|(capacities, costs, budget)| {
                capacities.iter().any(|&limit| limit < n)
                    || budget.is_some_and(|limit| {
                        let maximum_cost = costs.iter().copied().fold(0.0_f64, f64::max);
                        limit + 1e-12 < n as f64 * maximum_cost
                    })
            })
    }

    /// Validate action support, probability rows, availability, and constraints.
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.assignment.len();
        let k = self.action_labels.len();
        if n < 2 || k < 2 || self.actions.len() != n || self.reference.len() != n
            || self.propensities.len() != n * k || self.available.len() != n * k
            || self.costs.len() != k || self.reference_costs.len() != k
            || self.capacities.len() != k || self.reference_capacities.len() != k
            || self.action_labels.iter().any(|x| x.trim().is_empty())
            || self.action_labels.iter().collect::<std::collections::HashSet<_>>().len() != k
            || self.assignment.iter().chain(self.actions.iter()).chain(self.reference.iter()).any(|&a| a >= k)
            || (!self.cate_groups.is_empty() && (self.cate_groups.len() != n
                || self.cate_groups.iter().any(|group| group.trim().is_empty())))
        {
            return Err(QueryError::InvalidPolicyValue("multi-action policy rows, labels, probabilities, and constraints must align".into()));
        }
        if self.costs.iter().chain(self.reference_costs.iter()).any(|x| !x.is_finite() || *x < 0.0)
            || [self.budget, self.reference_budget].into_iter().flatten().any(|x| !x.is_finite() || x < 0.0)
            || self.propensities.chunks_exact(k).any(|row| {
                row.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
                    || (row.iter().sum::<f64>() - 1.0).abs() > 1e-8
            })
        {
            return Err(QueryError::InvalidPolicyValue("multi-action probabilities must be positive and sum to one; costs and budgets must be finite and non-negative".into()));
        }
        for (recommendations, costs, capacities, budget) in [
            (&self.actions, &self.costs, &self.capacities, self.budget),
            (&self.reference, &self.reference_costs, &self.reference_capacities, self.reference_budget),
        ] {
            let mut counts = vec![0usize; k];
            let mut total_cost = 0.0;
            for (i, &action) in recommendations.iter().enumerate() {
                if !self.available[i * k + action] {
                    return Err(QueryError::InvalidPolicyValue("policy selects an unavailable action".into()));
                }
                counts[action] += 1;
                total_cost += costs[action];
            }
            if counts.iter().zip(capacities.iter()).any(|(count, limit)| count > limit)
                || budget.is_some_and(|limit| total_cost > limit + 1e-12)
            {
                return Err(QueryError::InvalidPolicyValue("policy exceeds its action capacity or budget".into()));
            }
        }
        Ok(())
    }
}

/// Fixed randomized policy value query evaluated on independent subjects.
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
    /// A global capacity or budget selected recommendations jointly across
    /// evaluation rows; row-score pointwise intervals are withheld.
    pub global_constraints_present: bool,
    /// Multi-action randomized IPW inputs; binary fields are empty when present.
    pub multi_action: Option<MultiActionPolicyInputs>,
    /// Frozen descending-score rank bin for each binary evaluation row.
    pub uplift_bins: Arc<[usize]>,
    /// Number of nonempty, descending-score uplift bins.
    pub uplift_bin_count: usize,
    /// Subjects used to train the ranking model, declared disjoint from evaluation subjects.
    pub uplift_training_subject_ids: Arc<[Arc<str>]>,
    /// Prespecified finite candidate class for held-out regret, when requested.
    pub regret: Option<FixedCandidateRegretInputs>,
}

/// Fixed candidate recommendations and training ownership for finite-class regret.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedCandidateRegretInputs {
    /// Candidate action vectors in a prespecified order.
    pub candidates: Arc<[Arc<[bool]>]>,
    /// The selected candidate, which must equal `PolicyValueQuery::actions`.
    pub selected_index: usize,
    /// Caller-declared candidate construction and selection subjects.
    pub training_subject_ids: Arc<[Arc<str>]>,
}

impl PolicyValueQuery {
    /// Validate frozen row-aligned policy value inputs and positivity.
    pub fn validate(&self) -> Result<(), QueryError> {
        if let Some(multi) = &self.multi_action {
            if !self.assignment.is_empty() || !self.propensity.is_empty() || !self.actions.is_empty()
                || !self.reference.is_empty() || !self.mu0.is_empty() || !self.mu1.is_empty()
                || !self.costs.is_empty() || !self.reference_costs.is_empty()
            || self.disjoint_training_subjects || self.crossfit_fold_ownership_valid
                || self.global_constraints_present
                || !self.uplift_bins.is_empty() || self.uplift_bin_count != 0
                || !self.uplift_training_subject_ids.is_empty()
                || self.regret.is_some()
                || self.evaluation_subject_ids.len() != multi.assignment.len()
                || self.evaluation_subject_ids.iter().any(|id| id.trim().is_empty())
                || self.evaluation_subject_ids.iter().collect::<std::collections::HashSet<_>>().len() != multi.assignment.len()
            {
                return Err(QueryError::InvalidPolicyValue("multi-action policy must carry only multi-action inputs and unique evaluation subject IDs".into()));
            }
            return multi.validate();
        }
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
        if self.uplift_bin_count == 0 {
            if !self.uplift_bins.is_empty() || !self.uplift_training_subject_ids.is_empty() {
                return Err(QueryError::InvalidPolicyValue("uplift bins and training subjects require a positive bin count".into()));
            }
        } else {
            if self.uplift_bins.len() != n || self.uplift_bin_count > n
                || self.uplift_training_subject_ids.is_empty()
                || self.uplift_training_subject_ids.iter().any(|id| id.trim().is_empty() || self.evaluation_subject_ids.contains(id))
                || self.uplift_training_subject_ids.iter().collect::<std::collections::HashSet<_>>().len() != self.uplift_training_subject_ids.len()
                || self.uplift_bins.iter().any(|&bin| bin >= self.uplift_bin_count)
                || (0..self.uplift_bin_count).any(|bin| self.uplift_bins.iter().filter(|&&x| x == bin).count() < 2)
            {
                return Err(QueryError::InvalidPolicyValue("uplift bins must cover evaluation rows and each bin; declared training subjects must be unique and disjoint".into()));
            }
        }
        if !ipw && !(self.disjoint_training_subjects || self.crossfit_fold_ownership_valid) {
            return Err(QueryError::InvalidPolicyValue(
                "prediction ownership requires disjoint training IDs or matching excluded-fold metadata".into(),
            ));
        }
        if let Some(regret) = &self.regret {
            let training = &regret.training_subject_ids;
            if !ipw || self.global_constraints_present || self.uplift_bin_count != 0
                || !(2..=16).contains(&regret.candidates.len())
                || regret.selected_index >= regret.candidates.len()
                || regret.candidates.iter().any(|actions| actions.len() != n)
                || regret.candidates[regret.selected_index].as_ref() != self.actions.as_ref()
                || training.is_empty() || training.iter().any(|id| id.trim().is_empty()
                    || self.evaluation_subject_ids.contains(id))
                || training.iter().collect::<std::collections::HashSet<_>>().len() != training.len()
            {
                return Err(QueryError::InvalidPolicyValue(
                    "finite-class regret requires 2–16 fixed IPW candidates, selected-policy equality, and disjoint construction subjects".into(),
                ));
            }
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
            global_constraints_present: false,
            multi_action: None,
            uplift_bins: Arc::from([]),
            uplift_bin_count: 0,
            uplift_training_subject_ids: Arc::from([]),
            regret: None,
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
