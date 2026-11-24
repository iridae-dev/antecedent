//! The net-value reading of a fixed randomized policy.
//!
//! Selected by declaring [`EstimatorId::RandomizedIpwPolicyNet`] on a binary inverse-probability
//! `PolicyValueQuery`, or [`EstimatorId::RandomizedMultiActionIpwPolicyNet`] on a multi-action
//! one. The point values come from the point-only kernels
//! [`antecedent_estimate::policy_value::evaluate_binary_ipw_policy`] and
//! [`antecedent_estimate::policy_value::evaluate_multi_action_ipw_policy`]: each policy's value is
//! its Horvitz–Thompson sum divided by the row count minus its compensated total cost divided by
//! the row count, and the incremental value is the difference of the two net values. The
//! default route averages per-row net scores instead, the same estimator in a different order of
//! summation, so the two agree to rounding; this reading reproduces the point-only kernels to the
//! bit. The row-score standard errors are the default route's, and the pointwise intervals are
//! formed by the same rule around these points, so the result is classified under the same
//! graphless support row.
//!
//! The binary reading passes the study's declared
//! [`BinaryPolicyConstraints`](crate::analysis::binary_policy_constraints::BinaryPolicyConstraints)
//! (capacity, budget and availability of the policy and its reference) to the point-only kernel,
//! which refuses a policy that breaks one of them exactly as the direct call does. The
//! multi-action reading passes the availability, capacities and budgets of the query's
//! multi-action inputs, which query validation has already checked with the same conditions.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalError, CheckedPolicyValueOperation, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EstimatorId, Study,
};
use crate::analysis::binary_policy_constraints::BinaryPolicyConstraints;

/// The per-study settings a policy-value operation freezes at prepare: the Bayesian posterior
/// configuration, if any, and the declared binary policy constraints, if any.
#[derive(Clone, Debug)]
pub(crate) struct PolicyValueSettings {
    pub(super) bayesian: Option<super::super::BayesianConfig>,
    pub(super) constraints: Option<BinaryPolicyConstraints>,
}

impl PolicyValueSettings {
    /// The settings of `study`, with the admitted Bayesian configuration.
    pub(super) fn of(study: &Study, bayesian: Option<super::super::BayesianConfig>) -> Self {
        Self { bayesian, constraints: study.declared.binary_policy_constraints.clone() }
    }
}

/// The net-value estimator the study declares, if it declares one.
pub(super) fn selected(declared: Option<EstimatorId>) -> Option<EstimatorId> {
    declared.filter(|estimator| {
        matches!(
            estimator,
            EstimatorId::RandomizedIpwPolicyNet | EstimatorId::RandomizedMultiActionIpwPolicyNet
        )
    })
}

/// Refuse a net-value estimator on a query it does not answer: the binary reading answers a
/// fixed binary policy without outcome predictions, uplift bins or a candidate class; the
/// multi-action reading answers a multi-action policy without conditional groups.
pub(super) fn admit(
    query: &antecedent_core::PolicyValueQuery,
    estimator: EstimatorId,
) -> Result<(), CausalError> {
    let answers = match estimator {
        EstimatorId::RandomizedIpwPolicyNet => {
            query.multi_action.is_none()
                && query.mu0.is_empty()
                && query.mu1.is_empty()
                && query.uplift_bin_count == 0
                && query.regret.is_none()
        }
        EstimatorId::RandomizedMultiActionIpwPolicyNet => {
            query.multi_action.as_ref().is_some_and(|policy| policy.cate_groups.is_empty())
        }
        _ => false,
    };
    if answers {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "the net-value policy estimators answer the point value of a fixed policy only: \
         randomized.ipw_policy_net a binary policy without mu0/mu1, uplift bins or a candidate \
         class, randomized.multi_action_ipw_policy_net a multi-action policy without \
         conditional groups"
    ))
}

/// The diagnostic stating the arithmetic of the net-value point.
pub(super) fn diagnostic() -> Diagnostic {
    Diagnostic::new(
        "estimate.policy_value.net_value",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "policy and reference values are the inverse-probability value minus the mean assigned \
         cost, computed by the point-only evaluator; standard errors and intervals are the \
         independent-subject row-score ones of the same estimator",
    )
}

impl CheckedPolicyValueOperation {
    /// The net-value estimator this plan names, if it names one.
    pub(super) fn net_value_estimator(&self) -> Option<EstimatorId> {
        selected(self.physical.binding().estimator.named())
    }

    /// `score` with its point values replaced by the point-only kernel's when the plan names a
    /// net-value estimator; unchanged otherwise. Standard errors, propensity range and cost
    /// order are the row scores'.
    pub(super) fn with_net_points(
        &self,
        y: &[f64],
        mut score: antecedent_estimate::policy_value::PolicyValueScores,
    ) -> Result<antecedent_estimate::policy_value::PolicyValueScores, CausalError> {
        let Some(estimator) = self.net_value_estimator() else {
            return Ok(score);
        };
        admit(&self.query, estimator)?;
        if let Some(policy) = &self.query.multi_action {
            let k = policy.action_labels.len();
            let available: Vec<Vec<bool>> =
                policy.available.chunks_exact(k).map(<[bool]>::to_vec).collect();
            let value = antecedent_estimate::policy_value::evaluate_multi_action_ipw_policy(
                y,
                &antecedent_estimate::policy_value::MultiActionIpwDesign {
                    assigned: &policy.assignment,
                    actions: &policy.actions,
                    probabilities: &policy.propensities,
                    probability_rows: policy.assignment.len(),
                    action_count: k,
                    reference: Some(&policy.reference),
                    costs: Some(&policy.costs),
                    reference_costs: Some(&policy.reference_costs),
                    available: Some(&available),
                },
                &antecedent_estimate::policy_value::MultiActionLimits {
                    capacities: Some(&policy.capacities),
                    reference_capacities: Some(&policy.reference_capacities),
                    budget: policy.budget,
                    reference_budget: policy.reference_budget,
                },
            )
            .map_err(|message| CausalError::Unsupported { message })?;
            score.policy_value = value.policy_value;
            score.reference_value = value.reference_value;
            score.incremental_value = value.incremental_value;
            score.relative_value_gap = value.reference_value - value.policy_value;
            score.treatment_rate = value.non_control_rate;
            score.total_cost = value.policy_cost;
        } else {
            // The declared limits, if any, are checked by the kernel itself: its refusals and
            // their order are the direct call's.
            let constraints = self.extra.constraints.as_ref();
            let value = antecedent_estimate::policy_value::evaluate_binary_ipw_policy(
                y,
                &antecedent_estimate::policy_value::BinaryIpwDesign {
                    assignment: &self.query.assignment,
                    actions: &self.query.actions,
                    propensity: &self.query.propensity,
                    reference: Some(&self.query.reference),
                    costs: &self.query.costs,
                    reference_costs: Some(&self.query.reference_costs),
                    available: constraints.and_then(|declared| declared.available.as_deref()),
                },
                &constraints.map(BinaryPolicyConstraints::limits).unwrap_or_default(),
            )
            .map_err(|message| CausalError::Unsupported { message })?;
            score.policy_value = value.policy_net;
            score.reference_value = value.reference_net;
            score.incremental_value = value.incremental_value;
            score.relative_value_gap = value.regret;
            score.treatment_rate = value.treatment_rate;
            score.total_cost = value.total_cost;
        }
        Ok(score)
    }
}
