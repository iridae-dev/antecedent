//! Declared capacity, budget and availability limits of a fixed binary policy.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::StudyBuilder;
use crate::analysis::binary_policy_constraints::BinaryPolicyConstraints;

impl StudyBuilder {
    /// The capacity, budget and availability limits a fixed binary policy and its reference
    /// were chosen under, for a binary [`CausalQuery::PolicyValue`] study that declares the
    /// `randomized.ipw_policy_net` estimator.
    ///
    /// The route passes the limits to the point-only evaluator it reads the policy with
    /// (`antecedent_estimate::policy_value::evaluate_binary_ipw_policy`), so a policy or
    /// reference that treats an unavailable row, exceeds its capacity or exceeds its budget is
    /// refused with that evaluator's message, and an accepted policy has the values the
    /// evaluator returns. The limits enter the study's program identity. [`Self::build`]
    /// refuses them (`option_not_applicable`) on any other query or estimator.
    ///
    /// [`CausalQuery::PolicyValue`]: antecedent_core::CausalQuery::PolicyValue
    #[must_use]
    pub fn binary_policy_constraints(mut self, constraints: BinaryPolicyConstraints) -> Self {
        self.declared.binary_policy_constraints = Some(constraints);
        self
    }
}
