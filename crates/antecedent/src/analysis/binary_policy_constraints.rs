//! Declared treatment capacity, budget and availability of a fixed binary policy.
//!
//! The point-only evaluator `antecedent_estimate::policy_value::evaluate_binary_ipw_policy`
//! takes, beside the policy, the limits it was chosen under: the most rows the policy and its
//! reference may treat, the most total cost each may incur, and which rows may be treated at
//! all. It refuses a policy that breaks one of them. A [`antecedent_core::PolicyValueQuery`] carries
//! none of these (its fields are fixed), so a study states them with
//! [`crate::StudyBuilder::binary_policy_constraints`]. The compiled `PolicyValue` route under
//! the declared `randomized.ipw_policy_net` estimator passes them to that same kernel, so it
//! refuses exactly the policies the kernel refuses, with the kernel's own message, and returns
//! the same values for every policy it accepts.
//!
//! The constraints are checks, never selections: they do not change an accepted estimate or
//! the row-score intervals around it. A policy that was itself chosen jointly across rows under
//! a binding capacity or budget still states that on the query's `global_constraints_present`,
//! which withholds the independent-row intervals.
//!
//! A declaration enters the study's lowered program (a model slot named by the digest of the
//! limits), so two studies that differ only in their declared limits have different program
//! identities; a study that declares none keeps the identity it had.
//!
//! A multi-action policy carries its availability, capacities and budgets in
//! [`antecedent_core::MultiActionPolicyInputs`], which the multi-action route already checks.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, CausalQuery};
use antecedent_estimate::policy_value::BinaryPolicyLimits;

use crate::CausalError;
use crate::strategy_table::EstimatorId;

/// Domain tag of the digest that names a declaration in the lowered program.
const IDENTITY_TAG: &str = "antecedent.binary_policy_constraints";

/// The capacity, budget and availability limits a fixed binary policy and its reference were
/// chosen under, checked by the compiled net-value reading of the policy.
///
/// Every limit is optional; an omitted one is not checked. Budgets bound the total assigned
/// cost (the query's `costs` and `reference_costs` summed over the rows each policy treats) and
/// must be finite and non-negative. `available` holds one flag per evaluation row; neither the
/// policy nor the reference may treat a row whose flag is `false`.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct BinaryPolicyConstraints {
    /// Most rows the policy may treat.
    pub capacity: Option<usize>,
    /// Most total cost the policy may incur.
    pub budget: Option<f64>,
    /// Most rows the reference policy may treat.
    pub reference_capacity: Option<usize>,
    /// Most total cost the reference policy may incur.
    pub reference_budget: Option<f64>,
    /// Whether each evaluation row may be treated.
    pub available: Option<Arc<[bool]>>,
}

impl BinaryPolicyConstraints {
    /// No limits; add them with the `with_*` methods.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The policy treats at most `limit` rows.
    #[must_use]
    pub fn with_capacity(mut self, limit: usize) -> Self {
        self.capacity = Some(limit);
        self
    }

    /// The policy's total assigned cost is at most `limit`.
    #[must_use]
    pub fn with_budget(mut self, limit: f64) -> Self {
        self.budget = Some(limit);
        self
    }

    /// The reference policy treats at most `limit` rows.
    #[must_use]
    pub fn with_reference_capacity(mut self, limit: usize) -> Self {
        self.reference_capacity = Some(limit);
        self
    }

    /// The reference policy's total assigned cost is at most `limit`.
    #[must_use]
    pub fn with_reference_budget(mut self, limit: f64) -> Self {
        self.reference_budget = Some(limit);
        self
    }

    /// Which evaluation rows may be treated, one flag per row.
    #[must_use]
    pub fn with_available(mut self, available: impl Into<Arc<[bool]>>) -> Self {
        self.available = Some(available.into());
        self
    }

    /// The limits in the point-only kernel's form.
    pub(crate) const fn limits(&self) -> BinaryPolicyLimits {
        BinaryPolicyLimits {
            capacity: self.capacity,
            budget: self.budget,
            reference_capacity: self.reference_capacity,
            reference_budget: self.reference_budget,
        }
    }

    /// The model-slot family that names this declaration in the lowered program: a fixed
    /// prefix and the digest of every limit, so any change to a limit changes the program
    /// identity.
    pub(crate) fn program_family(&self) -> String {
        let mut writer = CanonicalWriter::new(IDENTITY_TAG);
        writer
            .opt_word(self.capacity.map(|limit| limit as u64))
            .opt_real(self.budget)
            .opt_word(self.reference_capacity.map(|limit| limit as u64))
            .opt_real(self.reference_budget);
        match self.available.as_deref() {
            Some(available) => {
                let flags: Vec<u8> = available.iter().map(|&flag| u8::from(flag)).collect();
                writer.word(1).word(available.len() as u64).bytes(&flags);
            }
            None => {
                writer.word(0);
            }
        }
        format!("declared_binary_policy_constraints;{}", writer.finish_hex())
    }
}

/// Refuse at build a study whose declared binary policy constraints would not be checked.
///
/// # Errors
///
/// `option_not_applicable` when the study declares constraints on anything but a binary
/// [`CausalQuery::PolicyValue`] read by the declared `randomized.ipw_policy_net` estimator.
pub(crate) fn admit(study: &crate::Study) -> Result<(), CausalError> {
    if study.declared.binary_policy_constraints.is_none() {
        return Ok(());
    }
    let binary =
        matches!(&study.query, CausalQuery::PolicyValue(query) if query.multi_action.is_none());
    if binary && study.declared.estimator == Some(EstimatorId::RandomizedIpwPolicyNet) {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "binary_policy_constraints are checked by the randomized.ipw_policy_net reading of a \
         binary PolicyValueQuery; declare that estimator on a binary policy, or state a \
         multi-action policy's availability, capacities and budgets in its \
         MultiActionPolicyInputs"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_limit_changes_the_program_family() {
        let base = BinaryPolicyConstraints::new().with_capacity(3).with_budget(1.5);
        let family = base.program_family();
        assert!(family.starts_with("declared_binary_policy_constraints;"));
        assert_eq!(family, base.clone().program_family());
        for changed in [
            base.clone().with_capacity(4),
            base.clone().with_budget(1.25),
            base.clone().with_reference_capacity(3),
            base.clone().with_reference_budget(1.5),
            base.clone().with_available([true, false]),
            base.clone().with_available([true, true]),
            BinaryPolicyConstraints::new().with_budget(1.5),
        ] {
            assert_ne!(changed.program_family(), family, "{changed:?}");
        }
        assert_ne!(
            base.clone().with_available([true, false]).program_family(),
            base.with_available([false, true]).program_family()
        );
    }

    #[test]
    fn the_kernel_limits_are_the_declared_ones() {
        let limits = BinaryPolicyConstraints::new()
            .with_capacity(2)
            .with_budget(0.5)
            .with_reference_capacity(1)
            .with_reference_budget(0.25)
            .limits();
        assert_eq!(
            limits,
            BinaryPolicyLimits {
                capacity: Some(2),
                budget: Some(0.5),
                reference_capacity: Some(1),
                reference_budget: Some(0.25),
            }
        );
    }
}
