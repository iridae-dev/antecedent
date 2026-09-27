//! Caller-supplied conditional continuous-dose response design.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

/// Frozen group-to-dose policy and reference rules for a local intervention.
/// Each group named by the query must appear exactly once in both rules.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedGroupDosePolicy {
    /// Prespecified policy target dose for each baseline group.
    pub policy_doses: Arc<[(Arc<str>, f64)]>,
    /// Prespecified reference target dose for each baseline group.
    pub reference_doses: Arc<[(Arc<str>, f64)]>,
}

impl FixedGroupDosePolicy {
    /// Check that both rules cover the frozen pre-treatment groups exactly.
    pub fn validate(&self, baseline_groups: &[Arc<str>]) -> Result<(), QueryError> {
        let observed = baseline_groups.iter().map(|group| group.as_ref())
            .collect::<std::collections::BTreeSet<_>>();
        if observed.is_empty() { return Err(QueryError::InvalidPolicyValue(
            "fixed dose rules require observed baseline groups".into())); }
        for rule in [&self.policy_doses, &self.reference_doses] {
            let mut seen = std::collections::BTreeSet::new();
            if rule.len() != observed.len() || rule.iter().any(|(group, target)|
                !target.is_finite() || !observed.contains(group.as_ref())
                    || !seen.insert(group.as_ref()))
            {
                return Err(QueryError::InvalidPolicyValue(
                    "fixed dose rules must name every observed baseline group exactly once with finite targets".into()));
            }
        }
        Ok(())
    }
}

/// A fixed local response grid under group-level exchangeability.
#[derive(Clone, Debug, PartialEq)]
pub struct ContinuousDoseResponseQuery {
    /// Outcome column.
    pub outcome: VariableId,
    /// Observed continuous dose column.
    pub dose: VariableId,
    /// Caller-supplied conditional dose density at observed doses.
    pub dose_density: VariableId,
    /// Pre-treatment group labels aligned to table rows.
    pub baseline_groups: Arc<[Arc<str>]>,
    /// Prespecified dose grid evaluated in every baseline group.
    pub target_doses: Arc<[f64]>,
    /// Triangular-kernel bandwidth.
    pub bandwidth: f64,
    /// Number of local rows required in every group-target cell.
    pub min_local_support: usize,
    /// How the caller obtained the density values.
    pub density_provenance: Arc<str>,
    /// Optional fixed policy and reference group-dose rules evaluated on these rows.
    pub fixed_policy: Option<FixedGroupDosePolicy>,
}

impl ContinuousDoseResponseQuery {
    /// Validate frozen columns, grid, and density provenance.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcome == self.dose
            || self.outcome == self.dose_density
            || self.dose == self.dose_density
            || self.baseline_groups.is_empty()
            || self.baseline_groups.iter().any(|group| group.trim().is_empty())
            || (self.target_doses.is_empty() && self.fixed_policy.is_none())
            || self.target_doses.iter().any(|dose| !dose.is_finite())
            || !self.bandwidth.is_finite()
            || self.bandwidth <= 0.0
            || self.min_local_support < 2
            || !matches!(self.density_provenance.as_ref(), "known" | "externally_estimated")
        {
            return Err(QueryError::InvalidPolicyValue(
                "continuous-dose response requires distinct columns, aligned non-empty baseline groups, finite targets, positive bandwidth, local support of at least two, and known or externally_estimated density provenance".into(),
            ));
        }
        if let Some(policy) = &self.fixed_policy {
            policy.validate(&self.baseline_groups)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    fn fixed_group_rules_require_exact_finite_group_coverage() {
        let groups: Arc<[Arc<str>]> = Arc::from([Arc::from("a"), Arc::from("b")]);
        let policy = FixedGroupDosePolicy {
            policy_doses: Arc::from([(Arc::from("a"), 0.3), (Arc::from("b"), 0.7)]),
            reference_doses: Arc::from([(Arc::from("a"), 0.2), (Arc::from("b"), 0.4)]),
        };
        assert!(policy.validate(&groups).is_ok());
        assert!(FixedGroupDosePolicy {
            reference_doses: Arc::from([(Arc::from("a"), 0.2), (Arc::from("a"), 0.4)]),
            ..policy.clone()
        }.validate(&groups).is_err());
        assert!(FixedGroupDosePolicy {
            policy_doses: Arc::from([(Arc::from("a"), f64::NAN), (Arc::from("b"), 0.7)]),
            ..policy
        }.validate(&groups).is_err());
    }
}
