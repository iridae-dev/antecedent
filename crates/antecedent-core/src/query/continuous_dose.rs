//! Caller-supplied conditional continuous-dose response design.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;
use std::sync::Arc;

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
}

impl ContinuousDoseResponseQuery {
    /// Validate frozen columns, grid, and density provenance.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcome == self.dose
            || self.outcome == self.dose_density
            || self.dose == self.dose_density
            || self.baseline_groups.is_empty()
            || self.baseline_groups.iter().any(|group| group.trim().is_empty())
            || self.target_doses.is_empty()
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
        Ok(())
    }
}
