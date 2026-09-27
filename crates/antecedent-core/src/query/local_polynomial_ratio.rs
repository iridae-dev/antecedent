//! Declared local-polynomial fuzzy discontinuity and regression kink design.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use crate::ids::VariableId;

/// A fixed-cutoff, fixed-bandwidth local ratio with observed treatment uptake.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalPolynomialRatioQuery {
    /// Continuous outcome variable.
    pub outcome: VariableId,
    /// Observed treatment receipt or dose.
    pub treatment: VariableId,
    /// Running variable that determines threshold exposure.
    pub running: VariableId,
    /// Prespecified threshold.
    pub cutoff: f64,
    /// Prespecified local fitting window radius.
    pub bandwidth: f64,
    /// Estimate a slope kink instead of a level jump.
    pub kink: bool,
}

impl LocalPolynomialRatioQuery {
    /// Validate the declared local design before compiling a study.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcome == self.treatment || self.outcome == self.running || self.treatment == self.running {
            return Err(QueryError::InvalidLocalPolynomialRatio("outcome, treatment, and running variable must be distinct".into()));
        }
        if !self.cutoff.is_finite() || !self.bandwidth.is_finite() || self.bandwidth <= 0.0 {
            return Err(QueryError::InvalidLocalPolynomialRatio("cutoff must be finite and bandwidth positive".into()));
        }
        Ok(())
    }
}
