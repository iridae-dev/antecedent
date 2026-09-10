//! Outcome functionals on licensed effect and response queries.
//!
//! Identification is unchanged: the same adjustment set is applied to a
//! transform of `Y` (`Y` itself, or `1{Y > c}`). Quantile inversion is not
//! licensed here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::QueryError;
use super::attribution::OrderedFloatBits;

/// Functional of the interventional outcome law.
///
/// [`Self::Mean`] is the licensed default. Exceedance functionals replace `Y`
/// with an indicator of the upper tail; the adjustment set does not change.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum OutcomeFunctional {
    /// `E[Y(a)]` (default).
    Mean,
    /// `P(Y(a) > c)` at a single threshold.
    Exceedance(OrderedFloatBits),
    /// `P(Y(a) > c)` on a strictly increasing threshold grid.
    ExceedanceGrid(Arc<[OrderedFloatBits]>),
}

impl Default for OutcomeFunctional {
    fn default() -> Self {
        Self::Mean
    }
}

impl OutcomeFunctional {
    /// Single-threshold exceedance.
    #[must_use]
    pub fn exceedance(threshold: f64) -> Self {
        Self::Exceedance(OrderedFloatBits::from_f64(threshold))
    }

    /// Exceedance on an explicit grid.
    #[must_use]
    pub fn exceedance_grid(thresholds: impl Into<Arc<[f64]>>) -> Self {
        let bits: Arc<[OrderedFloatBits]> = thresholds
            .into()
            .iter()
            .copied()
            .map(OrderedFloatBits::from_f64)
            .collect::<Vec<_>>()
            .into();
        Self::ExceedanceGrid(bits)
    }

    /// Thresholds in evaluation order (`None` for the mean functional).
    #[must_use]
    pub fn thresholds(&self) -> Option<Vec<f64>> {
        match self {
            Self::Mean => None,
            Self::Exceedance(c) => Some(vec![c.to_f64()]),
            Self::ExceedanceGrid(grid) => Some(grid.iter().map(|c| c.to_f64()).collect()),
        }
    }

    /// Whether this functional is the default mean.
    #[must_use]
    pub const fn is_mean(&self) -> bool {
        matches!(self, Self::Mean)
    }

    /// Validate finite, strictly increasing thresholds.
    ///
    /// # Errors
    ///
    /// Non-finite or non-increasing exceedance grids.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::Mean => Ok(()),
            Self::Exceedance(c) => {
                if c.to_f64().is_finite() {
                    Ok(())
                } else {
                    Err(QueryError::InvalidOutcomeFunctional)
                }
            }
            Self::ExceedanceGrid(grid) => {
                if grid.is_empty() {
                    return Err(QueryError::InvalidOutcomeFunctional);
                }
                let mut prev = f64::NEG_INFINITY;
                for c in grid.iter() {
                    let v = c.to_f64();
                    if !v.is_finite() || v <= prev {
                        return Err(QueryError::InvalidOutcomeFunctional);
                    }
                    prev = v;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_is_default() {
        assert_eq!(OutcomeFunctional::default(), OutcomeFunctional::Mean);
        assert!(OutcomeFunctional::Mean.is_mean());
    }

    #[test]
    fn grid_must_increase() {
        let bad = OutcomeFunctional::exceedance_grid([0.2, 0.1]);
        assert_eq!(bad.validate(), Err(QueryError::InvalidOutcomeFunctional));
        let ok = OutcomeFunctional::exceedance_grid([0.1, 0.2, 0.9]);
        assert!(ok.validate().is_ok());
    }
}
