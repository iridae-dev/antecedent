//! Randomized interference queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::VariableId;

use super::QueryError;

/// Known random assignment design.
#[derive(Clone, Debug, PartialEq)]
pub enum AssignmentDesign {
    /// Independent Bernoulli assignment, scalar or unit-specific probabilities.
    Bernoulli {
        /// One probability for all units or one per unit.
        probabilities: Arc<[f64]>,
    },
    /// Exactly `treated` units selected uniformly without replacement.
    CompleteRandomization {
        /// Number assigned to treatment.
        treated: usize,
    },
    /// Exactly `treated_clusters` clusters selected uniformly.
    ClusterRandomization {
        /// Cluster id per unit.
        clusters: Arc<[u32]>,
        /// Number of treated clusters.
        treated_clusters: usize,
    },
}

/// Built-in map from the global assignment vector to unit exposure.
#[derive(Clone, Debug, PartialEq)]
pub enum ExposureMapping {
    /// Unit's own binary treatment only.
    OwnTreatment,
    /// Own treatment plus count of treated incoming neighbors.
    NeighborCount,
    /// Own treatment plus fraction of treated incoming neighbors.
    NeighborFraction,
    /// Own treatment plus weighted mean neighbor treatment.
    WeightedNeighborExposure,
    /// Opaque custom mapping id resolved by a caller registry.
    Custom(Arc<str>),
}

/// Tolerance for treating two [`ExposureLevel`] values as the same level.
///
/// Exposure levels are frequently derived from floating-point neighbor fractions or weighted
/// means, so exact `==` is too brittle: a `from`/`to` pair that differs only by rounding error
/// would pass [`InterferenceQuery::validate`] as "distinct", yet
/// `antecedent_stats::interference` matches units to levels with a tolerance and would then
/// select the *same* unit set for both, silently producing an exact-zero contrast with no
/// warning. Validation therefore uses this same tolerance.
///
/// This constant lives here, in `antecedent-core`, because `antecedent-stats` depends on
/// `antecedent-core` (not the reverse), so the core crate cannot import a stats-crate constant.
/// It is not currently re-exported through `query/mod.rs`/`lib.rs` (out of scope for this
/// change), so `antecedent-stats` keeps its own same-named, same-valued constant with a doc
/// comment pointing back here; if the export chain opens up, that duplicate should be replaced
/// with an import of this one.
pub const EXPOSURE_LEVEL_TOLERANCE: f64 = 1e-12;

/// One exposure category/level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureLevel {
    /// Unit's own treatment.
    pub own: f64,
    /// Neighborhood summary; zero for [`ExposureMapping::OwnTreatment`].
    pub neighbors: f64,
}

/// Randomization-based interference estimand.
#[derive(Clone, Debug, PartialEq)]
pub enum InterferenceFunctional {
    /// Mean potential-outcome contrast between two exposure levels.
    ExposureContrast {
        /// Outcome variable.
        outcome: VariableId,
        /// Baseline exposure.
        from: ExposureLevel,
        /// Active exposure.
        to: ExposureLevel,
    },
}

/// Assignment design + exposure mapping + estimand.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferenceQuery {
    /// Known random assignment design.
    pub assignment: AssignmentDesign,
    /// Exposure mapping.
    pub exposure: ExposureMapping,
    /// Requested contrast.
    pub functional: InterferenceFunctional,
    /// Monte Carlo assignments used when exposure probabilities are not analytic.
    pub probability_draws: u32,
}

impl InterferenceQuery {
    /// Construct with 10,000 exposure-probability simulations when required.
    #[must_use]
    pub fn new(
        assignment: AssignmentDesign,
        exposure: ExposureMapping,
        functional: InterferenceFunctional,
    ) -> Self {
        Self { assignment, exposure, functional, probability_draws: 10_000 }
    }

    /// Validate design probabilities, exposure levels, and simulation budget.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidInterference`] when any design or exposure value is invalid.
    pub fn validate(&self) -> Result<(), QueryError> {
        match &self.assignment {
            AssignmentDesign::Bernoulli { probabilities }
                if probabilities.is_empty()
                    || probabilities.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) =>
            {
                return Err(QueryError::InvalidInterference(
                    "Bernoulli probabilities must be finite and strictly between zero and one"
                        .into(),
                ));
            }
            AssignmentDesign::CompleteRandomization { treated } if *treated == 0 => {
                return Err(QueryError::InvalidInterference(
                    "complete randomization requires at least one treated unit".into(),
                ));
            }
            AssignmentDesign::ClusterRandomization { clusters, treated_clusters }
                if clusters.is_empty() || *treated_clusters == 0 =>
            {
                return Err(QueryError::InvalidInterference(
                    "cluster randomization requires clusters and at least one treated cluster"
                        .into(),
                ));
            }
            _ => {}
        }
        let InterferenceFunctional::ExposureContrast { from, to, .. } = &self.functional;
        if [from.own, from.neighbors, to.own, to.neighbors].iter().any(|v| !v.is_finite())
            || exposure_levels_match(*from, *to)
            || self.probability_draws == 0
        {
            return Err(QueryError::InvalidInterference(
                "exposure levels must be finite/distinct and probability_draws must be positive"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// True when `a` and `b` are within [`EXPOSURE_LEVEL_TOLERANCE`] on both components.
fn exposure_levels_match(a: ExposureLevel, b: ExposureLevel) -> bool {
    (a.own - b.own).abs() <= EXPOSURE_LEVEL_TOLERANCE
        && (a.neighbors - b.neighbors).abs() <= EXPOSURE_LEVEL_TOLERANCE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VariableId;

    fn query_with_levels(from: ExposureLevel, to: ExposureLevel) -> InterferenceQuery {
        InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::OwnTreatment,
            InterferenceFunctional::ExposureContrast { outcome: VariableId::from_raw(0), from, to },
        )
    }

    #[test]
    fn distinct_exposure_levels_validate() {
        let query = query_with_levels(
            ExposureLevel { own: 0.0, neighbors: 0.0 },
            ExposureLevel { own: 1.0, neighbors: 0.0 },
        );
        assert!(query.validate().is_ok());
    }

    #[test]
    fn exposure_levels_within_tolerance_are_rejected_at_validation() {
        // `to` differs from `from` by 1e-15, well inside EXPOSURE_LEVEL_TOLERANCE. Exact `==`
        // would call this "distinct" and let it through, after which
        // `antecedent_stats::interference::same_exposure` (tolerance 1e-12) would match both
        // levels to the same unit set and silently report an exact-zero contrast.
        let from = ExposureLevel { own: 0.5, neighbors: 0.25 };
        let to = ExposureLevel { own: 0.5 + 1e-15, neighbors: 0.25 };
        assert_ne!(from, to, "test fixture must use exact f64 PartialEq inequality");
        let query = query_with_levels(from, to);
        let err = query.validate().unwrap_err();
        assert!(matches!(err, QueryError::InvalidInterference(_)));
    }
}
