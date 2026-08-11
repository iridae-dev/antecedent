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
            || from == to
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
