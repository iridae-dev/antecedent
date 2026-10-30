//! Replay inputs and exact license scope of the randomized-arm IPW pseudo-posterior.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::graphless_support::{GraphlessAssignmentSupport, GraphlessSupportKey, graphless_admits};
use crate::{IoError, RandomizationDesignWire, RandomizedEffectQueryWire, RandomizedEstimandWire};
use serde::{Deserialize, Serialize};

/// Estimator of the licensed conditional pseudo-posterior.
pub const ESTIMATOR: &str = "bayesian.randomized_superpopulation";
/// The working likelihood, distinct from a joint model of outcomes and missingness.
pub const METHOD: &str = "gaussian_ipow_modular_posterior";
/// A posterior-quantile claim; repeated-sampling coverage remains unmeasured.
pub const CLAIM: &str = "conditional_pseudo_posterior_pointwise_95_quantiles";
/// Target marker retained with the posterior row.
pub const TARGET: &str = "ipow_superpopulation_conditional_on_fitted_observation_model";

/// Source observations and declared prior needed to replay the whole IPW construction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RandomizedIpowWire {
    /// Exact assignment design (`bernoulli`, `complete`, `multi_arm`, `factorial_2x2`).
    pub design: String,
    /// Assigned arm of every original unit.
    pub assignment: Vec<usize>,
    /// Known arm probabilities in row order.
    pub probabilities: Vec<Vec<f64>>,
    /// Recorded outcomes; `None` means missing, never an imputed zero.
    pub outcomes: Vec<Option<f64>>,
    /// Baseline observation-model variable ids, in column order.
    pub covariate_ids: Vec<u32>,
    /// Complete baseline covariates, column-major.
    pub covariates: Vec<f64>,
    /// Contrast weights in the posterior row's coordinate order.
    pub contrasts: Vec<Vec<f64>>,
    /// Normal-Inverse-Gamma prior: mean, pseudo-units, variance shape, variance scale.
    pub prior: [f64; 4],
}

impl RandomizedIpowWire {
    /// Canonical arm-prior declaration bound by the inference identity.
    #[must_use]
    pub fn prior_identity(&self) -> String {
        format!(
            "arm_outcome;gaussian;mean={:016x};pseudo_units={:016x};shape={:016x};scale={:016x}",
            self.prior[0].to_bits(),
            self.prior[1].to_bits(),
            self.prior[2].to_bits(),
            self.prior[3].to_bits()
        )
    }
    /// Exact graphless license coordinate.
    #[must_use]
    pub fn key(&self) -> GraphlessSupportKey<'_> {
        GraphlessSupportKey {
            family: "randomized_effect",
            design: &self.design,
            method: METHOD,
            inference_claim: CLAIM,
        }
    }

    /// Support observed in the retained source table, after validating its shape.
    #[must_use]
    pub fn licensed(&self, intervals: usize) -> bool {
        let arms = self.probabilities.first().map_or(0, Vec::len);
        if arms != 2
            || self.contrasts != [vec![-1.0, 1.0]]
            || intervals != 1
            || self.assignment.len() != self.outcomes.len()
            || self.assignment.iter().any(|arm| *arm >= arms)
        {
            return false;
        }
        let counts: Vec<usize> =
            (0..arms).map(|arm| self.assignment.iter().filter(|a| **a == arm).count()).collect();
        let minimum = counts.into_iter().min().unwrap_or(0);
        graphless_admits(
            self.key(),
            GraphlessAssignmentSupport {
                assignment_unit: "unit",
                treated: minimum,
                control: minimum,
                rows: self.outcomes.len(),
                min_action_rows: minimum,
                interval_95_published: intervals == self.contrasts.len() && intervals > 0,
                reported_intervals: intervals,
                all_reported_intervals: intervals == self.contrasts.len(),
                ..Default::default()
            },
        )
    }

    /// Validate source shape, proper prior, response support, and the exact query facets.
    ///
    /// # Errors
    /// Any changed query facet, missing source row, invalid number, or unsupported design.
    pub fn validate_query(
        &self,
        query: &RandomizedEffectQueryWire,
        coordinates: &[String],
        variables: usize,
    ) -> Result<(), IoError> {
        let fail = || {
            IoError::Convert(
                "randomized IPW pseudo-posterior source does not match its licensed query".into(),
            )
        };
        let n = self.outcomes.len();
        if n == 0
            || self.assignment.len() != n
            || self.probabilities.len() != n
            || self.covariates.len() != n.saturating_mul(self.covariate_ids.len())
            || self
                .covariates
                .iter()
                .chain(self.outcomes.iter().flatten())
                .chain(self.prior.iter())
                .any(|x| !x.is_finite())
            || self.prior[1..].iter().any(|x| *x <= 0.0)
            || self.outcomes.iter().all(Option::is_some)
            || self.covariate_ids.iter().collect::<std::collections::BTreeSet<_>>().len()
                != self.covariate_ids.len()
            || self.covariate_ids.iter().any(|id| *id as usize >= variables || *id == query.outcome)
            || query.estimand != RandomizedEstimandWire::Itt
            || query.received_treatment.is_some()
            || query.fixed_cuped.is_some()
            || !query.ancova_covariates.is_empty()
            || query.exact_randomization_test
            || query.realized_assignment.len() != n
            || query.assignment_units.len() != n
            || query.outcome_units.len() != n
            || query.assignment_units.iter().collect::<std::collections::BTreeSet<_>>().len() != n
            || query.outcome_units.iter().collect::<std::collections::BTreeSet<_>>().len() != n
        {
            return Err(fail());
        }
        let (design, assignment, probabilities, contrasts, names) = match &query.design {
            RandomizationDesignWire::Bernoulli | RandomizationDesignWire::Complete { .. } => (
                if matches!(query.design, RandomizationDesignWire::Bernoulli) {
                    "bernoulli"
                } else {
                    "complete"
                },
                query.realized_assignment.iter().map(|v| usize::from(*v)).collect::<Vec<_>>(),
                query
                    .assignment_probabilities
                    .iter()
                    .map(|p| vec![1.0 - p, *p])
                    .collect::<Vec<_>>(),
                vec![vec![-1.0, 1.0]],
                vec!["itt".to_owned()],
            ),
            _ => return Err(fail()),
        };
        if self.design != design
            || self.assignment != assignment
            || self.probabilities != probabilities
            || self.contrasts != contrasts
            || coordinates != names
        {
            return Err(fail());
        }
        if let RandomizationDesignWire::Complete { treated_units } = query.design {
            if treated_units as usize != self.assignment.iter().filter(|arm| **arm == 1).count()
                || query
                    .assignment_probabilities
                    .iter()
                    .any(|p| (*p - treated_units as f64 / n as f64).abs() > 1e-10)
            {
                return Err(fail());
            }
        }
        let arms = self.probabilities.first().map_or(0, Vec::len);
        if arms != 2
            || self.probabilities.iter().any(|p| {
                p.len() != arms
                    || p.iter().any(|v| !v.is_finite() || *v <= 0.0 || *v >= 1.0)
                    || (p.iter().sum::<f64>() - 1.0).abs() > 1e-10
                    || p != &self.probabilities[0]
            })
            || self.assignment.iter().any(|arm| *arm >= arms)
            || (0..arms).any(|arm| {
                !self.assignment.iter().zip(&self.outcomes).any(|(a, y)| *a == arm && y.is_some())
            })
            || !self.licensed(coordinates.len())
        {
            return Err(fail());
        }
        Ok(())
    }
}
