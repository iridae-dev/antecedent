//! Covariate-distance matching with several nearest donors per unit.
//!
//! Each unit's counterfactual is the mean outcome of its `M` nearest donors on the
//! standardized covariates, with the linear within-arm bias adjustment of
//! [`super::DistanceMatching`] applied to that mean. The analytic standard error is the
//! Abadie–Imbens (2006) donor-reuse variance for `M` matches
//! ([`super::matching::matching_contrast`]): a donor used `Kⱼ` times carries weight
//! `Kⱼ / M`, and the outcome noise is estimated from the matched differences as
//! `Var(τ̂ᵢ) / (1 + 1/M)`. At `M = 1` the estimate equals [`super::DistanceMatching`] bit
//! for bit.
//!
//! The construction is its own: averaging `M` donors changes the point estimate, its bias and
//! its variance, so a calibration of single-neighbour matching says nothing about it. Its
//! identity is [`KNN_DISTANCE_MATCHING_CONSTRUCTION`] together with `M`
//! ([`KnnDistanceMatching::construction_id`]), and its interval is measured by its own
//! calibration designs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;

use super::distance::DistanceMatching;
use super::prepare::{PreparedPropensityProblem, PropensityEstimationWorkspace};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;

/// Identity of the `M`-nearest-neighbour covariate-distance matching construction; also the
/// stem of its provenance record (`provenance/estimate.knn_distance_matching.toml`).
pub const KNN_DISTANCE_MATCHING_CONSTRUCTION: &str = "estimate.knn_distance_matching";

/// Covariate-distance matching that averages the `neighbors` nearest donors of each unit.
///
/// Every setting of the single-neighbour estimator (caliper, overlap policy, sampling weights,
/// standard-error kind, ...) is [`Self::matching`]'s. With a caliper, a unit is matched only
/// when its `neighbors`-th nearest donor is within it, so every matched unit averages exactly
/// `neighbors` donors. Each donor arm must hold at least `neighbors` rows.
#[derive(Clone, Debug)]
pub struct KnnDistanceMatching {
    /// The covariate-distance configuration the neighbours are searched under.
    pub matching: DistanceMatching,
    /// Nearest donors averaged for each unit (at least 1).
    pub neighbors: usize,
}

impl KnnDistanceMatching {
    /// Average the `neighbors` nearest donors under the default [`DistanceMatching`].
    ///
    /// # Errors
    ///
    /// `neighbors == 0`.
    pub fn new(neighbors: usize) -> Result<Self, EstimationError> {
        if neighbors == 0 {
            return Err(EstimationError::unsupported(
                "knn matching needs at least one neighbour per unit",
            ));
        }
        Ok(Self { matching: DistanceMatching::new(), neighbors })
    }

    /// Identity of this construction: [`KNN_DISTANCE_MATCHING_CONSTRUCTION`] and the neighbour
    /// count, which changes the estimate and its variance.
    #[must_use]
    pub fn construction_id(&self) -> String {
        format!("{KNN_DISTANCE_MATCHING_CONSTRUCTION}/neighbors={}", self.neighbors)
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`DistanceMatching::prepare`].
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        self.matching.prepare(data, estimand, query)
    }

    /// Match each unit to its nearest donors and compute the matched effect.
    ///
    /// # Errors
    ///
    /// As [`DistanceMatching::fit`]; also `neighbors == 0` and a donor arm with fewer than
    /// `neighbors` rows.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        self.matching.fit_with_matches(problem, workspace, ctx, assumptions, self.neighbors)
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AssumptionSet, ExecutionContext};

    use antecedent_stats::MatchingDistance;

    use super::*;
    use crate::propensity::matching::{MatchSide, match_diffs};
    use crate::propensity::scale_fixture::{Design, Population, att_query};

    const NEIGHBORS: usize = 3;

    /// The covariates of `population` on the scale the estimator matches on: each column
    /// centred and divided by its population standard deviation, row-major.
    fn standardized(population: &Population) -> Vec<f64> {
        let rows = population.treatment.len();
        let dim = population.covariates.len();
        let mut features = vec![0.0; rows * dim];
        for (c, column) in population.covariates.iter().enumerate() {
            let mean = column.iter().sum::<f64>() / rows as f64;
            let variance = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / rows as f64;
            let sd = variance.sqrt().max(1e-12);
            for (r, v) in column.iter().enumerate() {
                features[r * dim + c] = (v - mean) / sd;
            }
        }
        features
    }

    /// Independent oracle for the ATT with `NEIGHBORS` matches: exhaustive neighbour search,
    /// per-arm OLS on the intercept and standardized covariates for the bias correction, and
    /// the Abadie–Imbens donor-reuse variance with `σ̂² = Var(τ̂ᵢ) / (1 + 1/M)`.
    struct Oracle {
        ate: f64,
        se: f64,
        matched: usize,
        /// Distance from each treated unit to its `M`-th nearest control.
        farthest: Vec<f64>,
    }

    fn oracle(population: &Population, caliper: Option<f64>) -> Oracle {
        let dim = population.covariates.len();
        let features = standardized(population);
        let treated: Vec<usize> =
            (0..population.treatment.len()).filter(|&r| population.treatment[r] > 0.5).collect();
        let controls: Vec<usize> =
            (0..population.treatment.len()).filter(|&r| population.treatment[r] <= 0.5).collect();
        let row = |r: usize| &features[r * dim..(r + 1) * dim];
        // OLS of the control outcome on [1, x].
        let p = dim + 1;
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];
        for &c in &controls {
            let design: Vec<f64> = std::iter::once(1.0).chain(row(c).iter().copied()).collect();
            for a in 0..p {
                xty[a] += design[a] * population.outcome[c];
                for b in 0..p {
                    xtx[a * p + b] += design[a] * design[b];
                }
            }
        }
        let beta = crate::util::solve_spd(&xtx, &xty, p).expect("the control design is full rank");
        let mean_function =
            |r: usize| beta[0] + row(r).iter().zip(&beta[1..]).map(|(x, b)| x * b).sum::<f64>();
        let mut diffs = Vec::new();
        let mut uses = vec![0usize; population.treatment.len()];
        let mut farthest = Vec::new();
        for &t in &treated {
            let mut by_distance: Vec<(f64, usize)> = controls
                .iter()
                .map(|&c| {
                    let d2: f64 = row(t).iter().zip(row(c)).map(|(a, b)| (a - b).powi(2)).sum();
                    (d2.sqrt(), c)
                })
                .collect();
            by_distance.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let nearest = &by_distance[..NEIGHBORS];
            farthest.push(nearest[NEIGHBORS - 1].0);
            if caliper.is_some_and(|limit| nearest[NEIGHBORS - 1].0 > limit) {
                continue;
            }
            let donor_outcome =
                nearest.iter().map(|&(_, c)| population.outcome[c]).sum::<f64>() / NEIGHBORS as f64;
            let donor_mean =
                nearest.iter().map(|&(_, c)| mean_function(c)).sum::<f64>() / NEIGHBORS as f64;
            diffs.push(population.outcome[t] - donor_outcome - (mean_function(t) - donor_mean));
            for &(_, c) in nearest {
                uses[c] += 1;
            }
        }
        let n = diffs.len() as f64;
        let ate = diffs.iter().sum::<f64>() / n;
        let variance = diffs.iter().map(|d| (d - ate).powi(2)).sum::<f64>() / (n - 1.0);
        let sigma2 = variance / (1.0 + 1.0 / NEIGHBORS as f64);
        let reuse: f64 = uses.iter().map(|&k| (k as f64 / NEIGHBORS as f64).powi(2)).sum();
        let se = (sigma2 * (n + reuse) / (n * n)).sqrt();
        Oracle { ate, se, matched: diffs.len(), farthest }
    }

    fn population() -> Population {
        Design {
            rows: 700,
            treated: 160,
            covariates: 3,
            seed: 0x4B4E_4E31,
            lattice: None,
            duplicated: false,
        }
        .population()
    }

    fn fit(
        knn: &KnnDistanceMatching,
        population: &Population,
    ) -> Result<EffectEstimate, EstimationError> {
        let (data, estimand) = population.table();
        let prep = knn.prepare(&data, &estimand, &att_query())?;
        knn.fit(
            &prep,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(11),
            AssumptionSet::new(),
        )
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn three_neighbour_estimate_and_standard_error_equal_the_independent_oracle() {
        let population = population();
        let knn = KnnDistanceMatching::new(NEIGHBORS).unwrap();
        let effect = fit(&knn, &population).expect("fits");
        let expected = oracle(&population, None);
        assert!(close(effect.ate, expected.ate), "ate {} vs {}", effect.ate, expected.ate);
        assert!(
            close(effect.se_analytic(), expected.se),
            "se {} vs {}",
            effect.se_analytic(),
            expected.se
        );
        assert_eq!(effect.provenance.n_obs, Some(expected.matched as u64));
    }

    #[test]
    fn a_caliper_keeps_only_units_whose_farthest_neighbour_is_within_it() {
        let population = population();
        let mut farthest = oracle(&population, None).farthest;
        farthest.sort_by(f64::total_cmp);
        // Between two adjacent distances, so no unit sits on the boundary.
        let middle = farthest.len() / 2;
        let limit = 0.5 * (farthest[middle - 1] + farthest[middle]);
        let mut knn = KnnDistanceMatching::new(NEIGHBORS).unwrap();
        knn.matching = knn.matching.with_caliper(limit);
        let effect = fit(&knn, &population).expect("fits");
        let expected = oracle(&population, Some(limit));
        assert_eq!(expected.matched, middle);
        assert_eq!(effect.provenance.n_obs, Some(middle as u64));
        assert!(close(effect.ate, expected.ate));
        assert!(close(effect.se_analytic(), expected.se));
    }

    #[test]
    fn one_neighbour_is_distance_matching_bit_for_bit() {
        let population = population();
        let knn = fit(&KnnDistanceMatching::new(1).unwrap(), &population).expect("fits");
        let plain = {
            let (data, estimand) = population.table();
            let est = DistanceMatching::new();
            let prep = est.prepare(&data, &estimand, &att_query()).unwrap();
            est.fit(
                &prep,
                &mut PropensityEstimationWorkspace::default(),
                &ExecutionContext::for_tests(11),
                AssumptionSet::new(),
            )
            .unwrap()
        };
        assert_eq!(knn.ate.to_bits(), plain.ate.to_bits());
        assert_eq!(knn.se_analytic().to_bits(), plain.se_analytic().to_bits());
        assert_eq!(knn.provenance.n_obs, plain.provenance.n_obs);
    }

    #[test]
    fn the_neighbour_count_changes_the_estimate_and_the_construction_identity() {
        let population = population();
        let one = KnnDistanceMatching::new(1).unwrap();
        let three = KnnDistanceMatching::new(3).unwrap();
        assert_ne!(one.construction_id(), three.construction_id());
        assert!(three.construction_id().starts_with(KNN_DISTANCE_MATCHING_CONSTRUCTION));
        let a = fit(&one, &population).unwrap();
        let b = fit(&three, &population).unwrap();
        assert_ne!(a.ate.to_bits(), b.ate.to_bits());
        assert_ne!(a.se_analytic().to_bits(), b.se_analytic().to_bits());
    }

    #[test]
    fn zero_neighbours_and_a_donor_arm_smaller_than_the_count_are_refused() {
        assert!(KnnDistanceMatching::new(0).is_err());
        let mut knn = KnnDistanceMatching::new(NEIGHBORS).unwrap();
        knn.neighbors = 0;
        let message = fit(&knn, &population()).expect_err("zero neighbours").to_string();
        assert!(message.contains("at least one match"), "{message}");
        // Two donors cannot supply three matches to a query.
        let refused = match match_diffs(
            MatchSide { features: &[0.0, 1.0], outcome: &[0.0, 1.0] },
            MatchSide { features: &[0.5], outcome: &[0.0] },
            1,
            MatchingDistance::Euclidean,
            NEIGHBORS,
            None,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(11),
        ) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("two donors matched three neighbours"),
        };
        assert!(refused.contains("needs at least 3 donor rows"), "{refused}");
    }
}
