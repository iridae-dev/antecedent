//! Design-based estimation under network interference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExposureLevel, InterferenceFunctional, InterferenceQuery};
use antecedent_data::{NetworkData, TableView};
use antecedent_prob::{LaplaceWorkspace, sample_gaussian_mvn};
use antecedent_stats::{
    ExposureProbabilityMethod, RandomizationContrast, exposure_probabilities, exposures,
    randomization_contrast, randomization_mean,
};
use std::collections::BTreeMap;

use crate::EstimationError;

/// Complete design-based estimate for an exposure contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct InterferenceEstimate {
    /// HT/Hájek point estimates and conservative variance.
    pub contrast: RandomizationContrast,
    /// Probability computation for the baseline exposure.
    pub from_probability_method: ExposureProbabilityMethod,
    /// Probability computation for the active exposure.
    pub to_probability_method: ExposureProbabilityMethod,
    /// Smallest exposure probability across either requested level.
    pub minimum_exposure_probability: f64,
}

/// Conservative pointwise total-effect interval for independently randomized
/// clusters under a complete cluster allocation.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterTotalInterval {
    /// Two-sided 95% interval for the Horvitz--Thompson total contrast.
    pub bounds: [f64; 2],
    /// Cluster-level Neyman standard error.
    pub standard_error: f64,
    /// Welch--Satterthwaite degrees of freedom.
    pub degrees_of_freedom: f64,
    /// Independent clusters observed in each assignment arm.
    pub control_clusters: usize,
    /// Independently randomized treated clusters.
    pub treated_clusters: usize,
}

/// Model based posterior for a finite network exposure contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianInterferenceEstimate {
    /// Posterior draws of the finite-network contrast (to minus from).
    pub contrast_draws: Vec<f64>,
    /// Posterior mean contrast.
    pub mean: f64,
}

/// Fit the declared Gaussian additive potential-outcome model on a fixed network.
///
/// Outcomes follow `Y_i(z,g) = alpha + beta*z + gamma*g + epsilon_i`, where the
/// same unit disturbance `epsilon_i` is shared across that unit's exposures. The
/// finite-network target is the mean potential-outcome contrast over these units.
/// This is model based and does not inherit the design-based HT guarantee. The
/// prior is independent `Normal(0, prior_sd²)` on all three coefficients, and the
/// residual variance is fixed to one. A full-rank observed design and empirical
/// support for both requested exposure levels are required.
pub fn estimate_interference_bayesian(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
    n_draws: usize,
    seed: u64,
    prior_sd: f64,
) -> Result<BayesianInterferenceEstimate, EstimationError> {
    query.validate()?;
    if n_draws < 2 || !prior_sd.is_finite() || prior_sd <= 0.0 {
        return Err(EstimationError::data_msg(
            "Bayesian interference requires at least two draws and a positive finite prior SD",
        ));
    }
    if assignment.len() != data.units().row_count() {
        return Err(EstimationError::data_msg("assignment/network row count mismatch"));
    }
    if !matches!(query.assignment, antecedent_core::AssignmentDesign::Bernoulli { .. })
        || query.exposure != antecedent_core::ExposureMapping::NeighborCount
    {
        return Err(EstimationError::unsupported(
            "Bayesian interference is licensed only for Bernoulli assignment with NeighborCount exposure",
        ));
    }
    let InterferenceFunctional::ExposureContrast { outcome, from, to } = query.functional;
    let n = assignment.len();
    let outcomes = data.units().float64_values(outcome)?;
    let incoming = (0..n)
        .map(|unit| {
            data.incoming(unit)
                .map(|edges| edges.iter().map(|e| (e.from as usize, e.weight)).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let observed = exposures(assignment, &incoming, &query.exposure)?;
    let level_supported = |level: ExposureLevel| {
        observed.iter().any(|e| {
            (e.own - level.own).abs() < 1e-12 && (e.neighbors - level.neighbors).abs() < 1e-12
        })
    };
    if !level_supported(from) || !level_supported(to) {
        return Err(EstimationError::unsupported(
            "Bayesian interference refuses an exposure level absent from the realized assignment",
        ));
    }
    // X = [1, own treatment, treated-neighbor count], row-major for local algebra.
    let x: Vec<[f64; 3]> = observed.iter().map(|e| [1.0, e.own, e.neighbors]).collect();
    let mut precision = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    let prior_precision = 1.0 / prior_sd.powi(2);
    for (j, row) in precision.iter_mut().enumerate() {
        row[j] = prior_precision;
    }
    for i in 0..n {
        for (j, (rhs_value, row)) in rhs.iter_mut().zip(precision.iter_mut()).enumerate() {
            *rhs_value += x[i][j] * outcomes[i];
            for (k, value) in row.iter_mut().enumerate() {
                *value += x[i][j] * x[i][k];
            }
        }
    }
    let mut observed_gram = precision;
    for (j, row) in observed_gram.iter_mut().enumerate() {
        row[j] -= prior_precision;
    }
    if inverse_3(observed_gram).is_none() {
        return Err(EstimationError::unsupported(
            "observed interference exposure design is rank deficient",
        ));
    }
    let covariance = inverse_3(precision)
        .ok_or_else(|| EstimationError::unsupported("posterior precision matrix is singular"))?;
    let mut mean = [0.0; 3];
    for (j, value) in mean.iter_mut().enumerate() {
        for (k, covariance_value) in covariance[j].iter().enumerate() {
            *value += covariance_value * rhs[k];
        }
    }
    let from_x = [0.0, from.own, from.neighbors];
    let to_x = [0.0, to.own, to.neighbors];
    let contrast = [0, 1, 2].map(|j| to_x[j] - from_x[j]);
    let effect_mean = (0..3).map(|j| contrast[j] * mean[j]).sum::<f64>();
    let mut effect_variance = 0.0;
    for j in 0..3 {
        for k in 0..3 {
            effect_variance += contrast[j] * covariance[j][k] * contrast[k];
        }
    }
    if !effect_variance.is_finite() || effect_variance <= 0.0 {
        return Err(EstimationError::unsupported(
            "posterior contrast variance is not positive and finite",
        ));
    }
    let draws = sample_gaussian_mvn(
        &[effect_mean],
        &[effect_variance],
        n_draws,
        seed,
        &mut LaplaceWorkspace::default(),
    )
    .map_err(|e| EstimationError::stats_msg(e.to_string()))?;
    let contrast_draws = (0..n_draws).map(|i| draws[i]).collect::<Vec<_>>();
    let mean = contrast_draws.iter().sum::<f64>() / n_draws as f64;
    Ok(BayesianInterferenceEstimate { contrast_draws, mean })
}

fn inverse_3(mut a: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let mut inv = [[0.0; 3]; 3];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..3 {
        let pivot =
            (col..3).max_by(|&lhs, &rhs| a[lhs][col].abs().total_cmp(&a[rhs][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let d = a[col][col];
        for j in 0..3 {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for row in 0..3 {
            if row == col {
                continue;
            }
            let f = a[row][col];
            for j in 0..3 {
                a[row][j] -= f * a[col][j];
                inv[row][j] -= f * inv[col][j];
            }
        }
    }
    Some(inv)
}

/// Estimate a randomized exposure contrast on a fixed network.
///
/// The caller supplies the realized binary assignment in unit-row order. Built-in exposure
/// mappings are evaluated from incoming network edges; custom mappings are refused.
pub fn estimate_interference(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
    seed: u64,
) -> Result<InterferenceEstimate, EstimationError> {
    query.validate()?;
    if assignment.len() != data.units().row_count() {
        return Err(EstimationError::data_msg("assignment/network row count mismatch"));
    }
    let InterferenceFunctional::ExposureContrast { outcome, from, to } = query.functional;
    let outcomes = data.units().float64_values(outcome)?;
    let incoming = (0..assignment.len())
        .map(|unit| {
            data.incoming(unit).map(|edges| {
                edges.iter().map(|edge| (edge.from as usize, edge.weight)).collect::<Vec<_>>()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let observed = exposures(assignment, &incoming, &query.exposure)?;
    let from_p = exposure_probabilities(
        &query.assignment,
        &incoming,
        &query.exposure,
        from,
        query.probability_draws,
        seed,
    )?;
    let to_p = exposure_probabilities(
        &query.assignment,
        &incoming,
        &query.exposure,
        to,
        query.probability_draws,
        seed.wrapping_add(1),
    )?;
    let from_mean = randomization_mean(&outcomes, &observed, &from_p.probabilities, from)?;
    let to_mean = randomization_mean(&outcomes, &observed, &to_p.probabilities, to)?;
    let minimum_exposure_probability = from_p
        .probabilities
        .iter()
        .chain(&to_p.probabilities)
        .copied()
        .fold(f64::INFINITY, f64::min);
    Ok(InterferenceEstimate {
        contrast: randomization_contrast(from_mean, to_mean),
        from_probability_method: from_p.method,
        to_probability_method: to_p.method,
        minimum_exposure_probability,
    })
}

/// Cluster-randomized total effect under partial interference. Every edge must stay within
/// its cluster and every unit must have a neighbor, so `NeighborFraction` is exactly zero or
/// one under the cluster assignment. The Neyman variance uses independent randomized clusters
/// as the units of inference and allows unequal cluster sizes through scaled cluster totals.
///
/// This deliberately has a separate entry point from [`estimate_interference`]. The general
/// direct estimator continues to handle cluster assignments with other exposure mappings.
pub fn estimate_cluster_interference_total(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
) -> Result<InterferenceEstimate, EstimationError> {
    estimate_cluster_interference_total_with_inference(query, data, assignment)
        .map(|(estimate, _)| estimate)
}

/// Estimate the same cluster total contrast and a pointwise interval when at
/// least eight independent clusters occur in each assignment arm. Thin-arm
/// designs retain their point estimate but withhold interval inference.
// length reflects the estimator's fixed statistical contract; refactor would change behavior
#[allow(clippy::too_many_lines)]
pub fn estimate_cluster_interference_total_with_inference(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
) -> Result<(InterferenceEstimate, Option<ClusterTotalInterval>), EstimationError> {
    use antecedent_core::{ExposureMapping, InterferenceFunctional};
    query.validate()?;
    let antecedent_core::AssignmentDesign::ClusterRandomization { clusters, treated_clusters } =
        &query.assignment
    else {
        return Err(EstimationError::unsupported(
            "cluster total-effect estimator requires cluster randomization",
        ));
    };
    let n = assignment.len();
    if n != data.units().row_count() {
        return Err(EstimationError::data_msg("assignment/network row count mismatch"));
    }
    if clusters.len() != n || query.exposure != ExposureMapping::NeighborFraction {
        return Err(EstimationError::unsupported(
            "cluster partial interference requires one cluster per unit and NeighborFraction exposure",
        ));
    }
    let InterferenceFunctional::ExposureContrast { outcome, from, to } = query.functional;
    let untreated = ExposureLevel { own: 0.0, neighbors: 0.0 };
    let treated = ExposureLevel { own: 1.0, neighbors: 1.0 };
    let forward = from == untreated && to == treated;
    let reverse = from == treated && to == untreated;
    if !forward && !reverse {
        return Err(EstimationError::unsupported(
            "cluster partial interference supports only the total-effect contrast between (0,0) and (1,1)",
        ));
    }
    let mut by_cluster: BTreeMap<u32, (bool, f64, usize)> = BTreeMap::new();
    let outcomes = data.units().float64_values(outcome)?;
    for (i, (&cluster, &assigned)) in clusters.iter().zip(assignment).enumerate() {
        if !outcomes[i].is_finite() {
            return Err(EstimationError::data_msg("cluster interference outcome must be finite"));
        }
        let entry = by_cluster.entry(cluster).or_insert((assigned, 0.0, 0));
        if entry.0 != assigned {
            return Err(EstimationError::data_msg(
                "cluster randomized assignment must be constant within each cluster",
            ));
        }
        entry.1 += outcomes[i];
        entry.2 += 1;
        let incoming = data.incoming(i)?;
        if incoming.is_empty()
            || incoming.iter().any(|edge| clusters[edge.from as usize] != cluster)
        {
            return Err(EstimationError::unsupported(
                "partial interference requires a nonempty within-cluster neighborhood for every unit and no cross-cluster edges",
            ));
        }
    }
    let k = by_cluster.len();
    let observed_treated = by_cluster.values().filter(|entry| entry.0).count();
    if *treated_clusters < 2
        || k.saturating_sub(*treated_clusters) < 2
        || observed_treated != *treated_clusters
    {
        return Err(EstimationError::unsupported(
            "cluster randomization requires at least two observed clusters in each arm and the declared treated-cluster count",
        ));
    }
    let scale = k as f64 / n as f64;
    let treated_totals = by_cluster
        .values()
        .filter(|entry| entry.0)
        .map(|entry| entry.1 * scale)
        .collect::<Vec<_>>();
    let control_totals = by_cluster
        .values()
        .filter(|entry| !entry.0)
        .map(|entry| entry.1 * scale)
        .collect::<Vec<_>>();
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let sample_variance = |values: &[f64]| {
        let center = mean(values);
        values.iter().map(|value| (value - center).powi(2)).sum::<f64>() / (values.len() - 1) as f64
    };
    let effect = mean(&treated_totals) - mean(&control_totals);
    let treated_component = sample_variance(&treated_totals) / treated_totals.len() as f64;
    let control_component = sample_variance(&control_totals) / control_totals.len() as f64;
    let variance = treated_component + control_component;
    let p = *treated_clusters as f64 / k as f64;
    let treated_weighted =
        by_cluster.values().filter(|entry| entry.0).map(|entry| entry.1 / p).sum::<f64>();
    let control_weighted =
        by_cluster.values().filter(|entry| !entry.0).map(|entry| entry.1 / (1.0 - p)).sum::<f64>();
    let treated_units =
        by_cluster.values().filter(|entry| entry.0).map(|entry| entry.2).sum::<usize>();
    let control_units = n - treated_units;
    let hajek = treated_weighted / (treated_units as f64 / p)
        - control_weighted / (control_units as f64 / (1.0 - p));
    let direction = if forward { 1.0 } else { -1.0 };
    let estimate = InterferenceEstimate {
        contrast: RandomizationContrast {
            horvitz_thompson: direction * effect,
            hajek: direction * hajek,
            conservative_variance: variance,
        },
        from_probability_method: ExposureProbabilityMethod::Exact,
        to_probability_method: ExposureProbabilityMethod::Exact,
        minimum_exposure_probability: p.min(1.0 - p),
    };
    let interval = if treated_totals.len() >= 8 && control_totals.len() >= 8 && variance > 0.0 && variance.is_finite() {
        let df = variance.powi(2) / (treated_component.powi(2) / (treated_totals.len() - 1) as f64
            + control_component.powi(2) / (control_totals.len() - 1) as f64);
        let critical = antecedent_stats::student_t_ppf(0.975, df);
        (critical.is_finite() && critical > 0.0).then(|| {
            let se = variance.sqrt();
            ClusterTotalInterval {
                bounds: [estimate.contrast.horvitz_thompson - critical * se,
                    estimate.contrast.horvitz_thompson + critical * se],
                standard_error: se,
                degrees_of_freedom: df,
                control_clusters: control_totals.len(),
                treated_clusters: treated_totals.len(),
            }
        })
    } else { None };
    Ok((estimate, interval))
}

/// Require the calibrated pointwise cluster interval. This entry point
/// explicitly refuses thin arms or degenerate between-cluster variation;
/// callers needing a point estimate may use [`estimate_cluster_interference_total`].
pub fn estimate_cluster_interference_total_pointwise(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
) -> Result<(InterferenceEstimate, ClusterTotalInterval), EstimationError> {
    let (estimate, interval) = estimate_cluster_interference_total_with_inference(query, data, assignment)?;
    let interval = interval.ok_or_else(|| EstimationError::unsupported(
        "cluster total-effect interval requires eight independent clusters in each arm and positive cluster variation",
    ))?;
    Ok((estimate, interval))
}

/// Convenience exposure level for an own-treatment contrast on an empty network.
#[must_use]
pub const fn own_treatment_level(treated: bool) -> ExposureLevel {
    ExposureLevel { own: if treated { 1.0 } else { 0.0 }, neighbors: 0.0 }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(test, allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "coverage fixtures build small cluster sizes and node ids from usize/u64 loop counters"))]
    use std::sync::Arc;

    use antecedent_core::{AssignmentDesign, ExposureMapping, InterferenceFunctional, VariableId};
    use antecedent_data::TabularData;

    use super::*;

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn cluster_total_interval_covers_known_truth() {
        // Fixed finite population of cluster potential outcomes with a constant
        // additive total effect (the unit-level effect variance is zero), so the
        // conservative cluster Neyman variance is unbiased and the pointwise 95%
        // Welch interval covers at the nominal rate. Only the complete cluster
        // allocation is re-randomized. Eight clusters per arm is the licensed
        // support boundary.
        const REPLICATES: usize = 2_000;
        const K: usize = 16;
        const TREATED: usize = 8;
        let truth = 3.0_f64;
        let baseline = (0..K)
            .map(|cluster| 0.7 * (cluster as f64 * 0.9).sin() + 0.2 * (cluster % 3) as f64)
            .collect::<Vec<_>>();
        let edges = (0..K)
            .flat_map(|cluster| {
                let first = (cluster * 2) as u32;
                [
                    antecedent_data::NetworkEdge { from: first, to: first + 1, weight: 1.0 },
                    antecedent_data::NetworkEdge { from: first + 1, to: first, weight: 1.0 },
                ]
            })
            .collect::<Vec<_>>();
        let clusters = (0..K).flat_map(|cluster| [cluster as u32; 2]).collect::<Vec<_>>();
        let mut rng = XorShift(0x0BAD_C0DE_D15E_A5E1);
        let mut covered = 0_usize;
        let mut supported = 0_usize;
        for _ in 0..REPLICATES {
            let mut order = (0..K).collect::<Vec<_>>();
            for i in (1..K).rev() {
                let j = (rng.next() as usize) % (i + 1);
                order.swap(i, j);
            }
            let mut treated_cluster = [false; K];
            for &cluster in &order[..TREATED] {
                treated_cluster[cluster] = true;
            }
            let mut assignment = Vec::with_capacity(K * 2);
            let mut outcomes = Vec::with_capacity(K * 2);
            for unit in 0..K * 2 {
                let cluster = unit / 2;
                let treated = treated_cluster[cluster];
                assignment.push(treated);
                let y0 = baseline[cluster] + 0.1 * (unit % 2) as f64;
                outcomes.push(if treated { y0 + truth } else { y0 });
            }
            let table = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
            let network = NetworkData::try_new(table, edges.clone()).unwrap();
            let query = InterferenceQuery::new(
                AssignmentDesign::ClusterRandomization {
                    clusters: Arc::from(clusters.clone()),
                    treated_clusters: TREATED,
                },
                ExposureMapping::NeighborFraction,
                InterferenceFunctional::ExposureContrast {
                    outcome: VariableId::from_raw(0),
                    from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                    to: ExposureLevel { own: 1.0, neighbors: 1.0 },
                },
            );
            let (_, interval) =
                estimate_cluster_interference_total_with_inference(&query, &network, &assignment)
                    .unwrap();
            if let Some(interval_95) = interval {
                supported += 1;
                covered += usize::from(
                    interval_95.bounds[0] <= truth && truth <= interval_95.bounds[1],
                );
            }
        }
        let rate = covered as f64 / supported as f64;
        eprintln!("cluster total: {covered}/{supported}, coverage={rate:.4}");
        assert!(supported >= 1_950, "too many degenerate refusals: {supported}");
        assert!((0.93..=0.985).contains(&rate), "pointwise 95% coverage {rate}");
    }

    #[test]
    fn cluster_total_pointwise_refuses_thin_assignment_arms() {
        let outcomes = [1.0, 2.0, 2.0, 3.0, 8.0, 9.0, 9.0, 10.0];
        let table = TabularData::from_f64_columns([("y", &outcomes[..])]).unwrap();
        let edges = (0..4).flat_map(|cluster| {
            let first = cluster * 2;
            [
                antecedent_data::NetworkEdge { from: first, to: first + 1, weight: 1.0 },
                antecedent_data::NetworkEdge { from: first + 1, to: first, weight: 1.0 },
            ]
        }).collect::<Vec<_>>();
        let network = NetworkData::try_new(table, edges).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from([0_u32, 0, 1, 1, 2, 2, 3, 3]),
                treated_clusters: 2,
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        );
        let assignment = [false, false, false, false, true, true, true, true];
        assert!(estimate_cluster_interference_total(&query, &network, &assignment).is_ok());
        assert!(estimate_cluster_interference_total_pointwise(&query, &network, &assignment).is_err());
    }

    #[test]
    fn empty_network_matches_ordinary_randomized_difference() {
        let table = TabularData::from_f64_columns([("y", &[1.0, 3.0, 2.0, 4.0][..])]).unwrap();
        let data = NetworkData::try_new(table, []).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::CompleteRandomization { treated: 2 },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: own_treatment_level(false),
                to: own_treatment_level(true),
            },
        );
        let estimate =
            estimate_interference(&query, &data, &[false, true, false, true], 9).unwrap();
        assert!((estimate.contrast.hajek - 2.0).abs() < 1e-12);
        assert_eq!(estimate.from_probability_method, ExposureProbabilityMethod::Exact);
    }

    #[test]
    fn network_exposure_contrast_uses_known_design() {
        let table = TabularData::from_f64_columns([("y", &[1.0, 4.0][..])]).unwrap();
        let data = NetworkData::try_new(
            table,
            [
                antecedent_data::NetworkEdge { from: 0, to: 1, weight: 1.0 },
                antecedent_data::NetworkEdge { from: 1, to: 0, weight: 1.0 },
            ],
        )
        .unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        );
        let estimate = estimate_interference(&query, &data, &[true, false], 1).unwrap();
        assert!((estimate.contrast.hajek + 3.0).abs() < 1e-12);
    }

    #[test]
    fn cluster_total_effect_uses_unit_weighted_target_with_unequal_clusters() {
        let outcomes = [4.0, 4.0, 6.0, 6.0, 6.0, 1.0, 1.0, 3.0, 3.0, 3.0];
        let clusters = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3];
        let assignment = [true, true, true, true, true, false, false, false, false, false];
        let table = TabularData::from_f64_columns([("y", &outcomes[..])]).unwrap();
        let edges =
            [(0, 1), (1, 0), (2, 3), (3, 4), (4, 2), (5, 6), (6, 5), (7, 8), (8, 9), (9, 7)]
                .into_iter()
                .map(|(from, to)| antecedent_data::NetworkEdge { from, to, weight: 1.0 })
                .collect::<Vec<_>>();
        let network = NetworkData::try_new(table, edges).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from(clusters),
                treated_clusters: 2,
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        );
        let estimate = estimate_cluster_interference_total(&query, &network, &assignment).unwrap();
        assert!((estimate.contrast.horvitz_thompson - 3.0).abs() < 1e-12);
        assert!((estimate.contrast.conservative_variance - 5.96).abs() < 1e-12);
        assert_eq!(estimate.from_probability_method, ExposureProbabilityMethod::Exact);
    }

    #[test]
    fn direct_cluster_neighbor_count_without_edges_keeps_general_design_estimator() {
        let table = TabularData::from_f64_columns([("y", &[4.0, 4.0, 1.0, 1.0][..])]).unwrap();
        let network =
            NetworkData::try_new(table, Vec::<antecedent_data::NetworkEdge>::new()).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from([0, 0, 1, 1]),
                treated_clusters: 1,
            },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        );
        let estimate = estimate_interference(&query, &network, &[true, true, false, false], 7)
            .expect("general direct cluster design remains available");
        assert!((estimate.contrast.horvitz_thompson - 3.0).abs() < 1e-12);
        assert_eq!(estimate.from_probability_method, ExposureProbabilityMethod::Exact);
    }

    #[test]
    fn matches_frozen_exact_design_calibration_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/response/randomized_interference/expected.json"
        ))
        .unwrap();
        let outcomes = fixture["outcomes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        let assignment = fixture["assignment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_bool().unwrap())
            .collect::<Vec<_>>();
        let table = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let data = NetworkData::try_new(
            table,
            [
                antecedent_data::NetworkEdge { from: 0, to: 1, weight: 1.0 },
                antecedent_data::NetworkEdge { from: 1, to: 0, weight: 1.0 },
            ],
        )
        .unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        );
        let estimate = estimate_interference(&query, &data, &assignment, 1).unwrap();
        let expected = &fixture["expected"];
        let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
        assert!(
            (estimate.contrast.horvitz_thompson
                - expected["horvitz_thompson_contrast"].as_f64().unwrap())
            .abs()
                <= atol
        );
        assert!(
            (estimate.contrast.hajek - expected["hajek_contrast"].as_f64().unwrap()).abs() <= atol
        );
        assert!(
            (estimate.contrast.conservative_variance
                - expected["conservative_variance"].as_f64().unwrap())
            .abs()
                <= atol
        );
        assert!(
            (estimate.minimum_exposure_probability
                - expected["minimum_exposure_probability"].as_f64().unwrap())
            .abs()
                <= atol
        );
        assert_eq!(estimate.from_probability_method, ExposureProbabilityMethod::Exact);
        assert_eq!(estimate.to_probability_method, ExposureProbabilityMethod::Exact);
    }

    #[test]
    fn bayesian_fixed_network_matches_independent_gaussian_calculation() {
        let table = TabularData::from_f64_columns([("y", &[5.0, 6.0, 9.0, 10.0][..])]).unwrap();
        let data = NetworkData::try_new(
            table,
            [
                antecedent_data::NetworkEdge { from: 0, to: 1, weight: 1.0 },
                antecedent_data::NetworkEdge { from: 0, to: 2, weight: 1.0 },
                antecedent_data::NetworkEdge { from: 0, to: 3, weight: 1.0 },
                antecedent_data::NetworkEdge { from: 2, to: 3, weight: 1.0 },
            ],
        )
        .unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        );
        let result = estimate_interference_bayesian(
            &query,
            &data,
            &[true, false, true, false],
            20_000,
            31,
            100.0,
        )
        .unwrap();
        // Rows have design [1,1,0], [1,0,1], [1,1,1], [1,0,2].
        // Their outcomes equal 2 + 3*own + 4*neighbors exactly, so the
        // independent finite-sample target contrast is 3 - 4 = -1.
        assert!((result.mean + 1.0).abs() < 0.08, "posterior mean {}", result.mean);
        assert_eq!(result.contrast_draws.len(), 20_000);
    }

    #[test]
    fn bayesian_interference_refuses_absent_exposure_support() {
        let table = TabularData::from_f64_columns([("y", &[1.0, 2.0][..])]).unwrap();
        let data = NetworkData::try_new(
            table,
            [antecedent_data::NetworkEdge { from: 0, to: 1, weight: 1.0 }],
        )
        .unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 2.0 },
            },
        );
        assert!(
            estimate_interference_bayesian(&query, &data, &[false, false], 100, 1, 10.0).is_err()
        );
    }

    #[test]
    fn bayesian_interference_refuses_rank_deficient_observed_design() {
        let table = TabularData::from_f64_columns([("y", &[1.0, 2.0][..])]).unwrap();
        let data = NetworkData::try_new(
            table,
            [
                antecedent_data::NetworkEdge { from: 0, to: 1, weight: 1.0 },
                antecedent_data::NetworkEdge { from: 1, to: 0, weight: 1.0 },
            ],
        )
        .unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        );
        let error = estimate_interference_bayesian(&query, &data, &[true, false], 100, 1, 10.0)
            .unwrap_err();
        assert!(error.to_string().contains("rank deficient"));
    }
}
