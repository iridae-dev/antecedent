//! Exact exposure probabilities under two-stage cluster saturation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery};
use antecedent_data::{NetworkData, TableView};
use antecedent_stats::{ExposureProbabilityMethod, RandomizationContrast, exposures};

use crate::{EstimationError, InterferenceEstimate};

/// Point estimate and observed support for one saturation exposure contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct SaturationInterferenceEstimate {
    /// Design-based HT/Hájek estimate and covariance-free variance proxy.
    pub estimate: InterferenceEstimate,
    /// Number of observed units at the baseline exposure.
    pub from_exposed_units: usize,
    /// Number of observed units at the active exposure.
    pub to_exposed_units: usize,
    /// Number of clusters containing observed baseline-exposed units.
    pub from_exposed_clusters: usize,
    /// Number of clusters containing observed active-exposed units.
    pub to_exposed_clusters: usize,
}

/// Estimate one contrast under complete high/low cluster allocation followed by
/// independent Bernoulli assignment within each cluster. Exposure probabilities
/// enumerate the cluster allocations and all incoming-neighbor assignments.
/// The variance is a covariance-free plug-in bound, not a calibrated interval.
///
/// # Errors
///
/// Refuses invalid randomization, cross-cluster edges, absent realized exposures,
/// nonpositive exposure probabilities, or designs above exact enumeration caps.
pub fn estimate_saturation_interference(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
) -> Result<SaturationInterferenceEstimate, EstimationError> {
    query.validate()?;
    let AssignmentDesign::TwoStageSaturation {
        clusters, low_probability, high_probability, high_clusters, realized_saturation,
    } = &query.assignment else {
        return Err(EstimationError::unsupported("two-stage saturation assignment is required"));
    };
    let n = data.units().row_count();
    if n == 0 || assignment.len() != n || clusters.len() != n || realized_saturation.len() != n {
        return Err(EstimationError::data_msg("outcome, assignment, clusters, and realized saturations must have equal non-zero length"));
    }
    if !matches!(query.exposure, ExposureMapping::NeighborCount | ExposureMapping::NeighborFraction | ExposureMapping::WeightedNeighborExposure) {
        return Err(EstimationError::unsupported("saturation effects require a built-in neighbor exposure mapping"));
    }
    let InterferenceFunctional::ExposureContrast { outcome, from, to } = query.functional;
    for level in [from, to] {
        if (level.own - 0.0).abs() > 1e-12 && (level.own - 1.0).abs() > 1e-12 {
            return Err(EstimationError::data_msg("saturation exposure own-treatment levels must be zero or one"));
        }
        if level.neighbors < 0.0 {
            return Err(EstimationError::data_msg("saturation neighbor exposure levels must be non-negative"));
        }
    }
    let outcomes = data.units().float64_values(outcome)?;
    if outcomes.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::data_msg("saturation outcomes must be finite"));
    }
    let mut ids = clusters.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let k = ids.len();
    if k < 2 || *high_clusters == 0 || *high_clusters >= k || k > 30 {
        return Err(EstimationError::unsupported("exact saturation requires two arms and at most 30 clusters"));
    }
    let positions = clusters.iter().map(|id| ids.binary_search(id).expect("collected cluster"))
        .collect::<Vec<_>>();
    let mut observed_cluster_saturation: Vec<Option<f64>> = vec![None; k];
    for (i, &position) in positions.iter().enumerate() {
        let value = realized_saturation[i];
        if !value.is_finite() || ((value - low_probability).abs() > 1e-12 && (value - high_probability).abs() > 1e-12) {
            return Err(EstimationError::data_msg("realized cluster saturation must equal declared low or high probability"));
        }
        if let Some(previous) = observed_cluster_saturation[position] {
            if (previous - value).abs() > 1e-12 {
                return Err(EstimationError::data_msg("all units in a cluster must share realized saturation"));
            }
        } else {
            observed_cluster_saturation[position] = Some(value);
        }
    }
    if observed_cluster_saturation.iter().filter(|value| value.is_some_and(|p| (p - high_probability).abs() <= 1e-12)).count() != *high_clusters {
        return Err(EstimationError::data_msg("realized saturation does not match declared high-cluster count"));
    }
    let mut incoming = Vec::with_capacity(n);
    for unit in 0..n {
        let neighbors = data.incoming(unit)?.iter().map(|edge| (edge.from as usize, edge.weight)).collect::<Vec<_>>();
        if neighbors.len() > 16 {
            return Err(EstimationError::unsupported("exact saturation refuses more than 16 incoming neighbors per unit"));
        }
        if neighbors.iter().any(|&(source, weight)| clusters[source] != clusters[unit] || !weight.is_finite() || weight < 0.0) {
            return Err(EstimationError::unsupported("partial interference requires nonnegative within-cluster edges only"));
        }
        incoming.push(neighbors);
    }
    let mut configs = Vec::new();
    enumerate_configs(0, *high_clusters, &mut vec![false; k], &mut configs);
    if configs.len() > 50_000 {
        return Err(EstimationError::unsupported("exact saturation refuses more than 50,000 cluster allocations"));
    }
    let from_p = level_probabilities(&configs, &positions, &incoming, *low_probability, *high_probability, &query.exposure, from);
    let to_p = level_probabilities(&configs, &positions, &incoming, *low_probability, *high_probability, &query.exposure, to);
    let observed = exposures(assignment, &incoming, &query.exposure)?;
    let baseline = weighted_mean(&outcomes, clusters, &observed, from, &from_p)?;
    let active = weighted_mean(&outcomes, clusters, &observed, to, &to_p)?;
    let variance = 2.0 * (baseline.variance + active.variance);
    let minimum_exposure_probability = from_p.iter().chain(&to_p).copied().fold(f64::INFINITY, f64::min);
    if !variance.is_finite() || !minimum_exposure_probability.is_finite() {
        return Err(EstimationError::data_msg("saturation estimate overflowed finite precision"));
    }
    Ok(SaturationInterferenceEstimate {
        estimate: InterferenceEstimate {
            contrast: RandomizationContrast {
                horvitz_thompson: active.ht - baseline.ht,
                hajek: active.hajek - baseline.hajek,
                conservative_variance: variance,
            },
            from_probability_method: ExposureProbabilityMethod::Exact,
            to_probability_method: ExposureProbabilityMethod::Exact,
            minimum_exposure_probability,
        },
        from_exposed_units: baseline.units,
        to_exposed_units: active.units,
        from_exposed_clusters: baseline.clusters,
        to_exposed_clusters: active.clusters,
    })
}

fn enumerate_configs(position: usize, remaining: usize, current: &mut [bool], out: &mut Vec<Vec<bool>>) {
    if out.len() > 50_000 || remaining > current.len() - position { return; }
    if position == current.len() {
        if remaining == 0 { out.push(current.to_vec()); }
        return;
    }
    current[position] = false;
    enumerate_configs(position + 1, remaining, current, out);
    if remaining > 0 {
        current[position] = true;
        enumerate_configs(position + 1, remaining - 1, current, out);
    }
}

fn level_probabilities(
    configs: &[Vec<bool>], positions: &[usize], incoming: &[Vec<(usize, f64)>],
    low: f64, high: f64, mapping: &ExposureMapping, level: ExposureLevel,
) -> Vec<f64> {
    (0..positions.len()).map(|unit| {
        let mut probability = 0.0;
        for config in configs {
            let own_p = if config[positions[unit]] { high } else { low };
            let own_mass = if level.own == 1.0 { own_p } else { 1.0 - own_p };
            let neighbors = &incoming[unit];
            let mut neighbor_mass = 0.0;
            for mask in 0..(1_usize << neighbors.len()) {
                let mut mass = 1.0;
                let mut treated = 0.0;
                let mut treated_weight = 0.0;
                let mut total_weight = 0.0;
                for (j, &(source, weight)) in neighbors.iter().enumerate() {
                    let p = if config[positions[source]] { high } else { low };
                    let selected = mask & (1 << j) != 0;
                    mass *= if selected { p } else { 1.0 - p };
                    treated += if selected { 1.0 } else { 0.0 };
                    treated_weight += if selected { weight } else { 0.0 };
                    total_weight += weight;
                }
                let exposure = match mapping {
                    ExposureMapping::NeighborCount => treated,
                    ExposureMapping::NeighborFraction => if neighbors.is_empty() { 0.0 } else { treated / neighbors.len() as f64 },
                    ExposureMapping::WeightedNeighborExposure => if total_weight == 0.0 { 0.0 } else { treated_weight / total_weight },
                    _ => unreachable!("mapping checked"),
                };
                if (exposure - level.neighbors).abs() <= 1e-12 { neighbor_mass += mass; }
            }
            probability += own_mass * neighbor_mass;
        }
        probability / configs.len() as f64
    }).collect()
}

struct WeightedMean { ht: f64, hajek: f64, variance: f64, units: usize, clusters: usize }

fn weighted_mean(
    outcomes: &[f64], clusters: &[u32], observed: &[ExposureLevel],
    level: ExposureLevel, probabilities: &[f64],
) -> Result<WeightedMean, EstimationError> {
    let n = outcomes.len() as f64;
    let mut total = 0.0;
    let mut weights = 0.0;
    let mut variance = 0.0;
    let mut units = 0;
    let mut observed_clusters = BTreeSet::new();
    for i in 0..outcomes.len() {
        let p = probabilities[i];
        if !p.is_finite() || p <= 0.0 {
            return Err(EstimationError::unsupported("saturation exposure positivity failure for a requested level"));
        }
        if (observed[i].own - level.own).abs() <= 1e-12 && (observed[i].neighbors - level.neighbors).abs() <= 1e-12 {
            let weight = 1.0 / p;
            total += outcomes[i] * weight;
            weights += weight;
            // Cauchy bounds every unknown pairwise covariance of the observed
            // HT score contributions by the product of their standard deviations.
            variance += (1.0 - p).sqrt() * outcomes[i].abs() / p;
            units += 1;
            observed_clusters.insert(clusters[i]);
        }
    }
    if units == 0 || weights == 0.0 {
        return Err(EstimationError::unsupported("requested saturation exposure is absent from realized assignments"));
    }
    Ok(WeightedMean { ht: total / n, hajek: total / weights, variance: variance.powi(2) / n.powi(2), units, clusters: observed_clusters.len() })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery, VariableId};
    use antecedent_data::{NetworkData, NetworkEdge, TabularData};

    use super::estimate_saturation_interference;

    fn fixture() -> (NetworkData, Vec<bool>, InterferenceQuery) {
        let clusters = [0_u32, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
        let assignment = vec![false, false, false, true, false, false, true, true, false, true, true, true];
        let edges = (0..4).flat_map(|cluster| {
            let first = cluster * 3;
            (first..first + 3).flat_map(move |from| {
                (first..first + 3).filter(move |&to| to != from).map(move |to| NetworkEdge { from: from as u32, to: to as u32, weight: 1.0 })
            })
        }).collect::<Vec<_>>();
        let outcomes = (0..12).map(|i| {
            let first = i / 3 * 3;
            let neighbors = (first..first + 3).filter(|&j| j != i && assignment[j]).count() as f64 / 2.0;
            let own = if assignment[i] { 1.0 } else { 0.0 };
            1.0 + 2.0 * own + 3.0 * neighbors + 4.0 * own * neighbors
        }).collect::<Vec<_>>();
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let network = NetworkData::try_new(data, edges).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::TwoStageSaturation {
                clusters: Arc::from(clusters), low_probability: 0.2, high_probability: 0.8,
                high_clusters: 2, realized_saturation: Arc::from([0.2; 6].into_iter().chain([0.8; 6]).collect::<Vec<_>>()),
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.5 },
                to: ExposureLevel { own: 1.0, neighbors: 0.5 },
            },
        );
        (network, assignment, query)
    }

    #[test]
    fn exact_two_stage_design_recovers_direct_spillover_and_total_known_truth() {
        let (network, assignment, mut query) = fixture();
        for (from, to, expected) in [
            ((0.0, 0.5), (1.0, 0.5), 4.0),
            ((0.0, 0.0), (0.0, 1.0), 3.0),
            ((0.0, 0.0), (1.0, 1.0), 9.0),
        ] {
            query.functional = InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: from.0, neighbors: from.1 },
                to: ExposureLevel { own: to.0, neighbors: to.1 },
            };
            let result = estimate_saturation_interference(&query, &network, &assignment).unwrap();
            assert!((result.estimate.contrast.hajek - expected).abs() < 1e-12);
            assert!(result.estimate.contrast.conservative_variance > 0.0);
            assert!(result.estimate.minimum_exposure_probability > 0.0);
        }
    }

    #[test]
    fn exact_saturation_refuses_cross_cluster_edge() {
        let (network, assignment, query) = fixture();
        let mut edges = network.edges().to_vec();
        edges.push(NetworkEdge { from: 0, to: 3, weight: 1.0 });
        let crossed = NetworkData::try_new(network.units().clone(), edges).unwrap();
        assert!(estimate_saturation_interference(&query, &crossed, &assignment).is_err());
    }
}
