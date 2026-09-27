//! Exact exposure probabilities under two-stage cluster saturation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

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
    /// Independent-cluster pointwise Neyman interval when its support gate passes.
    pub pointwise_interval: Option<SaturationClusterInterval>,
    /// Why the interval is withheld, while the point estimate remains available.
    pub interval_unavailable_reason: Option<&'static str>,
}

/// Conservative first-stage Neyman interval with within-cluster assignment
/// variation retained in the observed cluster scores.
#[derive(Clone, Debug, PartialEq)]
pub struct SaturationClusterInterval {
    /// Standard error of the Horvitz--Thompson exposure contrast.
    pub standard_error: f64,
    /// Two-sided pointwise 95% treatment-minus-control interval.
    pub bounds: [f64; 2],
    /// Welch--Satterthwaite degrees of freedom for the two cluster arms.
    pub degrees_of_freedom: f64,
    /// Independent high- and low-saturation cluster counts.
    pub high_clusters: usize,
    /// Independent low-saturation cluster count.
    pub low_clusters: usize,
}

/// Estimate one contrast under complete high/low cluster allocation followed by
/// independent Bernoulli assignment within each cluster. Exposure probabilities
/// marginalize the high/low allocation of each unit's own cluster and enumerate
/// its incoming-neighbor assignments. Other cluster allocations cancel because
/// partial interference forbids cross-cluster edges.
/// The legacy covariance-free variance remains descriptive. An independent-cluster
/// pointwise interval is supplied separately when both saturation arms and both
/// exposure levels pass their support gates.
///
/// # Errors
///
/// Refuses invalid randomization, cross-cluster edges, absent realized exposures,
/// nonpositive exposure probabilities, or neighborhoods above enumeration caps.
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
    if k < 2 || *high_clusters == 0 || *high_clusters >= k {
        return Err(EstimationError::unsupported("exact saturation requires both high and low cluster arms"));
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
    let high_share = *high_clusters as f64 / k as f64;
    let from_p = level_probabilities(&incoming, *low_probability, *high_probability, high_share, &query.exposure, from);
    let to_p = level_probabilities(&incoming, *low_probability, *high_probability, high_share, &query.exposure, to);
    let observed = exposures(assignment, &incoming, &query.exposure)?;
    let baseline = weighted_mean(&outcomes, clusters, &observed, from, &from_p)?;
    let active = weighted_mean(&outcomes, clusters, &observed, to, &to_p)?;
    let variance = 2.0 * (baseline.variance + active.variance);
    let minimum_exposure_probability = from_p.iter().chain(&to_p).copied().fold(f64::INFINITY, f64::min);
    if !variance.is_finite() || !minimum_exposure_probability.is_finite() {
        return Err(EstimationError::data_msg("saturation estimate overflowed finite precision"));
    }
    let ht_contrast = active.ht - baseline.ht;
    let interval = saturation_cluster_interval(
        &ids, &observed_cluster_saturation, *high_probability, *high_clusters,
        &baseline.cluster_totals, &active.cluster_totals, n, ht_contrast,
        baseline.clusters, active.clusters,
    );
    let (pointwise_interval, interval_unavailable_reason) = match interval {
        Ok(value) => (Some(value), None),
        Err(reason) => (None, Some(reason)),
    };
    Ok(SaturationInterferenceEstimate {
        estimate: InterferenceEstimate {
            contrast: RandomizationContrast {
                horvitz_thompson: ht_contrast,
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
        pointwise_interval,
        interval_unavailable_reason,
    })
}

/// Require a pointwise interval for a two-stage saturation contrast.
///
/// This keeps a point estimate available through [`estimate_saturation_interference`]
/// while giving callers that explicitly request interval inference an error with
/// the precise support condition that failed.
pub fn estimate_saturation_interference_pointwise(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
) -> Result<SaturationInterferenceEstimate, EstimationError> {
    let result = estimate_saturation_interference(query, data, assignment)?;
    if let Some(reason) = result.interval_unavailable_reason {
        return Err(EstimationError::unsupported(reason));
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn saturation_cluster_interval(
    ids: &[u32], observed_saturation: &[Option<f64>], high_probability: f64,
    high_count: usize, from_totals: &BTreeMap<u32, f64>, to_totals: &BTreeMap<u32, f64>,
    units: usize, point: f64, from_clusters: usize, to_clusters: usize,
) -> Result<SaturationClusterInterval, &'static str> {
    let k = ids.len();
    let low_count = k - high_count;
    if high_count < 8 || low_count < 8 {
        return Err("pointwise saturation inference requires eight independent clusters in each saturation arm");
    }
    if from_clusters < 8 || to_clusters < 8 {
        return Err("pointwise saturation inference requires eight exposed clusters at each requested level");
    }
    let scale = k as f64 / units as f64;
    let mut high_scores = Vec::with_capacity(high_count);
    let mut low_scores = Vec::with_capacity(low_count);
    for (position, id) in ids.iter().enumerate() {
        let score = scale * (to_totals.get(id).copied().unwrap_or(0.0)
            - from_totals.get(id).copied().unwrap_or(0.0));
        if observed_saturation[position].is_some_and(|p| (p - high_probability).abs() <= 1e-12) {
            high_scores.push(score);
        } else {
            low_scores.push(score);
        }
    }
    let component = |scores: &[f64]| {
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|score| (score - mean).powi(2)).sum::<f64>()
            / (scores.len() - 1) as f64;
        let arm_share = scores.len() as f64 / k as f64;
        arm_share.powi(2) * variance / scores.len() as f64
    };
    let high_variance = component(&high_scores);
    let low_variance = component(&low_scores);
    let variance = high_variance + low_variance;
    if !variance.is_finite() || variance <= 0.0 {
        return Err("pointwise saturation inference requires positive finite between-cluster score variation");
    }
    let denominator = high_variance.powi(2) / (high_count - 1) as f64
        + low_variance.powi(2) / (low_count - 1) as f64;
    let df = variance.powi(2) / denominator;
    let critical = antecedent_stats::student_t_ppf(0.975, df);
    if !critical.is_finite() || critical <= 0.0 {
        return Err("pointwise saturation inference critical value is unavailable");
    }
    let standard_error = variance.sqrt();
    Ok(SaturationClusterInterval {
        standard_error,
        bounds: [point - critical * standard_error, point + critical * standard_error],
        degrees_of_freedom: df,
        high_clusters: high_count,
        low_clusters: low_count,
    })
}

fn level_probabilities(
    incoming: &[Vec<(usize, f64)>], low: f64, high: f64,
    high_share: f64, mapping: &ExposureMapping, level: ExposureLevel,
) -> Vec<f64> {
    (0..incoming.len()).map(|unit| {
        let mut probability = 0.0;
        for (saturation, saturation_mass) in [(low, 1.0 - high_share), (high, high_share)] {
            let own_mass = if level.own == 1.0 { saturation } else { 1.0 - saturation };
            let neighbors = &incoming[unit];
            let mut neighbor_mass = 0.0;
            for mask in 0..(1_usize << neighbors.len()) {
                let mut mass = 1.0;
                let mut treated = 0.0;
                let mut treated_weight = 0.0;
                let mut total_weight = 0.0;
                for (j, &(_, weight)) in neighbors.iter().enumerate() {
                    let selected = mask & (1 << j) != 0;
                    mass *= if selected { saturation } else { 1.0 - saturation };
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
            probability += saturation_mass * own_mass * neighbor_mass;
        }
        probability
    }).collect()
}

struct WeightedMean {
    ht: f64,
    hajek: f64,
    variance: f64,
    units: usize,
    clusters: usize,
    cluster_totals: BTreeMap<u32, f64>,
}

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
    let mut cluster_totals = BTreeMap::new();
    for i in 0..outcomes.len() {
        let p = probabilities[i];
        if !p.is_finite() || p <= 0.0 {
            return Err(EstimationError::unsupported("saturation exposure positivity failure for a requested level"));
        }
        if (observed[i].own - level.own).abs() <= 1e-12 && (observed[i].neighbors - level.neighbors).abs() <= 1e-12 {
            let weight = 1.0 / p;
            total += outcomes[i] * weight;
            *cluster_totals.entry(clusters[i]).or_insert(0.0) += outcomes[i] * weight;
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
    Ok(WeightedMean { ht: total / n, hajek: total / weights, variance: variance.powi(2) / n.powi(2), units, clusters: observed_clusters.len(), cluster_totals })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery, VariableId};
    use antecedent_data::{NetworkData, NetworkEdge, TabularData};

    use super::{estimate_saturation_interference, estimate_saturation_interference_pointwise};

    #[test]
    fn local_mixture_matches_global_complete_allocation_enumeration() {
        let incoming = vec![vec![(1_usize, 1.0), (2_usize, 2.0)]; 3];
        for (mapping, level) in [
            (ExposureMapping::NeighborFraction, ExposureLevel { own: 0.0, neighbors: 0.5 }),
            (ExposureMapping::WeightedNeighborExposure, ExposureLevel { own: 1.0, neighbors: 1.0 / 3.0 }),
        ] {
            let observed = super::level_probabilities(&incoming, 0.2, 0.8, 0.5, &mapping, level)[0];
            let mut expected = 0.0;
            for high_a in [false, true] {
                for high_b in [false, true] {
                    for high_c in [false, true] {
                        for high_d in [false, true] {
                            if [high_a, high_b, high_c, high_d].iter().filter(|&&high| high).count() != 2 {
                                continue;
                            }
                            let p = if high_a { 0.8 } else { 0.2 };
                            let own_mass = if level.own == 1.0 { p } else { 1.0 - p };
                            for first in [false, true] {
                                for second in [false, true] {
                                    let g = match mapping {
                                        ExposureMapping::NeighborFraction =>
                                            (f64::from(first) + f64::from(second)) / 2.0,
                                        ExposureMapping::WeightedNeighborExposure =>
                                            (f64::from(first) + 2.0 * f64::from(second)) / 3.0,
                                        _ => unreachable!(),
                                    };
                                    if (g - level.neighbors).abs() <= 1e-12 {
                                        expected += own_mass * (if first { p } else { 1.0 - p })
                                            * (if second { p } else { 1.0 - p }) / 6.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!((observed - expected).abs() < 1e-12);
        }
    }

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

    #[test]
    fn exact_marginalization_accepts_many_clusters_and_withholds_thin_interval() {
        let (network, assignment, mut query) = fixture();
        let point = estimate_saturation_interference(&query, &network, &assignment).unwrap();
        assert!(point.pointwise_interval.is_none());
        assert!(point.interval_unavailable_reason.unwrap().contains("eight independent clusters"));
        let refused = estimate_saturation_interference_pointwise(&query, &network, &assignment)
            .unwrap_err();
        assert!(refused.to_string().contains("eight independent clusters"));
        // Forty independent clusters would require over 10^11 global high/low
        // allocations; the exact marginal mixture needs only two states per unit.
        let clusters = (0..40).flat_map(|cluster| [cluster; 3]).collect::<Vec<u32>>();
        let mut realized = vec![0.2; 60];
        realized.extend(vec![0.8; 60]);
        query.assignment = AssignmentDesign::TwoStageSaturation {
            clusters: Arc::from(clusters), low_probability: 0.2,
            high_probability: 0.8, high_clusters: 20,
            realized_saturation: Arc::from(realized),
        };
        let expanded_assignment = assignment.iter().copied().cycle().take(120).collect::<Vec<_>>();
        let outcomes = vec![1.0; 120];
        let expanded_data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let expanded_edges = (0..40).flat_map(|cluster| {
            let first = cluster * 3;
            (first..first + 3).flat_map(move |from| {
                (first..first + 3).filter(move |&to| to != from).map(move |to| NetworkEdge {
                    from: from as u32, to: to as u32, weight: 1.0,
                })
            })
        }).collect::<Vec<_>>();
        let expanded = NetworkData::try_new(expanded_data, expanded_edges).unwrap();
        // The observed assignment here is arbitrary; this assertion exercises
        // the exact marginal calculation beyond the old enumeration cap.
        let result = estimate_saturation_interference(&query, &expanded, &expanded_assignment);
        assert!(result.is_ok());
    }

    #[test]
    fn pointwise_interval_refuses_sparse_exposure_despite_adequate_cluster_arms() {
        let clusters = (0..20).flat_map(|cluster| [cluster; 3]).collect::<Vec<u32>>();
        let realized = [vec![0.2; 30], vec![0.8; 30]].concat();
        let mut assignment = vec![false; 60];
        assignment[30..33].fill(true);
        let edges = (0..20).flat_map(|cluster| {
            let first = cluster * 3;
            (first..first + 3).flat_map(move |from| {
                (first..first + 3).filter(move |&to| to != from).map(move |to| NetworkEdge {
                    from: from as u32, to: to as u32, weight: 1.0,
                })
            })
        }).collect::<Vec<_>>();
        let outcomes = (0..60).map(|unit| 1.0 + f64::from(assignment[unit])).collect::<Vec<_>>();
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let network = NetworkData::try_new(data, edges).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::TwoStageSaturation {
                clusters: Arc::from(clusters), low_probability: 0.2, high_probability: 0.8,
                high_clusters: 10, realized_saturation: Arc::from(realized),
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        );
        let point = estimate_saturation_interference(&query, &network, &assignment).unwrap();
        assert_eq!(point.to_exposed_clusters, 1);
        assert!(point.pointwise_interval.is_none());
        let refused = estimate_saturation_interference_pointwise(&query, &network, &assignment)
            .unwrap_err();
        assert!(refused.to_string().contains("eight exposed clusters"));
    }
}
