//! Exact two-stage saturation assignment utility for network interference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type ExposureEstimate = (f64, f64, f64, usize, usize, usize, usize, f64);

/// Estimate direct, spillover, and total exposure contrasts under a two-stage
/// cluster-saturation design. Exact design support is enumerated, not simulated.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn estimate_saturation_interference(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    edges: Vec<(u32, u32, f64)>,
    clusters: Vec<u32>,
    realized_saturation: Vec<f64>,
    low_saturation: f64,
    high_saturation: f64,
    high_clusters: usize,
    exposure: String,
    reference_neighbors: f64,
    low_neighbors: f64,
    high_neighbors: f64,
) -> PyResult<(ExposureEstimate, ExposureEstimate, ExposureEstimate)> {
    let y = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let n = y.len();
    if n == 0 || assignment.len() != n || clusters.len() != n || realized_saturation.len() != n {
        return Err(PyValueError::new_err(
            "outcome, assignment, clusters, and realized saturations must have equal non-zero length",
        ));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    if !low_saturation.is_finite()
        || !high_saturation.is_finite()
        || low_saturation <= 0.0
        || high_saturation >= 1.0
        || low_saturation >= high_saturation
    {
        return Err(PyValueError::new_err(
            "low and high saturation probabilities must satisfy 0 < low < high < 1",
        ));
    }
    if [reference_neighbors, low_neighbors, high_neighbors]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || (low_neighbors - high_neighbors).abs() <= 1e-12
    {
        return Err(PyValueError::new_err(
            "neighbor exposure levels must be finite, non-negative, and low/high must differ",
        ));
    }
    if !matches!(
        exposure.as_str(),
        "neighbor_count" | "neighbor_fraction" | "weighted_neighbor_exposure"
    ) {
        return Err(PyValueError::new_err(
            "saturation effects support NeighborCount, NeighborFraction, or WeightedNeighborExposure",
        ));
    }

    let mut cluster_ids = clusters.clone();
    cluster_ids.sort_unstable();
    cluster_ids.dedup();
    let cluster_count = cluster_ids.len();
    if cluster_count < 2 || high_clusters == 0 || high_clusters >= cluster_count {
        return Err(PyValueError::new_err(
            "two-stage saturation requires at least two clusters and both saturation arms",
        ));
    }
    if cluster_count > 30 {
        return Err(PyValueError::new_err(
            "exact saturation support refuses designs with more than 30 clusters",
        ));
    }
    let mut cluster_positions = Vec::with_capacity(n);
    let mut realized_by_cluster: Vec<Option<f64>> = vec![None; cluster_count];
    let mut realized_high_count = 0;
    for i in 0..n {
        let position = cluster_ids.binary_search(&clusters[i]).expect("cluster ID was collected");
        cluster_positions.push(position);
        let saturation = realized_saturation[i];
        if (saturation - low_saturation).abs() > 1e-12
            && (saturation - high_saturation).abs() > 1e-12
        {
            return Err(PyValueError::new_err(
                "realized cluster saturation must equal the declared low or high level",
            ));
        }
        if let Some(previous) = realized_by_cluster[position] {
            if (previous - saturation).abs() > 1e-12 {
                return Err(PyValueError::new_err(
                    "all units in a cluster must share its realized saturation probability",
                ));
            }
        } else {
            realized_by_cluster[position] = Some(saturation);
            if (saturation - high_saturation).abs() <= 1e-12 {
                realized_high_count += 1;
            }
        }
    }
    if realized_high_count != high_clusters {
        return Err(PyValueError::new_err(
            "realized saturation labels must contain the declared number of high-saturation clusters",
        ));
    }

    let mut incoming = vec![Vec::<(usize, f64)>::new(); n];
    for (source, target, weight) in edges {
        let (source, target) = (source as usize, target as usize);
        if source >= n || target >= n || source == target || !weight.is_finite() || weight < 0.0 {
            return Err(PyValueError::new_err("invalid network edge"));
        }
        if clusters[source] != clusters[target] {
            return Err(PyValueError::new_err(
                "partial-interference assumption violated: network edge crosses cluster boundary",
            ));
        }
        incoming[target].push((source, weight));
    }
    if incoming.iter().any(|neighbors| neighbors.len() > 16) {
        return Err(PyValueError::new_err(
            "exact saturation support refuses nodes with more than 16 incoming neighbors",
        ));
    }

    let mut configs = Vec::new();
    let mut current = vec![false; cluster_count];
    enumerate_saturation_configs(0, high_clusters, &mut current, &mut configs);
    if configs.len() > 50_000 {
        return Err(PyValueError::new_err(
            "exact saturation support refuses designs with more than 50,000 cluster allocations",
        ));
    }
    let target_specs = [
        ((0.0, reference_neighbors), (1.0, reference_neighbors)),
        ((0.0, low_neighbors), (0.0, high_neighbors)),
        ((0.0, low_neighbors), (1.0, high_neighbors)),
    ];
    let mut estimates = Vec::with_capacity(3);
    for (from, to) in target_specs {
        let p_from = exposure_probabilities(
            &configs,
            &cluster_positions,
            &incoming,
            low_saturation,
            high_saturation,
            &exposure,
            from,
        );
        let p_to = exposure_probabilities(
            &configs,
            &cluster_positions,
            &incoming,
            low_saturation,
            high_saturation,
            &exposure,
            to,
        );
        let from_mean =
            exposure_mean(&y, &assignment, &clusters, &incoming, &exposure, from, &p_from)?;
        let to_mean = exposure_mean(&y, &assignment, &clusters, &incoming, &exposure, to, &p_to)?;
        let result = (
            to_mean.0 - from_mean.0,
            to_mean.1 - from_mean.1,
            2.0 * (from_mean.2 + to_mean.2),
            from_mean.3,
            to_mean.3,
            from_mean.4,
            to_mean.4,
            p_from.iter().chain(&p_to).copied().fold(f64::INFINITY, f64::min),
        );
        if !result.0.is_finite()
            || !result.1.is_finite()
            || !result.2.is_finite()
            || !result.7.is_finite()
        {
            return Err(PyValueError::new_err("saturation estimate overflowed finite precision"));
        }
        estimates.push(result);
    }
    Ok((estimates[0], estimates[1], estimates[2]))
}

fn enumerate_saturation_configs(
    position: usize,
    remaining_high: usize,
    current: &mut [bool],
    output: &mut Vec<Vec<bool>>,
) {
    if output.len() > 50_000 {
        return;
    }
    if position == current.len() {
        if remaining_high == 0 {
            output.push(current.to_vec());
        }
        return;
    }
    if remaining_high > current.len() - position {
        return;
    }
    current[position] = false;
    enumerate_saturation_configs(position + 1, remaining_high, current, output);
    if remaining_high > 0 {
        current[position] = true;
        enumerate_saturation_configs(position + 1, remaining_high - 1, current, output);
    }
}

fn exposure_probabilities(
    configs: &[Vec<bool>],
    cluster_positions: &[usize],
    incoming: &[Vec<(usize, f64)>],
    low: f64,
    high: f64,
    exposure: &str,
    level: (f64, f64),
) -> Vec<f64> {
    let config_weight = 1.0 / configs.len() as f64;
    (0..cluster_positions.len())
        .map(|unit| {
            let mut probability = 0.0;
            for config in configs {
                let p_own = if config[cluster_positions[unit]] { high } else { low };
                let own_probability = if level.0 == 1.0 { p_own } else { 1.0 - p_own };
                let neighbors = &incoming[unit];
                let mut neighbor_probability = 0.0;
                for mask in 0..(1_usize << neighbors.len()) {
                    let mut assignment_probability = 1.0;
                    let mut treated = 0.0;
                    let mut treated_weight = 0.0;
                    let mut total_weight = 0.0;
                    for (j, &(source, weight)) in neighbors.iter().enumerate() {
                        let source_cluster = cluster_positions[source];
                        let p = if config[source_cluster] { high } else { low };
                        let is_treated = mask & (1 << j) != 0;
                        assignment_probability *= if is_treated { p } else { 1.0 - p };
                        treated += if is_treated { 1.0 } else { 0.0 };
                        treated_weight += if is_treated { weight } else { 0.0 };
                        total_weight += weight;
                    }
                    let neighbor_exposure = match exposure {
                        "neighbor_count" => treated,
                        "neighbor_fraction" => {
                            if neighbors.is_empty() {
                                0.0
                            } else {
                                treated / neighbors.len() as f64
                            }
                        }
                        _ => {
                            if total_weight == 0.0 {
                                0.0
                            } else {
                                treated_weight / total_weight
                            }
                        }
                    };
                    if (neighbor_exposure - level.1).abs() <= 1e-12 {
                        neighbor_probability += assignment_probability;
                    }
                }
                probability += config_weight * own_probability * neighbor_probability;
            }
            probability
        })
        .collect()
}

/// (HT, Hájek, diagonal variance term, observed exposed units, clusters).
fn exposure_mean(
    outcomes: &[f64],
    assignment: &[bool],
    clusters: &[u32],
    incoming: &[Vec<(usize, f64)>],
    exposure: &str,
    level: (f64, f64),
    probabilities: &[f64],
) -> PyResult<(f64, f64, f64, usize, usize)> {
    let n = outcomes.len();
    let mut ht_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut weighted_outcome_sum = 0.0;
    let mut variance_sum = 0.0;
    let mut observed_count = 0;
    let mut observed_clusters = std::collections::BTreeSet::new();
    for i in 0..n {
        let probability = probabilities[i];
        if !probability.is_finite() || probability <= 0.0 {
            return Err(PyValueError::new_err(
                "saturation exposure positivity failure: every unit needs positive probability for each requested exposure",
            ));
        }
        let own = if assignment[i] { 1.0 } else { 0.0 };
        let neighbors = match exposure {
            "neighbor_count" => incoming[i]
                .iter()
                .map(|&(source, _)| if assignment[source] { 1.0 } else { 0.0 })
                .sum::<f64>(),
            "neighbor_fraction" => {
                if incoming[i].is_empty() {
                    0.0
                } else {
                    incoming[i]
                        .iter()
                        .map(|&(source, _)| if assignment[source] { 1.0 } else { 0.0 })
                        .sum::<f64>()
                        / incoming[i].len() as f64
                }
            }
            _ => {
                let total_weight = incoming[i].iter().map(|edge| edge.1).sum::<f64>();
                if total_weight == 0.0 {
                    0.0
                } else {
                    incoming[i]
                        .iter()
                        .map(|&(source, weight)| {
                            weight * if assignment[source] { 1.0 } else { 0.0 }
                        })
                        .sum::<f64>()
                        / total_weight
                }
            }
        };
        if (own - level.0).abs() <= 1e-12 && (neighbors - level.1).abs() <= 1e-12 {
            observed_count += 1;
            observed_clusters.insert(clusters[i]);
            let weight = 1.0 / probability;
            ht_sum += outcomes[i] * weight;
            weight_sum += weight;
            weighted_outcome_sum += outcomes[i] * weight;
            variance_sum += (1.0 - probability) * outcomes[i].powi(2) / probability.powi(2);
        }
    }
    if observed_count == 0 || weight_sum == 0.0 {
        return Err(PyValueError::new_err(
            "requested saturation exposure is absent from realized assignments",
        ));
    }
    let ht = ht_sum / n as f64;
    let hajek = weighted_outcome_sum / weight_sum;
    // A worst-case covariance-free Young bound covers arbitrary dependence from
    // both saturation-level allocation and within-cluster assignment.
    let variance = variance_sum / (n as f64).powi(2);
    Ok((ht, hajek, variance, observed_count, observed_clusters.len()))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(estimate_saturation_interference, module)?)?;
    Ok(())
}
