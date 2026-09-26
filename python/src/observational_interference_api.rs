//! Observational exposure contrasts with explicit network-confounding inputs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

/// (HT, Hájek, cluster sandwich variance, from units, to units, from clusters,
/// to clusters, min propensity, max propensity, clusters total).
type ObservationalExposureResult = (f64, f64, f64, usize, usize, usize, usize, f64, f64, usize);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn estimate_observational_network_exposure(
    outcome: PyReadonlyArray1<'_, f64>,
    assignment: Vec<bool>,
    edges: Vec<(u32, u32, f64)>,
    clusters: Vec<u32>,
    exposure: String,
    from_level: (f64, f64),
    to_level: (f64, f64),
    propensity_from: PyReadonlyArray1<'_, f64>,
    propensity_to: PyReadonlyArray1<'_, f64>,
    propensity_provenance: String,
) -> PyResult<ObservationalExposureResult> {
    let y = outcome.as_array().iter().copied().collect::<Vec<_>>();
    let from_p = propensity_from.as_array().iter().copied().collect::<Vec<_>>();
    let to_p = propensity_to.as_array().iter().copied().collect::<Vec<_>>();
    let n = y.len();
    if n == 0
        || assignment.len() != n
        || clusters.len() != n
        || from_p.len() != n
        || to_p.len() != n
    {
        return Err(PyValueError::new_err(
            "outcome, assignment, clusters, and propensity vectors must have equal non-zero length",
        ));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("outcomes must be finite"));
    }
    if !matches!(
        exposure.as_str(),
        "neighbor_count" | "neighbor_fraction" | "weighted_neighbor_exposure"
    ) {
        return Err(PyValueError::new_err(
            "observational network exposure requires NeighborCount, NeighborFraction, or WeightedNeighborExposure",
        ));
    }
    if !matches!(propensity_provenance.as_str(), "known" | "externally_estimated") {
        return Err(PyValueError::new_err(
            "propensity_provenance must be known or externally_estimated",
        ));
    }
    if !from_level.0.is_finite()
        || !from_level.1.is_finite()
        || !to_level.0.is_finite()
        || !to_level.1.is_finite()
        || (from_level.0 - to_level.0).abs() <= 1e-12 && (from_level.1 - to_level.1).abs() <= 1e-12
    {
        return Err(PyValueError::new_err("exposure contrast levels must be finite and distinct"));
    }
    if from_p
        .iter()
        .chain(&to_p)
        .any(|probability| !probability.is_finite() || *probability <= 0.0 || *probability > 1.0)
    {
        return Err(PyValueError::new_err(
            "observational exposure positivity failure: probabilities must lie in (0, 1]",
        ));
    }
    let unique_clusters = clusters.iter().copied().collect::<BTreeSet<_>>();
    if unique_clusters.len() < 2 {
        return Err(PyValueError::new_err(
            "cluster variance requires at least two partial-interference clusters",
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
    let mut from_sum = 0.0;
    let mut to_sum = 0.0;
    let mut from_weight = 0.0;
    let mut to_weight = 0.0;
    let mut from_count = 0;
    let mut to_count = 0;
    let mut from_clusters = BTreeSet::new();
    let mut to_clusters = BTreeSet::new();
    let mut influence_by_cluster = BTreeMap::<u32, f64>::new();
    let mut minimum_probability = f64::INFINITY;
    let mut maximum_probability: f64 = 0.0;
    for i in 0..n {
        let neighbor_value = match exposure.as_str() {
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
        let own = if assignment[i] { 1.0 } else { 0.0 };
        let is_from =
            (own - from_level.0).abs() <= 1e-12 && (neighbor_value - from_level.1).abs() <= 1e-12;
        let is_to =
            (own - to_level.0).abs() <= 1e-12 && (neighbor_value - to_level.1).abs() <= 1e-12;
        let from_score = if is_from { y[i] / from_p[i] } else { 0.0 };
        let to_score = if is_to { y[i] / to_p[i] } else { 0.0 };
        from_sum += from_score;
        to_sum += to_score;
        if is_from {
            from_weight += 1.0 / from_p[i];
            from_count += 1;
            from_clusters.insert(clusters[i]);
        }
        if is_to {
            to_weight += 1.0 / to_p[i];
            to_count += 1;
            to_clusters.insert(clusters[i]);
        }
        *influence_by_cluster.entry(clusters[i]).or_default() += (to_score - from_score) / n as f64;
        minimum_probability = minimum_probability.min(from_p[i]).min(to_p[i]);
        maximum_probability = maximum_probability.max(from_p[i]).max(to_p[i]);
    }
    if from_count == 0 || to_count == 0 || from_weight == 0.0 || to_weight == 0.0 {
        return Err(PyValueError::new_err(
            "observational network contrast requires observed support at both requested exposure levels",
        ));
    }
    let ht = (to_sum - from_sum) / n as f64;
    let hajek = to_sum / to_weight - from_sum / from_weight;
    let cluster_count = influence_by_cluster.len();
    let mean_influence = influence_by_cluster.values().sum::<f64>() / cluster_count as f64;
    let cluster_variance = cluster_count as f64 / (cluster_count - 1) as f64
        * influence_by_cluster.values().map(|value| (value - mean_influence).powi(2)).sum::<f64>();
    if !ht.is_finite() || !hajek.is_finite() || !cluster_variance.is_finite() {
        return Err(PyValueError::new_err(
            "observational network estimate overflowed finite precision",
        ));
    }
    Ok((
        ht,
        hajek,
        cluster_variance,
        from_count,
        to_count,
        from_clusters.len(),
        to_clusters.len(),
        minimum_probability,
        maximum_probability,
        cluster_count,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(estimate_observational_network_exposure, module)?)?;
    Ok(())
}
