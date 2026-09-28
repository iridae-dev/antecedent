//! Exact two-stage saturation assignment utility for network interference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery,
    VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::estimate_saturation_interference as estimate_native;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type ExposureEstimate = (f64, f64, f64, usize, usize, usize, usize, f64);

/// Estimate direct, spillover, and total exposure contrasts under a two-stage
/// cluster-saturation design. The shared Rust estimator enumerates exact support.
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
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).map_err(crate::py_err)?;
    let network = NetworkData::try_new(
        data,
        edges
            .into_iter()
            .map(|(from, to, weight)| NetworkEdge { from, to, weight })
            .collect::<Vec<_>>(),
    )
    .map_err(crate::py_err)?;
    let mapping = match exposure.as_str() {
        "neighbor_count" => ExposureMapping::NeighborCount,
        "neighbor_fraction" => ExposureMapping::NeighborFraction,
        "weighted_neighbor_exposure" => ExposureMapping::WeightedNeighborExposure,
        _ => {
            return Err(PyValueError::new_err(
                "saturation effects support NeighborCount, NeighborFraction, or WeightedNeighborExposure",
            ));
        }
    };
    if [reference_neighbors, low_neighbors, high_neighbors]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
        || (low_neighbors - high_neighbors).abs() <= 1e-12
    {
        return Err(PyValueError::new_err(
            "neighbor exposure levels must be finite, non-negative, and low/high must differ",
        ));
    }
    let design = AssignmentDesign::TwoStageSaturation {
        clusters: Arc::from(clusters),
        low_probability: low_saturation,
        high_probability: high_saturation,
        high_clusters,
        realized_saturation: Arc::from(realized_saturation),
    };
    let mut estimates = Vec::with_capacity(3);
    for (from, to) in [
        ((0.0, reference_neighbors), (1.0, reference_neighbors)),
        ((0.0, low_neighbors), (0.0, high_neighbors)),
        ((0.0, low_neighbors), (1.0, high_neighbors)),
    ] {
        let query = InterferenceQuery::new(
            design.clone(),
            mapping.clone(),
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: from.0, neighbors: from.1 },
                to: ExposureLevel { own: to.0, neighbors: to.1 },
            },
        );
        let result = estimate_native(&query, &network, &assignment).map_err(crate::py_err)?;
        estimates.push((
            result.estimate.contrast.horvitz_thompson,
            result.estimate.contrast.hajek,
            result.estimate.contrast.conservative_variance,
            result.from_exposed_units,
            result.to_exposed_units,
            result.from_exposed_clusters,
            result.to_exposed_clusters,
            result.estimate.minimum_exposure_probability,
        ));
    }
    Ok((estimates[0], estimates[1], estimates[2]))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(estimate_saturation_interference, module)?)?;
    Ok(())
}
