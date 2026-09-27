//! Observational exposure contrasts with explicit network-confounding inputs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExposureLevel, ExposureMapping, ExposurePropensityProvenance, VariableId};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::{ObservationalExposureSpec, estimate_observational_exposure as estimate_native};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    let mapping = match exposure.as_str() {
        "neighbor_count" => ExposureMapping::NeighborCount,
        "neighbor_fraction" => ExposureMapping::NeighborFraction,
        "weighted_neighbor_exposure" => ExposureMapping::WeightedNeighborExposure,
        _ => return Err(PyValueError::new_err("observational network exposure requires NeighborCount, NeighborFraction, or WeightedNeighborExposure")),
    };
    let provenance = match propensity_provenance.as_str() {
        "known" => ExposurePropensityProvenance::Known,
        "externally_estimated" => ExposurePropensityProvenance::ExternallyEstimated,
        _ => return Err(PyValueError::new_err("propensity_provenance must be known or externally_estimated")),
    };
    let data = TabularData::from_f64_columns([("y", y.as_slice())]).map_err(crate::py_err)?;
    let network = NetworkData::try_new(data, edges.into_iter().map(|(from,to,weight)| NetworkEdge { from, to, weight }).collect::<Vec<_>>()).map_err(crate::py_err)?;
    let result = estimate_native(
        &network, VariableId::from_raw(0),
        &ObservationalExposureSpec {
            assignment: &assignment, clusters: &clusters, exposure: &mapping,
            from: ExposureLevel { own: from_level.0, neighbors: from_level.1 },
            to: ExposureLevel { own: to_level.0, neighbors: to_level.1 },
            propensity_from: &from_p, propensity_to: &to_p,
            propensity_provenance: provenance,
        },
    ).map_err(crate::py_err)?;
    Ok((
        result.horvitz_thompson, result.hajek, result.cluster_robust_variance,
        result.from_exposed_units, result.to_exposed_units,
        result.from_exposed_clusters, result.to_exposed_clusters,
        result.minimum_exposure_probability, result.maximum_exposure_probability,
        result.clusters,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(estimate_observational_network_exposure, module)?)?;
    Ok(())
}
