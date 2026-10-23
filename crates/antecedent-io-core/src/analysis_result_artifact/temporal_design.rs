//! Wire section of a sequentially randomized unit-period design or regime survival result.
//!
//! The section is optional on the result body and absent on every other result, so their bodies
//! and digests are unchanged. A `TemporalDesign` query requires it. A loader checks that the
//! section's design is the query's, that it claims no licence or calibration and publishes no
//! scalar interval beside it, that every coordinate array is aligned and finite, that each
//! pointwise interval is ordered, that a band names coordinates in range and contains their
//! pointwise intervals, and that Holm-adjusted p-values are probabilities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use super::AnalysisResultWire;
use crate::{CausalQueryWire, IoError};

/// A temporal-design result with its uncertainty (unlicensed, calibration unmeasured).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalDesignWire {
    /// Design name.
    pub design: String,
    /// Outcome functional.
    pub functional: String,
    /// Coordinate names.
    pub coordinates: Vec<String>,
    /// Estimate (frequentist) or posterior mean (Bayesian).
    pub estimate: Vec<f64>,
    /// Standard error or posterior standard deviation.
    pub standard_error: Vec<f64>,
    /// Pointwise lower endpoints.
    pub lower: Vec<f64>,
    /// Pointwise upper endpoints.
    pub upper: Vec<f64>,
    /// Row-major covariance across coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covariance: Option<Vec<f64>>,
    /// Coordinates a simultaneous band covers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub band_coordinates: Vec<u64>,
    /// Critical value of the band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_critical_value: Option<f64>,
    /// Lower band envelope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_lower: Option<Vec<f64>>,
    /// Upper band envelope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_upper: Option<Vec<f64>>,
    /// Holm-adjusted p-values of the contrast family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holm_adjusted_p: Option<Vec<f64>>,
    /// Worst-case smoothing bias per coordinate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoothing_bias_bound: Option<Vec<f64>>,
    /// Decision-outcome pairs (person-periods for regime survival).
    pub pairs: u64,
    /// Independent units or clusters.
    pub clusters: u64,
    /// Decisions whose outcome period was not observed.
    pub unpaired: u64,
    /// Smallest realized assignment probability, when the design reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_probability: Option<f64>,
    /// Successful and failed whole-procedure bootstrap replicates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicates: Option<[u64; 2]>,
    /// Posterior draws.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draws: Option<u64>,
    /// Uncertainty construction.
    pub uncertainty: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `false`.
    pub licensed: bool,
}

/// Check a temporal-design section against its query.
pub(in crate::analysis_result_artifact) fn validate_temporal_design(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let fail = |message: &str| Err(IoError::Convert(format!("temporal design section: {message}")));
    let query = match &result.query {
        CausalQueryWire::TemporalDesign(query) => Some(query),
        _ => None,
    };
    let (query, section) = match (query, &result.temporal_design) {
        (None, None) => return Ok(()),
        (Some(_), None) => return fail("a temporal-design artifact is missing its section"),
        (None, Some(_)) => return fail("the section is attached to another query"),
        (Some(query), Some(section)) => (query, section),
    };
    if section.design != query.design_name() {
        return fail("the design does not belong to the query");
    }
    if section.licensed
        || section.calibration != "unmeasured"
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail("a temporal-design result claims no licence, calibration or scalar interval");
    }
    let k = section.coordinates.len();
    if k == 0
        || [
            section.estimate.len(),
            section.standard_error.len(),
            section.lower.len(),
            section.upper.len(),
        ]
        .iter()
        .any(|len| *len != k)
        || section.covariance.as_ref().is_some_and(|values| values.len() != k * k)
        || section.smoothing_bias_bound.as_ref().is_some_and(|values| values.len() != k)
    {
        return fail("the coordinate arrays are not aligned");
    }
    if section
        .estimate
        .iter()
        .chain(&section.standard_error)
        .chain(&section.lower)
        .chain(&section.upper)
        .any(|value| !value.is_finite())
        || section.standard_error.iter().any(|value| *value < 0.0)
        || section.lower.iter().zip(&section.upper).any(|(low, high)| low > high)
    {
        return fail("a coordinate is not finite, or an interval is reversed");
    }
    match (&section.band_lower, &section.band_upper, section.band_critical_value) {
        (None, None, None) if section.band_coordinates.is_empty() => {}
        (Some(lower), Some(upper), Some(critical)) => {
            let n = section.band_coordinates.len();
            if n == 0 || lower.len() != n || upper.len() != n || !critical.is_finite() {
                return fail("the band is not aligned with its coordinates");
            }
            for (j, coordinate) in section.band_coordinates.iter().enumerate() {
                let Some(index) = usize::try_from(*coordinate).ok().filter(|index| *index < k)
                else {
                    return fail("a band coordinate is out of range");
                };
                let slack = 1e-9 * (1.0 + section.lower[index].abs() + section.upper[index].abs());
                if lower[j] > section.lower[index] + slack
                    || upper[j] < section.upper[index] - slack
                {
                    return fail("a band does not contain its pointwise interval");
                }
            }
        }
        _ => return fail("the band fields are incomplete"),
    }
    if section
        .holm_adjusted_p
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !(0.0..=1.0).contains(value)))
    {
        return fail("a Holm-adjusted p-value is not a probability");
    }
    if section.replicates.is_some() && section.draws.is_some() {
        return fail("a result is either a bootstrap or a posterior, not both");
    }
    Ok(())
}
