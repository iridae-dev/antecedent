//! Conditional continuous-dose response using supplied dose densities.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

/// One baseline-group response at a prespecified target dose.
#[derive(Clone, Debug, PartialEq)]
pub struct DoseResponsePoint {
    /// Baseline stratum.
    pub baseline_group: String,
    /// Prespecified dose.
    pub target_dose: f64,
    /// Local inverse-density weighted response.
    pub response: f64,
    /// Number of rows inside the kernel window.
    pub local_rows: usize,
    /// Kish effective sample size of local weights.
    pub effective_sample_size: f64,
    /// Smallest supplied local dose density.
    pub minimum_dose_density: f64,
    /// Largest normalized local weight.
    pub maximum_normalized_weight: f64,
    /// Descriptive local outcome deviation; not an inferential standard error.
    pub local_outcome_sd: f64,
}

/// Evaluate a local triangular-kernel response for every group and target.
///
/// The density is supplied by the caller; this function does not fit or
/// authenticate it. Every group-target cell must have the requested support.
pub fn conditional_dose_response(
    outcome: &[f64],
    dose: &[f64],
    groups: &[String],
    density: &[f64],
    target_doses: &[f64],
    bandwidth: f64,
    min_local_support: usize,
) -> Result<Vec<DoseResponsePoint>, String> {
    let n = outcome.len();
    if n == 0 || dose.len() != n || groups.len() != n || density.len() != n {
        return Err(
            "outcome, dose, groups, and dose density must have equal non-zero length".into()
        );
    }
    if target_doses.is_empty() || target_doses.iter().any(|value| !value.is_finite()) {
        return Err("target_doses must be non-empty and finite".into());
    }
    if !bandwidth.is_finite() || bandwidth <= 0.0 || min_local_support < 2 {
        return Err("bandwidth must be positive and min_local_support must be at least two".into());
    }
    if outcome.iter().chain(dose).any(|value| !value.is_finite()) {
        return Err("outcomes and doses must be finite".into());
    }
    if density.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err("continuous-dose positivity failure: supplied dose densities must be finite and positive".into());
    }
    if groups.iter().any(String::is_empty) {
        return Err("baseline group labels must be non-empty strings".into());
    }
    let strata: BTreeSet<&String> = groups.iter().collect();
    let mut output = Vec::with_capacity(strata.len() * target_doses.len());
    for stratum in strata {
        for &target in target_doses {
            let mut weighted_outcomes = Vec::new();
            let mut weights = Vec::new();
            let mut local_densities = Vec::new();
            for i in 0..n {
                let scaled_distance = (dose[i] - target) / bandwidth;
                if groups[i] != *stratum || scaled_distance.abs() >= 1.0 {
                    continue;
                }
                let kernel = 0.75 * (1.0 - scaled_distance.powi(2));
                let weight = kernel / density[i];
                if !weight.is_finite() || weight <= 0.0 {
                    return Err("dose-response weights overflowed or lost positivity".into());
                }
                weighted_outcomes.push(outcome[i]);
                weights.push(weight);
                local_densities.push(density[i]);
            }
            if weights.len() < min_local_support {
                return Err(format!(
                    "dose-response support failure: stratum {stratum:?} at dose {target} has {} local rows, below min_local_support={min_local_support}",
                    weights.len()
                ));
            }
            let sum_weight = weights.iter().sum::<f64>();
            let sum_weight_squared = weights.iter().map(|weight| weight.powi(2)).sum::<f64>();
            let response = weights
                .iter()
                .zip(&weighted_outcomes)
                .map(|(weight, outcome)| weight * outcome)
                .sum::<f64>()
                / sum_weight;
            let effective_n = sum_weight.powi(2) / sum_weight_squared;
            let local_sd = (weights
                .iter()
                .zip(&weighted_outcomes)
                .map(|(weight, outcome)| weight * (outcome - response).powi(2))
                .sum::<f64>()
                / sum_weight)
                .sqrt();
            let max_normalized_weight =
                weights.iter().map(|weight| weight / sum_weight).fold(0.0_f64, f64::max);
            let minimum_density = local_densities.iter().copied().fold(f64::INFINITY, f64::min);
            if !response.is_finite() || !effective_n.is_finite() || !local_sd.is_finite() {
                return Err("dose-response estimate overflowed finite precision".into());
            }
            output.push(DoseResponsePoint {
                baseline_group: stratum.clone(),
                target_dose: target,
                response,
                local_rows: weights.len(),
                effective_sample_size: effective_n,
                minimum_dose_density: minimum_density,
                maximum_normalized_weight: max_normalized_weight,
                local_outcome_sd: local_sd,
            });
        }
    }
    Ok(output)
}
