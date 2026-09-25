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

/// Convenience exposure level for an own-treatment contrast on an empty network.
#[must_use]
pub const fn own_treatment_level(treated: bool) -> ExposureLevel {
    ExposureLevel { own: if treated { 1.0 } else { 0.0 }, neighbors: 0.0 }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{AssignmentDesign, ExposureMapping, InterferenceFunctional, VariableId};
    use antecedent_data::TabularData;

    use super::*;

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
