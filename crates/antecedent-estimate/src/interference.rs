//! Design-based estimation under network interference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExposureLevel, InterferenceFunctional, InterferenceQuery};
use antecedent_data::{NetworkData, TableView};
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
}
