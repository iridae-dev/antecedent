//! Declared names parsed into typed values: an unsupported name is refused with its registered
//! reason code rather than ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::str::FromStr;

use antecedent_estimate::EstimationError;

/// The one supported matched case-control estimand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchedEstimand {
    /// The conditional odds ratio.
    ConditionalOddsRatio,
}

impl FromStr for MatchedEstimand {
    type Err = EstimationError;

    /// Accept the conditional odds ratio; every risk-scale estimand is refused
    /// (`effect_not_identified`) and any other name is unsupported (`route_not_supported`).
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        antecedent_estimate::parse_matched_estimand(name).map(|()| Self::ConditionalOddsRatio)
    }
}

/// The one supported learned-continuous transport target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LearnedContinuousTarget {
    /// The population mean contrast.
    PopulationMeanContrast,
}

impl FromStr for LearnedContinuousTarget {
    type Err = EstimationError;

    /// Refuse a heterogeneous or simultaneous target (`route_not_supported`).
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        antecedent_estimate::parse_learned_continuous_target(name)
            .map(|()| Self::PopulationMeanContrast)
    }
}

/// The one supported smoothed dose-response target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SmoothedDoseTarget {
    /// The fixed-bandwidth smoothed response.
    SmoothedResponse,
}

impl FromStr for SmoothedDoseTarget {
    type Err = EstimationError;

    /// Refuse every target but the fixed-bandwidth smoothed response (`route_not_supported`).
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        antecedent_estimate::smoothed_dose::parse_smoothed_dose_target(name)
            .map(|()| Self::SmoothedResponse)
    }
}

/// A declared IID sampling design of a transport trial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IidSampling {
    /// Trial participants nested in the source cohort.
    NestedCohort,
    /// Independent trial and target samples.
    IndependentSamples,
}

impl IidSampling {
    fn from_name(name: &str) -> Self {
        if name == "nested_cohort" { Self::NestedCohort } else { Self::IndependentSamples }
    }

    /// Parse the design of a learned-continuous transport trial; clustered, linked or
    /// undeclared sampling is refused (`sampling_dependence_unknown`).
    ///
    /// # Errors
    /// The refusal for a design other than the two declared IID designs.
    pub fn parse_learned_continuous(name: &str) -> Result<Self, EstimationError> {
        antecedent_estimate::parse_learned_continuous_sampling(name)?;
        Ok(Self::from_name(name))
    }

    /// Parse the design of a smoothed dose-response trial; clustered, linked or undeclared
    /// sampling is refused (`sampling_dependence_unknown`).
    ///
    /// # Errors
    /// The refusal for a design other than the two declared IID designs.
    pub fn parse_smoothed_dose(name: &str) -> Result<Self, EstimationError> {
        antecedent_estimate::smoothed_dose::parse_smoothed_dose_sampling(name)?;
        Ok(Self::from_name(name))
    }
}
