//! Wire form of the sequentially randomized unit-period design query.
//!
//! Every row-aligned array keeps its order, and decoding re-validates the query, so a stored
//! design that its own validation refuses never reaches a study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    OutcomeMisclassification, TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery,
    TemporalDoseTarget, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::{IoError, fit_u64};

/// Wire form of [`TemporalDoseTarget`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum TemporalDoseTargetWire {
    /// Kernel-smoothed curve.
    SmoothedCurve {
        /// Grid doses.
        grid: Vec<f64>,
    },
    /// Kernel-smoothed contrast.
    SmoothedContrast {
        /// Active dose.
        active: f64,
        /// Control dose.
        control: f64,
    },
    /// Local-quadratic grid.
    LocalQuadratic {
        /// Grid doses.
        grid: Vec<f64>,
        /// First derivative instead of the level.
        derivative: bool,
        /// Declared third-derivative bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        curvature_bound: Option<f64>,
    },
}

/// Wire form of [`TemporalAssignment`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum TemporalAssignmentWire {
    /// Two arms.
    TwoArm {
        /// Realized assignment.
        assignment: Vec<bool>,
        /// Known probability of treatment.
        probability: Vec<f64>,
    },
    /// Several arms or categorical levels.
    MultiArm {
        /// Realized arm index.
        assignment: Vec<u64>,
        /// Probability row per table row.
        probabilities: Vec<Vec<f64>>,
        /// Arm labels.
        arms: Vec<String>,
        /// Categorical levels.
        categorical: bool,
        /// Ordered levels.
        ordered: bool,
    },
    /// Two-factor factorial.
    Factorial {
        /// First factor.
        first: Vec<bool>,
        /// First factor probability.
        first_probability: Vec<f64>,
        /// Second factor.
        second: Vec<bool>,
        /// Second factor probability.
        second_probability: Vec<f64>,
    },
    /// Within-cluster interference.
    Interference {
        /// Realized assignment.
        assignment: Vec<bool>,
        /// Known probability of treatment.
        probability: Vec<f64>,
        /// Interference cluster labels.
        clusters: Vec<String>,
    },
    /// Continuous dose.
    Dose {
        /// Realized dose.
        dose: Vec<f64>,
        /// Density of the realized dose.
        density: Vec<f64>,
        /// Gaussian kernel bandwidth.
        bandwidth: f64,
        /// Dose target.
        target: TemporalDoseTargetWire,
    },
    /// Static-regime survival.
    RegimeSurvival {
        /// Treatment column.
        treatment: u32,
        /// Censoring column.
        censored: u32,
        /// Confounder columns.
        confounders: Vec<u32>,
        /// Horizon in periods.
        horizon: u32,
    },
}

/// Wire form of [`TemporalDesignQuery`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalDesignQueryWire {
    /// Outcome column.
    pub outcome: u32,
    /// Unit label per row.
    pub units: Vec<String>,
    /// Period per row.
    pub periods: Vec<i64>,
    /// Decision-to-outcome lag.
    pub lag: u32,
    /// Assignment record.
    pub assignment: TemporalAssignmentWire,
    /// Quantile level; absent for the mean functional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantile_level: Option<f64>,
    /// Positivity floor.
    pub min_probability: f64,
    /// Validation counts `[true positive, false negative, true negative, false positive]` of a
    /// misclassified binary outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_misclassification: Option<[u64; 4]>,
}

fn texts(values: &[Arc<str>]) -> Vec<String> {
    values.iter().map(std::string::ToString::to_string).collect()
}

fn arcs(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|value| Arc::<str>::from(value.as_str())).collect::<Vec<_>>().into()
}

impl TemporalDesignQueryWire {
    /// Wire form of a domain query.
    ///
    /// # Errors
    /// An arm index that does not fit the wire integer.
    pub fn from_domain(query: &TemporalDesignQuery) -> Result<Self, IoError> {
        let assignment = match &query.assignment {
            TemporalAssignment::TwoArm { assignment, probability } => {
                TemporalAssignmentWire::TwoArm {
                    assignment: assignment.to_vec(),
                    probability: probability.to_vec(),
                }
            }
            TemporalAssignment::MultiArm {
                assignment,
                probabilities,
                arms,
                categorical,
                ordered,
            } => TemporalAssignmentWire::MultiArm {
                assignment: assignment
                    .iter()
                    .map(|arm| fit_u64(*arm))
                    .collect::<Result<Vec<_>, _>>()?,
                probabilities: probabilities.to_vec(),
                arms: texts(arms),
                categorical: *categorical,
                ordered: *ordered,
            },
            TemporalAssignment::Factorial {
                first,
                first_probability,
                second,
                second_probability,
            } => TemporalAssignmentWire::Factorial {
                first: first.to_vec(),
                first_probability: first_probability.to_vec(),
                second: second.to_vec(),
                second_probability: second_probability.to_vec(),
            },
            TemporalAssignment::Interference { assignment, probability, clusters } => {
                TemporalAssignmentWire::Interference {
                    assignment: assignment.to_vec(),
                    probability: probability.to_vec(),
                    clusters: texts(clusters),
                }
            }
            TemporalAssignment::Dose { dose, density, bandwidth, target } => {
                TemporalAssignmentWire::Dose {
                    dose: dose.to_vec(),
                    density: density.to_vec(),
                    bandwidth: *bandwidth,
                    target: match target {
                        TemporalDoseTarget::SmoothedCurve { grid } => {
                            TemporalDoseTargetWire::SmoothedCurve { grid: grid.to_vec() }
                        }
                        TemporalDoseTarget::SmoothedContrast { active, control } => {
                            TemporalDoseTargetWire::SmoothedContrast {
                                active: *active,
                                control: *control,
                            }
                        }
                        TemporalDoseTarget::LocalQuadratic {
                            grid,
                            derivative,
                            curvature_bound,
                        } => TemporalDoseTargetWire::LocalQuadratic {
                            grid: grid.to_vec(),
                            derivative: *derivative,
                            curvature_bound: *curvature_bound,
                        },
                    },
                }
            }
            TemporalAssignment::RegimeSurvival { treatment, censored, confounders, horizon } => {
                TemporalAssignmentWire::RegimeSurvival {
                    treatment: treatment.raw(),
                    censored: censored.raw(),
                    confounders: vars_to_raw(confounders),
                    horizon: *horizon,
                }
            }
        };
        Ok(Self {
            outcome: query.outcome.raw(),
            units: texts(&query.units),
            periods: query.periods.to_vec(),
            lag: query.lag,
            assignment,
            quantile_level: match query.functional {
                TemporalDesignFunctional::Mean => None,
                TemporalDesignFunctional::Quantile { level } => Some(level),
            },
            min_probability: query.min_probability,
            outcome_misclassification: query.outcome_misclassification.map(|counts| {
                [
                    counts.true_positive,
                    counts.false_negative,
                    counts.true_negative,
                    counts.false_positive,
                ]
            }),
        })
    }

    /// The domain query, validated.
    ///
    /// # Errors
    /// An arm index that does not fit, or a query its own validation refuses.
    #[allow(
        clippy::too_many_lines,
        reason = "one exhaustive match converting every assignment wire arm to its domain arm"
    )]
    pub fn to_domain(&self) -> Result<TemporalDesignQuery, IoError> {
        let assignment = match &self.assignment {
            TemporalAssignmentWire::TwoArm { assignment, probability } => {
                TemporalAssignment::TwoArm {
                    assignment: assignment.as_slice().into(),
                    probability: probability.as_slice().into(),
                }
            }
            TemporalAssignmentWire::MultiArm {
                assignment,
                probabilities,
                arms,
                categorical,
                ordered,
            } => TemporalAssignment::MultiArm {
                assignment: assignment
                    .iter()
                    .map(|arm| usize::try_from(*arm).map_err(|_| IoError::TooLarge))
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
                probabilities: probabilities.clone().into(),
                arms: arcs(arms),
                categorical: *categorical,
                ordered: *ordered,
            },
            TemporalAssignmentWire::Factorial {
                first,
                first_probability,
                second,
                second_probability,
            } => TemporalAssignment::Factorial {
                first: first.as_slice().into(),
                first_probability: first_probability.as_slice().into(),
                second: second.as_slice().into(),
                second_probability: second_probability.as_slice().into(),
            },
            TemporalAssignmentWire::Interference { assignment, probability, clusters } => {
                TemporalAssignment::Interference {
                    assignment: assignment.as_slice().into(),
                    probability: probability.as_slice().into(),
                    clusters: arcs(clusters),
                }
            }
            TemporalAssignmentWire::Dose { dose, density, bandwidth, target } => {
                TemporalAssignment::Dose {
                    dose: dose.as_slice().into(),
                    density: density.as_slice().into(),
                    bandwidth: *bandwidth,
                    target: match target {
                        TemporalDoseTargetWire::SmoothedCurve { grid } => {
                            TemporalDoseTarget::SmoothedCurve { grid: grid.as_slice().into() }
                        }
                        TemporalDoseTargetWire::SmoothedContrast { active, control } => {
                            TemporalDoseTarget::SmoothedContrast {
                                active: *active,
                                control: *control,
                            }
                        }
                        TemporalDoseTargetWire::LocalQuadratic {
                            grid,
                            derivative,
                            curvature_bound,
                        } => TemporalDoseTarget::LocalQuadratic {
                            grid: grid.as_slice().into(),
                            derivative: *derivative,
                            curvature_bound: *curvature_bound,
                        },
                    },
                }
            }
            TemporalAssignmentWire::RegimeSurvival {
                treatment,
                censored,
                confounders,
                horizon,
            } => TemporalAssignment::RegimeSurvival {
                treatment: VariableId::from_raw(*treatment),
                censored: VariableId::from_raw(*censored),
                confounders: vars_from_raw(confounders),
                horizon: *horizon,
            },
        };
        let query = TemporalDesignQuery {
            outcome: VariableId::from_raw(self.outcome),
            units: arcs(&self.units),
            periods: self.periods.as_slice().into(),
            lag: self.lag,
            assignment,
            functional: self.quantile_level.map_or(TemporalDesignFunctional::Mean, |level| {
                TemporalDesignFunctional::Quantile { level }
            }),
            min_probability: self.min_probability,
            outcome_misclassification: self.outcome_misclassification.map(|counts| {
                OutcomeMisclassification {
                    true_positive: counts[0],
                    false_negative: counts[1],
                    true_negative: counts[2],
                    false_positive: counts[3],
                }
            }),
        };
        query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
        Ok(query)
    }

    /// Every table column the query names.
    #[must_use]
    pub fn variable_ids(&self) -> Vec<u32> {
        let mut ids = vec![self.outcome];
        if let TemporalAssignmentWire::RegimeSurvival { treatment, censored, confounders, .. } =
            &self.assignment
        {
            ids.push(*treatment);
            ids.push(*censored);
            ids.extend(confounders.iter().copied());
        }
        ids
    }

    /// The design name the identity labels record.
    #[must_use]
    pub fn design_name(&self) -> &'static str {
        match &self.assignment {
            TemporalAssignmentWire::TwoArm { .. } => "two_arm",
            TemporalAssignmentWire::MultiArm { categorical: true, .. } => "categorical",
            TemporalAssignmentWire::MultiArm { .. } => "multi_arm",
            TemporalAssignmentWire::Factorial { .. } => "factorial",
            TemporalAssignmentWire::Interference { .. } => "interference",
            TemporalAssignmentWire::Dose { .. } => "dose",
            TemporalAssignmentWire::RegimeSurvival { .. } => "regime_survival",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_two_arm_design_round_trips() {
        let query = TemporalDesignQuery {
            outcome: VariableId::from_raw(1),
            units: arcs(&["a".into(), "a".into(), "b".into(), "b".into()]),
            periods: Arc::from([0_i64, 1, 0, 1]),
            lag: 1,
            assignment: TemporalAssignment::TwoArm {
                assignment: Arc::from([true, false, false, true]),
                probability: Arc::from([0.5, 0.5, 0.5, 0.5]),
            },
            functional: TemporalDesignFunctional::Quantile { level: 0.5 },
            min_probability: 0.05,
            outcome_misclassification: None,
        };
        let wire = TemporalDesignQueryWire::from_domain(&query).unwrap();
        assert_eq!(wire.to_domain().unwrap(), query);
        assert_eq!(wire.variable_ids(), vec![1]);
        assert_eq!(wire.design_name(), "two_arm");
    }
}
