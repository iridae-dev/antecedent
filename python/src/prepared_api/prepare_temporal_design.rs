//! Prepare route of the sequentially randomized unit-period designs and regime survival.
//!
//! The outcome (and, for regime survival, the treatment, censoring and confounder columns) is
//! read from the table by name; the unit and period labels and the recorded assignment law are
//! part of the query. The design arrives as a JSON object whose `design` member names one of
//! `two_arm`, `multi_arm`, `factorial`, `interference`, `dose`, `regime_survival` and
//! `observational_excursion`. Inference
//! selects the excursion (or survival) estimator or its Bayesian counterpart.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::Study;
use antecedent_core::{
    CausalQuery, OutcomeMisclassification, TemporalAssignment, TemporalDesignFunctional,
    TemporalDesignQuery, TemporalDoseTarget,
};
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use serde::Deserialize;

use super::{PrepareRequest, PyPreparedAnalysis};
use crate::py_err;

/// A dose target as declared from Python.
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum DoseSpec {
    SmoothedCurve {
        grid: Vec<f64>,
    },
    SmoothedContrast {
        active: f64,
        control: f64,
    },
    LocalQuadratic {
        grid: Vec<f64>,
        #[serde(default)]
        derivative: bool,
        #[serde(default)]
        curvature_bound: Option<f64>,
    },
}

/// An assignment design as declared from Python.
#[derive(Deserialize)]
#[serde(tag = "design", rename_all = "snake_case", deny_unknown_fields)]
enum DesignSpec {
    TwoArm {
        assignment: Vec<bool>,
        probability: Vec<f64>,
    },
    MultiArm {
        assignment: Vec<usize>,
        probabilities: Vec<Vec<f64>>,
        arms: Vec<String>,
        #[serde(default)]
        categorical: bool,
        #[serde(default)]
        ordered: bool,
    },
    Factorial {
        first: Vec<bool>,
        first_probability: Vec<f64>,
        second: Vec<bool>,
        second_probability: Vec<f64>,
    },
    Interference {
        assignment: Vec<bool>,
        probability: Vec<f64>,
        clusters: Vec<String>,
    },
    Dose {
        dose: Vec<f64>,
        density: Vec<f64>,
        bandwidth: f64,
        target: DoseSpec,
    },
    RegimeSurvival {
        treatment: String,
        censored: String,
        confounders: Vec<String>,
        horizon: u32,
    },
    ObservationalExcursion {
        treatment: String,
        history: Vec<String>,
    },
    ObservationalRegime {
        treatment: String,
        confounders: Vec<String>,
        active: Vec<bool>,
        reference: Vec<bool>,
        method: String,
    },
}

fn labels(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|value| Arc::<str>::from(value.as_str())).collect::<Vec<_>>().into()
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Freeze a sequentially randomized unit-period design, or regime survival.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, units, periods, design, *, lag=0,
        quantile_level=None, min_probability=0.05, misclassification=None, seed=1,
        threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_temporal_design(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        units: Vec<String>,
        periods: Vec<i64>,
        design: String,
        lag: u32,
        quantile_level: Option<f64>,
        min_probability: f64,
        misclassification: Option<(u64, u64, u64, u64)>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("temporal design")?;
        let spec: DesignSpec = serde_json::from_str(&design)
            .map_err(|error| crate::value_err(format!("invalid temporal design: {error}")))?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let id = |name: &str| crate::graph_build::schema_var_id(data.schema(), name);
            let assignment = match spec {
                DesignSpec::TwoArm { assignment, probability } => TemporalAssignment::TwoArm {
                    assignment: assignment.into(),
                    probability: probability.into(),
                },
                DesignSpec::MultiArm { assignment, probabilities, arms, categorical, ordered } => {
                    TemporalAssignment::MultiArm {
                        assignment: assignment.into(),
                        probabilities: probabilities.into(),
                        arms: labels(&arms),
                        categorical,
                        ordered,
                    }
                }
                DesignSpec::Factorial { first, first_probability, second, second_probability } => {
                    TemporalAssignment::Factorial {
                        first: first.into(),
                        first_probability: first_probability.into(),
                        second: second.into(),
                        second_probability: second_probability.into(),
                    }
                }
                DesignSpec::Interference { assignment, probability, clusters } => {
                    TemporalAssignment::Interference {
                        assignment: assignment.into(),
                        probability: probability.into(),
                        clusters: labels(&clusters),
                    }
                }
                DesignSpec::Dose { dose, density, bandwidth, target } => TemporalAssignment::Dose {
                    dose: dose.into(),
                    density: density.into(),
                    bandwidth,
                    target: match target {
                        DoseSpec::SmoothedCurve { grid } => {
                            TemporalDoseTarget::SmoothedCurve { grid: grid.into() }
                        }
                        DoseSpec::SmoothedContrast { active, control } => {
                            TemporalDoseTarget::SmoothedContrast { active, control }
                        }
                        DoseSpec::LocalQuadratic { grid, derivative, curvature_bound } => {
                            TemporalDoseTarget::LocalQuadratic {
                                grid: grid.into(),
                                derivative,
                                curvature_bound,
                            }
                        }
                    },
                },
                DesignSpec::RegimeSurvival { treatment, censored, confounders, horizon } => {
                    TemporalAssignment::RegimeSurvival {
                        treatment: id(&treatment)?,
                        censored: id(&censored)?,
                        confounders: confounders
                            .iter()
                            .map(|name| id(name))
                            .collect::<Result<Vec<_>, _>>()?
                            .into(),
                        horizon,
                    }
                }
                DesignSpec::ObservationalExcursion { treatment, history } => {
                    TemporalAssignment::ObservationalExcursion {
                        treatment: id(&treatment)?,
                        history: history
                            .iter()
                            .map(|name| id(name))
                            .collect::<Result<Vec<_>, _>>()?
                            .into(),
                    }
                }
                DesignSpec::ObservationalRegime {
                    treatment,
                    confounders,
                    active,
                    reference,
                    method,
                } => TemporalAssignment::ObservationalRegime {
                    treatment: id(&treatment)?,
                    confounders: confounders
                        .iter()
                        .map(|name| id(name))
                        .collect::<Result<Vec<_>, _>>()?
                        .into(),
                    active: active.into(),
                    reference: reference.into(),
                    method: antecedent::TemporalRegimeMethod::from_name(&method).ok_or_else(
                        || crate::value_err(format!("unknown regime estimator {method:?}")),
                    )?,
                },
            };
            let query = TemporalDesignQuery {
                outcome: id(&outcome)?,
                units: labels(&units),
                periods: periods.into(),
                lag,
                assignment,
                functional: quantile_level.map_or(TemporalDesignFunctional::Mean, |level| {
                    TemporalDesignFunctional::Quantile { level }
                }),
                min_probability,
                outcome_misclassification: misclassification.map(
                    |(true_positive, false_negative, true_negative, false_positive)| {
                        OutcomeMisclassification {
                            true_positive,
                            false_negative,
                            true_negative,
                            false_positive,
                        }
                    },
                ),
            };
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let builder = Study::tabular(data).query(CausalQuery::TemporalDesign(query));
            req.finish_applied(builder, false)
        })
    }
}
