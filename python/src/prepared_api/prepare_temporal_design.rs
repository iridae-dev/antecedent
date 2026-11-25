//! Prepare route of the sequentially randomized unit-period designs and regime survival.
//!
//! The outcome (and, for regime survival, the treatment, censoring and confounder columns) is
//! read from the table by name; the unit and period labels and the recorded assignment law are
//! part of the query. The design arrives as a JSON object whose `design` member names one of
//! `two_arm`, `multi_arm`, `factorial`, `interference`, `dose`, `regime_survival` and
//! `observational_excursion`. Inference
//! selects the excursion (or survival) estimator or its Bayesian counterpart.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::Study;
use antecedent::data::TableView;
use antecedent::query::{
    CausalQuery, OutcomeMisclassification, ProbabilityRows, TemporalAssignment,
    TemporalDesignFunctional, TemporalDesignQuery, TemporalDoseTarget,
};
use numpy::PyReadonlyArray1;
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
    #[serde(skip)]
    Native(TemporalAssignment),
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

fn buffer_item<'py>(buffers: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    buffers
        .get_item(name)?
        .ok_or_else(|| crate::value_err(format!("missing temporal design column {name:?}")))
}

fn numeric_buffer<'py, T: numpy::Element + Copy>(
    buffers: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Arc<[T]>> {
    let array: PyReadonlyArray1<'py, T> = buffer_item(buffers, name)?.extract()?;
    Ok(crate::ingest::snapshot_shared(&array))
}

#[allow(clippy::float_cmp, reason = "assignment flags are exactly zero or one")]
fn flag_buffer(buffers: &Bound<'_, PyDict>, name: &str) -> PyResult<Arc<[bool]>> {
    let value = buffer_item(buffers, name)?;
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, bool>>() {
        return Ok(crate::ingest::snapshot_shared(&array));
    }
    let array: PyReadonlyArray1<'_, f64> = value.extract()?;
    let view = array.as_array();
    if let Some(value) = view.iter().find(|&&value| value != 0.0 && value != 1.0) {
        return Err(crate::value_err(format!("{name} must be 0/1; got {value}")));
    }
    let values: Arc<[bool]> = (0..view.len()).map(|row| view[row] == 1.0).collect();
    crate::ingest::record_input_copy::<bool>(values.len());
    Ok(values)
}

#[allow(clippy::cast_possible_truncation, reason = "the range and integrality are checked")]
fn periods_buffer(periods: &Bound<'_, PyAny>) -> PyResult<Arc<[i64]>> {
    if let Ok(array) = periods.extract::<PyReadonlyArray1<'_, i64>>() {
        return Ok(crate::ingest::snapshot_shared(&array));
    }
    if let Ok(array) = periods.extract::<PyReadonlyArray1<'_, f64>>() {
        let view = array.as_array();
        if let Some(value) = view.iter().find(|&&value| {
            value.fract() != 0.0
                || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&value)
        }) {
            return Err(crate::value_err(format!("period must be an integer; got {value}")));
        }
        let values: Arc<[i64]> = (0..view.len()).map(|row| view[row] as i64).collect();
        crate::ingest::record_input_copy::<i64>(values.len());
        return Ok(values);
    }
    Ok(periods.extract::<Vec<i64>>()?.into())
}

fn row_probabilities(buffers: &Bound<'_, PyDict>) -> PyResult<ProbabilityRows> {
    let value = buffers
        .get_item("probabilities")?
        .ok_or_else(|| crate::value_err("missing multi-arm probability columns"))?;
    let arrays: Vec<PyReadonlyArray1<'_, f64>> = value.extract()?;
    let columns: Vec<_> = arrays.iter().map(|array| array.as_array()).collect();
    let rows = columns.first().map_or(0, |column| column.len());
    let stride = columns.len();
    if columns.iter().any(|column| column.len() != rows) {
        return Err(crate::value_err("multi-arm probability columns must have equal row counts"));
    }
    let cells =
        rows.checked_mul(stride).ok_or_else(|| crate::value_err("probability size overflow"))?;
    let values: Arc<[f64]> =
        (0..cells).map(|index| columns[index % stride][index / stride]).collect();
    for _ in columns {
        crate::ingest::record_input_copy::<f64>(rows);
    }
    Ok(ProbabilityRows::new(values, stride))
}

fn dose_target(target: DoseSpec) -> TemporalDoseTarget {
    match target {
        DoseSpec::SmoothedCurve { grid } => TemporalDoseTarget::SmoothedCurve { grid: grid.into() },
        DoseSpec::SmoothedContrast { active, control } => {
            TemporalDoseTarget::SmoothedContrast { active, control }
        }
        DoseSpec::LocalQuadratic { grid, derivative, curvature_bound } => {
            TemporalDoseTarget::LocalQuadratic { grid: grid.into(), derivative, curvature_bound }
        }
    }
}

fn bind_numeric_design(spec: DesignSpec, buffers: &Bound<'_, PyDict>) -> PyResult<DesignSpec> {
    let assignment = match spec {
        DesignSpec::TwoArm { .. } => TemporalAssignment::TwoArm {
            assignment: flag_buffer(buffers, "assignment")?,
            probability: numeric_buffer(buffers, "probability")?,
        },
        DesignSpec::MultiArm { arms, categorical, ordered, .. } => TemporalAssignment::MultiArm {
            assignment: numeric_buffer(buffers, "assignment")?,
            probabilities: row_probabilities(buffers)?,
            arms: labels(&arms),
            categorical,
            ordered,
        },
        DesignSpec::Factorial { .. } => TemporalAssignment::Factorial {
            first: flag_buffer(buffers, "first")?,
            first_probability: numeric_buffer(buffers, "first_probability")?,
            second: flag_buffer(buffers, "second")?,
            second_probability: numeric_buffer(buffers, "second_probability")?,
        },
        DesignSpec::Interference { clusters, .. } => TemporalAssignment::Interference {
            assignment: flag_buffer(buffers, "assignment")?,
            probability: numeric_buffer(buffers, "probability")?,
            clusters: labels(&clusters),
        },
        DesignSpec::Dose { bandwidth, target, .. } => TemporalAssignment::Dose {
            dose: numeric_buffer(buffers, "dose")?,
            density: numeric_buffer(buffers, "density")?,
            bandwidth,
            target: dose_target(target),
        },
        other => return Ok(other),
    };
    Ok(DesignSpec::Native(assignment))
}

fn labels(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|value| Arc::<str>::from(value.as_str())).collect::<Vec<_>>().into()
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Freeze a sequentially randomized unit-period design, or regime survival.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, units, periods, design, *, lag=0,
        quantile_level=None, min_probability=0.05, misclassification=None, buffers=None, seed=1,
        threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_temporal_design(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        units: Vec<String>,
        periods: &Bound<'_, PyAny>,
        design: String,
        lag: u32,
        quantile_level: Option<f64>,
        min_probability: f64,
        misclassification: Option<(u64, u64, u64, u64)>,
        buffers: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        crate::catch_ffi(|| {
            let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
            req.refuse_design("temporal design")?;
            let spec: DesignSpec = serde_json::from_str(&design)
                .map_err(|error| crate::value_err(format!("invalid temporal design: {error}")))?;
            let spec = match &buffers {
                Some(buffers) => bind_numeric_design(spec, buffers)?,
                None => spec,
            };
            let periods = periods_buffer(periods)?;
            let data = req.load_columns(py, columns)?;
            req.off_gil(py, move |req| {
                let id = |name: &str| crate::graph_build::schema_var_id(data.schema(), name);
                let assignment = match spec {
                    DesignSpec::Native(assignment) => assignment,
                    DesignSpec::TwoArm { assignment, probability } => TemporalAssignment::TwoArm {
                        assignment: assignment.into(),
                        probability: probability.into(),
                    },
                    DesignSpec::MultiArm {
                        assignment,
                        probabilities,
                        arms,
                        categorical,
                        ordered,
                    } => TemporalAssignment::MultiArm {
                        assignment: assignment.into(),
                        probabilities: probabilities.into(),
                        arms: labels(&arms),
                        categorical,
                        ordered,
                    },
                    DesignSpec::Factorial {
                        first,
                        first_probability,
                        second,
                        second_probability,
                    } => TemporalAssignment::Factorial {
                        first: first.into(),
                        first_probability: first_probability.into(),
                        second: second.into(),
                        second_probability: second_probability.into(),
                    },
                    DesignSpec::Interference { assignment, probability, clusters } => {
                        TemporalAssignment::Interference {
                            assignment: assignment.into(),
                            probability: probability.into(),
                            clusters: labels(&clusters),
                        }
                    }
                    DesignSpec::Dose { dose, density, bandwidth, target } => {
                        TemporalAssignment::Dose {
                            dose: dose.into(),
                            density: density.into(),
                            bandwidth,
                            target: dose_target(target),
                        }
                    }
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
                    periods,
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
        })
    }
}
