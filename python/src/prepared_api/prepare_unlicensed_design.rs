//! Prepare routes of the observational sequential contrast and the within-period history match.
//!
//! Both read their columns from the table by name; the query holds the declaration (regimes,
//! folds, floor, caliper). Inference selects the sequential contrast's normal interval or its
//! modular posterior; a Bayesian history match is the typed matching boundary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent_core::CausalQuery;
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{FrameData, PrepareRequest, PyPreparedAnalysis};
use crate::py_err;

#[pymethods]
impl PyPreparedAnalysis {
    /// Freeze an observational two-decision contrast of two static regimes.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, first_treatment, second_treatment, baseline,
        intermediate, active, control, fold_of_row, fold_count, min_probability,
        *, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_sequential_contrast(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        first_treatment: String,
        second_treatment: String,
        baseline: Vec<String>,
        intermediate: Vec<String>,
        active: (u8, u8),
        control: (u8, u8),
        fold_of_row: Vec<u32>,
        fold_count: u32,
        min_probability: f64,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("sequential contrast")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let id = |name: &str| crate::graph_build::schema_var_id(data.schema(), name);
            let query = antecedent_core::SequentialContrastQuery {
                outcome: id(&outcome)?,
                first_treatment: id(&first_treatment)?,
                second_treatment: id(&second_treatment)?,
                baseline: baseline
                    .iter()
                    .map(|name| id(name))
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
                intermediate: intermediate
                    .iter()
                    .map(|name| id(name))
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
                active: [active.0, active.1],
                control: [control.0, control.1],
                fold_of_row: fold_of_row.into(),
                fold_count,
                min_probability,
            };
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::SequentialContrast(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a within-period nearest-neighbor contrast on pre-outcome histories.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, treated, history, period=None, weight=None,
        caliper=None, *, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_history_match(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        treated: String,
        history: Vec<String>,
        period: Option<String>,
        weight: Option<String>,
        caliper: Option<f64>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("history match")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let id = |name: &str| crate::graph_build::schema_var_id(data.schema(), name);
            let query = antecedent_core::HistoryMatchQuery {
                outcome: id(&outcome)?,
                treated: id(&treated)?,
                history: history.iter().map(|name| id(name)).collect::<Result<Vec<_>, _>>()?.into(),
                period: period.as_deref().map(id).transpose()?,
                weight: weight.as_deref().map(id).transpose()?,
                caliper,
            };
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::HistoryMatch(query));
            req.finish_applied(builder, false)
        })
    }

    /// Freeze bounds on the average effect of a binary treatment under a binary instrument.
    ///
    /// The instrument, treatment and outcome are 0/1 columns read by name; `weight` names an
    /// optional sampling-weight column. The answer is an identified set, never a point.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, treatment, instrument, weight=None, *,
        seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_iv_bounds(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        treatment: String,
        instrument: String,
        weight: Option<String>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("IV bounds")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let id = |name: &str| crate::graph_build::schema_var_id(data.schema(), name);
            let mut query = antecedent_core::IvBoundsQuery::new(
                id(&outcome)?,
                id(&treatment)?,
                id(&instrument)?,
            );
            if let Some(weight) = weight.as_deref() {
                query = query.with_weight(id(weight)?);
            }
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let builder = Study::tabular(data).query(CausalQuery::IvBounds(query));
            req.finish_applied(builder, false)
        })
    }
}

/// The functional of a Python association declaration.
fn association_functional(
    functional: &str,
    folds: Option<u32>,
) -> PyResult<antecedent_core::AssociationFunctional> {
    use antecedent_core::{AssociationFunctional, AssociationMeasure};
    Ok(match functional {
        "auto" => AssociationFunctional::Auto,
        "conditional" => AssociationFunctional::Conditional,
        "coefficients" => AssociationFunctional::Coefficients,
        "predictive" => AssociationFunctional::Predictive { folds: folds.unwrap_or(5) },
        other => AssociationFunctional::Measure(AssociationMeasure::from_name(other).ok_or_else(
            || {
                crate::value_err(format!(
                    "unknown association functional `{other}`; use auto, conditional, \
                     coefficients, predictive, pearson_correlation or distance_correlation"
                ))
            },
        )?),
    })
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Freeze a noncausal association of an outcome with predictors.
    ///
    /// The columns are read by name from a table (`series=false`) or are the consecutive points of
    /// one series (`series=true`). The declared selection rule picks the analysis from the outcome
    /// scale and the functional; the answer is a description, never an intervention effect.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, predictors, conditioning, functional, *,
        folds=None, outcome_scale=None, weight=None, refused_after=None, series=false, seed=1,
        threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_association(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        predictors: Vec<String>,
        conditioning: Vec<String>,
        functional: String,
        folds: Option<u32>,
        outcome_scale: Option<String>,
        weight: Option<String>,
        refused_after: Option<String>,
        series: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("association")?;
        let functional = association_functional(&functional, folds)?;
        let scale = match outcome_scale.as_deref() {
            None | Some("schema") => antecedent_core::AssociationOutcome::Schema,
            Some(other) => antecedent_core::AssociationOutcome::from_name(other)
                .ok_or_else(|| crate::value_err(format!("unknown outcome scale `{other}`")))?,
        };
        let build = move |req: PrepareRequest,
                          schema: &antecedent_core::CausalSchema|
              -> PyResult<(PrepareRequest, antecedent_core::AssociationQuery)> {
            let id = |name: &str| crate::graph_build::schema_var_id(schema, name);
            let predictors =
                predictors.iter().map(|name| id(name)).collect::<Result<Vec<_>, _>>()?;
            let conditioning =
                conditioning.iter().map(|name| id(name)).collect::<Result<Vec<_>, _>>()?;
            let mut query = antecedent_core::AssociationQuery::new(id(&outcome)?, predictors)
                .with_conditioning(conditioning)
                .with_functional(functional)
                .with_outcome_scale(scale);
            if let Some(weight) = weight.as_deref() {
                query = query.with_weight(id(weight)?);
            }
            if let Some(reason) = refused_after.as_deref() {
                query = query.following_refusal(reason);
            }
            query.validate().map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            Ok((req, query))
        };
        if series {
            let input = req.load_frame(py, columns, None)?;
            let FrameData::Series(data) = input.materialize()? else {
                return Err(crate::refusal(
                    antecedent_core::reason_code!("data_modality_not_licensed"),
                    "an association over time reads the consecutive points of one series",
                ));
            };
            req.off_gil(py, move |req| {
                let (req, query) = build(req, data.schema())?;
                let builder = Study::series(data).query(CausalQuery::Association(query));
                req.finish_applied(builder, true)
            })
        } else {
            let data = req.load_columns(py, columns)?;
            req.off_gil(py, move |req| {
                let (req, query) = build(req, data.schema())?;
                let builder = Study::tabular(data).query(CausalQuery::Association(query));
                req.finish_applied(builder, false)
            })
        }
    }
}
