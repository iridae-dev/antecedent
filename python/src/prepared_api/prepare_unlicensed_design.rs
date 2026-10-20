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

use super::{PrepareRequest, PyPreparedAnalysis};
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
                baseline: baseline.iter().map(|name| id(name)).collect::<Result<Vec<_>, _>>()?.into(),
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
}
