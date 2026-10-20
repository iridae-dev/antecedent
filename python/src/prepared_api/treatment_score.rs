//! Treatment-propensity nuisance product of a prepared study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::array_fits::{
    TemporalScoreModel, TemporalTreatmentScoreSpec, TreatmentScoreSpec,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

use super::PyPreparedAnalysis;
use crate::{detach_catch, py_err};

#[pymethods]
impl PyPreparedAnalysis {
    /// Posterior of the treatment propensity on the study's certified adjustment set, on the
    /// data the study was prepared with: `(report json, artifact bytes)`. A posterior of a
    /// nuisance, never of an effect.
    #[pyo3(signature = (*, prior_sd=2.0, draws=4000, seed=0, clip=0.05, summary_rows=None))]
    fn treatment_score_posterior(
        &self,
        py: Python<'_>,
        prior_sd: f64,
        draws: usize,
        seed: u64,
        clip: f64,
        summary_rows: Option<Vec<usize>>,
    ) -> PyResult<(String, Py<PyBytes>)> {
        let inner = Arc::clone(&self.inner);
        let spec = TreatmentScoreSpec {
            prior_sd,
            draws,
            seed,
            clip,
            summary_rows: summary_rows.unwrap_or_default(),
        };
        let fit = detach_catch(py, move || {
            inner.treatment_score_posterior_retained(&spec).map_err(py_err)
        })?;
        crate::array_fit_api::report_and_artifact(py, &fit)
    }

    /// Temporal treatment propensity on the independent-subject rows of a prepared panel study:
    /// `(report json, artifact bytes)`. `penalty` selects the ridge-penalized score; otherwise a
    /// Laplace posterior under `N(0, sd^2)` priors is fitted. A nuisance, never an effect.
    #[pyo3(signature = (histories, *, penalty=None, intercept_sd=1.0, slope_sd=1.0, draws=4000, seed=0))]
    fn temporal_treatment_score(
        &self,
        py: Python<'_>,
        histories: Vec<Vec<f64>>,
        penalty: Option<f64>,
        intercept_sd: f64,
        slope_sd: f64,
        draws: usize,
        seed: u64,
    ) -> PyResult<(String, Py<PyBytes>)> {
        let inner = Arc::clone(&self.inner);
        let model = match penalty {
            Some(penalty) => TemporalScoreModel::Penalized { penalty },
            None => TemporalScoreModel::Posterior { intercept_sd, slope_sd, draws, seed },
        };
        let spec = TemporalTreatmentScoreSpec { model, histories };
        let fit = detach_catch(py, move || inner.temporal_treatment_score(&spec).map_err(py_err))?;
        crate::array_fit_api::report_and_artifact(py, &fit)
    }
}
