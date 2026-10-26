//! Treatment-propensity nuisance product of a prepared study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::array_fits::{
    ArrayFit, TemporalScoreModel, TemporalTreatmentScoreSpec, TreatmentScoreSpec,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

use super::PyPreparedAnalysis;
use crate::{detach_catch, py_err};

/// Admit the real numerical work and preserve operational failures across the
/// legacy array-fit receipt layer, without changing its scientific encoding.
fn execute_score(
    seed: u64,
    threads: Option<u32>,
    stage: &'static str,
    fit: impl FnOnce(&antecedent_core::ExecutionContext) -> Result<ArrayFit, antecedent::CausalError>,
) -> PyResult<ArrayFit> {
    let ctx = crate::py_execution_context(seed, crate::resolve_user_threads(threads));
    let check_cancel = || {
        if ctx.cancellation.is_cancelled() {
            Err(py_err(antecedent::CausalError::Cancelled { stage }))
        } else {
            Ok(())
        }
    };
    check_cancel()?;
    if let Some(progress) = &ctx.progress {
        progress.report(0.0, stage);
    }
    check_cancel()?;
    let (result, failure) =
        antecedent_core::numerical_admission::with_context_and_failure(&ctx, || fit(&ctx));
    check_cancel()?;
    if let (true, Some(error)) = (result.is_err(), failure) {
        return Err(match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                py_err(antecedent::CausalError::Cancelled { stage })
            }
            error => crate::CausalResourceError::new_err(error.to_string()),
        });
    }
    let fit = result.map_err(py_err)?;
    if let Some(progress) = &ctx.progress {
        progress.report(1.0, stage);
    }
    check_cancel()?;
    Ok(fit)
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Posterior of the treatment propensity on the study's certified adjustment set, on the
    /// data the study was prepared with: `(report json, artifact bytes)`. A posterior of a
    /// nuisance, never of an effect.
    #[pyo3(signature = (*, prior_sd=2.0, draws=4000, seed=0, clip=0.05, summary_rows=None, threads=None))]
    fn treatment_score_posterior(
        &self,
        py: Python<'_>,
        prior_sd: f64,
        draws: usize,
        seed: u64,
        clip: f64,
        summary_rows: Option<Vec<usize>>,
        threads: Option<u32>,
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
            execute_score(seed, threads, "treatment_score_posterior", |_| {
                inner.treatment_score_posterior_retained(&spec)
            })
        })?;
        crate::array_fit_api::report_and_artifact(py, &fit)
    }

    /// Temporal treatment propensity on the independent-subject rows of a prepared panel study:
    /// `(report json, artifact bytes)`. `penalty` selects the ridge-penalized score; otherwise a
    /// Laplace posterior under `N(0, sd^2)` priors is fitted. A nuisance, never an effect.
    #[pyo3(signature = (histories, *, penalty=None, intercept_sd=1.0, slope_sd=1.0, draws=4000, seed=0, threads=None))]
    fn temporal_treatment_score(
        &self,
        py: Python<'_>,
        histories: Vec<Vec<f64>>,
        penalty: Option<f64>,
        intercept_sd: f64,
        slope_sd: f64,
        draws: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<(String, Py<PyBytes>)> {
        let inner = Arc::clone(&self.inner);
        let model = match penalty {
            Some(penalty) => TemporalScoreModel::Penalized { penalty },
            None => TemporalScoreModel::Posterior { intercept_sd, slope_sd, draws, seed },
        };
        let spec = TemporalTreatmentScoreSpec { model, histories };
        let fit = detach_catch(py, move || {
            execute_score(seed, threads, "temporal_treatment_score", |ctx| {
                inner.temporal_treatment_score_with_context(&spec, ctx)
            })
        })?;
        crate::array_fit_api::report_and_artifact(py, &fit)
    }
}
