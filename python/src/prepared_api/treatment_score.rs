//! Treatment-propensity nuisance product of a prepared study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::array_fits::{
    TemporalScoreModel, TemporalTreatmentScoreSpec, TreatmentScoreSpec,
};
use pyo3::prelude::*;
use std::sync::Arc;

use super::PyPreparedAnalysis;
use crate::detach_catch;

#[pymethods]
impl PyPreparedAnalysis {
    /// Posterior of the treatment propensity on the study's certified adjustment set, on the
    /// data the study was prepared with: an immutable native owner. A posterior of a
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
    ) -> PyResult<crate::array_fit_api::PyArrayFit> {
        let inner = Arc::clone(&self.inner);
        let spec = TreatmentScoreSpec {
            prior_sd,
            draws,
            seed,
            clip,
            summary_rows: summary_rows.unwrap_or_default(),
        };
        let fit = detach_catch(py, move || {
            crate::array_fit_api::execute_score(seed, threads, "treatment_score_posterior", |_| {
                inner.treatment_score_posterior_retained(&spec)
            })
        })?;
        Ok(crate::array_fit_api::PyArrayFit::produced(fit))
    }

    /// Temporal treatment propensity on the independent-subject rows of a prepared panel study:
    /// An immutable native owner. `penalty` selects the ridge-penalized score; otherwise a
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
    ) -> PyResult<crate::array_fit_api::PyArrayFit> {
        let inner = Arc::clone(&self.inner);
        let model = match penalty {
            Some(penalty) => TemporalScoreModel::Penalized { penalty },
            None => TemporalScoreModel::Posterior { intercept_sd, slope_sd, draws, seed },
        };
        let spec = TemporalTreatmentScoreSpec { model, histories };
        let fit = detach_catch(py, move || {
            crate::array_fit_api::execute_score(seed, threads, "temporal_treatment_score", |ctx| {
                inner.temporal_treatment_score_owned_with_context(spec, ctx)
            })
        })?;
        Ok(crate::array_fit_api::PyArrayFit::produced(fit))
    }
}
