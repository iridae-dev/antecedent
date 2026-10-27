//! faer-backed logistic factories. Hidden behind [`crate::LearnerSpec::Logistic`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmFit, GlmOptions, LeastSquaresWorkspace, fit_glm,
    fit_glm_ridge, fit_glm_ridge_weighted, fit_logistic_weighted,
};

use crate::dense::{gather_physical, materialize_dense_colmajor, require_binary_labels};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{FittedPredictor, LearnerCapabilities, LearnerFactory, PredictionTask};

/// Logistic regression via [`fit_glm`] (`BinomialLogit`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LogisticLearner {
    /// Accept a converged, unseparated fit whose probabilities reach within `1e-8` of 0 or 1;
    /// set only for a propensity nuisance whose declared overlap policy governs those rows.
    pub(crate) admit_boundary_saturation: bool,
}

/// Ridge-penalized logistic regression via [`fit_glm_ridge`]: the probability-task form of
/// a ridge learner. The penalty applies to every non-intercept coefficient; a constant
/// first design column is the intercept and is left unpenalized.
#[derive(Clone, Copy, Debug)]
pub struct RidgeLogisticLearner {
    lambda: f64,
    admit_boundary_saturation: bool,
}

impl RidgeLogisticLearner {
    /// Penalized logistic factory. `lambda` must be finite and positive.
    ///
    /// # Errors
    ///
    /// [`LearnError::Shape`] for a non-finite or non-positive penalty.
    pub fn new(lambda: f64) -> Result<Self, LearnError> {
        if lambda.is_finite() && lambda > 0.0 {
            Ok(Self { lambda, admit_boundary_saturation: false })
        } else {
            Err(LearnError::Shape { message: "logistic ridge penalty must be finite and > 0" })
        }
    }

    /// This factory admitting a boundary-saturated fit when `admit` holds (see
    /// [`antecedent_stats::GlmOptions::admit_boundary_saturation`]).
    #[must_use]
    pub const fn admitting_boundary_saturation(self, admit: bool) -> Self {
        Self { admit_boundary_saturation: admit, ..self }
    }
}

impl LogisticLearner {
    /// Logistic factory, or a typed refusal if `task` is not probability.
    ///
    /// # Errors
    ///
    /// [`LearnError::TaskMismatch`] when `task` is [`PredictionTask::Regression`].
    #[cfg(test)]
    pub fn for_task(task: PredictionTask) -> Result<Self, LearnError> {
        match task {
            PredictionTask::BinaryProbability => Ok(Self::default()),
            PredictionTask::Regression => Err(LearnError::TaskMismatch {
                requested: PredictionTask::Regression,
                supported: PredictionTask::BinaryProbability,
            }),
        }
    }
}

fn capabilities(sample_weights: bool) -> LearnerCapabilities {
    LearnerCapabilities {
        binary_probability: true,
        sample_weights,
        deterministic_seed: true,
        ..LearnerCapabilities::none()
    }
}

fn fit_logistic(
    x: DesignView<'_>,
    y: TargetView<'_>,
    weights: Option<&[f64]>,
    ridge: Option<f64>,
    admit_boundary_saturation: bool,
) -> Result<GlmFit, LearnError> {
    let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
    if y.len() != x.physical_nrows() {
        return Err(LearnError::Shape { message: "target length != physical rows" });
    }
    let gathered_y = gather_physical(y.values(), x, nrows)?;
    require_binary_labels(&gathered_y)?;
    let mut ws = LeastSquaresWorkspace::default();
    let design_ref = GlmDesignRef { x_colmajor: &design, nrows, ncols, y: &gathered_y };
    let fit = match ridge {
        None if weights.is_some() => {
            let w = weights.unwrap_or(&[]);
            if w.len() != x.physical_nrows() {
                return Err(LearnError::Shape { message: "weights length != physical rows" });
            }
            let gathered_w = gather_physical(w, x, nrows)?;
            fit_logistic_weighted(
                design_ref,
                &gathered_w,
                &FaerBackend,
                &mut ws,
                &GlmOptions::default().without_separation_ridge(),
            )?
        }
        None => fit_glm(
            GlmFamily::BinomialLogit,
            design_ref,
            &FaerBackend,
            &mut ws,
            &GlmOptions::default().without_separation_ridge(),
        )?,
        Some(lambda) if weights.is_some() => {
            let w = weights.unwrap_or(&[]);
            if w.len() != x.physical_nrows() {
                return Err(LearnError::Shape { message: "weights length != physical rows" });
            }
            let gathered_w = gather_physical(w, x, nrows)?;
            fit_glm_ridge_weighted(
                GlmFamily::BinomialLogit,
                design_ref,
                &gathered_w,
                &FaerBackend,
                &mut ws,
                &GlmOptions::default(),
                lambda,
            )?
        }
        Some(lambda) => fit_glm_ridge(
            GlmFamily::BinomialLogit,
            design_ref,
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
            lambda,
        )?,
    };
    if !(admit_boundary_saturation && fit.converged && !fit.separated) {
        fit.require_ok()?;
    }
    Ok(fit)
}

impl LearnerFactory for LogisticLearner {
    fn task(&self) -> PredictionTask {
        PredictionTask::BinaryProbability
    }

    fn capabilities(&self) -> LearnerCapabilities {
        capabilities(true)
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        let fit = fit_logistic(x, y, weights, None, self.admit_boundary_saturation)?;
        Ok(Box::new(crate::coefficient::CoefficientPredictor::new(
            fit.coefficients,
            true,
            "logistic",
            "faer",
            crate::learner::FAER_VERSION,
        )))
    }
}

impl LearnerFactory for RidgeLogisticLearner {
    fn task(&self) -> PredictionTask {
        PredictionTask::BinaryProbability
    }

    fn capabilities(&self) -> LearnerCapabilities {
        capabilities(true)
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        let fit = fit_logistic(x, y, weights, Some(self.lambda), self.admit_boundary_saturation)?;
        Ok(Box::new(crate::coefficient::CoefficientPredictor::new(
            fit.coefficients,
            true,
            "logistic",
            "faer_ridge",
            crate::learner::FAER_VERSION,
        )))
    }
}
