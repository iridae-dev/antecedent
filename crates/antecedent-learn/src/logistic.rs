//! faer-backed logistic factories. Hidden behind [`crate::LearnerSpec::Logistic`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmFit, GlmOptions, LeastSquaresWorkspace, fit_glm,
    fit_glm_ridge,
};

use crate::dense::{
    gather_physical, materialize_dense_colmajor, predict_linear, require_binary_labels,
};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};

/// Logistic regression via [`fit_glm`] (`BinomialLogit`).
#[derive(Clone, Copy, Debug, Default)]
pub struct LogisticLearner;

/// Ridge-penalized logistic regression via [`fit_glm_ridge`]: the probability-task form of
/// a ridge learner. The penalty applies to every non-intercept coefficient; a constant
/// first design column is the intercept and is left unpenalized.
#[derive(Clone, Copy, Debug)]
pub struct RidgeLogisticLearner {
    lambda: f64,
}

impl RidgeLogisticLearner {
    /// Penalized logistic factory. `lambda` must be finite and positive.
    ///
    /// # Errors
    ///
    /// [`LearnError::Shape`] for a non-finite or non-positive penalty.
    pub fn new(lambda: f64) -> Result<Self, LearnError> {
        if lambda.is_finite() && lambda > 0.0 {
            Ok(Self { lambda })
        } else {
            Err(LearnError::Shape { message: "logistic ridge penalty must be finite and > 0" })
        }
    }
}

impl LogisticLearner {
    /// Logistic factory, or a typed refusal if `task` is not probability.
    ///
    /// # Errors
    ///
    /// [`LearnError::TaskMismatch`] when `task` is [`PredictionTask::Regression`].
    pub fn for_task(task: PredictionTask) -> Result<Self, LearnError> {
        match task {
            PredictionTask::BinaryProbability => Ok(Self),
            PredictionTask::Regression => Err(LearnError::TaskMismatch {
                requested: PredictionTask::Regression,
                supported: PredictionTask::BinaryProbability,
            }),
        }
    }
}

fn capabilities() -> LearnerCapabilities {
    LearnerCapabilities {
        binary_probability: true,
        deterministic_seed: true,
        ..LearnerCapabilities::none()
    }
}

fn fit_logistic(
    x: DesignView<'_>,
    y: TargetView<'_>,
    weights: Option<&[f64]>,
    ridge: Option<f64>,
) -> Result<GlmFit, LearnError> {
    if weights.is_some() {
        return Err(LearnError::Unsupported {
            message: "logistic learner does not accept sample weights",
        });
    }
    let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
    if y.len() != x.physical_nrows() {
        return Err(LearnError::Shape { message: "target length != physical rows" });
    }
    let gathered_y = gather_physical(y.values(), x, nrows)?;
    require_binary_labels(&gathered_y)?;
    let mut ws = LeastSquaresWorkspace::default();
    let design_ref = GlmDesignRef { x_colmajor: &design, nrows, ncols, y: &gathered_y };
    let fit = match ridge {
        None => fit_glm(
            GlmFamily::BinomialLogit,
            design_ref,
            &FaerBackend,
            &mut ws,
            &GlmOptions::default().without_separation_ridge(),
        )?,
        Some(lambda) => fit_glm_ridge(
            GlmFamily::BinomialLogit,
            design_ref,
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
            lambda,
        )?,
    };
    fit.require_ok()?;
    Ok(fit)
}

impl LearnerFactory for LogisticLearner {
    fn task(&self) -> PredictionTask {
        PredictionTask::BinaryProbability
    }

    fn capabilities(&self) -> LearnerCapabilities {
        capabilities()
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        let fit = fit_logistic(x, y, weights, None)?;
        Ok(Box::new(LogisticPredictor { fit, implementation: "faer" }))
    }
}

impl LearnerFactory for RidgeLogisticLearner {
    fn task(&self) -> PredictionTask {
        PredictionTask::BinaryProbability
    }

    fn capabilities(&self) -> LearnerCapabilities {
        capabilities()
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        let fit = fit_logistic(x, y, weights, Some(self.lambda))?;
        Ok(Box::new(LogisticPredictor { fit, implementation: "faer_ridge" }))
    }
}

struct LogisticPredictor {
    fit: GlmFit,
    implementation: &'static str,
}

impl FittedPredictor for LogisticPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        predict_linear(&self.fit.coefficients, x, out)?;
        for slot in out.iter_mut() {
            *slot = (1.0 / (1.0 + (-*slot).exp())).clamp(1e-9, 1.0 - 1e-9);
        }
        Ok(())
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        Ok(crate::PortablePredictor {
            version: 1,
            columns: self.fit.coefficients.len(),
            provenance: self.provenance(),
            model: crate::PredictionMap::Linear {
                coefficients: self.fit.coefficients.clone(),
                logistic: true,
            },
        })
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "logistic".into(),
            implementation: self.implementation.into(),
            version: "0.24".into(),
        }
    }
}
