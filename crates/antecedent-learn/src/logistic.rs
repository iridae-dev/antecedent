//! faer-backed logistic factory. Hidden behind [`crate::LearnerSpec::Logistic`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmFit, GlmOptions, LeastSquaresWorkspace, fit_glm,
};

use crate::dense::{gather_physical, materialize_dense_colmajor, predict_linear};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};

/// Logistic regression via [`fit_glm`] (`BinomialLogit`).
#[derive(Clone, Copy, Debug, Default)]
pub struct LogisticLearner;

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

impl LearnerFactory for LogisticLearner {
    fn task(&self) -> PredictionTask {
        PredictionTask::BinaryProbability
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            binary_probability: true,
            deterministic_seed: true,
            ..LearnerCapabilities::none()
        }
    }

    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        _ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
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
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &design, nrows, ncols, y: &gathered_y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
        )?;
        fit.require_ok()?;
        Ok(Box::new(LogisticPredictor { fit }))
    }
}

struct LogisticPredictor {
    fit: GlmFit,
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

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "logistic".into(),
            implementation: "faer".into(),
            version: "0.24".into(),
        }
    }
}
