//! faer-backed OLS factory. Hidden behind [`crate::LearnerSpec::Linear`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresFit, LeastSquaresWorkspace, fit_wls,
};

use crate::dense::{gather_physical, materialize_dense_colmajor, predict_linear};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};

/// Ordinary least squares via [`FaerBackend`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearLearner;

impl LinearLearner {
    /// OLS factory, or a typed refusal if `task` is not regression.
    ///
    /// # Errors
    ///
    /// [`LearnError::TaskMismatch`] when `task` is [`PredictionTask::BinaryProbability`].
    pub fn for_task(task: PredictionTask) -> Result<Self, LearnError> {
        match task {
            PredictionTask::Regression => Ok(Self),
            PredictionTask::BinaryProbability => Err(LearnError::TaskMismatch {
                requested: PredictionTask::BinaryProbability,
                supported: PredictionTask::Regression,
            }),
        }
    }
}

impl LearnerFactory for LinearLearner {
    fn task(&self) -> PredictionTask {
        PredictionTask::Regression
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: true,
            sample_weights: true,
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
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        let mut ws = LeastSquaresWorkspace::default();
        let fit = if let Some(w) = weights {
            if w.len() != x.physical_nrows() {
                return Err(LearnError::Shape { message: "weights length != physical rows" });
            }
            let gathered_w = gather_physical(w, x, nrows)?;
            fit_wls(&design, nrows, ncols, &gathered_y, &gathered_w, &FaerBackend, &mut ws)?
        } else {
            FaerBackend.least_squares(&design, nrows, ncols, &gathered_y, &mut ws)?
        };
        Ok(Box::new(LinearPredictor { fit }))
    }
}

struct LinearPredictor {
    fit: LeastSquaresFit,
}

impl FittedPredictor for LinearPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        predict_linear(&self.fit.coefficients, x, out)
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "linear".into(),
            implementation: "faer".into(),
            version: "0.24".into(),
        }
    }
}
