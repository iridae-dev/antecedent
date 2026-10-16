//! faer-backed ridge factory. Hidden behind [`crate::LearnerSpec::Ridge`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{FaerBackend, LeastSquaresWorkspace, fit_ridge};

use crate::dense::{gather_physical, materialize_dense_colmajor};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{FittedPredictor, LearnerCapabilities, LearnerFactory, PredictionTask};
use crate::spec::RidgeSpec;

/// Ridge regression via [`fit_ridge`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct RidgeLearner {
    spec: RidgeSpec,
}

impl RidgeLearner {
    /// Ridge factory for a public spec.
    #[must_use]
    pub const fn new(spec: RidgeSpec) -> Self {
        Self { spec }
    }

    /// Ridge factory, or a typed refusal if `task` is not regression.
    ///
    /// # Errors
    ///
    /// [`LearnError::TaskMismatch`] when `task` is [`PredictionTask::BinaryProbability`].
    #[cfg(test)]
    pub fn for_task(task: PredictionTask, spec: RidgeSpec) -> Result<Self, LearnError> {
        match task {
            PredictionTask::Regression => Ok(Self { spec }),
            PredictionTask::BinaryProbability => Err(LearnError::TaskMismatch {
                requested: PredictionTask::BinaryProbability,
                supported: PredictionTask::Regression,
            }),
        }
    }
}

impl LearnerFactory for RidgeLearner {
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
        let (mut design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let mut gathered_y = gather_physical(y.values(), x, nrows)?;
        if let Some(w) = weights {
            // Prior-weighted ridge `Σ w_i (y_i − x_iβ)² + λ|β|²`: scale each row of the design
            // and the target by √w_i and solve the ordinary ridge problem.
            if w.len() != x.physical_nrows() {
                return Err(LearnError::Shape { message: "weights length != physical rows" });
            }
            let gathered_w = gather_physical(w, x, nrows)?;
            if gathered_w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(LearnError::Shape {
                    message: "prior weights must be finite and non-negative",
                });
            }
            for column in design.to_mut().chunks_exact_mut(nrows.max(1)) {
                for (value, weight) in column.iter_mut().zip(&gathered_w) {
                    *value *= weight.sqrt();
                }
            }
            for (value, weight) in gathered_y.iter_mut().zip(&gathered_w) {
                *value *= weight.sqrt();
            }
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit =
            fit_ridge(&design, nrows, ncols, &gathered_y, self.spec.lambda, &FaerBackend, &mut ws)?;
        Ok(Box::new(crate::coefficient::CoefficientPredictor::new(
            fit.coefficients,
            false,
            "ridge",
            "faer",
            crate::learner::FAER_VERSION,
        )))
    }
}
