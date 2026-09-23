//! faer-backed ridge factory. Hidden behind [`crate::LearnerSpec::Ridge`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{FaerBackend, LeastSquaresFit, LeastSquaresWorkspace, fit_ridge};

use crate::dense::{gather_physical, materialize_dense_colmajor, predict_linear};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::RidgeSpec;

/// Ridge regression via [`fit_ridge`].
#[derive(Clone, Copy, Debug)]
pub struct RidgeLearner {
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
                message: "ridge learner does not accept sample weights",
            });
        }
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        let mut ws = LeastSquaresWorkspace::default();
        let fit =
            fit_ridge(&design, nrows, ncols, &gathered_y, self.spec.lambda, &FaerBackend, &mut ws)?;
        Ok(Box::new(RidgePredictor { fit }))
    }
}

struct RidgePredictor {
    fit: LeastSquaresFit,
}

impl FittedPredictor for RidgePredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        predict_linear(&self.fit.coefficients, x, out)
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        Ok(crate::PortablePredictor {
            version: 1,
            columns: self.fit.coefficients.len(),
            provenance: self.provenance(),
            model: crate::PredictionMap::Linear {
                coefficients: self.fit.coefficients.clone(),
                logistic: false,
            },
        })
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "ridge".into(),
            implementation: "faer".into(),
            version: "0.24".into(),
        }
    }
}
