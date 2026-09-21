//! Coordinate-descent elastic net. Hidden behind [`crate::LearnerSpec::ElasticNet`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp)] // Intercept column is an exact ones pattern, not a tolerance.

use antecedent_core::ExecutionContext;
use antecedent_stats::{LassoFit, LassoOptions, fit_lasso, fit_lasso_with_ones_column};

use crate::dense::{gather_physical, materialize_dense_colmajor, predict_linear};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::ElasticNetSpec;

/// Elastic-net regression via coordinate descent.
#[derive(Clone, Copy, Debug)]
pub struct ElasticNetLearner {
    spec: ElasticNetSpec,
}

impl ElasticNetLearner {
    /// Elastic-net factory for a public spec.
    #[must_use]
    pub const fn new(spec: ElasticNetSpec) -> Self {
        Self { spec }
    }

    /// Elastic-net factory, or a typed refusal if `task` is not regression.
    ///
    /// # Errors
    ///
    /// [`LearnError::TaskMismatch`] when `task` is [`PredictionTask::BinaryProbability`].
    pub fn for_task(task: PredictionTask, spec: ElasticNetSpec) -> Result<Self, LearnError> {
        match task {
            PredictionTask::Regression => Ok(Self { spec }),
            PredictionTask::BinaryProbability => Err(LearnError::TaskMismatch {
                requested: PredictionTask::BinaryProbability,
                supported: PredictionTask::Regression,
            }),
        }
    }
}

impl LearnerFactory for ElasticNetLearner {
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
                message: "elastic-net learner does not accept sample weights",
            });
        }
        if !(self.spec.lambda.is_finite() && self.spec.lambda >= 0.0) {
            return Err(LearnError::Shape {
                message: "elastic-net lambda must be finite and ≥ 0"
            });
        }
        if !(self.spec.l1_ratio.is_finite() && (0.0..=1.0).contains(&self.spec.l1_ratio)) {
            return Err(LearnError::Shape { message: "elastic-net l1_ratio must be in [0, 1]" });
        }
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        let options = LassoOptions {
            lambda: self.spec.lambda,
            l1_ratio: self.spec.l1_ratio,
            fit_intercept: true,
            standardize: true,
            max_iter: 10_000,
            tol: 1e-8,
        };
        let coefficients = if first_col_is_exact_ones(design.as_ref(), nrows) {
            let fit =
                fit_lasso_with_ones_column(design.as_ref(), nrows, ncols, &gathered_y, &options)?;
            coefficients_with_ones_intercept(fit, ncols)
        } else {
            fit_lasso(
                design.as_ref(),
                nrows,
                ncols,
                &gathered_y,
                &LassoOptions { fit_intercept: false, ..options },
            )?
            .coefficients
        };
        if coefficients.len() != ncols {
            return Err(LearnError::Shape { message: "elastic-net coefficient length != ncols" });
        }
        Ok(Box::new(ElasticNetPredictor { coefficients }))
    }
}

struct ElasticNetPredictor {
    coefficients: Vec<f64>,
}

impl FittedPredictor for ElasticNetPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        predict_linear(&self.coefficients, x, out)
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        Ok(crate::PortablePredictor {
            version: 1,
            columns: self.coefficients.len(),
            provenance: self.provenance(),
            model: crate::PredictionMap::Linear {
                coefficients: self.coefficients.clone(),
                logistic: false,
            },
        })
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "elastic_net".into(),
            implementation: "coordinate_descent".into(),
            version: "0.24".into(),
        }
    }
}

fn first_col_is_exact_ones(x_colmajor: &[f64], nrows: usize) -> bool {
    nrows > 0 && x_colmajor.len() >= nrows && x_colmajor[..nrows].iter().all(|&v| v == 1.0)
}

fn coefficients_with_ones_intercept(fit: LassoFit, ncols: usize) -> Vec<f64> {
    let mut coefficients = vec![0.0; ncols];
    coefficients[0] = fit.intercept;
    for (slot, value) in coefficients.iter_mut().skip(1).zip(fit.coefficients) {
        *slot = value;
    }
    coefficients
}
