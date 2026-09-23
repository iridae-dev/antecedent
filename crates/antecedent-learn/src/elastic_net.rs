//! Coordinate-descent elastic net. Hidden behind [`crate::LearnerSpec::ElasticNet`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    LassoFit, LassoOptions, first_col_is_exact_ones, fit_lasso, fit_lasso_with_ones_column,
};

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
    max_iter: u32,
}

/// Coordinate-descent sweep budget.
const MAX_ITER: u32 = 10_000;

impl ElasticNetLearner {
    /// Elastic-net factory for a public spec.
    #[must_use]
    pub const fn new(spec: ElasticNetSpec) -> Self {
        Self { spec, max_iter: MAX_ITER }
    }

    #[cfg(test)]
    const fn with_max_iter(mut self, max_iter: u32) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Elastic-net factory, or a typed refusal if `task` is not regression.
    ///
    /// # Errors
    ///
    /// [`LearnError::TaskMismatch`] when `task` is [`PredictionTask::BinaryProbability`].
    pub fn for_task(task: PredictionTask, spec: ElasticNetSpec) -> Result<Self, LearnError> {
        match task {
            PredictionTask::Regression => Ok(Self::new(spec)),
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
            max_iter: self.max_iter,
            tol: 1e-8,
        };
        let fit = if first_col_is_exact_ones(design.as_ref(), nrows) {
            let fit =
                fit_lasso_with_ones_column(design.as_ref(), nrows, ncols, &gathered_y, &options)?;
            let converged = fit.converged;
            (coefficients_with_ones_intercept(fit, ncols), converged)
        } else {
            let fit = fit_lasso(
                design.as_ref(),
                nrows,
                ncols,
                &gathered_y,
                &LassoOptions { fit_intercept: false, ..options },
            )?;
            (fit.coefficients, fit.converged)
        };
        // A coordinate-descent fit that hit `max_iter` is not a solution of the penalized
        // problem; using it as a nuisance would silently bias the downstream estimate.
        // (The logistic learner refuses non-converged IRLS the same way.)
        if !fit.1 {
            return Err(LearnError::Backend(format!(
                "elastic-net coordinate descent did not converge in {} iterations",
                options.max_iter
            )));
        }
        let coefficients = fit.0;
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

fn coefficients_with_ones_intercept(fit: LassoFit, ncols: usize) -> Vec<f64> {
    let mut coefficients = vec![0.0; ncols];
    coefficients[0] = fit.intercept;
    for (slot, value) in coefficients.iter_mut().skip(1).zip(fit.coefficients) {
        *slot = value;
    }
    coefficients
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    /// One coordinate-descent sweep cannot solve a correlated two-feature problem, so a
    /// learner with a one-sweep budget must refuse instead of returning the half-solved
    /// coefficients; with the real budget the same problem fits and recovers the truth.
    #[test]
    fn non_converged_coordinate_descent_is_refused() {
        let n = 40;
        let x1: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let x2: Vec<f64> = (0..n).map(|i| x1[i] + 0.3 * (i as f64 * 1.9).cos()).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x1[i] - 1.5 * x2[i]).collect();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&x1);
        design.extend_from_slice(&x2);
        let view = DesignView::from_column_major(&design, n, 3).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let spec = ElasticNetSpec { l1_ratio: 0.5, lambda: 1e-6 };

        let err = ElasticNetLearner::new(spec)
            .with_max_iter(1)
            .fit(view, TargetView::new(&y), None, &ctx)
            .err()
            .expect("one sweep must not be accepted as a fit");
        assert!(
            matches!(err, LearnError::Backend(ref m) if m.contains("did not converge")),
            "{err}"
        );

        let fitted =
            ElasticNetLearner::new(spec).fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let mut pred = vec![0.0; n];
        fitted.predict(view, &mut pred, &ctx).unwrap();
        for i in 0..n {
            assert!((pred[i] - y[i]).abs() < 1e-3, "row {i}: {} vs {}", pred[i], y[i]);
        }
    }
}
