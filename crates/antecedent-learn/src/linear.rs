//! faer-backed OLS factory. Hidden behind [`crate::LearnerSpec::Linear`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, fit_wls};

use crate::dense::{gather_physical, materialize_dense_colmajor};
use crate::design::{DesignView, Layout, TargetView};
use crate::error::LearnError;
use crate::learner::{FittedPredictor, LearnerCapabilities, LearnerFactory, PredictionTask};

/// Ordinary least squares via [`FaerBackend`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LinearLearner;

impl LinearLearner {
    /// OLS factory, or a typed refusal if `task` is not regression.
    ///
    /// # Errors
    ///
    /// [`LearnError::TaskMismatch`] when `task` is [`PredictionTask::BinaryProbability`].
    #[cfg(test)]
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
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        if weights.is_some_and(|w| w.len() != x.physical_nrows()) {
            return Err(LearnError::Shape { message: "weights length != physical rows" });
        }
        // The fold hold accounts for row indices, not the materialized matrix or
        // gathered vectors. The weighted path also allocates root and scaled y.
        let rows = x.nrows();
        let copied_design =
            x.row_selection().is_some() || x.dense().layout() != Layout::ColumnMajor;
        let matrix = if copied_design { rows.checked_mul(x.ncols()) } else { Some(0) };
        let vectors = rows.checked_mul(if weights.is_some() { 4 } else { 1 });
        let bytes = matrix
            .and_then(|m| vectors.and_then(|v| m.checked_add(v)))
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(LearnError::Shape { message: "linear materialization size overflow" })?;
        let _materialization = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => LearnError::Cancelled,
            _ => LearnError::Backend(format!("linear materialization admission: {e}")),
        })?;
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        let mut ws = LeastSquaresWorkspace::default();
        let fit = if let Some(w) = weights {
            let gathered_w = gather_physical(w, x, nrows)?;
            antecedent_core::numerical_admission::with_context(ctx, || {
                fit_wls(&design, nrows, ncols, &gathered_y, &gathered_w, &FaerBackend, &mut ws)
            })?
        } else {
            antecedent_core::numerical_admission::with_context(ctx, || {
                FaerBackend.least_squares(&design, nrows, ncols, &gathered_y, &mut ws)
            })?
        };
        Ok(Box::new(crate::coefficient::CoefficientPredictor::new(
            fit.coefficients,
            false,
            "linear",
            "faer",
            crate::learner::FAER_VERSION,
        )))
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::design::RowSelection;
    use antecedent_core::execution::AdmissionPool;

    #[test]
    fn linear_materialization_is_admitted_before_copy_and_released() {
        let data = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let rows = [0, 1, 2, 3];
        let view = DesignView::from_column_major(&data, 4, 2)
            .unwrap()
            .with_rows(RowSelection::new(&rows))
            .unwrap();
        let pool = AdmissionPool::fixed(1, 95); // matrix (64) + gathered y (32)
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        assert!(matches!(LinearLearner.fit(view, TargetView::new(&y), None, &ctx),
            Err(LearnError::Backend(message)) if message.contains("materialization admission")));
        assert_eq!(pool.available().bytes(), 95);
        let ctx = ExecutionContext::for_tests(1);
        let model = LinearLearner.fit(view, TargetView::new(&y), None, &ctx).unwrap();
        let mut predictions = [0.0; 4];
        model.predict(view, &mut predictions, &ctx).unwrap();
        for (observed, expected) in predictions.iter().zip(y) {
            assert!((observed - expected).abs() < 1e-9);
        }
    }
}
