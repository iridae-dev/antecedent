//! `SmartCore` forest adapter. Hidden behind [`crate::LearnerSpec::RandomForest`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use smartcore::ensemble::extra_trees_regressor::{
    ExtraTreesRegressor, ExtraTreesRegressorParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::dense::{gather_physical, materialize_dense_colmajor};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::ForestSpec;

/// Random forest / extra-trees factory.
#[derive(Clone, Copy, Debug)]
pub struct ForestLearner {
    spec: ForestSpec,
    task: PredictionTask,
}

impl ForestLearner {
    /// Forest factory for a public spec and prediction task.
    #[must_use]
    pub const fn new(spec: ForestSpec, task: PredictionTask) -> Self {
        Self { spec, task }
    }
}

impl LearnerFactory for ForestLearner {
    fn task(&self) -> PredictionTask {
        self.task
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: true,
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
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if weights.is_some() {
            return Err(LearnError::Unsupported {
                message: "random forest does not accept sample weights",
            });
        }
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        let matrix = dense_row_major(&design, nrows, ncols)?;
        let mut rng = ctx.rng.stream(2);
        let seed = rng.next_u64();
        let model = if self.spec.extra_trees {
            FittedForest::Extra(
                ExtraTreesRegressor::fit(
                    &matrix,
                    &gathered_y,
                    ExtraTreesRegressorParameters::default().with_seed(seed).with_n_trees(64),
                )
                .map_err(|e| LearnError::Backend(e.to_string()))?,
            )
        } else {
            FittedForest::Rf(
                RandomForestRegressor::fit(
                    &matrix,
                    &gathered_y,
                    RandomForestRegressorParameters::default().with_seed(seed).with_n_trees(64),
                )
                .map_err(|e| LearnError::Backend(e.to_string()))?,
            )
        };
        Ok(Box::new(FittedForestPredictor { model, task: self.task }))
    }
}

enum FittedForest {
    Rf(RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    Extra(ExtraTreesRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
}

struct FittedForestPredictor {
    model: FittedForest,
    task: PredictionTask,
}

impl FittedPredictor for FittedForestPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        if out.len() != nrows {
            return Err(LearnError::Shape { message: "predict out length != logical rows" });
        }
        let matrix = dense_row_major(&design, nrows, ncols)?;
        let preds = match &self.model {
            FittedForest::Rf(m) => {
                m.predict(&matrix).map_err(|e| LearnError::Backend(e.to_string()))?
            }
            FittedForest::Extra(m) => {
                m.predict(&matrix).map_err(|e| LearnError::Backend(e.to_string()))?
            }
        };
        if preds.len() != out.len() {
            return Err(LearnError::Shape { message: "forest predict length mismatch" });
        }
        match self.task {
            PredictionTask::Regression => out.copy_from_slice(&preds),
            PredictionTask::BinaryProbability => {
                for (slot, p) in out.iter_mut().zip(preds) {
                    *slot = p.clamp(1e-9, 1.0 - 1e-9);
                }
            }
        }
        Ok(())
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "random_forest".into(),
            implementation: "smartcore".into(),
            version: "0.4".into(),
        }
    }
}

fn dense_row_major(
    colmajor: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<DenseMatrix<f64>, LearnError> {
    let mut rows = Vec::with_capacity(nrows);
    for r in 0..nrows {
        let mut row = Vec::with_capacity(ncols);
        for c in 0..ncols {
            row.push(colmajor[c * nrows + r]);
        }
        rows.push(row);
    }
    DenseMatrix::from_2d_vec(&rows).map_err(|e| LearnError::Backend(e.to_string()))
}
