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

use crate::dense::{gather_physical, materialize_dense_colmajor, require_binary_labels};
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
        if matches!(self.task, PredictionTask::BinaryProbability) {
            require_binary_labels(&gathered_y)?;
        }
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
        Ok(Box::new(FittedForestPredictor { model, task: self.task, columns: x.ncols() }))
    }
}

enum FittedForest {
    Rf(RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
    Extra(ExtraTreesRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>),
}

struct FittedForestPredictor {
    columns: usize,
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
            // Leaf means of a 0/1 outcome estimate E[T|X] = P(T=1|X). This is
            // not a log-loss classifier; clamp only guards floating error.
            PredictionTask::BinaryProbability => {
                for (slot, p) in out.iter_mut().zip(preds) {
                    *slot = p.clamp(1e-9, 1.0 - 1e-9);
                }
            }
        }
        Ok(())
    }

    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        use crate::{PortablePredictor, PredictionMap, PredictionNode};
        // Read the pinned provider's serde output only during export. Portable readers
        // never deserialize a foreign model or execute its unchecked tree traversal.
        let json = match &self.model {
            FittedForest::Rf(model) => serde_json::to_value(model),
            FittedForest::Extra(model) => serde_json::to_value(model),
        }
        .map_err(|e| LearnError::Backend(e.to_string()))?;
        #[derive(serde::Deserialize)]
        struct Forest {
            trees: Vec<Tree>,
        }
        #[derive(serde::Deserialize)]
        struct Tree {
            nodes: Vec<Node>,
        }
        #[derive(serde::Deserialize)]
        struct Node {
            output: f64,
            split_feature: usize,
            split_value: Option<f64>,
            true_child: Option<usize>,
            false_child: Option<usize>,
        }
        let forest: Forest = serde_json::from_value(json["forest_regressor"].clone())
            .map_err(|e| LearnError::Backend(e.to_string()))?;
        let trees = forest
            .trees
            .into_iter()
            .map(|tree| {
                tree.nodes
                    .into_iter()
                    .map(|n| match (n.true_child, n.false_child, n.split_value) {
                        (Some(left), Some(right), Some(threshold)) => Ok(PredictionNode::Split {
                            feature: n.split_feature,
                            threshold,
                            inclusive: true,
                            left,
                            right,
                            missing: right,
                        }),
                        (None, None, _) => Ok(PredictionNode::Leaf { value: Some(n.output) }),
                        _ => Err(LearnError::Shape { message: "incomplete provider tree" }),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = PortablePredictor {
            version: 1,
            columns: self.columns,
            provenance: self.provenance(),
            model: PredictionMap::Trees {
                trees,
                base: 0.0,
                average: true,
                logistic: false,
                probability: self.task == PredictionTask::BinaryProbability,
            },
        };
        result.validate()?;
        Ok(result)
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
