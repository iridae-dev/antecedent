//! Forust GBDT adapter. Hidden behind [`crate::LearnerSpec::GradientBoostedTrees`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::needless_pass_by_value)]

use antecedent_core::ExecutionContext;
use forust_ml::data::Matrix;
use forust_ml::gradientbooster::GradientBooster;
use forust_ml::objective::ObjectiveType;

use crate::dense::{gather_physical, materialize_dense_colmajor, require_binary_labels};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::GbtSpec;

/// Forust gradient-boosted trees.
#[derive(Clone, Copy, Debug)]
pub struct GbtLearner {
    spec: GbtSpec,
    task: PredictionTask,
}

impl GbtLearner {
    /// GBDT factory for a public spec and prediction task.
    #[must_use]
    pub const fn new(spec: GbtSpec, task: PredictionTask) -> Self {
        Self { spec, task }
    }
}

impl LearnerFactory for GbtLearner {
    fn task(&self) -> PredictionTask {
        self.task
    }

    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: true,
            binary_probability: true,
            sample_weights: true,
            missing_values: true,
            deterministic_seed: true,
            parallelism_control: true,
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
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        if matches!(self.task, PredictionTask::BinaryProbability) {
            require_binary_labels(&gathered_y)?;
        }
        let matrix = Matrix::new(&design, nrows, ncols);
        // Forust exposes a global-pool boolean, not a bounded thread lease.
        // Outer folds own the ExecutionContext parallelism budget.
        let parallel = false;
        let mut seed_rng = ctx.rng.stream(1);
        let objective = match self.task {
            PredictionTask::Regression => ObjectiveType::SquaredLoss,
            PredictionTask::BinaryProbability => ObjectiveType::LogLoss,
        };
        let mut model = GradientBooster::default()
            .set_objective_type(objective)
            .set_iterations(self.spec.trees as usize)
            .set_max_depth(self.spec.depth as usize)
            .set_learning_rate(self.spec.learning_rate as f32)
            .set_parallel(parallel)
            .set_seed(seed_rng.next_u64());
        if let Some(w) = weights {
            if w.len() != x.physical_nrows() {
                return Err(LearnError::Shape { message: "weights length != physical rows" });
            }
            let gathered_w = gather_physical(w, x, x.nrows())?;
            model.fit(&matrix, &gathered_y, &gathered_w, None).map_err(forust_err)?;
        } else {
            model.fit_unweighted(&matrix, &gathered_y, None).map_err(forust_err)?;
        }
        Ok(Box::new(GbtPredictor { model, task: self.task }))
    }
}

struct GbtPredictor {
    model: GradientBooster,
    task: PredictionTask,
}

impl FittedPredictor for GbtPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        _ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        if out.len() != x.nrows() {
            return Err(LearnError::Shape { message: "predict out length != logical rows" });
        }
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        let matrix = Matrix::new(&design, nrows, ncols);
        // Forust exposes a global-pool boolean, not a bounded thread lease.
        // Outer folds own the ExecutionContext parallelism budget.
        let parallel = false;
        let preds = self.model.predict(&matrix, parallel);
        if preds.len() != out.len() {
            return Err(LearnError::Shape { message: "forust predict length mismatch" });
        }
        match self.task {
            PredictionTask::Regression => out.copy_from_slice(&preds),
            PredictionTask::BinaryProbability => {
                for (slot, raw) in out.iter_mut().zip(preds) {
                    *slot = (1.0 / (1.0 + (-raw).exp())).clamp(1e-9, 1.0 - 1e-9);
                }
            }
        }
        Ok(())
    }

    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "gradient_boosted_trees".into(),
            implementation: "forust-ml".into(),
            version: "0.5".into(),
        }
    }
}

fn forust_err(err: forust_ml::errors::ForustError) -> LearnError {
    LearnError::Backend(format!("{err}"))
}
