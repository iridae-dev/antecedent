//! Burn MLP adapter. Hidden behind [`crate::LearnerSpec::NeuralNet`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value)]

use antecedent_core::ExecutionContext;
use antecedent_core::StreamDomain;
use antecedent_learn_burn::TrainedMlp;

use crate::dense::{gather_physical, materialize_dense_colmajor};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::NeuralSpec;

/// Neural-net factory. Provider identity stays out of `antecedent-estimate`.
#[derive(Clone, Copy, Debug)]
pub struct NeuralNetLearner {
    spec: NeuralSpec,
    task: PredictionTask,
}

impl NeuralNetLearner {
    /// Factory for a public spec and prediction task.
    #[must_use]
    pub const fn new(spec: NeuralSpec, task: PredictionTask) -> Self {
        Self { spec, task }
    }
}

impl LearnerFactory for NeuralNetLearner {
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
                message: "neural_net does not accept sample weights",
            });
        }
        if y.len() != x.physical_nrows() {
            return Err(LearnError::Shape { message: "target length != physical rows" });
        }
        let (design, nrows, ncols) = materialize_dense_colmajor(x)?;
        let gathered_y = gather_physical(y.values(), x, nrows)?;
        let mut rng = ctx.rng.stream_for(StreamDomain::Learner, 0x4E45_5500);
        let seed = rng.next_u64();
        let binary = matches!(self.task, PredictionTask::BinaryProbability);
        let model = antecedent_learn_burn::train(
            &design,
            nrows,
            ncols,
            &gathered_y,
            binary,
            self.spec.hidden as usize,
            self.spec.epochs as usize,
            self.spec.learning_rate,
            seed,
        )
        .map_err(LearnError::Backend)?;
        Ok(Box::new(NeuralNetPredictor { model, task: self.task }))
    }
}

struct NeuralNetPredictor {
    model: TrainedMlp,
    task: PredictionTask,
}

impl FittedPredictor for NeuralNetPredictor {
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
        self.model.predict(&design, nrows, ncols, out).map_err(LearnError::Backend)
    }

    fn provenance(&self) -> LearnerProvenance {
        let _ = self.task;
        LearnerProvenance {
            spec: "neural_net".into(),
            implementation: "burn".into(),
            version: "0.16".into(),
        }
    }
}
