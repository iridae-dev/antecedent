//! Burn MLP adapter. Hidden behind [`crate::LearnerSpec::NeuralNet`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::StreamDomain;
use antecedent_learn_burn::TrainedMlp;
use antecedent_learn_burn::policy::{
    AutoPolicy, ComputePolicy, FitRequest, PolicyContext, PolicyError, SystemProbe,
    fit_with_policy, system_device_pool,
};

use crate::dense::{gather_physical, materialize_dense_colmajor};
use crate::design::{DesignView, TargetView};
use crate::error::LearnError;
use crate::learner::{
    FittedPredictor, LearnerCapabilities, LearnerFactory, LearnerProvenance, PredictionTask,
};
use crate::spec::{DevicePolicy, NeuralSpec};

impl NeuralSpec {
    fn device_name(&self) -> &'static str {
        match self.device {
            DevicePolicy::Cpu => "cpu",
            DevicePolicy::Gpu => "gpu",
            DevicePolicy::Auto => "auto",
        }
    }
}

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
        if matches!(self.task, PredictionTask::BinaryProbability) {
            crate::dense::require_binary_labels(&gathered_y)?;
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Learner, crate::streams::NEURAL);
        let seed = rng.next_u64();
        let binary = matches!(self.task, PredictionTask::BinaryProbability);
        let mlp = antecedent_learn_burn::MlpSpec {
            binary,
            hidden: self.spec.hidden as usize,
            epochs: self.spec.epochs as usize,
            learning_rate: self.spec.learning_rate,
            seed,
        };
        let policy = match self.spec.device {
            DevicePolicy::Cpu => ComputePolicy::Cpu,
            DevicePolicy::Gpu => ComputePolicy::Gpu,
            DevicePolicy::Auto => ComputePolicy::Auto,
        };
        let request = FitRequest::neural(policy, &mlp, nrows, ncols);
        // The default CPU route polls no token, so cancellation accounting and the existing
        // fit are unchanged. A device route polls once per epoch.
        let stop = || policy != ComputePolicy::Cpu && ctx.cancellation.is_cancelled();
        let context = PolicyContext {
            probe: &SystemProbe,
            device_pool: system_device_pool(),
            auto: AutoPolicy::default(),
            stop: &stop,
        };
        let fit = fit_with_policy(&request, &design, nrows, ncols, &gathered_y, &mlp, &context)
            .map_err(|error| match error {
                PolicyError::Refused(reason) => LearnError::Backend(format!(
                    "neural_net device={}: {reason} [{}]",
                    self.spec.device_name(),
                    reason.code()
                )),
                PolicyError::Fit(antecedent_learn_burn::LearnBurnError::Cancelled) => {
                    LearnError::Cancelled
                }
                other => LearnError::Backend(other.to_string()),
            })?;
        Ok(Box::new(NeuralNetPredictor {
            model: fit.model,
            task: self.task,
            implementation: fit.provenance.implementation_label(),
        }))
    }
}

struct NeuralNetPredictor {
    model: TrainedMlp,
    task: PredictionTask,
    /// Library, device, dtype, and any fallback of the route that fit this model.
    implementation: String,
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
        self.model
            .predict(&design, nrows, ncols, out)
            .map_err(|error| LearnError::Backend(error.to_string()))
    }

    fn provenance(&self) -> LearnerProvenance {
        let _ = self.task;
        LearnerProvenance {
            spec: "neural_net".into(),
            implementation: self.implementation.clone(),
            version: "0.16".into(),
        }
    }
}
