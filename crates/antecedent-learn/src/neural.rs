//! Burn MLP adapter. Hidden behind [`crate::LearnerSpec::NeuralNet`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::StreamDomain;
use antecedent_core::execution::{AdmissionError, MapExecutionError};
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
        crate::spec::LearnerSpec::NeuralNet(self.spec).validate()?;
        let mut result = ctx
            .try_map_indexed(1, |_, worker| {
                antecedent_core::numerical_admission::with_context(worker, || {
                    self.fit_admitted(x, y, worker)
                })
            })
            .map_err(|error| match error {
                MapExecutionError::Work(error) => error,
                MapExecutionError::Admission(AdmissionError::Cancelled) => LearnError::Cancelled,
                MapExecutionError::Admission(error) => LearnError::Admission(error),
            })?;
        Ok(result.remove(0))
    }
}

impl NeuralNetLearner {
    fn fit_admitted(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        let footprint = antecedent_learn_burn::admission::footprint(
            x.nrows(),
            x.ncols(),
            self.spec.hidden as usize,
        )
        .map_err(|error| LearnError::Backend(format!("neural_net admission: {error}")))?;
        // Training state is host-resident on CPU, including Auto's CPU fallback. Charge it
        // conservatively on every route, in addition to the host and staging buffers.
        let epoch_bytes = u64::from(self.spec.epochs)
            .checked_mul(std::mem::size_of::<antecedent_learn_burn::device::EpochLoss>() as u64);
        let column_bytes = u64::try_from(x.ncols()).ok().and_then(|n| n.checked_mul(16));
        let bytes = footprint
            .host_bytes
            .checked_add(footprint.scratch_bytes)
            .and_then(|v| v.checked_add(footprint.device_bytes))
            .and_then(|v| epoch_bytes.and_then(|epoch| v.checked_add(epoch)))
            .and_then(|v| column_bytes.and_then(|cols| v.checked_add(cols)))
            .ok_or_else(|| {
                LearnError::Backend("neural_net admission: footprint overflow".into())
            })?;
        let _lease = antecedent_core::numerical_admission::acquire(bytes)
            .expect("worker context is installed")
            .map_err(|error| match error {
                AdmissionError::Cancelled => LearnError::Cancelled,
                other => LearnError::Admission(other),
            })?;
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
        let stop = || ctx.cancellation.is_cancelled();
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

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;

    #[test]
    fn neural_fit_obeys_admission_cancellation_and_reuses_a_worker() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let target = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x = DesignView::from_column_major(&data, 8, 1).unwrap();
        let y = TargetView::new(&target);
        let spec = NeuralSpec { hidden: 2, epochs: 1, ..NeuralSpec::default() };
        let learner = NeuralNetLearner::new(spec, PredictionTask::Regression);
        let cancelled = ExecutionContext::for_tests(1);
        cancelled.cancellation.cancel();
        assert!(matches!(learner.fit(x, y, None, &cancelled), Err(LearnError::Cancelled)));
        let mut during = ExecutionContext::for_tests(1);
        during.cancellation =
            antecedent_core::execution::CancellationToken::cancel_after_checks(10);
        let long =
            NeuralNetLearner::new(NeuralSpec { epochs: 100, ..spec }, PredictionTask::Regression);
        assert!(matches!(long.fit(x, y, None, &during), Err(LearnError::Cancelled)));
        for device in [DevicePolicy::Cpu, DevicePolicy::Auto] {
            let limited = ExecutionContext::for_tests(1).with_memory_limit(Some(1));
            let learner =
                NeuralNetLearner::new(NeuralSpec { device, ..spec }, PredictionTask::Regression);
            assert!(matches!(
                learner.fit(x, y, None, &limited),
                Err(LearnError::Admission(
                    antecedent_core::execution::AdmissionError::ExceedsRemainder { .. }
                ))
            ));
        }
        // Epoch bookkeeping is allocated before the first training iteration.
        let long_spec = NeuralSpec { epochs: u32::MAX, ..spec };
        let limited = ExecutionContext::for_tests(1).with_memory_limit(Some(1_000_000));
        assert!(matches!(
            NeuralNetLearner::new(long_spec, PredictionTask::Regression).fit(x, y, None, &limited),
            Err(LearnError::Admission(
                antecedent_core::execution::AdmissionError::ExceedsRemainder { .. }
            ))
        ));
        let pool = AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool);
        let result = ctx.try_map_indexed(1, |_, inner| learner.fit(x, y, None, inner));
        assert!(result.is_ok(), "nested neural fit must reuse its worker slot");
    }

    #[test]
    fn f32_learning_rates_must_remain_finite_and_positive() {
        for rate in [f64::MIN_POSITIVE, f64::MAX, 0.0, f64::NAN] {
            let neural = crate::spec::LearnerSpec::NeuralNet(NeuralSpec {
                learning_rate: rate,
                ..NeuralSpec::default()
            });
            let tree = crate::spec::LearnerSpec::GradientBoostedTrees(crate::spec::GbtSpec {
                learning_rate: rate,
                ..crate::spec::GbtSpec::default()
            });
            assert!(neural.validate().is_err());
            assert!(tree.validate().is_err());
        }
        assert!(crate::spec::LearnerSpec::NeuralNet(NeuralSpec::default()).validate().is_ok());
    }
}
