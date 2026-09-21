//! Public learner specification. Provider resolution is internal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::LearnError;
use crate::learner::{LearnerCapabilities, LearnerFactory, PredictionTask};
use crate::linear::LinearLearner;
use crate::logistic::LogisticLearner;
use crate::ridge::RidgeLearner;

/// Public linear (OLS) options. The design already includes an intercept if wanted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LinearSpec {}

/// Ridge options (milestone D).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RidgeSpec {
    /// Penalty strength. Zero is ordinary least squares, not a ridge default.
    pub lambda: f64,
}

impl Default for RidgeSpec {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

/// Logistic options (milestone D).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LogisticSpec {}

/// Elastic-net options (milestone D). Coordinate descent; not a dense factorization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElasticNetSpec {
    /// L1/L2 mix in `[0, 1]`. `1` is lasso; `0` is ridge.
    pub l1_ratio: f64,
    /// Penalty strength. Zero is ordinary least squares, not an elastic-net default.
    pub lambda: f64,
}

impl Default for ElasticNetSpec {
    fn default() -> Self {
        Self { l1_ratio: 0.5, lambda: 1.0 }
    }
}

/// Gradient-boosted trees (Forust in F). Public name is not the provider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GbtSpec {
    /// Number of trees.
    pub trees: u32,
    /// Maximum depth.
    pub depth: u32,
    /// Shrinkage.
    pub learning_rate: f64,
}

impl Default for GbtSpec {
    fn default() -> Self {
        Self { trees: 300, depth: 6, learning_rate: 0.05 }
    }
}

/// Random forest or extra-trees (`SmartCore` in H).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ForestSpec {
    /// Extra-trees split randomization when the forest provider is present.
    pub extra_trees: bool,
}

/// Neural net (Burn in L). Public name is not the provider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeuralSpec {
    /// Units in each of two hidden layers.
    pub hidden: u32,
    /// Training epochs.
    pub epochs: u32,
    /// Adam step size.
    pub learning_rate: f64,
}

impl Default for NeuralSpec {
    fn default() -> Self {
        Self { hidden: 32, epochs: 40, learning_rate: 0.05 }
    }
}

/// Public model specification. Resolution hides the provider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LearnerSpec {
    /// Restrained Auto (milestone I).
    Auto,
    /// Ordinary least squares.
    Linear(LinearSpec),
    /// Ridge (milestone D).
    Ridge(RidgeSpec),
    /// Logistic (milestone D).
    Logistic(LogisticSpec),
    /// Elastic net (milestone D).
    ElasticNet(ElasticNetSpec),
    /// Gradient-boosted trees (milestone F).
    GradientBoostedTrees(GbtSpec),
    /// Random forest (milestone H).
    RandomForest(ForestSpec),
    /// Neural net (milestone L).
    NeuralNet(NeuralSpec),
}

impl LearnerSpec {
    /// Public name for provenance and errors.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Linear(_) => "linear",
            Self::Ridge(_) => "ridge",
            Self::Logistic(_) => "logistic",
            Self::ElasticNet(_) => "elastic_net",
            Self::GradientBoostedTrees(_) => "gradient_boosted_trees",
            Self::RandomForest(_) => "random_forest",
            Self::NeuralNet(_) => "neural_net",
        }
    }

    /// Parse a public learner key (`ridge`, `gradient_boosted_trees`, …).
    ///
    /// # Errors
    ///
    /// Unknown key.
    pub fn parse(key: &str) -> Result<Self, LearnError> {
        match key {
            "auto" => Ok(Self::Auto),
            "linear" => Ok(Self::Linear(LinearSpec::default())),
            "ridge" => Ok(Self::Ridge(RidgeSpec { lambda: 1.0 })),
            "logistic" => Ok(Self::Logistic(LogisticSpec::default())),
            "elastic_net" => Ok(Self::ElasticNet(ElasticNetSpec::default())),
            "gradient_boosted_trees" => Ok(Self::GradientBoostedTrees(GbtSpec::default())),
            "random_forest" => Ok(Self::RandomForest(ForestSpec::default())),
            "neural_net" | "mlp" => Ok(Self::NeuralNet(NeuralSpec::default())),
            _ => Err(LearnError::Backend(format!("unknown learner {key:?}"))),
        }
    }

    /// Remap a parametric spec onto the task a nuisance actually needs.
    #[must_use]
    pub fn for_task(self, task: PredictionTask) -> Self {
        match (self, task) {
            (
                Self::Linear(_) | Self::Ridge(_) | Self::ElasticNet(_),
                PredictionTask::BinaryProbability,
            ) => Self::Logistic(LogisticSpec::default()),
            (Self::Logistic(_), PredictionTask::Regression) => Self::Linear(LinearSpec::default()),
            (other, _) => other,
        }
    }
}

/// Resolve a public spec to a factory. GBDT / forest use [`PredictionTask::Regression`].
///
/// # Errors
///
/// [`LearnError::ProviderUnavailable`] for specs that have no provider yet.
pub fn resolve(spec: LearnerSpec) -> Result<Box<dyn LearnerFactory>, LearnError> {
    resolve_for(spec, default_task(spec))
}

/// Resolve a public spec for an explicit prediction task.
///
/// # Errors
///
/// Task mismatch, or [`LearnError::ProviderUnavailable`].
pub fn resolve_for(
    spec: LearnerSpec,
    task: PredictionTask,
) -> Result<Box<dyn LearnerFactory>, LearnError> {
    match spec {
        LearnerSpec::Linear(_) => {
            require_resolved_task(task, PredictionTask::Regression)?;
            Ok(Box::new(LinearLearner))
        }
        LearnerSpec::Ridge(ridge) => {
            require_resolved_task(task, PredictionTask::Regression)?;
            Ok(Box::new(RidgeLearner::new(ridge)))
        }
        LearnerSpec::Logistic(_) => {
            require_resolved_task(task, PredictionTask::BinaryProbability)?;
            Ok(Box::new(LogisticLearner))
        }
        LearnerSpec::ElasticNet(elastic) => {
            require_resolved_task(task, PredictionTask::Regression)?;
            Ok(Box::new(crate::elastic_net::ElasticNetLearner::new(elastic)))
        }
        LearnerSpec::GradientBoostedTrees(gbt) => resolve_gbt(gbt, task),
        LearnerSpec::RandomForest(forest) => resolve_forest(forest, task),
        LearnerSpec::NeuralNet(neural) => resolve_neural(neural, task),
        LearnerSpec::Auto => Ok(Box::new(crate::auto::AutoLearner(task))),
    }
}

fn default_task(spec: LearnerSpec) -> PredictionTask {
    match spec {
        LearnerSpec::Logistic(_) => PredictionTask::BinaryProbability,
        _ => PredictionTask::Regression,
    }
}

fn require_resolved_task(
    requested: PredictionTask,
    supported: PredictionTask,
) -> Result<(), LearnError> {
    if requested != supported {
        return Err(LearnError::TaskMismatch { requested, supported });
    }
    Ok(())
}

#[allow(clippy::unnecessary_wraps, clippy::needless_return)]
fn resolve_forest(
    spec: crate::ForestSpec,
    task: PredictionTask,
) -> Result<Box<dyn LearnerFactory>, LearnError> {
    #[cfg(feature = "ml-forest")]
    {
        return Ok(Box::new(crate::forest::ForestLearner::new(spec, task)));
    }
    #[cfg(not(feature = "ml-forest"))]
    {
        let _ = (spec, task);
        Err(LearnError::ProviderUnavailable {
            spec: "random_forest",
            required: required_capabilities(
                LearnerSpec::RandomForest(crate::ForestSpec::default()),
            ),
        })
    }
}

#[allow(clippy::unnecessary_wraps, clippy::needless_return)]
fn resolve_gbt(
    spec: crate::GbtSpec,
    task: PredictionTask,
) -> Result<Box<dyn LearnerFactory>, LearnError> {
    #[cfg(feature = "ml-gbdt")]
    {
        return Ok(Box::new(crate::gbt::GbtLearner::new(spec, task)));
    }
    #[cfg(not(feature = "ml-gbdt"))]
    {
        let _ = (spec, task);
        Err(LearnError::ProviderUnavailable {
            spec: "gradient_boosted_trees",
            required: required_capabilities(LearnerSpec::GradientBoostedTrees(
                crate::GbtSpec::default(),
            )),
        })
    }
}

#[allow(clippy::unnecessary_wraps, clippy::needless_return)]
fn resolve_neural(
    spec: crate::NeuralSpec,
    task: PredictionTask,
) -> Result<Box<dyn LearnerFactory>, LearnError> {
    #[cfg(feature = "ml-gpu")]
    {
        return Ok(Box::new(crate::neural::NeuralNetLearner::new(spec, task)));
    }
    #[cfg(not(feature = "ml-gpu"))]
    {
        let _ = (spec, task);
        Err(LearnError::ProviderUnavailable {
            spec: "neural_net",
            required: required_capabilities(LearnerSpec::NeuralNet(crate::NeuralSpec::default())),
        })
    }
}

fn required_capabilities(spec: LearnerSpec) -> LearnerCapabilities {
    let mut cap = LearnerCapabilities::none();
    cap.deterministic_seed = true;
    match spec {
        LearnerSpec::Logistic(_) => cap.binary_probability = true,
        LearnerSpec::GradientBoostedTrees(_)
        | LearnerSpec::Auto
        | LearnerSpec::RandomForest(_)
        | LearnerSpec::NeuralNet(_) => {
            cap.regression = true;
            cap.binary_probability = true;
            cap.missing_values = matches!(
                spec,
                LearnerSpec::GradientBoostedTrees(_)
                    | LearnerSpec::Auto
                    | LearnerSpec::RandomForest(_)
            );
        }
        _ => cap.regression = true,
    }
    cap
}

/// Convenience: reject a factory that cannot serve `task` before fit.
///
/// # Errors
///
/// [`LearnError::TaskMismatch`].
pub fn require_task(factory: &dyn LearnerFactory, task: PredictionTask) -> Result<(), LearnError> {
    let supported = factory.task();
    if supported != task {
        return Err(LearnError::TaskMismatch { requested: task, supported });
    }
    Ok(())
}
