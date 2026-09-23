//! Public learner specification. Provider resolution is internal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::LearnError;
#[cfg(not(all(feature = "ml-gbdt", feature = "ml-forest", feature = "ml-neural")))]
use crate::learner::LearnerCapabilities;
use crate::learner::{LearnerFactory, PredictionTask};
use crate::linear::LinearLearner;
use crate::logistic::{LogisticLearner, RidgeLogisticLearner};
use crate::ridge::RidgeLearner;

/// Public linear (OLS) options. The design already includes an intercept if wanted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LinearSpec {}

/// Ridge options (milestone D).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RidgeSpec {
    /// Penalty strength. Zero is ordinary least squares, not a ridge default.
    ///
    /// Minimizes `‖y − Xβ‖² + λ‖β‖²` on the *raw* columns (a constant first column is the
    /// unpenalized intercept). The penalty is not divided by the sample size and the
    /// columns are not standardized, so `λ` is in the units of `‖y − Xβ‖²` and is not
    /// comparable with [`ElasticNetSpec::lambda`], whose columns are standardized.
    pub lambda: f64,
}

impl Default for RidgeSpec {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

/// Logistic options (milestone D).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LogisticSpec {
    /// Ridge penalty on the non-intercept coefficients. Zero is the unpenalized MLE.
    ///
    /// This is the probability-task form of [`RidgeSpec::lambda`]: a user who asks for
    /// a ridge learner and a binary nuisance gets a *penalized* logistic model, not the
    /// plain IRLS fit (which fails or yields extreme scores under near-separation).
    pub ridge_lambda: f64,
}

/// Elastic-net options (milestone D). Coordinate descent; not a dense factorization.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElasticNetSpec {
    /// L1/L2 mix in `[0, 1]`. `1` is lasso; `0` is ridge.
    pub l1_ratio: f64,
    /// Penalty strength. Zero is ordinary least squares, not an elastic-net default.
    ///
    /// Minimizes `½‖y − Xβ‖² + λ(α‖β‖₁ + (1 − α)/2 ‖β‖²)` on *standardized* columns
    /// (`α = l1_ratio`, unpenalized intercept). The data term is a sum, not a mean, so the
    /// same `λ` regularizes relatively less as `n` grows; scale `λ` with `n` to hold the
    /// penalty per observation fixed. It is not comparable with [`RidgeSpec::lambda`].
    pub lambda: f64,
}

impl Default for ElasticNetSpec {
    fn default() -> Self {
        Self { l1_ratio: 0.5, lambda: 1.0 }
    }
}

/// Gradient-boosted trees (Forust in F). Public name is not the provider.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForestSpec {
    /// Extra-trees split randomization when the forest provider is present.
    pub extra_trees: bool,
    /// Number of trees.
    pub n_trees: u32,
    /// Minimum rows in a leaf. `None` picks the task default: `1` for regression
    /// (fully grown trees) and [`FOREST_PROBABILITY_MIN_LEAF`] for a probability
    /// nuisance, where fully grown trees return exactly 0 or 1 out of fold and a
    /// propensity of 0/1 makes inverse-probability weights explode.
    pub min_samples_leaf: Option<u32>,
}

/// Default minimum leaf size of a forest used as a probability (propensity) model.
pub const FOREST_PROBABILITY_MIN_LEAF: u32 = 10;

impl Default for ForestSpec {
    fn default() -> Self {
        Self { extra_trees: false, n_trees: 64, min_samples_leaf: None }
    }
}

impl ForestSpec {
    /// Leaf size in effect for `task`.
    #[must_use]
    pub const fn resolved_min_samples_leaf(self, task: PredictionTask) -> u32 {
        match (self.min_samples_leaf, task) {
            (Some(leaf), _) => leaf,
            (None, PredictionTask::BinaryProbability) => FOREST_PROBABILITY_MIN_LEAF,
            (None, PredictionTask::Regression) => 1,
        }
    }
}

/// Neural net (Burn in L). Public name is not the provider.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
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
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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
    /// Stable exact-parameter identity, shared by compilation and provider caches.
    #[must_use]
    pub fn identity(self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::Linear(_) => "linear".into(),
            Self::Logistic(s) if s.ridge_lambda == 0.0 => "logistic".into(),
            Self::Logistic(s) => format!("logistic:{}", s.ridge_lambda.to_bits()),
            Self::Ridge(s) => format!("ridge:{}", s.lambda.to_bits()),
            Self::ElasticNet(s) => {
                format!("elastic_net:{}:{}", s.lambda.to_bits(), s.l1_ratio.to_bits())
            }
            Self::GradientBoostedTrees(s) => {
                format!("gbdt:{}:{}:{}", s.trees, s.depth, s.learning_rate.to_bits())
            }
            Self::RandomForest(s) => format!(
                "forest:{}:{}:{}",
                u8::from(s.extra_trees),
                s.n_trees,
                s.min_samples_leaf.map_or_else(|| "task".to_string(), |leaf| leaf.to_string())
            ),
            Self::NeuralNet(s) => {
                format!("neural:{}:{}:{}", s.hidden, s.epochs, s.learning_rate.to_bits())
            }
        }
    }

    /// Validate configuration independently of feature availability or task coercion.
    /// # Errors
    /// Nonfinite, negative, or otherwise invalid hyperparameters.
    pub fn validate(self) -> Result<(), LearnError> {
        let valid = match self {
            Self::Ridge(s) => s.lambda.is_finite() && s.lambda >= 0.0,
            Self::Logistic(s) => s.ridge_lambda.is_finite() && s.ridge_lambda >= 0.0,
            Self::RandomForest(s) => s.n_trees > 0 && s.min_samples_leaf != Some(0),
            Self::ElasticNet(s) => {
                s.lambda.is_finite()
                    && s.lambda >= 0.0
                    && s.l1_ratio.is_finite()
                    && (0.0..=1.0).contains(&s.l1_ratio)
            }
            Self::GradientBoostedTrees(s) => {
                s.trees > 0 && s.depth > 0 && s.learning_rate.is_finite() && s.learning_rate > 0.0
            }
            Self::NeuralNet(s) => {
                s.hidden > 0 && s.epochs > 0 && s.learning_rate.is_finite() && s.learning_rate > 0.0
            }
            _ => true,
        };
        if valid {
            Ok(())
        } else {
            Err(LearnError::Shape { message: "invalid learner hyperparameters" })
        }
    }

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
    ///
    /// A ridge spec becomes the *penalized* logistic model with the same penalty, so a
    /// user's `lambda` is honoured on a probability nuisance instead of silently
    /// replaced by an unpenalized fit; a pure-ridge elastic net (`l1_ratio = 0`) maps the
    /// same way. OLS maps to the unpenalized logistic MLE.
    ///
    /// An elastic net with an L1 component has no probability-task counterpart (there is
    /// no L1-penalized logistic learner). It is left as an elastic net so that resolving
    /// it for the probability task refuses with [`LearnError::TaskMismatch`] rather than
    /// substituting a different estimator than the one requested.
    #[must_use]
    pub const fn for_task(self, task: PredictionTask) -> Self {
        match (self, task) {
            (Self::Linear(_), PredictionTask::BinaryProbability) => {
                Self::Logistic(LogisticSpec { ridge_lambda: 0.0 })
            }
            (Self::Ridge(s), PredictionTask::BinaryProbability) => {
                Self::Logistic(LogisticSpec { ridge_lambda: s.lambda })
            }
            (Self::ElasticNet(s), PredictionTask::BinaryProbability) if s.l1_ratio == 0.0 => {
                Self::Logistic(LogisticSpec { ridge_lambda: s.lambda })
            }
            (Self::Logistic(_), PredictionTask::Regression) => Self::Linear(LinearSpec {}),
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
    spec.validate()?;
    match spec {
        LearnerSpec::Linear(_) => {
            require_resolved_task(task, PredictionTask::Regression)?;
            Ok(Box::new(LinearLearner))
        }
        LearnerSpec::Ridge(ridge) => {
            require_resolved_task(task, PredictionTask::Regression)?;
            Ok(Box::new(RidgeLearner::new(ridge)))
        }
        LearnerSpec::Logistic(logistic) => {
            require_resolved_task(task, PredictionTask::BinaryProbability)?;
            if logistic.ridge_lambda > 0.0 {
                Ok(Box::new(RidgeLogisticLearner::new(logistic.ridge_lambda)?))
            } else {
                Ok(Box::new(LogisticLearner))
            }
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
    #[cfg(feature = "ml-neural")]
    {
        return Ok(Box::new(crate::neural::NeuralNetLearner::new(spec, task)));
    }
    #[cfg(not(feature = "ml-neural"))]
    {
        let _ = (spec, task);
        Err(LearnError::ProviderUnavailable {
            spec: "neural_net",
            required: required_capabilities(LearnerSpec::NeuralNet(crate::NeuralSpec::default())),
        })
    }
}

// Only the stubs of a disabled learner feature report what the missing provider would offer.
#[cfg(not(all(feature = "ml-gbdt", feature = "ml-forest", feature = "ml-neural")))]
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
