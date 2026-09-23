//! Learner factory / fitted predictor contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use crate::design::{DesignView, TargetView};
use crate::error::LearnError;

/// What a nuisance model must emit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredictionTask {
    /// `E[Y | X]` — real-valued prediction.
    Regression,
    /// `P(T = 1 | X)` — a calibrated probability in `[0, 1]`.
    BinaryProbability,
}

/// Capabilities the planner checks before fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct LearnerCapabilities {
    /// Real-valued regression.
    pub regression: bool,
    /// Calibrated binary probabilities.
    pub binary_probability: bool,
    /// Sample weights at fit time.
    pub sample_weights: bool,
    /// Native missing-value handling.
    pub missing_values: bool,
    /// Sparse design input.
    pub sparse_input: bool,
    /// Native categorical columns (no one-hot).
    pub categorical_native: bool,
    /// Deterministic given `ExecutionContext` seed.
    pub deterministic_seed: bool,
    /// Honors `ExecutionContext` thread budget.
    pub parallelism_control: bool,
}

impl LearnerCapabilities {
    /// No capabilities.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            regression: false,
            binary_probability: false,
            sample_weights: false,
            missing_values: false,
            sparse_input: false,
            categorical_native: false,
            deterministic_seed: false,
            parallelism_control: false,
        }
    }
}

/// Hidden implementation identity recorded on artifacts.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnerProvenance {
    /// Public spec name (`"linear"`, `"gradient_boosted_trees"`, …).
    pub spec: String,
    /// Hidden provider (`"faer"`, `"forust-ml"`, …).
    pub implementation: String,
    /// Provider crate / pin version.
    pub version: String,
}

/// Unfitted learner. `antecedent-estimate` depends on this trait only.
pub trait LearnerFactory: Send + Sync {
    /// Prediction task this factory implements.
    fn task(&self) -> PredictionTask;

    /// Planner-visible capabilities.
    fn capabilities(&self) -> LearnerCapabilities;

    /// Fit on a (possibly row-selected) design.
    ///
    /// `y` and `weights` are aligned with the design's **physical** rows.
    ///
    /// # Errors
    ///
    /// Shape, task, or backend failure.
    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError>;
}

/// Fitted predictor. Writes one value per logical design row into `out`.
pub trait FittedPredictor: Send + Sync {
    /// Export a provider-independent numerical map, when supported.
    /// # Errors
    /// A provider without a codec explicitly refuses portable export.
    fn portable(&self) -> Result<crate::PortablePredictor, LearnError> {
        Err(LearnError::Unsupported { message: "portable export unavailable for this predictor" })
    }

    /// Predict into `out` (`out.len()` must equal `x.nrows()`).
    ///
    /// # Errors
    ///
    /// Shape or backend failure.
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), LearnError>;

    /// Implementation provenance.
    fn provenance(&self) -> LearnerProvenance;
}
