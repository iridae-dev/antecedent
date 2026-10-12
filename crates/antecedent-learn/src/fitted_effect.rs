//! Retained effect prediction, distinct from marginal-effect inference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId};
use serde::{Deserialize, Serialize};

use crate::design::DesignView;
use crate::error::LearnError;
use crate::learner::FittedPredictor;
use crate::portable::PortablePredictor;

/// Why a retained effect model could not be validated or applied.
///
/// The texts are the ones the estimator layer reported for the same refusals, so a
/// consumer that renders them sees no change.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FittedEffectError {
    /// The schema, the requested features or the prediction workspace are invalid.
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
    /// The portable predictor refused.
    #[error("backend error: {0}")]
    Predictor(String),
}

impl FittedEffectError {
    fn predictor(error: &LearnError) -> Self {
        Self::Predictor(error.to_string())
    }
}

/// A fitted CATE map and its exact ordered feature schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedEffect {
    /// Codec version.
    pub version: u32,
    /// Adjustment coordinates, excluding a generated intercept.
    pub features: Vec<u32>,
    /// Whether the fitted design includes a leading constant one.
    pub intercept: bool,
    /// Immutable model state. No refitting occurs during prediction.
    pub predictor: PortablePredictor,
}

impl FittedEffect {
    /// Validate schema and the numerical map.
    /// # Errors
    /// Unknown version, duplicate coordinates, or invalid predictor state.
    pub fn validate(&self) -> Result<(), FittedEffectError> {
        if self.version != 1
            || self.features.iter().collect::<std::collections::BTreeSet<_>>().len()
                != self.features.len()
            || self.predictor.columns != self.features.len() + usize::from(self.intercept)
        {
            return Err(FittedEffectError::InvalidArgument("invalid fitted effect schema"));
        }
        self.predictor.validate().map_err(|e| FittedEffectError::predictor(&e))
    }

    /// Predict from ordered raw feature columns, adding the retained intercept.
    /// # Errors
    /// Schema mismatch, nonfinite data, cancellation, or unavailable predictions.
    pub fn predict(
        &self,
        features: &[VariableId],
        columns: &[&[f64]],
        nrows: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, FittedEffectError> {
        self.validate()?;
        if features.iter().map(|v| v.raw()).collect::<Vec<_>>() != self.features
            || columns.len() != features.len()
            || columns.iter().any(|c| c.len() != nrows || c.iter().any(|v| !v.is_finite()))
        {
            return Err(FittedEffectError::InvalidArgument("prediction feature schema mismatch"));
        }
        let size = nrows
            .checked_mul(self.predictor.columns)
            .ok_or(FittedEffectError::InvalidArgument("prediction design too large"))?;
        let bytes = size
            .checked_add(nrows)
            .and_then(|n| n.checked_mul(8))
            .ok_or(FittedEffectError::InvalidArgument("prediction workspace overflow"))?;
        if ctx.cancellation.is_cancelled()
            || ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes as u64 > limit)
        {
            return Err(FittedEffectError::InvalidArgument(
                "prediction workspace budget/cancellation",
            ));
        }
        let mut design = Vec::with_capacity(size);
        if self.intercept {
            design.resize(nrows, 1.0);
        }
        for column in columns {
            design.extend_from_slice(column);
        }
        let view = DesignView::from_column_major(&design, nrows, self.predictor.columns)
            .map_err(|e| FittedEffectError::predictor(&e))?;
        let mut out = vec![0.0; nrows];
        self.predictor
            .predict(view, &mut out, ctx)
            .map_err(|e| FittedEffectError::predictor(&e))?;
        Ok(out)
    }
}
