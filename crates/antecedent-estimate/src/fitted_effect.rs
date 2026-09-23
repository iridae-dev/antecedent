//! Retained effect prediction, distinct from marginal-effect inference.

use crate::EstimationError;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_learn::{DesignView, FittedPredictor, PortablePredictor};
use serde::{Deserialize, Serialize};

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
    pub fn validate(&self) -> Result<(), EstimationError> {
        if self.version != 1
            || self.features.iter().collect::<std::collections::BTreeSet<_>>().len()
                != self.features.len()
            || self.predictor.columns != self.features.len() + usize::from(self.intercept)
        {
            return Err(EstimationError::data_msg("invalid fitted effect schema"));
        }
        self.predictor.validate().map_err(crate::learn_nuisance::learn_err)
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
    ) -> Result<Vec<f64>, EstimationError> {
        self.validate()?;
        if features.iter().map(|v| v.raw()).collect::<Vec<_>>() != self.features
            || columns.len() != features.len()
            || columns.iter().any(|c| c.len() != nrows || c.iter().any(|v| !v.is_finite()))
        {
            return Err(EstimationError::data_msg("prediction feature schema mismatch"));
        }
        let size = nrows
            .checked_mul(self.predictor.columns)
            .ok_or_else(|| EstimationError::data_msg("prediction design too large"))?;
        let bytes = size
            .checked_add(nrows)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| EstimationError::data_msg("prediction workspace overflow"))?;
        if ctx.cancellation.is_cancelled()
            || ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes as u64 > limit)
        {
            return Err(EstimationError::data_msg("prediction workspace budget/cancellation"));
        }
        let mut design = Vec::with_capacity(size);
        if self.intercept {
            design.resize(nrows, 1.0);
        }
        for column in columns {
            design.extend_from_slice(column);
        }
        let view = DesignView::from_column_major(&design, nrows, self.predictor.columns)
            .map_err(crate::learn_nuisance::learn_err)?;
        let mut out = vec![0.0; nrows];
        self.predictor.predict(view, &mut out, ctx).map_err(crate::learn_nuisance::learn_err)?;
        Ok(out)
    }
}
