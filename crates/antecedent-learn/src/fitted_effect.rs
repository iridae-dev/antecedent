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

/// Closed range one retained feature took over the rows the model was fitted on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureRange {
    /// Smallest fitted value.
    pub lo: f64,
    /// Largest fitted value.
    pub hi: f64,
}

/// The support contract of a retained effect model: the closed box of each feature over
/// the rows the final model was fitted on.
///
/// A box is a necessary condition for support, not a sufficient one: a row inside every
/// range can still lie in a region with no fitted rows. It is the checkable contract the
/// producer retains; a consumer must not describe it as a density or overlap guarantee.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectSupport {
    /// Codec version.
    pub version: u32,
    /// Number of rows the final model was fitted on.
    pub rows: u64,
    /// One range per [`FittedEffect::features`] entry, in the same order.
    pub ranges: Vec<FeatureRange>,
}

/// Where a candidate cohort falls against an [`EffectSupport`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SupportReport {
    /// Rows with at least one feature outside its fitted range.
    pub rows_outside: usize,
    /// First such row, in input order.
    pub first_row: Option<usize>,
    /// Index into [`FittedEffect::features`] of the first violated feature of that row.
    pub first_feature: Option<usize>,
}

impl EffectSupport {
    /// Version of the support codec.
    pub const VERSION: u32 = 1;

    /// Record the range of each feature column over `rows` (all rows when `None`).
    ///
    /// Returns `None` when a column is empty, has a nonfinite value, has a different
    /// length from the others, or no row is selected, so a producer never retains a
    /// contract it could not compute.
    #[must_use]
    pub fn from_rows(columns: &[&[f64]], rows: Option<&[usize]>) -> Option<Self> {
        let first = columns.first()?;
        let n = first.len();
        if columns.iter().any(|column| column.len() != n) {
            return None;
        }
        let selected: Vec<usize> = match rows {
            Some(rows) => rows.to_vec(),
            None => (0..n).collect(),
        };
        if selected.is_empty() || selected.iter().any(|&row| row >= n) {
            return None;
        }
        let mut ranges = Vec::with_capacity(columns.len());
        for column in columns {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &row in &selected {
                let value = column[row];
                if !value.is_finite() {
                    return None;
                }
                lo = lo.min(value);
                hi = hi.max(value);
            }
            ranges.push(FeatureRange { lo, hi });
        }
        Some(Self { version: Self::VERSION, rows: selected.len() as u64, ranges })
    }

    fn validate(&self, features: usize) -> bool {
        self.version == Self::VERSION
            && self.rows > 0
            && self.ranges.len() == features
            && self.ranges.iter().all(|r| r.lo.is_finite() && r.hi.is_finite() && r.lo <= r.hi)
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
    /// Support contract of the fitted rows. Absent on a model whose producer did not
    /// retain one; a consumer that needs row-level support refuses its absence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub support: Option<EffectSupport>,
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
        if self.support.as_ref().is_some_and(|support| !support.validate(self.features.len())) {
            return Err(FittedEffectError::InvalidArgument("invalid fitted effect support"));
        }
        self.predictor.validate().map_err(|e| FittedEffectError::predictor(&e))
    }

    /// Check rows against the retained support contract, or `None` when the model
    /// retained none. Columns are ordered as [`Self::features`] and must be finite.
    ///
    /// # Errors
    /// A column count or length that does not match the schema, or a nonfinite value.
    pub fn check_support(
        &self,
        columns: &[&[f64]],
        nrows: usize,
    ) -> Result<Option<SupportReport>, FittedEffectError> {
        let Some(support) = &self.support else {
            return Ok(None);
        };
        if columns.len() != support.ranges.len()
            || columns.iter().any(|c| c.len() != nrows || c.iter().any(|v| !v.is_finite()))
        {
            return Err(FittedEffectError::InvalidArgument("support feature schema mismatch"));
        }
        let mut report = SupportReport { rows_outside: 0, first_row: None, first_feature: None };
        for row in 0..nrows {
            let violated = columns
                .iter()
                .zip(&support.ranges)
                .position(|(column, range)| column[row] < range.lo || column[row] > range.hi);
            if let Some(feature) = violated {
                report.rows_outside += 1;
                if report.first_row.is_none() {
                    report.first_row = Some(row);
                    report.first_feature = Some(feature);
                }
            }
        }
        Ok(Some(report))
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
            || ctx.memory_limit_bytes().is_some_and(|limit| bytes as u64 > limit)
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
