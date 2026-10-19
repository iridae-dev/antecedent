//! Plain-data form of a cross-fitted score table.
//!
//! One definition shared by the estimator's decoder (`ScoreTable::from_wire`)
//! and the frozen-scores artifact, so the value checks that gate a stored table
//! are the checks the estimator applies to a live one. The embedded
//! `analysis_wire::ScoreTableWire` of the result body is a separate, defaulting
//! form with its own bytes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Column key: one interventional arm, optionally at an exceedance threshold.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreColumn {
    /// Arm label (`0`/`1` for binary, or a cell mask for joint treatments).
    pub arm: u32,
    /// Exceedance threshold. `None` is the mean functional.
    pub threshold: Option<f64>,
}

/// Why a score table's plain data is not a valid table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ScoreTableWireError {
    /// Row, column or fold vectors disagree in length.
    #[error("score table wire shape mismatch")]
    ShapeMismatch,
    /// A score, propensity, outcome, fold id, column or clip is out of range.
    #[error("invalid score table values or support shape")]
    InvalidValues,
}

/// The score table as plain data, for cross-language round trips and artifacts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreTableWire {
    /// Observed cell on each retained row.
    pub observed_arm: Vec<u32>,
    /// Out-of-fold raw propensity for each score column, column-major.
    pub propensities: Vec<f64>,
    /// Observed original outcome.
    pub observed_outcome: Vec<f64>,
    /// Complete-case row count.
    pub n_rows: u64,
    /// Original data-frame row index of each complete-case row (the row ids).
    pub row_index: Vec<u64>,
    /// Fold id of each row.
    pub fold_ids: Vec<u32>,
    /// Fold count.
    pub n_folds: u32,
    /// Column-major scores.
    pub scores: Vec<f64>,
    /// Column keys.
    pub columns: Vec<ScoreColumn>,
    /// Adjustment variable raw ids.
    pub adjustment_set: Vec<u32>,
    /// Nuisance provenance (the estimator identity label).
    pub nuisance_provenance: String,
    /// Applied propensity clip (`None`: unclipped).
    pub propensity_clip: Option<f64>,
    /// Treatment raw id.
    pub treatment: u32,
    /// Extra intervened raw ids.
    pub intervened: Vec<u32>,
}

impl ScoreTableWire {
    /// Check lengths, finiteness and ranges: the conditions under which the
    /// estimator accepts the table.
    ///
    /// # Errors
    /// [`ScoreTableWireError::ShapeMismatch`] for disagreeing lengths,
    /// [`ScoreTableWireError::InvalidValues`] for a non-finite or out-of-range value.
    pub fn check(&self) -> Result<(), ScoreTableWireError> {
        let n = usize::try_from(self.n_rows).unwrap_or(usize::MAX);
        if self.row_index.len() != n
            || self.fold_ids.len() != n
            || self.scores.len() != n.saturating_mul(self.columns.len())
        {
            return Err(ScoreTableWireError::ShapeMismatch);
        }
        if (!self.observed_arm.is_empty() && self.observed_arm.len() != n)
            || (!self.observed_outcome.is_empty() && self.observed_outcome.len() != n)
            || (!self.propensities.is_empty() && self.propensities.len() != self.scores.len())
            || self.scores.iter().any(|v| !v.is_finite())
            || self.propensities.iter().any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
            || self.observed_outcome.iter().any(|v| !v.is_finite())
            || self.n_folds < 2
            || self.fold_ids.iter().any(|&f| f >= self.n_folds)
            || self.columns.iter().any(|c| c.threshold.is_some_and(|v| !v.is_finite()))
            || self.propensity_clip.is_some_and(|c| !(c > 0.0 && c < 0.5))
        {
            return Err(ScoreTableWireError::InvalidValues);
        }
        Ok(())
    }
}
