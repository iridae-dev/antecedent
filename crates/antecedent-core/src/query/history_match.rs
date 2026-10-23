//! Declared nearest-neighbor contrast on pre-outcome histories within calendar periods.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::QueryError;
use crate::canonical_encode::{CanonicalEncode, encode_option, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// Nearest-neighbor contrast on the mean of each unit's history, matched inside its period.
///
/// The treatment indicator, the time-ordered history columns, the outcome, the optional calendar
/// period and the optional sampling weight are table columns. A treated unit is matched only to
/// a control of its own period. The estimand is the sampling-weighted effect on the matched
/// treated units, conditional on the period and the history-mean score; it is identified under
/// unconfoundedness given that score within a period, overlap, stable units and no interference.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryMatchQuery {
    /// Outcome contrasted after the history.
    pub outcome: VariableId,
    /// Treatment indicator column, coded 0 or 1.
    pub treated: VariableId,
    /// Time-ordered history columns; their row-wise mean is the match score.
    pub history: Arc<[VariableId]>,
    /// Calendar period column (integer valued); `None` puts every unit in one period.
    pub period: Option<VariableId>,
    /// Sampling-weight column; `None` weighs every unit `1`.
    pub weight: Option<VariableId>,
    /// Maximum absolute score distance of a match; `None` for no caliper.
    pub caliper: Option<f64>,
}

/// Every field is written. `history` is time-ordered and keeps its order.
impl CanonicalEncode for HistoryMatchQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        self.treated.encode(writer);
        encode_variables(writer, &self.history);
        encode_option(writer, self.period.as_ref());
        encode_option(writer, self.weight.as_ref());
        writer.opt_real(self.caliper);
    }
}

impl HistoryMatchQuery {
    /// Validate the declared match before compiling a study.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidHistoryMatch`] for no history column, a repeated column, or a
    /// caliper that is negative or non-finite.
    pub fn validate(&self) -> Result<(), QueryError> {
        let mut columns = vec![self.outcome, self.treated];
        columns.extend(self.history.iter().copied());
        columns.extend(self.period);
        columns.extend(self.weight);
        let mut sorted = columns.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if self.history.is_empty() || sorted.len() != columns.len() {
            return Err(QueryError::InvalidHistoryMatch(
                "declare at least one history column and distinct outcome, treatment, history, period and weight columns".into(),
            ));
        }
        if self.caliper.is_some_and(|limit| !(limit.is_finite() && limit >= 0.0)) {
            return Err(QueryError::InvalidHistoryMatch(
                "the caliper must be finite and non-negative".into(),
            ));
        }
        Ok(())
    }
}
