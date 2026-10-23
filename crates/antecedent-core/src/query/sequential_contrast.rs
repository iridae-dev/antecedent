//! Declared observational two-decision contrast of two static regimes.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::QueryError;
use crate::canonical_encode::{CanonicalEncode, encode_u32s, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// Contrast of two static two-decision regimes from one observed row per subject.
///
/// The first binary action, the covariates measured after it, the second binary action and the
/// terminal outcome are table columns. The query names them, the two regimes, the cross-fit fold
/// of every row and the positivity floor. Sequential conditional exchangeability given the
/// measured history, positivity, consistency and no interference are assumed, not derived.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialContrastQuery {
    /// Terminal outcome column.
    pub outcome: VariableId,
    /// First-decision action column, coded 0 or 1.
    pub first_treatment: VariableId,
    /// Second-decision action column, coded 0 or 1.
    pub second_treatment: VariableId,
    /// Baseline covariate columns.
    pub baseline: Arc<[VariableId]>,
    /// Covariate columns measured after the first decision and before the second.
    pub intermediate: Arc<[VariableId]>,
    /// Active regime: the actions at the first and second decision.
    pub active: [u8; 2],
    /// Control regime: the actions at the first and second decision.
    pub control: [u8; 2],
    /// Cross-fit fold of each row.
    pub fold_of_row: Arc<[u32]>,
    /// Number of folds; at least two.
    pub fold_count: u32,
    /// Positivity floor of every regime action probability, in `(0, 0.5]`.
    pub min_probability: f64,
}

/// Every field is written. `fold_of_row` is row-aligned and keeps its order.
/// `baseline` and `intermediate` keep their order too: they are model
/// covariate columns, and the conservative choice is never to merge two
/// spellings. The regimes keep their decision order.
impl CanonicalEncode for SequentialContrastQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        self.first_treatment.encode(writer);
        self.second_treatment.encode(writer);
        encode_variables(writer, &self.baseline);
        encode_variables(writer, &self.intermediate);
        writer.bytes(&self.active).bytes(&self.control);
        encode_u32s(writer, &self.fold_of_row);
        writer.word(u64::from(self.fold_count)).real(self.min_probability);
    }
}

impl SequentialContrastQuery {
    /// Validate the declared contrast before compiling a study.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidSequentialContrast`] for a repeated column, a regime action
    /// other than 0 or 1, fewer than two folds, a fold outside the count, no rows, or a floor
    /// outside `(0, 0.5]`.
    pub fn validate(&self) -> Result<(), QueryError> {
        let mut columns = vec![self.outcome, self.first_treatment, self.second_treatment];
        columns.extend(self.baseline.iter().copied());
        columns.extend(self.intermediate.iter().copied());
        let mut sorted = columns.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != columns.len() {
            return Err(QueryError::InvalidSequentialContrast(
                "outcome, both actions and every covariate must be distinct columns".into(),
            ));
        }
        if self.active.iter().chain(&self.control).any(|action| *action > 1) {
            return Err(QueryError::InvalidSequentialContrast(
                "a regime action must be 0 or 1".into(),
            ));
        }
        if self.fold_count < 2
            || self.fold_of_row.is_empty()
            || self.fold_of_row.iter().any(|fold| *fold >= self.fold_count)
        {
            return Err(QueryError::InvalidSequentialContrast(
                "cross-fitting needs at least two folds and a fold in range for every row".into(),
            ));
        }
        if !(self.min_probability.is_finite()
            && self.min_probability > 0.0
            && self.min_probability <= 0.5)
        {
            return Err(QueryError::InvalidSequentialContrast(
                "min_probability must be in (0, 0.5]".into(),
            ));
        }
        Ok(())
    }
}
