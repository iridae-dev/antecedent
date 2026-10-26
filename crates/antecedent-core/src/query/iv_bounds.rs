//! Sharp bounds on the average effect of a binary treatment under a binary instrument.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::fault::IvBoundsFault;
use super::QueryError;
use crate::canonical_encode::{CanonicalEncode, encode_option};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// Balke-Pearl bounds on `E[Y(1)] - E[Y(0)]` from the observed law of a binary instrument `Z`, a
/// binary treatment `D` and a binary outcome `Y`.
///
/// The three indicators are table columns coded 0 or 1, and the optional sampling weight is a
/// column. The estimand is the average effect of `D` on `Y` over the whole population. It is
/// partially identified, not a point: the interval is sharp under instrument independence of the
/// latent response types and the exclusion restriction (the instrument reaches the outcome only
/// through the treatment), with no monotonicity. The answer is the identified set of effects
/// those premises leave open, never an estimate of one of them.
#[derive(Clone, Debug, PartialEq)]
pub struct IvBoundsQuery {
    /// Binary outcome column.
    pub outcome: VariableId,
    /// Binary treatment column.
    pub treatment: VariableId,
    /// Binary instrument column.
    pub instrument: VariableId,
    /// Sampling-weight column; `None` weighs every row `1`.
    pub weight: Option<VariableId>,
}

/// Every field is written; the three roles are positional.
impl CanonicalEncode for IvBoundsQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        self.treatment.encode(writer);
        self.instrument.encode(writer);
        encode_option(writer, self.weight.as_ref());
    }
}

impl IvBoundsQuery {
    /// Bounds with every row weighed `1`.
    #[must_use]
    pub const fn new(outcome: VariableId, treatment: VariableId, instrument: VariableId) -> Self {
        Self { outcome, treatment, instrument, weight: None }
    }

    /// Weigh the rows by a sampling-weight column.
    #[must_use]
    pub const fn with_weight(mut self, weight: VariableId) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Validate the declared roles before compiling a study.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidIvBounds`] when the outcome, treatment, instrument and weight
    /// columns are not four distinct columns.
    pub fn validate(&self) -> Result<(), QueryError> {
        let mut columns = vec![self.outcome, self.treatment, self.instrument];
        columns.extend(self.weight);
        let mut sorted = columns.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != columns.len() {
            return Err(QueryError::InvalidIvBounds(IvBoundsFault::Columns));
        }
        Ok(())
    }
}
