//! A narrowly licensed natural direct effect query.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::error::QueryError;
use super::fault::InterventionFault;
use crate::canonical_encode::CanonicalEncode;
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::Intervention;
use crate::value::Value;

/// Natural direct effect `E[Y(1, M(0)) - Y(0, M(0))]` for one mediator.
///
/// Execution licenses the fixed Markovian DAG `X -> M`, `X -> Y`, `M -> Y`
/// with linear Gaussian mechanisms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NestedCounterfactualQuery {
    /// Treatment node.
    pub treatment: VariableId,
    /// Mediator held to its control-world value.
    pub mediator: VariableId,
    /// Outcome node.
    pub outcome: VariableId,
    control_bits: u64,
    active_bits: u64,
}

/// Every field is written; the two treatment levels as their exact bits.
impl CanonicalEncode for NestedCounterfactualQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.treatment.encode(writer);
        self.mediator.encode(writer);
        self.outcome.encode(writer);
        writer.word(self.control_bits).word(self.active_bits);
    }
}

impl NestedCounterfactualQuery {
    /// Construct the natural direct effect using treatment levels zero and one.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidIntervention`] if the variable roles are
    /// not distinct or either treatment level is non-finite.
    pub fn new(
        treatment: VariableId,
        mediator: VariableId,
        outcome: VariableId,
    ) -> Result<Self, QueryError> {
        Self::with_levels(treatment, mediator, outcome, 0.0, 1.0)
    }

    /// Construct with explicit finite, numerically distinct treatment values.
    /// The original IEEE-754 bits are retained in equality and query identity,
    /// including the sign of a zero in either otherwise valid contrast.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidIntervention`] if the variable roles are
    /// not distinct, a treatment level is non-finite, or the levels compare equal.
    pub fn with_levels(
        treatment: VariableId,
        mediator: VariableId,
        outcome: VariableId,
        control: f64,
        active: f64,
    ) -> Result<Self, QueryError> {
        // Exact ordering equality is the contract here: any distinct finite
        // levels define a valid contrast, however close they are.
        if !control.is_finite()
            || !active.is_finite()
            || control.partial_cmp(&active) == Some(std::cmp::Ordering::Equal)
        {
            return Err(QueryError::InvalidIntervention(InterventionFault::NestedLevels));
        }
        let query = Self {
            treatment,
            mediator,
            outcome,
            control_bits: control.to_bits(),
            active_bits: active.to_bits(),
        };
        query.validate()?;
        Ok(query)
    }

    /// Validate distinct treatment, mediator, and outcome roles.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidIntervention`] when a variable role is
    /// repeated or the treatment levels are non-finite or identical.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.mediator
            || self.treatment == self.outcome
            || self.mediator == self.outcome
        {
            return Err(QueryError::InvalidIntervention(InterventionFault::NestedRolesNotDistinct));
        }
        if !self.control_value().is_finite()
            || !self.active_value().is_finite()
            || self.control_bits == self.active_bits
        {
            return Err(QueryError::InvalidIntervention(InterventionFault::NestedLevels));
        }
        Ok(())
    }

    /// Control treatment level.
    #[must_use]
    pub fn control_value(&self) -> f64 {
        f64::from_bits(self.control_bits)
    }
    /// Active treatment level.
    #[must_use]
    pub fn active_value(&self) -> f64 {
        f64::from_bits(self.active_bits)
    }

    /// Represent this natural direct effect using the linear mediation contract.
    #[must_use]
    pub fn as_mediation_query(&self) -> super::MediationQuery {
        let mut q = super::MediationQuery::binary(
            self.treatment,
            self.outcome,
            [self.mediator],
            super::MediationContrast::NaturalDirect,
        );
        q.control = Intervention::set(self.treatment, Value::f64(self.control_value()));
        q.active = Intervention::set(self.treatment, Value::f64(self.active_value()));
        q
    }
}

#[cfg(test)]
mod signed_zero_tests {
    use super::*;
    #[test]
    fn distinct_levels_keep_their_bits_but_two_zero_levels_are_not_a_contrast() {
        let (t, m, y) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let negative = NestedCounterfactualQuery::with_levels(t, m, y, -0.0, 1.0).unwrap();
        let positive = NestedCounterfactualQuery::with_levels(t, m, y, 0.0, 1.0).unwrap();
        assert_ne!(negative, positive);
        assert_eq!(negative.control_value().to_bits(), (-0.0_f64).to_bits());
        assert!(matches!(
            NestedCounterfactualQuery::with_levels(t, m, y, -0.0, 0.0),
            Err(QueryError::InvalidIntervention(InterventionFault::NestedLevels))
        ));
        let mut a = CanonicalWriter::new("test.nested.signed-zero");
        let mut b = CanonicalWriter::new("test.nested.signed-zero");
        negative.encode(&mut a);
        positive.encode(&mut b);
        assert_ne!(a.finish(), b.finish());
    }
}
