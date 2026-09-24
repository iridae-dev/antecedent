//! A narrowly licensed natural direct effect query.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::error::QueryError;
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

impl NestedCounterfactualQuery {
    /// Construct the natural direct effect using treatment levels zero and one.
    pub fn new(
        treatment: VariableId,
        mediator: VariableId,
        outcome: VariableId,
    ) -> Result<Self, QueryError> {
        Self::with_levels(treatment, mediator, outcome, 0.0, 1.0)
    }

    /// Construct with explicit finite control and active treatment values.
    pub fn with_levels(
        treatment: VariableId,
        mediator: VariableId,
        outcome: VariableId,
        control: f64,
        active: f64,
    ) -> Result<Self, QueryError> {
        if !control.is_finite() || !active.is_finite() || control == active {
            return Err(QueryError::InvalidIntervention(
                "nested treatment levels must be finite and distinct".into(),
            ));
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
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.mediator
            || self.treatment == self.outcome
            || self.mediator == self.outcome
        {
            return Err(QueryError::InvalidIntervention(
                "nested treatment, mediator, and outcome must be distinct".into(),
            ));
        }
        if !self.control_value().is_finite()
            || !self.active_value().is_finite()
            || self.control_bits == self.active_bits
        {
            return Err(QueryError::InvalidIntervention(
                "nested treatment levels must be finite and distinct".into(),
            ));
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
