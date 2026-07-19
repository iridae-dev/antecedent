//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;
use crate::intervention::Intervention;

use super::error::QueryError;

#[derive(Clone, Debug, PartialEq)]
/// Counterfactual query over factual observations and interventions.
pub struct CounterfactualQuery {
    /// Outcome variable(s) to predict under the counterfactual world.
    pub outcomes: Arc<[VariableId]>,
    /// Interventions defining the counterfactual world (applied after abduction).
    pub interventions: Arc<[Intervention]>,
    /// When true, allow nested counterfactual interventions under invertible SCMs.
    pub allow_nested: bool,
}

impl CounterfactualQuery {
    /// Construct a single-outcome counterfactual query.
    #[must_use]
    pub fn new(outcome: VariableId, interventions: impl Into<Arc<[Intervention]>>) -> Self {
        Self {
            outcomes: Arc::from([outcome]),
            interventions: interventions.into(),
            allow_nested: false,
        }
    }

    /// Enable nested interventions where the model supports them.
    #[must_use]
    pub const fn with_nested(mut self, allow_nested: bool) -> Self {
        self.allow_nested = allow_nested;
        self
    }

    /// Validate interventions.
    ///
    /// # Errors
    ///
    /// Empty outcomes or invalid interventions.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcomes.is_empty() {
            return Err(QueryError::EmptyCounterfactualOutcomes);
        }
        for iv in self.interventions.iter() {
            iv.validate().map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
        }
        Ok(())
    }
}

