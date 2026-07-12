//! Interventions on causal variables (DESIGN.md §8.1).
//!
//! Phase 1 supports hard `Set` interventions for ATE contrasts. Richer variants
//! are represented but rejected by Phase 1 facades until later phases.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::VariableId;
use crate::value::Value;

/// An intervention applied to one or more variables.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Intervention {
    /// Hard assignment `do(variable := value)`.
    Set {
        /// Target variable.
        variable: VariableId,
        /// Assigned value.
        value: Value,
    },
}

impl Intervention {
    /// Hard set intervention.
    #[must_use]
    pub const fn set(variable: VariableId, value: Value) -> Self {
        Self::Set { variable, value }
    }

    /// Variable targeted by this intervention, when unique.
    #[must_use]
    pub const fn primary_variable(&self) -> VariableId {
        match self {
            Self::Set { variable, .. } => *variable,
        }
    }
}
