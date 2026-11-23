// SPDX-License-Identifier: MIT OR Apache-2.0

//! The unit of a population: one panel unit or one environment.

use std::fmt;

/// One panel unit or one environment.
///
/// The id is the unit's identity, not a row index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PopulationUnit {
    /// A panel unit.
    Panel {
        /// [`PanelUnit::unit_id`](crate::panel::PanelUnit::unit_id).
        unit_id: u32,
    },
    /// One environment in an environment list.
    Environment {
        /// Index of the environment in that list.
        env_index: u32,
    },
}

impl PopulationUnit {
    /// Panel unit `unit_id`.
    #[must_use]
    pub const fn panel(unit_id: u32) -> Self {
        Self::Panel { unit_id }
    }

    /// Environment `env_index`.
    #[must_use]
    pub const fn environment(env_index: u32) -> Self {
        Self::Environment { env_index }
    }
}

impl fmt::Display for PopulationUnit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Panel { unit_id } => write!(formatter, "panel unit {unit_id}"),
            Self::Environment { env_index } => write!(formatter, "environment {env_index}"),
        }
    }
}
