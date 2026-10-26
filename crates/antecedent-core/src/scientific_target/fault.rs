//! Typed reasons behind scientific-target refusals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::intervention::InterventionError;
use crate::query::QueryError;
use crate::value::InterventionValueError;

/// Why an intervention, policy, functional, or population declaration is unusable.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ActionFault {
    /// An intervention or temporal policy failed its own validation.
    #[error("{0}")]
    Intervention(InterventionError),
    /// An intervention level is not a concrete value.
    #[error("{0}")]
    Level(InterventionValueError),
    /// The target population failed validation.
    #[error("{0}")]
    Population(QueryError),
    /// The outcome functional is not a valid scientific quantity.
    #[error("outcome functional is not a valid scientific quantity")]
    OutcomeFunctional,
    /// A level or parameter of a structural action is NaN or infinite.
    #[error("{what} must be a finite level")]
    NonFiniteLevel {
        /// Which quantity was not finite.
        what: &'static str,
    },
    /// A world declares no intervention.
    #[error("a world needs at least one intervention")]
    NoIntervention,
    /// A mediation treatment intervention does not target exactly one variable.
    #[error("mediation treatment intervention must target one variable")]
    MediationTreatmentTarget,
}

/// Why the worlds of a contrast do not realise the contrast's formula.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ContrastIncoherence {
    /// The two worlds of a difference do not assign the same variables on the same schedules.
    #[error("the two worlds assign different variables or schedules")]
    TwoWorldsDiffer,
    /// The two worlds of a difference are the same intervention.
    #[error("the two worlds are the same intervention")]
    TwoWorldsIdentical,
    /// The four worlds of an interaction do not assign the same variables on the same schedules.
    #[error("the four worlds assign different variables or schedules")]
    FourWorldsDiffer,
    /// The two factors of an interaction move the same variable or schedule.
    #[error("the two factors move the same variable or schedule")]
    FactorsOverlap,
    /// The joint world is not the reference moved by both factors.
    #[error("the joint world is not the reference moved by both factors")]
    JointNotComposed,
    /// A factor does not move away from the reference.
    #[error("each factor must move away from the reference")]
    FactorNotMoved,
}

/// Why a dose × time declaration is not a rectangular grid of temporal worlds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DoseGridFault {
    /// Dose levels are not finite and strictly increasing.
    #[error("dose levels must be finite and strictly increasing")]
    DoseLevels,
    /// The grid has no dose or no horizon.
    #[error("a dose-time grid names at least one dose and one horizon")]
    EmptyGrid,
    /// Horizons are not strictly increasing and at least one.
    #[error("horizons must be strictly increasing and at least 1")]
    Horizons,
    /// The grid has more cells than the target admits.
    #[error("dose-time grid is too large")]
    TooLarge,
    /// The cells are not exactly one world per dose and horizon.
    #[error("cells must be one world per dose and horizon, and no others")]
    CellCoverage,
    /// A world fills more than one cell.
    #[error("a world is used for more than one cell")]
    SharedWorld,
    /// The dose and the outcome are one variable.
    #[error("dose and outcome are the same variable")]
    DoseIsOutcome,
    /// A cell's world does not match the dose and horizon of its position.
    #[error("a cell does not match its dose and horizon")]
    CellMismatch,
    /// Cells do not share a policy, history, and initial law.
    #[error("cells must share a policy, history, and initial law")]
    CellsNotComparable,
    /// A cell is not a temporal world.
    #[error("each cell is a temporal world")]
    NotTemporal,
    /// A cell is not a scheduled dose assignment.
    #[error("each cell is a scheduled dose assignment")]
    NotScheduledDose,
    /// A cell does not hard-set the dose.
    #[error("each cell hard-sets the dose")]
    NotHardSet,
}
