//! Inference-independent scientific target.
//!
//! A [`crate::CausalQuery`] is the question a route answers. This module is the
//! identity those questions are comparable under: population and observation
//! unit, quantity and units, intervention worlds, conditioning, decision time,
//! and graph or model assumptions. A static analysis is the one-decision-time
//! specialization. It is not implied by a horizon of one.
//!
//! A contrast assembles intervention worlds as one target. The reference
//! intervention is an explicit world, not the absence of treatment. A dose × time
//! surface is a declared grid of those worlds. Mediation is a separate target:
//! it carries cross-world premises, and a contrast does not construct it.
//!
//! Nothing here identifies, fits, or estimates. [`InferenceSelection`] is recorded
//! on a target and omitted from [`ScientificTarget::identity`]. A history-indexed
//! target law is optional and enters the identity only when it is attached.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod contrast;
mod identity;
mod mediation;
mod model;
#[cfg(test)]
mod tests;

pub use self::contrast::{ContrastForm, ContrastTerm, DoseTimeSurface, InterventionContrast};
pub use self::identity::{ScientificTarget, TargetForm};
pub use self::mediation::{CrossWorldPremise, MediationArms, MediationComponent, MediationTarget};
pub use self::model::{
    Conditioning, GraphFamily, HistoryIndexedExperiment, HistoryIndexedTargetLaw, HistoryWindow,
    InferenceSelection, InitialLaw, InterventionWorld, ModelAssumptions, ObservationUnit,
    OutcomeQuantity, QuantityUnits, ScientificWorldId, TimeFrame, WorldAction, WorldBinding,
    WorldCollection,
};

use crate::ids::VariableId;

/// Domain tag for [`ScientificTarget::identity`].
pub const SCIENTIFIC_TARGET_IDENTITY_TAG: &str = "antecedent.scientific_target.v1";

/// Payload version written after the domain tag. Bump when the canonical fields change.
///
/// Format 3 identifies a world, a world collection, and a dose × time surface by
/// scientific content. A caller-chosen world id and a reference flag do not enter.
pub const SCIENTIFIC_TARGET_FORMAT: u16 = 3;

/// Refusal from a scientific-target constructor.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ScientificTargetError {
    /// A name, unit, or law label is empty or whitespace.
    #[error("{field} must be non-blank")]
    EmptyName {
        /// Which label was blank.
        field: &'static str,
    },
    /// An intervention, policy, functional, or population declaration is unusable.
    #[error("{reason}")]
    InvalidAction {
        /// Stable explanation.
        reason: String,
    },
    /// A callback is a program binding, not a declared scientific action.
    #[error("a scientific world cannot carry a callback intervention")]
    CallbackAction,
    /// A temporal frame needs a horizon of at least one step.
    #[error("horizon_steps must be >= 1")]
    NonPositiveHorizon,
    /// Conditioning names the outcome.
    #[error("conditioning includes the outcome")]
    ConditioningIncludesOutcome,
    /// Decision time and the action disagree about whether a schedule exists.
    #[error("static decision time has no schedule, and temporal decision time must name one")]
    MismatchedActionTime,
    /// Two interventions in one world assign the same variable.
    #[error("variable {id} is intervened more than once in one world")]
    DuplicateInterventionVariable {
        /// Variable assigned twice.
        id: VariableId,
    },
    /// A collection was built with no worlds.
    #[error("a world collection names at least one world")]
    EmptyWorlds,
    /// Two worlds share an id.
    #[error("duplicate world id {id}")]
    DuplicateWorldId {
        /// Repeated id.
        id: ScientificWorldId,
    },
    /// Worlds disagree on population, unit, outcome, units, quantity, or conditioning.
    #[error(
        "worlds do not bind the same {field}; a collection needs one population, unit, outcome, units, quantity, and conditioning"
    )]
    IncompatibleBinding {
        /// `population`, `unit`, `outcome`, `units`, `quantity`, or `conditioning`.
        field: &'static str,
    },
    /// A contrast names a world the collection does not contain.
    #[error("contrast names unknown world {id}")]
    UnknownWorld {
        /// Missing id.
        id: ScientificWorldId,
    },
    /// A contrast repeats a world.
    #[error("a contrast uses each world once")]
    WorldsNotDistinct,
    /// The collection is not exactly the worlds the contrast formula names.
    #[error("a contrast's worlds are exactly the worlds in its formula")]
    ContrastWorldSet,
    /// A general linear functional has missing, zero, repeated or nonfinite coefficients.
    #[error("invalid linear world functional: {reason}")]
    InvalidLinearContrast {
        /// Exact invalid declaration.
        reason: &'static str,
    },
    /// The reference id is not flagged as the explicit reference intervention.
    #[error("reference world {id} is not marked as the explicit reference intervention")]
    ReferenceNotMarked {
        /// Id that was supposed to be the reference.
        id: ScientificWorldId,
    },
    /// A difference's reference is not the subtracted world.
    #[error("reference world {id} is not the subtracted baseline of this difference")]
    ReferenceIsNotBaseline {
        /// Id passed as the reference.
        id: ScientificWorldId,
    },
    /// A world other than the contrast's reference is flagged.
    #[error("world {id} is marked as a reference but is not this contrast's reference world")]
    UnexpectedReference {
        /// Extra flagged world.
        id: ScientificWorldId,
    },
    /// Contrasted worlds do not share decision time, horizon, history, and initial law.
    #[error("contrasted worlds do not share a decision time, horizon, history, and initial law")]
    IncompatibleTime,
    /// The dose × time declaration is not a rectangular grid of temporal worlds.
    #[error(
        "dose-time surface is a declared grid of worlds, not a contrast of intervals: {reason}"
    )]
    NotADoseTimeGrid {
        /// What about the grid failed.
        reason: &'static str,
    },
    /// A cross-world mediation component was declared without its premises.
    #[error(
        "mediation requires explicit cross-world premises; algebra of interventional expectations does not supply them"
    )]
    MediationNeedsCrossWorldPremises,
    /// Mediation names no mediator.
    #[error("mediation requires at least one mediator")]
    EmptyMediators,
    /// A mediator is the treatment or the outcome.
    #[error("a mediator overlaps the treatment or the outcome")]
    MediatorOverlap,
    /// Conditioning names the treatment of a mediation target.
    #[error("conditioning includes the treatment {id}")]
    ConditioningIncludesTreatment {
        /// Treatment named in the conditioning set.
        id: VariableId,
    },
    /// Conditioning names a mediator of a mediation target.
    #[error("conditioning includes the mediator {id}")]
    ConditioningIncludesMediator {
        /// Mediator named in the conditioning set.
        id: VariableId,
    },
    /// Treatment and outcome are one variable.
    #[error("treatment and outcome are the same variable {id}")]
    TreatmentEqualsOutcome {
        /// Shared id.
        id: VariableId,
    },
    /// A temporal policy reaches outside `[-horizon, +horizon]`.
    #[error("temporal policy {policy} lies outside the target horizon of {horizon_steps} steps")]
    PolicyOutsideHorizon {
        /// Horizon of the frame the policy was declared under.
        horizon_steps: u32,
        /// Short description of the offending policy.
        policy: String,
    },
    /// The worlds of a contrast do not realise the contrast's formula.
    #[error("contrast worlds do not realise the formula: {reason}")]
    IncoherentContrast {
        /// What about the four-world or two-world structure failed.
        reason: &'static str,
    },
    /// A mediation target compares an arm with itself.
    #[error("mediation control and active arms are the same intervention")]
    MediationArmsNotDistinct,
    /// A mediation arm does not assign the treatment.
    #[error("intervention targets {got}, expected treatment {expected}")]
    InterventionVariableMismatch {
        /// Treatment the arm declared.
        expected: VariableId,
        /// Variable the intervention assigns.
        got: VariableId,
    },
    /// A history-indexed law has no snapshot, coordinates, or experiments.
    #[error("a history-indexed target law needs a snapshot, coordinates, and experiments")]
    EmptyHistoryIndexedLaw,
    /// An experiment's level vector does not match the coordinate list.
    #[error("history-indexed experiment {index} does not match the coordinate width")]
    HistoryWidth {
        /// Position in the caller's experiment list.
        index: usize,
    },
    /// Two experiments name the same history.
    #[error("a history-indexed law names each history once")]
    DuplicateHistory,
    /// A mass is non-finite or negative.
    #[error("a history-indexed mass must be finite and non-negative")]
    InvalidHistoryMass,
    /// The experiments do not form a probability law.
    #[error("history-indexed masses must sum to one")]
    UnnormalizedHistoryLaw,
    /// A coordinate is listed twice.
    #[error("coordinate {id} is repeated in the history-indexed law")]
    DuplicateHistoryCoordinate {
        /// Repeated coordinate.
        id: VariableId,
    },
}
