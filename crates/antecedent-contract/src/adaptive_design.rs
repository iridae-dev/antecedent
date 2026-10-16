//! The analysis contract of an adaptive experiment and the matrix of
//! procedures the engine licenses, closes or carries forward.
//!
//! * [`matrix`]: assignment policy x outcome structure x monitoring strategy
//!   x stopping rule, every cell licensed (per inferential paradigm), closed
//!   with a typed refusal, or carried forward with an owner.
//! * [`declaration`]: the retained analysis contract (assignment-policy
//!   history, realized assignment probabilities, exposure history, outcome
//!   availability, interim analyses, stopping rule), its completeness and
//!   consistency checks, its canonical identity and the paradigm gate.
//! * [`accounting`]: the observations re-used by the interim analyses, held in
//!   the evidence ledger so each observation is counted once.
//!
//! The contract does no statistics. The numerical routes are in
//! `antecedent_stats::adaptive_inference` and `antecedent_estimate`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod accounting;
mod declaration;
mod matrix;

pub use accounting::{INTERIM_REUSE_STAGE, InterimEvidence};
pub use declaration::{
    ADAPTIVE_DESIGN_VERSION, AdaptiveContractError, AdaptiveDesignContract,
    AdaptiveDesignDeclaration, BoundarySidedness, DelayMechanism, InterimAnalysis,
    MAX_CONTRACT_ARMS, MAX_CONTRACT_LOOKS, MAX_CONTRACT_UNITS, MAX_UNIT_MEASUREMENTS,
    OutcomeAvailability, PolicyEpoch, RouteAuthorization, SpendingKind, StoppingRuleDeclaration,
    StoppingSpec,
};
pub use matrix::{
    AssignmentFamily, CellStatus, InferenceParadigm, MonitoringStrategy, OutcomeStructure,
    ProcedureCell, StoppingRuleKind, licensed_routes, procedure_cells, procedure_status,
};
