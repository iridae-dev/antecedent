//! The analysis contract of an adaptive experiment and the matrix of
//! procedures the engine supports, closes or carries forward.
//!
//! * [`matrix`]: assignment policy x outcome structure x monitoring strategy
//!   x stopping rule, every cell supported (per inferential paradigm), closed
//!   with a typed refusal, or carried forward with an owner.
//! * [`declaration`]: the retained analysis contract (assignment-policy
//!   history, realized assignment probabilities, exposure history, outcome
//!   availability, interim analyses, stopping rule), its completeness and
//!   consistency checks, its canonical identity and the paradigm gate.
//! * `choice`: the contract of a choice study (respondents, retained sources, looks, rule), whose
//!   evidence the ledger accounts for row by row.
//! * [`accounting`]: the observations re-used by the interim analyses, held in
//!   the evidence ledger so each observation is counted once.
//!
//! The contract does no statistics. The numerical routes are in
//! `antecedent_stats::adaptive_inference` and `antecedent_estimate`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod accounting;
mod choice;
mod declaration;
mod matrix;

pub use accounting::{INTERIM_REUSE_STAGE, InterimEvidence};
pub use choice::{
    CHOICE_DESIGN_VERSION, ChoiceContractError, ChoiceDesignContract, ChoiceDesignDeclaration,
    ChoiceEvidence, ChoiceLook, ChoiceRespondent, MAX_CHOICE_CONTRACT_OPTIONS,
    MAX_CHOICE_RESPONDENTS, MAX_CHOICE_SOURCES, MAX_CHOICES_PER_RESPONDENT, RetainedChoiceProof,
    RetainedChoiceSource,
};
pub use declaration::{
    ADAPTIVE_DESIGN_VERSION, AdaptiveContractError, AdaptiveDesignContract,
    AdaptiveDesignDeclaration, BoundarySidedness, DelayMechanism, InterimAnalysis,
    MAX_CONTRACT_UNITS, OutcomeAvailability, PolicyEpoch, RouteAuthorization, SpendingKind,
    StoppingRuleDeclaration, StoppingSpec,
};
pub use matrix::{
    AssignmentFamily, CellStatus, InferenceParadigm, MonitoringStrategy, OutcomeStructure,
    ProcedureCell, StoppingRuleKind, procedure_cells, procedure_status, supported_routes,
};
