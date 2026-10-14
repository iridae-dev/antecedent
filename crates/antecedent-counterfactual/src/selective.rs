//! General cross-world selective execution.
//!
//! One staged pipeline for coupled counterfactual worlds, with the stages kept
//! separate so that a change of one input rebuilds exactly the stages that
//! depend on it:
//!
//! 1. **mechanism fitting** (the mechanisms a request is evaluated under; a
//!    declared mechanism is bound, not fitted, and is counted as such);
//! 2. **factual abduction** (the exogenous posterior given the factual
//!    evidence of each unit);
//! 3. **exogenous draws** (the shared exogenous states every world reads, with
//!    semantic stream coordinates that exclude the world);
//! 4. **action-specific prediction** (each named world evaluated under the
//!    same exogenous states);
//! 5. **summary / decision** (the requested quantity from the coupled worlds).
//!
//! A change of the action definitions only reuses stages 1 to 3; changed
//! factual evidence, mechanism parameters, structure, noise law or a different
//! population or history is a different abduction identity and rebuilds it.
//! Missing identification or coupling is a scientific refusal, never a cache
//! miss, and no draws of independently produced marginals are paired: the one
//! pairing decision is `antecedent_contract::check_propagation`.
//!
//! The capability matrix ([`matrix`]) states which quantity, model class,
//! conditioning, coupling, population and history combinations are executed,
//! which keep their own theorem-scoped route, and which are closed with the
//! prerequisites that would open them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod finite_executor;
pub mod finite_scm;
pub mod matrix;
pub mod refusal;
pub mod stages;
pub mod whole_path_uncertainty;

pub use crate::temporal_cross_world::staged::{TemporalExecution, TemporalExecutor};
pub use finite_executor::{
    ExecutionReceipt, ExogenousInference, FiniteExecutionRequest, FiniteExecutionResult,
    FiniteScmExecutor, NoiseCoupling, RetainedAbduction, UnitAnswer,
};
pub use finite_scm::{
    ExogenousVariable, FactualUnit, FiniteQuantity, FiniteScm, FiniteVariable, FiniteWorld,
    WorldAtom, WorldRoute,
};
pub use matrix::{
    CapabilityStatus, ConditioningKind, CouplingKind, EvidenceKind, HistoryKind, InterventionKind,
    MatrixDecision, MatrixRefusal, MatrixRequest, MatrixRow, ModelClass, PopulationKind,
    QuantityClass, decide, rows,
};
pub use refusal::{CrossWorldExecutionError, CrossWorldRefusal};
pub use stages::{ExecutionStage, StageReceipt, StageStatus, WorkCounters, stream_identity};
