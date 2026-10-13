//! The runtime evidence ledger: one representation of which observations,
//! experiments, priors, fitted models and external claims contributed to a
//! result, how they overlap, and what changed since.
//!
//! * Sources carry row-level observation identities and a derivation
//!   lineage (duplicate, transform, subset, prior plus likelihood, derived
//!   from data already used), so overlap is computable rather than asserted.
//! * Overlap is independent, shared (with mass accounting) or unknown.
//!   Unknown stays unknown and is never promoted to independent.
//! * Operations (add, update, replace, withdraw, supersede,
//!   revalidate-downstream and combine) each have their own identity chained
//!   to the previous one, and their own downstream effect.
//! * Combinations are statistical pooling, Bayesian borrowing, causal
//!   transport, evidence reuse and repeated observation use. They are
//!   distinct operations with distinct overlap policies; silent double
//!   counting is refused with a typed reason.
//! * A source that carries a transport identity is never native target
//!   data, and the same source cannot be both.
//!
//! The ledger does no statistics. Numerical pooling lives in
//! `antecedent_stats::synthesis`; the ledger decides whether a pooling or
//! borrowing request may be made at all and records what it counted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod ids;
mod ledger;
mod operation;
mod overlap;
mod refusal;
mod source;

pub use ids::{
    LEDGER_IDENTITY_VERSION, ObservationId, OperationId, ResultId, SourceId, parse_digest_hex,
};
pub use ledger::{
    EvidenceLedger, MAX_COMBINATION_SOURCES, MAX_OBSERVATIONS_PER_SOURCE, MAX_OPERATIONS,
    MAX_RESULTS, MAX_SOURCES, MAX_TEXT_BYTES, MAX_TOTAL_OBSERVATIONS, MAX_TRANSPORT_ITEMS,
};
pub use operation::{
    AppliedOperation, ClaimClass, CombinationRequest, DetectedOverlap, LedgerOperation,
    OperationEffect, PoolingDependence, ResultRecord, ResultRevalidation, ResultStatus,
    Revalidation, SharedObservations, SynthesisKind, SynthesisOperation,
};
pub use overlap::{MassAccount, Overlap, ReentryKind, RelationKind};
pub use refusal::{LedgerError, RefusalKind};
pub use source::{
    EffectiveObservations, EvidenceSource, Lineage, ObservationBasis, SourceDeclaration,
    SourceKind, SourceStatus, TransportIdentity,
};
