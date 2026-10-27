//! Selective discovery: reuse of conditional-independence statistics across discovery
//! requests, with separate static and temporal frozen contracts, distinct stage identities,
//! observed work units and a portable replayable state.
//!
//! # Claims this module makes
//!
//! * **Algorithmic match.** A selective run returns, bit for bit, the result a full run of
//!   the same contract on the same data returns. A statistic is retained only at an
//!   identical input coordinate, and the retained value is the value the test returned for
//!   it. The result identity ([`outcome::DiscoveryOutcome::result_identity`]) makes the
//!   comparison executable.
//! * **Work.** The counters in [`work::WorkUnits`] are observed where tests reach the test
//!   implementation, where orientation rounds run and where search passes are recorded, and
//!   a retained statistic is not work.
//!
//! # Claims it does not make
//!
//! * **No recovery guarantee.** Matching a full run says nothing about whether the output is
//!   the true graph. No recovery, faithfulness-adequacy, false-discovery or error-control
//!   statement is made for the output graph, and none is licensed: the route that would
//!   state one is closed with a typed `cell_not_licensed` refusal until the complete
//!   discovery-plus-analysis method, including reuse of the discovery data in the later
//!   analysis and any sample splitting, has been calibrated (see
//!   `docs/selective-discovery.md` for the recorded calibration design).
//! * **No interval downstream.** A downstream estimate that rests on a discovered graph
//!   states no interval on account of discovery.
//! * **No incremental discovery.** New data invalidates the statistics; a previous graph is
//!   not a starting point.
//!
//! # Closed rows
//!
//! * score-based search (no score-cache injection seam; `score_evaluations` is zero),
//! * resumable skeleton state between runs,
//! * multi-population discovery (pooled or per-population search),
//! * transport of a discovered structure (refusal; see [`transport`]),
//! * derived sufficient-statistic updates (a separate row that needs its own validation).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod adapter;
pub mod claims;
pub mod contract;
pub mod identity;
pub mod outcome;
pub mod populations;
pub mod portable;
pub mod refusal;
pub mod reuse;
pub mod stages;
pub mod store;
pub mod structure;
pub mod transport;
pub mod work;

pub use adapter::{SelectiveRunReport, StaticSelectiveDiscovery, TemporalSelectiveDiscovery};
pub use claims::{request_downstream_interval, request_recovery_claim, request_score_search_reuse};
pub use contract::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, TemporalAlgorithm,
    TemporalDiscoveryContract, background_knowledge_digest, background_knowledge_identity,
};
pub use identity::{
    DataIdentity, Digest, column_digest, digest_from_hex, digest_hex, static_data_identity,
    temporal_data_identity,
};
pub use outcome::{DiscoveryOutcome, NativeResult};
pub use portable::PortableDiscoveryState;
pub use refusal::SelectiveDiscoveryError;
pub use stages::{
    DiscoveryPlan, DiscoveryStage, ORIENTATION_COMPONENTS, PlannedStage, SEARCH_COMPONENTS,
    STATISTIC_COMPONENTS, StageDecision, StageInputs,
};
pub use store::{
    DEFAULT_STORE_CAPACITY, MAX_PORTABLE_RECEIPTS, SelectiveCi, SelectiveLimits, StatisticReceipt,
    StatisticStore, StoreCounters, StoreMode,
};
pub use structure::{
    DiscoveredStructure, Mark, StructureClass, StructureEdge, StructureNode, UnresolvedEdge,
};
pub use transport::{DiscoveredStructureRefusal, request_selection_diagram};
pub use work::{OrientationWork, WorkUnits, observe_orientation};
