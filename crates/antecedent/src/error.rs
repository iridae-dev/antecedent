//! Facade errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_attribution::AttributionError;
use antecedent_core::SchemaError;
use antecedent_counterfactual::CounterfactualError;
use antecedent_data::DataError;
use antecedent_design::DesignError;
use antecedent_discovery::DiscoveryError;
use antecedent_estimate::EstimationError;
use antecedent_graph::GraphError;
use antecedent_identify::IdentificationError;
use antecedent_io::IoError;
use antecedent_model::ModelError;
use antecedent_state::StateError;
use antecedent_validate::ValidationError;
use thiserror::Error;

/// Which review artifact is blocking execution.
///
/// Mirrors the `kind` strings accepted by [`CausalError::ReviewRequired`]. Kept as a
/// closed enum so callers get exhaustiveness checking; [`Self::as_str`] is the bridge
/// to the wire-level `String` field (kept as `String` for the Python binding today).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ReviewKind {
    /// `DirectLiNGAM` / other full-DAG discovery pending edge acceptance.
    StaticDag,
    /// Static PC / GES CPDAG pending edge acceptance or undirected-mark orientation.
    StaticCpdag,
    /// Classic static FCI / RFCI PAG pending circle-mark review.
    StaticPag,
    /// PCMCI / temporal DAG discovery pending edge acceptance.
    TemporalDag,
    /// PCMCI+ temporal CPDAG pending edge acceptance or undirected-mark orientation.
    TemporalCpdag,
    /// LPCMCI temporal PAG pending circle-mark review.
    TemporalPag,
    /// Review required without a more specific structured kind.
    Generic,
}

impl ReviewKind {
    /// Canonical wire name (matches [`CausalError::ReviewRequired`]'s `kind` field values).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::StaticDag => "static_dag",
            Self::StaticCpdag => "static_cpdag",
            Self::StaticPag => "static_pag",
            Self::TemporalDag => "temporal_dag",
            Self::TemporalCpdag => "temporal_cpdag",
            Self::TemporalPag => "temporal_pag",
            Self::Generic => "generic",
        }
    }
}

impl std::fmt::Display for ReviewKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One unreviewed edge blocking a [`CausalError::ReviewRequired`], with its endpoint
/// marks.
///
/// Endpoint identifiers are display strings derived from the graph's own dense
/// variable id (`"V3"`) or lagged temporal key (`"V3@-1"`, `"V3@0"` for
/// contemporaneous) — user-facing names live in schemas and dictionaries, not in
/// these hot graph structures (see `antecedent_core::ids`), so resolving a bound
/// schema name is deliberately out of scope here; callers that need the resolved
/// name can look it up themselves from the identifier. Mark strings are `"tail"`,
/// `"arrow"`, `"circle"`, or `"conflict"`, matching [`antecedent_graph::Endpoint`]'s
/// wire vocabulary (the same one the Python `GraphEdge` binding already uses).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PendingEdge {
    /// Source endpoint identifier.
    pub source: String,
    /// Target endpoint identifier.
    pub target: String,
    /// Mark at the source endpoint.
    pub at_source: String,
    /// Mark at the target endpoint.
    pub at_target: String,
}

impl PendingEdge {
    /// Construct a pending edge from endpoint identifiers and marks.
    #[must_use]
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        at_source: impl Into<String>,
        at_target: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            at_source: at_source.into(),
            at_target: at_target.into(),
        }
    }
}

/// Pipeline and facade failures — structured sum over domain errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CausalError {
    /// Identification failed.
    #[error(transparent)]
    Identify(#[from] IdentificationError),
    /// Estimation failed.
    #[error(transparent)]
    Estimate(#[from] EstimationError),
    /// Validation / refutation failed.
    #[error(transparent)]
    Validate(#[from] ValidationError),
    /// Discovery failed.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    /// Structural / probabilistic model failure.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// Counterfactual evaluation failed.
    #[error(transparent)]
    Counterfactual(#[from] CounterfactualError),
    /// Attribution failed.
    #[error(transparent)]
    Attribution(#[from] AttributionError),
    /// Artifact serialization / deserialization.
    #[error(transparent)]
    Serialization(#[from] IoError),
    /// Tabular / time-series data construction or lookup.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Graph construction or validation.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Experiment / measurement design evaluation.
    #[error(transparent)]
    Design(#[from] DesignError),
    /// Incremental antecedent-state update.
    #[error(transparent)]
    State(#[from] StateError),
    /// Schema construction or name lookup at an API boundary.
    #[error(transparent)]
    Schema(#[from] SchemaError),
    /// Logical / physical plan compilation failed.
    #[error("{message}")]
    Compile {
        /// Message.
        message: String,
    },
    /// Memory or other resource refusal.
    #[error("{message}")]
    Resource {
        /// Message.
        message: String,
    },
    /// Graph review incomplete (structured for facade UX).
    #[error("{message}")]
    ReviewRequired {
        /// Review kind: `static_dag`, `static_cpdag`, `static_pag`, `temporal_dag`,
        /// `temporal_cpdag`, `temporal_pag`, `generic` (see [`ReviewKind::as_str`]).
        kind: String,
        /// Discovery / supply algorithm id when known.
        algorithm: Option<String>,
        /// Count of pending / ambiguous marks blocking estimation.
        pending_edge_count: usize,
        /// The actual pending edges blocking review, with endpoint marks.
        ///
        /// Kept consistent with `pending_edge_count` (same length) wherever the
        /// underlying edges are known. Empty only when the review is genuinely
        /// edge-free — e.g. [`Self::review_required_msg`]'s generic path — never used
        /// as a placeholder for edges that exist but weren't captured; a caller-visible
        /// empty list here must mean "nothing pending", not "unknown".
        pending_edges: std::sync::Arc<[PendingEdge]>,
        /// Human-readable message.
        message: String,
        /// Next-step hint for callers.
        hint: String,
    },
    /// Query / feature unsupported.
    #[error("{message}")]
    Unsupported {
        /// Message.
        message: &'static str,
    },
    /// Missing required builder input.
    #[error("missing required field: {field}")]
    Missing {
        /// Field name.
        field: &'static str,
    },
    /// Cooperative cancellation before a usable point estimate was available.
    #[error("cancelled during {stage}")]
    Cancelled {
        /// Pipeline stage where cancellation was observed.
        stage: &'static str,
    },
    /// Two sources of truth were set for the same setting.
    #[error("conflicting configuration for {what}: {detail}")]
    Conflict {
        /// The setting that was configured twice.
        what: &'static str,
        /// How to resolve it.
        detail: &'static str,
    },
    /// A structure was used with data it does not describe.
    #[error("{detail}")]
    SchemaMismatch {
        /// What disagreed, naming the specific variable or count.
        detail: String,
    },
}

impl CausalError {
    /// Build a structured review-required error.
    ///
    /// `pending_edges` should carry the real edges blocking review whenever the
    /// caller has them in hand — see the field doc on
    /// [`Self::ReviewRequired`]'s `pending_edges` for why an empty list is reserved
    /// for genuinely edge-free reviews.
    #[must_use]
    pub fn review_required(
        kind: impl Into<String>,
        algorithm: Option<impl Into<String>>,
        pending_edge_count: usize,
        pending_edges: impl Into<std::sync::Arc<[PendingEdge]>>,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self::ReviewRequired {
            kind: kind.into(),
            algorithm: algorithm.map(Into::into),
            pending_edge_count,
            pending_edges: pending_edges.into(),
            message: message.into(),
            hint: hint.into(),
        }
    }

    /// Convenience when only a message is available (generic review).
    #[must_use]
    pub fn review_required_msg(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::review_required(
            "generic",
            None::<String>,
            0,
            Vec::<PendingEdge>::new(),
            message,
            "complete graph review (finish_*_review) or supply a fully oriented graph",
        )
    }
}
