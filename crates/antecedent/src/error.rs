//! Facade errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_attribution::AttributionError;
use antecedent_core::{QueryError, SchemaError, TemporalIndexError};
use antecedent_counterfactual::CounterfactualError;
use antecedent_data::DataError;
use antecedent_design::DesignError;
use antecedent_discovery::DiscoveryError;
use antecedent_estimate::EstimationError;
use antecedent_expr::ProgramError;
use antecedent_graph::GraphError;
use antecedent_identify::IdentificationError;
use antecedent_io::IoError;
use antecedent_learn::LearnError;
use antecedent_model::ModelError;
use antecedent_prob::ProbError;
use antecedent_state::StateError;
use antecedent_stats::StatsError;
use antecedent_validate::ValidationError;
use thiserror::Error;

/// Reason codes of the identification-class refusals: the declared structure does not identify
/// the question. [`CausalError::is_identification_refusal`] reads this list, and the opt-in
/// descriptive fallback converts exactly these refusals (never a budget, a cancellation or an
/// invalid argument).
pub const IDENTIFICATION_REFUSAL_CODES: [&str; 2] =
    ["effect_not_identified", "cross_world_not_identified"];

/// Which review artifact is blocking execution.
///
/// The `kind` carried by [`CausalError::ReviewRequired`]. A closed enum so callers get
/// exhaustiveness checking; [`Self::as_str`] is the wire-level name.
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

/// The typed domain error that made a plan or program refuse to compile.
///
/// A compile refusal that used to flatten its cause with `e.to_string()` carries
/// it here instead, so the source chain survives. Every variant renders exactly as
/// the error it wraps.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompileCause {
    /// A query failed its own validation.
    #[error(transparent)]
    Query(#[from] QueryError),
    /// A prior, posterior or inference backend refused.
    #[error(transparent)]
    Probability(#[from] ProbError),
    /// A schema lookup or construction failed.
    #[error(transparent)]
    Schema(#[from] SchemaError),
    /// A table, column or series could not be built or read.
    #[error(transparent)]
    Data(#[from] DataError),
    /// A graph failed construction or validation.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// A lagged node could not be indexed.
    #[error(transparent)]
    TemporalIndex(#[from] TemporalIndexError),
    /// An attribution input was refused.
    #[error(transparent)]
    Attribution(#[from] AttributionError),
    /// A structural model failed.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A counterfactual input was refused.
    #[error(transparent)]
    Counterfactual(#[from] CounterfactualError),
    /// A learned basis or surrogate refused.
    #[error(transparent)]
    Learn(#[from] LearnError),
    /// A discovery input was refused.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    /// A statistical primitive refused.
    #[error(transparent)]
    Stats(#[from] StatsError),
    /// A functional program failed to build.
    #[error(transparent)]
    Program(#[from] ProgramError),
    /// An estimator refused its input.
    #[error(transparent)]
    Estimation(#[from] EstimationError),
    /// An exact discrete law or its transport data failed validation.
    #[error(transparent)]
    ExactLaw(#[from] antecedent_expr::ExactLawError),
    /// An exact expression plan failed to compile or evaluate.
    #[error(transparent)]
    ExactEvaluation(#[from] antecedent_expr::EvalError),
    /// An identification routine refused its input.
    #[error(transparent)]
    Identification(#[from] IdentificationError),
    /// An artifact could not be encoded or decoded.
    #[error(transparent)]
    Serialization(#[from] IoError),
    /// A decision demanded a carrier the statistical program cannot publish.
    #[error(transparent)]
    Decision(Box<crate::analysis::DecisionCarrierError>),
}

impl CompileCause {
    /// The family of the typed error that made the compile refuse, as a stable snake_case
    /// name (`graph`, `data`, `identification`, ...).
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Query(_) => "query",
            Self::Probability(_) => "probability",
            Self::Schema(_) => "schema",
            Self::Data(_) => "data",
            Self::Graph(_) => "graph",
            Self::TemporalIndex(_) => "temporal_index",
            Self::Attribution(_) => "attribution",
            Self::Model(_) => "model",
            Self::Counterfactual(_) => "counterfactual",
            Self::Learn(_) => "learn",
            Self::Discovery(_) => "discovery",
            Self::Stats(_) => "stats",
            Self::Program(_) => "program",
            Self::Estimation(_) => "estimation",
            Self::ExactLaw(_) => "exact_law",
            Self::ExactEvaluation(_) => "exact_evaluation",
            Self::Identification(_) => "identification",
            Self::Serialization(_) => "serialization",
            Self::Decision(_) => "decision",
        }
    }
}

impl From<crate::analysis::DecisionCarrierError> for CompileCause {
    fn from(error: crate::analysis::DecisionCarrierError) -> Self {
        Self::Decision(Box::new(error))
    }
}

/// A compile refusal raised from a typed cause ([`CausalError::Rejected`]).
///
/// The message is the cause's rendering, held once so the refusal classifies
/// ([`CausalError::reason_code`], the Python exception class) exactly as the
/// message-only [`CausalError::Compile`] it replaces; [`Self::cause`] keeps the
/// typed error and is the [`std::error::Error::source`] of the failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{message}")]
pub struct CompileFailure {
    message: String,
    #[source]
    cause: CompileCause,
}

impl CompileFailure {
    /// The rendered cause: the text of the refusal.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The typed error that made the compile refuse.
    #[must_use]
    pub const fn cause(&self) -> &CompileCause {
        &self.cause
    }
}

impl From<CompileCause> for CompileFailure {
    fn from(cause: CompileCause) -> Self {
        Self { message: cause.to_string(), cause }
    }
}

/// Pipeline and facade failures — structured sum over domain errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CausalError {
    /// Identification failed.
    #[error(transparent)]
    Identify(IdentificationError),
    /// Estimation failed.
    #[error(transparent)]
    Estimate(#[from] EstimationError),
    /// Validation / refutation failed.
    #[error(transparent)]
    Validate(ValidationError),
    /// Discovery failed.
    #[error(transparent)]
    Discovery(DiscoveryError),
    /// Structural / probabilistic model failure.
    #[error(transparent)]
    Model(ModelError),
    /// Counterfactual evaluation failed.
    #[error(transparent)]
    Counterfactual(CounterfactualError),
    /// Attribution failed.
    #[error(transparent)]
    Attribution(AttributionError),
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
    ///
    /// The residual form: a refusal with no typed domain error behind it (an
    /// invariant stated in prose, or a reason-coded message). A refusal that
    /// has a typed cause is [`Self::Rejected`].
    #[error("{message}")]
    Compile {
        /// Message.
        message: String,
    },
    /// Plan or program compilation failed because a typed domain error refused
    /// its input. Rendered, classified and raised exactly as [`Self::Compile`]
    /// with the cause's message; the typed cause is the error source.
    #[error(transparent)]
    Rejected(CompileFailure),
    /// The question has no identified estimand under the declared structure.
    ///
    /// A refusal, not a failure: it carries the identification outcome so a
    /// caller learns why. `search_capped` separates a search that finished
    /// without an estimand from one that stopped at a budget (a completion or
    /// history cap), which is not a proof of non-identification. The message
    /// carries the `effect_not_identified` reason code.
    #[error("{message}")]
    NotIdentified {
        /// Identification status the search ended with.
        status: antecedent_core::IdentificationStatus,
        /// Whether the search stopped at a budget rather than completing.
        search_capped: bool,
        /// Reason-coded message.
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
        /// Which review artifact is blocking (wire name: [`ReviewKind::as_str`]).
        kind: ReviewKind,
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
    /// Support-matrix refusal (stable id).
    #[error("{id}: {message}")]
    Support {
        /// Stable matrix id (`not_applicable` or `refused`).
        id: crate::support::SupportRefusal,
        /// Human-readable reason (n/a rule text, or a refused-cell explanation).
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
    /// In-process callback failed (design utility, not portable).
    #[error("callback {name}: {message}")]
    Callback {
        /// Callback name.
        name: String,
        /// Failure detail.
        message: String,
    },
    /// A refusal that also carries structured diagnostics
    /// ([`antecedent_estimate::RefusalFields`]). Renders exactly as the wrapped error, and
    /// every classification of it ([`Self::reason_code`], [`Self::blocker_id`], the Python
    /// exception class) is that of [`Self::peeled`]; the fields are read with
    /// [`Self::refusal_fields`]. Build one with [`Self::diagnosed`].
    #[error("{error}")]
    Diagnosed {
        /// The refusal being described.
        #[source]
        error: Box<CausalError>,
        /// Structured diagnostics of the failing step; absent entries stay absent.
        fields: Box<antecedent_estimate::RefusalFields>,
    },
}

/// Stage id reported on a [`CausalError::Cancelled`] raised while refuting /
/// running predictive checks. Matches `analysis::stage::STAGE_VALIDATE`'s value.
const STAGE_VALIDATE: &str = "validate";
/// Stage id reported on a [`CausalError::Cancelled`] raised by a standalone
/// discovery call (outside the `AverageEffect` progressive-execute pipeline,
/// which has no discovery stage of its own).
const STAGE_DISCOVER: &str = "discover";
/// Stage id reported on a [`CausalError::Cancelled`] raised by a standalone
/// attribution call (Shapley allocation, root-cause ranking).
const STAGE_ATTRIBUTE: &str = "attribute";
/// Stage id reported on a [`CausalError::Cancelled`] raised by identification.
const STAGE_IDENTIFY: &str = "identify";
/// Stage id reported on a [`CausalError::Cancelled`] raised by a model fit.
const STAGE_MODEL: &str = "model";
/// Stage id reported on a [`CausalError::Cancelled`] raised by counterfactual evaluation.
const STAGE_COUNTERFACTUAL: &str = "counterfactual";

/// How a stage error classifies at the facade: cancellation and resource exhaustion are
/// non-scientific outcomes, not findings of the stage, so they are peeled off before the stage
/// error is wrapped in its own [`CausalError`] variant. [`CausalError::blocker_id`] relies on
/// this to mark them correctly.
enum Classified<E> {
    /// Cooperative cancellation.
    Cancelled,
    /// A memory, budget or other resource refusal.
    Resource(String),
    /// A finding of the stage itself.
    Finding(E),
}

/// A stage error that reaches the facade through [`lift_stage_error`].
trait StageError: Sized {
    /// Stage id a cancellation of this stage reports.
    const STAGE: &'static str;

    /// Separate cancellation and resource exhaustion from findings.
    fn classify(self) -> Classified<Self>;

    /// The facade variant that carries a finding of this stage.
    fn wrap(self) -> CausalError;
}

/// The one path every [`StageError`] takes into a [`CausalError`].
fn lift_stage_error<E: StageError>(error: E) -> CausalError {
    match error.classify() {
        Classified::Cancelled => CausalError::Cancelled { stage: E::STAGE },
        Classified::Resource(message) => CausalError::Resource { message },
        Classified::Finding(finding) => finding.wrap(),
    }
}

impl StageError for ValidationError {
    const STAGE: &'static str = STAGE_VALIDATE;

    fn classify(self) -> Classified<Self> {
        match self {
            Self::Cancelled | Self::Discovery(DiscoveryError::Cancelled) => Classified::Cancelled,
            Self::Resource { message } | Self::Discovery(DiscoveryError::Resource(message)) => {
                Classified::Resource(message)
            }
            other => Classified::Finding(other),
        }
    }

    fn wrap(self) -> CausalError {
        CausalError::Validate(self)
    }
}

impl StageError for IdentificationError {
    const STAGE: &'static str = STAGE_IDENTIFY;

    fn classify(self) -> Classified<Self> {
        match self {
            Self::Cancelled => Classified::Cancelled,
            Self::Budget { .. } => Classified::Resource(self.to_string()),
            other => Classified::Finding(other),
        }
    }

    fn wrap(self) -> CausalError {
        CausalError::Identify(self)
    }
}

impl StageError for DiscoveryError {
    const STAGE: &'static str = STAGE_DISCOVER;

    fn classify(self) -> Classified<Self> {
        match self {
            Self::Cancelled => Classified::Cancelled,
            Self::Resource(message) => Classified::Resource(message),
            other => Classified::Finding(other),
        }
    }

    fn wrap(self) -> CausalError {
        CausalError::Discovery(self)
    }
}

impl StageError for ModelError {
    const STAGE: &'static str = STAGE_MODEL;

    fn classify(self) -> Classified<Self> {
        match self {
            Self::Cancelled => Classified::Cancelled,
            Self::ResourceLimit { .. } => Classified::Resource(self.to_string()),
            other => Classified::Finding(other),
        }
    }

    fn wrap(self) -> CausalError {
        CausalError::Model(self)
    }
}

impl StageError for CounterfactualError {
    const STAGE: &'static str = STAGE_COUNTERFACTUAL;

    fn classify(self) -> Classified<Self> {
        match self {
            Self::Cancelled => Classified::Cancelled,
            other => Classified::Finding(other),
        }
    }

    fn wrap(self) -> CausalError {
        CausalError::Counterfactual(self)
    }
}

impl StageError for AttributionError {
    const STAGE: &'static str = STAGE_ATTRIBUTE;

    fn classify(self) -> Classified<Self> {
        match self {
            Self::Cancelled => Classified::Cancelled,
            Self::Budget { .. } => Classified::Resource(self.to_string()),
            other => Classified::Finding(other),
        }
    }

    fn wrap(self) -> CausalError {
        CausalError::Attribution(self)
    }
}

macro_rules! lift_from_stage_error {
    ($($error:ty),+ $(,)?) => {
        $(
            impl From<$error> for CausalError {
                fn from(error: $error) -> Self {
                    lift_stage_error(error)
                }
            }
        )+
    };
}

lift_from_stage_error!(
    ValidationError,
    IdentificationError,
    DiscoveryError,
    ModelError,
    CounterfactualError,
    AttributionError,
);

/// A typed cause refuses compilation. Not provided for [`SchemaError`],
/// [`DataError`], [`GraphError`], [`ModelError`], [`CounterfactualError`],
/// [`AttributionError`], [`DiscoveryError`], [`EstimationError`],
/// [`IdentificationError`] or [`IoError`], which already convert to their own
/// variants: wrap those with [`CausalError::rejected`].
impl From<QueryError> for CausalError {
    fn from(error: QueryError) -> Self {
        Self::rejected(error)
    }
}

impl From<ProbError> for CausalError {
    fn from(error: ProbError) -> Self {
        Self::rejected(error)
    }
}

impl From<TemporalIndexError> for CausalError {
    fn from(error: TemporalIndexError) -> Self {
        Self::rejected(error)
    }
}

impl From<LearnError> for CausalError {
    fn from(error: LearnError) -> Self {
        Self::rejected(error)
    }
}

impl From<StatsError> for CausalError {
    fn from(error: StatsError) -> Self {
        Self::rejected(error)
    }
}

impl From<ProgramError> for CausalError {
    fn from(error: ProgramError) -> Self {
        Self::rejected(error)
    }
}

impl From<antecedent_expr::ExactLawError> for CausalError {
    fn from(error: antecedent_expr::ExactLawError) -> Self {
        Self::rejected(error)
    }
}

impl From<antecedent_expr::EvalError> for CausalError {
    fn from(error: antecedent_expr::EvalError) -> Self {
        Self::rejected(error)
    }
}

/// Build a [`CausalError::Unsupported`] refusal carrying a registered
/// runtime-refusal reason code.
///
/// The code is checked at compile time against `parity/reason_codes.toml` (the
/// generated `antecedent_core::reason_code` lists), so an unregistered code
/// cannot be emitted. The message is `reason=<code>: <message>`; the Python
/// layer parses that prefix into `reason_code`.
///
/// ```
/// let err = antecedent::unsupported_reason!("attested_not_reverifiable", "names differ");
/// assert_eq!(err.to_string(), "reason=attested_not_reverifiable: names differ");
/// ```
///
/// ```compile_fail
/// let _ = antecedent::unsupported_reason!("bogus", "not registered");
/// ```
#[macro_export]
macro_rules! unsupported_reason {
    ($code:literal, $message:literal) => {{
        const _: () = assert!(
            $crate::error::is_runtime_refusal_code($code),
            concat!("`", $code, "` is not a runtime_refusal code in parity/reason_codes.toml")
        );
        $crate::CausalError::Unsupported { message: concat!("reason=", $code, ": ", $message) }
    }};
}

/// [`unsupported_reason!`] for a support-matrix refusal, which carries the
/// cell's [`crate::support::SupportRefusal`] verdict alongside the reason.
///
/// ```
/// let err = antecedent::support_reason!(
///     "data_modality_not_licensed",
///     "that cell is not licensed for this data modality"
/// );
/// assert!(err.to_string().contains("reason=data_modality_not_licensed: "));
/// ```
#[macro_export]
macro_rules! support_reason {
    ($code:literal, $message:literal) => {{
        const _: () = assert!(
            $crate::error::is_runtime_refusal_code($code),
            concat!("`", $code, "` is not a runtime_refusal code in parity/reason_codes.toml")
        );
        $crate::CausalError::Support {
            id: $crate::support::SupportRefusal::Refused,
            message: concat!("reason=", $code, ": ", $message),
        }
    }};
}

/// Compile-time check used by [`unsupported_reason!`] and [`support_reason!`].
// doc(hidden): expanded by the exported `unsupported_reason!` and `support_reason!` macros.
#[doc(hidden)]
#[must_use]
pub const fn is_runtime_refusal_code(code: &str) -> bool {
    antecedent_core::reason_code::is_runtime_refusal(code)
}

/// Shared with [`crate::PreparedStudy::rank_designs`] and capability reports.
pub(crate) const RANK_DESIGNS_REQUIRES_LICENSE: &str =
    "design ranking requires a licensed prepared contract";
pub(crate) const RANK_DESIGNS_REQUIRES_PRODUCT: &str =
    "design ranking requires cached identification products";
pub(crate) const RANK_DESIGNS_INCOMPARABLE_TARGETS: &str =
    "width ranking refuses incomparable estimands; declare a common decision utility";
pub(crate) const RANK_DESIGNS_RENORMALIZED_MASS: &str =
    "design ranking must not renormalize unidentified mass onto favorable atoms";
pub(crate) const RETARGET_REQUIRES_LICENSE: &str = "retarget requires a licensed prepared contract";
pub(crate) const EXPORT_REQUIRES_LICENSE: &str = "export requires a licensed prepared contract";
pub(crate) const OPERATION_REQUIRES_LICENSE: &str =
    "operation requires a licensed prepared contract";

/// Refusals that block an operation for want of a license. Matched by
/// constant identity: a reworded message never changes a blocker id.
pub(crate) const OPERATION_UNLICENSED: [&str; 4] = [
    RANK_DESIGNS_REQUIRES_LICENSE,
    RETARGET_REQUIRES_LICENSE,
    EXPORT_REQUIRES_LICENSE,
    OPERATION_REQUIRES_LICENSE,
];

/// [`CausalError::Compile`] whose message carries a registered runtime-refusal
/// reason code, for refusals whose detail is formatted at run time.
///
/// The code is checked at compile time like [`unsupported_reason!`].
#[macro_export]
macro_rules! compile_reason {
    ($code:literal, $($detail:tt)+) => {{
        const _: () = assert!(
            $crate::error::is_runtime_refusal_code($code),
            concat!("`", $code, "` is not a runtime_refusal code in parity/reason_codes.toml")
        );
        $crate::CausalError::Compile {
            message: format!(
                "{}{}: {}",
                $crate::error::REASON_PREFIX,
                $code,
                format!($($detail)+)
            ),
        }
    }};
}

/// Wire prefix of a reason-coded message (see `antecedent_core::reason_code::PREFIX`).
pub(crate) const REASON_PREFIX: &str = antecedent_core::reason_code::PREFIX;

impl CausalError {
    /// Refuse compilation because `cause` refused its input, keeping the typed
    /// error as the source. Prefer `?` where [`CausalError`] converts from the
    /// cause directly; this is the explicit form for causes that already have
    /// a variant of their own (`.map_err(CausalError::rejected)`).
    #[must_use]
    pub fn rejected(cause: impl Into<CompileCause>) -> Self {
        Self::Rejected(CompileFailure::from(cause.into()))
    }

    /// The message of a plan or program compile refusal, whether it carries a
    /// typed cause ([`Self::Rejected`]) or only prose ([`Self::Compile`]).
    #[must_use]
    pub fn compile_message(&self) -> Option<&str> {
        match self.peeled() {
            Self::Compile { message } => Some(message),
            Self::Rejected(failure) => Some(failure.message()),
            _ => None,
        }
    }

    /// This refusal with structured diagnostics attached. The message and every
    /// classification are unchanged; match the refusal itself through [`Self::peeled`].
    #[must_use]
    pub fn diagnosed(self, fields: antecedent_estimate::RefusalFields) -> Self {
        Self::Diagnosed { error: Box::new(self.into_peeled()), fields: Box::new(fields) }
    }

    /// The refusal under any [`Self::Diagnosed`] wrapper (itself when it is not one).
    #[must_use]
    pub fn peeled(&self) -> &Self {
        let mut error = self;
        while let Self::Diagnosed { error: inner, .. } = error {
            error = inner.as_ref();
        }
        error
    }

    /// Owned [`Self::peeled`].
    #[must_use]
    pub fn into_peeled(self) -> Self {
        match self {
            Self::Diagnosed { error, .. } => (*error).into_peeled(),
            other => other,
        }
    }

    /// Refuse a question whose identification produced no estimand.
    ///
    /// `detail` names the route; the status and whether the search was
    /// complete or capped come from the identification outcome.
    #[must_use]
    pub fn not_identified(
        status: antecedent_core::IdentificationStatus,
        search_capped: bool,
        detail: &str,
    ) -> Self {
        let search = if search_capped {
            "the search stopped at a budget, so this is not a proof of non-identification"
        } else {
            "the search completed"
        };
        // A completed search can be described; a capped one is not converted, so it offers nothing.
        let opt_in = if search_capped {
            ""
        } else {
            ". To keep this refusal and also get a noncausal description of the observed data \
             and the evidence that would resolve it, use Study::analyze_with_fallback \
             (Python: analyze(..., on_refusal=\"describe\")); the description is not an answer \
             to the causal question"
        };
        Self::NotIdentified {
            status,
            search_capped,
            message: format!(
                "{REASON_PREFIX}effect_not_identified: {detail} (identification status {}; {search}){opt_in}",
                status.as_str()
            ),
        }
    }

    /// The reason code a refusal carries, when its message is reason-coded.
    #[must_use]
    pub fn reason_code(&self) -> Option<&str> {
        let message = match self.peeled() {
            Self::NotIdentified { .. } => return Some("effect_not_identified"),
            Self::Compile { message } => message.as_str(),
            Self::Rejected(failure) => failure.message(),
            Self::Unsupported { message } | Self::Support { message, .. } => message,
            Self::Estimate(
                EstimationError::Refused { code, .. }
                | EstimationError::RefusedWithFields { code, .. },
            )
            | Self::Serialization(IoError::Refused { code, .. })
            | Self::Discovery(DiscoveryError::Refused { code, .. }) => return Some(code),
            // The multi-source limited-experiment route's frozen (code, detail) pairs.
            Self::Identify(error) => {
                return antecedent_identify::mz_transport_refusal(error)
                    .or_else(|| antecedent_identify::counterfactual_transport_refusal(error))
                    .map(|(code, _)| code);
            }
            Self::Serialization(error @ IoError::MzTransport(_)) => return error.reason_code(),
            _ => return None,
        };
        antecedent_core::reason_code::split_prefix(message).map(|(code, _)| code)
    }

    /// Whether this is an identification-class refusal: the question has no identified
    /// estimand under the declared structure (`effect_not_identified`, or a cross-world query
    /// the declared structure does not identify), and the search finished.
    ///
    /// A search that stopped at a budget, a cancellation, a resource limit, an invalid
    /// argument and every data or estimation error are not identification-class: the opt-in
    /// descriptive fallback never converts them.
    #[must_use]
    pub fn is_identification_refusal(&self) -> bool {
        match self.peeled() {
            Self::NotIdentified { search_capped, .. } => !*search_capped,
            other => {
                other.reason_code().is_some_and(|code| IDENTIFICATION_REFUSAL_CODES.contains(&code))
            }
        }
    }

    /// Refusal for an inference mode the route does not perform.
    ///
    /// The record is unsupported inference, not a failed check or a computational failure.
    /// The refusal reports no mass, so the mass is unreported. The message is `summary`.
    #[must_use = "the refusal is dropped if the error is discarded"]
    pub fn unsupported_inference(summary: impl Into<String>) -> Self {
        Self::Estimate(EstimationError::Prob(ProbError::Route {
            summary: summary.into(),
            diagnostics: vec![
                antecedent_core::diagnostic::RouteNativeDiagnostic::unsupported_inference(
                    antecedent_core::diagnostic::RouteMass::unreported(),
                ),
            ],
        }))
    }

    /// Route records carried by this error.
    ///
    /// A failed check is included. Model restriction, computational failure, and
    /// unsupported inference stay the records they are. Empty when this error
    /// is not a route outcome.
    #[must_use = "a failed route check is dropped if the records are discarded"]
    pub fn route_diagnostics(&self) -> Vec<antecedent_core::diagnostic::RouteNativeDiagnostic> {
        match self.peeled() {
            Self::Estimate(EstimationError::Prob(ProbError::Route { diagnostics, .. })) => {
                diagnostics.clone()
            }
            Self::Estimate(EstimationError::Stats(error)) => {
                error.route_diagnostic().into_iter().collect()
            }
            Self::Data(error) | Self::Estimate(EstimationError::Data(error)) => {
                error.route_diagnostic().into_iter().collect()
            }
            _ => Vec::new(),
        }
    }

    /// Structured diagnostics (failing treatment or cell, stage, arm ESS, rank, implicated
    /// columns, remedy) of a refusal built with them. Additive: `None` for every other
    /// error, and absent entries inside stay absent rather than being derived.
    #[must_use]
    pub fn refusal_fields(
        &self,
    ) -> Option<std::borrow::Cow<'_, antecedent_estimate::RefusalFields>> {
        match self {
            Self::Diagnosed { fields, .. } => Some(std::borrow::Cow::Borrowed(fields)),
            Self::Estimate(error) => error.refusal_fields(),
            _ => None,
        }
    }

    /// What the caller can change to get past this refusal, when the refusal
    /// names one (an estimator refusal built with a remedy, or one whose structured
    /// diagnostics record it). Additive and
    /// optional: `None` for every other error, and never part of the message.
    #[must_use]
    pub fn remedy(&self) -> Option<&str> {
        match self {
            Self::Diagnosed { error, fields } => {
                fields.remedy.as_deref().or_else(|| error.remedy())
            }
            Self::Estimate(error) => error.remedy().or_else(|| error.field_remedy()),
            _ => None,
        }
    }

    /// Build a structured review-required error.
    ///
    /// `pending_edges` should carry the real edges blocking review whenever the
    /// caller has them in hand — see the field doc on
    /// [`Self::ReviewRequired`]'s `pending_edges` for why an empty list is reserved
    /// for genuinely edge-free reviews.
    #[must_use]
    pub fn review_required(
        kind: ReviewKind,
        algorithm: Option<impl Into<String>>,
        pending_edge_count: usize,
        pending_edges: impl Into<std::sync::Arc<[PendingEdge]>>,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self::ReviewRequired {
            kind,
            algorithm: algorithm.map(Into::into),
            pending_edge_count,
            pending_edges: pending_edges.into(),
            message: message.into(),
            hint: hint.into(),
        }
    }

    /// Stable capability-report blocker id, when this error is a preflight/execution
    /// refusal. Budget and cancel are marked non-scientific.
    #[must_use]
    pub fn blocker_id(&self) -> Option<antecedent_contract::BlockedOperation> {
        use antecedent_contract::BlockedOperation;
        match self.peeled() {
            Self::Support { id: crate::support::SupportRefusal::NotApplicable, message } => {
                Some(BlockedOperation::not_applicable(*message))
            }
            Self::Support { id: crate::support::SupportRefusal::Refused, message } => {
                Some(BlockedOperation::refused(*message))
            }
            Self::Unsupported { message } if OPERATION_UNLICENSED.contains(message) => {
                Some(BlockedOperation::operation_unlicensed(*message))
            }
            Self::Unsupported { message } if *message == RANK_DESIGNS_REQUIRES_PRODUCT => {
                Some(BlockedOperation::binding("identification_product"))
            }
            Self::ReviewRequired { message, .. } => {
                Some(BlockedOperation::new("review.required", message.clone(), true))
            }
            Self::Cancelled { stage } => Some(BlockedOperation::cancelled(*stage)),
            Self::Resource { message } => Some(BlockedOperation::resource(message.clone())),
            Self::Missing { field } => Some(BlockedOperation::binding(field)),
            _ => None,
        }
    }

    /// Convenience when only a message is available (generic review).
    #[must_use]
    pub fn review_required_msg(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::review_required(
            ReviewKind::Generic,
            None::<String>,
            0,
            Vec::<PendingEdge>::new(),
            message,
            "complete graph review (finish_*_review) or supply a fully oriented graph",
        )
    }
}

const _: () = assert!(
    is_runtime_refusal_code("effect_not_identified"),
    "`effect_not_identified` is not a runtime_refusal code in parity/reason_codes.toml"
);

#[cfg(test)]
mod tests {
    use super::{AttributionError, CausalError, DiscoveryError, ValidationError};

    // A cancellation token fired mid-refuter, mid-discovery, or mid-Shapley
    // computation must surface at the facade boundary as `CausalError::Cancelled`,
    // not as the stage's scientific error kind — a caller filtering on
    // `except CausalCancelledError` (or `blocker_id().scientific`) would otherwise
    // never see it, and a `CausalValidateError`/`CausalDiscoveryError` would wrongly
    // read as a refutation/discovery finding.

    /// The facade reads an estimator refusal's remedy through unchanged, and
    /// every other error names none.
    #[test]
    fn remedy_reads_through_from_the_estimator_refusal() {
        use antecedent_estimate::EstimationError;
        let remedied = CausalError::from(EstimationError::unsupported_with_remedy("no", "do this"));
        assert_eq!(remedied.remedy(), Some("do this"));
        assert_eq!(remedied.reason_code(), None);
        assert_eq!(remedied.to_string(), "no");
        assert_eq!(CausalError::from(EstimationError::unsupported("no")).remedy(), None);
        assert_eq!(CausalError::Unsupported { message: "no" }.remedy(), None);
    }

    /// A typed compile cause renders and classifies as the message it replaces,
    /// and keeps the typed error reachable as the source.
    #[test]
    fn a_typed_compile_cause_keeps_the_message_the_classification_and_the_source() {
        use antecedent_core::{QueryError, VariableId};
        use std::error::Error as _;
        let cause = QueryError::TreatmentEqualsOutcome { id: VariableId::from_raw(3) };
        let text = cause.to_string();
        let error = CausalError::from(cause.clone());
        assert_eq!(error.to_string(), text);
        assert_eq!(error.compile_message(), Some(text.as_str()));
        assert_eq!(error.reason_code(), None);
        assert!(error.blocker_id().is_none());
        assert!(matches!(
            &error,
            CausalError::Rejected(failure)
                if matches!(failure.cause(), crate::CompileCause::Query(q) if *q == cause)
        ));
        assert!(error.source().is_some_and(|source| source.to_string() == text));
        assert_eq!(error, CausalError::rejected(cause));
    }

    /// The residual message-only refusal still classifies by the code its
    /// message names, and reads through `compile_message` like a typed one.
    #[test]
    fn a_message_only_compile_refusal_is_classified_by_its_code() {
        let coded = CausalError::Compile {
            message: format!("{}effect_not_identified: x", super::REASON_PREFIX),
        };
        assert_eq!(coded.reason_code(), Some("effect_not_identified"));
        assert_eq!(coded.compile_message(), Some(coded.to_string().as_str()));
    }

    #[test]
    fn model_and_counterfactual_cancellation_are_top_level() {
        use antecedent_counterfactual::CounterfactualError;
        use antecedent_model::ModelError;
        let model = CausalError::from(ModelError::Cancelled);
        assert!(matches!(model, CausalError::Cancelled { stage: "model" }), "{model:?}");
        assert!(model.blocker_id().is_some_and(|b| !b.scientific));
        let cf = CausalError::from(CounterfactualError::Cancelled);
        assert!(matches!(cf, CausalError::Cancelled { stage: "counterfactual" }), "{cf:?}");
        assert!(cf.blocker_id().is_some_and(|b| !b.scientific));
    }

    #[test]
    fn validate_cancelled_is_top_level_cancelled_not_validate() {
        let error = CausalError::from(ValidationError::Cancelled);
        assert!(
            matches!(error, CausalError::Cancelled { stage: "validate" }),
            "expected top-level Cancelled, got {error:?}"
        );
        assert!(error.blocker_id().is_some_and(|b| !b.scientific), "must be non-scientific");
    }

    #[test]
    fn validate_wrapped_discovery_cancelled_is_top_level_cancelled() {
        // A stability refuter runs discovery internally; its cancellation must not
        // be reported as the outer refutation's own scientific failure either.
        let error = CausalError::from(ValidationError::Discovery(DiscoveryError::Cancelled));
        assert!(
            matches!(error, CausalError::Cancelled { stage: "validate" }),
            "expected top-level Cancelled, got {error:?}"
        );
        assert!(error.blocker_id().is_some_and(|b| !b.scientific), "must be non-scientific");
    }

    #[test]
    fn validate_wrapped_discovery_resource_is_top_level_resource() {
        let error =
            CausalError::from(ValidationError::Discovery(DiscoveryError::Resource("cap".into())));
        assert!(
            matches!(error, CausalError::Resource { ref message } if message == "cap"),
            "expected top-level Resource, got {error:?}"
        );
        assert!(error.blocker_id().is_some_and(|b| !b.scientific), "must be non-scientific");
    }

    #[test]
    fn discovery_cancelled_is_top_level_cancelled_not_discovery() {
        let error = CausalError::from(DiscoveryError::Cancelled);
        assert!(
            matches!(error, CausalError::Cancelled { stage: "discover" }),
            "expected top-level Cancelled, got {error:?}"
        );
        assert!(error.blocker_id().is_some_and(|b| !b.scientific), "must be non-scientific");
    }

    #[test]
    fn discovery_resource_is_top_level_resource_not_discovery() {
        let error = CausalError::from(DiscoveryError::Resource("budget exceeded".into()));
        assert!(
            matches!(error, CausalError::Resource { ref message } if message == "budget exceeded"),
            "expected top-level Resource, got {error:?}"
        );
        assert!(error.blocker_id().is_some_and(|b| !b.scientific), "must be non-scientific");
    }

    #[test]
    fn attribution_cancelled_is_top_level_cancelled_not_attribution() {
        let error = CausalError::from(AttributionError::Cancelled);
        assert!(
            matches!(error, CausalError::Cancelled { stage: "attribute" }),
            "expected top-level Cancelled, got {error:?}"
        );
        assert!(error.blocker_id().is_some_and(|b| !b.scientific), "must be non-scientific");
    }

    #[test]
    fn attribution_budget_is_a_top_level_resource_refusal() {
        let error = CausalError::from(AttributionError::Budget {
            message: "path enumeration truncated at max_paths=4 max_len=3".into(),
        });
        assert!(
            matches!(&error, CausalError::Resource { message }
                if message.contains("path enumeration truncated")),
            "expected top-level Resource, got {error:?}"
        );
        assert!(error.blocker_id().is_some_and(|b| !b.scientific), "must be non-scientific");
    }

    #[test]
    fn identification_cancel_and_budget_are_non_scientific_blockers() {
        use antecedent_identify::{IdentificationBudget, IdentificationError};
        let cancelled = CausalError::from(IdentificationError::Cancelled);
        assert!(matches!(cancelled, CausalError::Cancelled { stage: "identify" }), "{cancelled:?}");
        assert!(cancelled.blocker_id().is_some_and(|b| !b.scientific));
        let budget = CausalError::from(IdentificationError::budget(IdentificationBudget::Steps));
        assert!(
            matches!(&budget, CausalError::Resource { message } if message == "transport.identification_budget"),
            "{budget:?}"
        );
        assert!(budget.blocker_id().is_some_and(|b| !b.scientific));
        assert!(matches!(
            CausalError::from(IdentificationError::invalid_input("z_transport.invalid_query")),
            CausalError::Identify(IdentificationError::InvalidInput { .. })
        ));
    }

    #[test]
    fn non_cancellation_stage_errors_still_wrap_as_the_scientific_kind() {
        assert!(matches!(
            CausalError::from(ValidationError::NotApplicable { message: "n/a" }),
            CausalError::Validate(ValidationError::NotApplicable { .. })
        ));
        assert!(matches!(
            CausalError::from(DiscoveryError::unsupported("nope")),
            CausalError::Discovery(DiscoveryError::Unsupported { .. })
        ));
        assert!(matches!(
            CausalError::from(AttributionError::UnknownPlayer),
            CausalError::Attribution(AttributionError::UnknownPlayer)
        ));
    }
}
