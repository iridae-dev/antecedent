//! Estimation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::QueryError;
use antecedent_data::DataError;
use antecedent_expr::provider::ProviderSnapshotError;
use antecedent_expr::{EvalError, ExactLawError, ProgramError};
use antecedent_model::history_operators::HistoryOperatorError;
use antecedent_prob::ProbError;
use antecedent_stats::{GlmRefusalKind, StatsError};
use std::borrow::Cow;

use thiserror::Error;

/// Estimation failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum EstimationError {
    /// Data/schema issue.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Stats backend.
    #[error(transparent)]
    Stats(#[from] StatsError),
    /// Probability / posterior backend.
    #[error(transparent)]
    Prob(#[from] ProbError),
    /// Query validation failed (`AverageEffectQuery::validate`, …).
    #[error(transparent)]
    Query(#[from] QueryError),
    /// A graph operation (unfolding a temporal graph, reading a node) failed.
    #[error(transparent)]
    Graph(#[from] antecedent_graph::GraphError),
    /// A dense node id does not index the temporal schema.
    #[error(transparent)]
    TemporalIndex(#[from] antecedent_core::TemporalIndexError),
    /// An intervention schedule could not be resolved.
    #[error(transparent)]
    Intervention(#[from] antecedent_core::InterventionError),
    /// A score table's plain data is not a valid table.
    #[error(transparent)]
    ScoreWire(#[from] antecedent_io_core::score_wire::ScoreTableWireError),
    /// An exact discrete law refused a construction or use (see [`ExactLawError`]).
    #[error("{0}")]
    ExactLaw(ExactLawError),
    /// A causal expression could not be compiled or evaluated.
    #[error("{0}")]
    Evaluation(EvalError),
    /// A distribution provider could not snapshot the factor laws a program needs.
    #[error("{0}")]
    Provider(ProviderSnapshotError),
    /// A functional program failed its checked construction.
    #[error("{0}")]
    Program(ProgramError),
    /// A tiled prediction request was refused before any value was streamed.
    #[error(transparent)]
    Tile(#[from] crate::tiled_prediction::TileError),
    /// A causal expression leaf or program could not be built from the declared roles.
    #[error(transparent)]
    Expr(#[from] antecedent_expr::ExprError),
    /// An execution-admission lease was refused.
    #[error(transparent)]
    Admission(#[from] antecedent_core::execution::AdmissionError),
    /// The adaptive resampling wave procedure refused (an unusable rule, too many failed
    /// replicates, or a cancellation that is refused rather than reported).
    #[error(transparent)]
    Wave(#[from] crate::adaptive_resample_waves::WaveError),
    /// The out-of-core weighted-Gaussian reduction refused (see [`CapacityFault`]).
    #[error(transparent)]
    Capacity(#[from] CapacityFault),
    /// A nuisance learner failed (fit, prediction or shape); a cancelled fit is the coded
    /// `cancelled_no_claim` refusal instead.
    #[error(transparent)]
    Learner(#[from] antecedent_learn::LearnError),
    /// Identification could not state the quantity the estimator was asked for.
    #[error(transparent)]
    Identification(#[from] antecedent_identify::IdentificationError),
    /// A history operator (cadence, horizon, action time or weights) refused its inputs.
    #[error(transparent)]
    HistoryOperator(#[from] HistoryOperatorFault),
    /// A resampling loop could not size, admit or allocate its index, gather or draw buffers.
    #[error(transparent)]
    ResampleBuffer(#[from] ResampleBufferFault),
    /// Missing overlap override when required.
    #[error("{message}")]
    Overlap {
        /// Message.
        message: &'static str,
    },
    /// [`Self::Overlap`] raised on a fitted propensity that also carries structured
    /// diagnostics (see [`RefusalFields`]); the rendered message is that of the
    /// [`Self::Overlap`] it replaces.
    #[error("{message}")]
    OverlapWithFields {
        /// Message.
        message: &'static str,
        /// Structured diagnostics of the failing step; absent entries stay absent.
        fields: Box<RefusalFields>,
    },
    /// Incompatible estimand.
    #[error("{message}")]
    IncompatibleEstimand {
        /// Message.
        message: &'static str,
    },
    /// Effect modifiers not supported on this estimator path.
    #[error("effect modifiers are not supported on this estimator path")]
    EffectModifiers,
    /// A quantile posterior's threshold grid does not bracket the requested quantile in enough
    /// posterior draws. Callers that turn this into a next attempt match this variant, never
    /// its text.
    #[error(
        "quantile posterior refused: the threshold grid does not bracket the quantile in enough \
         posterior draws; widen the grid or move the level inside its range"
    )]
    QuantileGridUnresolved,
    /// Target population not supported on this estimator path: a refusal with
    /// the registered `population_not_estimable` reason code.
    #[error(
        "{}{}: only TargetPopulation::AllObserved is supported on this estimator path",
        antecedent_core::reason_code::PREFIX,
        antecedent_core::reason_code!("population_not_estimable")
    )]
    TargetPopulation,
    /// Query options unsupported by this estimator (fixed message).
    #[error("{message}")]
    Unsupported {
        /// Explanation.
        message: &'static str,
    },
    /// [`Self::Unsupported`] that also names what the caller can change to
    /// proceed. The rendered message is the refusal alone, byte-identical to the
    /// [`Self::Unsupported`] it replaces; the remedy is a separate structured
    /// field read with [`Self::remedy`], never parsed out of the message. It
    /// carries no reason code, as [`Self::Unsupported`] does not.
    #[error("{message}")]
    UnsupportedWithRemedy {
        /// Explanation (the refusal).
        message: &'static str,
        /// What the caller can do instead.
        remedy: &'static str,
    },
    /// [`Self::Unsupported`] whose raise site declares the registered reason code: a request
    /// that, as declared, this route cannot answer (an option, inference mode, estimand or
    /// support it does not provide). The rendered message is byte-identical to the
    /// [`Self::Unsupported`] it replaces; the code is read with [`Self::reason_code`].
    #[error("{message}")]
    UnsupportedCoded {
        /// Registered reason code (checked with `antecedent_core::reason_code!`).
        code: &'static str,
        /// Explanation.
        message: &'static str,
    },
    /// [`Self::Unsupported`] that also carries structured diagnostics (see
    /// [`RefusalFields`]). The rendered message is byte-identical to the
    /// [`Self::Unsupported`] with the same message and, as there, no reason code is
    /// attached; the fields are read with [`Self::refusal_fields`].
    #[error("{message}")]
    UnsupportedWithFields {
        /// Explanation (the refusal).
        message: &'static str,
        /// Structured diagnostics of the failing step; absent entries stay absent.
        fields: Box<RefusalFields>,
    },
    /// The estimator refused because identification was not certified for this query.
    ///
    /// Distinct from [`Self::Data`]: the inputs are well formed, and the refusal is about
    /// what the graph licenses. Owned rather than `&'static str` because it carries the
    /// certificate's own reason and message.
    #[error("{message}")]
    NotCertified {
        /// Refusal, including the certificate's reason and message.
        message: String,
    },
    /// A banked posterior's coefficient count does not match the target design's, so
    /// its coefficients cannot be mapped one-to-one onto a prior. Callers that treat
    /// "not this mechanism's prior" as a fallback match this variant, never its text;
    /// the message still carries the registered `prior_dimension_mismatch` reason so a
    /// caller with no fallback (the prior transfer is terminal) surfaces a coded refusal.
    #[error(
        "{}prior_dimension_mismatch: posterior coefficient dimension {posterior} != expected n_coef {design}",
        antecedent_core::reason_code::PREFIX
    )]
    PriorDimensionMismatch {
        /// Coefficients in the banked posterior.
        posterior: usize,
        /// Coefficients in the target design.
        design: usize,
    },
    /// A resampled empirical law received no rows, so it has no defined table. Bootstrap
    /// loops count this as a failed replicate; anywhere else it is an ordinary error.
    #[error("{message}")]
    EmptyEmpiricalSample {
        /// Located description of the empty sample.
        message: String,
    },
    /// A refusal carrying a registered runtime reason code
    /// (`parity/reason_codes.toml`), rendered `reason=<code>: <message>` so every
    /// boundary reads the code the same way.
    #[error("{}{code}: {message}", antecedent_core::reason_code::PREFIX)]
    Refused {
        /// Registered reason code (checked with `antecedent_core::reason_code!`).
        code: &'static str,
        /// What was refused and what the caller can do instead, with the stable detail
        /// identifier apart from the explanation when the refusal names one.
        message: RefusalMessage,
    },
    /// [`Self::Refused`] that also carries structured diagnostics (see
    /// [`RefusalFields`]). The rendered message is byte-identical to the
    /// [`Self::Refused`] with the same code and message; the fields are read with
    /// [`Self::refusal_fields`], never parsed out of the text.
    #[error("{}{code}: {message}", antecedent_core::reason_code::PREFIX)]
    RefusedWithFields {
        /// Registered reason code (checked with `antecedent_core::reason_code!`).
        code: &'static str,
        /// What was refused and what the caller can do instead.
        message: RefusalMessage,
        /// Structured diagnostics of the failing step; absent entries stay absent.
        fields: Box<RefusalFields>,
    },
}

/// The text of a coded refusal: an optional stable `detail` identifier (a dotted name such as
/// `nested_markov.cancelled`) and the `explanation` of what was refused.
///
/// The two parts are supplied separately by the producer and kept apart, so a boundary that
/// stores a refusal as `(code, detail, explanation)` reads them back and never splits the
/// rendered text. The rendering is `detail: explanation` when there is a detail and the
/// explanation alone otherwise; it is what [`Display`](std::fmt::Display) and the `str`
/// dereference give.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefusalMessage {
    /// The rendered text, `detail: explanation` or the explanation alone.
    text: String,
    /// Byte length of the detail prefix, when the refusal names one.
    detail_len: Option<usize>,
}

impl RefusalMessage {
    /// A refusal that names no detail identifier.
    #[must_use]
    pub fn plain(explanation: impl Into<String>) -> Self {
        Self { text: explanation.into(), detail_len: None }
    }

    /// A refusal with a stable `detail` identifier and its `explanation`. An empty explanation
    /// renders as the detail alone.
    #[must_use]
    pub fn detailed(detail: impl AsRef<str>, explanation: impl AsRef<str>) -> Self {
        let (detail, explanation) = (detail.as_ref(), explanation.as_ref());
        let text = if explanation.is_empty() {
            detail.to_owned()
        } else {
            format!("{detail}: {explanation}")
        };
        Self { text, detail_len: Some(detail.len()) }
    }

    /// The detail identifier, when the producer named one.
    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        self.detail_len.map(|len| &self.text[..len])
    }

    /// What was refused and what the caller can do instead, without the detail prefix.
    #[must_use]
    pub fn explanation(&self) -> &str {
        match self.detail_len {
            Some(len) => self.text.get(len + ": ".len()..).unwrap_or(""),
            None => &self.text,
        }
    }

    /// The rendered text, `detail: explanation` or the explanation alone.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// The rendered text, owned.
    #[must_use]
    pub fn into_string(self) -> String {
        self.text
    }
}

impl std::ops::Deref for RefusalMessage {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl std::fmt::Display for RefusalMessage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.text)
    }
}

impl AsRef<str> for RefusalMessage {
    fn as_ref(&self) -> &str {
        &self.text
    }
}

impl From<RefusalMessage> for String {
    fn from(message: RefusalMessage) -> Self {
        message.text
    }
}

impl serde::Serialize for RefusalMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.text)
    }
}

impl PartialEq<str> for RefusalMessage {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

impl PartialEq<&str> for RefusalMessage {
    fn eq(&self, other: &&str) -> bool {
        self.text == *other
    }
}

impl PartialEq<String> for RefusalMessage {
    fn eq(&self, other: &String) -> bool {
        &self.text == other
    }
}

/// A float compared by its bit pattern, so a refusal that carries diagnostics stays
/// `Eq` (NaN equals the same NaN; `0.0` and `-0.0` differ).
#[derive(Clone, Copy, Debug)]
pub struct ExactF64(pub f64);

impl PartialEq for ExactF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for ExactF64 {}

impl std::fmt::Display for ExactF64 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// A refusal of a history operator, compared exactly.
///
/// [`HistoryOperatorError`] carries `f64` values (a refused weight, level or coefficient), so it
/// is only `PartialEq`: a refused `NaN` weight does not equal itself. This wrapper compares those
/// values by bit pattern, like [`ExactF64`], so the refusal can sit in the `Eq` enum
/// [`EstimationError`].
#[derive(Clone, Debug, Error)]
#[error(transparent)]
pub struct HistoryOperatorFault(pub HistoryOperatorError);

impl HistoryOperatorFault {
    /// The operator's own refusal.
    #[must_use]
    pub const fn operator(&self) -> &HistoryOperatorError {
        &self.0
    }
}

impl PartialEq for HistoryOperatorFault {
    fn eq(&self, other: &Self) -> bool {
        use HistoryOperatorError as E;
        match (&self.0, &other.0) {
            (
                E::NonFiniteWeight { role: left_role, value: left },
                E::NonFiniteWeight { role: right_role, value: right },
            )
            | (
                E::NonPositiveWeight { role: left_role, value: left },
                E::NonPositiveWeight { role: right_role, value: right },
            ) => left_role == right_role && ExactF64(*left) == ExactF64(*right),
            (E::NonFiniteLevel { value: left }, E::NonFiniteLevel { value: right })
            | (
                E::NegativeInnovationVariance { value: left },
                E::NegativeInnovationVariance { value: right },
            )
            | (
                E::UnstableAutoregression { autoregression: left },
                E::UnstableAutoregression { autoregression: right },
            ) => ExactF64(*left) == ExactF64(*right),
            (left, right) => left == right,
        }
    }
}

impl Eq for HistoryOperatorFault {}

/// Why a resampling loop could not hold the buffers it retains.
///
/// Each variant renders the message the loop gave when these refusals were free text.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ResampleBufferFault {
    /// `n × workers` index slots do not fit an addressable allocation.
    #[error("bootstrap index allocation exceeds addressable capacity")]
    IndexCapacity,
    /// The allocator refused the resample index buffer.
    #[error("bootstrap index allocation failed: {0}")]
    IndexAllocation(std::collections::TryReserveError),
    /// A resampled row identity does not index the table.
    #[error("a resampled row identity exceeds the addressable table size")]
    RowIdentity,
    /// The admission pool cannot fund the index scratch of the workers.
    #[error("admission pool cannot fund the bootstrap scratch")]
    IndexScratchPool,
    /// The allocator refused the buffers of a resample wave.
    #[error("resample wave allocation failed: {0}")]
    WaveAllocation(std::collections::TryReserveError),
    /// The bytes of a resample window do not fit in `u64`.
    #[error("resample window bytes overflow")]
    WindowBytesOverflow,
    /// The admission pool refused the bytes of a resample window.
    #[error("the resample window of {draws} draws needs {bytes} bytes: {source}")]
    WindowAdmission {
        /// Draws in the window.
        draws: usize,
        /// Bytes the window needs.
        bytes: u64,
        /// The pool's refusal.
        source: antecedent_core::execution::AdmissionError,
    },
    /// The gather buffers of the workers do not fit an addressable allocation.
    #[error("bootstrap gather allocation exceeds addressable capacity")]
    GatherCapacity,
    /// The admission pool cannot fund the gather buffers of the workers.
    #[error("admission pool cannot fund the bootstrap gather buffers")]
    GatherPool,
    /// The byte size of a draw buffer overflows.
    #[error("draw buffer size overflow")]
    DrawBufferOverflow,
    /// The admission pool refused a draw buffer.
    #[error("admission pool cannot fund the draw buffers: {0}")]
    DrawBufferAdmission(antecedent_core::execution::AdmissionError),
}

/// The cause class of a refusal of the out-of-core weighted-Gaussian reduction.
///
/// The reduction's own error holds an I/O error and an allocator error, so it is neither
/// `Clone` nor `Eq`; this is the part of it a caller branches on.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CapacityFaultKind {
    /// The run was cancelled; nothing was written or kept.
    Cancelled,
    /// The memory charges do not fit the declared ceiling.
    OverCeiling {
        /// Declared ceiling in bytes.
        ceiling: u64,
        /// Bytes the refused reservation needed.
        total: u64,
        /// Bytes by which the reservation exceeds the ceiling.
        over_by: u64,
    },
    /// The weighted Gram is singular, or the reduced design was refused by the solve.
    Singular,
    /// No positively weighted rows.
    Empty,
    /// A weight, design entry or response is invalid or not finite.
    InvalidRow,
    /// The batch bound, window or design shape is unusable.
    Shape,
    /// The row source, spill file or row file failed or changed between passes.
    Source,
    /// Any other cause; the message names it.
    Other,
}

/// A refusal of the out-of-core weighted-Gaussian reduction: its cause class and the
/// engine's own text.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[error("{message}")]
pub struct CapacityFault {
    /// The cause class.
    pub kind: CapacityFaultKind,
    message: String,
}

impl CapacityFault {
    /// The engine's rendered refusal.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<antecedent_io_core::capacity_account::WeightedCapacityError> for CapacityFault {
    fn from(error: antecedent_io_core::capacity_account::WeightedCapacityError) -> Self {
        use antecedent_io_core::capacity_account::WeightedCapacityError as Engine;
        let kind = match &error {
            Engine::Cancelled => CapacityFaultKind::Cancelled,
            Engine::OverCeiling { ceiling, total, over_by, .. } => CapacityFaultKind::OverCeiling {
                ceiling: *ceiling,
                total: *total,
                over_by: *over_by,
            },
            Engine::Singular | Engine::Solve(_) => CapacityFaultKind::Singular,
            Engine::Empty => CapacityFaultKind::Empty,
            Engine::Weight { .. } | Engine::NonFinite { .. } => CapacityFaultKind::InvalidRow,
            Engine::BatchBound | Engine::ReadPastBatch { .. } | Engine::Shape(_) => {
                CapacityFaultKind::Shape
            }
            Engine::Spill(_)
            | Engine::SpillPool(_)
            | Engine::Io { .. }
            | Engine::RowFile(_)
            | Engine::SourceChanged { .. }
            | Engine::RowSource { .. } => CapacityFaultKind::Source,
            _ => CapacityFaultKind::Other,
        };
        Self { kind, message: error.to_string() }
    }
}

impl From<antecedent_io_core::capacity_account::WeightedCapacityError> for EstimationError {
    fn from(error: antecedent_io_core::capacity_account::WeightedCapacityError) -> Self {
        Self::Capacity(error.into())
    }
}

/// Structured diagnostics of a refusal: which treatment or cell failed, at which stage
/// and why, plus whatever numbers the failing step actually produced.
///
/// Every entry is optional or empty by default and stays that way unless the failing
/// step measured it. A diagnostic that needs a fitted nuisance is never invented from
/// a fit that failed before producing one. Build one with
/// `RefusalFields { stage: .., ..RefusalFields::default() }`; later fields are added
/// only as further optional or empty entries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RefusalFields {
    /// Pipeline stage that refused (`preflight`, `rank_drop`, ...).
    pub stage: Option<String>,
    /// Failing treatment or joint cell, by name.
    pub subject: Option<String>,
    /// Short statement of why it failed.
    pub reason: Option<String>,
    /// Effective sample size per arm, `(arm label, ess)`.
    pub arm_ess: Vec<(String, ExactF64)>,
    /// Smallest fitted propensity.
    pub propensity_min: Option<ExactF64>,
    /// Largest fitted propensity.
    pub propensity_max: Option<ExactF64>,
    /// Fitted propensity quantiles, `(probability, value)`.
    pub propensity_quantiles: Vec<(ExactF64, ExactF64)>,
    /// Number of clusters, when the estimator clusters.
    pub cluster_count: Option<u64>,
    /// Numerical rank of the design.
    pub numerical_rank: Option<u64>,
    /// Number of design columns the rank is out of.
    pub design_columns: Option<u64>,
    /// Columns the refusal implicates, by name.
    pub implicated_columns: Vec<String>,
    /// What the caller can change to get past the refusal.
    pub remedy: Option<String>,
    /// IRLS iterations of the refused GLM fit.
    pub glm_iterations: Option<u64>,
    /// Smallest `min(mu, 1 - mu)` over the refused binomial fit's rows (distance of the
    /// closest fitted probability to 0 or 1), at the returned coefficients.
    pub boundary_margin: Option<ExactF64>,
    /// Rows of the refused binomial fit whose fitted probability is within `1e-8` of 0 or 1.
    pub boundary_count: Option<u64>,
    /// Fewest clusters the refused cluster-robust variance needs (the failing count is
    /// [`Self::cluster_count`]).
    pub cluster_minimum: Option<u64>,
}

impl EstimationError {
    /// Fixed unsupported query option.
    #[must_use]
    pub const fn unsupported(message: &'static str) -> Self {
        Self::Unsupported { message }
    }

    /// [`Self::unsupported`] with the registered reason code its raise site declares (see
    /// [`Self::UnsupportedCoded`]). Pass the code through `antecedent_core::reason_code!` so
    /// an unregistered code fails to compile.
    #[must_use]
    pub const fn unsupported_coded(code: &'static str, message: &'static str) -> Self {
        Self::UnsupportedCoded { code, message }
    }

    /// Fixed unsupported query option that names a remedy (see
    /// [`Self::UnsupportedWithRemedy`]).
    #[must_use]
    pub const fn unsupported_with_remedy(message: &'static str, remedy: &'static str) -> Self {
        Self::UnsupportedWithRemedy { message, remedy }
    }

    /// What the caller can change to get past this refusal, when the refusal
    /// names one. Optional and additive: `None` for every error that names no
    /// remedy, and the rendered message never includes it.
    #[must_use]
    pub const fn remedy(&self) -> Option<&'static str> {
        match self {
            Self::UnsupportedWithRemedy { remedy, .. } => Some(remedy),
            _ => None,
        }
    }

    /// The remedy a refusal built with structured diagnostics records, when it records one.
    #[must_use]
    pub fn field_remedy(&self) -> Option<&str> {
        match self {
            Self::RefusedWithFields { fields, .. }
            | Self::UnsupportedWithFields { fields, .. }
            | Self::OverlapWithFields { fields, .. } => fields.remedy.as_deref(),
            _ => None,
        }
    }

    /// Refusal to estimate a quantity whose identification was not certified.
    #[must_use]
    pub fn not_certified(stage: &str, reason: &str, message: &str) -> Self {
        Self::NotCertified {
            message: format!(
                "{stage} refused: identification was not certified ({reason}): {message}"
            ),
        }
    }

    /// A refusal with a registered runtime reason code and no detail identifier.
    #[must_use]
    pub fn refused(code: &'static str, message: impl Into<String>) -> Self {
        Self::Refused { code, message: RefusalMessage::plain(message) }
    }

    /// A refusal with a registered runtime reason code, a stable `detail` identifier and its
    /// `explanation`, kept apart so a boundary reads them back without splitting text.
    #[must_use]
    pub fn refused_detail(
        code: &'static str,
        detail: impl AsRef<str>,
        explanation: impl AsRef<str>,
    ) -> Self {
        Self::Refused { code, message: RefusalMessage::detailed(detail, explanation) }
    }

    /// A refusal with a registered runtime reason code and an already built message.
    #[must_use]
    pub fn refused_message(code: &'static str, message: RefusalMessage) -> Self {
        Self::Refused { code, message }
    }

    /// A refusal with a registered runtime reason code and structured diagnostics.
    #[must_use]
    pub fn refused_with_fields(
        code: &'static str,
        message: impl Into<String>,
        fields: RefusalFields,
    ) -> Self {
        Self::RefusedWithFields {
            code,
            message: RefusalMessage::plain(message),
            fields: Box::new(fields),
        }
    }

    /// [`Self::refused_detail`] with structured diagnostics.
    #[must_use]
    pub fn refused_detail_with_fields(
        code: &'static str,
        detail: impl AsRef<str>,
        explanation: impl AsRef<str>,
        fields: RefusalFields,
    ) -> Self {
        Self::RefusedWithFields {
            code,
            message: RefusalMessage::detailed(detail, explanation),
            fields: Box::new(fields),
        }
    }

    /// The refusal's code and message, when it is a coded refusal (with or without
    /// diagnostics).
    #[must_use]
    pub fn coded_refusal(&self) -> Option<(&'static str, &RefusalMessage)> {
        match self {
            Self::Refused { code, message } | Self::RefusedWithFields { code, message, .. } => {
                Some((*code, message))
            }
            _ => None,
        }
    }

    /// The registered runtime reason code this refusal carries, when it is a scientific
    /// refusal that has one: a coded refusal's own code, or the code derived from the variant
    /// and its [`RefusalFields`] for a refusal built without one (a rank-deficient design, a
    /// refused GLM fit, too few clusters, a failed overlap or support gate, a retarget
    /// refusal). Never changes a rendered message; `None` for every other error.
    ///
    /// A stats-layer GLM refusal cannot say whether the fit was a propensity gate, so every
    /// separated, saturated or non-converged fit maps to `glm_fit_refused`; the propensity
    /// route that wants `positivity_violated` converts its fit error first.
    #[must_use]
    pub fn reason_code(&self) -> Option<&'static str> {
        use antecedent_core::reason_code;
        match self {
            Self::Refused { code, .. } | Self::RefusedWithFields { code, .. } => Some(code),
            Self::OverlapWithFields { .. } => Some(reason_code!("positivity_violated")),
            Self::UnsupportedWithFields { fields, .. } => {
                if fields.stage.as_deref() != Some("retarget") {
                    return None;
                }
                Some(match fields.reason.as_deref() {
                    Some(
                        "weights_depend_on_treatment_or_descendant"
                        | "weights_outside_adjustment_set"
                        | "descendant_closure_unavailable"
                        | "undeclared_nonconstant_weights",
                    ) => reason_code!("target_weights_not_licensed"),
                    _ => reason_code!("positivity_violated"),
                })
            }
            Self::Stats(StatsError::RankDeficient { .. }) => {
                Some(reason_code!("design_rank_deficient"))
            }
            Self::Stats(StatsError::GlmRefused { .. }) => Some(reason_code!("glm_fit_refused")),
            Self::Stats(StatsError::FewClusters { .. }) => Some(reason_code!("too_few_clusters")),
            Self::Overlap { message } => Some(if message.contains("RequireDiagnostics") {
                // An explicit override has no meaning on a route whose positivity is mandatory.
                reason_code!("option_not_applicable")
            } else {
                // A route that cannot check positivity needs the override declared.
                reason_code!("required_option_missing")
            }),
            Self::IncompatibleEstimand { .. } => Some(reason_code!("strategy_incompatible")),
            Self::EffectModifiers => Some(reason_code!("option_not_applicable")),
            Self::UnsupportedCoded { code, .. } => Some(code),
            // A plain `Unsupported` is malformed input or an operational limit, or a refusal
            // whose raise site has not declared its code; a code is never inferred from wording.
            _ => None,
        }
    }

    /// Structured diagnostics of a refusal that carries them: one built with
    /// [`Self::refused_with_fields`] or [`Self::UnsupportedWithFields`], or a stats-layer
    /// refusal whose own record holds them (a refused GLM fit, a rank-deficient design, too
    /// few clusters), read from that record and never parsed out of the message. Additive:
    /// `None` for every other error, and never part of the rendered message. A diagnostic the
    /// failing step did not measure stays absent.
    #[must_use]
    pub fn refusal_fields(&self) -> Option<Cow<'_, RefusalFields>> {
        match self {
            Self::RefusedWithFields { fields, .. }
            | Self::UnsupportedWithFields { fields, .. }
            | Self::OverlapWithFields { fields, .. } => Some(Cow::Borrowed(fields)),
            Self::Stats(error) => stats_refusal_fields(error).map(Cow::Owned),
            _ => None,
        }
    }

    /// [`Self::unsupported`] with structured diagnostics.
    #[must_use]
    pub fn unsupported_with_fields(message: &'static str, fields: RefusalFields) -> Self {
        Self::UnsupportedWithFields { message, fields: Box::new(fields) }
    }

    /// Ad-hoc data-layer message (maps to [`DataError::InvalidArgument`]).
    #[must_use]
    pub fn data_msg(message: impl Into<String>) -> Self {
        Self::Data(DataError::InvalidArgument { message: message.into() })
    }

    /// Ad-hoc stats-layer message (maps to [`StatsError::Backend`]).
    #[must_use]
    pub fn stats_msg(message: impl Into<String>) -> Self {
        Self::Stats(StatsError::Backend(message.into()))
    }
}

/// The structured diagnostics a stats-layer refusal records: stage, reason, remedy and the
/// numbers its own failing step produced. A subject (treatment or cell) is not known at this
/// layer and stays absent; so does every figure the fit did not measure.
fn stats_refusal_fields(error: &StatsError) -> Option<RefusalFields> {
    match error {
        StatsError::GlmRefused { kind, iterations, boundary_count, .. } => {
            let (reason, remedy) = match kind {
                GlmRefusalKind::NonConverged => (
                    "non_converged",
                    "standardize or drop near-collinear columns, or raise the iteration limit",
                ),
                GlmRefusalKind::Separated => (
                    "separated",
                    "drop or coarsen the separating columns, or (AIPW only) declare a penalized propensity \
                     (propensity_penalty)",
                ),
                GlmRefusalKind::BoundarySaturated => (
                    "boundary_saturated",
                    "drop or coarsen the near-deterministic columns, or (AIPW only) declare a penalized \
                     propensity (propensity_penalty)",
                ),
            };
            Some(RefusalFields {
                stage: Some("glm_fit".to_string()),
                reason: Some(reason.to_string()),
                remedy: Some(remedy.to_string()),
                glm_iterations: Some(u64::from(*iterations)),
                boundary_margin: error.glm_boundary_margin().map(ExactF64),
                boundary_count: *boundary_count,
                ..RefusalFields::default()
            })
        }
        StatsError::RankDeficient { rank, ncols } => Some(RefusalFields {
            stage: Some("design_rank".to_string()),
            reason: Some("rank_deficient".to_string()),
            remedy: Some(
                "run preflight to name the dependent columns, then drop them or use \
                 estimate_with_rank_drop"
                    .to_string(),
            ),
            numerical_rank: Some(u64::try_from(*rank).unwrap_or(u64::MAX)),
            design_columns: Some(u64::try_from(*ncols).unwrap_or(u64::MAX)),
            ..RefusalFields::default()
        }),
        StatsError::FewClusters { clusters, minimum, .. } => Some(RefusalFields {
            stage: Some("cluster_variance".to_string()),
            reason: Some("too_few_clusters".to_string()),
            remedy: Some(
                "supply more clusters, coarser-grained dependence, or a non-clustered standard \
                 error"
                    .to_string(),
            ),
            cluster_count: Some(*clusters),
            cluster_minimum: Some(*minimum),
            ..RefusalFields::default()
        }),
        _ => None,
    }
}

/// A coded estimation refusal keeps its code across the artifact boundary;
/// every other failure is a conversion failure there.
///
/// The conversion belongs to the crate that owns the source error: the estimators sit
/// above `antecedent-io-core`, so the core cannot name [`EstimationError`], and a
/// boxed or stringly source variant on `IoError` would only move the same match into
/// every caller.
impl From<EstimationError> for antecedent_io_core::IoError {
    fn from(error: EstimationError) -> Self {
        match error {
            EstimationError::Refused { code, message } => {
                Self::Refused { code, message: message.into_string() }
            }
            other => Self::Convert(other.to_string()),
        }
    }
}

impl From<ExactLawError> for EstimationError {
    fn from(error: ExactLawError) -> Self {
        Self::ExactLaw(error)
    }
}

impl From<ProviderSnapshotError> for EstimationError {
    fn from(error: ProviderSnapshotError) -> Self {
        Self::Provider(error)
    }
}

impl From<EvalError> for EstimationError {
    fn from(error: EvalError) -> Self {
        Self::Evaluation(error)
    }
}

impl From<ProgramError> for EstimationError {
    fn from(error: ProgramError) -> Self {
        Self::Program(error)
    }
}

impl From<antecedent_learn::FittedEffectError> for EstimationError {
    fn from(error: antecedent_learn::FittedEffectError) -> Self {
        use antecedent_learn::FittedEffectError;
        match error {
            FittedEffectError::InvalidArgument(message) => Self::data_msg(message),
            FittedEffectError::Predictor(message) => Self::stats_msg(message),
            // The enum is non-exhaustive; a future variant renders as its own text.
            other => Self::stats_msg(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EstimationError, HistoryOperatorFault, RefusalFields, RefusalMessage};
    use antecedent_model::history_operators::{HistoryOperatorError, WeightRole};

    /// A refused `NaN` weight equals itself, and differs from a refused infinity, so the
    /// refusal can sit in the `Eq` error enum.
    #[test]
    fn history_operator_fault_compares_refused_floats_by_bits() {
        let weight = |value| {
            EstimationError::from(HistoryOperatorFault(HistoryOperatorError::NonFiniteWeight {
                role: WeightRole::Censoring,
                value,
            }))
        };
        assert_eq!(weight(f64::NAN), weight(f64::NAN));
        assert_ne!(weight(f64::NAN), weight(f64::INFINITY));
        assert_ne!(
            weight(f64::INFINITY),
            EstimationError::from(HistoryOperatorFault(HistoryOperatorError::NonFiniteWeight {
                role: WeightRole::Assignment,
                value: f64::INFINITY,
            }))
        );
        assert!(weight(f64::NAN).to_string().contains("not finite"));
    }

    /// Only a refusal built with a remedy carries one; every other error reads
    /// `None`, and attaching a remedy never changes the rendered message.
    #[test]
    fn remedy_is_absent_unless_named_and_never_changes_the_message() {
        let plain = EstimationError::unsupported("refused");
        assert_eq!(plain.remedy(), None);
        assert_eq!(EstimationError::refused("route_not_supported", "no").remedy(), None);
        assert_eq!(EstimationError::stats_msg("backend").remedy(), None);
        assert_eq!(EstimationError::TargetPopulation.remedy(), None);
        assert_eq!(EstimationError::not_certified("s", "r", "m").remedy(), None);
        let remedied = EstimationError::unsupported_with_remedy("refused", "do this instead");
        assert_eq!(remedied.remedy(), Some("do this instead"));
        assert_eq!(remedied.to_string(), plain.to_string());
    }

    /// The refusal text keeps its detail apart from its explanation and renders both.
    #[test]
    fn refusal_message_keeps_detail_and_explanation_apart() {
        let detailed =
            EstimationError::refused_detail("route_not_supported", "area.reason", "no: go");
        let (code, message) = detailed.coded_refusal().expect("a coded refusal");
        assert_eq!(code, "route_not_supported");
        assert_eq!(message.detail(), Some("area.reason"));
        assert_eq!(message.explanation(), "no: go");
        assert_eq!(message.as_str(), "area.reason: no: go");
        assert!(detailed.to_string().ends_with("route_not_supported: area.reason: no: go"));
        let plain = EstimationError::refused("route_not_supported", "no: go");
        let (_, message) = plain.coded_refusal().expect("a coded refusal");
        assert_eq!(message.detail(), None);
        assert_eq!(message.explanation(), "no: go");
        let bare = RefusalMessage::detailed("area.reason", "");
        assert_eq!(bare.as_str(), "area.reason");
        assert_eq!(bare.explanation(), "");
    }

    /// Structured fields ride beside a refusal without changing its code or message, and
    /// a plain refusal reads no fields.
    #[test]
    fn refusal_fields_are_additive_to_the_coded_message() {
        let plain = EstimationError::refused("route_not_supported", "no");
        assert!(plain.refusal_fields().is_none());
        let fields = RefusalFields {
            numerical_rank: Some(174),
            implicated_columns: vec!["z174".to_string()],
            ..RefusalFields::default()
        };
        let rich = EstimationError::refused_with_fields("route_not_supported", "no", fields);
        assert_eq!(rich.to_string(), plain.to_string());
        let read = rich.refusal_fields().expect("fields were attached");
        assert_eq!(read.numerical_rank, Some(174));
        assert!(read.stage.is_none() && read.arm_ess.is_empty() && read.propensity_min.is_none());
    }

    /// A plain refusal that says the request as declared cannot be answered by the route
    /// carries a registered code, and a malformed-input or operational message carries none.
    #[test]
    fn declaration_refusals_carry_codes_by_variant_never_by_wording() {
        let code = |error: EstimationError| error.reason_code();
        assert_eq!(
            code(EstimationError::Overlap {
                message: "LinearAdjustmentAte requires ExplicitOverride overlap policy"
            }),
            Some("required_option_missing")
        );
        assert_eq!(
            code(EstimationError::Overlap {
                message: "propensity estimators require RequireDiagnostics overlap policy"
            }),
            Some("option_not_applicable")
        );
        assert_eq!(
            code(EstimationError::IncompatibleEstimand { message: "expects an iv estimand" }),
            Some("strategy_incompatible")
        );
        assert_eq!(code(EstimationError::EffectModifiers), Some("option_not_applicable"));
        // A plain `Unsupported` carries no inferred code.
        assert_eq!(
            code(EstimationError::unsupported("sharp RD does not support effect modifiers")),
            None
        );
    }
}
