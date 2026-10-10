//! Sampled observation recovery (2.3A X10): the exact 2.2 binary m-graph recovery
//! formula applied to finite counted observation-pattern rows, with a whole-method
//! outer bootstrap.
//!
//! The estimator is one composed procedure. The rows are tabulated into the
//! empirical observed pattern law over `R ∪ X* ∪ O`; the whole recovered law
//! `P(X(1), O)` is evaluated through [`crate::evaluate_exact_recovery`] (the one
//! checked formula of the derivation, nothing re-derived here) and the downstream
//! effect through [`crate::evaluate_recovered_effect`]. The interval resamples
//! whole ROWS with replacement, so every observed margin of a replicate is computed
//! from the same drawn rows and the covariance of margins computed from overlapping
//! rows is preserved. Each replicate reruns recovery and effect evaluation and the
//! requested interval is formed from the replicate effects, never from per-margin
//! variances.
//!
//! The empirical law is a finite-sample table, so the recovered masses need not
//! sum to one (the population identity is exact, the sample one is not). The
//! recovered law is never renormalized; its mass defect is reported and bounded by
//! an explicit tolerance, and the effect probabilities are normalized by the
//! evaluated total. A zero complete-case cell forces a zero denominator in the
//! recovery formula (every margin the formula divides by contains that cell), so
//! it is refused as `route_not_supported` /
//! `sampled_recovery.unrecoverable_pattern` with the offending patterns retained.
//!
//! Calibration of the interval (whole-method coverage) is UNMEASURED: no coverage
//! claim is made and the public interval route stays closed
//! ([`sampled_recovery_route_frozen`]). The digests are fingerprints (FNV-based),
//! not cryptographic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use antecedent_core::{EvidenceCatalog, ExecutionContext, Value, VariableId};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits,
    LawTolerance,
};
use antecedent_graph::Admg;
use antecedent_identify::recovery::MISSING_LEVEL;
use antecedent_identify::{
    ObservationRecoveryQuery, RecoveredEffectQuery, RecoveryDecision, RecoveryDerivation,
    RecoveryDetail, RecoveryError, RecoveryLimits, decide_observation_recovery,
};

use antecedent_kernels::{norm_cdf, norm_inv};

use crate::recovery::{RecoveredLaw, evaluate_exact_recovery, evaluate_recovered_effect};
use crate::splitmix::{mix64, seed_mix, splitmix64};

/// Most bootstrap replicates one request may run.
pub const SAMPLED_RECOVERY_MAX_REPLICATES: usize = 2000;
/// Fewest bootstrap replicates required by the supported `BCa` protocol.
pub const SAMPLED_RECOVERY_MIN_REPLICATES: usize = 2000;
/// Most observation rows one request may carry.
pub const SAMPLED_RECOVERY_MAX_ROWS: usize = 1_000_000;
/// Most binary substantive variables (`X ∪ O`) one request may recover.
pub const SAMPLED_RECOVERY_MAX_BINARY_VARIABLES: usize = 6;
/// Nominal level of the whole-row `BCa` interval.
pub const SAMPLED_RECOVERY_INTERVAL_LEVEL: f64 = 0.95;
/// Original source artifact status, distinct from measured-envelope authorization.
pub const SAMPLED_RECOVERY_CALIBRATION: &str = "unmeasured";

/// Why a sampled-recovery request was refused. The detail and reason code are
/// fixed per kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SampledRecoveryDetail {
    /// The sampled provider or interval route is requested before full-path
    /// truth, calibration and replay pass.
    RouteFrozen,
    /// An interval was requested from per-margin variances alone.
    ComponentVarianceOnly,
    /// An observation pattern, zero cell or graph violates the exact recovery
    /// formula.
    UnrecoverablePattern,
    /// Too many bootstrap replicates hit a zero cell or failed recovery.
    TooManyFailedReplicates,
    /// A declared bound is exceeded.
    BoundsExceeded,
    /// A malformed row set, snapshot, derivation or configuration.
    InvalidInput,
    /// A stored receipt does not match its recomputed digest or replay.
    ReceiptMismatch,
    /// The execution context was cancelled.
    Cancelled,
}

impl SampledRecoveryDetail {
    /// Stable namespaced detail.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::RouteFrozen => "sampled_recovery.route_frozen",
            Self::ComponentVarianceOnly => "sampled_recovery.component_variance_only",
            Self::UnrecoverablePattern => "sampled_recovery.unrecoverable_pattern",
            Self::TooManyFailedReplicates => "sampled_recovery.too_many_failed_replicates",
            Self::BoundsExceeded => "sampled_recovery.bounds_exceeded",
            Self::InvalidInput => "sampled_recovery.invalid_input",
            Self::ReceiptMismatch => "sampled_recovery.receipt_mismatch",
            Self::Cancelled => "sampled_recovery.cancelled",
        }
    }

    /// Registered top-level reason code.
    #[must_use]
    pub const fn reason_code(self) -> &'static str {
        match self {
            Self::RouteFrozen | Self::ComponentVarianceOnly => {
                antecedent_core::reason_code!("cell_not_licensed")
            }
            Self::UnrecoverablePattern | Self::TooManyFailedReplicates | Self::BoundsExceeded => {
                antecedent_core::reason_code!("route_not_supported")
            }
            Self::InvalidInput | Self::ReceiptMismatch => {
                antecedent_core::reason_code!("invalid_argument")
            }
            Self::Cancelled => antecedent_core::reason_code!("transport_budget_cancel"),
        }
    }
}

/// A typed sampled-recovery refusal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampledRecoveryError {
    /// Why.
    pub detail: SampledRecoveryDetail,
    /// Human explanation.
    pub message: String,
    /// The offending observation patterns (zero complete-case cells), retained.
    pub patterns: Vec<ObservationPattern>,
    /// The exact-recovery refusal this wraps, when one caused it.
    pub recovery: Option<RecoveryDetail>,
}

impl SampledRecoveryError {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.detail.reason_code()
    }
}

impl fmt::Display for SampledRecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.detail.detail(), self.message)
    }
}

impl std::error::Error for SampledRecoveryError {}

fn refusal(detail: SampledRecoveryDetail, message: impl Into<String>) -> SampledRecoveryError {
    SampledRecoveryError { detail, message: message.into(), patterns: Vec::new(), recovery: None }
}

fn cancelled() -> SampledRecoveryError {
    refusal(SampledRecoveryDetail::Cancelled, "sampled recovery cancelled")
}

fn unrecoverable(patterns: Vec<ObservationPattern>, message: String) -> SampledRecoveryError {
    SampledRecoveryError {
        detail: SampledRecoveryDetail::UnrecoverablePattern,
        message,
        patterns,
        recovery: None,
    }
}

/// Wrap an exact-recovery refusal, keeping its detail.
fn from_recovery(error: &RecoveryError) -> SampledRecoveryError {
    let detail = match error.detail {
        RecoveryDetail::Budget => SampledRecoveryDetail::Cancelled,
        RecoveryDetail::Positivity
        | RecoveryDetail::InvalidObservedLaw
        | RecoveryDetail::HandoffMismatch
        | RecoveryDetail::NonrecoverableWitness
        | RecoveryDetail::UnsupportedMechanism => SampledRecoveryDetail::UnrecoverablePattern,
        _ => SampledRecoveryDetail::InvalidInput,
    };
    SampledRecoveryError {
        detail,
        message: error.to_string(),
        patterns: Vec::new(),
        recovery: Some(error.detail),
    }
}

/// The route-level refusal of the public sampled interval: closed until full-path
/// truth, calibration and replay pass.
#[must_use]
pub fn sampled_recovery_route_frozen() -> SampledRecoveryError {
    refusal(
        SampledRecoveryDetail::RouteFrozen,
        "the sampled observation-recovery interval route is closed until the whole-method \
         calibration grid, public producer and artifact consumer pass",
    )
}

/// The refusal of an interval built from per-margin variances alone: the margins
/// of one recovered law are computed from overlapping rows, so their covariance
/// (carried by the whole-method bootstrap) is required.
#[must_use]
pub fn refuse_component_variance_only(margin_variances: &[f64]) -> SampledRecoveryError {
    refusal(
        SampledRecoveryDetail::ComponentVarianceOnly,
        format!(
            "{} per-margin variance(s) offered: an interval needs the covariance of margins \
             computed from overlapping rows and the whole recovered-effect path, not component \
             variances",
            margin_variances.len()
        ),
    )
}

/// One observation pattern. Bit `i` of each field refers to the `i`-th partially
/// observed variable in ascending variable-id order (the derivation's canonical
/// query order), and bit `j` of `fully` to the `j`-th fully observed variable.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ObservationPattern {
    /// Response indicators `R_i` (1 observed, 0 missing).
    pub responses: u8,
    /// Proxy values `X*_i` when `R_i = 1` (the bit must be 0 when `R_i = 0`,
    /// where the proxy is `?`).
    pub proxies: u8,
    /// Fully observed values `O_j`.
    pub fully: u8,
}

/// One observation row: a unit's identity and its observed pattern.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationRow {
    /// Row (unit) identity; unique within the input.
    pub id: u64,
    /// Observed pattern.
    pub pattern: ObservationPattern,
}

/// Finite counted observation rows of one snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampledObservationInput {
    /// Snapshot identity of the rows; must equal the snapshot the derivation's
    /// catalog binds.
    pub snapshot_id: String,
    /// The rows.
    pub rows: Vec<ObservationRow>,
}

/// Supported bias-corrected and accelerated whole-row interval procedure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampledIntervalMethod {
    /// Bias-corrected and accelerated whole-row bootstrap (artifact v3).
    Bca,
}

impl SampledIntervalMethod {
    /// Actual interval identity, never a coverage claim.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bca => "bootstrap_bca",
        }
    }
}

/// A delete-one-row effect shared by every row of this observation pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledJackknifeRecord {
    /// Original observed pattern.
    pub pattern: ObservationPattern,
    /// Number of rows with that pattern; every delete-one-row value is represented.
    pub multiplicity: u64,
    /// Effect from all original rows except one row of this pattern.
    pub effect: f64,
}

/// `BCa` arithmetic and the complete count-compressed delete-one jackknife.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledBcaReceipt {
    /// Bias correction: inverse-normal rank with half weight for exact ties.
    pub bias_correction: f64,
    /// Jackknife acceleration, with multiplicity weights.
    pub acceleration: f64,
    /// Adjusted probabilities used for type-7 quantiles.
    pub adjusted_probabilities: [f64; 2],
    /// Every nonempty observed pattern, in canonical pattern order.
    pub jackknife: Vec<SampledJackknifeRecord>,
}

/// Request configuration. Numerical tolerances are explicit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledRecoveryConfig {
    /// Explicit supported interval procedure.
    pub interval_method: SampledIntervalMethod,
    /// Bootstrap replicates, from `SAMPLED_RECOVERY_MIN_REPLICATES` to
    /// `SAMPLED_RECOVERY_MAX_REPLICATES`.
    pub replicates: usize,
    /// Bootstrap seed; replicate `r` draws from a stream depending only on
    /// `(seed, r)`.
    pub seed: u64,
    /// Must be zero: every failed replicate invalidates the supported `BCa` procedure.
    pub max_failed_fraction: f64,
    /// Absolute tolerance on the unit mass of the empirical observed law and the
    /// recovered law (the recovered law is not renormalized).
    pub normalization_tolerance: f64,
    /// Recovered cells whose expected count `n * p` is below this are reported as
    /// small cells (diagnostic only).
    pub small_cell_count: f64,
    /// Treatment level of the effect contrast.
    pub treated_level: f64,
    /// Reference level of the effect contrast.
    pub control_level: f64,
}

impl SampledRecoveryConfig {
    /// `replicates` bootstrap replicates from `seed`, with the default
    /// tolerances and the contrast `do(1) - do(0)`.
    #[must_use]
    pub const fn new(replicates: usize, seed: u64) -> Self {
        Self {
            interval_method: SampledIntervalMethod::Bca,
            replicates,
            seed,
            max_failed_fraction: 0.0,
            normalization_tolerance: 0.1,
            small_cell_count: 5.0,
            treated_level: 1.0,
            control_level: 0.0,
        }
    }
    /// Frozen `BCa` candidate protocol: 2,000 whole-row bootstrap draws; no failed
    /// draw is discarded to form a conditional distribution.
    #[must_use]
    pub const fn bca(seed: u64) -> Self {
        Self {
            interval_method: SampledIntervalMethod::Bca,
            max_failed_fraction: 0.0,
            ..Self::new(2000, seed)
        }
    }
}

/// Small-cell and support diagnostics of the point recovery.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledRecoveryDiagnostics {
    /// Number of rows.
    pub rows: usize,
    /// Count of every nonempty observed pattern, ascending by pattern.
    pub pattern_counts: Vec<(ObservationPattern, u64)>,
    /// Smallest count over the complete-case patterns (`R = 1` everywhere).
    pub min_complete_case_count: u64,
    /// Smallest recovered cell probability.
    pub min_recovered_cell: f64,
    /// Smallest recovered cell expected count (`rows * min_recovered_cell`).
    pub min_recovered_cell_count: f64,
    /// Indices (recovered-law cell order) of cells below the small-cell count.
    pub small_recovered_cells: Vec<usize>,
    /// Total mass of the recovered law.
    pub recovered_total: f64,
    /// `|recovered_total - 1|`.
    pub normalization_defect: f64,
    /// The tolerance the defect was bounded by.
    pub normalization_tolerance: f64,
}

/// The requested interval of the whole-method bootstrap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledEffectInterval {
    /// Nominal level.
    pub level: f64,
    /// Lower empirical quantile endpoint.
    pub lower: f64,
    /// Upper empirical quantile endpoint.
    pub upper: f64,
    /// Always [`SAMPLED_RECOVERY_CALIBRATION`]: no coverage claim is made.
    pub calibration: &'static str,
}

/// One bootstrap replicate: its identity, the digest of the rows it drew and its
/// effect, or why it failed.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledReplicateRecord {
    /// Replicate id.
    pub id: u32,
    /// Digest of the drawn row ids, in draw order.
    pub selection_digest: String,
    /// The recovered effect, `None` when the replicate failed.
    pub effect: Option<f64>,
    /// Why it failed.
    pub failure: Option<String>,
    /// Zero complete-case patterns of a failed replicate.
    pub zero_patterns: Vec<ObservationPattern>,
}

/// Replayable receipt of one sampled recovery.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledRecoveryReceipt {
    /// Snapshot identity of the rows.
    pub snapshot_id: String,
    /// Number of rows.
    pub rows: usize,
    /// Identity of the recovery derivation.
    pub derivation_identity: String,
    /// Digest of the snapshot, ids and patterns of every row, in order.
    pub input_digest: String,
    /// The configuration, including the seed.
    pub config: SampledRecoveryConfig,
    /// The recovered effect point.
    pub point_effect: f64,
    /// The requested interval.
    pub interval: SampledEffectInterval,
    /// Every replicate, failed ones included, in id order.
    pub replicates: Vec<SampledReplicateRecord>,
    /// Exact `BCa` settings and jackknife; required for every supported receipt.
    pub bca: Option<SampledBcaReceipt>,
    /// Digest of everything above.
    pub receipt_digest: String,
}

impl SampledRecoveryReceipt {
    /// Recompute the receipt digest from the stored fields (not from
    /// `receipt_digest`).
    #[must_use]
    pub fn recompute_digest(&self) -> String {
        let mut digest = Digest::new();
        digest.str("sampled_recovery.receipt.bca.v1");
        digest.str(&self.snapshot_id);
        digest.usize(self.rows);
        digest.str(&self.derivation_identity);
        digest.str(&self.input_digest);
        digest.usize(self.config.replicates);
        digest.u64(self.config.seed);
        digest.f64(self.config.max_failed_fraction);
        digest.f64(self.config.normalization_tolerance);
        digest.f64(self.config.small_cell_count);
        digest.f64(self.config.treated_level);
        digest.f64(self.config.control_level);
        digest.f64(self.point_effect);
        digest.f64(self.interval.level);
        digest.f64(self.interval.lower);
        digest.f64(self.interval.upper);
        digest.str(self.interval.calibration);
        digest.usize(self.replicates.len());
        for record in &self.replicates {
            digest.u64(u64::from(record.id));
            digest.str(&record.selection_digest);
            match record.effect {
                Some(effect) => {
                    digest.bytes(&[1]);
                    digest.f64(effect);
                }
                None => digest.bytes(&[0]),
            }
            if self.config.interval_method == SampledIntervalMethod::Bca {
                match &record.failure {
                    Some(reason) => {
                        digest.bytes(&[1]);
                        digest.str(reason);
                    }
                    None => digest.bytes(&[0]),
                }
            }
            digest.usize(record.zero_patterns.len());
            for pattern in &record.zero_patterns {
                digest.pattern(*pattern);
            }
        }
        if self.config.interval_method == SampledIntervalMethod::Bca {
            digest.str(self.config.interval_method.name());
            digest.str("midrank_exact_ties:type7:delete_one_row");
            match &self.bca {
                Some(bca) => {
                    digest.bytes(&[1]);
                    digest.f64(bca.bias_correction);
                    digest.f64(bca.acceleration);
                    for p in bca.adjusted_probabilities {
                        digest.f64(p);
                    }
                    digest.usize(bca.jackknife.len());
                    for row in &bca.jackknife {
                        digest.pattern(row.pattern);
                        digest.u64(row.multiplicity);
                        digest.f64(row.effect);
                    }
                }
                None => digest.bytes(&[0]),
            }
        }
        digest.finish()
    }

    /// Whether the stored digest is the recomputed one.
    #[must_use]
    pub fn verify_digest(&self) -> bool {
        self.recompute_digest() == self.receipt_digest
    }
}

/// The result of one sampled recovery.
#[derive(Clone, Debug)]
pub struct SampledRecoveryResult {
    /// The whole recovered law `P(X(1), O)` of the sample.
    pub recovered_law: RecoveredLaw,
    /// The recovered effect point `P(y | do(treated)) - P(y | do(control))`.
    pub effect: f64,
    /// Small-cell and support diagnostics.
    pub diagnostics: SampledRecoveryDiagnostics,
    /// The requested interval of the whole-method bootstrap.
    pub interval: SampledEffectInterval,
    /// Effects of the replicates that succeeded, in replicate order.
    pub replicate_effects: Vec<f64>,
    /// Number of replicates that failed and were dropped jointly.
    pub failed_replicates: usize,
    /// Bootstrap standard error of the effect.
    pub effect_standard_error: f64,
    /// Number of recovered-law cells (the side of the covariance matrix).
    pub recovered_cells: usize,
    /// Full bootstrap covariance matrix of the recovered-law cells, row-major
    /// `recovered_cells x recovered_cells`, from the successful replicates.
    pub recovered_cell_covariance: Vec<f64>,
    /// The replayable receipt.
    pub receipt: SampledRecoveryReceipt,
}

impl SampledRecoveryResult {
    /// Bind the internal whole-row candidate to its actual whole-method receipt.
    /// A checked recovery derivation and the complete executed row/replicate scope
    /// remain visible; this is not a public confidence-interval activation.
    #[must_use]
    pub fn calibration_basis(&self) -> antecedent_core::CalibrationBasis {
        let functional = format!(
            "recovered_effect:{}:treated={:016x}:control={:016x}",
            self.recovered_law.scientific_derivation_identity(),
            self.receipt.config.treated_level.to_bits(),
            self.receipt.config.control_level.to_bits()
        );
        antecedent_core::CalibrationBasis::new(
            [
                "RecoveredEffect",
                "MGraph",
                "fixed",
                "tabular",
                "Frequentist",
                "sampled_observation_recovery",
                self.receipt.config.interval_method.name(),
                "",
                "iid",
                "",
                &functional,
            ]
            .map(Arc::from),
            self.interval.level,
            Arc::from("point"),
            u64::try_from(self.receipt.rows).expect("validated bounded row count"),
            Some(
                u32::try_from(self.replicate_effects.len())
                    .expect("validated bounded replicate count"),
            ),
            None,
            0.,
        )
    }

    /// Covariance of recovered-law cells `i` and `j`.
    #[must_use]
    pub fn cell_covariance(&self, i: usize, j: usize) -> Option<f64> {
        if i >= self.recovered_cells || j >= self.recovered_cells {
            return None;
        }
        self.recovered_cell_covariance.get(i * self.recovered_cells + j).copied()
    }
}

/// Decide the exact recovery of a sampled request, mapping every graph-level
/// refusal (a self-censoring edge with a witness, an unsupported mechanism, a
/// missing margin, ...) to a sampled refusal that keeps the exact detail.
pub fn derive_sampled_recovery(
    graph: &Admg,
    query: &ObservationRecoveryQuery,
    catalog: &EvidenceCatalog,
    effect: &RecoveredEffectQuery,
    limits: RecoveryLimits,
    ctx: &ExecutionContext,
) -> Result<Box<RecoveryDerivation>, SampledRecoveryError> {
    match decide_observation_recovery(graph, query, catalog, Some(effect), limits, ctx) {
        Ok(RecoveryDecision::Recovered(derivation)) => Ok(derivation),
        Ok(RecoveryDecision::NonRecoverable(witness)) => Err(SampledRecoveryError {
            detail: SampledRecoveryDetail::UnrecoverablePattern,
            message: format!(
                "the self-censoring edge {:?} has a verified nonrecoverability witness",
                witness.edge
            ),
            patterns: Vec::new(),
            recovery: Some(RecoveryDetail::NonrecoverableWitness),
        }),
        Err(error) => Err(from_recovery(&error)),
    }
}

/// A fingerprint accumulator (two FNV-style lanes); not cryptographic.
struct Digest {
    a: u64,
    b: u64,
}

impl Digest {
    const fn new() -> Self {
        Self { a: 0xcbf2_9ce4_8422_2325, b: 0x8422_2325_cbf2_9ce5 }
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.a = (self.a ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
            self.b = (self.b ^ u64::from(*byte)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            self.b ^= self.b >> 29;
        }
    }
    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }
    fn usize(&mut self, value: usize) {
        self.u64(to_u64(value));
    }
    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }
    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.bytes(value.as_bytes());
    }
    fn pattern(&mut self, pattern: ObservationPattern) {
        self.bytes(&[pattern.responses, pattern.proxies, pattern.fully]);
    }
    fn finish(&self) -> String {
        format!("{:016x}{:016x}", self.a, mix64(self.b ^ self.a.rotate_left(17)))
    }
}

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// Bit mask of the lowest `n` bits.
fn mask(n: usize) -> u8 {
    (0..n).fold(0u8, |acc, i| acc | (1u8 << i))
}

#[derive(Clone, Copy)]
enum Role {
    Response(usize),
    Proxy(usize),
    Full(usize),
}

/// Dense layout of the observed pattern law: axes sorted by variable id, last
/// axis fastest, proxies with levels `[0, 1, ?]`.
struct Layout {
    axes: Vec<DiscreteAxis>,
    roles: Vec<Role>,
    strides: Vec<usize>,
    cards: Vec<usize>,
    cells: usize,
    k: usize,
    m: usize,
    complete: Vec<usize>,
}

impl Layout {
    fn new(query: &ObservationRecoveryQuery) -> Result<Self, SampledRecoveryError> {
        let k = query.partially_observed.len();
        let m = query.fully_observed.len();
        if k + m > SAMPLED_RECOVERY_MAX_BINARY_VARIABLES {
            return Err(refusal(
                SampledRecoveryDetail::BoundsExceeded,
                format!(
                    "{} binary variables; at most {SAMPLED_RECOVERY_MAX_BINARY_VARIABLES}",
                    k + m
                ),
            ));
        }
        let mut axes = Vec::new();
        let mut roles = Vec::new();
        let mut cards = Vec::new();
        for variable in query.observed() {
            let response = query.partially_observed.iter().position(|p| p.response == variable);
            let proxy = query.partially_observed.iter().position(|p| p.proxy == variable);
            let full = query.fully_observed.iter().position(|v| *v == variable);
            let (role, card) = match (response, proxy, full) {
                (Some(i), _, _) => (Role::Response(i), 2),
                (None, Some(i), _) => (Role::Proxy(i), 3),
                (None, None, Some(j)) => (Role::Full(j), 2),
                (None, None, None) => {
                    return Err(refusal(
                        SampledRecoveryDetail::InvalidInput,
                        "an observed coordinate has no role in the query",
                    ));
                }
            };
            let mut values = vec![Value::f64(0.0), Value::f64(1.0)];
            if card == 3 {
                values.push(Value::Label(Arc::from(MISSING_LEVEL)));
            }
            axes.push(DiscreteAxis { variable, values: Arc::from(values) });
            roles.push(role);
            cards.push(card);
        }
        let mut strides = vec![1usize; cards.len()];
        let mut cells = 1usize;
        for (axis, card) in cards.iter().enumerate().rev() {
            strides[axis] = cells;
            cells *= card;
        }
        let mut layout = Self { axes, roles, strides, cards, cells, k, m, complete: Vec::new() };
        let responses = mask(k);
        let mut complete = Vec::new();
        for proxies in 0..(1u8 << k) {
            for fully in 0..(1u8 << m) {
                complete.push(layout.cell(ObservationPattern { responses, proxies, fully }));
            }
        }
        layout.complete = complete;
        Ok(layout)
    }

    fn level(pattern: ObservationPattern, role: Role) -> usize {
        match role {
            Role::Response(i) => usize::from((pattern.responses >> i) & 1),
            Role::Proxy(i) => {
                if (pattern.responses >> i) & 1 == 1 {
                    usize::from((pattern.proxies >> i) & 1)
                } else {
                    2
                }
            }
            Role::Full(j) => usize::from((pattern.fully >> j) & 1),
        }
    }

    fn cell(&self, pattern: ObservationPattern) -> usize {
        self.roles
            .iter()
            .zip(&self.strides)
            .map(|(&role, &stride)| Self::level(pattern, role) * stride)
            .sum()
    }

    fn pattern(&self, cell: usize) -> ObservationPattern {
        let mut pattern = ObservationPattern::default();
        for (axis, role) in self.roles.iter().enumerate() {
            let level = (cell / self.strides[axis]) % self.cards[axis];
            if level != 1 {
                continue;
            }
            match *role {
                Role::Response(i) => pattern.responses |= 1u8 << i,
                Role::Proxy(i) => pattern.proxies |= 1u8 << i,
                Role::Full(j) => pattern.fully |= 1u8 << j,
            }
        }
        pattern
    }

    /// A pattern the proxy model allows: bits in range and a missing proxy
    /// carries no value.
    fn valid(&self, pattern: ObservationPattern) -> bool {
        pattern.responses >> self.k == 0
            && pattern.proxies >> self.k == 0
            && pattern.fully >> self.m == 0
            && pattern.proxies & !pattern.responses == 0
    }

    /// Complete-case patterns with no mass, ascending.
    fn zero_complete_cells(&self, counts: &[u64]) -> Vec<ObservationPattern> {
        let mut zero: Vec<ObservationPattern> = self
            .complete
            .iter()
            .filter(|cell| counts[**cell] == 0)
            .map(|cell| self.pattern(*cell))
            .collect();
        zero.sort_unstable();
        zero
    }
}

fn validate_config(config: &SampledRecoveryConfig) -> Result<(), SampledRecoveryError> {
    let invalid = |message: &str| Err(refusal(SampledRecoveryDetail::InvalidInput, message));
    if config.replicates > SAMPLED_RECOVERY_MAX_REPLICATES {
        return Err(refusal(
            SampledRecoveryDetail::BoundsExceeded,
            format!("{} replicates; at most {SAMPLED_RECOVERY_MAX_REPLICATES}", config.replicates),
        ));
    }
    if config.interval_method == SampledIntervalMethod::Bca
        && (config.replicates != 2000 || config.max_failed_fraction != 0.0)
    {
        return invalid("BCa requires exactly 2000 replicates and zero allowed failed replicates");
    }
    if config.replicates < SAMPLED_RECOVERY_MIN_REPLICATES {
        return invalid("too few replicates to form a 95% BCa interval");
    }
    if !(config.max_failed_fraction.is_finite()
        && (0.0..=0.5).contains(&config.max_failed_fraction))
    {
        return invalid("max_failed_fraction must be in [0, 0.5]");
    }
    if !(config.normalization_tolerance.is_finite()
        && config.normalization_tolerance >= 1e-12
        && config.normalization_tolerance < 1.0)
    {
        return invalid("normalization_tolerance must be in [1e-12, 1)");
    }
    if !config.small_cell_count.is_finite() || config.small_cell_count < 0.0 {
        return invalid("small_cell_count must be finite and nonnegative");
    }
    if !config.treated_level.is_finite()
        || !config.control_level.is_finite()
        || (config.treated_level - config.control_level).abs() < f64::EPSILON
    {
        return invalid("the effect contrast needs two distinct finite levels");
    }
    Ok(())
}

fn effect_variables(
    derivation: &RecoveryDerivation,
) -> Result<(VariableId, VariableId), SampledRecoveryError> {
    let Some(effect) = derivation.effect() else {
        return Err(refusal(
            SampledRecoveryDetail::InvalidInput,
            "the derivation identified no downstream effect",
        ));
    };
    match (effect.treatments(), effect.outcomes()) {
        ([treatment], [outcome]) => Ok((*treatment, *outcome)),
        _ => Err(refusal(
            SampledRecoveryDetail::InvalidInput,
            "the sampled effect contrast needs exactly one treatment and one outcome",
        )),
    }
}

fn check_input(
    derivation: &RecoveryDerivation,
    input: &SampledObservationInput,
    layout: &Layout,
) -> Result<(), SampledRecoveryError> {
    let invalid = |message: String| Err(refusal(SampledRecoveryDetail::InvalidInput, message));
    if input.rows.is_empty() {
        return invalid("no observation rows".to_string());
    }
    if input.rows.len() > SAMPLED_RECOVERY_MAX_ROWS {
        return Err(refusal(
            SampledRecoveryDetail::BoundsExceeded,
            format!("{} rows; at most {SAMPLED_RECOVERY_MAX_ROWS}", input.rows.len()),
        ));
    }
    if input.snapshot_id.is_empty() {
        return invalid("the snapshot id is empty".to_string());
    }
    if let Some(bound) = &derivation.observed().snapshot {
        if bound.as_ref() != input.snapshot_id {
            return invalid(format!(
                "the rows' snapshot {:?} is not the snapshot {:?} the catalog binds",
                input.snapshot_id,
                bound.as_ref()
            ));
        }
    }
    if let Some(row) = input.rows.iter().find(|row| !layout.valid(row.pattern)) {
        return invalid(format!("row {} has a pattern the proxy model excludes", row.id));
    }
    let mut ids: Vec<u64> = input.rows.iter().map(|row| row.id).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|pair| pair[0] == pair[1]) {
        return invalid("row ids are not unique".to_string());
    }
    Ok(())
}

fn input_digest(input: &SampledObservationInput) -> String {
    let mut digest = Digest::new();
    digest.str("sampled_recovery.input.v1");
    digest.str(&input.snapshot_id);
    digest.usize(input.rows.len());
    for row in &input.rows {
        digest.u64(row.id);
        digest.pattern(row.pattern);
    }
    digest.finish()
}

/// Probability that the outcome is 1, normalized by the evaluated total.
fn outcome_one_probability(distribution: &ExactDistribution, outcome: VariableId) -> f64 {
    let Some(axis) = distribution.outcomes.iter().position(|v| *v == outcome) else {
        return f64::NAN;
    };
    let mut one = 0.0;
    let mut total = 0.0;
    for (atom, p) in distribution.atoms.iter().zip(distribution.probabilities.iter()) {
        total += *p;
        if atom.get(axis).and_then(Value::as_f64).is_some_and(|value| value > 0.5) {
            one += *p;
        }
    }
    if total > 0.0 { one / total } else { f64::NAN }
}

struct Replicated {
    record: SampledReplicateRecord,
    values: Option<(f64, Vec<f64>)>,
}

struct Bootstrap {
    records: Vec<SampledReplicateRecord>,
    effects: Vec<f64>,
    cells: Vec<Vec<f64>>,
}

struct Run<'a> {
    derivation: &'a RecoveryDerivation,
    input: &'a SampledObservationInput,
    layout: Layout,
    cells_of_row: Vec<usize>,
    treatment: VariableId,
    outcome: VariableId,
    config: &'a SampledRecoveryConfig,
    ctx: &'a ExecutionContext,
}

impl Run<'_> {
    fn tabulate(&self) -> Vec<u64> {
        let mut counts = vec![0u64; self.layout.cells];
        for cell in &self.cells_of_row {
            counts[*cell] += 1;
        }
        counts
    }

    /// Recover the whole law and the effect from one table of counts.
    fn recover(&self, counts: &[u64]) -> Result<(RecoveredLaw, f64), RecoveryError> {
        let total = counts.iter().sum::<u64>() as f64;
        let probabilities: Vec<f64> = counts.iter().map(|count| *count as f64 / total).collect();
        let query = self.derivation.query();
        let law = ExactDiscreteLaw::try_new(
            Arc::clone(&query.population),
            query.observed_regime,
            Vec::new(),
            self.layout.axes.clone(),
            probabilities,
            self.input.snapshot_id.as_str(),
            LawTolerance { absolute: self.config.normalization_tolerance, relative: 0.0 },
        )
        .map_err(|e| RecoveryError::new(RecoveryDetail::InvalidObservedLaw, e.to_string()))?;
        let recovered = evaluate_exact_recovery(self.derivation, &law, self.ctx)?;
        let treated = self.level_probability(&recovered, self.config.treated_level)?;
        let control = self.level_probability(&recovered, self.config.control_level)?;
        Ok((recovered, treated - control))
    }

    fn level_probability(
        &self,
        recovered: &RecoveredLaw,
        level: f64,
    ) -> Result<f64, RecoveryError> {
        let request = Assignment::from_pairs([(self.treatment, Value::f64(level))]);
        let distribution = evaluate_recovered_effect(
            self.derivation,
            recovered,
            request,
            ExactEvaluationLimits::default(),
            self.ctx,
        )?;
        Ok(outcome_one_probability(&distribution, self.outcome))
    }

    fn diagnostics(&self, counts: &[u64], recovered: &RecoveredLaw) -> SampledRecoveryDiagnostics {
        let rows = self.input.rows.len();
        let mut pattern_counts: Vec<(ObservationPattern, u64)> = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(cell, count)| (self.layout.pattern(cell), *count))
            .collect();
        pattern_counts.sort_unstable();
        let cells = recovered.law().probabilities();
        let total: f64 = cells.iter().sum();
        let min_cell = cells.iter().copied().fold(f64::INFINITY, f64::min);
        SampledRecoveryDiagnostics {
            rows,
            pattern_counts,
            min_complete_case_count: self
                .layout
                .complete
                .iter()
                .map(|cell| counts[*cell])
                .min()
                .unwrap_or(0),
            min_recovered_cell: min_cell,
            min_recovered_cell_count: min_cell * rows as f64,
            small_recovered_cells: cells
                .iter()
                .enumerate()
                .filter(|(_, p)| **p * (rows as f64) < self.config.small_cell_count)
                .map(|(index, _)| index)
                .collect(),
            recovered_total: total,
            normalization_defect: (total - 1.0).abs(),
            normalization_tolerance: self.config.normalization_tolerance,
        }
    }

    /// Draw one row-level bootstrap sample: `n` rows with replacement, every
    /// observed margin of the replicate computed from the same drawn rows.
    fn resample(&self, id: usize) -> (Vec<u64>, String) {
        let rows = &self.input.rows;
        let n = to_u64(rows.len());
        let mut state = seed_mix(self.config.seed ^ mix64(to_u64(id).wrapping_add(1)));
        let mut counts = vec![0u64; self.layout.cells];
        let mut digest = Digest::new();
        digest.str("sampled_recovery.selection.v1");
        digest.u64(n);
        digest.usize(id);
        for _ in 0..rows.len() {
            let scaled = (u128::from(splitmix64(&mut state)) * u128::from(n)) >> 64;
            let draw = usize::try_from(scaled).unwrap_or(0);
            counts[self.cells_of_row[draw]] += 1;
            digest.u64(rows[draw].id);
        }
        (counts, digest.finish())
    }

    fn replicate(&self, id: usize) -> Result<Replicated, SampledRecoveryError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let (counts, selection_digest) = self.resample(id);
        let record_id = u32::try_from(id).unwrap_or(u32::MAX);
        let failed = |failure: String, zero_patterns: Vec<ObservationPattern>| Replicated {
            record: SampledReplicateRecord {
                id: record_id,
                selection_digest: selection_digest.clone(),
                effect: None,
                failure: Some(failure),
                zero_patterns,
            },
            values: None,
        };
        let zero = self.layout.zero_complete_cells(&counts);
        if !zero.is_empty() {
            return Ok(failed("zero_complete_case_cell".to_string(), zero));
        }
        match self.recover(&counts) {
            Ok((recovered, effect)) if effect.is_finite() => Ok(Replicated {
                record: SampledReplicateRecord {
                    id: record_id,
                    selection_digest: selection_digest.clone(),
                    effect: Some(effect),
                    failure: None,
                    zero_patterns: Vec::new(),
                },
                values: Some((effect, recovered.law().probabilities().to_vec())),
            }),
            Ok(_) => Ok(failed("nonfinite_effect".to_string(), Vec::new())),
            Err(error) if error.detail == RecoveryDetail::Budget => Err(cancelled()),
            Err(error) => Ok(failed(error.to_string(), Vec::new())),
        }
    }

    fn jackknife(
        &self,
        counts: &[u64],
    ) -> Result<Vec<SampledJackknifeRecord>, SampledRecoveryError> {
        let mut records = Vec::new();
        let mut reduced = counts.to_vec();
        for (cell, count) in counts.iter().copied().enumerate().filter(|(_, n)| *n > 0) {
            if self.ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            reduced[cell] -= 1;
            let zero = self.layout.zero_complete_cells(&reduced);
            if !zero.is_empty() {
                return Err(unrecoverable(
                    zero,
                    "a delete-one-row table leaves a zero complete-case cell".into(),
                ));
            }
            let (_, effect) = self.recover(&reduced).map_err(|e| from_recovery(&e))?;
            if !effect.is_finite() {
                return Err(unrecoverable(Vec::new(), "nonfinite delete-one-row effect".into()));
            }
            records.push(SampledJackknifeRecord {
                pattern: self.layout.pattern(cell),
                multiplicity: count,
                effect,
            });
            reduced[cell] += 1;
        }
        records.sort_by_key(|r| r.pattern);
        Ok(records)
    }

    fn bootstrap(&self) -> Result<Bootstrap, SampledRecoveryError> {
        let mut out = Bootstrap {
            records: Vec::with_capacity(self.config.replicates),
            effects: Vec::new(),
            cells: Vec::new(),
        };
        for id in 0..self.config.replicates {
            let replicated = self.replicate(id)?;
            out.records.push(replicated.record);
            if let Some((effect, cells)) = replicated.values {
                out.effects.push(effect);
                out.cells.push(cells);
            }
        }
        Ok(out)
    }
}

/// Linear-interpolation percentile (type 7) of a sorted, nonempty sample.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the position is in [0, len - 1]"
)]
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let position = p * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

/// Sample covariance (divisor `n - 1`) of equal-length rows, row-major.
fn covariance(samples: &[Vec<f64>]) -> Vec<f64> {
    let n = samples.len();
    let cells = samples.first().map_or(0, Vec::len);
    let mut means = vec![0.0; cells];
    for sample in samples {
        for (mean, value) in means.iter_mut().zip(sample) {
            *mean += value / n as f64;
        }
    }
    let mut out = vec![0.0; cells * cells];
    for sample in samples {
        for i in 0..cells {
            for j in 0..cells {
                out[i * cells + j] += (sample[i] - means[i]) * (sample[j] - means[j]);
            }
        }
    }
    let divisor = (n.max(2) - 1) as f64;
    out.iter().map(|value| value / divisor).collect()
}

/// Efron (1987) `BCa` transformation. This is an unmeasured candidate, not a
/// finite-sample coverage guarantee. Failures and unresolved tails are refused.
#[allow(
    clippy::float_cmp,
    reason = "the frozen bias correction uses exact floating-point ties, not a tolerance"
)]
fn bca_receipt(
    point: f64,
    samples: &[f64],
    jackknife: Vec<SampledJackknifeRecord>,
) -> Result<SampledBcaReceipt, SampledRecoveryError> {
    let unsupported = |message| refusal(SampledRecoveryDetail::UnrecoverablePattern, message);
    let n: u64 = jackknife.iter().map(|r| r.multiplicity).sum();
    let mean = jackknife.iter().map(|r| r.effect * r.multiplicity as f64).sum::<f64>() / n as f64;
    let squared =
        jackknife.iter().map(|r| (mean - r.effect).powi(2) * r.multiplicity as f64).sum::<f64>();
    let cubed =
        jackknife.iter().map(|r| (mean - r.effect).powi(3) * r.multiplicity as f64).sum::<f64>();
    if !squared.is_finite() || squared <= 0.0 {
        return Err(unsupported("degenerate BCa jackknife"));
    }
    let acceleration = cubed / (6.0 * squared.powf(1.5));
    let below = samples.iter().filter(|x| **x < point).count();
    let ties = samples.iter().filter(|x| **x == point).count();
    let rank = (below as f64 + 0.5 * ties as f64) / samples.len() as f64;
    if !(0.0..1.0).contains(&rank) || rank == 0.0 {
        return Err(unsupported("BCa bias rank is on a boundary"));
    }
    let bias_correction = norm_inv(rank);
    let mut adjusted_probabilities = [0.0; 2];
    for (index, tail) in [0.025, 0.975].into_iter().enumerate() {
        let z = bias_correction + norm_inv(tail);
        let denominator = 1.0 - acceleration * z;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(unsupported("BCa transformation reaches a pole"));
        }
        let p = norm_cdf(bias_correction + z / denominator);
        let resolution = 1.0 / (samples.len() + 1) as f64;
        if !p.is_finite() || !(resolution..=1.0 - resolution).contains(&p) {
            return Err(unsupported(
                "BCa adjusted tail is unresolved by the frozen bootstrap budget",
            ));
        }
        adjusted_probabilities[index] = p;
    }
    if !acceleration.is_finite() || adjusted_probabilities[0] >= adjusted_probabilities[1] {
        return Err(unsupported("invalid BCa adjusted probabilities"));
    }
    Ok(SampledBcaReceipt { bias_correction, acceleration, adjusted_probabilities, jackknife })
}

fn too_many_failed(records: &[SampledReplicateRecord], failed: usize) -> SampledRecoveryError {
    let patterns: BTreeSet<ObservationPattern> =
        records.iter().flat_map(|r| r.zero_patterns.iter().copied()).collect();
    SampledRecoveryError {
        detail: SampledRecoveryDetail::TooManyFailedReplicates,
        message: format!(
            "{failed} of {} bootstrap replicates failed jointly (a zero complete-case cell or \
             a refused recovery), above the allowed fraction",
            records.len()
        ),
        patterns: patterns.into_iter().collect(),
        recovery: None,
    }
}

/// Estimate the whole recovered law and effect from observation-pattern rows and
/// form the requested whole-method interval.
///
/// # Errors
///
/// `sampled_recovery.unrecoverable_pattern` for a zero complete-case cell (the
/// offending patterns are retained) or a refused exact recovery,
/// `sampled_recovery.too_many_failed_replicates`,
/// `sampled_recovery.bounds_exceeded`, `sampled_recovery.invalid_input` and
/// `sampled_recovery.cancelled`.
pub fn estimate_sampled_recovery(
    derivation: &RecoveryDerivation,
    input: &SampledObservationInput,
    config: &SampledRecoveryConfig,
    ctx: &ExecutionContext,
) -> Result<SampledRecoveryResult, SampledRecoveryError> {
    validate_config(config)?;
    let (treatment, outcome) = effect_variables(derivation)?;
    let layout = Layout::new(derivation.query())?;
    check_input(derivation, input, &layout)?;
    let cells_of_row = input.rows.iter().map(|row| layout.cell(row.pattern)).collect();
    let run = Run { derivation, input, layout, cells_of_row, treatment, outcome, config, ctx };
    let counts = run.tabulate();
    let zero = run.layout.zero_complete_cells(&counts);
    if !zero.is_empty() {
        let message = format!(
            "{} complete-case pattern(s) have zero count: the recovery formula divides by a \
             margin that contains them",
            zero.len()
        );
        return Err(unrecoverable(zero, message));
    }
    let (recovered_law, effect) = run.recover(&counts).map_err(|e| from_recovery(&e))?;
    if !effect.is_finite() {
        return Err(unrecoverable(Vec::new(), "the recovered effect is not finite".to_string()));
    }
    let diagnostics = run.diagnostics(&counts, &recovered_law);
    // Validate every delete-one domain before spending the bootstrap budget.
    let jackknife = run.jackknife(&counts)?;
    let boot = run.bootstrap()?;
    let failed = boot.records.len() - boot.effects.len();
    if boot.effects.len() < 2
        || failed as f64 / boot.records.len() as f64 > config.max_failed_fraction
    {
        return Err(too_many_failed(&boot.records, failed));
    }
    let mut sorted = boot.effects.clone();
    sorted.sort_by(f64::total_cmp);
    let corrected = bca_receipt(effect, &boot.effects, jackknife)?;
    let probabilities = corrected.adjusted_probabilities;
    let bca = Some(corrected);
    let interval = SampledEffectInterval {
        level: SAMPLED_RECOVERY_INTERVAL_LEVEL,
        lower: percentile(&sorted, probabilities[0]),
        upper: percentile(&sorted, probabilities[1]),
        calibration: SAMPLED_RECOVERY_CALIBRATION,
    };
    let used = boot.effects.len() as f64;
    let mean = boot.effects.iter().sum::<f64>() / used;
    let variance = boot.effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (used - 1.0);
    let mut receipt = SampledRecoveryReceipt {
        snapshot_id: input.snapshot_id.clone(),
        rows: input.rows.len(),
        derivation_identity: derivation.identity(),
        input_digest: input_digest(input),
        config: *config,
        point_effect: effect,
        interval,
        replicates: boot.records,
        bca,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = receipt.recompute_digest();
    Ok(SampledRecoveryResult {
        recovered_cells: recovered_law.law().probabilities().len(),
        recovered_cell_covariance: covariance(&boot.cells),
        recovered_law,
        effect,
        diagnostics,
        interval,
        replicate_effects: boot.effects,
        failed_replicates: failed,
        effect_standard_error: variance.sqrt(),
        receipt,
    })
}

/// Rerun a stored sampled recovery from its inputs and refuse unless the rerun
/// reproduces the stored receipt digest bit for bit. The configuration (seed
/// included) is the receipt's own.
///
/// # Errors
///
/// `sampled_recovery.receipt_mismatch` when the stored digest does not recompute
/// or the rerun differs; any refusal of [`estimate_sampled_recovery`].
pub fn replay_sampled_recovery(
    derivation: &RecoveryDerivation,
    input: &SampledObservationInput,
    expected: &SampledRecoveryReceipt,
    ctx: &ExecutionContext,
) -> Result<SampledRecoveryResult, SampledRecoveryError> {
    if (expected.config.interval_method == SampledIntervalMethod::Bca) != expected.bca.is_some() {
        return Err(refusal(
            SampledRecoveryDetail::ReceiptMismatch,
            "interval method and BCa receipt disagree",
        ));
    }
    if !expected.verify_digest() {
        return Err(refusal(
            SampledRecoveryDetail::ReceiptMismatch,
            "the stored receipt digest does not recompute from its fields",
        ));
    }
    let result = estimate_sampled_recovery(derivation, input, &expected.config, ctx)?;
    // Legacy v1 receipt fingerprints did not include failure text. Preserve their
    // digest format, but compare every replayed field as well.
    if result.receipt != *expected {
        return Err(refusal(
            SampledRecoveryDetail::ReceiptMismatch,
            "the rerun does not reproduce the stored receipt (changed pattern, snapshot, \
             derivation, seed or interval)",
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod bca_tests {
    use super::*;

    fn jack(values: &[f64]) -> Vec<SampledJackknifeRecord> {
        values
            .iter()
            .enumerate()
            .map(|(i, effect)| SampledJackknifeRecord {
                pattern: ObservationPattern {
                    responses: 0,
                    proxies: 0,
                    fully: u8::try_from(i).unwrap(),
                },
                multiplicity: 1,
                effect: *effect,
            })
            .collect()
    }

    #[test]
    fn bca_midrank_ties_and_domain_refusals_are_explicit() {
        let samples: Vec<f64> = (0..2000).map(|i| f64::from(i % 3) - 1.0).collect();
        let bca = bca_receipt(0.0, &samples, jack(&[-1.0, 0.0, 1.0])).unwrap();
        // 667 below + half of 667 tied; never silently strict-less z0.
        assert!((bca.bias_correction - norm_inv((667.0 + 0.5 * 667.0) / 2000.0)).abs() < 1e-12);
        assert!(bca.acceleration.abs() < f64::EPSILON);
        assert!(bca_receipt(0.0, &samples, jack(&[1.0, 1.0])).is_err());
        assert!(bca_receipt(-2.0, &samples, jack(&[-1.0, 0.0, 1.0])).is_err());
        assert!(bca_receipt(2.0, &samples, jack(&[-1.0, 0.0, 1.0])).is_err());
        // A resolvable bias rank alone does not ensure adjusted tails resolve.
        let tail: Vec<f64> = (0..2000).map(f64::from).collect();
        assert!(bca_receipt(1.5, &tail, jack(&[-1.0, 0.0, 1.0])).is_err());
    }
}
