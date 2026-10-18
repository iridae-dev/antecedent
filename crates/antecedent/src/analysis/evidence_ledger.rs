//! The evidence ledger as a facade object: sources, results and operations,
//! with the overlap check ahead of every number.
//!
//! Evidence that re-enters an analysis is not new evidence. An
//! [`EvidenceLedgerSession`] holds the sources (observation sets, experiments,
//! priors, fitted models, external claims) with their row-level observation
//! identities and derivation lineage, the results registered over them, and
//! the operations that changed them. It routes each request through the
//! layers that own the rule and adds no scientific rule of its own:
//!
//! * **Overlap first.** Every combination ([`pool_summaries`],
//!   [`random_effects`], [`hierarchical_borrowing`], [`dependence_range`],
//!   [`transport_claim`], [`reuse_result`], [`repeat_observations`]) is
//!   registered with the ledger *before* any number is computed. The ledger
//!   refuses shared or unknown overlap declared independent, duplicate and
//!   re-entered evidence, and a retired source, with a typed reason and the
//!   declaration that would license the request.
//! * **Then the numbers.** `antecedent_stats::synthesis` computes the pooled
//!   value from the study summaries: inverse-variance or generalized least
//!   squares pooling, DerSimonian-Laird or restricted-maximum-likelihood
//!   random effects, hierarchical borrowing with a declared finite prior on
//!   the between-study standard deviation, or the exact envelope over a
//!   declared correlation range. Everything is point-only; the envelope is
//!   labelled an assumption range. No interval, posterior or coverage
//!   statement is made ([`pooled_interval`] refuses).
//! * **Atomic.** A combination is staged on a copy of the ledger and
//!   committed only when the numbers also succeed; a refused request leaves
//!   the session exactly as it was.
//! * **No stale use.** After [`revise_source`] retires a source, results that
//!   rested on it are re-classified `still_valid`, `recompute` or `withdrawn`;
//!   [`EvidenceLedgerSession::synthesis`] refuses a result that is no longer
//!   valid, and a recomputation names the result it replaces.
//!
//! # Lifecycle
//!
//! [`EvidenceLedgerSession::refresh`] applies a batch of added and retired
//! evidence to a copy of the session. It **keeps** every result whose sources
//! are untouched (`still_valid`, with its stored numbers) and the operation
//! history, chaining the new operations onto the head identity. It
//! **recomputes** nothing on its own: a result whose input changed is
//! classified `recompute` and is *refused* for use until a new combination
//! over the replacing sources is registered with `replaces` set; a result
//! whose input it cannot do without is `withdrawn` and stays unusable. The
//! batch is all-or-nothing and the session it was called on is untouched
//! whether the refresh succeeds or refuses.
//!
//! # Artifact
//!
//! [`EvidenceLedgerSession::export`] writes `evidence_ledger_v1`: every
//! operation with its declaration, identity and effect, the head, the state
//! digest and the syntheses bound to ledger results.
//! [`consume_evidence_ledger_artifact`] replays every operation through the
//! ledger and recomputes the syntheses, refusing a stored identity, effect,
//! value, covariance or range that does not replay, including a consistently
//! resealed edit when the consumer retained the identity independently. A
//! hierarchical borrowing over prior sources is not carried numerically (the
//! stored synthesis holds one summary per source); the export lists it in
//! [`LedgerExport::not_carried`] rather than dropping it silently.
//!
//! [`pool_summaries`]: EvidenceLedgerSession::pool_summaries
//! [`random_effects`]: EvidenceLedgerSession::random_effects
//! [`hierarchical_borrowing`]: EvidenceLedgerSession::hierarchical_borrowing
//! [`dependence_range`]: EvidenceLedgerSession::dependence_range
//! [`transport_claim`]: EvidenceLedgerSession::transport_claim
//! [`reuse_result`]: EvidenceLedgerSession::reuse_result
//! [`repeat_observations`]: EvidenceLedgerSession::repeat_observations
//! [`pooled_interval`]: EvidenceLedgerSession::pooled_interval
//! [`revise_source`]: EvidenceLedgerSession::revise_source
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, LedgerError, LedgerOperation, MAX_COMBINATION_SOURCES,
    OperationId, PoolingDependence, SynthesisOperation,
};
use antecedent_core::ExecutionContext;
use antecedent_io::IoError;
use antecedent_io::evidence_ledger_artifact::{
    ComputedSynthesis, EvidenceLedgerArtifact, EvidenceLedgerArtifactError, SynthesisMethod,
    SynthesisRecord, covariance_digest, covariance_has_cross_terms,
};
use antecedent_stats::synthesis::{
    DependenceEnvelope, GaussianStudyEstimate, HierarchicalPosterior, MeanPrior, PooledEstimate,
    RandomEffectsEstimate, SynthesisError, TauPrior, equal_study_gaussian, fixed_effect_gls,
    fixed_effect_inverse_variance, hierarchical_normal_borrowing, pooled_dependence_envelope,
    random_effects_dersimonian_laird, random_effects_reml,
};

pub use antecedent_contract::evidence_ledger::{
    AppliedOperation, ClaimClass, DetectedOverlap, EvidenceSource, Lineage, MassAccount,
    ObservationBasis, ObservationId, Overlap, RelationKind, ResultId, ResultRecord,
    ResultRevalidation, ResultStatus, Revalidation, SourceDeclaration, SourceId, SourceKind,
    SourceStatus, SynthesisKind, TransportIdentity,
};

use super::StudyBuilder;

/// One refusal as plain data: registered reason code, namespaced
/// `evidence_ledger.*` detail and the explanation of what was refused and
/// what would license it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceLedgerRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `evidence_ledger.<rule>`.
    pub detail: String,
    /// What was refused and what would license it.
    pub message: String,
    /// Sources the refusal names.
    pub sources: Vec<SourceId>,
}

impl EvidenceLedgerRefusal {
    fn new(code: &'static str, detail: &str, message: impl Into<String>) -> Self {
        Self { code, detail: detail.to_owned(), message: message.into(), sources: Vec::new() }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(
            antecedent_core::reason_code!("invalid_argument"),
            "evidence_ledger.invalid_declaration",
            message,
        )
    }

    fn cancelled() -> Self {
        Self::new(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "evidence_ledger.cancelled",
            "cancelled before a synthesis was formed; the ledger is unchanged",
        )
    }

    fn from_ledger(error: &LedgerError) -> Self {
        Self {
            code: error.code,
            detail: error.detail.to_owned(),
            message: error.message.clone(),
            sources: error.sources.clone(),
        }
    }

    fn from_synthesis(error: &SynthesisError) -> Self {
        Self::invalid(error.to_string())
    }

    fn from_artifact(error: &EvidenceLedgerArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, message)) => Self::new(code, detail, message),
            None => Self::new(
                antecedent_core::reason_code!("invalid_argument"),
                "evidence_ledger.replay_mismatch",
                error.to_string(),
            ),
        }
    }
}

impl std::fmt::Display for EvidenceLedgerRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for EvidenceLedgerRefusal {}

impl From<EvidenceLedgerRefusal> for IoError {
    fn from(refusal: EvidenceLedgerRefusal) -> Self {
        Self::Refused {
            code: refusal.code,
            message: format!("{}: {}", refusal.detail, refusal.message),
        }
    }
}

impl From<EvidenceLedgerRefusal> for crate::CausalError {
    fn from(refusal: EvidenceLedgerRefusal) -> Self {
        Self::Serialization(refusal.into())
    }
}

/// A study's summary: the estimate and its sampling variance, bound to the
/// ledger source it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudySummary {
    /// The ledger source of the study.
    pub source: SourceId,
    /// The study's estimate.
    pub estimate: f64,
    /// The estimate's sampling variance.
    pub variance: f64,
}

/// A study's estimate with its standard error, for a dependence range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudyEstimate {
    /// The ledger source of the study.
    pub source: SourceId,
    /// The study's estimate.
    pub estimate: f64,
    /// The estimate's standard error.
    pub standard_error: f64,
}

/// How the dependence between pooled summaries is accounted for.
#[derive(Clone, Debug, PartialEq)]
pub enum PoolingDeclaration {
    /// Pairwise independent: every pair known and disjoint in the ledger.
    Independent,
    /// A full covariance of the summaries, row-major in the order of the
    /// summaries, preserved by generalized least squares. Licensed when it
    /// has cross terms (shared or unknown overlap) or the sources are known
    /// and disjoint.
    DeclaredCovariance {
        /// Row-major `k x k` covariance.
        covariance: Vec<f64>,
    },
}

/// The between-study variance estimator of a random-effects synthesis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BetweenStudy {
    /// DerSimonian-Laird moments.
    DerSimonianLaird,
    /// Restricted maximum likelihood.
    RestrictedMaximumLikelihood,
}

/// The computed value of a synthesis. Every variant is point-only; the
/// envelope is an assumption range.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SynthesisValue {
    /// Equal-study Gaussian mean; its empirical study-level variance supports
    /// conservative Student intervals under independent heterogeneous study variances.
    EqualStudyGaussian(GaussianStudyEstimate),
    /// Fixed-effect pooled estimate (inverse variance or generalized least squares).
    FixedEffect(PooledEstimate),
    /// Random-effects pooled estimate with heterogeneity.
    RandomEffects(RandomEffectsEstimate),
    /// Hierarchical normal-normal borrowing.
    Hierarchical(HierarchicalPosterior),
    /// Pooled estimate and variance extremes over a correlation range.
    DependenceEnvelope(DependenceEnvelope),
}

impl SynthesisValue {
    /// What the value licenses: `point_only` or `assumption_range`.
    #[must_use]
    pub const fn claim_label(&self) -> &'static str {
        match self {
            Self::DependenceEnvelope(_) => "assumption_range",
            _ => "point_only",
        }
    }
}

/// A registered combination and its numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisOutcome {
    /// The ledger result.
    pub result: ResultId,
    /// The computed value.
    pub value: SynthesisValue,
    /// Observation mass of the result, each observation counted once.
    pub mass: MassAccount,
    /// What the result licenses.
    pub claim_class: ClaimClass,
}

/// A request to pool the summaries of two or more studies.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolingRequest {
    /// Human label; part of the result identity.
    pub label: String,
    /// Study summaries, in the order of any declared covariance.
    pub summaries: Vec<StudySummary>,
    /// A result this one recomputes; it becomes `Replaced`.
    pub replaces: Option<ResultId>,
}

/// A request to borrow strength across exchangeable studies.
#[derive(Clone, Debug, PartialEq)]
pub struct BorrowingRequest {
    /// Human label; part of the result identity.
    pub label: String,
    /// Prior sources whose data origin must be disjoint from every
    /// likelihood source and from each other. Accounted in the ledger; the
    /// numeric prior is `mean_prior` and `tau_prior`.
    pub priors: Vec<SourceId>,
    /// Likelihood-bearing study summaries, pairwise independent.
    pub likelihood: Vec<StudySummary>,
    /// Prior on the common mean.
    pub mean_prior: MeanPrior,
    /// Declared finite prior on the between-study standard deviation.
    pub tau_prior: TauPrior,
    /// A result this one recomputes.
    pub replaces: Option<ResultId>,
}

/// A request for the pooled estimate over a declared correlation range.
#[derive(Clone, Debug, PartialEq)]
pub struct DependenceRangeRequest {
    /// Human label; part of the result identity.
    pub label: String,
    /// Study estimates with standard errors.
    pub studies: Vec<StudyEstimate>,
    /// Lower bound of the common pairwise correlation.
    pub rho_lo: f64,
    /// Upper bound of the common pairwise correlation.
    pub rho_hi: f64,
    /// A result this one recomputes.
    pub replaces: Option<ResultId>,
}

/// A revision of one source.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceRevision {
    /// The source grew: the replacement holds every earlier observation.
    Update {
        /// Source being updated.
        target: SourceId,
        /// Grown content.
        replacement: SourceDeclaration,
    },
    /// The source's content is swapped (a correction).
    Replace {
        /// Source being replaced.
        target: SourceId,
        /// Corrected content.
        replacement: SourceDeclaration,
    },
    /// A newer fitted model or external claim formally replaces an older one.
    Supersede {
        /// Source being superseded.
        target: SourceId,
        /// The newer model or claim.
        replacement: SourceDeclaration,
    },
    /// Retract a source.
    Withdraw {
        /// Source being withdrawn.
        target: SourceId,
        /// Why.
        reason: String,
    },
}

/// One change of a refresh batch.
#[derive(Clone, Debug, PartialEq)]
pub enum SourceChange {
    /// Add new evidence.
    Add(SourceDeclaration),
    /// Revise or retire existing evidence.
    Revise(SourceRevision),
}

/// What one applied operation did downstream.
#[derive(Clone, Debug, PartialEq)]
pub struct RevisionReport {
    /// Identity of the operation, chained to the previous one.
    pub operation: OperationId,
    /// Source created by the operation.
    pub created_source: Option<SourceId>,
    /// Source retired by the operation.
    pub retired_source: Option<SourceId>,
    /// Overlaps with earlier sources detected at entry (informational).
    pub detected_overlaps: Vec<DetectedOverlap>,
    /// Classification of every result after the operation, sorted by id.
    pub revalidation: Vec<ResultRevalidation>,
}

impl RevisionReport {
    fn of(applied: &AppliedOperation) -> Self {
        Self {
            operation: applied.id,
            created_source: applied.effect.created_source,
            retired_source: applied.effect.retired_source,
            detected_overlaps: applied.effect.detected_overlaps.clone(),
            revalidation: applied.effect.revalidation.clone(),
        }
    }
}

/// The result of a refresh batch.
#[derive(Clone, Debug)]
pub struct LedgerRefresh {
    /// The refreshed session; the one refreshed from is unchanged.
    pub session: EvidenceLedgerSession,
    /// One report per applied change, in order.
    pub reports: Vec<RevisionReport>,
    /// Classification of every result after the batch, sorted by id.
    pub classification: Vec<ResultRevalidation>,
}

/// The accounting of a set of sources: the mass counted once and how every
/// pair relates.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationAccount {
    /// Declared slots, distinct observations and unknown sources.
    pub mass: MassAccount,
    /// Every unordered pair of the requested sources, in request order.
    pub pairs: Vec<SourcePairRelation>,
}

/// How two sources overlap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourcePairRelation {
    /// First source.
    pub first: SourceId,
    /// Second source.
    pub second: SourceId,
    /// Independent, shared (with the count) or unknown.
    pub overlap: Overlap,
    /// The relation in the composition vocabulary.
    pub relation: RelationKind,
}

/// An exported ledger.
#[derive(Clone, Debug, PartialEq)]
pub struct LedgerExport {
    /// The artifact bytes.
    pub bytes: Vec<u8>,
    /// BLAKE3 identity (hex) of the artifact; a consumer that retains it
    /// independently refuses a consistently resealed ledger of another
    /// history.
    pub identity: String,
    /// Results whose synthesis is carried and recomputed by a consumer.
    pub carried: Vec<ResultId>,
    /// Results with numbers in the session that the artifact format cannot
    /// carry, with the reason. The ledger accounting of these results is
    /// carried; their numbers are not.
    pub not_carried: Vec<(ResultId, &'static str)>,
}

/// Nominal level of the measured pooled interval constructions.
const POOLED_INTERVAL_LEVEL: f64 = 0.95;

/// Study counts of the measured pooled-interval family, inclusive. Coverage
/// records are keyed by this range, not by an individual count: a count inside
/// it is licensed only because every count the measurement visited passed.
const MEASURED_STUDY_COUNTS: std::ops::RangeInclusive<usize> = 5..=30;

/// A pooled interval with the measured coverage record that governs it.
#[derive(Clone, Debug, PartialEq)]
pub struct PooledInterval {
    /// The ledger result.
    pub result: ResultId,
    /// `wald_z` or `hksj_t`.
    pub method: &'static str,
    /// Pooled mean.
    pub estimate: f64,
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// Readable measured report; its scalar carries the calibrated slot with
    /// the governing coverage record.
    pub report: antecedent_io::measured_inference::MeasuredInferenceReport,
}

/// A pooled interval construction before a coverage record is consulted.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct PooledIntervalCandidate {
    /// `wald_z` or `hksj_t`.
    pub method: &'static str,
    /// Pooled mean.
    pub estimate: f64,
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// Construction and scope a coverage record is matched against.
    pub basis: antecedent_core::CalibrationBasis,
    /// The same construction under the key of records measured at one exact study
    /// count with the total observations as scope. Such a record licenses exactly
    /// its count and sample range; new measurements use [`Self::basis`].
    pub exact_count_basis: antecedent_core::CalibrationBasis,
    /// Checked facts and declared assumptions of the construction.
    pub scope: serde_json::Value,
}

/// The evidence ledger with the numbers computed over it.
#[derive(Clone, Debug, Default)]
pub struct EvidenceLedgerSession {
    ledger: EvidenceLedger,
    values: BTreeMap<ResultId, SynthesisValue>,
    records: BTreeMap<ResultId, SynthesisRecord>,
    not_carried: BTreeMap<ResultId, &'static str>,
}

const NOT_CARRIED_PRIOR: &str = "hierarchical borrowing over prior sources: the stored synthesis carries one study summary per \
     source and a prior source has none";

impl StudyBuilder {
    /// An empty evidence ledger session.
    #[must_use]
    pub fn evidence_ledger() -> EvidenceLedgerSession {
        EvidenceLedgerSession::default()
    }
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), EvidenceLedgerRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(EvidenceLedgerRefusal::cancelled());
    }
    Ok(())
}

fn too_many(count: usize) -> Result<(), EvidenceLedgerRefusal> {
    if count > MAX_COMBINATION_SOURCES {
        return Err(EvidenceLedgerRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            "evidence_ledger.limit_exceeded",
            format!(
                "{count} sources in one request; at most {MAX_COMBINATION_SOURCES} are combined at once"
            ),
        ));
    }
    Ok(())
}

/// Source ids, estimates and variances of a list of study summaries, in input order.
type StudyColumns = (Vec<SourceId>, Vec<f64>, Vec<f64>);

impl EvidenceLedgerSession {
    /// The ledger: sources, results, operations, head and state digest.
    #[must_use]
    pub const fn ledger(&self) -> &EvidenceLedger {
        &self.ledger
    }

    /// Identity of the last applied operation, as lowercase hex.
    #[must_use]
    pub fn head_hex(&self) -> String {
        self.ledger.head().to_hex()
    }

    /// Digest of the whole ledger state, as lowercase hex.
    #[must_use]
    pub fn state_digest_hex(&self) -> String {
        self.ledger.state_digest().to_hex()
    }

    /// Add a source. A refusal leaves the session unchanged.
    ///
    /// # Errors
    /// A malformed declaration, a repeated source or row, or a lineage that
    /// disagrees with the declared rows.
    pub fn add_source(
        &mut self,
        source: SourceDeclaration,
    ) -> Result<SourceId, EvidenceLedgerRefusal> {
        self.ledger.add_source(source).map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))
    }

    /// Register a provider callback and its declared evidence as a ledger
    /// source (see
    /// [`register_callback_source`](super::provider_callbacks::register_callback_source)).
    /// A refusal leaves the session unchanged.
    ///
    /// # Errors
    /// An invalid declaration, an inner callback that is not registered, or a
    /// ledger refusal.
    #[allow(
        clippy::result_large_err,
        reason = "the registration refusal is the shared callback declaration error"
    )]
    pub fn register_callback(
        &mut self,
        declaration: &super::provider_callbacks::CallbackDeclaration,
        inner_sources: &BTreeMap<String, SourceId>,
    ) -> Result<SourceId, super::provider_callbacks::CallbackRegistrationError> {
        super::provider_callbacks::register_callback_source(
            &mut self.ledger,
            declaration,
            inner_sources,
        )
    }

    /// Supersede a registered callback with its replacement (see
    /// [`supersede_callback_source`](super::provider_callbacks::supersede_callback_source));
    /// results that rest on the old source are revalidated by the ledger.
    ///
    /// # Errors
    /// As [`Self::register_callback`].
    #[allow(
        clippy::result_large_err,
        reason = "the registration refusal is the shared callback declaration error"
    )]
    pub fn supersede_callback(
        &mut self,
        old: SourceId,
        replacement: &super::provider_callbacks::CallbackDeclaration,
        inner_sources: &BTreeMap<String, SourceId>,
    ) -> Result<SourceId, super::provider_callbacks::CallbackRegistrationError> {
        super::provider_callbacks::supersede_callback_source(
            &mut self.ledger,
            old,
            replacement,
            inner_sources,
        )
    }

    /// A declaration for a fitted model or external claim derived from a valid
    /// result, so that a later withdrawal of the result's sources reaches it
    /// through lineage.
    ///
    /// # Errors
    /// An unknown or no-longer-valid result, or a kind that is not a model or
    /// claim.
    pub fn declare_from_result(
        &self,
        result: ResultId,
        label: &str,
        kind: SourceKind,
    ) -> Result<SourceDeclaration, EvidenceLedgerRefusal> {
        self.ledger
            .declaration_from_result(result, label, kind)
            .map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))
    }

    /// The mass the sources contribute, each observation counted once, and how
    /// every pair overlaps. Overlap that is not declared is `unknown`, never
    /// independent.
    ///
    /// # Errors
    /// An unknown source or a request above the combination bound.
    pub fn account_observations(
        &self,
        sources: &[SourceId],
    ) -> Result<ObservationAccount, EvidenceLedgerRefusal> {
        too_many(sources.len())?;
        let mass = self
            .ledger
            .mass_account(sources)
            .map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
        let mut pairs = Vec::new();
        for (i, &first) in sources.iter().enumerate() {
            for &second in &sources[i + 1..] {
                let overlap = self
                    .ledger
                    .overlap(first, second)
                    .map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
                let relation = self
                    .ledger
                    .relation(first, second)
                    .map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
                pairs.push(SourcePairRelation { first, second, overlap, relation });
            }
        }
        Ok(ObservationAccount { mass, pairs })
    }

    /// Register `request` on a staged copy (the overlap check), then compute
    /// the numbers; commit only when both succeed.
    fn combine_numeric(
        &mut self,
        request: CombinationRequest,
        study_sources: Vec<SourceId>,
        compute: impl FnOnce() -> Result<(SynthesisValue, SynthesisMethod), SynthesisError>,
        carry: Option<&'static str>,
        ctx: &ExecutionContext,
    ) -> Result<SynthesisOutcome, EvidenceLedgerRefusal> {
        check_cancelled(ctx)?;
        let mut staged = self.ledger.clone();
        let result = staged.combine(request).map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
        check_cancelled(ctx)?;
        let (value, method) = compute().map_err(|e| EvidenceLedgerRefusal::from_synthesis(&e))?;
        let record = staged.result(result).ok_or_else(|| {
            EvidenceLedgerRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "evidence_ledger.unknown_result",
                "the registered result is not in the staged ledger",
            )
        })?;
        let outcome = SynthesisOutcome {
            result,
            value: value.clone(),
            mass: record.mass.clone(),
            claim_class: record.claim_class,
        };
        // Commit: nothing above has touched the session.
        self.ledger = staged;
        self.values.insert(result, value);
        match carry {
            None => {
                self.records
                    .insert(result, SynthesisRecord { result, sources: study_sources, method });
            }
            Some(reason) => {
                self.not_carried.insert(result, reason);
            }
        }
        Ok(outcome)
    }

    fn studies(summaries: &[StudySummary]) -> Result<StudyColumns, EvidenceLedgerRefusal> {
        too_many(summaries.len())?;
        Ok((
            summaries.iter().map(|s| s.source).collect(),
            summaries.iter().map(|s| s.estimate).collect(),
            summaries.iter().map(|s| s.variance).collect(),
        ))
    }

    /// Pool the summaries of two or more studies.
    ///
    /// The ledger decides first: independent pooling needs every pair known
    /// and disjoint; a declared covariance is licensed when it has cross terms
    /// or the sources are known and disjoint. Then the pooled value is
    /// `1' S^-1 y / 1' S^-1 1` with variance `1 / 1' S^-1 1` (inverse variance
    /// for independent studies, generalized least squares for a declared
    /// covariance). Point-only.
    ///
    /// # Errors
    /// `evidence_double_counted`, `evidence_overlap_unknown`,
    /// `evidence_source_withdrawn`, an invalid declaration or covariance, or
    /// cancellation. The session is unchanged on refusal.
    pub fn pool_summaries(
        &mut self,
        request: PoolingRequest,
        dependence: PoolingDeclaration,
        ctx: &ExecutionContext,
    ) -> Result<SynthesisOutcome, EvidenceLedgerRefusal> {
        let (sources, estimates, variances) = Self::studies(&request.summaries)?;
        let k = sources.len();
        let (ledger_dependence, covariance) = match dependence {
            PoolingDeclaration::Independent => (PoolingDependence::Independent, None),
            PoolingDeclaration::DeclaredCovariance { covariance } => {
                if covariance.len() != k * k {
                    return Err(EvidenceLedgerRefusal::invalid(format!(
                        "a covariance for {k} studies has {} entries, not {}",
                        covariance.len(),
                        k * k
                    )));
                }
                (
                    PoolingDependence::DeclaredCovariance {
                        covariance: covariance_digest(k, &covariance),
                        has_cross_covariance: covariance_has_cross_terms(k, &covariance),
                    },
                    Some(covariance),
                )
            }
        };
        let combination = CombinationRequest {
            label: request.label,
            operation: SynthesisOperation::StatisticalPooling {
                sources: sources.clone(),
                dependence: ledger_dependence,
            },
            replaces: request.replaces,
        };
        self.combine_numeric(
            combination,
            sources,
            move || match covariance {
                None => {
                    let fit = fixed_effect_inverse_variance(&estimates, &variances)?;
                    Ok((
                        SynthesisValue::FixedEffect(fit),
                        SynthesisMethod::FixedEffectInverseVariance { estimates, variances },
                    ))
                }
                Some(covariance) => {
                    let fit = fixed_effect_gls(&estimates, &covariance)?;
                    Ok((
                        SynthesisValue::FixedEffect(fit),
                        SynthesisMethod::FixedEffectGls { estimates, covariance },
                    ))
                }
            },
            None,
            ctx,
        )
    }

    /// Pool independent Gaussian study estimates with equal study weights.
    ///
    /// Every estimate has the same marginal mean; its unknown variance may
    /// differ arbitrarily across studies. Gaussian random effects are allowed.
    /// The study-level Student interval has at least 95% coverage under these
    /// assumptions (Ibragimov and Müller, 2010, section 2.1). Estimated within-
    /// study variances do not become noisy precision weights. The declared
    /// source independence is checked by the ledger as for other pooling.
    ///
    /// # Errors
    /// As [`Self::pool_summaries`]; fewer than two studies.
    pub fn pool_gaussian_studies(
        &mut self,
        request: PoolingRequest,
        ctx: &ExecutionContext,
    ) -> Result<SynthesisOutcome, EvidenceLedgerRefusal> {
        let (sources, estimates, _) = Self::studies(&request.summaries)?;
        let combination = CombinationRequest {
            label: request.label,
            operation: SynthesisOperation::StatisticalPooling {
                sources: sources.clone(),
                dependence: PoolingDependence::Independent,
            },
            replaces: request.replaces,
        };
        self.combine_numeric(
            combination,
            sources,
            move || {
                let fit = equal_study_gaussian(&estimates)?;
                Ok((
                    SynthesisValue::EqualStudyGaussian(fit),
                    SynthesisMethod::EqualStudyGaussian { estimates },
                ))
            },
            None,
            ctx,
        )
    }

    /// Random-effects pooling of independent studies, with Cochran's `Q`,
    /// `tau^2` and `I^2`. The ledger decides first (independence is known and
    /// disjoint observations, never assumed). Point-only; the variance is
    /// conditional on the estimated `tau^2`.
    ///
    /// # Errors
    /// As [`Self::pool_summaries`]; fewer than two studies.
    pub fn random_effects(
        &mut self,
        request: PoolingRequest,
        method: BetweenStudy,
        ctx: &ExecutionContext,
    ) -> Result<SynthesisOutcome, EvidenceLedgerRefusal> {
        let (sources, estimates, variances) = Self::studies(&request.summaries)?;
        let combination = CombinationRequest {
            label: request.label,
            operation: SynthesisOperation::StatisticalPooling {
                sources: sources.clone(),
                dependence: PoolingDependence::Independent,
            },
            replaces: request.replaces,
        };
        self.combine_numeric(
            combination,
            sources,
            move || match method {
                BetweenStudy::DerSimonianLaird => {
                    let fit = random_effects_dersimonian_laird(&estimates, &variances)?;
                    Ok((
                        SynthesisValue::RandomEffects(fit),
                        SynthesisMethod::RandomEffectsDerSimonianLaird { estimates, variances },
                    ))
                }
                BetweenStudy::RestrictedMaximumLikelihood => {
                    let fit = random_effects_reml(&estimates, &variances)?;
                    Ok((
                        SynthesisValue::RandomEffects(fit),
                        SynthesisMethod::RandomEffectsReml { estimates, variances },
                    ))
                }
            },
            None,
            ctx,
        )
    }

    /// Hierarchical normal-normal borrowing with a declared finite prior on
    /// the between-study standard deviation, evaluated as an exact finite
    /// mixture. The ledger decides first: likelihood sources must be pairwise
    /// independent and every prior's data origin disjoint from them (a prior
    /// built from the likelihood's own data is `evidence_double_counted`).
    /// Point-only: a computed posterior mean and variance, not a calibrated
    /// interval.
    ///
    /// # Errors
    /// As [`Self::pool_summaries`]; an invalid prior declaration.
    pub fn hierarchical_borrowing(
        &mut self,
        request: BorrowingRequest,
        ctx: &ExecutionContext,
    ) -> Result<SynthesisOutcome, EvidenceLedgerRefusal> {
        let (sources, estimates, variances) = Self::studies(&request.likelihood)?;
        too_many(request.priors.len() + sources.len())?;
        let carry = if request.priors.is_empty() { None } else { Some(NOT_CARRIED_PRIOR) };
        let combination = CombinationRequest {
            label: request.label,
            operation: SynthesisOperation::BayesianBorrowing {
                priors: request.priors,
                likelihood: sources.clone(),
            },
            replaces: request.replaces,
        };
        let (mean_prior, tau_prior) = (request.mean_prior, request.tau_prior);
        self.combine_numeric(
            combination,
            sources,
            move || {
                let fit =
                    hierarchical_normal_borrowing(&estimates, &variances, mean_prior, &tau_prior)?;
                Ok((
                    SynthesisValue::Hierarchical(fit),
                    SynthesisMethod::HierarchicalBorrowing {
                        estimates,
                        variances,
                        mean_prior,
                        tau_prior,
                    },
                ))
            },
            carry,
            ctx,
        )
    }

    /// The pooled estimate and its variance as a common pairwise correlation
    /// ranges over `[rho_lo, rho_hi]`: exact extremes, labelled an assumption
    /// range. It is never a point, interval or posterior, and it does not turn
    /// unknown dependence into independence.
    ///
    /// # Errors
    /// A range outside `(-1/(k-1), 1)` or an invalid study, the ledger
    /// refusals of retired sources, or cancellation.
    pub fn dependence_range(
        &mut self,
        request: DependenceRangeRequest,
        ctx: &ExecutionContext,
    ) -> Result<SynthesisOutcome, EvidenceLedgerRefusal> {
        too_many(request.studies.len())?;
        let sources: Vec<SourceId> = request.studies.iter().map(|s| s.source).collect();
        let estimates: Vec<f64> = request.studies.iter().map(|s| s.estimate).collect();
        let standard_errors: Vec<f64> = request.studies.iter().map(|s| s.standard_error).collect();
        let (rho_lo, rho_hi) = (request.rho_lo, request.rho_hi);
        let combination = CombinationRequest {
            label: request.label,
            operation: SynthesisOperation::StatisticalPooling {
                sources: sources.clone(),
                dependence: PoolingDependence::AssumptionRange { rho_lo, rho_hi },
            },
            replaces: request.replaces,
        };
        self.combine_numeric(
            combination,
            sources,
            move || {
                let envelope =
                    pooled_dependence_envelope(&estimates, &standard_errors, rho_lo, rho_hi)?;
                Ok((
                    SynthesisValue::DependenceEnvelope(envelope),
                    SynthesisMethod::DependenceEnvelope {
                        estimates,
                        standard_errors,
                        rho_lo,
                        rho_hi,
                    },
                ))
            },
            None,
            ctx,
        )
    }

    fn combine_accounting(
        &mut self,
        request: CombinationRequest,
    ) -> Result<ResultId, EvidenceLedgerRefusal> {
        too_many(request.operation.direct_sources().len())?;
        self.ledger.combine(request).map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))
    }

    /// Carry transported claims to the target alongside native target data.
    /// Native and transported evidence must share no observations, and a
    /// source that carries a transport identity is never native.
    ///
    /// # Errors
    /// `evidence_double_counted` for one source counted both ways or shared
    /// observations, a transported claim without its transport identity, or a
    /// retired source. The session is unchanged on refusal.
    pub fn transport_claim(
        &mut self,
        label: &str,
        transported: Vec<SourceId>,
        native: Vec<SourceId>,
        replaces: Option<ResultId>,
    ) -> Result<ResultId, EvidenceLedgerRefusal> {
        self.combine_accounting(CombinationRequest {
            label: label.to_owned(),
            operation: SynthesisOperation::CausalTransport { transported, native },
            replaces,
        })
    }

    /// Cite a valid existing result again; its observations are counted once.
    ///
    /// # Errors
    /// An unknown result or one that is stale, withdrawn or replaced.
    pub fn reuse_result(
        &mut self,
        label: &str,
        of: ResultId,
    ) -> Result<ResultId, EvidenceLedgerRefusal> {
        self.combine_accounting(CombinationRequest {
            label: label.to_owned(),
            operation: SynthesisOperation::EvidenceReuse { of },
            replaces: None,
        })
    }

    /// Declare that the same observations serve a later stage. The result is
    /// flagged `data_reuse`: it is not independent confirmation.
    ///
    /// # Errors
    /// Sources whose observations are unknown, or a retired source.
    pub fn repeat_observations(
        &mut self,
        label: &str,
        sources: Vec<SourceId>,
        stage: &str,
    ) -> Result<ResultId, EvidenceLedgerRefusal> {
        self.combine_accounting(CombinationRequest {
            label: label.to_owned(),
            operation: SynthesisOperation::RepeatedObservationUse {
                sources,
                stage: stage.to_owned(),
            },
            replaces: None,
        })
    }

    /// Apply one revision. Results that rested on the retired source are
    /// re-classified; the report carries the classification.
    ///
    /// # Errors
    /// An unknown or already retired source, an update that drops rows, a
    /// supersession of something that is not a model or claim. The session
    /// is unchanged on refusal.
    pub fn revise_source(
        &mut self,
        revision: SourceRevision,
    ) -> Result<RevisionReport, EvidenceLedgerRefusal> {
        let operation = match revision {
            SourceRevision::Update { target, replacement } => {
                LedgerOperation::Update { target, replacement }
            }
            SourceRevision::Replace { target, replacement } => {
                LedgerOperation::Replace { target, replacement }
            }
            SourceRevision::Supersede { target, replacement } => {
                LedgerOperation::Supersede { target, replacement }
            }
            SourceRevision::Withdraw { target, reason } => {
                LedgerOperation::Withdraw { target, reason }
            }
        };
        let applied =
            self.ledger.apply(operation).map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
        Ok(RevisionReport::of(applied))
    }

    /// Re-classify every result against the current source standings and
    /// record the operation.
    ///
    /// # Errors
    /// Only when the operation log is full.
    pub fn revalidate_downstream(
        &mut self,
    ) -> Result<Vec<ResultRevalidation>, EvidenceLedgerRefusal> {
        self.ledger.revalidate_downstream().map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))
    }

    /// A 95% interval for the pooled mean of a result, bound to the measured
    /// coverage record of exactly this construction.
    ///
    /// Fixed-effect (and declared-covariance) pooling reports the Wald `z`
    /// interval of the model variance; random-effects pooling reports the
    /// modified Hartung-Knapp-Sidik-Jonkman interval (`t` with `k - 1`
    /// degrees of freedom, variance floored at the conditional one). These
    /// legacy constructions are conditional on the declared variances being
    /// the studies' sampling variances. [`Self::pool_gaussian_studies`] instead
    /// uses study-level Student inference and does not treat estimated within-
    /// study variances as known precisions.
    ///
    /// Fixed-effect and Gaussian study pooling have measured constructions at
    /// the declared study counts from 5 to 30. The legacy random-effects HKSJ
    /// interval is published only at 20, 25 and 30 studies. At 5, 10 or 15
    /// studies, use [`Self::pool_gaussian_studies`] for the measured Gaussian
    /// construction. A legacy record for a strong-heterogeneity design does
    /// not identify that generating family from the observed summaries.
    /// Records also bound the observations of the smallest study (the sample
    /// size its variance estimate rests on). The random-effects interval uses
    /// the actual study count, with `k - 1` degrees of freedom.
    ///
    /// # Errors
    /// `cell_not_licensed` for a result with no numbers (accounting only), a
    /// hierarchical borrowing or dependence range (no measured construction),
    /// or a construction, study count or sample size without a currently
    /// attesting coverage record. An unknown result is `invalid_argument`.
    pub fn pooled_interval(
        &self,
        result: ResultId,
    ) -> Result<PooledInterval, EvidenceLedgerRefusal> {
        let candidate = self.pooled_interval_candidate(result)?;
        // Legacy HKSJ records share a broad `studies=5-30` key. Only the
        // baseline designs at these three counts passed; strong-heterogeneity
        // records at 5 and 15 cannot license other generating families there.
        // Keep the historical numerical candidate and its replay key intact.
        let unmeasured_hksj_count = matches!(
            self.values.get(&result),
            Some(SynthesisValue::RandomEffects(fit))
                if ![20, 25, 30].contains(&(fit.degrees_of_freedom + 1))
        );
        if unmeasured_hksj_count {
            return Err(EvidenceLedgerRefusal::new(
                antecedent_core::reason_code!("cell_not_licensed"),
                "evidence_ledger.hksj_study_count_not_measured",
                "legacy HKSJ intervals are measured at 20, 25 and 30 studies; use pool_gaussian_studies for the measured Gaussian construction at 5, 10 or 15 studies",
            ));
        }
        let observations = self
            .ledger
            .result(result)
            .and_then(|record| record.observations_digest)
            .map_or_else(|| "unknown".to_owned(), |digest| digest.to_hex());
        let report = [candidate.basis, candidate.exact_count_basis]
            .into_iter()
            .map(|basis| {
                antecedent_io::measured_inference::build_report(
                    "evidence_ledger_pooled",
                    result.to_hex().as_bytes(),
                    &result.to_hex(),
                    &observations,
                    candidate.scope.clone(),
                    vec![antecedent_io::measured_inference::ScalarExecution {
                        name: "pooled_mean".into(),
                        point: candidate.estimate,
                        interval: (candidate.lower, candidate.upper),
                        basis,
                    }],
                )
            })
            .reduce(Result::or)
            .unwrap_or_else(|| unreachable!("two bases were supplied"))
        .map_err(|error| {
            EvidenceLedgerRefusal::new(
                antecedent_core::reason_code!("cell_not_licensed"),
                "evidence_ledger.interval_not_measured",
                format!(
                    "no currently attesting coverage record governs this construction ({error:?})"
                ),
            )
        })?;
        Ok(PooledInterval {
            result,
            method: candidate.method,
            estimate: candidate.estimate,
            lower: candidate.lower,
            upper: candidate.upper,
            level: candidate.level,
            report,
        })
    }

    /// The interval construction of a pooled result before any coverage record
    /// is consulted, with the basis a record is matched against. This is what a
    /// calibration harness measures; it is never a published claim.
    ///
    /// # Errors
    /// As [`Self::pooled_interval`], except for the missing record.
    #[doc(hidden)]
    #[allow(clippy::too_many_lines, reason = "one construction per synthesis variant")]
    pub fn pooled_interval_candidate(
        &self,
        result: ResultId,
    ) -> Result<PooledIntervalCandidate, EvidenceLedgerRefusal> {
        let Some(value) = self.values.get(&result) else {
            return match self.ledger.request_interval(result) {
                Err(error) => Err(EvidenceLedgerRefusal::from_ledger(&error)),
                Ok(never) => match never {},
            };
        };
        self.ledger.require_valid(result).map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
        let record = self.ledger.result(result).ok_or_else(|| {
            EvidenceLedgerRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "evidence_ledger.unknown_result",
                "the result is not in the ledger",
            )
        })?;
        let unmeasured = |what: &str| {
            EvidenceLedgerRefusal::new(
                antecedent_core::reason_code!("cell_not_licensed"),
                "evidence_ledger.interval_not_measured",
                format!("no measured interval construction exists for {what}"),
            )
        };
        let level = POOLED_INTERVAL_LEVEL;
        let dependence =
            if record.dependence_declared { "declared_covariance" } else { "independent" };
        let (estimate, half, studies, estimator, interval_method, se_kind, variant, dependence) =
            match value {
                SynthesisValue::EqualStudyGaussian(p) => {
                    #[allow(clippy::cast_precision_loss, reason = "bounded study count")]
                    let t = antecedent_stats::special::student_t_ppf(
                        0.5 + level / 2.0,
                        p.degrees_of_freedom as f64,
                    );
                    (
                        p.estimate,
                        t * p.variance.sqrt(),
                        p.degrees_of_freedom + 1,
                        "evidence.synthesis.equal_study_gaussian",
                        "independent_gaussian_study_t",
                        "empirical_study_variance",
                        "equal_study_gaussian",
                        "independent",
                    )
                }
                SynthesisValue::FixedEffect(p) => {
                    let z = antecedent_stats::special::normal_ppf(0.5 + level / 2.0);
                    (
                        p.estimate,
                        z * p.variance.sqrt(),
                        p.degrees_of_freedom + 1,
                        "evidence.synthesis.fixed_effect_gls",
                        "wald_z",
                        "model_variance",
                        "fixed_effect",
                        dependence,
                    )
                }
                SynthesisValue::RandomEffects(r) => {
                    let (estimates, variances, variant) = match self.records.get(&result) {
                        Some(SynthesisRecord {
                            method:
                                SynthesisMethod::RandomEffectsDerSimonianLaird { estimates, variances },
                            ..
                        }) => (estimates, variances, "random_effects_dl"),
                        Some(SynthesisRecord {
                            method: SynthesisMethod::RandomEffectsReml { estimates, variances },
                            ..
                        }) => (estimates, variances, "random_effects_reml"),
                        _ => {
                            return Err(unmeasured(
                                "a random-effects result whose study summaries are not carried",
                            ));
                        }
                    };
                    let k = estimates.len();
                    if variances.len() != k {
                        return Err(unmeasured(
                            "a random-effects result whose study summaries are inconsistent",
                        ));
                    }
                    let precision: Vec<f64> =
                        variances.iter().map(|v| 1.0 / (v + r.tau_squared)).collect();
                    let total: f64 = precision.iter().sum();
                    let spread: f64 = precision
                        .iter()
                        .zip(estimates)
                        .map(|(w, y)| w * (y - r.estimate) * (y - r.estimate))
                        .sum();
                    #[allow(clippy::cast_precision_loss, reason = "bounded study count")]
                    let q = spread / (k - 1) as f64;
                    let variance = q.max(1.0) / total;
                    #[allow(clippy::cast_precision_loss, reason = "bounded study count")]
                    let t =
                        antecedent_stats::special::student_t_ppf(0.5 + level / 2.0, (k - 1) as f64);
                    (
                        r.estimate,
                        t * variance.sqrt(),
                        k,
                        "evidence.synthesis.random_effects",
                        "hksj_t",
                        "hartung_knapp",
                        variant,
                        "independent",
                    )
                }
                SynthesisValue::Hierarchical(_) => {
                    return Err(unmeasured("a hierarchical borrowing posterior"));
                }
                SynthesisValue::DependenceEnvelope(_) => {
                    return Err(unmeasured("an assumption range over a declared dependence"));
                }
            };
        if !(half.is_finite() && estimate.is_finite()) {
            return Err(unmeasured("a non-finite pooled value"));
        }
        if !MEASURED_STUDY_COUNTS.contains(&studies) {
            return Err(unmeasured(&format!(
                "{studies} studies (the measured family is {} to {})",
                MEASURED_STUDY_COUNTS.start(),
                MEASURED_STUDY_COUNTS.end()
            )));
        }
        let Some(study_rows) = self.smallest_study_rows(result) else {
            return Err(unmeasured("studies whose observations are not all declared"));
        };
        let make_basis = |functional: String, rows: u64| {
            antecedent_core::CalibrationBasis::new(
                [
                    "PooledEffect",
                    "NoGraph",
                    "fixed",
                    "summary",
                    "Frequentist",
                    estimator,
                    interval_method,
                    se_kind,
                    dependence,
                    "",
                    functional.as_str(),
                ]
                .map(std::sync::Arc::from),
                level,
                std::sync::Arc::from("point"),
                rows,
                None,
                None,
                0.0,
            )
        };
        let basis = make_basis(
            format!(
                "pooled_mean;{variant};studies={}-{}",
                MEASURED_STUDY_COUNTS.start(),
                MEASURED_STUDY_COUNTS.end()
            ),
            study_rows,
        );
        let exact_count_basis = make_basis(
            format!("pooled_mean;{variant};studies={studies}"),
            record.mass.distinct_observations,
        );
        let (lower, upper) = (estimate - half, estimate + half);
        let equal_study = matches!(value, SynthesisValue::EqualStudyGaussian(_));
        let mut scope = serde_json::json!({
            "validation_design": "independent_gaussian_studies_with_estimated_variances",
            "declared_model_assumptions": ["declared sampling variances (or covariance) are the studies' sampling variances", "studies are independent or the declared covariance holds"],
            "studies": studies,
            "observations": record.mass.distinct_observations,
            "method": variant,
            "level": level,
        });
        if equal_study {
            scope["coverage_claim"] = serde_json::json!("at_least_nominal");
            scope["declared_model_assumptions"] = serde_json::json!([
                "study estimates are independent Gaussian variables with a common marginal mean",
                "unknown study variances may differ; Gaussian between-study effects are included"
            ]);
            scope["reference"] = serde_json::json!(
                "Ibragimov-Mueller 2010, DOI 10.1198/jbes.2009.08046, section 2.1"
            );
        }
        Ok(PooledIntervalCandidate {
            method: interval_method,
            estimate,
            lower,
            upper,
            level,
            basis,
            exact_count_basis,
            scope,
        })
    }

    /// Distinct observations of the smallest pooled study: the sample size the
    /// per-study variance estimates rest on, which is what the measured
    /// family varies across study counts. `None` when a study does not
    /// declare its observations or the result pools no carried studies.
    fn smallest_study_rows(&self, result: ResultId) -> Option<u64> {
        let record = self.records.get(&result)?;
        record
            .sources
            .iter()
            .map(|&source| Some(self.ledger.source(source)?.effective.known()?.len() as u64))
            .collect::<Option<Vec<u64>>>()?
            .into_iter()
            .min()
    }

    /// The numbers of a result that is still valid.
    ///
    /// # Errors
    /// `evidence_source_withdrawn` for a result whose evidence changed after
    /// it was registered (stale, withdrawn or replaced), or an unknown result.
    /// `Ok(None)` for a valid result that was registered for accounting only
    /// (transport, reuse, repeated use) or whose numbers the session did not
    /// compute.
    pub fn synthesis(
        &self,
        result: ResultId,
    ) -> Result<Option<&SynthesisValue>, EvidenceLedgerRefusal> {
        self.ledger.require_valid(result).map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
        Ok(self.values.get(&result))
    }

    /// The ledger's record of a result, valid or not.
    #[must_use]
    pub fn result(&self, id: ResultId) -> Option<&ResultRecord> {
        self.ledger.result(id)
    }

    /// Classification of one result against the current source standings.
    #[must_use]
    pub fn standing(&self, id: ResultId) -> Option<Revalidation> {
        self.ledger.standing(id)
    }

    /// Apply a batch of added and retired evidence to a copy of this session.
    ///
    /// See the module documentation for what is kept, recomputed and refused.
    /// The batch is all-or-nothing; this session is not changed either way.
    ///
    /// # Errors
    /// The refusal of the first change that the ledger refuses.
    pub fn refresh(
        &self,
        changes: Vec<SourceChange>,
    ) -> Result<LedgerRefresh, EvidenceLedgerRefusal> {
        let mut session = self.clone();
        let mut reports = Vec::with_capacity(changes.len());
        for change in changes {
            match change {
                SourceChange::Add(source) => {
                    let applied = session
                        .ledger
                        .apply(LedgerOperation::Add { source })
                        .map_err(|e| EvidenceLedgerRefusal::from_ledger(&e))?;
                    reports.push(RevisionReport::of(applied));
                }
                SourceChange::Revise(revision) => reports.push(session.revise_source(revision)?),
            }
        }
        let classification = session.ledger.classification();
        Ok(LedgerRefresh { session, reports, classification })
    }

    /// Write the ledger and the syntheses bound to its results.
    ///
    /// # Errors
    /// A ledger or synthesis that does not seal, an encoding failure or a
    /// blank artifact id.
    pub fn export(&self, artifact_id: &str) -> Result<LedgerExport, EvidenceLedgerRefusal> {
        let records: Vec<SynthesisRecord> = self.records.values().cloned().collect();
        let (artifact, _) = EvidenceLedgerArtifact::seal(&self.ledger, &records)
            .map_err(|e| EvidenceLedgerRefusal::from_artifact(&e))?;
        let bytes = EvidenceLedgerArtifact::to_bytes(&self.ledger, &records, artifact_id)
            .map_err(|e| EvidenceLedgerRefusal::from_artifact(&e))?;
        Ok(LedgerExport {
            bytes,
            identity: artifact.identity().to_owned(),
            carried: self.records.keys().copied().collect(),
            not_carried: self.not_carried.iter().map(|(id, why)| (*id, *why)).collect(),
        })
    }
}

/// A consumed `evidence_ledger_v1` artifact: the replayed ledger and the
/// recomputed syntheses.
#[derive(Clone, Debug)]
pub struct ConsumedEvidenceLedger {
    ledger: EvidenceLedger,
    syntheses: Vec<ComputedSynthesis>,
    identity: String,
}

impl ConsumedEvidenceLedger {
    /// The replayed ledger.
    #[must_use]
    pub const fn ledger(&self) -> &EvidenceLedger {
        &self.ledger
    }

    /// The syntheses recomputed with the numerical routines, in stored order;
    /// `values` follow each method's fixed order.
    #[must_use]
    pub fn syntheses(&self) -> &[ComputedSynthesis] {
        &self.syntheses
    }

    /// The recomputed artifact identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The recomputed synthesis bound to `result`, when there is one.
    #[must_use]
    pub fn synthesis_of(&self, result: ResultId) -> Option<&ComputedSynthesis> {
        self.syntheses.iter().find(|s| s.result == result)
    }
}

/// Independently consume an evidence ledger artifact by replay and
/// recomputation.
///
/// `expected_identity` is an identity (hex) the consumer retained
/// independently of the bytes; with it a consistently resealed artifact of
/// another history is refused.
///
/// # Errors
/// Corruption, another major version, a bound exceeded, an operation the
/// ledger refuses on replay, a stored identity, effect or value that does not
/// replay, or an identity that does not match.
pub fn consume_evidence_ledger_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> Result<ConsumedEvidenceLedger, EvidenceLedgerRefusal> {
    let artifact = EvidenceLedgerArtifact::from_bytes(bytes, expected_identity)
        .map_err(|e| EvidenceLedgerRefusal::from_artifact(&e))?;
    Ok(ConsumedEvidenceLedger {
        ledger: artifact.ledger().clone(),
        syntheses: artifact.syntheses().to_vec(),
        identity: artifact.identity().to_owned(),
    })
}
