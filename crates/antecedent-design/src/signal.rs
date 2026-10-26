//! Typed candidate-signal providers: the scientific request/receipt contract for a
//! candidate-specific predictive observation law and its posterior update.
//!
//! A [`SignalProvider`] answers one exact [`SignalRequest`] (candidate, prior, state
//! and observation coordinates, sample size, RNG seed, evidence lineage, the
//! conditional-independence assumption and computational limits) with a
//! [`PreparedSignal`]: either a finite or continuous observation law that the
//! existing preposterior machinery updates natively
//! ([`crate::preposterior::DecisionSignal`]), or externally computed per-branch
//! decision values. The native Gaussian-mean and binomial families and the
//! [`ExternalSignal`] provider sit behind the same contract. An external provider's
//! trust label and exact request fingerprint travel on the [`SignalReceipt`]; a
//! representation never upgrades trust to native.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use antecedent_contract::{
    ExternalCapability, ExternalRefusal, ExternalScientificObject, ExternalTrustState,
    ProviderObjectIdentity, ScientificQuantity,
};
use antecedent_core::{CanonicalWriter, CausalRng, is_blank};

use crate::candidate::CandidateDesign;
use crate::error::DesignError;
use crate::preposterior::{BinomialSignal, DecisionSignal, GaussianMeanSignal};

/// Absolute tolerance for probability-mass and coherence checks on supplied tables.
const COHERENCE_TOLERANCE: f64 = 1e-9;

/// Computational limits a request imposes on the provider and the evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignalLimits {
    /// Largest sample size the request may declare.
    pub max_sample_size: u64,
    /// Largest number of distinct observation values enumerated exactly.
    pub max_support: usize,
}

impl Default for SignalLimits {
    fn default() -> Self {
        Self { max_sample_size: 10_000_000, max_support: 65_536 }
    }
}

/// One exact scientific request to a signal provider.
#[derive(Clone, Debug, PartialEq)]
pub struct SignalRequest {
    /// Stable proposed-study candidate identity.
    pub candidate_id: String,
    /// Prior or current-state identity the update consumes.
    pub prior_id: String,
    /// Coordinate of the decision state the prior and posterior are about.
    pub state_quantity: ScientificQuantity,
    /// Coordinate of the possible observation (the sufficient statistic).
    pub observation_quantity: ScientificQuantity,
    /// Number of observations the candidate collects.
    pub sample_size: u64,
    /// RNG seed the evaluation draws from.
    pub rng_seed: u64,
    /// Evidence lineage identities the signal depends on.
    pub evidence_lineage: Vec<String>,
    /// Stable identity of the conditional-independence assumption of the
    /// observations given the state.
    pub conditional_independence: String,
    /// Computational limits.
    pub limits: SignalLimits,
}

/// Digest of a native signal law: the family label, and for the Gaussian-mean family its noise
/// variance. The one implementation behind both the provider's `law_digest` and the design
/// ranking artifact's replay of it.
pub(crate) fn native_law_digest(family: &str, parameters: &[f64]) -> String {
    let mut hasher = CanonicalWriter::new("antecedent.signal.native_law.v1");
    hasher.text(family);
    if family == "gaussian_mean" {
        hasher.reals(parameters);
    }
    hasher.finish_hex()
}

/// A count-prefixed table of rows, each a count-prefixed sequence of reals.
fn rows(hasher: &mut CanonicalWriter, table: &[Vec<f64>]) {
    hasher.word(table.len() as u64);
    for row in table {
        hasher.reals(row);
    }
}

impl SignalRequest {
    /// Check the request is complete and within its own limits.
    ///
    /// # Errors
    ///
    /// Blank identities, missing lineage or conditional-independence assumption,
    /// invalid coordinates, a zero sample size, or one above the declared limit.
    pub fn validate(&self) -> Result<(), SignalError> {
        if is_blank(&self.candidate_id) || is_blank(&self.prior_id) {
            return Err(SignalError::InvalidRequest("candidate and prior identities"));
        }
        if self.state_quantity.validate().is_err() || self.observation_quantity.validate().is_err()
        {
            return Err(SignalError::InvalidRequest("state and observation coordinates"));
        }
        if self.evidence_lineage.is_empty() || self.evidence_lineage.iter().any(|l| is_blank(l)) {
            return Err(SignalError::MissingLineage);
        }
        if is_blank(&self.conditional_independence) {
            return Err(SignalError::ConditionalIndependenceMissing);
        }
        if self.sample_size == 0 {
            return Err(SignalError::SampleSizeInvalid);
        }
        if self.sample_size > self.limits.max_sample_size {
            return Err(SignalError::ComputationalLimit("sample_size"));
        }
        Ok(())
    }

    /// Exact request fingerprint (BLAKE3 hex). Reordering the evidence lineage
    /// leaves it unchanged; any other change alters it.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut hasher = CanonicalWriter::new("antecedent.signal_request.v1");
        hasher.text(&self.candidate_id).text(&self.prior_id);
        self.state_quantity.write_canonical(&mut hasher);
        self.observation_quantity.write_canonical(&mut hasher);
        hasher.word(self.sample_size).word(self.rng_seed);
        let mut lineage = self.evidence_lineage.clone();
        lineage.sort();
        hasher.texts(&lineage);
        hasher
            .text(&self.conditional_independence)
            .word(self.limits.max_sample_size)
            .word(self.limits.max_support as u64);
        hasher.finish_hex()
    }
}

/// How the posterior update behind a signal was computed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SignalUpdateMode {
    /// Antecedent updates natively from a supplied or built-in likelihood.
    NativeUpdate,
    /// The provider computed the updated posterior for every possible observation.
    ExternalPosterior,
    /// The provider computed the posterior decision-value distribution.
    ExternalDecisionValues,
}

impl SignalUpdateMode {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeUpdate => "native_update",
            Self::ExternalPosterior => "external_posterior",
            Self::ExternalDecisionValues => "external_decision_values",
        }
    }

    /// Whether the update was computed outside Antecedent.
    #[must_use]
    pub const fn externally_computed(self) -> bool {
        !matches!(self, Self::NativeUpdate)
    }
}

/// Trust label a signal carries on its receipt.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SignalTrustLabel {
    /// Implemented and licensed natively.
    NativeLicensed,
    /// Supplier-asserted only.
    ExternallyAttested,
    /// The exact contract passed object-level verification (still not native).
    ExactRequestVerified,
}

impl SignalTrustLabel {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeLicensed => "native_licensed",
            Self::ExternallyAttested => "externally_attested",
            Self::ExactRequestVerified => "exact_request_verified",
        }
    }
}

/// What a provider answered for one exact request.
#[derive(Clone, Debug, PartialEq)]
pub struct SignalReceipt {
    /// Fingerprint of the exact request answered.
    pub request_fingerprint: String,
    /// Candidate identity.
    pub candidate_id: String,
    /// Prior identity.
    pub prior_id: String,
    /// Sample size.
    pub sample_size: u64,
    /// RNG seed.
    pub rng_seed: u64,
    /// Signal family label (for example `gaussian_mean`, `external_finite_law`).
    pub family: String,
    /// Update mode.
    pub update_mode: SignalUpdateMode,
    /// Trust label.
    pub trust: SignalTrustLabel,
    /// Attesting party for an externally attested signal.
    pub attestor: Option<String>,
    /// Exact provider object identity for an external signal.
    pub provider_identity: Option<ProviderObjectIdentity>,
    /// Evidence lineage (sorted).
    pub evidence_lineage: Vec<String>,
    /// Conditional-independence assumption identity.
    pub conditional_independence: String,
    /// Computational limits.
    pub limits: SignalLimits,
    /// BLAKE3 digest of the observation law or table the provider supplied.
    pub law_digest: String,
}

impl SignalReceipt {
    /// Canonical identity of the signal and update (BLAKE3 hex). Any change of
    /// request, provider, update mode, trust or law changes it.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut hasher = CanonicalWriter::new("antecedent.signal_receipt.v1");
        hasher
            .text(&self.request_fingerprint)
            .text(&self.family)
            .text(self.update_mode.as_str())
            .text(self.trust.as_str())
            .opt_text(self.attestor.as_deref());
        match &self.provider_identity {
            Some(identity) => {
                hasher.word(1);
                for part in [
                    &identity.provider_id,
                    &identity.object_id,
                    &identity.version_id,
                    &identity.snapshot_id,
                    &identity.request_id,
                ] {
                    hasher.text(part);
                }
            }
            None => {
                hasher.word(0);
            }
        }
        hasher.text(&self.law_digest);
        hasher.finish_hex()
    }
}

/// Externally computed branch law and posterior decision values.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalDecisionValues {
    /// Probability of each possible observation.
    pub branch_probabilities: Vec<f64>,
    /// Action identities naming the value columns.
    pub action_ids: Vec<String>,
    /// `values[branch][action]` posterior expected utility given the branch.
    pub values: Vec<Vec<f64>>,
}

/// The observation law a provider supplied.
#[derive(Clone)]
pub enum PreparedLaw {
    /// A likelihood Antecedent updates natively.
    Likelihood(Arc<dyn DecisionSignal<f64>>),
    /// Externally computed per-branch decision values.
    DecisionValues(ExternalDecisionValues),
}

/// A provider's answer to one request.
#[derive(Clone)]
pub struct PreparedSignal {
    /// Observation law.
    pub law: PreparedLaw,
    /// Receipt retained with every result.
    pub receipt: SignalReceipt,
    /// Decision-state draws the law is defined on, when it is tied to specific draws.
    pub bound_states: Option<Vec<f64>>,
}

impl std::fmt::Debug for PreparedLaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Likelihood(signal) => write!(f, "Likelihood({})", signal.name()),
            Self::DecisionValues(values) => f.debug_tuple("DecisionValues").field(values).finish(),
        }
    }
}

impl std::fmt::Debug for PreparedSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedSignal")
            .field("law", &self.law)
            .field("receipt", &self.receipt)
            .field("bound_states", &self.bound_states)
            .finish()
    }
}

/// How many prepared laws to keep for a later seal. A ranking seals the laws it
/// just evaluated; older entries fall off so a process that evaluates without
/// sealing does not retain every law it has ever prepared.
const RETAINED_LAW_CAP: usize = 256;

struct RetainedLaws {
    order: VecDeque<String>,
    by_id: HashMap<String, PreparedSignal>,
}

fn retained_laws() -> &'static Mutex<RetainedLaws> {
    static LAWS: OnceLock<Mutex<RetainedLaws>> = OnceLock::new();
    LAWS.get_or_init(|| Mutex::new(RetainedLaws { order: VecDeque::new(), by_id: HashMap::new() }))
}

/// Whether a context can use the process-owned signal-law memo. A constrained
/// caller cannot transfer its scientific payload beyond its byte account.
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum ProcessLawReuse {
    Allowed,
    Bypass,
}

impl ProcessLawReuse {
    pub(crate) fn for_context(ctx: &antecedent_core::ExecutionContext) -> Self {
        if ctx.cache_policy.enabled
            && ctx.cache_policy.max_bytes.is_none()
            && ctx.admission.byte_budget() == u64::MAX
        {
            Self::Allowed
        } else {
            Self::Bypass
        }
    }
}

/// Remember the law `evaluate_evsi` just prepared, keyed by receipt identity.
///
/// The receipt identity covers the law digest. [`recall_prepared`] returns it to `seal`.
pub(crate) fn remember_prepared(
    prepared: &PreparedSignal,
    ctx: &antecedent_core::ExecutionContext,
) {
    if ProcessLawReuse::for_context(ctx) == ProcessLawReuse::Bypass {
        return;
    }
    let id = prepared.receipt.identity();
    let mut slot = retained_laws().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(existing) = slot.by_id.get_mut(&id) {
        *existing = prepared.clone();
        return;
    }
    while slot.order.len() >= RETAINED_LAW_CAP {
        if let Some(old) = slot.order.pop_front() {
            slot.by_id.remove(&old);
        } else {
            break;
        }
    }
    slot.order.push_back(id.clone());
    slot.by_id.insert(id, prepared.clone());
}

/// The prepared law whose receipt identity is `identity`, if it is still retained.
pub(crate) fn recall_prepared(identity: &str) -> Option<PreparedSignal> {
    let slot = retained_laws().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    slot.by_id.get(identity).cloned()
}

/// Candidate-specific predictive observation law and posterior update.
pub trait SignalProvider: Send + Sync {
    /// Answer one exact request.
    ///
    /// # Errors
    ///
    /// An invalid request, or an answer that does not align with it.
    fn prepare(&self, request: &SignalRequest) -> Result<PreparedSignal, SignalError>;
}

/// Why a signal request or provider answer is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SignalError {
    /// A required request field is blank or invalid.
    InvalidRequest(&'static str),
    /// No evidence lineage was declared.
    MissingLineage,
    /// No conditional-independence assumption was declared.
    ConditionalIndependenceMissing,
    /// The sample size is zero.
    SampleSizeInvalid,
    /// A declared computational limit is exceeded.
    ComputationalLimit(&'static str),
    /// The external object is not a valid signal declaration.
    InvalidObject,
    /// A supplied table is malformed.
    InvalidTable(&'static str),
    /// The object's exact request fingerprint differs from the request.
    RequestFingerprintMismatch,
    /// The object names a different candidate.
    CandidateMismatch,
    /// The object names a different prior.
    PriorMismatch,
    /// The supplied law is for a different sample size.
    SampleSizeMismatch {
        /// Sample size the law was supplied for.
        declared: u64,
        /// Sample size requested.
        requested: u64,
    },
    /// The observation coordinate differs from the request.
    ObservationQuantityMismatch,
    /// The returned state or posterior coordinate differs from the request.
    PosteriorQuantityMismatch,
    /// A required operation is not declared.
    CapabilityMissing(ExternalCapability),
    /// The supplier claimed native trust.
    TrustNativeClaimed,
    /// Verification covers another contract, or the attestor is blank.
    TrustMismatch,
    /// The predictive law does not sum to one for some state.
    LawIncoherent,
    /// The returned posterior does not update the prior coherently.
    PosteriorIncoherent,
}

/// The one place this crate fills an [`ExternalRefusal`].
///
/// The contract constructor owns the field set. Call sites set the optional fields
/// here, and [`Self::build`] forwards them to `ExternalRefusal::new`.
pub(crate) struct RefusalBuilder {
    code: &'static str,
    stage: &'static str,
    detail: String,
    offending: Option<String>,
    expected: Option<String>,
    supplied: Option<String>,
    capability: Option<ExternalCapability>,
    remedy: Option<&'static str>,
}

impl RefusalBuilder {
    /// A refusal with no optional fields.
    pub(crate) fn new(code: &'static str, stage: &'static str, detail: &str) -> Self {
        Self {
            code,
            stage,
            detail: detail.to_owned(),
            offending: None,
            expected: None,
            supplied: None,
            capability: None,
            remedy: None,
        }
    }

    /// Name the offending coordinate.
    pub(crate) fn offending(mut self, value: impl Into<String>) -> Self {
        self.offending = Some(value.into());
        self
    }

    /// Name the offending coordinate when there is one.
    pub(crate) fn offending_opt(mut self, value: Option<String>) -> Self {
        self.offending = value;
        self
    }

    /// Record the expected semantics.
    pub(crate) fn expected(mut self, value: impl Into<String>) -> Self {
        self.expected = Some(value.into());
        self
    }

    /// Record the supplied semantics.
    pub(crate) fn supplied(mut self, value: impl Into<String>) -> Self {
        self.supplied = Some(value.into());
        self
    }

    /// Record the missing operation.
    pub(crate) fn capability(mut self, value: ExternalCapability) -> Self {
        self.capability = Some(value);
        self
    }

    /// Record a remedy.
    pub(crate) fn remedy(mut self, value: &'static str) -> Self {
        self.remedy = Some(value);
        self
    }

    /// The structured refusal.
    pub(crate) fn build(self) -> ExternalRefusal {
        let mut value = ExternalRefusal::new(self.code, self.stage, self.detail);
        if let Some(offending) = self.offending {
            value = value.with_offending(offending);
        }
        if let Some(expected) = self.expected {
            value = value.with_expected(expected);
        }
        if let Some(supplied) = self.supplied {
            value = value.with_supplied(supplied);
        }
        if let Some(capability) = self.capability {
            value = value.with_capability(capability);
        }
        if let Some(remedy) = self.remedy {
            value = value.with_remedy(remedy);
        }
        value
    }
}

/// Keep a nested error's refusal and name the stage that surfaced it.
pub(crate) fn at_stage(mut refusal: ExternalRefusal, stage: &'static str) -> ExternalRefusal {
    refusal.stage = stage;
    refusal
}

impl SignalError {
    /// Structured refusal under the `signal_provider` namespace.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let signal = antecedent_core::reason_code!("design_signal_invalid");
        let at = |code, detail: &str| RefusalBuilder::new(code, "declare", detail);
        match self {
            Self::InvalidRequest(what) => at(
                antecedent_core::reason_code!("invalid_argument"),
                "signal_provider.invalid_request",
            )
            .offending(*what)
            .build(),
            Self::MissingLineage => at(signal, "signal_provider.missing_lineage").build(),
            Self::ConditionalIndependenceMissing => {
                at(signal, "signal_provider.conditional_independence_missing").build()
            }
            Self::SampleSizeInvalid => at(signal, "signal_provider.sample_size_invalid").build(),
            Self::ComputationalLimit(what) => {
                at(signal, "signal_provider.computational_limit").offending(*what).build()
            }
            Self::InvalidObject => at(
                antecedent_core::reason_code!("external_binding_mismatch"),
                "signal_provider.invalid_object",
            )
            .build(),
            Self::InvalidTable(what) => {
                at(signal, "signal_provider.invalid_table").offending(*what).build()
            }
            Self::RequestFingerprintMismatch => at(
                antecedent_core::reason_code!("external_binding_mismatch"),
                "signal_provider.request_fingerprint_mismatch",
            )
            .build(),
            Self::CandidateMismatch => at(signal, "signal_provider.candidate_mismatch").build(),
            Self::PriorMismatch => at(signal, "signal_provider.prior_mismatch").build(),
            Self::SampleSizeMismatch { declared, requested } => {
                at(signal, "signal_provider.sample_size_mismatch")
                    .expected(requested.to_string())
                    .supplied(declared.to_string())
                    .build()
            }
            Self::ObservationQuantityMismatch => at(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                "signal_provider.observation_quantity_mismatch",
            )
            .build(),
            Self::PosteriorQuantityMismatch => at(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                "signal_provider.posterior_quantity_mismatch",
            )
            .build(),
            Self::CapabilityMissing(capability) => at(
                antecedent_core::reason_code!("external_capability_missing"),
                "signal_provider.capability_missing",
            )
            .capability(*capability)
            .build(),
            Self::TrustNativeClaimed => at(
                antecedent_core::reason_code!("external_binding_mismatch"),
                "signal_provider.trust_native_claimed",
            )
            .build(),
            Self::TrustMismatch => at(
                antecedent_core::reason_code!("external_binding_mismatch"),
                "signal_provider.trust_mismatch",
            )
            .build(),
            Self::LawIncoherent => at(signal, "signal_provider.law_incoherent").build(),
            Self::PosteriorIncoherent => at(signal, "signal_provider.posterior_incoherent").build(),
        }
    }
}

// -- native providers -------------------------------------------------------------------

/// Signal wrapper that makes the request's sample size authoritative.
struct FixedSizeSignal {
    inner: Arc<dyn DecisionSignal<f64>>,
    n: u64,
}

impl DecisionSignal<f64> for FixedSizeSignal {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn sample_size(&self, _candidate: &CandidateDesign) -> Option<u64> {
        Some(self.n)
    }

    fn finite_support(&self, n: u64) -> Option<Vec<f64>> {
        self.inner.finite_support(n)
    }

    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        self.inner.log_likelihood(statistic, n, states, out)
    }

    fn sample_statistic(
        &self,
        state: &f64,
        n: u64,
        rng: &mut CausalRng,
    ) -> Result<f64, DesignError> {
        self.inner.sample_statistic(state, n, rng)
    }

    fn gaussian_noise_variance(&self) -> Option<f64> {
        self.inner.gaussian_noise_variance()
    }
}

fn native_prepared(
    request: &SignalRequest,
    family: &str,
    signal: Arc<dyn DecisionSignal<f64>>,
    law_digest: String,
) -> PreparedSignal {
    let wrapped = Arc::new(FixedSizeSignal { inner: signal, n: request.sample_size });
    let mut lineage = request.evidence_lineage.clone();
    lineage.sort();
    PreparedSignal {
        law: PreparedLaw::Likelihood(wrapped),
        receipt: SignalReceipt {
            request_fingerprint: request.fingerprint(),
            candidate_id: request.candidate_id.clone(),
            prior_id: request.prior_id.clone(),
            sample_size: request.sample_size,
            rng_seed: request.rng_seed,
            family: family.to_owned(),
            update_mode: SignalUpdateMode::NativeUpdate,
            trust: SignalTrustLabel::NativeLicensed,
            attestor: None,
            provider_identity: None,
            evidence_lineage: lineage,
            conditional_independence: request.conditional_independence.clone(),
            limits: request.limits,
            law_digest,
        },
        bound_states: None,
    }
}

/// Native Gaussian sample-mean signal `ȳ ~ N(θ, σ²/n)` behind the provider contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeGaussianMeanSignal {
    signal: GaussianMeanSignal,
}

impl NativeGaussianMeanSignal {
    /// Construct from the variance `σ²` of one observation.
    ///
    /// # Errors
    ///
    /// Non-positive or non-finite variance.
    pub fn new(noise_variance: f64) -> Result<Self, DesignError> {
        Ok(Self { signal: GaussianMeanSignal::new(noise_variance)? })
    }

    /// Variance `σ²` of one observation.
    #[must_use]
    pub const fn noise_variance(&self) -> f64 {
        self.signal.noise_variance()
    }
}

impl SignalProvider for NativeGaussianMeanSignal {
    fn prepare(&self, request: &SignalRequest) -> Result<PreparedSignal, SignalError> {
        request.validate()?;
        let digest = native_law_digest("gaussian_mean", &[self.signal.noise_variance()]);
        Ok(native_prepared(request, "gaussian_mean", Arc::new(self.signal), digest))
    }
}

/// Native binomial signal (successes in `n` Bernoulli trials) behind the provider contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeBinomialSignal;

impl SignalProvider for NativeBinomialSignal {
    fn prepare(&self, request: &SignalRequest) -> Result<PreparedSignal, SignalError> {
        request.validate()?;
        let support = request.sample_size.saturating_add(1);
        let too_large = match usize::try_from(support) {
            Ok(size) => size > request.limits.max_support,
            Err(_) => true,
        };
        if too_large {
            return Err(SignalError::ComputationalLimit("support"));
        }
        let digest = native_law_digest("binomial", &[]);
        Ok(native_prepared(request, "binomial", Arc::new(BinomialSignal), digest))
    }
}

// -- external provider ------------------------------------------------------------------

/// The law an external provider supplied, one variant per update mode.
#[derive(Clone, Debug, PartialEq)]
pub enum ExternalLaw {
    /// Predictive observation law `P(y | θ_k)` over the prior draws; Antecedent
    /// updates natively ([`SignalUpdateMode::NativeUpdate`]).
    Likelihood {
        /// Prior draws of the state the law is defined on.
        states: Vec<f64>,
        /// Possible observation values.
        statistics: Vec<f64>,
        /// `probabilities[y][k] = P(statistic_y | state_k)`.
        probabilities: Vec<Vec<f64>>,
    },
    /// Externally computed posterior over the equally weighted prior draws for each
    /// possible observation ([`SignalUpdateMode::ExternalPosterior`]).
    Posterior {
        /// Prior draws of the state (equally weighted).
        states: Vec<f64>,
        /// Possible observation values.
        statistics: Vec<f64>,
        /// Predictive probability `P(statistic_y)`.
        predictive: Vec<f64>,
        /// `posterior[y][k] = P(state_k | statistic_y)`.
        posterior: Vec<Vec<f64>>,
    },
    /// Externally computed decision-value distribution
    /// ([`SignalUpdateMode::ExternalDecisionValues`]).
    DecisionValues(ExternalDecisionValues),
}

/// Supplied external signal payload with the coordinates it claims to answer.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalSignalBody {
    /// Sample size the law is supplied for.
    pub sample_size: u64,
    /// Coordinate of the state the law or posterior is about.
    pub state_quantity: ScientificQuantity,
    /// Coordinate of the observation.
    pub observation_quantity: ScientificQuantity,
    /// The law.
    pub law: ExternalLaw,
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= COHERENCE_TOLERANCE
}

fn all_probabilities(values: &[f64]) -> bool {
    values.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p))
}

fn distinct_finite(values: &[f64]) -> bool {
    let mut keys: Vec<u64> = values.iter().map(|v| state_key(*v)).collect();
    keys.sort_unstable();
    keys.windows(2).all(|w| w[0] != w[1]) && values.iter().all(|v| v.is_finite())
}

/// Bit key of a state or statistic with `-0.0` folded onto `0.0`.
pub(crate) fn state_key(value: f64) -> u64 {
    (value + 0.0).to_bits()
}

fn rectangular(rows: &[Vec<f64>], n_rows: usize, n_cols: usize) -> bool {
    rows.len() == n_rows && rows.iter().all(|r| r.len() == n_cols)
}

// The finite signal is indexed by state value. Repeated prior draws are valid,
// but they must describe the same conditional observation law.
fn repeated_states_agree(
    states: &[f64],
    rows: &[Vec<f64>],
    include_row: impl Fn(usize) -> bool,
) -> bool {
    let mut first = BTreeMap::new();
    for (column, &state) in states.iter().enumerate() {
        if let Some(&previous) = first.get(&state_key(state)) {
            if rows
                .iter()
                .enumerate()
                .any(|(index, row)| include_row(index) && !near(row[column], row[previous]))
            {
                return false;
            }
        } else {
            first.insert(state_key(state), column);
        }
    }
    true
}

fn validate_likelihood(
    states: &[f64],
    statistics: &[f64],
    probabilities: &[Vec<f64>],
) -> Result<(), SignalError> {
    if states.is_empty() || statistics.is_empty() {
        return Err(SignalError::InvalidTable("empty states or statistics"));
    }
    if !states.iter().all(|s| s.is_finite()) || !distinct_finite(statistics) {
        return Err(SignalError::InvalidTable("non-finite state or repeated statistic"));
    }
    if !rectangular(probabilities, statistics.len(), states.len())
        || !probabilities.iter().all(|row| all_probabilities(row))
    {
        return Err(SignalError::InvalidTable("probability table shape or range"));
    }
    for k in 0..states.len() {
        let mass: f64 = probabilities.iter().map(|row| row[k]).sum();
        if !near(mass, 1.0) {
            return Err(SignalError::LawIncoherent);
        }
    }
    if !repeated_states_agree(states, probabilities, |_| true) {
        return Err(SignalError::LawIncoherent);
    }
    Ok(())
}

fn validate_posterior(
    states: &[f64],
    statistics: &[f64],
    predictive: &[f64],
    posterior: &[Vec<f64>],
) -> Result<(), SignalError> {
    if states.is_empty() || statistics.is_empty() {
        return Err(SignalError::InvalidTable("empty states or statistics"));
    }
    if !states.iter().all(|s| s.is_finite()) || !distinct_finite(statistics) {
        return Err(SignalError::InvalidTable("non-finite state or repeated statistic"));
    }
    if predictive.len() != statistics.len()
        || !all_probabilities(predictive)
        || !rectangular(posterior, statistics.len(), states.len())
        || !posterior.iter().all(|row| all_probabilities(row))
    {
        return Err(SignalError::InvalidTable("posterior table shape or range"));
    }
    if !near(predictive.iter().sum::<f64>(), 1.0) {
        return Err(SignalError::LawIncoherent);
    }
    if !posterior.iter().all(|row| near(row.iter().sum::<f64>(), 1.0)) {
        return Err(SignalError::PosteriorIncoherent);
    }
    // Averaging the posteriors over the predictive law must recover the prior
    // (equal weight on every draw): sum_y P(y) P(state_k | y) = 1 / K.
    let prior_mass = 1.0 / states.len() as f64;
    for k in 0..states.len() {
        let recovered: f64 = predictive.iter().zip(posterior).map(|(p, row)| p * row[k]).sum();
        if !near(recovered, prior_mass) {
            return Err(SignalError::PosteriorIncoherent);
        }
    }
    // At an observation with positive predictive mass, identical states must
    // have identical posterior weight under the equal-draw prior.
    if !repeated_states_agree(states, posterior, |index| predictive[index] > 0.0) {
        return Err(SignalError::PosteriorIncoherent);
    }
    Ok(())
}

fn validate_values(values: &ExternalDecisionValues) -> Result<(), SignalError> {
    let n_actions = values.action_ids.len();
    if n_actions == 0 || values.branch_probabilities.is_empty() {
        return Err(SignalError::InvalidTable("empty branches or actions"));
    }
    let mut ids: Vec<&String> = values.action_ids.iter().collect();
    ids.sort();
    if values.action_ids.iter().any(|id| is_blank(id)) || ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(SignalError::InvalidTable("blank or duplicate action id"));
    }
    if !all_probabilities(&values.branch_probabilities)
        || !rectangular(&values.values, values.branch_probabilities.len(), n_actions)
        || !values.values.iter().all(|row| row.iter().all(|v| v.is_finite()))
    {
        return Err(SignalError::InvalidTable("decision-value table shape or range"));
    }
    if !near(values.branch_probabilities.iter().sum::<f64>(), 1.0) {
        return Err(SignalError::LawIncoherent);
    }
    Ok(())
}

impl ExternalSignalBody {
    fn validate(&self) -> Result<(), SignalError> {
        if self.sample_size == 0 {
            return Err(SignalError::SampleSizeInvalid);
        }
        if self.state_quantity.validate().is_err() || self.observation_quantity.validate().is_err()
        {
            return Err(SignalError::InvalidTable("state or observation coordinate"));
        }
        match &self.law {
            ExternalLaw::Likelihood { states, statistics, probabilities } => {
                validate_likelihood(states, statistics, probabilities)
            }
            ExternalLaw::Posterior { states, statistics, predictive, posterior } => {
                validate_posterior(states, statistics, predictive, posterior)
            }
            ExternalLaw::DecisionValues(values) => validate_values(values),
        }
    }

    fn support_size(&self) -> usize {
        match &self.law {
            ExternalLaw::Likelihood { statistics, .. }
            | ExternalLaw::Posterior { statistics, .. } => statistics.len(),
            ExternalLaw::DecisionValues(values) => values.branch_probabilities.len(),
        }
    }

    fn mode(&self) -> SignalUpdateMode {
        match &self.law {
            ExternalLaw::Likelihood { .. } => SignalUpdateMode::NativeUpdate,
            ExternalLaw::Posterior { .. } => SignalUpdateMode::ExternalPosterior,
            ExternalLaw::DecisionValues(_) => SignalUpdateMode::ExternalDecisionValues,
        }
    }

    fn digest(&self) -> String {
        let mut hasher = CanonicalWriter::new("antecedent.signal.external_law.v1");
        hasher.text(self.mode().as_str()).word(self.sample_size);
        self.state_quantity.write_canonical(&mut hasher);
        self.observation_quantity.write_canonical(&mut hasher);
        match &self.law {
            ExternalLaw::Likelihood { states, statistics, probabilities } => {
                hasher.reals(states).reals(statistics);
                rows(&mut hasher, probabilities);
            }
            ExternalLaw::Posterior { states, statistics, predictive, posterior } => {
                hasher.reals(states).reals(statistics).reals(predictive);
                rows(&mut hasher, posterior);
            }
            ExternalLaw::DecisionValues(values) => {
                hasher.reals(&values.branch_probabilities).texts(&values.action_ids);
                rows(&mut hasher, &values.values);
            }
        }
        hasher.finish_hex()
    }
}

/// Finite external observation law updated natively over the prior draws.
struct ExternalFiniteSignal {
    n: u64,
    statistics: Vec<f64>,
    probabilities: Vec<Vec<f64>>,
    state_index: BTreeMap<u64, usize>,
    stat_index: BTreeMap<u64, usize>,
}

impl ExternalFiniteSignal {
    fn new(states: &[f64], statistics: &[f64], probabilities: Vec<Vec<f64>>, n: u64) -> Self {
        let mut state_index = BTreeMap::new();
        for (k, s) in states.iter().enumerate() {
            state_index.entry(state_key(*s)).or_insert(k);
        }
        let stat_index = statistics.iter().enumerate().map(|(y, s)| (state_key(*s), y)).collect();
        Self { n, statistics: statistics.to_vec(), probabilities, state_index, stat_index }
    }

    fn column(&self, state: f64) -> Result<usize, DesignError> {
        self.state_index.get(&state_key(state)).copied().ok_or_else(|| {
            DesignError::Shape(format!(
                "state {state} is not one of the draws the external law was supplied for"
            ))
        })
    }

    fn check_n(&self, n: u64) -> Result<(), DesignError> {
        if n == self.n {
            Ok(())
        } else {
            Err(DesignError::Shape(format!(
                "external law supplied for n = {}, evaluated at n = {n}",
                self.n
            )))
        }
    }

    fn probability(&self, row: usize, col: usize) -> Result<f64, DesignError> {
        self.probabilities
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .ok_or_else(|| DesignError::Shape("external law table index".into()))
    }
}

impl DecisionSignal<f64> for ExternalFiniteSignal {
    fn name(&self) -> &'static str {
        "external_finite_law"
    }

    fn sample_size(&self, _candidate: &CandidateDesign) -> Option<u64> {
        Some(self.n)
    }

    fn finite_support(&self, n: u64) -> Option<Vec<f64>> {
        (n == self.n).then(|| self.statistics.clone())
    }

    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        self.check_n(n)?;
        let Some(&row) = self.stat_index.get(&state_key(statistic)) else {
            out[..states.len()].fill(f64::NEG_INFINITY);
            return Ok(());
        };
        for (slot, &theta) in out.iter_mut().zip(states) {
            *slot = self.probability(row, self.column(theta)?)?.ln();
        }
        Ok(())
    }

    fn sample_statistic(
        &self,
        state: &f64,
        n: u64,
        rng: &mut CausalRng,
    ) -> Result<f64, DesignError> {
        self.check_n(n)?;
        let col = self.column(*state)?;
        let u = rng.next_f64();
        let mut cumulative = 0.0;
        for (row, statistic) in self.statistics.iter().enumerate() {
            cumulative += self.probability(row, col)?;
            if u < cumulative {
                return Ok(*statistic);
            }
        }
        self.statistics
            .last()
            .copied()
            .ok_or_else(|| DesignError::Shape("external law has no observations".into()))
    }
}

/// An external candidate signal and update, supplied with its provider object and
/// trust state. It supports native update from a supplied likelihood and externally
/// computed posterior or decision values; the update mode and trust are retained on
/// the receipt and the exact request fingerprint must match.
#[derive(Clone, Debug)]
pub struct ExternalSignal {
    object: ExternalScientificObject,
    trust: ExternalTrustState,
    body: ExternalSignalBody,
}

impl ExternalSignal {
    /// Bind a payload to a provider object and trust state.
    ///
    /// # Errors
    ///
    /// An object that is not a valid signal declaration, native trust claimed,
    /// verification covering another contract, a missing capability (`Factor` for a
    /// supplied likelihood, `Update` for a supplied posterior or values), a payload
    /// that disagrees with the object's observation coordinate, or an incoherent or
    /// malformed table.
    pub fn new(
        object: ExternalScientificObject,
        trust: ExternalTrustState,
        body: ExternalSignalBody,
    ) -> Result<Self, SignalError> {
        object.validate().map_err(|_| SignalError::InvalidObject)?;
        let ExternalScientificObject::Signal(contract) = &object else {
            return Err(SignalError::InvalidObject);
        };
        match &trust {
            ExternalTrustState::NativeLicensed => return Err(SignalError::TrustNativeClaimed),
            ExternalTrustState::ExternallyAttested { attestor } => {
                if is_blank(attestor) {
                    return Err(SignalError::TrustMismatch);
                }
            }
            ExternalTrustState::ExactRequestVerified(_) => {
                if !trust.verifies(&object) {
                    return Err(SignalError::TrustMismatch);
                }
            }
        }
        let needed = match body.law {
            ExternalLaw::Likelihood { .. } => ExternalCapability::Factor,
            _ => ExternalCapability::Update,
        };
        object.require_capability(needed).map_err(|_| SignalError::CapabilityMissing(needed))?;
        body.validate()?;
        if contract.observation.require_same_coordinate(&body.observation_quantity).is_err() {
            return Err(SignalError::ObservationQuantityMismatch);
        }
        Ok(Self { object, trust, body })
    }

    /// Update mode of the supplied law.
    #[must_use]
    pub fn update_mode(&self) -> SignalUpdateMode {
        self.body.mode()
    }

    fn trust_label(&self) -> (SignalTrustLabel, Option<String>) {
        match &self.trust {
            ExternalTrustState::ExternallyAttested { attestor } => {
                (SignalTrustLabel::ExternallyAttested, Some(attestor.clone()))
            }
            // Native trust is refused at construction; map defensively to the
            // weakest external label rather than ever claiming native.
            ExternalTrustState::ExactRequestVerified(_) => {
                (SignalTrustLabel::ExactRequestVerified, None)
            }
            ExternalTrustState::NativeLicensed => (SignalTrustLabel::ExternallyAttested, None),
        }
    }

    fn check_alignment(&self, request: &SignalRequest) -> Result<(), SignalError> {
        let ExternalScientificObject::Signal(contract) = &self.object else {
            return Err(SignalError::InvalidObject);
        };
        if contract.identity.request_id != request.fingerprint() {
            return Err(SignalError::RequestFingerprintMismatch);
        }
        if contract.candidate_id != request.candidate_id {
            return Err(SignalError::CandidateMismatch);
        }
        if contract.prior_id != request.prior_id {
            return Err(SignalError::PriorMismatch);
        }
        if self.body.sample_size != request.sample_size {
            return Err(SignalError::SampleSizeMismatch {
                declared: self.body.sample_size,
                requested: request.sample_size,
            });
        }
        if contract.observation.require_same_coordinate(&request.observation_quantity).is_err()
            || self
                .body
                .observation_quantity
                .require_same_coordinate(&request.observation_quantity)
                .is_err()
        {
            return Err(SignalError::ObservationQuantityMismatch);
        }
        if self.body.state_quantity.require_same_coordinate(&request.state_quantity).is_err() {
            return Err(SignalError::PosteriorQuantityMismatch);
        }
        if self.body.support_size() > request.limits.max_support {
            return Err(SignalError::ComputationalLimit("support"));
        }
        Ok(())
    }
}

impl SignalProvider for ExternalSignal {
    fn prepare(&self, request: &SignalRequest) -> Result<PreparedSignal, SignalError> {
        request.validate()?;
        self.check_alignment(request)?;
        let (law, bound_states) = match &self.body.law {
            ExternalLaw::Likelihood { states, statistics, probabilities } => {
                let signal = ExternalFiniteSignal::new(
                    states,
                    statistics,
                    probabilities.clone(),
                    self.body.sample_size,
                );
                (PreparedLaw::Likelihood(Arc::new(signal)), Some(states.clone()))
            }
            ExternalLaw::Posterior { states, statistics, predictive, posterior } => {
                // P(y | θ_k) = K · P(y) · P(θ_k | y) for equally weighted draws.
                let k = states.len() as f64;
                let probabilities: Vec<Vec<f64>> = predictive
                    .iter()
                    .zip(posterior)
                    .map(|(p, row)| row.iter().map(|q| k * p * q).collect())
                    .collect();
                let signal = ExternalFiniteSignal::new(
                    states,
                    statistics,
                    probabilities,
                    self.body.sample_size,
                );
                (PreparedLaw::Likelihood(Arc::new(signal)), Some(states.clone()))
            }
            ExternalLaw::DecisionValues(values) => {
                (PreparedLaw::DecisionValues(values.clone()), None)
            }
        };
        let (trust, attestor) = self.trust_label();
        let mut lineage = request.evidence_lineage.clone();
        lineage.sort();
        Ok(PreparedSignal {
            law,
            receipt: SignalReceipt {
                request_fingerprint: request.fingerprint(),
                candidate_id: request.candidate_id.clone(),
                prior_id: request.prior_id.clone(),
                sample_size: request.sample_size,
                rng_seed: request.rng_seed,
                family: "external_finite_law".to_owned(),
                update_mode: self.body.mode(),
                trust,
                attestor,
                provider_identity: Some(self.object.identity().clone()),
                evidence_lineage: lineage,
                conditional_independence: request.conditional_independence.clone(),
                limits: request.limits,
                law_digest: self.body.digest(),
            },
            bound_states,
        })
    }
}
