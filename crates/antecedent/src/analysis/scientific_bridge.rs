//! Typed bridges between scientific quantities: prepare, derive, inspect,
//! export and independently consume.
//!
//! A source claim answers a question about one quantity; a downstream consumer
//! asks about another. This module routes one request through the layers that
//! own each rule and adds no scientific rule of its own:
//!
//! 1. **Prepare** ([`PreparedScientificBridge::prepare`]) checks, before any
//!    number is formed, that the contract is well formed, that its class pair
//!    is one the kind connects, that the principal assumption of the kind is
//!    declared and names its evidence, that the mapping is one the kind allows
//!    and agrees with its own binding (a measurement model's variables,
//!    identity and population), that the source and target agree in population,
//!    regime, horizon and conditioning, and that evidence established in
//!    another population or regime than the claim's carries a declared
//!    transportability premise. Each refusal is a typed
//!    `scientific_bridge.*` detail with the declaration that would license it.
//! 2. **Derive** ([`PreparedScientificBridge::derive`]) maps the source value
//!    and returns a [`ScientificBridgeResult`]: a derived claim with its own
//!    identity (binding the contract, its assumptions and the source claim),
//!    its own provenance chain and receipt. The source claim is not modified.
//!    The derived claim is a point mean. Where the source claim declared a standard
//!    error and the contract declares the mapping's uncertainty and its dependence on
//!    the source, the result also carries the first-order propagated variance and mean
//!    ([`ScientificBridgeResult::propagated_uncertainty`]), read as an uncalibrated
//!    spread beside the plug-in value; the dependence is never assumed. Its interval
//!    ([`ScientificBridgeResult::bridged_interval`]) needs the source claim's declared
//!    standard error and is published only inside a measured design; an unscoped
//!    request, a posterior or a joint law refuses with `cell_not_licensed`, and a
//!    consumer that requires a predictive law, posterior, identified set or
//!    joint outcome law refuses with `bridge_substitution_refused`.
//! 3. **Export** ([`ScientificBridgeResult::export`]) writes
//!    `scientific_bridge_v1`: the contract, the source claim as the bridge saw
//!    it, and the derived identity, value and fit.
//! 4. **Consume** ([`consume_scientific_bridge_artifact`]) rebuilds the
//!    contract and the source claim, applies every licensing check again,
//!    derives the claim again and requires the stored identity, value bits,
//!    fit and transport flag to equal the derivation, against an artifact
//!    identity the consumer retained independently of the bytes.
//!
//! # Lifecycle
//!
//! * [`PreparedScientificBridge::derive`] with a new source claim about the
//!   same quantity (a refreshed estimate under the same program) **keeps** the
//!   contract, its identity, the declared assumptions, the licensing verdict
//!   and the transport flag, and **recomputes** the value, the trial-level fit,
//!   the derived claim identity and the provenance. It **refuses** a claim
//!   about another quantity or in other units, and a claim that is not a
//!   finite point value.
//! * [`PreparedScientificBridge::with_contract`] is the only way the
//!   assumptions, mapping or scope change. It **rebuilds** the contract
//!   identity (so every derived claim identity), re-runs every prepare check
//!   and refuses what no longer licenses; a claim derived under the old
//!   contract is not carried over.
//! * Both take `&self` and return a new value: a refused derivation or
//!   replacement leaves the prepared bridge exactly as it was.
//! * A source claim withdrawn or superseded makes the derived claim
//!   inapplicable ([`ScientificBridgeResult::freshness`]); the derived claim's
//!   own history is preserved.
//!
//! # Bridges of distributions
//!
//! A bridge whose source is the aligned draws of a posterior or a predictive law
//! ([`PreparedScientificBridgeDistribution`]) derives a claim whose value is the
//! same draw table with the bridged column in place of the source column: row for
//! row, with the table's alignment identity unchanged
//! ([`ScientificBridgeDistributionResult`]). The mapping adds the spread it
//! declares: a declared affine map with uncertain coefficients draws them jointly
//! from its declared law and the declared dependence on the source; a trial-level
//! line draws its coefficients from the Student-t law of its least-squares fit; a
//! measurement correction draws its rates. The source's spread comes only from its
//! draws and the mapping's only from the mapping, so neither is counted twice.
//! The result is read as an uncalibrated spread; its interval is published only
//! through the measured-coverage gate of a bridged mean, and a bridge expressed as
//! a measurement-mapping stage of a propagation chain
//! ([`ScientificBridgeDistributionResult::chain_stage`]) propagates its
//! coefficients jointly with the stages that read it.
//!
//! # What is closed
//!
//! A model score offered as a probability, and a predictive importance offered
//! as a causal effect, refuse under their own names whatever is declared: no
//! calibration bridge for a score and no route from an importance to a mechanism
//! is licensed. An associational claim is never a bridge source. An identified
//! set and a joint outcome law have no executed bridge.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    BridgeContract, BridgeKind, BridgeMapping, BridgeRefusal, BridgeSource, BridgedClaim,
    BridgedDistributionClaim, Licensing, MappingKind, NonlinearMap, PointPropagation,
    TrialLevelFit, ValueKind, derive_bridged_claim_propagating, point_propagation_unavailable,
};
use antecedent_contract::{ClaimEnvelope, ClaimFreshness, ClaimKind};
use antecedent_core::identity::SemanticDigest;
use antecedent_io::IoError;
use antecedent_io::bridge_artifact::{
    ScientificBridgeAny, ScientificBridgeArtifact, ScientificBridgeArtifactError,
    ScientificBridgeDrawsArtifact,
};
use antecedent_io::bridge_distribution::bridge_draws;
pub use antecedent_io::bridge_distribution::{
    AffineColumns, BridgeChainStage, BridgedDistribution,
};
use antecedent_io::bridge_distribution_interval::bridged_distribution_interval;
pub use antecedent_io::bridge_interval::BridgedClaimInterval;
use antecedent_io::distribution_artifact::DistributionArtifact;

/// One refusal as plain data: registered reason code, namespaced
/// `scientific_bridge.*` detail and the explanation of what was refused and
/// what would license it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScientificBridgeRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `scientific_bridge.<slot>`.
    pub detail: String,
    /// What was refused and what would license it.
    pub message: String,
    /// The declaration or evidence that would license the request, when known.
    pub remedy: Option<String>,
}

impl std::fmt::Display for ScientificBridgeRefusal {
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

impl std::error::Error for ScientificBridgeRefusal {}

impl From<ScientificBridgeRefusal> for IoError {
    fn from(refusal: ScientificBridgeRefusal) -> Self {
        Self::Refused {
            code: refusal.code,
            message: format!("{}: {}", refusal.detail, refusal.message),
        }
    }
}

impl From<ScientificBridgeRefusal> for crate::CausalError {
    fn from(refusal: ScientificBridgeRefusal) -> Self {
        Self::Serialization(refusal.into())
    }
}

impl ScientificBridgeRefusal {
    fn from_contract(refusal: &BridgeRefusal) -> Self {
        let mut message = format!("refused at stage `{}`", refusal.stage);
        if let Some(offending) = &refusal.offending {
            message.push_str(&format!("; offending `{offending}`"));
        }
        if let Some(expected) = &refusal.expected {
            message.push_str(&format!("; expected `{expected}`"));
        }
        if let Some(supplied) = &refusal.supplied {
            message.push_str(&format!("; supplied `{supplied}`"));
        }
        if let Some(remedy) = refusal.remedy {
            message.push_str(&format!("; {remedy}"));
        }
        Self {
            code: refusal.code,
            detail: refusal.detail.clone(),
            message,
            remedy: refusal.remedy.map(str::to_owned),
        }
    }

    fn from_artifact(error: &ScientificBridgeArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, message)) => {
                // Every refusal states what would license the request: the artifact's own
                // refusals name theirs; a library refusal on replay carries it as its message.
                let remedy = match detail.as_str() {
                    "scientific_bridge.identity_mismatch" => {
                        "supply the artifact whose recomputed identity equals the retained identity, or regenerate it from the intended bridge"
                            .to_owned()
                    }
                    "scientific_bridge.replay_mismatch" => {
                        "regenerate the artifact from the contract and the source claim; the stored identity, value, fit and flag must replay"
                            .to_owned()
                    }
                    _ => message.clone(),
                };
                Self { code, detail, message, remedy: Some(remedy) }
            }
            None => Self {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "scientific_bridge.artifact_invalid".to_owned(),
                message: error.to_string(),
                remedy: Some("supply an intact scientific_bridge_v1 artifact".to_owned()),
            },
        }
    }
}

/// A bridge whose contract was checked once and whose licensing verdict is
/// kept, ready to derive claims from source claims about its source quantity.
#[derive(Clone, Debug)]
pub struct PreparedScientificBridge {
    contract: BridgeContract,
    contract_identity: SemanticDigest,
    licensing: Licensing,
}

impl PreparedScientificBridge {
    /// Check a contract and keep its verdict.
    ///
    /// # Errors
    /// The first defect of the contract, as a coded refusal that names the
    /// declaration or evidence that would license it.
    pub fn prepare(contract: BridgeContract) -> Result<Self, ScientificBridgeRefusal> {
        let licensing = contract
            .check_licensing()
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))?;
        let contract_identity = contract.identity();
        Ok(Self { contract, contract_identity, licensing })
    }

    /// The contract.
    #[must_use]
    pub const fn contract(&self) -> &BridgeContract {
        &self.contract
    }

    /// The identity of the contract.
    #[must_use]
    pub const fn contract_identity(&self) -> SemanticDigest {
        self.contract_identity
    }

    /// Whether the bridge carries evidence established elsewhere than the
    /// claim's population or regime, under a declared premise.
    #[must_use]
    pub const fn transported(&self) -> bool {
        self.licensing.transported
    }

    /// Derive the claim for `source`.
    ///
    /// # Errors
    /// A source claim about another quantity or in other units, a source that
    /// is not a finite point value, or a value outside the mapping's validated
    /// or attainable range.
    pub fn derive(
        &self,
        source: &BridgeSource<'_>,
    ) -> Result<ScientificBridgeResult, ScientificBridgeRefusal> {
        self.derive_with_standard_error(source, None)
    }

    /// Derive the claim for `source` whose estimate carries `standard_error`.
    ///
    /// The standard error is the source claim's own sampling uncertainty, declared with
    /// the claim. It is what [`ScientificBridgeResult::bridged_interval`] propagates and,
    /// when the contract declares the mapping's uncertainty and its dependence on the
    /// source, what the propagated variance is built from; then it is part of the derived
    /// claim's identity and of the artifact. A bridge that declares no mapping
    /// uncertainty and no dependence holds it beside the result only.
    ///
    /// # Errors
    /// As [`Self::derive`]; a standard error that is not finite and non-negative; a
    /// declared mapping uncertainty with no declared dependence on the source (or the
    /// reverse), which is never assumed independent.
    pub fn derive_with_standard_error(
        &self,
        source: &BridgeSource<'_>,
        standard_error: Option<f64>,
    ) -> Result<ScientificBridgeResult, ScientificBridgeRefusal> {
        self.derive_propagating(source, standard_error, None)
    }

    /// Derive the claim for `source` with its propagated uncertainty, and the
    /// second-moment expectation of a nonlinear map of the bridged quantity.
    ///
    /// The claim value is the plug-in value of the mapping. `nonlinear` asks for
    /// `E[g(X)]` under the propagated mean and variance (the bias `E[g(X)] - g(E[X])` is
    /// reported, not dropped); a nonlinear map with no declared standard error is the map
    /// of a mean and refuses, and so does a map whose law the propagated moments do not
    /// determine.
    ///
    /// # Errors
    /// As [`Self::derive_with_standard_error`].
    pub fn derive_propagating(
        &self,
        source: &BridgeSource<'_>,
        standard_error: Option<f64>,
        nonlinear: Option<NonlinearMap>,
    ) -> Result<ScientificBridgeResult, ScientificBridgeRefusal> {
        let bridged =
            derive_bridged_claim_propagating(&self.contract, source, standard_error, nonlinear)
                .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))?;
        Ok(ScientificBridgeResult {
            bridged,
            source: source.claim.clone(),
            source_standard_error: standard_error,
        })
    }

    /// A prepared bridge for a replacement contract, re-checked in full.
    ///
    /// # Errors
    /// The first defect of the replacement; `self` is unchanged.
    pub fn with_contract(&self, contract: BridgeContract) -> Result<Self, ScientificBridgeRefusal> {
        Self::prepare(contract)
    }
}

/// The result of a derivation: the bridged claim and the source claim it was
/// derived from, both inspectable.
#[derive(Clone, Debug)]
pub struct ScientificBridgeResult {
    bridged: BridgedClaim,
    source: ClaimEnvelope,
    source_standard_error: Option<f64>,
}

/// What an inspection of a bridged claim reports.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeInspection {
    /// The bridge kind.
    pub kind: BridgeKind,
    /// The source value.
    pub source_value: f64,
    /// The bridged value.
    pub value: f64,
    /// The kind of object the value is.
    pub value_kind: ValueKind,
    /// The mapping that carried the value.
    pub mapping: MappingKind,
    /// The trial-level fit, when the mapping fitted one.
    pub fit: Option<TrialLevelFit>,
    /// Identity of the source claim.
    pub source_claim_id: String,
    /// Identity of the bridged claim.
    pub claim_id: String,
    /// Identity of the contract.
    pub contract_identity: String,
    /// Declared assumptions as `(identity, kind, evidence)`.
    pub assumptions: Vec<(String, String, Option<String>)>,
    /// The transportability premise identity, when the bridge is transported.
    pub transportability_premise: Option<String>,
    /// Whether the bridge carries evidence from another scope.
    pub transported: bool,
    /// Operations the derived claim refuses.
    pub unavailable: Vec<String>,
}

impl std::fmt::Display for BridgeInspection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "scientific bridge {} ({} -> {}) by {} mapping",
            self.kind.as_str(),
            self.source_value,
            self.value,
            self.mapping.as_str()
        )?;
        writeln!(
            f,
            "  claim {} derived from source claim {}",
            self.claim_id, self.source_claim_id
        )?;
        writeln!(
            f,
            "  contract {}; value kind {}",
            self.contract_identity,
            self.value_kind.as_str()
        )?;
        for (id, kind, evidence) in &self.assumptions {
            writeln!(f, "  assumption {id} ({kind}) on {}", evidence.as_deref().unwrap_or("-"))?;
        }
        if let Some(premise) = &self.transportability_premise {
            writeln!(f, "  transported under premise {premise}")?;
        }
        write!(f, "  unavailable: {}", self.unavailable.join(", "))
    }
}

/// An exported bridge.
#[derive(Clone, Debug, PartialEq)]
pub struct ScientificBridgeExport {
    /// The artifact bytes.
    pub bytes: Vec<u8>,
    /// BLAKE3 identity (hex) of the artifact; a consumer that retains it
    /// independently refuses a consistently resealed artifact of another
    /// bridge.
    pub identity: String,
}

impl ScientificBridgeResult {
    /// The bridged claim.
    #[must_use]
    pub const fn bridged(&self) -> &BridgedClaim {
        &self.bridged
    }

    /// The derived claim envelope.
    #[must_use]
    pub const fn claim(&self) -> &ClaimEnvelope {
        &self.bridged.claim
    }

    /// The source claim, unchanged and separately inspectable.
    #[must_use]
    pub const fn source_claim(&self) -> &ClaimEnvelope {
        &self.source
    }

    /// The bridged value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.bridged.value
    }

    /// A summary of the derivation.
    #[must_use]
    pub fn inspect(&self) -> BridgeInspection {
        let bridged = &self.bridged;
        BridgeInspection {
            kind: bridged.kind(),
            source_value: bridged.source_value,
            value: bridged.value,
            value_kind: bridged.value_kind,
            mapping: bridged.mapping,
            fit: bridged.fit,
            source_claim_id: bridged.source_claim_id.to_hex(),
            claim_id: bridged.claim.claim_id.to_hex(),
            contract_identity: bridged.contract_identity.to_hex(),
            assumptions: bridged
                .assumptions()
                .iter()
                .map(|a| (a.id.clone(), a.kind.as_str().to_owned(), a.evidence.clone()))
                .collect(),
            transportability_premise: bridged.contract.transport.as_ref().map(|p| p.id.clone()),
            transported: bridged.transported,
            unavailable: bridged
                .receipt
                .unavailable_operations
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
    }

    /// Require the kind of object a consumer needs.
    ///
    /// # Errors
    /// `bridge_substitution_refused` when the consumer needs a predictive law,
    /// posterior, identified set or joint outcome law: the derived claim is a
    /// mean, and none of these is interchangeable with it.
    pub fn require_value_kind(&self, requested: ValueKind) -> Result<(), ScientificBridgeRefusal> {
        self.bridged
            .require_value_kind(requested)
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))
    }

    /// The source claim's standard error declared with it, when there is one.
    #[must_use]
    pub const fn source_standard_error(&self) -> Option<f64> {
        self.source_standard_error
    }

    /// The propagated uncertainty of the bridged value, when there is one: the first-order
    /// variance and mean of the mapping applied to the source value and its declared
    /// standard error, with the mapping uncertainty and dependence it was built from.
    #[must_use]
    pub const fn propagation(&self) -> Option<&PointPropagation> {
        self.bridged.propagation.as_ref()
    }

    /// The propagated uncertainty of the bridged value. The claim value is the plug-in
    /// value; the variance is an uncalibrated spread (calibration unmeasured), never an
    /// interval, and the propagated mean is reported beside the value.
    ///
    /// # Errors
    /// `source_variance_invalid` when the source claim declared no standard error;
    /// `mapping_uncertainty_missing` when an affine map declared neither uncertainty nor
    /// dependence; `source_dependence_missing` when the dependence the propagation needs
    /// is undeclared: the bridge never assumes the source and the mapping independent.
    pub fn propagated_uncertainty(&self) -> Result<&PointPropagation, ScientificBridgeRefusal> {
        self.bridged.propagation.as_ref().ok_or_else(|| {
            ScientificBridgeRefusal::from_contract(&point_propagation_unavailable(
                &self.bridged.contract,
                self.source_standard_error.is_some(),
            ))
        })
    }

    /// The propagated variance of the bridged value, when there is one.
    #[must_use]
    pub fn propagated_variance(&self) -> Option<f64> {
        self.bridged.propagation.as_ref().map(|propagation| propagation.variance)
    }

    /// The propagated mean of the bridged value, when there is one: bias-corrected when a
    /// nonlinear map was requested.
    #[must_use]
    pub fn propagated_mean(&self) -> Option<f64> {
        self.bridged.propagation.as_ref().map(PointPropagation::propagated_mean)
    }

    /// The recorded uncertainty components of the derived claim, as `(source, target,
    /// omitted)`; a propagated variance is the component labelled `uncalibrated_spread`.
    #[must_use]
    pub fn components(&self) -> Vec<(&'static str, String, bool)> {
        match &self.bridged.claim.reasoning.uncertainty {
            antecedent_contract::SlotAvailability::Available(slot) => slot
                .components
                .iter()
                .map(|c| (c.source.as_str(), c.target.to_string(), c.omitted))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The interval of the bridged value at `level`: the source claim's declared
    /// standard error carried through the mapping together with the mapping's own
    /// estimation uncertainty.
    ///
    /// Published only where a current attesting coverage record measured this
    /// construction at this bridge kind, mapping and direction (and validated trial
    /// count); the record is resolved from the shipped coverage table.
    ///
    /// # Errors
    /// `cell_not_licensed` with `scientific_bridge.interval_source_uncertainty_required`
    /// when the source claim declared no standard error, and with
    /// `scientific_bridge.interval_scope_outside_measured` outside a measured record.
    pub fn bridged_interval(
        &self,
        level: f64,
    ) -> Result<BridgedClaimInterval, ScientificBridgeRefusal> {
        if let (BridgeMapping::Identity | BridgeMapping::Affine { .. }, Some(uncertainty)) =
            (&self.bridged.contract.mapping, &self.bridged.contract.mapping_uncertainty)
        {
            if uncertainty.intercept_variance > 0.0 || uncertainty.slope_variance > 0.0 {
                return Err(ScientificBridgeRefusal {
                    code: antecedent_core::reason_code!("cell_not_licensed"),
                    detail: "scientific_bridge.interval_scope_outside_measured".to_owned(),
                    message: format!(
                        "the affine map declares uncertainty `{}`: it is no exact transform of the \
                         source interval, and its propagated variance is an uncalibrated spread \
                         that no coverage record measured",
                        uncertainty.id
                    ),
                    remedy: Some(
                        "read the propagated variance as an uncalibrated spread, or ask for the \
                         interval of a bridge inside a measured design"
                            .to_owned(),
                    ),
                });
            }
        }
        antecedent_io::bridge_interval::bridged_claim_interval(
            &self.bridged,
            self.source_standard_error,
            level,
        )
        .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))
    }

    /// An unscoped request for an interval, posterior or joint law of the bridged claim.
    ///
    /// # Errors
    /// Always `cell_not_licensed` under `scientific_bridge.interval_not_licensed`: name
    /// the level and ask [`Self::bridged_interval`] for the interval of the value; a
    /// posterior and a joint law have no executed route.
    pub fn interval(&self) -> Result<(), ScientificBridgeRefusal> {
        self.bridged
            .request_interval()
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))
    }

    /// Freshness of the derived claim given the freshness of its source.
    #[must_use]
    pub fn freshness(&self, source: &ClaimFreshness) -> ClaimFreshness {
        self.bridged.freshness(source)
    }

    /// Write the `scientific_bridge_v1` artifact.
    ///
    /// # Errors
    /// An encoding failure or a bridge that does not seal.
    pub fn export(
        &self,
        artifact_id: &str,
    ) -> Result<ScientificBridgeExport, ScientificBridgeRefusal> {
        let source =
            BridgeSource { claim: &self.source, quantity: &self.bridged.contract.source.quantity };
        let (sealed, _) = ScientificBridgeArtifact::seal(&self.bridged, &source)
            .map_err(|e| ScientificBridgeRefusal::from_artifact(&e))?;
        let bytes = ScientificBridgeArtifact::to_bytes(&self.bridged, &source, artifact_id)
            .map_err(|e| ScientificBridgeRefusal::from_artifact(&e))?;
        Ok(ScientificBridgeExport { bytes, identity: sealed.identity().to_owned() })
    }
}

/// A bridge consumed from an artifact.
#[derive(Clone, Debug)]
pub struct ConsumedScientificBridge {
    /// The replayed derivation.
    pub bridged: BridgedClaim,
    /// The source claim as the bridge saw it.
    pub source: ClaimEnvelope,
    /// The artifact identity (hex).
    pub identity: String,
}

/// Consume a scientific bridge artifact by replaying its derivation.
///
/// `expected_identity` is an artifact identity (hex) the consumer retained
/// independently; when given, a consistently resealed artifact of another
/// bridge is refused.
///
/// # Errors
/// Corruption, another major version, a bound exceeded, a contract the library
/// refuses on replay, a stored identity, value, fit or flag that does not
/// replay, or an identity that does not match.
pub fn consume_scientific_bridge_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> Result<ConsumedScientificBridge, ScientificBridgeRefusal> {
    let artifact = ScientificBridgeArtifact::from_bytes(bytes, expected_identity)
        .map_err(|e| ScientificBridgeRefusal::from_artifact(&e))?;
    Ok(ConsumedScientificBridge {
        bridged: artifact.bridged().clone(),
        source: artifact.source_claim().clone(),
        identity: artifact.identity().to_owned(),
    })
}

// ---------------------------------------------------------------------------
// Bridges of distributions
// ---------------------------------------------------------------------------

/// A bridge of draws whose contract was checked once: the source is the aligned
/// draws of a posterior or a predictive law, and the mapping declares what it
/// adds to their spread.
#[derive(Clone, Debug)]
pub struct PreparedScientificBridgeDistribution {
    contract: BridgeContract,
    contract_identity: SemanticDigest,
    licensing: Licensing,
}

impl PreparedScientificBridgeDistribution {
    /// Check a contract that bridges a posterior or a predictive law.
    ///
    /// # Errors
    /// The first defect of the contract: those of a bridged mean, a value kind
    /// that is neither a posterior nor a predictive law (or that differs between
    /// the source and the target), an uncertain map that declares no dependence
    /// on its source, or a mapping that declares its uncertainty a second time.
    pub fn prepare(contract: BridgeContract) -> Result<Self, ScientificBridgeRefusal> {
        let licensing = contract
            .check_draws_licensing()
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))?;
        let contract_identity = contract.identity();
        Ok(Self { contract, contract_identity, licensing })
    }

    /// The contract.
    #[must_use]
    pub const fn contract(&self) -> &BridgeContract {
        &self.contract
    }

    /// The identity of the contract.
    #[must_use]
    pub const fn contract_identity(&self) -> SemanticDigest {
        self.contract_identity
    }

    /// Whether the bridge carries evidence established elsewhere than the claim's
    /// population or regime, under a declared premise.
    #[must_use]
    pub const fn transported(&self) -> bool {
        self.licensing.transported
    }

    /// Bridge the draws in `source`.
    ///
    /// `source` is the aligned table with a column about the contract's source
    /// quantity; `claim_kind` is the kind of the claim the draws come from, when
    /// they come from one (an associational claim is never a bridge source);
    /// `seed` selects the random stream of the mapping's coefficients, so the
    /// result is a function of the contract, the table and the seed.
    ///
    /// # Errors
    /// Draws of another quantity, of a meaning that is neither a posterior nor a
    /// predictive law or that is not the contract's value kind, independent
    /// marginals, a source claim that is not about draws, draws outside the
    /// range the mapping is licensed for, a dependence no joint law has, or a
    /// mapped draw that is not finite.
    pub fn derive(
        &self,
        source: &DistributionArtifact,
        claim_kind: Option<ClaimKind>,
        seed: u64,
    ) -> Result<ScientificBridgeDistributionResult, ScientificBridgeRefusal> {
        let bridged = bridge_draws(&self.contract, source, claim_kind, seed)
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))?;
        Ok(ScientificBridgeDistributionResult { bridged })
    }

    /// A prepared bridge for a replacement contract, re-checked in full.
    ///
    /// # Errors
    /// The first defect of the replacement; `self` is unchanged.
    pub fn with_contract(&self, contract: BridgeContract) -> Result<Self, ScientificBridgeRefusal> {
        Self::prepare(contract)
    }
}

/// What an inspection of a bridged distribution reports.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributionBridgeInspection {
    /// The bridge kind.
    pub kind: BridgeKind,
    /// The mapping that carried the draws.
    pub mapping: MappingKind,
    /// The kind of object the bridged draws are.
    pub value_kind: ValueKind,
    /// Rows mapped.
    pub n_draws: usize,
    /// Mean of the source column.
    pub source_mean: f64,
    /// Variance of the source column.
    pub source_variance: f64,
    /// Mean of the bridged column.
    pub mean: f64,
    /// Variance of the bridged column.
    pub variance: f64,
    /// The random stream of the mapping's coefficients.
    pub stream: u64,
    /// The pairing identity both tables share.
    pub alignment_identity: String,
    /// Digest of the mapped draw table.
    pub draws_digest: String,
    /// Identity of the source draws.
    pub source_claim_id: String,
    /// Identity of the bridged claim.
    pub claim_id: String,
    /// Identity of the contract.
    pub contract_identity: String,
    /// Declared assumptions as `(identity, kind, evidence)`.
    pub assumptions: Vec<(String, String, Option<String>)>,
    /// Whether the bridge carries evidence from another scope.
    pub transported: bool,
    /// How the spread of the draws is read.
    pub reading: &'static str,
    /// Operations the derived claim refuses.
    pub unavailable: Vec<String>,
}

/// The result of bridging draws: the derived claim and the aligned draw table.
#[derive(Clone, Debug)]
pub struct ScientificBridgeDistributionResult {
    bridged: BridgedDistribution,
}

impl From<BridgedDistribution> for ScientificBridgeDistributionResult {
    fn from(bridged: BridgedDistribution) -> Self {
        Self { bridged }
    }
}

impl ScientificBridgeDistributionResult {
    /// The bridged distribution: claim, source table and bridged table.
    #[must_use]
    pub const fn bridged(&self) -> &BridgedDistribution {
        &self.bridged
    }

    /// The derived claim and what the bridge retained of its derivation.
    #[must_use]
    pub const fn claim(&self) -> &BridgedDistributionClaim {
        self.bridged.claim()
    }

    /// The derived claim envelope.
    #[must_use]
    pub const fn envelope(&self) -> &ClaimEnvelope {
        self.bridged.envelope()
    }

    /// The bridged draw table: the source table with the bridged column in place
    /// of the source column, row for row and with the same alignment identity.
    #[must_use]
    pub const fn distribution(&self) -> &DistributionArtifact {
        self.bridged.distribution()
    }

    /// The source draw table, unchanged.
    #[must_use]
    pub const fn source(&self) -> &DistributionArtifact {
        self.bridged.source()
    }

    /// Mean of the bridged draws.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.bridged.claim().mean
    }

    /// Variance of the bridged draws.
    #[must_use]
    pub const fn variance(&self) -> f64 {
        self.bridged.claim().variance
    }

    /// How the spread of the bridged draws is read: an uncalibrated spread, never
    /// an interval, unless [`Self::bridged_interval`] publishes one.
    #[must_use]
    pub const fn reading(&self) -> &'static str {
        antecedent_contract::UNCALIBRATED_SPREAD_LABEL
    }

    /// A summary of the derivation.
    #[must_use]
    pub fn inspect(&self) -> DistributionBridgeInspection {
        let claim = self.bridged.claim();
        DistributionBridgeInspection {
            kind: claim.kind(),
            mapping: claim.mapping,
            value_kind: claim.value_kind,
            n_draws: claim.n_draws,
            source_mean: self.bridged.source_mean(),
            source_variance: self.bridged.source_variance(),
            mean: claim.mean,
            variance: claim.variance,
            stream: claim.stream,
            alignment_identity: claim.alignment_identity.clone(),
            draws_digest: claim.draws_digest.to_hex(),
            source_claim_id: claim.source_claim_id.to_hex(),
            claim_id: claim.claim.claim_id.to_hex(),
            contract_identity: claim.contract_identity.to_hex(),
            assumptions: claim
                .contract
                .assumptions
                .iter()
                .map(|a| (a.id.clone(), a.kind.as_str().to_owned(), a.evidence.clone()))
                .collect(),
            transported: claim.transported,
            reading: self.reading(),
            unavailable: claim
                .receipt
                .unavailable_operations
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
    }

    /// Require the kind of object a consumer needs.
    ///
    /// # Errors
    /// `bridge_substitution_refused` when the consumer needs a mean, an identified
    /// set, a joint outcome law or the other of a posterior and a predictive law.
    pub fn require_value_kind(&self, requested: ValueKind) -> Result<(), ScientificBridgeRefusal> {
        self.bridged
            .claim()
            .require_value_kind(requested)
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))
    }

    /// An unscoped request for an interval of the bridged draws.
    ///
    /// # Errors
    /// Always `cell_not_licensed` under `scientific_bridge.interval_not_licensed`:
    /// name a level and ask [`Self::bridged_interval`].
    pub fn interval(&self) -> Result<(), ScientificBridgeRefusal> {
        self.bridged
            .claim()
            .request_interval()
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))
    }

    /// The interval of the bridged value at `level`, through the measured-coverage
    /// gate of a bridged mean: the standard deviation of the source draws carried
    /// through the mapping with its own estimation uncertainty, at the source's
    /// mean. Published only where a current attesting coverage record measured the
    /// construction.
    ///
    /// # Errors
    /// `cell_not_licensed` with `scientific_bridge.interval_scope_outside_measured`
    /// for a mapping with uncertain coefficients, a declared source-mapping
    /// covariance, a level or design no record measured.
    pub fn bridged_interval(
        &self,
        level: f64,
    ) -> Result<BridgedClaimInterval, ScientificBridgeRefusal> {
        bridged_distribution_interval(&self.bridged, level)
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))
    }

    /// The bridge as a measurement-mapping stage of a propagation chain: the
    /// source draws, the drawn slope and the drawn intercept as columns of one
    /// aligned table, read by an affine stage whose coefficients are inputs.
    ///
    /// # Errors
    /// A bridge that is not an affine map of its inputs (a measurement
    /// correction, or a predictive law through a trial-level line).
    pub fn chain_stage(&self, stage_id: &str) -> Result<BridgeChainStage, ScientificBridgeRefusal> {
        self.bridged
            .chain_stage(stage_id)
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))
    }

    /// Freshness of the derived claim given the freshness of its source.
    #[must_use]
    pub fn freshness(&self, source: &ClaimFreshness) -> ClaimFreshness {
        self.bridged.claim().freshness(source)
    }

    /// Write the extended `scientific_bridge_v1` artifact: the contract, the
    /// source draw table, the seed and the summary of the mapped table.
    ///
    /// # Errors
    /// An encoding failure or a bridge that does not seal.
    pub fn export(
        &self,
        artifact_id: &str,
    ) -> Result<ScientificBridgeExport, ScientificBridgeRefusal> {
        let (sealed, _, _) = ScientificBridgeDrawsArtifact::seal(&self.bridged)
            .map_err(|e| ScientificBridgeRefusal::from_artifact(&e))?;
        let bytes = ScientificBridgeDrawsArtifact::to_bytes(&self.bridged, artifact_id)
            .map_err(|e| ScientificBridgeRefusal::from_artifact(&e))?;
        Ok(ScientificBridgeExport { bytes, identity: sealed.identity().to_owned() })
    }
}

/// A bridge of draws consumed from an artifact.
#[derive(Clone, Debug)]
pub struct ConsumedScientificBridgeDistribution {
    /// The replayed mapping of the stored draws.
    pub bridged: BridgedDistribution,
    /// The artifact identity (hex).
    pub identity: String,
}

/// Consume a bridge-of-draws artifact by mapping its stored draws again.
///
/// `expected_identity` is an artifact identity (hex) the consumer retained
/// independently; when given, a consistently resealed artifact of another bridge
/// is refused.
///
/// # Errors
/// Corruption, another major version, a bound exceeded, a contract the library
/// refuses on replay, a stored identity, summary or mapped table that does not
/// replay bit for bit, or an identity that does not match.
pub fn consume_scientific_bridge_distribution_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> Result<ConsumedScientificBridgeDistribution, ScientificBridgeRefusal> {
    let artifact = ScientificBridgeDrawsArtifact::from_bytes(bytes, expected_identity)
        .map_err(|e| ScientificBridgeRefusal::from_artifact(&e))?;
    Ok(ConsumedScientificBridgeDistribution {
        bridged: artifact.bridged().clone(),
        identity: artifact.identity().to_owned(),
    })
}

/// Which shape a scientific bridge artifact has, `point` or `draws`, without
/// replaying it.
///
/// # Errors
/// Corruption, another major version or a container that is not a bridge.
pub fn scientific_bridge_artifact_shape(
    bytes: &[u8],
) -> Result<&'static str, ScientificBridgeRefusal> {
    ScientificBridgeAny::shape(bytes).map_err(|e| ScientificBridgeRefusal::from_artifact(&e))
}

#[cfg(test)]
#[path = "scientific_bridge_distribution_tests.rs"]
mod distribution_tests;

#[cfg(test)]
#[path = "scientific_bridge_propagation_tests.rs"]
mod propagation_tests;
