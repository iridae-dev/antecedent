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
//!    The derived claim is a point mean: the request for an interval, a
//!    posterior or a joint law refuses with `cell_not_licensed`, and a
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
//! # What is closed
//!
//! A model score offered as a probability, and a predictive importance offered
//! as a causal effect, refuse under their own names whatever is declared: no
//! calibration bridge for a score and no route from an importance to a
//! mechanism is licensed. Values other than a point mean (a law, a posterior,
//! an identified set, a joint law) have no executed bridge.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    BridgeContract, BridgeKind, BridgeRefusal, BridgeSource, BridgedClaim, Licensing, MappingKind,
    TrialLevelFit, ValueKind, derive_bridged_claim,
};
use antecedent_contract::{ClaimEnvelope, ClaimFreshness};
use antecedent_core::identity::SemanticDigest;
use antecedent_io::IoError;
use antecedent_io::bridge_artifact::{ScientificBridgeArtifact, ScientificBridgeArtifactError};

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
        let bridged = derive_bridged_claim(&self.contract, source)
            .map_err(|refusal| ScientificBridgeRefusal::from_contract(&refusal))?;
        Ok(ScientificBridgeResult { bridged, source: source.claim.clone() })
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

    /// An interval, posterior or joint law of the bridged claim.
    ///
    /// # Errors
    /// Always `cell_not_licensed` under `scientific_bridge.interval_not_licensed`.
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
