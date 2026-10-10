//! Durable admissible decision contract and robust decision result artifacts.
//!
//! [`AdmissibleContractArtifact`] carries a [`AdmissibleDecisionContract`]
//! (the base contract plus its admissibility rules: support rules, declared
//! exclusions and the required uncertainty representation) under the contract's
//! canonical admissible identity. It is a separate artifact kind, so a contract
//! without rules keeps writing exactly the `decision_contract_v1` bytes of
//! [`crate::decision_artifact::DecisionContractArtifact`].
//!
//! [`RobustResultArtifact`] stores a [`RobustDecisionResult`] bound to the
//! admissible contract identity, to the digest of every structure's draws and to
//! the claim profile (uncertainty kind and per-structure support) it was assessed
//! under. It loads only under identities the consumer retained independently of
//! the bytes, refuses a resealed edit (the stored lineage and every number are
//! re-derived or recomputed), and replays by recomputation.
//!
//! A decision whose source value came from an external callback retains an
//! [`ExternalCallbackReceipt`]: the attested value, the exact request fingerprint
//! and the trust limit. The receipt enters the artifact lineage, so changing any
//! of its fields breaks the stored derivation. The trust vocabulary has no native
//! variant: such an artifact reports [`RobustResultArtifact::native_verified`] as
//! `false` and a replay says which structures it could not certify natively.

use std::collections::HashSet;

use antecedent_core::{
    CompositionLink, CompositionStage, ExternalRefusal, ProvenanceChain, ProvenanceChainError,
    SupportStatus,
};
use antecedent_io::convert::to_cbor;
use antecedent_io::error::IoError;
use antecedent_io::external_claim_artifact::{LineageLinkWire, wire_from_chain};
use serde::{Deserialize, Serialize};

use crate::decision_artifact::{
    ContractBody, MAX_DECISION_ARTIFACT_BYTES, body_from_contract, contract_from_body, decode,
    encode, refused, source_digest,
};
use crate::decision_contract::{
    AdmissibilityRules, AdmissibleDecisionContract, DeclaredExclusion, SupportRule,
    UncertaintyKind, UncertaintyRequirement,
};
use crate::decision_robustness::{
    ActionRobustness, AtomSupport, ClaimProfile, InputSupport, RobustDecisionResult, RobustVerdict,
    SupportShortfall, evaluate_robust,
};
use crate::decision_structural::{AtomEvidence, StructuralAtom};

const ADMISSIBLE_KIND: &str = "admissible_decision_contract_v1";
const ROBUST_KIND: &str = "robust_decision_result_v1";
/// Most atoms, actions, shortfalls or receipts a decoded artifact may list.
pub const MAX_ROBUST_ITEMS: usize = 65_536;

fn debug_error(error: impl std::fmt::Debug) -> IoError {
    IoError::Convert(format!("{error:?}"))
}

fn chain_error(error: &ProvenanceChainError) -> IoError {
    let refusal = error.to_refusal();
    IoError::Refused {
        code: refusal.code,
        message: format!("decision_artifact.lineage: {}", refusal.detail),
    }
}

// ---------------------------------------------------------------------------
// Admissible contract
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportRuleWire {
    action_id: String,
    input: usize,
    weakest_allowed: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExclusionWire {
    action_id: String,
    reason: String,
}

fn none_name() -> String {
    "none".to_owned()
}

fn one() -> u16 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RulesWire {
    #[serde(default)]
    default_weakest_support: Option<String>,
    #[serde(default)]
    support_rules: Vec<SupportRuleWire>,
    #[serde(default)]
    declared_exclusions: Vec<ExclusionWire>,
    #[serde(default = "none_name")]
    uncertainty: String,
}

impl Default for RulesWire {
    fn default() -> Self {
        Self {
            default_weakest_support: None,
            support_rules: Vec::new(),
            declared_exclusions: Vec::new(),
            uncertainty: none_name(),
        }
    }
}

fn requirement_from(name: &str) -> Result<UncertaintyRequirement, IoError> {
    Ok(match name {
        "none" => UncertaintyRequirement::None,
        "point_only" => UncertaintyRequirement::PointOnly,
        "structural_envelope" => UncertaintyRequirement::StructuralEnvelope,
        "credible" => UncertaintyRequirement::Credible,
        _ => return Err(IoError::Convert(format!("unknown uncertainty requirement `{name}`"))),
    })
}

fn kind_from(name: &str) -> Result<UncertaintyKind, IoError> {
    Ok(match name {
        "point" => UncertaintyKind::Point,
        "structural_envelope" => UncertaintyKind::StructuralEnvelope,
        "credible" => UncertaintyKind::Credible,
        _ => return Err(IoError::Convert(format!("unknown uncertainty kind `{name}`"))),
    })
}

fn status_from(name: &str) -> Result<SupportStatus, IoError> {
    SupportStatus::from_name(name)
        .ok_or_else(|| IoError::Convert(format!("unknown support status `{name}`")))
}

fn rules_to_wire(rules: &AdmissibilityRules) -> RulesWire {
    RulesWire {
        default_weakest_support: rules.default_weakest_support.map(|s| s.as_str().to_owned()),
        support_rules: rules
            .support_rules
            .iter()
            .map(|r| SupportRuleWire {
                action_id: r.action_id.clone(),
                input: r.input,
                weakest_allowed: r.weakest_allowed.as_str().to_owned(),
            })
            .collect(),
        declared_exclusions: rules
            .declared_exclusions
            .iter()
            .map(|e| ExclusionWire { action_id: e.action_id.clone(), reason: e.reason.clone() })
            .collect(),
        uncertainty: rules.uncertainty.name().to_owned(),
    }
}

fn rules_from_wire(wire: RulesWire) -> Result<AdmissibilityRules, IoError> {
    Ok(AdmissibilityRules {
        default_weakest_support: wire
            .default_weakest_support
            .as_deref()
            .map(status_from)
            .transpose()?,
        support_rules: wire
            .support_rules
            .into_iter()
            .map(|r| {
                Ok(SupportRule {
                    action_id: r.action_id,
                    input: r.input,
                    weakest_allowed: status_from(&r.weakest_allowed)?,
                })
            })
            .collect::<Result<Vec<_>, IoError>>()?,
        declared_exclusions: wire
            .declared_exclusions
            .into_iter()
            .map(|e| DeclaredExclusion { action_id: e.action_id, reason: e.reason })
            .collect(),
        uncertainty: requirement_from(&wire.uncertainty)?,
    })
}

/// Declaration and artifact body of an admissible contract: the base contract
/// declaration, its rules and the recomputed identity (ignored on parse).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmissibleBody {
    #[serde(default = "one")]
    version: u16,
    #[serde(default)]
    identity: String,
    contract: ContractBody,
    #[serde(default)]
    rules: RulesWire,
}

fn admissible_from_body(body: AdmissibleBody) -> Result<AdmissibleDecisionContract, IoError> {
    if body.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(body.version) });
    }
    Ok(AdmissibleDecisionContract {
        contract: contract_from_body(body.contract)?,
        rules: rules_from_wire(body.rules)?,
    })
}

fn admissible_to_body(contract: &AdmissibleDecisionContract) -> Result<AdmissibleBody, IoError> {
    let identity = contract.identity().map_err(debug_error)?;
    Ok(AdmissibleBody {
        version: 1,
        identity,
        contract: body_from_contract(&contract.contract)?,
        rules: rules_to_wire(&contract.rules),
    })
}

/// Parse an admissible contract declaration `{"contract": ..., "rules": ...}`
/// (the wire a host language builds). A stored identity is ignored.
///
/// # Errors
/// Malformed JSON, unknown fields or an invalid declaration refuse.
pub fn admissible_contract_from_json(json: &str) -> Result<AdmissibleDecisionContract, IoError> {
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let body: AdmissibleBody =
        serde_json::from_str(json).map_err(|e| IoError::Convert(e.to_string()))?;
    let contract = admissible_from_body(body)?;
    contract.validate().map_err(debug_error)?;
    Ok(contract)
}

/// Parse and validate an admissible declaration, refusing with the typed
/// structured refusal a Rust caller would get: a malformed declaration is
/// `decision_contract.invalid_declaration`; an invalid one keeps the exact code
/// and detail of its error.
///
/// # Errors
/// A malformed or invalid declaration, as a structured refusal.
// The refusal is the cold path of a once-per-declaration check; boxing would not pay.
#[allow(clippy::result_large_err)]
pub fn admissible_contract_from_json_refusal(
    json: &str,
) -> Result<AdmissibleDecisionContract, ExternalRefusal> {
    let malformed = |why: String| ExternalRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        stage: "declare",
        detail: "decision_contract.invalid_declaration".to_owned(),
        offending: Some(why),
        expected: None,
        supplied: None,
        capability: None,
        remedy: Some("declare the contract and its rules with the documented fields"),
    };
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(malformed("declaration is too large".to_owned()));
    }
    let body: AdmissibleBody = serde_json::from_str(json).map_err(|e| malformed(e.to_string()))?;
    let contract = admissible_from_body(body).map_err(|e| malformed(e.to_string()))?;
    contract.validate().map_err(|e| e.to_refusal())?;
    Ok(contract)
}

/// The JSON declaration of an admissible contract, including its recomputed
/// identity.
///
/// # Errors
/// An invalid contract refuses.
pub fn admissible_contract_to_json(
    contract: &AdmissibleDecisionContract,
) -> Result<String, IoError> {
    serde_json::to_string(&admissible_to_body(contract)?)
        .map_err(|e| IoError::Convert(e.to_string()))
}

/// Identity of an admissible decision contract bound to a causal contract's
/// target and identification identities (lowercase hex). Any change to any of
/// the three changes it.
#[must_use]
pub fn bound_decision_identity(
    decision_identity: &str,
    causal_target: &str,
    causal_identification: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    for part in [
        "antecedent.bound_decision_contract.v1",
        decision_identity,
        causal_target,
        causal_identification,
    ] {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

/// An admissible decision contract with its canonical identity.
#[derive(Clone, Debug, PartialEq)]
pub struct AdmissibleContractArtifact {
    contract: AdmissibleDecisionContract,
    identity: String,
}

impl AdmissibleContractArtifact {
    /// Own a validated admissible contract.
    ///
    /// # Errors
    /// An invalid contract or rule set refuses.
    pub fn new(contract: AdmissibleDecisionContract) -> Result<Self, IoError> {
        let identity = contract.identity().map_err(debug_error)?;
        Ok(Self { contract, identity })
    }

    /// The contract with its rules.
    #[must_use]
    pub fn contract(&self) -> &AdmissibleDecisionContract {
        &self.contract
    }

    /// Canonical identity covering the base contract and every rule.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An oversized payload or invalid contract refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let body = to_cbor(&admissible_to_body(&self.contract)?)?;
        encode(ADMISSIBLE_KIND, artifact_id, "admissible_decision_contract", &body)
    }

    /// Load under an independently retained identity.
    ///
    /// # Errors
    /// A corrupt, oversized or semantically changed artifact refuses, including
    /// one whose stored identity was resealed to match edited rules.
    pub fn from_bytes(bytes: &[u8], expected_identity: &str) -> Result<Self, IoError> {
        let body: AdmissibleBody = decode(bytes, ADMISSIBLE_KIND)?;
        let stored = body.identity.clone();
        let contract = admissible_from_body(body)?;
        let identity = contract.identity().map_err(debug_error)?;
        if identity != stored || identity != expected_identity {
            return Err(refused(
                "identity_expected",
                "admissible contract identity differs from the consumer's retained identity",
            ));
        }
        Ok(Self { contract, identity })
    }
}

// ---------------------------------------------------------------------------
// External callback receipts
// ---------------------------------------------------------------------------

/// The most an external callback's value can be trusted. There is deliberately no
/// native variant: a value from outside Antecedent is never natively verified.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExternalTrustLimit {
    /// Supplier assertion only; nothing was checked here.
    ExternallyAttested {
        /// Party making the assertion.
        attestor: String,
    },
    /// The exact request passed object-level verification. Still an extension,
    /// never native.
    VerifiedExtension,
}

impl ExternalTrustLimit {
    /// Stable wire label, shared with the Python provider-trust vocabulary.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::ExternallyAttested { .. } => "externally_attested",
            Self::VerifiedExtension => "verified_extension",
        }
    }

    /// Always `false`: no external trust limit is native verification.
    #[must_use]
    pub fn is_native(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        match self {
            Self::ExternallyAttested { attestor } => format!("externally_attested:{attestor}"),
            Self::VerifiedExtension => "verified_extension".to_owned(),
        }
    }
}

/// What a decision retains of a value an external callback supplied.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalCallbackReceipt {
    /// Structure whose draws or value came from the callback.
    pub atom_id: String,
    /// Provider service identity.
    pub provider_id: String,
    /// Provider input or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint the callback answered.
    pub request_fingerprint: String,
    /// The value the callback attested, as returned.
    pub attested_value: f64,
    /// Trust limit of the value.
    pub trust: ExternalTrustLimit,
}

impl ExternalCallbackReceipt {
    fn validate(&self) -> Result<(), IoError> {
        let blank = |value: &str| value.trim().is_empty();
        let attestor_blank = matches!(
            &self.trust,
            ExternalTrustLimit::ExternallyAttested { attestor } if blank(attestor)
        );
        if blank(&self.atom_id)
            || blank(&self.provider_id)
            || blank(&self.snapshot_id)
            || blank(&self.request_fingerprint)
            || attestor_blank
            || !self.attested_value.is_finite()
        {
            return Err(refused(
                "receipt_invalid",
                "a receipt names its structure, provider, snapshot, request and a finite value",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustWire {
    label: String,
    attestor: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    atom_id: String,
    provider_id: String,
    snapshot_id: String,
    request_fingerprint: String,
    attested_value: f64,
    trust: TrustWire,
}

fn receipt_to_wire(receipt: &ExternalCallbackReceipt) -> ReceiptWire {
    ReceiptWire {
        atom_id: receipt.atom_id.clone(),
        provider_id: receipt.provider_id.clone(),
        snapshot_id: receipt.snapshot_id.clone(),
        request_fingerprint: receipt.request_fingerprint.clone(),
        attested_value: receipt.attested_value,
        trust: TrustWire {
            label: receipt.trust.label().to_owned(),
            attestor: match &receipt.trust {
                ExternalTrustLimit::ExternallyAttested { attestor } => Some(attestor.clone()),
                ExternalTrustLimit::VerifiedExtension => None,
            },
        },
    }
}

fn receipt_from_wire(wire: ReceiptWire) -> Result<ExternalCallbackReceipt, IoError> {
    let trust = match (wire.trust.label.as_str(), wire.trust.attestor) {
        ("externally_attested", Some(attestor)) => {
            ExternalTrustLimit::ExternallyAttested { attestor }
        }
        ("verified_extension", None) => ExternalTrustLimit::VerifiedExtension,
        _ => {
            return Err(refused(
                "receipt_trust",
                "an external receipt is externally attested with an attestor or a verified \
                 extension; it is never native",
            ));
        }
    };
    let receipt = ExternalCallbackReceipt {
        atom_id: wire.atom_id,
        provider_id: wire.provider_id,
        snapshot_id: wire.snapshot_id,
        request_fingerprint: wire.request_fingerprint,
        attested_value: wire.attested_value,
        trust,
    };
    receipt.validate()?;
    Ok(receipt)
}

// ---------------------------------------------------------------------------
// Atom records and the claim profile
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AtomWire {
    id: String,
    probability: Option<f64>,
    status: String,
    reason: Option<String>,
    source_digest: Option<String>,
}

fn atom_wires(atoms: &[StructuralAtom]) -> Vec<AtomWire> {
    atoms
        .iter()
        .map(|atom| {
            let (status, reason, digest) = match &atom.evidence {
                AtomEvidence::Evaluated(source) => ("evaluated", None, Some(source_digest(source))),
                AtomEvidence::Unidentified => ("unidentified", None, None),
                AtomEvidence::Unevaluated(reason) => ("unevaluated", Some(reason.clone()), None),
            };
            AtomWire {
                id: atom.id.clone(),
                probability: atom.probability,
                status: status.to_owned(),
                reason,
                source_digest: digest,
            }
        })
        .collect()
}

/// Each structure's id with the digest of its draws when evaluated: what a
/// consumer retains independently to load a [`RobustResultArtifact`].
#[must_use]
pub fn atom_digests_of(atoms: &[StructuralAtom]) -> Vec<(String, Option<String>)> {
    atom_wires(atoms).into_iter().map(|a| (a.id, a.source_digest)).collect()
}

fn check_atom_wires(atoms: &[AtomWire]) -> Result<(), IoError> {
    let mut ids = HashSet::new();
    for atom in atoms {
        let consistent = match atom.status.as_str() {
            "evaluated" => atom.source_digest.is_some() && atom.reason.is_none(),
            "unidentified" => atom.source_digest.is_none() && atom.reason.is_none(),
            "unevaluated" => atom.source_digest.is_none() && atom.reason.is_some(),
            _ => false,
        };
        if !consistent || atom.id.trim().is_empty() || !ids.insert(atom.id.as_str()) {
            return Err(refused("atoms_invalid", "structure records are inconsistent"));
        }
    }
    Ok(())
}

fn check_receipts(atoms: &[AtomWire], receipts: &[ExternalCallbackReceipt]) -> Result<(), IoError> {
    let mut seen = HashSet::new();
    for receipt in receipts {
        receipt.validate()?;
        let Some(atom) = atoms.iter().find(|a| a.id == receipt.atom_id) else {
            return Err(refused("receipt_atom", "a receipt names no declared structure"));
        };
        if atom.source_digest.is_none() {
            return Err(refused("receipt_atom", "a receipt names a structure without draws"));
        }
        if !seen.insert(receipt.atom_id.as_str()) {
            return Err(refused("receipt_duplicate", "a structure has one receipt"));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputSupportWire {
    action_id: String,
    input: usize,
    status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportEntryWire {
    atom_id: String,
    overall: String,
    per_input: Vec<InputSupportWire>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileWire {
    uncertainty: String,
    support: Vec<SupportEntryWire>,
}

fn profile_to_wire(profile: &ClaimProfile) -> ProfileWire {
    let mut support: Vec<SupportEntryWire> = profile
        .support
        .iter()
        .map(|(id, s)| {
            let mut per_input: Vec<InputSupportWire> = s
                .per_input
                .iter()
                .map(|p| InputSupportWire {
                    action_id: p.action_id.clone(),
                    input: p.input,
                    status: p.status.as_str().to_owned(),
                })
                .collect();
            per_input.sort_by(|a, b| (&a.action_id, a.input).cmp(&(&b.action_id, b.input)));
            SupportEntryWire {
                atom_id: id.clone(),
                overall: s.overall.as_str().to_owned(),
                per_input,
            }
        })
        .collect();
    support.sort_by(|a, b| a.atom_id.cmp(&b.atom_id));
    ProfileWire { uncertainty: profile.uncertainty.name().to_owned(), support }
}

fn profile_from_wire(wire: ProfileWire) -> Result<ClaimProfile, IoError> {
    let support = wire
        .support
        .into_iter()
        .map(|entry| {
            let per_input = entry
                .per_input
                .into_iter()
                .map(|p| {
                    Ok(InputSupport {
                        action_id: p.action_id,
                        input: p.input,
                        status: status_from(&p.status)?,
                    })
                })
                .collect::<Result<Vec<_>, IoError>>()?;
            Ok((entry.atom_id, AtomSupport { overall: status_from(&entry.overall)?, per_input }))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    Ok(ClaimProfile { uncertainty: kind_from(&wire.uncertainty)?, support })
}

// ---------------------------------------------------------------------------
// Result wire
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum VerdictWire {
    StructurallyRobust(String),
    SupportRobust(String),
    SupportDependent { supported_choice: String, unrestricted_choice: Option<String> },
    GraphDependentChoice(Vec<(String, Vec<String>)>),
    UnsupportedExtrapolation,
    InsufficientClaims(String),
    NoAdmissibleAction,
    WorstCaseChoice(String),
    BayesChoice { action: String, evaluated_mass: f64 },
    ReportOnly,
}

fn verdict_to_wire(verdict: &RobustVerdict) -> VerdictWire {
    match verdict {
        RobustVerdict::StructurallyRobust(id) => VerdictWire::StructurallyRobust(id.clone()),
        RobustVerdict::SupportRobust(id) => VerdictWire::SupportRobust(id.clone()),
        RobustVerdict::SupportDependent { supported_choice, unrestricted_choice } => {
            VerdictWire::SupportDependent {
                supported_choice: supported_choice.clone(),
                unrestricted_choice: unrestricted_choice.clone(),
            }
        }
        RobustVerdict::GraphDependentChoice(leaders) => VerdictWire::GraphDependentChoice(
            leaders
                .iter()
                .map(|(atom, ids)| {
                    let mut ids = ids.clone();
                    ids.sort();
                    (atom.clone(), ids)
                })
                .collect(),
        ),
        RobustVerdict::UnsupportedExtrapolation => VerdictWire::UnsupportedExtrapolation,
        RobustVerdict::InsufficientClaims(why) => VerdictWire::InsufficientClaims(why.clone()),
        RobustVerdict::NoAdmissibleAction => VerdictWire::NoAdmissibleAction,
        RobustVerdict::WorstCaseChoice(id) => VerdictWire::WorstCaseChoice(id.clone()),
        RobustVerdict::BayesChoice { action, evaluated_mass } => {
            VerdictWire::BayesChoice { action: action.clone(), evaluated_mass: *evaluated_mass }
        }
        RobustVerdict::ReportOnly => VerdictWire::ReportOnly,
    }
}

fn verdict_from_wire(wire: VerdictWire) -> RobustVerdict {
    match wire {
        VerdictWire::StructurallyRobust(id) => RobustVerdict::StructurallyRobust(id),
        VerdictWire::SupportRobust(id) => RobustVerdict::SupportRobust(id),
        VerdictWire::SupportDependent { supported_choice, unrestricted_choice } => {
            RobustVerdict::SupportDependent { supported_choice, unrestricted_choice }
        }
        VerdictWire::GraphDependentChoice(leaders) => RobustVerdict::GraphDependentChoice(leaders),
        VerdictWire::UnsupportedExtrapolation => RobustVerdict::UnsupportedExtrapolation,
        VerdictWire::InsufficientClaims(why) => RobustVerdict::InsufficientClaims(why),
        VerdictWire::NoAdmissibleAction => RobustVerdict::NoAdmissibleAction,
        VerdictWire::WorstCaseChoice(id) => RobustVerdict::WorstCaseChoice(id),
        VerdictWire::BayesChoice { action, evaluated_mass } => {
            RobustVerdict::BayesChoice { action, evaluated_mass }
        }
        VerdictWire::ReportOnly => RobustVerdict::ReportOnly,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionRobustWire {
    id: String,
    declared_exclusion: Option<String>,
    range: Option<(f64, f64)>,
    supported_range: Option<(f64, f64)>,
    unsupported_in: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShortfallWire {
    atom: String,
    action: String,
    input: usize,
    status: String,
    weakest_allowed: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultWire {
    contract_identity: String,
    base_contract_identity: String,
    verdict: VerdictWire,
    actions: Vec<ActionRobustWire>,
    shortfalls: Vec<ShortfallWire>,
    unsupported_atoms: Vec<String>,
    unsupported_mass: Option<f64>,
    unidentified_mass: Option<f64>,
    unevaluated_mass: Option<f64>,
    evaluated_mass: Option<f64>,
    uncertainty_required: String,
    uncertainty_supplied: String,
    assumptions: Vec<String>,
}

/// Canonical wire of a result: actions, shortfalls and tied leaders are ordered
/// by identity, so reordering the contract's actions leaves it unchanged.
fn result_wire(result: &RobustDecisionResult) -> ResultWire {
    let mut actions: Vec<ActionRobustWire> = result
        .actions
        .iter()
        .map(|a| {
            let mut unsupported_in = a.unsupported_in.clone();
            unsupported_in.sort();
            ActionRobustWire {
                id: a.id.clone(),
                declared_exclusion: a.declared_exclusion.clone(),
                range: a.range,
                supported_range: a.supported_range,
                unsupported_in,
            }
        })
        .collect();
    actions.sort_by(|a, b| a.id.cmp(&b.id));
    let mut shortfalls: Vec<ShortfallWire> = result
        .shortfalls
        .iter()
        .map(|s| ShortfallWire {
            atom: s.atom.clone(),
            action: s.action.clone(),
            input: s.input,
            status: s.status.as_str().to_owned(),
            weakest_allowed: s.weakest_allowed.as_str().to_owned(),
        })
        .collect();
    shortfalls.sort_by(|a, b| (&a.atom, &a.action, a.input).cmp(&(&b.atom, &b.action, b.input)));
    let mut unsupported_atoms = result.unsupported_atoms.clone();
    unsupported_atoms.sort();
    ResultWire {
        contract_identity: result.contract_identity.clone(),
        base_contract_identity: result.base_contract_identity.clone(),
        verdict: verdict_to_wire(&result.verdict),
        actions,
        shortfalls,
        unsupported_atoms,
        unsupported_mass: result.unsupported_mass,
        unidentified_mass: result.unidentified_mass,
        unevaluated_mass: result.unevaluated_mass,
        evaluated_mass: result.evaluated_mass,
        uncertainty_required: result.uncertainty_required.name().to_owned(),
        uncertainty_supplied: result.uncertainty_supplied.name().to_owned(),
        assumptions: result.assumptions.clone(),
    }
}

fn result_from_wire(wire: ResultWire) -> Result<RobustDecisionResult, IoError> {
    let shortfalls = wire
        .shortfalls
        .into_iter()
        .map(|s| {
            Ok(SupportShortfall {
                atom: s.atom,
                action: s.action,
                input: s.input,
                status: status_from(&s.status)?,
                weakest_allowed: status_from(&s.weakest_allowed)?,
            })
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    Ok(RobustDecisionResult {
        contract_identity: wire.contract_identity,
        base_contract_identity: wire.base_contract_identity,
        verdict: verdict_from_wire(wire.verdict),
        actions: wire
            .actions
            .into_iter()
            .map(|a| ActionRobustness {
                id: a.id,
                declared_exclusion: a.declared_exclusion,
                range: a.range,
                supported_range: a.supported_range,
                unsupported_in: a.unsupported_in,
            })
            .collect(),
        shortfalls,
        unsupported_atoms: wire.unsupported_atoms,
        unsupported_mass: wire.unsupported_mass,
        unidentified_mass: wire.unidentified_mass,
        unevaluated_mass: wire.unevaluated_mass,
        evaluated_mass: wire.evaluated_mass,
        uncertainty_required: requirement_from(&wire.uncertainty_required)?,
        uncertainty_supplied: kind_from(&wire.uncertainty_supplied)?,
        assumptions: wire.assumptions,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RobustBody {
    version: u16,
    result: ResultWire,
    profile: ProfileWire,
    atoms: Vec<AtomWire>,
    external_receipts: Vec<ReceiptWire>,
    /// `true` exactly when no structure's value came from an external callback.
    native_verified: bool,
    /// Derived at serialization, verified on load by re-deriving it.
    #[serde(default)]
    lineage: Vec<LineageLinkWire>,
}

// ---------------------------------------------------------------------------
// Lineage
// ---------------------------------------------------------------------------

fn lineage_links(
    contract_identity: &str,
    atoms: &[AtomWire],
    receipts: &[ExternalCallbackReceipt],
) -> Vec<CompositionLink> {
    let link = |id: &str, stage: CompositionStage, parents: &[&str]| CompositionLink {
        id: id.to_owned(),
        stage,
        parents: parents.iter().map(|p| (*p).to_owned()).collect(),
        declared_parent_digests: None,
    };
    let decision = format!("decision:{contract_identity}");
    let mut links = vec![link(&decision, CompositionStage::DecisionContract, &[])];
    let mut claim_parents = vec![decision];
    for atom in atoms {
        let Some(digest) = &atom.source_digest else {
            continue;
        };
        let distribution = format!("distribution:{}:{digest}", atom.id);
        match receipts.iter().find(|r| r.atom_id == atom.id) {
            Some(r) => {
                let provider = format!(
                    "provider:{}:{}#{}@{}|{}|{:016x}",
                    atom.id,
                    r.provider_id,
                    r.snapshot_id,
                    r.request_fingerprint,
                    r.trust.describe(),
                    r.attested_value.to_bits()
                );
                links.push(link(&provider, CompositionStage::ExternalProvider, &[]));
                links.push(link(
                    &distribution,
                    CompositionStage::DistributionArtifact,
                    &[provider.as_str()],
                ));
            }
            None => links.push(link(&distribution, CompositionStage::DistributionArtifact, &[])),
        }
        claim_parents.push(distribution);
    }
    links.push(CompositionLink {
        id: RobustResultArtifact::CLAIM_LINK_ID.to_owned(),
        stage: CompositionStage::Claim,
        parents: claim_parents,
        declared_parent_digests: None,
    });
    links
}

fn lineage_wire(
    contract_identity: &str,
    atoms: &[AtomWire],
    receipts: &[ExternalCallbackReceipt],
) -> Result<Vec<LineageLinkWire>, IoError> {
    let chain = ProvenanceChain::new(lineage_links(contract_identity, atoms, receipts))
        .map_err(|e| chain_error(&e))?;
    wire_from_chain(&chain)
}

/// A stored lineage must hold under recomputation of its own digests and equal
/// the lineage re-derived from the contract identity, structures and receipts.
fn verify_lineage(stored: &[LineageLinkWire], derived: &[LineageLinkWire]) -> Result<(), IoError> {
    if stored.is_empty() {
        return Err(refused("lineage_missing", "a robust result stores its derivation"));
    }
    let links = stored
        .iter()
        .map(|l| {
            Ok(CompositionLink {
                id: l.id.clone(),
                stage: CompositionStage::from_name(&l.stage)
                    .ok_or_else(|| refused("lineage_stage", "unknown lineage stage"))?,
                parents: l.parents.clone(),
                declared_parent_digests: Some(l.parent_digests.clone()),
            })
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    let chain = ProvenanceChain::new(links).map_err(|e| chain_error(&e))?;
    for link in stored {
        chain.verify_digest(&link.id, &link.digest).map_err(|e| chain_error(&e))?;
    }
    if stored != derived {
        return Err(refused(
            "lineage_expected",
            "stored lineage differs from the contract, structures and receipts it claims",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// What a replay established.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayReceipt {
    /// The assessment was recomputed from the supplied contract, structures and
    /// profile and reproduced the stored result exactly.
    pub recomputed: bool,
    /// Every structure's draws were native; `false` when any came from an
    /// external callback, whose value the replay cannot certify.
    pub native_verified: bool,
    /// Structures whose values rest on an external callback, in structure order.
    pub external_atoms: Vec<String>,
}

/// A robust decision result bound to its admissible contract, structures and
/// claim profile, with the external callback receipts behind any structure.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustResultArtifact {
    result: RobustDecisionResult,
    profile: ClaimProfile,
    atoms: Vec<AtomWire>,
    receipts: Vec<ExternalCallbackReceipt>,
}

impl RobustResultArtifact {
    /// Identity of the reported decision within [`Self::provenance_chain`].
    pub const CLAIM_LINK_ID: &'static str = "robust_decision";

    /// Bind a result to the contract, structures and profile it was assessed
    /// under. `receipts` name the structures whose values an external callback
    /// supplied.
    ///
    /// # Errors
    /// A result computed under a different contract, structures with repeated or
    /// blank identities, or a receipt that is invalid, repeated or names a
    /// structure without draws refuses.
    pub fn new(
        contract: &AdmissibleDecisionContract,
        result: RobustDecisionResult,
        profile: &ClaimProfile,
        atoms: &[StructuralAtom],
        mut receipts: Vec<ExternalCallbackReceipt>,
    ) -> Result<Self, IoError> {
        let identity = contract.identity().map_err(debug_error)?;
        if identity != result.contract_identity {
            return Err(refused("result_contract", "result was assessed under another contract"));
        }
        let atoms = atom_wires(atoms);
        check_atom_wires(&atoms)?;
        check_receipts(&atoms, &receipts)?;
        receipts.sort_by(|a, b| a.atom_id.cmp(&b.atom_id));
        Ok(Self { result, profile: profile.clone(), atoms, receipts })
    }

    /// The stored result.
    #[must_use]
    pub fn result(&self) -> &RobustDecisionResult {
        &self.result
    }

    /// The claim profile the result was assessed under.
    #[must_use]
    pub fn profile(&self) -> &ClaimProfile {
        &self.profile
    }

    /// External callback receipts, ordered by structure.
    #[must_use]
    pub fn receipts(&self) -> &[ExternalCallbackReceipt] {
        &self.receipts
    }

    /// Structure ids with the digest of each evaluated structure's draws.
    #[must_use]
    pub fn atom_digests(&self) -> Vec<(String, Option<String>)> {
        self.atoms.iter().map(|a| (a.id.clone(), a.source_digest.clone())).collect()
    }

    /// Whether every value is native. `false` whenever any structure rests on an
    /// external callback: a decision over an external value is never labelled
    /// natively verified.
    #[must_use]
    pub fn native_verified(&self) -> bool {
        self.receipts.is_empty()
    }

    /// Derivation chain: the decision contract, each structure's draws (behind
    /// its external provider and receipt when external), and the reported
    /// decision derived from all of them.
    ///
    /// # Errors
    /// A blank identity refuses.
    pub fn provenance_chain(&self) -> Result<ProvenanceChain, ProvenanceChainError> {
        ProvenanceChain::new(lineage_links(
            &self.result.contract_identity,
            &self.atoms,
            &self.receipts,
        ))
    }

    fn body(&self) -> Result<RobustBody, IoError> {
        Ok(RobustBody {
            version: 1,
            result: result_wire(&self.result),
            profile: profile_to_wire(&self.profile),
            atoms: self.atoms.clone(),
            external_receipts: self.receipts.iter().map(receipt_to_wire).collect(),
            native_verified: self.native_verified(),
            lineage: lineage_wire(&self.result.contract_identity, &self.atoms, &self.receipts)?,
        })
    }

    /// The JSON form, including receipts, trust label and lineage.
    ///
    /// # Errors
    /// Serialization failure.
    pub fn to_json(&self) -> Result<String, IoError> {
        serde_json::to_string(&self.body()?).map_err(|e| IoError::Convert(e.to_string()))
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An oversized payload or a blank artifact id refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let body = to_cbor(&self.body()?)?;
        encode(ROBUST_KIND, artifact_id, "robust_decision_result", &body)
    }

    /// Load under the consumer's retained admissible contract identity and the
    /// digests of the structures' draws.
    ///
    /// # Errors
    /// A corrupt, oversized or changed artifact, one bound to a different
    /// contract or different structures, a receipt carrying a native trust
    /// label, a flipped `native_verified` label, or a lineage that does not
    /// match its contract, structures and receipts refuses.
    pub fn from_bytes(
        bytes: &[u8],
        expected_contract_identity: &str,
        expected_atom_digests: &[(String, Option<String>)],
    ) -> Result<Self, IoError> {
        let mut body: RobustBody = decode(bytes, ROBUST_KIND)?;
        if body.version != 1 {
            return Err(IoError::UnsupportedVersion { version: u32::from(body.version) });
        }
        if body.atoms.len() > MAX_ROBUST_ITEMS
            || body.result.actions.len() > MAX_ROBUST_ITEMS
            || body.result.shortfalls.len() > MAX_ROBUST_ITEMS
            || body.external_receipts.len() > MAX_ROBUST_ITEMS
        {
            return Err(IoError::TooLarge);
        }
        if body.result.contract_identity != expected_contract_identity {
            return Err(refused(
                "contract_expected",
                "result was assessed under a different admissible contract",
            ));
        }
        let digests: Vec<(String, Option<String>)> =
            body.atoms.iter().map(|a| (a.id.clone(), a.source_digest.clone())).collect();
        if digests.as_slice() != expected_atom_digests {
            return Err(refused(
                "atoms_expected",
                "result was assessed from different structures or draws",
            ));
        }
        check_atom_wires(&body.atoms)?;
        let receipts = body
            .external_receipts
            .into_iter()
            .map(receipt_from_wire)
            .collect::<Result<Vec<_>, IoError>>()?;
        check_receipts(&body.atoms, &receipts)?;
        if body.native_verified != receipts.is_empty() {
            return Err(refused(
                "native_label",
                "a decision over an external callback value is never natively verified",
            ));
        }
        let stored_lineage = std::mem::take(&mut body.lineage);
        let artifact = Self {
            result: result_from_wire(body.result)?,
            profile: profile_from_wire(body.profile)?,
            atoms: body.atoms,
            receipts,
        };
        let derived =
            lineage_wire(&artifact.result.contract_identity, &artifact.atoms, &artifact.receipts)?;
        verify_lineage(&stored_lineage, &derived)?;
        Ok(artifact)
    }

    /// Recompute the assessment from `contract`, `atoms` and `profile` and
    /// require the stored result to match exactly.
    ///
    /// The receipt reports whether the replay is native: a structure that rests
    /// on an external callback is recomputed over the draws supplied, but the
    /// callback itself is not re-run, so it is listed and the result is not
    /// labelled natively verified.
    ///
    /// # Errors
    /// A different contract, structures or claim profile, a refused evaluation
    /// or any differing number or verdict refuses.
    pub fn replay(
        &self,
        contract: &AdmissibleDecisionContract,
        atoms: &[StructuralAtom],
        profile: &ClaimProfile,
    ) -> Result<ReplayReceipt, IoError> {
        let identity = contract.identity().map_err(debug_error)?;
        if identity != self.result.contract_identity {
            return Err(refused("replay_contract", "contract differs from the stored result's"));
        }
        if atom_wires(atoms) != self.atoms {
            return Err(refused("replay_atoms", "structures differ from the stored result's"));
        }
        if to_cbor(&profile_to_wire(profile))? != to_cbor(&profile_to_wire(&self.profile))? {
            return Err(refused(
                "replay_profile",
                "claim profile differs from the stored result's",
            ));
        }
        let (_, fresh) = evaluate_robust(contract, atoms, profile)
            .map_err(|e| refused("replay_evaluation", &format!("{e:?}")))?;
        let stored = to_cbor(&result_wire(&self.result))?;
        let recomputed = to_cbor(&result_wire(&fresh))?;
        if stored != recomputed {
            return Err(refused(
                "replay_mismatch",
                "recomputation does not reproduce the stored result",
            ));
        }
        Ok(ReplayReceipt {
            recomputed: true,
            native_verified: self.receipts.is_empty(),
            external_atoms: self
                .atoms
                .iter()
                .filter(|a| self.receipts.iter().any(|r| r.atom_id == a.id))
                .map(|a| a.id.clone())
                .collect(),
        })
    }
}
