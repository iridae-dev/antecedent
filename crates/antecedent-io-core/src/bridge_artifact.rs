//! Portable scientific bridge (`scientific_bridge_v1`).
//!
//! A bounded, checksummed container holding one bridged claim with everything
//! needed to derive it again:
//!
//! * the bridge contract: kind, both quantities and their classes, the value
//!   kinds, the evidence scope, the mapping (every number as its `f64` bit
//!   pattern, a measurement correction with its full measurement model), the
//!   declared assumptions with their identities and the transportability
//!   premise;
//! * the source claim as the bridge saw it: its identity, point value, units,
//!   identity layers and evidence;
//! * the derived claim identity, value, trial-level fit and whether the
//!   bridge was transported.
//!
//! A consumer trusts none of the stored derived values. It rebuilds the
//! contract and the source claim, checks the contract licenses its bridge
//! (every refusal of the library applies), derives the claim again and
//! requires the stored contract identity, claim identity, value (bit for bit),
//! fit and transport flag to equal the derivation. A resealed edit of any
//! stored number, assumption, premise or scope is either refused by the
//! contract's licensing checks or changes the replayed identity or value; an
//! edit that is not resealed is an identity mismatch, and a consumer that
//! retained the artifact identity independently refuses a consistently
//! resealed artifact of another bridge.
//!
//! What replay protects against: corruption, unsealed edits and resealed edits
//! that change the identity or the answer. What it does not protect against: a
//! producer that seals a bridge whose declared assumptions are false (they are
//! declarations, carried as unresolved obligations), or a bug shared by
//! producer and consumer (the same derivation code recomputes the value; the
//! independent oracles are in the test suites). The artifact carries no
//! interval, posterior or joint law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, BridgedClaim, MAX_BRIDGE_ASSUMPTIONS, MAX_VALIDATION_TRIALS,
    QuantityClass, TransportabilityPremise, TrialEffectPair, TrialLevelFit, TrialLevelValidation,
    ValueKind, derive_bridged_claim,
};
use antecedent_contract::evidence_ledger::parse_digest_hex;
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ReasoningView, ScientificQuantity,
    SlotAvailability,
};
use antecedent_core::{CanonicalWriter, ContractIdentities, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::measurement_model_wire::{MeasurementModelWire, model_from_wire, model_to_wire};
use crate::quantity_wire::ScientificQuantityWire;

/// The artifact major version this reader writes and accepts.
pub const SCIENTIFIC_BRIDGE_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE: &str = "scientific_bridge_v1";
/// Most bytes a scientific bridge artifact may occupy.
pub const MAX_SCIENTIFIC_BRIDGE_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Most evidence strings a stored source claim may carry.
pub const MAX_SOURCE_EVIDENCE_ITEMS: usize = 1024;

const ARTIFACT_KIND: &str = crate::artifact_kinds::SCIENTIFIC_BRIDGE;
const META_SECTION: &str = "scientific_bridge_meta";
const TAG_ARTIFACT: &str = "antecedent.scientific_bridge.artifact";
const TAG_BODY: &str = "antecedent.scientific_bridge.artifact_body";

crate::sectioned_artifact_enum! {
    /// Why a scientific bridge artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    ScientificBridgeArtifactError {
        noun: "scientific bridge",
        identity: "scientific_bridge.identity_mismatch",
        variants {
            /// The library refused the stored contract or source claim on replay.
            Refused {
                /// Registered reason code.
                code: &'static str,
                /// Namespaced `scientific_bridge.*` detail.
                detail: String,
                /// The declaration or evidence that would license the request.
                message: String,
            },
            /// A stored derived identity, value, fit or flag differs from the replay.
            ReplayMismatch(&'static str),
        }
        display(f) {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
            Self::ReplayMismatch(what) => {
                write!(f, "scientific_bridge.replay_mismatch: stored {what} does not replay")
            }
        }
    }
}

impl ScientificBridgeArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a refusal of the stored declaration on replay, a changed
    /// identity and a stored value that does not replay; absent for
    /// corruption, unsupported versions, malformed values, limits and
    /// encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, detail, message } => {
                Some((*code, detail.clone(), message.clone()))
            }
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "scientific_bridge.identity_mismatch".to_owned(),
                format!("{field} changed"),
            )),
            Self::ReplayMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "scientific_bridge.replay_mismatch".to_owned(),
                format!("the stored {what} does not replay"),
            )),
            _ => None,
        }
    }
}

fn malformed(text: impl Into<String>) -> ScientificBridgeArtifactError {
    ScientificBridgeArtifactError::Malformed(text.into())
}

fn refused(refusal: antecedent_contract::ExternalRefusal) -> ScientificBridgeArtifactError {
    ScientificBridgeArtifactError::Refused {
        code: refusal.code,
        detail: refusal.detail,
        message: refusal.remedy.unwrap_or("the bridge is not licensed").to_owned(),
    }
}

// -- wire types ---------------------------------------------------------------------------

/// A population and regime on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeWire {
    /// Population identity.
    pub population_id: String,
    /// Regime identity.
    pub regime_id: String,
}

/// One bridge end on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointWire {
    /// The quantity coordinate.
    pub quantity: ScientificQuantityWire,
    /// `QuantityClass` name.
    pub class: String,
}

/// One declared assumption on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssumptionWire {
    /// Identity.
    pub id: String,
    /// `AssumptionKind` name.
    pub kind: String,
    /// The statement.
    pub statement: String,
    /// Evidence or identity it rests on.
    pub evidence: Option<String>,
}

/// The transportability premise on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PremiseWire {
    /// Identity.
    pub id: String,
    /// The statement.
    pub statement: String,
    /// Where the bridge's evidence was established.
    pub established_in: ScopeWire,
    /// Where the claim is about.
    pub applies_to: ScopeWire,
    /// Evidence it rests on.
    pub evidence: Option<String>,
}

/// One validation trial on the wire: both effects as `f64` bit patterns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialWire {
    /// Surrogate effect bits.
    pub surrogate_bits: u64,
    /// Terminal effect bits.
    pub terminal_bits: u64,
}

/// The mapping on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum MappingWire {
    /// The value is unchanged.
    Identity,
    /// `intercept + slope * value`.
    Affine {
        /// Intercept bits.
        intercept_bits: u64,
        /// Slope bits.
        slope_bits: u64,
    },
    /// Trial-level least squares.
    TrialLevelLinear {
        /// The validation trials.
        trials: Vec<TrialWire>,
        /// Fewest trials accepted.
        min_trials: u32,
        /// Smallest coefficient of determination accepted, as bits.
        min_r_squared_bits: u64,
    },
    /// Binary-misclassification correction by a measurement model.
    MeasurementCorrection {
        /// The full model, so the consumer recomputes its identity and rates.
        model: MeasurementModelWire,
    },
}

/// The bridge contract on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractWire {
    /// `BridgeKind` name.
    pub kind: String,
    /// Source end.
    pub source: EndpointWire,
    /// Target end.
    pub target: EndpointWire,
    /// `ValueKind` name of the source value.
    pub source_value: String,
    /// `ValueKind` name of the required value.
    pub target_value: String,
    /// Where the bridge's evidence was established.
    pub evidence_scope: ScopeWire,
    /// The mapping.
    pub mapping: MappingWire,
    /// Declared assumptions.
    pub assumptions: Vec<AssumptionWire>,
    /// Transportability premise.
    pub transport: Option<PremiseWire>,
}

/// The source claim as the bridge saw it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceWire {
    /// Claim identity (hex).
    pub claim_id: String,
    /// Point value bits.
    pub value_bits: u64,
    /// Outcome units.
    pub units: Option<String>,
    /// Target identity (hex).
    pub target: String,
    /// Identification identity (hex).
    pub identification: String,
    /// Identification product identity (hex).
    pub identification_product: Option<String>,
    /// Inference binding identity (hex).
    pub inference_binding: String,
    /// Observation identity (hex).
    pub observation: String,
    /// Data snapshot identity (hex).
    pub data_snapshot: String,
    /// Evaluated-domain status name.
    pub evaluated: String,
    /// Evidence references.
    pub evidence: Vec<String>,
    /// Execution lineage (hex).
    pub execution: Option<String>,
}

/// The trial-level fit on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitWire {
    /// Intercept bits.
    pub intercept_bits: u64,
    /// Slope bits.
    pub slope_bits: u64,
    /// Coefficient of determination bits.
    pub r_squared_bits: u64,
    /// Trials fitted.
    pub trials: u32,
    /// Smallest surrogate effect bits.
    pub surrogate_min_bits: u64,
    /// Largest surrogate effect bits.
    pub surrogate_max_bits: u64,
}

/// Stored metadata of a scientific bridge artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificBridgeMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The contract.
    pub contract: ContractWire,
    /// The source claim.
    pub source: SourceWire,
    /// Contract identity (hex).
    pub contract_identity: String,
    /// Derived claim identity (hex).
    pub claim_id: String,
    /// Derived value bits.
    pub value_bits: u64,
    /// Whether the bridge carries evidence from another scope.
    pub transported: bool,
    /// The trial-level fit, when the mapping fitted one.
    pub fit: Option<FitWire>,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

// -- hex and conversion helpers ---------------------------------------------------------------

fn hex(digest: SemanticDigest) -> String {
    digest.to_hex()
}

fn digest_of(
    text: &str,
    what: &'static str,
) -> Result<SemanticDigest, ScientificBridgeArtifactError> {
    parse_digest_hex(text).ok_or_else(|| malformed(format!("{what} is not 64 hex digits")))
}

fn scope_to_wire(scope: &BridgeScope) -> ScopeWire {
    ScopeWire { population_id: scope.population_id.clone(), regime_id: scope.regime_id.clone() }
}

fn scope_from_wire(wire: &ScopeWire) -> BridgeScope {
    BridgeScope { population_id: wire.population_id.clone(), regime_id: wire.regime_id.clone() }
}

fn endpoint_to_wire(endpoint: &BridgeEndpoint) -> EndpointWire {
    EndpointWire {
        quantity: ScientificQuantityWire::from(&endpoint.quantity),
        class: endpoint.class.as_str().to_owned(),
    }
}

fn endpoint_from_wire(
    wire: &EndpointWire,
) -> Result<BridgeEndpoint, ScientificBridgeArtifactError> {
    let quantity = ScientificQuantity::try_from(wire.quantity.clone()).map_err(malformed)?;
    let class = QuantityClass::from_name(&wire.class)
        .ok_or(ScientificBridgeArtifactError::UnsupportedSemantics("quantity class"))?;
    Ok(BridgeEndpoint { quantity, class })
}

fn mapping_to_wire(mapping: &BridgeMapping) -> Result<MappingWire, ScientificBridgeArtifactError> {
    Ok(match mapping {
        BridgeMapping::Identity => MappingWire::Identity,
        BridgeMapping::Affine { intercept, slope } => {
            MappingWire::Affine { intercept_bits: intercept.to_bits(), slope_bits: slope.to_bits() }
        }
        BridgeMapping::TrialLevelLinear(validation) => MappingWire::TrialLevelLinear {
            trials: validation
                .trials
                .iter()
                .map(|t| TrialWire {
                    surrogate_bits: t.surrogate_effect.to_bits(),
                    terminal_bits: t.terminal_effect.to_bits(),
                })
                .collect(),
            min_trials: validation.min_trials,
            min_r_squared_bits: validation.min_r_squared.to_bits(),
        },
        BridgeMapping::MeasurementCorrection(model) => {
            MappingWire::MeasurementCorrection { model: model_to_wire(model) }
        }
        _ => {
            return Err(ScientificBridgeArtifactError::UnsupportedSemantics("mapping variant"));
        }
    })
}

fn mapping_from_wire(wire: &MappingWire) -> Result<BridgeMapping, ScientificBridgeArtifactError> {
    Ok(match wire {
        MappingWire::Identity => BridgeMapping::Identity,
        MappingWire::Affine { intercept_bits, slope_bits } => BridgeMapping::Affine {
            intercept: f64::from_bits(*intercept_bits),
            slope: f64::from_bits(*slope_bits),
        },
        MappingWire::TrialLevelLinear { trials, min_trials, min_r_squared_bits } => {
            if trials.len() > MAX_VALIDATION_TRIALS {
                return Err(ScientificBridgeArtifactError::LimitsExceeded("validation trials"));
            }
            BridgeMapping::TrialLevelLinear(TrialLevelValidation {
                trials: trials
                    .iter()
                    .map(|t| TrialEffectPair {
                        surrogate_effect: f64::from_bits(t.surrogate_bits),
                        terminal_effect: f64::from_bits(t.terminal_bits),
                    })
                    .collect(),
                min_trials: *min_trials,
                min_r_squared: f64::from_bits(*min_r_squared_bits),
            })
        }
        MappingWire::MeasurementCorrection { model } => BridgeMapping::MeasurementCorrection(
            Box::new(model_from_wire(model).map_err(|e| malformed(e.to_string()))?),
        ),
    })
}

fn contract_to_wire(
    contract: &BridgeContract,
) -> Result<ContractWire, ScientificBridgeArtifactError> {
    Ok(ContractWire {
        kind: contract.kind.as_str().to_owned(),
        source: endpoint_to_wire(&contract.source),
        target: endpoint_to_wire(&contract.target),
        source_value: contract.source_value.as_str().to_owned(),
        target_value: contract.target_value.as_str().to_owned(),
        evidence_scope: scope_to_wire(&contract.evidence_scope),
        mapping: mapping_to_wire(&contract.mapping)?,
        assumptions: contract
            .assumptions
            .iter()
            .map(|a| AssumptionWire {
                id: a.id.clone(),
                kind: a.kind.as_str().to_owned(),
                statement: a.statement.clone(),
                evidence: a.evidence.clone(),
            })
            .collect(),
        transport: contract.transport.as_ref().map(|p| PremiseWire {
            id: p.id.clone(),
            statement: p.statement.clone(),
            established_in: scope_to_wire(&p.established_in),
            applies_to: scope_to_wire(&p.applies_to),
            evidence: p.evidence.clone(),
        }),
    })
}

fn contract_from_wire(
    wire: &ContractWire,
) -> Result<BridgeContract, ScientificBridgeArtifactError> {
    if wire.assumptions.len() > MAX_BRIDGE_ASSUMPTIONS {
        return Err(ScientificBridgeArtifactError::LimitsExceeded("assumptions"));
    }
    let unsupported = ScientificBridgeArtifactError::UnsupportedSemantics;
    let value = |name: &str| ValueKind::from_name(name).ok_or(unsupported("value kind"));
    let mut assumptions = Vec::with_capacity(wire.assumptions.len());
    for a in &wire.assumptions {
        assumptions.push(BridgeAssumption {
            id: a.id.clone(),
            kind: AssumptionKind::from_name(&a.kind).ok_or(unsupported("assumption kind"))?,
            statement: a.statement.clone(),
            evidence: a.evidence.clone(),
        });
    }
    Ok(BridgeContract {
        kind: BridgeKind::from_name(&wire.kind).ok_or(unsupported("bridge kind"))?,
        source: endpoint_from_wire(&wire.source)?,
        target: endpoint_from_wire(&wire.target)?,
        source_value: value(&wire.source_value)?,
        target_value: value(&wire.target_value)?,
        evidence_scope: scope_from_wire(&wire.evidence_scope),
        mapping: mapping_from_wire(&wire.mapping)?,
        assumptions,
        transport: wire.transport.as_ref().map(|p| TransportabilityPremise {
            id: p.id.clone(),
            statement: p.statement.clone(),
            established_in: scope_from_wire(&p.established_in),
            applies_to: scope_from_wire(&p.applies_to),
            evidence: p.evidence.clone(),
        }),
    })
}

fn source_to_wire(claim: &ClaimEnvelope) -> Result<SourceWire, ScientificBridgeArtifactError> {
    let Some(value) = claim.value else {
        return Err(malformed("a bridged source claim carries a point value"));
    };
    let identities = &claim.identities;
    Ok(SourceWire {
        claim_id: hex(claim.claim_id),
        value_bits: value.to_bits(),
        units: claim.outcome_units.as_ref().map(ToString::to_string),
        target: hex(identities.target),
        identification: hex(identities.identification),
        identification_product: identities.identification_product.map(hex),
        inference_binding: hex(identities.inference_binding),
        observation: hex(identities.observation),
        data_snapshot: hex(identities.data_snapshot),
        evaluated: claim.domains.evaluated.as_str().to_owned(),
        evidence: claim.evidence.iter().map(ToString::to_string).collect(),
        execution: claim.execution.map(hex),
    })
}

/// The source claim as far as the bridge depends on it. Reasoning, calibration
/// and the program identity do not enter a derivation, so they are not stored;
/// the replayed claim reports them unavailable and its identification and
/// support domains `Unknown`, never the original's statuses.
fn source_from_wire(wire: &SourceWire) -> Result<ClaimEnvelope, ScientificBridgeArtifactError> {
    if wire.evidence.len() > MAX_SOURCE_EVIDENCE_ITEMS {
        return Err(ScientificBridgeArtifactError::LimitsExceeded("source evidence"));
    }
    let identities = ContractIdentities::new(
        digest_of(&wire.target, "source target identity")?,
        digest_of(&wire.identification, "source identification identity")?,
        wire.identification_product
            .as_deref()
            .map(|text| digest_of(text, "source identification product identity"))
            .transpose()?,
        None,
        digest_of(&wire.inference_binding, "source inference binding identity")?,
        digest_of(&wire.observation, "source observation identity")?,
        digest_of(&wire.data_snapshot, "source data snapshot identity")?,
    );
    let evaluated = DomainStatus::from_name(&wire.evaluated)
        .ok_or(ScientificBridgeArtifactError::UnsupportedSemantics("domain status"))?;
    let reasoning = ReasoningView::new(
        SlotAvailability::unavailable("replayed_source"),
        SlotAvailability::unavailable("replayed_source"),
        SlotAvailability::unavailable("replayed_source"),
        SlotAvailability::unavailable("replayed_source"),
    );
    let evidence: Vec<Arc<str>> = wire.evidence.iter().map(|s| Arc::from(s.as_str())).collect();
    Ok(ClaimEnvelope::new(
        digest_of(&wire.claim_id, "source claim identity")?,
        identities,
        ClaimKind::Point,
        Some(f64::from_bits(wire.value_bits)),
        wire.units.as_deref().map(Arc::from),
        reasoning,
        ClaimDomains::new(DomainStatus::Unknown, DomainStatus::Unknown, evaluated),
        wire.execution.as_deref().map(|t| digest_of(t, "source execution identity")).transpose()?,
        evidence,
    ))
}

fn fit_to_wire(fit: &TrialLevelFit) -> FitWire {
    FitWire {
        intercept_bits: fit.intercept.to_bits(),
        slope_bits: fit.slope.to_bits(),
        r_squared_bits: fit.r_squared.to_bits(),
        trials: fit.trials,
        surrogate_min_bits: fit.surrogate_min.to_bits(),
        surrogate_max_bits: fit.surrogate_max.to_bits(),
    }
}

// -- identity -----------------------------------------------------------------------------------

fn compute_identity(meta: &ScientificBridgeMeta) -> Result<String, ScientificBridgeArtifactError> {
    let encode = |error: crate::IoError| ScientificBridgeArtifactError::Encode(error.to_string());
    let contract_digest =
        CanonicalWriter::digest_bytes_hex(TAG_BODY, &to_cbor(&meta.contract).map_err(encode)?);
    let source_digest =
        CanonicalWriter::digest_bytes_hex(TAG_BODY, &to_cbor(&meta.source).map_err(encode)?);
    let mut writer = CanonicalWriter::new(TAG_ARTIFACT);
    writer
        .word(u64::from(meta.version))
        .text(&meta.feature)
        .text(&contract_digest)
        .text(&source_digest)
        .text(&meta.contract_identity)
        .text(&meta.claim_id)
        .word(meta.value_bits)
        .word(u64::from(meta.transported));
    match &meta.fit {
        None => {
            writer.word(0);
        }
        Some(fit) => {
            writer
                .word(1)
                .word(fit.intercept_bits)
                .word(fit.slope_bits)
                .word(fit.r_squared_bits)
                .word(u64::from(fit.trials))
                .word(fit.surrogate_min_bits)
                .word(fit.surrogate_max_bits);
        }
    }
    Ok(writer.finish_hex())
}

// -- consumption ----------------------------------------------------------------------------------

fn check(
    meta: &ScientificBridgeMeta,
    expected_identity: Option<&str>,
) -> Result<ScientificBridgeArtifact, ScientificBridgeArtifactError> {
    if meta.version != SCIENTIFIC_BRIDGE_ARTIFACT_VERSION {
        return Err(ScientificBridgeArtifactError::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE {
        return Err(ScientificBridgeArtifactError::UnsupportedSemantics("feature marker"));
    }
    let contract = contract_from_wire(&meta.contract)?;
    let source = source_from_wire(&meta.source)?;
    let bridged = derive_bridged_claim(
        &contract,
        &BridgeSource { claim: &source, quantity: &contract.source.quantity },
    )
    .map_err(refused)?;
    let mismatch = ScientificBridgeArtifactError::ReplayMismatch;
    if hex(bridged.contract_identity) != meta.contract_identity {
        return Err(mismatch("contract identity"));
    }
    if hex(bridged.claim.claim_id) != meta.claim_id {
        return Err(mismatch("claim identity"));
    }
    if bridged.value.to_bits() != meta.value_bits {
        return Err(mismatch("value"));
    }
    if bridged.transported != meta.transported {
        return Err(mismatch("transport flag"));
    }
    if bridged.fit.as_ref().map(fit_to_wire) != meta.fit {
        return Err(mismatch("trial-level fit"));
    }
    let identity = compute_identity(meta)?;
    if meta.identity != identity {
        return Err(ScientificBridgeArtifactError::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(ScientificBridgeArtifactError::IdentityMismatch { field: "retained_identity" });
    }
    Ok(ScientificBridgeArtifact { bridged, source, identity })
}

fn encode_meta(
    meta: &ScientificBridgeMeta,
    artifact_id: &str,
) -> Result<Vec<u8>, ScientificBridgeArtifactError> {
    if artifact_id.trim().is_empty() {
        return Err(ScientificBridgeArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes =
        to_cbor(meta).map_err(|error| ScientificBridgeArtifactError::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "scientific_bridge",
            max_bytes: MAX_SCIENTIFIC_BRIDGE_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, ScientificBridgeArtifactError))
}

fn decode_meta(bytes: &[u8]) -> Result<ScientificBridgeMeta, ScientificBridgeArtifactError> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_SCIENTIFIC_BRIDGE_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, ScientificBridgeArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(ScientificBridgeArtifactError::Malformed(
            "unsupported container layout".into(),
        ));
    };
    crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == SCIENTIFIC_BRIDGE_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(ScientificBridgeArtifactError::UnsupportedVersion { version })
        }
    })
}

// -- public type ----------------------------------------------------------------------------------

/// A validated scientific bridge: the replayed derivation, the source claim it
/// was derived from and the identity that binds them.
#[derive(Clone, Debug)]
pub struct ScientificBridgeArtifact {
    bridged: BridgedClaim,
    source: ClaimEnvelope,
    identity: String,
}

impl ScientificBridgeArtifact {
    /// Seal a bridged claim with the source claim it was derived from. The
    /// result passes the same checks a consumer applies: a bridged claim that
    /// was not derived from this source under its contract does not seal.
    ///
    /// # Errors
    /// A source that is not a finite point value, a contract the library
    /// refuses, a stored value that does not replay, or an encoding failure.
    pub fn seal(
        bridged: &BridgedClaim,
        source: &BridgeSource<'_>,
    ) -> Result<(Self, ScientificBridgeMeta), ScientificBridgeArtifactError> {
        let mut meta = ScientificBridgeMeta {
            version: SCIENTIFIC_BRIDGE_ARTIFACT_VERSION,
            feature: SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE.to_owned(),
            contract: contract_to_wire(&bridged.contract)?,
            source: source_to_wire(source.claim)?,
            contract_identity: hex(bridged.contract_identity),
            claim_id: hex(bridged.claim.claim_id),
            value_bits: bridged.value.to_bits(),
            transported: bridged.transported,
            fit: bridged.fit.as_ref().map(fit_to_wire),
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta)?;
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// Serialize a sealed bridge through the checksummed container.
    ///
    /// # Errors
    /// A bridge that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        bridged: &BridgedClaim,
        source: &BridgeSource<'_>,
        artifact_id: &str,
    ) -> Result<Vec<u8>, ScientificBridgeArtifactError> {
        let (_, meta) = Self::seal(bridged, source)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact by replaying the derivation.
    ///
    /// `expected_identity` is an artifact identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another
    /// bridge is refused.
    ///
    /// # Errors
    /// Corruption, another major version, a bound exceeded, a contract the
    /// library refuses on replay, a stored identity, value, fit or flag that
    /// does not replay, or an identity that does not match.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, ScientificBridgeArtifactError> {
        check(&decode_meta(bytes)?, expected_identity)
    }

    /// The replayed bridged claim.
    #[must_use]
    pub const fn bridged(&self) -> &BridgedClaim {
        &self.bridged
    }

    /// The source claim as the bridge saw it, separately inspectable.
    #[must_use]
    pub const fn source_claim(&self) -> &ClaimEnvelope {
        &self.source
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &ScientificBridgeMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, ScientificBridgeArtifactError> {
        check(meta, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut ScientificBridgeMeta) {
        if let Ok(identity) = compute_identity(meta) {
            meta.identity = identity;
        }
    }
}
