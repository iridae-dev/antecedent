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
//! # Uncertain mappings and distributions
//!
//! A contract that declares mapping uncertainty, source dependence or an estimand
//! bridge, and a bridge whose source is the aligned draws of a posterior or a
//! predictive law, are carried in format version 2 under the feature marker
//! `scientific_bridge_extended_v1`. The declaration is part of the contract on
//! the wire, so a consumer rebuilds it, licenses it again and requires the
//! contract identity (which binds it) to replay. A bridge of draws additionally
//! embeds the source draw table, the seed of the mapping's random stream and the
//! summary of the mapped table; the consumer maps the draws again and requires the
//! mapped table's digest, mean and variance to equal the stored ones bit for bit.
//! A version-1 artifact keeps its bytes and its identity, and a version-1 reader
//! refuses a version-2 artifact before interpreting its payload.
//!
//! A bridge of a point source whose standard error was declared, through a mapping that
//! declares its uncertainty and dependence, also stores the propagation inputs (the
//! source standard error as its bit pattern and the nonlinear map, when one was asked
//! for) and the propagated mean and variance. The consumer derives the claim again with
//! those inputs and requires the propagated mean, variance, nonlinear expectation and
//! the claim identity (which binds them) to equal the stored bits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, BridgedClaim, EstimandBridge, ExposureSetting,
    MAX_BRIDGE_ASSUMPTIONS, MAX_VALIDATION_TRIALS, MappingUncertainty, NonlinearMap,
    PointPropagation, QuantityClass, SourceMappingDependence, TransportabilityPremise,
    TrialEffectPair, TrialLevelFit, TrialLevelValidation, ValueKind,
    derive_bridged_claim_propagating,
};
use antecedent_contract::evidence_ledger::parse_digest_hex;
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ReasoningView, ScientificQuantity,
    SlotAvailability,
};
use antecedent_core::refusal::{changed_identity, replay_mismatch};
use antecedent_core::{CanonicalWriter, ContractIdentities, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::bridge_distribution::{BridgedDistribution, bridge_draws};
use crate::convert::to_cbor;
use crate::distribution_artifact::{DistributionArtifact, DistributionIdentity};
use crate::measurement_model_wire::{MeasurementModelWire, model_from_wire, model_to_wire};
use crate::quantity_wire::ScientificQuantityWire;

/// The artifact major version of a point bridge with no uncertain mapping.
pub const SCIENTIFIC_BRIDGE_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the version-1 format.
pub const SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE: &str = "scientific_bridge_v1";
/// The artifact version that carries mapping uncertainty, source dependence, an
/// estimand bridge and the bridge of a distribution. A reader of
/// [`SCIENTIFIC_BRIDGE_ARTIFACT_VERSION`] alone refuses it before the payload is
/// interpreted.
pub const SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION: u16 = 2;
/// The feature marker of the extended format.
pub const SCIENTIFIC_BRIDGE_EXTENDED_FEATURE: &str = "scientific_bridge_extended_v1";
/// Most bytes a scientific bridge artifact may occupy.
pub const MAX_SCIENTIFIC_BRIDGE_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Most evidence strings a stored source claim may carry.
pub const MAX_SOURCE_EVIDENCE_ITEMS: usize = 1024;

const ARTIFACT_KIND: &str = crate::artifact_kinds::SCIENTIFIC_BRIDGE;
const META_SECTION: &str = "scientific_bridge_meta";
const DRAWS_SECTION: &str = "scientific_bridge_source_draws";
const SOURCE_DRAWS_ARTIFACT_ID: &str = "scientific_bridge_source_draws";
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
            Self::IdentityMismatch { field } => {
                Some(changed_identity("scientific_bridge.identity_mismatch", field))
            }
            Self::ReplayMismatch(what) => Some(replay_mismatch(
                "scientific_bridge.replay_mismatch",
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
#[allow(
    clippy::large_enum_variant,
    reason = "wire shape: boxing would change the serialized form and the public variant shape"
)]
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
    /// Variance of a declared affine map; carried in the extended format only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping_uncertainty: Option<MappingUncertaintyWire>,
    /// Dependence of the source estimate and the mapping; carried in the extended
    /// format only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_dependence: Option<DependenceWire>,
    /// Translation of an experiment into the target; carried in the extended
    /// format only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimand_bridge: Option<EstimandWire>,
}

impl ContractWire {
    /// Whether the contract declares anything only the extended format carries.
    #[must_use]
    pub const fn is_extended(&self) -> bool {
        self.mapping_uncertainty.is_some()
            || self.source_dependence.is_some()
            || self.estimand_bridge.is_some()
    }
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
    /// The source draws, the seed and the mapped summary, for a bridge of a
    /// distribution; carried in the extended format only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draws: Option<DrawsWire>,
    /// The propagation inputs and the propagated variance of a point source; carried
    /// in the extended format only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propagation: Option<PropagationWire>,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

/// The propagated uncertainty of a point source: the inputs the variance is built from
/// and the numbers it gave, as bit patterns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropagationWire {
    /// Source standard error bits.
    pub source_standard_error_bits: u64,
    /// The nonlinear map of the propagated law, by name, when one was requested.
    pub nonlinear: Option<String>,
    /// Propagated mean bits.
    pub mean_bits: u64,
    /// Propagated variance bits.
    pub variance_bits: u64,
    /// Bits of the nonlinear map of the propagated mean, when a map was requested.
    pub nonlinear_map_of_mean_bits: Option<u64>,
    /// Bits of the nonlinear expectation, when a map was requested.
    pub nonlinear_expectation_bits: Option<u64>,
}

/// What a bridge of draws stores beside the contract: the source table's
/// identity and digest (the table itself is a second section), the seed of the
/// mapping's random stream and the summary of the mapped table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrawsWire {
    /// Seed of the mapping's random stream.
    pub seed: u64,
    /// Scientific identity of the embedded source draw table.
    pub source_identity: DistributionIdentity,
    /// BLAKE3 digest (hex) of the embedded source table's bytes.
    pub source_digest: String,
    /// Rows of the table.
    pub n_draws: u64,
    /// Pairing identity both tables share.
    pub alignment_identity: String,
    /// Mean of the bridged column, as bits.
    pub mean_bits: u64,
    /// Variance of the bridged column, as bits.
    pub variance_bits: u64,
    /// Digest (hex) of the mapped draw table.
    pub draws_digest: String,
}

// -- hex and conversion helpers ---------------------------------------------------------------

fn propagation_to_wire(propagation: &PointPropagation) -> PropagationWire {
    PropagationWire {
        source_standard_error_bits: propagation.source_standard_error.to_bits(),
        nonlinear: propagation.nonlinear.map(|n| n.map.as_str().to_owned()),
        mean_bits: propagation.mean.to_bits(),
        variance_bits: propagation.variance.to_bits(),
        nonlinear_map_of_mean_bits: propagation.nonlinear.map(|n| n.map_of_mean.to_bits()),
        nonlinear_expectation_bits: propagation.nonlinear.map(|n| n.expectation.to_bits()),
    }
}

fn nonlinear_from_name(name: &str) -> Result<NonlinearMap, ScientificBridgeArtifactError> {
    match name {
        "square" => Ok(NonlinearMap::Square),
        "exponential" => Ok(NonlinearMap::Exponential),
        _ => Err(malformed("nonlinear map is not one the bridge propagates")),
    }
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

/// The contract without the extended declarations: the shape the version-1
/// format and the full contract request wrap.
fn contract_to_wire_core(
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
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    })
}

fn contract_to_wire(
    contract: &BridgeContract,
) -> Result<ContractWire, ScientificBridgeArtifactError> {
    let mut wire = contract_to_wire_core(contract)?;
    wire.mapping_uncertainty = contract.mapping_uncertainty.as_ref().map(uncertainty_to_wire);
    wire.source_dependence =
        contract.source_dependence.as_ref().map(dependence_to_wire).transpose()?;
    wire.estimand_bridge = contract.estimand_bridge.as_ref().map(estimand_to_wire);
    Ok(wire)
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
        mapping_uncertainty: wire.mapping_uncertainty.as_ref().map(uncertainty_from_wire),
        source_dependence: wire.source_dependence.as_ref().map(dependence_from_wire),
        estimand_bridge: wire.estimand_bridge.as_ref().map(estimand_from_wire),
    })
}

fn source_to_wire(claim: &ClaimEnvelope) -> Result<SourceWire, ScientificBridgeArtifactError> {
    let Some(value) = claim.value else {
        return Err(malformed("a bridged source claim carries a point value"));
    };
    let identities = &claim.identities;
    Ok(SourceWire {
        claim_id: claim.claim_id.to_hex(),
        value_bits: value.to_bits(),
        units: claim.outcome_units.as_ref().map(ToString::to_string),
        target: identities.target.to_hex(),
        identification: identities.identification.to_hex(),
        identification_product: identities.identification_product.map(|digest| digest.to_hex()),
        inference_binding: identities.inference_binding.to_hex(),
        observation: identities.observation.to_hex(),
        data_snapshot: identities.data_snapshot.to_hex(),
        evaluated: claim.domains.evaluated.as_str().to_owned(),
        evidence: claim.evidence.iter().map(ToString::to_string).collect(),
        execution: claim.execution.map(|digest| digest.to_hex()),
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

// -- extended declarations --------------------------------------------------------------------------

fn uncertainty_to_wire(value: &MappingUncertainty) -> MappingUncertaintyWire {
    MappingUncertaintyWire {
        id: value.id.clone(),
        intercept: value.intercept_variance.to_bits(),
        slope: value.slope_variance.to_bits(),
        covariance: value.intercept_slope_covariance.to_bits(),
    }
}

fn uncertainty_from_wire(value: &MappingUncertaintyWire) -> MappingUncertainty {
    MappingUncertainty {
        id: value.id.clone(),
        intercept_variance: f64::from_bits(value.intercept),
        slope_variance: f64::from_bits(value.slope),
        intercept_slope_covariance: f64::from_bits(value.covariance),
    }
}

fn dependence_to_wire(
    value: &SourceMappingDependence,
) -> Result<DependenceWire, ScientificBridgeArtifactError> {
    match value {
        SourceMappingDependence::Independent { evidence } => {
            Ok(DependenceWire::Independent { evidence: evidence.clone() })
        }
        SourceMappingDependence::SlopeCovariance { covariance, evidence } => {
            Ok(DependenceWire::SlopeCovariance {
                covariance: covariance.to_bits(),
                evidence: evidence.clone(),
            })
        }
        _ => Err(ScientificBridgeArtifactError::UnsupportedSemantics("source mapping dependence")),
    }
}

fn dependence_from_wire(value: &DependenceWire) -> SourceMappingDependence {
    match value {
        DependenceWire::Independent { evidence } => {
            SourceMappingDependence::Independent { evidence: evidence.clone() }
        }
        DependenceWire::SlopeCovariance { covariance, evidence } => {
            SourceMappingDependence::SlopeCovariance {
                covariance: f64::from_bits(*covariance),
                evidence: evidence.clone(),
            }
        }
    }
}

fn exposure_wire(value: &ExposureSetting) -> ExposureWire {
    ExposureWire {
        population: value.population_id.clone(),
        start: value.time_start.clone(),
        end: value.time_end.clone(),
        dose: value.dose.to_bits(),
        dose_units: value.dose_units.clone(),
        duration: value.duration.to_bits(),
        duration_units: value.duration_units.clone(),
        regime: value.exposure_regime.clone(),
    }
}

fn exposure_domain(value: &ExposureWire) -> ExposureSetting {
    ExposureSetting {
        population_id: value.population.clone(),
        time_start: value.start.clone(),
        time_end: value.end.clone(),
        dose: f64::from_bits(value.dose),
        dose_units: value.dose_units.clone(),
        duration: f64::from_bits(value.duration),
        duration_units: value.duration_units.clone(),
        exposure_regime: value.regime.clone(),
    }
}

fn estimand_to_wire(value: &EstimandBridge) -> EstimandWire {
    EstimandWire {
        id: value.id.clone(),
        experiment_id: value.experiment_id.clone(),
        experiment: exposure_wire(&value.experiment),
        target: exposure_wire(&value.target),
        recency: value.recency_delta.to_bits(),
        recency_units: value.recency_units.clone(),
        spend: value.spend_delta.to_bits(),
        spend_units: value.spend_units.clone(),
    }
}

fn estimand_from_wire(value: &EstimandWire) -> EstimandBridge {
    EstimandBridge {
        id: value.id.clone(),
        experiment_id: value.experiment_id.clone(),
        experiment: exposure_domain(&value.experiment),
        target: exposure_domain(&value.target),
        recency_delta: f64::from_bits(value.recency),
        recency_units: value.recency_units.clone(),
        spend_delta: f64::from_bits(value.spend),
        spend_units: value.spend_units.clone(),
    }
}

// -- identity -----------------------------------------------------------------------------------

/// Section word that opens the identity suffix of a bridge of draws. Written only
/// when the artifact stores draws, so a point bridge keeps its version-1 stream.
const DRAWS_IDENTITY_SECTION: u64 = 0xD2A5;

/// Section word that opens the identity suffix of a point bridge that stores a
/// propagation. Written only then, so every other point bridge keeps its stream.
const PROPAGATION_IDENTITY_SECTION: u64 = 0x9A07;

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
    if let Some(draws) = &meta.draws {
        let digest = CanonicalWriter::digest_bytes_hex(TAG_BODY, &to_cbor(draws).map_err(encode)?);
        writer.word(DRAWS_IDENTITY_SECTION).text(&digest);
    }
    if let Some(propagation) = &meta.propagation {
        let digest =
            CanonicalWriter::digest_bytes_hex(TAG_BODY, &to_cbor(propagation).map_err(encode)?);
        writer.word(PROPAGATION_IDENTITY_SECTION).text(&digest);
    }
    Ok(writer.finish_hex())
}

// -- consumption ----------------------------------------------------------------------------------

/// The format a stored version and feature marker name.
///
/// A version-1 artifact declares nothing the extended format carries; a
/// version-2 artifact declares at least one thing (one encoding per bridge). A
/// version-2 payload under the version-1 marker is another version, refused
/// before it is interpreted.
fn check_format(meta: &ScientificBridgeMeta) -> Result<(), ScientificBridgeArtifactError> {
    let extended =
        meta.contract.is_extended() || meta.draws.is_some() || meta.propagation.is_some();
    match meta.version {
        SCIENTIFIC_BRIDGE_ARTIFACT_VERSION => {
            if meta.feature != SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE {
                return Err(ScientificBridgeArtifactError::UnsupportedSemantics("feature marker"));
            }
            if extended {
                return Err(ScientificBridgeArtifactError::UnsupportedSemantics(
                    "extended declarations in a version-1 artifact",
                ));
            }
            Ok(())
        }
        SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION => {
            if meta.feature == SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE {
                return Err(ScientificBridgeArtifactError::UnsupportedVersion {
                    version: meta.version,
                });
            }
            if meta.feature != SCIENTIFIC_BRIDGE_EXTENDED_FEATURE {
                return Err(ScientificBridgeArtifactError::UnsupportedSemantics("feature marker"));
            }
            if !extended {
                return Err(ScientificBridgeArtifactError::UnsupportedSemantics(
                    "extended format without extended declarations",
                ));
            }
            Ok(())
        }
        version => Err(ScientificBridgeArtifactError::UnsupportedVersion { version }),
    }
}

/// The version and feature marker a point bridge is written under.
fn point_format(contract: &BridgeContract, propagated: bool) -> (u16, &'static str) {
    if propagated
        || contract.mapping_uncertainty.is_some()
        || contract.source_dependence.is_some()
        || contract.estimand_bridge.is_some()
    {
        (SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION, SCIENTIFIC_BRIDGE_EXTENDED_FEATURE)
    } else {
        (SCIENTIFIC_BRIDGE_ARTIFACT_VERSION, SCIENTIFIC_BRIDGE_ARTIFACT_FEATURE)
    }
}

fn check(
    meta: &ScientificBridgeMeta,
    expected_identity: Option<&str>,
) -> Result<ScientificBridgeArtifact, ScientificBridgeArtifactError> {
    check_format(meta)?;
    if meta.draws.is_some() {
        return Err(ScientificBridgeArtifactError::UnsupportedSemantics(
            "a bridge of draws is consumed with its draws",
        ));
    }
    let contract = contract_from_wire(&meta.contract)?;
    let source = source_from_wire(&meta.source)?;
    let (standard_error, nonlinear) = match &meta.propagation {
        None => (None, None),
        Some(wire) => (
            Some(f64::from_bits(wire.source_standard_error_bits)),
            wire.nonlinear.as_deref().map(nonlinear_from_name).transpose()?,
        ),
    };
    let bridged = derive_bridged_claim_propagating(
        &contract,
        &BridgeSource { claim: &source, quantity: &contract.source.quantity },
        standard_error,
        nonlinear,
    )
    .map_err(refused)?;
    let mismatch = ScientificBridgeArtifactError::ReplayMismatch;
    if bridged.contract_identity.to_hex() != meta.contract_identity {
        return Err(mismatch("contract identity"));
    }
    if bridged.claim.claim_id.to_hex() != meta.claim_id {
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
    if bridged.propagation.as_ref().map(propagation_to_wire) != meta.propagation {
        return Err(mismatch("propagated uncertainty"));
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
    source_draws: Option<&[u8]>,
    artifact_id: &str,
) -> Result<Vec<u8>, ScientificBridgeArtifactError> {
    if artifact_id.trim().is_empty() {
        return Err(ScientificBridgeArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes =
        to_cbor(meta).map_err(|error| ScientificBridgeArtifactError::Encode(error.to_string()))?;
    let mut sections = vec![crate::OutSection {
        id: META_SECTION,
        content_type: crate::CONTENT_CBOR,
        bytes: &meta_bytes,
    }];
    if let Some(draws) = source_draws {
        sections.push(crate::OutSection {
            id: DRAWS_SECTION,
            content_type: crate::CONTENT_BYTES,
            bytes: draws,
        });
    }
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "scientific_bridge",
            max_bytes: MAX_SCIENTIFIC_BRIDGE_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &sections,
    )
    .map_err(|error| crate::map_artifact_check!(error, ScientificBridgeArtifactError))
}

/// The stored metadata and, for a bridge of draws, the embedded source table.
type Decoded = (ScientificBridgeMeta, Option<Vec<u8>>);

fn decode_meta(bytes: &[u8]) -> Result<Decoded, ScientificBridgeArtifactError> {
    // A bridge of draws is a second section; try the layout of a single section first so a
    // version-1 artifact is read exactly as before.
    let (meta_bytes, source_draws) = match crate::decode_container(
        ARTIFACT_KIND,
        MAX_SCIENTIFIC_BRIDGE_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    ) {
        Ok(parts) => {
            let Ok([meta]) = <[Vec<u8>; 1]>::try_from(parts) else {
                return Err(ScientificBridgeArtifactError::Malformed(
                    "unsupported container layout".into(),
                ));
            };
            (meta, None)
        }
        Err(first) => match crate::decode_container(
            ARTIFACT_KIND,
            MAX_SCIENTIFIC_BRIDGE_ARTIFACT_BYTES,
            &[META_SECTION, DRAWS_SECTION],
            bytes,
        ) {
            Ok(parts) => {
                let Ok([meta, draws]) = <[Vec<u8>; 2]>::try_from(parts) else {
                    return Err(ScientificBridgeArtifactError::Malformed(
                        "unsupported container layout".into(),
                    ));
                };
                (meta, Some(draws))
            }
            Err(_) => {
                return Err(crate::map_artifact_check!(first, ScientificBridgeArtifactError));
            }
        },
    };
    let meta = crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == SCIENTIFIC_BRIDGE_ARTIFACT_VERSION
            || version == SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION
        {
            Ok(())
        } else {
            Err(ScientificBridgeArtifactError::UnsupportedVersion { version })
        }
    })?;
    Ok((meta, source_draws))
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
    /// A contract that declares mapping uncertainty, source dependence or an
    /// estimand bridge is sealed in the extended format; every other contract
    /// keeps the version-1 bytes.
    ///
    /// # Errors
    /// A source that is not a finite point value, a contract the library
    /// refuses, a stored value that does not replay, or an encoding failure.
    pub fn seal(
        bridged: &BridgedClaim,
        source: &BridgeSource<'_>,
    ) -> Result<(Self, ScientificBridgeMeta), ScientificBridgeArtifactError> {
        let (version, feature) = point_format(&bridged.contract, bridged.propagation.is_some());
        let mut meta = ScientificBridgeMeta {
            version,
            feature: feature.to_owned(),
            contract: contract_to_wire(&bridged.contract)?,
            source: source_to_wire(source.claim)?,
            contract_identity: bridged.contract_identity.to_hex(),
            claim_id: bridged.claim.claim_id.to_hex(),
            value_bits: bridged.value.to_bits(),
            transported: bridged.transported,
            fit: bridged.fit.as_ref().map(fit_to_wire),
            draws: None,
            propagation: bridged.propagation.as_ref().map(propagation_to_wire),
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
        encode_meta(&meta, None, artifact_id)
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
    /// does not replay, an artifact of a bridge of draws (consume it with
    /// [`ScientificBridgeDrawsArtifact`] or [`ScientificBridgeAny`]), or an
    /// identity that does not match.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, ScientificBridgeArtifactError> {
        let (meta, source_draws) = decode_meta(bytes)?;
        if source_draws.is_some() {
            return Err(ScientificBridgeArtifactError::UnsupportedSemantics(
                "a bridge of draws is consumed with its draws",
            ));
        }
        check(&meta, expected_identity)
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

/// A validated bridge of draws: the replayed mapping of the embedded source
/// table, bit for bit, and the identity that binds the contract, the table, the
/// seed and the mapped summary.
#[derive(Clone, Debug)]
pub struct ScientificBridgeDrawsArtifact {
    bridged: BridgedDistribution,
    identity: String,
}

fn check_draws(
    meta: &ScientificBridgeMeta,
    source_draws: &[u8],
    expected_identity: Option<&str>,
) -> Result<ScientificBridgeDrawsArtifact, ScientificBridgeArtifactError> {
    check_format(meta)?;
    let Some(draws) = &meta.draws else {
        return Err(ScientificBridgeArtifactError::Malformed(
            "a bridge of draws stores its draws".into(),
        ));
    };
    let mismatch = ScientificBridgeArtifactError::ReplayMismatch;
    if meta.fit.is_some() {
        return Err(mismatch("trial-level fit"));
    }
    if meta.propagation.is_some() {
        return Err(mismatch("propagated uncertainty"));
    }
    if source_draws.len() > crate::distribution_artifact::MAX_DISTRIBUTION_ARTIFACT_BYTES {
        return Err(ScientificBridgeArtifactError::LimitsExceeded("source draws"));
    }
    if CanonicalWriter::digest_bytes_hex(TAG_BODY, source_draws) != draws.source_digest {
        return Err(mismatch("source draws"));
    }
    let table = DistributionArtifact::from_bytes(source_draws, &draws.source_identity)
        .map_err(|error| malformed(error.to_string()))?;
    let contract = contract_from_wire(&meta.contract)?;
    let bridged = bridge_draws(&contract, &table, None, draws.seed).map_err(refused)?;
    let claim = bridged.claim();
    if claim.contract_identity.to_hex() != meta.contract_identity {
        return Err(mismatch("contract identity"));
    }
    if claim.claim.claim_id.to_hex() != meta.claim_id {
        return Err(mismatch("claim identity"));
    }
    if claim.mean.to_bits() != meta.value_bits || claim.mean.to_bits() != draws.mean_bits {
        return Err(mismatch("value"));
    }
    if claim.transported != meta.transported {
        return Err(mismatch("transport flag"));
    }
    if claim.variance.to_bits() != draws.variance_bits {
        return Err(mismatch("variance"));
    }
    if claim.n_draws as u64 != draws.n_draws {
        return Err(mismatch("draw count"));
    }
    if claim.alignment_identity != draws.alignment_identity {
        return Err(mismatch("alignment identity"));
    }
    if claim.draws_digest.to_hex() != draws.draws_digest {
        return Err(mismatch("mapped draws"));
    }
    if source_to_wire(&bridged.source_envelope())? != meta.source {
        return Err(mismatch("source draws summary"));
    }
    let identity = compute_identity(meta)?;
    if meta.identity != identity {
        return Err(ScientificBridgeArtifactError::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(ScientificBridgeArtifactError::IdentityMismatch { field: "retained_identity" });
    }
    Ok(ScientificBridgeDrawsArtifact { bridged, identity })
}

impl ScientificBridgeDrawsArtifact {
    /// Seal a bridge of draws: the contract, the source table it mapped, the seed
    /// of the mapping's random stream and the summary of the mapped table. The
    /// result passes the checks a consumer applies, which map the stored draws
    /// again.
    ///
    /// # Errors
    /// A bridge that does not replay from its stored draws, or an encoding failure.
    pub fn seal(
        bridged: &BridgedDistribution,
    ) -> Result<(Self, ScientificBridgeMeta, Vec<u8>), ScientificBridgeArtifactError> {
        let encode =
            |error: crate::IoError| ScientificBridgeArtifactError::Encode(error.to_string());
        let claim = bridged.claim();
        let source_bytes = bridged.source().to_bytes(SOURCE_DRAWS_ARTIFACT_ID).map_err(encode)?;
        let mut meta = ScientificBridgeMeta {
            version: SCIENTIFIC_BRIDGE_EXTENDED_ARTIFACT_VERSION,
            feature: SCIENTIFIC_BRIDGE_EXTENDED_FEATURE.to_owned(),
            contract: contract_to_wire(&claim.contract)?,
            source: source_to_wire(&bridged.source_envelope())?,
            contract_identity: claim.contract_identity.to_hex(),
            claim_id: claim.claim.claim_id.to_hex(),
            value_bits: claim.mean.to_bits(),
            transported: claim.transported,
            fit: None,
            propagation: None,
            draws: Some(DrawsWire {
                seed: claim.stream,
                source_identity: bridged.source().metadata().identity.clone(),
                source_digest: CanonicalWriter::digest_bytes_hex(TAG_BODY, &source_bytes),
                n_draws: claim.n_draws as u64,
                alignment_identity: claim.alignment_identity.clone(),
                mean_bits: claim.mean.to_bits(),
                variance_bits: claim.variance.to_bits(),
                draws_digest: claim.draws_digest.to_hex(),
            }),
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta)?;
        let artifact = check_draws(&meta, &source_bytes, None)?;
        Ok((artifact, meta, source_bytes))
    }

    /// Serialize a sealed bridge of draws through the checksummed container.
    ///
    /// # Errors
    /// A bridge that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        bridged: &BridgedDistribution,
        artifact_id: &str,
    ) -> Result<Vec<u8>, ScientificBridgeArtifactError> {
        let (_, meta, source_bytes) = Self::seal(bridged)?;
        encode_meta(&meta, Some(&source_bytes), artifact_id)
    }

    /// Consume an artifact by mapping its stored draws again.
    ///
    /// # Errors
    /// Corruption, another major version, a bound exceeded, a contract the library
    /// refuses on replay, a stored identity, summary or mapped table that does not
    /// replay bit for bit, or an identity that does not match the one the
    /// consumer retained.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, ScientificBridgeArtifactError> {
        let (meta, source_draws) = decode_meta(bytes)?;
        let Some(source_draws) = source_draws else {
            return Err(ScientificBridgeArtifactError::Malformed(
                "a bridge of draws stores its draws".into(),
            ));
        };
        check_draws(&meta, &source_draws, expected_identity)
    }

    /// The replayed bridged distribution.
    #[must_use]
    pub const fn bridged(&self) -> &BridgedDistribution {
        &self.bridged
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Validate metadata and the embedded source table as a consumer would, for
    /// tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &ScientificBridgeMeta,
        source_draws: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, ScientificBridgeArtifactError> {
        check_draws(meta, source_draws, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut ScientificBridgeMeta) {
        if let Ok(identity) = compute_identity(meta) {
            meta.identity = identity;
        }
    }
}

/// A scientific bridge artifact of either shape, consumed by replaying its
/// derivation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ScientificBridgeAny {
    /// A bridge of one mean.
    Point(ScientificBridgeArtifact),
    /// A bridge of aligned draws.
    Draws(ScientificBridgeDrawsArtifact),
}

impl ScientificBridgeAny {
    /// Consume the artifact, whichever shape it has.
    ///
    /// # Errors
    /// As [`ScientificBridgeArtifact::from_bytes`] and
    /// [`ScientificBridgeDrawsArtifact::from_bytes`].
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, ScientificBridgeArtifactError> {
        let (meta, source_draws) = decode_meta(bytes)?;
        match source_draws {
            None => check(&meta, expected_identity).map(Self::Point),
            Some(source_draws) => {
                check_draws(&meta, &source_draws, expected_identity).map(Self::Draws)
            }
        }
    }

    /// Which shape the container has, `point` or `draws`, without replaying it.
    ///
    /// # Errors
    /// Corruption, another major version or a layout that is not a bridge.
    pub fn shape(bytes: &[u8]) -> Result<&'static str, ScientificBridgeArtifactError> {
        let (_, source_draws) = decode_meta(bytes)?;
        Ok(if source_draws.is_some() { "draws" } else { "point" })
    }

    /// BLAKE3 identity (hex) of the artifact.
    #[must_use]
    pub fn identity(&self) -> &str {
        match self {
            Self::Point(artifact) => artifact.identity(),
            Self::Draws(artifact) => artifact.identity(),
        }
    }
}

/// Complete licensed bridge declaration, including uncertain estimand translation.
/// This versioned request wire extends the existing point-claim artifact owner; it
/// grants no posterior, interval or causal permission by serialization alone.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FullBridgeContractWire {
    version: u16,
    identity: [u8; 32],
    base: ContractWire,
    uncertainty: Option<MappingUncertaintyWire>,
    dependence: Option<DependenceWire>,
    estimand: Option<EstimandWire>,
}

/// The variance of a declared affine map on the wire, as bit patterns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappingUncertaintyWire {
    /// Identity of the declaration.
    pub id: String,
    /// Intercept variance bits.
    pub intercept: u64,
    /// Slope variance bits.
    pub slope: u64,
    /// Intercept-slope covariance bits.
    pub covariance: u64,
}

/// How the source estimate and the mapping share information, on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DependenceWire {
    /// The source and the mapping evidence are declared disjoint.
    Independent {
        /// The study or ledger entry the disjointness rests on.
        evidence: String,
    },
    /// Covariance of the slope estimator and the source estimator.
    SlopeCovariance {
        /// Covariance bits.
        covariance: u64,
        /// The study or ledger entry the covariance rests on.
        evidence: String,
    },
}

/// One side of an estimand bridge on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureWire {
    /// Population.
    pub population: String,
    /// Start of the time window.
    pub start: String,
    /// End of the time window.
    pub end: String,
    /// Dose bits.
    pub dose: u64,
    /// Dose units.
    pub dose_units: String,
    /// Duration bits.
    pub duration: u64,
    /// Duration units.
    pub duration_units: String,
    /// Exposure regime.
    pub regime: String,
}

/// The translation of an experiment into the bridge's target, on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EstimandWire {
    /// Identity of the translation.
    pub id: String,
    /// The experiment translated.
    pub experiment_id: String,
    /// What the experiment estimated.
    pub experiment: ExposureWire,
    /// What the bridge answers.
    pub target: ExposureWire,
    /// Recency difference bits.
    pub recency: u64,
    /// Recency units.
    pub recency_units: String,
    /// Spend difference bits.
    pub spend: u64,
    /// Spend units.
    pub spend_units: String,
}

impl FullBridgeContractWire {
    /// Encode all fields of an actually licensed contract, preserving IEEE values.
    /// # Errors
    /// Every native bridge licensing refusal and existing mapping conversion failure.
    pub fn from_domain(contract: &BridgeContract) -> Result<Self, ScientificBridgeArtifactError> {
        contract.check_licensing().map_err(refused)?;
        Ok(Self {
            version: 2,
            identity: *contract.identity().as_bytes(),
            base: contract_to_wire_core(contract)?,
            uncertainty: contract.mapping_uncertainty.as_ref().map(uncertainty_to_wire),
            dependence: contract.source_dependence.as_ref().map(dependence_to_wire).transpose()?,
            estimand: contract.estimand_bridge.as_ref().map(estimand_to_wire),
        })
    }
    /// Reconstruct the exact contract and repeat all licensing and identity checks.
    /// # Errors
    /// Unknown versions, malformed native declarations, unsupported mappings or mismatch.
    pub fn to_domain(&self) -> Result<BridgeContract, ScientificBridgeArtifactError> {
        if self.version != 2 {
            return Err(ScientificBridgeArtifactError::UnsupportedSemantics(
                "full bridge contract version",
            ));
        }
        let mut contract = contract_from_wire(&self.base)?;
        contract.mapping_uncertainty = self.uncertainty.as_ref().map(uncertainty_from_wire);
        contract.source_dependence = self.dependence.as_ref().map(dependence_from_wire);
        contract.estimand_bridge = self.estimand.as_ref().map(estimand_from_wire);
        contract.check_licensing().map_err(refused)?;
        if self.identity != *contract.identity().as_bytes() {
            return Err(ScientificBridgeArtifactError::ReplayMismatch(
                "full bridge contract identity",
            ));
        }
        Ok(contract)
    }
}

#[cfg(test)]
#[path = "bridge_artifact_tests.rs"]
mod tests;
