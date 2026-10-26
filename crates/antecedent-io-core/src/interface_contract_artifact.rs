//! Portable scientific interface contract (`interface_contract_v1`).
//!
//! A bounded, checksummed container holding one [`InterfaceContract`]: the
//! component identity and, for every produced and consumed port, the
//! quantity coordinate, the semantic identity, the support domain (interval
//! bounds as `f64` bit patterns), the operations, the meaning and paradigm of
//! the uncertainty, the calibration label or requirement, the
//! measurement-model identities, the evidence sources or dependence
//! requirement and the provenance stages.
//!
//! The artifact exists so a second party can inspect a component's contract
//! without running or importing the component: the meta section is one CBOR
//! map whose layout and identity rules are the normative specification in
//! `spec/` (JSON Schema and `canonicalization-and-identity.md`), and a
//! consumer that does not import this library can recompute every identity.
//!
//! A consumer of this crate trusts none of the stored identities. It rebuilds
//! the contract (every library validation applies), recomputes the contract
//! identity and the artifact identity, and refuses a stored identity that
//! differs, a newer format version, a feature marker it does not know and any
//! required feature this version does not define. A consumer that retained the
//! artifact identity independently also refuses a consistently resealed
//! artifact of another contract.
//!
//! What this protects against: corruption and edits, sealed or not. What it
//! does not: a producer that declares a population it did not use or a
//! calibration record it does not hold; those are declarations, decided at a
//! connection and verified by the artifacts they describe. The artifact
//! carries no interval, posterior or coverage statement of its own.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::ScientificQuantity;
use antecedent_contract::evidence_ledger::SourceId;
use antecedent_contract::interface_contract::{
    CalibrationRequirement, CalibrationStatus, ConsumedPort, ContractOperation,
    DependenceRequirement, EvidenceBasis, InterfaceContract, MAX_INTERFACE_PORTS,
    MAX_PORT_SET_ITEMS, ProducedPort, SupportDomain, UncertaintyMeaning, UncertaintyParadigm,
};
use antecedent_core::CanonicalWriter;
use antecedent_core::refusal::changed_identity;
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::quantity_wire::ScientificQuantityWire;

/// The artifact major version this reader writes and accepts.
pub const INTERFACE_CONTRACT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const INTERFACE_CONTRACT_ARTIFACT_FEATURE: &str = "interface_contract_v1";
/// Most bytes an interface contract artifact may occupy.
pub const MAX_INTERFACE_CONTRACT_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

const ARTIFACT_KIND: &str = crate::artifact_kinds::INTERFACE_CONTRACT;
const META_SECTION: &str = "interface_contract_meta";
/// Domain tag of the artifact identity stream.
pub const INTERFACE_CONTRACT_ARTIFACT_TAG: &str = "antecedent.interface_contract.artifact";

crate::sectioned_artifact_enum! {
    /// Why an interface contract artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    InterfaceContractArtifactError {
        noun: "interface contract",
        identity: "interface_contract.identity_mismatch",
        variants {
            /// The library refused the stored declaration.
            Refused {
                /// Registered reason code.
                code: &'static str,
                /// Namespaced `interface_contract.*` detail.
                detail: String,
                /// The declaration that would make the contract well formed.
                message: String,
            },
        }
        display(f) {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
        }
    }
}

impl InterfaceContractArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a refusal of the stored declaration and for a changed
    /// identity; absent for corruption, unsupported versions or features,
    /// malformed values, limits and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, detail, message } => {
                Some((*code, detail.clone(), message.clone()))
            }
            Self::IdentityMismatch { field } => {
                Some(changed_identity("interface_contract.identity_mismatch", field))
            }
            _ => None,
        }
    }
}

fn malformed(text: impl Into<String>) -> InterfaceContractArtifactError {
    InterfaceContractArtifactError::Malformed(text.into())
}

fn refused(refusal: antecedent_contract::ExternalRefusal) -> InterfaceContractArtifactError {
    InterfaceContractArtifactError::Refused {
        code: refusal.code,
        detail: refusal.detail,
        message: refusal.remedy.unwrap_or("the interface contract is not well formed").to_owned(),
    }
}

// -- wire types ---------------------------------------------------------------------------

/// A support domain on the wire. Fields that do not apply to `kind` are null.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupportWire {
    /// `real`, `non_negative`, `binary`, `interval` or `categorical`.
    pub kind: String,
    /// Lower bound bits (`interval`).
    pub lower_bits: Option<u64>,
    /// Upper bound bits (`interval`).
    pub upper_bits: Option<u64>,
    /// Levels (`categorical`).
    pub levels: Option<Vec<String>>,
}

/// An uncertainty meaning on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyWire {
    /// The distribution meaning name.
    pub meaning: String,
    /// The paradigm name, which must be the meaning's paradigm.
    pub paradigm: String,
}

/// A calibration label on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationWire {
    /// `none`, `point_only`, `unmeasured`, `measured` or `inherited`.
    pub status: String,
    /// Measured record identity (`measured`).
    pub record_id: Option<String>,
    /// Consumed port inherited from (`inherited`).
    pub input: Option<String>,
}

/// Declared evidence on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceWire {
    /// Whether the evidence sources are declared.
    pub declared: bool,
    /// Source identities (hex), sorted and distinct; empty when undeclared.
    pub sources: Vec<String>,
}

/// A produced port on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducedPortWire {
    /// Port name.
    pub name: String,
    /// The quantity coordinate.
    pub quantity: ScientificQuantityWire,
    /// Semantic identity.
    pub semantic_identity: String,
    /// Support domain.
    pub support: SupportWire,
    /// Operation names, in canonical order.
    pub operations: Vec<String>,
    /// Uncertainty meaning.
    pub uncertainty: Option<UncertaintyWire>,
    /// Calibration label.
    pub calibration: CalibrationWire,
    /// Measurement model identities (hex), sorted and distinct.
    pub measurement_models: Vec<String>,
    /// Evidence basis.
    pub evidence: EvidenceWire,
    /// Provenance stage names, sorted and distinct.
    pub provenance: Vec<String>,
}

/// A consumed port on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsumedPortWire {
    /// Port name.
    pub name: String,
    /// The quantity coordinate.
    pub quantity: ScientificQuantityWire,
    /// Semantic identity.
    pub semantic_identity: String,
    /// Accepted support domain.
    pub support: SupportWire,
    /// Required operation names, in canonical order.
    pub operations: Vec<String>,
    /// Required uncertainty meaning.
    pub uncertainty: Option<UncertaintyWire>,
    /// `not_required` or `measured`.
    pub calibration: String,
    /// Measurement model identities (hex), sorted and distinct.
    pub measurement_models: Vec<String>,
    /// `unconstrained`, `declared_evidence` or `disjoint_evidence`.
    pub dependence: String,
    /// Required provenance stage names, sorted and distinct.
    pub provenance: Vec<String>,
}

/// The contract on the wire. Ports are sorted by name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractWire {
    /// Component identity.
    pub component: String,
    /// Produced ports.
    pub produces: Vec<ProducedPortWire>,
    /// Consumed ports.
    pub consumes: Vec<ConsumedPortWire>,
}

/// Stored metadata of an interface contract artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InterfaceContractMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Features a reader must understand to interpret the contract. This
    /// version defines none; a reader refuses any entry.
    pub required_features: Vec<String>,
    /// The contract.
    pub contract: ContractWire,
    /// Contract identity (hex).
    pub contract_identity: String,
    /// BLAKE3 identity (hex) of the artifact: version, feature, required
    /// features and the contract identity.
    pub identity: String,
}

// -- contract -> wire ---------------------------------------------------------------------------

fn support_to_wire(support: &SupportDomain) -> SupportWire {
    let mut wire = SupportWire {
        kind: support.kind().to_owned(),
        lower_bits: None,
        upper_bits: None,
        levels: None,
    };
    match support {
        SupportDomain::Interval { lower_bits, upper_bits } => {
            wire.lower_bits = Some(*lower_bits);
            wire.upper_bits = Some(*upper_bits);
        }
        SupportDomain::Categorical { levels } => wire.levels = Some(levels.clone()),
        _ => {}
    }
    wire
}

fn uncertainty_to_wire(uncertainty: Option<UncertaintyMeaning>) -> Option<UncertaintyWire> {
    uncertainty.map(|meaning| UncertaintyWire {
        meaning: meaning.name().to_owned(),
        paradigm: meaning.paradigm().as_str().to_owned(),
    })
}

fn operations_to_wire(operations: &BTreeSet<ContractOperation>) -> Vec<String> {
    operations.iter().map(|op| op.as_str().to_owned()).collect()
}

fn calibration_to_wire(status: &CalibrationStatus) -> CalibrationWire {
    let mut wire =
        CalibrationWire { status: status.label().to_owned(), record_id: None, input: None };
    match status {
        CalibrationStatus::Measured { record_id } => wire.record_id = Some(record_id.clone()),
        CalibrationStatus::Inherited { input } => wire.input = Some(input.clone()),
        _ => {}
    }
    wire
}

/// The wire form of a contract: ports sorted by name.
#[must_use]
pub fn contract_to_wire(contract: &InterfaceContract) -> ContractWire {
    let mut produces: Vec<ProducedPortWire> = contract
        .produces
        .iter()
        .map(|port| ProducedPortWire {
            name: port.name.clone(),
            quantity: ScientificQuantityWire::from(&port.quantity),
            semantic_identity: port.semantic_identity.clone(),
            support: support_to_wire(&port.support),
            operations: operations_to_wire(&port.operations),
            uncertainty: uncertainty_to_wire(port.uncertainty),
            calibration: calibration_to_wire(&port.calibration),
            measurement_models: port.measurement_models.iter().cloned().collect(),
            evidence: match &port.evidence {
                EvidenceBasis::Undeclared => EvidenceWire { declared: false, sources: Vec::new() },
                EvidenceBasis::Declared(sources) => EvidenceWire {
                    declared: true,
                    sources: sources.iter().map(SourceId::to_hex).collect(),
                },
            },
            provenance: port.provenance.iter().cloned().collect(),
        })
        .collect();
    produces.sort_by(|a, b| a.name.cmp(&b.name));
    let mut consumes: Vec<ConsumedPortWire> = contract
        .consumes
        .iter()
        .map(|port| ConsumedPortWire {
            name: port.name.clone(),
            quantity: ScientificQuantityWire::from(&port.quantity),
            semantic_identity: port.semantic_identity.clone(),
            support: support_to_wire(&port.support),
            operations: operations_to_wire(&port.operations),
            uncertainty: uncertainty_to_wire(port.uncertainty),
            calibration: port.calibration.as_str().to_owned(),
            measurement_models: port.measurement_models.iter().cloned().collect(),
            dependence: port.dependence.as_str().to_owned(),
            provenance: port.provenance.iter().cloned().collect(),
        })
        .collect();
    consumes.sort_by(|a, b| a.name.cmp(&b.name));
    ContractWire { component: contract.component.clone(), produces, consumes }
}

// -- wire -> contract ---------------------------------------------------------------------------

type Failure = InterfaceContractArtifactError;

fn unsupported(what: &'static str) -> Failure {
    Failure::UnsupportedSemantics(what)
}

/// Whether `values` is sorted and distinct (strictly increasing).
pub(crate) fn is_strictly_increasing<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn strictly_increasing(
    values: &[String],
    what: &'static str,
    limits: &InterfaceContractIntakeLimits,
) -> Result<(), Failure> {
    if values.len() > limits.max_set_items {
        return Err(Failure::LimitsExceeded(what));
    }
    if !is_strictly_increasing(values) {
        return Err(malformed(format!("{what} are not sorted and distinct")));
    }
    Ok(())
}

fn support_from_wire(
    wire: &SupportWire,
    limits: &InterfaceContractIntakeLimits,
) -> Result<SupportDomain, Failure> {
    let no_levels = wire.levels.is_none();
    let no_bounds = wire.lower_bits.is_none() && wire.upper_bits.is_none();
    match wire.kind.as_str() {
        "real" | "non_negative" | "binary" if no_levels && no_bounds => {
            Ok(match wire.kind.as_str() {
                "real" => SupportDomain::Real,
                "non_negative" => SupportDomain::NonNegative,
                _ => SupportDomain::Binary,
            })
        }
        "interval" if no_levels => match (wire.lower_bits, wire.upper_bits) {
            (Some(lower_bits), Some(upper_bits)) => {
                Ok(SupportDomain::Interval { lower_bits, upper_bits })
            }
            _ => Err(malformed("an interval support carries both bounds")),
        },
        "categorical" if no_bounds => match &wire.levels {
            Some(levels) => {
                strictly_increasing(levels, "support levels", limits)?;
                Ok(SupportDomain::Categorical { levels: levels.clone() })
            }
            None => Err(malformed("a categorical support carries its levels")),
        },
        "real" | "non_negative" | "binary" | "interval" | "categorical" => {
            Err(malformed("a support carries fields its kind does not use"))
        }
        _ => Err(unsupported("support kind")),
    }
}

fn uncertainty_from_wire(
    wire: Option<&UncertaintyWire>,
) -> Result<Option<UncertaintyMeaning>, Failure> {
    let Some(wire) = wire else { return Ok(None) };
    let meaning =
        UncertaintyMeaning::from_name(&wire.meaning).ok_or(unsupported("uncertainty meaning"))?;
    let paradigm = UncertaintyParadigm::from_name(&wire.paradigm)
        .ok_or(unsupported("uncertainty paradigm"))?;
    if meaning.paradigm() != paradigm {
        return Err(malformed("the stated paradigm is not the meaning's paradigm"));
    }
    Ok(Some(meaning))
}

fn operations_from_wire(names: &[String]) -> Result<BTreeSet<ContractOperation>, Failure> {
    if names.len() > ContractOperation::ALL.len() {
        return Err(Failure::LimitsExceeded("operations"));
    }
    let mut operations = BTreeSet::new();
    let mut previous = 0_u64;
    for name in names {
        let operation = ContractOperation::from_name(name).ok_or(unsupported("operation"))?;
        if operation.tag() <= previous {
            return Err(malformed("operations are not in canonical order"));
        }
        previous = operation.tag();
        operations.insert(operation);
    }
    Ok(operations)
}

fn calibration_from_wire(wire: &CalibrationWire) -> Result<CalibrationStatus, Failure> {
    let stray = |present: bool| {
        if present {
            Err(malformed("a calibration label carries a field its status does not use"))
        } else {
            Ok(())
        }
    };
    match wire.status.as_str() {
        "none" | "point_only" | "unmeasured" => {
            stray(wire.record_id.is_some() || wire.input.is_some())?;
            Ok(match wire.status.as_str() {
                "none" => CalibrationStatus::None,
                "point_only" => CalibrationStatus::PointOnly,
                _ => CalibrationStatus::Unmeasured,
            })
        }
        "measured" => {
            stray(wire.input.is_some())?;
            let record_id =
                wire.record_id.clone().ok_or_else(|| malformed("a measured label has a record"))?;
            Ok(CalibrationStatus::Measured { record_id })
        }
        "inherited" => {
            stray(wire.record_id.is_some())?;
            let input =
                wire.input.clone().ok_or_else(|| malformed("an inherited label names an input"))?;
            Ok(CalibrationStatus::Inherited { input })
        }
        _ => Err(unsupported("calibration status")),
    }
}

fn quantity_from_wire(wire: &ScientificQuantityWire) -> Result<ScientificQuantity, Failure> {
    ScientificQuantity::try_from(wire.clone()).map_err(malformed)
}

fn string_set(
    values: &[String],
    what: &'static str,
    limits: &InterfaceContractIntakeLimits,
) -> Result<BTreeSet<String>, Failure> {
    strictly_increasing(values, what, limits)?;
    Ok(values.iter().cloned().collect())
}

fn produced_from_wire(
    wire: &ProducedPortWire,
    limits: &InterfaceContractIntakeLimits,
) -> Result<ProducedPort, Failure> {
    let evidence = if wire.evidence.declared {
        strictly_increasing(&wire.evidence.sources, "evidence sources", limits)?;
        let mut sources = BTreeSet::new();
        for text in &wire.evidence.sources {
            let id = SourceId::from_hex(text)
                .filter(|id| id.to_hex() == *text)
                .ok_or_else(|| malformed("an evidence source is not 64 lowercase hex digits"))?;
            sources.insert(id);
        }
        EvidenceBasis::Declared(sources)
    } else if wire.evidence.sources.is_empty() {
        EvidenceBasis::Undeclared
    } else {
        return Err(malformed("undeclared evidence lists sources"));
    };
    Ok(ProducedPort {
        name: wire.name.clone(),
        quantity: quantity_from_wire(&wire.quantity)?,
        semantic_identity: wire.semantic_identity.clone(),
        support: support_from_wire(&wire.support, limits)?,
        operations: operations_from_wire(&wire.operations)?,
        uncertainty: uncertainty_from_wire(wire.uncertainty.as_ref())?,
        calibration: calibration_from_wire(&wire.calibration)?,
        measurement_models: string_set(&wire.measurement_models, "measurement models", limits)?,
        evidence,
        provenance: string_set(&wire.provenance, "provenance stages", limits)?,
    })
}

fn consumed_from_wire(
    wire: &ConsumedPortWire,
    limits: &InterfaceContractIntakeLimits,
) -> Result<ConsumedPort, Failure> {
    Ok(ConsumedPort {
        name: wire.name.clone(),
        quantity: quantity_from_wire(&wire.quantity)?,
        semantic_identity: wire.semantic_identity.clone(),
        support: support_from_wire(&wire.support, limits)?,
        operations: operations_from_wire(&wire.operations)?,
        uncertainty: uncertainty_from_wire(wire.uncertainty.as_ref())?,
        calibration: CalibrationRequirement::from_name(&wire.calibration)
            .ok_or(unsupported("calibration requirement"))?,
        measurement_models: string_set(&wire.measurement_models, "measurement models", limits)?,
        dependence: DependenceRequirement::from_name(&wire.dependence)
            .ok_or(unsupported("dependence requirement"))?,
        provenance: string_set(&wire.provenance, "provenance stages", limits)?,
    })
}

/// Intake bounds for a stored interface contract. The defaults are the contract crate's
/// [`MAX_INTERFACE_PORTS`], [`MAX_PORT_SET_ITEMS`] and
/// [`MAX_INTERFACE_CONTRACT_ARTIFACT_BYTES`]; a caller with a larger contract passes its own
/// to [`contract_from_wire_within`] or [`InterfaceContractArtifact::from_bytes_within`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterfaceContractIntakeLimits {
    /// Most container bytes.
    pub max_bytes: usize,
    /// Most produced ports, and most consumed ports.
    pub max_ports: usize,
    /// Most items of one per-port set (models, sources, levels, stages).
    pub max_set_items: usize,
}

impl Default for InterfaceContractIntakeLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_INTERFACE_CONTRACT_ARTIFACT_BYTES,
            max_ports: MAX_INTERFACE_PORTS,
            max_set_items: MAX_PORT_SET_ITEMS,
        }
    }
}

/// Rebuild a contract from its wire form under the default intake bounds; see
/// [`contract_from_wire_within`].
///
/// # Errors
/// A wire that is not canonical, names an unknown vocabulary word or exceeds a bound.
pub fn contract_from_wire(wire: &ContractWire) -> Result<InterfaceContract, Failure> {
    contract_from_wire_within(wire, &InterfaceContractIntakeLimits::default())
}

/// Rebuild a contract from its wire form. The wire must be in canonical order
/// (ports sorted by name, every set sorted and distinct); the declaration is
/// not validated here.
///
/// # Errors
/// A wire that is not canonical, names an unknown vocabulary word or exceeds a bound of
/// `limits`.
pub fn contract_from_wire_within(
    wire: &ContractWire,
    limits: &InterfaceContractIntakeLimits,
) -> Result<InterfaceContract, Failure> {
    if wire.produces.len() > limits.max_ports || wire.consumes.len() > limits.max_ports {
        return Err(Failure::LimitsExceeded("ports"));
    }
    let names = |ports: &[&str]| -> Result<(), Failure> {
        if is_strictly_increasing(ports) {
            Ok(())
        } else {
            Err(malformed("ports are not sorted by name and distinct"))
        }
    };
    names(&wire.produces.iter().map(|p| p.name.as_str()).collect::<Vec<_>>())?;
    names(&wire.consumes.iter().map(|p| p.name.as_str()).collect::<Vec<_>>())?;
    let produces = wire.produces.iter().map(|p| produced_from_wire(p, limits)).collect::<Result<Vec<_>, _>>()?;
    let consumes = wire.consumes.iter().map(|p| consumed_from_wire(p, limits)).collect::<Result<Vec<_>, _>>()?;
    Ok(InterfaceContract { component: wire.component.clone(), produces, consumes })
}

// -- identity -----------------------------------------------------------------------------------

fn compute_identity(meta: &InterfaceContractMeta) -> String {
    let mut writer = CanonicalWriter::new(INTERFACE_CONTRACT_ARTIFACT_TAG);
    writer
        .word(u64::from(meta.version))
        .text(&meta.feature)
        .texts(&meta.required_features)
        .text(&meta.contract_identity);
    writer.finish_hex()
}

// -- consumption --------------------------------------------------------------------------------

fn check(
    meta: &InterfaceContractMeta,
    expected_identity: Option<&str>,
    limits: &InterfaceContractIntakeLimits,
) -> Result<InterfaceContractArtifact, Failure> {
    if meta.version != INTERFACE_CONTRACT_ARTIFACT_VERSION {
        return Err(Failure::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != INTERFACE_CONTRACT_ARTIFACT_FEATURE {
        return Err(unsupported("feature marker"));
    }
    if !meta.required_features.is_empty() {
        return Err(unsupported("required feature"));
    }
    let contract = contract_from_wire_within(&meta.contract, limits)?;
    contract.validate().map_err(refused)?;
    let contract_identity = contract.canonical_identity().to_hex();
    if meta.contract_identity != contract_identity {
        return Err(Failure::IdentityMismatch { field: "contract_identity" });
    }
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(Failure::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Failure::IdentityMismatch { field: "retained_identity" });
    }
    Ok(InterfaceContractArtifact { contract, contract_identity, identity })
}

fn encode_meta(meta: &InterfaceContractMeta, artifact_id: &str) -> Result<Vec<u8>, Failure> {
    if artifact_id.trim().is_empty() {
        return Err(Failure::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Failure::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "interface_contract",
            max_bytes: MAX_INTERFACE_CONTRACT_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, InterfaceContractArtifactError))
}

fn decode_meta(bytes: &[u8], max_bytes: usize) -> Result<InterfaceContractMeta, Failure> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        max_bytes,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, InterfaceContractArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(Failure::Malformed("unsupported container layout".into()));
    };
    crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == INTERFACE_CONTRACT_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(Failure::UnsupportedVersion { version })
        }
    })
}

// -- public type --------------------------------------------------------------------------------

/// A validated interface contract and the identities that bind it.
#[derive(Clone, Debug)]
pub struct InterfaceContractArtifact {
    contract: InterfaceContract,
    contract_identity: String,
    identity: String,
}

impl InterfaceContractArtifact {
    /// Seal a contract. The result passes the checks a consumer applies, so a
    /// contract the library refuses does not seal.
    ///
    /// # Errors
    /// A malformed declaration, or an encoding failure.
    pub fn seal(
        contract: &InterfaceContract,
    ) -> Result<(Self, InterfaceContractMeta), InterfaceContractArtifactError> {
        contract.validate().map_err(refused)?;
        let mut meta = InterfaceContractMeta {
            version: INTERFACE_CONTRACT_ARTIFACT_VERSION,
            feature: INTERFACE_CONTRACT_ARTIFACT_FEATURE.to_owned(),
            required_features: Vec::new(),
            contract: contract_to_wire(contract),
            contract_identity: contract.canonical_identity().to_hex(),
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta);
        let artifact = check(&meta, None, &InterfaceContractIntakeLimits::default())?;
        Ok((artifact, meta))
    }

    /// Serialize a sealed contract through the checksummed container.
    ///
    /// # Errors
    /// A contract that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        contract: &InterfaceContract,
        artifact_id: &str,
    ) -> Result<Vec<u8>, InterfaceContractArtifactError> {
        let (_, meta) = Self::seal(contract)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact: rebuild and validate the contract and recompute
    /// both identities.
    ///
    /// `expected_identity` is an artifact identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another
    /// contract is refused.
    ///
    /// # Errors
    /// Corruption, another major version, an unknown feature or a required
    /// feature, a bound exceeded, a declaration the library refuses, or a
    /// stored identity that does not match.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, InterfaceContractArtifactError> {
        Self::from_bytes_within(bytes, expected_identity, &InterfaceContractIntakeLimits::default())
    }

    /// [`Self::from_bytes`] under the caller's intake bounds.
    ///
    /// # Errors
    /// As [`Self::from_bytes`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn from_bytes_within(
        bytes: &[u8],
        expected_identity: Option<&str>,
        limits: &InterfaceContractIntakeLimits,
    ) -> Result<Self, InterfaceContractArtifactError> {
        check(&decode_meta(bytes, limits.max_bytes)?, expected_identity, limits)
    }

    /// Read the stored metadata without replaying it: the independent
    /// inspection of a contract artifact. Nothing here checks the identities
    /// or the declaration; use [`Self::from_bytes`] for that.
    ///
    /// # Errors
    /// Corruption, another major version or a malformed container.
    pub fn inspect(bytes: &[u8]) -> Result<InterfaceContractMeta, InterfaceContractArtifactError> {
        decode_meta(bytes, MAX_INTERFACE_CONTRACT_ARTIFACT_BYTES)
    }

    /// The validated contract.
    #[must_use]
    pub const fn contract(&self) -> &InterfaceContract {
        &self.contract
    }

    /// The contract identity (hex).
    #[must_use]
    pub fn contract_identity(&self) -> &str {
        &self.contract_identity
    }

    /// The artifact identity (hex).
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
        meta: &InterfaceContractMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, InterfaceContractArtifactError> {
        check(meta, expected_identity, &InterfaceContractIntakeLimits::default())
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut InterfaceContractMeta) {
        meta.identity = compute_identity(meta);
    }

    /// Serialize metadata as it is, for tests and conformance fixtures that
    /// write an edited or foreign layout.
    ///
    /// # Errors
    /// An encoding failure.
    #[doc(hidden)]
    pub fn meta_to_bytes(
        meta: &InterfaceContractMeta,
        artifact_id: &str,
    ) -> Result<Vec<u8>, InterfaceContractArtifactError> {
        encode_meta(meta, artifact_id)
    }
}
