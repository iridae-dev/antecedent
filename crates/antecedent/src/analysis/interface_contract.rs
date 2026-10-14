//! Scientific interface contracts: declare, check a connection or a whole
//! composition before anything runs, export and independently consume.
//!
//! A syntactically compatible interface is not scientific compatibility. This
//! module routes the declarations through the layers that own each rule and
//! adds no scientific rule of its own:
//!
//! 1. **Declare** an [`InterfaceContract`]: for each produced and consumed
//!    port, the scientific quantity (units, population, regime, functional,
//!    conditioning, transform), the semantic identity, the support domain, the
//!    operations offered or required (point, mean, law, draws, joint draws,
//!    covariance, interval, posterior: none implies another), what any
//!    uncertainty means and its paradigm, the calibration label held or
//!    required, the measurement-model identities, the evidence sources or the
//!    dependence required, and the provenance stages.
//! 2. **Check a connection** ([`check_interface_connection`]) or **a
//!    composition** ([`check_interface_composition`]) from the declarations
//!    alone, before any numerical execution. Every defect is reported as a
//!    registered reason code with an `interface_contract.*` detail and the
//!    declaration that would license the connection. A feedback cycle among
//!    components is refused with its path named.
//! 3. **Export** ([`export_interface_contract`]) writes `interface_contract_v1`
//!    (layout and identity rules: `spec/`), and **consume**
//!    ([`consume_interface_contract_artifact`]) rebuilds and re-validates the
//!    declaration and recomputes both identities against an identity the
//!    consumer retained independently of the bytes.
//!    [`inspect_interface_contract_artifact`] reads the stored metadata
//!    without replaying it.
//!
//! # What is closed
//!
//! A connection never raises a label. The calibration label a connection
//! carries is the producer's: a point-only producer stays point-only, an
//! unmeasured interval stays unmeasured, and a consumer that relies on
//! measured calibration is refused rather than served an unmeasured interval.
//! Calibration of a component is not calibration of a composition, and no
//! interval or posterior is licensed by declaring one. Cyclic scientific
//! composition is refused, not unrolled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ExternalRefusal;
pub use antecedent_contract::interface_contract::{
    CalibrationRequirement, CalibrationStatus, ComponentConnection, ComposedInterfaces, Connection,
    ConsumedPort, ContractOperation, DependenceRequirement, EvidenceBasis, InterfaceContract,
    MAX_COMPOSITION_COMPONENTS, MAX_COMPOSITION_CONNECTIONS, PortFacts, ProducedPort, SupportDomain,
    UncertaintyMeaning, UncertaintyParadigm, operations_of_capabilities,
};
use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_contract::interface_contract::{
    check_composition, check_composition_with_ledger, check_connection,
    disjoint_requirements_without_ledger,
};
use antecedent_io::IoError;
pub use antecedent_io::interface_contract_artifact::InterfaceContractMeta;
use antecedent_io::interface_contract_artifact::{
    InterfaceContractArtifact, InterfaceContractArtifactError,
};

/// One refusal as plain data: registered reason code, namespaced
/// `interface_contract.*` detail, the coordinates involved and what would
/// license the request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterfaceContractRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `interface_contract.<slot>`.
    pub detail: String,
    /// The offending port, identity, operation or cycle path, when there is one.
    pub offending: Option<String>,
    /// What the consumer expects, when a comparison failed.
    pub expected: Option<String>,
    /// What the producer declares, when a comparison failed.
    pub supplied: Option<String>,
    /// What was refused and what would license it.
    pub message: String,
    /// The declaration that would license the request, when known.
    pub remedy: Option<String>,
}

impl std::fmt::Display for InterfaceContractRefusal {
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

impl std::error::Error for InterfaceContractRefusal {}

impl From<InterfaceContractRefusal> for IoError {
    fn from(refusal: InterfaceContractRefusal) -> Self {
        Self::Refused {
            code: refusal.code,
            message: format!("{}: {}", refusal.detail, refusal.message),
        }
    }
}

impl From<InterfaceContractRefusal> for crate::CausalError {
    fn from(refusal: InterfaceContractRefusal) -> Self {
        Self::Serialization(refusal.into())
    }
}

impl InterfaceContractRefusal {
    fn from_contract(refusal: &ExternalRefusal) -> Self {
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
            offending: refusal.offending.clone(),
            expected: refusal.expected.clone(),
            supplied: refusal.supplied.clone(),
            message,
            remedy: refusal.remedy.map(str::to_owned),
        }
    }

    fn all(refusals: &[ExternalRefusal]) -> Vec<Self> {
        refusals.iter().map(Self::from_contract).collect()
    }

    fn from_artifact(error: &InterfaceContractArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, message)) => {
                let remedy = match detail.as_str() {
                    "interface_contract.identity_mismatch" => {
                        "supply the artifact whose recomputed identity equals the retained identity, or regenerate it from the intended contract"
                            .to_owned()
                    }
                    _ => message.clone(),
                };
                Self {
                    code,
                    detail,
                    offending: None,
                    expected: None,
                    supplied: None,
                    message,
                    remedy: Some(remedy),
                }
            }
            None => Self {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "interface_contract.artifact_invalid".to_owned(),
                offending: None,
                expected: None,
                supplied: None,
                message: error.to_string(),
                remedy: Some("supply an intact interface_contract_v1 artifact".to_owned()),
            },
        }
    }
}

/// Decide, from the declarations alone and before any numerical execution,
/// whether `produced_port` of `producer` may feed `consumed_port` of
/// `consumer`.
///
/// # Errors
/// Every defect found, each a coded refusal with an `interface_contract.*`
/// detail and the declaration that would license the connection.
pub fn check_interface_connection(
    producer: &InterfaceContract,
    produced_port: &str,
    consumer: &InterfaceContract,
    consumed_port: &str,
) -> Result<Connection, Vec<InterfaceContractRefusal>> {
    check_connection(producer, produced_port, consumer, consumed_port)
        .map_err(|refusals| InterfaceContractRefusal::all(&refusals))
}

/// Check a set of components and the connections between them: every
/// connection, inherited calibration resolved against its upstream, and
/// feedback cycles (refused with the path named).
///
/// Without a ledger the overlap of two declared sources is not computable, so
/// a component that requires disjoint evidence among its inputs is refused
/// (`interface_contract.dependence_unknown`) rather than checked on source
/// identities alone; use [`check_interface_composition_with_ledger`].
///
/// # Errors
/// Every defect of the group of checks that failed.
pub fn check_interface_composition(
    components: &[InterfaceContract],
    connections: &[ComponentConnection],
) -> Result<ComposedInterfaces, Vec<InterfaceContractRefusal>> {
    let unchecked = disjoint_requirements_without_ledger(components);
    if !unchecked.is_empty() {
        return Err(InterfaceContractRefusal::all(&unchecked));
    }
    check_composition(components, connections)
        .map_err(|refusals| InterfaceContractRefusal::all(&refusals))
}

/// As [`check_interface_composition`], deciding the disjoint-evidence
/// requirements from the evidence ledger's row-level overlap of the declared
/// sources: sources that share observations refuse as
/// `interface_contract.shared_evidence`, and a source whose observations are
/// not computable is unknown, never disjoint.
///
/// # Errors
/// Every defect of the group of checks that failed.
pub fn check_interface_composition_with_ledger(
    components: &[InterfaceContract],
    connections: &[ComponentConnection],
    ledger: &EvidenceLedger,
) -> Result<ComposedInterfaces, Vec<InterfaceContractRefusal>> {
    check_composition_with_ledger(components, connections, ledger)
        .map_err(|refusals| InterfaceContractRefusal::all(&refusals))
}

/// An exported interface contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterfaceContractExport {
    /// The artifact bytes.
    pub bytes: Vec<u8>,
    /// BLAKE3 identity (hex) of the artifact; a consumer that retains it
    /// independently refuses a consistently resealed artifact of another
    /// contract.
    pub identity: String,
    /// The identity (hex) of the declaration itself.
    pub contract_identity: String,
}

/// Seal a contract and write it as `interface_contract_v1`.
///
/// # Errors
/// A declaration that is not well formed, an empty artifact id or an encoding
/// failure.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
)]
pub fn export_interface_contract(
    contract: &InterfaceContract,
    artifact_id: &str,
) -> Result<InterfaceContractExport, InterfaceContractRefusal> {
    let (sealed, _) = InterfaceContractArtifact::seal(contract)
        .map_err(|e| InterfaceContractRefusal::from_artifact(&e))?;
    let bytes = InterfaceContractArtifact::to_bytes(contract, artifact_id)
        .map_err(|e| InterfaceContractRefusal::from_artifact(&e))?;
    Ok(InterfaceContractExport {
        bytes,
        identity: sealed.identity().to_owned(),
        contract_identity: sealed.contract_identity().to_owned(),
    })
}

/// A contract consumed from an artifact.
#[derive(Clone, Debug)]
pub struct ConsumedInterfaceContract {
    /// The re-validated declaration, ports in canonical order.
    pub contract: InterfaceContract,
    /// The identity (hex) of the declaration.
    pub contract_identity: String,
    /// The artifact identity (hex).
    pub identity: String,
}

/// Consume an `interface_contract_v1` artifact: rebuild and re-validate the
/// declaration and recompute both identities.
///
/// `expected_identity` is an artifact identity the consumer retained
/// independently of the bytes; when given, a consistently resealed artifact of
/// another contract is refused.
///
/// # Errors
/// Corruption, another major version, an unknown feature or a required
/// feature, a declaration the library refuses, or an identity that differs.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
)]
pub fn consume_interface_contract_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> Result<ConsumedInterfaceContract, InterfaceContractRefusal> {
    let artifact = InterfaceContractArtifact::from_bytes(bytes, expected_identity)
        .map_err(|e| InterfaceContractRefusal::from_artifact(&e))?;
    Ok(ConsumedInterfaceContract {
        contract: artifact.contract().clone(),
        contract_identity: artifact.contract_identity().to_owned(),
        identity: artifact.identity().to_owned(),
    })
}

/// Read the stored metadata of an artifact without replaying it: the
/// independent inspection of the contract. Identities are not checked.
///
/// # Errors
/// Corruption, another major version or a malformed container.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
)]
pub fn inspect_interface_contract_artifact(
    bytes: &[u8],
) -> Result<InterfaceContractMeta, InterfaceContractRefusal> {
    InterfaceContractArtifact::inspect(bytes)
        .map_err(|e| InterfaceContractRefusal::from_artifact(&e))
}
