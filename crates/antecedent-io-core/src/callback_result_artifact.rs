//! Portable result of one callback request (`callback_result_v1`).
//!
//! A bounded, checksummed container holding the callback's declaration, the
//! request it answered and, either the accepted output or an `unavailable`
//! stage that records why no output exists in this process. The declaration is
//! carried as an instance of the one interface-contract schema (the same
//! `ContractWire` as `interface_contract_v1`) plus the execution facts a
//! contract has no slot for (provider object, dependencies, replay contract,
//! trust, operation set).
//!
//! # What this is not
//!
//! The artifact is a declaration and a portable output. It is not a promise
//! that a closure runs elsewhere and it never executes anything: a consumer
//! decodes data, rebuilds the declaration, re-validates it and the output, and
//! recomputes every identity. Resuming computation in a fresh process needs a
//! compatible provider supplied by the caller (or a declared portable runtime
//! with resolved dependencies); without one the stage is `unavailable`, and the
//! historical output stays inspectable.
//!
//! # What a consumer does
//!
//! It trusts none of the stored identities. It rebuilds and validates the
//! interface contract and the declaration, checks that the contract is the
//! declaration's own projection, recomputes the declaration identity, the
//! request identity and the output digest, re-runs the output check
//! (finite, supported, a law) and recomputes the standing. A stored trust level
//! of `native_licensed` is refused, not downgraded. A stored standing of
//! `licensed_external` for a declaration that is not eligible is refused.
//!
//! Every decoded collection is bounded before it is used: total bytes, section
//! sizes (by the container), operations, inputs, dependencies, evidence
//! sources, draws and points. No CBOR value is evaluated as code.
//!
//! No interval, posterior guarantee, joint law or identification licence is
//! carried.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_contract::ScientificQuantity;
use antecedent_contract::callback_protocol::{
    CALLBACK_OUTPUT_PORT, CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind,
    CallbackOperation, CallbackOutput, CallbackQuery, CallbackRequest, CallbackTrust,
    DependencyDeclaration, DependencyKind, InputSupplier, MAX_CALLBACK_DEPENDENCIES,
    MAX_CALLBACK_DRAWS, MAX_CALLBACK_INPUTS, MAX_CALLBACK_POINTS, MAX_CALLBACK_TEXT_BYTES,
    ReplayContract, RetryContract, UpdateDeclaration, ValidatedOutput,
    refusal::native_trust_claimed, standing_label, validate_output,
};
use antecedent_contract::evidence_ledger::SourceId;
use antecedent_contract::interface_contract::{
    ContractOperation, EvidenceBasis, InterfaceContract, UncertaintyMeaning,
};
use antecedent_contract::{ExternalRefusal, ProviderObjectIdentity};
use antecedent_core::CanonicalWriter;
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::interface_contract_artifact::{ContractWire, contract_from_wire, contract_to_wire};
use crate::quantity_wire::ScientificQuantityWire;

/// The artifact major version this reader writes and accepts.
pub const CALLBACK_RESULT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const CALLBACK_RESULT_ARTIFACT_FEATURE: &str = "callback_result_v1";
/// Most bytes a callback result artifact may occupy.
pub const MAX_CALLBACK_RESULT_ARTIFACT_BYTES: usize = 24 * 1024 * 1024;
/// Most values one stored output may hold: a posterior mean and the most draws.
pub const MAX_CALLBACK_RESULT_VALUES: usize = MAX_CALLBACK_DRAWS as usize + 1;
/// Most evidence sources one declaration may list on the wire.
pub const MAX_CALLBACK_WIRE_EVIDENCE: usize = MAX_CALLBACK_DEPENDENCIES;
/// Domain tag of the artifact identity stream.
pub const CALLBACK_RESULT_ARTIFACT_TAG: &str = "antecedent.callback_result.artifact";
/// Domain tag of the output digest stream.
pub const CALLBACK_OUTPUT_DIGEST_TAG: &str = "antecedent.callback_result.output";

const ARTIFACT_KIND: &str = crate::artifact_kinds::CALLBACK_RESULT;
const META_SECTION: &str = "callback_result_meta";
const VALUES_SECTION: &str = "callback_result_values";

crate::sectioned_artifact_enum! {
    /// Why a callback result artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    CallbackResultArtifactError {
        noun: "callback result",
        identity: "callback_protocol.identity_mismatch",
        variants {
            /// The library refused the stored declaration, request or output.
            Refused {
                /// Registered reason code.
                code: &'static str,
                /// Namespaced `callback_protocol.*` detail.
                detail: String,
                /// The declaration or evidence that would make the result well formed.
                message: String,
            },
        }
        display(f) {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
        }
    }
}

impl CallbackResultArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a refusal of the stored content and for a changed identity;
    /// absent for corruption, unsupported versions or features, malformed
    /// values, limits and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, detail, message } => {
                Some((*code, detail.clone(), message.clone()))
            }
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "callback_protocol.identity_mismatch".to_owned(),
                format!("{field} changed"),
            )),
            _ => None,
        }
    }
}

type Failure = CallbackResultArtifactError;

fn malformed(text: impl Into<String>) -> Failure {
    Failure::Malformed(text.into())
}

fn refused(refusal: ExternalRefusal) -> Failure {
    Failure::Refused {
        code: refusal.code,
        detail: refusal.detail,
        message: refusal.remedy.unwrap_or("the callback result is not well formed").to_owned(),
    }
}

// -- the record -------------------------------------------------------------------------------

/// What happened to a request.
#[derive(Clone, Debug, PartialEq)]
pub enum CallbackOutcome {
    /// An output was produced and accepted.
    Computed {
        /// The accepted output.
        output: ValidatedOutput,
        /// Provider entries that produced it, failed attempts included.
        invocations: u64,
        /// State identity the provider reported after producing it.
        state_identity: Option<String>,
    },
    /// No output exists in this process: no compatible provider was supplied,
    /// or the request was refused before any invocation.
    Unavailable {
        /// Stable reason (`callback_protocol.*` detail).
        reason: String,
    },
}

/// One request's declaration, request and outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct CallbackResultRecord {
    /// The declaration the request was answered under.
    pub declaration: CallbackDeclaration,
    /// The request.
    pub request: CallbackRequest,
    /// What happened.
    pub outcome: CallbackOutcome,
}

// -- wire types ---------------------------------------------------------------------------------

/// A dependency on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyWire {
    /// Kind name.
    pub kind: String,
    /// Dependency name.
    pub name: String,
    /// Version or digest.
    pub version: String,
}

/// An input's execution facts on the wire (its quantity and support are in the
/// contract's consumed port of the same name).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputWire {
    /// Input name.
    pub name: String,
    /// Operation name of the supplier the input consumes.
    pub operation: String,
    /// `external` or `callback`.
    pub supplier: String,
    /// Inner declaration identity (hex) for a callback supplier.
    pub callback_identity: Option<String>,
    /// Whether the inner callback declares its evidence.
    pub evidence_declared: bool,
    /// The inner callback's evidence sources (hex), sorted and distinct.
    pub evidence_sources: Vec<String>,
}

/// The posterior-update declaration on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateWire {
    /// Candidate identity.
    pub candidate_id: String,
    /// Prior or state identity.
    pub prior_id: String,
    /// The possible observation.
    pub observation: ScientificQuantityWire,
}

/// The replay contract on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayWire {
    /// `deterministic_pure`, `seeded_stochastic`, `stateful` or `side_effecting`.
    pub kind: String,
    /// Semantic stream identity of a seeded callback.
    pub stream_id: Option<String>,
    /// Whether a stateful callback reports a state identity.
    pub exposes_state_identity: Option<bool>,
    /// `no_retry` or `idempotent_key` for a side-effecting callback.
    pub retry: Option<String>,
    /// Attempt bound of an idempotent retry.
    pub max_attempts: Option<u32>,
}

/// The trust the provider claims, on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustWire {
    /// `unverified`, `externally_attested` or `verified_extension`.
    pub level: String,
    /// Attestor of an attested callback.
    pub attestor: Option<String>,
    /// Receipt digest of a verified callback.
    pub receipt_digest: Option<String>,
}

/// Everything of the declaration that is not in the contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolWire {
    /// `law`, `posterior` or `nested`.
    pub kind: String,
    /// Provider identity.
    pub provider_id: String,
    /// Object identity.
    pub object_id: String,
    /// Version identity.
    pub version_id: String,
    /// Snapshot identity.
    pub snapshot_id: String,
    /// Request fingerprint of the provider object.
    pub request_id: String,
    /// Distribution meaning name.
    pub meaning: String,
    /// Declared operation names, sorted and distinct.
    pub operations: Vec<String>,
    /// Whether the callback's own evidence is declared.
    pub own_evidence_declared: bool,
    /// Its own evidence sources (hex), sorted and distinct.
    pub own_evidence: Vec<String>,
    /// The posterior-update declaration.
    pub update: Option<UpdateWire>,
    /// Inputs, sorted by name.
    pub inputs: Vec<InputWire>,
    /// `complete`, `opaque` or `undeclared`.
    pub dependencies_status: String,
    /// Dependencies of a complete declaration, sorted.
    pub dependencies: Vec<DependencyWire>,
    /// The replay contract.
    pub replay: ReplayWire,
    /// The claimed trust.
    pub trust: TrustWire,
}

/// A query on the wire; floating-point values are IEEE bit patterns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryWire {
    /// Operation name.
    pub operation: String,
    /// Draw count, for a sample or an update.
    pub draws: Option<u32>,
    /// Thresholds or levels, as bit patterns.
    pub points_bits: Vec<u64>,
    /// Observation of an update, as a bit pattern.
    pub observation_bits: Option<u64>,
}

/// One input digest on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputDigestWire {
    /// Input name.
    pub name: String,
    /// Digest (hex) of the value supplied.
    pub digest: String,
}

/// A request on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestWire {
    /// The query.
    pub query: QueryWire,
    /// Master seed.
    pub seed: u64,
    /// Input digests, sorted by name.
    pub input_digests: Vec<InputDigestWire>,
    /// Idempotency key.
    pub idempotency_key: Option<String>,
}

/// The stage outcome on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageWire {
    /// `computed` or `unavailable`.
    pub status: String,
    /// Reason an `unavailable` stage has no output.
    pub reason: Option<String>,
    /// Provider entries, failed attempts included.
    pub invocations: u64,
    /// Reported state identity.
    pub state_identity: Option<String>,
    /// Number of values in the values section.
    pub value_count: u64,
}

/// The stored metadata: a single CBOR map.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallbackResultMeta {
    /// Format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Features a reader must understand.
    pub required_features: Vec<String>,
    /// The declaration as an instance of the interface-contract schema.
    pub contract: ContractWire,
    /// The execution facts of the declaration.
    pub protocol: ProtocolWire,
    /// Declaration identity (hex).
    pub declaration_identity: String,
    /// The request.
    pub request: RequestWire,
    /// Request identity (hex).
    pub request_identity: String,
    /// The stage.
    pub stage: StageWire,
    /// `diagnostic` or `licensed_external`, best case under the declaration.
    pub standing: String,
    /// Digest (hex) of the output values, bit-exact.
    pub output_digest: String,
    /// Identity (hex) of the artifact.
    pub identity: String,
}

// -- declaration <-> wire -----------------------------------------------------------------------

fn evidence_to_wire(evidence: &EvidenceBasis) -> (bool, Vec<String>) {
    match evidence {
        EvidenceBasis::Undeclared => (false, Vec::new()),
        EvidenceBasis::Declared(sources) => (true, sources.iter().map(SourceId::to_hex).collect()),
    }
}

fn evidence_from_wire(declared: bool, sources: &[String]) -> Result<EvidenceBasis, Failure> {
    if sources.len() > MAX_CALLBACK_WIRE_EVIDENCE {
        return Err(Failure::LimitsExceeded("evidence"));
    }
    if !declared {
        if sources.is_empty() {
            return Ok(EvidenceBasis::Undeclared);
        }
        return Err(malformed("undeclared evidence lists sources"));
    }
    if sources.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(malformed("evidence sources are not sorted and distinct"));
    }
    let mut set = BTreeSet::new();
    for hex in sources {
        set.insert(
            SourceId::from_hex(hex).ok_or_else(|| malformed("evidence source is not a digest"))?,
        );
    }
    Ok(EvidenceBasis::Declared(set))
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat pass over every declared protocol field, each lowered or checked in turn"
)]
fn protocol_to_wire(declaration: &CallbackDeclaration) -> ProtocolWire {
    let (own_evidence_declared, own_evidence) = evidence_to_wire(&declaration.evidence);
    let mut inputs: Vec<InputWire> = declaration
        .inputs
        .iter()
        .map(|input| {
            let (supplier, callback_identity, evidence_declared, evidence_sources) =
                match &input.supplier {
                    InputSupplier::External => ("external", None, false, Vec::new()),
                    InputSupplier::Callback { identity, evidence } => {
                        let (declared, sources) = evidence_to_wire(evidence);
                        ("callback", Some(identity.clone()), declared, sources)
                    }
                };
            InputWire {
                name: input.name.clone(),
                operation: input.operation.as_str().to_owned(),
                supplier: supplier.to_owned(),
                callback_identity,
                evidence_declared,
                evidence_sources,
            }
        })
        .collect();
    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    let (dependencies_status, dependencies) = match &declaration.dependencies {
        DependencyDeclaration::Complete(items) => {
            let mut sorted: Vec<&CallbackDependency> = items.iter().collect();
            sorted.sort();
            (
                "complete",
                sorted
                    .into_iter()
                    .map(|d| DependencyWire {
                        kind: d.kind.as_str().to_owned(),
                        name: d.name.clone(),
                        version: d.version.clone(),
                    })
                    .collect::<Vec<_>>(),
            )
        }
        DependencyDeclaration::Opaque => ("opaque", Vec::new()),
        DependencyDeclaration::Undeclared => ("undeclared", Vec::new()),
    };
    let replay = match &declaration.replay {
        ReplayContract::DeterministicPure => ReplayWire {
            kind: "deterministic_pure".to_owned(),
            stream_id: None,
            exposes_state_identity: None,
            retry: None,
            max_attempts: None,
        },
        ReplayContract::SeededStochastic { stream_id } => ReplayWire {
            kind: "seeded_stochastic".to_owned(),
            stream_id: Some(stream_id.clone()),
            exposes_state_identity: None,
            retry: None,
            max_attempts: None,
        },
        ReplayContract::Stateful { exposes_state_identity } => ReplayWire {
            kind: "stateful".to_owned(),
            stream_id: None,
            exposes_state_identity: Some(*exposes_state_identity),
            retry: None,
            max_attempts: None,
        },
        ReplayContract::SideEffecting { retry } => {
            let (retry, max_attempts) = match retry {
                RetryContract::NoRetry => ("no_retry", None),
                RetryContract::IdempotentKey { max_attempts } => {
                    ("idempotent_key", Some(*max_attempts))
                }
            };
            ReplayWire {
                kind: "side_effecting".to_owned(),
                stream_id: None,
                exposes_state_identity: None,
                retry: Some(retry.to_owned()),
                max_attempts,
            }
        }
    };
    let trust = match &declaration.trust {
        CallbackTrust::Unverified => {
            TrustWire { level: "unverified".to_owned(), attestor: None, receipt_digest: None }
        }
        CallbackTrust::ExternallyAttested { attestor } => TrustWire {
            level: "externally_attested".to_owned(),
            attestor: Some(attestor.clone()),
            receipt_digest: None,
        },
        CallbackTrust::ExactRequestVerified { receipt_digest } => TrustWire {
            level: "verified_extension".to_owned(),
            attestor: None,
            receipt_digest: Some(receipt_digest.clone()),
        },
    };
    ProtocolWire {
        kind: declaration.kind.as_str().to_owned(),
        provider_id: declaration.identity.provider_id.clone(),
        object_id: declaration.identity.object_id.clone(),
        version_id: declaration.identity.version_id.clone(),
        snapshot_id: declaration.identity.snapshot_id.clone(),
        request_id: declaration.identity.request_id.clone(),
        meaning: UncertaintyMeaning::new(declaration.meaning).name().to_owned(),
        operations: {
            let mut names: Vec<String> =
                declaration.operations.iter().map(|op| op.as_str().to_owned()).collect();
            names.sort_unstable();
            names
        },
        own_evidence_declared,
        own_evidence,
        update: declaration.update.as_ref().map(|update| UpdateWire {
            candidate_id: update.candidate_id.clone(),
            prior_id: update.prior_id.clone(),
            observation: ScientificQuantityWire::from(&update.observation),
        }),
        inputs,
        dependencies_status: dependencies_status.to_owned(),
        dependencies,
        replay,
        trust,
    }
}

fn sorted_distinct(values: &[&str], what: &'static str) -> Result<(), Failure> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        Err(malformed(format!("{what} are not sorted and distinct")))
    } else {
        Ok(())
    }
}

fn replay_from_wire(wire: &ReplayWire) -> Result<ReplayContract, Failure> {
    let bare = wire.stream_id.is_none()
        && wire.exposes_state_identity.is_none()
        && wire.retry.is_none()
        && wire.max_attempts.is_none();
    match wire.kind.as_str() {
        "deterministic_pure" if bare => Ok(ReplayContract::DeterministicPure),
        "seeded_stochastic"
            if wire.exposes_state_identity.is_none()
                && wire.retry.is_none()
                && wire.max_attempts.is_none() =>
        {
            let stream_id = wire.stream_id.clone().ok_or_else(|| malformed("stream_id"))?;
            Ok(ReplayContract::SeededStochastic { stream_id })
        }
        "stateful"
            if wire.stream_id.is_none() && wire.retry.is_none() && wire.max_attempts.is_none() =>
        {
            let exposes = wire.exposes_state_identity.ok_or_else(|| malformed("state flag"))?;
            Ok(ReplayContract::Stateful { exposes_state_identity: exposes })
        }
        "side_effecting" if wire.stream_id.is_none() && wire.exposes_state_identity.is_none() => {
            match (wire.retry.as_deref(), wire.max_attempts) {
                (Some("no_retry"), None) => {
                    Ok(ReplayContract::SideEffecting { retry: RetryContract::NoRetry })
                }
                (Some("idempotent_key"), Some(max_attempts)) => Ok(ReplayContract::SideEffecting {
                    retry: RetryContract::IdempotentKey { max_attempts },
                }),
                _ => Err(malformed("retry contract")),
            }
        }
        _ => Err(Failure::UnsupportedSemantics("replay contract")),
    }
}

fn trust_from_wire(wire: &TrustWire) -> Result<CallbackTrust, Failure> {
    match (wire.level.as_str(), &wire.attestor, &wire.receipt_digest) {
        ("unverified", None, None) => Ok(CallbackTrust::Unverified),
        ("externally_attested", Some(attestor), None) => {
            Ok(CallbackTrust::ExternallyAttested { attestor: attestor.clone() })
        }
        ("verified_extension", None, Some(receipt_digest)) => {
            Ok(CallbackTrust::ExactRequestVerified { receipt_digest: receipt_digest.clone() })
        }
        // Metadata never confers native licence.
        ("native_licensed", _, _) => Err(refused(native_trust_claimed())),
        _ => Err(Failure::UnsupportedSemantics("trust level")),
    }
}

fn dependencies_from_wire(wire: &ProtocolWire) -> Result<DependencyDeclaration, Failure> {
    match wire.dependencies_status.as_str() {
        "complete" => {
            if wire.dependencies.len() > MAX_CALLBACK_DEPENDENCIES {
                return Err(Failure::LimitsExceeded("dependencies"));
            }
            let mut items = Vec::with_capacity(wire.dependencies.len());
            for dependency in &wire.dependencies {
                let kind = DependencyKind::from_name(&dependency.kind)
                    .ok_or(Failure::UnsupportedSemantics("dependency kind"))?;
                items.push(CallbackDependency::new(
                    kind,
                    dependency.name.clone(),
                    dependency.version.clone(),
                ));
            }
            if items.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(malformed("dependencies are not sorted and distinct"));
            }
            Ok(DependencyDeclaration::Complete(items))
        }
        "opaque" | "undeclared" if wire.dependencies.is_empty() => {
            Ok(if wire.dependencies_status == "opaque" {
                DependencyDeclaration::Opaque
            } else {
                DependencyDeclaration::Undeclared
            })
        }
        _ => Err(Failure::UnsupportedSemantics("dependency declaration")),
    }
}

fn contract_of(wire: &ContractWire) -> Result<InterfaceContract, Failure> {
    contract_from_wire(wire).map_err(|error| match error.refusal() {
        Some((code, detail, message)) => Failure::Refused { code, detail, message },
        None => malformed(error.to_string()),
    })
}

fn declaration_from_wire(
    contract_wire: &ContractWire,
    wire: &ProtocolWire,
) -> Result<CallbackDeclaration, Failure> {
    let contract = contract_of(contract_wire)?;
    let output = contract
        .produced(CALLBACK_OUTPUT_PORT)
        .filter(|_| contract.produces.len() == 1)
        .ok_or_else(|| {
        malformed("a callback contract has exactly one produced port `output`")
    })?;
    let kind = CallbackKind::from_name(&wire.kind).ok_or(Failure::UnsupportedSemantics("kind"))?;
    let meaning = UncertaintyMeaning::from_name(&wire.meaning)
        .ok_or(Failure::UnsupportedSemantics("meaning"))?
        .meaning;
    if wire.operations.len() > CallbackOperation::ALL.len() {
        return Err(Failure::LimitsExceeded("operations"));
    }
    let names: Vec<&str> = wire.operations.iter().map(String::as_str).collect();
    sorted_distinct(&names, "operations")?;
    let mut operations = BTreeSet::new();
    for name in &wire.operations {
        operations.insert(
            CallbackOperation::from_name(name).ok_or(Failure::UnsupportedSemantics("operation"))?,
        );
    }
    if wire.inputs.len() > MAX_CALLBACK_INPUTS || contract.consumes.len() != wire.inputs.len() {
        return Err(Failure::LimitsExceeded("inputs"));
    }
    let names: Vec<&str> = wire.inputs.iter().map(|i| i.name.as_str()).collect();
    sorted_distinct(&names, "inputs")?;
    let mut inputs = Vec::with_capacity(wire.inputs.len());
    for input in &wire.inputs {
        let port = contract
            .consumed(&input.name)
            .ok_or_else(|| malformed("an input has no consumed port"))?;
        let operation = CallbackOperation::from_name(&input.operation)
            .ok_or(Failure::UnsupportedSemantics("input operation"))?;
        let supplier = match (input.supplier.as_str(), &input.callback_identity) {
            ("external", None) if !input.evidence_declared && input.evidence_sources.is_empty() => {
                InputSupplier::External
            }
            ("callback", Some(identity)) => InputSupplier::Callback {
                identity: identity.clone(),
                evidence: evidence_from_wire(input.evidence_declared, &input.evidence_sources)?,
            },
            _ => return Err(Failure::UnsupportedSemantics("input supplier")),
        };
        inputs.push(CallbackInput {
            name: input.name.clone(),
            quantity: port.quantity.clone(),
            semantic_identity: port.semantic_identity.clone(),
            support: port.support.clone(),
            operation,
            uncertainty: port.uncertainty.map(|u| u.meaning),
            supplier,
        });
    }
    let update = match &wire.update {
        None => None,
        Some(update) => {
            let observation =
                ScientificQuantity::try_from(update.observation.clone()).map_err(malformed)?;
            Some(UpdateDeclaration {
                candidate_id: update.candidate_id.clone(),
                prior_id: update.prior_id.clone(),
                observation,
            })
        }
    };
    Ok(CallbackDeclaration {
        component: contract.component.clone(),
        kind,
        identity: ProviderObjectIdentity {
            provider_id: wire.provider_id.clone(),
            object_id: wire.object_id.clone(),
            version_id: wire.version_id.clone(),
            snapshot_id: wire.snapshot_id.clone(),
            request_id: wire.request_id.clone(),
        },
        quantity: output.quantity.clone(),
        semantic_identity: output.semantic_identity.clone(),
        meaning,
        support: output.support.clone(),
        operations,
        update,
        inputs,
        dependencies: dependencies_from_wire(wire)?,
        replay: replay_from_wire(&wire.replay)?,
        trust: trust_from_wire(&wire.trust)?,
        evidence: evidence_from_wire(wire.own_evidence_declared, &wire.own_evidence)?,
    })
}

// -- request <-> wire ---------------------------------------------------------------------------

fn request_to_wire(request: &CallbackRequest) -> RequestWire {
    let query = match &request.query {
        CallbackQuery::Mean => QueryWire {
            operation: "mean".to_owned(),
            draws: None,
            points_bits: Vec::new(),
            observation_bits: None,
        },
        CallbackQuery::Sample { draws } => QueryWire {
            operation: "sample".to_owned(),
            draws: Some(*draws),
            points_bits: Vec::new(),
            observation_bits: None,
        },
        CallbackQuery::Cdf { thresholds } => QueryWire {
            operation: "cdf".to_owned(),
            draws: None,
            points_bits: thresholds.iter().map(|v| v.to_bits()).collect(),
            observation_bits: None,
        },
        CallbackQuery::Quantile { levels } => QueryWire {
            operation: "quantile".to_owned(),
            draws: None,
            points_bits: levels.iter().map(|v| v.to_bits()).collect(),
            observation_bits: None,
        },
        CallbackQuery::PosteriorUpdate { observation, draws } => QueryWire {
            operation: "posterior_update".to_owned(),
            draws: Some(*draws),
            points_bits: Vec::new(),
            observation_bits: Some(observation.to_bits()),
        },
    };
    RequestWire {
        query,
        seed: request.seed,
        input_digests: request
            .input_digests
            .iter()
            .map(|(name, digest)| InputDigestWire { name: name.clone(), digest: digest.clone() })
            .collect(),
        idempotency_key: request.idempotency_key.clone(),
    }
}

fn request_from_wire(wire: &RequestWire) -> Result<CallbackRequest, Failure> {
    if wire.query.points_bits.len() > MAX_CALLBACK_POINTS {
        return Err(Failure::LimitsExceeded("points"));
    }
    if wire.query.draws.is_some_and(|d| d > MAX_CALLBACK_DRAWS) {
        return Err(Failure::LimitsExceeded("draws"));
    }
    if wire.input_digests.len() > MAX_CALLBACK_INPUTS {
        return Err(Failure::LimitsExceeded("input digests"));
    }
    let points: Vec<f64> = wire.query.points_bits.iter().map(|b| f64::from_bits(*b)).collect();
    let operation = CallbackOperation::from_name(&wire.query.operation)
        .ok_or(Failure::UnsupportedSemantics("query operation"))?;
    let shape_ok = match operation {
        CallbackOperation::Mean => {
            wire.query.draws.is_none() && points.is_empty() && wire.query.observation_bits.is_none()
        }
        CallbackOperation::Sample => {
            wire.query.draws.is_some() && points.is_empty() && wire.query.observation_bits.is_none()
        }
        CallbackOperation::Cdf | CallbackOperation::Quantile => {
            wire.query.draws.is_none()
                && !points.is_empty()
                && wire.query.observation_bits.is_none()
        }
        CallbackOperation::PosteriorUpdate => {
            wire.query.draws.is_some() && points.is_empty() && wire.query.observation_bits.is_some()
        }
    };
    if !shape_ok {
        return Err(malformed("query fields do not match its operation"));
    }
    let query = match operation {
        CallbackOperation::Mean => CallbackQuery::Mean,
        CallbackOperation::Sample => {
            CallbackQuery::Sample { draws: wire.query.draws.unwrap_or_default() }
        }
        CallbackOperation::Cdf => CallbackQuery::Cdf { thresholds: points },
        CallbackOperation::Quantile => CallbackQuery::Quantile { levels: points },
        CallbackOperation::PosteriorUpdate => CallbackQuery::PosteriorUpdate {
            observation: f64::from_bits(wire.query.observation_bits.unwrap_or_default()),
            draws: wire.query.draws.unwrap_or_default(),
        },
    };
    let names: Vec<&str> = wire.input_digests.iter().map(|d| d.name.as_str()).collect();
    sorted_distinct(&names, "input digests")?;
    let mut input_digests = BTreeMap::new();
    for digest in &wire.input_digests {
        if digest.digest.len() > MAX_CALLBACK_TEXT_BYTES {
            return Err(Failure::LimitsExceeded("input digest"));
        }
        input_digests.insert(digest.name.clone(), digest.digest.clone());
    }
    Ok(CallbackRequest {
        query,
        seed: wire.seed,
        input_digests,
        idempotency_key: wire.idempotency_key.clone(),
    })
}

// -- identity -----------------------------------------------------------------------------------

/// Bit-exact digest (hex) of output values, as stored in the artifact and used
/// to bind a nested callback's input in its request.
#[must_use]
pub fn output_values_digest(values: &[f64]) -> String {
    let mut writer = CanonicalWriter::new(CALLBACK_OUTPUT_DIGEST_TAG);
    writer.reals(values);
    writer.finish_hex()
}

fn compute_identity(meta: &CallbackResultMeta) -> String {
    let mut writer = CanonicalWriter::new(CALLBACK_RESULT_ARTIFACT_TAG);
    writer
        .word(u64::from(meta.version))
        .text(&meta.feature)
        .texts(&meta.required_features)
        .text(&meta.declaration_identity)
        .text(&meta.request_identity)
        .text(&meta.stage.status)
        .opt_text(meta.stage.reason.as_deref())
        .word(meta.stage.invocations)
        .opt_text(meta.stage.state_identity.as_deref())
        .word(meta.stage.value_count)
        .text(&meta.standing)
        .text(&meta.output_digest);
    writer.finish_hex()
}

// -- public type --------------------------------------------------------------------------------

/// A validated callback result and the identities that bind it.
#[derive(Clone, Debug)]
pub struct CallbackResultArtifact {
    record: CallbackResultRecord,
    declaration_identity: String,
    request_identity: String,
    identity: String,
}

fn values_of(record: &CallbackResultRecord) -> Vec<f64> {
    match &record.outcome {
        CallbackOutcome::Computed { output, .. } => output.output().values(),
        CallbackOutcome::Unavailable { .. } => Vec::new(),
    }
}

fn build_meta(record: &CallbackResultRecord) -> CallbackResultMeta {
    let declaration = &record.declaration;
    let (stage, values) = match &record.outcome {
        CallbackOutcome::Computed { invocations, state_identity, .. } => (
            StageWire {
                status: "computed".to_owned(),
                reason: None,
                invocations: *invocations,
                state_identity: state_identity.clone(),
                value_count: values_of(record).len() as u64,
            },
            values_of(record),
        ),
        CallbackOutcome::Unavailable { reason } => (
            StageWire {
                status: "unavailable".to_owned(),
                reason: Some(reason.clone()),
                invocations: 0,
                state_identity: None,
                value_count: 0,
            },
            Vec::new(),
        ),
    };
    let mut meta = CallbackResultMeta {
        version: CALLBACK_RESULT_ARTIFACT_VERSION,
        feature: CALLBACK_RESULT_ARTIFACT_FEATURE.to_owned(),
        required_features: Vec::new(),
        contract: contract_to_wire(&declaration.interface_contract()),
        protocol: protocol_to_wire(declaration),
        declaration_identity: declaration.canonical_identity().to_hex(),
        request: request_to_wire(&record.request),
        request_identity: record.request.identity(declaration).to_hex(),
        stage,
        standing: standing_label(declaration, false).as_str().to_owned(),
        output_digest: output_values_digest(&values),
        identity: String::new(),
    };
    meta.identity = compute_identity(&meta);
    meta
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat pass over every declared protocol field, each lowered or checked in turn"
)]
fn check(
    meta: &CallbackResultMeta,
    values: &[f64],
    expected_identity: Option<&str>,
) -> Result<CallbackResultArtifact, Failure> {
    if meta.version != CALLBACK_RESULT_ARTIFACT_VERSION {
        return Err(Failure::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != CALLBACK_RESULT_ARTIFACT_FEATURE {
        return Err(Failure::UnsupportedSemantics("feature marker"));
    }
    if !meta.required_features.is_empty() {
        return Err(Failure::UnsupportedSemantics("required feature"));
    }
    let declaration = declaration_from_wire(&meta.contract, &meta.protocol)?;
    declaration.validate().map_err(refused)?;
    // The stored contract is the declaration's own projection, not a separate claim.
    let stored_contract = contract_of(&meta.contract)?;
    if stored_contract.canonical_identity() != declaration.interface_contract().canonical_identity()
    {
        return Err(Failure::IdentityMismatch { field: "contract" });
    }
    let declaration_identity = declaration.canonical_identity().to_hex();
    if meta.declaration_identity != declaration_identity {
        return Err(Failure::IdentityMismatch { field: "declaration_identity" });
    }
    let request = request_from_wire(&meta.request)?;
    request.query.validate().map_err(refused)?;
    for text in [
        request.idempotency_key.as_deref(),
        meta.stage.reason.as_deref(),
        meta.stage.state_identity.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if text.len() > MAX_CALLBACK_TEXT_BYTES {
            return Err(Failure::LimitsExceeded("text"));
        }
    }
    let request_identity = request.identity(&declaration).to_hex();
    if meta.request_identity != request_identity {
        return Err(Failure::IdentityMismatch { field: "request_identity" });
    }
    if meta.stage.status == "computed"
        && !declaration.operations.contains(&request.query.operation())
    {
        return Err(malformed("the query's operation is not a declared operation"));
    }
    let outcome = match meta.stage.status.as_str() {
        "computed" => {
            if meta.stage.reason.is_some() || meta.stage.invocations == 0 {
                return Err(malformed("a computed stage has no reason and at least one entry"));
            }
            if usize::try_from(meta.stage.value_count).ok() != Some(values.len()) {
                return Err(malformed("value count disagrees with the values section"));
            }
            let output = rebuild_output(&request.query, values)?;
            let output = validate_output(
                &request.query,
                output,
                &declaration.support,
                MAX_CALLBACK_RESULT_VALUES,
            )
            .map_err(refused)?;
            CallbackOutcome::Computed {
                output,
                invocations: meta.stage.invocations,
                state_identity: meta.stage.state_identity.clone(),
            }
        }
        "unavailable" => {
            let reason = meta
                .stage
                .reason
                .clone()
                .ok_or_else(|| malformed("an unavailable stage names its reason"))?;
            if !values.is_empty()
                || meta.stage.invocations != 0
                || meta.stage.state_identity.is_some()
                || meta.stage.value_count != 0
            {
                return Err(malformed("an unavailable stage carries no output"));
            }
            CallbackOutcome::Unavailable { reason }
        }
        _ => return Err(Failure::UnsupportedSemantics("stage status")),
    };
    let record = CallbackResultRecord { declaration, request, outcome };
    // Recompute the output digest and the standing; a stored label is never believed.
    if meta.output_digest != output_values_digest(&values_of(&record)) {
        return Err(Failure::IdentityMismatch { field: "output_digest" });
    }
    // No verification receipt travels in the artifact, so a declared verification is
    // never confirmed here; the live path labels the same declaration `diagnostic`.
    if meta.standing != standing_label(&record.declaration, false).as_str() {
        return Err(Failure::IdentityMismatch { field: "standing" });
    }
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(Failure::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Failure::IdentityMismatch { field: "retained_identity" });
    }
    Ok(CallbackResultArtifact { record, declaration_identity, request_identity, identity })
}

fn rebuild_output(query: &CallbackQuery, values: &[f64]) -> Result<CallbackOutput, Failure> {
    let shape = || malformed("values do not have the shape of the query");
    match query {
        CallbackQuery::Mean => match values {
            [value] => Ok(CallbackOutput::Mean { value: *value }),
            _ => Err(shape()),
        },
        CallbackQuery::Sample { .. } => Ok(CallbackOutput::Sample { draws: values.to_vec() }),
        CallbackQuery::Cdf { .. } => Ok(CallbackOutput::Cdf { values: values.to_vec() }),
        CallbackQuery::Quantile { .. } => Ok(CallbackOutput::Quantile { values: values.to_vec() }),
        CallbackQuery::PosteriorUpdate { .. } => match values.split_first() {
            Some((mean, draws)) => {
                Ok(CallbackOutput::PosteriorUpdate { mean: *mean, draws: draws.to_vec() })
            }
            None => Err(shape()),
        },
    }
}

fn encode(
    meta: &CallbackResultMeta,
    values: &[f64],
    artifact_id: &str,
) -> Result<Vec<u8>, Failure> {
    if artifact_id.trim().is_empty() {
        return Err(Failure::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Failure::Encode(error.to_string()))?;
    let value_bytes = crate::artifact_codec::f64s_to_le_bytes(values);
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "callback_result",
            max_bytes: MAX_CALLBACK_RESULT_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[
            crate::OutSection {
                id: META_SECTION,
                content_type: crate::CONTENT_CBOR,
                bytes: &meta_bytes,
            },
            crate::OutSection {
                id: VALUES_SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: &value_bytes,
            },
        ],
    )
    .map_err(|error| crate::map_artifact_check!(error, CallbackResultArtifactError))
}

fn decode(bytes: &[u8]) -> Result<(CallbackResultMeta, Vec<f64>), Failure> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_CALLBACK_RESULT_ARTIFACT_BYTES,
        &[META_SECTION, VALUES_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, CallbackResultArtifactError))?;
    let Ok([meta_bytes, value_bytes]) = <[Vec<u8>; 2]>::try_from(parts) else {
        return Err(malformed("unsupported container layout"));
    };
    if value_bytes.len() % 8 != 0 || value_bytes.len() / 8 > MAX_CALLBACK_RESULT_VALUES {
        return Err(Failure::LimitsExceeded("values"));
    }
    let meta: CallbackResultMeta = crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == CALLBACK_RESULT_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(Failure::UnsupportedVersion { version })
        }
    })?;
    Ok((meta, crate::artifact_codec::f64s_from_le_bytes(&value_bytes)))
}

impl CallbackResultArtifact {
    /// Seal a record. The result passes the checks a consumer applies, so a
    /// record the library refuses does not seal.
    ///
    /// # Errors
    /// A declaration, request or output the protocol refuses, or an encoding
    /// failure.
    pub fn seal(
        record: &CallbackResultRecord,
    ) -> Result<(Self, CallbackResultMeta), CallbackResultArtifactError> {
        let meta = build_meta(record);
        let artifact = check(&meta, &values_of(record), None)?;
        Ok((artifact, meta))
    }

    /// Serialize a sealed record through the checksummed container.
    ///
    /// # Errors
    /// A record that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        record: &CallbackResultRecord,
        artifact_id: &str,
    ) -> Result<Vec<u8>, CallbackResultArtifactError> {
        let (_, meta) = Self::seal(record)?;
        encode(&meta, &values_of(record), artifact_id)
    }

    /// Consume an artifact: rebuild and validate the declaration, request and
    /// output and recompute every identity.
    ///
    /// `expected_identity` is an artifact identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another
    /// result is refused.
    ///
    /// # Errors
    /// Corruption, another major version, an unknown feature, a bound exceeded,
    /// a declaration, request or output the protocol refuses, a native trust
    /// claim, or a stored identity that does not match.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, CallbackResultArtifactError> {
        let (meta, values) = decode(bytes)?;
        check(&meta, &values, expected_identity)
    }

    /// Read the stored metadata without replaying it: the inspection of a
    /// historical result. Nothing here checks identities or declarations.
    ///
    /// # Errors
    /// Corruption, another major version or a malformed container.
    pub fn inspect(bytes: &[u8]) -> Result<CallbackResultMeta, CallbackResultArtifactError> {
        decode(bytes).map(|(meta, _)| meta)
    }

    /// The validated record.
    #[must_use]
    pub const fn record(&self) -> &CallbackResultRecord {
        &self.record
    }

    /// The declaration identity (hex).
    #[must_use]
    pub fn declaration_identity(&self) -> &str {
        &self.declaration_identity
    }

    /// The request identity (hex).
    #[must_use]
    pub fn request_identity(&self) -> &str {
        &self.request_identity
    }

    /// The artifact identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Validate metadata and values as a consumer would, for tests that reseal
    /// edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &CallbackResultMeta,
        values: &[f64],
        expected_identity: Option<&str>,
    ) -> Result<Self, CallbackResultArtifactError> {
        check(meta, values, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut CallbackResultMeta) {
        meta.identity = compute_identity(meta);
    }

    /// Serialize metadata and values as they are, for tests that write an
    /// edited or foreign layout.
    ///
    /// # Errors
    /// An encoding failure.
    #[doc(hidden)]
    pub fn meta_to_bytes(
        meta: &CallbackResultMeta,
        values: &[f64],
        artifact_id: &str,
    ) -> Result<Vec<u8>, CallbackResultArtifactError> {
        encode(meta, values, artifact_id)
    }

    /// The contract operations the result's declaration discloses, for a
    /// consumer that connects it through the interface contract.
    #[must_use]
    pub fn contract_operations(&self) -> BTreeSet<ContractOperation> {
        self.record
            .declaration
            .operations
            .iter()
            .copied()
            .map(CallbackOperation::contract_operation)
            .collect()
    }
}
