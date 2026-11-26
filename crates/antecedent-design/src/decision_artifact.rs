//! Portable decision contract and decision result artifacts.
//!
//! A contract artifact is identified by the contract's canonical digest; a
//! result artifact additionally binds the digest of the draws it was computed
//! from. Both load only under an identity the consumer retained independently
//! of the bytes. A result can be replayed: given the contract and the source,
//! the decision is recomputed and must reproduce the stored result exactly, so
//! a stored number never needs the producing process to be explained.

use antecedent_contract::{
    CompositionLink, CompositionStage, ProvenanceChain, ProvenanceChainError, ScientificQuantity,
};
use antecedent_core::CanonicalWriter;
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::distribution_artifact::DistributionArtifact;
use antecedent_io_core::error::IoError;
use antecedent_io_core::method_provenance::{METHOD_PROVENANCE_FORMAT, MethodProvenanceWire};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use crate::signal::RefusalBuilder;

use crate::decision_contract::{DecisionAction, DecisionContract, HardConstraint};
use crate::decision_eval::{
    ActionOutcome, ConstraintExclusion, DecisionResult, MeanSource, SourceReceipt, Verdict,
    criterion_label_of, evaluate_contract,
};
use crate::decision_wire::{
    ConstraintWire, CriterionWire, ExclusionWire, ExprWire, kind_name, kind_named, structural_name,
    structural_named,
};

/// Maximum artifact bytes accepted before any decode allocation.
pub const MAX_DECISION_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
const CONTRACT_KIND: &str = "decision_contract_v1";
const RESULT_KIND: &str = "decision_result_v1";
const RESULT_METHOD_SECTION: &str = "decision_result.method_provenance";
const BODY_SECTION: &str = "decision.body";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionWire {
    id: String,
    kind: String,
    inputs: Vec<ScientificQuantityWire>,
    utility: ExprWire,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContractBody {
    version: u16,
    #[serde(default)]
    identity: String,
    actions: Vec<ActionWire>,
    utility_units: String,
    criterion: CriterionWire,
    constraints: Vec<ConstraintWire>,
    target_population: String,
    horizon: u32,
    structural_policy: String,
}

pub(crate) fn contract_from_body(body: ContractBody) -> Result<DecisionContract, IoError> {
    if body.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(body.version) });
    }
    let actions = body
        .actions
        .into_iter()
        .map(|a| {
            let inputs = a
                .inputs
                .into_iter()
                .map(|q| ScientificQuantity::try_from(q).map_err(|e| IoError::Convert(e.into())))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(DecisionAction {
                id: a.id,
                kind: kind_named(&a.kind)
                    .ok_or_else(|| IoError::Convert(format!("unknown action kind `{}`", a.kind)))?,
                inputs,
                utility: a.utility.into(),
            })
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    Ok(DecisionContract {
        actions,
        utility_units: body.utility_units,
        criterion: body.criterion.into(),
        constraints: body.constraints.into_iter().map(HardConstraint::from).collect(),
        target_population: body.target_population,
        horizon: body.horizon,
        structural_policy: structural_named(&body.structural_policy).ok_or_else(|| {
            IoError::Convert(format!("unknown structural policy `{}`", body.structural_policy))
        })?,
    })
}

pub(crate) fn body_from_contract(contract: &DecisionContract) -> Result<ContractBody, IoError> {
    let identity = contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
    Ok(ContractBody {
        version: 1,
        identity,
        actions: contract
            .actions
            .iter()
            .map(|a| ActionWire {
                id: a.id.clone(),
                kind: kind_name(a.kind).to_owned(),
                inputs: a.inputs.iter().map(ScientificQuantityWire::from).collect(),
                utility: ExprWire::from(&a.utility),
            })
            .collect(),
        utility_units: contract.utility_units.clone(),
        criterion: contract.criterion.into(),
        constraints: contract.constraints.iter().map(ConstraintWire::from).collect(),
        target_population: contract.target_population.clone(),
        horizon: contract.horizon,
        structural_policy: structural_name(contract.structural_policy).to_owned(),
    })
}

pub(crate) fn encode(
    kind: &str,
    artifact_id: &str,
    note: &str,
    body: &[u8],
) -> Result<Vec<u8>, IoError> {
    encode_section(kind, BODY_SECTION, artifact_id, note, body)
}

pub(crate) fn encode_section(
    kind: &str,
    section: &str,
    artifact_id: &str,
    note: &str,
    body: &[u8],
) -> Result<Vec<u8>, IoError> {
    encode_parts(
        kind,
        artifact_id,
        note,
        &[antecedent_io_core::OutSection {
            id: section,
            content_type: antecedent_io_core::CONTENT_CBOR,
            bytes: body,
        }],
    )
}

fn encode_parts(
    kind: &str,
    artifact_id: &str,
    note: &str,
    sections: &[antecedent_io_core::OutSection<'_>],
) -> Result<Vec<u8>, IoError> {
    antecedent_io_core::encode_container(
        &antecedent_io_core::ArtifactContainer {
            kind,
            note,
            max_bytes: MAX_DECISION_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        sections,
    )
    .map_err(|error| error.into_io("unsupported decision artifact layout"))
}

pub(crate) fn decode<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    kind: &str,
) -> Result<T, IoError> {
    decode_section(bytes, kind, BODY_SECTION)
}

pub(crate) fn decode_section<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    kind: &str,
    section: &str,
) -> Result<T, IoError> {
    let parts =
        antecedent_io_core::decode_container(kind, MAX_DECISION_ARTIFACT_BYTES, &[section], bytes)
            .map_err(|error| error.into_io("unsupported decision artifact layout"))?;
    let Ok([body]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(IoError::Convert("unsupported decision artifact layout".into()));
    };
    decode_body(&body)
}

fn decode_body<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, IoError> {
    // The version word is read from the untyped item first, so a newer body
    // with a changed shape is refused as an unsupported version rather than
    // as a field error of this reader's layout.
    antecedent_io_core::decode_with_version(body, |version: u16| {
        if version == DECISION_BODY_VERSION {
            Ok(())
        } else {
            Err(IoError::UnsupportedVersion { version: u32::from(version) })
        }
    })
}

/// Version word of every decision artifact body this reader decodes.
const DECISION_BODY_VERSION: u16 = 1;

pub(crate) fn refused(slot: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        message: format!("decision_artifact.{slot}: {message}"),
    }
}

/// A decision contract with its canonical identity.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionContractArtifact {
    contract: DecisionContract,
    identity: String,
}

impl DecisionContractArtifact {
    /// Own a validated contract.
    ///
    /// # Errors
    /// An invalid contract refuses.
    pub fn new(contract: DecisionContract) -> Result<Self, IoError> {
        let identity = contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
        Ok(Self { contract, identity })
    }

    /// The contract.
    #[must_use]
    pub fn contract(&self) -> &DecisionContract {
        &self.contract
    }

    /// Canonical identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An oversized payload or invalid contract refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let body = to_cbor(&body_from_contract(&self.contract)?)?;
        encode(CONTRACT_KIND, artifact_id, "decision_contract", &body)
    }

    /// Load under an independently retained identity.
    ///
    /// # Errors
    /// A corrupt, oversized or semantically changed artifact refuses, including
    /// one whose stored identity was resealed to match an edited contract.
    pub fn from_bytes(bytes: &[u8], expected_identity: &str) -> Result<Self, IoError> {
        let body: ContractBody = decode(bytes, CONTRACT_KIND)?;
        let stored = body.identity.clone();
        let contract = contract_from_body(body)?;
        contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
        // A stored identity written before `applies_to` was read as a set still verifies.
        if stored != expected_identity || !contract.accepts_identity(&stored) {
            return Err(refused(
                "identity_expected",
                "contract identity differs from the consumer's retained identity",
            ));
        }
        Ok(Self { contract, identity: stored })
    }

    /// [`Self::from_bytes`], admitting only the artifact the consumer retained: the bytes bind
    /// to `retained` ([`antecedent_io_core::bind_retained_artifact`]) before they are read.
    ///
    /// # Errors
    ///
    /// As [`Self::from_bytes`], and `external_binding_mismatch` when the bytes are not the
    /// retained artifact.
    pub fn from_bytes_expecting(
        bytes: &[u8],
        expected_identity: &str,
        retained: &antecedent_io_core::RetainedArtifactIdentity,
    ) -> Result<Self, IoError> {
        antecedent_io_core::load_expecting(bytes, retained, |bytes| {
            Self::from_bytes(bytes, expected_identity)
        })
    }
}

/// BLAKE3 of a source's aligned draws and coordinates, retained by a result.
#[must_use]
pub fn source_digest(source: &DistributionArtifact) -> String {
    let mut hasher = CanonicalWriter::new("antecedent.decision_source.v1");
    hasher.word(source.quantities().len() as u64);
    for quantity in source.quantities() {
        hasher.bytes(&to_cbor(quantity).unwrap_or_default());
    }
    hasher.word(source.n_draws() as u64).reals(source.draws());
    hasher.opt_word(source.metadata().weights.as_ref().map(|w| w.len() as u64));
    if let Some(weights) = &source.metadata().weights {
        for &weight in weights {
            hasher.real(weight);
        }
    }
    hasher.finish_hex()
}

/// Stable digest of every published decision value and its declared source
/// identity. A composite artifact can retain this fingerprint when its source
/// is a replayable compiler program instead of a distribution artifact.
///
/// # Errors
/// A failure encoding the checked decision body.
pub fn decision_result_fingerprint(
    result: &DecisionResult,
    source_identity: &str,
) -> Result<String, IoError> {
    let body = to_cbor(&result_to_body(result, source_identity))?;
    Ok(CanonicalWriter::new("antecedent.decision_result_fingerprint.v1").bytes(&body).finish_hex())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeWire {
    id: String,
    admissible: bool,
    exclusions: Vec<ExclusionWire>,
    expected_utility: f64,
    value: f64,
    standard_error: Option<f64>,
    expected_regret: Option<f64>,
    max_regret: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum VerdictWire {
    UniquelyOptimal(String),
    Indistinguishable(Vec<String>),
    NoAdmissibleAction,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LinkWire {
    id: String,
    stage: String,
    parents: Vec<String>,
    /// Merkle digest of this link.
    digest: String,
    /// Digests of `parents`, in order.
    parent_digests: Vec<String>,
}

/// Derivation links behind a result computed from a distribution artifact,
/// parents before children. Derived only from the result's source receipt, its
/// contract identity and the source digest, so a stored and a recomputed result
/// over the same inputs yield identical links.
fn lineage_links(result: &DecisionResult, digest: &str) -> Vec<CompositionLink> {
    let contract = format!("contract:{}", result.source.causal_contract_id);
    let provider = format!("provider:{}#{}", result.source.provider_id, result.source.snapshot_id);
    let distribution = format!("distribution:{digest}");
    let decision = format!("decision:{}", result.contract_identity);
    let link = |id: &str, stage: CompositionStage, parents: &[&str]| CompositionLink {
        id: id.to_owned(),
        stage,
        parents: parents.iter().map(|p| (*p).to_owned()).collect(),
        declared_parent_digests: None,
    };
    vec![
        link(&contract, CompositionStage::CausalContract, &[]),
        link(&provider, CompositionStage::ExternalProvider, &[contract.as_str()]),
        link(&distribution, CompositionStage::DistributionArtifact, &[provider.as_str()]),
        link(&decision, CompositionStage::DecisionContract, &[]),
        link(
            DecisionResultArtifact::CLAIM_LINK_ID,
            CompositionStage::Claim,
            &[distribution.as_str(), decision.as_str()],
        ),
    ]
}

/// Wire lineage with each link's Merkle digest and its parents' digests. Empty
/// only when the chain itself is invalid (a blank identity), which
/// [`DecisionResultArtifact::provenance_chain`] reports.
fn lineage_wire(result: &DecisionResult, digest: &str) -> Vec<LinkWire> {
    let Ok(chain) = ProvenanceChain::new(lineage_links(result, digest)) else {
        return Vec::new();
    };
    let digest_of = |id: &str| chain.digest_of(id).unwrap_or_default().to_owned();
    chain
        .links()
        .iter()
        .map(|l| LinkWire {
            id: l.id.clone(),
            stage: l.stage.as_str().to_owned(),
            parents: l.parents.clone(),
            digest: digest_of(&l.id),
            parent_digests: l.parents.iter().map(|p| digest_of(p)).collect(),
        })
        .collect()
}

/// Check a stored lineage: its declared predecessor digests and own digests
/// must hold under recomputation, and it must equal the lineage re-derived from
/// the result's contract, provider and source. An empty stored lineage (a mean
/// grid result) has nothing to check.
fn verify_stored_lineage(stored: &[LinkWire], derived: &[LinkWire]) -> Result<(), IoError> {
    if stored.is_empty() {
        return Ok(());
    }
    let chain_refusal = |error: &ProvenanceChainError| {
        let refusal = error.to_refusal();
        IoError::Refused {
            code: refusal.code,
            message: format!("decision_artifact.lineage: {}", refusal.detail),
        }
    };
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
    let chain = ProvenanceChain::new(links).map_err(|e| chain_refusal(&e))?;
    for link in stored {
        chain.verify_digest(&link.id, &link.digest).map_err(|e| chain_refusal(&e))?;
    }
    if stored != derived {
        return Err(refused(
            "lineage_expected",
            "stored lineage differs from the contract, provider and source it claims",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultBody {
    version: u16,
    contract_identity: String,
    source_digest: String,
    criterion: CriterionWire,
    actions: Vec<OutcomeWire>,
    verdict: VerdictWire,
    evpi: Option<f64>,
    n_draws: u64,
    effective_draws: f64,
    provider_id: String,
    snapshot_id: String,
    rng_id: String,
    causal_contract_id: String,
    assumptions: Vec<String>,
    /// Derived at serialization and ignored on load, which re-derives it.
    #[serde(default)]
    lineage: Vec<LinkWire>,
}

fn result_to_body(result: &DecisionResult, digest: &str) -> ResultBody {
    ResultBody {
        version: 1,
        contract_identity: result.contract_identity.clone(),
        source_digest: digest.to_owned(),
        criterion: result.criterion.into(),
        actions: result
            .actions
            .iter()
            .map(|a| OutcomeWire {
                id: a.id.clone(),
                admissible: a.admissible,
                exclusions: a.exclusions.iter().map(ExclusionWire::from).collect(),
                expected_utility: a.expected_utility,
                value: a.value,
                standard_error: a.standard_error,
                expected_regret: a.expected_regret,
                max_regret: a.max_regret,
            })
            .collect(),
        verdict: match &result.verdict {
            Verdict::UniquelyOptimal(id) => VerdictWire::UniquelyOptimal(id.clone()),
            Verdict::Indistinguishable(ids) => VerdictWire::Indistinguishable(ids.clone()),
            Verdict::NoAdmissibleAction => VerdictWire::NoAdmissibleAction,
        },
        evpi: result.evpi,
        n_draws: result.n_draws as u64,
        effective_draws: result.effective_draws,
        provider_id: result.source.provider_id.clone(),
        snapshot_id: result.source.snapshot_id.clone(),
        rng_id: result.source.rng_id.clone(),
        causal_contract_id: result.source.causal_contract_id.clone(),
        assumptions: result.assumptions.clone(),
        lineage: lineage_wire(result, digest),
    }
}

fn body_to_result(body: ResultBody) -> Result<DecisionResult, IoError> {
    if body.version != 1 {
        return Err(IoError::UnsupportedVersion { version: u32::from(body.version) });
    }
    let criterion = body.criterion.into();
    let criterion_label = criterion_label_of(criterion, &body.assumptions);
    Ok(DecisionResult {
        method_provenance: None,
        contract_identity: body.contract_identity,
        criterion,
        criterion_label,
        actions: body
            .actions
            .into_iter()
            .map(|a| ActionOutcome {
                id: a.id,
                admissible: a.admissible,
                exclusions: a.exclusions.into_iter().map(ConstraintExclusion::from).collect(),
                expected_utility: a.expected_utility,
                value: a.value,
                standard_error: a.standard_error,
                expected_regret: a.expected_regret,
                max_regret: a.max_regret,
            })
            .collect(),
        verdict: match body.verdict {
            VerdictWire::UniquelyOptimal(id) => Verdict::UniquelyOptimal(id),
            VerdictWire::Indistinguishable(ids) => Verdict::Indistinguishable(ids),
            VerdictWire::NoAdmissibleAction => Verdict::NoAdmissibleAction,
        },
        evpi: body.evpi,
        n_draws: usize::try_from(body.n_draws).map_err(|_| IoError::TooLarge)?,
        effective_draws: body.effective_draws,
        source: SourceReceipt {
            provider_id: body.provider_id,
            snapshot_id: body.snapshot_id,
            rng_id: body.rng_id,
            causal_contract_id: body.causal_contract_id,
        },
        assumptions: body.assumptions,
    })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultMethodProvenance {
    result_fingerprint: String,
    provenance: MethodProvenanceWire,
}

fn result_method_binding(body: &[u8]) -> String {
    CanonicalWriter::new("antecedent.decision_method_snapshot.v1").bytes(body).finish_hex()
}

pub(crate) fn encode_with_method_provenance(
    kind: &str,
    artifact_id: &str,
    note: &str,
    body: &[u8],
    provenance: Option<&MethodProvenanceWire>,
) -> Result<Vec<u8>, IoError> {
    let Some(provenance) = provenance else {
        return encode(kind, artifact_id, note, body);
    };
    let metadata = to_cbor(&ResultMethodProvenance {
        result_fingerprint: result_method_binding(body),
        provenance: provenance.clone(),
    })?;
    encode_parts(
        kind,
        artifact_id,
        note,
        &[
            antecedent_io_core::OutSection {
                id: BODY_SECTION,
                content_type: antecedent_io_core::CONTENT_CBOR,
                bytes: body,
            },
            antecedent_io_core::OutSection {
                id: RESULT_METHOD_SECTION,
                content_type: antecedent_io_core::CONTENT_CBOR,
                bytes: &metadata,
            },
        ],
    )
}

pub(crate) fn decode_with_method_provenance<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    kind: &str,
) -> Result<(T, Option<MethodProvenanceWire>), IoError> {
    match antecedent_io_core::decode_container(
        kind,
        MAX_DECISION_ARTIFACT_BYTES,
        &[BODY_SECTION, RESULT_METHOD_SECTION],
        bytes,
    ) {
        Ok(parts) => {
            let [body, methods] = <[Vec<u8>; 2]>::try_from(parts)
                .map_err(|_| IoError::Convert("unsupported decision artifact layout".into()))?;
            let provenance = antecedent_io_core::from_cbor::<ResultMethodProvenance>(&methods)
                .ok()
                .filter(|methods| {
                    methods.provenance.format == METHOD_PROVENANCE_FORMAT
                        && methods.result_fingerprint == result_method_binding(&body)
                })
                .map(|methods| methods.provenance);
            Ok((decode_body(&body)?, provenance))
        }
        Err(antecedent_io_core::CodecError::Layout) => {
            decode(bytes, kind).map(|body| (body, None))
        }
        Err(error) => Err(error.into_io("unsupported decision artifact layout")),
    }
}

/// A decision result bound to its contract and source digest.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResultArtifact {
    result: DecisionResult,
    source_digest: String,
}

impl DecisionResultArtifact {
    /// Bind a result to the source it was computed from.
    #[must_use]
    pub fn new(result: DecisionResult, source: &DistributionArtifact) -> Self {
        Self { result, source_digest: source_digest(source) }
    }

    /// The stored result.
    #[must_use]
    pub fn result(&self) -> &DecisionResult {
        &self.result
    }

    /// Identity of the reported decision within [`Self::provenance_chain`].
    pub const CLAIM_LINK_ID: &'static str = "decision_result";

    /// Derivation chain behind the decision: the causal contract, the provider
    /// execution and the distribution artifact it supplied, the decision
    /// contract, and the reported result derived from the last two.
    ///
    /// # Errors
    /// A blank identity in the source receipt or contract refuses.
    pub fn provenance_chain(&self) -> Result<ProvenanceChain, ProvenanceChainError> {
        ProvenanceChain::new(lineage_links(&self.result, &self.source_digest))
    }

    /// Digest of the source draws the result was computed from.
    #[must_use]
    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An oversized payload refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        encode_with_method_provenance(
            RESULT_KIND,
            artifact_id,
            "decision_result",
            &to_cbor(&result_to_body(&self.result, &self.source_digest))?,
            self.result.method_provenance(),
        )
    }

    /// Load under the consumer's retained contract identity and source digest.
    ///
    /// # Errors
    /// A corrupt or changed artifact, or one bound to a different contract or
    /// source, refuses.
    pub fn from_bytes(
        bytes: &[u8],
        expected_contract_identity: &str,
        expected_source_digest: &str,
    ) -> Result<Self, IoError> {
        let (mut body, methods): (ResultBody, _) = decode_with_method_provenance(bytes, RESULT_KIND)?;
        if body.contract_identity != expected_contract_identity {
            return Err(refused(
                "contract_expected",
                "result was computed under a different decision contract",
            ));
        }
        if body.source_digest != expected_source_digest {
            return Err(refused("source_expected", "result was computed from different draws"));
        }
        let digest = body.source_digest.clone();
        let stored_lineage = std::mem::take(&mut body.lineage);
        let mut result = body_to_result(body)?;
        verify_stored_lineage(&stored_lineage, &lineage_wire(&result, &digest))?;
        if let Some(methods) = methods {
            result.set_method_provenance(methods);
        }
        Ok(Self { result, source_digest: digest })
    }

    /// [`Self::from_bytes`], admitting only the artifact the consumer retained: the bytes bind
    /// to `retained` ([`antecedent_io_core::bind_retained_artifact`]) before they are read.
    ///
    /// # Errors
    ///
    /// As [`Self::from_bytes`], and `external_binding_mismatch` when the bytes are not the
    /// retained artifact.
    pub fn from_bytes_expecting(
        bytes: &[u8],
        expected_contract_identity: &str,
        expected_source_digest: &str,
        retained: &antecedent_io_core::RetainedArtifactIdentity,
    ) -> Result<Self, IoError> {
        antecedent_io_core::load_expecting(bytes, retained, |bytes| {
            Self::from_bytes(bytes, expected_contract_identity, expected_source_digest)
        })
    }

    /// Recompute the decision from `contract` and `source` and require the
    /// stored result to match exactly.
    ///
    /// # Errors
    /// A different contract or source, a refused evaluation or any differing
    /// number refuses.
    pub fn replay(
        &self,
        contract: &DecisionContract,
        source: &DistributionArtifact,
    ) -> Result<(), IoError> {
        contract.identity().map_err(|e| IoError::Convert(format!("{e:?}")))?;
        if !contract.accepts_identity(&self.result.contract_identity) {
            return Err(refused("replay_contract", "contract differs from the stored result's"));
        }
        if source_digest(source) != self.source_digest {
            return Err(refused("replay_source", "source differs from the stored result's"));
        }
        let mut recomputed = evaluate_contract(contract, source)
            .map_err(|e| refused("replay_evaluation", &format!("{e:?}")))?;
        // The contract was accepted under the stored spelling of its identity.
        recomputed.contract_identity.clone_from(&self.result.contract_identity);
        let stored = to_cbor(&result_to_body(&self.result, &self.source_digest))?;
        let fresh = to_cbor(&result_to_body(&recomputed, &self.source_digest))?;
        if stored != fresh {
            return Err(refused(
                "replay_mismatch",
                "recomputation does not reproduce the stored result",
            ));
        }
        Ok(())
    }
}

/// Parse a contract from its JSON declaration (the wire a host language builds).
/// A stored `identity` field is ignored: the identity is always recomputed.
///
/// # Errors
/// Malformed JSON, unknown fields or an invalid declaration refuse.
pub fn contract_from_json(json: &str) -> Result<DecisionContract, IoError> {
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let body: ContractBody =
        serde_json::from_str(json).map_err(|e| IoError::Convert(e.to_string()))?;
    let contract = contract_from_body(body)?;
    contract.validate().map_err(|e| IoError::Convert(format!("{e:?}")))?;
    Ok(contract)
}

/// Parse and validate a contract declaration, refusing with the typed structured
/// refusal a Rust caller would get: a malformed declaration is
/// `decision_contract.invalid_declaration`, and a declaration that parses but is
/// invalid keeps the exact code and detail of its [`DecisionContractError`].
///
/// # Errors
/// A malformed or invalid declaration, as a structured refusal.
// The refusal is the cold path of a once-per-declaration check; boxing would not pay.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn contract_from_json_refusal(
    json: &str,
) -> Result<DecisionContract, antecedent_contract::ExternalRefusal> {
    let malformed = |why: String| {
        RefusalBuilder::new(
            antecedent_core::reason_code!("invalid_argument"),
            "declare",
            antecedent_core::reason_details::decision_contract::INVALID_DECLARATION,
        )
        .offending(why)
        .remedy("declare the contract with the documented fields")
        .build()
    };
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(malformed("declaration is too large".to_owned()));
    }
    let body: ContractBody = serde_json::from_str(json).map_err(|e| malformed(e.to_string()))?;
    let contract = contract_from_body(body).map_err(|e| malformed(e.to_string()))?;
    contract.validate().map_err(|e| e.to_refusal())?;
    Ok(contract)
}

/// The JSON declaration of a contract, including its recomputed identity.
///
/// # Errors
/// An invalid contract refuses.
pub fn contract_to_json(contract: &DecisionContract) -> Result<String, IoError> {
    serde_json::to_string(&body_from_contract(contract)?)
        .map_err(|e| IoError::Convert(e.to_string()))
}

/// The JSON form of a result bound to its source digest. Values that are
/// undefined for an excluded action are `null`.
///
/// # Errors
/// Serialization failure.
pub fn result_to_json(
    result: &DecisionResult,
    source: &DistributionArtifact,
) -> Result<String, IoError> {
    result_body_json(result, &result_to_body(result, &source_digest(source)))
}

/// BLAKE3 of a mean source's coordinates and means.
#[must_use]
pub(crate) fn mean_source_digest(source: &MeanSource) -> String {
    let mut hasher = CanonicalWriter::new("antecedent.decision_mean_source.v1");
    hasher.word(source.coordinates.len() as u64);
    for quantity in &source.coordinates {
        quantity.write_canonical(&mut hasher);
    }
    hasher.reals(&source.means);
    hasher.finish_hex()
}

/// The JSON form of a result computed from a mean source, bound to the mean
/// source's digest. Its `lineage` is empty: the distribution-artifact lineage
/// does not describe a mean grid, so the caller supplies the true derivation.
///
/// # Errors
/// Serialization failure.
pub fn mean_result_to_json(
    result: &DecisionResult,
    source: &MeanSource,
) -> Result<String, IoError> {
    let mut body = result_to_body(result, &mean_source_digest(source));
    body.lineage = Vec::new();
    result_body_json(result, &body)
}

fn result_body_json(result: &DecisionResult, body: &ResultBody) -> Result<String, IoError> {
    #[derive(Serialize)]
    struct Reading<'a> {
        #[serde(flatten)]
        body: &'a ResultBody,
        #[serde(skip_serializing_if = "Option::is_none")]
        method_provenance: Option<&'a MethodProvenanceWire>,
    }
    serde_json::to_string(&Reading { body, method_provenance: result.method_provenance() })
        .map_err(|error| IoError::Convert(error.to_string()))
}

/// The checksummed container of a result computed from a mean source: the same
/// `decision_result_v1` layout as [`DecisionResultArtifact::to_bytes`], bound to
/// the mean source's digest with an empty lineage. It loads through
/// [`DecisionResultArtifact::from_bytes`] under the mean source's digest, so a
/// point-only external decision can be carried in a composition bundle; it never
/// stands for aligned joint draws.
///
/// # Errors
/// A blank artifact id, an oversized payload or an encoding failure.
pub fn mean_result_to_bytes(
    result: &DecisionResult,
    source: &MeanSource,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    bound_mean_result_to_bytes(result, &mean_source_digest(source), artifact_id)
}

/// Encode a mean decision bound to a digest supplied by its verified producer.
///
/// # Errors
/// A blank artifact id, an oversized payload or an encoding failure.
#[doc(hidden)]
pub fn bound_mean_result_to_bytes(
    result: &DecisionResult,
    source_binding: &str,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    let mut body = result_to_body(result, source_binding);
    body.lineage = Vec::new();
    encode_with_method_provenance(
        RESULT_KIND,
        artifact_id,
        "decision_result",
        &to_cbor(&body)?,
        result.method_provenance(),
    )
}

#[cfg(test)]
mod retained_tests {
    use super::{DecisionContractArtifact, DecisionResultArtifact};
    use crate::retained_test_support::{other_artifact, refused_as_not_retained, same_outcome};
    use antecedent_io_core::RetainedArtifactIdentity;

    #[test]
    fn decision_artifact_expecting_binds_the_retained_bytes() {
        let bytes = b"not an artifact".to_vec();
        let retained = RetainedArtifactIdentity::of_bytes(&bytes);
        same_outcome(
            &DecisionContractArtifact::from_bytes(&bytes, "contract"),
            &DecisionContractArtifact::from_bytes_expecting(&bytes, "contract", &retained),
        );
        refused_as_not_retained(&DecisionContractArtifact::from_bytes_expecting(
            &bytes,
            "contract",
            &other_artifact(),
        ));
        same_outcome(
            &DecisionResultArtifact::from_bytes(&bytes, "contract", "source"),
            &DecisionResultArtifact::from_bytes_expecting(&bytes, "contract", "source", &retained),
        );
        refused_as_not_retained(&DecisionResultArtifact::from_bytes_expecting(
            &bytes,
            "contract",
            "source",
            &other_artifact(),
        ));
    }
}
