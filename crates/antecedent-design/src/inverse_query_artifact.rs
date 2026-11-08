//! `InverseQueryArtifact` (`inverse_functional_query_v1`).
//!
//! A durable record of one generalized finite inverse decision query and its answer:
//! the declared query (action grid and order, grid scope, constraints, selection
//! rule, tolerance, evaluation budget), the decision contract that declares the
//! actions, the forward evidence behind each feasibility field (finite laws and
//! means embedded compactly), the per-action status table with the *distinct*
//! point, interval-region, identified-set, all-scenario and posterior-probability
//! fields, the selected action, whether the selection is certified and whether
//! the grid was exhaustively decided.
//!
//! Layout of the bounded sectioned container:
//!
//! * `inverse_query.meta` (CBOR): version, feature, claim, contract identity, query,
//!   claim declarations (law metadata, mean coordinates), the evidence structure that
//!   points at those claims, the identity digests and the stored result table;
//! * `inverse_query.contract` (nested `decision_contract_v1` container);
//! * `inverse_query.data` (little-endian `f64`): the draws of every embedded law and
//!   the values of every embedded mean claim, in claim order.
//!
//! Identity keeps the *premises* (contract identity, query, claim declarations and
//! evidence structure) apart from the *data* (the numbers), and binds both in one
//! digest. A consumer trusts none of the stored results: it rebuilds the contract,
//! query and evidence, re-evaluates them through [`evaluate_inverse_query`], and
//! refuses unless the recomputed identity and the recomputed result table equal the
//! stored ones and, when the caller retained an identity of its own, that one too. A
//! resealed change of a target, a tolerance, a draw, a status or the selection is
//! therefore refused.
//!
//! A [`GridScope::ContinuousSample`] artifact never claims global feasibility: a
//! stored `exhaustive_over_declared_set` or `no_feasible_action_in_declared_set` on a
//! continuous sample is refused even before replay.
//!
//! A claim produced by a composed-mechanism model, a cohort-mean contrast or a transported
//! response may carry its [`ForwardOrigin`] (source kind, source identity, representation,
//! transport receipt, claim digest). The origin is bound into the premises digest, and a
//! consumer re-derives the digest of the embedded point claim and refuses an origin that
//! names another claim. Replaying the artifact checks that the embedded numbers are the
//! numbers the origin names; it does not re-execute the route that produced them.
//!
//! The artifact makes no coverage or calibration claim (`point_only`). The mean of a
//! posterior is never turned into a probability or quantile here: the shared engine
//! refuses that when the query is evaluated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod wire;

pub use wire::{
    ActionReportWire, AtomStateWire, AtomWire, ClaimWire, ComparisonWire, ConstraintValueWire,
    ConstraintWire, EvidenceWire, ExistenceWire, FeasibilityWire, GridScopeWire, IntervalWire,
    InverseQueryWire, InverseResultWire, LawWire, MeansWire, MemberStatusWire,
    PosteriorFeasibilityWire, SelectionOutcomeWire, SelectionWire, SetWire, SourceReceiptWire,
    TailWire,
};

use antecedent_contract::ScientificQuantity;

use crate::signal::RefusalBuilder;
use antecedent_core::CanonicalWriter;
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::distribution_artifact::{DistributionArtifact, DistributionMetadata};
use antecedent_io_core::error::IoError;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use crate::decision_artifact::DecisionContractArtifact;
use crate::decision_contract::{DecisionContract, Tail};
use crate::decision_eval::{MeanSource, SourceReceipt};
use crate::decision_structural::{AtomEvidence, StructuralAtom};
use crate::inverse_forward_source::{ForwardOrigin, forward_claim_digest};
use crate::inverse_query::{
    ActionReport, Comparison, ConstraintValue, ExistenceClaim, FeasibilityStatus, ForwardClaim,
    ForwardEvidence, GridScope, IdentifiedSet, IntervalRegion, InverseConstraint, InverseQuery,
    InverseQueryError, InverseResult, MemberStatus, PosteriorFeasibility, SelectionOutcome,
    SelectionRule, evaluate_inverse_query,
};

/// The artifact major version this reader writes and accepts.
pub const INVERSE_QUERY_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const INVERSE_QUERY_ARTIFACT_FEATURE: &str = "inverse_functional_query_v1";
/// The only claim an inverse-query artifact carries.
pub const INVERSE_QUERY_INFERENCE_CLAIM: &str = "point_only";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_INVERSE_QUERY_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Most embedded forward claims (laws and mean grids) of one artifact.
pub const MAX_INVERSE_QUERY_CLAIMS: usize = 256;
/// Most original source artifacts one inverse-query artifact may carry. Each is
/// replayed in full on consumption, so the count bounds the replay work inside
/// the byte cap.
pub const MAX_INVERSE_QUERY_SOURCE_EVIDENCE: usize = 8;

/// Counts a consumer accepts of a stored inverse-query artifact, refused before any claim or
/// source artifact is replayed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InverseQueryReadLimits {
    /// Most embedded forward claims.
    pub max_claims: usize,
    /// Most original source artifacts, each replayed in full.
    pub max_source_evidence: usize,
}

impl Default for InverseQueryReadLimits {
    fn default() -> Self {
        Self {
            max_claims: MAX_INVERSE_QUERY_CLAIMS,
            max_source_evidence: MAX_INVERSE_QUERY_SOURCE_EVIDENCE,
        }
    }
}

const ARTIFACT_KIND: &str = "inverse_functional_query_v1";
const META_SECTION: &str = "inverse_query.meta";
const CONTRACT_SECTION: &str = "inverse_query.contract";
const DATA_SECTION: &str = "inverse_query.data";
const CONTRACT_ARTIFACT_ID: &str = "inverse-query-contract";
const WRONG_CONTRACT: &str =
    antecedent_core::reason_details::functional_inverse_query::WRONG_CONTRACT;
const GLOBAL_CLAIM: &str =
    antecedent_core::reason_details::functional_inverse_query::GLOBAL_FEASIBILITY_CLAIM;
const BOUNDS: &str = antecedent_core::reason_details::functional_inverse_query::BOUNDS_EXCEEDED;
const ORIGIN_INVALID: &str =
    antecedent_core::reason_details::functional_inverse_query::FORWARD_ORIGIN_INVALID;
const KIND_MISMATCH: &str =
    antecedent_core::reason_details::functional_inverse_query::ARTIFACT_KIND_MISMATCH;

/// The kind of bytes that are a recognised artifact: a container's own kind, or the
/// bare-CBOR joint-law propagation and mechanism-composition artifacts, decoded
/// by their own readers. `None` for anything else.
#[must_use]
pub fn recognised_artifact_kind(bytes: &[u8]) -> Option<String> {
    if let Some(kind) = antecedent_io_core::container_kind(bytes) {
        return Some(kind);
    }
    if antecedent_io_core::joint_law_propagation_artifact::JointLawPropagationArtifactWire::decode(
        bytes,
    )
    .is_ok()
    {
        return Some("joint_law_propagation_v1".to_owned());
    }
    if antecedent_io_core::mechanism_composition_artifact::MechanismCompositionArtifactWire::decode(
        bytes,
    )
    .is_ok()
    {
        return Some("mechanism_composition_v1".to_owned());
    }
    None
}

/// What to do with a recognised artifact kind that a consumer does not read: the
/// consumer that does accept it, as a remedy sentence.
///
/// Shared by the consumers that refuse another kind's bytes with an
/// `artifact_kind_mismatch` detail, so each names the same destination.
#[must_use]
pub fn artifact_kind_remedy(kind: &str) -> &'static str {
    match kind {
        "joint_law_propagation_v1" => {
            "a propagation claim is consumed by the joint-law propagation consumer \
             (consume_joint_law_propagation_artifact)"
        }
        "decision_reliability_v1" => {
            "a decision-reliability artifact is consumed by the decision-reliability consumer \
             (consume_decision_reliability_artifact)"
        }
        "inverse_functional_query_v1" => {
            "an inverse-query artifact is consumed by the inverse-query consumer \
             (inverse_query::consume)"
        }
        "mechanism_composition_v1" => {
            "a mechanism-composition artifact is consumed by the mechanism-composition consumer \
             (consume_mechanism_composition_artifact)"
        }
        _ => "offer the artifact to the consumer of its own kind; this consumer reads only its own",
    }
}

fn refused(detail: &str, text: &str) -> IoError {
    crate::plan_common::refused_with(
        antecedent_core::reason_code!("inverse_functional_unsupported"),
        detail,
        text,
    )
}

fn wrong_contract(text: &str) -> IoError {
    refused(WRONG_CONTRACT, text)
}

/// Structured refusal of an inverse-functional artifact failure: the registered code,
/// the `functional_inverse_query.*` detail written in front of the message, and the
/// remainder of the message as the offending item. Any other I/O failure is
/// `functional_inverse_query.invalid_artifact` under `invalid_argument`.
#[must_use]
pub fn artifact_refusal(error: &IoError) -> antecedent_contract::ExternalRefusal {
    let (code, detail, offending) = match error {
        IoError::Refused { code, message } => {
            let (detail, text) =
                message.split_once(": ").unwrap_or((WRONG_CONTRACT, message.as_str()));
            if detail == KIND_MISMATCH {
                let supplied = text.rsplit_once("supplied ").map_or("", |(_, kind)| kind);
                return RefusalBuilder::new(code, "inverse_query", detail)
                    .offending(text)
                    .expected(ARTIFACT_KIND)
                    .supplied(supplied)
                    .remedy(artifact_kind_remedy(supplied))
                    .build();
            }
            (*code, detail.to_owned(), text.to_owned())
        }
        other => (
            antecedent_core::reason_code!("invalid_argument"),
            antecedent_core::reason_details::functional_inverse_query::INVALID_ARTIFACT.to_owned(),
            other.to_string(),
        ),
    };
    RefusalBuilder::new(code, "inverse_query", &detail).offending(offending).build()
}

fn wide(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

#[derive(Default)]
struct Encoded {
    claims: Vec<ClaimWire>,
    data: Vec<u8>,
    evidence: EvidenceWire,
}

impl Encoded {
    fn push_f64s(&mut self, values: &[f64]) {
        for value in values {
            self.data.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn law(&mut self, artifact: &DistributionArtifact) -> u64 {
        self.push_f64s(artifact.draws());
        self.claims
            .push(ClaimWire::Law(Box::new(LawWire { metadata: artifact.metadata().clone() })));
        wide(self.claims.len() - 1)
    }

    fn claim(&mut self, claim: &ForwardClaim) -> u64 {
        match claim {
            ForwardClaim::Law(artifact) => self.law(artifact),
            ForwardClaim::Gaussian(law) => {
                self.claims.push(ClaimWire::Gaussian(law.clone()));
                wide(self.claims.len() - 1)
            }
            ForwardClaim::Means(means) => {
                self.push_f64s(&means.means);
                self.claims.push(ClaimWire::Means(MeansWire {
                    coordinates: means
                        .coordinates
                        .iter()
                        .map(ScientificQuantityWire::from)
                        .collect(),
                    provider_id: means.provider_id.clone(),
                    snapshot_id: means.snapshot_id.clone(),
                    causal_contract_id: means.causal_contract_id.clone(),
                    rng_id: means.rng_id.clone(),
                    estimate: means.estimate,
                }));
                wide(self.claims.len() - 1)
            }
        }
    }

    fn atom(&mut self, atom: &StructuralAtom) -> AtomWire {
        let state = match &atom.evidence {
            AtomEvidence::Evaluated(artifact) => {
                AtomStateWire::Evaluated { claim: self.law(artifact) }
            }
            AtomEvidence::Unidentified => AtomStateWire::Unidentified,
            AtomEvidence::Unevaluated(reason) => {
                AtomStateWire::Unevaluated { reason: reason.clone() }
            }
        };
        AtomWire { id: atom.id.clone(), probability: atom.probability, state }
    }

    fn atoms(&mut self, atoms: &[StructuralAtom]) -> Vec<AtomWire> {
        atoms.iter().map(|atom| self.atom(atom)).collect()
    }
}

fn encode_evidence(evidence: &ForwardEvidence) -> Encoded {
    let mut encoded = Encoded::default();
    let point = evidence.point.as_ref().map(|claim| encoded.claim(claim));
    let interval_region = evidence.interval_region.as_ref().map(|region| {
        let lower = encoded.claim(&region.lower);
        let upper = encoded.claim(&region.upper);
        IntervalWire { lower, upper, endpoints_bound_functional: region.endpoints_bound_functional }
    });
    let identified_set = evidence
        .identified_set
        .as_ref()
        .map(|set| SetWire { members: encoded.atoms(&set.members), exhaustive: set.exhaustive });
    let scenarios = evidence.scenarios.as_ref().map(|atoms| encoded.atoms(atoms));
    encoded.evidence = EvidenceWire { point, interval_region, identified_set, scenarios };
    encoded
}

struct DataReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl DataReader<'_> {
    fn take(&mut self, count: usize) -> Result<Vec<f64>, IoError> {
        let end = count
            .checked_mul(8)
            .and_then(|size| self.at.checked_add(size))
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| wrong_contract("data section is shorter than the declared claims"))?;
        let values = self.bytes[self.at..end]
            .chunks_exact(8)
            .map(|chunk| {
                let mut word = [0_u8; 8];
                word.copy_from_slice(chunk);
                f64::from_le_bytes(word)
            })
            .collect();
        self.at = end;
        Ok(values)
    }

    fn exhausted(&self) -> bool {
        self.at == self.bytes.len()
    }
}

fn decode_claims(claims: &[ClaimWire], data: &[u8]) -> Result<Vec<ForwardClaim>, IoError> {
    let mut reader = DataReader { bytes: data, at: 0 };
    let mut decoded = Vec::with_capacity(claims.len());
    for claim in claims {
        decoded.push(match claim {
            ClaimWire::Law(law) => {
                let [rows, columns] = law.metadata.shape;
                let count = rows.checked_mul(columns).ok_or(IoError::TooLarge)?;
                let draws = reader.take(count)?;
                ForwardClaim::Law(Box::new(DistributionArtifact::new(law.metadata.clone(), draws)?))
            }
            ClaimWire::Gaussian(law) => {
                if !law.validate() {
                    return Err(wrong_contract("invalid analytic Gaussian law"));
                }
                ForwardClaim::Gaussian(law.clone())
            }
            ClaimWire::Means(means) => {
                let values = reader.take(means.coordinates.len())?;
                let coordinates = means
                    .coordinates
                    .iter()
                    .map(|wire| {
                        ScientificQuantity::try_from(wire.clone())
                            .map_err(|error| IoError::Convert(error.into()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ForwardClaim::Means(MeanSource {
                    coordinates,
                    means: values,
                    provider_id: means.provider_id.clone(),
                    snapshot_id: means.snapshot_id.clone(),
                    causal_contract_id: means.causal_contract_id.clone(),
                    rng_id: means.rng_id.clone(),
                    estimate: means.estimate,
                })
            }
        });
    }
    if !reader.exhausted() {
        return Err(wrong_contract("data section carries numbers no claim declares"));
    }
    Ok(decoded)
}

fn claim_at(claims: &[ForwardClaim], index: u64) -> Result<ForwardClaim, IoError> {
    usize::try_from(index)
        .ok()
        .and_then(|k| claims.get(k))
        .cloned()
        .ok_or_else(|| wrong_contract("evidence points at a claim that is not embedded"))
}

fn atom_from(claims: &[ForwardClaim], wire: &AtomWire) -> Result<StructuralAtom, IoError> {
    let evidence = match &wire.state {
        AtomStateWire::Evaluated { claim } => match claim_at(claims, *claim)? {
            ForwardClaim::Law(artifact) => AtomEvidence::Evaluated(artifact),
            ForwardClaim::Gaussian(_) => {
                return Err(wrong_contract(
                    "an analytic univariate Gaussian is not an explicitly joint scenario law",
                ));
            }
            ForwardClaim::Means(_) => {
                return Err(wrong_contract("a scenario or set member needs a law, not means"));
            }
        },
        AtomStateWire::Unidentified => AtomEvidence::Unidentified,
        AtomStateWire::Unevaluated { reason } => AtomEvidence::Unevaluated(reason.clone()),
    };
    Ok(StructuralAtom { id: wire.id.clone(), probability: wire.probability, evidence })
}

fn atoms_from(claims: &[ForwardClaim], wires: &[AtomWire]) -> Result<Vec<StructuralAtom>, IoError> {
    wires.iter().map(|wire| atom_from(claims, wire)).collect()
}

fn evidence_from(claims: &[ForwardClaim], wire: &EvidenceWire) -> Result<ForwardEvidence, IoError> {
    Ok(ForwardEvidence {
        point: wire.point.map(|index| claim_at(claims, index)).transpose()?,
        interval_region: wire
            .interval_region
            .as_ref()
            .map(|region| {
                Ok::<_, IoError>(IntervalRegion {
                    lower: claim_at(claims, region.lower)?,
                    upper: claim_at(claims, region.upper)?,
                    endpoints_bound_functional: region.endpoints_bound_functional,
                })
            })
            .transpose()?,
        identified_set: wire
            .identified_set
            .as_ref()
            .map(|set| {
                Ok::<_, IoError>(IdentifiedSet {
                    members: atoms_from(claims, &set.members)?,
                    exhaustive: set.exhaustive,
                })
            })
            .transpose()?,
        scenarios: wire.scenarios.as_ref().map(|atoms| atoms_from(claims, atoms)).transpose()?,
    })
}

// ---------------------------------------------------------------------------
// Identity and container
// ---------------------------------------------------------------------------

/// Identity digests; premises and numbers are kept apart.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InverseQueryIdentity {
    /// BLAKE3 of the canonical declarations (contract identity, query, claim
    /// declarations, evidence structure).
    pub premises_digest: String,
    /// BLAKE3 of the numerical section.
    pub data_digest: String,
    /// BLAKE3 of both digests.
    pub digest: String,
}

#[derive(Serialize)]
struct Premises<'a> {
    version: u16,
    feature: &'a str,
    inference_claim: &'a str,
    contract_identity: &'a str,
    query: &'a InverseQueryWire,
    claims: &'a [ClaimWire],
    evidence: &'a EvidenceWire,
}

fn tagged_digest(tag: &str, bytes: &[u8]) -> String {
    CanonicalWriter::new(tag).bytes(bytes).finish_hex()
}

fn whole_identity(premises_digest: &str, data_digest: &str) -> String {
    let mut whole = CanonicalWriter::new("antecedent.inverse_query.identity.v1");
    whole.text(premises_digest).text(data_digest);
    whole.finish_hex()
}

fn identity_of(
    contract_identity: &str,
    query: &InverseQueryWire,
    encoded: &Encoded,
) -> Result<InverseQueryIdentity, IoError> {
    let premises = to_cbor(&Premises {
        version: INVERSE_QUERY_ARTIFACT_VERSION,
        feature: INVERSE_QUERY_ARTIFACT_FEATURE,
        inference_claim: INVERSE_QUERY_INFERENCE_CLAIM,
        contract_identity,
        query,
        claims: &encoded.claims,
        evidence: &encoded.evidence,
    })?;
    let premises_digest = tagged_digest("antecedent.inverse_query.premises.v1", &premises);
    let data_digest = tagged_digest("antecedent.inverse_query.data.v1", &encoded.data);
    let digest = whole_identity(&premises_digest, &data_digest);
    Ok(InverseQueryIdentity { premises_digest, data_digest, digest })
}

fn identity_diff(
    left: &InverseQueryIdentity,
    right: &InverseQueryIdentity,
) -> Option<&'static str> {
    if left.premises_digest != right.premises_digest {
        Some("premises digest")
    } else if left.data_digest != right.data_digest {
        Some("data digest")
    } else if left.digest != right.digest {
        Some("identity digest")
    } else {
        None
    }
}

/// Stored metadata; the numbers are a separate `f64` section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InverseQueryMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Always [`INVERSE_QUERY_INFERENCE_CLAIM`].
    pub inference_claim: String,
    /// Identity of the decision contract (the contract itself is its own section).
    pub contract_identity: String,
    /// The declared query.
    pub query: InverseQueryWire,
    /// Embedded claim declarations in data order.
    pub claims: Vec<ClaimWire>,
    /// Evidence structure over those claims.
    pub evidence: EvidenceWire,
    /// Identity digests.
    pub identity: InverseQueryIdentity,
    /// The stored per-action status table and selection.
    pub result: InverseResultWire,
    /// Independently consumed original source evidence; no executable-state authority.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_evidence: Vec<Vec<u8>>,
    /// Where the point claim came from, when it came from a supported forward source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_origin: Option<ForwardOrigin>,
}

/// Encode the metadata, the nested contract artifact and the numbers as a
/// checksummed container.
///
/// Hidden: the producer path is [`InverseQueryArtifact::to_bytes`]; tests use this to
/// build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &InverseQueryMeta,
    contract: &[u8],
    data: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    if artifact_id.trim().is_empty() {
        return Err(IoError::Convert("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta)?;
    antecedent_io_core::encode_container(
        &antecedent_io_core::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "inverse_functional_query_point_only",
            max_bytes: MAX_INVERSE_QUERY_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[
            antecedent_io_core::OutSection {
                id: META_SECTION,
                content_type: antecedent_io_core::CONTENT_CBOR,
                bytes: &meta_bytes,
            },
            antecedent_io_core::OutSection {
                id: CONTRACT_SECTION,
                content_type: antecedent_io_core::CONTENT_BYTES,
                bytes: contract,
            },
            antecedent_io_core::OutSection {
                id: DATA_SECTION,
                content_type: antecedent_io_core::CONTENT_BYTES,
                bytes: data,
            },
        ],
    )
    .map_err(|error| error.into_io("unsupported inverse query artifact layout"))
}

/// Decode the three sections of a container, refusing another version before the
/// metadata is interpreted. No semantic validation happens here.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(bytes: &[u8]) -> Result<(InverseQueryMeta, Vec<u8>, Vec<u8>), IoError> {
    // A readable container of another kind is a recognised artifact this consumer
    // does not read; bytes that are no container at all keep the layout refusal.
    if let Some(kind) = recognised_artifact_kind(bytes)
        && kind != ARTIFACT_KIND
    {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("{KIND_MISMATCH}: expected {ARTIFACT_KIND}, supplied {kind}"),
        });
    }
    let parts = antecedent_io_core::decode_container(
        ARTIFACT_KIND,
        MAX_INVERSE_QUERY_ARTIFACT_BYTES,
        &[META_SECTION, CONTRACT_SECTION, DATA_SECTION],
        bytes,
    )
    .map_err(|error| error.into_io("unsupported inverse query artifact layout"))?;
    let Ok([meta_bytes, contract, data]) = <[Vec<u8>; 3]>::try_from(parts) else {
        return Err(IoError::Convert("unsupported inverse query artifact layout".into()));
    };
    let meta = antecedent_io_core::decode_with_version(&meta_bytes, |version: u16| {
        if version == INVERSE_QUERY_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(IoError::UnsupportedVersion { version: u32::from(version) })
        }
    })?;
    Ok((meta, contract, data))
}

fn query_refusal(error: &InverseQueryError) -> IoError {
    let refusal = error.to_refusal();
    IoError::Refused {
        code: refusal.code,
        message: format!("{}: {}", refusal.detail, refusal.offending.unwrap_or_default()),
    }
}

/// A validated inverse query, its forward evidence, its result table and identity.
#[derive(Clone, Debug)]
pub struct InverseQueryArtifact {
    query: InverseQuery,
    evidence: ForwardEvidence,
    result: InverseResult,
    identity: InverseQueryIdentity,
    source_evidence: Vec<crate::source_evidence::SourceEvidence>,
    forward_origin: Option<ForwardOrigin>,
}

fn claim_matches_source(
    claim: &ForwardClaim,
    source: &crate::source_evidence::SourceEvidence,
) -> bool {
    match claim {
        ForwardClaim::Means(means) => source.require_mean_source(means).is_ok(),
        ForwardClaim::Gaussian(law) => source
            .require_quantity_binding(std::slice::from_ref(&law.coordinate), &law.snapshot_id)
            .is_ok(),
        ForwardClaim::Law(law) => {
            let quantities = law
                .quantities()
                .iter()
                .cloned()
                .map(ScientificQuantity::try_from)
                .collect::<Result<Vec<_>, _>>();
            quantities.is_ok_and(|quantities| {
                source
                    .require_quantity_binding(&quantities, &law.metadata().identity.snapshot_id)
                    .is_ok()
            })
        }
    }
}

impl InverseQueryArtifact {
    /// Evaluate `query` on `evidence` through [`evaluate_inverse_query`] and seal the
    /// query, the evidence, the result and their identity.
    ///
    /// # Errors
    /// Every refusal of [`evaluate_inverse_query`] (a joint-law requirement the claim
    /// cannot meet, a mean-only claim asked for a quantile or probability, an invalid
    /// grid or constraint), and more embedded claims than
    /// [`MAX_INVERSE_QUERY_CLAIMS`].
    pub fn new(query: InverseQuery, evidence: ForwardEvidence) -> Result<Self, InverseQueryError> {
        let result = evaluate_inverse_query(&query, &evidence)?;
        let encoded = encode_evidence(&evidence);
        if encoded.claims.len() > MAX_INVERSE_QUERY_CLAIMS {
            return Err(InverseQueryError::InvalidParameter("claims"));
        }
        let identity =
            identity_of(&result.contract_identity, &InverseQueryWire::from_query(&query), &encoded)
                .map_err(|_| InverseQueryError::InvalidParameter("artifact_encoding"))?;
        Ok(Self {
            query,
            evidence,
            result,
            identity,
            source_evidence: Vec::new(),
            forward_origin: None,
        })
    }

    /// Where the point claim came from, when the artifact carries an origin.
    #[must_use]
    pub fn forward_origin(&self) -> Option<&ForwardOrigin> {
        self.forward_origin.as_ref()
    }

    /// Bind the origin of the point claim into the artifact identity.
    ///
    /// The origin's claim digest is recomputed from the embedded point claim and
    /// must equal the stored one, so an origin cannot be attached to numbers it
    /// does not describe. The origin is a declaration of provenance: this does
    /// not re-execute the producing route and grants no trust or calibration.
    ///
    /// # Errors
    /// An origin that is malformed, an artifact with no point claim, a claim
    /// digest that differs from the embedded claim's, or an origin already bound
    /// (`functional_inverse_query.forward_origin_invalid`).
    pub fn with_forward_origin(mut self, origin: ForwardOrigin) -> Result<Self, IoError> {
        origin.validate().map_err(|why| refused(ORIGIN_INVALID, why))?;
        if self.forward_origin.is_some() {
            return Err(refused(ORIGIN_INVALID, "an origin is already bound"));
        }
        let Some(point) = self.evidence.point.as_ref() else {
            return Err(refused(
                ORIGIN_INVALID,
                "an origin describes the point claim and none is embedded",
            ));
        };
        if forward_claim_digest(point)? != origin.claim_digest {
            return Err(refused(
                ORIGIN_INVALID,
                "the origin names a different claim than the one embedded",
            ));
        }
        self.identity.premises_digest = tagged_digest(
            "antecedent.inverse_query.forward_origin.v1",
            &to_cbor(&(&self.identity.premises_digest, origin.canonical_identity()))?,
        );
        self.identity.digest =
            whole_identity(&self.identity.premises_digest, &self.identity.data_digest);
        self.forward_origin = Some(origin);
        Ok(self)
    }

    /// Original diagnostic evidence retained alongside the actual inverse source.
    #[must_use]
    pub fn source_evidence(&self) -> &[crate::source_evidence::SourceEvidence] {
        &self.source_evidence
    }

    /// Attach original source evidence, rechecking semantic/numerical mean bindings and mappings.
    /// This never promotes imported evidence to executable state or native trust.
    pub fn with_source_evidence(
        mut self,
        sources: Vec<crate::source_evidence::SourceEvidence>,
    ) -> Result<Self, IoError> {
        if sources.len() > MAX_INVERSE_QUERY_SOURCE_EVIDENCE {
            return Err(refused(BOUNDS, "too many original source artifacts"));
        }
        let text = crate::decision_artifact::contract_to_json(&self.query.contract)?;
        for source in &sources {
            let projected =
                source.project(&text, &self.query.grid_order).map_err(source_refusal)?;
            if source.summary()["action_contributors"] != projected.summary()["action_contributors"]
                || source.summary()["contract_identity"] != projected.summary()["contract_identity"]
            {
                return Err(source_refusal(crate::source_evidence::SourceEvidenceError {
                    code: antecedent_core::reason_code!("invalid_argument"),
                    detail: antecedent_core::reason_details::source_evidence::ACTION_MISMATCH,
                }));
            }
            let matches_claim = |claim: &ForwardClaim| claim_matches_source(claim, source);
            let mut matched = self.evidence.point.as_ref().is_some_and(matches_claim);
            if let Some(region) = &self.evidence.interval_region {
                matched |= matches_claim(&region.lower) || matches_claim(&region.upper);
            }
            let atoms = self
                .evidence
                .identified_set
                .as_ref()
                .into_iter()
                .flat_map(|set| &set.members)
                .chain(self.evidence.scenarios.as_ref().into_iter().flatten());
            for atom in atoms {
                if let AtomEvidence::Evaluated(law) = &atom.evidence {
                    let quantities = law
                        .quantities()
                        .iter()
                        .cloned()
                        .map(ScientificQuantity::try_from)
                        .collect::<Result<Vec<_>, _>>();
                    matched |= quantities.is_ok_and(|quantities| {
                        source
                            .require_quantity_binding(
                                &quantities,
                                &law.metadata().identity.snapshot_id,
                            )
                            .is_ok()
                    });
                }
            }
            if !matched {
                return Err(source_refusal(crate::source_evidence::SourceEvidenceError {
                    code: antecedent_core::reason_code!("invalid_argument"),
                    detail:
                        antecedent_core::reason_details::source_evidence::FORWARD_BINDING_MISMATCH,
                }));
            }
        }
        self.identity = identity_of(
            &self.result.contract_identity,
            &InverseQueryWire::from_query(&self.query),
            &encode_evidence(&self.evidence),
        )?;
        if !sources.is_empty() {
            let original = identity_of(
                &self.result.contract_identity,
                &InverseQueryWire::from_query(&self.query),
                &encode_evidence(&self.evidence),
            )?;
            let bytes = sources
                .iter()
                .map(crate::source_evidence::SourceEvidence::export)
                .collect::<Result<Vec<_>, _>>()
                .map_err(source_refusal)?;
            self.identity.premises_digest = tagged_digest(
                "antecedent.inverse_query.source_evidence.v1",
                &to_cbor(&(original.premises_digest, bytes))?,
            );
            self.identity.digest =
                whole_identity(&self.identity.premises_digest, &self.identity.data_digest);
        }
        self.source_evidence = sources;
        match self.forward_origin.take() {
            Some(origin) => self.with_forward_origin(origin),
            None => Ok(self),
        }
    }

    /// The query.
    #[must_use]
    pub fn query(&self) -> &InverseQuery {
        &self.query
    }

    /// The forward evidence.
    #[must_use]
    pub fn evidence(&self) -> &ForwardEvidence {
        &self.evidence
    }

    /// The per-action status table, selection and flags.
    #[must_use]
    pub fn result(&self) -> &InverseResult {
        &self.result
    }

    /// The stored form of [`Self::result`].
    #[must_use]
    pub fn result_wire(&self) -> InverseResultWire {
        InverseResultWire::from(&self.result)
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &InverseQueryIdentity {
        &self.identity
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        let encoded = encode_evidence(&self.evidence);
        let contract = DecisionContractArtifact::new(self.query.contract.clone())?
            .to_bytes(CONTRACT_ARTIFACT_ID)?;
        let meta = InverseQueryMeta {
            version: INVERSE_QUERY_ARTIFACT_VERSION,
            feature: INVERSE_QUERY_ARTIFACT_FEATURE.into(),
            inference_claim: INVERSE_QUERY_INFERENCE_CLAIM.into(),
            contract_identity: self.result.contract_identity.clone(),
            query: InverseQueryWire::from_query(&self.query),
            claims: encoded.claims,
            evidence: encoded.evidence,
            identity: self.identity.clone(),
            result: InverseResultWire::from(&self.result),
            source_evidence: self
                .source_evidence
                .iter()
                .map(crate::source_evidence::SourceEvidence::export)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|cause| IoError::Convert(cause.to_string()))?,
            forward_origin: self.forward_origin.clone(),
        };
        encode_parts(&meta, &contract, &encoded.data, artifact_id)
    }

    /// Consume an artifact by re-evaluation.
    ///
    /// The contract, query and evidence are rebuilt, evaluated again through
    /// [`evaluate_inverse_query`], and the recomputed identity and result table must
    /// equal the stored ones. When the caller retained an `expected` identity of its
    /// own, that identity must match as well, so a resealed and self-consistent
    /// semantic change is refused. A continuous sample that stores a global
    /// feasibility claim is refused before replay.
    ///
    /// # Errors
    /// Corruption, another major version (`UnsupportedVersion`), a changed identity or
    /// result, a global claim on a continuous sample
    /// (`functional_inverse_query.global_feasibility_claim`) and any evaluation
    /// refusal, all with code `inverse_functional_unsupported` unless the engine's own
    /// refusal code applies.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&InverseQueryIdentity>,
    ) -> Result<Self, IoError> {
        Self::from_bytes_with_limits(bytes, expected, InverseQueryReadLimits::default())
    }

    /// [`Self::from_bytes`] accepting the counts `limits` allows instead of the defaults.
    ///
    /// # Errors
    /// Those of [`Self::from_bytes`].
    pub fn from_bytes_with_limits(
        bytes: &[u8],
        expected: Option<&InverseQueryIdentity>,
        limits: InverseQueryReadLimits,
    ) -> Result<Self, IoError> {
        let (meta, contract_bytes, data) = decode_parts(bytes)?;
        if meta.feature != INVERSE_QUERY_ARTIFACT_FEATURE
            || meta.inference_claim != INVERSE_QUERY_INFERENCE_CLAIM
        {
            return Err(wrong_contract("feature marker or inference claim is not this format's"));
        }
        if meta.claims.len() > limits.max_claims {
            return Err(refused(BOUNDS, "too many embedded claims"));
        }
        // Counted before any nested artifact is replayed.
        if meta.source_evidence.len() > limits.max_source_evidence {
            return Err(refused(BOUNDS, "too many original source artifacts"));
        }
        let contract =
            DecisionContractArtifact::from_bytes(&contract_bytes, &meta.contract_identity)?
                .contract()
                .clone();
        let claims = decode_claims(&meta.claims, &data)?;
        let evidence = evidence_from(&claims, &meta.evidence)?;
        let query = meta.query.clone().into_query(contract);
        if query.grid_scope == GridScope::ContinuousSample
            && (meta.result.exhaustive_over_declared_set
                || meta.result.existence == ExistenceWire::NoFeasibleActionInDeclaredSet)
        {
            return Err(refused(
                GLOBAL_CLAIM,
                "a sample of a continuous domain cannot claim global feasibility or infeasibility",
            ));
        }
        let sources = meta
            .source_evidence
            .iter()
            .map(|bytes| crate::source_evidence::SourceEvidence::consume(bytes))
            .collect::<Result<Vec<_>, _>>()
            .map_err(source_refusal)?;
        let mut artifact = Self::new(query, evidence)
            .map_err(|error| query_refusal(&error))?
            .with_source_evidence(sources)?;
        if let Some(origin) = meta.forward_origin.clone() {
            artifact = artifact.with_forward_origin(origin)?;
        }
        if let Some(field) = identity_diff(&meta.identity, &artifact.identity) {
            return Err(wrong_contract(&format!("stored {field} differs from the recomputed one")));
        }
        if meta.result != artifact.result_wire() {
            return Err(wrong_contract("stored result table does not replay from the evidence"));
        }
        if let Some(retained) = expected
            && let Some(field) = identity_diff(retained, &artifact.identity)
        {
            return Err(wrong_contract(&format!(
                "{field} differs from the consumer's retained identity"
            )));
        }
        Ok(artifact)
    }
}

/// The JSON form of a result table, for bridges that hand it to another language.
///
/// # Errors
/// A serialization failure.
pub fn result_to_json(result: &InverseResult) -> Result<String, IoError> {
    serde_json::to_string(&InverseResultWire::from(result))
        .map_err(|error| IoError::Convert(error.to_string()))
}

fn source_refusal(cause: crate::source_evidence::SourceEvidenceError) -> IoError {
    IoError::Refused {
        code: cause.code,
        message: format!("{}: original source evidence failed validation", cause.detail),
    }
}
