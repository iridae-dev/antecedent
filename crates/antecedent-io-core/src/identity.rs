//! Canonical encoding and domain-separated digests for contract identities.
//!
//! Payloads are existing CBOR wire types. Digests use BLAKE3 derive-key
//! domains from [`antecedent_core::IdentityDomain`]. `Debug`, Rust `Hash`,
//! arena offsets, and process-local keys are not durable identity.
//!
//! # Canonical encoding (`antecedent.identity.v2`)
//!
//! [`digest_canonical`] is the only advertised digest. Rules live next to it:
//!
//! | Rule | Encoding |
//! | --- | --- |
//! | Format tag | `IDENTITY_FORMAT` (little-endian `u16`) prefixes the hash; the same tag is stored on every payload. |
//! | Ordered fields | Schema variables, query lists, partitions, and estimands keep writer order. |
//! | Graphs | Edge lists are sorted; temporal-DAG, temporal-ADMG and temporal-class nodes are sorted by their key, repeated nodes merged, and edges re-indexed (class edges re-oriented to ascending endpoints with marks swapped); the class kinds are `temporal_cpdag.v2` and `temporal_pag.v2`. Insertion order never reaches a digest. |
//! | Set-valued fields | Observation tags (with their variable arguments), named prior-parameter pairs (a CBOR list of `(source, target)` tuples), selection targets, and class-prior pairs are sorted then deduplicated. Graph-posterior atoms are sorted by their encoding with the position-derived execution key cleared. |
//! | Configuration | Estimator, prior, response, observation, and execution settings are structured wires of scalar fields; data-sized vectors are [`PayloadDigestWire`]s. Nothing is rendered through `Debug` or `Display`. |
//! | Sub-digests | [`payload_digest`] (derive-key `antecedent.identity.payload`, a `kind` label, then little-endian bytes) is the only sub-digest convention. |
//! | Floats | Knobs, weights, masses, and claim values are IEEE-754 bits (`f64::to_bits`) carried as `u64`. Floats that remain inside reused wire types (for example RD design cutoffs and query values) use ciborium's shortest exact encoding: a value is written in the narrowest of f16/f32/f64 that represents it exactly, NaN payloads and the sign of zero are kept, so the encoding is injective on bits. |
//! | Collision | BLAKE3-256 equality is identity under this encoding, not a proof. Independent consume rehashes stored payloads; a colliding advertisement without a matching payload is unresolved. Bump [`IDENTITY_FORMAT`] to change a rule. |
//!
//! Graph-posterior atoms are not position-derived cache keys. Durable identity
//! is each atom's structure (static DAG, or lagged and contemporaneous DBN
//! edges) plus its posterior weight, over the premises' schema names.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalSchema, ExecutionContext, IDENTITY_FORMAT, IdentityDomain, NodeRef, SemanticDigest,
    VariableId,
};
use antecedent_discovery::adjacency_mask::{
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask,
    pag_from_adjacency_mask,
};
use antecedent_discovery::dbn_posterior::{
    temporal_cpdag_from_dbn_atoms, temporal_dag_from_dbn_atoms, temporal_pag_from_dbn_atoms,
};
use antecedent_discovery::{AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{
    Admg, Cpdag, Dag, MarkedEdge, Pag, TemporalAdmg, TemporalCpdag, TemporalDag, TemporalPag,
};
use antecedent_identify::IdentificationResult;
use serde::{Deserialize, Serialize};

use crate::analysis_wire::{IdentifiedEstimandWire, RdDesignWire};
use crate::convert::{
    admg_to_wire, cpdag_to_wire, dag_to_wire, endpoint_to_wire, pag_to_wire, schema_to_wire,
    vars_to_raw,
};
use crate::discovery_wire::{
    DiscoveryTemporalGraphWire, TemporalGraphWire, TemporalNodeKeyWire, temporal_dag_to_wire,
};
use crate::error::IoError;

pub use crate::claim_identity::{ClaimIdentityWire, claim_digest};
use crate::expr_wire::{ExprArenaWire, expr_arena_to_wire};
#[cfg(test)]
use crate::query_wire::InterventionWire;
use crate::query_wire::{CausalQueryWire, TargetPopulationWire};
use crate::to_cbor;
use crate::trace::{AssumptionRecordWire, assumptions_to_wire};
use crate::wire::{AdmgWire, CpdagWire, DagWire, EndpointWire, PagWire, SchemaWire};

/// Domain-separated BLAKE3 digest of a canonical CBOR payload.
///
/// Hash is `BLAKE3-derive-key(domain)(IDENTITY_FORMAT_le || payload)`. Changing
/// the format tag, domain key, ordered-versus-set rule, or float-bit rule
/// changes the advertised digest. See the module docs for the encoding table.
#[must_use]
pub fn digest_canonical(domain: IdentityDomain, payload: &[u8]) -> SemanticDigest {
    SemanticDigest::from_bytes(antecedent_core::derive_key_digest(
        domain.derive_key(),
        &[&IDENTITY_FORMAT.to_le_bytes(), payload],
    ))
}

/// Digest a serializable payload.
///
/// # Errors
///
/// CBOR encode failure.
pub fn digest_wire<T: Serialize>(
    domain: IdentityDomain,
    value: &T,
) -> Result<SemanticDigest, IoError> {
    let mut hasher = blake3::Hasher::new_derive_key(domain.derive_key());
    hasher.update(&IDENTITY_FORMAT.to_le_bytes());
    ciborium::into_writer(value, &mut hasher).map_err(IoError::cbor_encode)?;
    Ok(SemanticDigest::from_bytes(*hasher.finalize().as_bytes()))
}

/// Lower-case hex of [`digest_wire`], for artifact ids and wire-level identity strings.
///
/// # Errors
///
/// CBOR encode failure.
// doc(hidden): shared by antecedent-io wire digests.
#[doc(hidden)]
pub fn digest_wire_hex<T: Serialize>(domain: IdentityDomain, value: &T) -> Result<String, IoError> {
    Ok(digest_wire(domain, value)?.to_hex())
}

mod inference;
mod reuse;
mod snapshot;

pub use inference::{
    BayesianBindingWire, EstimatorConfigWire, EstimatorSpecWire, ExternalComposeIdentityWire,
    ExternalPriorSourceIdentityWire, GlmOptionsWire, InferenceBindingWire, ObservationOptionsWire,
    OverlapPolicyWire, PriorMappingIdentityWire, PriorSetIdentityWire, PriorSpecIdentityWire,
    ResponseOptionsWire, SplitIdentityWire, inference_binding_digest,
};

pub use reuse::{
    AdjustedFitReuseIdentityWire, ROW_WEIGHTS_PAYLOAD, ScoreReuseIdentityWire, ScoreTableKey,
    TargetWeightsIdentityWire, adjusted_fit_reuse_digest, score_reuse_digest,
    target_weights_digest,
};

pub use snapshot::{
    CROSS_PRODUCT_INTERCEPT, CrossProductMomentsWire, DataPartitionIdentityWire,
    DataSnapshotIdentityWire, ImputationSpecWire, InterferenceSnapshotWire, LinearFitMomentsWire,
    MissingPolicyWire, NestedCounterfactualFitWire, ObservationIdentityWire,
    ParticipantBindingWire, data_snapshot_digest, observation_identity_wire,
};

mod labels;

pub use labels::executed_functional_labels;
#[cfg(test)]
use labels::intervention_label;

/// Target-layer payload: schema bindings plus the typed query.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TargetIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Variable-name bindings for every dense id the query may mention.
    pub schema: SchemaWire,
    /// Typed query (population, interventions, functional, temporal policy).
    pub query: CausalQueryWire,
}

impl TargetIdentityWire {
    /// Population-free question: the same payload with the population set to
    /// all observed units. Identification premises hash this projection, so a
    /// population change keeps identification and moves target and program.
    #[must_use]
    pub fn question(&self) -> Self {
        self.clone().into_question()
    }

    /// [`Self::question`] for an owned payload: the population is replaced in
    /// place and nothing is cloned.
    #[must_use]
    pub fn into_question(mut self) -> Self {
        if let Some(population) = self.query.target_population_mut() {
            *population = TargetPopulationWire::AllObserved;
        }
        self
    }

    /// Target digest of the population-free question, without cloning the
    /// payload: the population is set aside while the question is hashed and put
    /// back before this returns, error or not.
    ///
    /// # Errors
    ///
    /// The payload does not encode.
    pub fn question_digest(&mut self) -> Result<antecedent_core::SemanticDigest, IoError> {
        let set_aside = self
            .query
            .target_population_mut()
            .map(|population| std::mem::replace(population, TargetPopulationWire::AllObserved));
        let digest = digest_wire(IdentityDomain::Target, self);
        if let (Some(original), Some(population)) = (set_aside, self.query.target_population_mut())
        {
            *population = original;
        }
        digest
    }
}

/// Digest of a data-sized payload embedded in an identity wire.
///
/// `BLAKE3-derive-key("antecedent.identity.payload")(IDENTITY_FORMAT_le ||
/// len(kind)_le_u64 || kind || bytes)`. The `kind` label separates payload
/// families (row indices, weights, cluster ids, prior bytes, ...), so equal
/// bytes of different families never share a digest. This is the only
/// sub-digest convention inside identity wires.
///
/// It frames exactly one homogeneous payload (an array of fixed-width values
/// or one opaque byte string) under a label, in the same derive-key family as
/// [`digest_canonical`]. A digest over several named fields is not a payload:
/// it is a [`antecedent_core::CanonicalWriter`] stream with its own domain tag.
#[must_use]
pub fn payload_digest(kind: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = payload_hasher(kind);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

/// Same framing as [`payload_digest`], writing the CBOR of `value` straight
/// into the hasher. The digest bytes match `payload_digest(kind, &to_cbor(value)?)`.
pub(crate) fn payload_digest_of<T: Serialize>(kind: &str, value: &T) -> Result<[u8; 32], IoError> {
    let mut hasher = payload_hasher(kind);
    ciborium::into_writer(value, &mut hasher).map_err(IoError::cbor_encode)?;
    Ok(*hasher.finalize().as_bytes())
}

/// Key-derivation context of [`payload_digest`] and [`payload_digest_of`].
const PAYLOAD_CONTEXT: &str = "antecedent.identity.payload";

fn payload_hasher(kind: &str) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new_derive_key(PAYLOAD_CONTEXT);
    hasher.update(&IDENTITY_FORMAT.to_le_bytes());
    hasher.update(&(kind.len() as u64).to_le_bytes());
    hasher.update(kind.as_bytes());
    hasher
}

/// Content digest plus element count for a data-sized payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PayloadDigestWire {
    /// [`payload_digest`] of the little-endian payload bytes.
    pub digest: [u8; 32],
    /// Element count.
    pub len: u64,
}

/// Digest `values` as consecutive `N`-byte words under `kind`, streaming through a small
/// stack buffer instead of materializing the payload. Digest-identical to
/// [`payload_digest`] over the concatenated words.
fn digest_words<T: Copy, const N: usize>(
    kind: &str,
    values: &[T],
    encode: impl Fn(T) -> [u8; N],
) -> [u8; 32] {
    const CHUNK: usize = 8192;
    let mut hasher = payload_hasher(kind);
    let mut chunk = [0_u8; CHUNK];
    let mut used = 0;
    for &value in values {
        if used + N > CHUNK {
            hasher.update(&chunk[..used]);
            used = 0;
        }
        chunk[used..used + N].copy_from_slice(&encode(value));
        used += N;
    }
    hasher.update(&chunk[..used]);
    *hasher.finalize().as_bytes()
}

impl PayloadDigestWire {
    /// Digest little-endian `u32` values.
    #[must_use]
    pub fn u32s(kind: &str, values: &[u32]) -> Self {
        Self { digest: digest_words(kind, values, u32::to_le_bytes), len: values.len() as u64 }
    }

    /// Digest little-endian `u64` row ids.
    #[must_use]
    pub fn u64s(kind: &str, values: &[u64]) -> Self {
        Self { digest: digest_words(kind, values, u64::to_le_bytes), len: values.len() as u64 }
    }

    /// Digest little-endian `i64` values.
    #[must_use]
    pub fn i64s(kind: &str, values: &[i64]) -> Self {
        Self { digest: digest_words(kind, values, i64::to_le_bytes), len: values.len() as u64 }
    }

    /// Digest exact IEEE-754 bits of `f64` values.
    #[must_use]
    pub fn f64s(kind: &str, values: &[f64]) -> Self {
        Self {
            digest: digest_words(kind, values, |value| value.to_bits().to_le_bytes()),
            len: values.len() as u64,
        }
    }

    /// Digest length-prefixed groups of `u32` values.
    #[must_use]
    pub fn u32_groups(kind: &str, groups: &[Vec<u32>]) -> Self {
        let mut hasher = payload_hasher(kind);
        let mut chunk = [0_u8; 8192];
        for group in groups {
            hasher.update(&(group.len() as u64).to_le_bytes());
            let mut used = 0;
            for value in group {
                if used + 4 > chunk.len() {
                    hasher.update(&chunk[..used]);
                    used = 0;
                }
                chunk[used..used + 4].copy_from_slice(&value.to_le_bytes());
                used += 4;
            }
            hasher.update(&chunk[..used]);
        }
        Self { digest: *hasher.finalize().as_bytes(), len: groups.len() as u64 }
    }

    /// Digest raw bytes.
    #[must_use]
    pub fn bytes(kind: &str, bytes: &[u8]) -> Self {
        Self { digest: payload_digest(kind, bytes), len: bytes.len() as u64 }
    }
}

mod graph;

use graph::canonical_admg_wire;
pub use graph::{
    GraphIdentityWire, IdentityMarkedEdgeWire, IdentityNodeWire, PosteriorAtomGraphWire,
    PosteriorAtomIdentityWire, TemporalAdmgIdentityWire, TemporalClassIdentityWire, admg_identity,
    canonical_dag_wire, canonical_temporal_dag_wire, cpdag_identity, dag_identity,
    graph_posterior_atom_identities, pag_identity, temporal_admg_identity, temporal_cpdag_identity,
    temporal_dag_identity, temporal_pag_identity,
};

/// RD configuration hashed with identification premises.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RdConfigWire {
    /// Running-variable id.
    pub running_variable: u32,
    /// Cutoff bits.
    pub cutoff_bits: u64,
    /// Bandwidth bits.
    pub bandwidth_bits: u64,
    /// Analytic SE kind, when set.
    pub se_kind: Option<String>,
}

/// Caller-supplied mass over temporal-class members.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClassPriorIdentityWire {
    /// Masses aligned with enumerated cases, as IEEE-754 bits.
    Ordered(Vec<u64>),
    /// `(completion fingerprint, mass bits)` sorted by fingerprint.
    Pairs(Vec<(u64, u64)>),
}

/// Transport selection diagram and trial column bindings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransportIdentityWire {
    /// Selection-diagram causal graph (canonical edge order), when frozen.
    pub selection_graph: Option<AdmgWire>,
    /// Selection targets (mechanisms that may differ), sorted.
    pub selection_targets: Vec<u32>,
    /// Trial-membership variable, when bound.
    pub trial: Option<u32>,
    /// Known `P(S=1 | X)` column, when bound.
    pub selection_probability: Option<u32>,
    /// Known `P(A=1 | X, S=1)` column, when bound.
    pub treatment_probability: Option<u32>,
}

/// Canonical transport premises from a frozen selection diagram and trial columns.
///
/// # Errors
///
/// Capacity overflow while encoding the selection graph.
pub fn transport_identity(
    selection: Option<&antecedent_graph::SelectionDiagram>,
    trial: Option<(VariableId, VariableId, VariableId)>,
) -> Result<TransportIdentityWire, IoError> {
    let mut selection_targets: Vec<u32> =
        selection.map(|diagram| vars_to_raw(diagram.selection_targets())).unwrap_or_default();
    selection_targets.sort_unstable();
    Ok(TransportIdentityWire {
        selection_graph: selection
            .map(|diagram| canonical_admg_wire(diagram.causal_graph()))
            .transpose()?,
        selection_targets,
        trial: trial.map(|(trial, _, _)| trial.raw()),
        selection_probability: trial.map(|(_, selection, _)| selection.raw()),
        treatment_probability: trial.map(|(_, _, treatment)| treatment.raw()),
    })
}

/// Identification-layer payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentificationIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Target digest of the population-free question ([`TargetIdentityWire::question`]).
    pub target_question: [u8; 32],
    /// Population `depends_on` variable ids (empty when the population is not weight-backed).
    #[serde(default)]
    pub population_depends_on: Vec<u32>,
    /// RD configuration, when the identifier is a discontinuity design.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rd_config: Option<RdConfigWire>,
    /// Caller-supplied class prior over temporal-class members.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_prior: Option<ClassPriorIdentityWire>,
    /// Transport selection diagram and trial bindings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<TransportIdentityWire>,
    /// Graph class wire name.
    pub graph_class: String,
    /// Audit metadata: how structure was supplied. Excluded from semantic identity.
    pub structure_source: String,
    /// Audit metadata: accepted-graph version. Excluded from semantic identity.
    pub accepted_version: u32,
    /// Audit metadata: discovery algorithm id. Excluded from semantic identity.
    pub algorithm_id: Option<String>,
    /// Optional schema-name binding on the accepted graph.
    pub schema_names: Option<Vec<String>>,
    /// Canonical graph encoding.
    pub graph: GraphIdentityWire,
    /// Observation-contract digest.
    pub observation: [u8; 32],
}

/// Digest identification premises.
///
/// # Errors
///
/// CBOR encode failure.
pub fn identification_digest(wire: &IdentificationIdentityWire) -> Result<SemanticDigest, IoError> {
    // Keep the public wire record round-trippable, but hash only scientific
    // premises. Re-accepting the same graph must not invalidate identification.
    #[derive(Serialize)]
    struct Premises<'a> {
        format: u16,
        target_question: &'a [u8; 32],
        population_depends_on: &'a [u32],
        rd_config: &'a Option<RdConfigWire>,
        class_prior: &'a Option<ClassPriorIdentityWire>,
        transport: &'a Option<TransportIdentityWire>,
        graph_class: &'a str,
        schema_names: &'a Option<Vec<String>>,
        graph: &'a GraphIdentityWire,
        observation: &'a [u8; 32],
    }
    let graph = wire.graph.premises();
    digest_wire(
        IdentityDomain::Identification,
        &Premises {
            format: wire.format,
            target_question: &wire.target_question,
            population_depends_on: &wire.population_depends_on,
            rd_config: &wire.rd_config,
            class_prior: &wire.class_prior,
            transport: &wire.transport,
            graph_class: &wire.graph_class,
            schema_names: &wire.schema_names,
            graph: &graph,
            observation: &wire.observation,
        },
    )
}

/// Identification-product payload. Excludes search-effort counters and prose.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentificationProductWire {
    /// Identity format.
    pub format: u16,
    /// Identification status.
    pub status: String,
    /// Estimands (variable ids are bound by the target schema).
    pub estimands: Vec<IdentifiedEstimandWire>,
    /// Expression arena owning functionals (not arena-local ids alone).
    pub arena: ExprArenaWire,
    /// Derivation rule ids in order. Step detail text is diagnostic prose
    /// (it may render statuses or error messages) and is not identity.
    pub derivation_rules: Vec<String>,
    /// Required assumptions.
    pub required_assumptions: Vec<AssumptionRecordWire>,
    /// Whether a hedge witness is attached.
    pub hedge: bool,
    /// Whether identification search was capped.
    pub search_capped: bool,
    /// Class-envelope shape, when the product came from an enumerated class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope: Option<IdentificationEnvelopeWire>,
}

/// Shape of the enumerated class an identification product covers.
///
/// Two envelopes over the same class that examined different completions, or
/// carry different identified / unidentified mass, are different products even
/// when their estimands agree.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentificationEnvelopeWire {
    /// Cases examined.
    pub cases: u64,
    /// Identified weight (IEEE bits).
    pub identified_weight_bits: u64,
    /// Unidentified weight (IEEE bits).
    pub unidentified_weight_bits: u64,
    /// Cases whose search was truncated before it could decide.
    pub truncated_completions: u64,
}

fn estimand_wire(estimand: &IdentifiedEstimand) -> IdentifiedEstimandWire {
    IdentifiedEstimandWire {
        method: estimand.method.to_string(),
        adjustment_set: vars_to_raw(&estimand.adjustment_set),
        instruments: vars_to_raw(&estimand.instruments),
        mediators: vars_to_raw(&estimand.mediators),
        functional: estimand.functional.raw(),
        rd_design: estimand.rd_design.map(|design| RdDesignWire {
            running_variable: design.running_variable.raw(),
            cutoff: design.cutoff,
            bandwidth: design.bandwidth,
        }),
    }
}

/// Digest a stored identification-product payload.
///
/// # Errors
///
/// CBOR encode failure.
pub fn identification_product_digest_wire(
    wire: &IdentificationProductWire,
) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::IdentificationProduct, wire)
}

/// Encode cached identification products for digest and durable rehash.
///
/// # Errors
///
/// Arena encode failure.
pub fn identification_product_wire(
    result: &IdentificationResult,
    search_capped: bool,
) -> Result<IdentificationProductWire, IoError> {
    Ok(IdentificationProductWire {
        format: IDENTITY_FORMAT,
        // `IdentificationStatus::as_str` is the one snake_case table contracts
        // and the artifact wire read; the digest reads the same one.
        status: result.status.as_str().into(),
        estimands: result.estimands.iter().map(estimand_wire).collect(),
        arena: expr_arena_to_wire(&result.arena)?,
        derivation_rules: result
            .derivation
            .steps
            .iter()
            .map(|step| step.rule.to_string())
            .collect(),
        required_assumptions: assumptions_to_wire(&result.required_assumptions),
        hedge: result.hedge.is_some(),
        search_capped,
        envelope: None,
    })
}

/// Digest cached identification products.
///
/// # Errors
///
/// Arena or CBOR encode failure.
pub fn identification_product_digest(
    result: &IdentificationResult,
    search_capped: bool,
) -> Result<SemanticDigest, IoError> {
    identification_product_digest_wire(&identification_product_wire(result, search_capped)?)
}

/// Supported inferential commitments (not numeric knobs).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferentialCommitmentsWire {
    /// Identity format.
    pub format: u16,
    /// Caller-selected estimator family id.
    pub estimator: Option<String>,
    /// Resolved estimator from the compiled plan.
    pub resolved_estimator: Option<String>,
    /// Identifier id.
    pub identifier: Option<String>,
    /// `frequentist` or `bayesian`.
    pub inference: String,
    /// Validation suite id.
    pub validation_suite: Option<String>,
    /// Interval method (`IntervalMethod::as_str()`).
    pub interval_method: String,
    /// Analytic SE kind, when the interval is analytic.
    pub se_kind: Option<String>,
    /// Whether a prior is required.
    pub prior_required: bool,
}

/// Versioned temporal attribution row semantics. An absent field in an older
/// program remains a legacy/unknown convention; it is not reinterpreted as v2.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TemporalAttributionRowsWire {
    /// Complete observed windows and run-aware block resampling. V2 keeps each
    /// primary block draw and derives additional segment streams by hashing its
    /// domain, replicate key, primary draw bits, block, run and segment positions
    /// in that order through `derive_stream`; population-specific draws derive
    /// the replicate key from the replicate and the population index. A sampled
    /// segment wraps circularly within its own run, and a global end-to-start
    /// wrap starts a derived segment even for a single run. Mechanism-change
    /// permutations retain the parent Fisher–Yates stream; split segments use
    /// separate `0x6d65_6368_7061_7232` parent and `0x6d65_6368_7365_6732`
    /// segment domains keyed by target stream, permutation and source positions.
    CompleteObservedDerivedRunBlocksV2,
}

/// Program-layer payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProgramIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Target digest, including the target population. Identification hashes
    /// the population-free question, so ATE and ATT share identification but
    /// are different programs.
    pub target: [u8; 32],
    /// Identification digest.
    pub identification: [u8; 32],
    /// Identification-product digest, when prepared.
    pub identification_product: Option<[u8; 32]>,
    /// Temporal-class completion search budget, when capped by the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_budget: Option<u64>,
    /// Supported inferential commitments.
    pub commitments: InferentialCommitmentsWire,
    /// Absent in older programs; newer temporal attribution binds its row law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_attribution_rows: Option<TemporalAttributionRowsWire>,
    /// Retained checked program for high-level functional-distribution and scalar
    /// functional-effect results. Omitted by older artifacts and other estimators.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub functional_program: Option<crate::FunctionalProgramWire>,
    /// Checked AIPW semantic lowering (target roles and procedure choices).
    /// Data-dependent complete-case rows are carried separately by the contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_aipw_lowering: Option<crate::CheckedAipwLoweringWire>,
    /// Checked semantic lowering for a prepared linear front-door program.
    /// Older artifacts omit it and cannot be independently verified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_frontdoor_lowering: Option<CheckedFrontDoorLoweringWire>,
    /// Checked semantic lowering for prepared IV Wald and single-instrument 2SLS.
    /// Older artifacts omit it and cannot be independently verified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_iv_lowering: Option<crate::CheckedIvLoweringWire>,
    /// Checked semantic lowering for prepared linear adjustment ATE routes.
    /// Missing in older artifacts, which remain readable but cannot be
    /// independently replayed as checked linear executions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_linear_adjustment_lowering: Option<CheckedLinearAdjustmentLoweringWire>,
    /// Ordered checked member programs for an ADMG mean response curve.
    /// Older artifacts omit the grid family and cannot independently replay it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_functional_response_grid: Option<CheckedFunctionalResponseGridWire>,
    /// Checked linear-Gaussian shared-exogenous natural direct effect operation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_nested_counterfactual: Option<CheckedNestedCounterfactualWire>,
    /// Checked semantic lowering of a cross-fitted doubly-robust learner average effect.
    /// Older artifacts and executions outside the supported operation omit it and cannot be
    /// independently replayed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_dr_lowering: Option<crate::CheckedDrLoweringWire>,
    /// Per-atom checked programs and the mixing rule of a functional effect or
    /// interventional distribution answered over a graph posterior. Such a result is
    /// answered per atom and mixed, so it has no single program; this carries what each
    /// atom was answered by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_posterior_programs: Option<GraphPosteriorProgramsWire>,
    /// Digest of the provenance of a discovered structure the program runs on:
    /// the search method, its configuration and bounds, the data snapshot it
    /// read, the discovery program and claim, and the review that accepted it.
    /// Absent for a supplied structure. Identification excludes the discovery
    /// algorithm as audit data (re-accepting a graph must not invalidate its
    /// identification); the program that fits and estimates on it binds it here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_provenance: Option<[u8; 32]>,
}

/// Mixing rule of a graph-posterior functional result: the posterior-weighted mean over the
/// atoms that identify and evaluate, with unidentified and unevaluable mass kept apart and
/// never redistributed.
pub const GRAPH_POSTERIOR_MIXING_RULE: &str = "posterior_weighted_mean_over_identified_atoms";

/// What one distinct graph of a posterior was answered by.
///
/// A single-structure atom (DAG, ADMG) is answered by one checked functional program; a
/// class atom (CPDAG, PAG) is answered per completion, so it carries the class's
/// identification product instead. An atom the identifier refuses carries neither.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GraphPosteriorAtomProgramWire {
    /// Key of the atom in the result's structural mixture.
    pub graph_key: u64,
    /// Total posterior weight of the atom as IEEE-754 bits: the sum of its samples.
    pub weight_bits: u64,
    /// Indices of the identification atoms (posterior samples) this atom aggregates.
    pub samples: Vec<u64>,
    /// Checked program the atom was answered by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub functional_program: Option<crate::FunctionalProgramWire>,
    /// Class identification product the atom was answered by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identification_product: Option<IdentificationProductWire>,
}

/// Per-atom programs of a graph-posterior functional result, chained by digest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GraphPosteriorProgramsWire {
    /// Identity format.
    pub format: u16,
    /// Mixing rule, [`GRAPH_POSTERIOR_MIXING_RULE`].
    pub mixing: String,
    /// One entry per distinct graph, in order of first appearance.
    pub atoms: Vec<GraphPosteriorAtomProgramWire>,
    /// Digest chain over the mixing rule and every atom's key, weight, samples, program and
    /// product, in order.
    pub chain: [u8; 32],
}

impl GraphPosteriorProgramsWire {
    /// Recompute the digest chain of the mixing rule and atoms.
    ///
    /// # Errors
    ///
    /// CBOR encode failure.
    pub fn compute_chain(
        mixing: &str,
        atoms: &[GraphPosteriorAtomProgramWire],
    ) -> Result<[u8; 32], IoError> {
        const DOMAIN: &str = "antecedent.identity.graph_posterior_programs.v1";
        let header = to_cbor(&(mixing, u64::try_from(atoms.len()).unwrap_or(u64::MAX)))?;
        let mut link = antecedent_core::derive_key_digest(DOMAIN, &[&header]);
        for atom in atoms {
            let encoded = to_cbor(atom)?;
            link = antecedent_core::derive_key_digest(DOMAIN, &[&link, &encoded]);
        }
        Ok(link)
    }

    /// Seal the atoms under the mixing rule.
    ///
    /// # Errors
    ///
    /// CBOR encode failure.
    pub fn seal(atoms: Vec<GraphPosteriorAtomProgramWire>) -> Result<Self, IoError> {
        let chain = Self::compute_chain(GRAPH_POSTERIOR_MIXING_RULE, &atoms)?;
        Ok(Self {
            format: IDENTITY_FORMAT,
            mixing: GRAPH_POSTERIOR_MIXING_RULE.to_owned(),
            atoms,
            chain,
        })
    }
}

/// Fixed three-node Markovian natural direct effect procedure.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckedNestedCounterfactualWire {
    /// Wire format.
    pub format: u16,
    /// Semantic variable roles, never remapped to design-column indices.
    pub treatment: u32,
    /// Mediator variable id.
    pub mediator: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// IEEE-754 treatment levels.
    pub control_bits: u64,
    /// Active treatment level, as IEEE-754 bits.
    pub active_bits: u64,
    /// Explicit model and cross-world procedure identity.
    pub model: String,
    /// Shared-exogenous nested-world procedure name.
    pub procedure: String,
}

/// Versioned checked member programs for a finite response curve.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CheckedFunctionalResponseGridWire {
    /// Grid-family wire format.
    pub format: u16,
    /// Intervened variable id.
    pub treatment: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// Members in the exact target-grid order.
    pub members: Vec<CheckedFunctionalResponseMemberWire>,
}

/// One checked response-curve intervention and its identified functional.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CheckedFunctionalResponseMemberWire {
    /// Intervention level bits, preserving exact IEEE-754 identity.
    pub grid_value_bits: u64,
    /// Exact one-level response query used to identify this member.
    pub query: crate::CausalQueryWire,
    /// Member-specific identification product.
    pub identification: IdentificationProductWire,
    /// Checked scalar functional evaluated for this member.
    pub program: crate::FunctionalProgramWire,
}

/// Durable target, design, estimator, and uncertainty choice for a checked
/// linear adjustment execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CheckedLinearAdjustmentLoweringWire {
    /// Wire format.
    pub format: u16,
    /// Source interventional target root.
    pub functional: u32,
    /// Executable observational adjustment root.
    pub executable: u32,
    /// Semantic variable roles.
    pub treatment: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// Adjustment variables in design order.
    pub adjustment: Vec<u32>,
    /// Active intervention value bits.
    pub active_bits: u64,
    /// Control intervention value bits.
    pub control_bits: u64,
    /// Population binding.
    pub population: crate::TargetPopulationWire,
    /// Ordered design column roles (`intercept`, `treatment`, `covariate:<id>`).
    pub design_columns: Vec<String>,
    /// Fit family and parameter bits (`ols`, `ridge:<bits>`, `lasso:<bits>`, `huber:<bits>`).
    pub fit_kind: String,
    /// Numerical backend.
    pub backend: String,
    /// Selected analytic SE kind.
    pub se_kind: String,
    /// HAC lag, where the selected SE requires one.
    pub se_lag: Option<u64>,
    /// Bootstrap replicate count.
    pub bootstrap_replicates: u32,
    /// Resolved uncertainty method commitment.
    pub interval_method: String,
    /// Complete-case row count used by the prepared design.
    pub complete_case_rows: u64,
    /// Checked arena retaining both source and executable roots.
    pub arena: crate::ExprArenaWire,
}

/// Factor domain for a portable finite-discrete law.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DistributionFactorDomainWire {
    /// Observational source law.
    Observational,
    /// Interventional source law.
    Interventional,
}

/// Exact identity of one provider factor table.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributionFactorKeyWire {
    /// Factor variables in provider order.
    pub variables: Vec<u32>,
    /// Conditioning variables in provider order.
    pub conditioned_on: Vec<u32>,
    /// Concrete intervention assignments.
    pub intervention: Vec<crate::expr_wire::InterventionAssignmentWire>,
    /// Probability domain.
    pub domain: DistributionFactorDomainWire,
    /// Population identity; empty means the default study population.
    pub population: String,
    /// Regime identity, when evidence is regime-bound.
    pub regime: Option<u32>,
}

/// One complete value row for a factor table, ordered as variables then conditioners.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributionFactorRowWire {
    /// Joint assignment values in the factor key's declared variable order.
    pub values: Vec<crate::ValueWire>,
    /// Conditional probability mass.
    pub probability: f64,
}

/// Complete factor table for one exact provider-factor identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributionFactorTableWire {
    /// Exact factor identity.
    pub key: DistributionFactorKeyWire,
    /// Complete joint support and probability rows.
    pub rows: Vec<DistributionFactorRowWire>,
}

/// Versioned finite-discrete provider snapshot used for detached replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributionFactorLawsWire {
    /// Wire format version.
    pub format: u16,
    /// Provider identity.
    pub provider: String,
    /// Source rows before complete-case filtering.
    pub source_rows: u64,
    /// Rows retained by the empirical provider.
    pub complete_case_rows: u64,
    /// Missing-row policy identifier.
    pub missing_row_policy: String,
    /// Complete finite domain for each semantic variable.
    pub domains: Vec<(u32, Vec<crate::ValueWire>)>,
    /// Exact factor requirements recorded by the checked estimator.
    pub requirements: Vec<DistributionFactorKeyWire>,
    /// Factor tables, including the optional free-variable weighting law.
    pub factors: Vec<DistributionFactorTableWire>,
}

/// Durable checked lowering for the linear front-door path-product route.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CheckedFrontDoorLoweringWire {
    /// Wire format.
    pub format: u16,
    /// Source root in the identification arena.
    pub functional: u32,
    /// Executable observational root.
    pub executable: u32,
    /// Treatment variable id.
    pub treatment: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// Mediator ids in selected-estimand order.
    pub mediators: Vec<u32>,
    /// Active intervention value bits.
    pub active_bits: u64,
    /// Control intervention value bits.
    pub control_bits: u64,
    /// Checked procedure tag.
    pub procedure: String,
    /// Checked-case count used by the fitted procedure.
    pub complete_case_rows: u64,
    /// Overlap handling tag.
    pub overlap: String,
    /// Uncertainty choice bound by the program commitments.
    pub uncertainty: String,
    /// Checked expression arena.
    pub arena: crate::ExprArenaWire,
}

/// Digest a program identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn program_digest(wire: &ProgramIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::Program, wire)
}

/// Kernel selection policy that ran.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KernelPolicyWire {
    /// Portable optimized kernels allowed.
    pub allow_portable_optimized: bool,
    /// Architecture SIMD allowed.
    pub allow_arch_simd: bool,
    /// Scalar kernels forced.
    pub force_scalar: bool,
}

/// Adaptive Monte Carlo early-stop budget, when enabled.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdaptiveBudgetWire {
    /// Minimum replicates or draws before a stop is allowed.
    pub minimum: u64,
    /// Relative precision target bits.
    pub epsilon_bits: u64,
    /// ESS target bits (draw budget only).
    pub ess_target_bits: Option<u64>,
}

/// Execution-lineage payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExecutionIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Library version.
    pub library_version: String,
    /// Seed stream identifier.
    pub seed: u64,
    /// Thread budget.
    pub threads: u32,
    /// Kernel / backend policy label.
    pub backend: String,
    /// Full kernel selection policy.
    pub kernel: KernelPolicyWire,
    /// Determinism policy (`strict` / `prefer_fast`).
    pub determinism: String,
    /// Adaptive bootstrap early-stop budget, when enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_bootstrap: Option<AdaptiveBudgetWire>,
    /// Adaptive posterior-draw early-stop budget, when enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_draws: Option<AdaptiveBudgetWire>,
}

/// Digest execution lineage.
///
/// # Errors
///
/// CBOR encode failure.
pub fn execution_digest(wire: &ExecutionIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::Execution, wire)
}

/// Bind execution identity to the context that actually ran (or will run).
#[must_use]
pub fn execution_identity_from_context(ctx: &ExecutionContext) -> ExecutionIdentityWire {
    let policy = ctx.kernel_policy;
    ExecutionIdentityWire {
        format: IDENTITY_FORMAT,
        library_version: antecedent_core::VERSION.into(),
        seed: ctx.rng.master_seed(),
        threads: ctx.parallelism.max_threads.get(),
        // The backend that runs, not the one requested: SIMD requested but not
        // compiled in is the portable path, so behaviourally identical contexts
        // share one execution identity.
        backend: if policy.force_scalar
            || !(policy.arch_simd_effective() || policy.allow_portable_optimized)
        {
            "scalar".into()
        } else if policy.arch_simd_effective() {
            "optimized".into()
        } else {
            "portable".into()
        },
        kernel: KernelPolicyWire {
            allow_portable_optimized: policy.allow_portable_optimized,
            allow_arch_simd: policy.arch_simd_effective(),
            force_scalar: policy.force_scalar,
        },
        determinism: match ctx.determinism {
            antecedent_core::Determinism::Strict => "strict".into(),
            antecedent_core::Determinism::PreferFast => "prefer_fast".into(),
        },
        adaptive_bootstrap: ctx.adaptive_bootstrap.enabled.then(|| AdaptiveBudgetWire {
            minimum: u64::from(ctx.adaptive_bootstrap.min_replicates),
            epsilon_bits: ctx.adaptive_bootstrap.se_rel_epsilon.to_bits(),
            ess_target_bits: None,
        }),
        adaptive_draws: ctx.adaptive_draws.enabled.then(|| AdaptiveBudgetWire {
            minimum: crate::error::lossless_u64(ctx.adaptive_draws.min_draws),
            epsilon_bits: ctx.adaptive_draws.quantile_width_rel_epsilon.to_bits(),
            ess_target_bits: Some(ctx.adaptive_draws.ess_target.to_bits()),
        }),
    }
}

/// Canonical prior-set identity: every spec's parameters, contrast,
/// categorical predictors, and restriction ids.
#[must_use]
pub fn prior_set_identity(prior: &antecedent_prob::PriorSet) -> PriorSetIdentityWire {
    use antecedent_prob::PriorSpec;
    PriorSetIdentityWire {
        specs: prior
            .specs
            .iter()
            .map(|spec| match spec {
                PriorSpec::GaussianCoefficients(gaussian) => {
                    PriorSpecIdentityWire::GaussianCoefficients {
                        mean: PayloadDigestWire::f64s("prior.coefficient_mean", &gaussian.mean),
                        variance: PayloadDigestWire::f64s(
                            "prior.coefficient_variance",
                            &gaussian.variance,
                        ),
                    }
                }
                PriorSpec::ResidualInvGamma(inv_gamma) => PriorSpecIdentityWire::ResidualInvGamma {
                    shape_bits: inv_gamma.shape.to_bits(),
                    scale_bits: inv_gamma.scale.to_bits(),
                },
                PriorSpec::KnownResidualVariance(variance) => {
                    PriorSpecIdentityWire::KnownResidualVariance(variance.to_bits())
                }
                PriorSpec::CoefficientCorrelation(correlation) => {
                    PriorSpecIdentityWire::CoefficientCorrelation {
                        dim: correlation.dim() as u64,
                        matrix: PayloadDigestWire::f64s(
                            "prior.coefficient_correlation",
                            correlation.matrix(),
                        ),
                    }
                }
            })
            .collect(),
        contrast: prior.contrast.map(|contrast| {
            match contrast {
                antecedent_prob::ContrastCoding::Treatment => "treatment",
                antecedent_prob::ContrastCoding::Sum => "sum",
            }
            .into()
        }),
        categorical: vars_to_raw(&prior.categorical),
        restrictions: prior
            .restrictions
            .iter()
            .map(|restriction| restriction.id.to_string())
            .collect(),
    }
}

/// External prior-bank composition inputs for the inference binding.
#[must_use]
pub fn external_compose_identity(
    sources: &[antecedent_prob::ExternalPriorSource],
    conflict_policy: Option<(f64, f64)>,
) -> ExternalComposeIdentityWire {
    ExternalComposeIdentityWire {
        sources: sources
            .iter()
            .map(|source| ExternalPriorSourceIdentityWire {
                id: source.id.to_string(),
                prior: prior_set_identity(&source.prior),
                alpha_bits: source.weight.alpha.to_bits(),
                mixture_weight_bits: source.weight.mixture_weight.map(f64::to_bits),
                ess_bits: source.ess.map(f64::to_bits),
            })
            .collect(),
        conflict_policy: conflict_policy
            .map(|(p_min, kl_scale)| (p_min.to_bits(), kl_scale.to_bits())),
        baseline: None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn imputation_domains_preserve_legacy_wire_and_identity_when_absent() {
        #[derive(serde::Serialize)]
        struct Legacy<'a> {
            model: &'a str,
            imputations: u32,
            iterations: u32,
            seed: u64,
            assumption: &'a str,
            unit: &'a str,
        }
        let old = Legacy {
            model: "chained_equations",
            imputations: 20,
            iterations: 10,
            seed: 0,
            assumption: "missing_at_random",
            unit: "row",
        };
        let mut wire = super::ImputationSpecWire {
            model: old.model.into(),
            imputations: old.imputations,
            iterations: old.iterations,
            seed: old.seed,
            assumption: old.assumption.into(),
            unit: old.unit.into(),
            bootstrap_replicates: 0,
            assumptions_digest: None,
            domains: Vec::new(),
        };
        assert_eq!(crate::to_cbor(&old).unwrap(), crate::to_cbor(&wire).unwrap());
        let old_digest =
            super::digest_wire(antecedent_core::IdentityDomain::DataSnapshot, &old).unwrap();
        assert_eq!(
            old_digest,
            super::digest_wire(antecedent_core::IdentityDomain::DataSnapshot, &wire).unwrap()
        );
        wire.domains.push(("x".into(), "binary".into()));
        let encoded = crate::to_cbor(&wire).unwrap();
        let decoded: super::ImputationSpecWire = crate::from_cbor(&encoded).unwrap();
        assert_eq!(decoded, wire);
        assert_ne!(
            old_digest,
            super::digest_wire(antecedent_core::IdentityDomain::DataSnapshot, &wire).unwrap()
        );
    }
    #[test]
    fn programmed_interventions_have_canonical_distinct_identities() {
        use antecedent_core::{Intervention, Value, ValueExpr, VariableId};
        let x = VariableId::from_raw(0);
        let z = VariableId::from_raw(1);
        let wire = |iv: Intervention| InterventionWire::from_domain(&iv).unwrap();
        let bump = ValueExpr::generated().add(ValueExpr::constant(1.0));
        let a = wire(Intervention::transform(x, bump.clone()));
        let same = wire(Intervention::transform(x, bump.clone()));
        let assigned =
            wire(Intervention::assign(x, ValueExpr::var(z).add(ValueExpr::constant(1.0))));
        let other =
            wire(Intervention::transform(x, ValueExpr::generated().add(ValueExpr::constant(2.0))));
        let moved = wire(Intervention::transform(z, bump));
        let cbor = |w: &InterventionWire| crate::convert::to_cbor(w).unwrap();
        assert_eq!(intervention_label(&a), intervention_label(&same));
        assert_eq!(cbor(&a), cbor(&same));
        for different in [&assigned, &other, &moved] {
            assert_ne!(intervention_label(&a), intervention_label(different));
            assert_ne!(cbor(&a), cbor(different));
        }
        // Transform and assign of one expression differ by kind alone.
        let e = ValueExpr::constant(3.0);
        let t = wire(Intervention::transform(x, e.clone()));
        let s = wire(Intervention::assign(x, e));
        assert_ne!(intervention_label(&t), intervention_label(&s));
        assert_ne!(cbor(&t), cbor(&s));
        // Released variants keep their labels.
        let set = wire(Intervention::set(x, Value::f64(1.0)));
        assert_eq!(intervention_label(&set), "set:0=1");
    }

    /// The released encoding of a policy-free intervention, written independently of the
    /// serde derive as a generic CBOR value: a program-free query must keep these bytes, so
    /// every identity digested over it is unchanged.
    #[test]
    fn a_policy_free_intervention_keeps_its_released_cbor_encoding() {
        use ciborium::Value;
        let text = |s: &str| Value::Text(s.into());
        let released = Value::Map(vec![(
            text("set"),
            Value::Map(vec![
                (text("variable"), Value::Integer(0.into())),
                (text("value"), Value::Map(vec![(text("float64"), Value::Float(1.0))])),
            ]),
        )]);
        let mut released_bytes = Vec::new();
        ciborium::into_writer(&released, &mut released_bytes).unwrap();
        let wire = InterventionWire::from_domain(&antecedent_core::Intervention::set(
            VariableId::from_raw(0),
            antecedent_core::Value::f64(1.0),
        ))
        .unwrap();
        assert_eq!(crate::convert::to_cbor(&wire).unwrap(), released_bytes);
        let decoded: InterventionWire = crate::convert::from_cbor(&released_bytes).unwrap();
        assert_eq!(decoded, wire);
    }

    #[test]
    fn a_policy_expression_moves_target_question_and_program_identities() {
        use antecedent_core::{Intervention, ResponseFunctional, ResponseQuery, ValueExpr};
        let (schema, _, _) = schema_and_query();
        let (t, y) = (VariableId::from_raw(0), VariableId::from_raw(1));
        let floor = |slope: f64| {
            let policy = Intervention::transform(
                t,
                ValueExpr::constant(10.0)
                    .max(ValueExpr::constant(slope).mul(ValueExpr::generated())),
            );
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: y,
                interventions: std::sync::Arc::from([policy]),
            }))
        };
        let digests = |slope: f64| {
            let target = target_wire(&schema, &floor(slope));
            let target_digest = digest_wire(IdentityDomain::Target, &target).unwrap();
            let question = digest_wire(IdentityDomain::Target, &target.question()).unwrap();
            let program = program_digest(&ProgramIdentityWire {
                format: IDENTITY_FORMAT,
                target: *target_digest.as_bytes(),
                identification: [0; 32],
                identification_product: None,
                completion_budget: None,
                commitments: InferentialCommitmentsWire {
                    format: IDENTITY_FORMAT,
                    estimator: None,
                    resolved_estimator: None,
                    identifier: None,
                    inference: "frequentist".into(),
                    validation_suite: None,
                    interval_method: "analytic_se".into(),
                    se_kind: None,
                    prior_required: false,
                },
                temporal_attribution_rows: None,
                functional_program: None,
                checked_aipw_lowering: None,
                checked_frontdoor_lowering: None,
                checked_iv_lowering: None,
                checked_linear_adjustment_lowering: None,
                checked_functional_response_grid: None,
                checked_nested_counterfactual: None,
                checked_dr_lowering: None,
                graph_posterior_programs: None,
                structure_provenance: None,
            })
            .unwrap();
            (target_digest, question, program)
        };
        let base = digests(0.85);
        assert_eq!(base, digests(0.85), "identical expressions give identical identities");
        let changed = digests(0.9);
        assert_ne!(base.0, changed.0, "target");
        assert_ne!(base.1, changed.1, "question");
        assert_ne!(base.2, changed.2, "program");
    }

    use antecedent_core::{
        AverageEffectQuery, CausalQuery, CausalSchemaBuilder, Lag, MeasurementSpec,
        RandomizedEffectQuery, RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;
    use crate::{causal_query_to_wire, from_cbor, to_cbor};
    use antecedent_discovery::SamplingSummary;

    #[test]
    fn graph_posterior_program_chain_matches_direct_framing() {
        let atoms = [
            GraphPosteriorAtomProgramWire {
                graph_key: 3,
                weight_bits: 0.25_f64.to_bits(),
                samples: vec![0, 2],
                functional_program: None,
                identification_product: None,
            },
            GraphPosteriorAtomProgramWire {
                graph_key: 7,
                weight_bits: 0.75_f64.to_bits(),
                samples: vec![1],
                functional_program: None,
                identification_product: None,
            },
        ];
        let mixing = GRAPH_POSTERIOR_MIXING_RULE;
        let reference = |items: &[GraphPosteriorAtomProgramWire]| {
            let mut hasher =
                blake3::Hasher::new_derive_key("antecedent.identity.graph_posterior_programs.v1");
            hasher.update(&to_cbor(&(mixing, u64::try_from(items.len()).unwrap())).unwrap());
            let mut link = *hasher.finalize().as_bytes();
            for atom in items {
                let mut hasher = blake3::Hasher::new_derive_key(
                    "antecedent.identity.graph_posterior_programs.v1",
                );
                hasher.update(&link);
                hasher.update(&to_cbor(atom).unwrap());
                link = *hasher.finalize().as_bytes();
            }
            link
        };
        assert_eq!(GraphPosteriorProgramsWire::compute_chain(mixing, &[]).unwrap(), reference(&[]));
        assert_eq!(
            GraphPosteriorProgramsWire::compute_chain(mixing, &atoms).unwrap(),
            reference(&atoms)
        );
        assert_ne!(
            GraphPosteriorProgramsWire::compute_chain(mixing, &atoms).unwrap(),
            GraphPosteriorProgramsWire::compute_chain(
                mixing,
                &[atoms[1].clone(), atoms[0].clone()]
            )
            .unwrap()
        );
    }

    fn schema_and_query() -> (CausalSchema, CausalQuery, Dag) {
        let mut builder = CausalSchemaBuilder::new();
        builder
            .add_variable(
                "t",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        builder
            .add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        let schema = builder.build().unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        (schema, query, dag)
    }

    fn target_wire(schema: &CausalSchema, query: &CausalQuery) -> TargetIdentityWire {
        TargetIdentityWire {
            format: IDENTITY_FORMAT,
            schema: schema_to_wire(schema),
            query: causal_query_to_wire(query).unwrap(),
        }
    }

    fn premises(graph: GraphIdentityWire) -> IdentificationIdentityWire {
        IdentificationIdentityWire {
            format: IDENTITY_FORMAT,
            target_question: [1; 32],
            population_depends_on: Vec::new(),
            rd_config: None,
            class_prior: None,
            transport: None,
            graph_class: "Dag".into(),
            structure_source: "explicit".into(),
            accepted_version: 1,
            algorithm_id: None,
            schema_names: None,
            graph,
            observation: [2; 32],
        }
    }

    #[test]
    fn identity_dictionary_covers_every_domain() {
        let text = include_str!("../../../parity/identity.toml");
        let listed: Vec<&str> = text
            .lines()
            .filter_map(|line| line.strip_prefix("domain = \""))
            .map(|line| line.trim_end_matches('"'))
            .collect();
        let domains: Vec<_> = IdentityDomain::ALL.iter().map(|domain| domain.as_str()).collect();
        assert_eq!(listed, domains);
    }

    #[test]
    fn streamed_cbor_digest_matches_the_buffered_payload() {
        let payload = vec![-0.0_f64, 1.0, f64::from_bits(0x7ff8_0000_0000_0001)];
        let streamed = digest_wire(IdentityDomain::Target, &payload).unwrap();
        let buffered = digest_canonical(IdentityDomain::Target, &to_cbor(&payload).unwrap());
        assert_eq!(streamed, buffered);
        let kind = "analysis_result.body";
        assert_eq!(
            payload_digest_of(kind, &payload).unwrap(),
            payload_digest(kind, &to_cbor(&payload).unwrap())
        );
    }

    #[test]
    fn domain_separation_changes_the_digest() {
        let payload = b"same-bytes";
        let target = digest_canonical(IdentityDomain::Target, payload);
        let program = digest_canonical(IdentityDomain::Program, payload);
        assert_ne!(target, program);
        assert_eq!(digest_canonical(IdentityDomain::Target, payload), target);
    }

    #[test]
    fn target_digest_is_stable_and_name_bound() {
        let (schema, query, _) = schema_and_query();
        let first = digest_wire(IdentityDomain::Target, &target_wire(&schema, &query)).unwrap();
        let second = digest_wire(IdentityDomain::Target, &target_wire(&schema, &query)).unwrap();
        assert_eq!(first, second);
        let mut renamed = CausalSchemaBuilder::new();
        for name in ["treatment", "outcome"] {
            renamed
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let renamed = renamed.build().unwrap();
        assert_ne!(
            digest_wire(IdentityDomain::Target, &target_wire(&renamed, &query)).unwrap(),
            first
        );
    }

    #[test]
    fn panel_did_target_identity_binds_timing_subject_and_cluster_vectors() {
        let (schema, _, _) = schema_and_query();
        let query = CausalQuery::PanelDid(antecedent_core::PanelDidQuery::new(
            VariableId::from_raw(1),
            [true, true, false, false],
            [false, true, false, true],
            ["a", "a", "b", "b"].map(std::sync::Arc::<str>::from),
            ["x", "x", "y", "y"].map(std::sync::Arc::<str>::from),
        ));
        let first = digest_wire(IdentityDomain::Target, &target_wire(&schema, &query)).unwrap();
        let CausalQuery::PanelDid(mut changed) = query.clone() else { unreachable!() };
        changed.post = [false, false, false, true].into();
        assert_ne!(
            first,
            digest_wire(
                IdentityDomain::Target,
                &target_wire(&schema, &CausalQuery::PanelDid(changed.clone()))
            )
            .unwrap()
        );
        changed.clusters = ["x", "x", "z", "z"].map(std::sync::Arc::<str>::from).into();
        assert_ne!(
            first,
            digest_wire(
                IdentityDomain::Target,
                &target_wire(&schema, &CausalQuery::PanelDid(changed))
            )
            .unwrap()
        );
        let labels: std::collections::HashMap<_, _> =
            executed_functional_labels(&causal_query_to_wire(&query).unwrap())
                .into_iter()
                .collect();
        assert_eq!(labels.get("query_kind").map(String::as_str), Some("panel_did"));
        assert_eq!(labels.get("design").map(String::as_str), Some("balanced_two_period_panel"));
    }

    #[test]
    fn repeated_cross_section_design_changes_target_identity() {
        let (schema, _, _) = schema_and_query();
        let panel = antecedent_core::PanelDidQuery::new(
            VariableId::from_raw(1),
            [true, true, false, false],
            [false, true, false, true],
            ["a", "b", "c", "d"].map(std::sync::Arc::<str>::from),
            ["x", "y", "z", "w"].map(std::sync::Arc::<str>::from),
        );
        let repeated = antecedent_core::PanelDidQuery::repeated_cross_section(
            panel.outcome,
            panel.treated.clone(),
            panel.post.clone(),
            panel.subjects.clone(),
            panel.clusters.clone(),
        );
        let panel_query = CausalQuery::PanelDid(panel);
        let repeated_query = CausalQuery::PanelDid(repeated);
        assert_ne!(
            digest_wire(IdentityDomain::Target, &target_wire(&schema, &panel_query)).unwrap(),
            digest_wire(IdentityDomain::Target, &target_wire(&schema, &repeated_query)).unwrap(),
        );
        let labels: std::collections::HashMap<_, _> =
            executed_functional_labels(&causal_query_to_wire(&repeated_query).unwrap())
                .into_iter()
                .collect();
        assert_eq!(labels.get("design").map(String::as_str), Some("repeated_cross_section"));
    }

    #[test]
    fn staggered_panel_did_labels_name_their_design_and_target() {
        let subjects = ["a", "a", "b", "b"].map(std::sync::Arc::<str>::from);
        let clusters = subjects.clone();
        let periods = [1_i64, 2, 1, 2];
        let cohorts = [2_i64, 2, 0, 0];
        let labels = |query: antecedent_core::PanelDidQuery| {
            executed_functional_labels(
                &causal_query_to_wire(&CausalQuery::PanelDid(query)).unwrap(),
            )
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>()
        };
        let group_time = labels(antecedent_core::PanelDidQuery::staggered_group_time(
            VariableId::from_raw(1),
            subjects.clone(),
            clusters.clone(),
            periods,
            cohorts,
            2,
            2,
        ));
        assert_eq!(group_time.get("design").map(String::as_str), Some("staggered_group_time"));
        assert_eq!(
            group_time.get("staggered_target").map(String::as_str),
            Some("cohort=2,period=2")
        );
        let event_study = labels(antecedent_core::PanelDidQuery::staggered_event_study(
            VariableId::from_raw(1),
            subjects,
            clusters,
            periods,
            cohorts,
        ));
        assert_eq!(event_study.get("design").map(String::as_str), Some("staggered_event_study"));
        assert_eq!(event_study.get("staggered_target"), None);
    }

    #[test]
    fn question_projection_drops_only_the_population() {
        let (schema, query, _) = schema_and_query();
        let CausalQuery::AverageEffect(ate) = &query else { unreachable!() };
        let treated = CausalQuery::AverageEffect(
            ate.clone().with_target_population(TargetPopulation::Treated),
        );
        let ate_target = target_wire(&schema, &query);
        let att_target = target_wire(&schema, &treated);
        let digest = |wire: &TargetIdentityWire| digest_wire(IdentityDomain::Target, wire).unwrap();
        assert_ne!(digest(&ate_target), digest(&att_target));
        assert_eq!(digest(&ate_target.question()), digest(&att_target.question()));
        assert_eq!(digest(&ate_target.question()), digest(&ate_target));
    }

    #[test]
    fn different_graphs_change_identification_not_target() {
        let (schema, query, dag) = schema_and_query();
        let target = digest_wire(IdentityDomain::Target, &target_wire(&schema, &query)).unwrap();
        let first = IdentificationIdentityWire {
            target_question: *target.as_bytes(),
            ..premises(dag_identity(&dag).unwrap())
        };
        let mut other = Dag::with_variables(2);
        other.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(0)).unwrap();
        let second =
            IdentificationIdentityWire { graph: dag_identity(&other).unwrap(), ..first.clone() };
        assert_eq!(identification_digest(&first).unwrap(), identification_digest(&first).unwrap());
        assert_ne!(identification_digest(&first).unwrap(), identification_digest(&second).unwrap());
        assert_eq!(first.target_question, second.target_question);
        let reviewed = IdentificationIdentityWire {
            structure_source: "accepted".into(),
            accepted_version: 9,
            algorithm_id: Some("pc".into()),
            ..first.clone()
        };
        assert_eq!(
            identification_digest(&first).unwrap(),
            identification_digest(&reviewed).unwrap()
        );
        let decoded: IdentificationIdentityWire = from_cbor(&to_cbor(&reviewed).unwrap()).unwrap();
        assert_eq!(decoded, reviewed, "audit metadata must survive serialization");
        for changed in [
            IdentificationIdentityWire { target_question: [9; 32], ..first.clone() },
            IdentificationIdentityWire { observation: [9; 32], ..first.clone() },
            IdentificationIdentityWire { graph_class: "Cpdag".into(), ..first.clone() },
            IdentificationIdentityWire {
                schema_names: Some(vec!["y".into(), "x".into()]),
                ..first.clone()
            },
            IdentificationIdentityWire {
                class_prior: Some(ClassPriorIdentityWire::Ordered(vec![0.5f64.to_bits()])),
                ..first.clone()
            },
            IdentificationIdentityWire {
                transport: Some(TransportIdentityWire {
                    selection_graph: None,
                    selection_targets: vec![1],
                    trial: None,
                    selection_probability: None,
                    treatment_probability: None,
                }),
                ..first.clone()
            },
        ] {
            assert_ne!(
                identification_digest(&first).unwrap(),
                identification_digest(&changed).unwrap()
            );
        }
    }

    #[test]
    fn graph_identity_ignores_edge_insertion_order() {
        let (a, b, c) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        let mut first = Dag::with_variables(3);
        first.insert_directed(c, a).unwrap();
        first.insert_directed(c, b).unwrap();
        first.insert_directed(a, b).unwrap();
        let mut second = Dag::with_variables(3);
        second.insert_directed(a, b).unwrap();
        second.insert_directed(c, b).unwrap();
        second.insert_directed(c, a).unwrap();
        assert_ne!(dag_to_wire(&first).unwrap().edges, dag_to_wire(&second).unwrap().edges);
        assert_eq!(dag_identity(&first).unwrap(), dag_identity(&second).unwrap());
        assert_eq!(
            identification_digest(&premises(dag_identity(&first).unwrap())).unwrap(),
            identification_digest(&premises(dag_identity(&second).unwrap())).unwrap()
        );

        let mut admg_first = Admg::with_variables(3);
        admg_first.insert_directed(c, b).unwrap();
        admg_first.insert_directed(a, b).unwrap();
        admg_first.insert_bidirected(c, a).unwrap();
        let mut admg_second = Admg::with_variables(3);
        admg_second.insert_bidirected(a, c).unwrap();
        admg_second.insert_directed(a, b).unwrap();
        admg_second.insert_directed(c, b).unwrap();
        assert_eq!(admg_identity(&admg_first).unwrap(), admg_identity(&admg_second).unwrap());

        let first_var = VariableId::from_raw(0);
        let second_var = VariableId::from_raw(1);
        let build = |order: [(VariableId, u32); 3]| {
            let mut graph = TemporalDag::empty();
            let mut ids = std::collections::HashMap::new();
            for (variable, lag) in order {
                ids.insert(
                    (variable.raw(), lag),
                    graph.add_lagged(variable, Lag::from_raw(lag)).unwrap(),
                );
            }
            graph.insert_directed(ids[&(0, 1)], ids[&(0, 0)]).unwrap();
            graph.insert_directed(ids[&(0, 0)], ids[&(1, 0)]).unwrap();
            graph
        };
        let temporal_first = build([(first_var, 1), (first_var, 0), (second_var, 0)]);
        let temporal_second = build([(second_var, 0), (first_var, 0), (first_var, 1)]);
        assert_ne!(
            temporal_dag_to_wire(&temporal_first).unwrap(),
            temporal_dag_to_wire(&temporal_second).unwrap()
        );
        assert_eq!(
            temporal_dag_identity(&temporal_first).unwrap(),
            temporal_dag_identity(&temporal_second).unwrap()
        );
    }

    #[test]
    fn temporal_class_identity_ignores_node_insertion_order() {
        let build_pag = |order: [(u32, u32); 3]| {
            let mut graph = TemporalPag::empty();
            let mut ids = std::collections::HashMap::new();
            for (variable, lag) in order {
                ids.insert(
                    (variable, lag),
                    graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap(),
                );
            }
            graph.insert_directed(ids[&(0, 1)], ids[&(0, 0)]).unwrap();
            graph
                .insert_circle_circle_with_middle(
                    ids[&(0, 0)],
                    ids[&(1, 0)],
                    antecedent_graph::MiddleMark::Empty,
                )
                .unwrap();
            graph
        };
        let first = build_pag([(0, 1), (0, 0), (1, 0)]);
        let second = build_pag([(1, 0), (0, 0), (0, 1)]);
        assert_ne!(first.nodes(), second.nodes());
        assert_eq!(temporal_pag_identity(&first).unwrap(), temporal_pag_identity(&second).unwrap());

        let build_cpdag = |order: [(u32, u32); 3]| {
            let mut graph = TemporalCpdag::empty();
            let mut ids = std::collections::HashMap::new();
            for (variable, lag) in order {
                ids.insert(
                    (variable, lag),
                    graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap(),
                );
            }
            graph.insert_directed(ids[&(0, 1)], ids[&(0, 0)]).unwrap();
            graph.insert_undirected(ids[&(0, 0)], ids[&(1, 0)]).unwrap();
            graph
        };
        assert_eq!(
            temporal_cpdag_identity(&build_cpdag([(0, 1), (0, 0), (1, 0)])).unwrap(),
            temporal_cpdag_identity(&build_cpdag([(1, 0), (0, 0), (0, 1)])).unwrap()
        );
        assert_ne!(
            temporal_cpdag_identity(&build_cpdag([(0, 1), (0, 0), (1, 0)])).unwrap(),
            temporal_pag_identity(&first).unwrap()
        );
    }

    #[test]
    fn a_temporal_class_that_repeats_a_node_digests_as_the_merged_graph() {
        let node = |graph: &mut TemporalPag, variable: u32, lag: u32| {
            graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
        };
        let mut repeated = TemporalPag::empty();
        let (t_a, z) = (node(&mut repeated, 0, 1), node(&mut repeated, 2, 1));
        let (t_b, y) = (node(&mut repeated, 0, 1), node(&mut repeated, 1, 0));
        repeated
            .insert_circle_circle_with_middle(t_a, z, antecedent_graph::MiddleMark::Empty)
            .unwrap();
        repeated.insert_directed(t_b, y).unwrap();
        let mut merged = TemporalPag::empty();
        let (t, y) = (node(&mut merged, 0, 1), node(&mut merged, 1, 0));
        let z = node(&mut merged, 2, 1);
        merged.insert_directed(t, y).unwrap();
        merged.insert_circle_circle_with_middle(t, z, antecedent_graph::MiddleMark::Empty).unwrap();
        assert_eq!(
            temporal_pag_identity(&repeated).unwrap(),
            temporal_pag_identity(&merged).unwrap()
        );
    }

    #[test]
    fn content_snapshot_round_trip_retains_partition_bindings() {
        let original = DataSnapshotIdentityWire {
            format: IDENTITY_FORMAT,
            observation: [1; 32],
            modality: "multi_env".into(),
            regularity: None,
            row_count: 6,
            unit_count: Some(2),
            partitions: vec![
                DataPartitionIdentityWire {
                    content: [2; 32],
                    row_count: 3,
                    unit_id: None,
                    regularity: Some("regular:1".into()),
                },
                DataPartitionIdentityWire {
                    content: [3; 32],
                    row_count: 3,
                    unit_id: None,
                    regularity: Some("regular:2".into()),
                },
            ],
            interference: None,
            distribution_factor_laws: None,
            nested_counterfactual_fit: None,
            linear_fit_moments: None,
            cross_product_moments: None,
            participants: None,
            missing: None,
            sampling_design: None,
        };
        let decoded: DataSnapshotIdentityWire = from_cbor(&to_cbor(&original).unwrap()).unwrap();
        assert_eq!(original, decoded);
        assert_eq!(
            data_snapshot_digest(&original).unwrap(),
            data_snapshot_digest(&decoded).unwrap()
        );
        let mut reordered = decoded.clone();
        reordered.partitions.reverse();
        assert_ne!(
            data_snapshot_digest(&original).unwrap(),
            data_snapshot_digest(&reordered).unwrap()
        );
        let mut changed_content = decoded.clone();
        changed_content.partitions[1].content = [4; 32];
        assert_ne!(
            data_snapshot_digest(&original).unwrap(),
            data_snapshot_digest(&changed_content).unwrap()
        );
        let mut networked = decoded;
        networked.interference = Some(InterferenceSnapshotWire {
            units: networked.partitions[0].clone(),
            edges: PayloadDigestWire::bytes("interference.edges", &[1]),
            assignment: PayloadDigestWire::bytes("interference.assignment", &[0, 1]),
        });
        let mut reassigned = networked.clone();
        reassigned.interference.as_mut().unwrap().assignment =
            PayloadDigestWire::bytes("interference.assignment", &[1, 0]);
        assert_ne!(
            data_snapshot_digest(&networked).unwrap(),
            data_snapshot_digest(&reassigned).unwrap()
        );
    }

    #[test]
    fn derive_key_helpers_reproduce_the_streamed_idioms() {
        let payload = b"payload bytes";
        let mut hasher = blake3::Hasher::new_derive_key(IdentityDomain::Target.derive_key());
        hasher.update(&IDENTITY_FORMAT.to_le_bytes());
        hasher.update(payload);
        assert_eq!(
            digest_canonical(IdentityDomain::Target, payload),
            SemanticDigest::from_bytes(*hasher.finalize().as_bytes())
        );

        let mut hasher = blake3::Hasher::new_derive_key("antecedent.identity.payload");
        hasher.update(&IDENTITY_FORMAT.to_le_bytes());
        hasher.update(&(4_u64).to_le_bytes());
        hasher.update(b"kind");
        hasher.update(payload);
        assert_eq!(payload_digest("kind", payload), *hasher.finalize().as_bytes());
    }

    #[test]
    fn encoding_rule_changes_change_the_advertised_digest() {
        let (schema, query, _) = schema_and_query();
        let canonical = target_wire(&schema, &query);
        let advertised = digest_wire(IdentityDomain::Target, &canonical).unwrap();
        let mut bumped = canonical.clone();
        bumped.format = IDENTITY_FORMAT + 1;
        assert_ne!(digest_wire(IdentityDomain::Target, &bumped).unwrap(), advertised);
        let payload = to_cbor(&canonical).unwrap();
        let mut hasher = blake3::Hasher::new_derive_key(IdentityDomain::Target.derive_key());
        hasher.update(&(IDENTITY_FORMAT + 1).to_le_bytes());
        hasher.update(&payload);
        assert_ne!(SemanticDigest::from_bytes(*hasher.finalize().as_bytes()), advertised);

        let observation = |tags: &[&str]| {
            digest_wire(
                IdentityDomain::Observation,
                &observation_identity_wire(&schema, tags.iter().copied()),
            )
            .unwrap()
        };
        assert_eq!(observation(&["b", "a"]), observation(&["a", "b", "a"]));
        let unsorted = ObservationIdentityWire {
            format: IDENTITY_FORMAT,
            schema: schema_to_wire(&schema),
            observation: vec!["b".into(), "a".into()],
        };
        assert_ne!(
            digest_wire(IdentityDomain::Observation, &unsorted).unwrap(),
            observation(&["a", "b"])
        );

        let mut binding = InferenceBindingWire {
            format: IDENTITY_FORMAT,
            inference: "bayesian".into(),
            bootstrap_replicates: 0,
            bayesian: Some(BayesianBindingWire {
                backend: "laplace".into(),
                likelihood: "gaussian_identity".into(),
                n_draws: 64,
                prior_scale_bits: Some(1.0f64.to_bits()),
                prior: None,
                prior_artifact: None,
                prior_source_snapshot: None,
                prior_mapping: None,
                external_compose: None,
                draw_controls: None,
                design_model: None,
                white_innovation: None,
                predictive_replicates: None,
            }),
            validation_suite: None,
            overlap_policy: None,
            estimator_spec: None,
            response_options: None,
            observation_options: ObservationOptionsWire {
                selected_correction: "aipw".into(),
                observation_probability_floor_bits: 0.01f64.to_bits(),
                censoring_survival_floor_bits: 0.01f64.to_bits(),
                crossfit_folds: 5,
            },
            split: None,
            joint_horizons: false,
        };
        let plus_zero = inference_binding_digest(&binding).unwrap();
        binding.bayesian.as_mut().unwrap().prior_scale_bits = Some((-0.0f64).to_bits());
        let minus_zero = inference_binding_digest(&binding).unwrap();
        assert_ne!(minus_zero, plus_zero);
        binding.bayesian.as_mut().unwrap().backend = "conjugate_gaussian".into();
        assert_ne!(inference_binding_digest(&binding).unwrap(), minus_zero);

        // Draw controls enter the identity only when requested: absent, the field is not
        // encoded, so a binding without them keeps the bytes it had before they existed.
        let uncontrolled = inference_binding_digest(&binding).unwrap();
        let uncontrolled_bytes = to_cbor(&binding.bayesian).unwrap();
        let encoded = serde_json::to_value(binding.bayesian.as_ref().unwrap()).unwrap();
        assert!(encoded.get("draw_controls").is_none());
        binding.bayesian.as_mut().unwrap().draw_controls = Some("measured".into());
        assert_ne!(inference_binding_digest(&binding).unwrap(), uncontrolled);
        assert_ne!(to_cbor(&binding.bayesian).unwrap(), uncontrolled_bytes);
    }

    #[test]
    fn named_prior_pairs_are_a_tuple_set_without_separator_collisions() {
        let named = |pairs: &[(&str, &str)]| {
            PriorMappingIdentityWire::from_mapping(&crate::PriorMapping::NamedParameters {
                pairs: pairs.iter().map(|(a, b)| ((*a).to_string(), (*b).to_string())).collect(),
            })
        };
        // `a -> "b->c"` and `"a->b" -> c` joined as text were both `a->b->c`.
        assert_ne!(
            to_cbor(&named(&[("a", "b->c")])).unwrap(),
            to_cbor(&named(&[("a->b", "c")])).unwrap()
        );
        assert_eq!(named(&[("x", "y"), ("a", "b")]), named(&[("a", "b"), ("x", "y"), ("a", "b")]));
    }

    #[test]
    fn product_hashes_derivation_rules_not_detail_prose() {
        let product = |details: [&str; 2], rules: [&str; 2]| {
            let mut derivation = antecedent_identify::DerivationTrace::default();
            for (rule, detail) in rules.into_iter().zip(details) {
                derivation.push(rule, detail);
            }
            let result = IdentificationResult::not_identified(
                CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                )),
                derivation,
                antecedent_core::AssumptionSet::default(),
                antecedent_identify::IdentificationPerformanceRecord::default(),
            );
            identification_product_digest(&result, false).unwrap()
        };
        let base = product(["backdoor: identified (1 estimand(s))", "ok"], ["auto.method", "done"]);
        assert_eq!(
            base,
            product(
                ["backdoor: identified (Debug text renamed)", "reworded"],
                ["auto.method", "done"]
            )
        );
        assert_ne!(base, product(["same", "same"], ["auto.method", "auto.stochastic"]));
    }

    #[test]
    fn executed_functional_labels_are_stable_tags() {
        let (_, query, _) = schema_and_query();
        let wire = causal_query_to_wire(&query).unwrap();
        let labels: std::collections::HashMap<_, _> =
            executed_functional_labels(&wire).into_iter().collect();
        assert_eq!(labels.get("query_kind").map(String::as_str), Some("average_effect"));
        assert_eq!(labels.get("treatment").map(String::as_str), Some("0"));
        assert_eq!(labels.get("outcome").map(String::as_str), Some("1"));
        assert_eq!(labels.get("control").map(String::as_str), Some("set:0=0"));
        assert_eq!(labels.get("active").map(String::as_str), Some("set:0=1"));
        assert_eq!(labels.get("population").map(String::as_str), Some("all_observed"));
        assert_eq!(labels.get("temporal_coordinates").map(String::as_str), Some("none"));
    }

    #[test]
    fn randomized_itt_identity_labels_have_no_fabricated_treatment() {
        let query = CausalQuery::RandomizedEffect(RandomizedEffectQuery::bernoulli_itt(
            VariableId::from_raw(1),
            [true, false],
            [0.5, 0.5],
            ["u1", "u2"].map(std::sync::Arc::<str>::from),
            ["y1", "y2"].map(std::sync::Arc::<str>::from),
            ("control", "treated"),
        ));
        let wire = causal_query_to_wire(&query).unwrap();
        let labels: std::collections::HashMap<_, _> =
            executed_functional_labels(&wire).into_iter().collect();
        assert_eq!(labels.get("query_kind").map(String::as_str), Some("randomized_effect"));
        assert_eq!(labels.get("outcome").map(String::as_str), Some("1"));
        assert!(!labels.contains_key("treatment"));
    }

    #[test]
    fn nested_counterfactual_has_distinct_functional_identity() {
        let query = CausalQuery::NestedCounterfactual(
            antecedent_core::NestedCounterfactualQuery::with_levels(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                VariableId::from_raw(2),
                0.0,
                1.0,
            )
            .unwrap(),
        );
        let wire = causal_query_to_wire(&query).unwrap();
        let labels: std::collections::HashMap<_, _> =
            executed_functional_labels(&wire).into_iter().collect();
        assert_eq!(labels.get("query_kind").map(String::as_str), Some("nested_counterfactual"));
        assert_eq!(
            labels.get("estimand").map(String::as_str),
            Some("natural_direct_shared_exogenous")
        );
        assert_eq!(labels.get("mediator").map(String::as_str), Some("1"));
        assert_eq!(labels.get("control").map(String::as_str), Some("0"));
        assert_eq!(labels.get("active").map(String::as_str), Some("1"));
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "the helper retains the error or fixture callback signature"
    )]
    fn static_posterior(weights: [f64; 3], graphs: [AdjacencyMask; 3]) -> GraphPosterior {
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        GraphPosterior::from_atoms(
            3,
            weights.to_vec(),
            graphs.to_vec(),
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0 / sum_sq,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("static"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
    }

    fn posterior_premises(posterior: &GraphPosterior) -> SemanticDigest {
        identification_digest(&premises(GraphIdentityWire::GraphPosterior {
            graph_class: "Dag".into(),
            n_atoms: posterior.n_graphs() as u64,
            atoms: graph_posterior_atom_identities(posterior).unwrap(),
        }))
        .unwrap()
    }

    #[test]
    fn static_posterior_identity_binds_atom_graphs_and_weights() {
        use antecedent_discovery::adjacency_mask::set_edge;
        let direct = set_edge(0, 3, 0, 1, true);
        let adjusted = set_edge(set_edge(&direct, 3, 2, 0, true), 3, 2, 1, true);
        let reversed = set_edge(0, 3, 1, 0, true);
        let confounded = set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true);
        let base =
            static_posterior([0.5, 0.3, 0.2], [direct.clone(), adjusted.clone(), reversed.clone()]);
        let reweighted =
            static_posterior([0.2, 0.3, 0.5], [direct.clone(), adjusted.clone(), reversed.clone()]);
        let regraphed =
            static_posterior([0.5, 0.3, 0.2], [direct.clone(), adjusted.clone(), confounded]);
        let reordered = static_posterior([0.2, 0.5, 0.3], [reversed, direct, adjusted]);
        assert_ne!(posterior_premises(&base), posterior_premises(&reweighted));
        assert_ne!(posterior_premises(&base), posterior_premises(&regraphed));
        // A weighted set: atom order and position-derived keys are not identity.
        assert_eq!(posterior_premises(&base), posterior_premises(&reordered));
        let atoms = graph_posterior_atom_identities(&base).unwrap();
        assert_eq!(atoms.iter().map(|atom| atom.execution_key).collect::<Vec<_>>(), [0, 1, 2]);
        assert!(matches!(atoms[0].graph, PosteriorAtomGraphWire::Static(_)));
    }

    /// The identity of an atom is its canonical graph, not its packing: a stored single-word
    /// adjacency, the same graph built through the wide edge set, and the canonical wire agree,
    /// and a twelve-variable atom (edge bits past the first word) encodes as its twelve nodes.
    #[test]
    fn static_posterior_atom_identity_is_independent_of_the_adjacency_packing() {
        use antecedent_discovery::adjacency_mask::set_edge;
        use antecedent_discovery::edge_bit;
        let n = 3;
        let stored: u64 = (1 << edge_bit(n, 0, 1)) | (1 << edge_bit(n, 2, 0));
        let from_word = AdjacencyMask::from_word(stored);
        let built = set_edge(set_edge(AdjacencyMask::EMPTY, n, 0, 1, true), n, 2, 0, true);
        assert_eq!(from_word, built);
        let empty = AdjacencyMask::EMPTY;
        let legacy = static_posterior([0.5, 0.3, 0.2], [from_word, empty.clone(), empty.clone()]);
        let wide = static_posterior([0.5, 0.3, 0.2], [built, empty.clone(), empty]);
        assert_eq!(posterior_premises(&legacy), posterior_premises(&wide));
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        let atoms = graph_posterior_atom_identities(&legacy).unwrap();
        assert_eq!(
            atoms[0].graph,
            PosteriorAtomGraphWire::Static(canonical_dag_wire(&dag).unwrap())
        );

        let n = 12;
        let chain =
            (0..n - 1).fold(AdjacencyMask::EMPTY, |mask, i| set_edge(mask, n, i, i + 1, true));
        assert_eq!(chain.to_word(), None, "twelve variables need more than one word");
        let posterior = GraphPosterior::from_atoms(
            n,
            vec![1.0],
            vec![chain],
            vec![0.0; n * n],
            vec![0.0; n * n],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("wide"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let atoms = graph_posterior_atom_identities(&posterior).unwrap();
        let PosteriorAtomGraphWire::Static(wire) = &atoms[0].graph else {
            panic!("a twelve-variable DAG atom encodes as a static graph");
        };
        assert_eq!(wire.node_count, 12);
        let expected: Vec<(u32, u32)> = (0..11).map(|i| (i, i + 1)).collect();
        assert_eq!(wire.edges, expected);
    }

    /// A posterior of `n` variables whose atoms are `graphs`, equally weighted.
    fn equal_weight_posterior(n: usize, graphs: Vec<AdjacencyMask>) -> GraphPosterior {
        let weight = 1.0 / graphs.len() as f64;
        GraphPosterior::from_atoms(
            n,
            vec![weight; graphs.len()],
            graphs,
            vec![0.0; n * n],
            vec![0.0; n * n],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("packing"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
    }

    /// Posteriors of eight variables or fewer keep the single-word packing every earlier
    /// artifact stored: the stored word, its key and the canonical graph an atom identifies as
    /// are the ones the layout `from * (n - 1) + (to, less one past the diagonal)` defines.
    /// A ninth variable needs edge bits past the first word, and such atoms carry the wide
    /// encoding and still identify as their canonical graph.
    #[test]
    fn eight_variable_atoms_keep_the_stored_word_and_wider_atoms_carry_wide_encodings() {
        use antecedent_discovery::adjacency_mask::set_edge;
        // The layout written out independently of the library's `edge_bit`.
        let bit =
            |n: usize, from: usize, to: usize| from * (n - 1) + if to < from { to } else { to - 1 };
        let eight = 8;
        // The edge 7 -> 6 sits at bit 55, the last of an eight-variable word.
        let edges: [(usize, usize); 4] = [(0, 1), (3, 4), (7, 6), (2, 0)];
        let stored: u64 =
            edges.iter().fold(0, |word, &(from, to)| word | (1 << bit(eight, from, to)));
        assert!((stored >> 55) & 1 == 1 && stored < 1 << 56);
        let mask = AdjacencyMask::from_word(stored);
        let built = edges
            .iter()
            .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, eight, from, to, true));
        assert_eq!(mask, built);
        assert_eq!(mask.to_word(), Some(stored));
        assert_eq!(mask.key(), stored, "an eight-variable atom keeps its word as its key");
        let posterior = equal_weight_posterior(eight, vec![mask.clone(), AdjacencyMask::EMPTY]);
        assert_eq!(posterior.graph_keys()[0], stored);
        let atoms = graph_posterior_atom_identities(&posterior).unwrap();
        let mut dag = Dag::with_variables(8);
        for &(from, to) in &edges {
            dag.insert_directed(
                DenseNodeId::from_raw(u32::try_from(from).unwrap()),
                DenseNodeId::from_raw(u32::try_from(to).unwrap()),
            )
            .unwrap();
        }
        assert_eq!(
            atoms[0].graph,
            PosteriorAtomGraphWire::Static(canonical_dag_wire(&dag).unwrap())
        );

        // Nine variables: the edge 8 -> 7 is at bit 71.
        let nine = 9;
        let wide_edges: [(usize, usize); 3] = [(0, 1), (8, 7), (4, 2)];
        let wide = wide_edges
            .iter()
            .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, nine, from, to, true));
        assert_eq!(wide.to_word(), None, "bit {} is past the first word", bit(nine, 8, 7));
        assert_eq!(bit(nine, 8, 7), 71);
        assert_ne!(wide.key(), wide.words()[0], "a wide atom's key is a hash of every word");
        let posterior = equal_weight_posterior(nine, vec![wide.clone(), AdjacencyMask::EMPTY]);
        let atoms = graph_posterior_atom_identities(&posterior).unwrap();
        let PosteriorAtomGraphWire::Static(wire) = &atoms[0].graph else {
            panic!("a nine-variable DAG atom encodes as a static graph");
        };
        assert_eq!(wire.node_count, 9);
        let mut expected: Vec<(u32, u32)> = wide_edges
            .iter()
            .map(|&(from, to)| (u32::try_from(from).unwrap(), u32::try_from(to).unwrap()))
            .collect();
        expected.sort_unstable();
        assert_eq!(wire.edges, expected);
        // The premises bind the wide edge: without it the posterior is another posterior.
        let without =
            equal_weight_posterior(nine, vec![AdjacencyMask::EMPTY, AdjacencyMask::EMPTY]);
        assert_ne!(posterior_premises(&posterior), posterior_premises(&without));
    }

    #[test]
    fn dbn_atom_identity_includes_lags_and_ignores_execution_key() {
        use antecedent_discovery::adjacency_mask::set_edge;

        let contemporaneous = set_edge(0, 3, 2, 0, true);
        // Packing is `(lag-1)*n² + from*n + to` for lag=1, n=3.
        let confounder_to_outcome = 1_u64 << 7;
        let treatment_to_outcome = 1_u64 << 1;
        let posterior = GraphPosterior::from_atoms(
            3,
            vec![0.5, 0.5],
            vec![contemporaneous.clone(), contemporaneous],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 2.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("dbn_atom"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_lagged_marginals(1, vec![0.0; 9])
        .unwrap()
        .with_lag_masks(vec![confounder_to_outcome, treatment_to_outcome])
        .unwrap();
        assert_eq!(posterior.graph_keys()[0], posterior.graph_keys()[1]);
        let atoms = graph_posterior_atom_identities(&posterior).unwrap();
        assert_eq!(atoms.len(), 2);
        assert!(matches!(atoms[0].graph, PosteriorAtomGraphWire::Temporal(_)));
        assert_ne!(atoms[0].graph, atoms[1].graph, "lag edges distinguish the atoms");
        let renumbered = GraphIdentityWire::GraphPosterior {
            graph_class: "TemporalDag".into(),
            n_atoms: 2,
            atoms: atoms
                .iter()
                .map(|atom| PosteriorAtomIdentityWire { execution_key: 77, ..atom.clone() })
                .collect(),
        };
        let original = GraphIdentityWire::GraphPosterior {
            graph_class: "TemporalDag".into(),
            n_atoms: 2,
            atoms,
        };
        assert_ne!(original, renumbered, "execution keys survive in the section");
        assert_eq!(
            identification_digest(&premises(original)).unwrap(),
            identification_digest(&premises(renumbered)).unwrap()
        );
    }

    #[test]
    fn score_reuse_keys_are_stricter_than_identification() {
        let identification = SemanticDigest::from_bytes([1; 32]);
        let snapshot = SemanticDigest::from_bytes([2; 32]);
        let inference = SemanticDigest::from_bytes([4; 32]);
        let treatment = VariableId::from_raw(0);
        let z = VariableId::from_raw(2);
        let table = ScoreReuseIdentityWire::score_table(&ScoreTableKey {
            identification,
            data_snapshot: snapshot,
            row_index: &[0, 1, 2, 3],
            fold_ids: &[0, 1, 0, 1],
            n_folds: 2,
            adjustment_set: &[z],
            nuisance_provenance: "crossfit.aipw",
            treatment,
            intervened: &[],
            bootstrap_replicates: 0,
            inference_binding: inference,
        });
        let share = ScoreReuseIdentityWire::batch_share(snapshot, &[0, 1, 0, 1], 2, &[z]);
        let table_digest = score_reuse_digest(&table).unwrap();
        let share_digest = score_reuse_digest(&share).unwrap();
        assert_ne!(table_digest, share_digest);
        assert_eq!(score_reuse_digest(&table).unwrap(), table_digest);

        let other_folds = ScoreReuseIdentityWire::score_table(&ScoreTableKey {
            identification,
            data_snapshot: snapshot,
            row_index: &[0, 1, 2, 3],
            fold_ids: &[1, 0, 1, 0],
            n_folds: 2,
            adjustment_set: &[z],
            nuisance_provenance: "crossfit.aipw",
            treatment,
            intervened: &[],
            bootstrap_replicates: 0,
            inference_binding: inference,
        });
        assert_ne!(score_reuse_digest(&other_folds).unwrap(), table_digest);
        let other_rows = ScoreReuseIdentityWire::score_table(&ScoreTableKey {
            identification,
            data_snapshot: snapshot,
            row_index: &[0, 1, 2, 4],
            fold_ids: &[0, 1, 0, 1],
            n_folds: 2,
            adjustment_set: &[z],
            nuisance_provenance: "crossfit.aipw",
            treatment,
            intervened: &[],
            bootstrap_replicates: 0,
            inference_binding: inference,
        });
        assert_ne!(score_reuse_digest(&other_rows).unwrap(), table_digest);

        let mut other_snapshot = table.clone();
        other_snapshot.data_snapshot = [3; 32];
        assert_ne!(score_reuse_digest(&other_snapshot).unwrap(), table_digest);

        // Same folds, rows and tag under another estimator / GLM / overlap configuration
        // produce different scores, so they must not share a key.
        let mut other_inference = table.clone();
        other_inference.inference_binding = Some([5; 32]);
        assert_ne!(score_reuse_digest(&other_inference).unwrap(), table_digest);

        let mut other_nuisance = table;
        other_nuisance.nuisance_provenance = Some("crossfit.cell_aipw".into());
        assert_ne!(score_reuse_digest(&other_nuisance).unwrap(), table_digest);
    }

    #[test]
    fn payload_digests_separate_kinds() {
        assert_ne!(payload_digest("a", b"xy"), payload_digest("ax", b"y"));
        assert_ne!(
            PayloadDigestWire::u32s("estimator.cluster_ids", &[1, 2]),
            PayloadDigestWire::u32s("estimator.cluster_ids", &[2, 1])
        );
    }

    #[test]
    fn target_weights_bind_bits_rows_snapshot_score_and_parents() {
        let snapshot = SemanticDigest::from_bytes([2; 32]);
        let score = SemanticDigest::from_bytes([4; 32]);
        let z = [VariableId::from_raw(2)];
        let base = TargetWeightsIdentityWire::new(&[1.0, 2.0, 3.0], snapshot, score, &z);
        let digest = target_weights_digest(&base).unwrap();
        assert_eq!(base.row_count, 3);
        assert_eq!(
            target_weights_digest(&TargetWeightsIdentityWire::new(
                &[1.0, 2.0, 3.0],
                snapshot,
                score,
                &z
            ))
            .unwrap(),
            digest
        );
        let variants = [
            TargetWeightsIdentityWire::new(&[7.0, 14.0, 21.0], snapshot, score, &z),
            TargetWeightsIdentityWire::new(&[1.0, 2.0, 3.0, 0.0], snapshot, score, &z),
            TargetWeightsIdentityWire::new(
                &[1.0, 2.0, 3.0],
                SemanticDigest::from_bytes([3; 32]),
                score,
                &z,
            ),
            TargetWeightsIdentityWire::new(
                &[1.0, 2.0, 3.0],
                snapshot,
                SemanticDigest::from_bytes([5; 32]),
                &z,
            ),
            TargetWeightsIdentityWire::new(&[1.0, 2.0, 3.0], snapshot, score, &[]),
        ];
        for variant in &variants {
            assert_ne!(target_weights_digest(variant).unwrap(), digest, "{variant:?}");
        }
        assert_ne!(
            PayloadDigestWire::f64s(ROW_WEIGHTS_PAYLOAD, &[0.0]).digest,
            PayloadDigestWire::f64s(ROW_WEIGHTS_PAYLOAD, &[-0.0]).digest,
            "no scale or sign canonicalization"
        );
    }
}
