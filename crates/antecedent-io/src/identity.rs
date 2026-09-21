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
//! | Graphs | Edge lists are sorted; temporal-DAG nodes are sorted by `(variable, lag)` and edges re-indexed. Insertion order never reaches a digest. |
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
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, admg_from_adjacency_mask, cpdag_from_adjacency_mask,
    dag_from_adjacency_mask, pag_from_adjacency_mask, temporal_cpdag_from_dbn_masks,
    temporal_dag_from_dbn_masks, temporal_pag_from_dbn_masks,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{
    Admg, Cpdag, Dag, MarkedEdge, Pag, TemporalCpdag, TemporalDag, TemporalPag,
};
use antecedent_identify::IdentificationResult;
use serde::{Deserialize, Serialize};

use crate::analysis_wire::{IdentifiedEstimandWire, RdDesignWire};
use crate::convert::{
    admg_to_wire, cpdag_to_wire, dag_to_wire, pag_to_wire, schema_to_wire, vars_to_raw,
};
use crate::discovery_wire::{TemporalGraphWire, temporal_dag_to_wire};
use crate::error::IoError;
use crate::expr_wire::{ExprArenaWire, expr_arena_to_wire};
use crate::query_wire::{
    CausalQueryWire, InterventionWire, OutcomeFunctionalWire, TargetPopulationWire, ValueWire,
};
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
    let mut hasher = blake3::Hasher::new_derive_key(domain.derive_key());
    hasher.update(&IDENTITY_FORMAT.to_le_bytes());
    hasher.update(payload);
    SemanticDigest::from_bytes(*hasher.finalize().as_bytes())
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
    Ok(digest_canonical(domain, &to_cbor(value)?))
}

/// Lower-case hex of [`digest_wire`], for artifact ids and wire-level identity strings.
///
/// # Errors
///
/// CBOR encode failure.
pub(crate) fn digest_wire_hex<T: Serialize>(
    domain: IdentityDomain,
    value: &T,
) -> Result<String, IoError> {
    Ok(digest_wire(domain, value)?.to_hex())
}

/// Host/Python view of the executed functional. Stable tags, not `Debug`.
///
/// Every query kind reports through this owner so consume and prepare do not
/// re-decide science in the adapter.
#[must_use]
pub fn executed_functional_labels(query: &CausalQueryWire) -> Vec<(String, String)> {
    match query {
        CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            control,
            active,
            target_population,
            outcome_functional,
            ..
        } => contrast_labels(ContrastLabels {
            kind: "average_effect",
            treatment: *treatment,
            outcome: *outcome,
            control,
            active,
            population: target_population,
            functional: Some(outcome_functional),
            temporal: "none".into(),
        }),
        CausalQueryWire::TemporalEffect {
            treatment,
            outcome,
            control,
            active,
            horizon_steps,
            target_population,
            ..
        } => contrast_labels(ContrastLabels {
            kind: "temporal_effect",
            treatment: *treatment,
            outcome: *outcome,
            control,
            active,
            population: target_population,
            functional: None,
            temporal: format!("horizon:{horizon_steps}"),
        }),
        CausalQueryWire::ConditionalEffect { inner } => {
            let mut labels = executed_functional_labels(inner);
            labels.retain(|(key, _)| key != "query_kind");
            let mut out = vec![("query_kind".into(), "conditional_effect".into())];
            out.extend(labels);
            out
        }
        CausalQueryWire::Mediation {
            treatment,
            outcome,
            control,
            active,
            target_population,
            horizons,
            ..
        } => contrast_labels(ContrastLabels {
            kind: "mediation",
            treatment: *treatment,
            outcome: *outcome,
            control,
            active,
            population: target_population,
            functional: None,
            temporal: horizons_label(horizons),
        }),
        CausalQueryWire::PathSpecific(path) => contrast_labels(ContrastLabels {
            kind: "path_specific",
            treatment: path.treatment,
            outcome: path.outcome,
            control: &path.control,
            active: &path.active,
            population: &path.target_population,
            functional: None,
            temporal: "none".into(),
        }),
        other => other_query_labels(other),
    }
}

struct ContrastLabels<'a> {
    kind: &'static str,
    treatment: u32,
    outcome: u32,
    control: &'a InterventionWire,
    active: &'a InterventionWire,
    population: &'a TargetPopulationWire,
    functional: Option<&'a OutcomeFunctionalWire>,
    temporal: String,
}

fn contrast_labels(contrast: ContrastLabels<'_>) -> Vec<(String, String)> {
    let mut out = vec![
        ("query_kind".into(), contrast.kind.into()),
        ("treatment".into(), contrast.treatment.to_string()),
        ("outcome".into(), contrast.outcome.to_string()),
        ("control".into(), intervention_label(contrast.control)),
        ("active".into(), intervention_label(contrast.active)),
        ("population".into(), population_label(contrast.population)),
        ("temporal_coordinates".into(), contrast.temporal),
    ];
    if let Some(functional) = contrast.functional {
        out.push(("outcome_functional".into(), outcome_functional_label(functional)));
    }
    out
}

fn other_query_labels(query: &CausalQueryWire) -> Vec<(String, String)> {
    match query {
        CausalQueryWire::Counterfactual { outcomes, interventions, .. } => vec![
            ("query_kind".into(), "counterfactual".into()),
            ("outcome".into(), join_ids(outcomes)),
            (
                "active".into(),
                interventions.iter().map(intervention_label).collect::<Vec<_>>().join(";"),
            ),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::Distribution(distribution) => vec![
            ("query_kind".into(), "distribution".into()),
            ("outcome".into(), join_ids(&distribution.outcomes)),
            ("population".into(), population_label(&distribution.target_population)),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::Response(response) => vec![
            ("query_kind".into(), "response".into()),
            ("population".into(), population_label(&response.target_population)),
            ("outcome_functional".into(), outcome_functional_label(&response.outcome_functional)),
            (
                "temporal_coordinates".into(),
                response
                    .temporal
                    .as_ref()
                    .map_or_else(|| "none".into(), |spec| horizons_label(&spec.horizons)),
            ),
        ],
        CausalQueryWire::Transport(transport) => vec![
            ("query_kind".into(), "transport".into()),
            (
                "population".into(),
                format!(
                    "source:{}->target:{}",
                    transport.source_population, transport.target_population
                ),
            ),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::Interference(_) => vec![
            ("query_kind".into(), "interference".into()),
            ("temporal_coordinates".into(), "none".into()),
        ],
        CausalQueryWire::AnomalyAttribution { targets, .. } => {
            named_outcomes("anomaly_attribution", targets)
        }
        CausalQueryWire::ChangeAttribution { outcome, .. } => {
            named_outcomes("change_attribution", &[*outcome])
        }
        CausalQueryWire::UnitChange { outcome, .. } => named_outcomes("unit_change", &[*outcome]),
        CausalQueryWire::MechanismChange { targets, .. } => {
            named_outcomes("mechanism_change", targets)
        }
        _ => unreachable!("contrast variants are handled by executed_functional_labels"),
    }
}

fn named_outcomes(kind: &'static str, outcomes: &[u32]) -> Vec<(String, String)> {
    vec![
        ("query_kind".into(), kind.into()),
        ("outcome".into(), join_ids(outcomes)),
        ("temporal_coordinates".into(), "none".into()),
    ]
}

fn join_ids(ids: &[u32]) -> String {
    ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

fn horizons_label(horizons: &[u32]) -> String {
    if horizons.is_empty() {
        "none".into()
    } else {
        format!(
            "horizons:{}",
            horizons.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        )
    }
}

fn population_label(population: &TargetPopulationWire) -> String {
    match population {
        TargetPopulationWire::AllObserved => "all_observed".into(),
        TargetPopulationWire::Treated => "treated".into(),
        TargetPopulationWire::Untreated => "untreated".into(),
        TargetPopulationWire::Environment(id) => format!("environment:{id}"),
        TargetPopulationWire::PredicateNamed { name, .. } => format!("predicate:{name}"),
        TargetPopulationWire::PredicateRows(rows) => {
            format!("rows:{}", rows.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))
        }
        TargetPopulationWire::CustomDistribution { handle, .. } => {
            format!("custom_distribution:{handle}")
        }
        TargetPopulationWire::RowWeights { .. } => "row_weights".into(),
        TargetPopulationWire::LocalAtCutoff { running, cutoff } => {
            format!("local_at_cutoff:{running}:{cutoff}")
        }
    }
}

fn outcome_functional_label(functional: &OutcomeFunctionalWire) -> String {
    match functional {
        OutcomeFunctionalWire::Mean => "mean".into(),
        OutcomeFunctionalWire::Exceedance(threshold) => format!("exceedance:{threshold}"),
        OutcomeFunctionalWire::ExceedanceGrid(grid) => format!(
            "exceedance_grid:{}",
            grid.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        ),
        OutcomeFunctionalWire::Quantile(tau) => format!("quantile:{tau}"),
    }
}

fn intervention_label(intervention: &InterventionWire) -> String {
    match intervention {
        InterventionWire::Set { variable, value } => {
            format!("set:{variable}={}", value_label(value))
        }
        InterventionWire::Shift { variable, delta } => {
            format!("shift:{variable}={}", value_label(delta))
        }
        InterventionWire::Stochastic { variable, .. } => format!("stochastic:{variable}"),
        InterventionWire::Soft { variable, .. } => format!("soft:{variable}"),
        InterventionWire::Sequence { steps } => format!(
            "sequence:{}",
            steps
                .iter()
                .map(|step| intervention_label(&step.intervention))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn value_label(value: &ValueWire) -> String {
    match value {
        ValueWire::Float64(value) => value.to_string(),
        ValueWire::Int64(value) => value.to_string(),
        ValueWire::Bool(value) => value.to_string(),
        ValueWire::Category(value) => format!("cat:{value}"),
        ValueWire::Label(value) => value.clone(),
    }
}

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
        let mut question = self.clone();
        if let Some(population) = question.query.target_population_mut() {
            *population = TargetPopulationWire::AllObserved;
        }
        question
    }
}

/// Digest of a data-sized payload embedded in an identity wire.
///
/// `BLAKE3-derive-key("antecedent.identity.payload")(IDENTITY_FORMAT_le ||
/// len(kind)_le_u64 || kind || bytes)`. The `kind` label separates payload
/// families (row indices, weights, cluster ids, prior bytes, ...), so equal
/// bytes of different families never share a digest. This is the only
/// sub-digest convention inside identity wires.
#[must_use]
pub fn payload_digest(kind: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key("antecedent.identity.payload");
    hasher.update(&IDENTITY_FORMAT.to_le_bytes());
    hasher.update(&(kind.len() as u64).to_le_bytes());
    hasher.update(kind.as_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

/// Content digest plus element count for a data-sized payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PayloadDigestWire {
    /// [`payload_digest`] of the little-endian payload bytes.
    pub digest: [u8; 32],
    /// Element count.
    pub len: u64,
}

impl PayloadDigestWire {
    /// Digest little-endian `u32` values.
    #[must_use]
    pub fn u32s(kind: &str, values: &[u32]) -> Self {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self { digest: payload_digest(kind, &bytes), len: values.len() as u64 }
    }

    /// Digest little-endian `i64` values.
    #[must_use]
    pub fn i64s(kind: &str, values: &[i64]) -> Self {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self { digest: payload_digest(kind, &bytes), len: values.len() as u64 }
    }

    /// Digest exact IEEE-754 bits of `f64` values.
    #[must_use]
    pub fn f64s(kind: &str, values: &[f64]) -> Self {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Self { digest: payload_digest(kind, &bytes), len: values.len() as u64 }
    }

    /// Digest length-prefixed groups of `u32` values.
    #[must_use]
    pub fn u32_groups(kind: &str, groups: &[Vec<u32>]) -> Self {
        let mut bytes = Vec::new();
        for group in groups {
            bytes.extend_from_slice(&(group.len() as u64).to_le_bytes());
            for value in group {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        Self { digest: payload_digest(kind, &bytes), len: groups.len() as u64 }
    }

    /// Digest raw bytes.
    #[must_use]
    pub fn bytes(kind: &str, bytes: &[u8]) -> Self {
        Self { digest: payload_digest(kind, bytes), len: bytes.len() as u64 }
    }
}

/// Graph payload for identification identity. Uses existing wire types.
///
/// Edge lists are in canonical (sorted) order, so a graph's identity never
/// depends on insertion order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GraphIdentityWire {
    /// Static DAG.
    Dag(DagWire),
    /// Static ADMG.
    Admg(AdmgWire),
    /// Static CPDAG.
    Cpdag(CpdagWire),
    /// Static PAG.
    Pag(PagWire),
    /// Temporal DAG (nodes sorted by `(variable, lag)`).
    TemporalDag(TemporalGraphWire),
    /// Temporal class with lagged/context nodes and marked edges.
    TemporalClass(TemporalClassIdentityWire),
    /// Graph-posterior atoms: every atom's structure and weight.
    GraphPosterior {
        /// Declared graph class of the posterior atoms.
        graph_class: String,
        /// Number of retained atoms.
        n_atoms: u64,
        /// Atom graphs and weights, in posterior order.
        atoms: Vec<PosteriorAtomIdentityWire>,
    },
}

impl GraphIdentityWire {
    /// Scientific projection hashed into identification premises.
    ///
    /// Posterior atoms are a weighted set: atoms are sorted canonically and
    /// the position-derived [`PosteriorAtomIdentityWire::execution_key`] is
    /// cleared, so reordering atoms or renumbering cache keys preserves
    /// identity while any graph or weight change moves it.
    #[must_use]
    pub fn premises(&self) -> Self {
        match self {
            Self::GraphPosterior { graph_class, n_atoms, atoms } => {
                let mut keyed: Vec<(Vec<u8>, PosteriorAtomIdentityWire)> = atoms
                    .iter()
                    .map(|atom| {
                        let atom = PosteriorAtomIdentityWire { execution_key: 0, ..atom.clone() };
                        (to_cbor(&atom).unwrap_or_default(), atom)
                    })
                    .collect();
                keyed.sort_by(|left, right| left.0.cmp(&right.0));
                Self::GraphPosterior {
                    graph_class: graph_class.clone(),
                    n_atoms: *n_atoms,
                    atoms: keyed.into_iter().map(|(_, atom)| atom).collect(),
                }
            }
            other => other.clone(),
        }
    }
}

/// Temporal CPDAG / PAG identity: lagged edges plus contemporaneous marks.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalClassIdentityWire {
    /// `temporal_cpdag` or `temporal_pag`.
    pub kind: String,
    /// Nodes in dense order.
    pub nodes: Vec<IdentityNodeWire>,
    /// Marked edges, including middle marks.
    pub edges: Vec<IdentityMarkedEdgeWire>,
}

/// Node identity that includes lagged and contemporaneous bindings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IdentityNodeWire {
    /// Static variable.
    Static {
        /// Variable raw id.
        variable: u32,
    },
    /// Lagged temporal node.
    Lagged {
        /// Variable raw id.
        variable: u32,
        /// Lag magnitude (`0` = contemporaneous).
        lag: u32,
    },
    /// Context / environment node.
    Context {
        /// Variable raw id.
        variable: u32,
        /// Optional environment id.
        environment: Option<u32>,
    },
}

impl IdentityNodeWire {
    fn from_node(node: NodeRef) -> Self {
        match node {
            NodeRef::Static(variable) => Self::Static { variable: variable.raw() },
            NodeRef::Lagged { variable, lag } => {
                Self::Lagged { variable: variable.raw(), lag: lag.raw() }
            }
            NodeRef::Context { variable, environment } => Self::Context {
                variable: variable.raw(),
                environment: environment.map(antecedent_core::EnvironmentId::raw),
            },
        }
    }
}

/// Marked edge with middle mark (LPCMCI).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentityMarkedEdgeWire {
    /// Endpoint A.
    pub a: u32,
    /// Endpoint B.
    pub b: u32,
    /// Mark at A.
    pub at_a: EndpointWire,
    /// Mark at B.
    pub at_b: EndpointWire,
    /// Middle mark (`unknown`, `left`, `right`, `both`, `empty`).
    pub middle: String,
}

fn endpoint_wire(mark: antecedent_graph::Endpoint) -> EndpointWire {
    match mark {
        antecedent_graph::Endpoint::Tail => EndpointWire::Tail,
        antecedent_graph::Endpoint::Arrow => EndpointWire::Arrow,
        antecedent_graph::Endpoint::Circle => EndpointWire::Circle,
        antecedent_graph::Endpoint::Conflict => EndpointWire::Conflict,
    }
}

fn middle_wire(mark: antecedent_graph::MiddleMark) -> &'static str {
    match mark {
        antecedent_graph::MiddleMark::Unknown => "unknown",
        antecedent_graph::MiddleMark::Left => "left",
        antecedent_graph::MiddleMark::Right => "right",
        antecedent_graph::MiddleMark::Both => "both",
        antecedent_graph::MiddleMark::Empty => "empty",
    }
}

fn marked_identity_edge(edge: MarkedEdge) -> IdentityMarkedEdgeWire {
    IdentityMarkedEdgeWire {
        a: edge.a.raw(),
        b: edge.b.raw(),
        at_a: endpoint_wire(edge.at_a),
        at_b: endpoint_wire(edge.at_b),
        middle: middle_wire(edge.middle).into(),
    }
}

/// DAG wire with edges in canonical `(from, to)` order.
///
/// # Errors
///
/// Non-directed edges.
pub fn canonical_dag_wire(dag: &Dag) -> Result<DagWire, IoError> {
    let mut wire = dag_to_wire(dag)?;
    wire.edges.sort_unstable();
    Ok(wire)
}

/// Temporal-DAG wire with nodes sorted by `(variable, lag)` and edges
/// re-indexed and sorted, so equal graphs built in any order encode equally.
///
/// # Errors
///
/// Non-lagged nodes.
pub fn canonical_temporal_dag_wire(graph: &TemporalDag) -> Result<TemporalGraphWire, IoError> {
    let wire = temporal_dag_to_wire(graph)?;
    let mut order: Vec<usize> = (0..wire.nodes.len()).collect();
    order.sort_by_key(|&index| (wire.nodes[index].variable, wire.nodes[index].lag));
    let mut position = vec![0u32; wire.nodes.len()];
    for (new, &old) in order.iter().enumerate() {
        position[old] = u32::try_from(new).map_err(|_| IoError::TooLarge)?;
    }
    let remap = |index: u32| {
        position.get(index as usize).copied().ok_or_else(|| {
            IoError::Convert("temporal edge references a node outside the graph".into())
        })
    };
    let mut directed = wire
        .directed
        .iter()
        .map(|&(from, to)| Ok((remap(from)?, remap(to)?)))
        .collect::<Result<Vec<_>, IoError>>()?;
    directed.sort_unstable();
    Ok(TemporalGraphWire {
        kind: wire.kind,
        nodes: order.iter().map(|&index| wire.nodes[index]).collect(),
        directed,
    })
}

/// Encode a DAG for identification identity.
///
/// # Errors
///
/// Non-directed edges.
pub fn dag_identity(dag: &Dag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Dag(canonical_dag_wire(dag)?))
}

/// Encode an ADMG for identification identity (sorted edge lists).
///
/// # Errors
///
/// Capacity overflow.
pub fn admg_identity(admg: &Admg) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Admg(canonical_admg_wire(admg)?))
}

fn canonical_admg_wire(admg: &Admg) -> Result<AdmgWire, IoError> {
    let mut wire = admg_to_wire(admg)?;
    wire.directed.sort_unstable();
    for pair in &mut wire.bidirected {
        if pair.0 > pair.1 {
            *pair = (pair.1, pair.0);
        }
    }
    wire.bidirected.sort_unstable();
    wire.bidirected.dedup();
    Ok(wire)
}

/// Encode a CPDAG for identification identity (sorted edge lists).
///
/// # Errors
///
/// Unexpected marks.
pub fn cpdag_identity(cpdag: &Cpdag) -> Result<GraphIdentityWire, IoError> {
    let mut wire = cpdag_to_wire(cpdag)?;
    wire.directed.sort_unstable();
    wire.undirected.sort_unstable();
    Ok(GraphIdentityWire::Cpdag(wire))
}

/// Encode a PAG for identification identity (edges sorted by endpoints).
///
/// # Errors
///
/// Non-static nodes.
pub fn pag_identity(pag: &Pag) -> Result<GraphIdentityWire, IoError> {
    let mut wire = pag_to_wire(pag)?;
    wire.edges.sort_by_key(|edge| (edge.a, edge.b));
    Ok(GraphIdentityWire::Pag(wire))
}

/// Encode a temporal DAG for identification identity.
///
/// # Errors
///
/// Non-lagged nodes.
pub fn temporal_dag_identity(graph: &TemporalDag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::TemporalDag(canonical_temporal_dag_wire(graph)?))
}

/// Encode a temporal CPDAG, including lagged and contemporaneous edges.
#[must_use]
pub fn temporal_cpdag_identity(graph: &TemporalCpdag) -> GraphIdentityWire {
    temporal_class_identity("temporal_cpdag", graph.nodes(), graph.edges())
}

/// Encode a temporal PAG, including lagged and contemporaneous edges.
#[must_use]
pub fn temporal_pag_identity(graph: &TemporalPag) -> GraphIdentityWire {
    temporal_class_identity("temporal_pag", graph.nodes(), graph.edges())
}

fn temporal_class_identity(
    kind: &'static str,
    nodes: &[NodeRef],
    edges: impl IntoIterator<Item = MarkedEdge>,
) -> GraphIdentityWire {
    GraphIdentityWire::TemporalClass(TemporalClassIdentityWire {
        kind: kind.into(),
        nodes: nodes.iter().copied().map(IdentityNodeWire::from_node).collect(),
        edges: edges.into_iter().map(marked_identity_edge).collect(),
    })
}

/// Structure of one graph-posterior atom.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PosteriorAtomGraphWire {
    /// Static DAG over the schema variables (canonical edge order).
    Static(DagWire),
    /// Static ADMG (canonical directed and bidirected edge order).
    Admg(AdmgWire),
    /// Static CPDAG (canonical directed and undirected edge order).
    Cpdag(CpdagWire),
    /// Static PAG (canonical marked edge order).
    Pag(PagWire),
    /// Lagged and contemporaneous DBN structure (canonical temporal wire).
    Temporal(TemporalGraphWire),
    /// Temporal class structure (lagged/context nodes and marked edges).
    TemporalClass(TemporalClassIdentityWire),
}

/// Durable graph-posterior atom: structure plus posterior weight.
///
/// The variable namespace is the enclosing premises' `schema_names`.
/// `execution_key` maps the atom to the local cache key used by the frozen
/// handle; it is audit data and is excluded from the hashed premises.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PosteriorAtomIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Posterior weight as IEEE-754 bits.
    pub weight_bits: u64,
    /// Atom structure.
    pub graph: PosteriorAtomGraphWire,
    /// Local envelope key for this frozen handle (not hashed).
    pub execution_key: u64,
}

/// Durable identities for every graph-posterior atom, static or DBN.
///
/// # Errors
///
/// Invalid adjacency or lag masks, or graph encode failure.
pub fn graph_posterior_atom_identities(
    posterior: &GraphPosterior,
) -> Result<Vec<PosteriorAtomIdentityWire>, IoError> {
    let ids: Vec<VariableId> = (0..posterior.n_vars)
        .map(|i| VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX)))
        .collect();
    let max_lag = posterior.max_lag.unwrap_or(1);
    let mut atoms = Vec::with_capacity(posterior.n_graphs);
    for (index, (&adjacency, &weight)) in
        posterior.adjacency.iter().zip(posterior.weights.iter()).enumerate()
    {
        let graph = posterior_atom_graph_wire(posterior, index, adjacency, max_lag, &ids)?;
        atoms.push(PosteriorAtomIdentityWire {
            format: IDENTITY_FORMAT,
            weight_bits: weight.to_bits(),
            graph,
            execution_key: u64::try_from(index).unwrap_or(u64::MAX),
        });
    }
    Ok(atoms)
}

fn posterior_atom_graph_wire(
    posterior: &GraphPosterior,
    index: usize,
    adjacency: u64,
    max_lag: u32,
    ids: &[VariableId],
) -> Result<PosteriorAtomGraphWire, IoError> {
    if let Some(lag_masks) = posterior.lag_masks.as_ref() {
        let lag_mask = *lag_masks
            .get(index)
            .ok_or_else(|| IoError::Convert("posterior lag masks must align with atoms".into()))?;
        return match posterior.atom_kind {
            GraphPosteriorAtomKind::Cpdag => {
                let graph = temporal_cpdag_from_dbn_masks(
                    adjacency,
                    lag_mask,
                    posterior.n_vars,
                    max_lag,
                    ids,
                )
                .map_err(|err| IoError::Convert(err.to_string()))?;
                let GraphIdentityWire::TemporalClass(wire) = temporal_cpdag_identity(&graph) else {
                    return Err(IoError::Convert(
                        "temporal CPDAG posterior atom must encode as TemporalClass".into(),
                    ));
                };
                Ok(PosteriorAtomGraphWire::TemporalClass(wire))
            }
            GraphPosteriorAtomKind::Pag => {
                let mark_mask = posterior
                    .mark_masks
                    .as_ref()
                    .and_then(|masks| masks.get(index))
                    .copied()
                    .unwrap_or(0);
                let graph = temporal_pag_from_dbn_masks(
                    adjacency,
                    lag_mask,
                    mark_mask,
                    posterior.n_vars,
                    max_lag,
                    ids,
                )
                .map_err(|err| IoError::Convert(err.to_string()))?;
                let GraphIdentityWire::TemporalClass(wire) = temporal_pag_identity(&graph) else {
                    return Err(IoError::Convert(
                        "temporal PAG posterior atom must encode as TemporalClass".into(),
                    ));
                };
                Ok(PosteriorAtomGraphWire::TemporalClass(wire))
            }
            _ => {
                let graph = temporal_dag_from_dbn_masks(
                    adjacency,
                    lag_mask,
                    posterior.n_vars,
                    max_lag,
                    ids,
                )
                .map_err(|err| IoError::Convert(err.to_string()))?;
                Ok(PosteriorAtomGraphWire::Temporal(canonical_temporal_dag_wire(&graph)?))
            }
        };
    }
    match posterior.atom_kind {
        GraphPosteriorAtomKind::Admg => {
            let admg = admg_from_adjacency_mask(adjacency, posterior.n_vars)
                .map_err(|err| IoError::Convert(err.to_string()))?;
            Ok(PosteriorAtomGraphWire::Admg(canonical_admg_wire(&admg)?))
        }
        GraphPosteriorAtomKind::Cpdag => {
            let cpdag = cpdag_from_adjacency_mask(adjacency, posterior.n_vars)
                .map_err(|err| IoError::Convert(err.to_string()))?;
            let GraphIdentityWire::Cpdag(wire) = cpdag_identity(&cpdag)? else {
                return Err(IoError::Convert("CPDAG posterior atom must encode as Cpdag".into()));
            };
            Ok(PosteriorAtomGraphWire::Cpdag(wire))
        }
        GraphPosteriorAtomKind::Pag => {
            let mark_mask = posterior
                .mark_masks
                .as_ref()
                .and_then(|masks| masks.get(index))
                .copied()
                .unwrap_or(0);
            let pag = pag_from_adjacency_mask(adjacency, mark_mask, posterior.n_vars)
                .map_err(|err| IoError::Convert(err.to_string()))?;
            let GraphIdentityWire::Pag(wire) = pag_identity(&pag)? else {
                return Err(IoError::Convert("PAG posterior atom must encode as Pag".into()));
            };
            Ok(PosteriorAtomGraphWire::Pag(wire))
        }
        _ => {
            let dag = dag_from_adjacency_mask(adjacency, posterior.n_vars)
                .map_err(|err| IoError::Convert(err.to_string()))?;
            Ok(PosteriorAtomGraphWire::Static(canonical_dag_wire(&dag)?))
        }
    }
}

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

/// Observation-contract payload (schema only; not row contents).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Schema.
    pub schema: SchemaWire,
    /// Observation tags with their variable arguments, sorted and deduplicated.
    pub observation: Vec<String>,
}

/// Observation-contract payload for digest and durable rehash.
///
/// Tags are a set: they are sorted and deduplicated. Each tag carries its
/// variable arguments (for example `independent_given:0,2`), so assumptions
/// that differ only in their conditioning set are distinct.
#[must_use]
pub fn observation_identity_wire(
    schema: &CausalSchema,
    observation: impl IntoIterator<Item = impl Into<String>>,
) -> ObservationIdentityWire {
    let mut observation: Vec<String> = observation.into_iter().map(Into::into).collect();
    observation.sort();
    observation.dedup();
    ObservationIdentityWire { format: IDENTITY_FORMAT, schema: schema_to_wire(schema), observation }
}

/// Ordered storage partition in a data snapshot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataPartitionIdentityWire {
    /// Digest from `OwnedColumnarStorage::content_digest` (storage encoding v1).
    pub content: [u8; 32],
    /// Row count for this partition.
    pub row_count: u64,
    /// Panel unit label; absent for tabular/series/environment partitions.
    pub unit_id: Option<u32>,
    /// Sampling regularity for this partition, when temporal.
    pub regularity: Option<String>,
}

/// Fixed interference network and realized assignment bound with the data.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InterferenceSnapshotWire {
    /// Unit table content partition.
    pub units: DataPartitionIdentityWire,
    /// Edges `(from, to, weight bits)` in `(from, to)` order.
    pub edges: PayloadDigestWire,
    /// Realized binary assignment in unit-row order.
    pub assignment: PayloadDigestWire,
}

/// Data-snapshot payload binding contents and ordered unit/environment partitions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DataSnapshotIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Observation digest.
    pub observation: [u8; 32],
    /// `tabular`, `series`, or `panel`.
    pub modality: String,
    /// Sampling regularity label, when temporal.
    pub regularity: Option<String>,
    /// Row or time-step count.
    pub row_count: u64,
    /// Unit count for panels; `None` for tabular/series.
    pub unit_count: Option<u64>,
    /// Ordered partitions, including content, masks, weights, and temporal metadata.
    pub partitions: Vec<DataPartitionIdentityWire>,
    /// Interference network and assignment, when the study carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interference: Option<InterferenceSnapshotWire>,
}

/// Digest a data snapshot identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn data_snapshot_digest(wire: &DataSnapshotIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::DataSnapshot, wire)
}

/// Score / batch reuse key. Stricter than identification: matching names and
/// shapes are not enough. Shared design is not shared nuisance fits.
///
/// Row-sized vectors enter as length-prefixed digests so the payload stays
/// small enough to travel in every contract section that advertises it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScoreReuseIdentityWire {
    /// Identity format.
    pub format: u16,
    /// `score_table` or `batch_share`.
    pub kind: String,
    /// Identification digest when the key is query-specific.
    pub identification: Option<[u8; 32]>,
    /// Data snapshot the scores or folds were fit on.
    pub data_snapshot: [u8; 32],
    /// Complete-case row index, or the full-table index for a shared design.
    pub row_index: PayloadDigestWire,
    /// Fold assignment aligned with [`Self::row_index`].
    pub fold_ids: PayloadDigestWire,
    /// Fold count used to assign [`Self::fold_ids`].
    pub n_folds: u32,
    /// Certified adjustment set, writer order.
    pub adjustment_set: Vec<u32>,
    /// Nuisance provenance tag. Absent on a design-only share key.
    pub nuisance_provenance: Option<String>,
    /// Treatment variable. Absent on a design-only share key.
    pub treatment: Option<u32>,
    /// Additional intervened coordinates.
    pub intervened: Vec<u32>,
    /// Requested bootstrap / shared-draw count.
    pub bootstrap_replicates: u32,
    /// Inference binding the scores were produced under: the estimator configuration, GLM
    /// options, overlap policy and backend all change the cross-fitted scores while the
    /// provenance tag stays constant. Absent on a design-only share key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_binding: Option<[u8; 32]>,
}

impl ScoreReuseIdentityWire {
    /// Query-specific score-table key.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn score_table(
        identification: SemanticDigest,
        data_snapshot: SemanticDigest,
        row_index: &[u32],
        fold_ids: &[u32],
        n_folds: u32,
        adjustment_set: &[VariableId],
        nuisance_provenance: &str,
        treatment: VariableId,
        intervened: &[VariableId],
        bootstrap_replicates: u32,
        inference_binding: SemanticDigest,
    ) -> Self {
        Self {
            format: IDENTITY_FORMAT,
            kind: "score_table".into(),
            identification: Some(*identification.as_bytes()),
            data_snapshot: *data_snapshot.as_bytes(),
            row_index: PayloadDigestWire::u32s("score_reuse.row_index", row_index),
            fold_ids: PayloadDigestWire::u32s("score_reuse.fold_ids", fold_ids),
            n_folds,
            adjustment_set: vars_to_raw(adjustment_set),
            nuisance_provenance: Some(nuisance_provenance.into()),
            treatment: Some(treatment.raw()),
            intervened: vars_to_raw(intervened),
            bootstrap_replicates,
            inference_binding: Some(*inference_binding.as_bytes()),
        }
    }

    /// Shared-design key: folds and covariates, not per-query nuisances.
    #[must_use]
    pub fn batch_share(
        data_snapshot: SemanticDigest,
        fold_ids: &[u32],
        n_folds: u32,
        adjustment_set: &[VariableId],
    ) -> Self {
        let row_index: Vec<u32> =
            (0..fold_ids.len()).map(|index| u32::try_from(index).unwrap_or(u32::MAX)).collect();
        Self {
            format: IDENTITY_FORMAT,
            kind: "batch_share".into(),
            identification: None,
            data_snapshot: *data_snapshot.as_bytes(),
            row_index: PayloadDigestWire::u32s("score_reuse.row_index", &row_index),
            fold_ids: PayloadDigestWire::u32s("score_reuse.fold_ids", fold_ids),
            n_folds,
            adjustment_set: vars_to_raw(adjustment_set),
            nuisance_provenance: None,
            treatment: None,
            intervened: Vec::new(),
            bootstrap_replicates: 0,
            inference_binding: None,
        }
    }
}

/// Digest a score / batch reuse identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn score_reuse_digest(wire: &ScoreReuseIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::ScoreReuse, wire)
}

/// Row-weight identity bound to one data snapshot and one score table.
///
/// Its digest is the `RowWeights` population's `weights` field, so the target
/// identity and every claim over it name exactly this weighting.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TargetWeightsIdentityWire {
    /// Identity format.
    pub format: u16,
    /// The weight vector under the one data-sized payload convention
    /// ([`PayloadDigestWire::f64s`], kind `target_weights.rows`).
    pub weights: [u8; 32],
    /// Row count the weights cover (score-table rows).
    pub row_count: u64,
    /// Data snapshot digest.
    pub data_snapshot: [u8; 32],
    /// Score-reuse digest of the table the weights reweight.
    pub score_reuse: [u8; 32],
    /// Covariates the weights are declared to depend on.
    pub depends_on: Vec<u32>,
}

impl TargetWeightsIdentityWire {
    /// Bind `weights` to a snapshot and score table.
    #[must_use]
    pub fn new(
        weights: &[f64],
        data_snapshot: SemanticDigest,
        score_reuse: SemanticDigest,
        depends_on: &[VariableId],
    ) -> Self {
        let rows = PayloadDigestWire::f64s(ROW_WEIGHTS_PAYLOAD, weights);
        Self {
            format: IDENTITY_FORMAT,
            weights: rows.digest,
            row_count: rows.len,
            data_snapshot: *data_snapshot.as_bytes(),
            score_reuse: *score_reuse.as_bytes(),
            depends_on: vars_to_raw(depends_on),
        }
    }
}

/// Payload kind of the row-weight vector inside a `RowWeights` population and
/// the target-weights identity. One owner, so the population digest a query
/// carries and the identity that binds it agree by construction.
pub const ROW_WEIGHTS_PAYLOAD: &str = "target_weights.rows";

/// Digest target weights.
///
/// # Errors
///
/// CBOR encode failure.
pub fn target_weights_digest(wire: &TargetWeightsIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::TargetWeights, wire)
}

/// Licensed inferential commitments (not numeric knobs).
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
    /// Licensed inferential commitments.
    pub commitments: InferentialCommitmentsWire,
}

/// Digest a program identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn program_digest(wire: &ProgramIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::Program, wire)
}

/// Coefficient / residual prior contents.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PriorSetIdentityWire {
    /// Prior specifications in declaration order.
    pub specs: Vec<PriorSpecIdentityWire>,
    /// Contrast coding for categorical predictors, when declared.
    pub contrast: Option<String>,
    /// Categorical predictor variable ids.
    pub categorical: Vec<u32>,
    /// Prior-restriction assumption ids.
    pub restrictions: Vec<String>,
}

/// One prior specification.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PriorSpecIdentityWire {
    /// Gaussian coefficient prior: mean and variance vectors.
    GaussianCoefficients {
        /// Coefficient means.
        mean: PayloadDigestWire,
        /// Coefficient variances.
        variance: PayloadDigestWire,
    },
    /// Inverse-gamma residual variance prior.
    ResidualInvGamma {
        /// Shape bits.
        shape_bits: u64,
        /// Scale bits.
        scale_bits: u64,
    },
    /// Known residual variance bits.
    KnownResidualVariance(u64),
}

/// Transferred-prior mapping.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PriorMappingIdentityWire {
    /// Identical coefficient subspace.
    IdenticalCoefficientSubspace,
    /// Effect functional bridge.
    EffectFunctional {
        /// Source posterior quantity name.
        source_quantity: String,
    },
    /// Named parameter pairs `(source, target)`, a sorted and deduplicated set.
    NamedParameters {
        /// Source → target parameter pairs.
        pairs: Vec<(String, String)>,
    },
}

impl PriorMappingIdentityWire {
    /// Canonical mapping (named pairs sorted and deduplicated).
    #[must_use]
    pub fn from_mapping(mapping: &crate::PriorMapping) -> Self {
        match mapping {
            crate::PriorMapping::IdenticalCoefficientSubspace => Self::IdenticalCoefficientSubspace,
            crate::PriorMapping::EffectFunctional { source_quantity } => {
                Self::EffectFunctional { source_quantity: source_quantity.to_string() }
            }
            crate::PriorMapping::NamedParameters { pairs } => {
                let mut pairs: Vec<(String, String)> = pairs
                    .iter()
                    .map(|(source, target)| (source.to_string(), target.to_string()))
                    .collect();
                pairs.sort();
                pairs.dedup();
                Self::NamedParameters { pairs }
            }
        }
    }
}

/// One external prior-bank source retained for conflict re-evaluation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalPriorSourceIdentityWire {
    /// Caller-stable source id.
    pub id: String,
    /// Mapped source prior.
    pub prior: PriorSetIdentityWire,
    /// Power-prior alpha bits.
    pub alpha_bits: u64,
    /// Mixture weight bits, when a mixture component.
    pub mixture_weight_bits: Option<u64>,
    /// Declared prior-strength ESS bits.
    pub ess_bits: Option<u64>,
}

/// External prior-bank composition inputs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalComposeIdentityWire {
    /// Sources in composition order.
    pub sources: Vec<ExternalPriorSourceIdentityWire>,
    /// Conflict policy `(p_min bits, kl_scale bits)`, when re-shrinking.
    pub conflict_policy: Option<(u64, u64)>,
}

/// Bayesian configuration bound into the inference layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BayesianBindingWire {
    /// Backend (`conjugate_gaussian`, `laplace`, `hmc`).
    pub backend: String,
    /// Likelihood / link.
    pub likelihood: String,
    /// Posterior draws requested.
    pub n_draws: u64,
    /// Isotropic prior scale bits, when no explicit prior or artifact is set.
    pub prior_scale_bits: Option<u64>,
    /// Explicit prior contents.
    pub prior: Option<PriorSetIdentityWire>,
    /// Posterior-artifact bytes used for a deferred mapped hydrate.
    pub prior_artifact: Option<PayloadDigestWire>,
    /// Mapping for the prior artifact.
    pub prior_mapping: Option<PriorMappingIdentityWire>,
    /// External prior-bank composition.
    pub external_compose: Option<ExternalComposeIdentityWire>,
}

/// Discovery / estimation window split.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SplitIdentityWire {
    /// Discovery window `[start, end)`.
    pub discovery: (u64, u64),
    /// Estimation window `[start, end)`.
    pub estimation: (u64, u64),
    /// Gap between the windows.
    pub gap: u64,
    /// Series length the split was validated against.
    pub series_len: u64,
}

/// Observation-mechanism estimator options.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservationOptionsWire {
    /// Selected-outcome correction (`ipw` / `aipw`).
    pub selected_correction: String,
    /// Observation-probability floor bits.
    pub observation_probability_floor_bits: u64,
    /// Censoring-survival floor bits.
    pub censoring_survival_floor_bits: u64,
    /// Cross-fitting folds.
    pub crossfit_folds: u64,
}

/// Inference-binding payload: numeric knobs, prior contents, and resampling.
///
/// Independent of structure: a different graph with the same numeric
/// configuration has the same inference binding and a different program.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceBindingWire {
    /// Identity format.
    pub format: u16,
    /// `frequentist` or `bayesian`; must equal the program commitments.
    pub inference: String,
    /// Bootstrap replicates (frequentist).
    pub bootstrap_replicates: u32,
    /// Bayesian backend, likelihood, draws, and prior contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bayesian: Option<BayesianBindingWire>,
    /// Validation suite id; must equal the program commitments.
    pub validation_suite: Option<String>,
    /// Overlap policy tag.
    pub overlap_policy: Option<String>,
    /// Typed estimator configuration, when the caller supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_spec: Option<EstimatorSpecWire>,
    /// Response options, when a response surface is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_options: Option<ResponseOptionsWire>,
    /// Observation-mechanism estimator options.
    pub observation_options: ObservationOptionsWire,
    /// Discovery / estimation split, when declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitIdentityWire>,
}

/// Overlap policy of a configured estimator.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OverlapPolicyWire {
    /// Caller overrides overlap diagnostics.
    ExplicitOverride,
    /// Diagnostics required, with optional clip / trim bits.
    RequireDiagnostics {
        /// Propensity clip bits.
        clip_bits: Option<u64>,
        /// Propensity trim bits.
        trim_bits: Option<u64>,
    },
}

/// GLM fitting options of a configured estimator.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlmOptionsWire {
    /// Maximum IRLS iterations.
    pub max_iter: u32,
    /// Convergence tolerance bits.
    pub tol_bits: u64,
    /// Negative-binomial dispersion policy.
    pub nb_alpha: String,
    /// Ridge applied on separation, as bits.
    pub ridge_on_separation_bits: Option<u64>,
}

/// Scalar configuration of one configured estimator.
///
/// Data-sized fields (cluster ids, multiway ids, panel times, registry
/// weights) are [`PayloadDigestWire`]s computed once when the study is built.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EstimatorConfigWire {
    /// Linear-algebra backend.
    pub backend: String,
    /// Bootstrap replicates.
    pub bootstrap_replicates: u32,
    /// Overlap policy.
    pub overlap: OverlapPolicyWire,
    /// GLM options, when the estimator fits a GLM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glm: Option<GlmOptionsWire>,
    /// Analytic SE kind, when configurable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub se_kind: Option<String>,
    /// Caliper bits, when matching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caliper_bits: Option<u64>,
    /// Caliper scale, when matching on the propensity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caliper_scale: Option<String>,
    /// Propensity strata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_strata: Option<u32>,
    /// GLM outcome family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Linear fit kind and penalty bits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_kind: Option<(String, Option<u64>)>,
    /// Cluster ids, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_ids: Option<PayloadDigestWire>,
    /// Multiway cluster ids, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiway_ids: Option<PayloadDigestWire>,
    /// Panel times, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub panel_times: Option<PayloadDigestWire>,
    /// Population registry contents, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population_registry: Option<PayloadDigestWire>,
    /// Cross-fit folds (`dml` / `dr.learner`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folds: Option<u32>,
    /// Outcome nuisance spec name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// Treatment nuisance spec name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment: Option<String>,
    /// DML score name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<String>,
    /// `dr.learner` final-stage spec name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_learner: Option<String>,
    /// Full outcome learner configuration, including hyperparameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_config: Option<String>,
    /// Full treatment learner configuration, including hyperparameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub treatment_config: Option<String>,
    /// Full final-stage learner configuration, including hyperparameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_learner_config: Option<String>,
    /// Causal-forest tree count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_trees: Option<u32>,
    /// Causal-forest minimum leaf size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_leaf: Option<u32>,
    /// Causal-forest maximum depth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<u32>,
    /// Causal-forest honesty flag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub honesty: Option<bool>,
}

/// Portable estimator-spec identity (mirrors `antecedent::EstimatorSpec` variants).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EstimatorSpecWire {
    /// Id only.
    Default(String),
    /// Caller-configured OLS / linear adjustment.
    LinearAdjustmentAte(EstimatorConfigWire),
    /// Caller-configured inverse-probability weighting.
    PropensityWeighting(EstimatorConfigWire),
    /// Caller-configured propensity-score matching.
    PropensityMatching(EstimatorConfigWire),
    /// Caller-configured propensity stratification.
    PropensityStratification(EstimatorConfigWire),
    /// Caller-configured covariate distance matching.
    DistanceMatching(EstimatorConfigWire),
    /// Caller-configured augmented IPW.
    Aipw(EstimatorConfigWire),
    /// Caller-configured GLM adjustment.
    GlmAdjustment(EstimatorConfigWire),
    /// Caller-configured front-door two-stage.
    FrontDoorTwoStage(EstimatorConfigWire),
    /// Caller-configured Wald IV.
    IvWald(EstimatorConfigWire),
    /// Caller-configured two-stage least squares.
    Iv2Sls(EstimatorConfigWire),
    /// Caller-configured DML / AIPW.
    Dml(EstimatorConfigWire),
    /// Caller-configured DR-Learner.
    DrLearner(EstimatorConfigWire),
    /// Caller-configured causal forest.
    CausalForest(EstimatorConfigWire),
}

/// Portable response-surface options (every field that changes the estimate).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseOptionsWire {
    /// Cross-fitting folds.
    pub folds: u64,
    /// Nuisance spline basis count.
    pub nuisance_basis: u64,
    /// Nuisance roughness penalty bits.
    pub nuisance_lambda_bits: u64,
    /// Kernel bandwidth bits, when set (normal reference otherwise).
    pub bandwidth_bits: Option<u64>,
    /// Weak-support ESS threshold bits.
    pub minimum_local_ess_bits: u64,
    /// Pointwise confidence level bits.
    pub confidence_level_bits: u64,
    /// Simultaneous-band multiplier replicates, when requested.
    pub simultaneous_replicates: Option<u32>,
    /// Multiplier seed for the simultaneous band.
    pub multiplier_seed: u64,
}

/// Digest an inference binding.
///
/// # Errors
///
/// CBOR encode failure.
pub fn inference_binding_digest(wire: &InferenceBindingWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::InferenceBinding, wire)
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
            minimum: u64::try_from(ctx.adaptive_draws.min_draws).unwrap_or(u64::MAX),
            epsilon_bits: ctx.adaptive_draws.quantile_width_rel_epsilon.to_bits(),
            ess_target_bits: Some(ctx.adaptive_draws.ess_target.to_bits()),
        }),
    }
}

/// Claim-id payload.
///
/// `seal` binds every contract identity, the four reasoning slots, and the
/// section's audit fields ([`crate::contract_seal`]); `claim` is the claim
/// section with a zeroed `claim_id`; `result` is [`crate::result_digest`] of
/// the executed result body. Nothing a verified consume reports is outside
/// these three.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClaimIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Contract seal.
    pub seal: [u8; 32],
    /// Claim fields with `claim_id` zeroed.
    pub claim: crate::ClaimSectionWire,
    /// Digest of the executed result body.
    pub result: [u8; 32],
}

impl ClaimIdentityWire {
    /// Bind a claim identity. Format is always [`IDENTITY_FORMAT`].
    #[must_use]
    pub fn new(seal: [u8; 32], claim: &crate::ClaimSectionWire, result: [u8; 32]) -> Self {
        Self {
            format: IDENTITY_FORMAT,
            seal,
            claim: crate::ClaimSectionWire { claim_id: [0; 32], ..claim.clone() },
            result,
        }
    }
}

/// Digest a claim envelope identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn claim_digest(wire: &ClaimIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::Claim, wire)
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
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint,
        SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;
    use crate::{causal_query_to_wire, from_cbor};

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
                prior_mapping: None,
                external_compose: None,
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
        };
        let plus_zero = inference_binding_digest(&binding).unwrap();
        binding.bayesian.as_mut().unwrap().prior_scale_bits = Some((-0.0f64).to_bits());
        let minus_zero = inference_binding_digest(&binding).unwrap();
        assert_ne!(minus_zero, plus_zero);
        binding.bayesian.as_mut().unwrap().backend = "conjugate_gaussian".into();
        assert_ne!(inference_binding_digest(&binding).unwrap(), minus_zero);
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

    fn static_posterior(weights: [f64; 3], graphs: [u64; 3]) -> GraphPosterior {
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        GraphPosterior::new(
            3,
            weights.to_vec(),
            graphs.to_vec(),
            vec![0.0; 9],
            vec![0.0; 9],
            1.0 / sum_sq,
            antecedent_prob::InferenceDiagnostics::analytic("static"),
            0,
        )
        .unwrap()
    }

    fn posterior_premises(posterior: &GraphPosterior) -> SemanticDigest {
        identification_digest(&premises(GraphIdentityWire::GraphPosterior {
            graph_class: "Dag".into(),
            n_atoms: posterior.n_graphs as u64,
            atoms: graph_posterior_atom_identities(posterior).unwrap(),
        }))
        .unwrap()
    }

    #[test]
    fn static_posterior_identity_binds_atom_graphs_and_weights() {
        use antecedent_discovery::set_edge;
        let direct = set_edge(0, 3, 0, 1, true);
        let adjusted = set_edge(set_edge(direct, 3, 2, 0, true), 3, 2, 1, true);
        let reversed = set_edge(0, 3, 1, 0, true);
        let confounded = set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true);
        let base = static_posterior([0.5, 0.3, 0.2], [direct, adjusted, reversed]);
        let reweighted = static_posterior([0.2, 0.3, 0.5], [direct, adjusted, reversed]);
        let regraphed = static_posterior([0.5, 0.3, 0.2], [direct, adjusted, confounded]);
        let reordered = static_posterior([0.2, 0.5, 0.3], [reversed, direct, adjusted]);
        assert_ne!(posterior_premises(&base), posterior_premises(&reweighted));
        assert_ne!(posterior_premises(&base), posterior_premises(&regraphed));
        // A weighted set: atom order and position-derived keys are not identity.
        assert_eq!(posterior_premises(&base), posterior_premises(&reordered));
        let atoms = graph_posterior_atom_identities(&base).unwrap();
        assert_eq!(atoms.iter().map(|atom| atom.execution_key).collect::<Vec<_>>(), [0, 1, 2]);
        assert!(matches!(atoms[0].graph, PosteriorAtomGraphWire::Static(_)));
    }

    #[test]
    fn dbn_atom_identity_includes_lags_and_ignores_execution_key() {
        use antecedent_discovery::set_edge;

        let contemporaneous = set_edge(0, 3, 2, 0, true);
        // Packing is `(lag-1)*n² + from*n + to` for lag=1, n=3.
        let confounder_to_outcome = 1_u64 << 7;
        let treatment_to_outcome = 1_u64 << 1;
        let posterior = GraphPosterior::new(
            3,
            vec![0.5, 0.5],
            vec![contemporaneous; 2],
            vec![0.0; 9],
            vec![0.0; 9],
            2.0,
            antecedent_prob::InferenceDiagnostics::analytic("dbn_atom"),
            0,
        )
        .unwrap()
        .with_lagged_marginals(1, vec![0.0; 9])
        .unwrap()
        .with_lag_masks(vec![confounder_to_outcome, treatment_to_outcome])
        .unwrap();
        assert_eq!(posterior.graph_keys[0], posterior.graph_keys[1]);
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
        let table = ScoreReuseIdentityWire::score_table(
            identification,
            snapshot,
            &[0, 1, 2, 3],
            &[0, 1, 0, 1],
            2,
            &[z],
            "crossfit.aipw",
            treatment,
            &[],
            0,
            inference,
        );
        let share = ScoreReuseIdentityWire::batch_share(snapshot, &[0, 1, 0, 1], 2, &[z]);
        let table_digest = score_reuse_digest(&table).unwrap();
        let share_digest = score_reuse_digest(&share).unwrap();
        assert_ne!(table_digest, share_digest);
        assert_eq!(score_reuse_digest(&table).unwrap(), table_digest);

        let other_folds = ScoreReuseIdentityWire::score_table(
            identification,
            snapshot,
            &[0, 1, 2, 3],
            &[1, 0, 1, 0],
            2,
            &[z],
            "crossfit.aipw",
            treatment,
            &[],
            0,
            inference,
        );
        assert_ne!(score_reuse_digest(&other_folds).unwrap(), table_digest);
        let other_rows = ScoreReuseIdentityWire::score_table(
            identification,
            snapshot,
            &[0, 1, 2, 4],
            &[0, 1, 0, 1],
            2,
            &[z],
            "crossfit.aipw",
            treatment,
            &[],
            0,
            inference,
        );
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
