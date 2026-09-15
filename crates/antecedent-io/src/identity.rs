//! Canonical encoding and domain-separated digests for contract identities.
//!
//! Payloads are existing CBOR wire types. Digests use BLAKE3 derive-key
//! domains from [`antecedent_core::IdentityDomain`]. `Debug`, Rust `Hash`,
//! arena offsets, and process-local keys are not durable identity.
//!
//! # Canonical encoding (`antecedent.identity.v1`)
//!
//! [`digest_canonical`] is the only advertised digest. Rules live next to it:
//!
//! | Rule | Encoding |
//! | --- | --- |
//! | Format tag | `IDENTITY_FORMAT` (little-endian `u16`) prefixes the hash; the same tag is stored on every payload. |
//! | Ordered fields | Schema variables, query lists, partitions, graph edge lists, and estimands keep writer order. |
//! | Set-valued fields | Observation tags and named prior-parameter pairs are sorted then deduplicated before hashing. |
//! | Floats | Claim values and overlap knobs use IEEE-754 bits (`f64::to_bits`). Other `f64` fields use CBOR float64 of those bits (`+0.0` ≠ `-0.0`). |
//! | Collision | BLAKE3-256 equality is identity under this encoding, not a proof. Independent consume rehashes stored payloads; a colliding advertisement without a matching payload is unresolved. Bump [`IDENTITY_FORMAT`] to change a rule. |
//!
//! DBN posterior atoms are not position-derived cache keys. Durable identity
//! includes lagged and contemporaneous edges, the variable namespace, and the
//! local envelope key used by the frozen handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalSchema, ExecutionContext, IDENTITY_FORMAT, IdentificationStatus, IdentityDomain, NodeRef,
    SemanticDigest, VariableId,
};
use antecedent_discovery::{GraphPosterior, temporal_dag_from_dbn_masks};
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
    causal_query_to_wire,
};
use crate::trace::{AssumptionRecordWire, DerivationStepWire, assumptions_to_wire};
use crate::wire::{AdmgWire, CpdagWire, DagWire, EndpointWire, PagWire, SchemaWire};
use crate::{from_cbor, to_cbor};

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
        TargetPopulationWire::PredicateNamed(name) => format!("predicate:{name}"),
        TargetPopulationWire::PredicateRows(rows) => format!(
            "rows:{}",
            rows.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
        ),
        TargetPopulationWire::CustomDistribution(id) => format!("custom_distribution:{id}"),
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
            steps.iter().map(|step| intervention_label(&step.intervention)).collect::<Vec<_>>().join(",")
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

/// Encode and digest a causal target.
///
/// # Errors
///
/// Query or CBOR encode failure.
pub fn target_digest(
    schema: &CausalSchema,
    query: &antecedent_core::CausalQuery,
) -> Result<SemanticDigest, IoError> {
    digest_wire(
        IdentityDomain::Target,
        &TargetIdentityWire {
            format: IDENTITY_FORMAT,
            schema: schema_to_wire(schema),
            query: causal_query_to_wire(query)?,
        },
    )
}

/// Graph payload for identification identity. Uses existing wire types.
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
    /// Temporal DAG.
    TemporalDag(TemporalGraphWire),
    /// Temporal class with lagged/context nodes and marked edges.
    TemporalClass(TemporalClassIdentityWire),
    /// Graph-posterior atoms. Static posteriors omit [`Self::GraphPosterior::atoms`].
    GraphPosterior {
        /// Declared graph class of the posterior atoms.
        graph_class: String,
        /// Number of retained atoms.
        n_atoms: u64,
        /// Durable DBN atom identities (lagged + contemporaneous + namespace).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        atoms: Vec<DbnAtomIdentityWire>,
    },
}

/// Temporal CPDAG / PAG identity: lagged edges plus contemporaneous marks.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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

/// Encode a DAG for identification identity.
///
/// # Errors
///
/// Non-directed edges.
pub fn dag_identity(dag: &Dag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Dag(dag_to_wire(dag)?))
}

/// Encode an ADMG for identification identity.
///
/// # Errors
///
/// Capacity overflow.
pub fn admg_identity(admg: &Admg) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Admg(admg_to_wire(admg)?))
}

/// Encode a CPDAG for identification identity.
///
/// # Errors
///
/// Unexpected marks.
pub fn cpdag_identity(cpdag: &Cpdag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Cpdag(cpdag_to_wire(cpdag)?))
}

/// Encode a PAG for identification identity.
///
/// # Errors
///
/// Non-static nodes.
pub fn pag_identity(pag: &Pag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Pag(pag_to_wire(pag)?))
}

/// Encode a temporal DAG for identification identity.
///
/// # Errors
///
/// Non-lagged nodes.
pub fn temporal_dag_identity(graph: &TemporalDag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::TemporalDag(temporal_dag_to_wire(graph)?))
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

/// Durable DBN posterior atom. Not a position-derived envelope key.
///
/// Edges include lagged and contemporaneous structure. `variables` is the
/// dense-id namespace. `execution_key` maps this identity back to the local
/// cache key used by the frozen handle (`dbn_envelope_key`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DbnAtomIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Variable names in dense-id order.
    pub variables: Vec<String>,
    /// Lagged and contemporaneous edges (existing temporal DAG wire).
    pub graph: TemporalGraphWire,
    /// Local envelope key for this frozen handle.
    pub execution_key: u64,
}

/// Digest one durable DBN atom.
///
/// # Errors
///
/// CBOR encode failure.
pub fn dbn_atom_digest(wire: &DbnAtomIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::IdentificationProduct, wire)
}

/// Durable identities for DBN posterior atoms.
///
/// Static graph posteriors (no lag masks) return an empty list so their
/// identification digest stays the pre-1.10 class + `n_atoms` encoding.
///
/// # Errors
///
/// Namespace length mismatch, invalid masks, or temporal encode failure.
pub fn dbn_atom_identities(
    posterior: &GraphPosterior,
    variables: &[String],
) -> Result<Vec<DbnAtomIdentityWire>, IoError> {
    let Some(lag_masks) = posterior.lag_masks.as_ref() else {
        return Ok(Vec::new());
    };
    if variables.len() != posterior.n_vars {
        return Err(IoError::Convert(
            "DBN atom namespace length must match posterior n_vars".into(),
        ));
    }
    let max_lag = posterior.max_lag.unwrap_or(1);
    let ids: Vec<VariableId> =
        (0..posterior.n_vars).map(|i| VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX))).collect();
    let mut atoms = Vec::with_capacity(posterior.n_graphs);
    for (index, (&adjacency, &lag_mask)) in
        posterior.adjacency.iter().zip(lag_masks.iter()).enumerate()
    {
        let graph = temporal_dag_from_dbn_masks(
            adjacency,
            lag_mask,
            posterior.n_vars,
            max_lag,
            &ids,
        )
        .map_err(|err| IoError::Convert(err.to_string()))?;
        atoms.push(DbnAtomIdentityWire {
            format: IDENTITY_FORMAT,
            variables: variables.to_vec(),
            graph: temporal_dag_to_wire(&graph)?,
            execution_key: u64::try_from(index).unwrap_or(u64::MAX),
        });
    }
    Ok(atoms)
}

/// Identification-layer payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentificationIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Target digest bytes.
    pub target: [u8; 32],
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
        target: &'a [u8; 32],
        graph_class: &'a str,
        schema_names: &'a Option<Vec<String>>,
        graph: &'a GraphIdentityWire,
        observation: &'a [u8; 32],
    }
    digest_wire(
        IdentityDomain::Identification,
        &Premises {
            format: wire.format,
            target: &wire.target,
            graph_class: &wire.graph_class,
            schema_names: &wire.schema_names,
            graph: &wire.graph,
            observation: &wire.observation,
        },
    )
}

/// Identification-product payload. Excludes search-effort counters.
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
    /// Derivation steps.
    pub derivation: Vec<DerivationStepWire>,
    /// Required assumptions.
    pub required_assumptions: Vec<AssumptionRecordWire>,
    /// Whether a hedge witness is attached.
    pub hedge: bool,
    /// Whether identification search was capped.
    pub search_capped: bool,
}

fn status_name(status: IdentificationStatus) -> &'static str {
    match status {
        IdentificationStatus::NonparametricallyIdentified => "nonparametrically_identified",
        IdentificationStatus::IdentifiedUnderParametricRestrictions => {
            "identified_under_parametric_restrictions"
        }
        IdentificationStatus::IdentifiedUnderPriorRestrictions => {
            "identified_under_prior_restrictions"
        }
        IdentificationStatus::PartiallyIdentified => "partially_identified",
        IdentificationStatus::GraphDependent => "graph_dependent",
        IdentificationStatus::NotIdentified => "not_identified",
    }
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
        status: status_name(result.status).into(),
        estimands: result.estimands.iter().map(estimand_wire).collect(),
        arena: expr_arena_to_wire(&result.arena)?,
        derivation: result
            .derivation
            .steps
            .iter()
            .map(|step| DerivationStepWire {
                rule: step.rule.to_string(),
                detail: step.detail.to_string(),
            })
            .collect(),
        required_assumptions: assumptions_to_wire(&result.required_assumptions),
        hedge: result.hedge.is_some(),
        search_capped,
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
    /// Observation assumption tags, when declared.
    pub observation: Vec<String>,
}

/// Observation-contract payload for digest and durable rehash.
#[must_use]
pub fn observation_identity_wire(
    schema: &CausalSchema,
    observation: impl IntoIterator<Item = impl Into<String>>,
) -> ObservationIdentityWire {
    let mut observation: Vec<String> = observation.into_iter().map(Into::into).collect();
    observation.sort();
    observation.dedup();
    ObservationIdentityWire {
        format: IDENTITY_FORMAT,
        schema: schema_to_wire(schema),
        observation,
    }
}

/// Digest the observation contract.
///
/// # Errors
///
/// CBOR encode failure.
pub fn observation_digest(
    schema: &CausalSchema,
    observation: impl IntoIterator<Item = impl Into<String>>,
) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::Observation, &observation_identity_wire(schema, observation))
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
}

/// Digest a data snapshot identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn data_snapshot_digest(wire: &DataSnapshotIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::DataSnapshot, wire)
}

/// Licensed inferential commitments (not numeric knobs).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferentialCommitmentsWire {
    /// Identity format.
    pub format: u16,
    /// Estimator family id.
    pub estimator: Option<String>,
    /// Identifier id.
    pub identifier: Option<String>,
    /// `frequentist` or `bayesian`.
    pub inference: String,
    /// Validation suite id.
    pub validation_suite: Option<String>,
    /// Interval target (`analytic_se`, `bootstrap_se`, `posterior_quantile`, …).
    pub interval_target: String,
    /// Whether a prior is required.
    pub prior_required: bool,
}

/// Program-layer payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProgramIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Identification digest.
    pub identification: [u8; 32],
    /// Identification-product digest, when prepared.
    pub identification_product: Option<[u8; 32]>,
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

/// Inference-binding payload (numeric knobs and prior mapping).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InferenceBindingWire {
    /// Identity format.
    pub format: u16,
    /// Program digest.
    pub program: [u8; 32],
    /// `frequentist` or `bayesian`.
    pub inference: String,
    /// Bootstrap replicates (frequentist).
    pub bootstrap_replicates: u32,
    /// Posterior draws (Bayesian); `None` on the frequentist path.
    pub n_draws: Option<u64>,
    /// Prior scale, when an isotropic prior is in force.
    pub prior_scale: Option<f64>,
    /// Prior-mapping tag, when a transferred prior is bound.
    pub prior_mapping: Option<String>,
    /// Validation suite id.
    pub validation_suite: Option<String>,
    /// Overlap policy tag.
    pub overlap_policy: Option<String>,
}

/// Digest an inference binding.
///
/// # Errors
///
/// CBOR encode failure.
pub fn inference_binding_digest(wire: &InferenceBindingWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::InferenceBinding, wire)
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
    ExecutionIdentityWire {
        format: IDENTITY_FORMAT,
        library_version: antecedent_core::VERSION.into(),
        seed: ctx.rng.master_seed(),
        threads: ctx.parallelism.max_threads.get(),
        backend: if ctx.kernel_policy.force_scalar {
            "scalar".into()
        } else if ctx.kernel_policy.allow_arch_simd {
            "optimized".into()
        } else {
            "portable".into()
        },
    }
}

/// Claim-id payload over the envelope's scientific fields.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClaimIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Program digest.
    pub program: [u8; 32],
    /// Target digest.
    pub target: [u8; 32],
    /// Claim kind.
    pub kind: String,
    /// Point value bits, when present (`None` = explicit absence).
    pub value_bits: Option<u64>,
    /// Execution digest, when an execution exists.
    pub execution: Option<[u8; 32]>,
}

impl ClaimIdentityWire {
    /// Bind a claim identity. Format is always [`IDENTITY_FORMAT`].
    #[must_use]
    pub fn new(
        program: [u8; 32],
        target: [u8; 32],
        kind: impl Into<String>,
        value_bits: Option<u64>,
        execution: Option<[u8; 32]>,
    ) -> Self {
        Self {
            format: IDENTITY_FORMAT,
            program,
            target,
            kind: kind.into(),
            value_bits,
            execution,
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

/// Mass-conservation check for structural mixtures.
///
/// # Errors
///
/// Any mass outside `[0, 1]`, a non-finite value, or a sum that is not 1
/// within `1e-12`.
pub fn validate_mixture_masses(
    identified: f64,
    unidentified: f64,
    unevaluable: f64,
    incomplete_search: f64,
) -> Result<(), IoError> {
    let masses = [identified, unidentified, unevaluable, incomplete_search];
    if masses.iter().any(|mass| !mass.is_finite() || *mass < 0.0 || *mass > 1.0) {
        return Err(IoError::Convert("mixture masses must be finite and inside [0, 1]".into()));
    }
    let sum = identified + unidentified + unevaluable + incomplete_search;
    if (sum - 1.0).abs() > 1e-12 {
        return Err(IoError::Convert(format!("mixture masses must sum to 1 (got {sum})")));
    }
    Ok(())
}

/// Round-trip helper used by tests: encode then decode a target payload.
///
/// # Errors
///
/// CBOR failure.
pub fn target_identity_round_trip(
    wire: &TargetIdentityWire,
) -> Result<TargetIdentityWire, IoError> {
    from_cbor(&to_cbor(wire)?)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;

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
        let first = target_digest(&schema, &query).unwrap();
        let second = target_digest(&schema, &query).unwrap();
        assert_eq!(first, second);
        let mut renamed = CausalSchemaBuilder::new();
        renamed
            .add_variable(
                "treatment",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        renamed
            .add_variable(
                "outcome",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        let renamed = renamed.build().unwrap();
        assert_ne!(target_digest(&renamed, &query).unwrap(), first);
    }

    #[test]
    fn different_graphs_change_identification_not_target() {
        let (schema, query, dag) = schema_and_query();
        let target = target_digest(&schema, &query).unwrap();
        let observation = observation_digest(&schema, std::iter::empty::<String>()).unwrap();
        let first = IdentificationIdentityWire {
            format: IDENTITY_FORMAT,
            target: *target.as_bytes(),
            graph_class: "Dag".into(),
            structure_source: "explicit".into(),
            accepted_version: 1,
            algorithm_id: None,
            schema_names: None,
            graph: dag_identity(&dag).unwrap(),
            observation: *observation.as_bytes(),
        };
        let mut other = Dag::with_variables(2);
        other.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(0)).unwrap();
        let second =
            IdentificationIdentityWire { graph: dag_identity(&other).unwrap(), ..first.clone() };
        assert_eq!(identification_digest(&first).unwrap(), identification_digest(&first).unwrap());
        assert_ne!(identification_digest(&first).unwrap(), identification_digest(&second).unwrap());
        assert_eq!(first.target, second.target);
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
            IdentificationIdentityWire { target: [9; 32], ..first.clone() },
            IdentificationIdentityWire { observation: [9; 32], ..first.clone() },
            IdentificationIdentityWire { graph_class: "Cpdag".into(), ..first.clone() },
            IdentificationIdentityWire {
                schema_names: Some(vec!["y".into(), "x".into()]),
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
        let mut changed_content = decoded;
        changed_content.partitions[1].content = [4; 32];
        assert_ne!(
            data_snapshot_digest(&original).unwrap(),
            data_snapshot_digest(&changed_content).unwrap()
        );
    }

    #[test]
    fn mixture_masses_must_sum_to_one() {
        validate_mixture_masses(0.5, 0.3, 0.2, 0.0).unwrap();
        assert!(validate_mixture_masses(0.5, 0.3, 0.1, 0.0).is_err());
        assert!(validate_mixture_masses(1.2, 0.0, 0.0, 0.0).is_err());
        assert!(validate_mixture_masses(f64::NAN, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn encoding_rule_changes_change_the_advertised_digest() {
        let (schema, query, _) = schema_and_query();
        let canonical = TargetIdentityWire {
            format: IDENTITY_FORMAT,
            schema: schema_to_wire(&schema),
            query: causal_query_to_wire(&query).unwrap(),
        };
        let advertised = digest_wire(IdentityDomain::Target, &canonical).unwrap();
        let mut bumped = canonical.clone();
        bumped.format = IDENTITY_FORMAT + 1;
        assert_ne!(digest_wire(IdentityDomain::Target, &bumped).unwrap(), advertised);
        let payload = to_cbor(&canonical).unwrap();
        let mut hasher = blake3::Hasher::new_derive_key(IdentityDomain::Target.derive_key());
        hasher.update(&(IDENTITY_FORMAT + 1).to_le_bytes());
        hasher.update(&payload);
        assert_ne!(SemanticDigest::from_bytes(*hasher.finalize().as_bytes()), advertised);

        let set_a = observation_digest(&schema, ["b", "a"]).unwrap();
        let set_b = observation_digest(&schema, ["a", "b", "a"]).unwrap();
        assert_eq!(set_a, set_b);
        let unsorted = ObservationIdentityWire {
            format: IDENTITY_FORMAT,
            schema: schema_to_wire(&schema),
            observation: vec!["b".into(), "a".into()],
        };
        assert_ne!(digest_wire(IdentityDomain::Observation, &unsorted).unwrap(), set_a);

        let mut binding = InferenceBindingWire {
            format: IDENTITY_FORMAT,
            program: [1; 32],
            inference: "bayesian".into(),
            bootstrap_replicates: 0,
            n_draws: Some(64),
            prior_scale: Some(1.0),
            prior_mapping: None,
            validation_suite: None,
            overlap_policy: None,
        };
        let plus_zero = inference_binding_digest(&binding).unwrap();
        binding.prior_scale = Some(-0.0);
        assert_ne!(inference_binding_digest(&binding).unwrap(), plus_zero);

        let claim = ClaimIdentityWire::new([2; 32], [3; 32], "point", Some(0.0f64.to_bits()), None);
        let flipped = ClaimIdentityWire::new(
            [2; 32],
            [3; 32],
            "point",
            Some((-0.0f64).to_bits()),
            None,
        );
        assert_ne!(claim_digest(&claim).unwrap(), claim_digest(&flipped).unwrap());
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
    fn dbn_atom_identity_includes_lags_namespace_and_execution_key() {
        use antecedent_discovery::set_edge;

        let contemporaneous = set_edge(0, 3, 2, 0, true);
        // Packing is `(lag-1)*n² + from*n + to` for lag=1, n=3.
        let confounder_to_outcome = 1_u64 << 7;
        let treatment_to_outcome = 1_u64 << 1;
        let names = ["treatment".into(), "outcome".into(), "confounder".into()];
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
        let atoms = dbn_atom_identities(&posterior, &names).unwrap();
        assert_eq!(atoms.len(), 2);
        assert_eq!(atoms[0].execution_key, 0);
        assert_eq!(atoms[1].execution_key, 1);
        assert_eq!(atoms[0].variables, names);
        assert_ne!(dbn_atom_digest(&atoms[0]).unwrap(), dbn_atom_digest(&atoms[1]).unwrap());
        let mut renamed = atoms[0].clone();
        renamed.variables[0] = "t".into();
        assert_ne!(dbn_atom_digest(&renamed).unwrap(), dbn_atom_digest(&atoms[0]).unwrap());
        assert!(dbn_atom_identities(&posterior, &["t".into()]).is_err());
        let static_posterior = GraphPosterior::new(
            3,
            vec![1.0],
            vec![contemporaneous],
            vec![0.0; 9],
            vec![0.0; 9],
            1.0,
            antecedent_prob::InferenceDiagnostics::analytic("static"),
            0,
        )
        .unwrap();
        assert!(dbn_atom_identities(&static_posterior, &names).unwrap().is_empty());
    }
}
