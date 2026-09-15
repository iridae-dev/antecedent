//! Canonical encoding and domain-separated digests for contract identities.
//!
//! Payloads are existing CBOR wire types. Digests use BLAKE3 derive-key
//! domains from [`antecedent_core::IdentityDomain`]. `Debug`, Rust `Hash`,
//! arena offsets, and process-local keys are not durable identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalSchema, ExecutionContext, IDENTITY_FORMAT, IdentificationStatus, IdentityDomain, NodeRef,
    SemanticDigest,
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
use crate::query_wire::{CausalQueryWire, causal_query_to_wire};
use crate::trace::{AssumptionRecordWire, DerivationStepWire, assumptions_to_wire};
use crate::wire::{AdmgWire, CpdagWire, DagWire, EndpointWire, PagWire, SchemaWire};
use crate::{from_cbor, to_cbor};

/// Domain-separated BLAKE3 digest of a canonical CBOR payload.
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
    /// Graph-posterior placeholder: class only; atoms live on the product layer.
    GraphPosterior {
        /// Declared graph class of the posterior atoms.
        graph_class: String,
        /// Number of retained atoms.
        n_atoms: u64,
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
    GraphIdentityWire::TemporalClass(TemporalClassIdentityWire {
        kind: "temporal_cpdag".into(),
        nodes: graph.nodes().iter().copied().map(IdentityNodeWire::from_node).collect(),
        edges: graph.edges().into_iter().map(marked_identity_edge).collect(),
    })
}

/// Encode a temporal PAG, including lagged and contemporaneous edges.
#[must_use]
pub fn temporal_pag_identity(graph: &TemporalPag) -> GraphIdentityWire {
    GraphIdentityWire::TemporalClass(TemporalClassIdentityWire {
        kind: "temporal_pag".into(),
        nodes: graph.nodes().iter().copied().map(IdentityNodeWire::from_node).collect(),
        edges: graph.edges().into_iter().map(marked_identity_edge).collect(),
    })
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
}
