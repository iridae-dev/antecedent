//! Independent artifacts for the DAG completions of a supplied CPDAG (2.3A X2).
//!
//! Format version 1. The artifact binds the CPDAG (variables, directed and
//! undirected edges), the shared coordinate schema, the question, the evidence
//! bound to each completion, the enumeration/decision limits and the supplied
//! exact laws, and stores the full report: every completion found (canonical
//! identity, edges, status, detail and completion-specific evidence identity),
//! the number of completions a stop never enumerated, the per-status counts
//! (never renormalized over the identified members), the unweighted structural
//! envelope and the stop receipt (explored versus not enumerated versus
//! unevaluated).
//!
//! A consumer trusts nothing: it rebuilds the CPDAG, re-enumerates the
//! completions, re-decides each one under the producer's recorded limits (none
//! may exceed its own), recompiles and re-evaluates, and accepts only a report
//! identical to the stored one. Two digests guard the stored premises: the
//! scientific premises digest (CPDAG, schema, question, request, limits) and a
//! separate data digest (evidence bindings with their catalogs, laws, provider).
//! A changed completion, snapshot, evidence binding or CPDAG, even with both
//! digests re-sealed, changes the replayed report and is refused.
//!
//! A report cut short by an operation, depth or memory bound exports: the limits
//! are recorded and the consumer reproduces the identical prefix, with the
//! unenumerated and unevaluated completions counted. A report cut short by
//! cancellation never exports: nothing recorded lets a consumer reproduce where
//! the interruption fell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    ExecutionContext, IdentityDomain, NodeRef, SearchLimits, SearchReceipt, SearchStop, VariableId,
    reason_code,
};
use antecedent_estimate::cpdag_scenarios::{
    CompletionReceiptEntry, CpdagScenarioReport, PreparedCpdagScenarios, prepare_cpdag_scenarios,
};
use antecedent_estimate::transport_scenarios::SCENARIO_EXACT_PROVIDER;
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData, LawOrigin};
use antecedent_graph::{Cpdag, DenseNodeId};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::cpdag_completion::{
    CPDAG_MAX_NODES, CompletionEvidenceBinding, CpdagCompletionInput, CpdagEvidence,
    CpdagScenarioError, decide_cpdag_completions,
};
use antecedent_identify::sid::scenarios::ScenarioCoordinate;
use serde::{Deserialize, Serialize};

use crate::IoError;
use crate::exact_law_wire::ExactLawWire;
use crate::query_wire::ValueWire;
use crate::transport_catalog_wire::EvidenceCatalogWire;
use crate::transport_scenario_artifact::{
    ScenarioCoordinateWire, ScenarioQueryWire, ScenarioReceiptWire, ScenarioReportWire,
    scenario_refusal,
};

/// The artifact format this reader writes and accepts.
pub const CPDAG_COMPLETION_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const CPDAG_COMPLETION_ARTIFACT_FEATURE: &str = "cpdag_completion_scenarios_v1";

/// Most completion records, laws or bindings a consumer accepts in one artifact.
const MAX_STORED_ITEMS: usize = 4096;

/// Why a completion artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CpdagArtifactError {
    /// The feature marker or a stored field is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A recorded limit or stored collection exceeds the consumer's bound; the
    /// consumer's search, evaluation or memory budget would be exceeded.
    #[error("consumer limit exceeded (budget): {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The stored evidence, laws or provider do not match the data digest.
    #[error("completion data identity mismatch")]
    DataIdentityMismatch,
    /// The recomputed report differs from the stored report.
    #[error("completion report does not replay")]
    ReportMismatch,
    /// The stored laws disagree with the supplied-exact provider.
    #[error("completion provider mismatch: {0}")]
    ProviderMismatch(&'static str),
    /// The report was truncated by cancellation, which no consumer can
    /// reproduce: it is never exported.
    #[error("a report truncated by cancellation cannot be independently verified")]
    CancelledNotReplayable,
}

impl CpdagArtifactError {
    /// The registered reason code and `cpdag_scenarios.*` detail of the refusal.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str) {
        match self {
            Self::UnsupportedSemantics(_) => {
                (reason_code!("route_not_supported"), "cpdag_scenarios.unsupported_semantics")
            }
            Self::LimitsExceeded(_) => {
                (reason_code!("transport_budget_cancel"), "cpdag_scenarios.consumer_limit_exceeded")
            }
            Self::PremisesMismatch => {
                (reason_code!("invalid_argument"), "cpdag_scenarios.premises_mismatch")
            }
            Self::DataIdentityMismatch => {
                (reason_code!("invalid_argument"), "cpdag_scenarios.data_identity_mismatch")
            }
            Self::ReportMismatch => {
                (reason_code!("invalid_argument"), "cpdag_scenarios.report_replay_mismatch")
            }
            Self::ProviderMismatch(_) => {
                (reason_code!("invalid_argument"), "cpdag_scenarios.provider_mismatch")
            }
            Self::CancelledNotReplayable => {
                (reason_code!("cancelled_no_claim"), "cpdag_scenarios.cancelled_not_exportable")
            }
        }
    }
}

impl From<CpdagArtifactError> for IoError {
    fn from(error: CpdagArtifactError) -> Self {
        let (code, detail) = error.refusal();
        Self::Refused { code, message: format!("{detail}: {error}") }
    }
}

fn cpdag_error(error: CpdagScenarioError) -> IoError {
    match error {
        CpdagScenarioError::Refused(refusal) => scenario_refusal(refusal),
        CpdagScenarioError::Identification(error) => error.into(),
    }
}

/// Consumer bounds. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct CpdagConsumeLimits {
    /// Largest shared enumeration-and-decision budget (operations and depth) the
    /// consumer will replay. A producer memory limit above the consumer
    /// context's hard limit is refused too.
    pub search_budget: SearchLimits,
    /// Largest exact-evaluation limits the consumer will replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
}

impl Default for CpdagConsumeLimits {
    fn default() -> Self {
        let sid = antecedent_identify::SidLimits::default();
        Self {
            search_budget: SearchLimits { operations: sid.steps, depth: sid.depth },
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
        }
    }
}

/// The stored CPDAG: variables ascending, `(parent, child)` directed edges and
/// `(low, high)` undirected edges, each sorted.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CpdagWire {
    /// Variable ids, ascending.
    pub variables: Vec<u32>,
    /// Directed `(parent, child)` edges, sorted.
    pub directed: Vec<(u32, u32)>,
    /// Undirected `(low, high)` edges, sorted.
    pub undirected: Vec<(u32, u32)>,
}

impl CpdagWire {
    /// Encode a static CPDAG over variable ids, independent of node and edge
    /// insertion order.
    ///
    /// # Errors
    /// A node that is not a static variable, or an edge with unknown endpoints
    /// or a mark that is neither directed nor undirected.
    pub fn from_cpdag(cpdag: &Cpdag) -> Result<Self, IoError> {
        let mut dense = Vec::with_capacity(cpdag.node_count());
        for node in cpdag.nodes() {
            match node {
                NodeRef::Static(variable) => dense.push(variable.raw()),
                _ => {
                    return Err(IoError::Convert(
                        "a CPDAG over static variables is required".into(),
                    ));
                }
            }
        }
        let variable = |id: DenseNodeId| dense.get(id.as_usize()).copied();
        let (mut directed, mut undirected) = (Vec::new(), Vec::new());
        for edge in cpdag.edges() {
            let (Some(a), Some(b)) = (variable(edge.a), variable(edge.b)) else {
                return Err(IoError::Convert("a CPDAG edge names an unknown node".into()));
            };
            if let Some((from, to)) = edge.parent_child() {
                match (variable(from), variable(to)) {
                    (Some(from), Some(to)) => directed.push((from, to)),
                    _ => return Err(IoError::Convert("a CPDAG edge names an unknown node".into())),
                }
            } else if edge.is_undirected() {
                undirected.push((a.min(b), a.max(b)));
            } else {
                return Err(IoError::Convert(
                    "a CPDAG edge is neither directed nor undirected".into(),
                ));
            }
        }
        let mut variables = dense;
        variables.sort_unstable();
        directed.sort_unstable();
        undirected.sort_unstable();
        Ok(Self { variables, directed, undirected })
    }

    /// Rebuild the CPDAG (nodes added in ascending variable order).
    ///
    /// # Errors
    /// More nodes than the completion route accepts, a repeated variable, or an
    /// edge naming a variable that is not a node.
    pub fn to_cpdag(&self) -> Result<Cpdag, IoError> {
        if self.variables.len() > CPDAG_MAX_NODES {
            return Err(CpdagArtifactError::LimitsExceeded("cpdag node count").into());
        }
        let mut graph = Cpdag::empty();
        let mut ids = BTreeMap::new();
        for raw in &self.variables {
            let id = graph
                .add_node(NodeRef::Static(VariableId::from_raw(*raw)))
                .map_err(|e| IoError::Convert(e.to_string()))?;
            if ids.insert(*raw, id).is_some() {
                return Err(IoError::Convert("a CPDAG variable repeats".into()));
            }
        }
        let node = |raw: u32| {
            ids.get(&raw)
                .copied()
                .ok_or_else(|| IoError::Convert("a CPDAG edge names an unknown variable".into()))
        };
        for (from, to) in &self.directed {
            graph
                .insert_directed(node(*from)?, node(*to)?)
                .map_err(|e| IoError::Convert(e.to_string()))?;
        }
        for (a, b) in &self.undirected {
            graph
                .insert_undirected(node(*a)?, node(*b)?)
                .map_err(|e| IoError::Convert(e.to_string()))?;
        }
        Ok(graph)
    }
}

/// One evidence catalog bound to one completion.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompletionEvidenceWire {
    /// The completion the evidence is supplied for.
    pub completion: String,
    /// The completion the evidence's graph certificate names.
    pub certified_for: String,
    /// Identity of the evidence.
    pub identity: String,
    /// The catalog of laws and regimes for this completion.
    pub catalog: EvidenceCatalogWire,
}

/// One catalog the caller declares valid for every completion.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SharedEvidenceWire {
    /// Identity of the shared evidence.
    pub identity: String,
    /// The catalog.
    pub catalog: EvidenceCatalogWire,
}

/// How evidence reached the completions: `shared` (one catalog) or
/// `per_completion` (a catalog per completion identity).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CpdagEvidenceWire {
    /// `shared` or `per_completion`.
    pub mode: String,
    /// The shared evidence (mode `shared`).
    pub shared: Option<SharedEvidenceWire>,
    /// The per-completion bindings (mode `per_completion`).
    pub bindings: Vec<CompletionEvidenceWire>,
}

impl CpdagEvidenceWire {
    /// Encode the evidence of a decision.
    #[must_use]
    pub fn from_evidence(evidence: &CpdagEvidence) -> Self {
        match evidence {
            CpdagEvidence::Shared { evidence_identity, catalog } => Self {
                mode: "shared".into(),
                shared: Some(SharedEvidenceWire {
                    identity: evidence_identity.to_string(),
                    catalog: EvidenceCatalogWire::from_catalog(catalog),
                }),
                bindings: Vec::new(),
            },
            CpdagEvidence::PerCompletion(bindings) => Self {
                mode: "per_completion".into(),
                shared: None,
                bindings: bindings
                    .iter()
                    .map(|b| CompletionEvidenceWire {
                        completion: b.completion.to_string(),
                        certified_for: b.certified_for.to_string(),
                        identity: b.evidence_identity.to_string(),
                        catalog: EvidenceCatalogWire::from_catalog(&b.catalog),
                    })
                    .collect(),
            },
        }
    }

    /// Decode the evidence.
    ///
    /// # Errors
    /// An unknown mode, a mode with the other mode's fields, or an invalid catalog.
    pub fn to_evidence(&self) -> Result<CpdagEvidence, IoError> {
        let unsupported = |what| Err(CpdagArtifactError::UnsupportedSemantics(what).into());
        match (self.mode.as_str(), &self.shared) {
            ("shared", Some(shared)) if self.bindings.is_empty() => Ok(CpdagEvidence::Shared {
                evidence_identity: Arc::from(shared.identity.as_str()),
                catalog: shared.catalog.to_catalog()?,
            }),
            ("per_completion", None) => {
                let bindings = self
                    .bindings
                    .iter()
                    .map(|b| -> Result<CompletionEvidenceBinding, IoError> {
                        Ok(CompletionEvidenceBinding {
                            completion: Arc::from(b.completion.as_str()),
                            certified_for: Arc::from(b.certified_for.as_str()),
                            evidence_identity: Arc::from(b.identity.as_str()),
                            catalog: b.catalog.to_catalog()?,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(CpdagEvidence::PerCompletion(bindings))
            }
            ("shared" | "per_completion", _) => unsupported("evidence mode and fields disagree"),
            _ => unsupported("evidence mode"),
        }
    }
}

/// One completion in the stored receipt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompletionResultWire {
    /// Canonical completion identity.
    pub id: String,
    /// Sorted `(parent, child)` edges.
    pub edges: Vec<(u32, u32)>,
    /// Scenario status.
    pub status: String,
    /// Why the completion did not produce a point, when it did not.
    pub detail: Option<String>,
    /// Identity of the evidence bound to this completion, if any.
    pub evidence_identity: Option<String>,
}

impl CompletionResultWire {
    fn from_entry(entry: &CompletionReceiptEntry) -> Self {
        Self {
            id: entry.id.to_string(),
            edges: entry.edges.iter().map(|(a, b)| (a.raw(), b.raw())).collect(),
            status: entry.status.into(),
            detail: entry.detail.clone(),
            evidence_identity: entry.evidence_identity.as_ref().map(ToString::to_string),
        }
    }
}

fn receipt_wire(receipt: &SearchReceipt) -> ScenarioReceiptWire {
    ScenarioReceiptWire {
        stop: receipt.stop.code().into(),
        operations_limit: receipt.operations_limit,
        depth_limit: receipt.depth_limit,
        memory_limit_bytes: receipt.memory_limit_bytes,
        operations_consumed: receipt.operations_consumed,
        depth_reached: receipt.depth_reached,
        explored: receipt.explored.clone(),
        unevaluated: receipt.unevaluated.clone(),
    }
}

/// The stored report: every completion, the counts, the unweighted envelope and
/// the stop receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpdagReportWire {
    /// Identity of the supplied CPDAG.
    pub cpdag_identity: String,
    /// Every completion found, in identity order, whatever its status.
    pub completions: Vec<CompletionResultWire>,
    /// Completions a stop never enumerated (graphs unknown).
    pub not_enumerated: usize,
    /// Identified completions.
    pub identified: usize,
    /// Completions decided but not identified.
    pub unidentified: usize,
    /// Completions left unevaluated, including those never enumerated.
    pub unevaluated: usize,
    /// The supplied-scenario report of the completions found: statuses, proofs,
    /// points, per-status counts and the unweighted structural envelope.
    pub scenario_report: Option<ScenarioReportWire>,
    /// The first stop's receipt (explored versus not enumerated versus unevaluated).
    pub receipt: Option<ScenarioReceiptWire>,
}

impl CpdagReportWire {
    /// Encode a report with the proofs of the prepared completions.
    ///
    /// # Errors
    /// The report and the prepared set disagree about whether completions were found.
    pub fn from_report(
        prepared: &PreparedCpdagScenarios,
        report: &CpdagScenarioReport,
    ) -> Result<Self, IoError> {
        let scenario_report = match (prepared.prepared(), report.report.as_ref()) {
            (Some(set), Some(inner)) => Some(ScenarioReportWire::from_report(set, inner)),
            (None, None) => None,
            _ => {
                return Err(IoError::Convert("completion report and prepared set disagree".into()));
            }
        };
        Ok(Self {
            cpdag_identity: report.cpdag_identity.to_string(),
            completions: report.completions.iter().map(CompletionResultWire::from_entry).collect(),
            not_enumerated: report.not_enumerated,
            identified: report.identified,
            unidentified: report.unidentified,
            unevaluated: report.unevaluated,
            scenario_report,
            receipt: report.receipt.as_ref().map(receipt_wire),
        })
    }
}

/// Every premise the completion route runs on: the supplied CPDAG, the shared
/// schema and question, the per-completion evidence, the limits and the exact
/// laws with their request.
#[derive(Clone, Debug)]
pub struct CpdagScenarioPremises {
    /// The supplied CPDAG with no selection or latent structure.
    pub input: CpdagCompletionInput,
    /// The shared coordinate schema.
    pub coordinates: Arc<[ScenarioCoordinate]>,
    /// The shared classical question.
    pub query: ClassicalTransportQuery,
    /// How evidence reaches the completions.
    pub evidence: CpdagEvidence,
    /// The one budget bounding enumeration and decisions together.
    pub budget: SearchLimits,
    /// The hard memory limit the producer decided under.
    pub memory_limit_bytes: Option<u64>,
    /// Supplied exact laws.
    pub data: ExactTransportData,
    /// Target request.
    pub request: Assignment,
    /// Exact-evaluation limits.
    pub evaluation: ExactEvaluationLimits,
}

impl CpdagScenarioPremises {
    /// Enumerate, decide and compile every completion once.
    ///
    /// # Errors
    /// The refusals of the completion route and of the scenario preparation.
    pub fn prepare(&self, ctx: &ExecutionContext) -> Result<PreparedCpdagScenarios, IoError> {
        let decision = decide_cpdag_completions(
            &self.input,
            &self.coordinates,
            &self.query,
            &self.evidence,
            self.budget,
            ctx,
        )
        .map_err(cpdag_error)?;
        prepare_cpdag_scenarios(
            decision,
            self.data.clone(),
            self.request.clone(),
            self.evaluation,
            ctx,
        )
        .map_err(IoError::from)
    }
}

/// Versioned completion-scenario execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpdagCompletionArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The supplied CPDAG.
    pub cpdag: CpdagWire,
    /// The shared named coordinate schema, in variable order.
    pub coordinates: Vec<ScenarioCoordinateWire>,
    /// Shared (classical) question.
    pub query: ScenarioQueryWire,
    /// Operation limit of the shared enumeration-and-decision budget.
    pub search_operations: usize,
    /// Depth limit of the shared budget.
    pub search_depth: usize,
    /// Hard memory limit the producer ran under.
    pub search_memory_bytes: Option<u64>,
    /// Evidence bound to the completions.
    pub evidence: CpdagEvidenceWire,
    /// Provider identity: supplied exact laws.
    pub provider: String,
    /// Laws in canonical order.
    pub laws: Vec<ExactLawWire>,
    /// Support budget.
    pub max_support_rows: usize,
    /// Target request.
    pub request: Vec<(u32, ValueWire)>,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// The full report.
    pub report: CpdagReportWire,
    /// Digest of the CPDAG, schema, question, request and limits: its
    /// scientific identity.
    pub premises_digest: String,
    /// Digest of the evidence bindings and catalogs, the laws and the provider.
    /// Separate from the premises so refreshed data replaces only it.
    pub data_digest: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

fn canonical_laws(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = data
        .laws()
        .iter()
        .map(|law| {
            let wire = ExactLawWire::from_law(law);
            crate::to_cbor(&(&wire.population, wire.regime, &wire.interventions))
                .map(|key| (key, wire))
        })
        .collect::<Result<Vec<_>, _>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

fn premises_identity(wire: &CpdagCompletionArtifactWire) -> Result<String, IoError> {
    let mut cpdag = wire.cpdag.clone();
    cpdag.variables.sort_unstable();
    cpdag.directed.sort_unstable();
    cpdag.undirected.sort_unstable();
    let mut coordinates = wire.coordinates.clone();
    coordinates.sort_by_key(|c| c.variable);
    let mut request = wire.request.clone();
    request.sort_by_key(|(v, _)| *v);
    let budget = (
        (wire.search_operations, wire.search_depth, wire.search_memory_bytes),
        (wire.operation_limit, wire.depth_limit, wire.max_support_rows),
    );
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("cpdag_completion_scenarios_v1", cpdag, coordinates, &wire.query, request, budget),
    )?
    .to_hex())
}

fn data_identity(wire: &CpdagCompletionArtifactWire) -> Result<String, IoError> {
    let mut evidence = wire.evidence.clone();
    evidence.bindings.sort_by(|a, b| a.completion.cmp(&b.completion));
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("cpdag_completion_data_v1", &wire.provider, evidence, &wire.laws),
    )?
    .to_hex())
}

fn query_wire(query: &ClassicalTransportQuery) -> ScenarioQueryWire {
    ScenarioQueryWire {
        outcomes: query.outcomes.iter().map(|v| v.raw()).collect(),
        treatments: query.treatments.iter().map(|v| v.raw()).collect(),
        source: query.source.to_string(),
        target: query.target.to_string(),
        conditioned_on: None,
    }
}

fn stopped_by_cancellation(report: &CpdagScenarioReport) -> bool {
    let cancelled = |r: &SearchReceipt| r.stop == SearchStop::Cancelled;
    report.receipt.as_ref().is_some_and(cancelled)
        || report.report.as_ref().and_then(|r| r.receipt.as_ref()).is_some_and(cancelled)
}

fn check_exact_provider(
    data: &ExactTransportData,
    provider: &str,
) -> Result<(), CpdagArtifactError> {
    let mismatch = CpdagArtifactError::ProviderMismatch;
    if provider != SCENARIO_EXACT_PROVIDER {
        return Err(mismatch("the completion route takes supplied exact laws only"));
    }
    if data.laws().iter().any(|law| law.origin() != LawOrigin::SuppliedExact) {
        return Err(mismatch("supplied exact laws carry no fitted tables or samples"));
    }
    Ok(())
}

impl CpdagCompletionArtifactWire {
    /// Build an artifact from the premises, the prepared completions and the
    /// report they produced.
    ///
    /// # Errors
    /// The premises do not encode, or the report was truncated by cancellation
    /// ([`CpdagArtifactError::CancelledNotReplayable`]).
    pub fn checked(
        premises: &CpdagScenarioPremises,
        prepared: &PreparedCpdagScenarios,
        report: &CpdagScenarioReport,
    ) -> Result<Self, IoError> {
        if stopped_by_cancellation(report) {
            return Err(CpdagArtifactError::CancelledNotReplayable.into());
        }
        check_exact_provider(&premises.data, SCENARIO_EXACT_PROVIDER)?;
        let mut wire = Self {
            version: CPDAG_COMPLETION_ARTIFACT_VERSION,
            required_features: vec![CPDAG_COMPLETION_ARTIFACT_FEATURE.into()],
            cpdag: CpdagWire::from_cpdag(&premises.input.cpdag)?,
            coordinates: premises
                .coordinates
                .iter()
                .map(ScenarioCoordinateWire::from_coordinate)
                .collect(),
            query: query_wire(&premises.query),
            search_operations: premises.budget.operations,
            search_depth: premises.budget.depth,
            search_memory_bytes: premises.memory_limit_bytes,
            evidence: CpdagEvidenceWire::from_evidence(&premises.evidence),
            provider: SCENARIO_EXACT_PROVIDER.into(),
            laws: canonical_laws(&premises.data)?,
            max_support_rows: premises.data.max_support_rows(),
            request: premises
                .request
                .entries()
                .iter()
                .map(|(v, x)| (v.raw(), ValueWire::from_value(x)))
                .collect(),
            operation_limit: premises.evaluation.operations,
            depth_limit: premises.evaluation.depth,
            report: CpdagReportWire::from_report(prepared, report)?,
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.premises_digest = premises_identity(&wire)?;
        wire.data_digest = data_identity(&wire)?;
        Ok(wire)
    }

    /// The premises digest these stored premises would carry. A consumer never
    /// trusts it: re-sealing a mutated artifact with it still fails replay.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        premises_identity(self)
    }

    /// The data digest these stored evidence, laws and provider would carry.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        data_identity(self)
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version first.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], a decoding failure, or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != CPDAG_COMPLETION_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [CPDAG_COMPLETION_ARTIFACT_FEATURE] {
            return Err(CpdagArtifactError::UnsupportedSemantics("required features").into());
        }
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &CpdagConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(), CpdagArtifactError> {
        let exceeded = CpdagArtifactError::LimitsExceeded;
        if self.search_operations > limits.search_budget.operations
            || self.search_depth > limits.search_budget.depth
        {
            return Err(exceeded("completion search budget"));
        }
        if let (Some(producer), Some(consumer)) =
            (self.search_memory_bytes, ctx.memory.hard_limit_bytes)
        {
            if producer > consumer {
                return Err(exceeded("completion search memory"));
            }
        }
        if self.operation_limit > limits.evaluation.operations
            || self.depth_limit > limits.evaluation.depth
        {
            return Err(exceeded("evaluation limits"));
        }
        if self.max_support_rows > limits.max_support_rows {
            return Err(exceeded("support rows"));
        }
        if self.laws.len() > limits.max_laws {
            return Err(exceeded("law count"));
        }
        if self.cpdag.variables.len() > CPDAG_MAX_NODES {
            return Err(exceeded("cpdag node count"));
        }
        if self.report.completions.len() > MAX_STORED_ITEMS
            || self.evidence.bindings.len() > MAX_STORED_ITEMS
        {
            return Err(exceeded("stored completion count"));
        }
        Ok(())
    }

    /// Rebuild every premise the producer ran on, validated.
    ///
    /// # Errors
    /// A stored CPDAG, schema, catalog, law or request that does not rebuild, or
    /// laws that are not supplied exact laws.
    pub fn to_premises(&self) -> Result<CpdagScenarioPremises, IoError> {
        let coordinates = self
            .coordinates
            .iter()
            .map(ScenarioCoordinateWire::to_coordinate)
            .collect::<Result<Vec<_>, _>>()?;
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Arc<[_]>>();
        let query = ClassicalTransportQuery {
            outcomes: ids(&self.query.outcomes),
            treatments: ids(&self.query.treatments),
            source: Arc::from(self.query.source.as_str()),
            target: Arc::from(self.query.target.as_str()),
        };
        if self.query.conditioned_on.is_some() {
            return Err(CpdagArtifactError::UnsupportedSemantics("a conditioned question").into());
        }
        let laws = self
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let data = ExactTransportData::try_new(laws, self.max_support_rows)
            .map_err(|e| IoError::Convert(e.to_string()))?;
        check_exact_provider(&data, &self.provider)?;
        Ok(CpdagScenarioPremises {
            input: CpdagCompletionInput::new(self.cpdag.to_cpdag()?),
            coordinates: Arc::from(coordinates),
            query,
            evidence: self.evidence.to_evidence()?,
            budget: SearchLimits { operations: self.search_operations, depth: self.search_depth },
            memory_limit_bytes: self.search_memory_bytes,
            data,
            request: Assignment::from_pairs(
                self.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
            ),
            evaluation: ExactEvaluationLimits {
                operations: self.operation_limit,
                depth: self.depth_limit,
            },
        })
    }

    /// Decode and replay everything, accepting only an identical report.
    ///
    /// # Errors
    /// A limit, digest, reconstruction, decision or report mismatch.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: CpdagConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, CpdagScenarioReport), IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits, ctx)?;
        if premises_identity(&wire)? != wire.premises_digest {
            return Err(CpdagArtifactError::PremisesMismatch.into());
        }
        if data_identity(&wire)? != wire.data_digest {
            return Err(CpdagArtifactError::DataIdentityMismatch.into());
        }
        let premises = wire.to_premises()?;
        // Replay under the producer's memory limit, which the consumer's own
        // hard limit already bounds, so a memory truncation reproduces.
        let mut replay = ctx.clone();
        if wire.search_memory_bytes.is_some() {
            replay.memory.hard_limit_bytes = wire.search_memory_bytes;
        }
        let ctx = &replay;
        let prepared = premises.prepare(ctx)?;
        let report = prepared.evaluate(ctx)?;
        let replayed = CpdagReportWire::from_report(&prepared, &report)?;
        if crate::to_cbor(&replayed)? != crate::to_cbor(&wire.report)? {
            return Err(CpdagArtifactError::ReportMismatch.into());
        }
        Ok((wire, report))
    }
}
