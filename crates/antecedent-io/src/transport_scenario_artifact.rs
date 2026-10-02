//! Independent artifacts for finite transport scenario sets.
//!
//! Format version 1. The artifact retains every scenario, failed as well as
//! successful, and the shared named coordinate schema. A consumer trusts
//! nothing: it rebuilds the scenario set, schema, question, catalog and laws,
//! re-decides every scenario under the producer's recorded limits (operations,
//! depth and memory of the one shared search budget; none may exceed its own,
//! and it needs them to reproduce any budget truncation), recompiles and
//! re-evaluates, and accepts
//! only a report identical to the stored one: statuses, proofs, points, masses,
//! envelope, weighted report and receipt. Two digests guard the stored
//! premises: the scientific premises digest (scenarios, weights, schema,
//! question, request, budgets) and a separate data-identity digest (catalog,
//! laws, provider, sample summaries), so refreshed data replaces only the
//! latter. For an empirical provider the artifact stores the fitted plug-in
//! tables and summaries of the samples, not the sample rows: the consumer
//! re-decides, recompiles and re-evaluates every scenario from those tables
//! and checks each table against exactly one sample summary of the same
//! world, snapshot and size, but it does not refit the tables. A report cut
//! short by cancellation is never exported: nothing recorded lets a consumer
//! reproduce where the interruption fell.
//!
//! Format version 2 (2.2B B1) is the same artifact for a conditional question
//! `P*(y | do(x), w)`: the question carries `conditioned_on`, an identified
//! scenario carries its checked conditional record (rule-2 moves, remaining
//! conditioned set and the reduced joint's sID record) in place of `proof`, a
//! not-certified scenario carries its inspection-only obstruction candidate, and
//! a structurally unidentified one its verified two-model witness.
//! Every version-2 field is absent from a classical artifact, which the writer
//! still emits as version 1 byte for byte (its digests are unchanged), so a
//! version-1-only reader keeps reading every classical artifact and refuses a
//! conditional one by its version (`UnsupportedVersion { version: 2 }`) before
//! decoding anything else. This reader accepts both and refuses a version-1
//! artifact that carries a version-2 field, and a version-2 artifact without a
//! conditional question.

use crate::{
    IoError, admg_from_wire, admg_to_wire, exact_law_wire::ExactLawWire,
    mz_transport_artifact::MzTransportPointWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, transport_grid_wire::SampleSummary,
    wire::AdmgWire,
};
use antecedent_core::{ExecutionContext, IdentityDomain, SearchLimits, VariableDomain, VariableId};
use antecedent_estimate::transport_scenarios::{
    PreparedScenarioSet, SCENARIO_EXACT_PROVIDER, ScenarioSetReport,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData, LawOrigin};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ClassicalTransportQuery, ConditionalNonTransportabilityRecord, ConditionalObstructionRecord,
    ConditionalTransportQuery, ConditionalTransportRecord, SidLimits,
    sid::SidDerivationRecord,
    sid::scenarios::{
        ScenarioCoordinate, ScenarioOutcome, ScenarioQuestion, ScenarioSetRefusal,
        TransportScenario, TransportScenarioSet, decide_scenario_question,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const TRANSPORT_SCENARIO_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const TRANSPORT_SCENARIO_ARTIFACT_FEATURE: &str = "transport_scenario_envelope_v1";
/// The format version of an artifact whose question is conditional (2.2B B1).
pub const TRANSPORT_SCENARIO_CONDITIONAL_ARTIFACT_VERSION: u32 = 2;
/// The feature marker a version-2 artifact requires beside
/// [`TRANSPORT_SCENARIO_ARTIFACT_FEATURE`].
pub const TRANSPORT_SCENARIO_CONDITIONAL_FEATURE: &str = "transport_scenario_admg_conditional_v2";

/// Why a scenario artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TransportScenarioArtifactError {
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A recorded limit or stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The recomputed report differs from the stored report.
    #[error("scenario report does not replay")]
    ReportMismatch,
    /// The stored laws disagree with the recorded provider or its samples.
    #[error("scenario provider mismatch: {0}")]
    ProviderMismatch(&'static str),
    /// The stored catalog, laws, provider or sample summaries do not match the
    /// data-identity digest.
    #[error("scenario data identity mismatch")]
    DataIdentityMismatch,
    /// The report was truncated by cancellation, which no consumer can
    /// reproduce: it is never exported.
    #[error("a report truncated by cancellation cannot be independently verified")]
    CancelledNotReplayable,
}

/// A refused scenario set, schema or law as a reason-coded refusal.
#[must_use]
#[doc(hidden)]
#[allow(clippy::needless_pass_by_value)] // A `map_err` adapter.
pub fn scenario_refusal(refusal: ScenarioSetRefusal) -> IoError {
    IoError::Refused {
        code: refusal.code,
        message: format!("{}: {}", refusal.detail, refusal.message),
    }
}

/// Consumer bounds. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct TransportScenarioConsumeLimits {
    /// Largest shared search budget (operations and depth) the consumer will
    /// replay. A producer memory limit above the consumer context's hard limit
    /// is refused too.
    pub scenario_budget: SearchLimits,
    /// Largest exact-evaluation limits the consumer will replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
}

impl Default for TransportScenarioConsumeLimits {
    fn default() -> Self {
        let sid = SidLimits::default();
        Self {
            scenario_budget: SearchLimits { operations: sid.steps, depth: sid.depth },
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
        }
    }
}

/// One stored coordinate of the shared schema.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioCoordinateWire {
    /// Variable id.
    pub variable: u32,
    /// Declared name.
    pub name: String,
    /// `unspecified`, `continuous`, `binary`, `count` or `categorical`.
    pub domain: String,
    /// Cardinality of a categorical domain.
    pub cardinality: Option<u32>,
    /// Declared unit.
    pub unit: Option<String>,
}

impl ScenarioCoordinateWire {
    /// Encode one coordinate.
    #[must_use]
    pub fn from_coordinate(c: &ScenarioCoordinate) -> Self {
        let (domain, cardinality) = match c.domain {
            VariableDomain::Unspecified => ("unspecified", None),
            VariableDomain::Continuous => ("continuous", None),
            VariableDomain::Binary => ("binary", None),
            VariableDomain::Count => ("count", None),
            VariableDomain::Categorical { cardinality } => ("categorical", Some(cardinality)),
        };
        Self {
            variable: c.variable.raw(),
            name: c.name.to_string(),
            domain: domain.into(),
            cardinality,
            unit: c.unit.as_ref().map(ToString::to_string),
        }
    }

    /// Decode one coordinate.
    ///
    /// # Errors
    /// An unknown domain tag or a cardinality on a non-categorical domain.
    pub fn to_coordinate(&self) -> Result<ScenarioCoordinate, IoError> {
        let domain = match (self.domain.as_str(), self.cardinality) {
            ("unspecified", None) => VariableDomain::Unspecified,
            ("continuous", None) => VariableDomain::Continuous,
            ("binary", None) => VariableDomain::Binary,
            ("count", None) => VariableDomain::Count,
            ("categorical", Some(cardinality)) => VariableDomain::Categorical { cardinality },
            _ => return Err(IoError::Convert("invalid scenario coordinate domain".into())),
        };
        Ok(ScenarioCoordinate {
            variable: VariableId::from_raw(self.variable),
            name: std::sync::Arc::from(self.name.as_str()),
            domain,
            unit: self.unit.as_deref().map(std::sync::Arc::from),
        })
    }
}

/// One stored scenario.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioWire {
    /// Scenario name.
    pub name: String,
    /// Causal graph.
    pub graph: AdmgWire,
    /// Selection targets.
    pub selections: Vec<u32>,
    /// Declared weight.
    pub weight: Option<f64>,
}

/// The shared transport question.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioQueryWire {
    /// Outcomes.
    pub outcomes: Vec<u32>,
    /// Treatments.
    pub treatments: Vec<u32>,
    /// Source population.
    pub source: String,
    /// Target population.
    pub target: String,
    /// Conditioned variables of a conditional question (version 2 only;
    /// absent, and not encoded, for a classical question).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditioned_on: Option<Vec<u32>>,
}

/// One scenario's stored result.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioResultWire {
    /// Scenario name.
    pub name: String,
    /// Declared weight.
    pub weight: Option<f64>,
    /// Status.
    pub status: String,
    /// Why no point was produced, when none was.
    pub detail: Option<String>,
    /// Checked derivation of an identified scenario.
    pub proof: Option<SidDerivationRecord>,
    /// Point of an identified, evaluated scenario.
    pub point: Option<MzTransportPointWire>,
    /// Checked conditional record of an identified conditional scenario
    /// (version 2 only; not encoded when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conditional_proof: Option<ConditionalTransportRecord>,
    /// Inspection-only obstruction candidate of a not-certified conditional
    /// scenario (version 2 only; not encoded when absent). Never a proof.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<ConditionalObstructionRecord>,
    /// Exactly verified two-model witness of a structurally unidentified
    /// conditional scenario (version 2 only; not encoded when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_transportability: Option<ConditionalNonTransportabilityRecord>,
}

/// One outcome's envelope.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MeanEnvelopeWire {
    /// Outcome.
    pub outcome: u32,
    /// Smallest identified mean.
    pub lower: f64,
    /// Largest identified mean.
    pub upper: f64,
    /// Scenario attaining `lower`.
    pub lower_scenario: String,
    /// Scenario attaining `upper`.
    pub upper_scenario: String,
}

/// Stored structural envelope.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeWire {
    /// Identified scenarios ranged over.
    pub scenarios: Vec<String>,
    /// Mean envelopes.
    pub means: Vec<MeanEnvelopeWire>,
    /// Atom-wise ranges when atoms agree.
    pub atoms: Option<Vec<(f64, f64)>>,
    /// Reading of the envelope.
    pub interpretation: String,
}

/// Stored declared-weight report.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WeightedWire {
    /// Weight of identified scenarios.
    pub identified_mass: f64,
    /// Every other scenario's weight plus residual.
    pub unaccounted_mass: f64,
    /// `(outcome, Σ w·mean)` over identified scenarios.
    pub identified_weighted_sums: Vec<(u32, f64)>,
    /// `(outcome, lower, upper)` with unaccounted mass at the support limits.
    pub ranges: Option<Vec<(u32, f64, f64)>>,
    /// Reading of the report.
    pub interpretation: String,
}

/// Stored scenario-budget receipt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioReceiptWire {
    /// Stop code.
    pub stop: String,
    /// Scenario budget.
    pub operations_limit: usize,
    /// Depth limit.
    pub depth_limit: usize,
    /// Hard memory limit in force.
    pub memory_limit_bytes: Option<u64>,
    /// Operations charged before the stop.
    pub operations_consumed: Option<usize>,
    /// Deepest level charged before the stop.
    pub depth_reached: Option<usize>,
    /// Scenarios decided, in order.
    pub explored: Vec<String>,
    /// Scenarios left unevaluated.
    pub unevaluated: Vec<String>,
}

/// The stored report: every scenario, masses, envelope, weights and receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioReportWire {
    /// Every scenario in canonical order.
    pub scenarios: Vec<ScenarioResultWire>,
    /// `(status, count, mass)` in canonical status order.
    pub masses: Vec<(String, usize, Option<f64>)>,
    /// Undeclared mass.
    pub residual_mass: Option<f64>,
    /// Envelope over identified scenarios.
    pub envelope: Option<EnvelopeWire>,
    /// Declared-weight report.
    pub weighted: Option<WeightedWire>,
    /// Scenario-budget receipt.
    pub receipt: Option<ScenarioReceiptWire>,
}

impl ScenarioReportWire {
    /// Encode a report together with the decision's proof records.
    #[must_use]
    pub(crate) fn from_report(prepared: &PreparedScenarioSet, report: &ScenarioSetReport) -> Self {
        let proofs = prepared
            .decision()
            .decisions
            .iter()
            .map(|d| match &d.outcome {
                ScenarioOutcome::Identified(f) => {
                    (Some(f.derivation().to_record()), None, None, None)
                }
                ScenarioOutcome::ConditionalIdentified(f) => {
                    (None, Some(f.derivation().to_record()), None, None)
                }
                ScenarioOutcome::ConditionalNotCertified { candidate, .. } => {
                    (None, None, candidate.as_ref().map(|c| c.to_record()), None)
                }
                ScenarioOutcome::ConditionalProvenNonTransportable(proof) => {
                    (None, None, None, Some(proof.to_record()))
                }
                _ => (None, None, None, None),
            })
            .collect::<Vec<_>>();
        Self {
            scenarios: report
                .scenarios
                .iter()
                .zip(proofs)
                .map(|(s, (proof, conditional_proof, candidate, non_transportability))| {
                    ScenarioResultWire {
                        name: s.name.to_string(),
                        weight: s.weight,
                        status: s.status.into(),
                        detail: s.detail.clone(),
                        proof,
                        point: s.distribution.as_ref().map(MzTransportPointWire::from_distribution),
                        conditional_proof,
                        candidate,
                        non_transportability,
                    }
                })
                .collect(),
            masses: report.masses.iter().map(|m| (m.status.to_owned(), m.count, m.mass)).collect(),
            residual_mass: report.residual_mass,
            envelope: report.envelope.as_ref().map(|e| EnvelopeWire {
                scenarios: e.scenarios.iter().map(ToString::to_string).collect(),
                means: e
                    .means
                    .iter()
                    .map(|m| MeanEnvelopeWire {
                        outcome: m.outcome.raw(),
                        lower: m.lower,
                        upper: m.upper,
                        lower_scenario: m.lower_scenario.to_string(),
                        upper_scenario: m.upper_scenario.to_string(),
                    })
                    .collect(),
                atoms: e.atoms.clone(),
                interpretation: e.interpretation.into(),
            }),
            weighted: report.weighted.as_ref().map(|w| WeightedWire {
                identified_mass: w.identified_mass,
                unaccounted_mass: w.unaccounted_mass,
                identified_weighted_sums: w
                    .identified_weighted_sums
                    .iter()
                    .map(|(v, x)| (v.raw(), *x))
                    .collect(),
                ranges: w
                    .ranges
                    .as_ref()
                    .map(|r| r.iter().map(|(v, lo, hi)| (v.raw(), *lo, *hi)).collect()),
                interpretation: w.interpretation.into(),
            }),
            receipt: report.receipt.as_ref().map(|r| ScenarioReceiptWire {
                stop: r.stop.code().into(),
                operations_limit: r.operations_limit,
                depth_limit: r.depth_limit,
                memory_limit_bytes: r.memory_limit_bytes,
                operations_consumed: r.operations_consumed,
                depth_reached: r.depth_reached,
                explored: r.explored.clone(),
                unevaluated: r.unevaluated.clone(),
            }),
        }
    }
}

/// Versioned scenario-set execution with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportScenarioArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Scenarios in canonical order.
    pub scenarios: Vec<ScenarioWire>,
    /// The shared named coordinate schema, in variable order.
    pub coordinates: Vec<ScenarioCoordinateWire>,
    /// Shared question.
    pub query: ScenarioQueryWire,
    /// Operation limit of the shared search budget the producer ran under.
    pub scenario_operations: usize,
    /// Depth limit of the shared search budget.
    pub scenario_depth: usize,
    /// Hard memory limit the producer decided under.
    pub scenario_memory_bytes: Option<u64>,
    /// Evidence catalog shared by every scenario.
    pub catalog: EvidenceCatalogWire,
    /// Provider identity: supplied exact laws or the empirical plug-in.
    pub provider: String,
    /// Laws in canonical order: supplied, or fitted by the empirical plug-in.
    pub laws: Vec<ExactLawWire>,
    /// Summaries of the samples the empirical plug-in fitted.
    pub samples: Vec<SampleSummary>,
    /// Support budget.
    pub max_support_rows: usize,
    /// Target request.
    pub request: Vec<(u32, ValueWire)>,
    /// Evaluation operation limit.
    pub operation_limit: usize,
    /// Evaluation depth limit.
    pub depth_limit: usize,
    /// The full report.
    pub report: ScenarioReportWire,
    /// Digest of the scenario set, weights, schema, question, request and
    /// budgets: its scientific identity.
    pub premises_digest: String,
    /// Digest of the data identity: the catalog (regimes, environments and
    /// snapshot bindings), every law, the provider and the fitted-sample
    /// summaries. Separate from the premises so refreshed data replaces it.
    pub data_digest: String,
}

/// Data identity of a scenario-set execution: the evidence catalog with its
/// regimes, environments and snapshot bindings, every stored law (its
/// population, regime, interventions, axes, snapshot, tolerances, origin and
/// masses), the provider identity and the fitted-sample summaries (population,
/// regime, snapshot, interventions, size and content digest). Deliberately not
/// part of [`scenario_set_identity`]: refresh replaces data and keeps the
/// scientific premises.
///
/// # Errors
/// Encoding failure.
pub(crate) fn scenario_data_identity(
    wire: &TransportScenarioArtifactWire,
) -> Result<String, IoError> {
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("transport_scenarios_data_v1", &wire.provider, &wire.catalog, &wire.laws, &wire.samples),
    )?
    .to_hex())
}

/// Scientific identity of a scenario-set execution: canonical scenarios (sorted
/// by their unique names) with their graphs, selections and weights; the shared
/// named coordinate schema (names, domains, cardinalities, units), which also
/// binds the variable names a report is read with; the question and the target
/// request; and every budget the report depends on (the shared search budget
/// with its memory limit, and evaluation).
///
/// # Errors
/// Encoding failure.
pub(crate) fn scenario_set_identity(
    wire: &TransportScenarioArtifactWire,
) -> Result<String, IoError> {
    let mut scenarios = wire.scenarios.clone();
    for scenario in &mut scenarios {
        scenario.graph.directed.sort_unstable();
        scenario.graph.bidirected.sort_unstable();
        scenario.selections.sort_unstable();
    }
    scenarios.sort_by(|a, b| a.name.cmp(&b.name));
    let mut coordinates = wire.coordinates.clone();
    coordinates.sort_by_key(|c| c.variable);
    let mut request = wire.request.clone();
    request.sort_by_key(|(v, _)| *v);
    let weights = scenarios.iter().map(|s| s.weight.map(f64::to_bits)).collect::<Vec<_>>();
    let shape = scenarios.iter().map(|s| (&s.name, &s.graph, &s.selections)).collect::<Vec<_>>();
    let budget = (
        (wire.scenario_operations, wire.scenario_depth, wire.scenario_memory_bytes),
        (wire.operation_limit, wire.depth_limit, wire.max_support_rows),
    );
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("transport_scenarios_v1", shape, weights, coordinates, &wire.query, request, budget),
    )?
    .to_hex())
}

fn scenario_wires(set: &TransportScenarioSet) -> Result<Vec<ScenarioWire>, IoError> {
    set.scenarios()
        .iter()
        .map(|s| {
            Ok(ScenarioWire {
                name: s.name.to_string(),
                graph: admg_to_wire(s.diagram.causal_graph())?,
                selections: s.diagram.selection_targets().iter().map(|v| v.raw()).collect(),
                weight: s.weight,
            })
        })
        .collect()
}

/// Every non-supplied law of an empirical provider must be accounted for by
/// exactly one sample summary of the same world, snapshot and size; a supplied
/// exact provider carries no samples.
fn check_provider(
    provider: &str,
    data: &ExactTransportData,
    samples: &[SampleSummary],
) -> Result<(), TransportScenarioArtifactError> {
    let mismatch = TransportScenarioArtifactError::ProviderMismatch;
    let fitted = data.laws().iter().filter(|law| law.origin() != LawOrigin::SuppliedExact);
    if provider == SCENARIO_EXACT_PROVIDER {
        if fitted.count() != 0 || !samples.is_empty() {
            return Err(mismatch("supplied exact laws carry no fitted tables or samples"));
        }
        return Ok(());
    }
    if provider != antecedent_estimate::EMPIRICAL_TABLE_PLUGIN {
        return Err(mismatch("unknown provider"));
    }
    if samples.is_empty() {
        return Err(mismatch("an empirical provider needs a sample"));
    }
    let mut unmatched = samples.to_vec();
    for law in fitted {
        if law.origin() != LawOrigin::EmpiricalPlugin {
            return Err(mismatch("only empirical plug-in tables are fitted"));
        }
        let wire = ExactLawWire::metadata(law);
        // A plug-in table is a frequency table of its sample: its counts sum
        // to the sample size, or every probability times the size is whole.
        let sized = |n: u32| match law.empirical_counts() {
            Some(counts) => counts.iter().sum::<u64>() == u64::from(n),
            None => {
                n > 0
                    && law.probabilities().iter().all(|p| {
                        let count = p * f64::from(n);
                        (count - count.round()).abs()
                            <= (8.0 * f64::EPSILON * f64::from(n)).max(1e-7)
                    })
            }
        };
        let index = unmatched
            .iter()
            .position(|s| {
                s.population == wire.population
                    && s.regime == wire.regime
                    && s.interventions == wire.interventions
                    && s.snapshot == wire.snapshot
                    && sized(s.n)
            })
            .ok_or(mismatch("a fitted table has no matching sample summary"))?;
        unmatched.swap_remove(index);
    }
    if !unmatched.is_empty() {
        return Err(mismatch("a sample summary has no fitted table"));
    }
    Ok(())
}

fn query_wire(question: &ScenarioQuestion) -> ScenarioQueryWire {
    let query = question.base();
    ScenarioQueryWire {
        outcomes: query.outcomes.iter().map(|v| v.raw()).collect(),
        treatments: query.treatments.iter().map(|v| v.raw()).collect(),
        source: query.source.to_string(),
        target: query.target.to_string(),
        conditioned_on: question
            .is_conditional()
            .then(|| question.conditioned_on().iter().map(|v| v.raw()).collect()),
    }
}

/// The stored question: classical, or conditional when it names a conditioned set.
fn question_of(wire: &ScenarioQueryWire) -> ScenarioQuestion {
    let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Arc<[_]>>();
    let base = ClassicalTransportQuery {
        outcomes: ids(&wire.outcomes),
        treatments: ids(&wire.treatments),
        source: Arc::from(wire.source.as_str()),
        target: Arc::from(wire.target.as_str()),
    };
    match &wire.conditioned_on {
        None => ScenarioQuestion::Classical(base),
        Some(conditioned) => ScenarioQuestion::Conditional(ConditionalTransportQuery {
            base,
            conditioned_on: ids(conditioned),
        }),
    }
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

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl TransportScenarioArtifactWire {
    /// Build an artifact from a prepared set and the report it produced. The
    /// limits recorded are the ones the decision ran under.
    ///
    /// A report truncated by an operation, depth or memory bound exports: the
    /// limits are recorded and the consumer reproduces the identical prefix. A
    /// report truncated by cancellation does not, because nothing recorded lets
    /// a consumer reproduce where the interruption fell.
    ///
    /// # Errors
    /// The premises do not encode, or the report was truncated by cancellation
    /// ([`TransportScenarioArtifactError::CancelledNotReplayable`]).
    pub fn checked(
        prepared: &PreparedScenarioSet,
        catalog: &antecedent_core::EvidenceCatalog,
        report: &ScenarioSetReport,
    ) -> Result<Self, IoError> {
        if report.receipt.as_ref().is_some_and(|r| r.stop == antecedent_core::SearchStop::Cancelled)
        {
            return Err(TransportScenarioArtifactError::CancelledNotReplayable.into());
        }
        let decision = prepared.decision();
        let decided = decision.limits;
        let limits = prepared.limits();
        let samples = prepared
            .empirical_input()
            .map(|input| input.samples.iter().map(SampleSummary::from_sample).collect())
            .transpose()?
            .unwrap_or_default();
        let conditional = decision.question.is_conditional();
        let mut wire = Self {
            version: if conditional {
                TRANSPORT_SCENARIO_CONDITIONAL_ARTIFACT_VERSION
            } else {
                TRANSPORT_SCENARIO_ARTIFACT_VERSION
            },
            required_features: if conditional {
                vec![
                    TRANSPORT_SCENARIO_ARTIFACT_FEATURE.into(),
                    TRANSPORT_SCENARIO_CONDITIONAL_FEATURE.into(),
                ]
            } else {
                vec![TRANSPORT_SCENARIO_ARTIFACT_FEATURE.into()]
            },
            scenarios: scenario_wires(&decision.set)?,
            coordinates: decision
                .set
                .schema()
                .iter()
                .map(ScenarioCoordinateWire::from_coordinate)
                .collect(),
            query: query_wire(&decision.question),
            scenario_operations: decided.budget.operations,
            scenario_depth: decided.budget.depth,
            scenario_memory_bytes: decided.memory_limit_bytes,
            catalog: EvidenceCatalogWire::from_catalog(catalog),
            provider: prepared.provider().into(),
            laws: canonical_laws(prepared.data())?,
            samples,
            max_support_rows: prepared.data().max_support_rows(),
            request: prepared
                .request()
                .entries()
                .iter()
                .map(|(v, x)| (v.raw(), ValueWire::from_value(x)))
                .collect(),
            operation_limit: limits.operations,
            depth_limit: limits.depth,
            report: ScenarioReportWire::from_report(prepared, report),
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.premises_digest = scenario_set_identity(&wire)?;
        wire.data_digest = scenario_data_identity(&wire)?;
        Ok(wire)
    }

    /// The data-identity digest these stored data would carry.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        scenario_data_identity(self)
    }

    /// The premises digest these stored premises would carry (the scientific
    /// identity of the scenario set; see the module docs). A consumer never
    /// trusts it: re-sealing a mutated artifact with it still fails replay.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        scenario_set_identity(self)
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
    /// Version 1 (classical) and version 2 (conditional) are accepted, each
    /// with exactly its own feature markers. A version-1 artifact carrying a
    /// conditioned set, a conditional record or a candidate, or a version-2
    /// artifact without a conditioned set, is refused as unsupported semantics.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], a decoding failure, or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        let conditional = match peek.version {
            TRANSPORT_SCENARIO_ARTIFACT_VERSION => false,
            TRANSPORT_SCENARIO_CONDITIONAL_ARTIFACT_VERSION => true,
            version => return Err(IoError::UnsupportedVersion { version }),
        };
        let wire: Self = crate::from_cbor(bytes)?;
        let unsupported =
            |what| Err(TransportScenarioArtifactError::UnsupportedSemantics(what).into());
        let features: &[&str] = if conditional {
            &[TRANSPORT_SCENARIO_ARTIFACT_FEATURE, TRANSPORT_SCENARIO_CONDITIONAL_FEATURE]
        } else {
            &[TRANSPORT_SCENARIO_ARTIFACT_FEATURE]
        };
        if wire.required_features != features {
            return unsupported("required features");
        }
        let carries_conditional = wire.query.conditioned_on.is_some()
            || wire.report.scenarios.iter().any(|s| {
                s.conditional_proof.is_some()
                    || s.candidate.is_some()
                    || s.non_transportability.is_some()
            });
        if !conditional && carries_conditional {
            return unsupported("a version 1 artifact carries a conditional question or record");
        }
        if conditional && wire.query.conditioned_on.as_ref().is_none_or(Vec::is_empty) {
            return unsupported("a version 2 artifact needs a conditioned set");
        }
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &TransportScenarioConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(), TransportScenarioArtifactError> {
        let exceeded = TransportScenarioArtifactError::LimitsExceeded;
        if self.scenario_operations > limits.scenario_budget.operations
            || self.scenario_depth > limits.scenario_budget.depth
        {
            return Err(exceeded("scenario budget"));
        }
        if let (Some(producer), Some(consumer)) =
            (self.scenario_memory_bytes, ctx.memory.hard_limit_bytes)
        {
            if producer > consumer {
                return Err(exceeded("scenario memory"));
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
        Ok(())
    }

    /// Decode and replay everything, accepting only an identical report.
    ///
    /// # Errors
    /// A limit, digest, reconstruction, decision or report mismatch.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: TransportScenarioConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, ScenarioSetReport), IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits, ctx)?;
        if scenario_set_identity(&wire)? != wire.premises_digest {
            return Err(TransportScenarioArtifactError::PremisesMismatch.into());
        }
        if scenario_data_identity(&wire)? != wire.data_digest {
            return Err(TransportScenarioArtifactError::DataIdentityMismatch.into());
        }
        let coordinates = wire
            .coordinates
            .iter()
            .map(ScenarioCoordinateWire::to_coordinate)
            .collect::<Result<Vec<_>, _>>()?;
        let coordinates = Arc::<[ScenarioCoordinate]>::from(coordinates);
        let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        let scenarios = wire
            .scenarios
            .iter()
            .map(|s| {
                let diagram = SelectionDiagram::try_new(
                    admg_from_wire(&s.graph)?,
                    Arc::<[VariableId]>::from(ids(&s.selections)),
                )
                .map_err(|e| IoError::Convert(e.to_string()))?;
                Ok(TransportScenario {
                    name: Arc::from(s.name.as_str()),
                    diagram,
                    weight: s.weight,
                    coordinates: Arc::clone(&coordinates),
                })
            })
            .collect::<Result<Vec<_>, IoError>>()?;
        let set = TransportScenarioSet::try_new(scenarios).map_err(scenario_refusal)?;
        let question = question_of(&wire.query);
        let catalog = wire.catalog.to_catalog()?;
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let data = ExactTransportData::try_new(laws, wire.max_support_rows)
            .map_err(|e| IoError::Convert(e.to_string()))?;
        check_provider(&wire.provider, &data, &wire.samples)?;
        // Replay under the producer's memory limit, which the consumer's own
        // hard limit already bounds, so a memory truncation reproduces.
        let mut replay = ctx.clone();
        if wire.scenario_memory_bytes.is_some() {
            replay.memory.hard_limit_bytes = wire.scenario_memory_bytes;
        }
        let ctx = &replay;
        let request = Assignment::from_pairs(
            wire.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        let decision = decide_scenario_question(
            &set,
            &question,
            &catalog,
            SearchLimits { operations: wire.scenario_operations, depth: wire.scenario_depth },
            ctx,
        )
        .map_err(|error| {
            if question.is_conditional() {
                crate::admg_conditional_transport_artifact::admg_conditional_identification_error(
                    error,
                )
            } else {
                error.into()
            }
        })?;
        let prepared = antecedent_estimate::transport_scenarios::prepare_transport_scenarios(
            decision,
            data,
            request,
            ExactEvaluationLimits { operations: wire.operation_limit, depth: wire.depth_limit },
            ctx,
        )?;
        let report = prepared.evaluate(ctx)?;
        let replayed = ScenarioReportWire::from_report(&prepared, &report);
        if crate::to_cbor(&replayed)? != crate::to_cbor(&wire.report)? {
            return Err(TransportScenarioArtifactError::ReportMismatch.into());
        }
        Ok((wire, report))
    }
}
