//! Independent artifacts for finite transport scenario sets.
//!
//! Format version 1. The artifact retains every scenario, failed as well as
//! successful. A consumer trusts nothing: it rebuilds the scenario set, question,
//! catalog and laws, re-decides every scenario under the producer's recorded
//! limits (which must not exceed its own, and which it needs to reproduce any
//! budget truncation), recompiles and re-evaluates, and accepts only a report
//! identical to the stored one: statuses, proofs, points, masses, envelope,
//! weighted report and receipt.

use crate::{
    IoError, admg_from_wire, admg_to_wire, exact_law_wire::ExactLawWire,
    mz_transport_artifact::MzTransportPointWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, wire::AdmgWire,
};
use antecedent_core::{ExecutionContext, IdentityDomain, SearchLimits, VariableId};
use antecedent_estimate::transport_scenarios::{PreparedScenarioSet, ScenarioSetReport};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    ClassicalTransportQuery, SidLimits,
    sid::SidDerivationRecord,
    sid::scenarios::{
        ScenarioOutcome, TransportScenario, TransportScenarioSet, decide_transport_scenarios,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const TRANSPORT_SCENARIO_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const TRANSPORT_SCENARIO_ARTIFACT_FEATURE: &str = "transport_scenario_envelope_v1";

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
}

/// Consumer bounds. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct TransportScenarioConsumeLimits {
    /// Largest per-scenario identification limits the consumer will replay.
    pub identification: SidLimits,
    /// Largest scenario budget the consumer will replay.
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
        Self {
            identification: SidLimits::default(),
            scenario_budget: SearchLimits { operations: 64, depth: 1 },
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
        }
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
    /// Scenarios decided before the stop.
    pub operations_consumed: Option<usize>,
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
    pub fn from_report(prepared: &PreparedScenarioSet, report: &ScenarioSetReport) -> Self {
        let proofs = prepared
            .decision()
            .decisions
            .iter()
            .map(|d| match &d.outcome {
                ScenarioOutcome::Identified(f) => Some(f.derivation().to_record()),
                _ => None,
            })
            .collect::<Vec<_>>();
        Self {
            scenarios: report
                .scenarios
                .iter()
                .zip(proofs)
                .map(|(s, proof)| ScenarioResultWire {
                    name: s.name.to_string(),
                    weight: s.weight,
                    status: s.status.into(),
                    detail: s.detail.clone(),
                    proof,
                    point: s.distribution.as_ref().map(MzTransportPointWire::from_distribution),
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
                operations_consumed: r.operations_consumed,
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
    /// Shared question.
    pub query: ScenarioQueryWire,
    /// Per-scenario identification step limit the producer ran under.
    pub identification_steps: usize,
    /// Per-scenario identification depth limit.
    pub identification_depth: usize,
    /// Scenario budget.
    pub scenario_budget: usize,
    /// Evidence catalog shared by every scenario.
    pub catalog: EvidenceCatalogWire,
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
    pub report: ScenarioReportWire,
    /// Digest of the scenario set, weights and question: its scientific identity.
    pub premises_digest: String,
}

/// Scientific identity of a scenario set and question: canonical scenarios with
/// their graphs, selections and weights, and the question.
///
/// # Errors
/// Encoding failure.
pub fn scenario_set_identity(
    scenarios: &[ScenarioWire],
    query: &ScenarioQueryWire,
) -> Result<String, IoError> {
    let mut scenarios = scenarios.to_vec();
    for scenario in &mut scenarios {
        scenario.graph.directed.sort_unstable();
        scenario.graph.bidirected.sort_unstable();
        scenario.selections.sort_unstable();
    }
    scenarios.sort_by(|a, b| a.name.cmp(&b.name));
    let weights = scenarios.iter().map(|s| s.weight.map(f64::to_bits)).collect::<Vec<_>>();
    let shape = scenarios.iter().map(|s| (&s.name, &s.graph, &s.selections)).collect::<Vec<_>>();
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("transport_scenarios_v1", shape, weights, query),
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

fn query_wire(query: &ClassicalTransportQuery) -> ScenarioQueryWire {
    ScenarioQueryWire {
        outcomes: query.outcomes.iter().map(|v| v.raw()).collect(),
        treatments: query.treatments.iter().map(|v| v.raw()).collect(),
        source: query.source.to_string(),
        target: query.target.to_string(),
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
    /// Build an artifact from a prepared set and the report it produced.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn checked(
        prepared: &PreparedScenarioSet,
        query: &ClassicalTransportQuery,
        catalog: &antecedent_core::EvidenceCatalog,
        identification: SidLimits,
        scenario_budget: SearchLimits,
        report: &ScenarioSetReport,
    ) -> Result<Self, IoError> {
        let scenarios = scenario_wires(&prepared.decision().set)?;
        let query = query_wire(query);
        let premises_digest = scenario_set_identity(&scenarios, &query)?;
        let limits = prepared.limits();
        Ok(Self {
            version: TRANSPORT_SCENARIO_ARTIFACT_VERSION,
            required_features: vec![TRANSPORT_SCENARIO_ARTIFACT_FEATURE.into()],
            scenarios,
            query,
            identification_steps: identification.steps,
            identification_depth: identification.depth,
            scenario_budget: scenario_budget.operations,
            catalog: EvidenceCatalogWire::from_catalog(catalog),
            laws: canonical_laws(prepared.data())?,
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
            premises_digest,
        })
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
        if peek.version != TRANSPORT_SCENARIO_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [TRANSPORT_SCENARIO_ARTIFACT_FEATURE] {
            return Err(
                TransportScenarioArtifactError::UnsupportedSemantics("required features").into()
            );
        }
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &TransportScenarioConsumeLimits,
    ) -> Result<(), TransportScenarioArtifactError> {
        let exceeded = TransportScenarioArtifactError::LimitsExceeded;
        if self.identification_steps > limits.identification.steps
            || self.identification_depth > limits.identification.depth
        {
            return Err(exceeded("identification limits"));
        }
        if self.scenario_budget > limits.scenario_budget.operations {
            return Err(exceeded("scenario budget"));
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
        wire.check_limits(&limits)?;
        if scenario_set_identity(&wire.scenarios, &wire.query)? != wire.premises_digest {
            return Err(TransportScenarioArtifactError::PremisesMismatch.into());
        }
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
                })
            })
            .collect::<Result<Vec<_>, IoError>>()?;
        let set = TransportScenarioSet::try_new(scenarios)?;
        let query = ClassicalTransportQuery {
            outcomes: ids(&wire.query.outcomes).into(),
            treatments: ids(&wire.query.treatments).into(),
            source: Arc::from(wire.query.source.as_str()),
            target: Arc::from(wire.query.target.as_str()),
        };
        let catalog = wire.catalog.to_catalog()?;
        let laws = wire
            .laws
            .iter()
            .map(ExactLawWire::to_law)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let data = ExactTransportData::try_new(laws, wire.max_support_rows)
            .map_err(|e| IoError::Convert(e.to_string()))?;
        let request = Assignment::from_pairs(
            wire.request.iter().map(|(v, x)| (VariableId::from_raw(*v), x.to_value())),
        );
        let decision = decide_transport_scenarios(
            &set,
            &query,
            &catalog,
            SidLimits { steps: wire.identification_steps, depth: wire.identification_depth },
            SearchLimits { operations: wire.scenario_budget, depth: 1 },
            ctx,
        )?;
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
