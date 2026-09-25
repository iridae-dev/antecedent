//! Failure snapshots and evidence-arrival checks for bounded z-transport.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, sync::Arc};

use antecedent_core::{EvidenceCatalog, EvidenceCatalogDelta, EvidenceKind, RegimeKind};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    bind_z_transport_catalog, decide_z_transport_with_catalog, identify_classical_transport,
    identify_z_transport, validate_z_experiment_family, BoundZTransportFunctional,
    ClassicalTransportQuery, ClassicalTransportResult, SidLimits, ZTransportDecision,
    ZTransportDerivation, ZTransportObstruction, ZTransportObstructionRecord,
    ZTransportProofInspection, ZTransportQuery,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{CandidateDesign, TransportEvidenceCandidate};

/// Why the original z-transport attempt did not yield an available result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZTransportFailureStatus {
    /// A checked formula exists, but one or more required joint laws are absent.
    MissingEvidence,
    /// The graph/query is outside the supported theorem input.
    UnsupportedInput,
    /// The complete theorem premises establish an obstruction.
    ProofObstruction,
    /// The bounded computation stopped before reaching a decision.
    ExhaustedComputation,
    /// The bounded search has neither a positive formula nor a checked obstruction.
    UnresolvedIdentification,
}

/// Portable record of one failed z-transport request and its exact evidence state.
#[derive(Clone, Debug)]
pub struct ZTransportFailureSnapshot {
    diagram: SelectionDiagram,
    query: ZTransportQuery,
    catalog: EvidenceCatalog,
    status: ZTransportFailureStatus,
    obligations: Arc<[Arc<str>]>,
    proof_graph: Option<ZTransportProofInspection>,
    obstruction: Option<antecedent_identify::sid::SHedgeRecord>,
    z_obstruction: Option<ZTransportObstructionRecord>,
    catalog_digest: String,
    graph_digest: String,
}

/// Serializable representation of a failure snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportFailureSnapshotWire {
    /// Schema version.
    pub version: u32,
    /// Query coordinates and concrete source assignment.
    pub query: ZTransportQueryWire,
    /// Shared source/target ADMG.
    pub graph: antecedent_io::wire::AdmgWire,
    /// Population-specific mechanisms.
    pub selection_targets: Vec<u32>,
    /// Exact catalog, including currently available bindings.
    pub catalog: antecedent_io::transport_catalog_wire::EvidenceCatalogWire,
    /// Failure classification and missing factor obligations.
    pub status: ZTransportFailureStatus,
    /// Human-readable and machine-stable obligation records.
    pub obligations: Vec<String>,
    /// Reachable checked formula graph with factor-level binding failures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_graph: Option<ZTransportProofInspection>,
    /// Checked stronger-family s-hedge, if one proves this z query impossible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obstruction: Option<antecedent_identify::sid::SHedgeRecord>,
    /// Checked bounded TRz line-11 obstruction, with complete-family evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub z_obstruction: Option<ZTransportObstructionRecord>,
    /// Canonical digest of the frozen catalog.
    pub catalog_digest: String,
    /// Digest of the graph wire and selection targets.
    pub graph_digest: String,
    /// Digest of every snapshot field, including query, status, and obligations.
    pub snapshot_digest: String,
}

/// Stable query wire for the snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportQueryWire {
    /// Outcome variable IDs.
    pub outcomes: Vec<u32>,
    /// Treatment variable IDs.
    pub treatments: Vec<u32>,
    /// Controllable variable IDs.
    pub controllable: Vec<u32>,
    /// Exact concrete source intervention assignments.
    pub experiment_assignment: Vec<(u32, antecedent_io::query_wire::ValueWire)>,
    /// Source population.
    pub source: String,
    /// Target population.
    pub target: String,
}

impl ZTransportFailureSnapshot {
    /// Freeze an outcome, catalog, query, and graph as the planning anchor.
    fn new(
        diagram: &SelectionDiagram,
        query: &ZTransportQuery,
        catalog: &EvidenceCatalog,
        status: ZTransportFailureStatus,
        obligations: impl Into<Arc<[Arc<str>]>>,
    ) -> Result<Self, ZTransportPlanningError> {
        catalog.validate().map_err(|error| ZTransportPlanningError::Invalid(error.to_string()))?;
        Ok(Self {
            diagram: diagram.clone(),
            query: query.clone(),
            catalog: catalog.clone(),
            status,
            obligations: obligations.into(),
            proof_graph: None,
            obstruction: None,
            z_obstruction: None,
            catalog_digest: catalog_digest(catalog)?,
            graph_digest: graph_digest(diagram)?,
        })
    }

    /// Snapshot status.
    #[must_use]
    pub const fn status(&self) -> &ZTransportFailureStatus {
        &self.status
    }

    /// Missing factor obligations from the original attempt.
    #[must_use]
    pub fn obligations(&self) -> &[Arc<str>] {
        &self.obligations
    }

    /// Frozen catalog identity.
    #[must_use]
    pub fn catalog_digest(&self) -> &str {
        &self.catalog_digest
    }

    /// Encode all inputs needed to independently inspect this failure.
    pub fn to_wire(&self) -> Result<ZTransportFailureSnapshotWire, ZTransportPlanningError> {
        let assignments = self
            .query
            .experiment_assignment
            .iter()
            .map(|a| (a.variable.raw(), antecedent_io::query_wire::ValueWire::from_value(&a.value)))
            .collect();
        let mut wire = ZTransportFailureSnapshotWire {
            version: if self.z_obstruction.is_some() { 2 } else { 1 },
            query: ZTransportQueryWire {
                outcomes: ids(&self.query.outcomes),
                treatments: ids(&self.query.treatments),
                controllable: ids(&self.query.controllable),
                experiment_assignment: assignments,
                source: self.query.source.to_string(),
                target: self.query.target.to_string(),
            },
            graph: antecedent_io::admg_to_wire(self.diagram.causal_graph())?,
            selection_targets: ids(self.diagram.selection_targets()),
            catalog: antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(
                &self.catalog,
            ),
            status: self.status.clone(),
            obligations: self.obligations.iter().map(ToString::to_string).collect(),
            proof_graph: self.proof_graph.clone(),
            obstruction: self.obstruction.clone(),
            z_obstruction: self.z_obstruction.clone(),
            catalog_digest: self.catalog_digest.clone(),
            graph_digest: self.graph_digest.clone(),
            snapshot_digest: String::new(),
        };
        wire.snapshot_digest = failure_snapshot_digest(&wire)?;
        Ok(wire)
    }

    /// Restore and verify a portable failure snapshot.
    pub fn from_wire(
        wire: &ZTransportFailureSnapshotWire,
    ) -> Result<Self, ZTransportPlanningError> {
        if !matches!(wire.version, 1 | 2) || (wire.version == 1 && wire.z_obstruction.is_some()) {
            return Err(ZTransportPlanningError::Invalid(
                "unsupported failure snapshot version".into(),
            ));
        }
        let graph = antecedent_io::admg_from_wire(&wire.graph)?;
        let diagram = SelectionDiagram::try_new(
            graph,
            wire.selection_targets
                .iter()
                .copied()
                .map(antecedent_core::VariableId::from_raw)
                .collect::<Vec<_>>(),
        )
        .map_err(|e| ZTransportPlanningError::Invalid(e.to_string()))?;
        let catalog = wire.catalog.to_catalog()?;
        let query = ZTransportQuery {
            outcomes: wire
                .query
                .outcomes
                .iter()
                .copied()
                .map(antecedent_core::VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            treatments: wire
                .query
                .treatments
                .iter()
                .copied()
                .map(antecedent_core::VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            controllable: wire
                .query
                .controllable
                .iter()
                .copied()
                .map(antecedent_core::VariableId::from_raw)
                .collect::<Vec<_>>()
                .into(),
            experiment_assignment: wire
                .query
                .experiment_assignment
                .iter()
                .map(|(variable, value)| antecedent_core::InterventionAssignment {
                    variable: antecedent_core::VariableId::from_raw(*variable),
                    value: value.to_value(),
                })
                .collect::<Vec<_>>()
                .into(),
            source: Arc::from(wire.query.source.as_str()),
            target: Arc::from(wire.query.target.as_str()),
        };
        let snapshot = Self::new(
            &diagram,
            &query,
            &catalog,
            wire.status.clone(),
            wire.obligations
                .iter()
                .map(|value| Arc::<str>::from(value.as_str()))
                .collect::<Vec<_>>(),
        )?;
        let mut snapshot = snapshot;
        snapshot.proof_graph = wire.proof_graph.clone();
        if let Some(record) = &wire.obstruction {
            let classical = ClassicalTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                source: Arc::clone(&query.source),
                target: Arc::clone(&query.target),
            };
            antecedent_identify::sid::SHedgeCertificate::from_record_checked(
                record.clone(),
                &diagram,
                &classical,
                &antecedent_core::ExecutionContext::for_tests(0),
            )?;
            snapshot.obstruction = Some(record.clone());
        }
        if let Some(record) = &wire.z_obstruction {
            ZTransportObstruction::from_record_checked(
                record.clone(),
                &diagram,
                &query,
                &catalog,
                SidLimits::default(),
                &antecedent_core::ExecutionContext::for_tests(0),
            )?;
            snapshot.z_obstruction = Some(record.clone());
        }
        let recomputed = snapshot_z_transport_failure(&diagram, &query, &catalog)?;
        if recomputed.status != snapshot.status
            || recomputed.obligations != snapshot.obligations
            || recomputed.obstruction.is_some() != snapshot.obstruction.is_some()
            || recomputed.z_obstruction != snapshot.z_obstruction
            || serde_json::to_value(&recomputed.proof_graph).ok()
                != serde_json::to_value(&snapshot.proof_graph).ok()
        {
            return Err(ZTransportPlanningError::Invalid(
                "failure snapshot outcome differs from current checked identification".into(),
            ));
        }
        if snapshot.catalog_digest != wire.catalog_digest
            || snapshot.graph_digest != wire.graph_digest
            || failure_snapshot_digest(wire)? != wire.snapshot_digest
        {
            return Err(ZTransportPlanningError::Invalid(
                "failure snapshot digest mismatch".into(),
            ));
        }
        Ok(snapshot)
    }
}

/// A proposal tied to one failed snapshot and one candidate delta.
#[derive(Clone, Debug)]
pub struct ZTransportProposal {
    pub(crate) candidate: TransportEvidenceCandidate,
    pub(crate) snapshot: ZTransportFailureSnapshot,
    pub(crate) derivation: ZTransportDerivation,
    pub(crate) checked_preview: BoundZTransportFunctional,
}

/// Portable proposal proof, study action, and original failure binding.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ZTransportProposalWire {
    /// Wire schema version.
    pub version: u32,
    /// Exact failure that led to the proposal.
    pub snapshot: ZTransportFailureSnapshotWire,
    /// Stable candidate ID and action.
    pub candidate_id: String,
    /// Candidate action description.
    pub design: crate::transport_planner::CandidateDesignRecord,
    /// Declared cost and feasibility premises.
    pub cost: crate::DesignCost,
    /// Recruitment and sampling premise.
    pub recruitment_sampling: String,
    /// Feasibility conditions.
    pub feasibility_constraints: Vec<String>,
    /// Proposed evidence regimes; never available by themselves.
    pub delta: antecedent_io::transport_catalog_wire::EvidenceCatalogDeltaWire,
    /// Checked proof premises.
    pub proof: antecedent_identify::ZTransportDerivationRecord,
    /// Expression arena independently replayed with the proof.
    pub arena: antecedent_io::ExprArenaWire,
}

/// Result for evidence arriving from the proposed study.
#[derive(Clone, Debug)]
pub struct ZTransportArrival {
    /// Functional bound against the actual, available catalog.
    pub functional: BoundZTransportFunctional,
    /// Identity of the provider snapshot that supplied the actual evidence.
    pub provider_snapshot: Arc<str>,
}

/// One candidate's result when planning from a frozen z-transport failure.
#[derive(Clone, Debug)]
pub enum ZTransportCandidateOutcome {
    /// Hypothetical catalog binds a checked formula.
    VerifiedSufficient(ZTransportProposal),
    /// Candidate action or evidence does not satisfy the frozen request.
    Rejected {
        /// Why this declared action or evidence could not yield a checked result.
        reason: Arc<str>,
    },
}

/// Candidate assessment in declared order.
#[derive(Clone, Debug)]
pub struct ZTransportCandidateAssessment {
    /// Stable candidate identity.
    pub id: Arc<str>,
    /// Declared candidate cost.
    pub cost: crate::DesignCost,
    /// Checked sufficiency outcome.
    pub outcome: ZTransportCandidateOutcome,
}

/// Candidate results and verified sufficient IDs sorted by cost then ID.
#[derive(Clone, Debug)]
pub struct ZTransportPlanResult {
    /// All candidate assessments in input order.
    pub assessments: Vec<ZTransportCandidateAssessment>,
    /// IDs for proposals that produced a checked and bound formula.
    pub ranked_sufficient: Vec<Arc<str>>,
}

/// z-transport planning and arrival validation failures.
#[derive(Debug, Error)]
pub enum ZTransportPlanningError {
    /// Mismatch between declared action, delta, snapshot, or supplied evidence.
    #[error("invalid z-transport planning input: {0}")]
    Invalid(String),
    /// Failed wire conversion.
    #[error(transparent)]
    Io(#[from] antecedent_io::IoError),
    /// Failed structural identification or provider binding.
    #[error(transparent)]
    Identification(#[from] antecedent_identify::IdentificationError),
    /// Invalid catalog delta.
    #[error(transparent)]
    Catalog(#[from] antecedent_core::QueryError),
}

/// Capture an explicit failed outcome from the currently supported identifier.
pub fn snapshot_z_transport_failure(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
) -> Result<ZTransportFailureSnapshot, ZTransportPlanningError> {
    let identification = match identify_z_transport(diagram, query) {
        Ok(result) => result,
        Err(error) => {
            let code = error.to_string();
            let status = if code.starts_with("z_transport.unsupported_") {
                Some(ZTransportFailureStatus::UnsupportedInput)
            } else if code == "z_transport.exhausted_computation" {
                Some(ZTransportFailureStatus::ExhaustedComputation)
            } else {
                None
            };
            if let Some(status) = status {
                return ZTransportFailureSnapshot::new(
                    diagram,
                    query,
                    catalog,
                    status,
                    [Arc::from(code)],
                );
            }
            return Err(error.into());
        }
    };
    match identification {
        antecedent_identify::ZTransportResult::Identified(derivation) => {
            match bind_z_transport_catalog(diagram, query, &derivation, catalog) {
                Ok(_) => Err(ZTransportPlanningError::Invalid(
                    "query already binds to available evidence".into(),
                )),
                Err(error) => {
                    let message = error.to_string();
                    let status = if message.contains("unsupported_domain")
                        || message.contains("unsupported_observed_count")
                        || message.contains("unsupported_controllable_count")
                    {
                        ZTransportFailureStatus::UnsupportedInput
                    } else {
                        ZTransportFailureStatus::MissingEvidence
                    };
                    let graph = derivation.inspect_proof(catalog);
                    let obligations = graph
                        .factors
                        .iter()
                        .filter(|factor| factor.failure.is_some())
                        .map(|factor| {
                            Arc::<str>::from(
                                serde_json::to_string(factor).unwrap_or_else(|_| message.clone()),
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut snapshot = ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        status,
                        if obligations.is_empty() { vec![Arc::from(message)] } else { obligations },
                    )?;
                    snapshot.proof_graph = Some(graph);
                    Ok(snapshot)
                }
            }
        }
        antecedent_identify::ZTransportResult::NotCertified { reason } => {
            match decide_z_transport_with_catalog(
                diagram,
                query,
                catalog,
                SidLimits::default(),
                &antecedent_core::ExecutionContext::for_tests(0),
            )? {
                ZTransportDecision::ProvenNonTransportable(obstruction) => {
                    let mut snapshot = ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        ZTransportFailureStatus::ProofObstruction,
                        [Arc::from("z_transport.checked_trz_line11_obstruction")],
                    )?;
                    snapshot.z_obstruction = Some(obstruction.to_record());
                    return Ok(snapshot);
                }
                ZTransportDecision::MissingEvidence { missing } => {
                    return ZTransportFailureSnapshot::new(
                        diagram,
                        query,
                        catalog,
                        ZTransportFailureStatus::MissingEvidence,
                        [Arc::from(format!("z_transport.experimental_family: {missing:?}"))],
                    );
                }
                ZTransportDecision::Identified(derivation) => {
                    return match bind_z_transport_catalog(
                        diagram,
                        derivation.query(),
                        &derivation,
                        catalog,
                    ) {
                        Ok(_) => Err(ZTransportPlanningError::Invalid(
                            "catalog-aware TRz decision found an available formula; prepare from the decided derivation".into(),
                        )),
                        Err(error) => ZTransportFailureSnapshot::new(
                            diagram,
                            query,
                            catalog,
                            ZTransportFailureStatus::MissingEvidence,
                            [Arc::from(error.to_string())],
                        ),
                    };
                }
                ZTransportDecision::NotCertified { .. } => {}
            }
            if let Err(family) = validate_z_experiment_family(diagram, query, catalog) {
                let status = match family {
                    antecedent_identify::ZExperimentFamilyError::UnsupportedDomain { .. } => {
                        ZTransportFailureStatus::UnsupportedInput
                    }
                    antecedent_identify::ZExperimentFamilyError::FamilyExceedsBudget { .. } => {
                        ZTransportFailureStatus::ExhaustedComputation
                    }
                    antecedent_identify::ZExperimentFamilyError::MissingJointLaw { .. } => {
                        ZTransportFailureStatus::MissingEvidence
                    }
                };
                return ZTransportFailureSnapshot::new(
                    diagram,
                    query,
                    catalog,
                    status,
                    [Arc::from(format!("z_transport.experimental_family: {family:?}"))],
                );
            }
            if !target_joint_observational_law(diagram, query, catalog) {
                return ZTransportFailureSnapshot::new(
                    diagram,
                    query,
                    catalog,
                    ZTransportFailureStatus::MissingEvidence,
                    [Arc::from("z_transport.target_joint_observational_law_required")],
                );
            }
            let classical = ClassicalTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                source: Arc::clone(&query.source),
                target: Arc::clone(&query.target),
            };
            if let ClassicalTransportResult::ProvenNonTransportable(witness) =
                identify_classical_transport(
                    diagram,
                    &classical,
                    SidLimits::default(),
                    &antecedent_core::ExecutionContext::for_tests(0),
                )?
            {
                let mut snapshot = ZTransportFailureSnapshot::new(
                    diagram,
                    query,
                    catalog,
                    ZTransportFailureStatus::ProofObstruction,
                    [Arc::from("z_transport.checked_stronger_family_s_hedge")],
                )?;
                snapshot.obstruction = Some(witness.to_record());
                return Ok(snapshot);
            }
            ZTransportFailureSnapshot::new(
                diagram,
                query,
                catalog,
                ZTransportFailureStatus::UnresolvedIdentification,
                [Arc::from(reason)],
            )
        }
    }
}

fn target_joint_observational_law(
    diagram: &SelectionDiagram,
    query: &ZTransportQuery,
    catalog: &EvidenceCatalog,
) -> bool {
    let observed = diagram
        .causal_graph()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            antecedent_graph::NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect::<Vec<_>>();
    catalog.regimes.iter().any(|regime| {
        regime.population.as_ref() == query.target.as_ref()
            && regime.kind == RegimeKind::Observational
            && regime.evidence_kind == EvidenceKind::Available
            && regime.interventions.is_empty()
            && regime.intervention_values.is_empty()
            && regime.conditioned_on.is_empty()
            && regime.distribution == antecedent_core::DistributionAvailability::Joint
            && observed.iter().all(|v| regime.measured.contains(v))
            && catalog.bindings.iter().any(|binding| binding.regime == regime.id)
    })
}

/// Verify that a candidate is feasible and its delta matches the declared action.
pub fn validate_z_transport_candidate(
    snapshot: &ZTransportFailureSnapshot,
    candidate: &TransportEvidenceCandidate,
) -> Result<(), ZTransportPlanningError> {
    if candidate.id.trim().is_empty()
        || candidate.recruitment_sampling.trim().is_empty()
        || !candidate.cost.amount.is_finite()
        || candidate.cost.amount < 0.0
        || candidate.delta.proposed_regimes.is_empty()
    {
        return Err(ZTransportPlanningError::Invalid(
            "candidate identity, cost, or delta is invalid".into(),
        ));
    }
    candidate.delta.preview_catalog(&snapshot.catalog)?;
    for regime in candidate.delta.proposed_regimes.iter() {
        let matches = match &candidate.design {
            CandidateDesign::Intervene(plan) => {
                regime.kind == RegimeKind::Experimental
                    && same_vars(&plan.targets, &regime.interventions)
            }
            CandidateDesign::Measure(plan) => {
                regime.kind == RegimeKind::Observational
                    && regime.interventions.is_empty()
                    && same_vars(&plan.variables, &regime.measured)
            }
            _ => false,
        };
        if !matches
            || regime.population.as_ref() != snapshot.query.source.as_ref()
            || regime.evidence_kind != EvidenceKind::Proposed
            || !regime.conditioned_on.is_empty()
            || regime.distribution != antecedent_core::DistributionAvailability::Joint
            || !valid_intervention_values(&snapshot.catalog, regime)
        {
            return Err(ZTransportPlanningError::Invalid(format!(
                "candidate action cannot produce proposed regime {}",
                regime.id.raw()
            )));
        }
    }
    Ok(())
}

fn valid_intervention_values(
    catalog: &EvidenceCatalog,
    regime: &antecedent_core::EvidenceRegime,
) -> bool {
    if regime.kind == RegimeKind::Experimental
        && regime.intervention_values.len() != regime.interventions.len()
    {
        return false;
    }
    let Some(environment) = catalog.environments.iter().find(|e| e.identity == regime.population)
    else {
        return false;
    };
    regime.intervention_values.iter().all(|assignment| {
        let Some(coordinate) =
            environment.variables.iter().find(|c| c.variable == assignment.variable)
        else {
            return false;
        };
        let Some(value) = assignment.value.as_f64() else { return false };
        match coordinate.domain {
            antecedent_core::VariableDomain::Unspecified => value.is_finite(),
            antecedent_core::VariableDomain::Continuous => value.is_finite(),
            antecedent_core::VariableDomain::Binary => value == 0.0 || value == 1.0,
            antecedent_core::VariableDomain::Count => value >= 0.0 && value.fract() == 0.0,
            antecedent_core::VariableDomain::Categorical { cardinality } => {
                value >= 0.0 && value < f64::from(cardinality) && value.fract() == 0.0
            }
        }
    })
}

/// Check candidate sufficiency on an isolated hypothetical catalog.
pub fn propose_z_transport_evidence(
    snapshot: &ZTransportFailureSnapshot,
    candidate: &TransportEvidenceCandidate,
) -> Result<ZTransportProposal, ZTransportPlanningError> {
    if snapshot.status != ZTransportFailureStatus::MissingEvidence {
        return Err(ZTransportPlanningError::Invalid(
            "evidence planning requires a missing-evidence failure snapshot".into(),
        ));
    }
    validate_z_transport_candidate(snapshot, candidate)?;
    let antecedent_identify::ZTransportResult::Identified(derivation) =
        identify_z_transport(&snapshot.diagram, &snapshot.query)?
    else {
        return Err(ZTransportPlanningError::Invalid(
            "snapshot query has no checked positive formula".into(),
        ));
    };
    let preview = hypothetical_catalog(&snapshot.catalog, &candidate.delta)?;
    let checked_preview =
        bind_z_transport_catalog(&snapshot.diagram, &snapshot.query, &derivation, &preview)?;
    Ok(ZTransportProposal {
        candidate: candidate.clone(),
        snapshot: snapshot.clone(),
        derivation: *derivation,
        checked_preview,
    })
}

/// Evaluate every candidate against the frozen failure and rank only checked proposals.
pub fn plan_z_transport_evidence(
    snapshot: &ZTransportFailureSnapshot,
    candidates: &[TransportEvidenceCandidate],
) -> ZTransportPlanResult {
    let mut ids = BTreeSet::new();
    let mut assessments = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let outcome = if !ids.insert(candidate.id.clone()) {
            ZTransportCandidateOutcome::Rejected { reason: Arc::from("duplicate candidate ID") }
        } else {
            match propose_z_transport_evidence(snapshot, candidate) {
                Ok(proposal) => ZTransportCandidateOutcome::VerifiedSufficient(proposal),
                Err(error) => {
                    ZTransportCandidateOutcome::Rejected { reason: Arc::from(error.to_string()) }
                }
            }
        };
        assessments.push(ZTransportCandidateAssessment {
            id: Arc::clone(&candidate.id),
            cost: candidate.cost,
            outcome,
        });
    }
    let mut ranked = assessments
        .iter()
        .filter_map(|assessment| {
            matches!(assessment.outcome, ZTransportCandidateOutcome::VerifiedSufficient(_))
                .then_some((assessment.cost, Arc::clone(&assessment.id)))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        a.0.amount
            .total_cmp(&b.0.amount)
            .then_with(|| a.0.sample_budget.cmp(&b.0.sample_budget))
            .then_with(|| a.1.cmp(&b.1))
    });
    ZTransportPlanResult {
        assessments,
        ranked_sufficient: ranked.into_iter().map(|entry| entry.1).collect(),
    }
}

impl ZTransportProposal {
    /// Checked formula available only in the hypothetical preview.
    #[must_use]
    pub const fn checked_preview(&self) -> &BoundZTransportFunctional {
        &self.checked_preview
    }

    /// Serialize the snapshot, study delta, and checked proof for portable replay.
    pub fn to_wire(&self) -> Result<ZTransportProposalWire, ZTransportPlanningError> {
        Ok(ZTransportProposalWire {
            version: 1,
            snapshot: self.snapshot.to_wire()?,
            candidate_id: self.candidate.id.to_string(),
            design: crate::transport_planner::candidate_design_record(&self.candidate.design),
            cost: self.candidate.cost,
            recruitment_sampling: self.candidate.recruitment_sampling.to_string(),
            feasibility_constraints: self
                .candidate
                .feasibility_constraints
                .iter()
                .map(ToString::to_string)
                .collect(),
            delta: antecedent_io::transport_catalog_wire::EvidenceCatalogDeltaWire::from_delta(
                &self.candidate.delta,
            ),
            proof: self.derivation.to_record(),
            arena: antecedent_io::expr_arena_to_wire(self.derivation.arena())?,
        })
    }

    /// Recheck snapshot binding and the hypothetical formula.
    pub fn replay(&self) -> Result<(), ZTransportPlanningError> {
        if catalog_digest(&self.snapshot.catalog)? != self.snapshot.catalog_digest
            || graph_digest(&self.snapshot.diagram)? != self.snapshot.graph_digest
        {
            return Err(ZTransportPlanningError::Invalid(
                "failure snapshot identity changed".into(),
            ));
        }
        validate_z_transport_candidate(&self.snapshot, &self.candidate)?;
        bind_z_transport_catalog(
            &self.snapshot.diagram,
            &self.snapshot.query,
            &self.derivation,
            &hypothetical_catalog(&self.snapshot.catalog, &self.candidate.delta)?,
        )?;
        Ok(())
    }

    /// Reidentify and bind after evidence arrives in a real available catalog.
    pub fn receive(
        &self,
        actual_catalog: &EvidenceCatalog,
        provider_snapshot: impl Into<Arc<str>>,
    ) -> Result<ZTransportArrival, ZTransportPlanningError> {
        self.replay()?;
        actual_catalog.validate().map_err(|e| ZTransportPlanningError::Invalid(e.to_string()))?;
        validate_base_preserved(&self.snapshot.catalog, actual_catalog)?;
        validate_actual_delta(&self.candidate.delta, actual_catalog)?;
        let provider_snapshot = provider_snapshot.into();
        if provider_snapshot.trim().is_empty()
            || !self.candidate.delta.proposed_regimes.iter().all(|regime| {
                actual_catalog.bindings.iter().any(|binding| {
                    binding.regime == regime.id
                        && binding.snapshot_identity.as_ref() == provider_snapshot.as_ref()
                })
            })
        {
            return Err(ZTransportPlanningError::Invalid(
                "provider snapshot does not bind every arriving proposed regime".into(),
            ));
        }
        let antecedent_identify::ZTransportResult::Identified(derivation) =
            identify_z_transport(&self.snapshot.diagram, &self.snapshot.query)?
        else {
            return Err(ZTransportPlanningError::Invalid(
                "query no longer has a checked positive formula".into(),
            ));
        };
        let functional = bind_z_transport_catalog(
            &self.snapshot.diagram,
            &self.snapshot.query,
            &derivation,
            actual_catalog,
        )?;
        Ok(ZTransportArrival { functional, provider_snapshot })
    }
}

impl ZTransportProposalWire {
    /// Restore and recheck a portable proposal against its frozen failure.
    pub fn replay(&self) -> Result<ZTransportProposal, ZTransportPlanningError> {
        if self.version != 1 {
            return Err(ZTransportPlanningError::Invalid("unsupported proposal version".into()));
        }
        let snapshot = ZTransportFailureSnapshot::from_wire(&self.snapshot)?;
        let delta = self.delta.to_delta(&snapshot.catalog)?;
        let design = match self.design.kind.as_str() {
            "intervene" => CandidateDesign::Intervene(crate::ExperimentPlan {
                targets: self
                    .design
                    .identities
                    .iter()
                    .copied()
                    .map(antecedent_core::VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                cost: self.cost,
                tag: self.design.tag,
            }),
            "measure" => CandidateDesign::Measure(crate::MeasurementPlan {
                variables: self
                    .design
                    .identities
                    .iter()
                    .copied()
                    .map(antecedent_core::VariableId::from_raw)
                    .collect::<Vec<_>>()
                    .into(),
                cost: self.cost,
                tag: self.design.tag,
            }),
            _ => {
                return Err(ZTransportPlanningError::Invalid("unsupported proposal action".into()));
            }
        };
        let candidate = TransportEvidenceCandidate {
            id: Arc::from(self.candidate_id.as_str()),
            delta,
            design,
            recruitment_sampling: Arc::from(self.recruitment_sampling.as_str()),
            feasibility_constraints: self
                .feasibility_constraints
                .iter()
                .map(|v| Arc::<str>::from(v.as_str()))
                .collect::<Vec<_>>()
                .into(),
            cost: self.cost,
        };
        validate_z_transport_candidate(&snapshot, &candidate)?;
        let arena = antecedent_io::expr_arena_from_wire(&self.arena)?;
        let derivation = antecedent_identify::ZTransportDerivation::from_record_checked(
            &snapshot.diagram,
            &snapshot.query,
            &self.proof,
            arena,
        )?;
        let preview = hypothetical_catalog(&snapshot.catalog, &candidate.delta)?;
        let checked_preview =
            bind_z_transport_catalog(&snapshot.diagram, &snapshot.query, &derivation, &preview)?;
        Ok(ZTransportProposal { candidate, snapshot, derivation, checked_preview })
    }
}

fn validate_actual_delta(
    delta: &EvidenceCatalogDelta,
    actual: &EvidenceCatalog,
) -> Result<(), ZTransportPlanningError> {
    for expected in delta.proposed_regimes.iter() {
        let Some(found) = actual.regimes.iter().find(|r| r.id == expected.id) else {
            return Err(ZTransportPlanningError::Invalid(format!(
                "arriving catalog lacks proposed regime {}",
                expected.id.raw()
            )));
        };
        if found.evidence_kind != EvidenceKind::Available
            || found.kind != expected.kind
            || found.population != expected.population
            || !same_vars(&found.interventions, &expected.interventions)
            || !same_assignments(&found.intervention_values, &expected.intervention_values)
            || !same_vars(&found.measured, &expected.measured)
            || found.conditioned_on != expected.conditioned_on
            || found.distribution != expected.distribution
        {
            return Err(ZTransportPlanningError::Invalid(format!(
                "arriving regime {} differs from proposal",
                expected.id.raw()
            )));
        }
        if !actual.bindings.iter().any(|b| b.regime == found.id) {
            return Err(ZTransportPlanningError::Invalid(
                "arriving regime has no provider binding".into(),
            ));
        }
    }
    Ok(())
}

/// Add isolated placeholder provider bindings so structural planning can check
/// whether a proposed law would satisfy the formula. These identities never
/// leave a proposal as available evidence and are replaced by actual bindings
/// during arrival.
fn hypothetical_catalog(
    base: &EvidenceCatalog,
    delta: &EvidenceCatalogDelta,
) -> Result<EvidenceCatalog, ZTransportPlanningError> {
    let preview = delta.preview_catalog(base)?;
    let mut bindings = preview.bindings.to_vec();
    for regime in delta.proposed_regimes.iter() {
        bindings.push(antecedent_core::RegimeBinding {
            dataset_identity: None,
            regime: regime.id,
            snapshot_identity: Arc::from(format!("hypothetical:{}", regime.id.raw())),
            schema_names: Arc::from([]),
            sampling: antecedent_core::SamplingDesign::Independent,
            weights: None,
            dependence: antecedent_core::DependenceGroup::IndependentStudies,
        });
    }
    Ok(EvidenceCatalog::try_new(
        Arc::clone(&preview.environments),
        Arc::clone(&preview.regimes),
        bindings,
        preview.target_sampling,
    )?)
}

fn validate_base_preserved(
    base: &EvidenceCatalog,
    actual: &EvidenceCatalog,
) -> Result<(), ZTransportPlanningError> {
    if base.environments != actual.environments || base.target_sampling != actual.target_sampling {
        return Err(ZTransportPlanningError::Invalid(
            "arrival catalog does not preserve the proposal base catalog".into(),
        ));
    }
    for regime in base.regimes.iter() {
        if !actual.regimes.contains(regime) {
            return Err(ZTransportPlanningError::Invalid(format!(
                "base regime {} changed or disappeared",
                regime.id.raw()
            )));
        }
    }
    for binding in base.bindings.iter() {
        if !actual.bindings.contains(binding) {
            return Err(ZTransportPlanningError::Invalid(
                "base provider binding changed or disappeared".into(),
            ));
        }
    }
    Ok(())
}

fn same_assignments(
    a: &[antecedent_core::InterventionAssignment],
    b: &[antecedent_core::InterventionAssignment],
) -> bool {
    a.len() == b.len()
        && a.iter().all(|x| {
            b.iter().any(|y| x.variable == y.variable && x.value.as_f64() == y.value.as_f64())
        })
}
fn same_vars(a: &[antecedent_core::VariableId], b: &[antecedent_core::VariableId]) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x))
}
fn ids(values: &[antecedent_core::VariableId]) -> Vec<u32> {
    values.iter().map(|v| v.raw()).collect()
}
fn catalog_digest(catalog: &EvidenceCatalog) -> Result<String, ZTransportPlanningError> {
    let canonical =
        catalog.canonicalized().map_err(|e| ZTransportPlanningError::Invalid(e.to_string()))?;
    let wire = antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&canonical);
    Ok(blake3::hash(
        &serde_json::to_vec(&wire).map_err(|e| ZTransportPlanningError::Invalid(e.to_string()))?,
    )
    .to_hex()
    .to_string())
}
fn graph_digest(diagram: &SelectionDiagram) -> Result<String, ZTransportPlanningError> {
    let wire = antecedent_io::admg_to_wire(diagram.causal_graph())?;
    let payload = serde_json::to_vec(&(wire, ids(diagram.selection_targets())))
        .map_err(|e| ZTransportPlanningError::Invalid(e.to_string()))?;
    Ok(blake3::hash(&payload).to_hex().to_string())
}

fn failure_snapshot_digest(
    wire: &ZTransportFailureSnapshotWire,
) -> Result<String, ZTransportPlanningError> {
    let payload = serde_json::to_vec(&(
        wire.version,
        &wire.query,
        &wire.graph,
        &wire.selection_targets,
        &wire.catalog,
        &wire.status,
        &wire.obligations,
        &wire.catalog_digest,
        &wire.graph_digest,
    ))
    .map_err(|e| ZTransportPlanningError::Invalid(e.to_string()))?;
    Ok(blake3::hash(&payload).to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceRegime,
        InterventionAssignment, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value,
        VariableCoordinate, VariableDomain, VariableId,
    };
    use antecedent_graph::{Admg, DenseNodeId};

    #[test]
    fn line11_obstruction_is_structural_with_or_without_the_experiment_family() {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram =
            SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([VariableId::from_raw(1)]))
                .unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(1)]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            controllable: Arc::from([VariableId::from_raw(2)]),
            experiment_assignment: Arc::from([InterventionAssignment {
                variable: VariableId::from_raw(2),
                value: Value::Bool(false),
            }]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let coords = (0..3)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let source =
            Environment::try_new("source", coords.clone(), Arc::<[VariableId]>::from([])).unwrap();
        let target = Environment::try_new("target", coords, Arc::<[VariableId]>::from([])).unwrap();
        let regimes = (0..3)
            .map(|raw| {
                let intervention = if raw == 0 { vec![] } else { vec![VariableId::from_raw(2)] };
                let values = if raw == 0 {
                    vec![]
                } else {
                    vec![InterventionAssignment {
                        variable: VariableId::from_raw(2),
                        value: Value::Bool(raw == 2),
                    }]
                };
                EvidenceRegime::try_new(
                    RegimeId::from_raw(raw),
                    if raw == 0 { RegimeKind::Observational } else { RegimeKind::Experimental },
                    EvidenceKind::Available,
                    intervention,
                    values,
                    [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)],
                    if raw == 0 { "target" } else { "source" },
                    DistributionAvailability::Joint,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let bindings = (0..3)
            .map(|raw| RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(raw),
                snapshot_identity: Arc::from(format!("obstruction-{raw}")),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            })
            .collect::<Vec<_>>();
        let complete = EvidenceCatalog::try_new([source, target], regimes, bindings, None).unwrap();
        let checked = snapshot_z_transport_failure(&diagram, &query, &complete).unwrap();
        assert_eq!(checked.status(), &ZTransportFailureStatus::ProofObstruction);
        let wire = checked.to_wire().unwrap();
        assert!(wire.z_obstruction.is_some());
        assert!(wire.obstruction.is_none());
        ZTransportFailureSnapshot::from_wire(&wire).unwrap();

        let mut incomplete = complete;
        incomplete.regimes = incomplete.regimes[..2].to_vec().into();
        incomplete.bindings = incomplete.bindings[..2].to_vec().into();
        let unresolved = snapshot_z_transport_failure(&diagram, &query, &incomplete).unwrap();
        assert_eq!(unresolved.status(), &ZTransportFailureStatus::ProofObstruction);
        assert!(unresolved.to_wire().unwrap().z_obstruction.is_some());
    }

    #[test]
    fn restricted_family_obstruction_survives_snapshot_replay_and_rejects_tampering() {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(1)]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            controllable: Arc::from([VariableId::from_raw(2)]),
            experiment_assignment: Arc::from([]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let variables = (0..3)
            .map(|raw| VariableCoordinate {
                variable: VariableId::from_raw(raw),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let environments = [
            Environment::try_new("source", variables.clone(), []).unwrap(),
            Environment::try_new("target", variables, []).unwrap(),
        ];
        let measured: Vec<_> = (0..3).map(VariableId::from_raw).collect();
        let regimes = (0..3)
            .map(|raw| {
                EvidenceRegime::try_new(
                    RegimeId::from_raw(raw),
                    if raw == 0 { RegimeKind::Observational } else { RegimeKind::Experimental },
                    EvidenceKind::Available,
                    if raw == 0 { vec![] } else { vec![VariableId::from_raw(2)] },
                    if raw == 0 {
                        vec![]
                    } else {
                        vec![InterventionAssignment {
                            variable: VariableId::from_raw(2),
                            value: Value::Bool(raw == 2),
                        }]
                    },
                    measured.clone(),
                    if raw == 0 { "target" } else { "source" },
                    DistributionAvailability::Joint,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let bindings = (0..3)
            .map(|raw| RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(raw),
                snapshot_identity: Arc::from(format!("restricted-{raw}")),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            })
            .collect::<Vec<_>>();
        let catalog = EvidenceCatalog::try_new(environments, regimes, bindings, None).unwrap();
        let snapshot = snapshot_z_transport_failure(&diagram, &query, &catalog).unwrap();
        assert_eq!(snapshot.status(), &ZTransportFailureStatus::ProofObstruction);
        let wire = snapshot.to_wire().unwrap();
        assert_eq!(wire.version, 2);
        assert!(wire.obstruction.is_none());
        assert!(wire.z_obstruction.is_some());
        ZTransportFailureSnapshot::from_wire(&wire).unwrap();

        let mut tampered = wire.clone();
        tampered.z_obstruction.as_mut().unwrap().terminal.treatments.clear();
        assert!(ZTransportFailureSnapshot::from_wire(&tampered).is_err());
    }

    fn fixture() -> (SelectionDiagram, ZTransportQuery) {
        let mut graph = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        for (a, b) in [(0, 3), (1, 3), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(3)]),
            treatments: Arc::from([VariableId::from_raw(2)]),
            controllable: Arc::from([VariableId::from_raw(1)]),
            experiment_assignment: Arc::from([InterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            }]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        (diagram, query)
    }

    fn catalog_with_high_regime() -> EvidenceCatalog {
        let source = Environment::try_new(
            "source",
            (0..4)
                .map(|raw| VariableCoordinate {
                    variable: VariableId::from_raw(raw),
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            Arc::<[VariableId]>::from([]),
        )
        .unwrap();
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(9),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [VariableId::from_raw(1)],
            [InterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(true),
            }],
            (0..4).map(VariableId::from_raw).collect::<Vec<_>>(),
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let binding = RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(9),
            snapshot_identity: Arc::from("source-high"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        };
        EvidenceCatalog::try_new([source], [regime], [binding], None).unwrap()
    }

    #[test]
    fn failure_snapshot_wire_freezes_query_graph_catalog_and_obligation() {
        let (diagram, query) = fixture();
        let source = Environment::try_new(
            "source",
            (0..4)
                .map(|raw| VariableCoordinate {
                    variable: VariableId::from_raw(raw),
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            Arc::<[VariableId]>::from([]),
        )
        .unwrap();
        let catalog = EvidenceCatalog::try_new([source], [], [], None).unwrap();
        let snapshot = snapshot_z_transport_failure(&diagram, &query, &catalog).unwrap();
        assert_eq!(snapshot.status(), &ZTransportFailureStatus::MissingEvidence);
        let wire = snapshot.to_wire().unwrap();
        assert_eq!(wire.version, 1);
        assert_eq!(wire.query.source, "source");
        assert_eq!(wire.query.experiment_assignment.len(), 1);
        assert_eq!(wire.graph.node_count, 4);
        assert!(!wire.catalog_digest.is_empty());
        assert_eq!(wire.obligations.len(), 2);
        assert_eq!(wire.proof_graph.as_ref().unwrap().factors.len(), 2);
        assert!(ZTransportFailureSnapshot::from_wire(&wire).is_ok());
        let mut changed_query = wire.clone();
        changed_query.query.source = "other-source".into();
        assert!(ZTransportFailureSnapshot::from_wire(&changed_query).is_err());
        let mut changed_snapshot = wire.clone();
        changed_snapshot.obligations[0].push_str(" changed");
        assert!(ZTransportFailureSnapshot::from_wire(&changed_snapshot).is_err());
    }

    #[test]
    fn proposal_replay_and_arrival_reidentify_against_actual_provider() {
        let (diagram, query) = fixture();
        let base = catalog_with_high_regime();
        let snapshot = snapshot_z_transport_failure(&diagram, &query, &base).unwrap();
        let proposed = EvidenceRegime::try_new(
            RegimeId::from_raw(10),
            RegimeKind::Experimental,
            EvidenceKind::Proposed,
            [VariableId::from_raw(1)],
            [InterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            }],
            (0..4).map(VariableId::from_raw).collect::<Vec<_>>(),
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let candidate = TransportEvidenceCandidate {
            id: Arc::from("low-z"),
            delta: EvidenceCatalogDelta::try_new(&base, [proposed.clone()]).unwrap(),
            design: CandidateDesign::Intervene(crate::ExperimentPlan {
                targets: Arc::from([VariableId::from_raw(1)]),
                cost: crate::DesignCost { amount: 1.0, sample_budget: 20 },
                tag: 7,
            }),
            recruitment_sampling: Arc::from("source population, joint measurement"),
            feasibility_constraints: Arc::from([]),
            cost: crate::DesignCost { amount: 1.0, sample_budget: 20 },
        };
        let measurement_regime = EvidenceRegime::try_new(
            RegimeId::from_raw(11),
            RegimeKind::Observational,
            EvidenceKind::Proposed,
            [],
            [],
            [VariableId::from_raw(1), VariableId::from_raw(3)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let mismatched_measurement = TransportEvidenceCandidate {
            id: Arc::from("incomplete-measurement-action"),
            delta: EvidenceCatalogDelta::try_new(&base, [measurement_regime]).unwrap(),
            design: CandidateDesign::Measure(crate::MeasurementPlan {
                variables: Arc::from([VariableId::from_raw(1)]),
                cost: crate::DesignCost { amount: 1.0, sample_budget: 20 },
                tag: 8,
            }),
            recruitment_sampling: Arc::from("source population"),
            feasibility_constraints: Arc::from([]),
            cost: crate::DesignCost { amount: 1.0, sample_budget: 20 },
        };
        assert!(validate_z_transport_candidate(&snapshot, &mismatched_measurement).is_err());
        let plan = plan_z_transport_evidence(&snapshot, &[candidate]);
        assert_eq!(
            plan.ranked_sufficient,
            [Arc::<str>::from("low-z")],
            "assessments: {:?}",
            plan.assessments
        );
        let ZTransportCandidateOutcome::VerifiedSufficient(proposal) = &plan.assessments[0].outcome
        else {
            panic!("proposed joint evidence should complete the family");
        };
        proposal.replay().unwrap();
        let proposal_wire = proposal.to_wire().unwrap();
        let encoded = serde_json::to_vec(&proposal_wire).unwrap();
        let decoded: ZTransportProposalWire = serde_json::from_slice(&encoded).unwrap();
        let restored = decoded.replay().unwrap();
        restored.replay().unwrap();
        let mut stale_proposal = decoded.clone();
        stale_proposal.snapshot.query.source = "other-source".into();
        assert!(stale_proposal.replay().is_err());

        let mut actual_regime = proposed;
        actual_regime.evidence_kind = EvidenceKind::Available;
        let actual_binding = RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(10),
            snapshot_identity: Arc::from("source-low-arrival"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        };
        let actual = EvidenceCatalog::try_new(
            Arc::clone(&base.environments),
            [base.regimes[0].clone(), actual_regime],
            [base.bindings[0].clone(), actual_binding],
            None,
        )
        .unwrap();
        let arrived = proposal.receive(&actual, "source-low-arrival").unwrap();
        assert_eq!(arrived.provider_snapshot.as_ref(), "source-low-arrival");
        assert_eq!(arrived.functional.catalog().regimes.len(), 2);
        assert!(restored.receive(&actual, "source-low-arrival").is_ok());
        assert!(proposal.receive(&actual, "source-high").is_err());

        let changed_value = EvidenceRegime::try_new(
            RegimeId::from_raw(10),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [VariableId::from_raw(1)],
            [InterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(true),
            }],
            (0..4).map(VariableId::from_raw).collect::<Vec<_>>(),
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let mismatched = EvidenceCatalog::try_new(
            Arc::clone(&base.environments),
            [base.regimes[0].clone(), changed_value],
            [
                base.bindings[0].clone(),
                RegimeBinding {
                    dataset_identity: None,
                    regime: RegimeId::from_raw(10),
                    snapshot_identity: Arc::from("source-low-arrival"),
                    schema_names: Arc::from([]),
                    sampling: SamplingDesign::Independent,
                    weights: None,
                    dependence: DependenceGroup::IndependentStudies,
                },
            ],
            None,
        )
        .unwrap();
        assert!(proposal.receive(&mismatched, "source-low-arrival").is_err());

        let changed_source = EvidenceRegime::try_new(
            RegimeId::from_raw(10),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [VariableId::from_raw(1)],
            [InterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            }],
            (0..4).map(VariableId::from_raw).collect::<Vec<_>>(),
            "other-source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let other_source = Environment::try_new(
            "other-source",
            (0..4)
                .map(|raw| VariableCoordinate {
                    variable: VariableId::from_raw(raw),
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            Arc::<[VariableId]>::from([]),
        )
        .unwrap();
        let wrong_source = EvidenceCatalog::try_new(
            [base.environments[0].clone(), other_source],
            [base.regimes[0].clone(), changed_source],
            [base.bindings[0].clone(), actual.bindings[1].clone()],
            None,
        )
        .unwrap();
        assert!(proposal.receive(&wrong_source, "source-low-arrival").is_err());

        let insufficient_law = EvidenceRegime::try_new(
            RegimeId::from_raw(10),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [VariableId::from_raw(1)],
            [InterventionAssignment {
                variable: VariableId::from_raw(1),
                value: Value::Bool(false),
            }],
            [VariableId::from_raw(1)],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let insufficient = EvidenceCatalog::try_new(
            Arc::clone(&base.environments),
            [base.regimes[0].clone(), insufficient_law],
            [base.bindings[0].clone(), actual.bindings[1].clone()],
            None,
        )
        .unwrap();
        assert!(proposal.receive(&insufficient, "source-low-arrival").is_err());
    }
}
