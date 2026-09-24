//! Cost ranking for finite hypothetical transport-evidence additions.
//!
//! Only a candidate whose proposed catalog produces a checked, bound transport
//! functional is called sufficient. This is a structural sufficiency and cost
//! ranking; it assigns no probability to candidate success.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, sync::Arc};

use antecedent_core::{EvidenceCatalog, EvidenceCatalogDelta};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery, SidLimits,
    identify_catalog_transport, verify_classical_transport,
};
use thiserror::Error;

use crate::{CandidateDesign, DesignCost};

/// Check that a declared study action can produce every regime in its delta.
///
/// This guards against a planner claiming sufficiency from catalog rows that its
/// stated intervention or measurement action could not have collected.
fn validate_candidate_delta(
    candidate: &TransportEvidenceCandidate,
) -> Result<(), TransportPlanningError> {
    if candidate.delta.proposed_regimes.is_empty() {
        return Err(TransportPlanningError::InvalidSpec(
            "candidate delta must contain at least one proposed regime".into(),
        ));
    }
    for regime in candidate.delta.proposed_regimes.iter() {
        let matches = match &candidate.design {
            CandidateDesign::Intervene(plan) => {
                regime.kind == antecedent_core::RegimeKind::Experimental
                    && same_variables(&plan.targets, &regime.interventions)
            }
            CandidateDesign::Measure(plan) => {
                regime.kind == antecedent_core::RegimeKind::Observational
                    && regime.interventions.is_empty()
                    && plan.variables.iter().all(|variable| regime.measured.contains(variable))
            }
            CandidateDesign::ObserveEnvironment(_) | CandidateDesign::IncreaseSamplingRate(_) => {
                false
            }
        };
        if !matches {
            return Err(TransportPlanningError::InvalidSpec(format!(
                "candidate {} design does not produce proposed regime {}",
                candidate.id,
                regime.id.raw()
            )));
        }
    }
    Ok(())
}

fn same_variables(
    left: &[antecedent_core::VariableId],
    right: &[antecedent_core::VariableId],
) -> bool {
    left.len() == right.len() && left.iter().all(|variable| right.contains(variable))
}

/// One feasible, caller-declared transport evidence addition and its cost.
#[derive(Clone, Debug)]
pub struct TransportEvidenceCandidate {
    /// Stable identity used to report and break equal-cost ties.
    pub id: Arc<str>,
    /// Immutable hypothetical addition. Every regime must remain `Proposed`.
    pub delta: EvidenceCatalogDelta,
    /// Design action represented by the proposed catalog addition.
    pub design: CandidateDesign,
    /// Recruitment and sampling assumptions for this candidate.
    pub recruitment_sampling: Arc<str>,
    /// Explicit feasibility premises; empty means no additional constraints.
    pub feasibility_constraints: Arc<[Arc<str>]>,
    /// Caller-defined cost units and sample budget.
    pub cost: DesignCost,
}

/// Finite candidate universe and computational limit for a transport plan.
#[derive(Clone, Debug)]
pub struct TransportPlanSpec {
    /// All feasible candidates considered by this plan, in declared order.
    pub candidates: Vec<TransportEvidenceCandidate>,
    /// Maximum number of candidates to evaluate. Remaining items stay unevaluated.
    pub max_evaluated: usize,
    /// Search limits passed to the checked catalog-aware identifier.
    pub identification_limits: SidLimits,
}

/// Outcome for one evaluated candidate.
#[derive(Clone, Debug)]
pub enum TransportCandidateOutcome {
    /// The hypothetical addition produced a checked and bound functional.
    VerifiedSufficient(TransportProposal),
    /// Search found no available binding; this candidate is not sufficient.
    MissingEvidence {
        /// Unmet source-factor obligations from the bounded alternatives.
        obligations: Arc<[Arc<str>]>,
    },
    /// Bounded search did not certify sufficiency; no impossibility is implied.
    NotCertified {
        /// Scope notes for the attempted bounded search.
        obligations: Arc<[Arc<str>]>,
    },
    /// Identification or validation failed before a sufficiency decision.
    Unresolved {
        /// Stable diagnostic string from validation or search.
        reason: Arc<str>,
    },
}

/// Per-candidate assessment in declared input order.
#[derive(Clone, Debug)]
pub struct TransportCandidateAssessment {
    /// Candidate identity.
    pub id: Arc<str>,
    /// Declared cost.
    pub cost: DesignCost,
    /// Structural sufficiency result.
    pub outcome: TransportCandidateOutcome,
}

/// Finite transport planning result.
#[derive(Clone, Debug)]
pub struct TransportPlanResult {
    /// Assessments for evaluated candidates, in declared order.
    pub assessments: Vec<TransportCandidateAssessment>,
    /// Candidate IDs of verified sufficient additions, sorted by cost then ID.
    pub ranked_sufficient: Vec<Arc<str>>,
    /// Number of candidates in the declared universe.
    pub candidate_universe_size: usize,
    /// The configured maximum candidate evaluations.
    pub search_limit: usize,
    /// True when some declared candidates were not evaluated.
    pub truncated: bool,
}

/// Replayable verified proposal. It retains the exact base catalog, delta,
/// query and graph used to check the proposed addition.
#[derive(Clone, Debug)]
pub struct TransportProposal {
    candidate: TransportEvidenceCandidate,
    base_catalog: EvidenceCatalog,
    diagram: SelectionDiagram,
    query: ClassicalTransportQuery,
    limits: SidLimits,
    checked: BoundTransportFunctional,
}

/// Durable proposal payload. Replay requires the original diagram, query, base
/// catalog, and an execution context; this payload never promotes evidence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransportProposalWire {
    /// Wire schema version.
    pub version: u32,
    /// Stable candidate identity.
    pub candidate_id: String,
    /// Declared intervention or measurement design.
    pub design: CandidateDesignRecord,
    /// Recruitment and sampling premise.
    pub recruitment_sampling: String,
    /// Feasibility constraints declared by the planner.
    pub feasibility_constraints: Vec<String>,
    /// Candidate cost.
    pub cost: DesignCost,
    /// Digest of the exact base catalog needed to replay this proposal.
    pub base_catalog_digest: String,
    /// Proposed catalog delta.
    pub delta: antecedent_io::transport_catalog_wire::EvidenceCatalogDeltaWire,
    /// Checked derivation to replay against the original query and graph.
    pub proof: antecedent_io::transport_proof::TransportProofWire,
}

/// Portable candidate-design description using stable numeric identities.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateDesignRecord {
    /// Stable design kind.
    pub kind: String,
    /// Variable or environment identifiers, depending on kind.
    pub identities: Vec<u32>,
    /// Additional sample or row count when the design declares one.
    pub additional_samples: Option<u64>,
    /// Semantic random-stream tag.
    pub tag: u64,
}

impl TransportProposalWire {
    /// Recheck a deserialized proposal against its original inputs and bind its
    /// proposed evidence in a temporary catalog. The base catalog is untouched.
    ///
    /// # Errors
    /// The base digest, delta, checked proof, or provider binding does not match.
    pub fn replay(
        &self,
        diagram: &SelectionDiagram,
        query: &ClassicalTransportQuery,
        base_catalog: &EvidenceCatalog,
        ctx: &antecedent_core::ExecutionContext,
        limits: SidLimits,
    ) -> Result<BoundTransportFunctional, TransportPlanningError> {
        let supplied_digest = catalog_digest(base_catalog)
            .map_err(|error| TransportPlanningError::InvalidSpec(error.to_string()))?;
        if self.version != 1 || supplied_digest != self.base_catalog_digest {
            return Err(TransportPlanningError::InvalidSpec(
                "proposal version or base catalog digest does not match".into(),
            ));
        }
        let delta = self
            .delta
            .to_delta(base_catalog)
            .map_err(|error| TransportPlanningError::InvalidSpec(error.to_string()))?;
        let preview = delta.preview_catalog(base_catalog)?;
        let derivation = self
            .proof
            .check(diagram, query, limits, ctx)
            .map_err(|error| TransportPlanningError::InvalidSpec(error.to_string()))?;
        derivation.bind_catalog(&preview).map_err(TransportPlanningError::Identification)
    }
}

impl TransportProposal {
    /// Candidate whose proposed evidence passed the sufficiency check.
    #[must_use]
    pub const fn candidate(&self) -> &TransportEvidenceCandidate {
        &self.candidate
    }

    /// Checked theorem and provider-binding witness.
    #[must_use]
    pub const fn checked_functional(&self) -> &BoundTransportFunctional {
        &self.checked
    }

    /// Serialize the proposed addition and checked derivation for later replay.
    ///
    /// # Errors
    /// The checked expression exceeds the proof wire's supported capacity.
    pub fn to_wire(&self) -> Result<TransportProposalWire, antecedent_io::IoError> {
        Ok(TransportProposalWire {
            version: 1,
            candidate_id: self.candidate.id.to_string(),
            design: candidate_design_record(&self.candidate.design),
            recruitment_sampling: self.candidate.recruitment_sampling.to_string(),
            feasibility_constraints: self
                .candidate
                .feasibility_constraints
                .iter()
                .map(ToString::to_string)
                .collect(),
            cost: self.candidate.cost,
            base_catalog_digest: catalog_digest(&self.base_catalog)?,
            delta: antecedent_io::transport_catalog_wire::EvidenceCatalogDeltaWire::from_delta(
                &self.candidate.delta,
            ),
            proof: antecedent_io::transport_proof::TransportProofWire::from_checked(
                self.checked.derivation(),
            )?,
        })
    }

    /// Recheck the stored proof against its saved base catalog and hypothetical delta.
    ///
    /// # Errors
    /// The graph proof no longer verifies or its required factors no longer bind.
    pub fn replay(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(), TransportPlanningError> {
        let preview = self.candidate.delta.preview_catalog(&self.base_catalog)?;
        verify_classical_transport(
            &self.diagram,
            &self.query,
            self.checked.derivation(),
            self.limits,
            ctx,
        )?;
        self.checked.derivation().bind_catalog(&preview)?;
        Ok(())
    }
}

fn catalog_digest(catalog: &EvidenceCatalog) -> Result<String, antecedent_io::IoError> {
    let wire = antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(catalog);
    let bytes = serde_json::to_vec(&wire)
        .map_err(|error| antecedent_io::IoError::Convert(error.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

pub(crate) fn candidate_design_record(design: &CandidateDesign) -> CandidateDesignRecord {
    match design {
        CandidateDesign::Measure(plan) => CandidateDesignRecord {
            kind: "measure".into(),
            identities: plan.variables.iter().map(|variable| variable.raw()).collect(),
            additional_samples: None,
            tag: plan.tag,
        },
        CandidateDesign::Intervene(plan) => CandidateDesignRecord {
            kind: "intervene".into(),
            identities: plan.targets.iter().map(|variable| variable.raw()).collect(),
            additional_samples: None,
            tag: plan.tag,
        },
        CandidateDesign::ObserveEnvironment(plan) => CandidateDesignRecord {
            kind: "observe_environment".into(),
            identities: vec![plan.environment.raw()],
            additional_samples: Some(plan.additional_rows),
            tag: plan.tag,
        },
        CandidateDesign::IncreaseSamplingRate(plan) => CandidateDesignRecord {
            kind: "increase_sampling_rate".into(),
            identities: Vec::new(),
            additional_samples: Some(plan.additional_samples),
            tag: plan.tag,
        },
    }
}

/// Invalid plan specification or an error while replaying a checked proposal.
#[derive(Debug, Error)]
pub enum TransportPlanningError {
    /// Empty or duplicate candidate identity, invalid cost, or unsupported zero limit.
    #[error("invalid transport plan specification: {0}")]
    InvalidSpec(String),
    /// The hypothetical catalog violates the base catalog contract.
    #[error(transparent)]
    Catalog(#[from] antecedent_core::QueryError),
    /// The checked transport proof did not verify or bind.
    #[error(transparent)]
    Identification(#[from] antecedent_identify::IdentificationError),
}

/// Evaluate a finite list of hypothetical evidence additions and rank only
/// those that establish a checked sufficient route.
///
/// Proposed evidence is promoted to available only in each temporary preview
/// catalog. The caller's `base_catalog` and prepared evidence remain unchanged.
///
/// # Errors
/// The candidate catalog itself is malformed (including duplicate IDs or bad costs).
pub fn plan_transport_evidence(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    base_catalog: &EvidenceCatalog,
    spec: &TransportPlanSpec,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<TransportPlanResult, TransportPlanningError> {
    base_catalog
        .validate()
        .map_err(|error| TransportPlanningError::InvalidSpec(error.to_string()))?;
    if spec.max_evaluated == 0 {
        return Err(TransportPlanningError::InvalidSpec(
            "max_evaluated must be greater than zero".into(),
        ));
    }
    let mut ids = BTreeSet::<Arc<str>>::new();
    for candidate in &spec.candidates {
        if candidate.id.trim().is_empty()
            || candidate.recruitment_sampling.trim().is_empty()
            || !ids.insert(Arc::clone(&candidate.id))
            || !candidate.cost.amount.is_finite()
            || candidate.cost.amount < 0.0
        {
            return Err(TransportPlanningError::InvalidSpec(
                "candidate IDs must be unique and costs finite and nonnegative".into(),
            ));
        }
        // Validate the delta even when it falls beyond the evaluation cap.
        candidate.delta.preview_catalog(base_catalog)?;
        validate_candidate_delta(candidate)?;
    }

    let count = spec.max_evaluated.min(spec.candidates.len());
    let mut assessments = Vec::with_capacity(count);
    for candidate in spec.candidates.iter().take(count) {
        let preview = candidate.delta.preview_catalog(base_catalog)?;
        let outcome = match identify_catalog_transport(
            diagram,
            query,
            &preview,
            spec.identification_limits,
            ctx,
        ) {
            Ok(CatalogTransportResult::Identified(checked)) => {
                TransportCandidateOutcome::VerifiedSufficient(TransportProposal {
                    candidate: candidate.clone(),
                    base_catalog: base_catalog.clone(),
                    diagram: diagram.clone(),
                    query: query.clone(),
                    limits: spec.identification_limits,
                    checked: *checked,
                })
            }
            Ok(CatalogTransportResult::MissingEvidence { obligations, .. }) => {
                TransportCandidateOutcome::MissingEvidence { obligations }
            }
            Ok(CatalogTransportResult::NotCertified { obligations, .. }) => {
                TransportCandidateOutcome::NotCertified { obligations }
            }
            Err(error) => {
                TransportCandidateOutcome::Unresolved { reason: Arc::from(error.to_string()) }
            }
        };
        assessments.push(TransportCandidateAssessment {
            id: Arc::clone(&candidate.id),
            cost: candidate.cost,
            outcome,
        });
    }

    let mut sufficient: Vec<_> = assessments
        .iter()
        .filter(|assessment| {
            matches!(assessment.outcome, TransportCandidateOutcome::VerifiedSufficient(_))
        })
        .map(|assessment| (assessment.cost, Arc::clone(&assessment.id)))
        .collect();
    sufficient.sort_by(|a, b| {
        a.0.amount
            .total_cmp(&b.0.amount)
            .then_with(|| a.0.sample_budget.cmp(&b.0.sample_budget))
            .then_with(|| a.1.cmp(&b.1))
    });
    Ok(TransportPlanResult {
        assessments,
        ranked_sufficient: sufficient.into_iter().map(|assessment| assessment.1).collect(),
        candidate_universe_size: spec.candidates.len(),
        search_limit: spec.max_evaluated,
        truncated: count < spec.candidates.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        Environment, EvidenceRegime, RegimeId, RegimeKind, VariableCoordinate, VariableDomain,
        VariableId,
    };
    use antecedent_graph::{Admg, DenseNodeId};

    fn fixture() -> (SelectionDiagram, ClassicalTransportQuery, EvidenceCatalog, VariableId) {
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let coordinate =
            |variable| VariableCoordinate { variable, domain: VariableDomain::Binary, unit: None };
        let source = Environment::try_new("source", [coordinate(x), coordinate(y)], [x]).unwrap();
        let target = Environment::try_new("target", [coordinate(x), coordinate(y)], []).unwrap();
        let catalog = EvidenceCatalog::try_new([source, target], [], [], None).unwrap();
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [x]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([x]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        (diagram, query, catalog, x)
    }

    fn candidate(
        catalog: &EvidenceCatalog,
        x: VariableId,
        id: &str,
        measured: &[VariableId],
        cost: f64,
    ) -> TransportEvidenceCandidate {
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(if id == "good-z" { 2 } else { 1 }),
            RegimeKind::Experimental,
            antecedent_core::EvidenceKind::Proposed,
            [x],
            [],
            measured.to_vec(),
            "source",
            antecedent_core::DistributionAvailability::Joint,
        )
        .unwrap();
        TransportEvidenceCandidate {
            id: Arc::from(id),
            delta: EvidenceCatalogDelta::try_new(catalog, [regime]).unwrap(),
            design: crate::CandidateDesign::Intervene(crate::ExperimentPlan {
                targets: Arc::from([x]),
                cost: DesignCost { amount: cost, sample_budget: 100 },
                tag: 0,
            }),
            recruitment_sampling: Arc::from(
                "randomized source recruitment; joint regime measurement",
            ),
            feasibility_constraints: Arc::from([Arc::from("treatment is manipulable")]),
            cost: DesignCost { amount: cost, sample_budget: 100 },
        }
    }

    #[test]
    fn proposed_experiment_repairs_missing_law_and_proposal_replays() {
        let (diagram, query, catalog, x) = fixture();
        let y = VariableId::from_raw(1);
        let repair = candidate(&catalog, x, "repair", &[y], 2.0);
        let impossible = candidate(&catalog, x, "unmeasured", &[], 0.1);
        let result = plan_transport_evidence(
            &diagram,
            &query,
            &catalog,
            &TransportPlanSpec {
                candidates: vec![repair, impossible],
                max_evaluated: 2,
                identification_limits: SidLimits::default(),
            },
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(result.ranked_sufficient, [Arc::<str>::from("repair")]);
        assert!(!result.truncated);
        assert_eq!(result.candidate_universe_size, 2);
        assert!(matches!(
            result.assessments[0].outcome,
            TransportCandidateOutcome::VerifiedSufficient(_)
        ));
        assert!(!matches!(
            result.assessments[1].outcome,
            TransportCandidateOutcome::VerifiedSufficient(_)
        ));
        let TransportCandidateOutcome::VerifiedSufficient(proposal) =
            &result.assessments[0].outcome
        else {
            unreachable!();
        };
        proposal.replay(&antecedent_core::ExecutionContext::for_tests(1)).unwrap();
        let wire = proposal.to_wire().unwrap();
        let encoded = serde_json::to_vec(&wire).unwrap();
        let decoded: TransportProposalWire = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.candidate_id, "repair");
        assert_eq!(decoded.design.kind, "intervene");
        assert_eq!(decoded.delta.proposed_regimes.len(), 1);
        decoded
            .replay(
                &diagram,
                &query,
                &catalog,
                &antecedent_core::ExecutionContext::for_tests(1),
                SidLimits::default(),
            )
            .unwrap();
        let mut wrong_base = catalog.clone();
        wrong_base.target_sampling = Some(antecedent_core::TargetSampling::ConvenienceSample);
        assert!(
            decoded
                .replay(
                    &diagram,
                    &query,
                    &wrong_base,
                    &antecedent_core::ExecutionContext::for_tests(1),
                    SidLimits::default(),
                )
                .is_err()
        );
        // Preview never promotes evidence in the caller's catalog.
        assert!(catalog.regimes.is_empty());
    }

    #[test]
    fn equal_cost_candidates_rank_by_id_and_cap_is_reported() {
        let (diagram, query, catalog, x) = fixture();
        let candidates = vec![
            candidate(&catalog, x, "z", &[VariableId::from_raw(1)], 1.0),
            candidate(&catalog, x, "a", &[VariableId::from_raw(1)], 1.0),
        ];
        let full = plan_transport_evidence(
            &diagram,
            &query,
            &catalog,
            &TransportPlanSpec {
                candidates: candidates.clone(),
                max_evaluated: 2,
                identification_limits: SidLimits::default(),
            },
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(full.ranked_sufficient, [Arc::<str>::from("a"), Arc::<str>::from("z")]);

        let result = plan_transport_evidence(
            &diagram,
            &query,
            &catalog,
            &TransportPlanSpec {
                candidates,
                max_evaluated: 1,
                identification_limits: SidLimits::default(),
            },
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(result.ranked_sufficient, [Arc::<str>::from("z")]);
        assert!(result.truncated);
        assert_eq!(result.search_limit, 1);
        assert_eq!(result.candidate_universe_size, 2);
    }

    #[test]
    fn zero_search_limit_is_rejected() {
        let (diagram, query, catalog, _) = fixture();
        let error = plan_transport_evidence(
            &diagram,
            &query,
            &catalog,
            &TransportPlanSpec {
                candidates: vec![],
                max_evaluated: 0,
                identification_limits: SidLimits::default(),
            },
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(matches!(error, TransportPlanningError::InvalidSpec(_)));
    }

    #[test]
    fn candidate_design_must_match_its_proposed_regime() {
        let (diagram, query, catalog, x) = fixture();
        let mut bad = candidate(&catalog, x, "bad", &[VariableId::from_raw(1)], 1.0);
        bad.design = crate::CandidateDesign::Measure(crate::MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(1)]),
            cost: bad.cost,
            tag: 0,
        });
        let error = plan_transport_evidence(
            &diagram,
            &query,
            &catalog,
            &TransportPlanSpec {
                candidates: vec![bad],
                max_evaluated: 1,
                identification_limits: SidLimits::default(),
            },
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(matches!(error, TransportPlanningError::InvalidSpec(_)));
    }
}
