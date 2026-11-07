use antecedent_core::{AssumptionSet, SearchLimits};
use antecedent_identify::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};

use super::*;
use crate::obligation_adapters::identification_failure_obligations;
use crate::repair::{RepairClassification, RepairLimits, RepairObjective, repair};
use crate::study_candidate::{ExpectedEvidence, StudyCostDeclaration, StudyKind, UnitRules};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut dag = Dag::with_variables(nodes);
    for (a, b) in edges {
        dag.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    dag
}

fn admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(nodes);
    for (a, b) in directed {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    for (a, b) in bidirected {
        graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
}

fn failed(treatment: u32, outcome: u32) -> IdentificationResult {
    IdentificationResult::not_identified(
        CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(treatment), v(outcome))),
        DerivationTrace::default(),
        AssumptionSet::new(),
        IdentificationPerformanceRecord::default(),
    )
}

fn steps(obligations: &[EvidenceObligation], family: &str) -> Vec<String> {
    obligations
        .iter()
        .filter(|o| o.provenance.family.as_ref() == family)
        .filter_map(|o| o.provenance.proof_step.as_ref().map(ToString::to_string))
        .collect()
}

fn candidate(
    kind: StudyKind,
    interventions: &[u32],
    measured: &[u32],
    label: &str,
) -> DurableStudyCandidate {
    let ids = |raw: &[u32]| raw.iter().copied().map(v).collect::<Arc<[VariableId]>>();
    DurableStudyCandidate {
        label: Arc::from(label),
        kind,
        population: Arc::from("pop"),
        interventions: ids(interventions),
        measured: ids(measured),
        joint_measurement: true,
        sample_size: 100,
        recruitment: Arc::from("census"),
        timing: Arc::from("one wave"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units: 5, unit_label: Arc::from("usd"), sample_budget: 100 },
        feasible: true,
        feasibility_notes: Arc::from([]),
        expected_evidence: Arc::from([ExpectedEvidence {
            population: Arc::from("pop"),
            interventions: ids(interventions),
            intervention_values: Arc::from([]),
            conditioned_on: Arc::from([]),
            measured: ids(measured),
            distribution: DistributionAvailability::Joint,
        }]),
        external_provider: None,
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn limits() -> RepairLimits {
    RepairLimits {
        search: SearchLimits { operations: 1000, depth: 2 },
        memory_limit_bytes: 16 * 1024 * 1024,
    }
}

/// T=0 -> Y=1 confounded by the unmeasured U=2.
#[test]
fn a_confounded_graph_owes_the_back_door_set_with_its_proof_step() {
    let graph = Admg::from_dag(&dag(3, &[(0, 1), (2, 0), (2, 1)])).unwrap();
    let obligations =
        identification_failure_obligations(&failed(0, 1), &graph, &[v(0), v(1)], "pop", None)
            .unwrap();
    assert_eq!(steps(&obligations, "backdoor"), vec!["backdoor.adjustment_set:0,1,2"]);
    let joint = obligations.iter().find(|o| o.provenance.family.as_ref() == "backdoor").unwrap();
    assert_eq!(joint.kind, EvidenceObligationKind::ProvideJointLaw);
    assert!(joint.variables.contains(&v(2)));
    // Unmeasured confounding also owes the panel alternative, with its assumption.
    assert!(obligations.iter().any(|o| o.provenance.family.as_ref() == "panel"));
}

/// T=0 -> M=1 -> Y=2 with U=3 confounding T and Y; only T and Y are observed.
#[test]
fn an_unmeasured_front_door_mediator_is_owed_as_a_joint_law() {
    let graph = Admg::from_dag(&dag(4, &[(0, 1), (1, 2), (3, 0), (3, 2)])).unwrap();
    let obligations =
        identification_failure_obligations(&failed(0, 2), &graph, &[v(0), v(2)], "pop", None)
            .unwrap();
    assert_eq!(steps(&obligations, "frontdoor"), vec!["frontdoor.mediator_set:1"]);
    let front = obligations.iter().find(|o| o.provenance.family.as_ref() == "frontdoor").unwrap();
    assert_eq!(front.kind, EvidenceObligationKind::ProvideJointLaw);
    assert_eq!(front.variables.iter().map(|x| x.raw()).collect::<Vec<_>>(), vec![0, 1, 2]);
}

#[test]
fn a_base_law_that_already_identifies_refuses() {
    let graph = dag(4, &[(0, 1), (1, 2), (3, 0), (3, 2)]);
    let observed = [v(0), v(1), v(2)];
    let refused =
        FrontDoorRepairFamily::try_new(graph.clone(), v(0), v(2), "pop", observed).unwrap_err();
    assert_eq!(
        refused.detail,
        antecedent_core::reason_details::identification_repair::NOT_A_FAILURE
    );
    // Back-door adjustment over everything observed, with U measured as well.
    let all = [v(0), v(1), v(2), v(3)];
    let refused = BackdoorRepairFamily::try_new(graph, v(0), v(2), "pop", all, &[]).unwrap_err();
    assert_eq!(
        refused.detail,
        antecedent_core::reason_details::identification_repair::NOT_A_FAILURE
    );
    // An identified result owes nothing.
    let mut identified = failed(0, 1);
    identified.status = IdentificationStatus::NonparametricallyIdentified;
    let graph = Admg::from_dag(&dag(2, &[(0, 1)])).unwrap();
    let refused =
        identification_failure_obligations(&identified, &graph, &[v(0), v(1)], "pop", None)
            .unwrap_err();
    assert_eq!(
        refused.detail,
        antecedent_core::reason_details::identification_repair::NOT_A_FAILURE
    );
}

/// Z=0 -> T=1 -> Y=2 with U=3 confounding T and Y; Z is not observed.
#[test]
fn an_instrument_is_owed_with_its_validity_recorded_as_assumptions() {
    let graph = dag(4, &[(0, 1), (1, 2), (3, 1), (3, 2)]);
    let family = InstrumentRepairFamily::try_new(graph, v(1), v(2), "pop", [v(1), v(2)]).unwrap();
    let obligations = family.unresolved_obligations();
    let steps: Vec<_> =
        obligations.iter().filter_map(|o| o.provenance.proof_step.as_deref()).collect();
    assert_eq!(
        steps,
        vec!["iv.instrument:0", "iv.exclusion:0", "iv.relevance:0", "iv.effect_restriction"]
    );
    let assumptions: Vec<_> = obligations
        .iter()
        .filter(|o| o.kind == EvidenceObligationKind::EstablishAssumption)
        .collect();
    assert_eq!(assumptions.len(), 3);
    assert!(assumptions.iter().all(|o| !o.satisfiable_by_study()));

    // A study that delivers the joint law is the data the Wald ratio needs, yet
    // no study certifies the family: validity stays an assumption.
    let study = candidate(StudyKind::Observation, &[], &[0, 1, 2], "instrumented survey");
    let report = repair(
        &family,
        std::slice::from_ref(&study),
        RepairObjective::MinimizeCost,
        limits(),
        &ctx(),
    )
    .unwrap();
    assert_eq!(report.outcomes[0].classification, RepairClassification::NotCertified);
    assert!(report.best().is_none());
    assert_eq!(
        report.status(),
        Some((
            antecedent_core::reason_code!("transport_missing_evidence"),
            antecedent_core::reason_details::evidence_obligations::WRONG_CONTRACT
        ))
    );
}

#[test]
fn a_graph_without_an_instrument_owes_the_search_for_one() {
    let graph = dag(3, &[(1, 2), (0, 1), (0, 2)]);
    let family = InstrumentRepairFamily::try_new(graph, v(1), v(2), "pop", [v(1), v(2)]).unwrap();
    let obligations = family.unresolved_obligations();
    assert_eq!(obligations.len(), 1);
    assert_eq!(obligations[0].kind, EvidenceObligationKind::EstablishAssumption);
    assert_eq!(obligations[0].provenance.proof_step.as_deref(), Some("iv.existence"));
}

/// T=0 -> W=1 -> Y=2 with T<->Y and W<->Y: a hedge on {T, W, Y}.
#[test]
fn a_hedge_owes_experiments_on_the_treatment_and_on_a_verified_surrogate() {
    let graph = admg(3, &[(0, 1), (1, 2)], &[(0, 2), (1, 2)]);
    let family =
        HedgeRepairFamily::try_new(graph.clone(), v(0), v(2), "pop", [v(0), v(1), v(2)]).unwrap();
    assert!(family.certificate().is_some());
    assert_eq!(family.surrogates(), &[v(1)]);
    let obligations = family.unresolved_obligations();
    assert!(obligations.iter().all(|o| o.kind == EvidenceObligationKind::Intervene));
    let targets: Vec<u32> = obligations.iter().map(|o| o.regime.interventions[0].raw()).collect();
    assert_eq!(targets, vec![0, 1]);
    assert!(obligations[0].provenance.proof_step.as_deref().unwrap().starts_with("hedge:f=["));
    assert_eq!(obligations[1].provenance.proof_step.as_deref(), Some("hedge.surrogate:1"));

    let direct = candidate(StudyKind::Experiment, &[0], &[2], "randomized trial");
    let on_w = candidate(StudyKind::Experiment, &[1], &[2], "mediator trial");
    let elsewhere = candidate(StudyKind::Experiment, &[2], &[0], "wrong direction");
    let report = repair(
        &family,
        &[direct, on_w, elsewhere],
        RepairObjective::MinimizeCost,
        limits(),
        &ctx(),
    )
    .unwrap();
    let verified: Vec<_> = report
        .outcomes
        .iter()
        .filter(|o| o.classification == RepairClassification::VerifiedSufficient)
        .collect();
    assert_eq!(verified.len(), 2);
    assert!(verified.iter().all(|o| o.candidates.len() == 1));
    // An observation never answers a hedge.
    let observed = candidate(StudyKind::Observation, &[], &[0, 1, 2], "bigger survey");
    let report =
        repair(&family, &[observed], RepairObjective::MinimizeCost, limits(), &ctx()).unwrap();
    assert!(report.best().is_none());
}

#[test]
fn an_identified_hedge_contract_refuses() {
    let graph = admg(2, &[(0, 1)], &[]);
    let refused = HedgeRepairFamily::try_new(graph, v(0), v(1), "pop", [v(0), v(1)]).unwrap_err();
    assert_eq!(
        refused.detail,
        antecedent_core::reason_details::identification_repair::NOT_A_FAILURE
    );
}

#[test]
fn a_sensitivity_finding_owes_the_confounder_and_a_robust_one_refuses() {
    // T=0 -> Y=1 adjusted for the measured Z=2; the analysis says a confounder
    // of e-value 1.4 would overturn the conclusion.
    let graph = dag(3, &[(0, 1), (2, 0), (2, 1)]);
    let finding =
        SensitivityFinding { method: SensitivityMethod::EValue, strength: 1.4, robust: false };
    let family = SensitivityRepairFamily::try_new(
        graph.clone(),
        v(0),
        v(1),
        "pop",
        [v(0), v(1), v(2)],
        finding,
    )
    .unwrap();
    assert_eq!(family.confounder(), v(3));
    let obligations = family.unresolved_obligations();
    assert_eq!(obligations.len(), 1);
    assert_eq!(obligations[0].provenance.family.as_ref(), "sensitivity");
    assert_eq!(
        obligations[0].provenance.proof_step.as_deref(),
        Some("sensitivity.e_value:strength=1.4")
    );
    assert!(obligations[0].variables.contains(&v(3)));
    // Measuring the confounder jointly repairs it; the measured covariates alone do not.
    let measured = candidate(StudyKind::Observation, &[], &[0, 1, 2, 3], "assay");
    let unchanged = candidate(StudyKind::Observation, &[], &[0, 1, 2], "more rows");
    let report =
        repair(&family, &[measured, unchanged], RepairObjective::MinimizeCost, limits(), &ctx())
            .unwrap();
    let verified: Vec<_> = report
        .outcomes
        .iter()
        .filter(|o| o.classification == RepairClassification::VerifiedSufficient)
        .collect();
    assert_eq!(verified.len(), 1);
    let robust = SensitivityFinding { robust: true, ..finding };
    let refused =
        SensitivityRepairFamily::try_new(graph, v(0), v(1), "pop", [v(0), v(1), v(2)], robust)
            .unwrap_err();
    assert_eq!(
        refused.detail,
        antecedent_core::reason_details::identification_repair::NOT_A_FAILURE
    );
}
