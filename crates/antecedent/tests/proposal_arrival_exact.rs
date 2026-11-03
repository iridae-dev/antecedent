//! Arrival estimation with unequal arms, derived by hand from the generating model.
//!
//! The source experiment randomizes X and Y = I(U < 0.3 + 0.25 X) with U uniform, so the arm
//! means are 0.3 and 0.55 and the contrast is 0.25. The delivered counts are the expected
//! counts of arms of unequal size: control 120 units (84, 36) and treated 80 units (36, 44),
//! 200 in all, the planned sample size. Selection acts on X alone, so the transported
//! P*(y = 1 | do(x)) is each arm's own proportion: 36/120 = 0.3 and 44/80 = 0.55 (not 44/200,
//! which would be the share of the pooled sample). The treated distribution is [0.45, 0.55].
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::design_ranking::{DesignRankingRequestWire, evaluate};
use antecedent::analysis::proposal_arrival::{
    ArrivalArtifact, ArrivalLimits, ArrivalPoint, ArrivalRequest, execute_with_limits,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    DependenceGroup, Environment, EvidenceCatalog, EvidenceKind, ExecutionContext, RegimeBinding,
    SamplingDesign, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_design::{
    DurableStudyCandidate, ExpectedEvidence, RepairFamily, RepairFamilyRef, RepairLimits,
    RepairObjective, RepairReportArtifact, StudyCostDeclaration, StudyKind, TransportRepairFamily,
    UnitRules, repair,
};
use antecedent_expr::execution_counts::count_static_work;
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{ClassicalTransportQuery, SidLimits};
use antecedent_io::query_wire::ValueWire;
use antecedent_io::{
    IoError, exact_law_wire::ExactLawWire, transport_catalog_wire::EvidenceCatalogWire,
};
use serde_json::json;
use std::sync::Arc;
fn execute(r: &ArrivalRequest, ctx: &ExecutionContext) -> Result<ArrivalPoint, IoError> {
    execute_with_limits(r, ctx, ArrivalLimits::default())
}
fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}
fn candidate() -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: "arrive-experiment".into(),
        kind: StudyKind::Experiment,
        population: "source".into(),
        interventions: Arc::from([v(0)]),
        measured: Arc::from([v(1)]),
        joint_measurement: true,
        sample_size: 200,
        recruitment: "independent patients".into(),
        timing: "baseline".into(),
        unit_rules: UnitRules {
            unit: "patient".into(),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units: 1, unit_label: "utility".into(), sample_budget: 200 },
        feasible: true,
        feasibility_notes: Arc::from([]),
        expected_evidence: Arc::from([ExpectedEvidence {
            population: "source".into(),
            interventions: Arc::from([v(0)]),
            intervention_values: Arc::from([]),
            conditioned_on: Arc::from([]),
            measured: Arc::from([v(1)]),
            distribution: antecedent_core::DistributionAvailability::Joint,
        }]),
        external_provider: None,
    }
}
fn ranking(id: &str) -> antecedent::analysis::design_ranking::DesignRankingEvaluation {
    let q = ScientificQuantity {
        variable_id: "theta".into(),
        variable_name: "theta".into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "decision-state".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    };
    let q = antecedent_io::quantity_wire::ScientificQuantityWire::from(&q);
    let rank:DesignRankingRequestWire=serde_json::from_value(json!({
        "decision":{"contract_identity":"arrival-ranking","utility_unit":"utility","action_ids":["wait","act"],"intercepts":[0.,-0.5],"slopes":[0.,1.],"prior":{"kind":"draws","states":[0.25,0.75]}},
        "candidates":[{"semantic_id":id,"sample_size":200,"cost":{"amount":1.,"unit":"utility"},"signal":{"prior_id":"decision-prior","state_quantity":q,"observation_quantity":q,"rng_seed":3,"evidence_lineage":["original-study-proposal"],"conditional_independence":"iid_given_state"},"provider":{"kind":"binomial"}}],"rng_seed":3,"source_digests":["original-prior"]
    })).unwrap();
    evaluate(rank).unwrap()
}
fn law(rid: antecedent_core::RegimeId, active: bool) -> ExactLawWire {
    let counts: Vec<u64> = if active { vec![36, 44] } else { vec![84, 36] };
    let total = counts.iter().sum::<u64>() as f64;
    ExactLawWire::from_law(
        &ExactDiscreteLaw::try_empirical(
            "source",
            rid,
            vec![antecedent_expr::InterventionAssignment {
                variable: v(0),
                value: antecedent_core::Value::Int64(i64::from(active)),
            }],
            vec![DiscreteAxis {
                variable: v(1),
                values: Arc::from([
                    antecedent_core::Value::Int64(0),
                    antecedent_core::Value::Int64(1),
                ]),
            }],
            counts.iter().map(|x| *x as f64 / total).collect::<Vec<_>>(),
            "arrived-1",
            LawTolerance::default(),
        )
        .unwrap()
        .with_empirical_counts(counts)
        .unwrap(),
    )
}
fn fixture() -> ArrivalRequest {
    fixture_family(false)
}
fn fixture_family(z: bool) -> ArrivalRequest {
    let coord =
        |i| VariableCoordinate { variable: v(i), domain: VariableDomain::Binary, unit: None };
    let source = Environment::try_new("source", [coord(0), coord(1)], [v(0)]).unwrap();
    let target = Environment::try_new("target", [coord(0), coord(1)], []).unwrap();
    let base = EvidenceCatalog::try_new([source, target], [], [], None).unwrap();
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    if z {
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, [v(0)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: "source".into(),
        target: "target".into(),
    };
    let ctx = ExecutionContext::for_tests(3);
    let family =
        TransportRepairFamily::try_new(diagram, query, base.clone(), SidLimits::default(), &ctx)
            .unwrap();
    let study = candidate();
    let z_query = antecedent_identify::ZTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        controllable: Arc::from([v(0)]),
        experiment_assignment: Arc::from([]),
        source: "source".into(),
        target: "target".into(),
    };
    let z_family = if z {
        let snapshot = antecedent_design::snapshot_z_transport_failure(
            family.diagram(),
            &z_query,
            &base,
            SidLimits::default(),
            &ctx,
        )
        .unwrap();
        Some(
            antecedent_design::ZTransportRepairFamily::try_new(snapshot, SidLimits::default())
                .unwrap(),
        )
    } else {
        None
    };
    let actual: &dyn antecedent_design::RepairFamily =
        z_family.as_ref().map_or(&family as &dyn antecedent_design::RepairFamily, |f| f);
    let reference =
        z_family.as_ref().map_or(RepairFamilyRef::Transport(&family), RepairFamilyRef::ZTransport);
    let report = repair(
        actual,
        &[study.clone()],
        RepairObjective::MinimizeCost,
        RepairLimits::default(),
        &ctx,
    )
    .unwrap();
    let artifact = RepairReportArtifact::build(reference, &[study.clone()], &report).unwrap();
    let id = study.semantic_id().to_string();
    let ranked = ranking(&id);
    let mut regimes = family.hypothetical_regimes(&[&study]).unwrap();
    regimes[0].evidence_kind = EvidenceKind::Available;
    let rid = regimes[0].id;
    let catalog = EvidenceCatalog::try_new(
        base.environments.clone(),
        regimes,
        [RegimeBinding {
            dataset_identity: Some("arrived-dataset".into()),
            regime: rid,
            snapshot_identity: "arrived-1".into(),
            schema_names: Arc::from(["y".into()]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap();
    ArrivalRequest {
        repair_artifact: artifact.to_bytes("repair").unwrap(),
        ranking_artifact: ranked.export("rank").unwrap(),
        candidate_id: id,
        catalog: EvidenceCatalogWire::from_catalog(&catalog),
        base_laws: vec![],
        arrived_laws: vec![law(rid, false), law(rid, true)],
        assignment: vec![(0, ValueWire::Int64(1))],
        operations: 100_000,
        depth: 128,
        support_rows: 1024,
    }
}
fn detail(e: IoError) -> String {
    match e {
        IoError::Refused { message, .. } => message,
        _ => panic!("not a typed refusal: {e}"),
    }
}
#[test]
fn unequal_arms_are_normalized_within_each_world_and_match_the_generating_model() {
    for z in [false, true] {
        let request = fixture_family(z);
        let treated = execute(&request, &ExecutionContext::for_tests(3)).unwrap();
        assert_eq!(treated.sample_size, 200);
        assert!((treated.mean.unwrap() - 0.55).abs() < 1e-12, "z={z}: {:?}", treated.mean);
        assert!((treated.point.probabilities[0] - 0.45).abs() < 1e-12);
        assert!((treated.point.probabilities[1] - 0.55).abs() < 1e-12);
        let mut control = request;
        control.assignment[0].1 = ValueWire::Int64(0);
        let control = execute(&control, &ExecutionContext::for_tests(3)).unwrap();
        assert!((control.mean.unwrap() - 0.3).abs() < 1e-12, "z={z}: {:?}", control.mean);
        assert!((treated.mean.unwrap() - control.mean.unwrap() - 0.25).abs() < 1e-12);
    }
}
