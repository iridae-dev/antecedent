//! Original checked transport arrival estimation and independently replayed raw counts.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::design_ranking::{DesignRankingRequestWire, evaluate};
use antecedent::analysis::proposal_arrival::{ArrivalArtifact, ArrivalRequest, execute};
use antecedent_core::{
    DependenceGroup, Environment, EvidenceCatalog, EvidenceKind, ExecutionContext, QuantityRole,
    RegimeBinding, SamplingDesign, ScientificQuantity, VariableCoordinate, VariableDomain,
    VariableId,
};
use antecedent_design::{
    DurableStudyCandidate, ExpectedEvidence, RepairFamilyRef, RepairLimits, RepairObjective,
    RepairReportArtifact, StudyCostDeclaration, StudyKind, TransportRepairFamily, UnitRules,
    repair,
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
fn pin(key: &str) -> serde_json::Value {
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("../../../conformance/proposals/arrival/expected.json"))
            .unwrap();
    expected[key].clone()
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
    evaluate(&rank).unwrap()
}
fn law(rid: antecedent_core::RegimeId, active: bool) -> ExactLawWire {
    let counts = if active { vec![20, 80] } else { vec![80, 20] };
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
            counts.iter().map(|x| *x as f64 / 100.).collect::<Vec<_>>(),
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
fn arrival_checked_transport_evaluates_actual_counts_and_full_atom_truth() {
    let request = fixture();
    let (point, actual) =
        count_static_work(|| execute(&request, &ExecutionContext::for_tests(3)).unwrap());
    assert_eq!(point.sample_size, pin("rust_sample_size").as_u64().unwrap());
    assert!((point.mean.unwrap() - pin("active_mean").as_f64().unwrap()).abs() < 1e-12);
    assert_eq!(point.point.probabilities, vec![0.2, 0.8]);
    assert_eq!(point.inference, "empirical_point_only");
    assert_eq!(point.calibration, "unmeasured");
    assert_eq!(point.work.program_compilations, actual.program_compilations);
    assert_eq!(point.work.provider_calls, actual.provider_calls);
    assert!(actual.program_compilations > 0 && actual.provider_calls > 0);
    let mut control = request;
    control.assignment[0].1 = ValueWire::Int64(0);
    let old = execute(&control, &ExecutionContext::for_tests(3)).unwrap();
    assert!((old.mean.unwrap() - pin("control_mean").as_f64().unwrap()).abs() < 1e-12);
    assert!(
        (point.mean.unwrap() - old.mean.unwrap() - pin("contrast").as_f64().unwrap()).abs() < 1e-12
    );
}
#[test]
fn arrival_original_artifact_and_raw_provider_replay_refuses_resealed_changes() {
    let artifact = ArrivalArtifact::produce(fixture(), &ExecutionContext::for_tests(3)).unwrap();
    let bytes = artifact.to_bytes().unwrap();
    let replay = ArrivalArtifact::consume(
        &bytes,
        &artifact.result.proposal_identity,
        &ExecutionContext::for_tests(99),
    )
    .unwrap();
    assert_eq!(artifact.result, replay.result);
    assert!(
        ArrivalArtifact::consume(&bytes, "wrong-proposal", &ExecutionContext::for_tests(3))
            .is_err()
    );
    for change in 0..4 {
        let mut forged = artifact.clone();
        match change {
            0 => forged.result.mean = Some(999.),
            1 => forged.result.point.probabilities = vec![0.9, 0.1],
            2 => {
                forged.request.arrived_laws[1].probabilities = vec![0.5, 0.5];
                forged.request.arrived_laws[1].empirical_counts = Some(vec![50, 50]);
            }
            _ => forged.request.assignment[0].1 = ValueWire::Int64(0),
        }
        // A producer can reseal bytes, but must reproduce the changed scientific result.
        forged.identity = blake3::hash(
            &antecedent_io::convert::to_cbor(&(forged.version, &forged.request, &forged.result))
                .unwrap(),
        )
        .to_hex()
        .to_string();
        assert!(
            detail(
                ArrivalArtifact::consume(
                    &forged.to_bytes().unwrap(),
                    &artifact.result.proposal_identity,
                    &ExecutionContext::for_tests(3)
                )
                .unwrap_err()
            )
            .contains("proposal_arrival.replay_mismatch")
        );
    }
    assert!(
        ArrivalArtifact::consume(
            &bytes[..bytes.len() - 1],
            &artifact.result.proposal_identity,
            &ExecutionContext::for_tests(3)
        )
        .is_err()
    );
}
#[test]
fn arrival_count_snapshot_population_assignment_and_sampling_refusals() {
    let original = fixture();
    for change in 0..8 {
        let mut r = original.clone();
        match change {
            0 => r.arrived_laws[0].empirical_counts = None,
            1 => {
                r.arrived_laws[0].probabilities = vec![0.5, 0.5];
                r.arrived_laws[0].empirical_counts = Some(vec![25, 25]);
            }
            2 => r.arrived_laws[0].snapshot = "wrong-snapshot".into(),
            3 => r.arrived_laws[0].population = "target".into(),
            4 => r.assignment[0].0 = 1,
            5 => r.catalog.bindings[0].sampling = "clustered".into(),
            6 => r.arrived_laws[0].origin = "supplied_exact".into(),
            _ => r.operations = 0,
        }
        assert!(execute(&r, &ExecutionContext::for_tests(3)).is_err(), "change {change}");
    }
    let ctx = ExecutionContext::for_tests(3);
    ctx.cancellation.cancel();
    assert!(detail(execute(&original, &ctx).unwrap_err()).contains("proposal_arrival.cancelled"));
}

#[test]
fn arrival_checked_z_transport_replays_original_proof_and_actual_count_worlds() {
    let artifact =
        ArrivalArtifact::produce(fixture_family(true), &ExecutionContext::for_tests(3)).unwrap();
    assert!((artifact.result.mean.unwrap() - 0.8).abs() < 1e-12);
    assert_eq!(artifact.result.point.probabilities, vec![0.2, 0.8]);
    let replay = ArrivalArtifact::consume(
        &artifact.to_bytes().unwrap(),
        &artifact.result.proposal_identity,
        &ExecutionContext::for_tests(21),
    )
    .unwrap();
    assert_eq!(replay.result, artifact.result);
}
