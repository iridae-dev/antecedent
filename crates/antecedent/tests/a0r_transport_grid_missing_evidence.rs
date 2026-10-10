//! A0 remainder: a transported grid with a coordinate that has no value keeps that
//! coordinate, labelled `missing_evidence`, through the Rust result and its artifact.
//!
//! Hand example. Graph `x -> y` with `x <-> y`. The source supplies exactly one law,
//! `P(y | do(x = 1)) = (0.3, 0.7)`. Two coordinates are requested in order,
//! `do(x = 0)` then `do(x = 1)`: no law exists under `do(x = 0)`, so coordinate 0 has
//! no value and no evidence for one (missing evidence, not a verdict about supplied
//! evidence); coordinate 1 is executable with `E[y] = 0.7`.

use std::sync::Arc;

use antecedent::PreparedStudy;
use antecedent::StudyBuilder;
use antecedent::analysis::{
    TransportGridData, TransportGridPoint, TransportGridQuery, TransportGridState,
};
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, RegimeId, RegimeKind, SupportStatus, Value, VariableCoordinate,
    VariableDomain, VariableId, check_static_point_labels,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{ClassicalTransportQuery, ClassicalTransportResult, SidLimits};

fn v(n: u32) -> VariableId {
    VariableId::from_raw(n)
}

fn grid() -> PreparedStudy<TransportGridState> {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, []).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(0);
    let ClassicalTransportResult::Identified(proof) =
        antecedent_identify::identify_classical_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &ctx,
        )
        .unwrap()
    else {
        panic!("the direct transport question is identified");
    };
    let env = |name| {
        Environment::try_new(
            name,
            [0, 1].map(|n| VariableCoordinate {
                variable: v(n),
                domain: VariableDomain::Binary,
                unit: None,
            }),
            [],
        )
        .unwrap()
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [v(0)],
        [],
        [v(1)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog =
        EvidenceCatalog::try_new([env("source"), env("target")], [regime], [], None).unwrap();
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [InterventionAssignment { variable: v(0), value: Value::Int64(1) }],
        [DiscreteAxis { variable: v(1), values: Arc::from([Value::Int64(0), Value::Int64(1)]) }],
        [0.3, 0.7],
        "snapshot",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new([law], 1000).unwrap();
    StudyBuilder::transport_grid(
        TransportGridQuery {
            diagram,
            functional: proof.bind_catalog(&catalog).unwrap(),
            at: vec![
                Assignment::from_pairs([(v(0), Value::Int64(0))]),
                Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ],
            limits: ExactEvaluationLimits::default(),
        },
        TransportGridData::Exact(data),
        &ctx,
    )
    .unwrap()
}

#[test]
fn a0r_the_coordinate_without_a_value_is_kept_at_its_requested_position() {
    let ctx = ExecutionContext::for_tests(0);
    let result = grid().estimate(&ctx).unwrap();
    // Both requested coordinates are retained, in request order; none is dropped.
    assert_eq!(result.points().len(), 2);
    let TransportGridPoint::Unavailable(missing) = &result.points()[0] else {
        panic!("the unbound coordinate stays in the result as unavailable");
    };
    assert_eq!(missing.kind, "missing_evidence");
    assert!(result.points()[0].distribution().is_none());
    assert_eq!(result.points()[1].support_status(), SupportStatus::Supported);
    // The per-coordinate labels and their summary agree: the summary is the worst label,
    // and missing evidence cannot be reported as a support failure.
    let labels: Vec<SupportStatus> =
        result.points().iter().map(TransportGridPoint::support_status).collect();
    assert_eq!(labels, [SupportStatus::MissingEvidence, SupportStatus::Supported]);
    assert_eq!(check_static_point_labels(&labels, SupportStatus::MissingEvidence, 2), Ok(()));
    let relabeled =
        check_static_point_labels(&labels, SupportStatus::OutsideEmpiricalSupport, 2).unwrap_err();
    assert_eq!(relabeled.detail, "coordinate_support.missing_evidence_relabeled");
    // The executable coordinate carries the hand-derived mean E[y | do(x = 1)] = 0.7.
    let mean = result.points()[1].distribution().unwrap().mean(v(1)).unwrap();
    assert!((mean - 0.7).abs() < 1e-12, "{mean}");
}

#[test]
fn a0r_missing_evidence_survives_the_artifact_round_trip_unchanged() {
    let ctx = ExecutionContext::for_tests(0);
    let result = grid().estimate(&ctx).unwrap();
    let bytes = result.export().unwrap();
    let (_, loaded) = PreparedStudy::<TransportGridState>::consume(
        &bytes,
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(loaded.points().len(), 2);
    assert_eq!(loaded.points()[0].support_status(), SupportStatus::MissingEvidence);
    assert_eq!(loaded.points()[1].support_status(), SupportStatus::Supported);
    assert_eq!(loaded.identity(), result.identity());
    assert_eq!(loaded.export().unwrap(), bytes);
    // A support failure is a different kind and a different label.
    let TransportGridPoint::Unavailable(missing) = &loaded.points()[0] else {
        panic!("unavailable point");
    };
    let mut failed = missing.clone();
    failed.kind = "support_failure".into();
    assert_eq!(
        TransportGridPoint::Unavailable(failed).support_status(),
        SupportStatus::OutsideEmpiricalSupport
    );
}
