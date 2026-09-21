//! Complementary sources plus a retained grid. Four reasoning slots; exact laws.
//!
//! Run: `cargo run -p antecedent --example transport_meta_grid`
//! Source: `examples/rust/transport_meta_grid.rs`

use std::sync::Arc;

use antecedent::analysis::{
    PreparedStudy, StudyBuilder, TransportGridData, TransportGridPoint, TransportGridQuery,
    TransportGridState,
};
use antecedent::prelude::*;
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ReasoningView, RegimeId, RegimeKind, VariableCoordinate, VariableDomain,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportResult, MetaSource, MetaTransportQuery, SidLimits, identify_meta_transport,
};

fn v(id: u32) -> VariableId {
    VariableId::from_raw(id)
}

fn slots(view: &ReasoningView, uncertainty: bool) {
    assert!(view.identification.is_available());
    assert!(view.assumptions.is_available());
    assert_eq!(view.uncertainty.is_available(), uncertainty);
    println!(
        "identification={} support={} uncertainty={} assumptions={}",
        view.identification.is_available(),
        view.support.is_available(),
        view.uncertainty.is_available(),
        view.assumptions.is_available()
    );
}

fn coordinates() -> [VariableCoordinate; 3] {
    [0, 1, 2].map(|n| VariableCoordinate {
        variable: v(n),
        domain: VariableDomain::Binary,
        unit: None,
    })
}

fn law(
    population: &str,
    regime: u32,
    treatment: u32,
    outcome: u32,
    x: i64,
    p: [f64; 2],
) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        population,
        RegimeId::from_raw(regime),
        [InterventionAssignment { variable: v(treatment), value: Value::Int64(x) }],
        [DiscreteAxis {
            variable: v(outcome),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        }],
        p,
        "supplied-v1",
        LawTolerance::default(),
    )
    .expect("normalized binary law")
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))?;
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2))?;
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))?;
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2))?;
    let query = MetaTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        target: Arc::from("target"),
        sources: vec![
            MetaSource { population: "a".into(), selections: vec![2] },
            MetaSource { population: "b".into(), selections: vec![1] },
        ],
    };
    let ctx = ExecutionContext::for_tests(0);
    let ClassicalTransportResult::Identified(proof) =
        identify_meta_transport(&graph, &query, SidLimits::default(), &ctx)?
    else {
        return Err("expected complementary sources to identify".into());
    };
    let catalog = EvidenceCatalog::try_new(
        [
            Environment::try_new("a", coordinates(), [v(2)])?,
            Environment::try_new("b", coordinates(), [v(1)])?,
            Environment::try_new("target", coordinates(), [])?,
        ],
        [
            EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Experimental,
                EvidenceKind::Available,
                [v(0)],
                [],
                [v(1)],
                "a",
                DistributionAvailability::Joint,
            )?,
            EvidenceRegime::try_new(
                RegimeId::from_raw(1),
                RegimeKind::Experimental,
                EvidenceKind::Available,
                [v(1)],
                [],
                [v(2)],
                "b",
                DistributionAvailability::Joint,
            )?,
        ],
        [],
        None,
    )?;
    let functional = proof.bind_catalog(&catalog)?;
    let diagram = SelectionDiagram::try_new(graph, [v(2)])?;
    let data = ExactTransportData::try_new(
        [
            law("a", 0, 0, 1, 0, [0.8, 0.2]),
            law("a", 0, 0, 1, 1, [0.2, 0.8]),
            law("b", 1, 1, 2, 0, [0.9, 0.1]),
            law("b", 1, 1, 2, 1, [0.1, 0.9]),
        ],
        1000,
    )?;
    let prepared: PreparedStudy<TransportGridState> = StudyBuilder::transport_grid(
        TransportGridQuery {
            diagram,
            functional,
            at: vec![
                Assignment::from_pairs([(v(0), Value::Int64(0))]),
                Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ],
            limits: ExactEvaluationLimits::default(),
        },
        TransportGridData::Exact(data),
        &ctx,
    )?;
    let result = prepared.estimate(&ctx)?;
    slots(&result.reasoning(), false);
    let means: Vec<f64> = result
        .points()
        .iter()
        .map(|point| match point {
            TransportGridPoint::Exact(_, executed) => {
                executed.distribution().mean(v(2)).expect("P(Y=1)")
            }
            other => panic!("expected exact grid point, got {other:?}"),
        })
        .collect();
    assert!((means[0] - 0.26).abs() < 1e-12);
    assert!((means[1] - 0.74).abs() < 1e-12);
    println!("means = {means:?}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
