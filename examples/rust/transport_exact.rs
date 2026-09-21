//! Exact supplied laws. Four reasoning slots; no sampling interval.
//!
//! Run: `cargo run -p antecedent --example transport_exact`
//! Source: `examples/rust/transport_exact.rs`

use std::sync::Arc;

use antecedent::analysis::{ExactPreparedState, PreparedStudy, StudyBuilder};
use antecedent::prelude::*;
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ReasoningView, RegimeId, RegimeKind, TargetSampling, VariableCoordinate, VariableDomain,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery, SidLimits,
    identify_catalog_transport,
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

fn coordinates() -> [VariableCoordinate; 2] {
    [0, 1].map(|n| VariableCoordinate {
        variable: v(n),
        domain: VariableDomain::Binary,
        unit: None,
    })
}

fn law(x: i64, p: [f64; 2], snapshot: &str) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [InterventionAssignment { variable: v(0), value: Value::Int64(x) }],
        [DiscreteAxis { variable: v(1), values: Arc::from([Value::Int64(0), Value::Int64(1)]) }],
        p,
        snapshot,
        LawTolerance::default(),
    )
    .expect("normalized binary law")
}

fn identify() -> Result<(SelectionDiagram, BoundTransportFunctional), Box<dyn std::error::Error>> {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))?;
    let diagram = SelectionDiagram::try_new(graph, [])?;
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(0);
    let catalog = EvidenceCatalog::try_new(
        [
            Environment::try_new("source", coordinates(), [])?,
            Environment::try_new("target", coordinates(), [])?,
        ],
        [EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [v(0)],
            [],
            [v(1)],
            "source",
            DistributionAvailability::Joint,
        )?],
        [],
        Some(TargetSampling::RepresentativeSample),
    )?;
    let CatalogTransportResult::Identified(functional) =
        identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx)?
    else {
        return Err("expected catalog-bound exact transport".into());
    };
    Ok((diagram, *functional))
}

fn prepare(
    functional: BoundTransportFunctional,
    diagram: SelectionDiagram,
    snapshot: &str,
    p0: [f64; 2],
    p1: [f64; 2],
    x: i64,
) -> Result<PreparedStudy<ExactPreparedState>, Box<dyn std::error::Error>> {
    let ctx = ExecutionContext::for_tests(0);
    Ok(StudyBuilder::exact_transport(
        diagram,
        functional,
        ExactTransportData::try_new([law(0, p0, snapshot), law(1, p1, snapshot)], 1000)?,
        Assignment::from_pairs([(v(0), Value::Int64(x))]),
        ExactEvaluationLimits::default(),
        &ctx,
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (diagram, functional) = identify()?;
    let ctx = ExecutionContext::for_tests(0);
    let mut means = Vec::new();
    for x in [0, 1] {
        let prepared =
            prepare(functional.clone(), diagram.clone(), "source-v1", [0.5, 0.5], [0.2, 0.8], x)?;
        let before = prepared.inspect();
        slots(&before.reasoning, false);
        let result = prepared.estimate_checked(&before.identities.execution, &ctx)?;
        slots(result.reasoning(), false);
        means.push(result.distribution().mean(v(1)).expect("P(Y=1)"));
    }
    assert!((means[0] - 0.5).abs() < 1e-12);
    assert!((means[1] - 0.8).abs() < 1e-12);
    println!("means = {means:?}");

    let refreshed = prepare(functional, diagram, "source-v2", [0.6, 0.4], [0.3, 0.7], 1)?;
    let before = refreshed.inspect();
    let result = refreshed.estimate_checked(&before.identities.execution, &ctx)?;
    assert!((result.distribution().mean(v(1)).expect("P(Y=1)") - 0.7).abs() < 1e-12);
    Ok(())
}
