//! Single-source empirical table. Four reasoning slots with a bootstrap interval.
//!
//! Run: `cargo run -p antecedent --example transport_statistical`
//! Source: `examples/rust/transport_statistical.rs`

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::analysis::{PreparedStudy, StatisticalPreparedState, StudyBuilder};
use antecedent::prelude::*;
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ReasoningView, RegimeBinding, RegimeId, RegimeKind, SamplingDesign,
    TargetSampling, VariableCoordinate, VariableDomain,
};
use antecedent_estimate::{EmpiricalTableOptions, RegimeSample, StatisticalTransportInput};
use antecedent_expr::{Assignment, ExactEvaluationLimits, InterventionAssignment};
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

fn sample(x: i64, y0: usize, y1: usize, snapshot: &str) -> RegimeSample {
    let y = std::iter::repeat_n(Some(0.0), y0).chain(std::iter::repeat_n(Some(1.0), y1)).collect();
    RegimeSample {
        population: Arc::from("source"),
        regime: RegimeId::from_raw(0),
        snapshot_identity: Arc::from(snapshot),
        interventions: Arc::from([InterventionAssignment {
            variable: v(0),
            value: Value::Int64(x),
        }]),
        columns: BTreeMap::from([(v(1), y)]),
    }
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
    let ctx = ExecutionContext::for_tests(7);
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
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("v1"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        Some(TargetSampling::RepresentativeSample),
    )?;
    let CatalogTransportResult::Identified(functional) =
        identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx)?
    else {
        return Err("expected catalog-bound statistical transport".into());
    };
    Ok((diagram, *functional))
}

fn prepare(
    functional: BoundTransportFunctional,
    diagram: SelectionDiagram,
    snapshot: &str,
    counts: [(i64, usize, usize); 2],
    x: i64,
) -> Result<PreparedStudy<StatisticalPreparedState>, Box<dyn std::error::Error>> {
    let ctx = ExecutionContext::for_tests(7);
    Ok(StudyBuilder::statistical_transport(
        diagram,
        functional,
        StatisticalTransportInput {
            supplied: Vec::new(),
            samples: counts.into_iter().map(|(xi, y0, y1)| sample(xi, y0, y1, snapshot)).collect(),
        },
        Assignment::from_pairs([(v(0), Value::Int64(x))]),
        ExactEvaluationLimits::default(),
        EmpiricalTableOptions { bootstrap_replicates: 39, ..EmpiricalTableOptions::default() },
        &ctx,
    )?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (diagram, functional) = identify()?;
    let ctx = ExecutionContext::for_tests(7);
    let mut means = Vec::new();
    for x in [0, 1] {
        let prepared =
            prepare(functional.clone(), diagram.clone(), "v1", [(0, 25, 25), (1, 20, 80)], x)?;
        let before = prepared.inspect();
        slots(&before.reasoning, true);
        let result = prepared.estimate_checked(&before.identities.execution, &ctx)?;
        slots(result.reasoning(), true);
        means.push(result.distribution().mean(v(1)).expect("P(Y=1)"));
    }
    assert!((means[0] - 0.5).abs() < 1e-12);
    assert!((means[1] - 0.8).abs() < 1e-12);
    println!("means = {means:?}");
    Ok(())
}
