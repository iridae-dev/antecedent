//! Transport an experimental result to a target population from a sampled source table.
//!
//! Question: what is `P(Y = 1 | do(X = x))` in a target population, estimated from the rows of a
//! source experiment, and how uncertain is it?
//!
//! Data and truth: the graph is `X -> Y` with no selection node. The source experiment has 50
//! rows at `X = 0` with 25 successes and 100 rows at `X = 1` with 80 successes, so the plug-in
//! probabilities are 0.5 and 0.8. The catalog declares independent sampling.
//!
//! Asserted: the transported point estimates equal 0.5 and 0.8 to within `1e-12`. Before and
//! after estimation the reasoning view has identification, assumptions and uncertainty
//! available: 40 bootstrap replicates meet the minimum for a percentile interval.
//!
//! What this establishes: the same identified functional as `transport_exact`, evaluated on an
//! empirical table, carries a bootstrap interval. The example does not check the interval's
//! endpoints or its coverage, and the absence of selection nodes remains an assumption.
//!
//! Run: `cargo run -p antecedent --example transport_statistical`
//! Source: `examples/rust/transport_statistical.rs`

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::analysis::{PreparedStudy, StatisticalPreparedState, StudyBuilder};
use antecedent::prelude::*;
use antecedent_contract::ReasoningView;
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, RegimeBinding, RegimeId, RegimeKind, SamplingDesign, TargetSampling,
    VariableCoordinate, VariableDomain,
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
        // A percentile interval needs at least `PERCENTILE_95_MIN_REPLICATES` (40) bootstrap
        // replicates. With fewer, the uncertainty slot is withheld at inspection and after
        // estimation.
        EmpiricalTableOptions { bootstrap_replicates: 40, ..EmpiricalTableOptions::default() },
        &ctx,
    )?)
}

/// Total sample rows per regime table: `ANTECEDENT_LANE_ROWS` (the performance lanes' row
/// tier) split in the registered 50/50 and 20/80 proportions, else the registered 50 and 100.
fn regime_counts() -> [(i64, usize, usize); 2] {
    let Some(rows) =
        std::env::var("ANTECEDENT_LANE_ROWS").ok().and_then(|v| v.parse::<usize>().ok())
    else {
        return [(0, 25, 25), (1, 20, 80)];
    };
    // Multiples of ten keep both proportions exact, so the example's checks still hold.
    let half = (rows / 20).max(1) * 10;
    [(0, half / 2, half / 2), (1, half / 5, half - half / 5)]
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (diagram, functional) = identify()?;
    let ctx = ExecutionContext::for_tests(7);
    let mut means = Vec::new();
    for x in [0, 1] {
        let prepared = prepare(functional.clone(), diagram.clone(), "v1", regime_counts(), x)?;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
