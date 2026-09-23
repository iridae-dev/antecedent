//! Transport identification, catalog search, exact evaluation, and bootstrap.
//!
//! Workloads stay inside the licensed classical/catalog subset. Absolute times
//! live in `benches/baselines/transport.md`.
#![allow(missing_docs, clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::analysis::{PreparedStudy, StatisticalPreparedState, StudyBuilder};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, RegimeBinding, RegimeId, RegimeKind, SamplingDesign,
    TargetSampling, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::{EmpiricalTableOptions, RegimeSample, StatisticalTransportInput};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactTransportData, InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery,
    ClassicalTransportResult, SidLimits, identify_catalog_transport, identify_classical_transport,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}
fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn coordinates(n: u32) -> Vec<VariableCoordinate> {
    (0..n)
        .map(|i| VariableCoordinate { variable: v(i), domain: VariableDomain::Binary, unit: None })
        .collect()
}

fn three_node_frontdoor() -> SelectionDiagram {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    SelectionDiagram::try_new(graph, [v(2)]).unwrap()
}

fn six_node_districts() -> SelectionDiagram {
    let mut graph = Admg::with_variables(6);
    for offset in [0, 3] {
        graph.insert_directed(d(offset), d(offset + 1)).unwrap();
        graph.insert_directed(d(offset + 1), d(offset + 2)).unwrap();
        graph.insert_bidirected(d(offset), d(offset + 2)).unwrap();
    }
    SelectionDiagram::try_new(graph, [v(2), v(5)]).unwrap()
}

fn three_query() -> ClassicalTransportQuery {
    ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn six_query() -> ClassicalTransportQuery {
    ClassicalTransportQuery {
        outcomes: Arc::from([v(2), v(5)]),
        treatments: Arc::from([v(0), v(3)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn catalog(regimes: impl Into<Vec<EvidenceRegime>>) -> EvidenceCatalog {
    EvidenceCatalog::try_new([], regimes.into(), [], None).unwrap()
}

fn observational(n: u32, regime: u32, population: &'static str) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(regime),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        (0..n).map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

fn experimental(mask: u32, n: u32) -> EvidenceRegime {
    let interventions: Vec<_> = (0..n).filter(|i| mask & (1 << i) != 0).map(v).collect();
    let measured: Vec<_> = (0..n).filter(|i| mask & (1 << i) == 0).map(v).collect();
    EvidenceRegime::try_new(
        RegimeId::from_raw(mask),
        if mask == 0 { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        interventions,
        [],
        measured,
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap()
}

fn identify_three(ctx: &ExecutionContext) -> antecedent_identify::ClassicalTransportDerivation {
    let ClassicalTransportResult::Identified(proof) = identify_classical_transport(
        &three_node_frontdoor(),
        &three_query(),
        SidLimits::default(),
        ctx,
    )
    .unwrap() else {
        panic!("three-node frontdoor")
    };
    *proof
}

fn bound_three(ctx: &ExecutionContext) -> BoundTransportFunctional {
    identify_three(ctx).bind_catalog(&catalog([observational(3, 7, "target")])).unwrap()
}

fn uniform_law(n: u32, cells: usize, regime: u32) -> ExactDiscreteLaw {
    let p = 1.0 / cells as f64;
    ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(regime),
        [],
        (0..n)
            .map(|i| DiscreteAxis {
                variable: v(i),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            })
            .collect::<Vec<_>>(),
        vec![p; cells],
        "bench",
        LawTolerance::default(),
    )
    .unwrap()
}

fn evaluate(
    functional: &BoundTransportFunctional,
    data: ExactTransportData,
    outcomes: Vec<VariableId>,
    request: Assignment,
    ctx: &ExecutionContext,
) {
    let plan = ExactEvaluationPlan::compile(
        functional.arena(),
        functional.root(),
        data,
        outcomes,
        request,
        ExactEvaluationLimits::default(),
        LawTolerance::default(),
        ctx,
    )
    .unwrap();
    black_box(plan.evaluate(ctx).unwrap());
}

fn statistical(bootstrap: u32, ctx: &ExecutionContext) -> PreparedStudy<StatisticalPreparedState> {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(d(0), d(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, []).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let coords = coordinates(2);
    let catalog = EvidenceCatalog::try_new(
        [
            Environment::try_new("source", coords.clone(), []).unwrap(),
            Environment::try_new("target", coords, []).unwrap(),
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
        )
        .unwrap()],
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
    )
    .unwrap();
    let CatalogTransportResult::Identified(functional) =
        identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), ctx).unwrap()
    else {
        panic!("catalog statistical")
    };
    let sample = |x: i64, y0: usize, y1: usize| RegimeSample {
        population: Arc::from("source"),
        regime: RegimeId::from_raw(0),
        snapshot_identity: Arc::from("v1"),
        interventions: Arc::from([InterventionAssignment {
            variable: v(0),
            value: Value::Int64(x),
        }]),
        columns: BTreeMap::from([(
            v(1),
            std::iter::repeat_n(Some(0.0), y0).chain(std::iter::repeat_n(Some(1.0), y1)).collect(),
        )]),
    };
    StudyBuilder::statistical_transport(
        diagram,
        *functional,
        StatisticalTransportInput {
            supplied: Vec::new(),
            samples: vec![sample(0, 25, 25), sample(1, 20, 80)],
        },
        Assignment::from_pairs([(v(0), Value::Int64(1))]),
        ExactEvaluationLimits::default(),
        EmpiricalTableOptions {
            bootstrap_replicates: bootstrap,
            ..EmpiricalTableOptions::default()
        },
        ctx,
    )
    .unwrap()
}

fn bench_transport(c: &mut Criterion) {
    let ctx = ExecutionContext::for_tests(3);
    let three = three_node_frontdoor();
    let six = six_node_districts();
    let q3 = three_query();
    let q6 = six_query();
    c.bench_function("identify_classical_3node", |b| {
        b.iter(|| {
            let ClassicalTransportResult::Identified(proof) = identify_classical_transport(
                black_box(&three),
                black_box(&q3),
                SidLimits::default(),
                &ctx,
            )
            .unwrap() else {
                panic!("identifiable")
            };
            black_box(*proof);
        });
    });
    c.bench_function("identify_classical_6node_districts", |b| {
        b.iter(|| {
            let ClassicalTransportResult::Identified(proof) = identify_classical_transport(
                black_box(&six),
                black_box(&q6),
                SidLimits::default(),
                &ctx,
            )
            .unwrap() else {
                panic!("identifiable")
            };
            black_box(*proof);
        });
    });

    let proof = identify_three(&ctx);
    let one = catalog([observational(3, 0, "target")]);
    let four = catalog((0..4).map(|mask| experimental(mask, 3)).collect::<Vec<_>>());
    c.bench_function("catalog_search_1_regime", |b| {
        b.iter(|| {
            black_box(
                proof.search_catalog(&three, black_box(&one), SidLimits::default(), &ctx).unwrap(),
            )
        });
    });
    c.bench_function("catalog_search_4_regimes", |b| {
        b.iter(|| {
            black_box(
                proof.search_catalog(&three, black_box(&four), SidLimits::default(), &ctx).unwrap(),
            )
        });
    });

    let bound3 = bound_three(&ctx);
    let data3 = ExactTransportData::try_new([uniform_law(3, 8, 7)], 1000).unwrap();
    let ClassicalTransportResult::Identified(proof6) =
        identify_classical_transport(&six, &q6, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("six")
    };
    let bound6 = proof6.bind_catalog(&catalog([observational(6, 7, "target")])).unwrap();
    let data6 = ExactTransportData::try_new([uniform_law(6, 64, 7)], 1000).unwrap();
    c.bench_function("evaluate_exact_3node_8cell", |b| {
        b.iter(|| {
            evaluate(
                &bound3,
                black_box(data3.clone()),
                vec![v(2)],
                Assignment::from_pairs([(v(0), Value::Int64(1))]),
                &ctx,
            );
        });
    });
    c.bench_function("evaluate_exact_6node_64cell", |b| {
        b.iter(|| {
            evaluate(
                &bound6,
                black_box(data6.clone()),
                vec![v(2), v(5)],
                Assignment::from_pairs([(v(0), Value::Int64(1)), (v(3), Value::Int64(0))]),
                &ctx,
            );
        });
    });

    let prepared0 = statistical(0, &ctx);
    let prepared19 = statistical(19, &ctx);
    let exec0 = prepared0.inspect().identities.execution.clone();
    let exec19 = prepared19.inspect().identities.execution.clone();
    c.bench_function("statistical_plugin_bootstrap_0", |b| {
        b.iter(|| black_box(prepared0.estimate_checked(&exec0, &ctx).unwrap()));
    });
    c.bench_function("statistical_plugin_bootstrap_19", |b| {
        b.iter(|| black_box(prepared19.estimate_checked(&exec19, &ctx).unwrap()));
    });
}

criterion_group!(benches, bench_transport);
criterion_main!(benches);
