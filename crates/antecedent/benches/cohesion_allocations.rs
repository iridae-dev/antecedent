//! Fixed release workloads with native allocator accounting. No speedup claim.
#![allow(missing_docs, clippy::too_many_lines)]
use antecedent::analysis::{TransportGridData, TransportGridQuery, TransportGridState};
use antecedent::{PreparedStudy, StudyBuilder};
use antecedent_core::{AssumptionSet, AverageEffectQuery, ContinuousDomain, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext, GridSpec, RegimeId, RegimeKind, ResponseFunctional, ResponseQuery, TransportQuery, Value, VariableCoordinate, VariableDomain, VariableId};
use antecedent_estimate::{DrLearner, TrialAipwInput, TrialAipwOptions, TrialSampling};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData, ExprId,
    IdentifiedEstimand, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportQuery, ClassicalTransportResult, SidLimits, identify_classical_transport,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

struct Meter;
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(size, Ordering::Relaxed);
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
// SAFETY: Every operation delegates to System with the original pointer/layout;
// atomics only observe successful allocations and never allocate themselves.
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, size) };
        if !out.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            allocated(size);
        }
        out
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;
fn measure(mut f: impl FnMut()) -> serde_json::Value {
    f();
    let mut times = Vec::with_capacity(25);
    for _ in 0..25 {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1000.);
    }
    times.sort_by(f64::total_cmp);
    let calls = CALLS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    f();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    let calls = CALLS.load(Ordering::Relaxed) - calls;
    let bytes = BYTES.load(Ordering::Relaxed) - bytes;
    serde_json::json!({"median_ms":times[12],"p95_ms":times[23],"allocation_calls":calls,
        "allocated_bytes":bytes,"peak_extra_live_bytes":peak,"repeats":25})
}
fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}
fn grid_fixture(ctx: &ExecutionContext) -> (TransportGridQuery, TransportGridData) {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, []).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ClassicalTransportResult::Identified(proof) =
        identify_classical_transport(&diagram, &query, SidLimits::default(), ctx).unwrap()
    else {
        panic!("fixture must identify")
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
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        [v(0), v(1)],
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog =
        EvidenceCatalog::try_new([env("source"), env("target")], [regime], [], None).unwrap();
    let law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [v(0), v(1)].map(|variable| DiscreteAxis {
            variable,
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        }),
        [0.4, 0.1, 0.1, 0.4],
        "snapshot",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new(vec![law], 1000).unwrap();
    (
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
    )
}
fn main() {
    let ctx = ExecutionContext::production(9, 1);
    let n = 600;
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).sin()).collect();
    let t: Vec<f64> = (0..n).map(|i| f64::from((i * 17) % 101 < 50)).collect();
    let y: Vec<f64> = z.iter().zip(&t).map(|(z, t)| z + t * (2. + z)).collect();
    let data = antecedent_data::TabularData::from_f64_columns([
        ("a", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([v(2)]), ExprId::from_raw(0));
    let query = AverageEffectQuery::binary_ate(v(0), v(1));
    let estimator = DrLearner::new();
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    let result = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
    let model = result.fitted_effect.as_ref().unwrap();
    let model_bytes = antecedent_io::to_cbor(model.as_ref()).unwrap();
    let loaded: antecedent_estimate::FittedEffect = antecedent_io::from_cbor(&model_bytes).unwrap();
    loaded.validate().unwrap();
    let expected = model.predict(&[v(2)], &[&z], n, &ctx).unwrap();
    assert_eq!(loaded.predict(&[v(2)], &[&z], n, &ctx).unwrap(), expected);
    let (grid_query, grid_data) = grid_fixture(&ctx);
    let grid = StudyBuilder::transport_grid(grid_query.clone(), grid_data.clone(), &ctx).unwrap();
    let grid_result = grid.estimate(&ctx).unwrap();
    let grid_bytes = grid_result.export().unwrap();
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let trial_query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(1),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0., 1.]))),
        }),
        "trial",
        "target",
        [v(0)],
    );
    let trial = StudyBuilder::learned_trial_transport(
        SelectionDiagram::try_new(graph, []).unwrap(),
        trial_query,
        TrialAipwInput {
            features: vec![],
            covariates: vec![],
            outcome: (0..200)
                .map(|i| if i < 120 { 1. + 2. * f64::from(i % 2) } else { 0. })
                .collect(),
            treatment: (0..200).map(|i| i % 2 == 1).collect(),
            source: (0..200).map(|i| i < 120).collect(),
            randomization: vec![0.5; 200],
            sampling: TrialSampling::IndependentSamples,
        },
        TrialAipwOptions {
            bootstrap: 9,
            folds: 3,
            outcome: antecedent_estimate::LearnerSpec::Linear(Default::default()),
            ..Default::default()
        },
        &ctx,
    )
    .unwrap();
    let mut measurements = serde_json::Map::new();
    measurements.insert(
        "ml_cold_preparation".into(),
        measure(|| {
            std::hint::black_box(estimator.prepare(&data, &estimand, &query).unwrap());
        }),
    );
    measurements.insert(
        "ml_cold_nuisance_fit".into(),
        measure(|| {
            let p = estimator.prepare(&data, &estimand, &query).unwrap();
            let r = estimator.fit(&p, &ctx, AssumptionSet::new()).unwrap();
            assert_eq!(r.ate, result.ate);
        }),
    );
    measurements.insert(
        "ml_reused_nuisance_fit".into(),
        measure(|| {
            let r = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
            assert_eq!(r.ate, result.ate);
        }),
    );
    measurements.insert(
        "loaded_numerical_model_prediction".into(),
        measure(|| {
            assert_eq!(loaded.predict(&[v(2)], &[&z], n, &ctx).unwrap(), expected);
        }),
    );
    measurements.insert(
        "grid_cold_preparation".into(),
        measure(|| {
            std::hint::black_box(
                StudyBuilder::transport_grid(grid_query.clone(), grid_data.clone(), &ctx).unwrap(),
            );
        }),
    );
    measurements.insert(
        "grid_repeated_evaluation".into(),
        measure(|| {
            let r = grid.estimate(&ctx).unwrap();
            assert_eq!(r.identity(), grid_result.identity());
        }),
    );
    measurements.insert(
        "grid_reasoning".into(),
        measure(|| {
            std::hint::black_box(grid_result.reasoning());
        }),
    );
    measurements.insert(
        "grid_export".into(),
        measure(|| {
            assert_eq!(grid_result.export().unwrap(), grid_bytes);
        }),
    );
    measurements.insert(
        "grid_verified_load".into(),
        measure(|| {
            std::hint::black_box(
                PreparedStudy::<TransportGridState>::consume(
                    &grid_bytes,
                    ExactEvaluationLimits::default(),
                    &ctx,
                )
                .unwrap(),
            );
        }),
    );
    measurements.insert(
        "trial_joint_bootstrap_9".into(),
        measure(|| {
            assert!((trial.estimate(&ctx).unwrap().estimate().estimate - 2.).abs() < 1e-10);
        }),
    );
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"profile":"release bench","threads":1,"seed":9,"ml_rows":600,"grid_points":2,"trial_rows":200,"workloads":measurements,
        "allocation_scope":"Rust System global allocator; excludes foreign direct malloc and process RSS","comparison":"current tree; warm cache compared with fresh fitting on identical inputs; no branch speedup claim"})).unwrap());
}
