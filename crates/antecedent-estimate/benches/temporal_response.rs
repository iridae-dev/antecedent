//! Temporal dose × horizon response benches (ADR 0021).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use antecedent_core::{
    AssumptionSet, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec,
    IdentificationStatus, Intervention, Lag, MeasurementSpec, MechanismOverride,
    ResponseFunctional, ResponseQuery, RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TemporalIndexer,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::TemporalResponseEstimator;
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn series(n: usize) -> (TimeSeriesData, ResponseQuery, IdentifiedEstimand, TemporalIndexer) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 2..n {
        t[i] = 0.2 * t[i - 1] + 0.05 * (i as f64).sin();
        y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2] + 0.01 * (i as f64).cos();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    let temporal =
        TemporalResponseSpec::new(Arc::from([1_u32, 2, 4, 8]), TemporalPolicy::pulse(0), None)
            .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(vec![0.0_f64, 0.5, 1.0, 1.5])),
        ),
    })
    .with_temporal(temporal);
    let id_query =
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(8)
            .with_policy(TemporalPolicy::pulse(0));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&graph, &id_query).unwrap();
    let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
    (data, query, estimand, id_res.indexer)
}

/// Same underlying series/graph shape as [`series`], but for the `InterventionResponse`
/// additive-shift path — the path that was O(n^2 * p) before it was collapsed to the
/// closed form `mu_hat(Abar + delta)` (an O(p) evaluation per horizon, not an O(n) loop
/// over observed treatment levels). `n` is chosen large enough that a reintroduced O(n)
/// (let alone O(n^2)) per-horizon averaging loop would blow the budget decisively, while
/// the closed-form path stays comfortably fast.
fn intervention_shift_series(
    n: usize,
) -> (TimeSeriesData, ResponseQuery, IdentifiedEstimand, TemporalIndexer) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 2..n {
        t[i] = 0.2 * t[i - 1] + 0.05 * (i as f64).sin();
        y[i] = 1.0 + 2.0 * t[i - 1] + 3.0 * t[i - 2] + 0.01 * (i as f64).cos();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    let temporal =
        TemporalResponseSpec::new(Arc::from([1_u32, 2, 4, 8]), TemporalPolicy::pulse(0), None)
            .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from(vec![Intervention::soft(
            VariableId::from_raw(0),
            MechanismOverride::additive_shift(0.5),
        )]),
    })
    .with_temporal(temporal);
    let id_query =
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(8)
            .with_policy(TemporalPolicy::pulse(0));
    let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&graph, &id_query).unwrap();
    let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
    (data, query, estimand, id_res.indexer)
}

fn bench_temporal_response(c: &mut Criterion) {
    // Soft budgets (temporal_response.md); 2× headroom for gate `--test` noise.
    const MULTI_HORIZON_BUDGET: Duration = Duration::from_millis(25);
    // n = 100_000 is comfortably fast for the closed-form O(n*p) path (fit-dominated,
    // same order as the MeanCurve bench above) but would be ~100x+ over budget if the
    // additive-shift path regressed to an O(n) (or worse, the original O(n^2)) loop
    // averaging g-comp over every observed treatment level.
    const INTERVENTION_SHIFT_BUDGET: Duration = Duration::from_millis(200);

    let est = TemporalResponseEstimator::new();

    c.bench_function("temporal_response_multi_horizon_n800", |b| {
        let (data, query, estimand, indexer) = series(800);
        b.iter(|| {
            let r = est
                .estimate(
                    &data,
                    &[(&estimand, &indexer); 4],
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                    &ExecutionContext::for_tests(1),
                )
                .unwrap();
            black_box(r.provenance_id);
        });
    });

    let (data, query, estimand, indexer) = series(800);
    let started = Instant::now();
    let _ = est
        .estimate(
            &data,
            &[(&estimand, &indexer); 4],
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < MULTI_HORIZON_BUDGET,
        "temporal_response_multi_horizon_n800 exceeded soft budget: {elapsed:?} >= {MULTI_HORIZON_BUDGET:?}"
    );

    c.bench_function("temporal_response_intervention_shift_n100000", |b| {
        let (data, query, estimand, indexer) = intervention_shift_series(100_000);
        b.iter(|| {
            let r = est
                .estimate(
                    &data,
                    &[(&estimand, &indexer); 4],
                    &query,
                    IdentificationStatus::NonparametricallyIdentified,
                    AssumptionSet::new(),
                    &ExecutionContext::for_tests(1),
                )
                .unwrap();
            black_box(r.provenance_id);
        });
    });

    let (data, query, estimand, indexer) = intervention_shift_series(100_000);
    let started = Instant::now();
    let _ = est
        .estimate(
            &data,
            &[(&estimand, &indexer); 4],
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < INTERVENTION_SHIFT_BUDGET,
        "temporal_response_intervention_shift_n100000 exceeded soft budget: {elapsed:?} >= {INTERVENTION_SHIFT_BUDGET:?}"
    );
}

criterion_group!(benches, bench_temporal_response);
criterion_main!(benches);
