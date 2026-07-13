//! Manufacturing-style temporal effect example (DESIGN.md §34.2 / Phase 3).
//!
//! Run: `cargo +1.85 test -p causal-analysis --test manufacturing_temporal -- --nocapture`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use causal_analysis::CausalAnalysis;
use causal_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use causal_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use causal_graph::{TemporalDag, ensure_lagged};

#[test]
fn manufacturing_pressure_defect() {
    let n = 400usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 1..n {
        pressure[t] = ((t as f64) * 0.04).sin();
        defect[t] = 0.9 * pressure[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(pressure),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(defect),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 3_600_000_000_000 }, length: n },
    )
    .unwrap();

    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p1, d0).unwrap();

    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);

    let analysis = CausalAnalysis::builder()
        .series(series)
        .temporal_graph(g)
        .temporal_query(q)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(42);
    let result = analysis.run(&ctx).unwrap();

    // Y ≈ 0.9 X_{t-1}; unit pulse ⇒ ATE ≈ 0.9
    assert!(
        (result.estimate.ate - 0.9).abs() < 0.05,
        "ate={} expected ~0.9",
        result.estimate.ate
    );
    assert_eq!(&*result.logical_plan.plan_id, "phase3.temporal_effect");
    assert!(result.physical_plan.estimated_peak_memory_bytes.is_some());
    assert!(result.physical_plan.estimated_copy_bytes.is_some());
    assert!(!result.physical_plan.task_schedule.is_empty());
    println!(
        "manufacturing ATE={:.4} plan={} peak_mem={:?} schedule={:?}",
        result.estimate.ate,
        result.logical_plan.plan_id,
        result.physical_plan.estimated_peak_memory_bytes,
        result.physical_plan.task_schedule,
    );
}
