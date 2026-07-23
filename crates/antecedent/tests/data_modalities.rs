//! Event align + Panel stacked estimate facade paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent::CausalAnalysis;
use causal_core::{
    CausalSchemaBuilder, DataClassification, ExecutionContext, Lag, MeasurementSpec, RoleHint,
    SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use causal_data::{
    EventData, Float64Column, OwnedColumn, OwnedColumnarStorage, PanelData, PanelUnit,
    SamplingRegularity, TableView, TimeIndex, TimeSeriesData, ValidityBitmap,
};
use causal_graph::{TemporalDag, ensure_lagged};

fn xy_series(n: usize, seed: f64) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.07 + seed).sin();
        y[t] = 0.8 * x[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn lagged_xy_graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

#[test]
fn event_align_then_temporal_effect() {
    // Dense regular events at every ns → align_to_grid(1) recovers the series.
    let series = xy_series(200, 0.0);
    let n = series.row_count();
    let times: Vec<i64> =
        (0..n).map(|i| i64::try_from(i).expect("test row count fits i64")).collect();
    let event = EventData::try_new(series.storage().clone(), Arc::from(times)).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let analysis = CausalAnalysis::builder()
        .events(event, 1)
        .temporal_graph(lagged_xy_graph())
        .temporal_query(q)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!((result.estimate.ate - 0.8).abs() < 0.08, "ate={}", result.estimate.ate);
    assert_eq!(result.logical_plan.data_classification, DataClassification::Event);
}

#[test]
fn panel_stacked_estimate_with_cluster_se() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(180, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(180, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(180, 0.7) },
    ]))
    .unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let analysis = CausalAnalysis::builder()
        .panel(panel)
        .temporal_graph(lagged_xy_graph())
        .temporal_query(q)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(2)).unwrap();
    assert!((result.estimate.ate - 0.8).abs() < 0.08, "ate={}", result.estimate.ate);
    assert_eq!(result.logical_plan.data_classification, DataClassification::Panel);
    assert!(result.estimate.se_analytic.is_finite());
}
