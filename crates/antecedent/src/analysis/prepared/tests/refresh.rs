//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};

use crate::analysis::builder::{DataInput, RefuteSuite};
use crate::analysis::execute::Study;

fn xy_series(n: usize) -> TimeSeriesData {
    xy_series_with_slope(n, 0.8)
}

#[allow(clippy::cast_precision_loss)]
fn xy_series_with_slope(n: usize, slope: f64) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [("x", RoleHint::TreatmentCandidate), ("y", RoleHint::OutcomeCandidate)] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.07).sin();
        y[t] = slope * x[t - 1];
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

fn retained_rows(prepared: &crate::analysis::prepared::PreparedStudy) -> usize {
    match &prepared.analysis.data {
        DataInput::Temporal(data) => data.row_count(),
        _ => panic!("series handle must retain series data"),
    }
}

#[test]
fn refresh_recomputes_numerics_and_preserves_cloned_snapshot_after_cached_contract() {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let study = |data| {
        Study::series(data)
            .graph(graph.clone())
            .temporal_query(query.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    };
    let ctx = ExecutionContext::for_tests(2);
    let original = xy_series(160);
    let changed = xy_series_with_slope(160, 2.3);
    let mut warm = study(original.clone()).prepare(&ctx).unwrap();
    // Prime both the result path and the lazy contract/program payload cache.
    assert!((warm.estimate_series(&original, &ctx).unwrap().effect() - 0.8).abs() < 1e-10);
    let before = warm.contract().unwrap().identities;
    assert_eq!(before, warm.contract().unwrap().identities);
    let retained = warm.clone();

    let refreshed = warm.refresh_series(changed.clone(), &ctx).unwrap();
    let cold = study(changed.clone()).prepare(&ctx).unwrap();
    let fresh = cold.estimate_series(&changed, &ctx).unwrap();
    // Independent structural truth: Y[t] = 2.3 X[t-1], with unit pulse contrast.
    assert!((refreshed.effect() - 2.3).abs() < 1e-10);
    assert_eq!(refreshed.effect().to_bits(), fresh.effect().to_bits());
    let after = warm.contract().unwrap().identities;
    assert_eq!(before.program, after.program);
    assert_ne!(before.data_snapshot, after.data_snapshot);
    assert_eq!(after, cold.contract().unwrap().identities);
    assert_eq!(before, retained.contract().unwrap().identities);
    assert!((retained.estimate_series(&original, &ctx).unwrap().effect() - 0.8).abs() < 1e-10);
}

#[test]
fn failed_series_refresh_leaves_handle_unchanged() {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let ctx = ExecutionContext::for_tests(2);
    let mut prepared = Study::series(xy_series(160))
        .graph(graph)
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert_eq!(retained_rows(&prepared), 160);
    let original = xy_series(160);
    let before = prepared.contract().unwrap().identities;

    // Same schema and regularity, so the compatibility gate passes and the
    // failure comes from estimation itself.
    assert!(prepared.refresh_series(xy_series(2), &ctx).is_err());
    assert_eq!(retained_rows(&prepared), 160, "failed refresh must not replace data");
    assert_eq!(before, prepared.contract().unwrap().identities);
    let recovered = prepared.estimate_series(&original, &ctx).unwrap();
    assert!(recovered.effect().is_finite(), "old-data estimate must survive a failed refresh");
    assert_eq!(before, prepared.contract().unwrap().identities);

    prepared.refresh_series(xy_series(140), &ctx).unwrap();
    assert_eq!(retained_rows(&prepared), 140);
    let after = prepared.contract().unwrap().identities;
    assert_eq!(before.program, after.program);
    assert_ne!(before.data_snapshot, after.data_snapshot);
}
