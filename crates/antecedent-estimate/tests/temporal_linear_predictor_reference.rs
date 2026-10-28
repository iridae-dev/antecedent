//! The lagged linear predictor and its held-level prediction against a 40-digit reference.
//!
//! `oracle_data/temporal_linear_predictor_reference.py` (frozen in the JSON beside it) regresses
//! `y(t)` on an intercept, `x(t-1)`, `x(t-2)` and `z(t-1)` over the lag-aligned rows, and
//! evaluates the fitted line with both lags of `x` set to a level while `z(t-1)` keeps its
//! observed value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, KernelPolicy, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, LaggedColumn, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::TemporalLinearPredictor;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn texts(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect()
}

fn series(columns: Vec<(&str, Vec<f64>)>) -> TimeSeriesData {
    let n = columns[0].1.len();
    let mut builder = CausalSchemaBuilder::new();
    for (name, _) in &columns {
        builder
            .add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let cols = columns
        .into_iter()
        .enumerate()
        .map(|(index, (_, values))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(index as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn lagged(variable: u32, lag: u32) -> LaggedColumn {
    LaggedColumn { variable: VariableId::from_raw(variable), lag: Lag::from_raw(lag) }
}

#[test]
fn coefficients_and_held_level_predictions_equal_the_multiprecision_fit() {
    let fx: Value =
        serde_json::from_str(include_str!("oracle_data/temporal_linear_predictor_reference.json"))
            .unwrap();
    let data =
        series(vec![("x", floats(&fx["x"])), ("z", floats(&fx["z"])), ("y", floats(&fx["y"]))]);
    let policy = KernelPolicy::default_policy();
    let predictor = TemporalLinearPredictor::fit(
        &data,
        VariableId::from_raw(2),
        [lagged(0, 1), lagged(0, 2), lagged(1, 1)],
        &policy,
    )
    .unwrap();
    assert_eq!(predictor.max_lag, 2);
    for (got, want) in predictor.coefficients.iter().zip(texts(&fx["coefficients"])) {
        assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "coefficient {got} vs {want}");
    }
    let levels = floats(&fx["levels"]);
    for (level, want) in levels.iter().zip(fx["predictions"].as_array().unwrap()) {
        let got =
            predictor.predict_conditional(&data, VariableId::from_raw(0), *level, &policy).unwrap();
        let want = texts(want);
        assert_eq!(got.len(), want.len());
        assert_eq!(got.len(), 62);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-9 * w.abs().max(1.0), "level {level}: {g} vs {w}");
        }
    }
    // Holding z instead leaves both x lags observed and sets only the z column.
    let held_z =
        predictor.predict_conditional(&data, VariableId::from_raw(1), 0.0, &policy).unwrap();
    let c = &predictor.coefficients;
    let x = floats(&fx["x"]);
    for (i, g) in held_z.iter().enumerate() {
        let t = i + 2;
        let want = c[0] + c[1] * x[t - 1] + c[2] * x[t - 2];
        assert!((g - want).abs() < 1e-12 * want.abs().max(1.0), "held z row {i}");
    }
}
