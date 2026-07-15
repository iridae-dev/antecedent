//! Fuzz temporal SamplePlan / LaggedFrame construction.
#![no_main]

use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::{
    Float64Column, LaggedFrame, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    // Cap dimensions to avoid unbounded allocation under the fuzzer.
    let n_vars = (usize::from(data[0]) % 4).max(1);
    let n = (usize::from(data[1]) % 48).max(4);
    let max_lag = u32::from(data[2] % 4);
    let mut b = CausalSchemaBuilder::new();
    for i in 0..n_vars {
        let _ = b.add_variable(
            format!("v{i}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        );
    }
    let Ok(schema) = b.build() else {
        return;
    };
    let mut cols = Vec::with_capacity(n_vars);
    for i in 0..n_vars {
        let values: Vec<f64> = (0..n)
            .map(|r| f64::from(data.get(3 + (r + i) % (data.len() - 3).max(1)).copied().unwrap_or(0)))
            .collect();
        let Ok(col) = Float64Column::new(
            VariableId::from_raw(i as u32),
            Arc::from(values),
            ValidityBitmap::all_valid(n),
        ) else {
            return;
        };
        cols.push(OwnedColumn::Float64(col));
    }
    let Ok(storage) = OwnedColumnarStorage::try_new(schema, cols, None, None) else {
        return;
    };
    let Ok(series) = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    ) else {
        return;
    };
    let vars: Vec<VariableId> = (0..n_vars as u32).map(VariableId::from_raw).collect();
    let _ = LaggedFrame::from_series(&series, &vars, max_lag);
});
