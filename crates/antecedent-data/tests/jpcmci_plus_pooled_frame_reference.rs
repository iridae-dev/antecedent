//! The pooled multi-environment lagged frame of J-PCMCI+ against direct construction.
//!
//! `oracle_data/jpcmci_plus_pooled_frame_reference.py` (frozen in the JSON beside it) builds, from
//! three environments of different lengths, every column of the pooled frame by its definition:
//! variable-major lag slots read inside each environment (no window crosses an environment
//! boundary), environments stacked, the space dummy as one indicator column per environment but
//! the last, the time dummy as the raw time index or as indicators of every distinct raw time but
//! the last, each dummy repeated on every lag slot, and synthetic ids starting past the whole
//! schema. The product's `pool_multi_env_lagged_frame` must match every value and every id.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, KernelPolicy, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    DummyOptions, Float64Column, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage,
    SamplingRegularity, TimeDummyEncoding, TimeIndex, TimeSeriesData, ValidityBitmap,
    pool_multi_env_lagged_frame,
};
use serde_json::Value;

fn environment(values: &[Vec<f64>]) -> TimeSeriesData {
    let n = values[0].len();
    let mut schema = CausalSchemaBuilder::new();
    for i in 0..values.len() {
        schema
            .add_variable(
                format!("v{i}").as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let columns = values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(v.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(schema.build().unwrap(), columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

#[test]
fn the_pooled_frame_equals_its_definition_with_every_dummy_encoding() {
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/jpcmci_plus_pooled_frame_reference.json"))
            .unwrap();
    let envs: Vec<TimeSeriesData> = doc["envs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|env| {
            let values: Vec<Vec<f64>> = env
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
                .collect();
            environment(&values)
        })
        .collect();
    let multi = MultiEnvironmentData::try_new(Arc::from(envs)).unwrap();
    let variables: Vec<VariableId> = doc["variables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| VariableId::from_raw(v.as_u64().unwrap() as u32))
        .collect();
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for case in cases {
        let name = format!("{} depth {}", case["name"].as_str().unwrap(), case["depth"]);
        let encoding = match case["time"].as_str() {
            Some("one_hot") => TimeDummyEncoding::OneHot,
            _ => TimeDummyEncoding::IntegerIndex,
        };
        let options = DummyOptions {
            include_space_dummy: case["space"].as_bool().unwrap(),
            include_time_dummy: case["time"].is_string(),
            time_dummy_encoding: encoding,
            max_time_one_hot_levels: 512,
        };
        let pooled = pool_multi_env_lagged_frame(
            &multi,
            &variables,
            case["depth"].as_u64().unwrap() as u32,
            options,
            &KernelPolicy::default_policy(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let want_rows: Vec<usize> = case["env_rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            pooled.env_effective_rows.as_ref(),
            want_rows.as_slice(),
            "{name}: rows per environment"
        );
        let ids = |key: &str| -> Vec<u32> {
            case[key].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
        };
        let raw = |ids: &[VariableId]| ids.iter().map(|v| v.raw()).collect::<Vec<_>>();
        assert_eq!(raw(&pooled.space_dummy_variables), ids("space_ids"), "{name}: space ids");
        assert_eq!(raw(&pooled.time_dummy_variables), ids("time_ids"), "{name}: time ids");
        let columns = case["columns"].as_array().unwrap();
        assert_eq!(pooled.frame.ncols(), columns.len(), "{name}: columns");
        assert_eq!(pooled.frame.n_effective(), want_rows.iter().sum::<usize>(), "{name}");
        for (c, want) in columns.iter().enumerate() {
            let want: Vec<f64> =
                want.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let got = pooled.frame.column(c);
            assert_eq!(got.len(), want.len(), "{name}: column {c} length");
            for (row, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "{name}: column {c} row {row}: {g} vs {w}");
            }
        }
    }
}
