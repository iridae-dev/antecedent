//! Frozen Tigramite J-PCMCI+ multi-context motif conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage, SamplingRegularity,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, JpcmciPlus, MultiDatasetConstraints,
    TemporalConstraints,
};
use serde_json::Value as JsonValue;

type LinkKey = (String, u32, String);

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/jpcmci_plus_matrix/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("J-PCMCI+ fixture"))
        .expect("parse J-PCMCI+ fixture")
}

fn canonical(source: &str, lag: u32, target: &str) -> LinkKey {
    if lag == 0 && source > target {
        (target.to_owned(), 0, source.to_owned())
    } else {
        (source.to_owned(), lag, target.to_owned())
    }
}

fn reference_links(case: &JsonValue) -> BTreeSet<LinkKey> {
    case["reference"]["links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            canonical(
                link["source"].as_str().unwrap(),
                link["source_lag"].as_u64().unwrap() as u32,
                link["target"].as_str().unwrap(),
            )
        })
        .collect()
}

fn schema(names: &[String]) -> antecedent_core::CausalSchema {
    let mut builder = CausalSchemaBuilder::new();
    for name in names {
        builder
            .add_variable(
                name.as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    builder.build().unwrap()
}

fn multi_data(case: &JsonValue) -> (MultiEnvironmentData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = case["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let mut environments = Vec::new();
    for env in case["environments"].as_array().unwrap() {
        let rows = env.as_array().unwrap();
        let n = rows.len();
        let columns: Vec<OwnedColumn> = (0..names.len())
            .map(|column| {
                let values: Vec<f64> = rows
                    .iter()
                    .map(|row| row.as_array().unwrap()[column].as_f64().unwrap())
                    .collect();
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(column as u32),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema(&names), columns, None, None).unwrap();
        environments.push(
            TimeSeriesData::try_new(
                storage,
                TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
            )
            .unwrap(),
        );
    }
    let data = MultiEnvironmentData::try_new(Arc::from(environments)).unwrap();
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (data, variables, names)
}

#[test]
fn jpcmci_plus_tigramite_matrix_is_reproducible() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names) = multi_data(case);
        let params = &case["parameters"];
        let algorithm = JpcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                min_lag: Lag::from_raw(params["tau_min"].as_u64().unwrap() as u32),
                max_lag: Lag::from_raw(params["tau_max"].as_u64().unwrap() as u32),
            },
            alpha: params["pc_alpha"].as_f64().unwrap(),
            max_cond_size: 3,
            multi_dataset: MultiDatasetConstraints {
                include_space_dummy: false,
                include_time_dummy: false,
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        });
        let mut workspace = DiscoveryWorkspace::default();
        let result = algorithm
            .run(&data, &variables, &mut workspace, &ExecutionContext::for_tests(0x4A_C0))
            .unwrap();
        let native: BTreeSet<LinkKey> = result
            .evidence
            .links
            .iter()
            .filter(|link| {
                link.link.source.raw() < names.len() as u32
                    && link.link.target.raw() < names.len() as u32
            })
            .map(|link| {
                canonical(
                    &names[link.link.source.raw() as usize],
                    link.link.source_lag.raw(),
                    &names[link.link.target.raw() as usize],
                )
            })
            .collect();
        let reference = reference_links(case);
        assert_eq!(
            native,
            reference,
            "{} canonical temporal skeleton",
            case["name"].as_str().unwrap()
        );
        assert_eq!(result.algorithm.id.as_ref(), "jpcmci_plus");
    }
}

#[test]
fn jpcmci_plus_fixture_records_pins_and_no_generator() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(oracle["packages"]["tigramite"]["version"], "5.2.9.7");
    assert_eq!(oracle["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(oracle["generation_location"], "temporary external harness; not retained");
    for package in ["tigramite", "numpy", "scipy", "joblib"] {
        assert_eq!(oracle["packages"][package]["metadata_sha256"].as_str().unwrap().len(), 64);
    }
}
