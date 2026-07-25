//! Frozen per-regime Tigramite PCMCI+ parity for fixed-label RPCMCI.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RegimeId, RoleHint, SmallRoleSet,
    ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, PcmciPlus, RegimeAssignment, Rpcmci,
    TemporalConstraints,
};
use serde_json::Value as JsonValue;

type LinkKey = (String, u32, String);

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/rpcmci_fixed_regime_matrix/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("fixed-regime fixture"))
        .expect("parse fixed-regime fixture")
}

fn canonical(source: &str, lag: u32, target: &str) -> LinkKey {
    if lag == 0 && source > target {
        (target.to_owned(), 0, source.to_owned())
    } else {
        (source.to_owned(), lag, target.to_owned())
    }
}

fn reference_links(reference: &JsonValue) -> BTreeSet<LinkKey> {
    reference["links"]
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

fn case_data(case: &JsonValue) -> (TimeSeriesData, Vec<VariableId>, Vec<String>, RegimeAssignment) {
    let names: Vec<String> = case["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let segments = case["segments"].as_array().unwrap();
    let n: usize = segments.iter().map(|segment| segment.as_array().unwrap().len()).sum();
    let mut builder = CausalSchemaBuilder::new();
    for name in &names {
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
    let schema = builder.build().unwrap();
    let columns: Vec<OwnedColumn> = (0..names.len())
        .map(|column| {
            let values: Vec<f64> = segments
                .iter()
                .flat_map(|segment| segment.as_array().unwrap().iter())
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
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let labels: Vec<RegimeId> = segments
        .iter()
        .enumerate()
        .flat_map(|(regime, segment)| {
            std::iter::repeat_n(
                RegimeId::from_raw(regime as u32),
                segment.as_array().unwrap().len(),
            )
        })
        .collect();
    let assignment = RegimeAssignment::try_new(Arc::from(labels)).unwrap();
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (data, variables, names, assignment)
}

#[test]
fn fixed_label_rpcmci_matches_independent_per_regime_tigramite() {
    let fixture = fixture();
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names, assignment) = case_data(case);
        let params = &case["parameters"];
        let nested = PcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                min_lag: Lag::from_raw(params["tau_min"].as_u64().unwrap() as u32),
                max_lag: Lag::from_raw(params["tau_max"].as_u64().unwrap() as u32),
            },
            alpha: params["pc_alpha"].as_f64().unwrap(),
            max_cond_size: 3,
            ..DiscoveryConstraints::default()
        });
        let algorithm =
            Rpcmci::new().with_pcmci_plus(nested).with_min_regime_len(40).with_alternating_iters(0);
        let mut workspace = DiscoveryWorkspace::default();
        let result = algorithm
            .run(
                &data,
                &variables,
                &assignment,
                &mut workspace,
                &ExecutionContext::for_tests(0x52_C0),
            )
            .unwrap();
        let references = case["reference_by_regime"].as_array().unwrap();
        assert_eq!(result.per_regime.len(), references.len());
        for (regime, (native, reference)) in result.per_regime.iter().zip(references).enumerate() {
            let native_links: BTreeSet<LinkKey> = native
                .evidence
                .links
                .iter()
                .map(|link| {
                    canonical(
                        &names[link.link.source.raw() as usize],
                        link.link.source_lag.raw(),
                        &names[link.link.target.raw() as usize],
                    )
                })
                .collect();
            let expected = reference_links(reference);
            assert_eq!(
                native_links,
                expected,
                "{} regime {regime} canonical skeleton",
                case["name"].as_str().unwrap()
            );
        }
        assert_eq!(result.assignments, assignment, "fixed-label mode must preserve assignments");
    }
}

#[test]
fn fixed_regime_fixture_records_scope_pins_and_no_generator() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(oracle["packages"]["tigramite"]["version"], "5.2.9.7");
    assert_eq!(oracle["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(oracle["generation_location"], "temporary external harness; not retained");
    assert!(oracle["scope"].as_str().unwrap().contains("caller-supplied"));
    for package in ["tigramite", "numpy", "scipy"] {
        assert_eq!(oracle["packages"][package]["metadata_sha256"].as_str().unwrap().len(), 64);
    }
}
