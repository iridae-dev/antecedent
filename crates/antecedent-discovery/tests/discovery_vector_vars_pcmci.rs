//! Vector-variable PCMCI conformance (`Exact` logical edge set).
//!
//! The recovered logical parents must equal both the fixture's true parents and
//! the frozen upstream reference run's (the pinned baseline) `reference.outputs.recovered_parents` (scalar
//! logical parents at the same alpha, max_lag and BH-FDR). The recorded
//! `vector_vars_probe` dump is informational and not compared.
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
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap, VectorVariableGroups,
};
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Pcmci, TemporalConstraints};
use serde_json::Value as JsonValue;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/discovery/vector_vars_pcmci")
}

fn load_expected() -> JsonValue {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn load_series(expected: &JsonValue) -> (TimeSeriesData, Vec<VariableId>, VectorVariableGroups) {
    let csv = fs::read_to_string(fixture_dir().join("data.csv")).expect("data.csv");
    let mut x0 = Vec::new();
    let mut x1 = Vec::new();
    let mut y = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let mut parts = line.split(',');
        x0.push(parts.next().unwrap().parse::<f64>().unwrap());
        x1.push(parts.next().unwrap().parse::<f64>().unwrap());
        y.push(parts.next().unwrap().parse::<f64>().unwrap());
    }
    let n = x0.len();
    let expected_n = expected["n"].as_u64().expect("n") as usize;
    assert_eq!(n, expected_n, "csv row count vs expected.n");

    let mut b = CausalSchemaBuilder::new();
    for name in ["x0", "x1", "y"] {
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
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(x0),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(x1),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let groups = VectorVariableGroups::try_new(Arc::from([Arc::from([
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    ])]))
    .unwrap();
    (data, vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)], groups)
}

fn name_to_id(name: &str) -> VariableId {
    match name {
        "x0" => VariableId::from_raw(0),
        "x1" => VariableId::from_raw(1),
        "y" => VariableId::from_raw(2),
        other => panic!("unknown variable {other}"),
    }
}

#[test]
fn vector_vars_pcmci_exact_logical_parents() {
    let expected = load_expected();
    assert_eq!(expected["tolerance_class"].as_str().unwrap(), "Exact");
    let max_lag = expected["max_lag"].as_u64().unwrap() as u32;
    let alpha = expected["alpha"].as_f64().unwrap();
    let fdr = expected["fdr"].as_bool().unwrap_or(false);

    let (data, vars, groups) = load_series(&expected);
    let pcmci = Pcmci::new().with_fdr(fdr).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(max_lag),
            min_lag: Lag::from_raw(1),
        },
        alpha,
        max_cond_size: 2,
        vector_groups: groups,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(42);
    let result = pcmci.run(&data, &vars, &mut ws, &ctx).unwrap();

    let recovered: BTreeSet<(u32, u32, u32, u32)> = result
        .evidence
        .links
        .iter()
        .map(|s| {
            (
                s.link.source.raw(),
                s.link.source_lag.raw(),
                s.link.target.raw(),
                s.link.target_lag.raw(),
            )
        })
        .collect();

    // Secondary component x1 must not appear as a search-node endpoint.
    assert!(
        recovered.iter().all(|(s, _, t, _)| *s != 1 && *t != 1),
        "secondary component appeared in links: {recovered:?}"
    );

    let true_set = parent_set(&expected["true_parents"]);
    assert_eq!(
        recovered, true_set,
        "Exact edge-set mismatch: recovered={recovered:?} true={true_set:?}"
    );

    // Frozen external oracle: the recorded upstream reference run on the same data.csv, at the
    // same alpha and max_lag, must recover exactly the parents this crate recovers.
    let reference = &expected["reference"];
    assert_eq!(
        Some(reference["project"].as_str().expect("reference.project")),
        std::path::Path::new(expected["generation"]["baseline_pin"].as_str().unwrap())
            .file_stem()
            .and_then(|s| s.to_str()),
        "the reference is the pinned baseline's run"
    );
    assert_eq!(reference["available"].as_bool(), Some(true));
    let outputs = &reference["outputs"];
    assert_eq!(outputs["alpha"].as_f64(), Some(alpha), "reference alpha");
    assert_eq!(outputs["max_lag"].as_u64(), Some(u64::from(max_lag)), "reference max_lag");
    assert!(fdr && outputs["fdr_method"].as_str() == Some("fdr_bh"), "reference FDR control");
    let upstream = parent_set(&outputs["recovered_parents"]);
    assert_eq!(
        recovered, upstream,
        "upstream reference mismatch: recovered={recovered:?} reference={upstream:?}"
    );
}

fn parent_set(parents: &JsonValue) -> BTreeSet<(u32, u32, u32, u32)> {
    parents
        .as_array()
        .expect("parent list")
        .iter()
        .map(|p| {
            (
                name_to_id(p["source"].as_str().unwrap()).raw(),
                p["source_lag"].as_u64().unwrap() as u32,
                name_to_id(p["target"].as_str().unwrap()).raw(),
                p["target_lag"].as_u64().unwrap() as u32,
            )
        })
        .collect()
}
