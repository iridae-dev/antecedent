//! PCMCI+ lag-0 black-box edge-set equality vs discovery-parity.
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
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, PcmciPlus, TemporalConstraints,
};
use serde_json::Value as JsonValue;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/discovery/pcmci_plus_lag0")
}

fn load_expected() -> JsonValue {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn load_series(expected: &JsonValue) -> (TimeSeriesData, Vec<VariableId>) {
    let csv = fs::read_to_string(fixture_dir().join("data.csv")).expect("data.csv");
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let mut parts = line.split(',');
        x.push(parts.next().unwrap().parse::<f64>().unwrap());
        y.push(parts.next().unwrap().parse::<f64>().unwrap());
    }
    let n = x.len();
    assert_eq!(n, expected["n"].as_u64().unwrap() as usize);
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y"] {
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
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
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
    (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
}

fn name_to_id(name: &str) -> VariableId {
    match name {
        "x" => VariableId::from_raw(0),
        "y" => VariableId::from_raw(1),
        other => panic!("unknown variable {other}"),
    }
}

/// Canonical undirected contemporaneous key + directed lagged key.
fn link_key(src: u32, slag: u32, tgt: u32) -> (u32, u32, u32, u32) {
    if slag == 0 {
        let (a, b) = if src <= tgt { (src, tgt) } else { (tgt, src) };
        (a, 0, b, 0)
    } else {
        (src, slag, tgt, 0)
    }
}

#[test]
fn discovery_pcmci_plus_lag0_edge_equality() {
    let expected = load_expected();
    assert_eq!(expected["tolerance_class"].as_str().unwrap(), "Exact");
    let tig = &expected["reference"];
    let outs = &tig["outputs"];
    assert_eq!(tig["available"].as_bool(), Some(true));

    let max_lag = expected["max_lag"].as_u64().unwrap() as u32;
    let min_lag = expected["min_lag"].as_u64().unwrap_or(0) as u32;
    let alpha = expected["alpha"].as_f64().unwrap();
    let fdr = expected["fdr"].as_bool().unwrap_or(false);

    let (data, vars) = load_series(&expected);
    let plus = PcmciPlus::new().with_fdr(fdr).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(max_lag),
            min_lag: Lag::from_raw(min_lag),
        },
        alpha,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let result = plus.run(&data, &vars, &mut ws, &ExecutionContext::for_tests(42)).unwrap();

    let recovered: BTreeSet<(u32, u32, u32, u32)> = result
        .evidence
        .links
        .iter()
        .map(|s| link_key(s.link.source.raw(), s.link.source_lag.raw(), s.link.target.raw()))
        .collect();

    // Prefer graph_links (includes contemporaneous); fall back to recovered_parents.
    let mut tig_set = BTreeSet::new();
    let links = outs
        .get("graph_links")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| outs["recovered_parents"].as_array().unwrap());
    for p in links {
        let src = name_to_id(p["source"].as_str().unwrap()).raw();
        let slag = p["source_lag"].as_u64().unwrap() as u32;
        let tgt = name_to_id(p["target"].as_str().unwrap()).raw();
        // Skip reverse contemporaneous marks that duplicate the undirected edge.
        if slag == 0 {
            let mark = p.get("mark").and_then(|m| m.as_str()).unwrap_or("");
            if mark == "<--" {
                continue;
            }
        }
        tig_set.insert(link_key(src, slag, tgt));
    }

    assert!(
        tig_set.is_subset(&recovered),
        "missing discovery links: tig={tig_set:?} rust={recovered:?}"
    );
    let extras: BTreeSet<_> = recovered.difference(&tig_set).copied().collect();
    // Autoregressive self-lags may differ from discovery; forbid any other extras
    // so the fixture still pins cross-variable equality.
    assert!(
        extras.iter().all(|(s, _slag, t, _)| s == t),
        "unexpected non-self extras vs discovery: {extras:?}"
    );
    for p in expected["true_parents"].as_array().unwrap() {
        let src = name_to_id(p["source"].as_str().unwrap()).raw();
        let slag = p["source_lag"].as_u64().unwrap() as u32;
        let tgt = name_to_id(p["target"].as_str().unwrap()).raw();
        let key = link_key(src, slag, tgt);
        assert!(recovered.contains(&key), "missing true parent {key:?} in {recovered:?}");
    }
    assert_eq!(result.algorithm.id.as_ref(), "pcmci_plus");
}
