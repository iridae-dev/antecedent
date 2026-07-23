//! discovery-parity PCMCI multi-var conformance (edge set + val/p matrix entries).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::{BTreeMap, BTreeSet};
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
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Pcmci, TemporalConstraints};
use serde_json::Value as JsonValue;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/discovery/pcmci_multivar")
}

fn load_expected() -> JsonValue {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn close(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn load_series(expected: &JsonValue) -> (TimeSeriesData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = expected["variables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let csv = fs::read_to_string(fixture_dir().join("data.csv")).expect("data.csv");
    let mut cols: Vec<Vec<f64>> = names.iter().map(|_| Vec::new()).collect();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        for (j, part) in line.split(',').enumerate() {
            cols[j].push(part.parse().unwrap());
        }
    }
    let n = cols[0].len();
    assert_eq!(n, expected["n"].as_u64().unwrap() as usize);

    let mut b = CausalSchemaBuilder::new();
    for name in &names {
        b.add_variable(
            name.as_str(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let owned: Vec<OwnedColumn> = cols
        .into_iter()
        .enumerate()
        .map(|(i, data)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(data),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let vars: Vec<VariableId> = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (data, vars, names)
}

fn name_idx(names: &[String], name: &str) -> usize {
    names.iter().position(|n| n == name).unwrap_or_else(|| panic!("unknown var {name}"))
}

#[test]
fn discovery_pcmci_multivar_edges_and_matrices() {
    let expected = load_expected();
    let tig = &expected["reference"];
    let outs = &tig["outputs"];
    assert_eq!(tig["available"].as_bool(), Some(true));

    let max_lag = expected["max_lag"].as_u64().unwrap() as u32;
    let alpha = expected["alpha"].as_f64().unwrap();
    let fdr = expected["fdr"].as_bool().unwrap_or(false);
    let (data, vars, names) = load_series(&expected);

    let pcmci = Pcmci::new().with_fdr(fdr).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(max_lag),
            min_lag: Lag::from_raw(1),
        },
        alpha,
        max_cond_size: 3,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let result = pcmci.run(&data, &vars, &mut ws, &ExecutionContext::for_tests(42)).unwrap();

    let recovered: BTreeSet<(u32, u32, u32)> = result
        .evidence
        .links
        .iter()
        .map(|s| (s.link.source.raw(), s.link.source_lag.raw(), s.link.target.raw()))
        .collect();

    let mut tig_set = BTreeSet::new();
    for p in outs["recovered_parents"].as_array().unwrap() {
        tig_set.insert((
            name_idx(&names, p["source"].as_str().unwrap()) as u32,
            p["source_lag"].as_u64().unwrap() as u32,
            name_idx(&names, p["target"].as_str().unwrap()) as u32,
        ));
    }
    assert_eq!(recovered, tig_set, "edge-set mismatch rust={recovered:?} discovery={tig_set:?}");

    // True structural parents must be recovered.
    for p in expected["true_parents"].as_array().unwrap() {
        let key = (
            name_idx(&names, p["source"].as_str().unwrap()) as u32,
            p["source_lag"].as_u64().unwrap() as u32,
            name_idx(&names, p["target"].as_str().unwrap()) as u32,
        );
        assert!(recovered.contains(&key), "missing true parent {key:?}");
    }

    let val_matrix = outs["val_matrix"].as_array().unwrap();
    let p_matrix = outs["p_matrix"].as_array().unwrap();
    let q_matrix = outs.get("q_matrix").and_then(|v| v.as_array());
    let atol_s = expected["atol_stat"].as_f64().unwrap();
    let rtol_s = expected["rtol_stat"].as_f64().unwrap();
    let atol_p = expected["atol_p"].as_f64().unwrap();
    let rtol_p = expected["rtol_p"].as_f64().unwrap();

    let by_key: BTreeMap<(u32, u32, u32), _> = result
        .evidence
        .links
        .iter()
        .map(|s| ((s.link.source.raw(), s.link.source_lag.raw(), s.link.target.raw()), s))
        .collect();

    for ((src, slag, tgt), scored) in &by_key {
        let i = *src as usize;
        let j = *tgt as usize;
        let tau = *slag as usize;
        let ref_val = val_matrix[i][j][tau].as_f64().unwrap();
        let ref_p = p_matrix[i][j][tau].as_f64().unwrap();
        assert!(
            close(scored.statistic, ref_val, atol_s, rtol_s),
            "stat mismatch link=({src},{slag}->{tgt}): rust={} tig={ref_val}",
            scored.statistic
        );
        assert!(
            close(scored.p_value, ref_p, atol_p, rtol_p),
            "p mismatch link=({src},{slag}->{tgt}): rust={} tig={ref_p}",
            scored.p_value
        );
        if fdr {
            if let (Some(qmat), Some(adj)) = (q_matrix, scored.adjusted_p_value) {
                let ref_q = qmat[i][j][tau].as_f64().unwrap();
                assert!(
                    close(adj, ref_q, atol_p, rtol_p),
                    "q mismatch link=({src},{slag}->{tgt}): rust={adj} tig={ref_q}"
                );
            }
        }
    }
}
