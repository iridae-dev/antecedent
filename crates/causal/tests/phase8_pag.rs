//! Phase 8 conformance: LPCMCI, latent projection, envelope mass, DAG-only reject.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::{
    GraphInput, reject_dag_only_on_pag, GeneralizedAdjustmentIdentifier, Lpcmci,
};
use causal_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use causal_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use causal_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};
use causal_graph::{CompletionSampler, Dag, DenseNodeId, Pag, latent_project, projection_preserves_msep_sample};
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/phase8")
        .join(name)
}

fn load_expected(name: &str) -> JsonValue {
    let raw = fs::read_to_string(fixture_dir(name).join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse")
}

fn tiny_series(n: usize) -> (TimeSeriesData, Vec<VariableId>) {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.5 * x[t - 1] + 0.1 * (t as f64).sin();
        y[t] = 0.7 * x[t] + 0.2 * y[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(x),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(y),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 1 },
            length: n,
        },
    )
    .unwrap();
    (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
}

#[test]
fn lpcmci_chain() {
    let expected = load_expected("lpcmci_chain");
    let (data, vars) = tiny_series(80);
    let alg = Lpcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(1),
            min_lag: Lag::CONTEMPORANEOUS,
        },
        alpha: 0.2,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
    assert_eq!(
        result.algorithm.id.as_ref(),
        expected["algorithm_id"].as_str().unwrap()
    );
    assert!(result.evidence.graph.node_count() >= expected["min_nodes"].as_u64().unwrap() as usize);
    assert!(
        result.evidence.links.len() >= expected["min_links_retained"].as_u64().unwrap() as usize
    );
    assert!(
        result.review.pending_circles.len()
            <= expected["max_pending_circles"].as_u64().unwrap() as usize
    );
    let rules = expected["orientation_rule_ids"].as_array().unwrap();
    assert_eq!(rules.len(), 5);
    assert!(rules.iter().any(|r| r.as_str() == Some("lpcmci.r2")));
    assert!(rules.iter().any(|r| r.as_str() == Some("lpcmci.r3")));
}

#[test]
fn latent_projection_msep() {
    let expected = load_expected("latent_projection_msep");
    let mut dag = Dag::with_variables(3);
    let l = DenseNodeId::from_raw(0);
    let x = DenseNodeId::from_raw(1);
    let y = DenseNodeId::from_raw(2);
    dag.insert_directed(l, x).unwrap();
    dag.insert_directed(l, y).unwrap();
    let _ = latent_project(&dag, &[x, y]).unwrap();
    assert!(expected["preserve_msep"].as_bool().unwrap());
    assert!(projection_preserves_msep_sample(&dag, &[x, y], &[(x, y, vec![])]).unwrap());
}

#[test]
fn envelope_unidentified_mass() {
    let expected = load_expected("envelope_unidentified_mass");
    let mut pag = Pag::with_variables(2);
    pag.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
        .unwrap();
    let id = GeneralizedAdjustmentIdentifier::new();
    let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let env = id.identify_pag_envelope(&pag, &q).unwrap();
    let total = env.identified_weight.0 + env.unidentified_weight.0;
    assert!(expected["require_mass_accounted"].as_bool().unwrap());
    assert!((total - env.cases.len() as f64).abs() < 1e-9);
}

#[test]
fn dag_only_pag_reject() {
    let expected = load_expected("dag_only_pag_reject");
    let pag = Pag::with_variables(2);
    let id = expected["identifier"].as_str().unwrap();
    let err = reject_dag_only_on_pag(&GraphInput::Pag(pag), id);
    assert!(expected["expect_compile_error"].as_bool().unwrap());
    assert!(err.is_err());
}

#[test]
fn completion_sampler_respects_bound() {
    let mut pag = Pag::with_variables(3);
    pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
        .unwrap();
    pag.insert_circle_arrow(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2))
        .unwrap();
    let max = 3usize;
    let n = CompletionSampler::new(pag, max).unwrap().count();
    assert!(n <= max);
}

