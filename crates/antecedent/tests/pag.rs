//! conformance: LPCMCI, latent projection, envelope mass, DAG-only reject.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::AcceptedGraph;
use antecedent::discovery::Lpcmci;
use antecedent::identify::GeneralizedAdjustmentIdentifier;
use antecedent::planner::reject_dag_only_on_pag;
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};
use antecedent_graph::{
    CompletionSampler, Dag, DenseNodeId, Endpoint, NodeRef, Pag, latent_project,
    projection_preserves_msep_sample,
};
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/pag").join(name)
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

/// Shape checks only (algorithm id, retained graph, orientation-rule vocabulary) on an
/// in-test deterministic series; parity with the recorded upstream reference run on the
/// fixture's data is `lpcmci_chain_matches_upstream_reference_links_and_marks`.
#[test]
fn lpcmci_chain() {
    let expected = load_expected("lpcmci_chain");
    let (data, vars) = tiny_series(80);
    let alg = Lpcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::CONTEMPORANEOUS },
        alpha: 0.2,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
    assert_eq!(result.algorithm.id.as_ref(), expected["algorithm_id"].as_str().unwrap());
    assert!(result.evidence.graph.node_count() >= expected["min_nodes"].as_u64().unwrap() as usize);
    assert!(
        result.evidence.links.len() >= expected["min_links_retained"].as_u64().unwrap() as usize
    );
    assert!(
        result.review.pending_circles.len()
            <= expected["max_pending_circles"].as_u64().unwrap() as usize
    );
    let rules = expected["orientation_rule_ids"].as_array().unwrap();
    assert!(rules.len() >= 5);
    assert!(rules.iter().any(|r| r.as_str() == Some("lpcmci.r2")));
    assert!(rules.iter().any(|r| r.as_str() == Some("lpcmci.r3")));
    assert!(rules.iter().any(|r| r.as_str() == Some("lpcmci.r8")));
    assert!(rules.iter().any(|r| r.as_str() == Some("lpcmci.r10")));
    if expected["require_true_edge_subset"].as_bool() == Some(true) {
        let recovered: std::collections::BTreeSet<(u32, u32, u32)> = result
            .evidence
            .links
            .iter()
            .map(|s| (s.link.source.raw(), s.link.source_lag.raw(), s.link.target.raw()))
            .collect();
        for edge in expected["true_links"].as_array().unwrap() {
            let src = edge["source"].as_u64().unwrap() as u32;
            let slag = edge["source_lag"].as_u64().unwrap() as u32;
            let tgt = edge["target"].as_u64().unwrap() as u32;
            let forward = (src, slag, tgt);
            let reverse = (tgt, slag, src);
            let ok = recovered.contains(&forward) || (slag == 0 && recovered.contains(&reverse));
            assert!(ok, "missing true link {forward:?} in {recovered:?}");
        }
    }
}

/// `data.csv` of the `lpcmci_chain` fixture as a two-variable series (`x`, `y`).
fn chain_fixture_series(expected_rows: usize) -> (TimeSeriesData, Vec<VariableId>) {
    let csv = fs::read_to_string(fixture_dir("lpcmci_chain").join("data.csv")).expect("data.csv");
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some("x,y"), "lpcmci_chain data.csv header");
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for line in lines {
        let (a, b) = line.split_once(',').expect("two columns");
        x.push(a.parse::<f64>().unwrap());
        y.push(b.parse::<f64>().unwrap());
    }
    let n = x.len();
    assert_eq!(n, expected_rows, "data.csv rows vs expected n");
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
    let cols = [x, y]
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
}

fn endpoint_symbol(mark: Endpoint) -> char {
    match mark {
        Endpoint::Tail => '-',
        Endpoint::Arrow => '>',
        Endpoint::Circle => 'o',
        _ => 'x',
    }
}

/// Frozen external oracle: LPCMCI on the fixture's own `data.csv`, at the recorded
/// upstream-reference `alpha` and `max_lag`, must return the reference's links with its
/// endpoint marks. Links are keyed `(source, lag, target)` from the earlier node
/// (contemporaneous links once, the smaller variable name as source); the mark is
/// `<source end>-<target end>` (`o-o`, `o->`, `-->` ...).
#[test]
fn lpcmci_chain_matches_upstream_reference_links_and_marks() {
    let expected = load_expected("lpcmci_chain");
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
    let (data, vars) = chain_fixture_series(expected["n"].as_u64().unwrap() as usize);
    let alg = Lpcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(outputs["max_lag"].as_u64().unwrap() as u32),
            min_lag: Lag::CONTEMPORANEOUS,
        },
        alpha: outputs["alpha"].as_f64().unwrap(),
        max_cond_size: 3,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let result = alg.run(&data, &vars, &mut ws, &ExecutionContext::for_tests(3)).unwrap();
    let names = ["x", "y"];
    let graph = &result.evidence.graph;
    let mut native = BTreeMap::new();
    for edge in graph.edges() {
        let (NodeRef::Lagged { variable: av, lag: al }, NodeRef::Lagged { variable: bv, lag: bl }) =
            (graph.nodes()[edge.a.raw() as usize], graph.nodes()[edge.b.raw() as usize])
        else {
            unreachable!("temporal PAG nodes are lagged")
        };
        let a_first = al.raw() > bl.raw()
            || (al == bl && names[av.raw() as usize] <= names[bv.raw() as usize]);
        let (src, src_mark, lag, tgt, tgt_mark) = if a_first {
            (av, edge.at_a, al.raw() - bl.raw(), bv, edge.at_b)
        } else {
            (bv, edge.at_b, bl.raw() - al.raw(), av, edge.at_a)
        };
        native.insert(
            (names[src.raw() as usize].to_owned(), lag, names[tgt.raw() as usize].to_owned()),
            format!("{}-{}", endpoint_symbol(src_mark), endpoint_symbol(tgt_mark)),
        );
    }
    let mut upstream = BTreeMap::new();
    for link in outputs["links"].as_array().unwrap() {
        let (source, target) = (link["source"].as_str().unwrap(), link["target"].as_str().unwrap());
        let lag = link["lag"].as_u64().unwrap() as u32;
        // The reference lists a contemporaneous link from both ends; keep the canonical one.
        if lag == 0 && source > target {
            continue;
        }
        let mark = link["mark"].as_str().unwrap().to_owned();
        upstream.insert((source.to_owned(), lag, target.to_owned()), mark);
    }
    assert_eq!(native, upstream, "LPCMCI links and marks vs the pinned upstream reference");
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
    pag.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
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
    let err = reject_dag_only_on_pag(&AcceptedGraph::pag(pag), id.parse().unwrap());
    assert!(expected["expect_compile_error"].as_bool().unwrap());
    assert!(err.is_err());
}

#[test]
fn completion_sampler_respects_bound() {
    let mut pag = Pag::with_variables(3);
    pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    pag.insert_circle_arrow(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let max = 3usize;
    let n = CompletionSampler::new(pag, max).unwrap().count();
    assert!(n <= max);
}
