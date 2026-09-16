//! Graph-posterior known-truth conformance (`conformance/bayesian/dag_posterior`).
//!
//! Every engine the fixture names runs on data drawn from a known SEM and must put
//! the fixture's minimum posterior mass on the true structure: the chain skeleton
//! (orientation is not identified), both collider arrows, and the lag-1 DBN edge.
//! The thresholds are the fixture's; the SEMs are defined here. There is no external
//! oracle: this is internal known-truth evidence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    CausalRng, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
    ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TabularData, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{
    CiScreenedPosterior, DbnPosterior, DiscoveryConstraints, DiscoveryWorkspace,
    EXACT_ENUM_MAX_NODES, ExactDagPosterior, GraphPosterior, GraphPrior, OrderMcmc, StructureMcmc,
    TemporalConstraints,
};
use antecedent_state::GraphScoreFamily;
use serde_json::Value as JsonValue;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/bayesian/dag_posterior/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("dag_posterior fixture"))
        .expect("parse dag_posterior fixture")
}

fn schema(names: &[&str]) -> antecedent_core::CausalSchema {
    let mut b = CausalSchemaBuilder::new();
    for name in names {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    b.build().unwrap()
}

fn storage(names: &[&str], columns: Vec<Vec<f64>>) -> (OwnedColumnarStorage, Vec<VariableId>) {
    let n = columns[0].len();
    let vars: Vec<_> = (0..names.len() as u32).map(VariableId::from_raw).collect();
    let cols = columns
        .into_iter()
        .zip(&vars)
        .map(|(values, &id)| {
            OwnedColumn::Float64(
                Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    (OwnedColumnarStorage::try_new(schema(names), cols, None, None).unwrap(), vars)
}

fn uniform(rng: &mut CausalRng) -> f64 {
    rng.next_f64() * 2.0 - 1.0
}

/// `a -> b -> c`.
fn chain(n: usize) -> (TabularData, Vec<VariableId>) {
    let mut rng = CausalRng::from_seed(7);
    let (mut a, mut b, mut c) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        a[i] = uniform(&mut rng);
        b[i] = 1.5 * a[i] + 0.15 * uniform(&mut rng);
        c[i] = 1.2 * b[i] + 0.15 * uniform(&mut rng);
    }
    let (s, vars) = storage(&["a", "b", "c"], vec![a, b, c]);
    (TabularData::new(s), vars)
}

/// `a -> c <- b`, `a` and `b` independent.
fn collider(n: usize) -> (TabularData, Vec<VariableId>) {
    let mut rng = CausalRng::from_seed(11);
    let (mut a, mut b, mut c) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        a[i] = uniform(&mut rng);
        b[i] = uniform(&mut rng);
        c[i] = a[i] + b[i] + 0.15 * uniform(&mut rng);
    }
    let (s, vars) = storage(&["a", "b", "c"], vec![a, b, c]);
    (TabularData::new(s), vars)
}

/// `x_{t-1} -> y_t`.
fn lag1(n: usize) -> (TimeSeriesData, Vec<VariableId>) {
    let mut rng = CausalRng::from_seed(99);
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
    x[0] = uniform(&mut rng);
    y[0] = uniform(&mut rng);
    for t in 1..n {
        x[t] = 0.2 * uniform(&mut rng);
        y[t] = 1.4 * x[t - 1] + 0.15 * uniform(&mut rng);
    }
    let (s, vars) = storage(&["x", "y"], vec![x, y]);
    let data = TimeSeriesData::try_new(
        s,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    (data, vars)
}

fn index(names: &[&str], name: &str) -> usize {
    names.iter().position(|n| *n == name).unwrap_or_else(|| panic!("unknown variable {name}"))
}

/// Posterior mass of the directed edge `from -> to` (`edge_marginals[from * n + to]`).
fn directed(post: &GraphPosterior, n: usize, from: usize, to: usize) -> f64 {
    post.edge_marginals[from * n + to]
}

/// Runs every static engine the fixture names with its default configuration.
///
/// `screened_gate = false` turns off the CI-screened engine's MCMC publication gate.
/// On the collider its screened space leaves one graph, every chain stays on it,
/// and the gate refuses a trace in which nothing varied ("stuck" and
/// "concentrated" look the same there); with the gate off the posterior itself is
/// checked.
fn static_engines(
    data: &TabularData,
    vars: &[VariableId],
    engines: &[&str],
    screened_gate: bool,
) -> Vec<(String, GraphPosterior)> {
    let prior = GraphPrior::uniform();
    let family = GraphScoreFamily::GaussianBic;
    let ctx = ExecutionContext::for_tests(1);
    let mut ws = DiscoveryWorkspace::default();
    engines
        .iter()
        .filter(|engine| **engine != "DbnPosterior")
        .map(|engine| {
            let post = match *engine {
                "ExactDagPosterior" => {
                    ExactDagPosterior::new().run(data, vars, &prior, family, &mut ws, &ctx)
                }
                "StructureMcmc" => {
                    StructureMcmc::new().run(data, vars, &prior, family, &mut ws, &ctx)
                }
                "OrderMcmc" => OrderMcmc::new().run(data, vars, &prior, family, &mut ws, &ctx),
                "CiScreenedPosterior" => CiScreenedPosterior::new()
                    .with_mcmc(StructureMcmc::new().with_diagnostics_gate(screened_gate))
                    .run(data, vars, &prior, family, &mut ws, &ctx),
                other => panic!("fixture names engine {other}, which this test does not run"),
            };
            ((*engine).to_owned(), post.unwrap_or_else(|e| panic!("{engine}: {e}")))
        })
        .collect()
}

#[test]
fn dag_posterior_engines_put_fixture_mass_on_the_true_structure() {
    let fx = fixture();
    assert_eq!(fx["score_family"].as_str(), Some("GaussianBic"));
    assert_eq!(fx["exact_max_nodes"].as_u64(), Some(EXACT_ENUM_MAX_NODES as u64));
    let engines: Vec<&str> =
        fx["engines"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect();
    assert!(engines.contains(&"DbnPosterior"), "fixture names the DBN engine");
    let names = ["a", "b", "c"];

    // Chain: skeleton mass on each true adjacency, for every static engine.
    let spec = &fx["chain_fixture"];
    assert_eq!(spec["n_vars"].as_u64(), Some(3));
    let min_mass = spec["min_undirected_mass"].as_f64().unwrap();
    let (data, vars) = chain(200);
    let posts = static_engines(&data, &vars, &engines, true);
    assert_eq!(posts.len(), engines.len() - 1, "every static engine ran");
    for (engine, post) in &posts {
        for pair in spec["expect_skeleton"].as_array().unwrap() {
            let (i, j) = (
                index(&names, pair[0].as_str().unwrap()),
                index(&names, pair[1].as_str().unwrap()),
            );
            let mass = directed(post, 3, i, j) + directed(post, 3, j, i);
            assert!(mass >= min_mass, "{engine}: P({pair}) = {mass} < {min_mass}");
        }
    }

    // Collider: both arrows into the collider are oriented, for every static engine.
    let spec = &fx["collider_fixture"];
    assert_eq!(spec["n_vars"].as_u64(), Some(3));
    let min_mass = spec["min_directed_mass"].as_f64().unwrap();
    let (data, vars) = collider(400);
    for (engine, post) in &static_engines(&data, &vars, &engines, false) {
        for arrow in spec["expect_oriented"].as_array().unwrap() {
            let (from, to) = (
                index(&names, arrow[0].as_str().unwrap()),
                index(&names, arrow[1].as_str().unwrap()),
            );
            let mass = directed(post, 3, from, to);
            assert!(mass >= min_mass, "{engine}: P({arrow}) = {mass} < {min_mass}");
        }
    }

    // DBN: the lag-1 edge.
    let spec = &fx["dbn_lag1_fixture"];
    assert_eq!(spec["n_vars"].as_u64(), Some(2));
    let max_lag = spec["max_lag"].as_u64().unwrap() as u32;
    let min_mass = spec["min_lag_mass"].as_f64().unwrap();
    let (data, vars) = lag1(200);
    let prior = GraphPrior::uniform().with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::from_raw(max_lag),
            min_lag: Lag::from_raw(1),
        },
        ..Default::default()
    });
    let post = DbnPosterior::new(max_lag)
        .run(&data, &vars, &prior, GraphScoreFamily::GaussianBic, &ExecutionContext::for_tests(1))
        .unwrap();
    let edge = spec["expect_lag_edge"].as_array().unwrap();
    let series = ["x", "y"];
    let (from, to) =
        (index(&series, edge[0].as_str().unwrap()), index(&series, edge[1].as_str().unwrap()));
    let lagged = post.lagged_edge_marginals.as_ref().expect("lagged marginals");
    let mass = lagged[from * 2 + to];
    assert!(mass >= min_mass, "DbnPosterior: P(x_(t-1) -> y_t) = {mass} < {min_mass}");
}
