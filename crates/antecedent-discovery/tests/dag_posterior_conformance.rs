//! Graph-posterior known-truth conformance (`conformance/bayesian/dag_posterior`).
//!
//! Every engine the fixture names runs on data drawn from a known SEM and must put
//! the fixture's minimum posterior mass on the true structure: the chain skeleton
//! (orientation is not identified), both collider arrows, and the lag-1 DBN edge.
//! The thresholds are the fixture's; the SEMs are defined here. There is no external
//! oracle: this is internal known-truth evidence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::fs;
use std::path::PathBuf;

use antecedent_core::{CausalRng, ExecutionContext, Lag, VariableId};
use antecedent_data::{
    OwnedColumnarStorage, SamplingRegularity, TabularData, TimeIndex, TimeSeriesData,
};

mod common;
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

fn storage(names: &[&str], columns: Vec<Vec<f64>>) -> (OwnedColumnarStorage, Vec<VariableId>) {
    let vars: Vec<_> = (0..names.len() as u32).map(VariableId::from_raw).collect();
    (common::continuous_storage(names, columns), vars)
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
    post.edge_marginals()[from * n + to]
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
    let lagged = post.lagged_edge_marginals().expect("lagged marginals");
    let mass = lagged[from * 2 + to];
    assert!(mass >= min_mass, "DbnPosterior: P(x_(t-1) -> y_t) = {mass} < {min_mass}");

    // Exact reference: the configured min_lag=1 excludes contemporaneous edges. Enumerate
    // the empty contemporaneous DAG times the 2^4 lag-1 edge sets under the uniform prior,
    // and weight each by the Gaussian BIC of the aligned design, exactly as for the
    // static enumeration. The design rows are t = 1..n-1 with columns (x_t, y_t, x_(t-1),
    // y_(t-1)); a node's parents are its contemporaneous parents and its lagged sources.
    use antecedent_data::TableView;
    let series_x = data.float64_values(vars[0]).unwrap().to_vec();
    let series_y = data.float64_values(vars[1]).unwrap().to_vec();
    let rows = series_x.len() - 1;
    let design: Vec<Vec<f64>> = vec![
        series_x[1..].to_vec(),
        series_y[1..].to_vec(),
        series_x[..rows].to_vec(),
        series_y[..rows].to_vec(),
    ];
    let n = rows as f64;
    let contemporaneous: [&[(usize, usize)]; 1] = [&[]];
    let mut logs = Vec::new();
    let mut lag_sets = Vec::new();
    let mut cmasks = Vec::new();
    for (ci, edges) in contemporaneous.iter().enumerate() {
        for lword in 0_u32..16 {
            // lag bit = from * 2 + to
            let lag_edge = |from: usize, to: usize| lword >> (from * 2 + to) & 1 == 1;
            let log_weight: f64 = (0..2)
                .map(|node| {
                    let mut parents: Vec<usize> =
                        edges.iter().filter(|e| e.1 == node).map(|e| e.0).collect();
                    parents
                        .extend((0..2).filter(|&from| lag_edge(from, node)).map(|from| 2 + from));
                    -(n / 2.0) * (ols_rss(&design, node, &parents) / n).ln()
                        - (parents.len() as f64 / 2.0) * n.ln()
                })
                .sum();
            logs.push(log_weight);
            lag_sets.push(lword);
            cmasks.push(ci);
        }
    }
    let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = logs.iter().map(|w| (w - top).exp()).sum();
    let mut lag_marginals = [0.0_f64; 4];
    for ((w, lword), _) in logs.iter().zip(&lag_sets).zip(&cmasks) {
        for (bit, slot) in lag_marginals.iter_mut().enumerate() {
            if lword >> bit & 1 == 1 {
                *slot += (w - top).exp() / total;
            }
        }
    }
    for (bit, want) in lag_marginals.iter().enumerate() {
        assert!(
            (lagged[bit] - want).abs() < 1e-6,
            "DbnPosterior lag-1 marginal of edge {} -> {}: {} vs enumeration {want}",
            bit / 2,
            bit % 2,
            lagged[bit]
        );
    }
    // The true edge x_(t-1) -> y_t carries essentially all the mass, and the enumeration says
    // exactly how much.
    assert!((mass - lag_marginals[1]).abs() < 1e-6);
    assert!(lag_marginals[1] > 0.999, "enumerated mass {}", lag_marginals[1]);
}

/// Residual sum of squares of `target` regressed on an intercept and `parents` (normal
/// equations by Gaussian elimination with partial pivoting; the designs here are 1-3 columns of
/// well-conditioned data). Independent of the crate's score code.
fn ols_rss(columns: &[Vec<f64>], target: usize, parents: &[usize]) -> f64 {
    let n = columns[0].len();
    let k = parents.len() + 1;
    let design = |row: usize, j: usize| if j == 0 { 1.0 } else { columns[parents[j - 1]][row] };
    let mut a = vec![vec![0.0; k + 1]; k];
    for row in 0..n {
        for i in 0..k {
            for j in 0..k {
                a[i][j] += design(row, i) * design(row, j);
            }
            a[i][k] += design(row, i) * columns[target][row];
        }
    }
    for col in 0..k {
        let pivot = (col..k).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs())).unwrap();
        a.swap(col, pivot);
        for r in 0..k {
            if r != col {
                let f = a[r][col] / a[col][col];
                for c in col..=k {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
    }
    let beta: Vec<f64> = (0..k).map(|i| a[i][k] / a[i][i]).collect();
    (0..n)
        .map(|row| {
            let fit: f64 = (0..k).map(|j| beta[j] * design(row, j)).sum();
            (columns[target][row] - fit).powi(2)
        })
        .sum()
}

/// Exact BIC posterior edge marginals over all DAGs on `columns.len() <= 4` nodes with a uniform
/// prior over DAGs: weight `exp(sum_v [-(n/2) ln(RSS_v / n) - (|pa(v)| / 2) ln n])`. Terms that
/// do not depend on the graph (2-pi, the intercept and variance parameters) cancel.
fn reference_edge_marginals(columns: &[Vec<f64>]) -> Vec<f64> {
    let (v, n) = (columns.len(), columns[0].len() as f64);
    let pairs: Vec<(usize, usize)> =
        (0..v).flat_map(|i| (0..v).filter(move |&j| j != i).map(move |j| (i, j))).collect();
    let mut log_weights = Vec::new();
    let mut graphs = Vec::new();
    for bits in 0..(1_u32 << pairs.len()) {
        let edges: Vec<(usize, usize)> =
            pairs.iter().enumerate().filter(|(b, _)| bits >> b & 1 == 1).map(|(_, e)| *e).collect();
        // Acyclic: repeatedly remove a node with no remaining parents.
        let mut remaining: Vec<usize> = (0..v).collect();
        while let Some(pos) = remaining
            .iter()
            .position(|&x| !edges.iter().any(|&(f, t)| t == x && remaining.contains(&f)))
        {
            remaining.remove(pos);
        }
        if !remaining.is_empty() {
            continue;
        }
        let log_weight: f64 = (0..v)
            .map(|node| {
                let parents: Vec<usize> =
                    edges.iter().filter(|e| e.1 == node).map(|e| e.0).collect();
                -(n / 2.0) * (ols_rss(columns, node, &parents) / n).ln()
                    - (parents.len() as f64 / 2.0) * n.ln()
            })
            .sum();
        log_weights.push(log_weight);
        graphs.push(edges);
    }
    let max = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = log_weights.iter().map(|w| (w - max).exp()).sum();
    let mut marginals = vec![0.0; v * v];
    for (edges, w) in graphs.iter().zip(&log_weights) {
        for &(f, t) in edges {
            marginals[f * v + t] += (w - max).exp() / total;
        }
    }
    marginals
}

fn columns_of(data: &TabularData, names: &[&str]) -> Vec<Vec<f64>> {
    use antecedent_data::TableView;
    (0..names.len() as u32)
        .map(|i| data.float64_values(VariableId::from_raw(i)).unwrap().to_vec())
        .collect()
}

#[test]
fn exact_engine_reproduces_an_independent_bic_enumeration_and_samplers_agree_with_it() {
    let fx = fixture();
    let engines: Vec<&str> =
        fx["engines"].as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect();
    let names = ["a", "b", "c"];
    for (label, (data, vars), gate) in
        [("chain", chain(200), true), ("collider", collider(400), false)]
    {
        let reference = reference_edge_marginals(&columns_of(&data, &names));
        // The in-test enumeration must itself agree with the fixture's marginals, which the
        // retained exact-rational generator (reference.py) writes from the SplitMix64 stream.
        let frozen: Vec<f64> = fx["exact_marginals"][label]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(frozen.len(), reference.len(), "{label}");
        for (cell, (a, b)) in reference.iter().zip(&frozen).enumerate() {
            assert!((a - b).abs() < 1e-9, "{label}: in-test enumeration cell {cell}: {a} vs {b}");
        }
        for (engine, post) in static_engines(&data, &vars, &engines, gate) {
            // The exact engine is the enumeration itself; the samplers estimate it from
            // finite chains, so they get a Monte Carlo budget.
            // Observed seeded deviations are at most 0.04 (finite chains, a few thousand draws).
            let budget = if engine == "ExactDagPosterior" { 1e-6 } else { 0.06 };
            if engine == "ExactDagPosterior" {
                for (cell, want) in frozen.iter().enumerate() {
                    assert!(
                        (post.edge_marginals()[cell] - want).abs() < 1e-6,
                        "{label}: exact engine cell {cell} vs frozen enumeration {want}"
                    );
                }
            }
            for cell in 0..9 {
                let got = post.edge_marginals()[cell];
                if engine == "CiScreenedPosterior" {
                    // Screening removes graphs, so only the edges the full posterior keeps
                    // must survive; edges it rules out may be zeroed or sharpened.
                    if reference[cell] >= 0.5 {
                        assert!(got >= reference[cell] - budget, "{label} {engine}: edge {cell}");
                    }
                    continue;
                }
                assert!(
                    (got - reference[cell]).abs() < budget,
                    "{label} {engine}: P(edge {cell}) = {got}, enumeration {}",
                    reference[cell]
                );
            }
        }
    }
}
