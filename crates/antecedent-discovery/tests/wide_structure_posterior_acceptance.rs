//! Ten- and twelve-variable structure posteriors on 2,000 rows, end to end.
//!
//! For each planted structure (the interacting ring at ten and twelve variables, independent
//! colliders at twelve), structure MCMC and order MCMC run under the published conditions
//! (at most three parents, four chains, 3 000 warmup and 12 000 retained draws thinned by two;
//! order MCMC at twice the warmup and draws). The checks:
//!
//! * the posterior is valid: weights sum to one, every atom is acyclic, and the edge marginals
//!   equal the weighted sum of the atoms' edges;
//! * the publication gate's diagnostics are published and clear their bars (R-hat below 1.2,
//!   bulk ESS above 10, every edge MCSE within 0.05) for every edge cell;
//! * no atom exceeds the parent cap;
//! * every planted edge is adjacent with posterior probability at least 0.95;
//! * every edge marginal is within `0.03 + 5 x` its Monte-Carlo standard error of the exact
//!   enumerating route, which also recovers the planted skeleton;
//! * a schedule far shorter than the published one is refused by the diagnostics gate.
//!
//! The generators are the planted-structure families used throughout the wide-posterior tests;
//! expected values are the generating structure and the exact route, never the samplers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "test scaffolding indexes small matrices by node"
)]

use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{mask_is_dag, parents_of};
use antecedent_discovery::{
    exact_dag_marginals, DiscoveryConstraints, DiscoveryWorkspace, GraphPosterior, GraphPrior,
    OrderMcmc, StructureMcmc,
};
use antecedent_state::GraphScoreFamily;

mod common;

const ROWS: usize = 2000;
const MAX_PARENTS: usize = 3;
const SKELETON_FLOOR: f64 = 0.95;

fn uniform(rng: &mut CausalRng) -> f64 {
    rng.next_f64() * 2.0 - 1.0
}

/// Roots `x_0..x_{k-1}` and children `y_j = coef (x_j + x_{j+1 mod k}) + noise`; returns the
/// columns (roots first) and the directed edges.
fn ring(k: usize, coef: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<(usize, usize)>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut cols = vec![vec![0.0; ROWS]; 2 * k];
    for r in 0..ROWS {
        for j in 0..k {
            cols[j][r] = uniform(&mut rng);
        }
        for j in 0..k {
            cols[k + j][r] = coef * (cols[j][r] + cols[(j + 1) % k][r]) + uniform(&mut rng);
        }
    }
    let edges = (0..k).flat_map(|j| [(j, k + j), ((j + 1) % k, k + j)]).collect();
    (cols, edges)
}

/// `blocks` independent colliders `a_j -> c_j <- b_j` with `c_j = 0.5 (a_j + b_j) + noise`;
/// columns are all `a`, then all `b`, then all `c`.
fn colliders(blocks: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<(usize, usize)>) {
    let mut rng = CausalRng::from_seed(seed);
    let mut cols = vec![vec![0.0; ROWS]; 3 * blocks];
    for r in 0..ROWS {
        for j in 0..blocks {
            cols[j][r] = uniform(&mut rng);
            cols[blocks + j][r] = uniform(&mut rng);
            cols[2 * blocks + j][r] = 0.5 * (cols[j][r] + cols[blocks + j][r]) + uniform(&mut rng);
        }
    }
    let edges =
        (0..blocks).flat_map(|j| [(j, 2 * blocks + j), (blocks + j, 2 * blocks + j)]).collect();
    (cols, edges)
}

fn prior() -> GraphPrior {
    GraphPrior::uniform().with_constraints(DiscoveryConstraints {
        max_parents: Some(MAX_PARENTS),
        ..Default::default()
    })
}

fn structure_engine() -> StructureMcmc {
    StructureMcmc::new().with_schedule(4, 3000, 12_000, 2)
}

fn order_engine() -> OrderMcmc {
    OrderMcmc::new().with_schedule(4, 6000, 24_000, 2)
}

/// Weights, atoms and marginals are consistent through a decoder that shares no code with the
/// sampler's accumulation.
fn assert_valid(label: &str, post: &GraphPosterior, n: usize) {
    assert_eq!(post.n_vars(), n, "{label}");
    assert_eq!(post.n_graphs(), post.atoms().len(), "{label}");
    let total: f64 = post.weights().iter().sum();
    assert!((total - 1.0).abs() <= 1e-9, "{label}: weights sum to {total}");
    let mut recovered = vec![0.0; n * n];
    for (mask, &weight) in post.atoms().iter().zip(post.weights().iter()) {
        assert!(mask_is_dag(mask, n), "{label}: a cyclic atom");
        for node in 0..n {
            for parent in parents_of(mask, n, node) {
                recovered[parent as usize * n + node] += weight;
            }
        }
    }
    for (cell, (&got, &want)) in recovered.iter().zip(post.edge_marginals().iter()).enumerate() {
        assert!((got - want).abs() <= 1e-9, "{label}: cell {cell} atoms {got} vs marginal {want}");
    }
}

/// The diagnostics the publication gate reads are present for every cell and clear its bars.
fn assert_gate_diagnostics(label: &str, post: &GraphPosterior, n: usize) {
    assert!(post.diagnostics().converged, "{label}: {:?}", post.diagnostics());
    let cells = n * n;
    for (name, field) in
        [("R-hat", post.edge_rhat()), ("ESS", post.edge_ess()), ("MCSE", post.edge_mcse())]
    {
        let field = field.unwrap_or_else(|| panic!("{label}: no per-edge {name}"));
        assert_eq!(field.len(), cells, "{label}: {name}");
        assert!(field.iter().all(|v| v.is_finite()), "{label}: non-finite {name}");
    }
    let max = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(max(post.edge_rhat().unwrap()) < 1.2, "{label}: max R-hat");
    assert!(min(post.edge_ess().unwrap()) > 10.0, "{label}: min ESS");
    assert!(max(post.edge_mcse().unwrap()) <= 0.05, "{label}: max MCSE");
}

fn assert_parent_cap(label: &str, post: &GraphPosterior, n: usize) {
    for mask in post.atoms().iter() {
        for node in 0..n {
            assert!(parents_of(mask, n, node).len() <= MAX_PARENTS, "{label}: node {node}");
        }
    }
}

/// Every planted edge is adjacent (either direction) with probability at least the floor.
fn assert_true_skeleton(label: &str, marginals: &[f64], n: usize, edges: &[(usize, usize)]) {
    for &(from, to) in edges {
        let adjacent = marginals[from * n + to] + marginals[to * n + from];
        assert!(adjacent >= SKELETON_FLOOR, "{label}: P({from} - {to}) = {adjacent}");
    }
}

fn assert_matches_exact(label: &str, post: &GraphPosterior, exact: &[f64]) {
    let mcse = post.edge_mcse().expect("a sampled posterior carries edge MCSE");
    for (cell, (&got, &want)) in post.edge_marginals().iter().zip(exact).enumerate() {
        let band = 0.03 + 5.0 * mcse[cell];
        assert!(
            (got - want).abs() <= band,
            "{label}: cell {cell} is {got}, exact {want}, band {band}"
        );
    }
}

fn run_case(n: usize, cols: Vec<Vec<f64>>, edges: &[(usize, usize)]) {
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let storage = common::continuous_storage(&names, cols);
    let data = TabularData::new(storage);
    let vars: Vec<VariableId> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
    let prior = prior();
    let ws = || DiscoveryWorkspace::default();

    let structure = structure_engine()
        .run(
            &data,
            &vars,
            &prior,
            GraphScoreFamily::GaussianBic,
            &mut ws(),
            &ExecutionContext::for_tests(5),
        )
        .expect("structure MCMC publishes under the published conditions");
    let order = order_engine()
        .run(
            &data,
            &vars,
            &prior,
            GraphScoreFamily::GaussianBic,
            &mut ws(),
            &ExecutionContext::for_tests(5),
        )
        .expect("order MCMC publishes under the published conditions");
    for (label, post) in [("structure", &structure), ("order", &order)] {
        let label = format!("n={n} {label}");
        assert_valid(&label, post, n);
        assert_gate_diagnostics(&label, post, n);
        assert_parent_cap(&label, post, n);
        assert_true_skeleton(&label, post.edge_marginals(), n, edges);
    }

    // The exact route is the reference the samplers match, and it recovers the skeleton too.
    // The cap of three parents does not bind: every planted node has at most two parents.
    let exact = exact_dag_marginals(
        &data,
        &vars,
        &prior,
        GraphScoreFamily::GaussianBic,
        &ExecutionContext::for_tests(1),
    )
    .expect("the exact route answers at this size");
    assert_eq!(exact.n_vars, n);
    assert!(exact.log_evidence.is_finite());
    assert!((0..n).all(|i| exact.edge_marginals[i * n + i] == 0.0));
    assert!(exact.edge_marginals.iter().all(|p| (0.0..=1.0).contains(p)));
    assert_true_skeleton(&format!("n={n} exact"), &exact.edge_marginals, n, edges);
    assert_matches_exact(&format!("n={n} structure"), &structure, &exact.edge_marginals);
    assert_matches_exact(&format!("n={n} order"), &order, &exact.edge_marginals);

    // A schedule far shorter than the published one is refused rather than published.
    let short_structure = StructureMcmc::new().with_schedule(2, 20, 40, 1).run(
        &data,
        &vars,
        &prior,
        GraphScoreFamily::GaussianBic,
        &mut ws(),
        &ExecutionContext::for_tests(3),
    );
    let short_order = OrderMcmc::new().with_schedule(2, 20, 40, 1).run(
        &data,
        &vars,
        &prior,
        GraphScoreFamily::GaussianBic,
        &mut ws(),
        &ExecutionContext::for_tests(3),
    );
    for (label, result) in [("structure", short_structure), ("order", short_order)] {
        let error = result.expect_err(label);
        assert!(error.to_string().contains("diagnostics gate"), "n={n} {label}: {error}");
    }
}

#[test]
fn ten_variable_ring_publishes_a_posterior_with_diagnostics() {
    let (cols, edges) = ring(5, 0.5, 4_867_969);
    run_case(10, cols, &edges);
}

#[test]
fn twelve_variable_ring_publishes_a_posterior_with_diagnostics() {
    let (cols, edges) = ring(6, 1.0, 5_398_576);
    run_case(12, cols, &edges);
}

#[test]
fn twelve_variable_colliders_publish_a_posterior_with_diagnostics() {
    let (cols, edges) = colliders(4, 7_573_334);
    run_case(12, cols, &edges);
}
