//! Graph posteriors above eight variables.
//!
//! Adjacency is a multi-word [`AdjacencyMask`] (up to 16 variables). These tests check the
//! samplers against references that do not share their code:
//!
//! * five variables: order and structure MCMC against exhaustive enumeration of all 29 281 DAGs;
//! * twelve variables: both samplers against an exhaustive enumeration of the 3^12 states of a
//!   twelve-pair candidate skeleton, scored by a separate Gaussian-BIC implementation;
//! * the published ten-variable workflow conditions against the publication gate;
//! * sixteen variables, the full packing.
//!
//! Closed-form prior-only references for the proposal kernels are unit tests in
//! `structure_mcmc.rs` and `order_mcmc.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    reason = "test scaffolding indexes small matrices by node"
)]

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{
    dag_from_adjacency_mask, mask_is_dag, parents_of, set_edge,
};
use antecedent_discovery::{
    AdjacencyMask, DiscoveryConstraints, DiscoveryWorkspace, EXACT_MARGINAL_MAX_NODES,
    ExactDagPosterior, GraphPosterior, GraphPrior, LaggedLink, MAX_POSTERIOR_NODES, OrderMcmc,
    StructureMcmc, exact_dag_marginals,
};
use antecedent_state::GraphScoreFamily;

mod common;

fn uniform(rng: &mut CausalRng) -> f64 {
    rng.next_f64() * 2.0 - 1.0
}

fn table(names: &[String], columns: &[Vec<f64>]) -> (TabularData, Vec<VariableId>) {
    let storage = common::continuous_storage(names, columns.to_vec());
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (TabularData::new(storage), variables)
}

/// `a -> c <- b`, `c -> d -> e`, `a -> e`: a collider with a tail, informative but not
/// degenerate at 120 rows.
fn five_variable_columns(rows: usize) -> Vec<Vec<f64>> {
    let mut rng = CausalRng::from_seed(2024);
    let mut cols = vec![vec![0.0; rows]; 5];
    for r in 0..rows {
        let a = uniform(&mut rng);
        let b = uniform(&mut rng);
        let c = 0.8 * a + 0.8 * b + 0.8 * uniform(&mut rng);
        let d = 0.7 * c + 0.8 * uniform(&mut rng);
        let e = 0.6 * d + 0.4 * a + 0.8 * uniform(&mut rng);
        for (col, v) in cols.iter_mut().zip([a, b, c, d, e]) {
            col[r] = v;
        }
    }
    cols
}

/// Roots `x_0..x_{k-1}` (variables `0..k`) and children `y_j = coef (x_j + x_{j+1 mod k}) + noise`
/// (variables `k..2k`). Every edge sits in an unshielded collider, so the skeleton fixes the
/// orientation of the full structure.
fn ring_columns(k: usize, coef: f64, rows: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = CausalRng::from_seed(seed);
    let mut cols = vec![vec![0.0; rows]; 2 * k];
    for r in 0..rows {
        for j in 0..k {
            cols[j][r] = uniform(&mut rng);
        }
        for j in 0..k {
            cols[k + j][r] = coef * (cols[j][r] + cols[(j + 1) % k][r]) + uniform(&mut rng);
        }
    }
    cols
}

/// `b` independent colliders `a_j -> c_j <- b_j` (variables `0..b`, `b..2b`, `2b..3b`, with
/// `c_j = coef (a_j + b_j) + noise`) followed by `extra` unrelated variables. Every edge is in an
/// unshielded collider, so the whole structure is identified, and the blocks do not interact,
/// which keeps single-edge samplers out of the deep local modes a ring of colliders has.
fn collider_block_columns(
    b: usize,
    extra: usize,
    coef: f64,
    rows: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = CausalRng::from_seed(seed);
    let mut cols = vec![vec![0.0; rows]; 3 * b + extra];
    for r in 0..rows {
        for j in 0..b {
            cols[j][r] = uniform(&mut rng);
            cols[b + j][r] = uniform(&mut rng);
            cols[2 * b + j][r] = coef * (cols[j][r] + cols[b + j][r]) + uniform(&mut rng);
        }
        for e in 0..extra {
            cols[3 * b + e][r] = uniform(&mut rng);
        }
    }
    cols
}

/// The `2b` directed edges of [`collider_block_columns`].
fn collider_block_edges(b: usize) -> Vec<(usize, usize)> {
    (0..b).flat_map(|j| [(j, 2 * b + j), (b + j, 2 * b + j)]).collect()
}

/// The `2k` directed edges `x_j -> y_j` and `x_{j+1} -> y_j` of [`ring_columns`].
fn ring_edges(k: usize) -> Vec<(usize, usize)> {
    (0..k).flat_map(|j| [(j, k + j), ((j + 1) % k, k + j)]).collect()
}

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("v{i}")).collect()
}

fn edge_mask(n: usize, edges: &[(usize, usize)]) -> AdjacencyMask {
    edges.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, n, from, to, true))
}

fn max_parents(limit: usize) -> GraphPrior {
    GraphPrior::uniform()
        .with_constraints(DiscoveryConstraints { max_parents: Some(limit), ..Default::default() })
}

fn link(vars: &[VariableId], from: usize, to: usize) -> LaggedLink {
    LaggedLink {
        source: vars[from],
        source_lag: Lag::CONTEMPORANEOUS,
        target: vars[to],
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

fn run_structure(
    engine: &StructureMcmc,
    data: &TabularData,
    vars: &[VariableId],
    prior: &GraphPrior,
    seed: u64,
) -> GraphPosterior {
    engine
        .run(
            data,
            vars,
            prior,
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(seed),
        )
        .unwrap()
}

fn run_order(
    engine: &OrderMcmc,
    data: &TabularData,
    vars: &[VariableId],
    prior: &GraphPrior,
    seed: u64,
) -> GraphPosterior {
    engine
        .run(
            data,
            vars,
            prior,
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(seed),
        )
        .unwrap_or_else(|error| {
            panic!("order MCMC n_vars={} config={engine:?}: {error}", vars.len())
        })
}

/// Every cell within `0.03 + 5 x` its own Monte-Carlo standard error of the reference.
fn assert_marginals_match(label: &str, post: &GraphPosterior, exact: &[f64]) {
    let mcse = post.edge_mcse.as_ref().expect("a sampled posterior carries edge MCSE");
    for (cell, (&got, &want)) in post.edge_marginals.iter().zip(exact).enumerate() {
        let band = 0.03 + 5.0 * mcse[cell];
        assert!(
            (got - want).abs() <= band,
            "{label}: cell {cell} (from {} to {}) is {got}, exact {want}, band {band}",
            cell / post.n_vars,
            cell % post.n_vars
        );
    }
}

/// The per-edge chain diagnostics agree with the summary the publication gate reads.
fn assert_chain_diagnostics_consistent(label: &str, post: &GraphPosterior) {
    let cells = post.n_vars * post.n_vars;
    let rhat = post.edge_rhat.as_ref().expect("per-edge R-hat");
    let ess = post.edge_ess.as_ref().expect("per-edge ESS");
    assert_eq!((rhat.len(), ess.len()), (cells, cells), "{label}");
    let rhat_max = post.diagnostics.rhat_max.expect("R-hat summary");
    let worst = rhat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Cells that never varied report 1, so the worst cell is the summary or 1.
    assert!(
        worst >= rhat_max - 1e-9 && worst <= rhat_max.max(1.0) + 1e-9,
        "{label}: max edge R-hat {worst} vs {rhat_max}"
    );
    let ess_min = post.diagnostics.ess_bulk_min.expect("ESS summary");
    let smallest = ess.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(smallest <= ess_min + 1e-9, "{label}: min edge ESS {smallest} vs {ess_min}");
}

#[test]
fn mcmc_matches_exhaustive_enumeration_on_five_variables() {
    let cols = five_variable_columns(120);
    let (data, vars) = table(&names(5), &cols);
    let prior = GraphPrior::uniform();
    let exact = ExactDagPosterior::new()
        .run(
            &data,
            &vars,
            &prior,
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    assert_eq!(exact.n_graphs, 29_281 - exact.rejected_invalid as usize);
    let support: HashSet<AdjacencyMask> = exact.atoms.iter().copied().collect();
    assert!(exact.atoms.iter().all(|&mask| mask_is_dag(mask, 5)));

    let structure =
        StructureMcmc::new().with_schedule(4, 1000, 8000, 1).with_diagnostics_gate(false);
    let post = run_structure(&structure, &data, &vars, &prior, 11);
    assert_marginals_match("structure", &post, &exact.edge_marginals);
    assert!(post.atoms.iter().all(|mask| support.contains(mask)));
    assert_chain_diagnostics_consistent("structure", &post);

    let order = OrderMcmc::new().with_schedule(4, 1000, 8000, 1).with_diagnostics_gate(false);
    let post = run_order(&order, &data, &vars, &prior, 12);
    assert_marginals_match("order", &post, &exact.edge_marginals);
    assert!(post.atoms.iter().all(|mask| support.contains(mask)));
    assert_chain_diagnostics_consistent("order", &post);
}

/// `ln` Gaussian-BIC local score of `node` on `parents` plus an intercept:
/// `-n/2 (1 + ln 2π + ln σ²) - (|pa| + 1)/2 ln n`, `σ² = SSE / n`. Normal equations with
/// partial pivoting; deliberately not the library's Gram/Cholesky path.
fn local_bic(cols: &[Vec<f64>], node: usize, parents: &[usize]) -> f64 {
    let n = cols[0].len();
    let q = parents.len() + 1;
    let design = |row: usize, k: usize| if k == 0 { 1.0 } else { cols[parents[k - 1]][row] };
    let mut a = vec![vec![0.0; q + 1]; q];
    for row in 0..n {
        for i in 0..q {
            for j in 0..q {
                a[i][j] += design(row, i) * design(row, j);
            }
            a[i][q] += design(row, i) * cols[node][row];
        }
    }
    for col in 0..q {
        let pivot = (col..q).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs())).unwrap();
        a.swap(col, pivot);
        for row in 0..q {
            if row != col {
                let f = a[row][col] / a[col][col];
                for k in col..=q {
                    let sub = f * a[col][k];
                    a[row][k] -= sub;
                }
            }
        }
    }
    let beta: Vec<f64> = (0..q).map(|i| a[i][q] / a[i][i]).collect();
    let sse: f64 = (0..n)
        .map(|row| {
            let fit: f64 = (0..q).map(|k| beta[k] * design(row, k)).sum();
            (cols[node][row] - fit).powi(2)
        })
        .sum();
    let nf = n as f64;
    -0.5 * nf * (1.0 + (2.0 * std::f64::consts::PI).ln() + (sse / nf).ln())
        - 0.5 * q as f64 * nf.ln()
}

/// Exact edge marginals of the uniform-prior Gaussian-BIC posterior restricted to DAGs whose
/// edges lie on `pairs` (each pair absent, `a -> b`, or `b -> a`), with at most `max_parents`
/// parents per node: all `3^pairs.len()` states are scored.
fn restricted_exact_marginals(
    cols: &[Vec<f64>],
    pairs: &[(usize, usize)],
    max_parents: usize,
) -> Vec<f64> {
    let n = cols.len();
    let m = pairs.len();
    let states = 3usize.pow(m as u32);
    let mut local = vec![f64::NAN; n << n];
    let mut log_w = Vec::with_capacity(states);
    let mut valid = Vec::with_capacity(states);
    for state in 0..states {
        let mut parent_bits = [0u16; MAX_POSTERIOR_NODES];
        let mut rest = state;
        for &(a, b) in pairs {
            match rest % 3 {
                1 => parent_bits[b] |= 1 << a,
                2 => parent_bits[a] |= 1 << b,
                _ => {}
            }
            rest /= 3;
        }
        let mut remaining: u16 = (1 << n) - 1;
        loop {
            let before = remaining;
            for v in 0..n {
                if remaining & (1 << v) != 0 && parent_bits[v] & remaining == 0 {
                    remaining &= !(1 << v);
                }
            }
            if remaining == 0 || remaining == before {
                break;
            }
        }
        let acyclic = remaining == 0;
        let within_limit = (0..n).all(|v| parent_bits[v].count_ones() as usize <= max_parents);
        if !(acyclic && within_limit) {
            valid.push(false);
            log_w.push(f64::NEG_INFINITY);
            continue;
        }
        let mut total = 0.0;
        for v in 0..n {
            let slot = &mut local[(v << n) | usize::from(parent_bits[v])];
            if slot.is_nan() {
                let pa: Vec<usize> = (0..n).filter(|&p| parent_bits[v] & (1 << p) != 0).collect();
                *slot = local_bic(cols, v, &pa);
            }
            total += *slot;
        }
        valid.push(true);
        log_w.push(total);
    }
    let top = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let norm: f64 = log_w.iter().map(|lw| (lw - top).exp()).sum();
    let mut marginals = vec![0.0; n * n];
    for state in 0..states {
        if !valid[state] {
            continue;
        }
        let w = (log_w[state] - top).exp() / norm;
        let mut rest = state;
        for &(a, b) in pairs {
            match rest % 3 {
                1 => marginals[a * n + b] += w,
                2 => marginals[b * n + a] += w,
                _ => {}
            }
            rest /= 3;
        }
    }
    marginals
}

/// Both samplers under the published conditions (gate on), restricted to the candidate `pairs`,
/// against the exhaustive Gaussian-BIC enumeration of every state of those pairs.
fn assert_restricted_samplers_match_exact(cols: &[Vec<f64>], pairs: &[(usize, usize)]) {
    let n = cols.len();
    let exact = restricted_exact_marginals(cols, pairs, 3);
    // A reference worth checking: some pairs are neither pinned to 0 nor to 1.
    let undecided = pairs
        .iter()
        .filter(|&&(a, b)| (0.15..0.85).contains(&(exact[a * n + b] + exact[b * n + a])))
        .count();
    assert!(undecided >= 2, "reference is degenerate: {exact:?}");

    let (data, vars) = table(&names(n), cols);
    let candidate: Vec<(u32, u32)> =
        pairs.iter().map(|&(a, b)| (a.min(b) as u32, a.max(b) as u32)).collect();
    let structure =
        published_structure().with_candidate_pairs(Arc::<[(u32, u32)]>::from(candidate));
    let post = run_structure(&structure, &data, &vars, &max_parents(3), 5);
    assert!(post.diagnostics.converged, "structure: {:?}", post.diagnostics);
    assert_marginals_match("structure", &post, &exact);
    assert!(
        n <= 8 || post.atoms.iter().any(|mask| mask.to_word().is_none()),
        "atoms use words past the first"
    );
    assert!(post.atoms.iter().all(|&mask| mask_is_dag(mask, n)));
    assert_chain_diagnostics_consistent("structure", &post);

    // The order sampler has no candidate list: forbid every directed pair off the candidates.
    let allowed: HashSet<(usize, usize)> =
        pairs.iter().flat_map(|&(a, b)| [(a, b), (b, a)]).collect();
    let mut prior = max_parents(3);
    let forbidden: Vec<LaggedLink> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|&(i, j)| i != j && !allowed.contains(&(i, j)))
        .map(|(i, j)| link(&vars, i, j))
        .collect();
    prior.constraints.forbidden = Arc::from(forbidden);
    let post = run_order(&published_order(), &data, &vars, &prior, 6);
    assert!(post.diagnostics.converged, "order: {:?}", post.diagnostics);
    assert_marginals_match("order", &post, &exact);
    assert!(post.atoms.iter().all(|&mask| mask_is_dag(mask, n)));
    assert_chain_diagnostics_consistent("order", &post);
}

/// A twelve-variable interacting target with real posterior uncertainty: the twelve edges of a
/// six-pair collider ring with weak coefficients, so each edge is near the BIC threshold and a
/// lone edge is score-equivalent in both directions. Exact reference: the 3^12 states.
#[test]
fn samplers_match_an_exhaustive_reference_on_the_twelve_variable_ring_skeleton() {
    let cols = ring_columns(6, 0.13, 400, 77);
    assert_restricted_samplers_match_exact(&cols, &ring_edges(6));
}

/// The ten-variable ring with the edge between each collider's two parents allowed as well (so
/// the saturated triangles of the unrestricted posterior are in the reference): 3^15 states.
#[test]
#[ignore = "acceptance workload: 14 million states; run with --ignored in a release build"]
fn samplers_match_an_exhaustive_reference_on_the_ten_variable_ring_with_triangles() {
    let k = 5;
    let cols = ring_columns(k, 0.5, 300, 41);
    let mut pairs = ring_edges(k);
    pairs.extend((0..k).map(|j| (j, (j + 1) % k)));
    assert_restricted_samplers_match_exact(&cols, &pairs);
}

/// With strong signal the posterior is adjacent on the true skeleton, favours the collider
/// directions, and its maximum-weight atom contains every true adjacency, through the
/// multi-word adjacency (twelve variables, edge bits past the first word).
#[test]
fn strong_signal_concentrates_on_the_true_twelve_variable_structure() {
    let b = 4;
    let n = 3 * b;
    let cols = collider_block_columns(b, 0, 0.5, 400, 41);
    let (data, vars) = table(&names(n), &cols);
    let truth_edges = collider_block_edges(b);
    let truth = edge_mask(n, &truth_edges);
    assert!(truth.to_word().is_none());

    let structure =
        StructureMcmc::new().with_schedule(4, 3000, 8_000, 2).with_diagnostics_gate(false);
    let order = OrderMcmc::new().with_schedule(4, 3000, 8_000, 2).with_diagnostics_gate(false);
    for (label, post) in [
        ("structure", run_structure(&structure, &data, &vars, &max_parents(3), 8)),
        ("order", run_order(&order, &data, &vars, &max_parents(3), 9)),
    ] {
        // BIC does not make a collider certain: the saturated triangle on its three nodes (six
        // score-equivalent DAGs) costs only one extra parameter, so a share of the mass sits on
        // reversed edges. What the data do fix is adjacency, and the collider direction stays
        // the more probable one.
        for &(from, to) in &truth_edges {
            let (forward, backward) =
                (post.edge_marginals[from * n + to], post.edge_marginals[to * n + from]);
            assert!(
                forward + backward >= 0.95,
                "{label}: P({from} - {to}) = {}",
                forward + backward
            );
            assert!(
                forward > backward,
                "{label}: P({from} -> {to}) = {forward} vs reversed {backward}"
            );
        }
        let on_truth: HashSet<(usize, usize)> = truth_edges.iter().copied().collect();
        let stray: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|&(i, j)| i != j && !on_truth.contains(&(i, j)) && !on_truth.contains(&(j, i)))
            .map(|(i, j)| post.edge_marginals[i * n + j])
            .sum();
        // 120 off-skeleton directed cells; the saturated triangle puts a share of each block's
        // mass on the edge between its two parents, and BIC admits a few chance edges at 400 rows.
        assert!(stray < 14.0, "{label}: off-skeleton mass {stray}");
        let top = post
            .weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| post.atoms[i])
            .unwrap();
        for &(from, to) in &truth_edges {
            assert!(
                antecedent_discovery::adjacency_mask::has_edge(top, n, from, to)
                    || antecedent_discovery::adjacency_mask::has_edge(top, n, to, from),
                "{label}: the maximum-weight graph lacks the adjacency {from} - {to}"
            );
        }
        assert!(post.atoms.iter().all(|&mask| {
            mask_is_dag(mask, n) && (0..n).all(|v| parents_of(mask, n, v).len() <= 3)
        }));
        let dag = dag_from_adjacency_mask(top, n).unwrap();
        assert!(dag.edges().count() >= truth_edges.len());
        let keys: HashSet<u64> = post.graph_keys.iter().copied().collect();
        let masks: HashSet<AdjacencyMask> = post.atoms.iter().copied().collect();
        assert_eq!(keys.len(), masks.len(), "{label}: one graph key per distinct graph");
    }
}

/// Published conditions for ten and twelve variables: at most three parents per node, four
/// chains, 3 000 warmup iterations and 12 000 retained draws per chain thinned by two, and the
/// default four-replica temperature ladder. Under them the publication gate (R-hat below 1.2,
/// bulk ESS above 10, every edge MCSE within 0.05) clears on the data below.
fn published_structure() -> StructureMcmc {
    StructureMcmc::new().with_schedule(4, 3000, 12_000, 2)
}

fn published_order() -> OrderMcmc {
    OrderMcmc::new().with_schedule(4, 6000, 24_000, 2)
}

/// Independent colliders: the easy case, where even untempered single-edge moves mix.
fn assert_colliders_clear_the_gate(blocks: usize, extra: usize, rows: usize) {
    let n = 3 * blocks + extra;
    let cols = collider_block_columns(blocks, extra, 0.5, rows, 41);
    let (data, vars) = table(&names(n), &cols);
    let post = run_structure(&published_structure(), &data, &vars, &max_parents(3), 21);
    assert!(post.diagnostics.converged, "structure n={n}: {:?}", post.diagnostics);
    assert_chain_diagnostics_consistent("structure", &post);
    let post = run_order(&published_order(), &data, &vars, &max_parents(3), 22);
    assert!(post.diagnostics.converged, "order n={n}: {:?}", post.diagnostics);
    assert_chain_diagnostics_consistent("order", &post);
}

#[test]
fn published_conditions_clear_the_gate_on_independent_colliders_at_ten_variables() {
    assert_colliders_clear_the_gate(3, 1, 300);
}

#[test]
fn published_conditions_clear_the_gate_on_independent_colliders_at_twelve_variables() {
    assert_colliders_clear_the_gate(4, 0, 400);
}

/// The interacting ring of colliders, where every `y_j` shares a parent with each neighbour:
/// the case that traps untempered single-edge moves in local modes 100 nats from the truth.
/// Both samplers must clear the gate under the published conditions, recover the skeleton and
/// the collider directions, and agree with each other (they are different algorithms).
fn assert_ring_clears_the_gate_and_is_recovered(k: usize, coef: f64, rows: usize, seed: u64) {
    let n = 2 * k;
    let cols = ring_columns(k, coef, rows, seed);
    let (data, vars) = table(&names(n), &cols);
    let structure = run_structure(&published_structure(), &data, &vars, &max_parents(3), 21);
    let order = run_order(&published_order(), &data, &vars, &max_parents(3), 22);
    for (label, post) in [("structure", &structure), ("order", &order)] {
        assert!(post.diagnostics.converged, "{label} n={n}: {:?}", post.diagnostics);
        assert_chain_diagnostics_consistent(label, post);
        // Adjacency is the sharp claim: the saturated triangle on a collider's nodes keeps some
        // mass on reversed edges, so BIC does not make the direction certain (the exhaustive
        // reference tests assert it against an exact enumeration instead).
        for (from, to) in ring_edges(k) {
            let adjacent = post.edge_marginals[from * n + to] + post.edge_marginals[to * n + from];
            assert!(adjacent >= 0.95, "{label}: P({from} - {to}) = {adjacent}");
        }
    }
    let (mcse_s, mcse_o) =
        (structure.edge_mcse.as_ref().unwrap(), order.edge_mcse.as_ref().unwrap());
    for cell in 0..n * n {
        let band = 0.03 + 5.0 * (mcse_s[cell] + mcse_o[cell]);
        let gap = (structure.edge_marginals[cell] - order.edge_marginals[cell]).abs();
        assert!(
            gap <= band,
            "cell {cell}: structure {} vs order {} (band {band})",
            structure.edge_marginals[cell],
            order.edge_marginals[cell]
        );
    }
}

#[test]
fn published_conditions_clear_the_gate_on_the_interacting_ring_at_ten_variables() {
    assert_ring_clears_the_gate_and_is_recovered(5, 0.5, 300, 41);
}

#[test]
fn published_conditions_clear_the_gate_on_the_interacting_ring_at_twelve_variables() {
    assert_ring_clears_the_gate_and_is_recovered(6, 1.0, 800, 3);
}

/// Five times the published draws leave the twelve-variable ring's marginals where the published
/// schedule put them: the published conditions are long enough, not merely gate-passing.
#[test]
#[ignore = "acceptance workload: five-fold longer chains; run with --ignored in a release build"]
fn published_schedule_agrees_with_a_five_times_longer_run_on_the_twelve_variable_ring() {
    let k = 6;
    let n = 2 * k;
    let cols = ring_columns(k, 1.0, 800, 3);
    let (data, vars) = table(&names(n), &cols);
    let published = run_structure(&published_structure(), &data, &vars, &max_parents(3), 31);
    let long = StructureMcmc::new().with_schedule(4, 15_000, 60_000, 2);
    let long = run_structure(&long, &data, &vars, &max_parents(3), 32);
    let (a, b) = (published.edge_mcse.as_ref().unwrap(), long.edge_mcse.as_ref().unwrap());
    for cell in 0..n * n {
        let band = 0.02 + 5.0 * (a[cell] + b[cell]);
        let gap = (published.edge_marginals[cell] - long.edge_marginals[cell]).abs();
        assert!(
            gap <= band,
            "cell {cell}: published {} vs long {} (band {band})",
            published.edge_marginals[cell],
            long.edge_marginals[cell]
        );
    }
}

/// The gate is not a formality at this size: a schedule far shorter than the published one is
/// refused rather than published.
#[test]
fn a_short_twelve_variable_schedule_is_refused_by_the_gate() {
    let k = 6;
    let n = 2 * k;
    let cols = ring_columns(k, 0.13, 300, 9);
    let (data, vars) = table(&names(n), &cols);
    for err in [
        StructureMcmc::new()
            .with_schedule(2, 20, 40, 1)
            .run(
                &data,
                &vars,
                &max_parents(3),
                GraphScoreFamily::GaussianBic,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap_err(),
        OrderMcmc::new()
            .with_schedule(2, 20, 40, 1)
            .run(
                &data,
                &vars,
                &max_parents(3),
                GraphScoreFamily::GaussianBic,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap_err(),
    ] {
        assert!(err.to_string().contains("diagnostics gate"), "{err}");
    }
}

/// The full node limit: sixteen variables, 240 edge bits.
#[test]
fn structure_mcmc_runs_on_the_full_sixteen_variable_packing() {
    let k = 8;
    let n = 2 * k;
    assert_eq!(n, MAX_POSTERIOR_NODES);
    let cols = ring_columns(k, 0.6, 400, 13);
    let (data, vars) = table(&names(n), &cols);
    let structure = StructureMcmc::new().with_schedule(2, 300, 300, 1).with_diagnostics_gate(false);
    let post = run_structure(&structure, &data, &vars, &max_parents(3), 4);
    assert_eq!(post.n_vars, 16);
    assert_eq!(post.edge_marginals.len(), 256);
    assert!(post.atoms.iter().all(|&mask| mask_is_dag(mask, n)));
    assert!(post.atoms.iter().any(|mask| mask.words()[1] != 0), "edge bits past word 0");
    let top = post.atoms[0];
    assert!(dag_from_adjacency_mask(top, n).is_ok());

    let seventeen = names(17);
    let wide: Vec<Vec<f64>> = (0..17).map(|i| cols[i % cols.len()].clone()).collect();
    let (data, vars) = table(&seventeen, &wide);
    let err = structure
        .run(
            &data,
            &vars,
            &GraphPrior::uniform(),
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(4),
        )
        .unwrap_err();
    assert!(err.to_string().contains("at most 16 variables"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// Exact marginals by dynamic programming, against enumeration and against the samplers.
// ---------------------------------------------------------------------------------------------

/// Seven variables of a connected linear model with a collider at `x2`: roots `x0`, `x1`;
/// `x2 <- x0, x1`; `x3 <- x2`; `x4 <- x3, x0`; `x5 <- x4`; `x6 <- x5, x2`. The first `n` are used.
fn tractable_columns(n: usize, rows: usize, seed: u64) -> Vec<Vec<f64>> {
    assert!((5..=7).contains(&n));
    let mut rng = CausalRng::from_seed(seed);
    let mut cols = vec![vec![0.0; rows]; 7];
    for r in 0..rows {
        let x0 = uniform(&mut rng);
        let x1 = uniform(&mut rng);
        let x2 = 0.7 * x0 + 0.7 * x1 + 0.6 * uniform(&mut rng);
        let x3 = 0.8 * x2 + 0.6 * uniform(&mut rng);
        let x4 = 0.6 * x3 + 0.5 * x0 + 0.6 * uniform(&mut rng);
        let x5 = 0.7 * x4 + 0.6 * uniform(&mut rng);
        let x6 = 0.6 * x5 + 0.5 * x2 + 0.6 * uniform(&mut rng);
        for (col, value) in cols.iter_mut().zip([x0, x1, x2, x3, x4, x5, x6]) {
            col[r] = value;
        }
    }
    cols.truncate(n);
    cols
}

/// Every labelled DAG visited once, by assigning each node its parent set in turn: the state
/// is the ancestors and descendants of every node, so a parent set that would close a cycle is
/// never offered. The mass of each `(node, parent set)` is the total weight of the DAGs that
/// give the node that set, kept relative to the best weight seen so far.
struct Enumeration {
    n: usize,
    log_factor: Vec<Vec<f64>>,
    mass: Vec<Vec<f64>>,
    top: f64,
    chosen: [usize; 8],
}

impl Enumeration {
    fn visit(&mut self, node: usize, weight: f64, descendants: [usize; 8], ancestors: [usize; 8]) {
        // A parent of `node` can be neither `node` itself nor one of its descendants.
        let allowed = ((1usize << self.n) - 1) & !descendants[node] & !(1 << node);
        let mut parents = allowed;
        loop {
            let factor = self.log_factor[node][parents];
            if factor > f64::NEG_INFINITY {
                self.chosen[node] = parents;
                if node + 1 == self.n {
                    self.record(weight + factor);
                } else {
                    let (below, above) = Self::add_edges(descendants, ancestors, node, parents);
                    self.visit(node + 1, weight + factor, below, above);
                }
            }
            if parents == 0 {
                break;
            }
            parents = (parents - 1) & allowed;
        }
    }

    /// The closure after the edges `parent -> node` for every parent in the set.
    fn add_edges(
        mut descendants: [usize; 8],
        mut ancestors: [usize; 8],
        node: usize,
        parents: usize,
    ) -> ([usize; 8], [usize; 8]) {
        let below = (1 << node) | descendants[node];
        let mut rest = parents;
        while rest != 0 {
            let parent = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            let above = (1 << parent) | ancestors[parent];
            let mut sinks = below;
            while sinks != 0 {
                ancestors[sinks.trailing_zeros() as usize] |= above;
                sinks &= sinks - 1;
            }
            let mut sources = above;
            while sources != 0 {
                descendants[sources.trailing_zeros() as usize] |= below;
                sources &= sources - 1;
            }
        }
        (descendants, ancestors)
    }

    fn record(&mut self, log_weight: f64) {
        if log_weight > self.top {
            let scale = (self.top - log_weight).exp();
            for row in &mut self.mass {
                for cell in row.iter_mut() {
                    *cell *= scale;
                }
            }
            self.top = log_weight;
        }
        let weight = (log_weight - self.top).exp();
        for node in 0..self.n {
            self.mass[node][self.chosen[node]] += weight;
        }
    }
}

/// `P(i -> j)` of the uniform-prior Gaussian-BIC posterior with at most `max_parents` parents,
/// by visiting every DAG, scored by [`local_bic`] (an implementation that shares no code with
/// the library).
fn enumerated_marginals(cols: &[Vec<f64>], max_parents: usize) -> Vec<f64> {
    let n = cols.len();
    let mut log_factor = vec![vec![f64::NEG_INFINITY; 1 << n]; n];
    for (node, row) in log_factor.iter_mut().enumerate() {
        for (parents, slot) in row.iter_mut().enumerate() {
            if parents & (1 << node) != 0 || parents.count_ones() as usize > max_parents {
                continue;
            }
            let list: Vec<usize> = (0..n).filter(|p| (parents >> p) & 1 == 1).collect();
            *slot = local_bic(cols, node, &list);
        }
    }
    let mut walk = Enumeration {
        n,
        log_factor,
        mass: vec![vec![0.0; 1 << n]; n],
        top: f64::NEG_INFINITY,
        chosen: [0; 8],
    };
    walk.visit(0, 0.0, [0; 8], [0; 8]);
    let total: f64 = walk.mass[0].iter().sum();
    let mut marginals = vec![0.0; n * n];
    for target in 0..n {
        for (parents, mass) in walk.mass[target].iter().enumerate() {
            for source in (0..n).filter(|s| (parents >> s) & 1 == 1) {
                marginals[source * n + target] += mass / total;
            }
        }
    }
    marginals
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (cell, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-9, "{label}: cell {cell} is {g}, enumeration {w}");
    }
}

/// The dynamic programme (and, up to six variables, the library's enumerating engine) against
/// the independent enumeration, cell by cell.
fn assert_exact_posterior_matches_enumeration(n: usize, seed: u64) {
    let cols = tractable_columns(n, 1000, seed);
    let (data, vars) = table(&names(n), &cols);
    let prior = GraphPrior::uniform();
    let reference = enumerated_marginals(&cols, n - 1);
    let total_edge_mass: f64 = reference.iter().sum();
    assert!(total_edge_mass > 0.5, "the posterior is not degenerate: {reference:?}");
    let dp = exact_dag_marginals(
        &data,
        &vars,
        &prior,
        GraphScoreFamily::GaussianBic,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_eq!(dp.n_vars, n);
    assert_close(&format!("dynamic programme, {n} variables"), &dp.edge_marginals, &reference);
    if n <= 6 {
        let engine = ExactDagPosterior::new()
            .run(
                &data,
                &vars,
                &prior,
                GraphScoreFamily::GaussianBic,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        assert_close(
            &format!("enumerating engine, {n} variables"),
            &engine.edge_marginals,
            &reference,
        );
    }
}

#[test]
fn exact_marginals_equal_an_independent_enumeration_on_five_and_six_variables() {
    assert_exact_posterior_matches_enumeration(5, 4_604_603);
    assert_exact_posterior_matches_enumeration(6, 4_604_604);
}

/// Seven variables have 1,138,779,265 labelled DAGs: the enumeration visits every one.
#[test]
#[ignore = "acceptance workload: 1.1 billion DAGs at seven variables; run with --ignored in a release build"]
fn exact_posterior_matches_independent_enumeration_on_tractable_cases() {
    assert_exact_posterior_matches_enumeration(5, 4_604_603);
    assert_exact_posterior_matches_enumeration(6, 4_604_604);
    assert_exact_posterior_matches_enumeration(7, 4_604_605);
    // The enumerating engine stops at six; the recursion is what reaches seven.
    let cols = tractable_columns(7, 200, 1);
    let (data, vars) = table(&names(7), &cols);
    let refused = ExactDagPosterior::new()
        .run(
            &data,
            &vars,
            &GraphPrior::uniform(),
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
    assert!(refused.to_string().contains("n ≤ 6"), "{refused}");
}

#[test]
fn exact_marginals_respect_the_parent_cap_and_refuse_beyond_their_size() {
    let cols = tractable_columns(6, 1000, 77);
    let (data, vars) = table(&names(6), &cols);
    let capped = max_parents(2);
    let dp = exact_dag_marginals(
        &data,
        &vars,
        &capped,
        GraphScoreFamily::GaussianBic,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_close("parent cap of two", &dp.edge_marginals, &enumerated_marginals(&cols, 2));

    let too_wide = EXACT_MARGINAL_MAX_NODES + 1;
    let cols: Vec<Vec<f64>> = (0..too_wide).map(|i| cols[i % cols.len()].clone()).collect();
    let (data, vars) = table(&names(too_wide), &cols);
    let refused = exact_dag_marginals(
        &data,
        &vars,
        &max_parents(3),
        GraphScoreFamily::GaussianBic,
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert!(refused.to_string().contains("at most 12 variables"), "{refused}");
}

/// Both samplers against the exact posterior at ten and twelve variables on 2,000 rows:
/// every edge marginal within `0.03 + 5 x` its Monte-Carlo standard error. The order
/// sampler gets longer chains here than the separate published-workflow checks.
#[test]
fn samplers_agree_with_the_exact_marginals_at_ten_and_twelve_variables() {
    for (label, k, coef, seed) in [("ten", 5, 0.5, 4_867_969), ("twelve", 6, 1.0, 5_398_576)] {
        let n = 2 * k;
        let cols = ring_columns(k, coef, 2000, seed);
        let (data, vars) = table(&names(n), &cols);
        let prior = max_parents(3);
        let exact = exact_dag_marginals(
            &data,
            &vars,
            &prior,
            GraphScoreFamily::GaussianBic,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let structure = run_structure(&published_structure(), &data, &vars, &prior, 21);
        assert!(structure.diagnostics.converged, "{label} structure: {:?}", structure.diagnostics);
        assert_marginals_match(&format!("{label} structure"), &structure, &exact.edge_marginals);
        assert_chain_diagnostics_consistent(&format!("{label} structure"), &structure);
        // This exact-oracle comparison uses twice the order-chain effort of the
        // workflow checks. Floating score differences across platforms can change
        // a finite chain trajectory; retain the production publication gate and
        // give its convergence checks more sampling headroom, never a wider bar.
        let order_engine = OrderMcmc::new().with_schedule(4, 12_000, 48_000, 2);
        let order = run_order(&order_engine, &data, &vars, &prior, 22);
        assert!(order.diagnostics.converged, "{label} order: {:?}", order.diagnostics);
        assert_marginals_match(&format!("{label} order"), &order, &exact.edge_marginals);
        assert_chain_diagnostics_consistent(&format!("{label} order"), &order);
    }
}
