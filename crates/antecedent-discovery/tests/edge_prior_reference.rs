//! Per-edge inclusion priors and soft tiers against independent oracles.
//!
//! The oracle is the exact enumeration of every labelled DAG (`ExactDagPosterior` under the
//! uniform prior: its weights are `exp(score)` of each DAG and carry no edge factor), with the
//! per-edge or soft-tier prior multiplied in by hand, edge by edge, in the test. The subset
//! dynamic programme with the same prior must reproduce those marginals; the samplers must
//! agree with it within Monte-Carlo error; hard constraints must win over soft tiers; and a
//! prior that sets neither factor must keep printing as it always did, since that text enters
//! the digest of a discovery query.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "small graph counts and exact posterior ratios of identical computations"
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{
    CiScreenedPosterior, DiscoveryConstraints, DiscoveryWorkspace, EdgePriors, ExactDagPosterior,
    GraphPosterior, GraphPrior, OrderMcmc, SoftTiers, StructureMcmc, exact_dag_marginals,
};
use antecedent_state::GraphScoreFamily;

mod common;

const FAMILY: GraphScoreFamily = GraphScoreFamily::GaussianBic;

fn vars(n: usize) -> Vec<VariableId> {
    (0..n).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect()
}

/// Rows of a linear-Gaussian SEM in node order `0, 1, ...`: `parents[j]` lists `(i, coefficient)`.
fn sem_rows(rows: usize, parents: &[Vec<(usize, f64)>], seed: u64) -> TabularData {
    let n = parents.len();
    let mut rng = CausalRng::from_seed(seed);
    let mut cols = vec![vec![0.0; rows]; n];
    for row in 0..rows {
        for node in 0..n {
            let noise = rng.next_f64() * 2.0 - 1.0;
            let mean: f64 = parents[node].iter().map(|&(p, b)| b * cols[p][row]).sum();
            cols[node][row] = mean + noise;
        }
    }
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    TabularData::new(common::continuous_storage(&names, cols))
}

/// A chain `0 -> 1 -> ... -> n-1` with a chord `0 -> 2`.
fn chain_with_chord(n: usize, rows: usize, seed: u64) -> TabularData {
    let mut parents: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for j in 1..n {
        parents[j].push((j - 1, 0.6));
    }
    if n > 2 {
        parents[2].push((0, 0.4));
    }
    sem_rows(rows, &parents, seed)
}

/// Every enumerated DAG with its likelihood-only weight (uniform prior).
fn enumerated(data: &TabularData, n: usize) -> GraphPosterior {
    ExactDagPosterior::new()
        .run(
            data,
            &vars(n),
            &GraphPrior::uniform(),
            FAMILY,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap()
}

/// Marginals of the enumerated DAGs reweighted by `factor(mask)`.
fn reweighted_marginals(
    post: &GraphPosterior,
    n: usize,
    factor: impl Fn(&antecedent_discovery::AdjacencyMask) -> f64,
) -> Vec<f64> {
    let mut mass = 0.0;
    let mut edge = vec![0.0; n * n];
    for (mask, weight) in post.atoms().iter().zip(post.weights().iter()) {
        let w = weight * factor(mask);
        mass += w;
        for i in 0..n {
            for j in 0..n {
                if i != j && mask.has_edge(n, i, j) {
                    edge[i * n + j] += w;
                }
            }
        }
    }
    edge.iter().map(|v| v / mass).collect()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, label: &str) {
    assert_eq!(got.len(), want.len());
    for (cell, (g, w)) in got.iter().zip(want).enumerate() {
        let n = (got.len() as f64).sqrt() as usize;
        assert!((g - w).abs() <= tol, "{label}: P({} -> {}) = {g} vs {w}", cell / n, cell % n);
    }
}

/// A per-edge map with a deliberate spread, listing about half the ordered pairs.
fn listed_edges(n: usize) -> Vec<(VariableId, VariableId, f64)> {
    let v = vars(n);
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j && (i + 2 * j) % 3 != 0 {
                out.push((v[i], v[j], 0.05 + 0.9 * (((3 * i + 5 * j) % 7) as f64) / 7.0));
            }
        }
    }
    out
}

fn edge_prior_probability(
    listed: &[(VariableId, VariableId, f64)],
    default: f64,
    i: usize,
    j: usize,
) -> f64 {
    listed
        .iter()
        .find(|&&(a, b, _)| a.raw() as usize == i && b.raw() as usize == j)
        .map_or(default, |&(_, _, p)| p)
}

#[test]
fn per_edge_marginals_equal_brute_force_enumeration_over_all_dags() {
    for n in [3usize, 4, 5] {
        let data = chain_with_chord(n, 120, 11 + n as u64);
        let post = enumerated(&data, n);
        let listed = listed_edges(n);
        let default = 0.2;
        let prior = GraphPrior::uniform()
            .with_edge_priors(EdgePriors::new(listed.clone()).with_default(default));
        let want = reweighted_marginals(&post, n, |mask| {
            let mut factor = 1.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let p = edge_prior_probability(&listed, default, i, j);
                        factor *= if mask.has_edge(n, i, j) { p } else { 1.0 - p };
                    }
                }
            }
            factor
        });
        let got =
            exact_dag_marginals(&data, &vars(n), &prior, FAMILY, &ExecutionContext::for_tests(1))
                .unwrap();
        assert_close(&got.edge_marginals, &want, 1e-9, &format!("n={n} per-edge"));
    }
}

#[test]
fn unlisted_edges_fall_back_to_the_global_inclusion_probability() {
    let n = 4;
    let data = chain_with_chord(n, 120, 3);
    let post = enumerated(&data, n);
    let listed = vec![(vars(n)[0], vars(n)[1], 0.8)];
    let global = 0.15;
    let prior = GraphPrior::bernoulli_edges(global)
        .unwrap()
        .with_edge_priors(EdgePriors::new(listed.clone()));
    let want = reweighted_marginals(&post, n, |mask| {
        let mut factor = 1.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let p = edge_prior_probability(&listed, global, i, j);
                    factor *= if mask.has_edge(n, i, j) { p } else { 1.0 - p };
                }
            }
        }
        factor
    });
    let got = exact_dag_marginals(&data, &vars(n), &prior, FAMILY, &ExecutionContext::for_tests(1))
        .unwrap();
    assert_close(&got.edge_marginals, &want, 1e-9, "global fallback");
}

#[test]
fn a_map_listing_every_edge_at_one_probability_is_the_global_bernoulli_prior() {
    let n = 4;
    let data = chain_with_chord(n, 100, 5);
    let p = 0.3;
    let v = vars(n);
    let all: Vec<_> = (0..n)
        .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
        .map(|(i, j)| (v[i], v[j], p))
        .collect();
    let mapped = exact_dag_marginals(
        &data,
        &v,
        &GraphPrior::uniform().with_edge_priors(EdgePriors::new(all)),
        FAMILY,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let global = exact_dag_marginals(
        &data,
        &v,
        &GraphPrior::bernoulli_edges(p).unwrap(),
        FAMILY,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_close(&mapped.edge_marginals, &global.edge_marginals, 1e-12, "mapped vs global");
    assert!((mapped.log_evidence - global.log_evidence).abs() < 1e-9);
}

fn tiers(v: &[VariableId], groups: &[&[usize]]) -> Arc<[Arc<[VariableId]>]> {
    groups.iter().map(|g| Arc::from(g.iter().map(|&i| v[i]).collect::<Vec<_>>())).collect()
}

/// Edges `i -> j` with `i` in a later tier than `j` of `[[0], [1, 2], [3]]`-style tiers.
fn against_order(tier_of: &[usize], mask: &antecedent_discovery::AdjacencyMask, n: usize) -> usize {
    let mut count = 0;
    for i in 0..n {
        for j in 0..n {
            if i != j && mask.has_edge(n, i, j) && tier_of[i] > tier_of[j] {
                count += 1;
            }
        }
    }
    count
}

#[test]
fn soft_tier_penalty_is_the_hand_computed_posterior_ratio() {
    let n = 4;
    let data = chain_with_chord(n, 150, 9);
    let post = enumerated(&data, n);
    let v = vars(n);
    let tier_of = [0usize, 1, 1, 2];
    let penalty = 1.7;
    let prior = GraphPrior::uniform()
        .with_soft_tiers(SoftTiers::with_log_penalty(tiers(&v, &[&[0], &[1, 2], &[3]]), penalty));

    // The posterior of two DAGs differ by exp(-penalty) per extra edge against the order.
    let soft = ExactDagPosterior::new()
        .run(
            &data,
            &v,
            &prior,
            FAMILY,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    assert_eq!(soft.atoms().len(), post.atoms().len());
    let mut offset = None;
    for (k, mask) in post.atoms().iter().enumerate() {
        let ratio = (soft.weights()[k] / post.weights()[k]).ln();
        let against = against_order(&tier_of, mask, n) as f64;
        let shifted = ratio + penalty * against;
        match offset {
            None => offset = Some(shifted),
            Some(first) => {
                assert!((shifted - first).abs() < 1e-9, "atom {k}: {shifted} vs {first}")
            }
        }
    }

    // The exact marginals under the soft tiers equal the hand-reweighted enumeration.
    let want = reweighted_marginals(&post, n, |mask| {
        (-penalty * against_order(&tier_of, mask, n) as f64).exp()
    });
    let got =
        exact_dag_marginals(&data, &v, &prior, FAMILY, &ExecutionContext::for_tests(1)).unwrap();
    assert_close(&got.edge_marginals, &want, 1e-9, "soft tiers");
}

#[test]
fn an_odds_factor_is_a_log_penalty_of_minus_its_log() {
    let v = vars(3);
    let by_factor = SoftTiers::with_odds_factor(tiers(&v, &[&[0], &[1, 2]]), 0.25).unwrap();
    assert!((by_factor.log_penalty - 4.0_f64.ln()).abs() < 1e-15);
    assert!(SoftTiers::with_odds_factor(tiers(&v, &[&[0], &[1, 2]]), 1.0).is_err());
    assert!(SoftTiers::with_odds_factor(tiers(&v, &[&[0], &[1, 2]]), 0.0).is_err());
}

#[test]
fn hard_tiers_win_over_soft_tiers() {
    let n = 4;
    let data = chain_with_chord(n, 150, 21);
    let v = vars(n);
    let order = tiers(&v, &[&[0], &[1, 2], &[3]]);
    let hard = GraphPrior::uniform().with_constraints(DiscoveryConstraints {
        tiers: Arc::clone(&order),
        ..GraphPrior::uniform().constraints
    });
    let both = GraphPrior {
        soft_tiers: Some(SoftTiers::with_log_penalty(Arc::clone(&order), 3.0)),
        ..hard.clone()
    };
    let ctx = ExecutionContext::for_tests(1);
    let alone = exact_dag_marginals(&data, &v, &hard, FAMILY, &ctx).unwrap();
    let composed = exact_dag_marginals(&data, &v, &both, FAMILY, &ctx).unwrap();
    assert_close(&composed.edge_marginals, &alone.edge_marginals, 1e-12, "hard wins");
    assert!((composed.log_evidence - alone.log_evidence).abs() < 1e-9);
    // And every edge the hard tiers forbid keeps exactly zero mass.
    assert_eq!(composed.edge_marginals[3 * n], 0.0);
    assert_eq!(composed.edge_marginals[3 * n + 1], 0.0);
}

#[test]
fn a_soft_tier_shifts_mass_toward_the_declared_order() {
    let n = 3;
    let data = chain_with_chord(n, 20, 33);
    let v = vars(n);
    let ctx = ExecutionContext::for_tests(1);
    let free = exact_dag_marginals(&data, &v, &GraphPrior::uniform(), FAMILY, &ctx).unwrap();
    let soft = exact_dag_marginals(
        &data,
        &v,
        &GraphPrior::uniform()
            .with_soft_tiers(SoftTiers::with_log_penalty(tiers(&v, &[&[0], &[1], &[2]]), 4.0)),
        FAMILY,
        &ctx,
    )
    .unwrap();
    assert!(soft.edge_marginals[n] < free.edge_marginals[n], "1 -> 0 loses mass");
    assert!(soft.edge_marginals[1] > 0.0 && soft.edge_marginals[n] > 0.0, "short of forbidding");
}

#[test]
fn invalid_per_edge_priors_and_soft_tiers_are_refused() {
    let n = 3;
    let data = chain_with_chord(n, 60, 2);
    let v = vars(n);
    let ctx = ExecutionContext::for_tests(1);
    let run = |prior: &GraphPrior| exact_dag_marginals(&data, &v, prior, FAMILY, &ctx);
    let with = |edges: Vec<(VariableId, VariableId, f64)>| {
        GraphPrior::uniform().with_edge_priors(EdgePriors::new(edges))
    };
    let stranger = VariableId::from_raw(99);
    for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
        assert!(run(&with(vec![(v[0], v[1], bad)])).is_err(), "p = {bad}");
        assert!(
            run(&GraphPrior::uniform().with_edge_priors(
                EdgePriors::new(Vec::<(VariableId, VariableId, f64)>::new()).with_default(bad)
            ))
            .is_err(),
            "default = {bad}"
        );
    }
    assert!(run(&with(vec![(v[0], stranger, 0.3)])).is_err(), "unknown target");
    assert!(run(&with(vec![(stranger, v[0], 0.3)])).is_err(), "unknown source");
    assert!(run(&with(vec![(v[1], v[1], 0.3)])).is_err(), "self edge");
    assert!(run(&with(vec![(v[0], v[1], 0.3), (v[0], v[1], 0.4)])).is_err(), "repeated edge");

    // An edge the hard constraints forbid cannot carry a prior probability.
    let forbidding = DiscoveryConstraints {
        tiers: tiers(&v, &[&[0], &[1], &[2]]),
        ..GraphPrior::uniform().constraints
    };
    let contradicting = with(vec![(v[2], v[0], 0.3)]).with_constraints(forbidding.clone());
    assert!(run(&contradicting).is_err(), "edge against a hard tier");
    assert!(run(&with(vec![(v[0], v[2], 0.3)]).with_constraints(forbidding)).is_ok());

    let soft = |penalty: f64| {
        GraphPrior::uniform()
            .with_soft_tiers(SoftTiers::with_log_penalty(tiers(&v, &[&[0], &[1, 2]]), penalty))
    };
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(run(&soft(bad)).is_err(), "penalty = {bad}");
    }
    let outsider: Arc<[VariableId]> = Arc::from(vec![stranger]);
    let unknown =
        GraphPrior::uniform().with_soft_tiers(SoftTiers::with_log_penalty(vec![outsider], 1.0));
    assert!(run(&unknown).is_err(), "soft tier naming an unknown variable");
    assert!(run(&soft(0.0)).is_ok());
}

#[test]
fn a_prior_without_per_edge_factors_prints_as_it_always_did() {
    // The Debug text of the prior is hashed into the configuration digest of a discovery query.
    for prior in [GraphPrior::uniform(), GraphPrior::bernoulli_edges(0.2).unwrap()] {
        let text = format!("{prior:?}");
        assert!(text.starts_with("GraphPrior { constraints: "), "{text}");
        assert!(text.contains(", edge_inclusion: "), "{text}");
        assert!(!text.contains("edge_priors") && !text.contains("soft_tiers"), "{text}");
        assert!(text.ends_with(" }"), "{text}");
    }
    let v = vars(2);
    let with_map = GraphPrior::uniform().with_edge_priors(EdgePriors::new(vec![(v[0], v[1], 0.4)]));
    assert!(format!("{with_map:?}").contains("edge_priors"));
}

fn small_problem() -> (TabularData, Vec<VariableId>, GraphPrior) {
    let n = 4;
    let data = chain_with_chord(n, 200, 77);
    let v = vars(n);
    let listed = vec![(v[0], v[1], 0.7), (v[1], v[2], 0.6), (v[3], v[2], 0.1), (v[2], v[0], 0.02)];
    let prior = GraphPrior::uniform()
        .with_edge_priors(EdgePriors::new(listed).with_default(0.25))
        .with_soft_tiers(SoftTiers::with_log_penalty(tiers(&v, &[&[0], &[1, 2], &[3]]), 1.5));
    (data, v, prior)
}

/// The samplers target the per-edge posterior: every cell within `0.03 + 5 x` its Monte-Carlo
/// standard error of the exact dynamic-programme marginal.
#[test]
fn order_and_structure_samplers_match_the_exact_marginals_under_per_edge_priors() {
    let (data, v, prior) = small_problem();
    let n = v.len();
    let exact =
        exact_dag_marginals(&data, &v, &prior, FAMILY, &ExecutionContext::for_tests(1)).unwrap();
    let structure =
        StructureMcmc::new().with_schedule(4, 1000, 8000, 1).with_diagnostics_gate(false);
    let order = OrderMcmc::new().with_schedule(4, 1000, 8000, 1).with_diagnostics_gate(false);
    let posteriors = [
        (
            "structure",
            structure
                .run(
                    &data,
                    &v,
                    &prior,
                    FAMILY,
                    &mut DiscoveryWorkspace::default(),
                    &ExecutionContext::for_tests(201),
                )
                .unwrap(),
        ),
        (
            "order",
            order
                .run(
                    &data,
                    &v,
                    &prior,
                    FAMILY,
                    &mut DiscoveryWorkspace::default(),
                    &ExecutionContext::for_tests(202),
                )
                .unwrap(),
        ),
    ];
    for (label, post) in posteriors {
        let mcse = post.edge_mcse().expect("a sampled posterior carries edge MCSE");
        for (cell, (&got, &want)) in
            post.edge_marginals().iter().zip(&*exact.edge_marginals).enumerate()
        {
            let band = 0.03 + 5.0 * mcse[cell];
            assert!(
                (got - want).abs() <= band,
                "{label}: P({} -> {}) is {got}, exact {want}, band {band}",
                cell / n,
                cell % n
            );
        }
    }
}

/// Eight variables: `0 -> 1 -> 2` carry signal, `3 -> 4` a second pair, `5, 6, 7` are noise.
fn screened_problem() -> (TabularData, Vec<VariableId>) {
    let mut parents: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 8];
    parents[1].push((0, 0.9));
    parents[2].push((1, 0.9));
    parents[4].push((3, 0.9));
    (sem_rows(400, &parents, 5), vars(8))
}

#[test]
fn the_screened_route_runs_the_exact_marginals_on_the_selected_variables() {
    let (data, v) = screened_problem();
    let engine = CiScreenedPosterior::new();
    let out = engine
        .screen_then_exact_marginals(
            &data,
            &v,
            &[v[1]],
            3,
            &GraphPrior::uniform(),
            0.5,
            0.02,
            FAMILY,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(3),
        )
        .unwrap();
    // The neighbours of the focus in the screen's skeleton are the chain's other two variables.
    assert_eq!(&*out.variables, &[v[0], v[1], v[2]]);
    assert_eq!(out.marginals.n_vars, 3);
    assert!(out.screened_in_pairs >= 2);
    assert!(out.conditional_on_screen.contains("conditional on the CI screen"));
    // Marginals are those of the subset posterior under the screen-derived prior: the same
    // numbers as calling the exact route on the subset with the same per-edge prior.
    let m = &out.marginals.edge_marginals;
    let adjacent = |i: usize, j: usize| m[i * 3 + j] + m[j * 3 + i];
    assert!(adjacent(0, 1) > 0.9 && adjacent(1, 2) > 0.9);
    assert!(adjacent(0, 2) < 0.5);
}

#[test]
fn the_screened_route_respects_the_exact_ceiling_and_refuses_bad_questions() {
    let (data, v) = screened_problem();
    let engine = CiScreenedPosterior::new();
    let ask = |focus: &[VariableId], max: usize, p_in: f64, p_out: f64| {
        engine.screen_then_exact_marginals(
            &data,
            &v,
            focus,
            max,
            &GraphPrior::uniform(),
            p_in,
            p_out,
            FAMILY,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(3),
        )
    };
    assert!(ask(&[v[1]], 16, 0.5, 0.02).is_err(), "above the exact ceiling");
    assert!(ask(&[v[1]], 0, 0.5, 0.02).is_err(), "no variables");
    assert!(ask(&[], 4, 0.5, 0.02).is_err(), "no focus");
    assert!(ask(&[v[0], v[1], v[2]], 2, 0.5, 0.02).is_err(), "focus larger than the set");
    assert!(ask(&[VariableId::from_raw(99)], 4, 0.5, 0.02).is_err(), "unknown focus");
    assert!(ask(&[v[1]], 4, 1.0, 0.02).is_err(), "probability one");
    assert!(ask(&[v[1]], 4, 0.5, 0.0).is_err(), "probability zero");
}

#[test]
fn the_screened_route_keeps_a_callers_own_edge_probability() {
    let (data, v) = screened_problem();
    let engine = CiScreenedPosterior::new();
    let run = |prior: &GraphPrior| {
        engine
            .screen_then_exact_marginals(
                &data,
                &v,
                &[v[1]],
                3,
                prior,
                0.5,
                0.02,
                FAMILY,
                &mut DiscoveryWorkspace::default(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap()
    };
    let plain = run(&GraphPrior::uniform());
    let stated = run(&GraphPrior::uniform()
        .with_edge_priors(EdgePriors::new(vec![(v[2], v[1], 0.001), (v[7], v[6], 0.9)])));
    // An edge naming a variable outside the selected set is ignored, not refused.
    assert_eq!(&*stated.variables, &*plain.variables);
    let n = 3;
    assert!(stated.marginals.edge_marginals[2 * n + 1] < plain.marginals.edge_marginals[2 * n + 1]);
}
