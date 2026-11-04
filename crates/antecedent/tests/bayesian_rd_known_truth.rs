//! Exact-posterior evidence for the fixed-bandwidth Bayesian local-linear sharp RD.
//!
//! The model is `y ~ 1 + T + x + T x` on the rows with `|x| <= h`, an independent
//! `N(0, v sd^2)` prior on each coefficient (`sd` is the study's prior scale, `v` the residual
//! variance of the posterior-mean fit with `n - 4` degrees of freedom), so the posterior of
//! the jump (the `T` coefficient) is exactly normal with mean `beta_1` of the ridge system
//! `(X'X + I / sd^2) beta = X'y` and variance `v [(X'X + I / sd^2)^-1]_11`. The test solves that
//! 4x4 system by Gaussian elimination from the rows, and compares the mean of 20 000 posterior
//! draws with `beta_1` to six Monte-Carlo standard errors, `6 sqrt(var / 20000)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "known_truth_cells/mod.rs"]
mod support;

use antecedent::{AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use support::{GraphKind, Structure, for_each_matching, on_large_stack};

const N: usize = 400;
const BANDWIDTH: f64 = 0.8;
const PRIOR_SD: f64 = 20.0;
const DRAWS: usize = 20_000;

fn columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let r: Vec<f64> = (0..N).map(|i| (i as f64 - 200.0) / 100.0).collect();
    let t: Vec<f64> = r.iter().map(|x| f64::from(*x >= 0.0)).collect();
    let y: Vec<f64> = (0..N)
        .map(|i| 1.0 + 2.5 * t[i] + 0.8 * r[i] + 1.2 * t[i] * r[i] + ((i % 5) as f64 - 2.0) * 0.01)
        .collect();
    (t, y, r)
}

/// Solve `a x = b` (4x4) by Gaussian elimination with partial pivoting.
fn solve(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    for p in 0..4 {
        let piv = (p..4).max_by(|&u, &v| a[u][p].abs().total_cmp(&a[v][p].abs())).unwrap();
        a.swap(p, piv);
        b.swap(p, piv);
        for r in p + 1..4 {
            let f = a[r][p] / a[p][p];
            for c in p..4 {
                let v = a[p][c];
                a[r][c] -= f * v;
            }
            b[r] -= f * b[p];
        }
    }
    let mut x = [0.0; 4];
    for r in (0..4).rev() {
        x[r] = (b[r] - (r + 1..4).map(|c| a[r][c] * x[c]).sum::<f64>()) / a[r][r];
    }
    x
}

/// Exact posterior mean and sd of the jump, from the window rows.
fn exact_posterior() -> (f64, f64) {
    let (t, y, r) = columns();
    let mut gram = [[0.0; 4]; 4];
    let mut moment = [0.0; 4];
    let mut rows = Vec::new();
    for i in 0..N {
        if r[i].abs() <= BANDWIDTH {
            let row = [1.0, t[i], r[i], t[i] * r[i]];
            for a in 0..4 {
                moment[a] += row[a] * y[i];
                for b in 0..4 {
                    gram[a][b] += row[a] * row[b];
                }
            }
            rows.push((row, y[i]));
        }
    }
    for (a, gram_row) in gram.iter_mut().enumerate().take(4) {
        gram_row[a] += 1.0 / (PRIOR_SD * PRIOR_SD);
    }
    let beta = solve(gram, moment);
    let rss: f64 = rows
        .iter()
        .map(|(row, yi)| (yi - (0..4).map(|j| row[j] * beta[j]).sum::<f64>()).powi(2))
        .sum();
    let v = (rss / (rows.len() - 4) as f64).max(1e-12);
    let column = solve(gram, [0.0, 1.0, 0.0, 0.0]);
    (beta[1], (v * column[1]).sqrt())
}

#[test]
fn bayesian_rd_posterior_mean_matches_the_exact_ridge_posterior() {
    let (mean, sd) = exact_posterior();
    assert!((mean - 2.5).abs() < 0.05, "exact posterior mean {mean} is near the structural jump");
    let tolerance = 6.0 * sd / (DRAWS as f64).sqrt() + 1e-9;
    on_large_stack(move || {
        let (t, y, r) = columns();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("r", r.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (from, to) in [(2, 0), (0, 1), (2, 1)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        for_each_matching(
            "rd.bayesian_local_linear|AverageEffect",
            |c| {
                c.graph == GraphKind::Dag
                    && c.structure == Structure::Accepted
                    && c.suite_label == "none"
            },
            |c| {
                let result = Study::tabular(data.clone())
                    .graph(AcceptedGraph::from(graph.clone()))
                    .query(AverageEffectQuery::binary_ate(
                        VariableId::from_raw(0),
                        VariableId::from_raw(1),
                    ))
                    .identifier(IdentifierId::RdSharp)
                    .estimator(EstimatorId::BayesianRdLocalLinear)
                    .rd_config(VariableId::from_raw(2), 0.0, BANDWIDTH)
                    .inference(InferenceMode::Bayesian(
                        BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(PRIOR_SD),
                    ))
                    .refute(c.suite)
                    .build()
                    .unwrap_or_else(|e| panic!("{}: build: {e}", c.text))
                    .run(&ExecutionContext::for_tests(9))
                    .unwrap_or_else(|e| panic!("{}: run: {e}", c.text));
                assert_eq!(
                    result.logical_plan.estimator.as_deref(),
                    Some("rd.bayesian_local_linear"),
                    "{}",
                    c.text
                );
                let got = result.estimate().as_effect().unwrap().ate;
                assert!(
                    (got - mean).abs() <= tolerance,
                    "{}: posterior mean {got} vs exact {mean} (tolerance {tolerance})",
                    c.text
                );
            },
        );
    });
}
