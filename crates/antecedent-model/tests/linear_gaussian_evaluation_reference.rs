// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ModelEvaluator` against an independent high-precision evaluation of a linear Gaussian
//! structural model on a small data set.
//!
//! The constants below were produced by a script with `mpmath` at 40 digits: it draws the
//! data from a seeded generator (with an extra edge 2 -> 0 that the model omits, so some
//! implied independencies fail), then computes
//! * the mean per-term Gaussian log-likelihood, `-log(2 pi s^2)/2 - e^2/(2 s^2)`, over every
//!   node and row, with `e = y - (intercept + coeffs . parents)`;
//! * the mean absolute residual over the non-root nodes;
//! * for every ordered pair `(i, j)`, the two-sided Student-t p-value
//!   `I_{df/(df+t^2)}(df/2, 1/2)` of the partial correlation of `i` and `j` given the parents
//!   of `i` (`df = n - 2 - |parents|`, residualising by ordinary least squares with an
//!   intercept) for the local Markov table, and of the residual of `i` with `j` given nothing
//!   for the residual-independence table.
//!
//! The pairs the evaluator is obliged to test are selected here from the graph and the
//! compiled topological order: local Markov pairs are the nodes earlier in the order that are
//! not parents; residual pairs are the nodes that are neither parents, descendants nor the
//! node itself.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ModelEvaluator,
};

const N: usize = 14;
const COL0: [f64; N] = [-1.1908, -0.2822, 0.8592, -0.5744, -0.2361, 1.2491, 2.2031, -1.6639, -0.9728, 4.184, 2.0273, 0.113, -0.662, 0.0491];
const COL1: [f64; N] = [1.45, -0.7279, 2.9914, 0.5933, 2.0907, -0.9281, 0.6055, -2.5445, -2.5091, 2.5173, 0.8033, -2.1618, 0.0253, -1.1579];
const COL2: [f64; N] = [-1.293, -0.0021, -1.6607, -0.5514, -1.8574, -0.2388, 0.9266, 0.8873, 0.2378, 1.0245, 1.3992, 1.8839, 1.1521, -0.8011];
const COL3: [f64; N] = [-0.7478, 1.0757, 1.1392, 0.8521, 2.4613, 1.5981, 0.2471, -0.4344, -1.716, 0.1209, 0.6972, -0.6519, -2.4509, -1.9709];
const COL4: [f64; N] = [1.7597, 0.174, 0.6976, 0.2434, 0.8175, -0.3363, 0.1678, -1.2636, -0.1538, 0.5925, 0.3178, -0.1029, -1.0658, -0.1358];
const MEAN_LOGLIK: f64 = -1.649165982956021;
const MEAN_ABS_RESIDUAL: f64 = 1.0417160714285714;
const LOCAL_MARKOV_P: [[f64; 5]; 5] = [
    [f64::NAN, f64::NAN, 0.01641255812285868, 0.778180915310058, 0.5590191334891852],
    [0.10004983476661612, f64::NAN, 0.6933283836342682, 0.31662946657151664, f64::NAN],
    [0.08287766490590338, 0.6933283836342682, f64::NAN, 0.47748838691024126, f64::NAN],
    [f64::NAN, 0.8769010710552717, f64::NAN, f64::NAN, 0.9490312927812511],
    [0.443249039636397, 0.0068015689697342105, 0.043779878082807255, 0.17416746748226894, f64::NAN],
];
const RESIDUAL_P: [[f64; 5]; 5] = [
    [f64::NAN, f64::NAN, 0.005606795082616415, 0.7739995830898754, 0.215872417839122],
    [0.06276007847766504, f64::NAN, 0.37071324089472546, 0.1797800420426447, f64::NAN],
    [0.12009548499894296, 0.4573814717278548, f64::NAN, 0.35228361561150895, f64::NAN],
    [f64::NAN, 0.8661386139848344, f64::NAN, f64::NAN, 0.7568385042774421],
    [0.443249039636397, 0.0068015689697342105, 0.043779878082807255, 0.17416746748226894, f64::NAN],
];

const INTERCEPT: [f64; 5] = [0.3, -0.2, 0.1, 0.0, 0.5];
const SIGMA: [f64; 5] = [0.8, 1.3, 0.9, 1.1, 1.0];
/// Edges `(parent, child)`; coefficient by edge.
const EDGES: [(u32, u32, f64); 5] =
    [(1, 0, 0.7), (4, 1, 1.1), (4, 2, -0.6), (0, 3, 0.5), (2, 3, -0.4)];

fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

fn model() -> CompiledCausalModel {
    let mut graph = Dag::with_variables(5);
    for (p, c, _) in EDGES {
        graph.insert_directed(DenseNodeId::from_raw(p), DenseNodeId::from_raw(c)).unwrap();
    }
    let compiled = CompiledCausalModel::compile(graph).unwrap();
    let slots: Vec<MechanismSlot> = (0..5)
        .map(|i| {
            let coeffs: Vec<f64> = compiled
                .gather_for(id(i))
                .unwrap()
                .parents
                .iter()
                .map(|parent| {
                    EDGES
                        .iter()
                        .find(|(p, c, _)| *p as usize == parent.as_usize() && *c as usize == i)
                        .unwrap()
                        .2
                })
                .collect();
            MechanismSlot::LinearGaussian {
                intercept: INTERCEPT[i],
                coeffs: Arc::from(coeffs),
                sigma: SIGMA[i],
            }
        })
        .collect();
    compiled.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

fn data(rows: usize) -> TabularData {
    let cols: [&[f64]; 5] = [&COL0, &COL1, &COL2, &COL3, &COL4];
    let names = ["x0", "x1", "x2", "x3", "x4"];
    TabularData::from_f64_columns(
        names.iter().zip(cols).map(|(name, col)| (*name, &col[..rows])),
    )
    .unwrap()
}

fn is_parent(p: usize, c: usize) -> bool {
    EDGES.iter().any(|(a, b, _)| *a as usize == p && *b as usize == c)
}

fn is_descendant(of: usize, node: usize) -> bool {
    let mut stack = vec![of];
    let mut seen = [false; 5];
    while let Some(x) = stack.pop() {
        for (a, b, _) in EDGES {
            if a as usize == x && !seen[b as usize] {
                seen[b as usize] = true;
                stack.push(b as usize);
            }
        }
    }
    seen[node]
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: count");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-9 * (1.0 + w.abs()) + 1e-12, "{label}: {g} vs {w}");
    }
}

#[test]
fn the_report_matches_the_high_precision_evaluation() {
    let model = model();
    let report = ModelEvaluator { alpha: 0.05, n_permutations: 5, seed: 3 }
        .evaluate(&model, &data(N), &ExecutionContext::for_tests(1))
        .unwrap();
    assert_eq!(report.zero_probability_rows, 0);
    assert_close("log-likelihood", &[report.in_sample_loglik], &[MEAN_LOGLIK]);
    assert_close("mean absolute residual", &[report.mean_abs_residual], &[MEAN_ABS_RESIDUAL]);

    let order: Vec<usize> = model.node_order.iter().map(|n| n.as_usize()).collect();
    let mut local = Vec::new();
    for (position, &node) in order.iter().enumerate() {
        for &other in &order[..position] {
            if !is_parent(other, node) {
                local.push(LOCAL_MARKOV_P[node][other]);
            }
        }
    }
    let mut residual = Vec::new();
    for node in 0..5 {
        for other in 0..5 {
            if other != node && !is_parent(other, node) && !is_descendant(node, other) {
                residual.push(RESIDUAL_P[node][other]);
            }
        }
    }
    assert!(local.iter().chain(&residual).all(|p| p.is_finite()));
    assert_close("local Markov p", &sorted(report.local_markov_p.to_vec()), &sorted(local.clone()));
    assert_close(
        "residual independence p",
        &sorted(report.residual_independence_p.to_vec()),
        &sorted(residual.clone()),
    );

    // Bonferroni over the union: falsified exactly when the smallest p is below alpha / m.
    let family: Vec<f64> = local.iter().chain(&residual).copied().collect();
    let smallest = family.iter().copied().fold(f64::INFINITY, f64::min);
    assert_eq!(report.falsified, smallest < 0.05 / family.len() as f64);
    for alpha in [0.01, 0.2, 0.8] {
        let again = ModelEvaluator { alpha, n_permutations: 0, seed: 3 }
            .evaluate(&model, &data(N), &ExecutionContext::for_tests(1))
            .unwrap();
        assert_eq!(again.falsified, smallest < alpha / family.len() as f64, "alpha {alpha}");
        assert_eq!(again.permutation_loglik, f64::NEG_INFINITY);
    }
}

#[test]
fn too_few_rows_for_the_conditioning_set_is_an_error_not_a_verdict() {
    // Node 3 has two parents; its local Markov tests need n - 2 - 2 > 0 residual degrees.
    let result = ModelEvaluator::default().evaluate(&model(), &data(4), &ExecutionContext::for_tests(1));
    let message = result.expect_err("df = 0 must not produce a p-value").to_string();
    assert!(message.contains("non-positive residual df"), "{message}");
    let empty = ModelEvaluator::default().evaluate(&model(), &data(0), &ExecutionContext::for_tests(1));
    let message = empty.expect_err("empty data").to_string();
    assert!(message.contains("empty data"), "{message}");
}

#[test]
fn a_permutation_baseline_is_a_mean_over_shuffles_of_the_last_node_only() {
    let model = model();
    let report = ModelEvaluator { alpha: 0.05, n_permutations: 400, seed: 11 }
        .evaluate(&model, &data(N), &ExecutionContext::for_tests(1))
        .unwrap();
    // The last node's outcome is shuffled; the other nodes' terms are unchanged. Exact
    // bounds: every shuffle of the last node's column scores a log-likelihood between the
    // best and the worst assignment of its own values to its own parents' means.
    let last = *model.node_order.last().unwrap();
    let li = last.as_usize();
    let cols: [&[f64]; 5] = [&COL0, &COL1, &COL2, &COL3, &COL4];
    let mean = |r: usize| -> f64 {
        let mut m = INTERCEPT[li];
        for (p, c, w) in EDGES {
            if c as usize == li {
                m += w * cols[p as usize][r];
            }
        }
        m
    };
    let term = |y: f64, m: f64| {
        -(2.0 * std::f64::consts::PI * SIGMA[li] * SIGMA[li]).ln() / 2.0
            - (y - m) * (y - m) / (2.0 * SIGMA[li] * SIGMA[li])
    };
    let others: f64 = (0..5)
        .filter(|&i| i != li)
        .map(|i| {
            (0..N)
                .map(|r| {
                    let mut m = INTERCEPT[i];
                    for (p, c, w) in EDGES {
                        if c as usize == i {
                            m += w * cols[p as usize][r];
                        }
                    }
                    let y = cols[i][r];
                    -(2.0 * std::f64::consts::PI * SIGMA[i] * SIGMA[i]).ln() / 2.0
                        - (y - m) * (y - m) / (2.0 * SIGMA[i] * SIGMA[i])
                })
                .sum::<f64>()
        })
        .sum();
    // Expectation over a uniform shuffle: E[(y_pi(r) - m_r)^2] = mean_r over all pairs.
    let mut pair_sum = 0.0;
    for r in 0..N {
        for s in 0..N {
            pair_sum += term(cols[li][s], mean(r));
        }
    }
    let expected = (others + pair_sum / N as f64) / (5 * N) as f64;
    // 400 shuffles of 14 values: observed deviation 1.3e-3, so 0.01 is several standard errors.
    assert!(
        (report.permutation_loglik - expected).abs() < 0.01,
        "baseline {} vs exact expectation {expected}",
        report.permutation_loglik
    );
}

/* Generator of the constants above (run with `uv run --with mpmath python`):

import random
from mpmath import mp, mpf, betainc, sqrt, log, pi, matrix, lu_solve
mp.dps = 40
random.seed(20261028)
N = 14
# DAG edges parent -> child, dense ids 0..4
parents = {0: [1], 1: [4], 2: [4], 3: [0, 2], 4: []}
intercept = {0: "0.3", 1: "-0.2", 2: "0.1", 3: "0.0", 4: "0.5"}
coef = {0: {1: "0.7"}, 1: {4: "1.1"}, 2: {4: "-0.6"}, 3: {0: "0.5", 2: "-0.4"}, 4: {}}
sigma = {0: "0.8", 1: "1.3", 2: "0.9", 3: "1.1", 4: "1.0"}
# data: roughly from the model, plus a perturbation so some dependencies are violated
cols = {i: [] for i in range(5)}
for r in range(N):
    v = {}
    v[4] = 0.5 + random.gauss(0, 1.0)
    v[1] = -0.2 + 1.1 * v[4] + random.gauss(0, 1.3)
    v[2] = 0.1 - 0.6 * v[4] + random.gauss(0, 0.9)
    v[0] = 0.3 + 0.7 * v[1] + random.gauss(0, 0.8) + 0.5 * v[2]   # misspecified: extra edge 2 -> 0
    v[3] = 0.5 * v[0] - 0.4 * v[2] + random.gauss(0, 1.1)
    for i in range(5):
        cols[i].append(round(v[i], 4))
X = {i: [mpf(str(x)) for x in cols[i]] for i in cols}
def mu(i, r):
    m = mpf(intercept[i])
    for p, c in coef[i].items():
        m += mpf(c) * X[p][r]
    return m
ll = mpf(0); resid = {i: [] for i in range(5)}
for i in range(5):
    s = mpf(sigma[i])
    for r in range(N):
        e = X[i][r] - mu(i, r)
        resid[i].append(e)
        ll += -log(2*pi*s*s)/2 - e*e/(2*s*s)
mean_ll = ll / (5*N)
nonroot = [i for i in range(5) if parents[i]]
mabs = sum(abs(e) for i in nonroot for e in resid[i]) / (len(nonroot)*N)
def ols_resid(y, Z):
    n = len(y)
    k = len(Z)
    A = matrix(n, k+1)
    for r in range(n):
        A[r,0] = 1
        for c in range(k):
            A[r,c+1] = Z[c][r]
    AtA = A.T*A
    Aty = A.T*matrix(y)
    b = lu_solve(AtA, Aty)
    fit = A*b
    return [y[r]-fit[r] for r in range(n)]
def pcorr_p(x, y, Z):
    rx = ols_resid(x, Z); ry = ols_resid(y, Z)
    mx = sum(rx)/len(rx); my = sum(ry)/len(ry)
    sxy = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    sxx = sum((a-mx)**2 for a in rx); syy = sum((b-my)**2 for b in ry)
    r = sxy/sqrt(sxx*syy)
    df = len(x)-2-len(Z)
    t2 = r*r*df/(1-r*r)
    p = betainc(mpf(df)/2, mpf(1)/2, 0, mpf(df)/(df+t2), regularized=True)
    return p
def fmt(p): return "f64::NAN" if p is None else repr(float(p))
lm = [[None]*5 for _ in range(5)]
rs = [[None]*5 for _ in range(5)]
for i in range(5):
    for j in range(5):
        if i != j and j not in parents[i]:
            lm[i][j] = pcorr_p(X[i], X[j], [X[p] for p in parents[i]])
            rs[i][j] = pcorr_p(resid[i], X[j], [])
print("const N: usize = %d;" % N)
for i in range(5):
    print("const COL%d: [f64; N] = [%s];" % (i, ", ".join(repr(x) for x in cols[i])))
print("const MEAN_LOGLIK: f64 = %r;" % float(mean_ll))
print("const MEAN_ABS_RESIDUAL: f64 = %r;" % float(mabs))
print("const LOCAL_MARKOV_P: [[f64; 5]; 5] = [")
for i in range(5): print("    [%s]," % ", ".join(fmt(p) for p in lm[i]))
print("];")
print("const RESIDUAL_P: [[f64; 5]; 5] = [")
for i in range(5): print("    [%s]," % ", ".join(fmt(p) for p in rs[i]))
print("];")
*/
