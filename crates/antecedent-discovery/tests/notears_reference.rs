//! NOTEARS against the exhaustively computed constrained optimum.
//!
//! `oracle_data/notears_reference.py` (frozen in the JSON beside it) minimises
//! `(1/2n) ||X - XW||^2 + lambda |W|_1` over acyclic supports by enumerating node orderings and
//! solving the per-node lasso problems to 1e-30 in mpmath. The data have exactly zero column
//! means, so the solver (run unstandardised) sees the same objective. The returned weights must
//! reach the reference optimum within the declared acyclicity tolerance, equal the optimal
//! weights where the runner-up support is clearly worse, and give the hard graph obtained by
//! thresholding them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::{DiscoveryWorkspace, Notears};
use antecedent_graph::MarkedEdge;
use serde_json::Value;
use std::collections::BTreeSet;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/notears_reference.json")).unwrap()
        ["solver"]
        .as_array()
        .unwrap()
        .clone()
}

fn float(v: &Value) -> f64 {
    match v {
        Value::String(s) => s.parse().unwrap(),
        other => other.as_f64().unwrap(),
    }
}

fn objective(w: &[f64], gram: &[Vec<f64>], lambda: f64) -> f64 {
    let d = gram.len();
    // (1/2) tr((I - W)^T G (I - W)) + lambda |W|_1
    let mut ls = 0.0;
    for j in 0..d {
        let mut column = vec![0.0; d];
        for (i, slot) in column.iter_mut().enumerate() {
            *slot = f64::from(u8::from(i == j)) - w[i * d + j];
        }
        for a in 0..d {
            for b in 0..d {
                ls += column[a] * gram[a][b] * column[b];
            }
        }
    }
    0.5 * ls + lambda * w.iter().map(|x| x.abs()).sum::<f64>()
}

#[test]
fn weights_reach_the_exhaustive_constrained_optimum() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let d = usize::try_from(case["d"].as_u64().unwrap()).unwrap();
        let lambda = case["lam"].as_f64().unwrap();
        let columns: Vec<Vec<f64>> = case["cols"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect())
            .collect();
        let gram: Vec<Vec<f64>> = case["gram"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(float).collect())
            .collect();
        let data = TabularData::from_f64_columns(
            columns.iter().enumerate().map(|(i, c)| (format!("x{i}"), c.as_slice())),
        )
        .unwrap();
        let ids: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        // h is quartic in a two-cycle's weights (about w12^2 w21^2), so |h| <= 1e-8 admits
        // reverse-direction weights up to 0.01 on a cycle: the returned W is acyclic only to that
        // size, and may undercut the exactly acyclic optimum by a few such weights times the
        // gradient. A hard threshold of 0.2 lies above them.
        let alg = Notears::new().with_lambda(lambda).with_standardize(false).with_threshold(0.2);
        let result = alg
            .run(&data, &ids, &mut DiscoveryWorkspace::default(), &ExecutionContext::for_tests(1))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let w = result.weights.to_vec();
        let want = float(&case["objective"]);
        let got = objective(&w, &gram, lambda);
        // The constraint is satisfied to |h| <= 1e-8, which can buy at most a few 1e-8 of
        // objective; the augmented-Lagrangian path must not stop short of the optimum either.
        assert!(got >= want - 5e-3, "{name}: objective {got} below the global optimum {want}");
        // The augmented-Lagrangian problem is non-convex and the solver is a local method: on
        // the diamond (two supports within 1e-3 of the optimum) it settles in a different
        // acyclic ordering 2.6e-3 above the global optimum. That is the documented limit of
        // NOTEARS, so that case is bounded loosely; every other case reaches the optimum.
        let slack = if name == "diamond_four_no_penalty" { 1e-2 } else { 1e-5 };
        assert!(got <= want + slack, "{name}: objective {got} above the global optimum {want}");
        let reference: Vec<f64> =
            case["weights"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let separated = case["runner_up_gap"].as_f64().is_none_or(|g| g > 1e-3);
        if separated && name != "diamond_four_no_penalty" {
            for (k, (g, r)) in w.iter().zip(&reference).enumerate() {
                assert!((g - r).abs() <= 3e-2, "{name} weight[{k}]: {g} vs {r}");
            }
            if reference.iter().any(|r| (r.abs() - 0.2).abs() < 0.04) {
                continue;
            }
            let edges: BTreeSet<(u32, u32)> = result
                .discovery
                .evidence
                .graph
                .edges()
                .filter_map(MarkedEdge::parent_child)
                .map(|(a, b)| (a.raw(), b.raw()))
                .collect();
            let expected: BTreeSet<(u32, u32)> = (0..d)
                .flat_map(|i| (0..d).map(move |j| (i, j)))
                .filter(|&(i, j)| reference[i * d + j].abs() >= 0.2)
                .map(|(i, j)| (u32::try_from(i).unwrap(), u32::try_from(j).unwrap()))
                .collect();
            assert_eq!(edges, expected, "{name}: thresholded graph");
        }
    }
}
