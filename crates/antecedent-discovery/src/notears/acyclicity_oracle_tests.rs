//! `h(W)` and its gradient against a 40-digit matrix exponential.
//!
//! `tests/oracle_data/notears_reference.py` evaluates `tr(exp(W o W)) - d` and
//! `exp(W o W)^T o 2W` with mpmath's matrix exponential for an acyclic matrix, tiny and unit
//! cycles, a dense matrix, entries of size 5 (1-norm of `W o W` in the hundreds) and twelve
//! variables.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde_json::Value;

use super::{AcyclicityWorkspace, grad_h, h_of_w};

fn number(v: &Value) -> f64 {
    match v {
        Value::String(s) => s.parse().unwrap(),
        other => other.as_f64().unwrap(),
    }
}

#[test]
fn constraint_and_gradient_match_the_multiprecision_exponential() {
    let doc: Value =
        serde_json::from_str(include_str!("../../tests/oracle_data/notears_reference.json"))
            .unwrap();
    for case in doc["acyclicity"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let d = usize::try_from(case["d"].as_u64().unwrap()).unwrap();
        let w: Vec<f64> = case["w"].as_array().unwrap().iter().map(number).collect();
        let mut ws = AcyclicityWorkspace::default();
        let h = h_of_w(&w, d, &mut ws).unwrap();
        let want = number(&case["h"]);
        // tr(exp(A)) - d subtracts two numbers of size d + h: its rounding error is a few ulps of
        // (d + h), absolute, however small the difference is.
        let tolerance = 64.0 * f64::EPSILON * (d as f64 + want.abs());
        assert!((h - want).abs() <= tolerance, "{name}: h {h} want {want} tol {tolerance}");
        grad_h(&w, d, &mut ws);
        let grad: Vec<f64> = case["grad"].as_array().unwrap().iter().map(number).collect();
        let scale = grad.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
        for (k, want) in grad.iter().enumerate() {
            assert!(
                (ws.grad[k] - want).abs() <= 1e-12 * scale.max(1e-300) + 1e-15,
                "{name} gradient[{k}]: {} want {want}",
                ws.grad[k]
            );
        }
    }
}
