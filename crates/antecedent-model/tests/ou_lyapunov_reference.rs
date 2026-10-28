//! The Ornstein-Uhlenbeck Lyapunov solve against an exact rational reference.
//!
//! `oracle_data/ou_lyapunov_reference.py` (frozen in the JSON beside it) solves
//! `M X + X M^T = -Q` for 8 x 8 drifts in exact `Fraction` arithmetic: a real spectrum,
//! complex-conjugate pairs, a non-normal drift and an indefinite direction. The product's
//! Schur-based `OrnsteinUhlenbeck::lyapunov_direction` must agree with the exact solution to
//! a small multiple of the rounding of the largest entry.

use antecedent_core::ExecutionContext;
use antecedent_model::continuous_cyclic::OrnsteinUhlenbeck;
use serde_json::Value;

fn matrix(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect()
}

#[test]
fn schur_lyapunov_solve_matches_the_exact_rational_solution() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/ou_lyapunov_reference.json")).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;
        let drift = matrix(&case["drift"]);
        let direction = matrix(&case["direction"]);
        let exact = matrix(&case["solution"]);
        let process = OrnsteinUhlenbeck {
            drift,
            center: vec![0.0; n],
            diffusion: (0..n * n).map(|k| if k / n == k % n { 1.0 } else { 0.0 }).collect(),
        };
        let solved = process.lyapunov_direction(&direction, &ctx).unwrap();
        let scale = exact.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let error = solved.iter().zip(&exact).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("{name}: max abs error {error:e} (scale {scale:e}, relative {:e})", error / scale);
        assert!(error <= 1e-12 * scale.max(1.0), "{name}: error {error:e} at scale {scale:e}");
    }
}
