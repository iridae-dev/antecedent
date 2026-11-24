//! Linear-program optima and vertex sets against exact rational brute force.
//!
//! The frozen values in `oracle_data/linear_program_reference.json` come from
//! `oracle_data/linear_program_reference.py`: every choice of `n` hyperplanes among the
//! constraints and finite bounds is solved in exact rational arithmetic, feasible solutions are
//! the vertices, and the optimum is the best vertex (infeasible when there is none). The programs
//! are hand-written cases (production, Beale's cycling example, covering, equalities with free
//! variables, degenerate and redundant rows, fixed bounds, contradictions) and 36 programs from
//! a fixed-seed generator with small integer coefficients in dimensions 2 to 5.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::linear_program::{
    ConstraintSense, LinearConstraint, LinearProgram, LpOptions, LpOutcome, solve, verify, vertices,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/linear_program_reference.json")).unwrap()
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// `null` is an absent bound.
fn bounds(v: &Value, absent: f64) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(absent)).collect()
}

fn constraints(v: &Value) -> Vec<LinearConstraint> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| LinearConstraint {
            coefficients: numbers(&c["coefficients"]),
            sense: match c["sense"].as_str().unwrap() {
                "at_most" => ConstraintSense::AtMost,
                "at_least" => ConstraintSense::AtLeast,
                "equal" => ConstraintSense::Equal,
                other => panic!("unknown sense {other}"),
            },
            rhs: c["rhs"].as_f64().unwrap(),
        })
        .collect()
}

#[test]
fn optima_match_exact_vertex_enumeration() {
    let data = fixture();
    let programs = data["programs"].as_array().unwrap();
    assert!(programs.len() >= 43);
    for case in programs {
        let name = case["name"].as_str().unwrap();
        let program = LinearProgram {
            objective: numbers(&case["objective"]),
            constraints: constraints(&case["constraints"]),
            lower: bounds(&case["lower"], f64::NEG_INFINITY),
            upper: bounds(&case["upper"], f64::INFINITY),
        };
        let outcome = solve(&program, LpOptions::default()).unwrap();
        verify(&program, &outcome, LpOptions::default()).unwrap();
        match (case["status"].as_str().unwrap(), &outcome) {
            ("optimal", LpOutcome::Optimal(solution)) => {
                let want = case["objective_value"].as_f64().unwrap();
                assert!(
                    (solution.objective - want).abs() <= 1e-9 * (1.0 + want.abs()),
                    "{name}: objective {} want {want} ({})",
                    solution.objective,
                    case["objective_exact"]
                );
            }
            ("infeasible", LpOutcome::Infeasible(_)) => {}
            (status, other) => panic!("{name}: expected {status}, got {other:?}"),
        }
    }
}

#[test]
fn vertex_sets_match_exact_enumeration() {
    let data = fixture();
    for case in data["polytopes"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want: Vec<Vec<f64>> =
            case["vertices"].as_array().unwrap().iter().map(numbers).collect();
        let got = vertices(
            &constraints(&case["constraints"]),
            &bounds(&case["lower"], f64::NEG_INFINITY),
            &bounds(&case["upper"], f64::INFINITY),
            1_000_000,
            LpOptions::default(),
        )
        .unwrap();
        assert_eq!(got.len(), want.len(), "{name}: vertex count");
        for expected in &want {
            assert!(
                got.iter().any(|v| v.iter().zip(expected).all(|(a, b)| (a - b).abs() <= 1e-9)),
                "{name}: missing vertex {expected:?}"
            );
        }
        for w in got.windows(2) {
            let order = w[0].iter().zip(&w[1]).map(|(a, b)| a.total_cmp(b)).find(|o| o.is_ne());
            assert!(order.is_some_and(std::cmp::Ordering::is_lt), "{name}: not sorted");
        }
    }
}
