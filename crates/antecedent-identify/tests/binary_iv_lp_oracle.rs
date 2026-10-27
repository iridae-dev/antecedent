//! Sharp binary-IV bounds against an independent linear-programming solve.
//!
//! Cases and optima come from a linear program over the same 16 latent response types
//! (`conformance/identify/binary_iv_lp/reference.py`), a different algorithm from the
//! product's enumeration of basic solutions. Feasible and infeasible laws are both asserted.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_identify::{BinaryIvLaw, binary_iv_ate_bounds};

fn law_of(case: &serde_json::Value) -> BinaryIvLaw {
    let mut cells = [[0.0; 4]; 2];
    for (z, arm) in case["cells"].as_array().unwrap().iter().enumerate() {
        for (cell, value) in arm.as_array().unwrap().iter().enumerate() {
            cells[z][cell] = value.as_f64().unwrap();
        }
    }
    BinaryIvLaw { cells }
}

#[test]
fn matches_independent_linear_program_cases() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/identify/binary_iv_lp/expected.json"
    ))
    .unwrap();
    let atol = fixture["atol"].as_f64().unwrap();
    assert!((atol - 1e-8).abs() < 1e-20, "reviewed tolerance must not be widened");
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 30, "case selection must not shrink");
    let (mut feasible, mut infeasible, mut wide) = (0, 0, 0);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let law = law_of(case);
        if case["feasible"].as_bool().unwrap() {
            feasible += 1;
            let bounds = binary_iv_ate_bounds(law)
                .unwrap_or_else(|error| panic!("{name}: feasible law refused: {error}"));
            let lower = case["lower"].as_f64().unwrap();
            let upper = case["upper"].as_f64().unwrap();
            assert!(
                (bounds.lower - lower).abs() <= atol,
                "{name}: lower {} vs {lower}",
                bounds.lower
            );
            assert!(
                (bounds.upper - upper).abs() <= atol,
                "{name}: upper {} vs {upper}",
                bounds.upper
            );
            if upper - lower > 0.05 {
                wide += 1;
            }
        } else {
            infeasible += 1;
            let error = binary_iv_ate_bounds(law).expect_err(name);
            assert!(
                error.to_string().contains("instrumental inequality"),
                "{name}: refusal must name the instrumental inequality: {error}"
            );
        }
    }
    assert!(feasible >= 20 && infeasible >= 2 && wide >= 10);
}
