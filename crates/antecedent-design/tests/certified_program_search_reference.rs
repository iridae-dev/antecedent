//! The certified global bound of `PreparedCertifiedSearch` against truth values found without
//! interval arithmetic (`oracle_data/certified_program_search_reference.py`: a 65,537-point scan
//! with a golden-section polish for one control, a 1201 x 1201 scan with coordinate polish for
//! two).
//!
//! Twenty bounded objectives: a parabola, a cubic with two interior peaks, competing peaks under
//! max and a valley under min, a binding and a non-binding linear budget, a box with a negative
//! lower end, ratios with a separated denominator, a 1e8 offset under a 1e-2 variation, a box of
//! width 1e-7, a fixed control, a kink at the optimum, a decreasing objective, and two-control
//! separable, bilinear, signed, saddle, min-of-controls and two-peak objectives with budgets.
//! For each: the global upper bound never falls below the true maximum, the incumbent is
//! feasible, its enclosure contains the objective at the controls and does not exceed the true
//! maximum, and a `Certified` stop means the true maximum is within the gap of the incumbent.

use antecedent_core::{ArithOp, ExecutionContext, ValueExpr, VariableId};
use antecedent_design::program_search::certified::{
    CertificateOptions, CertificateStop, CertifiedControl, CertifiedRestriction, ObjectiveInterval,
    PreparedCertifiedSearch,
};
use serde_json::Value;

fn build(e: &Value) -> ValueExpr {
    match e["op"].as_str().expect("op") {
        "var" => ValueExpr::var(VariableId::from_raw(e["i"].as_u64().unwrap() as u32)),
        "const" => ValueExpr::constant(e["v"].as_f64().unwrap()),
        "neg" => build(&e["x"]).neg(),
        op => {
            let arith = match op {
                "add" => ArithOp::Add,
                "sub" => ArithOp::Sub,
                "mul" => ArithOp::Mul,
                "div" => ArithOp::Div,
                "min" => ArithOp::Min,
                "max" => ArithOp::Max,
                other => panic!("unknown op {other}"),
            };
            ValueExpr::Arith {
                op: arith,
                lhs: Box::new(build(&e["l"])),
                rhs: Box::new(build(&e["r"])),
            }
        }
    }
}

fn eval(e: &Value, x: &[f64]) -> f64 {
    match e["op"].as_str().unwrap() {
        "var" => x[e["i"].as_u64().unwrap() as usize],
        "const" => e["v"].as_f64().unwrap(),
        "neg" => -eval(&e["x"], x),
        op => {
            let (a, b) = (eval(&e["l"], x), eval(&e["r"], x));
            match op {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "div" => a / b,
                "min" => a.min(b),
                "max" => a.max(b),
                other => panic!("unknown op {other}"),
            }
        }
    }
}

#[test]
fn the_certified_bound_is_sound_and_the_certificate_is_tight_against_independent_truth() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/certified_program_search_reference.json"))
            .expect("frozen reference");
    assert_eq!(cases.len(), 20);
    let ctx = ExecutionContext::for_tests(1);
    let mut certified = 0;
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let controls: Vec<CertifiedControl> = case["controls"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, b)| CertifiedControl {
                variable: VariableId::from_raw(i as u32),
                bounds: ObjectiveInterval {
                    lower: b[0].as_f64().unwrap(),
                    upper: b[1].as_f64().unwrap(),
                },
            })
            .collect();
        let restrictions: Vec<CertifiedRestriction> = case["restrictions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| CertifiedRestriction {
                coefficients: r["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_f64().unwrap())
                    .collect(),
                bound: r["bound"].as_f64().unwrap(),
            })
            .collect();
        let plan = PreparedCertifiedSearch::compile(
            build(&case["expr"]),
            controls.clone(),
            restrictions.clone(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let initial: Vec<f64> =
            case["initial"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let gap = case["gap"].as_f64().unwrap();
        let result = plan
            .execute(
                initial,
                CertificateOptions {
                    absolute_gap: gap,
                    max_splits: case["max_splits"].as_u64().unwrap() as usize,
                },
                &ctx,
            )
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let truth = case["truth"].as_f64().unwrap();
        let error = case["truth_error"].as_f64().unwrap();
        let slack = error + 1e-12 * truth.abs().max(1.0);

        assert!(
            result.global_upper >= truth - slack,
            "{name}: bound {} < truth {truth}",
            result.global_upper
        );
        assert!(
            result.incumbent.lower <= truth + slack,
            "{name}: incumbent {} > truth {truth}",
            result.incumbent.lower
        );
        let value = eval(&case["expr"], &result.controls);
        let scale = value.abs().max(1.0) * 1e-12;
        assert!(
            result.incumbent.lower <= value + scale && value <= result.incumbent.upper + scale,
            "{name}: enclosure [{}, {}] misses the objective {value} at {:?}",
            result.incumbent.lower,
            result.incumbent.upper,
            result.controls
        );
        for (x, c) in result.controls.iter().zip(&controls) {
            assert!(
                c.bounds.lower <= *x && *x <= c.bounds.upper,
                "{name}: control outside its box"
            );
        }
        for r in &restrictions {
            let used: f64 = r.coefficients.iter().zip(&result.controls).map(|(a, b)| a * b).sum();
            assert!(used <= r.bound, "{name}: incumbent violates a restriction");
        }
        assert!(
            result.gap
                >= result.global_upper - result.incumbent.lower - 1e-9 * truth.abs().max(1.0)
        );
        if result.stop == CertificateStop::Certified {
            certified += 1;
            assert!(result.gap <= gap, "{name}: certified gap {} > {gap}", result.gap);
            assert!(
                truth - result.incumbent.lower <= gap + slack,
                "{name}: certified but the incumbent {} is {} below the true maximum {truth}",
                result.incumbent.lower,
                truth - result.incumbent.lower
            );
        }
    }
    assert!(certified >= 15, "only {certified} of 20 certified");
}
