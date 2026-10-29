//! Intervention-program expression semantics against an independent interpreter.
//!
//! `oracle_data/program_expression_reference.py` (frozen in the JSON beside it) generates 220
//! random typed expression trees over two input columns (arithmetic including division, minimum
//! and maximum, clipping, comparisons, boolean connectives, conditionals whose unselected branch
//! may be non-finite, indicators) and evaluates each on 24 rows of awkward inputs (signed zeros,
//! subnormals, values near the overflow threshold, ties) with the documented definitions in
//! IEEE binary64. This test lowers the same trees with `NodeProgram::compile_columns` and runs
//! the shared postfix executor row by row: every finite result must equal the reference value and
//! every row the reference refuses (a non-finite final value) must be refused by the executor.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ValueExpr, VariableId};
use antecedent_model::NodeProgram;
use serde_json::Value;

fn build(e: &Value) -> ValueExpr {
    let a = e.as_array().unwrap();
    let arg = |i: usize| build(&a[i]);
    match a[0].as_str().unwrap() {
        "const" => ValueExpr::constant(a[1].as_f64().unwrap()),
        "var" => ValueExpr::var(VariableId::from_raw(a[1].as_u64().unwrap() as u32)),
        "neg" => arg(1).neg(),
        "add" => arg(1).add(arg(2)),
        "sub" => arg(1).sub(arg(2)),
        "mul" => arg(1).mul(arg(2)),
        "div" => arg(1).div(arg(2)),
        "min" => arg(1).min(arg(2)),
        "max" => arg(1).max(arg(2)),
        "clip" => arg(1).clip(arg(2), arg(3)),
        "lt" => arg(1).lt(arg(2)),
        "le" => arg(1).le(arg(2)),
        "gt" => arg(1).gt(arg(2)),
        "ge" => arg(1).ge(arg(2)),
        "eq" => arg(1).equals(arg(2)),
        "ne" => arg(1).not_equals(arg(2)),
        "not" => arg(1).not(),
        "and" => arg(1).and(arg(2)),
        "or" => arg(1).or(arg(2)),
        "if" => arg(1).if_then_else(arg(2), arg(3)),
        "ind" => arg(1).indicator(),
        other => panic!("unknown node {other}"),
    }
}

#[test]
fn the_executor_agrees_with_an_independent_interpreter_on_awkward_inputs() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/program_expression_reference.json"))
            .unwrap();
    let axis = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let (mut finite, mut refused) = (0_usize, 0_usize);
    for case in cases.as_array().unwrap() {
        let expr = build(&case["expr"]);
        let program = NodeProgram::compile_columns(&axis, &expr)
            .unwrap_or_else(|e| panic!("{}: {e:?}", case["expr"]));
        for row in case["rows"].as_array().unwrap() {
            let input: Vec<f64> =
                row["in"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let got = program.evaluate_columns(&input, None);
            match row["out"].as_f64() {
                Some(want) => {
                    let got = got.unwrap_or_else(|e| {
                        panic!(
                            "{} on {input:?}: refused ({e}) but the reference is {want}",
                            case["expr"]
                        )
                    });
                    assert!(got == want, "{} on {input:?}: {got} vs {want}", case["expr"]);
                    finite += 1;
                }
                None => {
                    let error = got.err().unwrap_or_else(|| {
                        panic!(
                            "{} on {input:?}: reference refuses but a value came back",
                            case["expr"]
                        )
                    });
                    // The program is the dense node 2 at row 0; the non-finite value is NaN or
                    // an infinity, exactly where the reference holds no finite value.
                    let text = error.to_string();
                    let value = text
                        .strip_prefix(
                            "numerical error: intervention program for dense node 2 is not finite \
                             at row 0 (",
                        )
                        .and_then(|rest| {
                            rest.strip_suffix("); guard a division or restrict the clip bounds")
                        })
                        .unwrap_or_else(|| panic!("{} on {input:?}: {text}", case["expr"]));
                    assert!(
                        ["NaN", "inf", "-inf"].contains(&value),
                        "{} on {input:?}: {text}",
                        case["expr"]
                    );
                    refused += 1;
                }
            }
        }
    }
    assert!(finite > 3000 && refused > 500, "{finite} finite, {refused} refused");
}
