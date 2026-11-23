//! Linear forms of declared value models against an exact-rational reference
//! (`oracle_data/preference_model_reference.py`, `fractions.Fraction`, dyadic inputs).
//!
//! The reference writes each model as a closed-form function and recovers the fixed part and
//! the per-parameter terms by evaluating at unit parameter vectors; the product walks the
//! expression symbolically. Both are compared at fixed attribute and parameter points: rates,
//! a three-segment curve with one slope per segment, an excess and a shortfall kink, and an
//! interaction with a minimum.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_design::preference_model::{
    Attribute, Curve, PreferenceExpr as P, PreferenceParameter, PreferenceSet, ValueModel, shape,
};
use serde_json::Value;

fn model(attributes: &[&str], parameters: &[&str], value: P) -> ValueModel {
    ValueModel {
        attributes: attributes
            .iter()
            .map(|id| Attribute { id: (*id).into(), units: "units".into() })
            .collect(),
        preferences: PreferenceSet {
            parameters: parameters
                .iter()
                .map(|id| PreferenceParameter {
                    id: (*id).into(),
                    units: "usd/unit".into(),
                    lower: -3.0,
                    upper: 3.0,
                    assumption_id: format!("{id}-judgment"),
                })
                .collect(),
            relations: vec![],
        },
        value,
        utility_units: "usd".into(),
    }
}

fn declared(name: &str) -> ValueModel {
    match name {
        "rates" => model(
            &["benefit", "harm", "cost"],
            &["w_benefit", "w_harm"],
            P::difference(
                P::difference(
                    P::product(P::Parameter(0), shape::rate(0)),
                    P::product(P::Parameter(1), shape::rate(1)),
                ),
                shape::rate(2),
            ),
        ),
        "curve" => {
            let curve = Curve {
                attribute: 0,
                breakpoints: vec![0.0, 1.0, 2.5, 4.0],
                slopes: vec![0, 1, 2],
            };
            assert_eq!(curve.validate(), Ok(()));
            model(
                &["dose", "cost"],
                &["s0", "s1", "s2", "w_cost"],
                P::difference(curve.expr(), P::product(P::Parameter(3), shape::rate(1))),
            )
        }
        "excess" => model(
            &["benefit", "wait"],
            &["w_benefit", "w_wait"],
            P::difference(
                P::difference(
                    P::product(P::Parameter(0), P::sum(shape::rate(0), P::Const(0.5))),
                    P::product(P::Parameter(1), shape::excess(1, 3.0)),
                ),
                P::product(shape::shortfall(0, 1.0), P::Const(0.25)),
            ),
        ),
        "interaction" => model(
            &["benefit", "adherence"],
            &["w_joint", "w_both"],
            P::sum(
                P::difference(
                    P::sum(
                        P::product(P::Parameter(0), shape::interaction(0, 1)),
                        P::product(P::Parameter(1), shape::both(0, 1)),
                    ),
                    P::product(P::Parameter(0), shape::rate(1)),
                ),
                P::product(P::Const(2.0), P::Parameter(1)),
            ),
        ),
        other => panic!("no declared model named {other}"),
    }
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
        "{what}: got {got}, reference {want}"
    );
}

#[test]
fn linear_forms_match_the_exact_rational_reference() {
    let root: Value =
        serde_json::from_str(include_str!("oracle_data/preference_model_reference.json"))
            .expect("frozen reference");
    let models = root["models"].as_array().expect("models");
    assert_eq!(models.len(), 4);
    let mut compared = 0;
    for entry in models {
        let name = entry["name"].as_str().expect("name");
        let m = declared(name);
        assert_eq!(m.validate(), Ok(()), "{name}");
        assert!(m.shape_parameters().is_empty(), "{name}");
        let form = m.linear_form().expect("affine in the parameters");
        for point in entry["points"].as_array().expect("points") {
            let x = floats(&point["attributes"]);
            let theta = floats(&point["parameters"]);
            let terms = floats(&point["terms"]);
            assert_eq!(terms.len(), form.terms.len(), "{name}");
            close(form.fixed.evaluate(&x, &[]).unwrap(), point["fixed"].as_f64().unwrap(), name);
            for (j, (term, want)) in form.terms.iter().zip(&terms).enumerate() {
                close(term.evaluate(&x, &[]).unwrap(), *want, &format!("{name} term {j}"));
            }
            let value = point["value"].as_f64().unwrap();
            close(m.value_at(&x, &theta).unwrap(), value, name);
            compared += 1;
        }
    }
    assert_eq!(compared, 32);
}
