//! The transported additive path-specific contrast against an exact-rational structural
//! reference.
//!
//! `oracle_data/transported_additive_reference.py` (frozen in the JSON beside it) evaluates
//! each world with a recursive structural evaluator in which the edge `A -> c` carries the
//! treated or the control value, gives every mechanism an explicit three-point noise law, and
//! sums the contrast over the product of target covariate points and all noise points, feeding
//! one noise draw to both worlds. The product never sees the noise; it relies on the contrast
//! being free of it, so agreement with the reference checks that claim on covariate-dependent
//! intercepts, slopes and coefficients, a two-mediator chain with a direct edge, covariate
//! offsets near 1e6, target weights 2^-20 apart and a treatment contrast offset by 1e6. The
//! selection-on-mediator cases check the refusal witness: its node, parent and perturbed
//! coefficient and the two target contrasts it reports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_counterfactual::transported_gate::{PopulationRole, RegimeFactorKey};
use antecedent_counterfactual::transported_path_specific::{
    AdditiveLinearScm, Affine, CovariateLaw, DeclaredAssumptions, EdgeAssignment, LinearMechanism,
    PathSpecificQuery, PathSpecificSelectionDiagram, SelectionNode, TransportedPathSpecificInput,
    evaluate_transported_path_specific,
};
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/transported_additive_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn affine(v: &Value) -> Affine {
    let mut slopes: Vec<(String, f64)> =
        v["slopes"].as_object().unwrap().iter().map(|(k, s)| (k.clone(), f(s))).collect();
    slopes.sort_by(|a, b| a.0.cmp(&b.0));
    Affine { constant: f(&v["constant"]), covariate_slopes: slopes }
}

fn model(v: &Value) -> AdditiveLinearScm {
    AdditiveLinearScm {
        treatment: v["treatment"].as_str().unwrap().to_owned(),
        covariates: v["covariates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().to_owned())
            .collect(),
        mechanisms: v["mechanisms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m["name"].as_str().unwrap().to_owned(),
                    LinearMechanism {
                        intercept: affine(&m["intercept"]),
                        parents: m["parents"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|p| (p[0].as_str().unwrap().to_owned(), affine(&p[1])))
                            .collect(),
                    },
                )
            })
            .collect(),
    }
}

fn law(v: &Value) -> CovariateLaw {
    CovariateLaw {
        points: v
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let point: BTreeMap<String, f64> =
                    p[0].as_object().unwrap().iter().map(|(k, x)| (k.clone(), f(x))).collect();
                (point, f(&p[1]))
            })
            .collect(),
    }
}

fn query(v: &Value) -> PathSpecificQuery {
    let set = |k: &str| {
        EdgeAssignment::treated_on(
            v[k].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()),
        )
    };
    PathSpecificQuery {
        outcome: v["outcome"].as_str().unwrap().to_owned(),
        treated_value: f(&v["a1"]),
        control_value: f(&v["a0"]),
        plus: set("plus"),
        minus: set("minus"),
    }
}

fn evidence() -> BTreeMap<RegimeFactorKey, String> {
    [PopulationRole::Source, PopulationRole::Target]
        .into_iter()
        .map(|role| (RegimeFactorKey { role, regime: "observational".to_owned() }, "e".to_owned()))
        .collect()
}

const DECLARED: DeclaredAssumptions = DeclaredAssumptions {
    additive_noise: true,
    noise_laws_shared: true,
    cross_world_independence: true,
};

fn diagram(case: &Value) -> PathSpecificSelectionDiagram {
    match case.get("selection_node").and_then(Value::as_str) {
        Some(node) => PathSpecificSelectionDiagram {
            selections: vec![SelectionNode { label: "S".to_owned(), target: node.to_owned() }],
        },
        None => PathSpecificSelectionDiagram::default(),
    }
}

fn tol(expected: f64) -> f64 {
    // a few ulps of the largest intermediate: the covariate offsets enter with weight ~ 1e6
    1e-12 * expected.abs().max(1.0) + 64.0 * f64::EPSILON * 1.0e6
}

#[test]
fn product_matches_the_noise_summing_structural_reference() {
    let mut checked = 0;
    for case in cases() {
        if case.get("selection_node").is_some() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let (m, s, t, q) = (
            model(&case["model"]),
            law(&case["source"]),
            law(&case["target"]),
            query(&case["query"]),
        );
        let d = diagram(&case);
        let input = TransportedPathSpecificInput {
            model: &m,
            diagram: &d,
            source_law: &s,
            target_law: &t,
            query: &q,
            assumptions: DECLARED,
            supplied_factors: &evidence(),
        };
        let r = evaluate_transported_path_specific(&input).unwrap();
        let (wt, ws) = (f(&case["target_contrast"]), f(&case["source_contrast"]));
        // the offset cases carry ~1e11-sized contrasts; compare relatively there
        let rel = |w: f64| if w.abs() > 1.0e6 { 1e-13 * w.abs() } else { tol(w) };
        assert!((r.target_contrast - wt).abs() <= rel(wt), "{name}: {} vs {wt}", r.target_contrast);
        assert!((r.source_contrast - ws).abs() <= rel(ws), "{name}: {} vs {ws}", r.source_contrast);
        assert_eq!(r.unit_contrasts.len(), t.points.len(), "{name}");
        eprintln!(
            "{name}: target err {:e} source err {:e} (contrast {wt})",
            (r.target_contrast - wt).abs(),
            (r.source_contrast - ws).abs()
        );
        checked += 1;
    }
    assert!(checked >= 10);
}

#[test]
fn target_and_source_laws_are_not_interchangeable() {
    // the answer must move when only the target law moves (no silently ignored law)
    let case = cases().into_iter().find(|c| c["name"] == "diamond_nie").unwrap();
    let (m, s, t, q) =
        (model(&case["model"]), law(&case["source"]), law(&case["target"]), query(&case["query"]));
    let d = PathSpecificSelectionDiagram::default();
    let run = |target: &CovariateLaw| {
        evaluate_transported_path_specific(&TransportedPathSpecificInput {
            model: &m,
            diagram: &d,
            source_law: &s,
            target_law: target,
            query: &q,
            assumptions: DECLARED,
            supplied_factors: &evidence(),
        })
        .unwrap()
    };
    let base = run(&t);
    let mut other = t.clone();
    other.points = vec![(other.points[0].0.clone(), 0.5), (other.points[1].0.clone(), 0.5)];
    let moved = run(&other);
    assert!((base.target_contrast - moved.target_contrast).abs() > 1e-3);
    assert_eq!(base.source_contrast, moved.source_contrast);
}

#[test]
fn selection_on_a_mediator_returns_the_reference_witness_or_no_witness() {
    for case in cases() {
        let Some(node) = case.get("selection_node").and_then(Value::as_str) else { continue };
        let name = case["name"].as_str().unwrap();
        let (m, s, t, q) = (
            model(&case["model"]),
            law(&case["source"]),
            law(&case["target"]),
            query(&case["query"]),
        );
        let d = diagram(&case);
        let err = evaluate_transported_path_specific(&TransportedPathSpecificInput {
            model: &m,
            diagram: &d,
            source_law: &s,
            target_law: &t,
            query: &q,
            assumptions: DECLARED,
            supplied_factors: &evidence(),
        })
        .expect_err("selection on a mediator is outside the class");
        assert_eq!(err.refusal.detail, "transported_counterfactual.selection_on_mechanism");
        match case["witness"].as_object() {
            Some(w) => {
                assert_eq!(err.refusal.code, "transport_proven_non_transportable", "{name}");
                let got = err.witness.as_ref().expect("witness");
                assert_eq!(got.selected_node, node);
                assert_eq!(got.selected_node, w["node"].as_str().unwrap());
                assert_eq!(got.perturbed_parent, w["parent"].as_str().unwrap());
                assert_eq!(got.perturbed_slope_covariate.as_deref(), w["slope"].as_str());
                assert!((got.target_contrast_a - f(&w["base"])).abs() < 1e-12, "{name}");
                assert!((got.target_contrast_b - f(&w["moved"])).abs() < 1e-12, "{name}");
                assert!((got.source_contrast - f(&case["source_contrast"])).abs() < 1e-12, "{name}");
            }
            None => {
                assert_eq!(err.refusal.code, "cell_not_licensed", "{name}");
                assert!(err.witness.is_none(), "{name}");
            }
        }
    }
}
