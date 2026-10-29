//! The fixed two-slice temporal counterfactual against an exact-rational forward-simulation
//! reference.
//!
//! `oracle_data/temporal_fixed_population_reference.py` (frozen in the JSON beside it) generates
//! each unit's history by forward simulation of the structural equations from chosen noise,
//! rounds it to doubles, abduces the noise from those doubles by exact rational residuals,
//! replays both named action histories with that same noise and averages exactly. The product
//! must reproduce the per-unit outcomes and the mean contrast to a few ulps of the magnitude of
//! the terms (large-offset case included), must recover the abduced residual through its
//! support check (a residual equal to the half-width is accepted, one slightly larger refuses),
//! and must not depend on the order in which units are supplied.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use TemporalNode::{Action0, Action1, Covariate0, Covariate1, Outcome};
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::temporal_cross_world::general::{
    GeneralActionHistory, GeneralFactualHistory, GeneralMechanismFit, GeneralNodeMechanism,
    GeneralTemporalSpec, evaluate_general_temporal_counterfactual,
};
use antecedent_counterfactual::temporal_cross_world::staged::{
    GeneralTemporalExecutor, TemporalExecutor,
};
use antecedent_counterfactual::temporal_cross_world::{
    FactualUnitHistory, HistoryId, NamedActionHistory, NodeMechanism, SnapshotId,
    TemporalCounterfactualError, TemporalCounterfactualSpec, TemporalGraph, TemporalMechanismFit,
    TemporalNode, UnitId, evaluate_temporal_counterfactual,
};
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "oracle_data/temporal_fixed_population_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

fn spec(case: &Value) -> TemporalCounterfactualSpec {
    let c = floats(&case["intercepts"]);
    let b = floats(&case["coefs"]);
    let hw = case["halfwidth"].as_f64();
    let graph = TemporalGraph {
        horizon: 2,
        edges: vec![
            (Covariate0, Covariate1),
            (Action0, Covariate1),
            (Covariate0, Outcome),
            (Action0, Outcome),
            (Covariate1, Outcome),
            (Action1, Outcome),
        ],
        latent_confounding: false,
    };
    let fit = TemporalMechanismFit {
        fit_id: "reference-fit".to_owned(),
        mechanisms: vec![
            NodeMechanism {
                node: Covariate0,
                intercept: c[0],
                parent_coefficients: vec![],
                noise_halfwidth: hw,
            },
            NodeMechanism {
                node: Covariate1,
                intercept: c[1],
                parent_coefficients: vec![(Covariate0, b[0]), (Action0, b[1])],
                noise_halfwidth: hw,
            },
            NodeMechanism {
                node: Outcome,
                intercept: c[2],
                parent_coefficients: vec![
                    (Covariate0, b[2]),
                    (Action0, b[3]),
                    (Covariate1, b[4]),
                    (Action1, b[5]),
                ],
                noise_halfwidth: hw,
            },
        ],
    };
    let units = case["units"].as_array().unwrap();
    let ids: Vec<UnitId> = (0..units.len()).map(|i| UnitId::new(format!("u{i:02}"))).collect();
    let factual = units
        .iter()
        .zip(&ids)
        .enumerate()
        .map(|(i, (u, id))| {
            let v = floats(&u["factual"]);
            FactualUnitHistory {
                unit: id.clone(),
                history: HistoryId::new(format!("h{i:02}")),
                times: [0, 1],
                values: [v[0], v[1], v[2], v[3], v[4]],
            }
        })
        .collect();
    let world = |name: &str, key: &str| {
        let a = floats(&case[key]);
        NamedActionHistory {
            name: name.to_owned(),
            times: [0, 1],
            actions: [a[0], a[1]],
            units: ids.clone(),
        }
    };
    TemporalCounterfactualSpec {
        graph,
        fit,
        snapshot: SnapshotId::new("snap"),
        factual,
        plus: world("plus", "plus"),
        minus: world("minus", "minus"),
    }
}

fn run(
    s: &TemporalCounterfactualSpec,
) -> Result<
    antecedent_counterfactual::temporal_cross_world::TemporalCounterfactualResult,
    TemporalCounterfactualError,
> {
    evaluate_temporal_counterfactual(s, &ExecutionContext::for_tests(1))
}

fn scale(case: &Value) -> f64 {
    let mut s = 1.0f64;
    for u in case["units"].as_array().unwrap() {
        for key in ["factual", "residuals"] {
            for v in floats(&u[key]) {
                s = s.max(v.abs());
            }
        }
    }
    for v in floats(&case["intercepts"]).into_iter().chain(floats(&case["coefs"])) {
        s = s.max(v.abs());
    }
    s * 8.0
}

#[test]
fn product_matches_exact_abduction_action_prediction() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let s = spec(&case);
        let r = run(&s).unwrap();
        let tol = 64.0 * f64::EPSILON * scale(&case);
        let mut worst = 0.0f64;
        for (got, want) in r.units.iter().zip(case["units"].as_array().unwrap()) {
            for (g, w, what) in [
                (got.plus_outcome, f(&want["plus"]), "plus"),
                (got.minus_outcome, f(&want["minus"]), "minus"),
                (got.factual_outcome, floats(&want["factual"])[4], "factual"),
            ] {
                worst = worst.max((g - w).abs());
                assert!((g - w).abs() <= tol, "{name} {what}: {g} vs {w} (tol {tol})");
            }
        }
        for (g, key) in [
            (r.mean_plus, "mean_plus"),
            (r.mean_minus, "mean_minus"),
            (r.mean_contrast, "mean_contrast"),
        ] {
            let w = f(&case[key]);
            worst = worst.max((g - w).abs());
            assert!((g - w).abs() <= tol, "{name} {key}: {g} vs {w} (tol {tol})");
        }
        assert!(r.receipt.is_self_consistent(), "{name}");
        eprintln!("{name}: worst abs error {worst:e} (tol {tol:e})");
    }
}

#[test]
fn identical_histories_have_exactly_zero_contrast() {
    let case = cases().into_iter().find(|c| c["name"] == "identical_histories").unwrap();
    let r = run(&spec(&case)).unwrap();
    assert_eq!(r.mean_contrast, 0.0);
    for u in &r.units {
        assert_eq!(u.plus_outcome, u.minus_outcome);
    }
}

#[test]
fn unit_order_does_not_change_the_answer() {
    let case = cases().into_iter().find(|c| c["name"] == "non_dyadic_seven_units").unwrap();
    let s = spec(&case);
    let mut rev = s.clone();
    rev.factual.reverse();
    rev.plus.units.reverse();
    rev.minus.units.reverse();
    let (a, b) = (run(&s).unwrap(), run(&rev).unwrap());
    assert_eq!(a.units, b.units);
    assert_eq!(a.mean_contrast.to_bits(), b.mean_contrast.to_bits());
    assert_eq!(a.receipt.digest, b.receipt.digest);
}

#[test]
fn noise_support_boundary_accepts_equal_and_refuses_larger() {
    let case = cases().into_iter().find(|c| c["name"] == "support_boundary_accepts_equal").unwrap();
    let s = spec(&case);
    assert!(run(&s).is_ok());
    // the exact residual of unit 0 at the outcome node is 0.25; push it one dyadic step beyond
    let mut over = s.clone();
    over.factual[0].values[4] += 2f64.powi(-20);
    let err = run(&over).unwrap_err();
    let w = err.witness().expect("witness");
    assert_eq!(w.node, Some(Outcome));
    assert_eq!(w.bound, Some(0.25));
    assert!((w.residual.unwrap() - (0.25 + 2f64.powi(-20))).abs() < 1e-15);
    assert_eq!(w.unit.as_ref().unwrap().as_str(), "u00");
}

#[test]
fn each_world_must_cover_exactly_the_factual_units() {
    let case = cases().into_iter().find(|c| c["name"] == "non_dyadic_seven_units").unwrap();
    let s = spec(&case);
    let mut short = s.clone();
    short.plus.units.pop();
    assert_refusal(&short, "temporal_counterfactual.unpaired_histories", "u06", "plus");
    let mut extra = s.clone();
    extra.minus.units.push(UnitId::new("ghost"));
    assert_refusal(&extra, "temporal_counterfactual.unpaired_histories", "ghost", "minus");
    let mut shifted = s;
    shifted.minus.times = [0, 2];
    assert_refusal(&shifted, "temporal_counterfactual.time_misaligned", "u00", "minus");
}

fn assert_refusal(s: &TemporalCounterfactualSpec, detail: &str, unit: &str, world: &str) {
    let err = run(s).unwrap_err();
    let refusal = err.refusal().expect("a typed refusal");
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, detail);
    assert_eq!(refusal.offending.as_deref(), Some(unit));
    let witness = err.witness().expect("witness");
    assert_eq!(witness.unit.as_ref().unwrap().as_str(), unit);
    assert_eq!(witness.world.as_deref(), Some(world));
}

/// The staged executor runs the same arithmetic as separate stages and must return, bit for bit,
/// what the one-shot evaluator returns, on every reference case; a changed action history must
/// reuse the retained abduction and still equal the exact reference for the new worlds.
#[test]
fn staged_executor_is_bit_identical_to_the_one_shot_on_every_reference_case() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let s = spec(&case);
        let one_shot = run(&s).unwrap();
        let mut exec = TemporalExecutor::new();
        let staged = exec.execute(&s, &ExecutionContext::for_tests(2)).unwrap();
        assert_eq!(staged.result, one_shot, "{name}");
        assert_eq!(staged.work.abductions as usize, s.factual.len(), "{name}");
        assert_eq!(staged.work.world_predictions as usize, 2 * s.factual.len(), "{name}");
        // the same request again is a pure reuse
        let again = exec.execute(&s, &ExecutionContext::for_tests(2)).unwrap();
        assert_eq!(again.result, one_shot, "{name}");
        assert_eq!(
            (again.work.abductions, again.work.abductions_reused as usize),
            (0, s.factual.len())
        );
    }
}

#[test]
fn staged_replay_of_a_new_action_history_equals_the_exact_reference() {
    let case = cases().into_iter().find(|c| c["name"] == "closed_form_four_units").unwrap();
    let s = spec(&case);
    let mut exec = TemporalExecutor::new();
    exec.execute(&s, &ExecutionContext::for_tests(2)).unwrap();
    // new worlds (1, 0) and (0, 1); the exact answer needs only the abduced residuals
    let (c, b) = (floats(&case["intercepts"]), floats(&case["coefs"]));
    let mut changed = s.clone();
    changed.plus.name = "first_only".to_owned();
    changed.plus.actions = [1.0, 0.0];
    changed.minus.name = "second_only".to_owned();
    changed.minus.actions = [0.0, 1.0];
    let out = exec.execute(&changed, &ExecutionContext::for_tests(2)).unwrap();
    assert_eq!(out.work.abductions, 0);
    for (row, unit) in out.result.units.iter().zip(case["units"].as_array().unwrap()) {
        let u = floats(&unit["residuals"]);
        let y = |a: [f64; 2]| {
            let l0 = c[0] + u[0];
            let l1 = c[1] + b[0] * l0 + b[1] * a[0] + u[1];
            c[2] + b[2] * l0 + b[3] * a[0] + b[4] * l1 + b[5] * a[1] + u[2]
        };
        assert!((row.plus_outcome - y([1.0, 0.0])).abs() < 1e-12);
        assert!((row.minus_outcome - y([0.0, 1.0])).abs() < 1e-12);
    }
}

fn general_spec(case: &Value) -> GeneralTemporalSpec {
    let h = case["horizon"].as_u64().unwrap() as usize;
    let units = case["units"].as_array().unwrap();
    let ids: Vec<UnitId> = (0..units.len()).map(|i| UnitId::new(format!("u{i:02}"))).collect();
    GeneralTemporalSpec {
        horizon: h,
        latent_confounding: false,
        fit: GeneralMechanismFit {
            fit_id: "general-reference-fit".to_owned(),
            mechanisms: case["mechanisms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| GeneralNodeMechanism {
                    node: m["node"].as_u64().unwrap() as usize,
                    intercept: f(&m["intercept"]),
                    parent_coefficients: m["parents"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| (p[0].as_u64().unwrap() as usize, f(&p[1])))
                        .collect(),
                    noise_halfwidth: None,
                })
                .collect(),
        },
        snapshot: SnapshotId::new("snap"),
        factual: units
            .iter()
            .zip(&ids)
            .enumerate()
            .map(|(i, (u, id))| GeneralFactualHistory {
                unit: id.clone(),
                history: HistoryId::new(format!("h{i:02}")),
                times: (0..h as u32).collect(),
                values: floats(&u["factual"]),
            })
            .collect(),
        worlds: case["worlds"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(w, a)| GeneralActionHistory {
                name: format!("world{w}"),
                times: (0..h as u32).collect(),
                actions: floats(a),
                units: ids.clone(),
            })
            .collect(),
    }
}

#[test]
fn general_horizon_route_matches_the_exact_reference_and_its_staged_executor() {
    let cases: Vec<Value> = serde_json::from_str(include_str!(
        "oracle_data/temporal_fixed_population_general_reference.json"
    ))
    .unwrap();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let s = general_spec(case);
        let r =
            evaluate_general_temporal_counterfactual(&s, &ExecutionContext::for_tests(3)).unwrap();
        let tol = 1e-12;
        for (got, want) in r.units.iter().zip(case["units"].as_array().unwrap()) {
            let outs = floats(&want["outcomes"]);
            assert_eq!(got.outcomes.len(), outs.len(), "{name}");
            for (g, w) in got.outcomes.iter().zip(&outs) {
                assert!((g - w).abs() <= tol * w.abs().max(1.0), "{name}: {g} vs {w}");
            }
            let fac = floats(&want["factual"]);
            assert_eq!(got.factual_outcome, *fac.last().unwrap(), "{name}");
        }
        for (g, w) in r.world_means.iter().zip(floats(&case["world_means"])) {
            assert!((g - w).abs() <= tol * w.abs().max(1.0), "{name} mean: {g} vs {w}");
        }
        assert!(r.receipt.is_self_consistent(), "{name}");
        let mut exec = GeneralTemporalExecutor::new();
        let staged = exec.execute(&s, &ExecutionContext::for_tests(3)).unwrap();
        assert_eq!(staged.result, r, "{name}: staged differs from one-shot");
        let again = exec.execute(&s, &ExecutionContext::for_tests(3)).unwrap();
        assert_eq!((again.work.abductions, again.result == r), (0, true), "{name}");
    }
}

/// At horizon two with two worlds the general route is the fixed route: same numbers.
#[test]
fn general_route_at_horizon_two_equals_the_fixed_route_bit_for_bit() {
    let case = cases().into_iter().find(|c| c["name"] == "non_dyadic_seven_units").unwrap();
    let fixed = run(&spec(&case)).unwrap();
    let (c, b) = (floats(&case["intercepts"]), floats(&case["coefs"]));
    let units = case["units"].as_array().unwrap();
    let ids: Vec<UnitId> = (0..units.len()).map(|i| UnitId::new(format!("u{i:02}"))).collect();
    let mech = |node, intercept, parents: Vec<(usize, f64)>| GeneralNodeMechanism {
        node,
        intercept,
        parent_coefficients: parents,
        noise_halfwidth: None,
    };
    let world = |name: &str, key: &str| {
        let a = floats(&case[key]);
        GeneralActionHistory {
            name: name.to_owned(),
            times: vec![0, 1],
            actions: a,
            units: ids.clone(),
        }
    };
    let g = GeneralTemporalSpec {
        horizon: 2,
        latent_confounding: false,
        fit: GeneralMechanismFit {
            fit_id: "reference-fit".to_owned(),
            mechanisms: vec![
                mech(0, c[0], vec![]),
                mech(2, c[1], vec![(0, b[0]), (1, b[1])]),
                mech(4, c[2], vec![(0, b[2]), (1, b[3]), (2, b[4]), (3, b[5])]),
            ],
        },
        snapshot: SnapshotId::new("snap"),
        factual: units
            .iter()
            .zip(&ids)
            .enumerate()
            .map(|(i, (u, id))| GeneralFactualHistory {
                unit: id.clone(),
                history: HistoryId::new(format!("h{i:02}")),
                times: vec![0, 1],
                values: floats(&u["factual"]),
            })
            .collect(),
        worlds: vec![world("plus", "plus"), world("minus", "minus")],
    };
    let r = evaluate_general_temporal_counterfactual(&g, &ExecutionContext::for_tests(3)).unwrap();
    for (a, u) in fixed.units.iter().zip(&r.units) {
        assert_eq!(a.plus_outcome.to_bits(), u.outcomes[0].to_bits());
        assert_eq!(a.minus_outcome.to_bits(), u.outcomes[1].to_bits());
    }
    assert_eq!(fixed.mean_contrast.to_bits(), r.mean_contrast(0, 1).unwrap().to_bits());
}
