//! Known-truth evidence for recursive z-transport derivations with two source
//! exchanges, where the second exchanged coordinate is summed over.
//!
//! The structural models are asymmetric on purpose: `P(Y = 1 | do(x, z))`
//! varies with `z`, so a formula that evaluated the inner factor at one fixed
//! `z` would miss the truth. An earlier fixture used XOR mechanisms whose
//! interventional risk was one half at every level and could not tell the two
//! apart.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_core::{ExecutionContext, RegimeKind, Value};
use antecedent_estimate::evaluate_exact_z_transport;
use antecedent_expr::{Assignment, EvalError, ExactEvaluationLimits};
use common::z_scm::{
    Scm, Spec, bit, build, diagram, evaluate_risk, family_specs, identify_and_bind, query, vid,
};

/// Majority vote with threshold `k`.
fn majority(votes: &[u8], k: u8) -> u8 {
    u8::from(votes.iter().copied().sum::<u8>() >= k)
}

// 0=X, 1=Z, 2=V, 3=Y, 4=R. X→Z→Y; X↔Z (e0), X↔Y (e1), Z↔V (e2), R (e3); private e4..e6.
fn asymmetric_two_exchange_scm() -> Scm {
    Scm {
        n: 5,
        exo_p: vec![0.3, 0.6, 0.5, 0.5, 0.4, 0.7, 0.25],
        f: vec![
            Box::new(|_, e| bit(e[0] == 1) ^ bit(e[1] == 1 && e[4] == 1)),
            Box::new(|v, e| bit(v[0] == 1 && e[5] == 1) ^ bit(e[0] == 1)),
            Box::new(|v, e| bit(v[1] == 1) ^ bit(e[2] == 1)),
            Box::new(|v, e| bit((v[1] == 1 && e[1] == 1) || e[6] == 1)),
            Box::new(|_, e| e[3]),
        ],
    }
}

fn two_exchange_diagram() -> antecedent_graph::SelectionDiagram {
    diagram(5, &[(0, 1), (1, 3)], &[(0, 1), (0, 3), (1, 2)], &[])
}

#[test]
fn recursive_formula_uses_two_source_exchange_factors_and_matches_exact_scm_truth() {
    let scm = asymmetric_two_exchange_scm();
    let diagram = two_exchange_diagram();
    let query = query(3, 0, &[0, 1], &[(0, false), (1, false)]);
    let specs = family_specs("source", &[0, 1], &["target", "source"]);
    let (catalog, data) = build(&|_| &scm, 5, &specs, &["source", "target"]);
    let (derivation, bound) = identify_and_bind(&diagram, &query, &catalog);
    let rules = derivation.to_record().rules;
    assert_eq!(
        rules.iter().filter(|rule| rule.contains("line10.source_exchange")).count(),
        2,
        "each exchange factor must remain explicit in the recursive proof: {rules:?}"
    );
    // The treatment is bound by the request and the second exchanged coordinate
    // by the enclosing summation, so neither is fixed at a declared level.
    assert!(rules.iter().any(|rule| rule.contains("[(0, None)]")), "{rules:?}");
    assert!(rules.iter().any(|rule| rule.contains("[(0, None), (1, None)]")), "{rules:?}");
    // Independent truth from the structural equations, and its frozen value.
    let truth = [scm.risk(&[(0, 0)], 3), scm.risk(&[(0, 1)], 3)];
    assert!((truth[0] - 0.385).abs() < 5e-4 && (truth[1] - 0.511).abs() < 5e-4, "{truth:?}");
    let fixed = scm.risk(&[(0, 0), (1, 0)], 3);
    assert!((fixed - truth[0]).abs() > 0.05, "the fixture must separate a fixed z from the sum");
    for (level, expected) in [(false, truth[0]), (true, truth[1])] {
        let risk = evaluate_risk(&bound, &data, 0, level).unwrap();
        assert!((risk - expected).abs() < 1e-12, "P(Y=1 | do(X={level}))={risk} != {expected}");
    }
}

// Case C of the empirical review: X→Z→Y, X↔Z, X↔Y, Z↔V, R isolated, selection
// on V and R (irrelevant to Y). Exogenous bits: 0=u_xz, 1=u_xy, 2=u_zv, 3=u_r,
// 4=u_x, 5=u_z, 6=u_y.
fn case_c_scm(target: bool) -> Scm {
    Scm {
        n: 5,
        exo_p: vec![0.3, 0.4, 0.5, if target { 0.7 } else { 0.2 }, 0.25, 0.35, 0.15],
        f: vec![
            Box::new(|_, e| majority(&[e[0], e[1], e[4]], 2)),
            Box::new(|v, e| majority(&[v[0], e[0], e[5]], 2)),
            Box::new(move |v, e| if target { e[2] } else { v[1] ^ e[2] }),
            Box::new(|v, e| majority(&[v[1], e[1], e[6]], 2)),
            Box::new(|_, e| e[3]),
        ],
    }
}

#[test]
fn summed_exchange_coordinate_follows_the_summation_variable_not_the_declared_level() {
    let source = case_c_scm(false);
    let target = case_c_scm(true);
    let pick = |population: &str| if population == "target" { &target } else { &source };
    let diagram = diagram(5, &[(0, 1), (1, 3)], &[(0, 1), (0, 3), (1, 2)], &[2, 4]);
    let query = query(3, 0, &[0, 1], &[(0, false), (1, false)]);
    let specs = family_specs("source", &[0, 1], &["target"]);
    let (catalog, data) = build(&pick, 5, &specs, &["source", "target"]);
    let (_, bound) = identify_and_bind(&diagram, &query, &catalog);
    let truth = target.risk(&[(0, 0)], 3);
    assert!((truth - 0.105_150).abs() < 5e-7, "target truth {truth}");
    let risk = evaluate_risk(&bound, &data, 0, false).unwrap();
    assert!((risk - truth).abs() < 1e-12, "P*(Y=1 | do(X=0))={risk} != {truth}");
    // Evaluating the inner factor at the declared level z=0 would give 0.060.
    let fixed_z = source.risk(&[(0, 0), (1, 0)], 3);
    assert!((fixed_z - 0.06).abs() < 5e-7 && (risk - fixed_z).abs() > 0.04);
    let risk_one = evaluate_risk(&bound, &data, 0, true).unwrap();
    assert!((risk_one - target.risk(&[(0, 1)], 3)).abs() < 1e-12);
}

// Case B of the empirical review: W→Y, X→Y, X↔Y, selection on W. Exogenous bits:
// 0=u_xy, 1=u_x, 2=u_w, 3=u_y.
fn case_b_scm(target: bool) -> Scm {
    Scm {
        n: 3,
        exo_p: vec![0.4, 0.3, if target { 0.65 } else { 0.2 }, 0.1],
        f: vec![
            Box::new(|_, e| e[2]),
            Box::new(|_, e| e[1] | e[0]),
            Box::new(|v, e| majority(&[v[1], v[0], e[0], e[3]], 2)),
        ],
    }
}

#[test]
fn exchanged_treatment_is_bound_by_the_request_and_unsupplied_levels_are_refused() {
    let source = case_b_scm(false);
    let target = case_b_scm(true);
    let pick = |population: &str| if population == "target" { &target } else { &source };
    let diagram = diagram(3, &[(0, 2), (1, 2)], &[(1, 2)], &[0]);
    let query = query(2, 1, &[1], &[(1, false)]);
    let specs = vec![
        Spec { population: "target", kind: RegimeKind::Observational, assignments: vec![] },
        Spec { population: "source", kind: RegimeKind::Experimental, assignments: vec![(1, 0)] },
        Spec { population: "source", kind: RegimeKind::Experimental, assignments: vec![(1, 1)] },
    ];
    let (catalog, data) = build(&pick, 3, &specs, &["source", "target"]);
    let (_, bound) = identify_and_bind(&diagram, &query, &catalog);
    let truth = [target.risk(&[(1, 0)], 2), target.risk(&[(1, 1)], 2)];
    assert!((truth[0] - 0.313).abs() < 5e-4 && (truth[1] - 0.811).abs() < 5e-4, "{truth:?}");
    // A request at the other treatment level is answered from that level's
    // cited experiment, never silently from the declared assignment.
    for (level, expected) in [(false, truth[0]), (true, truth[1])] {
        let risk = evaluate_risk(&bound, &data, 1, level).unwrap();
        assert!((risk - expected).abs() < 1e-12, "P*(Y=1 | do(X={level}))={risk} != {expected}");
    }
    // A level no cited experiment supplies is a typed refusal, not a number.
    let ctx = ExecutionContext::for_tests(1);
    for level in [Value::f64(7.0), Value::f64(0.5)] {
        let request = Assignment::from_pairs([(vid(1), level)]);
        let error = evaluate_exact_z_transport(
            &bound,
            data.clone(),
            request,
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap_err();
        assert!(
            matches!(error, EvalError::ProviderKind(message) if message.contains("no cited source experiment")),
            "{error}"
        );
    }
    // With only the do(X=0) experiment supplied, X=1 is refused rather than
    // answered with the X=0 law.
    let (catalog, data) = build(&pick, 3, &specs[..2], &["source", "target"]);
    let (_, bound) = identify_and_bind(&diagram, &query, &catalog);
    assert!((evaluate_risk(&bound, &data, 1, false).unwrap() - truth[0]).abs() < 1e-12);
    assert!(matches!(
        evaluate_risk(&bound, &data, 1, true),
        Err(EvalError::ProviderKind(message)) if message.contains("no cited source experiment")
    ));
}
