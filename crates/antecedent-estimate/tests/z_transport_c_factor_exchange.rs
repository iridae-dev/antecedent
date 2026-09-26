//! Known-truth evidence for the source c-factor taken at a `TRz` line-10 exchange
//! and for rule 3 leaving the child formula unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_core::RegimeKind;
use common::z_scm::{Scm, Spec, bit, build, diagram, evaluate_risk, identify_and_bind, query};

// G5: 0=Z, 1=X, 2=A, 3=M, 4=Y. Z→X, X→A, A→M, M→Y, X→Y; Z↔X (e0), Z↔Y (e1), A↔Y (e2).
// After the line-8 reduction the district {Z, A, Y} keeps X and M as external
// parents; M is a descendant of A, so conditioning the exchanged source law on
// M would not be the c-factor Q^s_z[{X, A, Y}] = P^s_{z, m}(x, a, y).
fn g5_scm() -> Scm {
    Scm {
        n: 5,
        exo_p: vec![0.3, 0.6, 0.4, 0.2, 0.7, 0.5, 0.15, 0.35],
        f: vec![
            Box::new(|_, e| bit((e[0] == 1 && e[1] == 1) || e[3] == 1)),
            Box::new(|v, e| bit(v[0] == 1) ^ bit(e[0] == 1 && e[4] == 1)),
            Box::new(|v, e| bit(v[1] == 1 && e[5] == 1) ^ bit(e[2] == 1)),
            Box::new(|v, e| bit(v[2] == 1) ^ bit(e[6] == 1)),
            Box::new(|v, e| {
                bit((v[1] == 1 && v[3] == 1) || e[1] == 1) ^ bit(e[2] == 1 && e[7] == 1)
            }),
        ],
    }
}

#[test]
fn line10_exchange_takes_the_source_c_factor_with_external_parents_intervened() {
    let scm = g5_scm();
    let diagram =
        diagram(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)], &[(0, 1), (0, 4), (2, 4)], &[]);
    let query = query(4, 1, &[0], &[(0, false)]);
    let specs = vec![
        Spec { population: "target", kind: RegimeKind::Observational, assignments: vec![] },
        Spec { population: "source", kind: RegimeKind::Observational, assignments: vec![] },
        Spec { population: "source", kind: RegimeKind::Experimental, assignments: vec![(0, 0)] },
        Spec { population: "source", kind: RegimeKind::Experimental, assignments: vec![(0, 1)] },
    ];
    let (catalog, data) = build(&|_| &scm, 5, &specs, &["source", "target"]);
    let (derivation, bound) = identify_and_bind(&diagram, &query, &catalog);
    let rules = derivation.to_record().rules;
    assert!(rules.iter().any(|rule| rule == "ztr.line8.recurse"), "{rules:?}");
    assert!(rules.iter().any(|rule| rule.starts_with("ztr.line10.c_factor:")), "{rules:?}");
    let truth = [scm.risk(&[(1, 0)], 4), scm.risk(&[(1, 1)], 4)];
    assert!((truth[0] - 0.572).abs() < 5e-4 && (truth[1] - 0.716).abs() < 5e-4, "{truth:?}");
    for (level, expected) in [(false, truth[0]), (true, truth[1])] {
        let risk = evaluate_risk(&bound, &data, 1, level).unwrap();
        assert!((risk - expected).abs() < 1e-12, "P(Y=1 | do(X={level}))={risk} != {expected}");
    }
    // The conditional replacement Σ_{a,m} P(m|a) P^s_z(a|x,m) P^s_z(y|x,a,m)
    // evaluates to 0.8397 at x=0; the c-factor formula recovers the truth.
    assert!((truth[0] - 0.8397).abs() > 0.2);
}

// Surrogate graph 0=W, 1=Z, 2=X, 3=Y: W→Z, Z→X, X→Y, W→Y; W↔Y (e0), Z↔Y (e1),
// Z↔X (e2); private e3..e6. The target's Z mechanism differs.
fn surrogate_scm(target: bool) -> Scm {
    Scm {
        n: 4,
        exo_p: vec![0.35, 0.6, 0.45, 0.3, 0.55, 0.2, 0.4],
        f: vec![
            Box::new(|_, e| bit(e[0] == 1 || e[3] == 1)),
            Box::new(move |v, e| {
                if target {
                    bit(v[0] == 1) ^ bit(e[1] == 1) ^ bit(e[2] == 1)
                } else {
                    bit((v[0] == 1 && e[1] == 1) || (e[2] == 1 && e[4] == 1))
                }
            }),
            Box::new(|v, e| bit(v[1] == 1) ^ bit(e[2] == 1 && e[5] == 1)),
            Box::new(|v, e| {
                bit((v[2] == 1) && (v[0] == 1 || e[0] == 1)) ^ bit(e[1] == 1 && e[6] == 1)
            }),
        ],
    }
}

#[test]
fn rule3_is_the_identity_and_a_selected_controllable_needs_no_target_law() {
    let source = surrogate_scm(false);
    let target = surrogate_scm(true);
    let pick = |population: &str| if population == "target" { &target } else { &source };
    let diagram = diagram(4, &[(0, 1), (1, 2), (2, 3), (0, 3)], &[(0, 3), (1, 3), (1, 2)], &[1]);
    let query = query(3, 2, &[1], &[(1, false)]);
    let experiments = vec![
        Spec { population: "source", kind: RegimeKind::Experimental, assignments: vec![(1, 0)] },
        Spec { population: "source", kind: RegimeKind::Experimental, assignments: vec![(1, 1)] },
    ];
    let truth = [target.risk(&[(2, 0)], 3), target.risk(&[(2, 1)], 3)];
    // Only the two source do(Z) laws: the formula Σ_w P^s_z(y | w, x) P^s_z(w)
    // cites no target factor, so no target law is required.
    let (catalog, data) = build(&pick, 4, &experiments, &["source", "target"]);
    let (derivation, bound) = identify_and_bind(&diagram, &query, &catalog);
    let rules = derivation.to_record().rules;
    assert!(rules.iter().any(|rule| rule == "ztr.line3.enlarge"), "{rules:?}");
    assert!(!rules.iter().any(|rule| rule == "ztr.line3.kernel_weighted"), "{rules:?}");
    let inspection = derivation.inspect_proof(&catalog);
    assert!(
        inspection.factors.iter().all(|factor| factor.population == "source"),
        "{inspection:?}"
    );
    for (level, expected) in [(false, truth[0]), (true, truth[1])] {
        let risk = evaluate_risk(&bound, &data, 2, level).unwrap();
        assert!((risk - expected).abs() < 1e-12, "P*(Y=1 | do(X={level}))={risk} != {expected}");
    }
    // A target observational law in the catalog changes nothing.
    let mut with_target =
        vec![Spec { population: "target", kind: RegimeKind::Observational, assignments: vec![] }];
    with_target.extend(experiments);
    let (catalog, data) = build(&pick, 4, &with_target, &["source", "target"]);
    let (_, bound) = identify_and_bind(&diagram, &query, &catalog);
    for (level, expected) in [(false, truth[0]), (true, truth[1])] {
        assert!((evaluate_risk(&bound, &data, 2, level).unwrap() - expected).abs() < 1e-12);
    }
}
