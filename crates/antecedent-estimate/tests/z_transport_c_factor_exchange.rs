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
