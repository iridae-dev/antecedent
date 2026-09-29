//! Exact and empirical execution of multi-source limited-experiment formulas.
//!
//! The graph is Bareinboim & Pearl (`NeurIPS` 2014, R-443) Figure 1(c,d):
//! `Z1 -> X -> Z2 -> Y` with `Z1 <-> X`, `Z1 <-> Z2`, `Z1 <-> Y`. Source `a`
//! changes the `Z1` and `Z2` mechanisms and can experiment on `Z2`; source `b`
//! changes `Z1` and `Y` and can experiment on `Z1`. Every law is enumerated from
//! each population's structural model, so the formula is checked against the
//! target's own interventional truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, EvidenceCatalog, ExecutionContext, InterventionAssignment, RegimeKind, Value,
};
use antecedent_estimate::{
    Z_TRANSPORT_INTERVAL_NOT_MEASURED, evaluate_exact_mz_transport, mz_sampling_dependence,
    mz_transport_bootstrap_interval,
};
use antecedent_expr::{Assignment, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    BoundMzTransportFunctional, MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision, MzTransportQuery,
    MzTransportRoute, ZTransportSourceSpec, bind_mz_transport_catalog, decide_mz_transport,
};
use common::z_scm::{Scm, Spec, bit, build, diagram, risk_of, vid};

const Z1: usize = 0;
const X: usize = 1;
const Z2: usize = 2;
const Y: usize = 3;

// Exogenous bits: e0 = Z1<->X, e1 = Z1<->Z2, e2 = Z1<->Y; e3..e6 private to Z1, X, Z2, Y.
const EXO: [f64; 7] = [0.35, 0.6, 0.45, 0.3, 0.7, 0.65, 0.4];

fn x_mechanism() -> common::z_scm::Mechanism {
    Box::new(|v, e| bit(v[Z1] == 1) ^ bit(e[0] == 1 && e[4] == 1))
}
fn z2_target() -> common::z_scm::Mechanism {
    Box::new(|v, e| bit((v[X] == 1 && e[5] == 1) || (v[X] == 0 && e[1] == 1)))
}
fn y_target() -> common::z_scm::Mechanism {
    Box::new(|v, e| bit((v[Z2] == 1 && e[6] == 1) || (e[2] == 1 && e[6] == 0)))
}

fn target_scm() -> Scm {
    Scm {
        n: 4,
        exo_p: EXO.to_vec(),
        f: vec![
            Box::new(|_, e| bit((e[0] == 1 && e[3] == 1) || (e[1] == 1 && e[2] == 1))),
            x_mechanism(),
            z2_target(),
            y_target(),
        ],
    }
}

/// Source `a`: the Z1 and Z2 mechanisms differ from the target.
fn source_a_scm() -> Scm {
    Scm {
        n: 4,
        exo_p: EXO.to_vec(),
        f: vec![
            Box::new(|_, e| bit(e[3] == 1 || e[1] == 1)),
            x_mechanism(),
            Box::new(|v, e| bit(v[X] == 1) ^ bit(e[1] == 1 && e[5] == 1)),
            y_target(),
        ],
    }
}

/// Source `b`: the Z1 and Y mechanisms differ from the target.
fn source_b_scm() -> Scm {
    Scm {
        n: 4,
        exo_p: EXO.to_vec(),
        f: vec![
            Box::new(|_, e| bit(e[0] == 1 && e[2] == 1)),
            x_mechanism(),
            z2_target(),
            Box::new(|v, e| bit(v[Z2] == 1) ^ bit(e[2] == 1 && e[6] == 1)),
        ],
    }
}

fn graph() -> antecedent_graph::Admg {
    diagram(4, &[(Z1, X), (X, Z2), (Z2, Y)], &[(Z1, X), (Z1, Z2), (Z1, Y)], &[])
        .causal_graph()
        .clone()
}

fn sources() -> Vec<ZTransportSourceSpec> {
    vec![
        ZTransportSourceSpec {
            population: Arc::from("a"),
            controllable: Arc::from([vid(Z2)]),
            experiment_assignment: Arc::from([]),
            selection_targets: Arc::from([vid(Z1), vid(Z2)]),
        },
        ZTransportSourceSpec {
            population: Arc::from("b"),
            controllable: Arc::from([vid(Z1)]),
            experiment_assignment: Arc::from([InterventionAssignment {
                variable: vid(Z1),
                value: Value::Bool(false),
            }]),
            selection_targets: Arc::from([vid(Z1), vid(Y)]),
        },
    ]
}

fn query(sources: Vec<ZTransportSourceSpec>) -> MzTransportQuery {
    MzTransportQuery {
        outcomes: Arc::from([vid(Y)]),
        treatments: Arc::from([vid(X)]),
        target: Arc::from("target"),
        sources: sources.into(),
    }
}

/// Target observational law; `do(Z2 = 0/1)` in `a`; `do(Z1 = 0)` in `b`.
fn evidence() -> (EvidenceCatalog, ExactTransportData) {
    let specs = [
        Spec { population: "target", kind: RegimeKind::Observational, assignments: vec![] },
        Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(Z2, 0)] },
        Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(Z2, 1)] },
        Spec { population: "b", kind: RegimeKind::Experimental, assignments: vec![(Z1, 0)] },
    ];
    let (target, a, b) = (target_scm(), source_a_scm(), source_b_scm());
    let scm_for = |population: &str| -> &Scm {
        match population {
            "a" => &a,
            "b" => &b,
            _ => &target,
        }
    };
    build(&scm_for, 4, &specs, &["target", "a", "b"])
}

fn bound(
    sources: Vec<ZTransportSourceSpec>,
    catalog: &EvidenceCatalog,
) -> BoundMzTransportFunctional {
    let q = query(sources);
    let ctx = ExecutionContext::for_tests(3);
    let MzTransportDecision::Identified { derivation, .. } =
        decide_mz_transport(&graph(), &q, catalog, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap()
    else {
        panic!("the complementary catalog identifies");
    };
    assert!(matches!(derivation.route(), MzTransportRoute::Combined { .. }));
    bind_mz_transport_catalog(&graph(), &derivation, catalog).unwrap()
}

fn request(x: bool) -> Assignment {
    Assignment::from_pairs([(vid(X), Value::Bool(x))])
}

fn truth(x: u8) -> f64 {
    target_scm().risk(&[(X, x)], Y)
}

#[test]
fn combined_formula_matches_the_target_interventional_truth() {
    let (catalog, data) = evidence();
    let functional = bound(sources(), &catalog);
    let ctx = ExecutionContext::for_tests(3);
    for x in [0u8, 1] {
        let point = evaluate_exact_mz_transport(
            &functional,
            data.clone(),
            request(x == 1),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let risk = risk_of(&point);
        assert!((risk - truth(x)).abs() < 1e-12, "do(X={x}): formula {risk}, truth {}", truth(x));
    }
    // The fixture is not trivial: the target's observational conditional differs.
    let observational = target_scm().law(&[], &[X, Y]);
    let naive = observational[3] / (observational[2] + observational[3]);
    assert!((naive - truth(1)).abs() > 1e-3);
    // Each source's own law differs from the target's too.
    assert!((source_b_scm().risk(&[(X, 1)], Y) - truth(1)).abs() > 1e-3);
}

#[test]
fn source_order_does_not_change_the_number() {
    let (catalog, data) = evidence();
    let ctx = ExecutionContext::for_tests(3);
    let mut reversed = sources();
    reversed.reverse();
    let evaluate = |functional: &BoundMzTransportFunctional| {
        risk_of(
            &evaluate_exact_mz_transport(
                functional,
                data.clone(),
                request(true),
                ExactEvaluationLimits::default(),
                &ctx,
            )
            .unwrap(),
        )
    };
    let forward = bound(sources(), &catalog);
    let backward = bound(reversed, &catalog);
    assert_eq!(forward.cited_regimes(), backward.cited_regimes());
    assert_eq!(evaluate(&forward).to_bits(), evaluate(&backward).to_bits());
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "fixture probabilities are in [0, 1] and the sample sizes fit u64"
)]
fn empirical(data: &ExactTransportData, n: f64) -> ExactTransportData {
    let laws = data
        .laws()
        .iter()
        .map(|law| {
            let counts =
                law.probabilities().iter().map(|p| (p * n).round() as u64).collect::<Vec<_>>();
            let total = counts.iter().sum::<u64>() as f64;
            ExactDiscreteLaw::try_empirical(
                law.population(),
                law.regime(),
                law.interventions().to_vec(),
                law.axes().to_vec(),
                counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>(),
                law.snapshot_identity(),
                law.tolerance(),
            )
            .unwrap()
            .with_empirical_counts(counts)
            .unwrap()
        })
        .collect::<Vec<_>>();
    ExactTransportData::try_new(laws, data.max_support_rows()).unwrap()
}

/// Declare every regime its own study, so independence between them is known.
fn with_studies(catalog: &EvidenceCatalog) -> EvidenceCatalog {
    let mut out = catalog.clone();
    let mut regimes = out.regimes.to_vec();
    for regime in &mut regimes {
        regime.study = Some(Arc::from(format!("study-{}", regime.id.raw())));
    }
    out.regimes = regimes.into();
    out
}

#[test]
fn empirical_tables_run_through_the_same_formula_with_a_joint_bootstrap() {
    let (catalog, exact) = evidence();
    let catalog = with_studies(&catalog);
    let data = empirical(&exact, 40_000.0);
    let functional = bound(sources(), &catalog);
    let ctx = ExecutionContext::for_tests(11);
    let limits = ExactEvaluationLimits::default();
    let point = |x: bool| {
        risk_of(
            &evaluate_exact_mz_transport(&functional, data.clone(), request(x), limits, &ctx)
                .unwrap(),
        )
    };
    let (p0, p1) = (point(false), point(true));
    assert!((p0 - truth(0)).abs() < 1e-2 && (p1 - truth(1)).abs() < 1e-2);

    let intervals = mz_transport_bootstrap_interval(
        &functional,
        &data,
        &[request(false), request(true)],
        limits,
        99,
        0.95,
        &ctx,
    )
    .unwrap()
    .expect("declared independent studies publish a joint interval");
    assert_eq!(intervals.reason.as_ref(), Z_TRANSPORT_INTERVAL_NOT_MEASURED);
    assert_eq!(intervals.replicates_ok + intervals.replicates_failed, 99);
    let means0 = &intervals.requests[0].mean_intervals;
    let means1 = &intervals.requests[1].mean_intervals;
    assert!(means0[0].1 <= p0 && p0 <= means0[0].2);
    assert!(means1[0].1 <= p1 && p1 <= means1[0].2);
    // The contrast is a within-replicate difference, so it contains the point contrast.
    let &(k, outcome, lower, upper) = &intervals.contrasts[0];
    assert_eq!((k, outcome), (1, vid(Y)));
    assert!(lower <= p1 - p0 && p1 - p0 <= upper);

    // The same seed reproduces the same interval.
    let again = mz_transport_bootstrap_interval(
        &functional,
        &data,
        &[request(false), request(true)],
        limits,
        99,
        0.95,
        &ctx,
    )
    .unwrap()
    .unwrap();
    assert_eq!(again.contrasts[0].2.to_bits(), lower.to_bits());
}

#[test]
fn exact_laws_publish_no_interval() {
    let (catalog, data) = evidence();
    let functional = bound(sources(), &with_studies(&catalog));
    let withheld = mz_transport_bootstrap_interval(
        &functional,
        &data,
        &[request(true)],
        ExactEvaluationLimits::default(),
        99,
        0.95,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_eq!(withheld.unwrap_err(), "exact_supplied_law_no_sampling_uncertainty");
}

#[test]
fn undeclared_or_unsupported_dependence_withholds_the_interval_not_the_point() {
    let (catalog, exact) = evidence();
    let data = empirical(&exact, 40_000.0);
    let ctx = ExecutionContext::for_tests(5);
    let check = |catalog: &EvidenceCatalog| {
        let functional = bound(sources(), catalog);
        let dependence = mz_sampling_dependence(&functional);
        let interval = mz_transport_bootstrap_interval(
            &functional,
            &data,
            &[request(true)],
            ExactEvaluationLimits::default(),
            49,
            0.95,
            &ctx,
        )
        .unwrap();
        // The point is always available.
        evaluate_exact_mz_transport(
            &functional,
            data.clone(),
            request(true),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        (dependence, interval.err())
    };
    // Declared studies are independent.
    assert_eq!(check(&with_studies(&catalog)), (Ok(()), None));
    // Two regimes of source `a` with no study identity: independence is not known.
    let unknown = check(&catalog);
    assert_eq!(unknown, (Err("sampling_dependence_unknown"), Some("sampling_dependence_unknown")));
    let with_binding = |edit: &dyn Fn(&mut antecedent_core::RegimeBinding)| {
        let mut out = with_studies(&catalog);
        let mut bindings = out.bindings.to_vec();
        edit(&mut bindings[1]);
        out.bindings = bindings.into();
        out
    };
    // Linked units cannot be resampled jointly from count tables.
    let linked = check(&with_binding(&|b| b.dependence = DependenceGroup::LinkedUnits));
    assert_eq!(linked.0, Err("transport.unsupported_dependence"));
    // An explicitly unknown dependence is unknown.
    let declared_unknown =
        check(&with_binding(&|b| b.dependence = DependenceGroup::UnknownDependence));
    assert_eq!(declared_unknown.0, Err("sampling_dependence_unknown"));
    // The same snapshot without a forwarded dataset identity shares units that a
    // table bootstrap cannot resample jointly; with one it is resampled as one table.
    let shared = with_binding(&|b| b.snapshot_identity = Arc::from("a-2"));
    assert_eq!(
        mz_sampling_dependence(&bound(sources(), &shared)),
        Err("transport.unsupported_dependence")
    );
    let forwarded = with_binding(&|b| b.dataset_identity = Some(Arc::from("a-trial")));
    let mut forwarded_bindings = forwarded.bindings.to_vec();
    forwarded_bindings[2].dataset_identity = Some(Arc::from("a-trial"));
    let mut forwarded = forwarded;
    forwarded.bindings = forwarded_bindings.into();
    assert_eq!(mz_sampling_dependence(&bound(sources(), &forwarded)), Ok(()));
}
