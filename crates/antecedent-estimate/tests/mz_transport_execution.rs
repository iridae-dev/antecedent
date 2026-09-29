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

use antecedent_core::{DependenceGroup, EvidenceCatalog, ExecutionContext, Value};
use antecedent_estimate::{
    Z_TRANSPORT_INTERVAL_NOT_MEASURED, evaluate_exact_mz_transport, mz_sampling_dependence,
    mz_transport_bootstrap_interval,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_identify::{
    BoundMzTransportFunctional, MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision, MzTransportRoute,
    ZTransportSourceSpec, bind_mz_transport_catalog, decide_mz_transport,
};
use common::mz_fixture::{
    X, Y, empirical, evidence, graph, query, source_b_scm, sources, target_scm, with_studies,
};
use common::z_scm::{risk_of, vid};

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
