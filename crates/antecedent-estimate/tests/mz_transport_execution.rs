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
    Z_TRANSPORT_INTERVAL_NOT_MEASURED, evaluate_exact_mz_transport, mz_interval_withheld_reason,
    mz_sampling_dependence, mz_transport_bootstrap_interval, mz_transport_bootstrap_law_draws,
};
use antecedent_expr::{Assignment, ExactEvaluationLimits};
use antecedent_identify::{
    BoundMzTransportFunctional, MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision, MzTransportRoute,
    ZTransportSourceSpec, bind_mz_transport_catalog, decide_mz_transport,
    verify_mz_transport_obstruction,
};
use common::mz_fixture::{
    B_TRIAL_REGIME, SHARED_REGIME, SharedTable, X, Y, empirical, evidence, evidence_with_twin,
    graph, query, source_b_scm, sources, target_scm, twin_sources, with_conflicting_a_trial,
    with_shared_b_trial, with_studies,
};
use common::z_scm::{Mechanism, Scm, Spec, build, diagram, risk_of, vid};

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

/// `(point risk, published interval or withheld reason)` of the internal joint
/// bootstrap over `do(X=0)` and `do(X=1)`.
fn joint_bootstrap(
    catalog: &EvidenceCatalog,
    data: &antecedent_expr::ExactTransportData,
) -> (f64, Result<antecedent_estimate::MzTransportIntervals, &'static str>) {
    let functional = bound(sources(), catalog);
    let ctx = ExecutionContext::for_tests(17);
    let limits = ExactEvaluationLimits::default();
    let point = evaluate_exact_mz_transport(&functional, data.clone(), request(true), limits, &ctx)
        .unwrap();
    let run = mz_transport_bootstrap_interval(
        &functional,
        data,
        &[request(false), request(true)],
        limits,
        49,
        0.95,
        &ctx,
    )
    .unwrap();
    (risk_of(&point), run)
}

#[test]
fn identical_shared_dataset_tables_are_resampled_as_one() {
    let (catalog, exact) = evidence();
    let (catalog, data) =
        with_shared_b_trial(&catalog, &empirical(&exact, 40_000.0), SharedTable::Identical);
    let (point, run) = joint_bootstrap(&catalog, &data);
    assert!((point - truth(1)).abs() < 1e-2);
    let intervals = run.expect("one table under two regimes is resampled jointly");
    assert_eq!(intervals.replicates_ok + intervals.replicates_failed, 49);
    let functional = bound(sources(), &catalog);
    assert_eq!(mz_interval_withheld_reason(&functional, &data).unwrap(), None);
    // Both regimes read the same resampled table in every replicate.
    let draws =
        mz_transport_bootstrap_law_draws(&functional, &data, 25, &ExecutionContext::for_tests(3))
            .unwrap()
            .unwrap();
    let counts = |drawn: &antecedent_expr::ExactTransportData, regime: u32| {
        drawn
            .laws()
            .iter()
            .find(|law| law.regime().raw() == regime)
            .and_then(|law| law.empirical_counts().map(<[u64]>::to_vec))
            .unwrap()
    };
    let mut moved = false;
    for drawn in draws.iter().flatten() {
        assert_eq!(counts(drawn, B_TRIAL_REGIME), counts(drawn, SHARED_REGIME));
        moved |= counts(drawn, B_TRIAL_REGIME)
            != data
                .laws()
                .iter()
                .find(|l| l.regime().raw() == B_TRIAL_REGIME)
                .unwrap()
                .empirical_counts()
                .unwrap();
    }
    assert!(moved, "the shared table is resampled, not frozen");
}

#[test]
fn a_recorded_margin_of_a_shared_joint_is_projected_within_every_replicate() {
    let (catalog, exact) = evidence();
    let (catalog, data) =
        with_shared_b_trial(&catalog, &empirical(&exact, 40_000.0), SharedTable::Margin);
    let (point, run) = joint_bootstrap(&catalog, &data);
    assert!((point - truth(1)).abs() < 1e-2);
    let intervals = run.expect("a recorded margin is projected from the resampled joint");
    let p1 = point;
    let means = &intervals.requests[1].mean_intervals;
    assert!(means[0].1 <= p1 && p1 <= means[0].2);
    let functional = bound(sources(), &catalog);
    assert_eq!(mz_interval_withheld_reason(&functional, &data).unwrap(), None);
    let draws =
        mz_transport_bootstrap_law_draws(&functional, &data, 25, &ExecutionContext::for_tests(4))
            .unwrap()
            .unwrap();
    assert!(draws.iter().flatten().count() == 25);
    for drawn in draws.iter().flatten() {
        let law =
            |regime: u32| drawn.laws().iter().find(|law| law.regime().raw() == regime).unwrap();
        let joint = law(B_TRIAL_REGIME).empirical_counts().unwrap();
        let margin = law(SHARED_REGIME).empirical_counts().unwrap();
        // The margin is exactly the (X, Z2) margin of this replicate's joint.
        let expected = joint.chunks(2).map(|pair| pair.iter().sum()).collect::<Vec<u64>>();
        assert_eq!(margin, expected.as_slice());
        assert_eq!(joint.iter().sum::<u64>(), margin.iter().sum::<u64>());
    }
    // A margin whose counts are not the joint's margin is not that projection.
    let mut laws = data.laws().to_vec();
    let last = laws.pop().unwrap();
    let mut counts = last.empirical_counts().unwrap().to_vec();
    counts[0] += 1;
    counts[1] -= 1;
    #[expect(clippy::cast_precision_loss, reason = "fixture counts fit f64 exactly")]
    let total = counts.iter().sum::<u64>() as f64;
    #[expect(clippy::cast_precision_loss, reason = "fixture counts fit f64 exactly")]
    let probabilities = counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>();
    laws.push(
        antecedent_expr::ExactDiscreteLaw::try_empirical(
            last.population(),
            last.regime(),
            last.interventions().to_vec(),
            last.axes().to_vec(),
            probabilities,
            last.snapshot_identity(),
            last.tolerance(),
        )
        .unwrap()
        .with_empirical_counts(counts)
        .unwrap(),
    );
    let tampered =
        antecedent_expr::ExactTransportData::try_new(laws, data.max_support_rows()).unwrap();
    assert_eq!(
        mz_interval_withheld_reason(&functional, &tampered).unwrap(),
        Some("transport.unsupported_dependence")
    );
}

#[test]
fn unrelatable_tables_of_one_dataset_withhold_the_interval_not_the_point() {
    let (catalog, exact) = evidence();
    let catalog = with_conflicting_a_trial(&catalog);
    let data = empirical(&exact, 40_000.0);
    let functional = bound(sources(), &catalog);
    // The catalog declares one dataset for both arms, so dependence is known ...
    assert_eq!(mz_sampling_dependence(&functional), Ok(()));
    // ... but the two arms' different tables cannot be one resampled table.
    let (point, run) = joint_bootstrap(&catalog, &data);
    assert!((point - truth(1)).abs() < 1e-2);
    assert_eq!(run.unwrap_err(), "transport.unsupported_dependence");
    assert_eq!(
        mz_interval_withheld_reason(&functional, &data).unwrap(),
        Some("transport.unsupported_dependence")
    );
}

/// Two sources that both certify `Q[Y]` (departure from Fig. 3 line 11). The paper
/// combines the certifying sources by a weighted sum; this contract returns the
/// first in canonical order. Each is a valid exact formula, so the numbers agree
/// with the enumerated target truth whichever source is used, whatever the
/// declaration order, and any weighted combination of the two.
#[test]
fn two_sources_certifying_one_factor_give_the_same_truth_either_way() {
    let (catalog, data) = evidence_with_twin();
    let ctx = ExecutionContext::for_tests(3);
    let by_name = |names: &[&str]| {
        let pool = twin_sources();
        names
            .iter()
            .map(|name| pool.iter().find(|s| s.population.as_ref() == *name).unwrap().clone())
            .collect::<Vec<_>>()
    };
    let point = |functional: &BoundMzTransportFunctional, x: u8| {
        risk_of(
            &evaluate_exact_mz_transport(
                functional,
                data.clone(),
                request(x == 1),
                ExactEvaluationLimits::default(),
                &ctx,
            )
            .unwrap(),
        )
    };
    // a and c both certify Q[Y]; the canonical first (a) is the one cited.
    let with_both = bound(by_name(&["a", "b", "c"]), &catalog);
    assert!(
        with_both.cited_regimes().iter().all(|r| r.raw() <= 3),
        "{:?}",
        with_both.cited_regimes()
    );
    let reversed = bound(by_name(&["c", "b", "a"]), &catalog);
    assert_eq!(with_both.cited_regimes(), reversed.cited_regimes());
    // Each certifying source alone, in both orders.
    let via_a = bound(by_name(&["a", "b"]), &catalog);
    let via_c = bound(by_name(&["c", "b"]), &catalog);
    assert!(via_c.cited_regimes().iter().any(|r| r.raw() >= 4), "c's regimes are cited");
    assert!(via_c.cited_regimes().iter().all(|r| r.raw() == 0 || r.raw() >= 3));
    for x in [0u8, 1] {
        let (both, backward) = (point(&with_both, x), point(&reversed, x));
        let (a, c) = (point(&via_a, x), point(&via_c, x));
        assert_eq!(both.to_bits(), backward.to_bits(), "order invariance at do(X={x})");
        assert_eq!(both.to_bits(), a.to_bits(), "the first canonical source is the one used");
        // The two certifying sources agree with each other and with the truth.
        assert!((a - c).abs() < 1e-12, "do(X={x}): a {a}, c {c}");
        assert!((a - truth(x)).abs() < 1e-12, "do(X={x}): a {a}, truth {}", truth(x));
        // The paper's weighted combination of the two is the same number.
        for weight in [0.25, 0.5, 0.75] {
            let mixed = weight * a + (1.0 - weight) * c;
            assert!((mixed - truth(x)).abs() < 1e-12, "weight {weight} at do(X={x})");
        }
    }
    // The twin differs from `a` as a population: its own laws are not a's.
    let law_of = |regime: u32| {
        data.laws()
            .iter()
            .find(|law| law.regime().raw() == regime)
            .unwrap()
            .probabilities()
            .to_vec()
    };
    assert!(law_of(1).iter().zip(law_of(4)).any(|(p, q)| (p - q).abs() > 1e-3));
}

/// R-443 Figure 2 analogue (`Z -> X -> Y`, `X <-> Y`; source `a` runs `do(X)` but
/// its selection node points into `Y`, source `b` runs `do(Z)` and its selection
/// node points into `X`). The obstruction the search certifies is real: two fully
/// specified SCMs agree on the target's observational law and on every
/// experiment either source supplies, yet give different `P*(y | do(x))`. The
/// exact Figure 2 graph was not available offline, so the counterexample is our
/// own for this graph family, not the paper's Eqs. 3-4. The bow arc `X <-> Y` is
/// the witness: model 1 has no effect of `X` on `Y` (`Y = U`), model 2 copies `X`
/// (`Y = X`), and with `X = U` the two are observationally identical. Source `a`
/// draws `Y` from an independent coin (the mechanism its selection node may
/// change) and source `b` runs the target's own mechanisms, so every law
/// supplied is the same in both models.
#[test]
fn fig_2_analogue_obstruction_is_real_two_models_agree_on_every_experiment() {
    const Z: usize = 0;
    const XV: usize = 1;
    const YV: usize = 2;
    // e0 = Z's coin, e1 = the X <-> Y confounder U, e2 = source a's own Y coin.
    let model = |copies_x: bool| -> [Scm; 3] {
        let x_is_u = || -> Mechanism { Box::new(|_, e| e[1]) };
        let z_coin = || -> Mechanism { Box::new(|_, e| e[0]) };
        let target_y = move || -> Mechanism {
            if copies_x { Box::new(|v, _| v[XV]) } else { Box::new(|_, e| e[1]) }
        };
        let scm =
            |y: Mechanism| Scm { n: 3, exo_p: vec![0.5, 0.4, 0.3], f: vec![z_coin(), x_is_u(), y] };
        // target, source a (Y differs), source b (the target's mechanisms)
        [scm(target_y()), scm(Box::new(|_, e| e[2])), scm(target_y())]
    };
    let evidence_of = |copies_x: bool| {
        let [target, a, b] = model(copies_x);
        let specs = [
            Spec { population: "target", kind: RegimeKind::Observational, assignments: vec![] },
            Spec { population: "a", kind: RegimeKind::Observational, assignments: vec![] },
            Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(XV, 0)] },
            Spec { population: "a", kind: RegimeKind::Experimental, assignments: vec![(XV, 1)] },
            Spec { population: "b", kind: RegimeKind::Observational, assignments: vec![] },
            Spec { population: "b", kind: RegimeKind::Experimental, assignments: vec![(Z, 0)] },
            Spec { population: "b", kind: RegimeKind::Experimental, assignments: vec![(Z, 1)] },
        ];
        let scm_for = |population: &str| -> &Scm {
            match population {
                "a" => &a,
                "b" => &b,
                _ => &target,
            }
        };
        let evidence = build(&scm_for, 3, &specs, &["target", "a", "b"]);
        let truth = |x: u8| model(copies_x)[0].risk(&[(XV, x)], YV);
        (evidence, truth(0), truth(1))
    };
    let ((catalog, data_1), t1_0, t1_1) = evidence_of(false);
    let ((_, data_2), t2_0, t2_1) = evidence_of(true);

    // Every supplied law is identical across the two models ...
    assert_eq!(data_1.laws().len(), data_2.laws().len());
    for (l1, l2) in data_1.laws().iter().zip(data_2.laws()) {
        assert_eq!((l1.population(), l1.regime()), (l2.population(), l2.regime()));
        let worst = l1
            .probabilities()
            .iter()
            .zip(l2.probabilities())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-15,
            "{} regime {}: laws differ by {worst}",
            l1.population(),
            l1.regime().raw()
        );
    }
    // ... yet the target's interventional effect is not: X has no effect in
    // model 1 and copies itself in model 2.
    assert!((t1_0 - 0.4).abs() < 1e-12 && (t1_1 - 0.4).abs() < 1e-12);
    assert!((t2_0 - 0.0).abs() < 1e-12 && (t2_1 - 1.0).abs() < 1e-12);
    assert!((t1_1 - t2_1).abs() > 0.5);

    // The search certifies exactly this: a checked obstruction, replayed.
    let graph = diagram(3, &[(Z, XV), (XV, YV)], &[(XV, YV)], &[]).causal_graph().clone();
    let source = |name: &str, controllable: usize, target: usize| ZTransportSourceSpec {
        population: Arc::from(name),
        controllable: Arc::from([vid(controllable)]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: vid(controllable),
            value: Value::Bool(false),
        }]),
        selection_targets: Arc::from([vid(target)]),
    };
    let q = antecedent_identify::MzTransportQuery {
        outcomes: Arc::from([vid(YV)]),
        treatments: Arc::from([vid(XV)]),
        target: Arc::from("target"),
        sources: Arc::from([source("a", XV, YV), source("b", Z, XV)]),
    };
    let ctx = ExecutionContext::for_tests(1);
    let decision =
        decide_mz_transport(&graph, &q, &catalog, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap();
    let MzTransportDecision::ProvenNonTransportable(obstruction) = decision else {
        panic!("the Figure 2 analogue is non-transportable, got {decision:?}");
    };
    verify_mz_transport_obstruction(
        &graph,
        &obstruction,
        antecedent_identify::SidLimits::default(),
        &ctx,
    )
    .unwrap();
}
