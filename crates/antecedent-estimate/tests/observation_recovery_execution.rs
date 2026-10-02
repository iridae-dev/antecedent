//! 2.2B X10: exact evaluation of recovery formulas against enumerated SCM truth,
//! the downstream effect, the bounded-class sweep and the observed-law refusals.
#![allow(clippy::cast_possible_truncation, reason = "test indices are small")]

mod support {
    pub mod recovery_scm;
}

use std::sync::Arc;

use antecedent_core::{ExecutionContext, LawOrigin, Value};
use antecedent_estimate::{evaluate_exact_recovery, evaluate_recovered_effect};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactTransportData, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    RecoveredEffectQuery, RecoveryDecision, RecoveryDerivation, RecoveryDetail, RecoveryLimits,
    WitnessMechanism, decide_observation_recovery, verify_recovery_witness,
};
use support::recovery_scm::{MModel, POPULATION, Rng, dags, v};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn derive(model: &MModel, effect: Option<&RecoveredEffectQuery>) -> Box<RecoveryDerivation> {
    match decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        effect,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap()
    {
        RecoveryDecision::Recovered(d) => d,
        RecoveryDecision::NonRecoverable(_) => panic!("expected recoverable"),
    }
}

fn close(a: &[f64], b: &[f64], tolerance: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
}

/// The complete-case analysis `P(X, O | R = 1)` a wrong implementation would give.
fn complete_case(model: &MModel) -> Vec<f64> {
    let law = model.observed_law();
    let s = (model.k + model.m) as usize;
    let mut out = vec![0.0; 1 << s];
    let vars = model.observed_variables();
    let proxy_start = 2 * model.k + model.m;
    let card = |n: u32| if n >= proxy_start { 3 } else { 2 };
    let mut total = 0.0;
    for (index, p) in law.probabilities().iter().enumerate() {
        let mut rest = index;
        let mut levels = vec![0usize; vars.len()];
        for (i, n) in vars.iter().enumerate().rev() {
            levels[i] = rest % card(*n);
            rest /= card(*n);
        }
        let level = |node: u32| levels[vars.iter().position(|n| *n == node).unwrap()];
        if *p == 0.0 || (0..model.k).any(|i| level(model.r(i)) != 1) {
            continue;
        }
        let target = (0..model.k)
            .map(|i| level(model.proxy(i)))
            .chain((0..model.m).map(|j| level(model.o(j))));
        out[target.fold(0usize, |acc, b| (acc << 1) | b)] += p;
        total += p;
    }
    out.iter().map(|p| p / total).collect()
}

#[test]
fn k1_recovery_matches_the_enumerated_truth() {
    // O -> X, O -> R: R _||_ X | O, so P(X, O) = P(X | O, R = 1) P(O).
    let model = MModel::new(1, 1, &[(1, 0)], &[vec![1]], 21).unwrap();
    let derivation = derive(&model, None);
    let recovered = evaluate_exact_recovery(&derivation, &model.observed_law(), &ctx()).unwrap();
    let truth = model.truth();
    assert!(
        close(recovered.law().probabilities(), &truth, 1e-12),
        "{:?} vs {truth:?}",
        recovered.law().probabilities()
    );
    // The fixture discriminates: complete-case analysis is wrong here.
    let naive = complete_case(&model);
    assert!(naive.iter().zip(&truth).any(|(a, b)| (a - b).abs() > 1e-3), "{naive:?} {truth:?}");
    // The derived law carries its provenance and never claims to be observed.
    assert!(
        matches!(&recovered.descriptor().origin, LawOrigin::Recovered { derivation: d } if d.as_ref() == derivation.identity())
    );
    // The table's own provenance comes from the derivation, never the observed
    // snapshot, so it cannot pose as the observed pattern law: offered back as the
    // observed law, it is refused on its snapshot before anything is evaluated.
    assert_eq!(recovered.law().snapshot_identity(), format!("recovered:{}", derivation.identity()));
    assert_ne!(recovered.law().snapshot_identity(), model.observed_law().snapshot_identity());
    assert_eq!(
        recovered.law().snapshot_identity(),
        format!("recovered:{}", recovered.derivation_identity())
    );
    let posed = evaluate_exact_recovery(&derivation, recovered.law(), &ctx()).unwrap_err();
    assert_eq!(posed.detail, antecedent_identify::RecoveryDetail::InvalidObservedLaw);
    assert!(posed.message.contains("snapshot"), "{posed}");
    // The descriptor never supplies a population law to a catalog route.
    assert!(recovered.descriptor().origin != LawOrigin::Measured);
    // Closed form P(x | o, R = 1) P(o) straight from the table.
    let law = model.observed_law();
    let p = law.probabilities(); // axes O, R, X* (levels 0, 1, ?)
    let at = |o: usize, r: usize, x: usize| p[o * 6 + r * 3 + x];
    for x in 0..2 {
        for o in 0..2 {
            let p_o: f64 =
                (0..2).flat_map(|r| (0..3).map(move |s| (r, s))).map(|(r, s)| at(o, r, s)).sum();
            let cc = at(o, 1, x) / (at(o, 1, 0) + at(o, 1, 1));
            assert!((recovered.law().probabilities()[x * 2 + o] - cc * p_o).abs() < 1e-12);
        }
    }
}

#[test]
fn k2_cross_censoring_matches_the_enumerated_truth() {
    // X0 -> X1, X0 -> R1, X1 -> R0: each missingness depends on the other variable.
    let model = MModel::new(2, 0, &[(0, 1)], &[vec![1], vec![0]], 8).unwrap();
    let derivation = derive(&model, None);
    let record = derivation.record();
    assert_eq!(record.factors[0].conditioning, vec![3, 5], "p(R0 = 1 | X1) via (R1, X*1)");
    assert_eq!(record.factors[1].conditioning, vec![2, 4], "p(R1 = 1 | X0) via (R0, X*0)");
    let recovered = evaluate_exact_recovery(&derivation, &model.observed_law(), &ctx()).unwrap();
    let truth = model.truth();
    assert!(
        close(recovered.law().probabilities(), &truth, 1e-12),
        "{:?} vs {truth:?}",
        recovered.law().probabilities()
    );
    let naive = complete_case(&model);
    assert!(naive.iter().zip(&truth).any(|(a, b)| (a - b).abs() > 1e-3));
    // Hand-derived: P(x0, x1) = P(R = 11, x0, x1) / [p(R0 = 1 | x1) p(R1 = 1 | x0)],
    // p(R0 = 1 | x1) = P(R0 = 1, R1 = 1, X*1 = x1) / P(R1 = 1, X*1 = x1).
    let law = model.observed_law();
    let p = law.probabilities(); // axes R0, R1, X*0, X*1
    let at = |r0: usize, r1: usize, a: usize, b: usize| p[((r0 * 2 + r1) * 3 + a) * 3 + b];
    let sum = |f: &dyn Fn(usize, usize, usize, usize) -> bool| -> f64 {
        let mut total = 0.0;
        for r0 in 0..2 {
            for r1 in 0..2 {
                for a in 0..3 {
                    for b in 0..3 {
                        if f(r0, r1, a, b) {
                            total += at(r0, r1, a, b);
                        }
                    }
                }
            }
        }
        total
    };
    for x0 in 0..2 {
        for x1 in 0..2 {
            let p_r0 = sum(&|r0, r1, _, b| r0 == 1 && r1 == 1 && b == x1)
                / sum(&|_, r1, _, b| r1 == 1 && b == x1);
            let p_r1 = sum(&|r0, r1, a, _| r0 == 1 && r1 == 1 && a == x0)
                / sum(&|r0, _, a, _| r0 == 1 && a == x0);
            let hand = at(1, 1, x0, x1) / (p_r0 * p_r1);
            assert!((hand - truth[x0 * 2 + x1]).abs() < 1e-12);
            assert!((recovered.law().probabilities()[x0 * 2 + x1] - hand).abs() < 1e-12);
        }
    }
}

fn effect_graph(model: &MModel, edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in edges {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
}

fn at_level(variable: u32, level: f64) -> Assignment {
    Assignment::from_pairs([(v(variable), Value::f64(level))])
}

#[test]
fn the_recovered_law_feeds_a_backdoor_effect_equal_to_the_complete_data_effect() {
    // Treatment X0 and outcome X1 both partially observed, confounder O fully
    // observed: O -> X0, O -> X1, X0 -> X1. The treatment's missingness depends on O,
    // the outcome's on the treatment.
    let edges = [(2, 0), (2, 1), (0, 1)];
    let model = MModel::new(2, 1, &edges, &[vec![2], vec![0]], 17).unwrap();
    let effect = RecoveredEffectQuery {
        graph: effect_graph(&model, &edges),
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
    };
    let derivation = derive(&model, Some(&effect));
    let recovered = evaluate_exact_recovery(&derivation, &model.observed_law(), &ctx()).unwrap();
    let identified = derivation.effect().unwrap();
    for level in [0usize, 1] {
        let point = evaluate_recovered_effect(
            &derivation,
            &recovered,
            at_level(0, level as f64),
            ExactEvaluationLimits::default(),
            &ctx(),
        )
        .unwrap();
        let p1 = point.probabilities
            [point.atoms.iter().position(|a| a[0].as_f64() == Some(1.0)).unwrap()];
        // Equal to the enumerated interventional truth ...
        assert!((p1 - model.interventional(1, 0, level)).abs() < 1e-12, "level {level}");
        // ... and to the same identified formula on the complete-data law.
        let complete = ExactTransportData::try_new(vec![model.complete_law()], 1 << 10)
            .unwrap()
            .with_world_bound_leaves(vec![model.complete_law().regime()]);
        let plan = ExactEvaluationPlan::compile(
            identified.arena(),
            identified.root(),
            complete,
            vec![v(1)],
            at_level(0, level as f64),
            ExactEvaluationLimits::default(),
            LawTolerance::default(),
            &ctx(),
        )
        .unwrap();
        let reference = plan.evaluate(&ctx()).unwrap();
        assert!(close(&point.probabilities, &reference.probabilities, 1e-12));
    }
    // The fixture discriminates: the confounded conditional P(X1 = 1 | X0 = 1)
    // (axes X0, X1, O) differs from the effect.
    let truth = model.truth();
    let cell = |x0: usize, x1: usize| truth[(x0 * 2 + x1) * 2] + truth[(x0 * 2 + x1) * 2 + 1];
    let conditional = cell(1, 1) / (cell(1, 0) + cell(1, 1));
    assert!((conditional - model.interventional(1, 0, 1)).abs() > 1e-3, "{conditional}");
}

#[test]
fn the_effect_handoff_refuses_a_mismatched_law() {
    let edges = [(2, 0), (2, 1), (0, 1)];
    let model = MModel::new(2, 1, &edges, &[vec![2], vec![0]], 17).unwrap();
    let effect = RecoveredEffectQuery {
        graph: effect_graph(&model, &edges),
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
    };
    let derivation = derive(&model, Some(&effect));
    let recovered = evaluate_exact_recovery(&derivation, &model.observed_law(), &ctx()).unwrap();
    // A law recovered by another derivation (no effect requested): refused.
    let other = derive(&model, None);
    let elsewhere = evaluate_exact_recovery(&other, &model.observed_law(), &ctx()).unwrap();
    let error = evaluate_recovered_effect(
        &derivation,
        &elsewhere,
        at_level(0, 1.0),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::HandoffMismatch);
    // A derivation without an effect cannot evaluate one.
    let error = evaluate_recovered_effect(
        &other,
        &elsewhere,
        at_level(0, 1.0),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::InvalidQuery);
    // The matching law evaluates.
    evaluate_recovered_effect(
        &derivation,
        &recovered,
        at_level(0, 1.0),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_eq!(RecoveryDetail::HandoffMismatch.reason_code(), "invalid_argument");
}

fn relabel(law: &ExactDiscreteLaw, probabilities: Vec<f64>) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        law.population(),
        law.regime(),
        Vec::new(),
        law.axes().to_vec(),
        probabilities,
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap()
}

#[test]
fn observed_laws_outside_the_contract_are_refused() {
    let model = MModel::new(2, 0, &[(0, 1)], &[vec![1], vec![0]], 8).unwrap();
    let derivation = derive(&model, None);
    let law = model.observed_law();
    let refuse = |observed: &ExactDiscreteLaw| {
        evaluate_exact_recovery(&derivation, observed, &ctx()).unwrap_err()
    };
    // Positivity: a complete-case cell with no mass (its mass moved to a
    // missing-pattern cell, so the table stays a law consistent with the proxies).
    let mut probabilities = law.probabilities().to_vec();
    // Axes R0, R1, X*0, X*1 (last fastest): index ((r0 * 2 + r1) * 3 + a) * 3 + b.
    let complete = 3 * 3 * 3; // R0 = R1 = 1, X*0 = X*1 = 0
    let missing = 2 * 3 + 2; // R0 = R1 = 0, X*0 = X*1 = ?
    probabilities[missing] += probabilities[complete];
    probabilities[complete] = 0.0;
    let error = refuse(&relabel(&law, probabilities));
    assert_eq!(error.detail, RecoveryDetail::Positivity);
    assert_eq!(error.reason_code(), "transport_support_failure");
    // Mass on a cell the deterministic proxy excludes (R0 = 1, X*0 = ?).
    let mut probabilities = law.probabilities().to_vec();
    let impossible = (3 * 3 + 2) * 3; // R0 = R1 = 1, X*0 = ?, X*1 = 0
    probabilities[impossible] += probabilities[complete];
    probabilities[complete] = 0.0;
    assert_eq!(refuse(&relabel(&law, probabilities)).detail, RecoveryDetail::InvalidObservedLaw);
    // A counted (empirical) law is a sampled provider: not licensed.
    let counts: Vec<u64> = law.probabilities().iter().map(|_| 1).collect();
    let uniform = relabel(&law, vec![1.0 / counts.len() as f64; counts.len()]);
    let empirical = uniform.with_empirical_counts(counts).unwrap();
    let error = refuse(&empirical);
    assert_eq!(error.detail, RecoveryDetail::EmpiricalNotLicensed);
    assert_eq!(error.reason_code(), "cell_not_licensed");
    // Another snapshot than the catalog binds.
    let other = ExactDiscreteLaw::try_new(
        POPULATION,
        law.regime(),
        Vec::new(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        "snap-other",
        LawTolerance::default(),
    )
    .unwrap();
    assert_eq!(refuse(&other).detail, RecoveryDetail::InvalidObservedLaw);
    // The missing level spelled differently ("NA"): never aligned.
    let mut axes = law.axes().to_vec();
    axes[2] = DiscreteAxis {
        variable: axes[2].variable,
        values: Arc::from([Value::f64(0.0), Value::f64(1.0), Value::Label(Arc::from("NA"))]),
    };
    let na = ExactDiscreteLaw::try_new(
        POPULATION,
        law.regime(),
        Vec::new(),
        axes,
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        LawTolerance::default(),
    )
    .unwrap();
    assert_eq!(refuse(&na).detail, RecoveryDetail::InvalidObservedLaw);
    // A non-binary substantive axis (a third level on R0's response coordinate
    // would also do): the binary bound is enforced on the table, not only on the
    // declared domain.
    let mut axes = law.axes().to_vec();
    axes[0] = DiscreteAxis {
        variable: axes[0].variable,
        values: Arc::from([Value::f64(0.0), Value::f64(1.0), Value::f64(2.0)]),
    };
    let mut wide = vec![0.0; law.probabilities().len() / 2 * 3];
    // Axes R0 (now 3 levels), R1, X*0, X*1: the old cells keep their mass at R0 in {0, 1}.
    for (i, p) in law.probabilities().iter().enumerate() {
        wide[i] = *p;
    }
    let ternary = ExactDiscreteLaw::try_new(
        POPULATION,
        law.regime(),
        Vec::new(),
        axes,
        wide,
        law.snapshot_identity(),
        LawTolerance::default(),
    )
    .unwrap();
    let error = refuse(&ternary);
    assert_eq!(error.detail, RecoveryDetail::InvalidObservedLaw);
    assert!(error.message.contains("outside its domain"), "{error}");
    // A law consistent with the proxies that is no law of the m-graph: the
    // recovered masses do not normalize.
    let mut rng = Rng::new(99);
    let mut arbitrary = vec![0.0; law.probabilities().len()];
    for (i, p) in law.probabilities().iter().enumerate() {
        if *p > 0.0 {
            arbitrary[i] = rng.prob();
        }
    }
    let total: f64 = arbitrary.iter().sum();
    let arbitrary: Vec<f64> = arbitrary.iter().map(|p| p / total).collect();
    let error = refuse(&relabel(&law, arbitrary));
    assert_eq!(error.detail, RecoveryDetail::InvalidObservedLaw);
    assert!(error.message.contains("not the law of any model of the m-graph"), "{error}");
}

#[test]
fn variable_and_response_permutations_leave_the_recovered_law_unchanged() {
    // The same SCM with X0 and X1 (and their responses and proxies) swapped.
    let model = MModel::new(2, 1, &[(2, 0), (0, 1)], &[vec![2], vec![0]], 31).unwrap();
    let mut swapped = MModel::new(2, 1, &[(2, 1), (1, 0)], &[vec![1], vec![2]], 31).unwrap();
    // Carry the mechanisms across the relabelling 0 <-> 1, 3 <-> 4.
    let map = |n: u32| match n {
        0 => 1,
        1 => 0,
        3 => 4,
        4 => 3,
        other => other,
    };
    for (node, table) in &model.mechanisms {
        swapped.mechanisms.insert(map(*node), table.clone());
    }
    let a = evaluate_exact_recovery(&derive(&model, None), &model.observed_law(), &ctx()).unwrap();
    let b =
        evaluate_exact_recovery(&derive(&swapped, None), &swapped.observed_law(), &ctx()).unwrap();
    // Axes (X0, X1, O): swapping X0 and X1 permutes the cells accordingly.
    for x0 in 0..2 {
        for x1 in 0..2 {
            for o in 0..2 {
                let pa = a.law().probabilities()[(x0 * 2 + x1) * 2 + o];
                let pb = b.law().probabilities()[(x1 * 2 + x0) * 2 + o];
                assert!((pa - pb).abs() < 1e-12);
            }
        }
    }
    assert!(close(a.law().probabilities(), &model.truth(), 1e-12));
}

#[test]
fn the_colluder_archetype_has_a_recoverable_target() {
    // X0 -> R1 <- R0 (a colluder). The class refuses it as an unsupported
    // mechanism; this checks, by enumeration, that the refusal is not hiding a
    // nonrecoverable target: P(X0, X1) = P(R = 11, x0, x1) / [p(R0 = 1) p(R1 = 1 | x0, R0 = 1)].
    let mut model = MModel::new(2, 0, &[(0, 1)], &[vec![], vec![0]], 13).unwrap();
    model.graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    model.mechanisms.insert(3, vec![0.3, 0.6, 0.7, 0.2]);
    let truth = model.truth();
    let law = model.observed_law();
    let p = law.probabilities(); // R0, R1, X*0, X*1
    let at = |r0: usize, r1: usize, a: usize, b: usize| p[((r0 * 2 + r1) * 3 + a) * 3 + b];
    let r0: f64 = (0..2)
        .flat_map(|r1| (0..3).flat_map(move |a| (0..3).map(move |b| (r1, a, b))))
        .map(|(r1, a, b)| at(1, r1, a, b))
        .sum();
    for x0 in 0..2 {
        let joint: f64 = (0..3).map(|b| at(1, 1, x0, b)).sum();
        let cond: f64 = (0..2)
            .flat_map(|r1| (0..3).map(move |b| (r1, b)))
            .map(|(r1, b)| at(1, r1, x0, b))
            .sum();
        for x1 in 0..2 {
            let formula = at(1, 1, x0, x1) / (r0 * (joint / cond));
            assert!((formula - truth[x0 * 2 + x1]).abs() < 1e-12);
        }
    }
    let error = decide_observation_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        None,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, RecoveryDetail::UnsupportedMechanism);
}

/// The test SCM with a witness model's mechanisms (`k / 60`, graph parent order),
/// checked to be positive and one mechanism per non-proxy node on its graph parents.
fn witness_model(model: &MModel, mechanisms: &[WitnessMechanism]) -> MModel {
    let mut out = model.clone();
    assert_eq!(mechanisms.len(), (2 * model.k + model.m) as usize);
    for mechanism in mechanisms {
        let parents: Vec<u32> = model
            .graph
            .parents(DenseNodeId::from_raw(mechanism.node))
            .iter()
            .map(|p| p.raw())
            .collect();
        assert_eq!(mechanism.parents, parents, "a witness mechanism is not on its graph parents");
        assert!(mechanism.numerators.iter().all(|k| *k > 0 && *k < 60), "non-positive witness");
        let table = mechanism.numerators.iter().map(|k| f64::from(*k) / 60.0).collect();
        assert!(out.mechanisms.insert(mechanism.node, table).is_some(), "not a non-proxy node");
    }
    out
}

/// Every graph of one (k, m) size, or a seeded sample of `limit` of them.
fn sweep(k: u32, m: u32, limit: Option<usize>, seed: u64) -> (usize, usize, usize) {
    let s = k + m;
    let substantive = dags(s);
    let parent_sets = 1u64 << s;
    let total = substantive.len() as u64 * parent_sets.pow(k);
    let mut rng = Rng::new(seed);
    let picks: Vec<u64> = match limit {
        None => (0..total).collect(),
        Some(n) => (0..n).map(|_| rng.below(total)).collect(),
    };
    let (mut recovered, mut witnessed, mut gaps) = (0, 0, 0);
    for (i, pick) in picks.iter().enumerate() {
        let mut rest = *pick;
        let edges = &substantive[(rest % substantive.len() as u64) as usize];
        rest /= substantive.len() as u64;
        let responses: Vec<Vec<u32>> = (0..k)
            .map(|_| {
                let mask = rest % parent_sets;
                rest /= parent_sets;
                (0..s).filter(|b| (mask >> b) & 1 == 1).collect()
            })
            .collect();
        let model = MModel::new(k, m, edges, &responses, seed ^ (i as u64)).unwrap();
        let violating = (0..k).any(|i| responses[i as usize].contains(&i));
        match decide_observation_recovery(
            &model.graph,
            &model.query(),
            &model.catalog(),
            None,
            RecoveryLimits::default(),
            &ctx(),
        ) {
            Ok(RecoveryDecision::Recovered(d)) => {
                assert!(
                    !violating,
                    "a self-censoring graph was recovered: {edges:?} {responses:?}"
                );
                let law = evaluate_exact_recovery(&d, &model.observed_law(), &ctx()).unwrap();
                assert!(
                    close(law.law().probabilities(), &model.truth(), 1e-10),
                    "{edges:?} {responses:?}"
                );
                recovered += 1;
            }
            Ok(RecoveryDecision::NonRecoverable(w)) => {
                assert!(violating, "a compliant graph was declared nonrecoverable");
                verify_recovery_witness(&model.graph, &model.query(), &w).unwrap();
                // Independently of the library's verifier: both witness models,
                // enumerated by the test SCM, give the same observed pattern law
                // and different targets.
                assert_eq!(w.edge.1, model.r(w.edge.0), "the witness names a self-censoring edge");
                let (a, b) = (witness_model(&model, &w.first), witness_model(&model, &w.second));
                assert!(
                    close(
                        a.observed_law().probabilities(),
                        b.observed_law().probabilities(),
                        1e-12
                    ),
                    "witness models disagree on the observed law: {edges:?} {responses:?}"
                );
                assert!(!close(&a.truth(), &b.truth(), 1e-6), "witness targets agree: {edges:?}");
                witnessed += 1;
            }
            Err(e) if e.detail == RecoveryDetail::WitnessUnavailable => gaps += 1,
            Err(e) => panic!("{e}"),
        }
    }
    (recovered, witnessed, gaps)
}

#[test]
fn every_graph_of_the_bounded_class_matches_truth_or_has_a_verified_witness() {
    // Exhaustive for up to three substantive nodes with k <= 2, and k = 3 with no
    // fully observed variable; seeded samples at the larger sizes.
    let mut totals = (0, 0, 0);
    for (k, m, limit) in [
        (1, 0, None),
        (1, 1, None),
        (1, 2, None),
        (2, 0, None),
        (2, 1, None),
        (3, 0, None),
        (2, 2, Some(400)),
        (3, 1, Some(300)),
        (3, 2, Some(150)),
    ] {
        let (r, w, g) = sweep(k, m, limit, 7 + u64::from(k * 10 + m));
        assert!(r > 0 && w > 0, "k={k} m={m}: {r} recovered, {w} witnessed");
        totals = (totals.0 + r, totals.1 + w, totals.2 + g);
    }
    // Non-vacuous counts; the listed-gap set is empty in this class.
    assert!(totals.0 >= 2000, "{totals:?}");
    assert!(totals.1 >= 12000, "{totals:?}");
    assert_eq!(totals.2, 0, "listed gaps (witness unavailable): {totals:?}");
}
