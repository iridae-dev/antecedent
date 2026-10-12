//! Tests of the cross-world cell: check, evaluation, artifact replay and the
//! independent closed form.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::closed_form::{closed_form_agrees, closed_form_point, ols_on_parents};
use super::*;
use crate::gcm::NestedCounterfactualOperation;
use antecedent_core::NestedCounterfactualQuery;
use antecedent_graph::DenseNodeId;

#[test]
fn ill_posed_abduction_keeps_the_invalid_query_detail() {
    let error =
        evaluation_error(CounterfactualError::AbductionNotExact { message: "probe".into() });
    let text = error.to_string();
    assert!(
        text.contains("cross_world.invalid_query: abduction is not exact inversion: probe"),
        "{text}"
    );
    assert!(text.contains("invalid_argument"), "{text}");
    assert!(matches!(
        evaluation_error(CounterfactualError::Cancelled),
        CausalError::Cancelled { stage: "cross_world" }
    ));
}

fn mediation_columns(noisy: bool) -> Vec<Vec<f64>> {
    let n = 400usize;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    let m: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&m)
        .enumerate()
        .map(|(i, (x, m))| {
            1.7 * x + 4.0 * m + if noisy { (i as f64 * 1.7 + 0.3).sin() } else { 0.0 }
        })
        .collect();
    vec![x, m, y]
}

/// The closed-form implementation is checked against the structural truth
/// (the sample disturbances are only approximately orthogonal to the
/// regressors, so within 0.1) and against the evaluator (1e-9) for every edge
/// set: the outcome disturbance cancels, the mediator path contributes
/// 4 * 0.8 * 3.
#[test]
fn the_closed_form_recomputation_matches_the_structural_truth() {
    let edges = [(0, 1), (0, 2), (1, 2)];
    let v = VariableId::from_raw;
    let all = edges.map(|(a, b)| (v(a), v(b)));
    for noisy in [false, true] {
        let columns = mediation_columns(noisy);
        for mask in 0u32..8 {
            let chosen: Vec<_> = all
                .iter()
                .enumerate()
                .filter(|(k, _)| (mask >> k) & 1 == 1)
                .map(|(_, e)| *e)
                .collect();
            let query =
                CrossWorldQuery::path_specific(v(0), v(2), -1.0, 2.0, &all, &chosen).unwrap();
            let truth = 1.7 * 3.0 * f64::from(u8::from(chosen.contains(&all[1])))
                + 4.0
                    * 0.8
                    * 3.0
                    * f64::from(u8::from(chosen.contains(&all[0]) && chosen.contains(&all[2])));
            let closed = closed_form_point(
                3,
                &edges,
                &columns,
                &query,
                NestedOutcomeMechanism::LinearGaussian,
                2,
            )
            .unwrap();
            assert!((closed - truth).abs() < 0.1, "noisy={noisy} mask={mask}: {closed} vs {truth}");
            let data = TabularData::from_f64_columns([
                ("x", columns[0].as_slice()),
                ("m", columns[1].as_slice()),
                ("y", columns[2].as_slice()),
            ])
            .unwrap();
            let mut graph = Dag::with_variables(3);
            for (a, b) in edges {
                graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
            let evaluated = evaluate_cross_world_effect(
                graph,
                &data,
                &query,
                CrossWorldOptions::default(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap()
            .point;
            assert!(closed_form_agrees(closed, evaluated), "{mask}: {closed} vs {evaluated}");
            assert!((closed - evaluated).abs() < 1e-9, "{mask}: {closed} vs {evaluated}");
        }
    }
}

/// A singular design has no closed form to compare with, and the comparison
/// tolerance is relative.
#[test]
fn the_closed_form_declines_singular_designs_and_compares_relatively() {
    let v = VariableId::from_raw;
    let x: Vec<f64> = (0..50).map(|i| (f64::from(i) * 0.37).sin()).collect();
    let y: Vec<f64> = (0..50).map(|i| (f64::from(i) * 1.1).cos()).collect();
    // The outcome has two identical parents: X twice cannot happen in a DAG,
    // so make the mediator an exact copy of the treatment.
    let columns = vec![x.clone(), x, y];
    let all = [(v(0), v(1)), (v(0), v(2)), (v(1), v(2))];
    let query = CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &all, &all[1..2]).unwrap();
    assert!(
        closed_form_point(
            3,
            &[(0, 1), (0, 2), (1, 2)],
            &columns,
            &query,
            NestedOutcomeMechanism::LinearGaussian,
            2
        )
        .is_none()
    );
    assert!(closed_form_agrees(1000.0, 1000.0 + 5e-4));
    assert!(!closed_form_agrees(1000.0, 1000.0 + 5e-2));
    assert!(!closed_form_agrees(0.0, 1e-3));
}

/// The natural direct effect as an edge contrast equals the existing nested
/// operation on both the separable and the non-separable mechanism.
#[test]
fn natural_direct_edge_contrast_reproduces_the_existing_nested_operation() {
    let n = 2000usize;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
    let m: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
    // Outcome noise, so the regression covers a per-unit outcome disturbance.
    let y: Vec<f64> = x
        .iter()
        .zip(&m)
        .enumerate()
        .map(|(i, (x, m))| 1.7 * x + 0.5 * m + 0.9 * x * m * m + (i as f64 * 1.7 + 0.3).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let (t, mid, out) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
    let nested = NestedCounterfactualQuery::with_levels(t, mid, out, -1.0, 2.0).unwrap();
    let direct = CrossWorldQuery::natural_direct(t, mid, out, -1.0, 2.0).unwrap();
    let ctx = ExecutionContext::for_tests(8);
    for mechanism in
        [NestedOutcomeMechanism::LinearGaussian, NestedOutcomeMechanism::NonSeparableBasis]
    {
        let mut operation = NestedCounterfactualOperation::compile(graph.clone(), nested).unwrap();
        if mechanism == NestedOutcomeMechanism::NonSeparableBasis {
            operation = operation.with_non_separable_outcome();
        }
        let existing = operation.execute(&data, &ctx).unwrap();
        let mine = evaluate_cross_world_effect(
            graph.clone(),
            &data,
            &direct,
            CrossWorldOptions { mechanism, interval_requested: false },
            &ctx,
        )
        .unwrap()
        .point;
        assert!((mine - existing).abs() < 1e-9, "{mechanism:?}: {mine} vs {existing}");
    }
}

/// A noisy non-separable table: the outcome interacts the treatment with the
/// squared mediator, with a per-unit outcome disturbance.
fn non_separable_table(n: usize) -> (TabularData, Vec<Vec<f64>>, Dag) {
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
    let m: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&m)
        .enumerate()
        .map(|(i, (x, m))| 1.7 * x + 0.5 * m + 0.9 * x * m * m + (i as f64 * 1.7 + 0.3).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    (data, vec![x, m, y], graph)
}

/// The independent basis regression (Givens QR, restated basis, knots,
/// scaling and ridge) reproduces the fitter's point for every edge set, and
/// the check is not vacuous: the point is far from the linear closed form.
#[test]
fn the_non_separable_closed_form_matches_the_fit_for_every_edge_set() {
    let v = VariableId::from_raw;
    let (data, columns, graph) = non_separable_table(1500);
    let edges = [(0, 1), (0, 2), (1, 2)];
    let all = edges.map(|(a, b)| (v(a), v(b)));
    let ctx = ExecutionContext::for_tests(4);
    let mut worst = 0.0_f64;
    for mask in 0u32..8 {
        let chosen: Vec<_> =
            all.iter().enumerate().filter(|(k, _)| (mask >> k) & 1 == 1).map(|(_, e)| *e).collect();
        let query = CrossWorldQuery::path_specific(v(0), v(2), -1.0, 2.0, &all, &chosen).unwrap();
        let replayed = evaluate_cross_world_effect(
            graph.clone(),
            &data,
            &query,
            CrossWorldOptions {
                mechanism: NestedOutcomeMechanism::NonSeparableBasis,
                interval_requested: false,
            },
            &ctx,
        )
        .unwrap()
        .point;
        let closed = closed_form_point(
            3,
            &edges,
            &columns,
            &query,
            NestedOutcomeMechanism::NonSeparableBasis,
            2,
        )
        .expect("a well-conditioned table has a closed form");
        worst = worst.max((closed - replayed).abs() / (1.0 + replayed.abs()));
        assert!(closed_form_agrees(closed, replayed), "{mask}: {closed} vs {replayed}");
        let linear = closed_form_point(
            3,
            &edges,
            &columns,
            &query,
            NestedOutcomeMechanism::LinearGaussian,
            2,
        )
        .unwrap();
        if mask == 0b101 {
            // The natural indirect edge set reads the interaction, which a
            // linear outcome cannot represent.
            assert!(
                (linear - replayed).abs() > 1.0,
                "the linear closed form must not pass as the basis one: {linear} vs {replayed}"
            );
        }
    }
    assert!(worst < 1e-8, "independent solves should agree far inside the tolerance: {worst}");
}

/// Mutating the fitted mechanism (an edited outcome coefficient, and an edited
/// knot) gives a point the replay would accept, because the same evaluator
/// recomputes it from the same edited mechanism, while the independent
/// recomputation from the table disagrees.
#[test]
fn an_edited_mechanism_is_caught_by_the_independent_check_but_not_by_replay() {
    use antecedent_model::{MechanismSlot, ParentBasis};
    let v = VariableId::from_raw;
    let (data, columns, graph) = non_separable_table(1500);
    let edges = [(0, 1), (0, 2), (1, 2)];
    let all = edges.map(|(a, b)| (v(a), v(b)));
    let query = CrossWorldQuery::path_specific(v(0), v(2), -1.0, 2.0, &all, &all[1..2]).unwrap();
    let compiled = CompiledCausalModel::compile(graph).unwrap();
    let store = crate::gcm::fit_nested_mechanisms(
        &compiled,
        &data,
        NestedOutcomeMechanism::NonSeparableBasis,
        v(2),
    )
    .unwrap();
    let outcome = compiled.dense_of(v(2)).unwrap();
    let MechanismSlot::LinearBasis { intercept, basis, coeffs, sigma } = store.get(outcome).clone()
    else {
        panic!("the outcome is a basis mechanism");
    };
    let closed = closed_form_point(
        3,
        &edges,
        &columns,
        &query,
        NestedOutcomeMechanism::NonSeparableBasis,
        2,
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(5);
    let point_of = |slot: MechanismSlot| {
        let engine = CounterfactualEngine::new(
            compiled.clone().with_mechanisms(store.with_replaced(outcome, slot).unwrap()),
        );
        let first = evaluate_cross_world(&engine, &data, &query, &ctx).unwrap().point();
        // Replay by the same evaluator over the same (edited) mechanism is
        // bit-identical, so replay alone accepts the edit.
        let second = evaluate_cross_world(&engine, &data, &query, &ctx).unwrap().point();
        assert_eq!(first.to_bits(), second.to_bits());
        first
    };
    let untouched = point_of(MechanismSlot::LinearBasis {
        intercept,
        basis: basis.clone(),
        coeffs: coeffs.clone(),
        sigma,
    });
    assert!(closed_form_agrees(closed, untouched), "unmutated: {closed} vs {untouched}");
    // (1) The coefficient of the first cross-parent product column edited.
    let mut edited: Vec<f64> = coeffs.to_vec();
    let product = basis
        .terms()
        .iter()
        .position(|t| matches!(t, antecedent_model::BasisTerm::Product { .. }))
        .expect("a cross-parent product column");
    edited[product] += 0.5;
    let coefficient = point_of(MechanismSlot::LinearBasis {
        intercept,
        basis: basis.clone(),
        coeffs: Arc::from(edited),
        sigma,
    });
    assert!(!closed_form_agrees(closed, coefficient), "coefficient edit: {coefficient}");
    // (2) One knot of the basis moved.
    let mut knots: Vec<Arc<[f64]>> = basis.knots().to_vec();
    let moved: Vec<f64> = knots[1].iter().map(|k| k + 0.4).collect();
    knots[1] = Arc::from(moved);
    let shifted = ParentBasis::new(
        basis.n_parents(),
        Arc::from(basis.centers().to_vec()),
        Arc::from(basis.scales().to_vec()),
        Arc::from(knots),
        Arc::from(basis.terms().to_vec()),
    )
    .unwrap();
    let knot = point_of(MechanismSlot::LinearBasis {
        intercept,
        basis: shifted,
        coeffs: coeffs.clone(),
        sigma,
    });
    assert!(!closed_form_agrees(closed, knot), "knot edit: {knot}");
}

/// The consumer reports the non-separable family as independently verified,
/// and declines (replay only) when the outcome design has no closed form.
#[test]
fn the_non_separable_closed_form_declines_designs_the_fitter_refuses() {
    let v = VariableId::from_raw;
    let edges = [(0, 1), (0, 2), (1, 2)];
    let all = edges.map(|(a, b)| (v(a), v(b)));
    let query = CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &all, &all[..1]).unwrap();
    let (_, columns, _) = non_separable_table(1500);
    // Too few rows for the columns of the expansion.
    let small: Vec<Vec<f64>> = columns.iter().map(|c| c[..60].to_vec()).collect();
    let decline = |columns: &[Vec<f64>], outcome: u32| {
        closed_form_point(
            3,
            &edges,
            columns,
            &query,
            NestedOutcomeMechanism::NonSeparableBasis,
            outcome,
        )
    };
    assert!(decline(&small, 2).is_none());
    // Every parent of the outcome binary: too few distinct values for a spline.
    let binary = |k: usize| -> Vec<f64> { (0..1500).map(|i| ((i / k) % 2) as f64).collect() };
    let flat = vec![binary(1), binary(3), columns[2].clone()];
    let direct = [(v(0), v(2)), (v(1), v(2))];
    let flat_query =
        CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &direct, &direct[..1]).unwrap();
    assert!(
        closed_form_point(
            3,
            &[(0, 2), (1, 2)],
            &flat,
            &flat_query,
            NestedOutcomeMechanism::NonSeparableBasis,
            2
        )
        .is_none()
    );
    // An outcome with a single parent has no cross-parent product.
    assert!(
        closed_form_point(
            3,
            &[(0, 1), (1, 2)],
            &columns,
            &query,
            NestedOutcomeMechanism::NonSeparableBasis,
            2
        )
        .is_none()
    );
}

/// A cancellation raised in the middle of the fit stops it with the typed
/// error and no effect, for both mechanism families; the same evaluation
/// without cancellation polls the token many more times and answers.
#[test]
fn a_context_cancelled_mid_fit_stops_the_fit_without_a_result() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let v = VariableId::from_raw;
    let (data, _, graph) = non_separable_table(400);
    let all = [(v(0), v(1)), (v(0), v(2)), (v(1), v(2))];
    let query = CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &all, &all[..1]).unwrap();
    for mechanism in
        [NestedOutcomeMechanism::LinearGaussian, NestedOutcomeMechanism::NonSeparableBasis]
    {
        let options = CrossWorldOptions { mechanism, interval_requested: false };
        let total = AtomicUsize::new(0);
        let ok = evaluate_effect_probed(
            graph.clone(),
            &data,
            &query,
            options,
            &ExecutionContext::for_tests(6),
            &|_| {
                total.fetch_add(1, Ordering::Relaxed);
            },
        )
        .unwrap();
        let total = total.into_inner();
        // Three nodes: one poll per node and candidate family, and one per fold.
        assert!(total >= 20, "{mechanism:?}: only {total} fit polls");
        assert!(ok.point.is_finite());
        for at in [0, 1, total / 3, total / 2, total - 1] {
            let ctx = ExecutionContext::for_tests(6);
            let seen = AtomicUsize::new(0);
            let outcome =
                evaluate_effect_probed(graph.clone(), &data, &query, options, &ctx, &|k| {
                    seen.fetch_add(1, Ordering::Relaxed);
                    if k == at {
                        ctx.cancellation.cancel();
                    }
                });
            assert!(
                matches!(outcome, Err(CausalError::Cancelled { stage: "cross_world" })),
                "{mechanism:?} at poll {at}: {outcome:?}"
            );
            assert_eq!(
                seen.into_inner(),
                at + 1,
                "{mechanism:?}: the fit kept going after the cancelling poll {at}"
            );
        }
    }
}

#[test]
fn ols_on_parents_recovers_a_known_fit_at_any_covariate_scale_and_declines_collinearity() {
    // y = 2 + 3e9 x0 - 4 x1 exactly, with a covariate in units of 1e-9: the scale-free
    // rank rule fits it where an absolute pivot floor on the Gram would decline it.
    let n = 40;
    let x0: Vec<f64> = (0..n).map(|i| 1e-9 * ((i * 7) % 11) as f64).collect();
    let x1: Vec<f64> = (0..n).map(|i| ((i * 5) % 13) as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 + 3.0e9 * x0[i] - 4.0 * x1[i]).collect();
    let columns = vec![x0.clone(), x1.clone(), y];
    let (intercept, slopes) = ols_on_parents(&columns, 2, &[0, 1]).expect("full rank");
    assert!((intercept - 2.0).abs() < 1e-6, "intercept {intercept}");
    assert!((slopes[0] / 3.0e9 - 1.0).abs() < 1e-9, "slope0 {}", slopes[0]);
    assert!((slopes[1] + 4.0).abs() < 1e-6, "slope1 {}", slopes[1]);
    // An exact copy of a parent is rank deficient and is declined.
    let collinear = vec![x1.clone(), x1.clone(), x1.iter().map(|v| 2.0 * v + 1.0).collect()];
    assert!(ols_on_parents(&collinear, 2, &[0, 1]).is_none());
    // No parents: the mean.
    let (mean, none) = ols_on_parents(&columns, 2, &[]).unwrap();
    assert!(none.is_empty());
    assert!((mean - columns[2].iter().sum::<f64>() / n as f64).abs() < 1e-9);
}
