//! Abduction, action, prediction against a hand-written structural recursion.
//!
//! The model is a confounded mediation diamond with known linear-Gaussian mechanisms
//! (`w -> x`, `w -> y`, `x -> m`, `x -> y`, `m -> y`). The reference recovers each unit's noise
//! by inverting the structural equations on the observed values, applies the intervention by
//! overwriting the intervened nodes, and recomputes every descendant from its own mechanism with
//! the recovered noise, writing the recursion out directly rather than calling the engine's
//! overlay or replay code. The engine must agree to rounding for single, double and nested
//! interventions, for unit subsets and for the individual treatment effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Intervention, Value, VariableId};
use antecedent_counterfactual::{
    AbductionMissingPolicy, CounterfactualEngine, CounterfactualWorld,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, MechanismWorkspace,
};

const W: u32 = 0;
const X: u32 = 1;
const M: u32 = 2;
const Y: u32 = 3;

// Coefficients, intercepts as written in the structural equations.
const X0: f64 = 0.4;
const XW: f64 = 1.5;
const M0: f64 = -0.7;
const MX: f64 = 2.0;
const Y0: f64 = 0.25;
const YW: f64 = -1.2;
const YX: f64 = 0.9;
const YM: f64 = 1.7;

fn engine() -> CounterfactualEngine {
    let mut g = Dag::with_variables(4);
    for (a, b) in [(W, X), (W, Y), (X, M), (X, Y), (M, Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let slots = vec![
        MechanismSlot::LinearGaussian { intercept: 0.0, coeffs: Arc::from([]), sigma: 1.0 },
        MechanismSlot::LinearGaussian { intercept: X0, coeffs: Arc::from([XW]), sigma: 1.0 },
        MechanismSlot::LinearGaussian { intercept: M0, coeffs: Arc::from([MX]), sigma: 1.0 },
        // Parents of y in ascending node order: w, x, m.
        MechanismSlot::LinearGaussian {
            intercept: Y0,
            coeffs: Arc::from([YW, YX, YM]),
            sigma: 1.0,
        },
    ];
    let compiled = CompiledCausalModel::compile(g)
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });
    CounterfactualEngine::new(compiled)
}

struct Units {
    w: Vec<f64>,
    x: Vec<f64>,
    m: Vec<f64>,
    y: Vec<f64>,
}

/// Observed units whose noise is not zero: the structural equations hold with these shocks.
fn units(n: usize) -> Units {
    let shock = |i: usize, k: f64| ((i as f64) * k).sin() * 0.8 + ((i % 7) as f64 - 3.0) * 0.05;
    let w: Vec<f64> = (0..n).map(|i| shock(i, 0.91)).collect();
    let x: Vec<f64> = (0..n).map(|i| X0 + XW * w[i] + shock(i, 1.7)).collect();
    let m: Vec<f64> = (0..n).map(|i| M0 + MX * x[i] + shock(i, 2.3)).collect();
    let y: Vec<f64> =
        (0..n).map(|i| Y0 + YW * w[i] + YX * x[i] + YM * m[i] + shock(i, 3.1)).collect();
    Units { w, x, m, y }
}

fn table(u: &Units) -> TabularData {
    TabularData::from_f64_columns([
        ("w", u.w.as_slice()),
        ("x", u.x.as_slice()),
        ("m", u.m.as_slice()),
        ("y", u.y.as_slice()),
    ])
    .unwrap()
}

/// Unit `i` after the hard interventions `set`, by the structural recursion on recovered noise.
fn reference(u: &Units, i: usize, set: &[(u32, f64)]) -> [f64; 4] {
    let u_x = u.x[i] - (X0 + XW * u.w[i]);
    let u_m = u.m[i] - (M0 + MX * u.x[i]);
    let u_y = u.y[i] - (Y0 + YW * u.w[i] + YX * u.x[i] + YM * u.m[i]);
    let pick =
        |node: u32, natural: f64| set.iter().find(|(n, _)| *n == node).map_or(natural, |(_, v)| *v);
    let w = pick(W, u.w[i]);
    let x = pick(X, X0 + XW * w + u_x);
    let m = pick(M, M0 + MX * x + u_m);
    let y = pick(Y, Y0 + YW * w + YX * x + YM * m + u_y);
    [w, x, m, y]
}

fn world(rows: Option<&[usize]>, set: &[(u32, f64)]) -> CounterfactualWorld {
    CounterfactualWorld {
        unit_rows: rows.map(Arc::from),
        interventions: set
            .iter()
            .map(|(n, v)| Intervention::set(VariableId::from_raw(*n), Value::f64(*v)))
            .collect(),
    }
}

#[test]
fn every_unit_and_node_matches_the_structural_recursion_under_single_and_joint_interventions() {
    let n = 40;
    let u = units(n);
    let engine = engine();
    let ctx = ExecutionContext::for_tests(3);
    let exo = engine.abduct(&table(&u), AbductionMissingPolicy::Error, &ctx).unwrap();
    let outcomes: Vec<VariableId> = (0..4).map(VariableId::from_raw).collect();
    let cases: [&[(u32, f64)]; 5] =
        [&[], &[(X, 1.25)], &[(M, -0.5)], &[(X, 1.25), (M, -0.5)], &[(W, 2.0), (X, 0.0)]];
    let mut ws = MechanismWorkspace::default();
    for set in cases {
        let res =
            engine.predict(&exo, &[world(None, set)], &outcomes, false, &mut ws, &ctx).unwrap();
        for i in 0..n {
            let want = reference(&u, i, set);
            for node in 0..4u32 {
                let got = res.get(0, DenseNodeId::from_raw(node), i);
                assert!(
                    (got - want[node as usize]).abs() < 1e-12,
                    "set={set:?} unit={i} node={node}: {got} vs {}",
                    want[node as usize]
                );
            }
        }
    }
}

#[test]
fn the_empty_world_returns_the_observed_units() {
    let u = units(25);
    let engine = engine();
    let ctx = ExecutionContext::for_tests(3);
    let exo = engine.abduct(&table(&u), AbductionMissingPolicy::Error, &ctx).unwrap();
    let outcomes: Vec<VariableId> = (0..4).map(VariableId::from_raw).collect();
    let res = engine
        .predict(
            &exo,
            &[world(None, &[])],
            &outcomes,
            false,
            &mut MechanismWorkspace::default(),
            &ctx,
        )
        .unwrap();
    for (node, observed) in [&u.w, &u.x, &u.m, &u.y].into_iter().enumerate() {
        for (i, v) in observed.iter().enumerate() {
            let got = res.get(0, DenseNodeId::from_raw(node as u32), i);
            assert!((got - v).abs() < 1e-12, "node {node} unit {i}");
        }
    }
}

#[test]
fn a_unit_subset_and_the_individual_treatment_effect_follow_the_same_recursion() {
    let n = 30;
    let u = units(n);
    let engine = engine();
    let ctx = ExecutionContext::for_tests(3);
    let exo = engine.abduct(&table(&u), AbductionMissingPolicy::Error, &ctx).unwrap();
    let mut ws = MechanismWorkspace::default();
    let rows = [3usize, 17, 4];
    let res = engine
        .predict(
            &exo,
            &[world(Some(&rows), &[(X, 1.0)])],
            &[VariableId::from_raw(Y)],
            false,
            &mut ws,
            &ctx,
        )
        .unwrap();
    // The result keeps the factual row coordinates: selected rows hold the counterfactual,
    // every other row is NaN.
    for i in 0..n {
        let got = res.get(0, DenseNodeId::from_raw(Y), i);
        if rows.contains(&i) {
            let want = reference(&u, i, &[(X, 1.0)])[3];
            assert!((got - want).abs() < 1e-12, "row {i}: {got} vs {want}");
        } else {
            assert!(got.is_nan(), "row {i} is outside the subset");
        }
    }
    let ite = engine
        .individual_treatment_effect(
            &exo,
            VariableId::from_raw(Y),
            Intervention::set(VariableId::from_raw(X), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(X), Value::f64(0.0)),
            &mut ws,
            &ctx,
        )
        .unwrap();
    // Linear additive noise: the unit effect is the total effect for every unit.
    for effect in ite.iter() {
        assert!((effect - (YX + YM * MX)).abs() < 1e-12, "{effect}");
    }
}
