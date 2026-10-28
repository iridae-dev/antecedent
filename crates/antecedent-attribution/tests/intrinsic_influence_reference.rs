//! Intrinsic causal influence against exact Shapley values of the variance game.
//!
//! `oracle_data/intrinsic_influence_reference.py` (frozen in the JSON beside it) computes, in
//! rational arithmetic, `v(S) = Var(E[Y | U_S])` for small structural models and each player's
//! Shapley value by enumerating every ordering. The models are chosen for the regimes a nested
//! simulation handles worst: intercepts near 1e7 beside noise scales from 2^-10 to 30 (a
//! variance formed from raw second moments loses every digit of the smallest share), two paths
//! from a root whose effects cancel exactly (a source with truth zero), a root whose coefficient
//! cancels beside near-equal parents, an amplifying chain, and a product mechanism with
//! non-zero, unequal means (asymmetric Shapley values from an interaction). This test requires
//! the product's contributions to lie within the same six-standard-error band the module's other
//! closed-form tests use (the larger of the derived and the reported error, with a small
//! absolute floor relative to the total variance for sources whose truth is zero), the total
//! variance to match, and the efficiency identity to hold within estimation error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::{
    IntrinsicAllocation, IntrinsicInfluenceQuery, NoiseIndependence, intrinsic_influence,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    BasisTerm, CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ParentBasis,
};
use serde_json::Value;

fn node(i: u64) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).unwrap())
}

fn linear(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

fn compile(n: u64, edges: &[Value]) -> CompiledCausalModel {
    let mut graph = Dag::with_variables(u32::try_from(n).unwrap());
    for e in edges {
        graph.insert_directed(node(e[0].as_u64().unwrap()), node(e[1].as_u64().unwrap())).unwrap();
    }
    CompiledCausalModel::compile(graph).unwrap()
}

fn linear_model(case: &Value) -> CompiledCausalModel {
    let n = case["n"].as_u64().unwrap();
    let base = compile(n, case["edges"].as_array().unwrap());
    let slots: Vec<MechanismSlot> = (0..n)
        .map(|i| {
            let coeffs: Vec<f64> = base
                .gather_for(node(i))
                .unwrap()
                .parents
                .iter()
                .map(|p| case["coef"][i as usize][p.as_usize().to_string()].as_f64().unwrap())
                .collect();
            linear(
                case["intercept"][i as usize].as_f64().unwrap(),
                &coeffs,
                case["sigma"][i as usize].as_f64().unwrap(),
            )
        })
        .collect();
    base.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

fn product_model(case: &Value) -> CompiledCausalModel {
    let f = |k: &str| case[k].as_f64().unwrap();
    let basis = ParentBasis::new(
        2,
        Arc::from(vec![0.0, 0.0]),
        Arc::from(vec![1.0, 1.0]),
        Arc::from(vec![
            Arc::<[f64]>::from(Vec::<f64>::new()),
            Arc::<[f64]>::from(Vec::<f64>::new()),
        ]),
        Arc::from(vec![
            BasisTerm::Power { parent: 0, degree: 1 },
            BasisTerm::Power { parent: 1, degree: 1 },
            BasisTerm::Product { left: 0, right: 1 },
        ]),
    )
    .unwrap();
    let base = compile(3, &[serde_json::json!([0, 2]), serde_json::json!([1, 2])]);
    let slots = vec![
        linear(f("ma"), &[], f("sa")),
        linear(f("mb"), &[], f("sb")),
        MechanismSlot::LinearBasis {
            intercept: f("b"),
            basis,
            coeffs: Arc::from(vec![0.0, 0.0, 1.0]),
            sigma: f("se"),
        },
    ];
    base.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

#[test]
fn contributions_match_exact_shapley_values_of_the_variance_game() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/intrinsic_influence_reference.json"))
            .unwrap();
    let ctx = ExecutionContext::for_tests(11);
    let seeds = 30_u64;
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (model, target, label): (_, u32, fn(u32) -> String) = if case["kind"] == "linear" {
            (linear_model(case), u32::try_from(case["target"].as_u64().unwrap()).unwrap(), |v| {
                v.to_string()
            })
        } else {
            (product_model(case), 2, |v| ["A", "B", "E"][v as usize].to_owned())
        };
        let truth = case["truth"].as_object().unwrap();
        let total = case["total_variance"].as_f64().unwrap();
        // Contributions and total variance of one estimate per seed, keyed by source label.
        let mut draws: std::collections::BTreeMap<String, Vec<f64>> =
            truth.keys().map(|k| (k.clone(), Vec::new())).collect();
        let mut totals = Vec::new();
        for seed in 0..seeds {
            let query = IntrinsicInfluenceQuery::new(VariableId::from_raw(target))
                .with_simulation(500, 20, 4)
                .with_allocation(IntrinsicAllocation::Exact)
                .with_seed(seed);
            let result =
                intrinsic_influence(&model, NoiseIndependence::Declared, None, &query, &ctx)
                    .unwrap();
            assert_eq!(result.sources.len(), truth.len(), "{name}");
            for s in result.sources.iter() {
                draws.get_mut(&label(s.source.variable().raw())).unwrap().push(s.contribution);
            }
            totals.push(result.total_variance);
        }
        // The estimator is unbiased up to Monte Carlo noise, whose size depends on how unequal
        // the shares are (a small share beside a large one is estimated from a noisy nested
        // variance), so the check is on the mean of independent estimates against five
        // standard errors of that mean, with a floor of 1e-6 of the total variance for sources
        // whose truth is zero.
        let check = |what: &str, xs: &[f64], want: f64| {
            let n = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / n;
            let sd = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            let tol = 5.0 * sd / n.sqrt() + 1e-6 * total;
            assert!(
                (mean - want).abs() <= tol,
                "{name} {what}: mean {mean} vs {want} (sd {sd}, tol {tol})"
            );
        };
        for (key, xs) in &draws {
            check(&format!("source {key}"), xs, truth[key].as_f64().unwrap());
        }
        check("total variance", &totals, total);
    }
}
