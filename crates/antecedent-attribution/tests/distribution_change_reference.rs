//! Mechanism-Shapley attribution of a distribution change against 80-digit evaluation of its
//! definition (`oracle_data/distribution_change_reference.py`).
//!
//! Two populations of `X -> Y -> Z` with the chord `X -> Z` differ in every mechanism. The
//! reference fits each node by least squares on each population, forms the exact outcome law of
//! every hybrid system, and averages marginal contributions over all orders. The linear
//! (regression-hybrid) path must agree to rounding; the sampling path must agree within the
//! sampling error of its declared budget, for the mean, the variance and the Gaussian KL measures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::DifferenceMeasure;
use antecedent_attribution::{
    DistributionChangeOptions, RobustChangeOptions, distribution_change, distribution_change_robust,
};
use antecedent_core::{
    AllocationMethod, CausalSchemaBuilder, ChangeAttributionQuery, ExecutionContext,
    MeasurementSpec, PopulationSelector, RoleHint, ShapleyConfig, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, MechanismRegistry, MechanismSlot, SelectionPolicy};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/distribution_change_reference.json"))
        .expect("frozen reference")
}

fn f(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn rows_of(r: &Value) -> Vec<Vec<f64>> {
    r["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(f).collect())
        .collect()
}

fn table(rows: &[Vec<f64>]) -> TabularData {
    let n = rows.len();
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y", "z"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(n);
    let cols = (0..3)
        .map(|k| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(k as u32),
                    Arc::from(rows.iter().map(|row| row[k]).collect::<Vec<_>>()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

fn setup(r: &Value) -> (CompiledCausalModel, TabularData, usize) {
    let data = table(&rows_of(r));
    let mut g = Dag::with_variables(3);
    for (a, c) in [(0, 1), (0, 2), (1, 2)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(c)).unwrap();
    }
    (CompiledCausalModel::compile(g).unwrap(), data, r["n"].as_u64().unwrap() as usize)
}

fn query(n: usize) -> ChangeAttributionQuery {
    ChangeAttributionQuery::new(
        VariableId::from_raw(2),
        PopulationSelector::Rows(Arc::from((0..n).collect::<Vec<_>>())),
        PopulationSelector::Rows(Arc::from((n..2 * n).collect::<Vec<_>>())),
    )
    .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() })
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{what}: {got} vs {want}");
}

#[test]
fn registry_fits_equal_least_squares_with_the_unbiased_residual_variance() {
    let r = reference();
    let (model, _data, n) = setup(&r);
    for (pop, range) in [("baseline", 0..n), ("comparison", n..2 * n)] {
        let sub = table(&rows_of(&r)[range]);
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &sub, SelectionPolicy::BestScore)
            .unwrap();
        for (i, node) in ["x", "y", "z"].iter().enumerate() {
            let want = &r["fit"][pop][node];
            let MechanismSlot::LinearGaussian { intercept, coeffs, sigma } = &store.slots[i] else {
                panic!("{pop} {node}: registry did not select the linear Gaussian family");
            };
            let beta: Vec<f64> =
                std::iter::once(*intercept).chain(coeffs.iter().copied()).collect();
            for (j, b) in want["beta"].as_array().unwrap().iter().enumerate() {
                close(beta[j], f(b), 1e-9, &format!("{pop} {node} beta {j}"));
            }
            close(*sigma, f(&want["sigma"]), 1e-9, &format!("{pop} {node} sigma"));
        }
    }
}

#[test]
fn regression_hybrid_path_equals_the_exact_mean_shapley_values() {
    let r = reference();
    let (model, data, n) = setup(&r);
    let q = query(n);
    let out = distribution_change_robust(
        &model,
        &data,
        &q,
        &RobustChangeOptions::default(),
        &ExecutionContext::for_tests(3),
    )
    .unwrap();
    let want = &r["mean"];
    close(out.total_change, f(&want["total"]), 1e-10, "total");
    assert_eq!(out.contributions.len(), 3);
    let mut sum = 0.0;
    for (i, c) in out.contributions.iter().enumerate() {
        close(c.contribution, f(&want["phi"][i]), 1e-10, &format!("phi {i}"));
        sum += c.contribution;
    }
    close(sum, out.total_change, 1e-10, "efficiency");
}

#[test]
fn sampling_path_matches_the_exact_laws_within_its_sampling_error() {
    let r = reference();
    let (model, data, n) = setup(&r);
    let q = query(n);
    for (name, measure, tol) in [
        ("mean", DifferenceMeasure::MeanDiff, 0.02),
        ("variance", DifferenceMeasure::VarianceDiff, 0.03),
        ("kl", DifferenceMeasure::GaussianKl, 0.02),
    ] {
        let out = distribution_change(
            &model,
            &data,
            &q,
            &DistributionChangeOptions { measure, n_samples: 200_000, seed: 17 },
            &ExecutionContext::for_tests(5),
        )
        .unwrap();
        let want = &r[name];
        close(out.total_change, f(&want["total"]), tol, &format!("{name} total"));
        let mut sum = 0.0;
        for (i, c) in out.contributions.iter().enumerate() {
            close(c.contribution, f(&want["phi"][i]), tol, &format!("{name} phi {i}"));
            sum += c.contribution;
        }
        close(sum, out.total_change, 1e-9, &format!("{name} efficiency"));
    }
}
