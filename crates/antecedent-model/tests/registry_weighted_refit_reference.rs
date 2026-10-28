//! Weighted refit of linear-Gaussian mechanisms (the refit that row-weight and bootstrap
//! uncertainty rely on) against exact weighted least squares.
//!
//! `oracle_data/registry_weighted_refit_reference.py` (frozen in the JSON beside it) solves the
//! weighted normal equations of seven designs in rational arithmetic, rescales the weights to mean
//! one as the refit documents, and takes the residual scale from the weighted residual sum of
//! squares over `max(n - columns, 1)` with the `1e-8 * rms(y)` floor. The designs are unit
//! weights, weights spanning 2^-20 to 2^20, frequency counts with zeros, intercepts near 1e8, two
//! parents equal to within 2^-22 (near-collinear), four rows (one residual degree of freedom) and
//! three rows for three coefficients (none). This test refits the same data through
//! `MechanismRegistry::refit_weighted` (and the frequency route when a count is zero) and requires
//! the intercepts, coefficients (the fitted values for the near-collinear design, whose
//! coefficients are individually ill-determined), residual scales and the root mechanisms to
//! match.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, MechanismFamily, MechanismRegistry, MechanismSlot, SelectionPolicy,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn table(columns: &[&[f64]]) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for name in ["x1", "x2", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(columns[0].len());
    let cols = columns
        .iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v.to_vec()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1.0), "{what}: {got} vs {want}");
}

#[test]
fn weighted_linear_gaussian_refits_match_exact_weighted_least_squares() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/registry_weighted_refit_reference.json"))
            .unwrap();
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let registry = MechanismRegistry::standard();
    // Selected families are taken from a fit on a larger table and then pinned to the linear
    // Gaussian family, which is what the weighted route refits.
    let warm: Vec<f64> = (0..60).map(|i| f64::from(i % 7) + 0.37 * f64::from(i)).collect();
    let warm_y: Vec<f64> = warm.iter().map(|v| 2.0 * v + (v * 1.3).sin()).collect();
    let warm_x2: Vec<f64> = warm.iter().map(|v| (v * 0.7).cos()).collect();
    let (_, mut assignments) = registry
        .assign_and_fit(
            &model,
            &table(&[warm.as_slice(), warm_x2.as_slice(), warm_y.as_slice()]),
            SelectionPolicy::BestScore,
        )
        .unwrap();
    for a in &mut assignments {
        a.selected = MechanismFamily::LinearGaussian;
    }
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (x1, x2, y) = (floats(&case["x1"]), floats(&case["x2"]), floats(&case["y"]));
        let weights = floats(&case["weights"]);
        let data = table(&[&x1, &x2, &y]);
        let store = if weights.iter().any(|w| *w == 0.0) {
            registry
                .refit_frequency_weighted_polled(&model, &data, &assignments, &weights, &|| false)
                .unwrap()
        } else {
            registry.refit_weighted(&model, &data, &assignments, &weights).unwrap()
        };
        let slot = |i: usize| &store.slots[i];
        let MechanismSlot::LinearGaussian { intercept, sigma, .. } = slot(0) else {
            panic!("{name}: root family");
        };
        close(*intercept, case["x1_intercept"].as_f64().unwrap(), 1e-9, name);
        close(*sigma, case["x1_sigma"].as_f64().unwrap(), 1e-8, name);
        let MechanismSlot::LinearGaussian { intercept, sigma, .. } = slot(1) else {
            panic!("{name}: root family");
        };
        close(*intercept, case["x2_intercept"].as_f64().unwrap(), 1e-9, name);
        close(*sigma, case["x2_sigma"].as_f64().unwrap(), 1e-8, name);
        let MechanismSlot::LinearGaussian { intercept, coeffs, sigma } = slot(2) else {
            panic!("{name}: outcome family");
        };
        let near_collinear = name == "near_collinear_parents";
        let want = floats(&case["y_coeffs"]);
        let order: Vec<usize> = model
            .gather_for(DenseNodeId::from_raw(2))
            .unwrap()
            .parents
            .iter()
            .map(|p| p.as_usize())
            .collect();
        let got: Vec<f64> =
            (0..2).map(|j| coeffs[order.iter().position(|&p| p == j).unwrap()]).collect();
        let columns = [&x1, &x2];
        for (i, fitted) in floats(&case["y_fitted"]).into_iter().enumerate() {
            let pred = intercept + got[0] * columns[0][i] + got[1] * columns[1][i];
            // The prediction error is bounded by the conditioning of the design times rounding.
            close(pred, fitted, if near_collinear { 1e-6 } else { 1e-9 }, name);
        }
        if !near_collinear {
            close(*intercept, case["y_intercept"].as_f64().unwrap(), 1e-8, name);
            for j in 0..2 {
                close(got[j], want[j], 1e-8, name);
            }
        }
        close(
            *sigma,
            case["y_sigma"].as_f64().unwrap(),
            if near_collinear { 1e-4 } else { 1e-7 },
            name,
        );
    }
}
