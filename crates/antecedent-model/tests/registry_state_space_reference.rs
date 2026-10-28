//! Scalar linear-Gaussian state space against exact recursions and the joint Gaussian law.
//!
//! `oracle_data/registry_state_space_reference.py` (frozen in the JSON beside it) holds
//! * Kalman filter and Rauch-Tung-Striebel smoother outputs as exact rationals, each asserted
//!   inside the script equal to Gaussian conditioning on the whole state path, compared with
//!   `kalman_filter` and `rts_smooth` (including the lag-one smoothed covariance);
//! * the one-step predictive log density of `log_prob_column` at unit scale and at 1e9 and 1e-9;
//! * the registry's fitted conditional state space: OLS of the outcome on its parents, residuals
//!   normalised by their largest magnitude, then 25 EM iterations whose E-step is the joint-law
//!   conditioning in 60-digit arithmetic, compared with `assign_and_fit_with` of the required
//!   `LinearGaussianStateSpace` family.
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
    CompiledCausalModel, MechanismFamily, MechanismRegistry, MechanismSlot, NodeFamilyConstraint,
    ParentBatch, SelectionOptions, SelectionPolicy, kalman_filter, log_prob_column, rts_smooth,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn table(columns: &[Vec<f64>], names: &[&str]) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for name in names {
        b.add_variable(
            *name,
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
                    Arc::from(v.clone()),
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

fn all_close(got: &[f64], want: &[f64], rel: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (t, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, rel, &format!("{what}[{t}]"));
    }
}

#[test]
fn registry_state_space_reference_filter_and_smoother_match_exact_rational_recursions() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/registry_state_space_reference.json"))
            .unwrap();
    for case in data["filter_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = floats(&case["y"]);
        let (a, q, r, x0) = (
            case["a"].as_f64().unwrap(),
            case["process_var"].as_f64().unwrap(),
            case["obs_var"].as_f64().unwrap(),
            case["x0"].as_f64().unwrap(),
        );
        let (x_f, p_f, x_pred, p_pred) = kalman_filter(&y, a, q, r, x0, q);
        all_close(&x_f, &floats(&case["x_filtered"]), 1e-13, &format!("{name} filtered mean"));
        all_close(&p_f, &floats(&case["p_filtered"]), 1e-13, &format!("{name} filtered variance"));
        all_close(&x_pred, &floats(&case["x_predicted"]), 1e-13, &format!("{name} predicted mean"));
        all_close(
            &p_pred,
            &floats(&case["p_predicted"]),
            1e-13,
            &format!("{name} predicted variance"),
        );
        let (x_s, p_s, p_lag) = rts_smooth(a, &x_f, &p_f, &x_pred, &p_pred);
        all_close(&x_s, &floats(&case["x_smoothed"]), 1e-12, &format!("{name} smoothed mean"));
        all_close(&p_s, &floats(&case["p_smoothed"]), 1e-12, &format!("{name} smoothed variance"));
        all_close(&p_lag, &floats(&case["p_lag"]), 1e-12, &format!("{name} lag covariance"));
    }
}

#[test]
fn registry_state_space_reference_predictive_density_is_unit_free() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/registry_state_space_reference.json"))
            .unwrap();
    for case in data["density_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = floats(&case["y"]);
        let slot = MechanismSlot::LinearGaussianStateSpace {
            a: case["a"].as_f64().unwrap(),
            process_std: case["process_std"].as_f64().unwrap(),
            obs_std: case["obs_std"].as_f64().unwrap(),
            initial_mean: case["initial_mean"].as_f64().unwrap(),
        };
        let mut out = vec![0.0; y.len()];
        log_prob_column(
            &slot,
            &y,
            ParentBatch { n_rows: y.len(), n_parents: 0, values: &[] },
            &mut out,
        )
        .unwrap();
        all_close(&out, &floats(&case["log_density"]), 1e-11, &format!("{name} log density"));
    }
}

#[test]
fn registry_state_space_reference_em_fit_matches_joint_law_em() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/registry_state_space_reference.json"))
            .unwrap();
    let registry = MechanismRegistry::standard();
    for case in data["em_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = floats(&case["y"]);
        let xs: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let (model, frame, y_node) = if xs.is_empty() {
            (
                CompiledCausalModel::compile(Dag::with_variables(1)).unwrap(),
                table(&[y.clone()], &["y"]),
                0,
            )
        } else {
            let mut g = Dag::with_variables(2);
            g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            (
                CompiledCausalModel::compile(g).unwrap(),
                table(&[xs[0].clone(), y.clone()], &["x", "y"]),
                1,
            )
        };
        let mut constraints = vec![(
            VariableId::from_raw(y_node),
            NodeFamilyConstraint {
                required: Some(MechanismFamily::LinearGaussianStateSpace),
                ..NodeFamilyConstraint::default()
            },
        )];
        if y_node == 1 {
            constraints.push((
                VariableId::from_raw(0),
                NodeFamilyConstraint {
                    required: Some(MechanismFamily::LinearGaussian),
                    ..NodeFamilyConstraint::default()
                },
            ));
        }
        let options = SelectionOptions { constraints, ..SelectionOptions::default() };
        let (store, _) = registry
            .assign_and_fit_with(&model, &frame, SelectionPolicy::BestScore, &options)
            .unwrap();
        let MechanismSlot::ConditionalLinearGaussianStateSpace {
            intercept,
            coeffs,
            a,
            process_std,
            obs_std,
            initial_mean,
        } = &store.slots[y_node as usize]
        else {
            panic!("{name}: conditional state-space slot");
        };
        close(*intercept, case["intercept"].as_f64().unwrap(), 1e-10, name);
        let want_coeffs = floats(&case["coeffs"]);
        assert_eq!(coeffs.len(), want_coeffs.len(), "{name}");
        for (j, w) in want_coeffs.iter().enumerate() {
            close(coeffs[j], *w, 1e-10, &format!("{name} coefficient {j}"));
        }
        close(*a, case["a"].as_f64().unwrap(), 1e-7, &format!("{name} a"));
        close(*process_std, case["process_std"].as_f64().unwrap(), 1e-7, &format!("{name} q"));
        close(*obs_std, case["obs_std"].as_f64().unwrap(), 1e-7, &format!("{name} r"));
        close(*initial_mean, case["initial_mean"].as_f64().unwrap(), 1e-7, &format!("{name} x0"));
    }
}
