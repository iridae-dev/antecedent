//! Gaussian-process mechanism against exact Cholesky marginal-likelihood selection.
//!
//! `oracle_data/registry_gaussian_process_reference.py` (frozen in the JSON beside it) evaluates
//! the exact negative log marginal likelihood of every cell of the 25-cell hyperparameter grid in
//! 60-digit arithmetic (LU solve for the dual weights, LDL' factor for the log determinant),
//! takes the first minimiser and asserts it leads by more than 1e-6. The registry fit must pick
//! the same length scale and noise level, store the same signal variance, prior mean and dual
//! weights, and then predict at held-out points and score them as the reference does: posterior
//! mean plus the fitted noise as a Gaussian (the posterior variance is *not* used by the library).
//! One design has parents in years and an outcome near 1000.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "gaussian-process")]

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, MechanismFamily, MechanismRegistry, MechanismSlot, NodeFamilyConstraint,
    ParentBatch, SelectionOptions, SelectionPolicy, evaluate_column, log_prob_column,
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

#[test]
fn registry_gaussian_process_reference_selects_the_exact_marginal_likelihood_cell() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/registry_gaussian_process_reference.json"))
            .unwrap();
    let registry = MechanismRegistry::with_bayesian_families();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let p = rows[0].len();
        let y = floats(&case["y"]);
        let mut columns: Vec<Vec<f64>> =
            (0..p).map(|c| rows.iter().map(|r| r[c]).collect()).collect();
        columns.push(y);
        let names: Vec<&str> = if p == 1 { vec!["x1", "y"] } else { vec!["x1", "x2", "y"] };
        let mut g = Dag::with_variables(u32::try_from(p + 1).unwrap());
        for c in 0..p {
            g.insert_directed(
                DenseNodeId::from_raw(u32::try_from(c).unwrap()),
                DenseNodeId::from_raw(u32::try_from(p).unwrap()),
            )
            .unwrap();
        }
        let model = CompiledCausalModel::compile(g).unwrap();
        let data = table(&columns, &names);
        let mut constraints: Vec<(VariableId, NodeFamilyConstraint)> = (0..p)
            .map(|c| {
                (
                    VariableId::from_raw(u32::try_from(c).unwrap()),
                    NodeFamilyConstraint {
                        required: Some(MechanismFamily::LinearGaussian),
                        ..NodeFamilyConstraint::default()
                    },
                )
            })
            .collect();
        constraints.push((
            VariableId::from_raw(u32::try_from(p).unwrap()),
            NodeFamilyConstraint {
                required: Some(MechanismFamily::GaussianProcess),
                ..NodeFamilyConstraint::default()
            },
        ));
        let (store, _) = registry
            .assign_and_fit_with(
                &model,
                &data,
                SelectionPolicy::BestScore,
                &SelectionOptions { constraints, ..SelectionOptions::default() },
            )
            .unwrap();
        let slot = &store.slots[p];
        let MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            noise_std,
            mean,
            n_train,
            n_parents,
            alpha,
            ..
        } = slot
        else {
            panic!("{name}: Gaussian-process slot");
        };
        assert_eq!((*n_train, *n_parents), (rows.len(), p), "{name}");
        close(*length_scale, case["length_scale"].as_f64().unwrap(), 1e-12, name);
        close(*noise_std, case["noise_std"].as_f64().unwrap(), 1e-12, name);
        close(*variance, case["variance"].as_f64().unwrap(), 1e-12, name);
        close(*mean, case["mean"].as_f64().unwrap(), 1e-12, name);
        let want_alpha = floats(&case["alpha"]);
        let scale = want_alpha.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (i, w) in want_alpha.iter().enumerate() {
            assert!((alpha[i] - w).abs() <= 1e-7 * scale, "{name}: alpha[{i}] {} vs {w}", alpha[i]);
        }
        // Held-out points: predictive mean, then the Gaussian score about it.
        for (t, test) in case["tests"].as_array().unwrap().iter().enumerate() {
            let point = floats(&test["x"]);
            let batch = ParentBatch { n_rows: 1, n_parents: p, values: &point };
            let mut predicted = [0.0];
            let mut workspace = antecedent_model::batch::MechanismWorkspace::default();
            evaluate_column(slot, batch, &[0.0], &mut predicted, &mut workspace).unwrap();
            let scale = test["mean"].as_f64().unwrap().abs().max(1.0);
            assert!(
                (predicted[0] - test["mean"].as_f64().unwrap()).abs() <= 1e-8 * scale,
                "{name}: test {t}: mean {} vs {}",
                predicted[0],
                test["mean"]
            );
            let mut lp = [0.0];
            log_prob_column(slot, &[test["y"].as_f64().unwrap()], batch, &mut lp).unwrap();
            close(
                lp[0],
                test["log_density"].as_f64().unwrap(),
                1e-7,
                &format!("{name}: log density {t}"),
            );
        }
    }
}
