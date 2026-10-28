//! Categorical and binary-logit mechanism fits against exact maximum likelihood.
//!
//! `oracle_data/registry_glm_reference.py` (frozen in the JSON beside it) finds the unique
//! maximiser of each concave log-likelihood by Newton's method in 50-digit arithmetic on the raw
//! design: the baseline-category multinomial logit of the `Discrete` family (unweighted fit and
//! the weighted refit, three, four and two categories, a parent measured in years beside a
//! unit-scale parent), and the always-penalised binary logit of `HierarchicalGlm` with the
//! empirical-Bayes ridge strength recomputed exactly (interior value and the upper clamp). The
//! registry fits are taken from `assign_and_fit_with` with the family required, and the weighted
//! refit from `refit_weighted`.
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
    SelectionOptions, SelectionPolicy,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn table(columns: &[Vec<f64>], outcome_type: ValueType) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    for (i, name) in ["x1", "x2", "y"].into_iter().enumerate() {
        let vt = if i == 2 { outcome_type.clone() } else { ValueType::Continuous };
        b.add_variable(
            name,
            vt,
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

fn required(family: MechanismFamily) -> NodeFamilyConstraint {
    NodeFamilyConstraint { required: Some(family), ..NodeFamilyConstraint::default() }
}

#[test]
fn registry_glm_reference_categorical_and_binary_logit_fits_match_exact_maximum_likelihood() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/registry_glm_reference.json")).unwrap();
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let registry = MechanismRegistry::standard();
    let order: Vec<usize> = model
        .gather_for(DenseNodeId::from_raw(2))
        .unwrap()
        .parents
        .iter()
        .map(|p| p.as_usize())
        .collect();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let kind = case["kind"].as_str().unwrap();
        let xs: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let y = floats(&case["y"]);
        if kind == "discrete_interactions" || kind == "discrete_spline" {
            let family = if kind == "discrete_interactions" {
                MechanismFamily::DiscreteInteractions
            } else {
                MechanismFamily::DiscreteSpline
            };
            let data = table(&[xs[0].clone(), xs[1].clone(), y.clone()], ValueType::Binary);
            let options = SelectionOptions {
                constraints: vec![
                    (VariableId::from_raw(0), required(MechanismFamily::LinearGaussian)),
                    (VariableId::from_raw(1), required(MechanismFamily::LinearGaussian)),
                    (VariableId::from_raw(2), required(family)),
                ],
                ..SelectionOptions::default()
            };
            let (store, assignments) = registry
                .assign_and_fit_with(&model, &data, SelectionPolicy::BestScore, &options)
                .unwrap();
            let check = |slot: &MechanismSlot, want: &Value, tag: &str| {
                let MechanismSlot::DiscreteBasis { support, basis, logit_coeffs, .. } = slot else {
                    panic!("{name} {tag}: DiscreteBasis slot");
                };
                assert_eq!(support.to_vec(), floats(&case["support"]));
                assert_eq!(basis.n_terms(), case["n_terms"].as_u64().unwrap() as usize);
                let ncols = 1 + basis.n_terms();
                let mut expanded = vec![0.0; basis.n_terms()];
                for (k, row) in want.as_array().unwrap().iter().enumerate() {
                    let row = floats(row);
                    let big = row.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
                    let tol = if kind == "discrete_spline" { 1e-5 } else { 1e-7 };
                    for (j, w) in row.iter().enumerate() {
                        assert!(
                            (logit_coeffs[k * ncols + j] - w).abs() <= tol * big,
                            "{name} {tag}: logit {k}/{j}: {} vs {w}",
                            logit_coeffs[k * ncols + j]
                        );
                    }
                    // The conditional law row by row.
                    for i in 0..y.len() {
                        basis.expand_row(&[xs[0][i], xs[1][i]], &mut expanded).unwrap();
                        let got = logit_coeffs[k * ncols]
                            + expanded
                                .iter()
                                .enumerate()
                                .map(|(c, v)| logit_coeffs[k * ncols + 1 + c] * v)
                                .sum::<f64>();
                        let want_eta = row[0]
                            + expanded.iter().enumerate().map(|(c, v)| row[1 + c] * v).sum::<f64>();
                        assert!(
                            (got - want_eta).abs() <= 1e-5 * want_eta.abs().max(1.0),
                            "{name} {tag}: row {i} logit {got} vs {want_eta}"
                        );
                    }
                }
            };
            check(&store.slots[2], &case["logits"], "unweighted");
            let weights = floats(&case["weights"]);
            let refit = registry.refit_weighted(&model, &data, &assignments, &weights).unwrap();
            check(&refit.slots[2], &case["weighted_logits"], "weighted");
            continue;
        }
        let family = if kind == "discrete" {
            MechanismFamily::Discrete
        } else {
            MechanismFamily::HierarchicalGlm
        };
        // `Binary` is a declared discrete type that needs no category domain; the level count
        // is the support, whatever it is.
        let outcome_type = ValueType::Binary;
        let data = table(&[xs[0].clone(), xs[1].clone(), y.clone()], outcome_type);
        let options = SelectionOptions {
            constraints: vec![
                (VariableId::from_raw(0), required(MechanismFamily::LinearGaussian)),
                (VariableId::from_raw(1), required(MechanismFamily::LinearGaussian)),
                (VariableId::from_raw(2), required(family)),
            ],
            ..SelectionOptions::default()
        };
        let (store, assignments) = registry
            .assign_and_fit_with(&model, &data, SelectionPolicy::BestScore, &options)
            .unwrap();
        let ncols = 1 + xs.len();
        let check_logits = |slot: &MechanismSlot, want: &Value, probs: &Value, tag: &str| {
            let MechanismSlot::Discrete { support, probs: got_probs, logit_coeffs } = slot else {
                panic!("{name} {tag}: a discrete slot");
            };
            if let Some(support_want) = case.get("support") {
                assert_eq!(support.to_vec(), floats(support_want), "{name} {tag}: support");
            }
            for (k, want_p) in floats(probs).into_iter().enumerate() {
                close(got_probs[k], want_p, 1e-12, &format!("{name} {tag}: probability {k}"));
            }
            let logits = logit_coeffs.as_ref().expect("parent-conditional logits");
            for (k, row) in want.as_array().unwrap().iter().enumerate() {
                let row = floats(row);
                // The reference lists [intercept, slope of x1, slope of x2]; the slot lists the
                // slopes in gather order.
                close(logits[k * ncols], row[0], 1e-7, &format!("{name} {tag}: intercept {k}"));
                for (pos, &parent) in order.iter().enumerate() {
                    close(
                        logits[k * ncols + 1 + pos],
                        row[1 + parent],
                        1e-7,
                        &format!("{name} {tag}: slope {k}/{parent}"),
                    );
                }
                // The conditional law is what matters: logits at every training row.
                for i in 0..y.len() {
                    let eta_got: f64 = logits[k * ncols]
                        + order
                            .iter()
                            .enumerate()
                            .map(|(pos, &parent)| logits[k * ncols + 1 + pos] * xs[parent][i])
                            .sum::<f64>();
                    let eta_want: f64 =
                        row[0] + (0..xs.len()).map(|c| row[1 + c] * xs[c][i]).sum::<f64>();
                    assert!(
                        (eta_got - eta_want).abs() <= 1e-6 * eta_want.abs().max(1.0),
                        "{name} {tag}: row {i} logit {eta_got} vs {eta_want}"
                    );
                }
            }
        };
        let fitted = &store.slots[2];
        if kind == "discrete" {
            check_logits(fitted, &case["logits"], &case["probs"], "unweighted");
            if let Some(weights) = case.get("weights") {
                let weights = floats(weights);
                let refit = registry.refit_weighted(&model, &data, &assignments, &weights).unwrap();
                check_logits(
                    &refit.slots[2],
                    &case["weighted_logits"],
                    &case["weighted_probs"],
                    "weighted",
                );
            }
        } else {
            let MechanismSlot::Discrete { support, probs, logit_coeffs } = fitted else {
                panic!("{name}: HierarchicalGlm is encoded as a two-category discrete slot");
            };
            assert_eq!(support.to_vec(), vec![0.0, 1.0]);
            let p1 = case["p1"].as_f64().unwrap();
            close(probs[1], p1, 1e-12, name);
            close(probs[0], 1.0 - p1, 1e-12, name);
            let logits = logit_coeffs.as_ref().unwrap();
            assert!(logits[..ncols].iter().all(|v| *v == 0.0), "{name}: reference category");
            let want = floats(&case["coefficients"]);
            close(logits[ncols], want[0], 1e-7, &format!("{name}: intercept"));
            for (pos, &parent) in order.iter().enumerate() {
                close(
                    logits[ncols + 1 + pos],
                    want[1 + parent],
                    1e-7,
                    &format!("{name}: slope {parent} (lambda {})", case["lambda"]),
                );
            }
        }
    }
}

/// A declared `UnitId` column is a statement about the data's structure. The empirical-Bayes
/// ridge logit shrinks slopes but has no unit-level term, so under a usable grouping it must
/// refuse rather than fit as if the units were not declared.
#[test]
fn registry_glm_reference_hierarchical_glm_refuses_a_declared_unit_structure() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/registry_glm_reference.json")).unwrap();
    let case = cases
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "hierarchical" && c["name"] == "signal")
        .unwrap();
    let xs: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
    let y = floats(&case["y"]);
    let units: Vec<f64> = (0..y.len()).map(|i| (i % 4) as f64).collect();
    let mut b = CausalSchemaBuilder::new();
    for (i, name) in ["x1", "x2", "y", "unit"].into_iter().enumerate() {
        let (vt, hint) = match i {
            2 => (ValueType::Binary, RoleHint::OutcomeCandidate),
            3 => (ValueType::Continuous, RoleHint::UnitId),
            _ => (ValueType::Continuous, RoleHint::OutcomeCandidate),
        };
        b.add_variable(
            name,
            vt,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let validity = ValidityBitmap::all_valid(y.len());
    let cols = [xs[0].clone(), xs[1].clone(), y.clone(), units]
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(v),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TabularData::new(
        OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap(),
    );
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let registry = MechanismRegistry::standard();
    let options = SelectionOptions {
        constraints: vec![
            (VariableId::from_raw(0), required(MechanismFamily::LinearGaussian)),
            (VariableId::from_raw(1), required(MechanismFamily::LinearGaussian)),
            (VariableId::from_raw(2), required(MechanismFamily::HierarchicalGlm)),
        ],
        ..SelectionOptions::default()
    };
    let error = registry
        .assign_and_fit_with(&model, &data, SelectionPolicy::BestScore, &options)
        .map(|_| ())
        .unwrap_err();
    assert!(
        error.to_string().contains("UnitId"),
        "the refusal names the declaration it cannot honour: {error}"
    );
    // Without the requirement the family is a recorded failure and another candidate is chosen.
    let options = SelectionOptions {
        constraints: vec![
            (VariableId::from_raw(0), required(MechanismFamily::LinearGaussian)),
            (VariableId::from_raw(1), required(MechanismFamily::LinearGaussian)),
            (
                VariableId::from_raw(2),
                NodeFamilyConstraint {
                    allowed: Some(Arc::from([
                        MechanismFamily::Discrete,
                        MechanismFamily::HierarchicalGlm,
                    ])),
                    ..NodeFamilyConstraint::default()
                },
            ),
        ],
        ..SelectionOptions::default()
    };
    let (_, assignments) =
        registry.assign_and_fit_with(&model, &data, SelectionPolicy::BestScore, &options).unwrap();
    assert_eq!(assignments[2].selected, MechanismFamily::Discrete);
    assert!(
        assignments[2]
            .failed_families
            .iter()
            .any(|(f, why)| *f == MechanismFamily::HierarchicalGlm && why.contains("UnitId"))
    );
}
