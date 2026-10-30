//! Penalised and expanded linear-Gaussian mechanism fits against exact normal-equation solves.
//!
//! `oracle_data/registry_linear_families_reference.py` (frozen in the JSON beside it) solves, in
//! 60-digit arithmetic and from the normal equations rather than a QR of augmented rows: the
//! Minnesota-style conjugate ridge of `Bvar` on scale-standardised parents, the empirical-Bayes
//! ridge of `HierarchicalLinear` (using centered-design sampling variance for its raw-slope prior, without and with a `UnitId` column
//! whose within-unit regression supplies the slopes), the interaction expansion of
//! `LinearInteractions` and the cubic truncated-power spline of `LinearSpline` (its conditioning
//! ridge on root-mean-square scaled columns), the last two both as a fresh fit and as a refit
//! under row weights. The registry fits are taken from `assign_and_fit_with` with the family
//! required, and the weighted refits from `refit_weighted`.
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

fn table(columns: &[Vec<f64>], units: Option<&[f64]>) -> TabularData {
    let mut b = CausalSchemaBuilder::new();
    let mut names = vec!["x1", "x2", "y"];
    if units.is_some() {
        names.push("unit");
    }
    for (i, name) in names.iter().enumerate() {
        let hint = if i == 3 { RoleHint::UnitId } else { RoleHint::OutcomeCandidate };
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(columns[0].len());
    let mut all: Vec<Vec<f64>> = columns.to_vec();
    if let Some(u) = units {
        all.push(u.to_vec());
    }
    let cols = all
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

fn fit(
    registry: &MechanismRegistry,
    model: &CompiledCausalModel,
    data: &TabularData,
    family: MechanismFamily,
) -> (antecedent_model::CompiledMechanismStore, Vec<antecedent_model::MechanismAssignment>) {
    let options = SelectionOptions {
        constraints: vec![
            (VariableId::from_raw(0), required(MechanismFamily::LinearGaussian)),
            (VariableId::from_raw(1), required(MechanismFamily::LinearGaussian)),
            (VariableId::from_raw(2), required(family)),
        ],
        ..SelectionOptions::default()
    };
    registry.assign_and_fit_with(model, data, SelectionPolicy::BestScore, &options).unwrap()
}

#[test]
fn registry_linear_families_reference_penalised_and_expanded_fits_match_exact_solves() {
    let cases: Value =
        serde_json::from_str(include_str!("oracle_data/registry_linear_families_reference.json"))
            .unwrap();
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let order: Vec<usize> = model
        .gather_for(DenseNodeId::from_raw(2))
        .unwrap()
        .parents
        .iter()
        .map(|p| p.as_usize())
        .collect();
    assert_eq!(order, vec![0, 1], "the references list parents in node order");
    let registry = MechanismRegistry::standard();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let family = case["family"].as_str().unwrap();
        let xs: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let y = floats(&case["y"]);
        let columns = [xs[0].clone(), xs[1].clone(), y.clone()];
        match family {
            "shrinkage" | "grouped" => {
                let units = case.get("groups").map(|g| floats(g));
                let data = table(&columns, units.as_deref());
                if let Some(bvar) = case.get("bvar") {
                    let (store, _) = fit(&registry, &model, &data, MechanismFamily::Bvar);
                    let MechanismSlot::Bvar { intercept, coeffs, sigma } = &store.slots[2] else {
                        panic!("{name}: Bvar slot");
                    };
                    close(*intercept, bvar["intercept"].as_f64().unwrap(), 1e-9, name);
                    for (j, want) in floats(&bvar["coeffs"]).into_iter().enumerate() {
                        close(coeffs[j], want, 1e-9, &format!("{name} bvar slope {j}"));
                    }
                    close(*sigma, bvar["sigma"].as_f64().unwrap(), 1e-9, name);
                }
                let hier = &case["hierarchical"];
                let (store, _) = fit(&registry, &model, &data, MechanismFamily::HierarchicalLinear);
                let MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, shrinkage } =
                    &store.slots[2]
                else {
                    panic!("{name}: HierarchicalLinear slot");
                };
                close(*shrinkage, hier["shrinkage"].as_f64().unwrap(), 1e-9, name);
                close(*intercept, hier["intercept"].as_f64().unwrap(), 1e-9, name);
                for (j, want) in floats(&hier["coeffs"]).into_iter().enumerate() {
                    close(coeffs[j], want, 1e-9, &format!("{name} hierarchical slope {j}"));
                }
                close(*sigma, hier["sigma"].as_f64().unwrap(), 1e-9, name);
            }
            "interactions" | "spline" => {
                let data = table(&columns, None);
                let which = if family == "interactions" {
                    MechanismFamily::LinearInteractions
                } else {
                    MechanismFamily::LinearSpline
                };
                let (store, assignments) = fit(&registry, &model, &data, which);
                let check = |slot: &MechanismSlot, want: &Value, tag: &str| {
                    let MechanismSlot::LinearBasis { intercept, basis, coeffs, sigma } = slot
                    else {
                        panic!("{name} {tag}: LinearBasis slot");
                    };
                    assert_eq!(basis.n_terms(), want["n_terms"].as_u64().unwrap() as usize);
                    let want_coeffs = floats(&want["coeffs"]);
                    // The truncated-power columns are strongly correlated, so a coefficient is
                    // judged against the largest one; the fitted surface is judged row by row
                    // below at a much tighter tolerance.
                    let big = want_coeffs.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
                    let tol = if family == "spline" { 1e-6 } else { 1e-9 };
                    close(*intercept, want["intercept"].as_f64().unwrap(), tol * big, name);
                    for (j, w) in want_coeffs.iter().enumerate() {
                        assert!(
                            (coeffs[j] - w).abs() <= tol * big,
                            "{name} {tag}: coefficient {j}: {} vs {w}",
                            coeffs[j]
                        );
                    }
                    close(*sigma, want["sigma"].as_f64().unwrap(), 1e-7, &format!("{name} {tag}"));
                    let mut row = [0.0; 2];
                    let mut expanded = vec![0.0; basis.n_terms()];
                    for (i, fitted) in floats(&want["fitted"]).into_iter().enumerate() {
                        row[0] = xs[0][i];
                        row[1] = xs[1][i];
                        basis.expand_row(&row, &mut expanded).unwrap();
                        let pred = intercept
                            + expanded.iter().zip(coeffs.iter()).map(|(a, b)| a * b).sum::<f64>();
                        close(pred, fitted, 1e-7, &format!("{name} {tag}: fitted {i}"));
                    }
                };
                check(&store.slots[2], &case["unweighted"], "unweighted");
                let weights = floats(&case["weights"]);
                let refit = registry.refit_weighted(&model, &data, &assignments, &weights).unwrap();
                check(&refit.slots[2], &case["weighted"], "weighted");
            }
            other => panic!("unknown family {other}"),
        }
    }
}
