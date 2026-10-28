//! The anomaly score, its exact noise-Shapley attribution and the arrow strength against
//! 80-digit evaluation of their definitions (`oracle_data/anomaly_arrow_reference.py`).
//!
//! Covers a fixed and an empirical (median / 1.4826 MAD) reference on a three-node model with a
//! chord, targets from the bulk to a far tail past the log-space asymptotic switch, the
//! degenerate-bulk scale fallbacks, and arrow strengths on a diamond with a chord whose parents
//! are correlated. The arrow strength is the cut-edge Kullback-Leibler divergence, evaluated
//! independently from the Gaussian divergence of the joint covariances; the variance contribution
//! `beta^2 Var(parent)` is checked alongside, and the test records where the normalised form with
//! the child's whole variance departs from the divergence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::{arrow_strengths, score_anomalies};
use antecedent_core::{
    AnomalyAttributionQuery, AnomalyReference, CausalSchemaBuilder, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, CompiledMechanismStore, MechanismSlot};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/anomaly_arrow_reference.json"))
        .expect("frozen reference")
}

fn data_for(columns: &[Vec<f64>]) -> TabularData {
    let n = columns[0].len();
    let mut b = CausalSchemaBuilder::new();
    for k in 0..columns.len() {
        b.add_variable(
            format!("v{k}"),
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
    let cols = columns
        .iter()
        .enumerate()
        .map(|(k, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(k as u32),
                    Arc::from(v.clone()),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

fn linear(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs), sigma }
}

fn f(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * (1.0 + want.abs()), "{what}: {got} vs {want}");
}

/// x -> y -> z with the chord x -> z, mechanisms from the reference's model block.
fn chain(reference: &Value) -> (CompiledCausalModel, TabularData) {
    let m = &reference["model"];
    let rows = reference["rows"].as_array().expect("rows");
    let col = |k: usize| rows.iter().map(|r| f(&r[k])).collect::<Vec<_>>();
    let data = data_for(&[col(0), col(1), col(2)]);
    let mut g = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let store = CompiledMechanismStore {
        slots: Arc::from([
            linear(f(&m["a0"]), &[], 1.0),
            linear(f(&m["a1"]), &[f(&m["b1"])], 1.0),
            linear(f(&m["a2"]), &[f(&m["b2"]), f(&m["c2"])], 1.0),
        ]),
    };
    (CompiledCausalModel::compile(g).unwrap().with_mechanisms(store), data)
}

fn check_case(reference: &Value, case: &Value, query_reference: AnomalyReference) {
    let (model, data) = chain(reference);
    let q = AnomalyAttributionQuery::new([VariableId::from_raw(2)], 100)
        .with_reference(query_reference);
    let scores = &antecedent_attribution::score_anomalies(&model, &data, &q).unwrap()[0];
    assert_eq!(scores.noise_components.len(), 3);
    for (i, want) in case["rows"].as_array().unwrap().iter().enumerate() {
        close(scores.scores[i], f(&want["score"]), 1e-11, &format!("row {i} score"));
        close(scores.baseline_scores[i], f(&want["baseline"]), 1e-11, &format!("row {i} baseline"));
        let mut sum = 0.0;
        for (j, phi) in want["phi"].as_array().unwrap().iter().enumerate() {
            let got = scores.noise_contributions[i * 3 + j];
            close(got, f(phi), 1e-10, &format!("row {i} phi {j}"));
            sum += got;
        }
        // Efficiency of the exact game, from the product's published numbers.
        close(sum, scores.scores[i] - scores.baseline_scores[i], 1e-10, "efficiency");
    }
}

#[test]
fn fixed_reference_scores_and_exact_shapley_equal_the_definition() {
    let r = reference();
    let case = &r["fixed"];
    check_case(&r, case, AnomalyReference::fixed(f(&case["center"]), f(&case["scale"])));
}

#[test]
fn empirical_reference_uses_the_median_and_scaled_mad_of_the_target_column() {
    let r = reference();
    let case = &r["empirical"];
    // The reference scores against its own recomputed (center, scale): a Fixed query at the
    // reference's median / 1.4826 MAD must agree with the Empirical query to the last bits.
    check_case(&r, case, AnomalyReference::Empirical);
    let (model, data) = chain(&r);
    let q = AnomalyAttributionQuery::new([VariableId::from_raw(2)], 100);
    let emp = &score_anomalies(&model, &data, &q).unwrap()[0];
    let fixed = &score_anomalies(
        &model,
        &data,
        &q.clone().with_reference(AnomalyReference::fixed(f(&case["center"]), f(&case["scale"]))),
    )
    .unwrap()[0];
    for i in 0..emp.scores.len() {
        close(emp.scores[i], fixed.scores[i], 1e-12, "empirical vs fixed at the reference values");
    }
}

#[test]
fn degenerate_bulk_falls_back_to_the_mean_absolute_deviation_then_to_one() {
    let r = reference();
    let d = &r["degenerate"];
    let z: Vec<f64> = d["z"].as_array().unwrap().iter().map(f).collect();
    // A single-node model so that the target column is exactly the sample under test.
    let data = data_for(&[z.clone()]);
    let model = CompiledCausalModel::compile(Dag::with_variables(1))
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from([linear(0.0, &[], 1.0)]) });
    let q = AnomalyAttributionQuery::new([VariableId::from_raw(0)], 100);
    let emp = &score_anomalies(&model, &data, &q).unwrap()[0];
    let fixed_q =
        q.clone().with_reference(AnomalyReference::fixed(f(&d["center"]), f(&d["scale"])));
    let fixed = &score_anomalies(&model, &data, &fixed_q).unwrap()[0];
    for i in 0..z.len() {
        close(emp.scores[i], fixed.scores[i], 1e-12, "mean-absolute-deviation fallback");
    }
    // Every value identical: scale 1 and centre at the value, so every score is exactly zero.
    let same = data_for(&[vec![5.0; 6]]);
    let out = &score_anomalies(&model, &same, &q).unwrap()[0];
    assert!(out.scores.iter().all(|s| *s == 0.0), "{:?}", out.scores);
    assert_eq!(f(&d["all_same_scale"]), 1.0);
}

#[test]
fn far_tail_scores_follow_the_exact_tail_across_the_asymptotic_switch() {
    let r = reference();
    let data = data_for(&[vec![0.0, 1.0, -1.0]]);
    let model = CompiledCausalModel::compile(Dag::with_variables(1))
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from([linear(0.0, &[], 1.0)]) });
    for point in r["tail"].as_array().unwrap() {
        let z = f(&point["z"]);
        // A one-row table whose value sits z scales from a unit-scale centre.
        let one = data_for(&[vec![z]]);
        let q = AnomalyAttributionQuery::new([VariableId::from_raw(0)], 10)
            .with_reference(AnomalyReference::fixed(0.0, 1.0));
        let got = score_anomalies(&model, &one, &q).unwrap()[0].scores[0];
        close(got, f(&point["score"]), 1e-9, &format!("z={z}"));
    }
    let _ = data;
    // A deviation of 1e200 scales overflows `z^2`: refused rather than scored as infinite.
    let huge = data_for(&[vec![1e200]]);
    let q = AnomalyAttributionQuery::new([VariableId::from_raw(0)], 10)
        .with_reference(AnomalyReference::fixed(0.0, 1.0));
    assert!(score_anomalies(&model, &huge, &q).is_err());
}

#[test]
fn arrow_strength_is_the_cut_edge_divergence_and_the_variance_contribution_is_reported_too() {
    let r = reference();
    let a = &r["arrow"];
    let order = ["a", "b", "c", "d"];
    let betas: Vec<(&str, &str, f64)> = a["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e["parent"].as_str().unwrap(), e["child"].as_str().unwrap(), f(&e["beta"])))
        .collect();
    let sigma = [1.3, 0.6, 0.9, 0.4];
    let data = data_for(&vec![vec![0.0, 1.0]; 4]);
    let mut g = Dag::with_variables(4);
    let index = |s: &str| order.iter().position(|o| *o == s).unwrap();
    for &(p, c, _) in &betas {
        g.insert_directed(
            DenseNodeId::from_raw(index(p) as u32),
            DenseNodeId::from_raw(index(c) as u32),
        )
        .unwrap();
    }
    let compiled = CompiledCausalModel::compile(g).unwrap();
    // Coefficients in the parent order the compiled gather uses.
    let slots: Vec<MechanismSlot> = (0..4)
        .map(|child| {
            let gather = compiled.gather_for(DenseNodeId::from_raw(child as u32)).unwrap();
            let coeffs: Vec<f64> = gather
                .parents
                .iter()
                .map(|p| {
                    betas
                        .iter()
                        .find(|(pp, cc, _)| index(pp) == p.as_usize() && index(cc) == child)
                        .unwrap()
                        .2
                })
                .collect();
            linear(0.0, &coeffs, sigma[child])
        })
        .collect();
    let model = compiled.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });
    let _ = data;
    let got = arrow_strengths(&model).unwrap();
    assert_eq!(got.len(), 5);
    let mut exceeds_child_variance = 0;
    let mut departs_from_normalised = 0;
    for want in a["edges"].as_array().unwrap() {
        let parent = VariableId::from_raw(index(want["parent"].as_str().unwrap()) as u32);
        let child = VariableId::from_raw(index(want["child"].as_str().unwrap()) as u32);
        let e = got.iter().find(|e| e.parent == parent && e.child == child).expect("edge");
        close(e.strength, f(&want["strength"]), 1e-13, "cut-edge divergence");
        close(e.variance_contribution, f(&want["variance_contribution"]), 1e-13, "beta^2 Var");
        close(e.coefficient, f(&want["beta"]), 0.0, "coefficient");
        // The normalised form -1/2 ln(1 - beta^2 Var(parent) / Var(child)) equals the divergence
        // only when the edge is the child's single parent; with several parents the other
        // parents stay connected and the divergence is not that number (and the normalised form
        // does not exist when the contribution exceeds Var(child)).
        match want["normalised_form"].as_f64() {
            Some(normalised) if want["child_parents"].as_u64() == Some(1) => {
                close(e.strength, normalised, 1e-13, "single-parent normalised form");
            }
            Some(normalised) => {
                assert!((e.strength - normalised).abs() > 1e-3, "{} vs {normalised}", e.strength);
                departs_from_normalised += 1;
            }
            None => exceeds_child_variance += 1,
        }
    }
    assert_eq!(departs_from_normalised, 2, "the two later edges into d with an existing form");
    assert_eq!(exceeds_child_variance, 1, "the correlated-parent chord exceeds Var(child)");
}
