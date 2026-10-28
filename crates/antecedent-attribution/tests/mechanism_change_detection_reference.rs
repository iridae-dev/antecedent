//! Mechanism-change detection against exact rational residuals and 50-digit statistics.
//!
//! `oracle_data/mechanism_change_detection_reference.py` (frozen in the JSON beside it) builds a
//! chain `x -> y`, `x -> z` with an intercept shift in `y` in the comparison period, forms the
//! cross-fitted baseline residuals (fold = row mod k) and the full-baseline comparison residuals
//! of each target by exact rational least squares, and evaluates from their definitions the Welch
//! mean difference and p-value, the Gaussian likelihood-ratio statistic with its chi-square(2)
//! tail, the known-split change-point statistic, the biased RBF MMD^2 at the median bandwidth and
//! the Benjamini-Hochberg adjusted p-values of the family. The product fits its own mechanisms
//! and reaches the same numbers through its residual construction, so this checks the route:
//! which rows are held out, which model scores them, the pairing of baseline and comparison
//! residuals per target, and the family correction of the raw p-values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::{MechanismChangeMethod, detect_mechanism_changes};
use antecedent_core::MechanismChangeQuery;
use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, PopulationSelector, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::CompiledCausalModel;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1e-300),
        "{what}: {actual} vs {expected}"
    );
}

/// Benjamini-Hochberg step-up adjusted p-values, written from the definition.
fn bh(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut adjusted = vec![0.0; m];
    let mut running = 1.0_f64;
    for rank in (1..=m).rev() {
        let i = order[rank - 1];
        running = running.min(p[i] * m as f64 / rank as f64);
        adjusted[i] = running;
    }
    adjusted
}

#[test]
fn detection_statistics_and_family_correction_match_the_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/mechanism_change_detection_reference.json"))
            .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n_base = usize::try_from(case["n_base"].as_u64().unwrap()).unwrap();
        let n_comp = usize::try_from(case["n_comp"].as_u64().unwrap()).unwrap();
        let n = n_base + n_comp;
        let mut b = CausalSchemaBuilder::new();
        for var in ["x", "y", "z"] {
            b.add_variable(
                var,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let cols: Vec<OwnedColumn> = ["x", "y", "z"]
            .iter()
            .enumerate()
            .map(|(i, key)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Arc::from(floats(&case[*key])),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let query = MechanismChangeQuery::new(
            [VariableId::from_raw(1), VariableId::from_raw(2)],
            PopulationSelector::TimeRange { start: 0, end: n_base },
            PopulationSelector::TimeRange { start: n_base, end: n },
            0.05,
            10,
        );
        let large = n_base.min(n_comp) >= 30;
        for (method, key) in [
            (MechanismChangeMethod::MeanDiff, "mean_diff"),
            (MechanismChangeMethod::LikelihoodRatio, "likelihood_ratio"),
            (MechanismChangeMethod::ChangePoint, "change_point"),
            (MechanismChangeMethod::KernelTwoSample, "kernel"),
        ] {
            let found = detect_mechanism_changes(
                &model,
                &data,
                &query,
                method,
                &ExecutionContext::for_tests(3),
            )
            .unwrap();
            assert_eq!(found.len(), 2, "{name} {key}");
            let raw: Vec<f64> = found.iter().map(|d| d.p_value).collect();
            let adjusted = bh(&raw);
            for (detection, target) in found.iter().zip(["y", "z"]) {
                let want = &case["targets"][target][key];
                let at = format!("{name} {key} {target}");
                close(detection.statistic, want["statistic"].as_f64().unwrap(), 1e-7, &at);
                if let Some(p) = want.get("p_value").and_then(Value::as_f64) {
                    if key == "mean_diff" || large {
                        close(detection.p_value, p, 1e-5, &format!("{at} p"));
                    }
                }
                if key == "kernel" {
                    assert_eq!(detection.n_permutations, Some(999), "{at}");
                    assert!(detection.p_value >= 0.001 - 1e-15 && detection.p_value <= 1.0);
                }
            }
            // The family correction is Benjamini-Hochberg over the raw per-target p-values.
            for (detection, want) in found.iter().zip(&adjusted) {
                close(
                    detection.adjusted_p_value.unwrap(),
                    *want,
                    1e-12,
                    &format!("{name} {key} adjusted"),
                );
                assert_eq!(detection.changed, *want < 0.05, "{name} {key} verdict");
            }
            if key == "mean_diff" {
                let want = floats(&case["mean_diff_adjusted"]);
                for (detection, w) in found.iter().zip(&want) {
                    close(detection.adjusted_p_value.unwrap(), *w, 1e-5, &format!("{name} BH"));
                }
            }
        }
    }
}
