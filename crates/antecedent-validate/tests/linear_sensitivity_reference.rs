//! The simulated-confounder sensitivity grid against explicit 50-digit perturbation refits.
//!
//! `oracle_data/linear_sensitivity_reference.py` (frozen in the JSON beside it) builds one
//! explicit unit-variance confounder orthogonal to the design, perturbs treatment and outcome at
//! every grid strength, refits by the normal equations in 50-digit arithmetic and records the
//! first strength that flips the estimate's sign, with the closed-form tipping value
//! `|rho| / (1 + |rho|)` alongside. The same data are also supplied with the outcome and the
//! treatment rescaled by factors from 1e-12 to 1e8: a tipping partial R^2 is dimensionless, so the
//! verdict must not move.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_expr::ExprId;
use antecedent_identify::IdentifiedEstimand;
use antecedent_validate::{
    LinearSensitivity, NonparametricSensitivity, PartialLinearSensitivity, RefutationProblem,
};
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/linear_sensitivity_reference.json"))
        .unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn table(case: &Value) -> (TabularData, IdentifiedEstimand) {
    let t = floats(&case["t"]);
    let y = floats(&case["y"]);
    let zs: Vec<Vec<f64>> = case["z"].as_array().unwrap().iter().map(floats).collect();
    let n = t.len();
    let mut b = CausalSchemaBuilder::new();
    let mut columns = vec![t, y];
    columns.extend(zs.iter().cloned());
    for (j, _) in columns.iter().enumerate() {
        let (name, role) = match j {
            0 => ("t".to_owned(), RoleHint::TreatmentCandidate),
            1 => ("y".to_owned(), RoleHint::OutcomeCandidate),
            _ => (format!("z{j}"), RoleHint::Context),
        };
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let cols = columns
        .into_iter()
        .enumerate()
        .map(|(j, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(j).unwrap()),
                    Arc::from(v),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap();
    let adjustment: Vec<VariableId> =
        (0..zs.len()).map(|j| VariableId::from_raw(u32::try_from(j + 2).unwrap())).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    (TabularData::new(storage), estimand)
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

#[test]
fn the_tipping_strength_and_refit_estimate_match_the_explicit_perturbation() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (data, estimand) = table(&case);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(7);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        close(original.ate, case["ate"].as_f64().unwrap(), 1e-9, &format!("{name} original"));
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let grid: Vec<f64> =
            case["grid"].as_array().unwrap().iter().map(|g| g["r"].as_f64().unwrap()).collect();
        let expected_tip = case["tip"].as_f64();
        for (label, report) in [
            (
                "linear",
                LinearSensitivity {
                    partial_r2_grid: grid.clone(),
                    pass_threshold: 0.1,
                    estimator: est.clone(),
                }
                .refute(&problem, &mut ws, &ctx)
                .unwrap(),
            ),
            (
                "partial linear",
                PartialLinearSensitivity {
                    partial_r2_grid: grid.clone(),
                    pass_threshold: 0.1,
                    estimator: est.clone(),
                }
                .refute(&problem, &mut ws, &ctx)
                .unwrap(),
            ),
        ] {
            let what = format!("{name} {label}");
            match expected_tip {
                Some(tip) => {
                    assert_eq!(report.comparison, tip, "{what}: tipping strength");
                    let row = case["grid"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|g| g["r"].as_f64().unwrap() == tip)
                        .unwrap();
                    let want = row["ate"].as_f64().unwrap();
                    // The refit slope at the tipping strength is a small difference of large
                    // terms; compare against the scale of the original estimate.
                    assert!(
                        (report.refuted_ate - want).abs()
                            <= 1e-8 * case["ate"].as_f64().unwrap().abs(),
                        "{what}: refuted {} want {want}",
                        report.refuted_ate
                    );
                }
                None => assert!(report.comparison.is_infinite(), "{what}"),
            }
            // The passing rule follows the cleared grid value, never the tipped one.
            let cleared = grid
                .iter()
                .copied()
                .filter(|&g| expected_tip.is_none_or(|t| g < t))
                .fold(0.0_f64, f64::max);
            assert_eq!(report.passed, cleared >= 0.1, "{what}");
        }
    }
}

#[test]
fn the_realized_partial_r_squared_equals_the_grid_value_and_the_tip_matches_the_closed_form() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        for row in case["grid"].as_array().unwrap() {
            let (r, p) = (row["r"].as_f64().unwrap(), row["partial_r2"].as_f64().unwrap());
            close(p, r, 1e-9, &format!("{name} partial R2 at {r}"));
        }
        let closed = case["closed_form_tip"].as_f64().unwrap();
        let tip = case["tip"].as_f64().unwrap();
        let grid: Vec<f64> =
            case["grid"].as_array().unwrap().iter().map(|g| g["r"].as_f64().unwrap()).collect();
        // The first grid value at or above the closed-form tipping strength.
        let first = grid.iter().copied().find(|&g| g >= closed).unwrap();
        assert_eq!(tip, first, "{name}");
    }
}

#[test]
fn a_grid_that_stops_below_the_tipping_strength_reports_no_tipping() {
    let case = &cases()[1];
    let (data, estimand) = table(case);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(7);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let short = LinearSensitivity {
        partial_r2_grid: vec![0.01, 0.05, 0.1],
        pass_threshold: 0.1,
        estimator: est.clone(),
    }
    .refute(&problem, &mut ws, &ctx)
    .unwrap();
    assert!(short.comparison.is_infinite());
    // Untipped, but the grid only certified 0.05 cleared... its largest cleared value is 0.1 here.
    assert!(short.passed);
    // The nonparametric variant is a different construction and is not asserted here.
    let _ = NonparametricSensitivity::new();
}
