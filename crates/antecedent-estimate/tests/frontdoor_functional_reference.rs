//! Front-door plug-in functional against its definition-level reference.
//!
//! `oracle_data/frontdoor_functional_reference.py` (frozen in the JSON beside it) writes the
//! functional as a smooth map of per-row masses at 80 digits, takes the effect at unit masses and
//! the influence of each row as the derivative of the effect with respect to its mass by central
//! differences, and compares the saturated cell-mean outcome model and the per-arm least-squares
//! outcome model, including a three-arm two-mediator contrast, a treated arm of five rows,
//! mediator offsets of 1e3 and 1e4 and two nearly collinear mediators. The standard error is the
//! square root of the summed squared influences over n.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint};
use antecedent_estimate::{FrontDoorFunctional, FrontDoorOutcomeModel};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-12), "{what}: got {got} want {want}");
}

fn data_and_estimand(
    t: &[f64],
    y: &[f64],
    m: &[Vec<f64>],
) -> (antecedent_data::TabularData, IdentifiedEstimand) {
    let k = m[0].len();
    let mut columns = vec![
        ("t", RoleHint::TreatmentCandidate, t.to_vec()),
        ("y", RoleHint::OutcomeCandidate, y.to_vec()),
    ];
    for j in 0..k {
        columns.push(("m", RoleHint::Context, m.iter().map(|r| r[j]).collect()));
    }
    let mediators: Vec<_> = (0..k).map(|j| id(2 + u32::try_from(j).unwrap())).collect();
    (
        table(&columns),
        IdentifiedEstimand::frontdoor("frontdoor", Arc::from(mediators), ExprId::from_raw(0)),
    )
}

#[test]
fn plug_in_effect_and_influence_standard_error_match_the_mass_derivative_reference() {
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/frontdoor_functional_reference.json"))
            .unwrap();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let m: Vec<Vec<f64>> = case["m"].as_array().unwrap().iter().map(floats).collect();
        let (data, estimand) = data_and_estimand(&t, &y, &m);
        let model = match case["model"].as_str().unwrap() {
            "saturated" => FrontDoorOutcomeModel::Saturated,
            _ => FrontDoorOutcomeModel::ArmLinear,
        };
        let estimator =
            FrontDoorFunctional::new().with_bootstrap_replicates(0).with_outcome_model(model);
        let query = AverageEffectQuery::with_levels(
            id(0),
            id(1),
            case["control"].as_f64().unwrap(),
            case["active"].as_f64().unwrap(),
        );
        let problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let got = estimator
            .fit(&problem, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        close(got.ate, case["ate"].as_f64().unwrap(), 1e-8, &format!("{name} effect"));
        close(got.se_analytic(), case["se"].as_f64().unwrap(), 1e-6, &format!("{name} se"));
    }
}

#[test]
fn a_mediator_cell_missing_from_an_arm_is_refused_not_extrapolated() {
    let t = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let y = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
    let m = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0].map(|v| vec![v]);
    let (data, estimand) = data_and_estimand(&t, &y, &m);
    let estimator = FrontDoorFunctional::new()
        .with_bootstrap_replicates(0)
        .with_outcome_model(FrontDoorOutcomeModel::Saturated);
    let query = AverageEffectQuery::with_levels(id(0), id(1), 0.0, 1.0);
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    let refused = estimator
        .fit(&problem, &ExecutionContext::for_tests(3), AssumptionSet::new())
        .err()
        .expect("a mediator cell missing from an arm must be refused");
    assert_eq!(
        refused.to_string(),
        "frontdoor.functional (saturated): a mediator value observed under a contrasted \
         treatment level never occurs under another treatment level, so E[Y | M, T] has no data \
         there; coarsen the mediator or request the arm-linear outcome model"
    );
}
