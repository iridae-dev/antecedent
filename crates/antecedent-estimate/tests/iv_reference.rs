//! Wald and two-stage least squares fits, first-stage diagnostics and the Anderson-Rubin set
//! against an exact rational / 60-digit reference.
//!
//! `oracle_data/iv_reference.py` (frozen in the JSON beside it) fits every case by exact
//! rational least squares and finds the Anderson-Rubin set by evaluating the statistic with two
//! direct regressions per trial value, so it shares no algebra with the moment form the product
//! uses. Cases: one binary instrument, continuous instruments with covariates, weak instruments
//! giving a bounded, an unbounded and a two-piece acceptance set, and one design rescaled by
//! 1e-9 and 1e9 in treatment and outcome (the set must rescale exactly).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint};
use antecedent_estimate::{TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/iv_reference.json")).unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

struct Built {
    data: antecedent_data::TabularData,
    estimand: IdentifiedEstimand,
}

fn build(case: &Value) -> Built {
    let mut columns = vec![
        ("t", RoleHint::TreatmentCandidate, floats(&case["t"])),
        ("y", RoleHint::OutcomeCandidate, floats(&case["y"])),
    ];
    let mut instruments = Vec::new();
    for (i, z) in case["z"].as_array().unwrap().iter().enumerate() {
        instruments.push(id(2 + i as u32));
        columns.push(("z", RoleHint::Context, floats(z)));
    }
    let mut adjustment = Vec::new();
    for (i, x) in case["x"].as_array().unwrap().iter().enumerate() {
        adjustment.push(id(2 + instruments.len() as u32 + i as u32));
        columns.push(("x", RoleHint::Context, floats(x)));
    }
    let mut estimand =
        IdentifiedEstimand::instrumental("iv", Arc::from(instruments), ExprId::from_raw(0));
    estimand.adjustment_set = Arc::from(adjustment);
    Built { data: table(&columns), estimand }
}

fn query(active: f64, control: f64) -> AverageEffectQuery {
    AverageEffectQuery::with_levels(id(0), id(1), control, active)
}

fn check(case: &Value, active: f64, control: f64, use_wald: bool) {
    let name = case["name"].as_str().unwrap();
    let delta = active - control;
    let b = build(case);
    let ctx = ExecutionContext::for_tests(5);
    let effect = if use_wald {
        let est = WaldIv::new().with_bootstrap_replicates(0);
        let prep = est.prepare(&b.data, &b.estimand, &query(active, control)).unwrap();
        est.fit(&prep, &ctx, AssumptionSet::new()).unwrap()
    } else {
        let est = TwoStageLeastSquares::new().with_bootstrap_replicates(0);
        let prep = est.prepare(&b.data, &b.estimand, &query(active, control)).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap()
    };
    let coef = case["coef"].as_f64().unwrap();
    close(effect.ate, coef * delta, 1e-9, &format!("{name} coefficient"));
    let diag = effect.diagnostics.first_stage.as_ref().expect("first stage");
    assert_eq!(diag.df1, case["k"].as_u64().unwrap() as usize, "{name}");
    assert_eq!(diag.df2, case["df2"].as_u64().unwrap() as usize, "{name}");
    close(diag.f_statistic, case["f_stat"].as_f64().unwrap(), 1e-8, &format!("{name} F"));
    close(diag.partial_r2, case["partial_r2"].as_f64().unwrap(), 1e-8, &format!("{name} R2"));
    let kind = case["ar_kind"].as_str().unwrap();
    match kind {
        "interval" => {
            let (lo, hi, level) = diag.anderson_rubin.unwrap_or_else(|| {
                panic!("{name}: AR set missing ({:?})", diag.uncertainty_withheld)
            });
            assert_eq!(level, 0.95);
            let (mut wl, mut wh) =
                (case["ar_lo"].as_f64().unwrap() * delta, case["ar_hi"].as_f64().unwrap() * delta);
            if delta < 0.0 {
                std::mem::swap(&mut wl, &mut wh);
            }
            close(lo, wl, 1e-7, &format!("{name} AR lower"));
            close(hi, wh, 1e-7, &format!("{name} AR upper"));
            assert!(lo <= effect.ate && effect.ate <= hi, "{name}: estimate outside its set");
        }
        "whole" => {
            let (lo, hi, _) = diag.anderson_rubin.unwrap_or_else(|| {
                panic!("{name}: AR set missing ({:?})", diag.uncertainty_withheld)
            });
            assert!(lo == f64::NEG_INFINITY && hi == f64::INFINITY, "{name}: [{lo}, {hi}]");
        }
        "union" => {
            assert!(diag.anderson_rubin.is_none(), "{name}");
            assert_eq!(diag.uncertainty_withheld, Some("anderson_rubin_set_is_union"), "{name}");
        }
        other => panic!("unexpected reference class {other}"),
    }
}

#[test]
fn two_stage_fit_first_stage_and_anderson_rubin_set_match_the_exact_reference() {
    for case in cases() {
        check(&case, 1.0, 0.0, false);
    }
}

#[test]
fn the_wald_estimator_matches_for_one_binary_instrument_and_delegates_otherwise() {
    for case in cases() {
        check(&case, 1.0, 0.0, true);
    }
}

#[test]
fn the_set_scales_with_the_contrast_and_swaps_when_the_contrast_is_negative() {
    for case in cases().iter().filter(|c| c["ar_kind"] == "interval").take(3) {
        check(case, 3.0, 1.0, false);
        check(case, 1.0, 3.5, false);
    }
}
