//! Sharp regression discontinuity against exact rational one-sided line fits and sandwiches.
//!
//! `oracle_data/rd_reference.py` (frozen in the JSON beside it) fits a straight line on each side
//! of the cutoff over the rows with `|R - c| <= h` and takes the difference of the fitted values
//! at `c`, then writes the HC0-HC3 and classical variances of the pooled interaction regression
//! from their definitions in exact rational arithmetic. Cases put rows exactly on the cutoff and
//! on both window edges, rows outside the window with extreme outcomes, unequal sides,
//! heteroskedastic noise, a cutoff of 2^20 and a saturated window.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, TargetPopulation,
};
use antecedent_estimate::{AnalyticSeKind, RdWorkspace, SharpRegressionDiscontinuity};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/rd_reference.json")).unwrap()["cases"]
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

fn data_for(r: &[f64], y: &[f64], cutoff: f64) -> antecedent_data::TabularData {
    let t: Vec<f64> = r.iter().map(|&v| f64::from(u8::from(v >= cutoff))).collect();
    table(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y.to_vec()),
        ("r", RoleHint::Context, r.to_vec()),
    ])
}

fn fit(
    r: &[f64],
    y: &[f64],
    cutoff: f64,
    bandwidth: f64,
    kind: AnalyticSeKind,
) -> Result<antecedent_estimate::EffectEstimate, antecedent_estimate::EstimationError> {
    let data = data_for(r, y, cutoff);
    let est = SharpRegressionDiscontinuity::new(id(2), cutoff, bandwidth)
        .with_se_kind(kind)
        .with_bootstrap_replicates(0);
    let estimand = IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), ExprId::from_raw(0));
    let query = AverageEffectQuery::binary_ate(id(0), id(1))
        .with_target_population(TargetPopulation::local_at_cutoff(id(2), cutoff));
    let prep = est.prepare(&data, &estimand, &query)?;
    est.fit(
        &prep,
        &mut RdWorkspace::default(),
        &ExecutionContext::for_tests(3),
        AssumptionSet::new(),
    )
}

#[test]
fn jump_and_every_variance_match_the_exact_references() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (r, y) = (floats(&case["r"]), floats(&case["y"]));
        let (c, h) = (case["cutoff"].as_f64().unwrap(), case["bandwidth"].as_f64().unwrap());
        let saturated = case["leverage_one"].as_bool().unwrap();
        for (kind, key) in [
            (AnalyticSeKind::Hc0, "var_hc0"),
            (AnalyticSeKind::Hc1, "var_hc1"),
            (AnalyticSeKind::Hc2, "var_hc2"),
            (AnalyticSeKind::Hc3, "var_hc3"),
            (AnalyticSeKind::Homoskedastic, "var_homoskedastic"),
        ] {
            let result = fit(&r, &y, c, h, kind);
            if saturated {
                // Four rows, four coefficients: nothing is left to estimate a variance from.
                // The jump is the difference of two interpolating lines, but no variance is
                // reported: the residuals are rounding noise.
                if let Ok(effect) = result {
                    close(effect.ate, case["jump"].as_f64().unwrap(), 1e-10, name);
                    let se = effect.se_analytic();
                    assert!(se.is_nan(), "{name} {key}: se {se}");
                }
                continue;
            }
            let effect = result.unwrap_or_else(|e| panic!("{name} {key}: {e:?}"));
            close(effect.ate, case["jump"].as_f64().unwrap(), 1e-9, &format!("{name} jump"));
            close(
                effect.se_analytic(),
                case[key].as_f64().unwrap().sqrt(),
                1e-8,
                &format!("{name} {key}"),
            );
        }
    }
}

#[test]
fn rows_outside_the_window_do_not_move_the_fit_and_rows_on_the_edges_do() {
    let case = &cases()[0];
    let (r, y) = (floats(&case["r"]), floats(&case["y"]));
    let (c, h) = (case["cutoff"].as_f64().unwrap(), case["bandwidth"].as_f64().unwrap());
    let base = fit(&r, &y, c, h, AnalyticSeKind::Hc1).unwrap();
    // Drop the three rows beyond the window: nothing changes.
    let keep: Vec<usize> = (0..r.len()).filter(|&i| (r[i] - c).abs() <= h).collect();
    let (r2, y2): (Vec<f64>, Vec<f64>) = keep.iter().map(|&i| (r[i], y[i])).unzip();
    let trimmed = fit(&r2, &y2, c, h, AnalyticSeKind::Hc1).unwrap();
    assert_eq!(base.ate.to_bits(), trimmed.ate.to_bits());
    assert_eq!(base.se_analytic().to_bits(), trimmed.se_analytic().to_bits());
    // Shrink the window to exclude the rows exactly at -h and +h: the fit is on fewer rows.
    let narrower = fit(&r, &y, c, h - 1e-9, AnalyticSeKind::Hc1).unwrap();
    assert_ne!(base.ate.to_bits(), narrower.ate.to_bits());
}

#[test]
fn a_side_with_one_row_cannot_support_a_line_and_is_not_estimated() {
    let r = [-0.0625, 0.0, 0.0625, 0.125];
    let y = [1.0, 2.0, 3.0, 4.5];
    match fit(&r, &y, 0.0, 0.2, AnalyticSeKind::Hc1) {
        Err(_) => {}
        Ok(effect) => {
            assert!(!effect.ate.is_finite() || !effect.se_analytic().is_finite(), "{effect:?}")
        }
    }
}
