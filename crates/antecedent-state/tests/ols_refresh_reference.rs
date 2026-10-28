//! Refreshed least-squares results against exact rational arithmetic after every operation.
//!
//! `oracle_data/ols_refresh_reference.py` (frozen in the JSON beside it) replays appends,
//! windows, evictions, count weights and masks on three data sets and, after each operation,
//! recomputes the sample size, the coefficients and the residual variance from the live rows
//! in `Fraction` arithmetic. The data sets have a year regressor near 2020 and near 1e6 with
//! responses near 1e8 and 1e12, and four columns scaled over twelve decades.

use antecedent_state::{OlsRefresh, RefreshRow};
use serde_json::Value;

fn agree(label: &str, name: &str, step: usize, actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs(),
        "{name} step {step} {label}: {actual:.17e} vs {expected:.17e}"
    );
}

#[test]
fn every_operation_leaves_the_exact_least_squares_result() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/ols_refresh_reference.json")).unwrap();
    for scenario in reference["scenarios"].as_array().unwrap() {
        let name = scenario["name"].as_str().unwrap();
        let ncols = scenario["ncols"].as_u64().unwrap() as usize;
        let rows: Vec<RefreshRow> = scenario["rows"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(id, r)| {
                RefreshRow::new(
                    id as u64,
                    r["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect(),
                    r["y"].as_f64().unwrap(),
                )
            })
            .collect();
        let beta_tolerance = scenario["beta_tolerance"].as_f64().unwrap();
        let variance_tolerance = scenario["variance_tolerance"].as_f64().unwrap();
        let mut fit = OlsRefresh::new("source", ncols).unwrap();
        for (step, (op, want)) in scenario["ops"]
            .as_array()
            .unwrap()
            .iter()
            .zip(scenario["expected"].as_array().unwrap())
            .enumerate()
        {
            let op = op.as_array().unwrap();
            match op[0].as_str().unwrap() {
                "append" => {
                    for id in op[1].as_array().unwrap() {
                        fit.append_row("source", &rows[id.as_u64().unwrap() as usize]).unwrap();
                    }
                }
                "window" => {
                    let window: Vec<RefreshRow> = op[1]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|id| rows[id.as_u64().unwrap() as usize].clone())
                        .collect();
                    fit.append_window("source", &window).unwrap();
                }
                "evict" => {
                    fit.evict_row(op[1].as_u64().unwrap()).unwrap();
                }
                "count" => {
                    fit.set_count(op[1].as_u64().unwrap(), op[2].as_u64().unwrap() as u32).unwrap();
                }
                "mask" => {
                    fit.set_mask(op[1].as_u64().unwrap(), op[2].as_bool().unwrap()).unwrap();
                }
                other => panic!("unknown operation {other}"),
            }
            let snapshot = fit.snapshot();
            assert_eq!(snapshot.n, want["n"].as_u64().unwrap(), "{name} step {step}: n");
            match want["beta"].as_array() {
                Some(beta) => {
                    let got = snapshot
                        .beta
                        .as_deref()
                        .unwrap_or_else(|| panic!("{name} step {step}: no coefficients"));
                    for (g, w) in got.iter().zip(beta) {
                        agree("beta", name, step, *g, w.as_f64().unwrap(), beta_tolerance);
                    }
                }
                None => assert!(snapshot.beta.is_none(), "{name} step {step}: rank deficient"),
            }
            if let (Some(g), Some(w)) =
                (snapshot.residual_variance, want["residual_variance"].as_f64())
            {
                eprintln!("{name} step {step}: variance error {:.2e}", (g - w).abs() / w);
            }
            match want["residual_variance"].as_f64() {
                Some(w) => agree(
                    "residual variance",
                    name,
                    step,
                    snapshot.residual_variance.expect("residual variance"),
                    w,
                    variance_tolerance,
                ),
                None => assert!(snapshot.residual_variance.is_none(), "{name} step {step}"),
            }
            let (cached, fresh) = (snapshot.residual_variance, fit.fresh_snapshot().unwrap());
            if scenario["check_fresh"].as_bool().unwrap() {
                fit.check_against_fresh().unwrap_or_else(|e| panic!("{name} step {step}: {e}"));
            } else if let (Some(c), Some(f)) = (cached, fresh.residual_variance) {
                eprintln!("{name} step {step}: cached {c:.12e} fresh {f:.12e}");
            }
        }
    }
}
