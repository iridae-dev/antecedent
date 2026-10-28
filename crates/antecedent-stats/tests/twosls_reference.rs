//! Two-stage least squares, its first-stage test, the F critical value and the Anderson-Rubin
//! statistic and set against exact rational and 50-digit references.
//!
//! The frozen values come from `oracle_data/twosls_reference.py` (its output is
//! `oracle_data/twosls_reference.json`).  Designs include a single instrument, several
//! instruments, a covariate with a large offset, and weak instruments whose Anderson-Rubin set is
//! unbounded in one or both directions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    FaerBackend, IvBlocks, IvMoments, LeastSquaresWorkspace, anderson_rubin_confidence_set,
    anderson_rubin_statistic, f_critical, fit_2sls, fit_2sls_from_moments,
};
use serde_json::Value;

const REFERENCE: &str = include_str!("oracle_data/twosls_reference.json");

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn rel(got: f64, want: f64) -> f64 {
    ((got - want) / want.abs().max(f64::MIN_POSITIVE)).abs()
}

fn normwise(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0_f64, |m, (g, w)| m.max((g - w).abs())) / scale
}

#[test]
fn two_stage_least_squares_matches_exact_arithmetic() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (n, k, px) = (
            case["n"].as_u64().unwrap() as usize,
            case["k"].as_u64().unwrap() as usize,
            case["x_ncols"].as_u64().unwrap() as usize,
        );
        let (z, x, t, y) =
            (floats(&case["z"]), floats(&case["x"]), floats(&case["t"]), floats(&case["y"]));
        let blocks = IvBlocks {
            nrows: n,
            instruments_colmajor: &z,
            z_ncols: k,
            exogenous_colmajor: &x,
            x_ncols: px,
        };
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_2sls(&blocks, &t, &y, &FaerBackend, &mut ws).unwrap();
        let coef = normwise(&fit.second_stage.coefficients, &floats(&case["coef"]));
        let rss = rel(fit.structural_rss, case["structural_rss"].as_f64().unwrap());
        let diag = &fit.first_stage_diagnostics;
        let f = rel(diag.f_statistic, case["f_statistic"].as_f64().unwrap());
        let r2 = rel(diag.partial_r2, case["partial_r2"].as_f64().unwrap());
        let df_ok = diag.df1 == k && diag.df2 == case["df2"].as_u64().unwrap() as usize;
        eprintln!("{name:28} coef {coef:.2e} rss {rss:.2e} F {f:.2e} R2 {r2:.2e}");
        if coef > 1e-9 || rss > 1e-9 || f > 1e-9 || r2 > 1e-9 || !df_ok {
            failures.push(format!(
                "{name}: coef {coef:.2e} rss {rss:.2e} F {f:.2e} R2 {r2:.2e} df_ok {df_ok}"
            ));
        }

        // The moment-based entry point agrees to the squared condition number of the design.
        let p = px + k + 1;
        let gram = floats(&case["gram"]);
        let cross = floats(&case["cross"]);
        let moments = IvMoments {
            gram: &gram,
            cross_outcome: &cross,
            outcome_sum_squares: case["yy"].as_f64().unwrap(),
            x_ncols: px,
            z_ncols: k,
            nrows: n,
        };
        assert_eq!(gram.len(), p * p);
        match fit_2sls_from_moments(&moments) {
            Ok(m) => {
                let mc = normwise(&m.coefficients, &floats(&case["coef"]));
                let mf = rel(
                    m.first_stage_diagnostics.f_statistic,
                    case["f_statistic"].as_f64().unwrap(),
                );
                eprintln!(
                    "{name:28} moments: coef {mc:.2e} F {mf:.2e} resolved {}",
                    m.rss_resolved
                );
                let tol = if name.contains("offset") { 1e-3 } else { 1e-8 };
                if mc > tol || mf > tol || !m.rss_resolved {
                    failures.push(format!(
                        "{name}: moments coef {mc:.2e} F {mf:.2e} resolved {}",
                        m.rss_resolved
                    ));
                }
            }
            Err(e) => failures.push(format!("{name}: moments refused ({e})")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn anderson_rubin_statistic_critical_value_and_set_match_exact_arithmetic() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (n, k, px) = (
            case["n"].as_u64().unwrap() as usize,
            case["k"].as_u64().unwrap() as usize,
            case["x_ncols"].as_u64().unwrap() as usize,
        );
        let (z, x, t, y) =
            (floats(&case["z"]), floats(&case["x"]), floats(&case["t"]), floats(&case["y"]));
        let blocks = IvBlocks {
            nrows: n,
            instruments_colmajor: &z,
            z_ncols: k,
            exogenous_colmajor: &x,
            x_ncols: px,
        };
        let mut ws = LeastSquaresWorkspace::default();
        let level = case["level"].as_f64().unwrap();
        let df = case["df2"].as_u64().unwrap() as usize;

        let crit = rel(f_critical(level, k, df), case["f_critical"].as_f64().unwrap());
        let at_beta0 = anderson_rubin_statistic(
            &blocks,
            &y,
            &t,
            case["ar_beta0"].as_f64().unwrap(),
            &FaerBackend,
            &mut ws,
        )
        .unwrap();
        let at_zero =
            anderson_rubin_statistic(&blocks, &y, &t, 0.0, &FaerBackend, &mut ws).unwrap();
        let (e0, ez) = (
            rel(at_beta0, case["ar_at_beta0"].as_f64().unwrap()),
            rel(at_zero, case["ar_at_zero"].as_f64().unwrap()),
        );
        let (set, why) =
            anderson_rubin_confidence_set(&blocks, &y, &t, level, &FaerBackend, &mut ws).unwrap();
        let kind = case["ar_kind"].as_str().unwrap();
        eprintln!(
            "{name:28} F_crit {crit:.2e} AR(2) {e0:.2e} AR(0) {ez:.2e} set {set:?} {why:?} want {kind}"
        );
        if crit > 1e-12 || e0 > 1e-9 || ez > 1e-9 {
            failures.push(format!("{name}: F_crit {crit:.2e} AR(beta0) {e0:.2e} AR(0) {ez:.2e}"));
        }
        match (kind, set, why) {
            ("interval", Some((lo, hi, l)), None) => {
                let (el, eh) = (
                    rel(lo, case["ar_lo"].as_f64().unwrap()),
                    rel(hi, case["ar_hi"].as_f64().unwrap()),
                );
                if el > 1e-9 || eh > 1e-9 || l != level {
                    failures.push(format!("{name}: set endpoints {el:.2e}, {eh:.2e}"));
                }
            }
            ("union", None, Some("anderson_rubin_set_is_union")) => {}
            ("whole_line", Some((lo, hi, _)), None)
                if lo == f64::NEG_INFINITY && hi == f64::INFINITY => {}
            (want, got, why) => {
                failures.push(format!("{name}: wanted a {want} set, got {got:?} ({why:?})"))
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
