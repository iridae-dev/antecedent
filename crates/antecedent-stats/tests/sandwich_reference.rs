//! Sandwich covariances, least-squares solves and the sample mean/SE helpers against exact
//! rational references.
//!
//! The frozen values come from `oracle_data/sandwich_reference.py` (its output is
//! `oracle_data/sandwich_reference.json`).  Each design, residual, weight and cluster label is an
//! exact binary64 value; every expected value is computed in exact rational arithmetic from the
//! defining formulas and rounded once.  The designs include the worst numeric regimes for these
//! routines: a large common offset (condition number up to 1.6e8 after column equilibration),
//! near-collinear columns, a very high-leverage row, six-decade column scales and six-decade
//! weights, rank-deficient designs, subnormal and near-overflow sample variances.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, MEstimateOptions, SandwichKind,
    StatsError, coefficient_covariance, fit_huber_m, hc2_coefficient_bell_mccaffrey, iid_se,
    mean_and_se, score_coefficient_covariance,
};
use serde_json::Value;

const REFERENCE: &str = include_str!("oracle_data/sandwich_reference.json");

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn ints<T: TryFrom<i64>>(v: &Value) -> Vec<T> {
    v.as_array().unwrap().iter().map(|x| T::try_from(x.as_i64().unwrap()).ok().unwrap()).collect()
}

/// Largest entry-wise error relative to the largest expected entry of the matrix.
fn normwise(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0.0_f64, |m, (g, w)| m.max((g - w).abs())) / scale
}

/// Largest error of a diagonal entry relative to that entry (the variances themselves).
fn diagonal(got: &[f64], want: &[f64], p: usize) -> f64 {
    (0..p).fold(0.0_f64, |m, k| {
        let (g, w) = (got[k * p + k], want[k * p + k]);
        m.max(((g - w) / w).abs())
    })
}

#[test]
fn sandwich_covariances_match_exact_arithmetic() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    let mut worst: Vec<(String, f64, f64)> = Vec::new();
    for case in doc["covariance"].as_array().unwrap() {
        let kind_name = case["kind"].as_str().unwrap();
        let design = case["design"].as_str().unwrap();
        let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
        let (x, e, want) = (floats(&case["x"]), floats(&case["e"]), floats(&case["cov"]));
        let cond = case["cond"].as_f64().unwrap();
        let groups: Vec<u32> = case.get("groups").map(ints).unwrap_or_default();
        let groups2: Vec<u32> = case.get("groups2").map(ints).unwrap_or_default();
        let time: Vec<i64> = case.get("time").map(ints).unwrap_or_default();
        let lag = case.get("lag").map_or(0, |l| l.as_u64().unwrap() as usize);
        let weights = case.get("weights").map(floats);
        let dims: [&[u32]; 2] = [&groups, &groups2];
        let got = match kind_name {
            "hc0" => coefficient_covariance(&x, n, p, &e, SandwichKind::Hc0),
            "hc1" => coefficient_covariance(&x, n, p, &e, SandwichKind::Hc1),
            "hc2" => coefficient_covariance(&x, n, p, &e, SandwichKind::Hc2),
            "hc3" => coefficient_covariance(&x, n, p, &e, SandwichKind::Hc3),
            "homoskedastic" => coefficient_covariance(&x, n, p, &e, SandwichKind::Homoskedastic),
            "cluster" | "cluster_alt" => {
                coefficient_covariance(&x, n, p, &e, SandwichKind::Cluster { groups: &groups })
            }
            "multiway" => {
                coefficient_covariance(&x, n, p, &e, SandwichKind::Multiway { dimensions: &dims })
            }
            "newey_west" => coefficient_covariance(&x, n, p, &e, SandwichKind::NeweyWest { lag }),
            "panel_hac" | "panel_hac_lag0" => coefficient_covariance(
                &x,
                n,
                p,
                &e,
                SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag },
            ),
            "score_hc0" | "score_hc2" | "score_hc3" => {
                let kind = match kind_name {
                    "score_hc0" => SandwichKind::Hc0,
                    "score_hc2" => SandwichKind::Hc2,
                    _ => SandwichKind::Hc3,
                };
                score_coefficient_covariance(&x, n, p, &e, weights.as_deref().unwrap(), kind)
            }
            other => panic!("unknown kind {other}"),
        };
        let label = format!("{kind_name}/{design}/lag{lag}");
        if let Some(ratio) = case.get("min_eig_ratio").map(|r| r.as_f64().unwrap()) {
            // The inclusion-exclusion matrix is not always positive semidefinite; where its
            // exact smallest generalised eigenvalue is materially negative the estimator must
            // refuse (publishing it would give a negative variance for some contrast), however
            // ill-conditioned the columns are, rather than return the indefinite matrix.
            if ratio < -1e-12 {
                if !matches!(got, Err(StatsError::NonPositiveVariance { .. })) {
                    failures
                        .push(format!("{label}: indefinite (ratio {ratio:.2e}) but got {got:?}"));
                }
                continue;
            }
        }
        let got = match got {
            Ok(g) => g,
            Err(err) => {
                failures.push(format!("{label}: refused ({err}) where an exact value exists"));
                continue;
            }
        };
        let (norm, diag) = (normwise(&got, &want), diagonal(&got, &want, p));
        worst.push((label.clone(), cond, norm.max(diag)));
        // The sandwich is formed from a QR factorisation of the equilibrated design, so its
        // error scales with the condition number (about nine rounding units per unit of it);
        // HC2 and HC3 also divide by `1 - h`, whose own rounding error is `eps / (1 - h)`.
        let omh = case["one_minus_max_leverage"].as_f64().unwrap();
        let leverage_term = if kind_name.contains("hc2") || kind_name.contains("hc3") {
            16.0 * f64::EPSILON / omh
        } else {
            0.0
        };
        let tol = 1e-13 + 2e-15 * cond + leverage_term;
        if norm.max(diag) > tol {
            failures
                .push(format!("{label}: cond {cond:.3e} error {:.3e} > {tol:.1e}", norm.max(diag)));
        }
    }
    for (label, cond, err) in &worst {
        eprintln!("{label:40} cond {cond:10.3e} err {err:.3e}");
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn bell_mccaffrey_variance_and_degrees_of_freedom_match_the_dense_definition() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["bell_mccaffrey"].as_array().unwrap() {
        let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
        let coef = case["coef"].as_u64().unwrap() as usize;
        let (x, e) = (floats(&case["x"]), floats(&case["e"]));
        let (var, df) = (case["variance"].as_f64().unwrap(), case["df"].as_f64().unwrap());
        let cond = case["cond"].as_f64().unwrap();
        let label = format!("{}/coef{coef}", case["design"].as_str().unwrap());
        match hc2_coefficient_bell_mccaffrey(&x, n, p, &e, coef) {
            Ok(fit) => {
                let (ev, ed) = (
                    ((fit.variance - var) / var).abs(),
                    ((fit.degrees_of_freedom - df) / df).abs(),
                );
                eprintln!("{label:30} cond {cond:9.2e} var {ev:.2e} df {ed:.2e}");
                // The degrees of freedom subtract two terms of order `d_i^2` that nearly cancel
                // when a leverage approaches one, hence the `eps / (1 - h)^2` allowance.
                let omh = case["one_minus_max_leverage"].as_f64().unwrap();
                let tol_var = 1e-12 + 2e-15 * cond + 16.0 * f64::EPSILON / omh;
                let tol_df = 1e-12 + 2e-15 * cond + 64.0 * f64::EPSILON / (omh * omh);
                if ev > tol_var || ed > tol_df {
                    failures.push(format!(
                        "{label}: variance {ev:.2e} (> {tol_var:.1e}), df {ed:.2e} (> {tol_df:.1e})"
                    ));
                }
            }
            Err(err) => failures.push(format!("{label}: refused ({err})")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn least_squares_solves_match_exact_normal_equations() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["least_squares"].as_array().unwrap() {
        let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
        let (x, y, w) = (floats(&case["x"]), floats(&case["y"]), floats(&case["w"]));
        let (coef, coef_w) = (floats(&case["coef"]), floats(&case["coef_w"]));
        let cond = case["cond"].as_f64().unwrap();
        let design = case["design"].as_str().unwrap();
        let mut ws = LeastSquaresWorkspace::default();
        let mut check =
            |what: &str, got: Result<Vec<f64>, antecedent_stats::StatsError>, want: &[f64]| {
                match got {
                    Ok(g) => {
                        let err = normwise(&g, want);
                        eprintln!("{what:12} {design:22} cond {cond:9.2e} err {err:.2e}");
                        // Column-equilibrated QR: the error is a few hundred rounding units per unit
                        // of condition number (the residual part of the error carries cond^2).
                        let tol = 1e-13 * (1.0 + cond);
                        if err > tol {
                            failures.push(format!(
                                "{what}/{design}: cond {cond:.2e} err {err:.2e} > {tol:.1e}"
                            ));
                        }
                    }
                    Err(e) => failures.push(format!("{what}/{design}: refused ({e})")),
                }
            };
        check(
            "ols",
            FaerBackend.least_squares(&x, n, p, &y, &mut ws).map(|f| f.coefficients),
            &coef,
        );
        check(
            "ols_min_norm",
            FaerBackend.least_squares_or_min_norm(&x, n, p, &y, &mut ws).map(|f| f.coefficients),
            &coef,
        );
        let root: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
        let yw: Vec<f64> = y.iter().zip(&root).map(|(a, b)| a * b).collect();
        // The solver is given sqrt(w) as an f64, so the exact weights it realises are
        // fl(sqrt(w))^2; compare against those exact weights through a reference-side check
        // with the same relative perturbation bound: sqrt rounding is 1.1e-16 relative.
        check(
            "wls",
            FaerBackend
                .row_scaled_least_squares(&x, n, p, &root, &yw, &mut ws)
                .map(|f| f.coefficients),
            &coef_w,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn rank_deficient_designs_are_refused_or_solved_to_the_exact_minimum_norm() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["rank_deficient"].as_array().unwrap() {
        let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
        let rank = case["rank"].as_u64().unwrap() as usize;
        let (x, y, coef) = (floats(&case["x"]), floats(&case["y"]), floats(&case["coef"]));
        let k = case["case"].as_u64().unwrap();
        let mut ws = LeastSquaresWorkspace::default();
        if FaerBackend.least_squares(&x, n, p, &y, &mut ws).is_ok() {
            failures.push(format!("case {k}: the full-rank solver accepted a rank-{rank} design"));
        }
        match FaerBackend.least_squares_or_min_norm(&x, n, p, &y, &mut ws) {
            Ok(fit) => {
                let err = normwise(&fit.coefficients, &coef);
                eprintln!("rank-deficient case {k}: rank {} err {err:.2e}", fit.rank);
                if fit.rank != rank || err > 1e-12 {
                    failures
                        .push(format!("case {k}: rank {} (want {rank}), err {err:.2e}", fit.rank));
                }
            }
            Err(e) => failures.push(format!("case {k}: min-norm refused ({e})")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn sample_mean_and_standard_error_match_exact_arithmetic() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["describe"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let values = floats(&case["values"]);
        let (mean, se) = (case["mean"].as_f64().unwrap(), case["se"].as_f64().unwrap());
        let rel = |g: f64, w: f64| if w == 0.0 { g.abs() } else { ((g - w) / w).abs() };
        match mean_and_se(&values) {
            Some((m, s)) => {
                let (em, es) = (rel(m, mean), rel(s, se));
                eprintln!("{name:12} mean {em:.2e} se {es:.2e}");
                // The sum behind the mean is rounded n times: allow 4 eps sqrt(n) of the mean
                // absolute value, then the standard error to a few rounding units.
                let n = values.len() as f64;
                let mean_abs = values.iter().map(|v| v.abs()).sum::<f64>() / n;
                let mean_ok = (m - mean).abs() <= 4.0 * f64::EPSILON * n.sqrt() * mean_abs;
                if !mean_ok || es > 1e-14 || (rel(iid_se(&values), se) > 1e-14) {
                    failures.push(format!("{name}: mean {em:.2e}, se {es:.2e}"));
                }
            }
            None => failures.push(format!("{name}: refused a finite sample with an exact value")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn huber_regression_reaches_the_exact_fixed_point_of_its_estimating_equation() {
    let doc: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut failures = Vec::new();
    for case in doc["huber"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (n, p) = (case["n"].as_u64().unwrap() as usize, case["p"].as_u64().unwrap() as usize);
        let (x, y, want) = (floats(&case["x"]), floats(&case["y"]), floats(&case["coef"]));
        let want_scale = case["scale"].as_f64().unwrap();
        let mut ws = LeastSquaresWorkspace::default();
        let options = MEstimateOptions { max_iter: 500, tol: 1e-9, ..MEstimateOptions::default() };
        let fit = fit_huber_m(&x, n, p, &y, &options, &FaerBackend, &mut ws).unwrap();
        let err = normwise(&fit.coefficients, &want);
        let scale_err = ((fit.scale - want_scale) / want_scale).abs();
        eprintln!(
            "huber {name}: coef {err:.2e} scale {scale_err:.2e} fallback {} iterations {}",
            fit.scale_fallback, fit.iterations
        );
        // The reference iteration is run to a 1e-40 step, so it sits on the fixed point; the
        // product stops at a relative step of 1e-9 of a linearly convergent iteration.  (An
        // absolute step criterion could not be met for the offset design: its intercept is
        // about -2e6 and the QR solve is only good to about 5e-5 in it.)
        if !fit.converged
            || fit.scale_fallback != case["scale_fallback"].as_bool().unwrap()
            || err > 1e-7
            || scale_err > 1e-7
        {
            failures.push(format!(
                "{name}: converged {} fallback {} coef {err:.2e} scale {scale_err:.2e}",
                fit.converged, fit.scale_fallback
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_dependent_or_underdetermined_design_has_no_sandwich() {
    let e = [0.5, -1.0, 0.25, 1.5, -0.75, 0.0];
    let ones = [1.0; 6];
    let u = [0.1, 0.9, 0.3, 0.7, 0.2, 0.8];
    // An exactly duplicated column, an all-zero column and an exact multiple.
    let dup: Vec<f64> = ones.iter().chain(&u).chain(&u).copied().collect();
    let zero: Vec<f64> = ones.iter().chain(&[0.0; 6]).copied().collect();
    let multiple: Vec<f64> = ones
        .iter()
        .chain(&u)
        .chain(u.iter().map(|v| v * 2.0).collect::<Vec<_>>().iter())
        .copied()
        .collect();
    for (x, p) in [(&dup, 3), (&zero, 2), (&multiple, 3)] {
        for kind in [SandwichKind::Hc0, SandwichKind::Hc3, SandwichKind::Homoskedastic] {
            let err = coefficient_covariance(x, 6, p, &e, kind).unwrap_err();
            assert!(
                matches!(&err, StatsError::Backend(m) if m == "singular sandwich bread"),
                "p {p} {kind:?}: {err:?}"
            );
        }
        let err = hc2_coefficient_bell_mccaffrey(x, 6, p, &e, 0).unwrap_err();
        assert!(
            matches!(&err, StatsError::Backend(m) if m == "singular sandwich bread"),
            "p {p}: {err:?}"
        );
    }
    // Fewer rows than columns.
    let wide: Vec<f64> = (0..8).map(f64::from).collect();
    let err = coefficient_covariance(&wide, 2, 4, &e[..2], SandwichKind::Hc0).unwrap_err();
    assert!(matches!(&err, StatsError::Backend(m) if m == "singular sandwich bread"), "{err:?}");
    // A leverage-one row (its own dummy) is refused by HC2/HC3 and fine for HC0/HC1.
    let x = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let r = [0.0, 0.5, -0.5, 0.0];
    for kind in [SandwichKind::Hc2, SandwichKind::Hc3] {
        let err = coefficient_covariance(&x, 4, 2, &r, kind).unwrap_err();
        assert!(
            matches!(&err, StatsError::Backend(m)
                if m.starts_with("HC2/HC3 are undefined for 1 row(s) with leverage 1 (rows 0)")),
            "{kind:?}: {err:?}"
        );
    }
    assert!(coefficient_covariance(&x, 4, 2, &r, SandwichKind::Hc0).is_ok());
    assert!(coefficient_covariance(&x, 4, 2, &r, SandwichKind::Hc1).is_ok());
}
