//! Penalized additive cubic-spline fits against exact rational references.
//!
//! `oracle_data/gam_reference.py` evaluates the B-spline bases from the Cox-de Boor definition,
//! places the quantile knots from the type-7 definition and solves the joint penalized weighted
//! least-squares problem exactly, so neither backfitting nor the Cholesky and spectral
//! shortcuts of the implementation are shared with the reference.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    FaerBackend, GamFit, GamOptions, GamWorkspace, PredictorMatrix, SmoothSpec, expand_bspline,
    fit_gam, fit_gam_weighted, predict_gam,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/gam_reference.json")).unwrap()
}

fn vec_f(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn fit_case(case: &Value, auto: bool) -> (GamFit, Vec<f64>, usize) {
    let cols: Vec<Vec<f64>> = case["x_cols"].as_array().unwrap().iter().map(vec_f).collect();
    let nrows = cols[0].len();
    let x: Vec<f64> = cols.iter().flatten().copied().collect();
    let y = vec_f(&case["y"]);
    let lambdas = case.get("lambdas").map(vec_f);
    let specs: Vec<SmoothSpec> = case["specs"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(j, s)| {
            let (col, k) =
                (s["raw_col"].as_u64().unwrap() as usize, s["n_basis"].as_u64().unwrap() as usize);
            if auto {
                SmoothSpec::auto(col, k)
            } else {
                SmoothSpec::new(col, k, lambdas.as_ref().unwrap()[j])
            }
        })
        .collect();
    let weights = (!case["weights"].is_null()).then(|| vec_f(&case["weights"]));
    let fit = fit_gam_weighted(
        &PredictorMatrix::new(&x, nrows, cols.len()),
        &y,
        &specs,
        &GamOptions { max_iter: 20_000, tol: 1e-13 },
        weights.as_deref(),
        &FaerBackend,
        &mut GamWorkspace::default(),
    )
    .unwrap();
    (fit, x, nrows)
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-300) + rel * 1e-12,
        "{what}: {got} vs {want}"
    );
}

#[test]
fn fixed_lambda_fits_equal_the_exact_joint_penalized_solution() {
    for case in fixture()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (fit, _, nrows) = fit_case(case, false);
        assert!(fit.converged, "{name}");
        let want = vec_f(&case["fitted"]);
        // An outcome offset lives in the intercept; the smooths are held to the scale of the
        // centred fit, not to the offset.
        let offset = case.get("offset").and_then(Value::as_f64).unwrap_or(0.0);
        let scale = want.iter().fold(0.0_f64, |a, v| a.max((v - offset).abs()));
        // The large-lambda case solves a system whose condition number grows with lambda.
        let rel = if name == "large_lambda" { 1e-6 } else { 1e-8 };
        for r in 0..nrows {
            assert!(
                (fit.fitted[r] - want[r]).abs() <= rel * scale,
                "{name} row {r}: {} vs {}",
                fit.fitted[r],
                want[r]
            );
        }
        if !case["xnew"].is_null() {
            let xn: Vec<Vec<f64>> = case["xnew"].as_array().unwrap().iter().map(vec_f).collect();
            let flat: Vec<f64> = xn.iter().flatten().copied().collect();
            let got = predict_gam(&fit, &flat, xn[0].len(), xn.len()).unwrap();
            for (r, (g, w)) in got.iter().zip(vec_f(&case["prediction"])).enumerate() {
                assert!((g - w).abs() <= rel * scale.max(1.0), "{name} prediction {r}: {g} vs {w}");
            }
        }
    }
}

#[test]
fn quantile_knots_are_the_type_seven_quantiles() {
    for case in fixture()["cases"].as_array().unwrap() {
        let cols: Vec<Vec<f64>> = case["x_cols"].as_array().unwrap().iter().map(vec_f).collect();
        for (j, s) in case["specs"].as_array().unwrap().iter().enumerate() {
            let k = s["n_basis"].as_u64().unwrap() as usize;
            let (_, knots) =
                expand_bspline(&cols[s["raw_col"].as_u64().unwrap() as usize], k, None).unwrap();
            let want = vec_f(&case["knots"][j]);
            assert_eq!(knots.len(), want.len());
            let span = want[want.len() - 1] - want[0];
            for (g, w) in knots.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-12 * span, "{}: knot {g} vs {w}", case["name"]);
            }
        }
    }
}

#[test]
fn edf_is_the_hat_trace_for_one_smooth_and_the_sum_of_single_traces_for_several() {
    for case in fixture()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if name == "large_lambda" {
            continue;
        }
        let (fit, _, _) = fit_case(case, false);
        match case.get("sum_of_single_smooth_traces").filter(|v| !v.is_null()) {
            None => close(fit.edf_approx, case["hat_trace"].as_f64().unwrap(), 1e-8, name),
            Some(sum) => {
                close(fit.edf_approx, sum.as_f64().unwrap(), 1e-8, name);
                // With correlated predictors the joint trace is smaller: the field is an upper
                // approximation, never the joint trace itself.
                assert!(fit.edf_approx > case["hat_trace"].as_f64().unwrap() + 1e-3, "{name}");
            }
        }
    }
}

#[test]
fn very_large_lambda_collapses_to_a_straight_line_only_on_equally_spaced_knots() {
    let case = fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "large_lambda")
        .unwrap()
        .clone();
    let x = vec_f(&case["x_cols"][0]);
    let y = vec_f(&case["y"]);
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let slope = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>()
        / x.iter().map(|a| (a - mx) * (a - mx)).sum::<f64>();
    let line = |v: f64| my + slope * (v - mx);
    let fit_with = |knots: Option<Vec<f64>>| {
        let mut spec = SmoothSpec::new(0, 8, 1e9);
        if let Some(k) = knots {
            spec = spec.with_knots(k);
        }
        fit_gam(
            &PredictorMatrix::new(&x, x.len(), 1),
            &y,
            &[spec],
            &GamOptions { max_iter: 20_000, tol: 1e-13 },
            &FaerBackend,
            &mut GamWorkspace::default(),
        )
        .unwrap()
    };
    let (lo, hi) = (
        x.iter().cloned().fold(f64::INFINITY, f64::min),
        x.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    // Unclamped equally spaced knots: the Greville abscissae are then equally spaced as well,
    // so coefficients linear in their index give a straight line in x.
    let h = (hi - lo) / 5.0;
    let uniform: Vec<f64> = (0..12).map(|i| lo + (f64::from(i) - 3.0) * h).collect();
    let even = fit_with(Some(uniform));
    for (r, v) in x.iter().enumerate() {
        assert!((even.fitted[r] - line(*v)).abs() < 1e-4, "uniform knots, row {r}");
    }
    assert!((even.edf_approx - 2.0).abs() < 1e-4, "{}", even.edf_approx);
    // The penalty acts on coefficient differences, so its null space is a line in the knot
    // index. Sample-quantile knots are unevenly spaced and the limit is a curve in x.
    let quantile = fit_with(None);
    let worst = x
        .iter()
        .enumerate()
        .map(|(r, v)| (quantile.fitted[r] - line(*v)).abs())
        .fold(0.0, f64::max);
    assert!(worst > 1e-3, "quantile-knot limit is not a straight line: {worst}");
    assert!((quantile.edf_approx - 2.0).abs() < 1e-3, "{}", quantile.edf_approx);
}

#[test]
fn gcv_picks_the_grid_minimiser_of_the_exact_score() {
    for g in fixture()["gcv"].as_array().unwrap() {
        let name = g["name"].as_str().unwrap();
        let case = serde_json::json!({
            "x_cols": [g["x"]], "y": g["y"], "weights": g["weights"],
            "specs": [{"raw_col": 0, "n_basis": g["n_basis"]}],
        });
        let (fit, _, _) = fit_case(&case, true);
        let lambda = fit.smooths[0].lambda;
        let scores = vec_f(&g["scores_with_intercept"]);
        let grid = vec_f(&fixture()["grid"]);
        let i = grid
            .iter()
            .position(|v| (v - lambda).abs() <= 1e-12 * v)
            .expect("lambda is a grid point");
        let best = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        // Two conventions for the degrees of freedom (with and without the intercept's one) pick
        // the same grid point on every case here; require a score within a relative 1e-9 of the
        // exact minimum so a near tie cannot fail on rounding.
        assert!(
            scores[i] <= best * (1.0 + 1e-9),
            "{name}: picked {lambda} (score {}) vs minimum {best}",
            scores[i]
        );
        let at_edge = i == 0 || i == grid.len() - 1;
        assert_eq!(fit.boundary_lambda_smooths.contains(&0), at_edge, "{name}");
    }
}

#[test]
fn unweighted_and_unit_weight_fits_coincide() {
    let case = &fixture()["cases"][0];
    let (a, x, nrows) = fit_case(case, false);
    let y = vec_f(&case["y"]);
    let b = fit_gam(
        &PredictorMatrix::new(&x, nrows, 1),
        &y,
        &[SmoothSpec::new(0, 8, 0.7)],
        &GamOptions { max_iter: 20_000, tol: 1e-13 },
        &FaerBackend,
        &mut GamWorkspace::default(),
    )
    .unwrap();
    for (p, q) in a.fitted.iter().zip(&b.fitted) {
        assert!((p - q).abs() < 1e-12);
    }
}

#[test]
fn an_outcome_offset_does_not_loosen_the_smooths_under_the_default_stopping_rule() {
    let fixture = fixture();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "offset_million_outcome")
        .unwrap();
    let offset = case["offset"].as_f64().unwrap();
    let cols: Vec<Vec<f64>> = case["x_cols"].as_array().unwrap().iter().map(vec_f).collect();
    let nrows = cols[0].len();
    let x: Vec<f64> = cols.iter().flatten().copied().collect();
    let y = vec_f(&case["y"]);
    let centred: Vec<f64> = y.iter().map(|v| v - offset).collect();
    let lambdas = vec_f(&case["lambdas"]);
    let specs = [SmoothSpec::new(0, 6, lambdas[0]), SmoothSpec::new(1, 7, lambdas[1])];
    let run = |outcome: &[f64]| {
        fit_gam(
            &PredictorMatrix::new(&x, nrows, cols.len()),
            outcome,
            &specs,
            &GamOptions::default(),
            &FaerBackend,
            &mut GamWorkspace::default(),
        )
        .unwrap()
    };
    let (shifted, plain) = (run(&y), run(&centred));
    // The same stopping rule on the same centred problem: the iterates agree to rounding of the
    // offset itself (the ulp of 1e6 is 1.2e-10), not to a tolerance scaled by the offset.
    for r in 0..nrows {
        let got = shifted.fitted[r] - offset;
        assert!((got - plain.fitted[r]).abs() <= 1e-8, "row {r}: {got} vs {}", plain.fitted[r]);
    }
    assert_eq!(shifted.iterations, plain.iterations);
}
