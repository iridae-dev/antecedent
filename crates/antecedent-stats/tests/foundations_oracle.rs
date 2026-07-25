//! Frozen external-oracle parity for shared statistical foundations.
//!
//! Oracle packages are used only during external fixture generation. These
//! tests consume committed outputs and do not install or execute them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::fs;
use std::path::{Path, PathBuf};

use antecedent_core::CausalRng;
use antecedent_kernels::fill_standard_normal;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GamOptions, GamWorkspace, GlmDesignRef, GlmFamily, GlmOptions,
    LeastSquaresWorkspace, NbAlphaPolicy, SandwichKind, SmoothSpec, coefficient_covariance,
    expand_bspline, fit_gam, fit_glm, fit_wls, mean_var, predict_gam, sample_std,
};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/stats")
        .join(name)
        .join("expected.json");
    let raw = fs::read_to_string(path).expect("read oracle fixture");
    serde_json::from_str(&raw).expect("parse oracle fixture")
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn rowmajor(value: &Value) -> Vec<Vec<f64>> {
    value.as_array().unwrap().iter().map(floats).collect()
}

fn colmajor(rows: &[Vec<f64>]) -> Vec<f64> {
    let nrows = rows.len();
    let ncols = rows[0].len();
    let mut out = vec![0.0; nrows * ncols];
    for r in 0..nrows {
        for c in 0..ncols {
            out[c * nrows + r] = rows[r][c];
        }
    }
    out
}

fn close(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    (actual - expected).abs() <= atol + rtol * expected.abs()
}

fn assert_slice_close(actual: &[f64], expected: &[f64], atol: f64, rtol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(close(a, e, atol, rtol), "{label}[{i}] {a} vs {e}");
    }
}

fn flatten_matrix(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().flat_map(floats).collect()
}

#[test]
fn descriptive_reductions_match_frozen_numpy_outputs() {
    let expected = fixture("descriptive");
    assert_eq!(expected["reference"]["available"].as_bool(), Some(true));
    let atol = expected["atol"].as_f64().unwrap();
    let rtol = expected["rtol"].as_f64().unwrap();
    for case in expected["cases"].as_array().unwrap() {
        let values = floats(&case["values"]);
        let (mean, variance) = mean_var(&values);
        assert!(close(mean, case["mean"].as_f64().unwrap(), atol, rtol), "{} mean", case["name"]);
        assert!(
            close(variance, case["population_variance"].as_f64().unwrap(), atol, rtol),
            "{} variance",
            case["name"]
        );
        let std = sample_std(&values);
        if let Some(reference) = case["sample_std"].as_f64() {
            assert!(close(std, reference, atol, rtol), "{} sample std", case["name"]);
        } else {
            assert!(std.is_nan(), "{} singleton sample std", case["name"]);
        }
    }
}

#[test]
fn least_squares_matches_frozen_statsmodels_outputs() {
    let expected = fixture("ols_wls");
    assert_eq!(expected["reference"]["available"].as_bool(), Some(true));
    let atol = expected["atol"].as_f64().unwrap();
    let rtol = expected["rtol"].as_f64().unwrap();
    let mut workspace = LeastSquaresWorkspace::default();

    for case in expected["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = rowmajor(&case["x_rowmajor"]);
        let nrows = rows.len();
        let ncols = rows[0].len();
        let x = colmajor(&rows);
        let y = floats(&case["y"]);
        if name == "rank_deficient_oracle_pseudoinverse" {
            let error =
                FaerBackend.least_squares(&x, nrows, ncols, &y, &mut workspace).unwrap_err();
            assert!(matches!(error, antecedent_stats::StatsError::RankDeficient { .. }));
            continue;
        }

        let fit = if let Some(weights) = case["weights"].as_array() {
            let weights: Vec<f64> = weights.iter().map(|v| v.as_f64().unwrap()).collect();
            fit_wls(&x, nrows, ncols, &y, &weights, &FaerBackend, &mut workspace).unwrap()
        } else {
            FaerBackend.least_squares(&x, nrows, ncols, &y, &mut workspace).unwrap()
        };

        let reference_coefficients = floats(&case["coefficients"]);
        if name == "ill_conditioned" {
            let reference_fitted = floats(&case["fitted"]);
            let fitted: Vec<f64> = (0..nrows)
                .map(|r| (0..ncols).map(|c| x[c * nrows + r] * fit.coefficients[c]).sum::<f64>())
                .collect();
            assert_slice_close(&fitted, &reference_fitted, 2e-8, 2e-8, name);
        } else {
            assert_slice_close(&fit.coefficients, &reference_coefficients, atol, rtol, name);
        }

        if case["weights"].is_array() {
            assert!(close(fit.rss, case["weighted_rss"].as_f64().unwrap(), atol, rtol));
        } else {
            assert!(close(fit.rss, case["rss"].as_f64().unwrap(), 2e-9, 2e-9));
        }
    }
}

#[test]
fn sandwich_covariances_match_frozen_statsmodels_outputs() {
    let expected = fixture("sandwich_covariance");
    assert_eq!(expected["reference"]["available"].as_bool(), Some(true));
    let rows = rowmajor(&expected["x_rowmajor"]);
    let nrows = rows.len();
    let ncols = rows[0].len();
    let x = colmajor(&rows);
    let residuals = floats(&expected["residuals"]);
    let groups: Vec<u32> =
        expected["groups"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let atol = expected["atol"].as_f64().unwrap();
    let rtol = expected["rtol"].as_f64().unwrap();

    let cases = [
        ("homoskedastic", SandwichKind::Homoskedastic),
        ("hc0", SandwichKind::Hc0),
        ("hc1", SandwichKind::Hc1),
        ("hc2", SandwichKind::Hc2),
        ("hc3", SandwichKind::Hc3),
        ("cluster", SandwichKind::Cluster { groups: &groups }),
        ("newey_west_lag_0", SandwichKind::NeweyWest { lag: 0 }),
        ("newey_west_lag_1", SandwichKind::NeweyWest { lag: 1 }),
        ("newey_west_lag_3", SandwichKind::NeweyWest { lag: 3 }),
    ];

    for (name, kind) in cases {
        let actual = coefficient_covariance(&x, nrows, ncols, &residuals, kind).unwrap();
        let oracle = flatten_matrix(&expected["covariances"][name]);
        assert_slice_close(&actual, &oracle, atol, rtol, name);
    }
}

fn family_and_options(name: &str) -> (GlmFamily, GlmOptions) {
    let mut options = GlmOptions { max_iter: 200, tol: 1e-12, ..GlmOptions::default() };
    let family = match name {
        "gaussian_identity" => GlmFamily::GaussianIdentity,
        "binomial_logit" => GlmFamily::BinomialLogit,
        "binomial_probit" => GlmFamily::BinomialProbit,
        "poisson_log" => GlmFamily::PoissonLog,
        "negative_binomial_alpha_0_7" => {
            options.nb_alpha = NbAlphaPolicy::Fixed(0.7);
            GlmFamily::NegativeBinomial
        }
        _ => panic!("unknown GLM oracle case {name}"),
    };
    (family, options)
}

#[test]
fn glm_families_match_frozen_statsmodels_outputs() {
    let expected = fixture("glm_irls");
    assert_eq!(expected["reference"]["available"].as_bool(), Some(true));
    let coef_abs_tol = expected["coefficient_atol"].as_f64().unwrap();
    let coef_rel_tol = expected["coefficient_rtol"].as_f64().unwrap();
    let fitted_atol = expected["fitted_atol"].as_f64().unwrap();
    let deviance_atol = expected["deviance_atol"].as_f64().unwrap();
    let mut workspace = LeastSquaresWorkspace::default();

    for case in expected["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = rowmajor(&case["x_rowmajor"]);
        let nrows = rows.len();
        let ncols = rows[0].len();
        let x = colmajor(&rows);
        let y = floats(&case["y"]);
        let (family, options) = family_and_options(name);
        let fit = fit_glm(
            family,
            GlmDesignRef { x_colmajor: &x, nrows, ncols, y: &y },
            &FaerBackend,
            &mut workspace,
            &options,
        )
        .unwrap();
        assert!(fit.converged, "{name}");
        assert_slice_close(
            &fit.coefficients,
            &floats(&case["coefficients"]),
            coef_abs_tol,
            coef_rel_tol,
            name,
        );
        let fitted: Vec<f64> = rows
            .iter()
            .map(|row| {
                let eta = row.iter().zip(&fit.coefficients).map(|(x, b)| x * b).sum::<f64>();
                family.mean_from_eta(eta)
            })
            .collect();
        assert_slice_close(&fitted, &floats(&case["fitted_mean"]), fitted_atol, fitted_atol, name);
        assert!(
            close(fit.deviance, case["deviance"].as_f64().unwrap(), deviance_atol, deviance_atol),
            "{name} deviance {} vs {}",
            fit.deviance,
            case["deviance"]
        );
    }
}

#[test]
fn fixture_provenance_records_pins_and_no_retained_harness() {
    for name in ["descriptive", "ols_wls", "sandwich_covariance", "glm_irls", "rng_gaussian", "gam"]
    {
        let expected = fixture(name);
        let reference = &expected["reference"];
        assert!(reference["command"].as_str().unwrap().contains("uv run --python 3.12"));
        assert!(reference["installed_metadata_sha256"].is_object());
        assert_eq!(
            reference["generation_location"].as_str(),
            Some("temporary external harness; not retained")
        );
    }
    assert!(!Path::new("scripts/conformance/generate_stats_oracles.py").exists());
}

fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let mut cross = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for (&a, &b) in x.iter().zip(y) {
        cross += (a - mx) * (b - my);
        sx += (a - mx) * (a - mx);
        sy += (b - my) * (b - my);
    }
    cross / (sx * sy).sqrt()
}

#[test]
fn rng_stream_and_gaussian_sampler_match_frozen_scipy_battery() {
    let expected = fixture("rng_gaussian");
    assert_eq!(expected["reference"]["available"].as_bool(), Some(true));
    let n = expected["generation"]["n"].as_u64().unwrap() as usize;
    let uniform_seed = expected["generation"]["uniform_seed"].as_u64().unwrap();
    let normal_seed = expected["generation"]["normal_seed"].as_u64().unwrap();

    let mut uniform_rng = CausalRng::from_seed(uniform_seed);
    let uniforms: Vec<f64> = (0..n).map(|_| uniform_rng.next_f64()).collect();
    let mut normal_rng = CausalRng::from_seed(normal_seed);
    let mut normals = vec![0.0; n];
    fill_standard_normal(&mut normal_rng, &mut normals);

    let (uniform_mean, uniform_variance) = mean_var(&uniforms);
    assert!(close(uniform_mean, expected["uniform"]["mean"].as_f64().unwrap(), 1e-12, 1e-12));
    assert!(close(
        uniform_variance,
        expected["uniform"]["population_variance"].as_f64().unwrap(),
        1e-12,
        1e-12
    ));
    let lag1 = correlation(&uniforms[..n - 1], &uniforms[1..]);
    assert!(close(lag1, expected["uniform"]["lag1_correlation"].as_f64().unwrap(), 1e-12, 1e-12));

    let mut bins = [0usize; 20];
    for &u in &uniforms {
        let bin = ((u * 20.0).floor() as i32).clamp(0, 19);
        bins[usize::try_from(bin).expect("bin in 0..20")] += 1;
    }
    let oracle_bins: Vec<usize> = expected["uniform"]["bin20_counts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(bins.as_slice(), oracle_bins);

    let (normal_mean, normal_variance) = mean_var(&normals);
    assert!(close(normal_mean, expected["normal"]["mean"].as_f64().unwrap(), 1e-12, 1e-12));
    assert!(close(
        normal_variance,
        expected["normal"]["population_variance"].as_f64().unwrap(),
        1e-12,
        1e-12
    ));
    for (threshold, key) in [(1.0, "gt_1"), (2.0, "gt_2"), (3.0, "gt_3")] {
        let actual = normals.iter().filter(|&&z| z.abs() > threshold).count();
        let oracle = expected["normal"]["absolute_tail_counts"][key].as_u64().unwrap() as usize;
        assert_eq!(actual, oracle, "{key}");
    }

    let acceptance = &expected["acceptance"];
    assert!(
        expected["uniform"]["ks_p_value"].as_f64().unwrap()
            >= acceptance["uniform_ks_p_min"].as_f64().unwrap()
    );
    assert!(
        expected["uniform"]["bin20_p_value"].as_f64().unwrap()
            >= acceptance["uniform_bin20_p_min"].as_f64().unwrap()
    );
    assert!(
        expected["uniform"]["pair_10x10_p_value"].as_f64().unwrap()
            >= acceptance["uniform_pair_p_min"].as_f64().unwrap()
    );
    assert!(
        expected["normal"]["ks_p_value"].as_f64().unwrap()
            >= acceptance["normal_ks_p_min"].as_f64().unwrap()
    );
    assert!(
        expected["normal"]["normaltest_p_value"].as_f64().unwrap()
            >= acceptance["normaltest_p_min"].as_f64().unwrap()
    );
}

#[test]
fn gam_basis_fit_edf_and_prediction_match_frozen_scipy_oracle() {
    let expected = fixture("gam");
    assert_eq!(expected["reference"]["available"].as_bool(), Some(true));
    let atol = expected["atol"].as_f64().unwrap();
    let rtol = expected["rtol"].as_f64().unwrap();
    let mut workspace = GamWorkspace::default();

    for case in expected["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let x_columns: Vec<Vec<f64>> =
            case["x_colmajor"].as_array().unwrap().iter().map(floats).collect();
        let nrows = x_columns[0].len();
        let x: Vec<f64> = x_columns.iter().flatten().copied().collect();
        let y = floats(&case["y"]);
        let mut specs = Vec::new();
        for (index, spec) in case["specs"].as_array().unwrap().iter().enumerate() {
            let raw_col = spec["raw_col"].as_u64().unwrap() as usize;
            let n_basis = spec["n_basis"].as_u64().unwrap() as usize;
            let lambda = spec["lambda"].as_f64().unwrap();
            let knots = floats(&spec["knots"]);
            specs.push(SmoothSpec::new(raw_col, n_basis, lambda).with_knots(knots.clone()));

            let (basis, actual_knots) =
                expand_bspline(&x_columns[raw_col], n_basis, Some(&knots)).unwrap();
            let oracle_rows = rowmajor(&case["basis_matrices_rowmajor"][index]);
            let oracle = colmajor(&oracle_rows);
            assert_slice_close(&basis, &oracle, atol, rtol, &format!("{name} basis {index}"));
            assert_slice_close(
                actual_knots.as_ref(),
                &knots,
                0.0,
                0.0,
                &format!("{name} knots {index}"),
            );
        }

        let fit = fit_gam(
            &x,
            nrows,
            x_columns.len(),
            &y,
            &specs,
            &GamOptions { max_iter: 5000, tol: 1e-10 },
            &FaerBackend,
            &mut workspace,
        )
        .unwrap();
        assert!(fit.converged, "{name}");
        assert!(close(fit.intercept, case["intercept"].as_f64().unwrap(), atol, rtol));
        let oracle_coefficients: Vec<f64> =
            case["coefficients_by_smooth"].as_array().unwrap().iter().flat_map(floats).collect();
        assert_slice_close(
            &fit.coefficients,
            &oracle_coefficients,
            atol,
            rtol,
            &format!("{name} coefficients"),
        );
        assert_slice_close(
            &fit.fitted,
            &floats(&case["fitted"]),
            atol,
            rtol,
            &format!("{name} fitted"),
        );
        assert!(close(fit.edf_approx, case["edf"].as_f64().unwrap(), atol, rtol));

        let prediction_columns: Vec<Vec<f64>> =
            case["prediction_x_colmajor"].as_array().unwrap().iter().map(floats).collect();
        let prediction_nrows = prediction_columns[0].len();
        let prediction_x: Vec<f64> = prediction_columns.iter().flatten().copied().collect();
        let prediction =
            predict_gam(&fit, &prediction_x, prediction_nrows, prediction_columns.len()).unwrap();
        assert_slice_close(
            &prediction,
            &floats(&case["prediction"]),
            atol,
            rtol,
            &format!("{name} prediction"),
        );
    }
}

/// Rewrite GAM oracle fit fields after intentional roughness-penalty contract changes.
///
/// ```text
/// UPDATE_GAM_ORACLE=1 cargo test -p antecedent-stats --test foundations_oracle \
///   update_gam_oracle_fixture -- --ignored --nocapture
/// ```
#[test]
#[ignore = "run explicitly with UPDATE_GAM_ORACLE=1 to rewrite the fixture"]
fn update_gam_oracle_fixture() {
    assert_eq!(
        std::env::var("UPDATE_GAM_ORACLE").ok().as_deref(),
        Some("1"),
        "refusing to rewrite fixture without UPDATE_GAM_ORACLE=1"
    );
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/stats/gam/expected.json");
    let raw = fs::read_to_string(&path).expect("read oracle fixture");
    let mut root: Value = serde_json::from_str(&raw).expect("parse oracle fixture");
    let mut workspace = GamWorkspace::default();

    for case in root["cases"].as_array_mut().unwrap() {
        if case.get("x_colmajor").is_none() {
            continue;
        }
        let x_columns: Vec<Vec<f64>> =
            case["x_colmajor"].as_array().unwrap().iter().map(floats).collect();
        let nrows = x_columns[0].len();
        let x: Vec<f64> = x_columns.iter().flatten().copied().collect();
        let y = floats(&case["y"]);
        let mut specs = Vec::new();
        for spec in case["specs"].as_array().unwrap() {
            let raw_col = spec["raw_col"].as_u64().unwrap() as usize;
            let n_basis = spec["n_basis"].as_u64().unwrap() as usize;
            let lambda = spec["lambda"].as_f64().unwrap();
            let knots = floats(&spec["knots"]);
            specs.push(SmoothSpec::new(raw_col, n_basis, lambda).with_knots(knots));
        }
        let fit = fit_gam(
            &x,
            nrows,
            x_columns.len(),
            &y,
            &specs,
            &GamOptions { max_iter: 5000, tol: 1e-10 },
            &FaerBackend,
            &mut workspace,
        )
        .unwrap();

        let mut coefs_by_smooth = Vec::new();
        let mut off = 0usize;
        for spec in &specs {
            coefs_by_smooth.push(fit.coefficients[off..off + spec.n_basis].to_vec());
            off += spec.n_basis;
        }
        case["coefficients_by_smooth"] = Value::from(coefs_by_smooth);
        case["fitted"] = Value::from(fit.fitted.clone());
        case["intercept"] = Value::from(fit.intercept);
        case["edf"] = Value::from(fit.edf_approx);
        case["iterations"] = Value::from(fit.iterations);

        let prediction_columns: Vec<Vec<f64>> =
            case["prediction_x_colmajor"].as_array().unwrap().iter().map(floats).collect();
        let prediction_nrows = prediction_columns[0].len();
        let prediction_x: Vec<f64> = prediction_columns.iter().flatten().copied().collect();
        let prediction =
            predict_gam(&fit, &prediction_x, prediction_nrows, prediction_columns.len()).unwrap();
        case["prediction"] = Value::from(prediction);
    }

    root["reference"]["project"] = Value::from(
        "SciPy BSpline bases + Rust second-difference roughness (D2'D2) clean-room oracle",
    );
    root["reference"]["penalty"] = Value::from("second_difference_D2T_D2");
    let out = serde_json::to_string_pretty(&root).unwrap();
    fs::write(&path, out + "\n").expect("write oracle fixture");
}
