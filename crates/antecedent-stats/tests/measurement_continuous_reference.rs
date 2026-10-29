//! Classical continuous measurement error, replicate variance components and Rubin's rules
//! against 60-digit evaluation of their definitions
//! (`oracle_data/measurement_continuous_reference.py`).
//!
//! The reference builds the corrected normal equations `(M - s2 kbar e_W e_W') beta = r`, the
//! sandwich covariance, the derivative of the coefficients in the error variance (a numerical
//! derivative of the solve, not the closed form the product uses) and the one-way random-effects
//! components of unbalanced replicate groups from their definitions. The cases cover an error
//! variance estimated from the main sample's own replicates (influence retained), from an
//! independent replicate study, from a validation sample and declared; mixed replicate counts of
//! one to three; and Rubin's rules over five posterior draws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::measurement_error::{
    ErrorVariance, LinearErrorData, LinearMoments, MeasurementErrorError,
    combine_imputations_linear, posterior_means, replicate_variance_components, rubin_combine,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/measurement_continuous_reference.json"))
        .expect("frozen reference")
}

fn vec_of(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn counts_of(value: &Value) -> Vec<u32> {
    value.as_array().expect("array").iter().map(|v| v.as_u64().expect("count") as u32).collect()
}

fn close(got: f64, want: f64, relative: f64, what: &str) {
    assert!((got - want).abs() <= 1e-13 + relative * want.abs(), "{what}: {got} vs {want}");
}

fn all_close(got: &[f64], want: &[f64], relative: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        close(*g, *w, relative, &format!("{what}[{i}]"));
    }
}

#[test]
fn unbalanced_replicates_give_the_one_way_random_effects_components() {
    let r = reference();
    let rep = &r["replicates"];
    let (c, influence) =
        replicate_variance_components(&vec_of(&rep["values"]), &counts_of(&rep["sizes"])).unwrap();
    close(c.within_variance, rep["within"].as_f64().unwrap(), 1e-12, "within");
    close(c.between_variance, rep["between"].as_f64().unwrap(), 1e-11, "between");
    close(c.intraclass_correlation, rep["icc"].as_f64().unwrap(), 1e-11, "icc");
    close(c.effective_replicates, rep["n0"].as_f64().unwrap(), 1e-12, "n0");
    close(c.grand_mean, rep["grand"].as_f64().unwrap(), 1e-12, "grand mean");
    close(
        c.within_variance_estimate_variance,
        rep["var_within"].as_f64().unwrap(),
        1e-11,
        "variance of the within variance",
    );
    assert_eq!(c.within_df, rep["df"].as_f64().unwrap());
    all_close(&influence, &vec_of(&rep["xi"]), 1e-10, "influence");
}

#[test]
fn validation_error_variance_and_its_nonparametric_variance() {
    let r = reference();
    let v = &r["validation"];
    let (e, summary) = ErrorVariance::from_validation(&vec_of(&v["w"]), &vec_of(&v["x"])).unwrap();
    close(e.variance, v["variance"].as_f64().unwrap(), 1e-12, "variance");
    close(e.estimate_variance, v["estimate_variance"].as_f64().unwrap(), 1e-11, "its variance");
    assert!(e.influence.is_none(), "an independent validation sample retains no influence");
    close(summary.error_variance, e.variance, 0.0, "summary variance");
}

fn solve_case(main: &Value, error: &ErrorVariance, want: &Value, what: &str, tolerance: f64) {
    let y = vec_of(&main["y"]);
    let w = vec_of(&main["wbar"]);
    let z = vec_of(&main["z"]);
    let counts = counts_of(&main["counts"]);
    let data = LinearErrorData {
        y: &y,
        w: &w,
        covariates: &z,
        covariate_count: 1,
        replicate_counts: Some(&counts),
    };
    let fit = LinearMoments::accumulate(&data, error, 0.0).unwrap().solve().unwrap();
    all_close(&fit.coefficients, &vec_of(&want["beta"]), tolerance, &format!("{what} beta"));
    all_close(
        &fit.naive_coefficients,
        &vec_of(&want["naive"]),
        tolerance,
        &format!("{what} naive"),
    );
    all_close(&fit.covariance, &vec_of(&want["cov"]), 1e-7, &format!("{what} covariance"));
    all_close(
        &fit.covariance_given_error_variance,
        &vec_of(&want["given"]),
        tolerance,
        &format!("{what} covariance given s2"),
    );
    all_close(
        &fit.naive_covariance,
        &vec_of(&want["naive_cov"]),
        tolerance,
        &format!("{what} HC0"),
    );
    close(
        fit.reliability,
        want["reliability"].as_f64().unwrap(),
        tolerance,
        &format!("{what} reliability"),
    );
    // The functional variance is w' Cov w under the full covariance.
    let (value, variance) = fit.linear_functional(&[0.0, 1.0, 0.0]).unwrap();
    close(value, fit.coefficients[1], 0.0, "functional value");
    close(variance, fit.covariance[4], 1e-14, "functional variance");
}

#[test]
fn regression_calibration_matches_the_corrected_normal_equations_and_sandwich() {
    let r = reference();
    let main = &r["main"];
    // Error variance from the main sample's own replicates: the influence is retained, so the
    // covariance of the corrected score with the variance estimate enters.
    let (own, _) = ErrorVariance::from_main_sample_replicates(
        &vec_of(&main["values"]),
        &counts_of(&main["counts"]),
    )
    .unwrap();
    close(own.variance, main["within"].as_f64().unwrap(), 1e-12, "own within");
    assert!(own.influence.is_some());
    solve_case(main, &own, &main["fit_main_replicates"], "own replicates", 1e-10);

    // The same number from an independent replicate study keeps no influence.
    let rep = &r["replicates"];
    let (external, _) =
        ErrorVariance::from_external_replicates(&vec_of(&rep["values"]), &counts_of(&rep["sizes"]))
            .unwrap();
    solve_case(main, &external, &main["fit_external"], "external replicates", 1e-10);

    // A declared exact variance.
    let known = ErrorVariance::known(0.64, 0.0).unwrap();
    solve_case(main, &known, &main["fit_known"], "declared", 1e-10);

    // A validation sample.
    let v = &r["validation"];
    let (from_validation, _) =
        ErrorVariance::from_validation(&vec_of(&v["w"]), &vec_of(&v["x"])).unwrap();
    solve_case(main, &from_validation, &main["fit_validation"], "validation", 1e-10);
}

#[test]
fn a_reliability_at_or_below_zero_is_refused() {
    let r = reference();
    let main = &r["main"];
    let y = vec_of(&main["y"]);
    let w = vec_of(&main["wbar"]);
    let z = vec_of(&main["z"]);
    let counts = counts_of(&main["counts"]);
    let data = LinearErrorData {
        y: &y,
        w: &w,
        covariates: &z,
        covariate_count: 1,
        replicate_counts: Some(&counts),
    };
    // An error variance larger than the variance of W given the covariate leaves no latent signal.
    let huge = ErrorVariance::known(50.0, 0.0).unwrap();
    assert!(matches!(
        LinearMoments::accumulate(&data, &huge, 0.0),
        Err(MeasurementErrorError::ReliabilityNotPositive { reliability, floor })
            if reliability < 0.0 && floor == 0.0
    ));
}

#[test]
fn rubin_rules_combine_per_draw_fits_and_the_posterior_mean_is_the_draw_average() {
    let r = reference();
    let rub = &r["rubin"];
    let main = &r["main"];
    let draws: Vec<Vec<f64>> = rub["draws"].as_array().unwrap().iter().map(vec_of).collect();
    let flat: Vec<f64> = draws.iter().flatten().copied().collect();
    let combined =
        combine_imputations_linear(&vec_of(&main["y"]), &flat, draws.len(), &vec_of(&main["z"]), 1)
            .unwrap();
    all_close(&combined.coefficients, &vec_of(&rub["mean"]), 1e-10, "mean");
    all_close(&combined.within, &vec_of(&rub["within"]), 1e-9, "within");
    all_close(&combined.between, &vec_of(&rub["between"]), 1e-8, "between");
    all_close(&combined.covariance, &vec_of(&rub["total"]), 1e-8, "total");
    // The combination step alone reproduces the same numbers from the stored per-draw fits.
    let again = rubin_combine(&combined.draw_coefficients, &combined.draw_covariances).unwrap();
    assert_eq!(again.covariance, combined.covariance);
    all_close(
        &posterior_means(&flat, draws[0].len(), draws.len()).unwrap(),
        &vec_of(&rub["posterior_mean"]),
        1e-12,
        "posterior mean",
    );
    // One imputation cannot estimate the between-imputation variance.
    assert!(matches!(
        combine_imputations_linear(&vec_of(&main["y"]), &draws[0], 1, &vec_of(&main["z"]), 1),
        Err(MeasurementErrorError::InsufficientReplicates(
            "Rubin's rules need at least two imputations"
        ))
    ));
}
