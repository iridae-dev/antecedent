// SPDX-License-Identifier: MIT OR Apache-2.0
//! The time-varying panel likelihood against a dense-covariance reference.
//!
//! The path changes only the mean mechanism; the residuals of every group keep the stationary
//! AR(1) plus shared random-intercept covariance. The reference builds the dense covariance
//! `sigma^2/(1-rho^2) rho^|i-j| + tau^2` of each group's residual vector (clustered: the
//! intercept variance on every within-group pair; serial: the AR(1) decay) and evaluates the
//! Gaussian log-likelihood by Cholesky factorization, which shares nothing with the closed-form
//! precision and rank-one update the product uses.

use super::*;

fn dense_group_log_likelihood(r: &[f64], rho: f64, sigma: f64, tau: f64) -> f64 {
    let n = r.len();
    let covariance: Vec<f64> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                tau * tau
                    + sigma * sigma / (1.0 - rho * rho)
                        * rho.powi(i32::try_from(i.abs_diff(j)).unwrap())
            })
        })
        .collect();
    let lower = antecedent_stats::cholesky_spd(&covariance, n).unwrap();
    let inverse = antecedent_kernels::chol_inverse(&lower, n);
    let quadratic: f64 =
        (0..n).map(|i| r[i] * (0..n).map(|j| inverse[i * n + j] * r[j]).sum::<f64>()).sum();
    -0.5 * (n as f64 * std::f64::consts::TAU.ln()
        + antecedent_stats::chol_log_det(&lower, n)
        + quadratic)
}

fn specification() -> TemporalPanelSpec {
    TemporalPanelSpec {
        varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
        change_points: vec![8, 12, 16],
        minimum_segment: 4,
        weak_change_gap: 4.0,
        smooth_knots: vec![0, 8, 16, 23],
        smoothing: vec![0.5, 2.0],
        path_prior_sd: 1.0,
    }
}

/// Changed-mechanism objective against the dense covariance of the same residuals.
fn assert_matches_dense(rho: f64, sigma: f64, tau: f64) {
    let (data, baseline, _) = super::super::tests::fixture();
    let mut parameters = baseline.parameters.clone();
    parameters.rho = rho;
    parameters.innovation_sd = sigma;
    parameters.group_sd = tau;
    let spec = specification();
    let choice = TemporalModelChoice::Piecewise { at: 12 };
    let mut values = likelihood::encode(&data, &parameters).unwrap();
    values.push(1.5);
    let compiled = likelihood::CompiledPanel::new(&data).unwrap();
    let actual = objective(&compiled, &spec, &choice, &values);
    let expected: f64 = compiled
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let residuals: Vec<f64> = (0..group.outcome.len())
                .map(|time| {
                    let p = parameters_at(&data, &spec, &choice, &values, time).unwrap();
                    group.outcome[time]
                        - likelihood::observed_response(&compiled, &p, group, index, time)
                })
                .collect();
            let decoded =
                likelihood::decode(&data, &values[..likelihood::parameter_count(&data)]).unwrap();
            dense_group_log_likelihood(
                &residuals,
                decoded.rho,
                decoded.innovation_sd,
                decoded.group_sd,
            )
        })
        .sum();
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "rho {rho} sigma {sigma} tau {tau}: product {actual} vs dense {expected}"
    );
    // The change is not vacuous: the no-change objective differs on the same dependence law.
    let mut unchanged = values.clone();
    *unchanged.last_mut().unwrap() = 0.0;
    assert!((objective(&compiled, &spec, &choice, &unchanged) - actual).abs() > 1e-6);
}

#[test]
fn time_varying_serial_dependence_matches_the_dense_ar1_covariance() {
    // Strong positive and negative serial correlation with a negligible shared intercept.
    assert_matches_dense(0.7, 0.4, 0.02);
    assert_matches_dense(-0.5, 0.35, 0.02);
}

#[test]
fn time_varying_clustered_dependence_matches_the_dense_shared_intercept_covariance() {
    // A dominant shared group intercept with weak serial correlation.
    assert_matches_dense(0.05, 0.3, 0.9);
    assert_matches_dense(0.3, 0.25, 0.6);
}
