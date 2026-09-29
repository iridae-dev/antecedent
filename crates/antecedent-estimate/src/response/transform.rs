//! Derivative-scale transforms and their interval disclosures.
//!
//! The transform / delta-method / Fieller math that turns a local-polynomial
//! coordinate into an elasticity, semi-elasticity, or transformed second
//! derivative, together with the runtime notes published alongside a
//! transformed or bias-corrected point-derivative interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{DerivativeScale, Diagnostic, DiagnosticKind, DiagnosticSeverity};

use crate::EstimationError;

pub(super) fn transform_derivative(
    derivative: f64,
    treatment: f64,
    response: f64,
    scale: DerivativeScale,
) -> Result<f64, EstimationError> {
    let value = match scale {
        DerivativeScale::Identity => derivative,
        DerivativeScale::LogTreatment => treatment * derivative,
        DerivativeScale::LogOutcome => {
            if response <= 0.0 {
                return Err(EstimationError::unsupported(
                    "log-outcome derivative scale requires a positive fitted response",
                ));
            }
            derivative / response
        }
        DerivativeScale::LogLog => {
            if response <= 0.0 {
                return Err(EstimationError::unsupported(
                    "elasticity requires a positive fitted response",
                ));
            }
            treatment * derivative / response
        }
    };
    Ok(value)
}

/// Runtime disclosure attached whenever a transformed point-derivative interval
/// (elasticity, semi-elasticity, or a transformed second derivative) is published
/// from the delta method on the joint local-coordinate covariance.
pub(super) fn delta_method_interval_note(conventional: f64, interval_center: f64) -> Diagnostic {
    let mut note = Diagnostic::new(
        "response.derivative_interval_delta_method",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "the transformed-derivative interval is a delta-method interval: it is centred at the bias-corrected transform (a nonlinear function of the local level m and derivatives m', m'') and its standard_error is sqrt(∇τᵀ Σ_θ ∇τ), where Σ_θ is the heteroskedasticity-robust joint influence covariance of the bias-corrected local-cubic slope and local-quartic level and curvature and ∇τ the transform gradient at the bias-corrected coordinate. [lower, upper] is not a CI for the printed conventional point; it targets the true transformed derivative, conditions on the caller-fixed bandwidth, treats the cross-fitted pseudo-outcome as data, and its coverage carries the transform's linearization error in addition to the coordinate-covariance error",
    );
    note.fields = Arc::from(vec![
        (Arc::from("conventional_point"), Arc::from(conventional.to_string())),
        (Arc::from("interval_center"), Arc::from(interval_center.to_string())),
    ]);
    note
}

pub(super) fn fieller_interval_note(conventional: f64, interval_center: f64) -> Diagnostic {
    let mut note = Diagnostic::new(
        "response.derivative_interval_fieller",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "the elasticity confidence interval inverts the joint normal test for a*m'/m using the bias-corrected level and slope and their full robust covariance (Fieller's method); the reported standard_error is the local delta-method approximation, while the interval bounds account for denominator uncertainty and need not be symmetric around the printed conventional point or the bias-corrected ratio. The interval conditions on the caller-fixed bandwidth and treats the cross-fitted pseudo-outcome as data",
    );
    note.fields = Arc::from(vec![
        (Arc::from("conventional_point"), Arc::from(conventional.to_string())),
        (Arc::from("bias_corrected_ratio"), Arc::from(interval_center.to_string())),
    ]);
    note
}

/// Bounded Fieller confidence interval for the elasticity `at * slope / level`.
/// The complementary case is an unbounded confidence set, which cannot be
/// encoded by the scalar interval contract.
pub(super) fn fieller_elasticity_interval(
    level: f64,
    slope: f64,
    at: f64,
    covariance: &[[f64; 3]; 3],
    z: f64,
) -> Option<(f64, f64)> {
    let var_level = covariance[0][0];
    let var_slope = covariance[1][1];
    let cov_level_slope = covariance[0][1];
    let a = level * level - z * z * var_level;
    let b = -2.0 * at * (level * slope - z * z * cov_level_slope);
    let c = at * at * (slope * slope - z * z * var_slope);
    let discriminant = b.mul_add(b, -4.0 * a * c);
    if !(a > 0.0 && discriminant >= 0.0 && discriminant.is_finite()) {
        return None;
    }
    let radius = discriminant.sqrt();
    let lower = (-b - radius) / (2.0 * a);
    let upper = (-b + radius) / (2.0 * a);
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some((lower, upper))
}

/// Runtime disclosure attached whenever a point-derivative interval is published.
pub(super) fn bias_corrected_interval_note(
    bayesian: bool,
    conventional: f64,
    interval_center: f64,
) -> Diagnostic {
    let mut note = Diagnostic::new(
        "response.derivative_interval_bias_corrected",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        if bayesian {
            "the derivative credible interval is robust bias-corrected: every Dirichlet row-weight draw refits the cross-fitted outcome and Gaussian treatment nuisances, rebuilds the Kennedy pseudo-outcome, and evaluates both the local-quadratic coordinate (averaged into the reported value) and its bias-corrected coordinate (Calonico-Cattaneo-Titiunik, pilot bandwidth equal to the caller bandwidth: the local-cubic slope for a first derivative, the local-quartic level and curvature otherwise; its quantiles give the interval and its SD the standard_error); [lower, upper] is not a CI for the reported conventional point; the interval targets the true derivative and holds only the caller-fixed bandwidth, fold assignment, spline knots, and penalty fixed"
        } else {
            "the derivative interval is robust bias-corrected (Calonico-Cattaneo-Titiunik, pilot bandwidth equal to the caller bandwidth): it is centered at the bias-corrected coordinate (the local-cubic slope for a first derivative, the local-quartic curvature for a second derivative) rather than at the reported local-quadratic point estimate, so [lower, upper] is not a CI for the printed value; standard_error is the bias-corrected standard error; it targets the true derivative, conditions on the caller-fixed bandwidth, and treats the cross-fitted pseudo-outcome as data"
        },
    );
    note.fields = Arc::from(vec![
        (Arc::from("conventional_point"), Arc::from(conventional.to_string())),
        (Arc::from("interval_center"), Arc::from(interval_center.to_string())),
    ]);
    note
}

pub(super) fn transform_point_derivative(
    response: f64,
    first: f64,
    second: f64,
    treatment: f64,
    order: u8,
    scale: DerivativeScale,
) -> Result<f64, EstimationError> {
    if order == 1 {
        return transform_derivative(first, treatment, response, scale);
    }
    if matches!(scale, DerivativeScale::LogOutcome | DerivativeScale::LogLog) && response <= 0.0 {
        return Err(EstimationError::unsupported(
            "log-outcome derivative scale requires a positive fitted response",
        ));
    }
    Ok(match scale {
        DerivativeScale::Identity => second,
        DerivativeScale::LogTreatment => treatment * first + treatment * treatment * second,
        DerivativeScale::LogOutcome => second / response - (first / response).powi(2),
        DerivativeScale::LogLog => {
            treatment * first / response
                + treatment * treatment * (second / response - (first / response).powi(2))
        }
    })
}

/// Gradient `∇τ = (∂τ/∂m, ∂τ/∂m', ∂τ/∂m'')` of the transformed point-derivative
/// estimand `τ` with respect to the local-polynomial coordinates `θ = (m, m', m'')`
/// (the level, first derivative, and second derivative), evaluated at `(m, first,
/// second)` and treatment `at`.
///
/// These are the exact partials of [`transform_point_derivative`], used by the
/// delta method to turn the joint coordinate covariance `Σ_θ` into a variance for
/// `τ`. `m > 0` is a precondition for the log-outcome and log-log scales (their
/// transform errors otherwise); the caller establishes it before calling.
pub(super) fn transform_point_derivative_gradient(
    m: f64,
    first: f64,
    second: f64,
    at: f64,
    order: u8,
    scale: DerivativeScale,
) -> [f64; 3] {
    if order == 1 {
        // τ is a function of (m, m') only, so ∂τ/∂m'' = 0.
        return match scale {
            DerivativeScale::Identity => [0.0, 1.0, 0.0],
            DerivativeScale::LogTreatment => [0.0, at, 0.0],
            // s = m'/m
            DerivativeScale::LogOutcome => [-first / (m * m), 1.0 / m, 0.0],
            // η = at·m'/m
            DerivativeScale::LogLog => [-at * first / (m * m), at / m, 0.0],
        };
    }
    match scale {
        DerivativeScale::Identity => [0.0, 0.0, 1.0],
        // g = at·m' + at²·m''
        DerivativeScale::LogTreatment => [0.0, at, at * at],
        // g = m''/m − (m'/m)²
        DerivativeScale::LogOutcome => {
            [-second / (m * m) + 2.0 * first * first / (m * m * m), -2.0 * first / (m * m), 1.0 / m]
        }
        // g = at·m'/m + at²·(m''/m − (m'/m)²)
        DerivativeScale::LogLog => [
            -at * first / (m * m)
                + at * at * (-second / (m * m) + 2.0 * first * first / (m * m * m)),
            at / m - 2.0 * at * at * first / (m * m),
            at * at / m,
        ],
    }
}

/// Delta-method standard error `sqrt(∇τᵀ Σ_θ ∇τ)` of a transformed point
/// derivative, from the joint coordinate covariance `Σ_θ` and gradient `∇τ`.
pub(super) fn delta_method_standard_error(covariance: &[[f64; 3]; 3], gradient: &[f64; 3]) -> f64 {
    let mut variance = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            variance += gradient[a] * covariance[a][b] * gradient[b];
        }
    }
    variance.max(0.0).sqrt()
}
