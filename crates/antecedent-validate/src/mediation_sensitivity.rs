//! Sensitivity of a linear mediation effect to mediator-outcome confounding (Imai, Keele
//! and Yamamoto 2010, "Identification, inference and sensitivity analysis for causal
//! mediation effects", Statistical Science 25(1)).
//!
//! The indirect (mediated) effect is identified under sequential ignorability: no
//! unmeasured confounding of the treatment, and no unmeasured confounding of the mediator
//! and the outcome given the treatment and covariates. The second part also carries the
//! cross-world independence the natural effects need, and it cannot be tested. In the
//! linear structural model
//!
//! `M = α₂ + β₂ T + ξ₂ᵀX + ε₂`, `Y = α₃ + β₃ T + γ M + ξ₃ᵀX + ε₃`,
//!
//! a violation is a correlation `ρ = corr(ε₂, ε₃) ≠ 0`, and the average causal mediation
//! effect becomes (their Theorem 2)
//!
//! `ACME(ρ) = β₂ σ₁ / σ₂ · (ρ̃ − ρ · sqrt((1 − ρ̃²) / (1 − ρ²)))`,
//!
//! with `σ₁` the residual standard deviation of the total-effect regression
//! `Y = α₁ + β₁ T + ξ₁ᵀX + ε₁`, `σ₂` that of the mediator regression, and
//! `ρ̃ = corr(ε₁, ε₂)`. At `ρ = 0` it is the product of coefficients `β₂ γ`; it vanishes at
//! the break-even `ρ = ρ̃`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::ValidationError;
use crate::assumption_range::{AssumptionRange, RangePoint, SensitivityAssumption};
use crate::target_native::RefutationTarget;

/// Summary statistics of the linear mediation model the curve is read from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MediationSensitivitySummary {
    /// Treatment coefficient of the mediator regression, `β₂`.
    pub beta_mediator: f64,
    /// Residual standard deviation of the total-effect regression, `σ₁`.
    pub sigma_total: f64,
    /// Residual standard deviation of the mediator regression, `σ₂`.
    pub sigma_mediator: f64,
    /// Correlation of the two residuals, `ρ̃` (the break-even `ρ`).
    pub rho_tilde: f64,
}

impl MediationSensitivitySummary {
    /// Mediated effect when the mediator-outcome error correlation is `rho`.
    ///
    /// # Errors
    ///
    /// `rho` outside `(−1, 1)`.
    pub fn acme(&self, rho: f64) -> Result<f64, ValidationError> {
        if !(rho > -1.0 && rho < 1.0) {
            return Err(ValidationError::NotApplicable {
                message: "the mediator-outcome error correlation must lie in (-1, 1)",
            });
        }
        let scale = self.beta_mediator * self.sigma_total / self.sigma_mediator;
        let shrink = ((1.0 - self.rho_tilde * self.rho_tilde) / (1.0 - rho * rho)).sqrt();
        Ok(scale * (self.rho_tilde - rho * shrink))
    }
}

/// Fit the total-effect and mediator regressions and summarize them.
///
/// # Errors
///
/// Mismatched lengths, too few rows, non-finite values, a rank-deficient design or a
/// residual with no variance.
pub fn mediation_sensitivity_fit(
    treatment: &[f64],
    mediator: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<MediationSensitivitySummary, ValidationError> {
    let n = outcome.len();
    if treatment.len() != n || mediator.len() != n || covariates.iter().any(|c| c.len() != n) {
        return Err(ValidationError::NotApplicable {
            message: "treatment, mediator, outcome and covariates must have the same length",
        });
    }
    let p = 2 + covariates.len();
    if n <= p + 1 {
        return Err(ValidationError::NotApplicable {
            message: "mediation sensitivity needs more rows than regression parameters plus one",
        });
    }
    if treatment
        .iter()
        .chain(mediator)
        .chain(outcome)
        .chain(covariates.iter().flatten())
        .any(|v| !v.is_finite())
    {
        return Err(ValidationError::NotApplicable {
            message: "mediation sensitivity needs finite values",
        });
    }
    let mut design = vec![1.0; n];
    design.extend_from_slice(treatment);
    for column in covariates {
        design.extend_from_slice(column);
    }
    let mut ws = LeastSquaresWorkspace::default();
    let total = FaerBackend.least_squares(&design, n, p, outcome, &mut ws)?;
    let med = FaerBackend.least_squares(&design, n, p, mediator, &mut ws)?;
    if total.rank < p || med.rank < p {
        return Err(ValidationError::NotApplicable {
            message: "mediation sensitivity needs a full-rank treatment and covariate design",
        });
    }
    let dof = (n - p) as f64;
    let sigma_total = (total.rss / dof).sqrt();
    let sigma_mediator = (med.rss / dof).sqrt();
    if !(sigma_total > 0.0 && sigma_mediator > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "the outcome or the mediator is fitted exactly by treatment and covariates",
        });
    }
    let cross: f64 = total.residuals.iter().zip(&med.residuals).map(|(a, b)| a * b).sum();
    let rho_tilde = cross / (total.rss.sqrt() * med.rss.sqrt());
    Ok(MediationSensitivitySummary {
        beta_mediator: med.coefficients[1],
        sigma_total,
        sigma_mediator,
        rho_tilde,
    })
}

/// Probed mediator-outcome error correlations.
const RHO_GRID: [f64; 13] =
    [-0.9, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7];

/// The mediation sensitivity curve as an [`AssumptionRange`] on `target`.
///
/// # Errors
///
/// None beyond a degenerate summary (non-finite values).
pub fn mediation_range(
    target: RefutationTarget,
    summary: &MediationSensitivitySummary,
) -> Result<AssumptionRange, ValidationError> {
    let mut tested = Vec::with_capacity(RHO_GRID.len());
    for &rho in &RHO_GRID {
        tested.push(RangePoint { parameter: rho, target: summary.acme(rho)?, test: None });
    }
    if tested.iter().any(|p| !p.target.is_finite()) {
        return Err(ValidationError::estimation_msg("mediation sensitivity curve is not finite"));
    }
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::MediatorOutcomeConfounding,
        "rho",
        0.0,
        tested,
        Some(summary.rho_tilde),
        None,
    ))
}

/// Slope of the treatment in the regression of `outcome` on a constant, the treatment and the
/// covariates: the total effect the mediator-outcome correlation `ρ` splits into its indirect
/// and direct parts.
///
/// # Errors
///
/// Mismatched lengths, too few rows, non-finite values or a rank-deficient design.
pub fn total_effect_slope(
    treatment: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<f64, ValidationError> {
    let n = outcome.len();
    let p = 2 + covariates.len();
    if treatment.len() != n
        || covariates.iter().any(|c| c.len() != n)
        || n <= p + 1
        || treatment
            .iter()
            .chain(outcome)
            .chain(covariates.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err(ValidationError::NotApplicable {
            message: "the total effect needs finite, equal-length columns and more rows than \
                      regression parameters plus one",
        });
    }
    let mut design = vec![1.0; n];
    design.extend_from_slice(treatment);
    for column in covariates {
        design.extend_from_slice(column);
    }
    let fit =
        FaerBackend.least_squares(&design, n, p, outcome, &mut LeastSquaresWorkspace::default())?;
    if fit.rank < p {
        return Err(ValidationError::NotApplicable {
            message: "the total effect needs a full-rank treatment and covariate design",
        });
    }
    Ok(fit.coefficients[1])
}

/// Correlation of the treatment and outcome residuals after a constant and the covariates:
/// the partial correlation that is the break-even of [`cross_world_noise_range`].
///
/// # Errors
///
/// As [`total_effect_slope`], or a residual with no variance.
pub fn residual_correlation(
    treatment: &[f64],
    outcome: &[f64],
    covariates: &[Vec<f64>],
) -> Result<f64, ValidationError> {
    let n = outcome.len();
    let p = 1 + covariates.len();
    if treatment.len() != n
        || covariates.iter().any(|c| c.len() != n)
        || n <= p + 1
        || treatment
            .iter()
            .chain(outcome)
            .chain(covariates.iter().flatten())
            .any(|v| !v.is_finite())
    {
        return Err(ValidationError::NotApplicable {
            message: "the residual correlation needs finite, equal-length columns and more rows \
                      than regression parameters plus one",
        });
    }
    let mut design = vec![1.0; n];
    for column in covariates {
        design.extend_from_slice(column);
    }
    let mut ws = LeastSquaresWorkspace::default();
    let t = FaerBackend.least_squares(&design, n, p, treatment, &mut ws)?;
    let y = FaerBackend.least_squares(&design, n, p, outcome, &mut ws)?;
    if !(t.rss > 0.0 && y.rss > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "the treatment or the outcome is fitted exactly by the covariates",
        });
    }
    let cross: f64 = t.residuals.iter().zip(&y.residuals).map(|(a, b)| a * b).sum();
    Ok(cross / (t.rss.sqrt() * y.rss.sqrt()))
}

/// Finest grid and bisection depth of the break-even search.
const BREAK_EVEN_STEPS: usize = 398;

/// Smallest-magnitude `ρ` in `(−0.995, 0.995)` at which `f` changes sign, by a grid scan and
/// bisection; `None` when `f` keeps one sign over the grid.
fn first_root(f: impl Fn(f64) -> f64) -> Option<f64> {
    let at = |k: usize| -0.995 + 1.99 * (k as f64) / (BREAK_EVEN_STEPS as f64);
    let mut best: Option<f64> = None;
    let mut previous = (at(0), f(at(0)));
    for k in 1..=BREAK_EVEN_STEPS {
        let x = at(k);
        let value = f(x);
        if previous.1 == 0.0 || previous.1 * value < 0.0 {
            let (mut lo, mut hi) = (previous.0, x);
            let mut f_lo = previous.1;
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                let f_mid = f(mid);
                if f_lo * f_mid <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    f_lo = f_mid;
                }
            }
            let root = 0.5 * (lo + hi);
            if best.is_none_or(|b| root.abs() < b.abs()) {
                best = Some(root);
            }
        }
        previous = (x, value);
    }
    best
}

/// The natural direct effect as a function of the mediator-outcome error correlation `ρ`, bound
/// to the sequential-ignorability (cross-world independence) assumption it relaxes.
///
/// `direct(ρ) = total − ACME(ρ)` (Imai, Keele and Yamamoto 2010): the total effect is
/// identified without the assumption, so the direct effect inherits the indirect effect's
/// dependence on `ρ`. At `ρ = 0` it is the direct coefficient of the outcome regression.
///
/// # Errors
///
/// A non-finite curve.
pub fn natural_direct_range(
    target: RefutationTarget,
    summary: &MediationSensitivitySummary,
    total_effect: f64,
) -> Result<AssumptionRange, ValidationError> {
    let direct = |rho: f64| summary.acme(rho).map(|acme| total_effect - acme);
    let mut tested = Vec::with_capacity(RHO_GRID.len());
    for &rho in &RHO_GRID {
        tested.push(RangePoint { parameter: rho, target: direct(rho)?, test: None });
    }
    if tested.iter().any(|p| !p.target.is_finite()) {
        return Err(ValidationError::estimation_msg(
            "direct-effect sensitivity curve is not finite",
        ));
    }
    let break_even = first_root(|rho| direct(rho).unwrap_or(f64::NAN));
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::MediatorOutcomeConfounding,
        "rho",
        0.0,
        tested,
        break_even,
        None,
    ))
}

/// A linear-Gaussian effect as a function of the treatment-outcome error correlation `ρ`.
///
/// With `r` the residual correlation of treatment and outcome after the covariates, the
/// least-squares slope is `β̂ = r s_y / s_t`. If the structural outcome error has correlation
/// `ρ` with the treatment error, then `cov(t, y) = β s_t² + ρ s_t σ_u` and
/// `var(y) = β² s_t² + 2 β ρ s_t σ_u + σ_u²`; eliminating `σ_u` gives
/// `β̂ − β = (s_y / s_t) · sqrt(1 − r²) · g(ρ)` with `g(x) = x / sqrt(1 − x²)`. The effect,
/// proportional to the slope, is therefore `effect · (1 − g(ρ) / g(r))`, which vanishes at
/// the break-even `ρ = r`.
///
/// # Errors
///
/// A non-finite effect, or `r` outside `(−1, 1)`.
pub fn cross_world_noise_range(
    target: RefutationTarget,
    effect: f64,
    residual_correlation: f64,
) -> Result<AssumptionRange, ValidationError> {
    if !effect.is_finite() || !(residual_correlation > -1.0 && residual_correlation < 1.0) {
        return Err(ValidationError::NotApplicable {
            message: "the cross-world range needs a finite effect and a residual correlation \
                      inside (-1, 1)",
        });
    }
    let r = residual_correlation;
    let mut tested = Vec::with_capacity(RHO_GRID.len());
    for &rho in &RHO_GRID {
        let g = |x: f64| x / (1.0 - x * x).sqrt();
        let value = if r == 0.0 { effect } else { effect * (1.0 - g(rho) / g(r)) };
        tested.push(RangePoint { parameter: rho, target: value, test: None });
    }
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::CrossWorldNoiseDependence,
        "rho",
        0.0,
        tested,
        (r != 0.0).then_some(r),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut t = Vec::new();
        let mut m = Vec::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        for i in 0..80 {
            let a = f64::from(i);
            let xi = (a * 0.3).sin();
            let ti = f64::from(i % 2);
            let mi = 0.8 * ti + 0.5 * xi + 0.4 * (a * 1.7).sin();
            let yi = 0.3 * ti + 1.2 * mi - 0.7 * xi + 0.5 * (a * 0.9).cos();
            t.push(ti);
            m.push(mi);
            y.push(yi);
            x.push(xi);
        }
        (t, m, y, x)
    }

    #[test]
    fn at_zero_correlation_the_curve_is_the_product_of_coefficients() {
        let (t, m, y, x) = data();
        let s = mediation_sensitivity_fit(&t, &m, &y, &[x.clone()]).unwrap();
        // Product of coefficients from the two structural regressions.
        let n = y.len();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&t);
        design.extend_from_slice(&m);
        design.extend_from_slice(&x);
        let outcome = FaerBackend
            .least_squares(&design, n, 4, &y, &mut LeastSquaresWorkspace::default())
            .unwrap();
        let gamma = outcome.coefficients[2];
        assert!((s.acme(0.0).unwrap() - s.beta_mediator * gamma).abs() < 1e-10);
    }

    #[test]
    fn the_effect_vanishes_at_the_break_even_correlation() {
        let (t, m, y, x) = data();
        let s = mediation_sensitivity_fit(&t, &m, &y, &[x]).unwrap();
        assert!(s.acme(s.rho_tilde).unwrap().abs() < 1e-12);
        assert!(s.acme(-1.0).is_err());
    }

    #[test]
    fn the_direct_effect_at_zero_is_the_outcome_regression_coefficient_and_vanishes_at_break_even()
    {
        let (t, m, y, x) = data();
        let s = mediation_sensitivity_fit(&t, &m, &y, &[x.clone()]).unwrap();
        let total = total_effect_slope(&t, &y, &[x.clone()]).unwrap();
        let n = y.len();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&t);
        design.extend_from_slice(&m);
        design.extend_from_slice(&x);
        let outcome = FaerBackend
            .least_squares(&design, n, 4, &y, &mut LeastSquaresWorkspace::default())
            .unwrap();
        let target = RefutationTarget::scalar_average_effect(
            crate::target_native::TargetQuadrant::STATIC_FREQUENTIST,
            [9; 32],
        );
        let range = natural_direct_range(target, &s, total).unwrap();
        // In-sample, total = direct + beta_mediator * gamma exactly, so direct(0) is the
        // coefficient of the treatment in the outcome regression on treatment and mediator.
        let at_zero = range.tested.iter().find(|p| p.parameter == 0.0).unwrap().target;
        assert!((at_zero - outcome.coefficients[1]).abs() < 1e-9, "{at_zero}");
        let rho = range.break_even.expect("the direct effect changes sign on the grid");
        assert!((total - s.acme(rho).unwrap()).abs() < 1e-9);
    }

    #[test]
    fn the_cross_world_range_recovers_the_slope_of_errors_correlated_at_rho() {
        // y = beta t + e_y with corr(e_t, e_y) = rho0 built exactly from orthogonal columns.
        let n = 400;
        let center = |v: Vec<f64>| {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            v.into_iter().map(|x| x - mean).collect::<Vec<f64>>()
        };
        let a = center((0..n).map(|i| (f64::from(i) * 0.37).sin()).collect());
        let mut b = center((0..n).map(|i| (f64::from(i) * 1.13).cos()).collect());
        let ab: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let aa: f64 = a.iter().map(|x| x * x).sum();
        for (bi, ai) in b.iter_mut().zip(&a) {
            *bi -= ab / aa * ai;
        }
        let norm = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        let (sa, sb) = (norm(&a), norm(&b));
        let rho0 = 0.4;
        let beta = 1.5;
        // Treatment error a (unit scale), outcome error rho0 * a + sqrt(1 - rho0^2) * b.
        let t: Vec<f64> = a.iter().map(|x| x / sa).collect();
        let e_y: Vec<f64> = t
            .iter()
            .zip(&b)
            .map(|(ti, bi)| rho0 * ti + (1.0 - rho0 * rho0).sqrt() * bi / sb)
            .collect();
        let y: Vec<f64> = t.iter().zip(&e_y).map(|(ti, e)| beta * ti + e).collect();
        let slope = total_effect_slope(&t, &y, &[]).unwrap();
        let r = residual_correlation(&t, &y, &[]).unwrap();
        // The OLS slope is biased by rho0 * sigma_u / sigma_t = rho0 * sd(e_y) (sd(t) = 1).
        let bias = rho0 * norm(&e_y);
        assert!((slope - (beta + bias)).abs() < 1e-9, "{slope}");
        let target = RefutationTarget::scalar_average_effect(
            crate::target_native::TargetQuadrant::STATIC_FREQUENTIST,
            [9; 32],
        );
        let range = cross_world_noise_range(target, slope, r).unwrap();
        assert_eq!(range.break_even, Some(r));
        // At the true correlation the range returns the structural slope beta.
        let g = |x: f64| x / (1.0 - x * x).sqrt();
        let at = |rho: f64| slope * (1.0 - g(rho) / g(r));
        assert!((at(rho0) - beta).abs() < 1e-9, "{}", at(rho0));
        let point = range.tested.iter().find(|p| p.parameter == 0.3).unwrap();
        assert!((point.target - at(0.3)).abs() < 1e-12);
    }
}
