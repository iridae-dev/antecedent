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
}
