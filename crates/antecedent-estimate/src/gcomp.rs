//! Shared g-computation contrast kernel (GLM + Bayesian).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::GlmFamily;

/// A column-major design with its treatment column and the two intervention values a
/// g-computation contrast compares.
///
/// The one description of "predict every row with the treatment column set to `active`,
/// then to `control`" shared by the GLM adjustment, its delta-method gradient and the
/// Bayesian posterior-draw evaluator.
#[derive(Clone, Copy, Debug)]
pub struct GcompContrast<'a> {
    /// Mean/link family of the fitted model.
    pub family: GlmFamily,
    /// Column-major design (`nrows * ncols`), treatment column included.
    pub x_colmajor: &'a [f64],
    /// Row count.
    pub nrows: usize,
    /// Column count.
    pub ncols: usize,
    /// Treatment column index within the design.
    pub t_col: usize,
    /// Active intervention value.
    pub active: f64,
    /// Control intervention value.
    pub control: f64,
}

impl GcompContrast<'_> {
    /// Per-row mean-scale contrast `μ(T=active, Z) − μ(T=control, Z)`.
    #[must_use]
    pub fn diffs(&self, coefficients: &[f64]) -> Vec<f64> {
        (0..self.nrows).map(|row| self.row_contrast(coefficients, row)).collect()
    }

    /// Single-row mean-scale contrast.
    #[must_use]
    pub fn row_contrast(&self, coefficients: &[f64], row: usize) -> f64 {
        let mut eta_active = 0.0;
        let mut eta_control = 0.0;
        for c in 0..self.ncols {
            let coef = coefficients[c];
            if c == self.t_col {
                eta_active += self.active * coef;
                eta_control += self.control * coef;
            } else {
                let val = self.x_colmajor[c * self.nrows + row];
                eta_active += val * coef;
                eta_control += val * coef;
            }
        }
        self.family.mean_from_eta(eta_active) - self.family.mean_from_eta(eta_control)
    }

    /// Row-averaged mean-scale contrast over every row.
    ///
    /// The identity link needs no data pass: only the treatment column differs, so the
    /// average is exactly `(active − control) β_T`. Nonlinear links share one
    /// residualized `η` (covariates only) between the two arms.
    #[must_use]
    pub fn mean_contrast(&self, beta: &[f64]) -> f64 {
        if matches!(self.family, GlmFamily::GaussianIdentity) {
            return (self.active - self.control) * beta[self.t_col];
        }
        let beta_t = beta[self.t_col];
        let mut sum = 0.0;
        for r in 0..self.nrows {
            let mut eta = 0.0;
            for c in 0..self.ncols {
                if c == self.t_col {
                    continue;
                }
                eta += self.x_colmajor[c * self.nrows + r] * beta[c];
            }
            sum += self.family.mean_from_eta(eta + beta_t * self.active)
                - self.family.mean_from_eta(eta + beta_t * self.control);
        }
        sum / self.nrows as f64
    }
}
