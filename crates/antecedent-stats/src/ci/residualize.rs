//! Shared (weighted) residualization on an intercept plus conditioning columns.
//!
//! One implementation serves the weighted, multivariate and Bayesian CI tests so they agree
//! on the same data. Columns are weighted-centred and scaled to unit weighted variance
//! before the normal equations are formed, so the conditioning of the Gram matrix is that of
//! the columns' correlation matrix and does not degrade with the mean-to-sd ratio of a level
//! series (`[1 | Z]'[1 | Z]` has condition number ~ (mean/sd)^4 when formed raw).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::trivially_copy_pass_by_ref,
    clippy::neg_cmp_op_on_partial_ord
)]

use antecedent_core::KernelPolicy;
use antecedent_kernels::{sanitize_weight, weighted_mean};

use crate::error::StatsError;
use crate::gram::invert_square;

/// A conditioning column whose weighted sd is below this fraction of its mean magnitude is
/// constant to working precision: it is collinear with the intercept.
const CONSTANT_COLUMN_REL_SD: f64 = 1e-12;

/// A fitted `[1 | Z]` weighted least-squares design, reusable across response columns.
#[derive(Clone, Debug)]
pub(crate) struct ZDesign {
    n: usize,
    qz: usize,
    /// Sanitised row weights (all ones when unweighted).
    weights: Vec<f64>,
    sum_w: f64,
    /// Centred, unit-weighted-variance Z columns, column-major `n × qz`.
    u: Vec<f64>,
    /// Inverse of the weighted correlation matrix of `u`, row-major `qz × qz`.
    g_inv: Vec<f64>,
}

impl ZDesign {
    /// Fit the design from the columns `z` of `columns` (all `n` rows).
    ///
    /// `weights = None` is the unweighted (unit-weight) design.
    ///
    /// # Errors
    ///
    /// [`StatsError::Shape`] on non-finite columns, no positive weight mass, or a constant /
    /// collinear conditioning set.
    pub(crate) fn fit(
        columns: &[&[f64]],
        z: &[usize],
        weights: Option<&[f64]>,
        n: usize,
    ) -> Result<Self, StatsError> {
        let w: Vec<f64> = match weights {
            Some(ws) => {
                if ws.len() < n {
                    return Err(StatsError::Shape { message: "weights length < nrows" });
                }
                ws[..n].iter().map(|&v| sanitize_weight(v)).collect()
            }
            None => vec![1.0; n],
        };
        let sum_w: f64 = w.iter().sum();
        if !(sum_w > 0.0 && sum_w.is_finite()) {
            return Err(StatsError::Shape { message: "no positive observation weight" });
        }
        let qz = z.len();
        let mut u = vec![0.0; n * qz];
        for (j, &zc) in z.iter().enumerate() {
            let col = columns
                .get(zc)
                .ok_or(StatsError::Shape { message: "conditioning column out of range" })?;
            if col.len() < n || col[..n].iter().any(|v| !v.is_finite()) {
                return Err(StatsError::Shape { message: "non-finite conditioning column" });
            }
            let mean = col[..n].iter().zip(&w).map(|(v, wi)| wi * v).sum::<f64>() / sum_w;
            let var = col[..n]
                .iter()
                .zip(&w)
                .map(|(v, wi)| {
                    let d = v - mean;
                    wi * d * d
                })
                .sum::<f64>()
                / sum_w;
            let sd = var.sqrt();
            if !(sd.is_finite() && sd > CONSTANT_COLUMN_REL_SD * mean.abs()) {
                return Err(StatsError::Shape {
                    message: "singular Z design: a conditioning column is constant",
                });
            }
            for r in 0..n {
                u[j * n + r] = (col[r] - mean) / sd;
            }
        }
        let mut g = vec![0.0; qz * qz];
        for a in 0..qz {
            for b in a..qz {
                let mut s = 0.0;
                for r in 0..n {
                    s += w[r] * u[a * n + r] * u[b * n + r];
                }
                s /= sum_w;
                g[a * qz + b] = s;
                g[b * qz + a] = s;
            }
        }
        let g_inv = if qz == 0 {
            Vec::new()
        } else {
            invert_square(&g, qz).ok_or(StatsError::Shape {
                message: "singular Z design: conditioning columns are collinear",
            })?
        };
        Ok(Self { n, qz, weights: w, sum_w, u, g_inv })
    }

    /// Residuals of `target` after weighted least squares on `[1 | Z]`.
    ///
    /// # Errors
    ///
    /// [`StatsError::Shape`] on a non-finite or short target.
    pub(crate) fn residuals(&self, target: &[f64]) -> Result<Vec<f64>, StatsError> {
        let n = self.n;
        if target.len() < n || target[..n].iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "non-finite response in residualization" });
        }
        let mean =
            target[..n].iter().zip(&self.weights).map(|(v, w)| w * v).sum::<f64>() / self.sum_w;
        let tc: Vec<f64> = target[..n].iter().map(|v| v - mean).collect();
        let mut rhs = vec![0.0; self.qz];
        for a in 0..self.qz {
            let mut s = 0.0;
            for r in 0..n {
                s += self.weights[r] * self.u[a * n + r] * tc[r];
            }
            rhs[a] = s / self.sum_w;
        }
        let mut beta = vec![0.0; self.qz];
        for a in 0..self.qz {
            for b in 0..self.qz {
                beta[a] += self.g_inv[a * self.qz + b] * rhs[b];
            }
        }
        let mut out = tc;
        for a in 0..self.qz {
            for r in 0..n {
                out[r] -= beta[a] * self.u[a * n + r];
            }
        }
        Ok(out)
    }
}

/// Weighted Pearson correlation with weighted centering (`weights` are sanitised).
pub(crate) fn weighted_pearson(
    policy: &KernelPolicy,
    x: &[f64],
    y: &[f64],
    weights: &[f64],
) -> Option<f64> {
    let n = x.len();
    let mx = weighted_mean(policy, x, weights)?;
    let my = weighted_mean(policy, y, weights)?;
    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for i in 0..n {
        let w = sanitize_weight(weights[i]);
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cxx += w * dx * dx;
        cyy += w * dy * dy;
        cxy += w * dx * dy;
    }
    if !(cxx > 0.0 && cyy > 0.0) {
        return None;
    }
    let denom = (cxx * cyy).sqrt();
    if !denom.is_finite() || denom == 0.0 {
        return None;
    }
    Some((cxy / denom).clamp(-1.0, 1.0))
}

/// Sum of squares of `v` about its plain mean (the information in an unconditioned column).
pub(crate) fn centered_sum_squares(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (x - mean) * (x - mean)).sum()
}

/// Whether a residual carries no information about its column: the column is constant, or
/// the conditioning set explains it to working precision.
pub(crate) fn residual_is_uninformative(raw: &[f64], resid: &[f64]) -> bool {
    let raw_ss = centered_sum_squares(raw);
    let res_ss: f64 = resid.iter().map(|v| v * v).sum();
    !(raw_ss > 0.0) || res_ss <= EXPLAINED_BY_Z_REL_SS * raw_ss
}

/// A residual carrying less than this fraction of its column's variance is numerically
/// fully explained by the conditioning set; the correlation of what remains is rounding.
pub(crate) const EXPLAINED_BY_Z_REL_SS: f64 = 1e-20;

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 33) as f64) / ((1u64 << 31) as f64) - 0.5
            })
            .collect()
    }

    /// Ordinary least squares on `[1, z]` for one conditioner, from closed-form simple
    /// regression: `beta = cov(z,t) / var(z)`.
    fn simple_ols_residuals(t: &[f64], z: &[f64]) -> Vec<f64> {
        let n = t.len() as f64;
        let mt = t.iter().sum::<f64>() / n;
        let mz = z.iter().sum::<f64>() / n;
        let szz: f64 = z.iter().map(|v| (v - mz) * (v - mz)).sum();
        let szt: f64 = z.iter().zip(t).map(|(a, b)| (a - mz) * (b - mt)).sum();
        let beta = szt / szz;
        t.iter().zip(z).map(|(tv, zv)| (tv - mt) - beta * (zv - mz)).collect()
    }

    #[test]
    fn matches_closed_form_simple_regression() {
        let n = 60;
        let z = lcg(n, 1);
        let t: Vec<f64> = lcg(n, 2).iter().zip(&z).map(|(e, zv)| 1.5 * zv + e).collect();
        let cols: [&[f64]; 1] = [&z];
        let fit = ZDesign::fit(&cols, &[0], None, n).unwrap();
        let got = fit.residuals(&t).unwrap();
        let want = simple_ols_residuals(&t, &z);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12, "{g} vs {w}");
        }
    }

    /// The uncentred normal equations rejected this design ("singular Z design") once the
    /// level's mean-to-sd ratio passed ~2e4. Residuals must not depend on the offset.
    #[test]
    fn residuals_are_offset_invariant_at_large_levels() {
        let n = 80;
        let z0 = lcg(n, 3);
        let t: Vec<f64> = lcg(n, 4).iter().zip(&z0).map(|(e, zv)| -0.7 * zv + e).collect();
        let want = simple_ols_residuals(&t, &z0);
        for offset in [1e3, 1e5, 1.7e9] {
            let z: Vec<f64> = z0.iter().map(|v| v + offset).collect();
            let cols: [&[f64]; 1] = [&z];
            let fit = ZDesign::fit(&cols, &[0], None, n).expect("offset design must fit");
            let got = fit.residuals(&t).unwrap();
            // The offset costs ~eps * offset of absolute precision in the centred column.
            let tol = 100.0 * f64::EPSILON * offset + 1e-12;
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < tol, "offset {offset}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn weighted_fit_matches_weighted_simple_regression() {
        let n = 50;
        let z = lcg(n, 5);
        let t: Vec<f64> = lcg(n, 6).iter().zip(&z).map(|(e, zv)| 2.0 * zv + e).collect();
        let w: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64).collect();
        let sw: f64 = w.iter().sum();
        let mz = z.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / sw;
        let mt = t.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / sw;
        let szz: f64 = z.iter().zip(&w).map(|(a, b)| b * (a - mz) * (a - mz)).sum();
        let szt: f64 = (0..n).map(|i| w[i] * (z[i] - mz) * (t[i] - mt)).sum();
        let beta = szt / szz;
        let cols: [&[f64]; 1] = [&z];
        let fit = ZDesign::fit(&cols, &[0], Some(&w), n).unwrap();
        let got = fit.residuals(&t).unwrap();
        for i in 0..n {
            let want = (t[i] - mt) - beta * (z[i] - mz);
            assert!((got[i] - want).abs() < 1e-12, "{i}: {} vs {want}", got[i]);
        }
    }

    #[test]
    fn constant_and_collinear_conditioners_are_refused() {
        let n = 30;
        let a = lcg(n, 7);
        let constant = vec![4.0; n];
        let twin: Vec<f64> = a.iter().map(|v| 2.0 * v + 1.0).collect();
        let cols: [&[f64]; 3] = [&a, &constant, &twin];
        assert!(ZDesign::fit(&cols, &[1], None, n).is_err());
        assert!(ZDesign::fit(&cols, &[0, 2], None, n).is_err());
        assert!(ZDesign::fit(&cols, &[0], None, n).is_ok());
    }
}
