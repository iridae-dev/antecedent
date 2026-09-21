//! Scalar linear-Gaussian state-space helpers (Kalman 1960 filter / Rauch–Tung–Striebel
//! 1965 smoother / residual-path sampling).
//!
//! Model: `x_t = a x_{t-1} + σ_proc ε_t`, `y_t = x_t + σ_obs η_t` with `ε, η ~ N(0,1)`.
//!
//! The structural noise of an LGSSM mechanism is the *residual path*
//! `r_t = x_t + σ_obs η_t` itself, one lossless `f64` per row. Parents enter only
//! through an additive mean, so `y_t = mean(pa_t) + r_t` and the residual is
//! identified exactly from an observed row (`r_t = y_t − mean(pa_t)`): abduction is
//! exact inversion, and replaying it with unchanged parents reproduces the data to
//! machine precision. The latent split of `r_t` into `(ε, η)` is not identified by
//! one series and does not affect any counterfactual of the outcome, so it is not
//! carried.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::needless_range_loop)]

use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;

use crate::error::ModelError;

/// Sample a residual path `r_t = x_t + σ_obs η_t` from the LGSSM prior into
/// `output` (`x_0 = initial_mean + σ_proc ε_0`, `x_t = a x_{t-1} + σ_proc ε_t`).
///
/// # Errors
///
/// `output` shorter than `n_rows`.
pub fn sample_lgssm_noise(
    n_rows: usize,
    a: f64,
    process_std: f64,
    obs_std: f64,
    initial_mean: f64,
    rng: &mut CausalRng,
    output: &mut [f64],
) -> Result<(), ModelError> {
    if output.len() < n_rows {
        return Err(ModelError::Shape { message: "lgssm noise output too short".into() });
    }
    let mut x = initial_mean;
    for t in 0..n_rows {
        let eps = standard_normal(rng);
        x = if t == 0 { initial_mean + process_std * eps } else { a * x + process_std * eps };
        output[t] = x + obs_std * standard_normal(rng);
    }
    Ok(())
}

/// Forward Kalman (1960) filter for a scalar LGSSM.
///
/// Returns filtered means/variances and one-step predictive means/variances.
#[must_use]
pub fn kalman_filter(
    y: &[f64],
    a: f64,
    process_var: f64,
    obs_var: f64,
    x0: f64,
    p0: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut x_f = vec![0.0; n];
    let mut p_f = vec![0.0; n];
    let mut x_pred = vec![0.0; n];
    let mut p_pred = vec![0.0; n];
    let mut x = x0;
    let mut p = p0;
    for t in 0..n {
        let xp = if t == 0 { x } else { a * x };
        let pp = if t == 0 { p } else { a * a * p + process_var };
        x_pred[t] = xp;
        p_pred[t] = pp;
        let s = pp + obs_var;
        let k = if s > 0.0 { pp / s } else { 0.0 };
        x = xp + k * (y[t] - xp);
        // Equivalent to (1-k)*pp without cancellation when k rounds to one.
        p = if s > 0.0 { (pp / s) * obs_var } else { 0.0 };
        x_f[t] = x;
        p_f[t] = p.max(0.0);
    }
    (x_f, p_f, x_pred, p_pred)
}

/// Rauch–Tung–Striebel (1965) smoother given filter outputs.
///
/// Returns smoothed means, variances, and lag-1 cross-covariances `p_lag[t] = Cov(x_t, x_{t-1})`.
#[must_use]
pub fn rts_smooth(
    a: f64,
    x_f: &[f64],
    p_f: &[f64],
    x_pred: &[f64],
    p_pred: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x_f.len();
    let mut x_s = x_f.to_vec();
    let mut p_s = p_f.to_vec();
    let mut p_lag = vec![0.0; n];
    for t in (0..n.saturating_sub(1)).rev() {
        let pp = p_pred[t + 1];
        let j = if pp > 0.0 { p_f[t] * a / pp } else { 0.0 };
        x_s[t] = x_f[t] + j * (x_s[t + 1] - x_pred[t + 1]);
        p_s[t] = p_f[t] + j * j * (p_s[t + 1] - p_pred[t + 1]);
        p_lag[t + 1] = j * p_s[t + 1];
    }
    (x_s, p_s, p_lag)
}

/// Scale the state and observations before squaring noise scales, preserving
/// units without imposing a positive variance floor on the statistical model.
pub(crate) struct ScaledLgssm {
    pub scale: f64,
    pub values: Vec<f64>,
    pub process_var: f64,
    pub obs_var: f64,
    pub initial_mean: f64,
}

pub(crate) fn scaled_lgssm(
    y: &[f64],
    a: f64,
    process_std: f64,
    obs_std: f64,
    initial_mean: f64,
) -> Result<ScaledLgssm, ModelError> {
    if !a.is_finite()
        || !initial_mean.is_finite()
        || !process_std.is_finite()
        || process_std <= 0.0
        || !obs_std.is_finite()
        || obs_std <= 0.0
        || y.iter().any(|value| !value.is_finite())
    {
        return Err(ModelError::Numerical {
            message: "LGSSM requires finite observations/parameters and positive noise scales"
                .into(),
        });
    }
    let scale = process_std.max(obs_std);
    Ok(ScaledLgssm {
        scale,
        values: y.iter().map(|value| value / scale).collect(),
        process_var: (process_std / scale).powi(2),
        obs_var: (obs_std / scale).powi(2),
        initial_mean: initial_mean / scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{MechanismWorkspace, ParentBatch};
    use crate::compile::MechanismSlot;
    use crate::mechanism::{
        NoiseInferenceMode, evaluate_column, infer_noise_column_rng, log_prob_column,
    };
    use antecedent_core::CausalRng;

    /// The residual-path prior has `Var(r_0) = q² + r²` and
    /// `Cov(r_0, r_1) = a·q²` (`x_0 = μ + qε`, `x_1 = a x_0 + qε'`, independent obs noise).
    /// Draws are seeded; the tolerance is ~6 Monte-Carlo standard errors.
    #[test]
    fn sampled_residual_paths_have_the_prior_variance_and_lag_covariance() {
        let (a, q, r) = (0.7, 0.3, 0.2);
        let mut rng = CausalRng::from_seed(287);
        let draws = 20_000;
        let mut var0 = 0.0;
        let mut cov01 = 0.0;
        for _ in 0..draws {
            let mut path = [0.0; 2];
            sample_lgssm_noise(2, a, q, r, 0.0, &mut rng, &mut path).unwrap();
            var0 += path[0] * path[0];
            cov01 += path[0] * path[1];
        }
        let n = f64::from(draws);
        assert!((var0 / n - (q * q + r * r)).abs() < 0.006, "var0={}", var0 / n);
        assert!((cov01 / n - a * q * q).abs() < 0.006, "cov01={}", cov01 / n);
    }

    /// Abduction is exact inversion in full `f64` precision: replaying the
    /// abduced residual with unchanged parents reproduces the observations to
    /// rounding error, and shifting a parent moves the outcome by exactly its
    /// coefficient times the shift.
    #[test]
    fn abduction_is_lossless_and_reports_invertible_noise() {
        let slot = MechanismSlot::ConditionalLinearGaussianStateSpace {
            intercept: 4.0,
            coeffs: std::sync::Arc::from([2.0]),
            a: 0.999,
            process_std: 0.3,
            obs_std: 0.2,
            initial_mean: 0.1,
        };
        let n = 400;
        let parents: Vec<f64> = (0..n).map(|t| f64::from(t as u32).sin()).collect();
        let batch = ParentBatch { n_rows: n, n_parents: 1, values: &parents };
        let mut rng = CausalRng::from_seed(7);
        let mut noise = vec![0.0; n];
        crate::mechanism::sample_noise_column(&slot, n, &mut rng, &mut noise).unwrap();
        let mut ws = MechanismWorkspace::default();
        let mut y = vec![0.0; n];
        evaluate_column(&slot, batch, &noise, &mut y, &mut ws).unwrap();

        let mut inferred = vec![0.0; n];
        let mode =
            infer_noise_column_rng(&slot, &y, batch, &mut inferred, &mut CausalRng::from_seed(1))
                .unwrap();
        assert_eq!(mode, NoiseInferenceMode::Invertible);
        let mut replay = vec![0.0; n];
        evaluate_column(&slot, batch, &inferred, &mut replay, &mut ws).unwrap();
        for t in 0..n {
            assert!((replay[t] - y[t]).abs() < 1e-12, "t={t}: {} vs {}", replay[t], y[t]);
        }
        let shifted: Vec<f64> = parents.iter().map(|p| p + 1.0).collect();
        let mut cf = vec![0.0; n];
        evaluate_column(
            &slot,
            ParentBatch { n_rows: n, n_parents: 1, values: &shifted },
            &inferred,
            &mut cf,
            &mut ws,
        )
        .unwrap();
        for t in 0..n {
            assert!((cf[t] - y[t] - 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn likelihood_respects_changes_of_units() {
        let parents = ParentBatch { values: &[], n_rows: 2, n_parents: 0 };
        let mut base_logp = [0.0; 2];
        log_prob_column(
            &MechanismSlot::LinearGaussianStateSpace {
                a: 0.5,
                process_std: 2.0,
                obs_std: 1.0,
                initial_mean: 0.0,
            },
            &[1.0, 1.0],
            parents,
            &mut base_logp,
        )
        .unwrap();
        for scale in [1e-150, 1e-12, 1e12, 1e150] {
            let mut logp = [0.0; 2];
            log_prob_column(
                &MechanismSlot::LinearGaussianStateSpace {
                    a: 0.5,
                    process_std: 2.0 * scale,
                    obs_std: scale,
                    initial_mean: 0.0,
                },
                &[scale, scale],
                parents,
                &mut logp,
            )
            .unwrap();
            for t in 0..2 {
                assert!((logp[t] + scale.ln() - base_logp[t]).abs() < 1e-12);
            }
        }
    }
}
