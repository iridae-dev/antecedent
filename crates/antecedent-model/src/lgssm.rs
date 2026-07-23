//! Scalar linear-Gaussian state-space helpers (Kalman / RTS / innovation packing).
//!
//! Model: `x_t = a x_{t-1} + σ_proc ε_t`, `y_t = x_t + σ_obs η_t` with `ε, η ~ N(0,1)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;

use crate::error::ModelError;

/// Pack unit-normal process and observation innovations into one `f64` (f32×2).
#[must_use]
pub fn pack_innovations(process_eps: f64, obs_eta: f64) -> f64 {
    let bits =
        u64::from((process_eps as f32).to_bits()) | (u64::from((obs_eta as f32).to_bits()) << 32);
    f64::from_bits(bits)
}

/// Unpack innovations packed by [`pack_innovations`].
#[must_use]
pub fn unpack_innovations(packed: f64) -> (f64, f64) {
    let bits = packed.to_bits();
    let eps = f64::from(f32::from_bits(bits as u32));
    let eta = f64::from(f32::from_bits((bits >> 32) as u32));
    (eps, eta)
}

/// Sample packed LGSSM innovations into `output`.
pub fn sample_lgssm_noise(
    n_rows: usize,
    rng: &mut CausalRng,
    output: &mut [f64],
) -> Result<(), ModelError> {
    if output.len() < n_rows {
        return Err(ModelError::Shape { message: "lgssm noise output too short".into() });
    }
    for i in 0..n_rows {
        output[i] = pack_innovations(standard_normal(rng), standard_normal(rng));
    }
    Ok(())
}

/// Forward Kalman filter for a scalar LGSSM.
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
        let k = if s > 1e-18 { pp / s } else { 0.0 };
        x = xp + k * (y[t] - xp);
        p = (1.0 - k) * pp;
        x_f[t] = x;
        p_f[t] = p.max(0.0);
    }
    (x_f, p_f, x_pred, p_pred)
}

/// Rauch–Tung–Striebel smoother given filter outputs.
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
        let pp = p_pred[t + 1].max(1e-18);
        let j = p_f[t] * a / pp;
        x_s[t] = x_f[t] + j * (x_s[t + 1] - x_pred[t + 1]);
        p_s[t] = p_f[t] + j * j * (p_s[t + 1] - p_pred[t + 1]);
        p_lag[t + 1] = j * p_s[t + 1];
    }
    (x_s, p_s, p_lag)
}

/// Abduce packed innovations from observations via RTS smoother + residual reconstruction.
///
/// Samples latents from smoother marginals when `rng` is provided (posterior draws);
/// otherwise uses smoothed means (still marked posterior because the map is many-to-one).
pub fn infer_lgssm_innovations(
    y: &[f64],
    a: f64,
    process_std: f64,
    obs_std: f64,
    initial_mean: f64,
    output: &mut [f64],
    rng: Option<&mut CausalRng>,
) -> Result<(), ModelError> {
    let n = y.len();
    if output.len() < n {
        return Err(ModelError::Shape { message: "lgssm infer output too short".into() });
    }
    if n == 0 {
        return Ok(());
    }
    let q = (process_std * process_std).max(1e-16);
    let r = (obs_std * obs_std).max(1e-16);
    let (x_f, p_f, x_pred, p_pred) = kalman_filter(y, a, q, r, initial_mean, 1.0);
    let (x_s, p_s, _) = rts_smooth(a, &x_f, &p_f, &x_pred, &p_pred);

    let mut x_draw = x_s.clone();
    if let Some(rng) = rng {
        for t in 0..n {
            let s = p_s[t].max(0.0).sqrt();
            x_draw[t] = x_s[t] + s * standard_normal(rng);
        }
    }

    let inv_proc = 1.0 / process_std.max(1e-12);
    let inv_obs = 1.0 / obs_std.max(1e-12);
    for t in 0..n {
        let eps = if t == 0 {
            (x_draw[0] - initial_mean) * inv_proc
        } else {
            (x_draw[t] - a * x_draw[t - 1]) * inv_proc
        };
        let eta = (y[t] - x_draw[t]) * inv_obs;
        output[t] = pack_innovations(eps, eta);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::CausalRng;

    #[test]
    fn pack_unpack_round_trip() {
        let p = pack_innovations(0.5, -1.25);
        let (a, b) = unpack_innovations(p);
        assert!((a - 0.5).abs() < 1e-5);
        assert!((b - (-1.25)).abs() < 1e-5);
    }

    #[test]
    fn generative_abduction_recovers_observations() {
        let a = 0.7;
        let process_std = 0.3;
        let obs_std = 0.2;
        let initial_mean = 0.0;
        let mut rng = CausalRng::from_seed(7);
        let n = 32;
        let mut noise = vec![0.0; n];
        sample_lgssm_noise(n, &mut rng, &mut noise).unwrap();

        let mut y = vec![0.0; n];
        let mut x = initial_mean;
        for t in 0..n {
            let (eps, eta) = unpack_innovations(noise[t]);
            x = if t == 0 { initial_mean + process_std * eps } else { a * x + process_std * eps };
            y[t] = x + obs_std * eta;
        }

        let mut inferred = vec![0.0; n];
        infer_lgssm_innovations(&y, a, process_std, obs_std, initial_mean, &mut inferred, None)
            .unwrap();

        let mut x2 = initial_mean;
        for t in 0..n {
            let (eps, eta) = unpack_innovations(inferred[t]);
            x2 = if t == 0 { initial_mean + process_std * eps } else { a * x2 + process_std * eps };
            let yhat = x2 + obs_std * eta;
            assert!((yhat - y[t]).abs() < 1e-4, "t={t}: yhat={yhat} y={}", y[t]);
        }
    }
}
