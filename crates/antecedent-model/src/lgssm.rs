//! Scalar linear-Gaussian state-space helpers (Kalman 1960 filter / Rauch–Tung–Striebel 1965
//! smoother / innovation packing).
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

/// Abduce packed innovations from observations via Kalman conditioning.
///
/// With an RNG, forward filtering/backward sampling draws a joint latent path
/// conditional on all observations, retaining cross-time posterior covariance.
/// Without an RNG, uses RTS smoothed means (a conditional mean reconstruction,
/// not a posterior draw). Innovations retain the existing f32 packing precision.
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
    let scaled = scaled_lgssm(y, a, process_std, obs_std, initial_mean)?;
    let q = scaled.process_var;
    let r = scaled.obs_var;
    // x_0 has variance q, matching the generative mechanism.
    let (x_f, p_f, x_pred, p_pred) = kalman_filter(&scaled.values, a, q, r, scaled.initial_mean, q);
    let x_draw = if let Some(rng) = rng {
        let mut path = vec![0.0; n];
        path[n - 1] = x_f[n - 1] + p_f[n - 1].max(0.0).sqrt() * standard_normal(rng);
        for t in (0..n - 1).rev() {
            // p(x_t | x_{t+1}, y_{0:t}); the future is conditionally independent
            // of x_t given x_{t+1}. Independently sampling smoothed marginals
            // would erase the lag covariance and corrupt process innovations.
            let gain = if p_pred[t + 1] > 0.0 { p_f[t] * a / p_pred[t + 1] } else { 0.0 };
            let mean = x_f[t] + gain * (path[t + 1] - x_pred[t + 1]);
            let variance = if p_pred[t + 1] > 0.0 { p_f[t] * (q / p_pred[t + 1]) } else { 0.0 };
            path[t] = mean + variance.max(0.0).sqrt() * standard_normal(rng);
        }
        path
    } else {
        rts_smooth(a, &x_f, &p_f, &x_pred, &p_pred).0
    };
    for t in 0..n {
        let eps = if t == 0 {
            (x_draw[0] - scaled.initial_mean) / (process_std / scaled.scale)
        } else {
            (x_draw[t] - a * x_draw[t - 1]) / (process_std / scaled.scale)
        };
        let eta = (scaled.values[t] - x_draw[t]) / (obs_std / scaled.scale);
        output[t] = pack_innovations(eps, eta);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::CausalRng;

    #[test]
    fn sampled_paths_preserve_joint_smoothing_covariance() {
        // Prior covariance for a=1, q=1 is [[1,1],[1,2]]. With r=1,
        // posterior covariance = (prior^-1 + I)^-1 = [[2,1],[1,3]] / 5.
        let mut rng = CausalRng::from_seed(287);
        let mut product = 0.0;
        let mut first_square = 0.0;
        let mut second_square = 0.0;
        let draws = 40_000;
        for _ in 0..draws {
            let mut packed = [0.0; 2];
            infer_lgssm_innovations(&[0.0, 0.0], 1.0, 1.0, 1.0, 0.0, &mut packed, Some(&mut rng))
                .unwrap();
            let x0 = unpack_innovations(packed[0]).0;
            let x1 = x0 + unpack_innovations(packed[1]).0;
            product += x0 * x1;
            first_square += x0 * x0;
            second_square += x1 * x1;
        }
        assert!((product / f64::from(draws) - 0.2).abs() < 0.015);
        assert!((first_square / f64::from(draws) - 0.4).abs() < 0.015);
        assert!((second_square / f64::from(draws) - 0.6).abs() < 0.015);
    }

    #[test]
    fn innovations_and_likelihood_respect_changes_of_units() {
        use crate::{MechanismSlot, ParentBatch, log_prob_column};
        let mut reference = [0.0; 2];
        infer_lgssm_innovations(&[1.0, 1.0], 0.5, 2.0, 1.0, 0.0, &mut reference, None).unwrap();
        let mut base_logp = [0.0; 2];
        let parents = ParentBatch { values: &[], n_rows: 2, n_parents: 0 };
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
            let mut packed = [0.0; 2];
            infer_lgssm_innovations(
                &[scale, scale],
                0.5,
                2.0 * scale,
                scale,
                0.0,
                &mut packed,
                None,
            )
            .unwrap();
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
                let (eps, eta) = unpack_innovations(packed[t]);
                let (base_eps, base_eta) = unpack_innovations(reference[t]);
                assert!((eps - base_eps).abs() < 1e-6);
                assert!((eta - base_eta).abs() < 1e-6);
                assert!((logp[t] + scale.ln() - base_logp[t]).abs() < 1e-12);
            }
        }
    }

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

    /// MM-A3: `infer_lgssm_innovations` must seed the Kalman filter's initial state variance
    /// from `process_std²` (matching the generative model `x_0 = initial_mean + process_std *
    /// eps`, i.e. `Var(x_0) = process_std²`), not a hardcoded `1.0`.
    ///
    /// Expected `eps`/`eta` below are hand-derived (exact rationals) from the Kalman
    /// filter/RTS-smoother recursion for `a=0.5, q=process_std²=4.0, r=obs_std²=1.0,
    /// initial_mean=0.0, y=[1.0, 1.0]` with `p0 = q = 4.0`:
    ///   `x_pred`=[0, 2/5], `p_pred`=[4, 21/5], `x_f`=[4/5, 23/26], `p_f`=[4/5, 21/26]
    ///   `x_s`=[11/13, 23/26]  ⇒  eps0=11/26, eps1=3/13, eta0=2/13, eta1=3/26
    ///
    /// The pre-fix code hardcoded `p0=1.0`, which gives a materially different `x_s[0]` (and
    /// hence `eps0 ≈ 0.268293`, not `11/26 ≈ 0.423077`) — a self-consistency round trip
    /// (reconstructing `y` from the inferred innovations) cannot distinguish the two, so this
    /// test compares against the independently-computed rationals instead.
    #[test]
    fn infer_lgssm_innovations_seeds_initial_variance_from_process_std() {
        let a = 0.5;
        let process_std = 2.0; // ≠ 1.0, so the old hardcoded p0 would be wrong.
        let obs_std = 1.0;
        let initial_mean = 0.0;
        let y = [1.0_f64, 1.0];
        let mut inferred = [0.0; 2];
        infer_lgssm_innovations(&y, a, process_std, obs_std, initial_mean, &mut inferred, None)
            .unwrap();
        let (eps0, eta0) = unpack_innovations(inferred[0]);
        let (eps1, eta1) = unpack_innovations(inferred[1]);
        let expected_eps0 = 11.0 / 26.0;
        let expected_eps1 = 3.0 / 13.0;
        let expected_eta0 = 2.0 / 13.0;
        let expected_eta1 = 3.0 / 26.0;
        assert!((eps0 - expected_eps0).abs() < 1e-4, "eps0={eps0} expected={expected_eps0}");
        assert!((eps1 - expected_eps1).abs() < 1e-4, "eps1={eps1} expected={expected_eps1}");
        assert!((eta0 - expected_eta0).abs() < 1e-4, "eta0={eta0} expected={expected_eta0}");
        assert!((eta1 - expected_eta1).abs() < 1e-4, "eta1={eta1} expected={expected_eta1}");
    }
}
