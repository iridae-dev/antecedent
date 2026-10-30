//! Split out of the mechanism registry. Behavior is unchanged.
use super::{data_rss, fit_ar1, fit_linear_gaussian, residual_sigma};

use std::sync::Arc;

use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use super::super::cv::FitFrame;
use crate::basis::column_moments;
use crate::compile::MechanismSlot;
use crate::error::ModelError;

/// Minnesota-style single-equation Bayesian regression: a zero-mean normal prior
/// on each parent's coefficient, with the intercept unpenalized.
///
/// The prior is placed on the coefficient *per standard deviation of the parent*
/// and scales with the error variance (`b'ⱼ ~ N(0, φ σ²)`, the conjugate
/// normal–inverse-gamma form), so the posterior mode is a ridge fit on
/// scale-standardized columns with pseudo-row weight `1/√φ`. That makes the fit
/// equivariant to a change of units of the outcome and of every parent, which the
/// unstandardized pseudo-rows were not. The Minnesota lag decay `φ/ℓ²` is not
/// applied: the gather plan carries no lag metadata, a parent's position in the
/// list is not a lag, and letting it set the shrinkage made the fit depend on the
/// order the parents happened to be listed in.
pub(super) fn fit_bvar_minnesota(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = frame.n_parents();
    let ncols = 1 + p;
    let phi: f64 = 0.2; // overall tightness
    let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
    let (_, scales) = column_moments(&refs, n);
    let extra = p;
    let rows = n + extra;
    let mut x2 = vec![0.0; rows * ncols];
    let mut y2 = vec![0.0; rows];
    x2[..n].fill(1.0);
    for (j, col) in refs.iter().enumerate() {
        for r in 0..n {
            x2[(1 + j) * rows + r] = col[r] / scales[j];
        }
        x2[(1 + j) * rows + n + j] = (1.0 / phi).sqrt();
    }
    y2[..n].copy_from_slice(y);
    let fit = backend.least_squares(&x2, rows, ncols, &y2, ls_ws).map_err(ModelError::from)?;
    let intercept = fit.coefficients[0];
    let coeffs: Vec<f64> =
        fit.coefficients[1..].iter().zip(scales.iter()).map(|(b, s)| b / s).collect();
    // σ from the data rows: the augmented RSS also holds Σ b'²/φ from the prior rows.
    let rss = data_rss(frame, y, intercept, &coeffs, None);
    let sigma = residual_sigma(rss, n, ncols, y);
    Ok(MechanismSlot::Bvar { intercept, coeffs: Arc::from(coeffs), sigma })
}

/// Scalar LGSSM on parent-adjusted residuals via EM (Kalman 1960 filter / Rauch–Tung–Striebel
/// 1965 smoother).
pub(super) fn fit_lgssm_kalman_em(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    if y.len() < 3 {
        return Err(ModelError::Unsupported {
            message: "state-space fitting requires at least three observations to estimate its dynamics and noise".into(),
        });
    }
    let lg = fit_linear_gaussian(frame, y, backend, ls_ws, 0.0)?;
    let MechanismSlot::LinearGaussian { intercept, coeffs, .. } = lg else {
        return Err(ModelError::Unsupported { message: "lgssm fit requires linear base".into() });
    };
    let mut resid = vec![0.0; y.len()];
    for r in 0..y.len() {
        let mut pred = intercept;
        for (p, col) in frame.parents.iter().enumerate() {
            pred += coeffs[p] * col[r];
        }
        resid[r] = y[r] - pred;
    }
    let (a, process_std, obs_std, initial_mean) = lgssm_em(&resid, 25);
    Ok(MechanismSlot::ConditionalLinearGaussianStateSpace {
        intercept,
        coeffs,
        a,
        process_std,
        obs_std,
        initial_mean,
    })
}

/// EM for scalar LGSSM: `x_t` = a x_{t-1} + q ε, `y_t` = `x_t` + r η.
/// The internal short-series fallback is heuristic; the production fitter refuses it.
pub(in crate::registry) fn lgssm_em(y: &[f64], max_iters: usize) -> (f64, f64, f64, f64) {
    // Fit in relative units so variance floors and initialization do not change
    // the model when the outcome's measurement units change.
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let normalized: Vec<f64> = y.iter().map(|v| v / scale).collect();
    let (a, process_std, obs_std, initial_mean) = lgssm_em_normalized(&normalized, max_iters);
    (a, process_std * scale, obs_std * scale, initial_mean * scale)
}

/// EM fit of a local-level state-space model on a normalized series.
pub(super) fn lgssm_em_normalized(y: &[f64], max_iters: usize) -> (f64, f64, f64, f64) {
    let n = y.len();
    if n < 3 {
        let (a, q) = fit_ar1(y);
        return (a, q, q.max(1e-8), y.first().copied().unwrap_or(0.0));
    }
    let mut a = 0.8;
    let mut q = 1.0; // process variance
    let mut r = 1.0; // obs variance
    let mut x0 = y[0];
    for _ in 0..max_iters {
        let (x_f, p_f, x_pred, p_pred) = crate::lgssm::kalman_filter(y, a, q, r, x0, q);
        let (x_s, p_s, p_lag) = crate::lgssm::rts_smooth(a, &x_f, &p_f, &x_pred, &p_pred);
        // M-step
        let mut num = 0.0;
        let mut den = 0.0;
        for t in 1..n {
            num += p_lag[t] + x_s[t] * x_s[t - 1];
            den += p_s[t - 1] + x_s[t - 1] * x_s[t - 1];
        }
        a = if den > 1e-12 { (num / den).clamp(-0.999, 0.999) } else { a };
        // Initial state variance is q in the emitted model as well. Its
        // expected squared residual contributes one of the n process terms.
        x0 = x_s[0];
        let mut q_acc = p_s[0];
        for t in 1..n {
            q_acc += p_s[t] + x_s[t] * x_s[t] + a * a * (p_s[t - 1] + x_s[t - 1] * x_s[t - 1])
                - 2.0 * a * (p_lag[t] + x_s[t] * x_s[t - 1]);
        }
        q = (q_acc / n as f64).max(1e-8);
        let mut r_acc = 0.0;
        for t in 0..n {
            r_acc += p_s[t] + (y[t] - x_s[t]).powi(2);
        }
        r = (r_acc / n as f64).max(1e-8);
    }
    (a, q.sqrt(), r.sqrt(), x0)
}

#[cfg(test)]
mod short_fit_tests {
    use super::*;
    use crate::registry::ValidationSplits;

    #[test]
    fn lgssm_fitter_refuses_short_unidentified_series() {
        for n in [1, 2] {
            let frame =
                FitFrame { parents: vec![], groups: None, n, splits: ValidationSplits::Shuffled };
            let y = vec![1.0; n];
            let result =
                fit_lgssm_kalman_em(&frame, &y, FaerBackend, &mut LeastSquaresWorkspace::default());
            assert!(matches!(result, Err(ModelError::Unsupported { .. })));
        }
        let frame =
            FitFrame { parents: vec![], groups: None, n: 3, splits: ValidationSplits::Shuffled };
        let result = fit_lgssm_kalman_em(
            &frame,
            &[0.0, 1.0, 0.0],
            FaerBackend,
            &mut LeastSquaresWorkspace::default(),
        );
        assert!(matches!(result, Ok(MechanismSlot::ConditionalLinearGaussianStateSpace { .. })));
    }
}
