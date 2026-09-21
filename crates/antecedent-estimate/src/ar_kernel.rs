//! Autoregressive models of a score series and the Bartlett kernel bias they imply.
//!
//! The one owner of the autoregressive family every temporal dispersion correction
//! reads: the Kendall-corrected AR(1) lag-1 coefficient, the BIC-selected Yule–Walker
//! AR(q ≤ 4), the model autocorrelations, and the share of the model's long-run variance
//! that a Bartlett kernel keeps. The Bayesian long-run tempering
//! ([`crate::serial_dependence`]) prewhitens with the same fits; the frequentist
//! circular-block SEs (scalar Pulse / Sustained / mediation and the response bands) take
//! [`kernel_bias_factor`] from here, so one score gets one correction wherever it is
//! read.
//!
//! The circular-block bootstrap variance of a level is (asymptotically) the Bartlett
//! long-run variance of that level's influence series at bandwidth `ℓ`, the block
//! length. The Bartlett kernel misses `O(1/ℓ)` of a persistent influence's long-run
//! variance, a bias the fixed-b critical value does not repair: for an AR(1) ρ = 0.9
//! influence at `ℓ = 13` the kernel keeps 46% of the long-run variance. Lengthening
//! the block does not close the gap either — at `ℓ = n/3` the kernel keeps 82% but
//! the fixed-b interval was measured at 0.930 coverage of nominal 0.95 (n = 160) against
//! 0.954 at `ℓ = ⌈√n⌉` with the correction below — and it degrades the sup-t band,
//! which needs many blocks per replicate.
//!
//! [`kernel_bias_factor`] fits the influence with a Kendall-corrected AR(1) and a
//! BIC-selected AR(q ≤ 4), reads the fraction of the fitted model's long-run variance
//! the Bartlett kernel at `ℓ` keeps, and returns `1/√fraction` (never below 1). A weakly
//! dependent influence keeps a factor within 1% of 1; the factor is model-based, so it
//! does not carry the sampling noise of a wide-bandwidth long-run-variance estimate.
//! What the model does not see, the factor does not correct: a persistent component that
//! is a small share of the influence (an AR(1) ρ = 0.9 residual under omitted iid
//! treatment lags) is invisible to the fit at short `n`; the short-series warning, not
//! this factor, says the series is too short for the dependence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

/// Largest absolute autoregressive coefficient a fit reports (Andrews & Monahan 1992 cap
/// the AR(1) coefficient the same way, so a near-unit-root score cannot make the model
/// long-run variance unbounded).
pub(crate) const MAX_AR_RHO: f64 = 0.97;

/// Largest autoregressive order the BIC search considers.
pub(crate) const MAX_AR_ORDER: usize = 4;

/// Fewest rows on which a score model is fitted; shorter series keep factor 1.
const MIN_ROWS: usize = 8;

/// Longest autocorrelation tail summed for the model long-run variance.
const MAX_ACF_LAGS: usize = 20_000;

/// Kernel-bias factor of one influence series at block length `block`:
/// `1/√f`, `f` the fraction of the fitted autoregression's long-run variance that the
/// Bartlett kernel at `block` keeps, `f = (1 + 2 Σ_{k<ℓ} (1 − k/ℓ) ρ_k) / (1 + 2 Σ_k ρ_k)`
/// with `ρ_k` the model autocorrelation. Two fits are read and the larger factor kept:
/// a Kendall-corrected AR(1) (`ρ̂ + (1 + 3ρ̂)/n`, capped at 0.97) and, when BIC selects
/// an order of at least 2 (Yule–Walker, `q ≤ 4`), that AR(q). A model with no positive
/// long-run excess, a constant or non-finite series, or fewer than 8 rows gives `1`.
#[must_use]
pub fn kernel_bias_factor(influence: &[f64], block: usize) -> f64 {
    let n = influence.len();
    if n < MIN_ROWS || block == 0 || influence.iter().any(|v| !v.is_finite()) {
        return 1.0;
    }
    let mean = influence.iter().sum::<f64>() / n as f64;
    let s: Vec<f64> = influence.iter().map(|v| v - mean).collect();
    let gamma0 = dot(&s, &s) / n as f64;
    if !gamma0.is_finite() || gamma0 <= 0.0 {
        return 1.0;
    }
    let mut fraction = 1.0_f64;
    let rho1 = kendall_rho(&s);
    if rho1 > 0.0 {
        fraction = fraction.min(bartlett_capture(&ar1_acf(rho1, block), block));
    }
    let model = bic_autoregression(&s);
    if model.phi.len() >= 2 {
        fraction = fraction.min(bartlett_capture(&model.acf(block), block));
    }
    if fraction.is_finite() && fraction > 0.0 && fraction < 1.0 {
        fraction.sqrt().recip()
    } else {
        1.0
    }
}

/// Fraction of the long-run variance `1 + 2 Σ_k ρ_k` (over the whole `acf`, `acf[0] = 1`)
/// that the Bartlett kernel at `block` keeps; `1` when the model carries no positive
/// long-run excess.
pub(crate) fn bartlett_capture(acf: &[f64], block: usize) -> f64 {
    let full = 1.0 + 2.0 * acf.iter().skip(1).sum::<f64>();
    let kept = 1.0
        + 2.0
            * acf
                .iter()
                .enumerate()
                .skip(1)
                .take(block.saturating_sub(1))
                .map(|(k, rho)| (1.0 - k as f64 / block as f64) * rho)
                .sum::<f64>();
    if !full.is_finite() || !kept.is_finite() || full <= 1.0 || kept <= 0.0 {
        1.0
    } else {
        (kept / full).min(1.0)
    }
}

/// Model autocorrelations `ρ_0 = 1, ρ_1, …` of an AR(1) with coefficient `rho`, summed
/// until the tail is negligible (at least `block` lags).
pub(crate) fn ar1_acf(rho: f64, block: usize) -> Vec<f64> {
    let mut acf = vec![1.0];
    let mut value = 1.0;
    for k in 1..MAX_ACF_LAGS {
        value *= rho;
        if k > block && value.abs() < 1e-12 {
            break;
        }
        acf.push(value);
    }
    acf
}

/// Bias-corrected lag-1 autocorrelation of a mean-zero series (Kendall 1954:
/// `E[ρ̂] ≈ ρ − (1 + 3ρ)/n`), clamped to `±MAX_AR_RHO`. The uncentred lag-1 moment is
/// the convention for OLS residuals and scores, which have mean zero.
pub(crate) fn kendall_rho(s: &[f64]) -> f64 {
    let (num, den) = if s.len() < 2 {
        (0.0, 0.0)
    } else {
        let lead = &s[..s.len() - 1];
        (dot(&s[1..], lead), dot(lead, lead))
    };
    let rho_hat = if den > 0.0 { num / den } else { 0.0 };
    (rho_hat + (1.0 + 3.0 * rho_hat) / s.len() as f64).clamp(-MAX_AR_RHO, MAX_AR_RHO)
}

/// Yule–Walker autoregression of a mean-zero series at a BIC-selected order.
#[derive(Clone, Debug, Default)]
pub(crate) struct Autoregression {
    /// Coefficients `φ₁ … φ_q` (empty for `q = 0`).
    pub(crate) phi: Vec<f64>,
    /// Sample autocorrelations `r_0 = 1, r_1 … r_q` (the model's own up to lag `q`).
    autocorrelation: Vec<f64>,
}

impl Autoregression {
    /// Model autocorrelations extended by the AR recursion until the tail is negligible
    /// (at least `block` lags).
    pub(crate) fn acf(&self, block: usize) -> Vec<f64> {
        let q = self.phi.len();
        let mut rho = self.autocorrelation.clone();
        for k in (q + 1)..MAX_ACF_LAGS {
            let next: f64 = self.phi.iter().enumerate().map(|(j, f)| f * rho[k - 1 - j]).sum();
            if !next.is_finite() {
                break;
            }
            rho.push(next);
            if k > block && rho[k + 1 - q..=k].iter().all(|r| r.abs() < 1e-12) {
                break;
            }
        }
        rho
    }
}

/// Levinson–Durbin fit of a mean-zero series `s` (uncentred autocovariances, the
/// convention of [`kendall_rho`]) with `q ≤ MAX_AR_ORDER` minimizing
/// `n ln σ̂²_q + q ln n`. The Toeplitz autocovariance is positive definite, so every
/// fitted model is stationary.
pub(crate) fn bic_autoregression(s: &[f64]) -> Autoregression {
    let n = s.len();
    let max_order = MAX_AR_ORDER.min(n.saturating_sub(2));
    let gamma: Vec<f64> = (0..=max_order).map(|k| dot(&s[k..], s) / n as f64).collect();
    if gamma[0] <= 0.0 || !gamma[0].is_finite() {
        return Autoregression::default();
    }
    let nf = n as f64;
    let mut phi: Vec<f64> = Vec::new();
    let mut variance = gamma[0];
    let (mut best_bic, mut best) = (nf * variance.ln(), Vec::new());
    for order in 1..=max_order {
        let acc = gamma[order]
            - phi.iter().enumerate().map(|(j, f)| f * gamma[order - 1 - j]).sum::<f64>();
        let reflection = acc / variance;
        if !reflection.is_finite() || reflection.abs() >= 1.0 {
            break;
        }
        let mut next: Vec<f64> =
            (0..order - 1).map(|j| phi[j] - reflection * phi[order - 2 - j]).collect();
        next.push(reflection);
        phi = next;
        variance *= 1.0 - reflection * reflection;
        if variance <= 0.0 || variance.is_nan() {
            break;
        }
        let bic = nf * variance.ln() + order as f64 * nf.ln();
        if bic < best_bic {
            best_bic = bic;
            best.clone_from(&phi);
        }
    }
    let autocorrelation = gamma[..=best.len()].iter().map(|g| g / gamma[0]).collect();
    Autoregression { phi: best, autocorrelation }
}

/// Dot product over the common length of `a` and `b` with four independent
/// accumulators (the score HAC passes are otherwise latency-bound on one
/// floating-point chain).
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0_f64; 4];
    let (head_a, tail_a) = a.split_at(n - n % 4);
    let (head_b, tail_b) = b.split_at(n - n % 4);
    for (ca, cb) in head_a.chunks_exact(4).zip(head_b.chunks_exact(4)) {
        for k in 0..4 {
            acc[k] += ca[k] * cb[k];
        }
    }
    let tail: f64 = tail_a.iter().zip(tail_b).map(|(u, v)| u * v).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic AR(1) series with Gaussian innovations (Box–Muller on splitmix64).
    fn gaussian_series(n: usize, rho: f64, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n);
        let mut previous = 0.0;
        for _ in 0..n {
            let (u, v) = (next().max(1e-12), next());
            let innovation = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            previous = rho * previous + innovation;
            out.push(previous);
        }
        out
    }

    #[test]
    fn bartlett_capture_matches_the_ar1_closed_form() {
        // AR(1) ρ: full ratio (1+ρ)/(1−ρ); kept 1 + 2 Σ_{k<ℓ} (1 − k/ℓ) ρ^k.
        let (rho, block) = (0.9_f64, 13_usize);
        let full = (1.0 + rho) / (1.0 - rho);
        let kept = 1.0
            + 2.0
                * (1..block)
                    .map(|k| (1.0 - k as f64 / block as f64) * rho.powf(k as f64))
                    .sum::<f64>();
        let acf = ar1_acf(rho, block);
        assert!((bartlett_capture(&acf, block) - kept / full).abs() < 1e-9);
        assert!((kept / full - 0.456).abs() < 0.01, "{}", kept / full);
        // No positive excess: the kernel keeps everything.
        assert!((bartlett_capture(&[1.0, -0.3, 0.09], 5) - 1.0).abs() < 1e-12);
        assert!((bartlett_capture(&[1.0], 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kernel_bias_factor_is_near_one_for_white_noise_and_large_for_persistence() {
        let white = gaussian_series(400, 0.0, 3);
        let f0 = kernel_bias_factor(&white, 20);
        assert!((1.0..1.03).contains(&f0), "white noise factor {f0}");
        let persistent = gaussian_series(400, 0.9, 5);
        let f9 = kernel_bias_factor(&persistent, 20);
        // True capture at ℓ = 20 is 0.58: factor about 1.3.
        assert!((1.15..1.6).contains(&f9), "persistent factor {f9}");
        assert!(kernel_bias_factor(&persistent, 400) < f9, "longer blocks keep more");
        // Negative dependence, constants, short and non-finite series keep 1.
        let alternating: Vec<f64> = (0..100).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((kernel_bias_factor(&alternating, 10) - 1.0).abs() < 1e-12);
        assert!((kernel_bias_factor(&[2.0; 50], 5) - 1.0).abs() < 1e-12);
        assert!((kernel_bias_factor(&persistent[..6], 2) - 1.0).abs() < 1e-12);
        assert!(
            (kernel_bias_factor(&[1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2) - 1.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn bic_autoregression_recovers_an_ar2_and_its_acf_follows_the_recursion() {
        let (a, b) = (0.3_f64, 0.5_f64);
        let mut state = 0.0;
        let mut prev = 0.0;
        let mut white = gaussian_series(4_500, 0.0, 11).into_iter();
        let series: Vec<f64> = (0..4_500)
            .map(|_| {
                let next = a * state + b * prev + white.next().unwrap();
                prev = state;
                state = next;
                next
            })
            .skip(500)
            .collect();
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let centered: Vec<f64> = series.iter().map(|v| v - mean).collect();
        let fit = bic_autoregression(&centered);
        assert_eq!(fit.phi.len(), 2, "{fit:?}");
        assert!((fit.phi[0] - a).abs() < 0.06 && (fit.phi[1] - b).abs() < 0.06, "{fit:?}");
        let acf = fit.acf(10);
        for k in 3..acf.len() {
            let recursion = fit.phi[0] * acf[k - 1] + fit.phi[1] * acf[k - 2];
            assert!((acf[k] - recursion).abs() < 1e-12);
        }
        // The AR(2) model sees dependence the AR(1) fit under-reads: its factor is larger.
        let ar1_only = bartlett_capture(&ar1_acf(kendall_rho(&centered), 20), 20);
        let ar2 = bartlett_capture(&fit.acf(20), 20);
        assert!(ar2 < ar1_only, "AR(2) capture {ar2} vs AR(1) {ar1_only}");
        assert!(kernel_bias_factor(&centered, 20) >= ar2.sqrt().recip() - 1e-12);
    }

    /// The factor of a known AR(1) is the closed form `1/√f` with `f` the kernel's
    /// share of `(1 + ρ)/(1 − ρ)`; a long exact-AR(1) sample recovers it.
    #[test]
    fn kernel_bias_factor_matches_the_ar1_closed_form_on_a_long_sample() {
        let (rho, block) = (0.8_f64, 10_usize);
        let series = gaussian_series(20_000, rho, 21);
        let full = (1.0 + rho) / (1.0 - rho);
        let kept = 1.0
            + 2.0
                * (1..block)
                    .map(|k| (1.0 - k as f64 / block as f64) * rho.powf(k as f64))
                    .sum::<f64>();
        let truth = (full / kept).sqrt();
        let factor = kernel_bias_factor(&series, block);
        assert!((factor - truth).abs() < 0.06 * truth, "factor {factor} vs closed form {truth}");
    }

    /// Kendall (1954): `ρ̂ + (1 + 3ρ̂)/n`. A geometric series with ratio 0.2 has raw
    /// lag-1 moment ratio exactly 0.2, so with `n = 20` the corrected coefficient is
    /// `0.2 + 1.6/20 = 0.28`; a near-unit-root reading is capped at 0.97.
    #[test]
    fn kendall_rho_applies_the_small_sample_correction_and_the_cap() {
        let geometric: Vec<f64> = (0..20).map(|t| 0.2_f64.powi(t)).collect();
        assert!((kendall_rho(&geometric) - 0.28).abs() < 1e-12);
        let ramp: Vec<f64> = (0..50).map(f64::from).collect();
        assert!((kendall_rho(&ramp) - MAX_AR_RHO).abs() < 1e-12);
    }
}
