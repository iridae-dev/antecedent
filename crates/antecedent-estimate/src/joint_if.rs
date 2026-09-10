//! Joint influence-function covariance for fits that share rows.
//!
//! One implementation serves per-arm, per-threshold, per-cell, and cross-claim
//! covariances, and frozen-weight mixture uncertainty for static multi-atom
//! Frequentist aggregates. Graph weights are treated as frozen modeling
//! choices: the reported SE is for the weighted aggregate, not a probability
//! distribution over completion-specific effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]
#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use std::sync::Arc;

use crate::error::EstimationError;

/// Column-major `k × k` covariance of `k` influence sequences of length `n`.
#[derive(Clone, Debug, PartialEq)]
pub struct JointCovariance {
    /// Number of functionals.
    pub dim: usize,
    /// Column-major covariance matrix, length `dim * dim`.
    pub values: Arc<[f64]>,
}

impl JointCovariance {
    /// Entry `(i, j)`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[j * self.dim + i]
    }

    /// Analytic SE for functional `i`.
    #[must_use]
    pub fn se(&self, i: usize) -> f64 {
        let variance = self.get(i, i);
        if variance.is_finite() && variance >= 0.0 { variance.sqrt() } else { f64::NAN }
    }
}

/// Weighted mean of a score column: `θ = Σ w_i φ_i / Σ w_i`.
///
/// # Errors
///
/// Empty input, length mismatch, or non-positive total weight.
pub fn weighted_mean(scores: &[f64], weights: Option<&[f64]>) -> Result<f64, EstimationError> {
    if scores.is_empty() {
        return Err(EstimationError::data_msg("joint IF requires at least one row"));
    }
    if scores.iter().any(|x| !x.is_finite()) {
        return Err(EstimationError::data_msg("scores must be finite"));
    }
    match weights {
        None => {
            let mean = scores.iter().map(|v| v / scores.len() as f64).sum::<f64>();
            if !mean.is_finite() {
                return Err(EstimationError::data_msg("score mean overflowed"));
            }
            Ok(mean)
        }
        Some(w) => {
            if w.len() != scores.len() {
                return Err(EstimationError::data_msg("weight length does not match scores"));
            }
            let mut num = 0.0;
            let mut den = 0.0;
            for (&phi, &wi) in scores.iter().zip(w) {
                if !wi.is_finite() || wi < 0.0 {
                    return Err(EstimationError::data_msg(
                        "target weights must be finite and non-negative",
                    ));
                }
                num += wi * phi;
                den += wi;
            }
            if !den.is_finite() || den <= 0.0 || !num.is_finite() {
                return Err(EstimationError::data_msg("target weights have no mass"));
            }
            Ok(num / den)
        }
    }
}

/// Kish effective sample size of non-negative weights.
#[must_use]
pub fn kish_n_eff(weights: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &w in weights {
        if w > 0.0 && w.is_finite() {
            sum += w;
            sum_sq += w * w;
        }
    }
    if sum_sq <= 0.0 { 0.0 } else { (sum * sum) / sum_sq }
}

/// Joint covariance of `k` score columns that share rows.
///
/// `scores[j]` is a length-`n` score sequence for functional `j`.
/// Scores may be uncentered; validity as an estimated influence function
/// requires the estimator’s nuisance and sampling assumptions. The estimator is the weighted mean; the IF of that
/// mean is `ψ_i = (n w_i / W) (φ_i − θ)` when weights are present, else
/// `φ_i − θ`. Homoskedastic SE is `sqrt(Σ ψ² / n²)` via the sample SD of `ψ`
/// over `sqrt(n)` with the same `n/(n-1)` correction for weighted and
/// unweighted calls, matching [`crate::se::influence_se_kind`] for one column.
///
/// # Errors
///
/// Empty family, mismatched lengths, or invalid weights.
pub fn joint_influence_covariance(
    scores: &[&[f64]],
    weights: Option<&[f64]>,
) -> Result<JointCovariance, EstimationError> {
    let dim = scores.len();
    if dim == 0 {
        return Err(EstimationError::data_msg("joint IF requires at least one functional"));
    }
    let n = scores[0].len();
    if n < 2 {
        return Err(EstimationError::data_msg("joint IF requires at least two rows"));
    }
    for col in scores {
        if col.len() != n {
            return Err(EstimationError::data_msg("score columns must share a row count"));
        }
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(EstimationError::data_msg("weight length does not match scores"));
        }
    }

    let mut means = Vec::with_capacity(dim);
    for col in scores {
        means.push(weighted_mean(col, weights)?);
    }

    let mut values = vec![0.0; dim * dim];
    let nf = n as f64;
    match weights {
        None => {
            for j in 0..dim {
                for i in 0..=j {
                    let mut acc = 0.0;
                    for r in 0..n {
                        acc += (scores[i][r] - means[i]) * (scores[j][r] - means[j]);
                    }
                    let cov = acc / (nf * (nf - 1.0));
                    values[j * dim + i] = cov;
                    values[i * dim + j] = cov;
                }
            }
        }
        Some(w) => {
            let w_sum: f64 = w.iter().sum();
            if w_sum <= 0.0 {
                return Err(EstimationError::data_msg("target weights have no mass"));
            }
            for j in 0..dim {
                for i in 0..=j {
                    let mut acc = 0.0;
                    for r in 0..n {
                        acc +=
                            (w[r] * (scores[i][r] - means[i])) * (w[r] * (scores[j][r] - means[j]));
                    }
                    let cov = (nf / (nf - 1.0)) * acc / (w_sum * w_sum);
                    values[j * dim + i] = cov;
                    values[i * dim + j] = cov;
                }
            }
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::data_msg("joint covariance overflowed"));
    }
    Ok(JointCovariance { dim, values: Arc::from(values) })
}

/// Frozen-weight mixture of per-atom score columns that share rows.
///
/// `atom_scores[g][i]` is atom `g`'s IF on row `i`. `atom_weights` are frozen
/// graph / completion weights (renormalized over contributing atoms). The
/// mixture IF is `ψ_i = Σ_g w_g φ_i^g`. Unidentified mass is not mixed into
/// `ψ`; the caller retains it on the envelope.
///
/// # Errors
///
/// Shape mismatch or non-positive contributing weight.
pub fn frozen_weight_mixture_scores(
    atom_scores: &[&[f64]],
    atom_weights: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    if atom_scores.len() != atom_weights.len() || atom_scores.is_empty() {
        return Err(EstimationError::data_msg("mixture scores and weights must align"));
    }
    let n = atom_scores[0].len();
    let mut mass = 0.0;
    for (scores, &w) in atom_scores.iter().zip(atom_weights) {
        if scores.len() != n {
            return Err(EstimationError::data_msg("mixture atoms must share a row count"));
        }
        if !w.is_finite() || w < 0.0 {
            return Err(EstimationError::data_msg(
                "mixture weights must be finite and non-negative",
            ));
        }
        mass += w;
    }
    if mass <= 0.0 {
        return Err(EstimationError::data_msg("mixture has no contributing mass"));
    }
    let mut out = vec![0.0; n];
    for (scores, &w) in atom_scores.iter().zip(atom_weights) {
        if w == 0.0 {
            continue;
        }
        let scale = w / mass;
        for i in 0..n {
            out[i] += scale * scores[i];
        }
    }
    Ok(out)
}

/// Max-t critical value for simultaneous bands over `k` functionals.
///
/// Uses a Gaussian multiplier draw: `c = quantile_{1-α}(max_j |Z_j|)` where
/// `Z ~ N(0, R)` and `R` is the correlation matrix of `cov`. `replicates`
/// multiplier draws, deterministic from `seed`.
///
/// # Errors
///
/// Degenerate covariance or non-finite level.
pub fn max_t_critical(
    cov: &JointCovariance,
    level: f64,
    replicates: u32,
    seed: u64,
) -> Result<f64, EstimationError> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 || replicates == 0 {
        return Err(EstimationError::unsupported(
            "max-t bands require level in (0, 1) and a positive replicate count",
        ));
    }
    let k = cov.dim;
    if k == 0
        || cov.values.len() != k.saturating_mul(k)
        || cov.values.iter().any(|v| !v.is_finite())
    {
        return Err(EstimationError::data_msg("invalid covariance shape or values"));
    }
    for i in 0..k {
        for j in 0..i {
            if (cov.get(i, j) - cov.get(j, i)).abs() > 1e-10 * (1.0 + cov.get(i, j).abs()) {
                return Err(EstimationError::data_msg("covariance must be symmetric"));
            }
        }
    }
    let mut corr = vec![0.0; k * k];
    let mut se = vec![0.0; k];
    for i in 0..k {
        se[i] = cov.se(i);
        if !(se[i] > 0.0 && se[i].is_finite()) {
            return Err(EstimationError::data_msg("max-t requires positive finite SEs"));
        }
    }
    for j in 0..k {
        for i in 0..k {
            corr[j * k + i] = cov.get(i, j) / (se[i] * se[j]);
        }
    }
    let chol = cholesky_corr(&corr, k)?;
    let mut rng = SplitMix64 { state: seed | 1 };
    let mut maxima = Vec::with_capacity(replicates as usize);
    let mut z = vec![0.0; k];
    for _ in 0..replicates {
        for zi in &mut z {
            *zi = standard_normal(&mut rng);
        }
        let mut max_abs: f64 = 0.0;
        for i in 0..k {
            let mut acc = 0.0;
            for j in 0..=i {
                acc += chol[i * k + j] * z[j];
            }
            max_abs = max_abs.max(acc.abs());
        }
        maxima.push(max_abs);
    }
    maxima.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx =
        ((level * f64::from(replicates)).ceil() as usize).saturating_sub(1).min(maxima.len() - 1);
    Ok(maxima[idx])
}

/// Equal-weight least-squares projection onto decreasing sequences using PAVA.
///
/// For exceedance `P(Y > c)` on an increasing `c` grid. Disclosed by callers.
#[must_use]
pub fn monotone_decreasing(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    // PAVA for decreasing: negate, isotone increasing, negate back.
    let mut y: Vec<f64> = values.iter().map(|v| -v).collect();
    pava_increasing(&mut y);
    y.iter().map(|v| -v).collect()
}

/// Isotone increasing rearrangement (PAVA).
#[must_use]
pub fn monotone_increasing(values: &[f64]) -> Vec<f64> {
    let mut y = values.to_vec();
    pava_increasing(&mut y);
    y
}

fn pava_increasing(y: &mut [f64]) {
    let n = y.len();
    if n < 2 {
        return;
    }
    let mut val = y.to_vec();
    let mut weight = vec![1.0; n];
    let mut len = n;
    let mut i = 0;
    while i + 1 < len {
        if val[i] <= val[i + 1] {
            i += 1;
            continue;
        }
        let w = weight[i] + weight[i + 1];
        let v = (val[i] * weight[i] + val[i + 1] * weight[i + 1]) / w;
        val[i] = v;
        weight[i] = w;
        val.remove(i + 1);
        weight.remove(i + 1);
        len -= 1;
        i = i.saturating_sub(1);
    }
    let mut out_i = 0;
    for (block, &w) in val.iter().zip(weight.iter()) {
        let count = w.round() as usize;
        for _ in 0..count {
            if out_i < y.len() {
                y[out_i] = *block;
                out_i += 1;
            }
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn standard_normal(rng: &mut SplitMix64) -> f64 {
    let u = (rng.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
    let v = (rng.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
    let u = u.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn cholesky_corr(corr: &[f64], k: usize) -> Result<Vec<f64>, EstimationError> {
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = corr[i * k + j];
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if sum < -1e-10 {
                    return Err(EstimationError::data_msg(
                        "covariance must be positive semidefinite",
                    ));
                }
                if sum <= 1e-12 {
                    l[i * k + j] = 0.0;
                } else {
                    l[i * k + j] = sum.sqrt();
                }
            } else if l[j * k + j] > 0.0 {
                l[i * k + j] = sum / l[j * k + j];
            } else if sum.abs() > 1e-10 {
                return Err(EstimationError::data_msg("covariance must be positive semidefinite"));
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_weights_preserve_covariance_and_invalid_inputs_refuse() {
        let a = [1.0, 2.0, 4.0, 8.0];
        let plain = joint_influence_covariance(&[&a], None).unwrap();
        for w in [[1.0; 4], [20.0; 4]] {
            let weighted = joint_influence_covariance(&[&a], Some(&w)).unwrap();
            assert!((plain.get(0, 0) - weighted.get(0, 0)).abs() < 1e-12);
        }
        assert!(weighted_mean(&[f64::NAN], None).is_err());
        let bad = JointCovariance { dim: 2, values: Arc::from([1.0, 2.0, 2.0, 1.0]) };
        assert!(max_t_critical(&bad, 0.95, 100, 1).is_err());
        assert!(max_t_critical(&plain, 0.0, 100, 1).is_err());
    }

    #[test]
    fn unweighted_mean_and_se_match_iid() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 2.0, 2.0, 6.0];
        let cov = joint_influence_covariance(&[&a, &b], None).unwrap();
        assert!((weighted_mean(&a, None).unwrap() - 2.5).abs() < 1e-12);
        assert!(cov.se(0) > 0.0 && cov.se(1) > 0.0);
        assert!((cov.get(0, 1) - cov.get(1, 0)).abs() < 1e-12);
    }

    #[test]
    fn weighted_mean_is_standardized() {
        let phi = [0.0, 2.0, 4.0];
        let w = [1.0, 1.0, 2.0];
        let theta = weighted_mean(&phi, Some(&w)).unwrap();
        assert!((theta - 2.5).abs() < 1e-12);
    }

    #[test]
    fn monotone_decreasing_fixes_a_dip() {
        let raw = [0.4, 0.5, 0.3];
        let adj = monotone_decreasing(&raw);
        assert!(adj[0] >= adj[1] && adj[1] >= adj[2]);
    }

    #[test]
    fn mixture_is_mass_weighted() {
        let a = [1.0, 1.0];
        let b = [3.0, 3.0];
        let mix = frozen_weight_mixture_scores(&[&a, &b], &[0.25, 0.75]).unwrap();
        assert!((mix[0] - 2.5).abs() < 1e-12);
    }
}
