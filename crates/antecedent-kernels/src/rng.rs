//! Small RNG helpers for sampling kernels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;

use crate::view::ViewError;

/// One standard-normal draw via Box–Muller (cosine component only).
#[must_use]
pub fn standard_normal(rng: &mut CausalRng) -> f64 {
    let (z, _) = standard_normal_pair(rng);
    z
}

/// Box–Muller pair `(cos, sin)` from one uniform `(u1, u2)` draw.
///
/// Prefer [`fill_standard_normal`] when filling a buffer — it uses both
/// components and halves RNG consumption vs repeated [`standard_normal`].
#[must_use]
pub(crate) fn standard_normal_pair(rng: &mut CausalRng) -> (f64, f64) {
    let u1 = rng.next_f64();
    let u2 = rng.next_f64();
    box_muller(u1, u2)
}

/// Smallest positive value [`CausalRng::next_f64`] can return: `2^-53`.
const UNIFORM_RESOLUTION: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Box–Muller transform of two uniforms in `[0, 1)` into a `(cos, sin)` normal pair.
///
/// The single owner of the transform. `ln(0)` is the only singularity, and a
/// 53-bit uniform is either 0 or at least `2^-53`, so `u1` is floored at that
/// resolution: no draw is altered except the exact-zero one, and the tail is
/// truncated at `|z| <= sqrt(2 * 53 * ln 2) ~ 8.57`, the point the generator's own
/// resolution cannot reach anyway.
#[must_use]
pub(crate) fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let r = (-2.0 * u1.max(UNIFORM_RESOLUTION).ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

/// One standard exponential draw `-ln U`, with `U` floored at `f64::MIN_POSITIVE`
/// so an exact-zero uniform yields a large finite value instead of `+∞`.
#[must_use]
pub(crate) fn exp1(rng: &mut CausalRng) -> f64 {
    -rng.next_f64().max(f64::MIN_POSITIVE).ln()
}

/// `n` Bayesian-bootstrap (Dirichlet(1,…,1) up to scale) weights from [`exp1`] draws.
///
/// With `normalize == false` the raw exponentials are returned. With
/// `normalize == true` they are rescaled to mean one, `wᵢ *= n / Σw` (the
/// unguarded `n as f64 / total` of the call sites this replaces: a zero or
/// non-finite total is left to the caller to check on the raw variant).
#[must_use]
pub fn bayesian_bootstrap_weights(rng: &mut CausalRng, n: usize, normalize: bool) -> Vec<f64> {
    let mut weights: Vec<f64> = (0..n).map(|_| exp1(rng)).collect();
    if normalize {
        let total: f64 = weights.iter().sum();
        for w in &mut weights {
            *w *= n as f64 / total;
        }
    }
    weights
}

/// Fill `out` with i.i.d. standard normals, emitting both Box–Muller components.
pub fn fill_standard_normal(rng: &mut CausalRng, out: &mut [f64]) {
    let mut i = 0;
    while i < out.len() {
        let (z0, z1) = standard_normal_pair(rng);
        out[i] = z0;
        i += 1;
        if i < out.len() {
            out[i] = z1;
            i += 1;
        }
    }
}

/// Unbiased uniform index in `0..n` (rejects modulo bias).
///
/// # Panics
///
/// Panics if `n == 0`.
#[must_use]
pub fn unbiased_index(rng: &mut CausalRng, n: usize) -> usize {
    assert!(n > 0, "unbiased_index requires n > 0");
    // Every supported target has `usize` no wider than `u64`, so both casts are exact.
    const _: () = assert!(usize::BITS <= u64::BITS);
    let n64 = n as u64;
    // Largest multiple of n that fits in u64.
    let limit = u64::MAX - (u64::MAX % n64);
    loop {
        let v = rng.next_u64();
        if v < limit {
            return (v % n64) as usize;
        }
    }
}

/// Fisher–Yates shuffle in place.
pub fn shuffle<T>(rng: &mut CausalRng, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = unbiased_index(rng, i + 1);
        items.swap(i, j);
    }
}

/// Assign each of `n` rows to one of `folds` cross-validation folds through a
/// seeded shuffle, optionally stratified.
///
/// A position-based `i % folds` assignment makes fold membership a function of
/// row order, so any periodic or sorted structure aligned with the fold count
/// (alternating treated/control rows, weekly data with seven folds) puts whole
/// classes into single folds. The rows are shuffled first; with `strata`, rows
/// are then grouped by stratum (stable within a stratum) so every stratum is
/// dealt round-robin across the folds and each fold's training rows retain
/// every stratum that has at least `folds` members.
///
/// # Errors
///
/// [`ViewError::InvalidShape`] when `folds == 0` or `strata` has a length other
/// than `n`. Those checks run before the shuffle, so a refused call draws nothing.
#[must_use]
pub fn shuffled_fold_assignment(
    rng: &mut CausalRng,
    n: usize,
    folds: usize,
    strata: Option<&[u32]>,
) -> Result<Vec<usize>, ViewError> {
    if folds == 0 {
        return Err(ViewError::InvalidShape { message: "fold count must be positive" });
    }
    if strata.is_some_and(|s| s.len() != n) {
        return Err(ViewError::InvalidShape { message: "strata must align with rows" });
    }
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(rng, &mut order);
    if let Some(strata) = strata {
        order.sort_by_key(|&row| strata[row]);
    }
    let mut assignment = vec![0usize; n];
    for (rank, &row) in order.iter().enumerate() {
        assignment[row] = rank % folds;
    }
    Ok(assignment)
}

/// A weight counts only if it is finite and positive; NaN, negative and infinite
/// weights carry no probability mass.
fn categorical_mass(weight: f64) -> f64 {
    if weight.is_finite() && weight > 0.0 { weight } else { 0.0 }
}

/// Inverse-CDF category for `target` in `[0, total)` over the sanitized masses.
///
/// `target < cumulative` skips zero-mass categories by construction. Rounding
/// that leaves `target` at or above the final cumulative mass resolves to the
/// last category that has mass.
fn categorical_index(weights: &[f64], target: f64) -> usize {
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += categorical_mass(w);
        if target < acc {
            return i;
        }
    }
    weights.iter().rposition(|&w| categorical_mass(w) > 0.0).unwrap_or(0)
}

/// Total sanitized mass, or `None` when there is none (or it overflows).
fn categorical_total(weights: &[f64]) -> Option<f64> {
    let total: f64 = weights.iter().map(|&w| categorical_mass(w)).sum();
    (total > 0.0 && total.is_finite()).then_some(total)
}

/// Draw a categorical index given non-negative weights (normalized internally).
///
/// Negative, NaN and infinite weights carry no mass. Returns `None` when there is
/// no positive mass (empty, all-zero or all-invalid weights); a category with zero
/// mass is never returned.
#[must_use]
pub fn sample_categorical(rng: &mut CausalRng, weights: &[f64]) -> Option<usize> {
    let total = categorical_total(weights)?;
    Some(categorical_index(weights, rng.next_f64() * total))
}

/// Draw a categorical index from a unit-interval `u` and (possibly unnormalized) weights.
///
/// The inverse CDF: category `i` owns `[F(i-1), F(i))`, so `u == 0` never selects a
/// leading zero-mass category and `u == 1` selects the last category with mass.
/// `None` when the weights carry no positive mass or `u` is NaN.
#[must_use]
pub fn categorical_from_u(u: f64, probs: &[f64]) -> Option<usize> {
    if u.is_nan() {
        return None;
    }
    let total = categorical_total(probs)?;
    Some(categorical_index(probs, u.clamp(0.0, 1.0) * total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_cdf_skips_zero_mass_categories() {
        // u = 0 belongs to the first category WITH mass, not the zero-mass index 0.
        assert_eq!(categorical_from_u(0.0, &[0.0, 0.5, 0.5]), Some(1));
        // F = [0, 0.5, 1]: u = 0.5 opens the interval of category 2.
        assert_eq!(categorical_from_u(0.5, &[0.0, 0.5, 0.5]), Some(2));
        assert_eq!(categorical_from_u(0.25, &[0.0, 0.5, 0.5]), Some(1));
        // u = 1 lands on the last category with mass, never a trailing zero-mass one.
        assert_eq!(categorical_from_u(1.0, &[0.5, 0.5, 0.0]), Some(1));
        // Unnormalized weights use their own total.
        assert_eq!(categorical_from_u(0.75, &[2.0, 6.0]), Some(1));
        assert_eq!(categorical_from_u(0.2, &[2.0, 6.0]), Some(0));
    }

    #[test]
    fn categorical_without_mass_is_none() {
        assert_eq!(categorical_from_u(0.5, &[]), None);
        assert_eq!(categorical_from_u(0.5, &[0.0, 0.0]), None);
        assert_eq!(categorical_from_u(0.5, &[f64::NAN, -1.0]), None);
        assert_eq!(categorical_from_u(f64::NAN, &[1.0]), None);
        let mut rng = CausalRng::from_seed(1);
        assert_eq!(sample_categorical(&mut rng, &[0.0, 0.0]), None);
        assert_eq!(sample_categorical(&mut rng, &[f64::NAN]), None);
    }

    #[test]
    fn invalid_weights_carry_no_mass_in_the_same_total() {
        // The negative weight is dropped from both the total and the running sum.
        assert_eq!(categorical_from_u(0.9, &[3.0, -2.0, 1.0]), Some(2));
        assert_eq!(categorical_from_u(0.7, &[3.0, -2.0, 1.0]), Some(0));
        let mut rng = CausalRng::from_seed(7);
        let n = 20_000;
        let mut counts = [0usize; 3];
        for _ in 0..n {
            counts[sample_categorical(&mut rng, &[3.0, -2.0, 1.0]).unwrap()] += 1;
        }
        assert_eq!(counts[1], 0, "the negative-weight category must never be drawn");
        // P(2) = 1/4; the binomial sd at n = 20000 is 0.0031.
        let freq = counts[2] as f64 / f64::from(n);
        assert!((freq - 0.25).abs() < 0.02, "freq {freq}");
    }

    #[test]
    fn box_muller_matches_the_closed_form() {
        let (c, s) = box_muller(0.5, 0.25);
        let r = (-2.0 * 0.5_f64.ln()).sqrt();
        assert!((c - r * (std::f64::consts::FRAC_PI_2).cos()).abs() < 1e-15);
        assert!((s - r).abs() < 1e-15);
        // u1 = 0 is finite and bounded by the 2^-53 floor: r = sqrt(2 * 53 * ln 2).
        let (z, _) = box_muller(0.0, 0.0);
        let bound = (2.0 * 53.0 * std::f64::consts::LN_2).sqrt();
        assert!((z - bound).abs() < 1e-12, "{z} vs {bound}");
    }
}

/// One draw from `Gamma(shape, rate)` (Marsaglia–Tsang; `shape < 1` via the
/// `G(shape + 1) · U^{1/shape}` boost).
///
/// Returns NaN when `shape` or `rate` is not a positive finite number: no
/// gamma law exists there, and the rejection loop would otherwise spin or
/// recurse without bound.
#[must_use]
pub fn sample_gamma(shape: f64, rate: f64, rng: &mut CausalRng) -> f64 {
    if !(shape.is_finite() && shape > 0.0 && rate.is_finite() && rate > 0.0) {
        return f64::NAN;
    }
    if shape < 1.0 {
        let u = rng.next_f64().max(f64::EPSILON);
        return sample_gamma(shape + 1.0, rate, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = standard_normal(rng);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v / rate;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v / rate;
        }
    }
}

/// One draw from `InvGamma(shape, scale)` (`1 / Gamma(shape, rate = scale)`; mean
/// `scale / (shape − 1)` for `shape > 1`).
///
/// NaN for invalid parameters (see [`sample_gamma`]); `+∞` only if the gamma
/// draw underflows to zero, never a clamped finite stand-in.
#[must_use]
pub fn sample_inv_gamma(shape: f64, scale: f64, rng: &mut CausalRng) -> f64 {
    1.0 / sample_gamma(shape, scale, rng)
}

#[cfg(test)]
mod gamma_tests {
    use super::*;

    #[test]
    fn invalid_gamma_parameters_are_nan_not_a_hang() {
        let mut rng = CausalRng::from_seed(1);
        for (shape, rate) in [
            (f64::NAN, 1.0),
            (f64::NEG_INFINITY, 1.0),
            (f64::INFINITY, 1.0),
            (0.0, 1.0),
            (-2.0, 1.0),
            (2.0, 0.0),
            (2.0, -1.0),
            (2.0, f64::NAN),
        ] {
            assert!(sample_gamma(shape, rate, &mut rng).is_nan(), "shape={shape} rate={rate}");
            assert!(sample_inv_gamma(shape, rate, &mut rng).is_nan(), "shape={shape} rate={rate}");
        }
    }

    #[test]
    fn gamma_mean_and_variance_match_closed_form() {
        // Gamma(shape 3, rate 2): mean 3/2, variance 3/4. n = 40 000 -> SE(mean) ~ 0.0043.
        let mut rng = CausalRng::from_seed(7);
        let n = 40_000usize;
        let draws: Vec<f64> = (0..n).map(|_| sample_gamma(3.0, 2.0, &mut rng)).collect();
        let mean = draws.iter().sum::<f64>() / n as f64;
        let var = draws.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!((mean - 1.5).abs() < 0.03, "mean {mean}");
        assert!((var - 0.75).abs() < 0.04, "var {var}");
        // shape < 1 boost: Gamma(0.5, rate 1) has mean 0.5.
        let m: f64 = (0..n).map(|_| sample_gamma(0.5, 1.0, &mut rng)).sum::<f64>() / n as f64;
        assert!((m - 0.5).abs() < 0.02, "shape<1 mean {m}");
    }
}

#[cfg(test)]
mod fold_tests {
    use super::*;

    /// Alternating classes with two folds put every treated row in one fold
    /// under `i % 2`; a stratified shuffle deals each class across both folds.
    #[test]
    fn stratified_shuffled_folds_split_every_class_across_folds() {
        let n = 40;
        let strata: Vec<u32> = (0..n).map(|i| u32::from(i % 2 == 0)).collect();
        let mut rng = CausalRng::from_seed(5);
        let fold = shuffled_fold_assignment(&mut rng, n, 2, Some(&strata)).unwrap();
        for class in 0..2u32 {
            for f in 0..2usize {
                let count = (0..n).filter(|&i| strata[i] == class && fold[i] == f).count();
                assert_eq!(count, 10, "class {class} fold {f}");
            }
        }
    }

    /// Without strata, fold sizes are balanced to within one row and the
    /// assignment is not the position-periodic `i % folds`.
    #[test]
    fn unstratified_shuffled_folds_are_balanced_and_not_periodic() {
        let n = 103;
        let mut rng = CausalRng::from_seed(9);
        let fold = shuffled_fold_assignment(&mut rng, n, 5, None).unwrap();
        for f in 0..5usize {
            let size = fold.iter().filter(|&&g| g == f).count();
            assert!(size == 20 || size == 21, "fold {f} has {size} rows");
        }
        assert!((0..n).any(|i| fold[i] != i % 5));
    }

    /// A zero fold count or a short stratum vector is refused before any draw.
    #[test]
    fn fold_assignment_refuses_before_drawing() {
        let mut rng = CausalRng::from_seed(3);
        let mut twin = CausalRng::from_seed(3);
        assert_eq!(
            shuffled_fold_assignment(&mut rng, 4, 0, None).unwrap_err(),
            ViewError::InvalidShape { message: "fold count must be positive" }
        );
        assert_eq!(
            shuffled_fold_assignment(&mut rng, 4, 2, Some(&[1])).unwrap_err(),
            ViewError::InvalidShape { message: "strata must align with rows" }
        );
        assert_eq!(rng.next_u64(), twin.next_u64());
    }

    /// The helper reproduces the inline `-ln(max(u, MIN_POSITIVE))` loop and the
    /// mean-one rescale bit for bit.
    #[test]
    fn bootstrap_weights_match_inline_loop() {
        let n = 17;
        let mut a = CausalRng::from_seed(11);
        let mut b = CausalRng::from_seed(11);
        let raw: Vec<f64> = (0..n).map(|_| -a.next_f64().max(f64::MIN_POSITIVE).ln()).collect();
        let got_raw = bayesian_bootstrap_weights(&mut b, n, false);
        assert!(raw.iter().zip(&got_raw).all(|(x, y)| x.to_bits() == y.to_bits()));
        let mut norm = raw.clone();
        let total: f64 = norm.iter().sum();
        for w in &mut norm {
            *w *= n as f64 / total;
        }
        let mut c = CausalRng::from_seed(11);
        let got = bayesian_bootstrap_weights(&mut c, n, true);
        assert!(norm.iter().zip(&got).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}
