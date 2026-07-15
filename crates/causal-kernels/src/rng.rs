//! Small RNG helpers for sampling kernels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::CausalRng;

/// One standard-normal draw via Box–Muller.
#[must_use]
pub fn standard_normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(f64::EPSILON);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Unbiased uniform index in `0..n` (rejects modulo bias).
///
/// # Panics
///
/// Panics if `n == 0`.
#[must_use]
pub fn unbiased_index(rng: &mut CausalRng, n: usize) -> usize {
    assert!(n > 0, "unbiased_index requires n > 0");
    // Largest multiple of n that fits in u64.
    let n64 = n as u64;
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

/// Draw a categorical index given non-negative weights (normalized internally).
///
/// Returns `0` if all weights are zero / empty.
#[must_use]
pub fn sample_categorical(rng: &mut CausalRng, weights: &[f64]) -> usize {
    if weights.is_empty() {
        return 0;
    }
    let sum: f64 = weights.iter().copied().sum();
    if !(sum > 0.0) {
        return 0;
    }
    let u = rng.next_f64() * sum;
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w.max(0.0);
        if u < acc {
            return i;
        }
    }
    weights.len() - 1
}

/// Draw a categorical index from a unit-interval `u` and (possibly unnormalized) weights.
#[must_use]
pub fn categorical_from_u(u: f64, probs: &[f64]) -> usize {
    if probs.is_empty() {
        return 0;
    }
    let sum: f64 = probs.iter().copied().sum::<f64>().max(f64::EPSILON);
    let target = u.clamp(0.0, 1.0) * sum;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p.max(0.0);
        if target <= acc {
            return i;
        }
    }
    probs.len() - 1
}
