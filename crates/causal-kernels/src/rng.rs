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
