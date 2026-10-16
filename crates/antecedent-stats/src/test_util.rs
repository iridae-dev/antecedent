//! Seeded fixtures shared by the unit tests in this crate.
//!
//! One generator, [`antecedent_core::CausalRng`], so no test carries its own LCG.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;

/// `n` independent uniform draws on `[-0.5, 0.5)` from the stream seeded by `seed`.
pub(crate) fn centered_uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = CausalRng::from_seed(seed);
    (0..n).map(|_| rng.next_f64() - 0.5).collect()
}
