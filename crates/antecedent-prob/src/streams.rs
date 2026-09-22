//! Random streams for posterior computation, all from the library's stream mixer.
//!
//! Every posterior sampler (HMC chains, Laplace / conjugate / multivariate-normal draws) seeds
//! from [`RngFactory`] so that its stream is an unrelated offset of the run seed rather than the
//! run seed itself: two samplers given the same seed must not replay the same underlying
//! uniform sequence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, RngFactory, StreamDomain};

/// Stream index of the single-stream direct draws (Laplace, conjugate, MVN). Chain streams use
/// the chain number, which never reaches this index.
const DIRECT_DRAW_STREAM: u64 = 0xD1EC_7000;

/// One stream per HMC chain.
pub(crate) fn chain_rng(seed: u64, chain: usize) -> CausalRng {
    RngFactory::from_seed(seed).stream_for(StreamDomain::Bayesian, chain as u64)
}

/// The stream for direct posterior draws (Laplace normal approximation, conjugate NIG and
/// known-variance normal, multivariate normal).
pub(crate) fn direct_draw_rng(seed: u64) -> CausalRng {
    RngFactory::from_seed(seed).stream_for(StreamDomain::Bayesian, DIRECT_DRAW_STREAM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_draws_do_not_replay_the_seed_stream_or_a_chain_stream() {
        let seed = 42;
        let first = |mut rng: CausalRng| rng.next_u64();
        let raw = first(CausalRng::from_seed(seed));
        let direct = first(direct_draw_rng(seed));
        assert_ne!(direct, raw);
        for chain in 0..8 {
            assert_ne!(direct, first(chain_rng(seed, chain)));
        }
        // Reproducible per seed, distinct across seeds.
        assert_eq!(direct, first(direct_draw_rng(seed)));
        assert_ne!(direct, first(direct_draw_rng(seed + 1)));
    }
}
