//! Seed and stream derivation for temporal mediation fits.
//!
//! Every mediation route (the sealed `TemporalDag` operation, the one-shot
//! route, the class and DBN routes) draws its random streams here, so one
//! derivation owns the independence guarantees.
//!
//! Additive offsets (`seed + h * K + i * K'`) are not used: two
//! `(horizon, mechanism)` pairs can land on the same sum, and adjacent master
//! seeds then share almost every stream. A mechanism seed is instead the first
//! word of the [`StreamDomain::Mediation`] stream keyed by the master seed and
//! the packed `(horizon, mechanism)` index, the same discipline
//! `completion_fit_seed` uses for completions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{RngFactory, StreamDomain, derive_stream};

/// The mediation route a block-bootstrap stream belongs to. Routes keep
/// distinct streams so a class estimate never replays a single-graph one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MediationRoute {
    /// A fixed `TemporalDag` (sealed and one-shot).
    Temporal,
    /// A temporal class envelope.
    TemporalClass,
    /// A DBN structure posterior.
    Dbn,
}

impl MediationRoute {
    fn domain_tag(self) -> u64 {
        match self {
            Self::Temporal => 0x6D65_6469_6174_6501,
            Self::TemporalClass => 0x6D65_6469_6174_6502,
            Self::Dbn => 0x6D65_6469_6174_6503,
        }
    }
}

/// Fit seed of mediation mechanism `mechanism` (mediator, then outcome) at
/// `horizon`, an independent stream per `(master seed, horizon, mechanism)`.
///
/// `master_seed` is the study seed, or a per-completion seed on envelope
/// routes; `horizon` is `0` for a route without horizons.
pub(super) fn mediation_mechanism_seed(master_seed: u64, horizon: u32, mechanism: usize) -> u64 {
    let index = (u64::from(horizon) << 32) | (mechanism as u64 & 0xFFFF_FFFF);
    RngFactory::from_seed(master_seed).stream_for(StreamDomain::Mediation, index).next_u64()
}

/// Stream base of the shared circular-block bootstrap of one mediation
/// horizon.
///
/// The resampler keys the master seed in and XORs the replicate number into
/// the low 32 bits of the base, so the base keeps those bits zero and carries
/// `(route tag, horizon)` in the high word: distinct horizons of one route are
/// distinct by construction, and no `(horizon, replicate)` pair can alias
/// another.
pub(super) fn mediation_block_stream(route: MediationRoute, horizon: u32) -> u64 {
    #[allow(clippy::cast_possible_truncation, reason = "the high word of a mixed tag")]
    let tag = (derive_stream(route.domain_tag(), 0) >> 32) as u32;
    u64::from(tag ^ horizon) << 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mechanism_seeds_are_distinct_across_horizon_and_mechanism() {
        let mut seen = std::collections::HashSet::new();
        for horizon in 0..16 {
            for mechanism in 0..4 {
                assert!(seen.insert(mediation_mechanism_seed(7, horizon, mechanism)));
            }
        }
        assert_ne!(mediation_mechanism_seed(7, 1, 0), mediation_mechanism_seed(8, 1, 0));
    }

    #[test]
    fn block_streams_keep_the_replicate_word_free_and_differ_by_route_and_horizon() {
        let mut seen = std::collections::HashSet::new();
        for route in [MediationRoute::Temporal, MediationRoute::TemporalClass, MediationRoute::Dbn]
        {
            for horizon in 1..32 {
                let base = mediation_block_stream(route, horizon);
                assert_eq!(base & 0xFFFF_FFFF, 0);
                assert!(seen.insert(base));
            }
        }
    }
}
