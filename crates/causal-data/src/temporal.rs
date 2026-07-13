//! Temporal sampling metadata (series length / regularity).
//!
//! Node keys and dense unfolding indexes live in `causal-core` (DESIGN.md §3.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use causal_core::{TemporalIndexError, TemporalIndexer, TemporalNodeKey};

/// Sampling regularity metadata (not used as lag duration for irregular data).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplingRegularity {
    /// Regular sampling with interval nanoseconds.
    Regular {
        /// Interval between samples in nanoseconds.
        interval_ns: u64,
    },
    /// Irregular sampling.
    Irregular,
}

/// Time index metadata for series data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeIndex {
    /// Regularity.
    pub regularity: SamplingRegularity,
    /// Number of time points.
    pub length: usize,
}

#[cfg(test)]
mod tests {
    use causal_core::VariableId;

    use super::*;

    #[test]
    fn reexport_indexer_round_trip() {
        let idx = TemporalIndexer::new(3, 2, 4).unwrap();
        let key = TemporalNodeKey { variable: VariableId::from_raw(1), offset: -1 };
        let dense = idx.dense_id(key).unwrap();
        assert_eq!(idx.key_of(dense).unwrap(), key);
    }
}
