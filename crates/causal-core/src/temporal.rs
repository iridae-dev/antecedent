//! Temporal node identity and dense unfolding indexes (ADR 0005 / ).
//!
//! Dense indexes are process-local and **must not** be serialized.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use crate::ids::{Lag, VariableId};

/// Errors from temporal indexing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TemporalIndexError {
    /// Invalid constructor arguments.
    Invalid {
        /// Message.
        message: &'static str,
    },
    /// Variable id outside the indexer.
    UnknownVariable {
        /// Variable.
        id: VariableId,
    },
}

impl fmt::Display for TemporalIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { message } => write!(f, "{message}"),
            Self::UnknownVariable { id } => write!(f, "unknown variable id {}", id.raw()),
        }
    }
}

impl std::error::Error for TemporalIndexError {}

/// Stable unfolded temporal node identity (serializable).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TemporalNodeKey {
    /// Variable.
    pub variable: VariableId,
    /// Offset relative to the analysis origin (may be negative for history).
    pub offset: i32,
}

impl TemporalNodeKey {
    /// Contemporaneous node at the origin.
    #[must_use]
    pub const fn contemporaneous(variable: VariableId) -> Self {
        Self { variable, offset: 0 }
    }

    /// Node at `t - lag` when lag is non-negative and fits in `i32`.
    #[must_use]
    pub fn lagged(variable: VariableId, lag: Lag) -> Option<Self> {
        let lag_i = i32::try_from(lag.raw()).ok()?;
        Some(Self { variable, offset: -lag_i })
    }
}

/// Finite unfolding indexer: time-major dense layout.
///
/// `dense_index = time_slice_index * variable_count + variable_index`
/// where `time_slice_index = offset + history`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalIndexer {
    variable_count: u32,
    /// Number of historical slices before offset 0.
    history: u32,
    /// Number of forward slices including offset 0.
    horizon: u32,
}

impl TemporalIndexer {
    /// Create an indexer.
    ///
    /// # Errors
    ///
    /// When counts are zero or products overflow.
    pub fn new(
        variable_count: u32,
        history: u32,
        horizon: u32,
    ) -> Result<Self, TemporalIndexError> {
        if variable_count == 0 || horizon == 0 {
            return Err(TemporalIndexError::Invalid {
                message: "variable_count and horizon must be non-zero",
            });
        }
        let slices = history
            .checked_add(horizon)
            .ok_or(TemporalIndexError::Invalid { message: "history+horizon overflow" })?;
        let _ = variable_count
            .checked_mul(slices)
            .ok_or(TemporalIndexError::Invalid { message: "dense index space overflow" })?;
        Ok(Self { variable_count, history, horizon })
    }

    /// Variable count.
    #[must_use]
    pub const fn variable_count(&self) -> u32 {
        self.variable_count
    }

    /// History depth.
    #[must_use]
    pub const fn history(&self) -> u32 {
        self.history
    }

    /// Horizon (including t=0).
    #[must_use]
    pub const fn horizon(&self) -> u32 {
        self.horizon
    }

    /// Total dense nodes.
    #[must_use]
    pub fn dense_len(&self) -> usize {
        let slices = u64::from(self.history) + u64::from(self.horizon);
        usize::try_from(slices * u64::from(self.variable_count)).expect("checked at construction")
    }

    /// Convert a stable key to a dense index.
    ///
    /// # Errors
    ///
    /// Out of range variable or offset.
    pub fn dense_id(&self, key: TemporalNodeKey) -> Result<u32, TemporalIndexError> {
        let v = key.variable.raw();
        if v >= self.variable_count {
            return Err(TemporalIndexError::UnknownVariable { id: key.variable });
        }
        let slice = i64::from(key.offset) + i64::from(self.history);
        if slice < 0 || slice >= i64::from(self.history) + i64::from(self.horizon) {
            return Err(TemporalIndexError::Invalid {
                message: "temporal offset outside unfolding window",
            });
        }
        let dense = u64::try_from(slice).expect("non-negative") * u64::from(self.variable_count)
            + u64::from(v);
        u32::try_from(dense)
            .map_err(|_| TemporalIndexError::Invalid { message: "dense id exceeds u32" })
    }

    /// Invert a dense index to a stable key.
    ///
    /// # Errors
    ///
    /// Out of range dense id.
    pub fn key_of(&self, dense: u32) -> Result<TemporalNodeKey, TemporalIndexError> {
        if dense as usize >= self.dense_len() {
            return Err(TemporalIndexError::Invalid { message: "dense id out of range" });
        }
        let vc = self.variable_count;
        let slice = dense / vc;
        let var = dense % vc;
        let offset = i32::try_from(i64::from(slice) - i64::from(self.history))
            .map_err(|_| TemporalIndexError::Invalid { message: "offset overflow" })?;
        Ok(TemporalNodeKey { variable: VariableId::from_raw(var), offset })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_major_dense_round_trip() {
        let idx = TemporalIndexer::new(3, 2, 4).unwrap();
        assert_eq!(idx.dense_len(), 18);
        let key = TemporalNodeKey { variable: VariableId::from_raw(1), offset: -1 };
        let dense = idx.dense_id(key).unwrap();
        assert_eq!(dense, 4);
        assert_eq!(idx.key_of(dense).unwrap(), key);
    }
}
