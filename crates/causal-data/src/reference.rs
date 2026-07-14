//! Reference-point policies for temporal sample alignment (DESIGN.md §5.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// How lag-aligned samples choose their contemporaneous origin row.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum ReferencePointPolicy {
    /// Origin at row `max_lag` so every lag in `0..=max_lag` is in-bounds
    /// (default stationary series alignment).
    #[default]
    SeriesOrigin,
    /// Absolute origin row; effective samples are
    /// `origin_row .. series_len` clipped so lags stay in-bounds.
    AbsoluteOrigin {
        /// Row treated as contemporaneous lag-0.
        origin_row: usize,
    },
}

impl ReferencePointPolicy {
    /// Base contemporaneous row and effective sample count for a series.
    ///
    /// # Errors
    ///
    /// Empty series, `max_lag` too large, or origin out of range.
    pub fn base_and_n(
        self,
        series_len: usize,
        max_lag: u32,
    ) -> Result<(usize, usize), crate::error::DataError> {
        use crate::error::DataError;
        if series_len == 0 {
            return Err(DataError::InvalidArgument { message: "empty time series".into() });
        }
        let max_lag_usize = max_lag as usize;
        if max_lag_usize >= series_len {
            return Err(DataError::InvalidArgument {
                message: "max_lag must be strictly less than series length".into(),
            });
        }
        match self {
            Self::SeriesOrigin => {
                let base = max_lag_usize;
                Ok((base, series_len - base))
            }
            Self::AbsoluteOrigin { origin_row } => {
                if origin_row < max_lag_usize || origin_row >= series_len {
                    return Err(DataError::InvalidArgument {
                        message: "absolute origin must satisfy max_lag <= origin < series_len"
                            .into(),
                    });
                }
                Ok((origin_row, series_len - origin_row))
            }
        }
    }
}
