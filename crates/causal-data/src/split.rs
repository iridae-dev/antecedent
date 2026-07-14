//! Temporal discovery / estimation splits (DESIGN.md §5.6).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

use crate::error::DataError;

/// Half-open index range `[start, end)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TimeRange {
    /// Inclusive start index.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
}

impl TimeRange {
    /// Length of the range.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// Discovery / estimation split with a temporal gap (DESIGN.md §5.6).
///
/// Layout over `0..series_len`:
/// `[discovery) | gap | [estimation)`.
///
/// This is metadata only — no data copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DiscoveryEstimationSplit {
    /// Contiguous discovery window.
    pub discovery: TimeRange,
    /// Contiguous estimation window (after the gap).
    pub estimation: TimeRange,
    /// Number of time steps skipped between discovery end and estimation start.
    pub gap: usize,
    /// Full series length used to validate the split.
    pub series_len: usize,
}

impl DiscoveryEstimationSplit {
    /// Build a split from absolute ranges.
    ///
    /// # Errors
    ///
    /// Empty windows, inverted ranges, overlapping discovery/estimation, or
    /// ranges outside `series_len`.
    pub fn try_new(
        series_len: usize,
        discovery: TimeRange,
        estimation: TimeRange,
    ) -> Result<Self, DataError> {
        validate_range(series_len, discovery, "discovery")?;
        validate_range(series_len, estimation, "estimation")?;
        if discovery.end > estimation.start {
            return Err(DataError::InvalidArgument {
                message:
                    "discovery window must end at or before estimation start (with optional gap)"
                        .into(),
            });
        }
        let gap = estimation.start - discovery.end;
        Ok(Self { discovery, estimation, gap, series_len })
    }

    /// Split `series_len` into discovery / gap / estimation by sizes.
    ///
    /// # Errors
    ///
    /// When sizes do not sum to `series_len`, or either window is empty.
    pub fn from_sizes(
        series_len: usize,
        discovery_len: usize,
        gap: usize,
        estimation_len: usize,
    ) -> Result<Self, DataError> {
        if discovery_len == 0 || estimation_len == 0 {
            return Err(DataError::InvalidArgument {
                message: "discovery and estimation windows must be non-empty".into(),
            });
        }
        let need = discovery_len
            .checked_add(gap)
            .and_then(|v| v.checked_add(estimation_len))
            .ok_or(DataError::InvalidArgument { message: "split sizes overflow".into() })?;
        if need != series_len {
            return Err(DataError::InvalidArgument {
                message: "discovery + gap + estimation must equal series_len".into(),
            });
        }
        let discovery = TimeRange { start: 0, end: discovery_len };
        let estimation = TimeRange { start: discovery_len + gap, end: series_len };
        Self::try_new(series_len, discovery, estimation)
    }

    /// Proportional split: discovery gets `discovery_frac` of the length after
    /// reserving `gap` steps in the middle (remainder → estimation).
    ///
    /// # Errors
    ///
    /// Invalid fraction, insufficient length for gap + two non-empty windows.
    pub fn from_fraction(
        series_len: usize,
        discovery_frac: f64,
        gap: usize,
    ) -> Result<Self, DataError> {
        if !(discovery_frac > 0.0 && discovery_frac < 1.0) {
            return Err(DataError::InvalidArgument {
                message: "discovery_frac must be in (0, 1)".into(),
            });
        }
        if series_len <= gap + 1 {
            return Err(DataError::InvalidArgument {
                message: "series too short for gap and two non-empty windows".into(),
            });
        }
        let usable = series_len - gap;
        let mut discovery_len = ((usable as f64) * discovery_frac).floor() as usize;
        if discovery_len == 0 {
            discovery_len = 1;
        }
        if discovery_len >= usable {
            discovery_len = usable - 1;
        }
        let estimation_len = usable - discovery_len;
        Self::from_sizes(series_len, discovery_len, gap, estimation_len)
    }
}

fn validate_range(series_len: usize, range: TimeRange, label: &str) -> Result<(), DataError> {
    if range.is_empty() {
        return Err(DataError::InvalidArgument {
            message: format!("{label} window must be non-empty"),
        });
    }
    if range.end > series_len {
        return Err(DataError::InvalidArgument {
            message: format!("{label} window exceeds series_len"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_sizes_layout() {
        let s = DiscoveryEstimationSplit::from_sizes(100, 60, 10, 30).unwrap();
        assert_eq!(s.discovery, TimeRange { start: 0, end: 60 });
        assert_eq!(s.gap, 10);
        assert_eq!(s.estimation, TimeRange { start: 70, end: 100 });
    }

    #[test]
    fn from_fraction_respects_gap() {
        let s = DiscoveryEstimationSplit::from_fraction(100, 0.5, 10).unwrap();
        assert_eq!(s.gap, 10);
        assert_eq!(s.discovery.len() + s.gap + s.estimation.len(), 100);
        assert!(!s.discovery.is_empty());
        assert!(!s.estimation.is_empty());
    }

    #[test]
    fn from_fraction_rejects_boundary_fractions() {
        assert!(DiscoveryEstimationSplit::from_fraction(100, 0.0, 10).is_err());
        assert!(DiscoveryEstimationSplit::from_fraction(100, 1.0, 10).is_err());
        assert!(DiscoveryEstimationSplit::from_fraction(100, f64::NAN, 10).is_err());
    }

    #[test]
    fn rejects_overlap() {
        let err = DiscoveryEstimationSplit::try_new(
            50,
            TimeRange { start: 0, end: 30 },
            TimeRange { start: 20, end: 50 },
        );
        assert!(err.is_err());
    }

    #[test]
    fn rejects_bad_sum() {
        assert!(DiscoveryEstimationSplit::from_sizes(100, 40, 10, 40).is_err());
    }
}
