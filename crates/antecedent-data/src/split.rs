//! Split strategies for discovery / estimation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{DeclaredSamplingDesign, SplittingRule};

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

/// Whether random row splits are allowed on temporal / panel / event data.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub(crate) enum TemporalRandomPolicy {
    /// Refuse random IID resampling on temporal-like data (default, §5.6).
    #[default]
    Refuse,
    /// Explicit opt-in for random resampling on temporal / panel / event data.
    #[cfg(test)]
    Allow,
}

/// Guard applied before an IID resample of temporal data.
///
/// # Errors
///
/// When policy is [`TemporalRandomPolicy::Refuse`].
pub(crate) fn ensure_random_allowed_on_temporal(
    policy: TemporalRandomPolicy,
) -> Result<(), DataError> {
    match policy {
        #[cfg(test)]
        TemporalRandomPolicy::Allow => Ok(()),
        TemporalRandomPolicy::Refuse => Err(DataError::unsupported(
            "random IID split",
            "refused on temporal, panel or event data without explicit opt-in",
        )),
    }
}

/// Discovery / estimation split with a temporal gap.
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
            return Err(DataError::invalid_value(
                "discovery window",
                "must end at or before the estimation start (with the optional gap)",
            ));
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
            return Err(DataError::invalid_value(
                "discovery and estimation windows",
                "must be non-empty",
            ));
        }
        let need = discovery_len
            .checked_add(gap)
            .and_then(|v| v.checked_add(estimation_len))
            .ok_or(DataError::overflow("split sizes"))?;
        if need != series_len {
            return Err(DataError::invalid_value(
                "discovery + gap + estimation",
                "must equal series_len",
            ));
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
            return Err(DataError::invalid_value("discovery_frac", "must be in (0, 1)"));
        }
        if series_len <= gap + 1 {
            return Err(DataError::invalid_value(
                "series length",
                "is too short for the gap and two non-empty windows",
            ));
        }
        let usable = series_len - gap;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the product is floored/rounded from a fraction in (0, 1) of a length, so it is non-negative and at most that length"
        )]
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

    /// Bind this split to a contiguous-block design with an embargo of `embargo` steps.
    ///
    /// The gap between the windows is the embargo. A shorter gap leaves a training
    /// observation inside the dependence of the held-out block and is refused. The
    /// observations in the gap belong to neither window.
    ///
    /// # Errors
    ///
    /// The design does not allow a contiguous-block split, `embargo` does not apply
    /// to that rule, or the gap is shorter than the embargo.
    pub fn bind_embargo(
        self,
        design: DeclaredSamplingDesign,
        embargo: u32,
    ) -> Result<Self, DataError> {
        let bound = design
            .apply_splitting_with_embargo(SplittingRule::ContiguousBlock, embargo)
            .map_err(DataError::SamplingDesign)?;
        let required = usize::try_from(bound.embargo())
            .map_err(|_| DataError::overflow("embargo as usize"))?;
        if self.gap < required {
            return Err(DataError::invalid_value("contiguous split", "leaks across the embargo"));
        }
        Ok(self)
    }

    /// Discovery, then an embargo gap, then estimation.
    ///
    /// The gap rows are in neither window. `discovery_len + embargo + estimation_len`
    /// must equal `series_len`.
    ///
    /// # Errors
    ///
    /// As [`Self::bind_embargo`] and [`Self::from_sizes`].
    pub fn dropping_embargo(
        design: DeclaredSamplingDesign,
        series_len: usize,
        discovery_len: usize,
        embargo: u32,
        estimation_len: usize,
    ) -> Result<Self, DataError> {
        let gap = usize::try_from(embargo).map_err(|_| DataError::overflow("embargo as usize"))?;
        let split = Self::from_sizes(series_len, discovery_len, gap, estimation_len)?;
        split.bind_embargo(design, embargo)
    }
}

/// Environment holdout: discovery vs estimation environment index sets (no data copy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvHoldoutSplit {
    /// Environment indices used for discovery / CI.
    pub discovery_envs: Arc<[usize]>,
    /// Environment indices held out for estimation / evaluation.
    pub estimation_envs: Arc<[usize]>,
}

impl EnvHoldoutSplit {
    /// Split environments: first `n_discovery` indices for discovery, rest for estimation.
    ///
    /// # Errors
    ///
    /// Empty total, or `n_discovery` not in `1..n_env`.
    pub fn try_prefix(n_env: usize, n_discovery: usize) -> Result<Self, DataError> {
        if n_env == 0 {
            return Err(DataError::empty("env holdout environments"));
        }
        if n_discovery == 0 || n_discovery >= n_env {
            return Err(DataError::invalid_value("n_discovery", "must be in 1..n_env"));
        }
        Ok(Self {
            discovery_envs: Arc::from((0..n_discovery).collect::<Vec<_>>()),
            estimation_envs: Arc::from((n_discovery..n_env).collect::<Vec<_>>()),
        })
    }
}

fn validate_range(
    series_len: usize,
    range: TimeRange,
    label: &'static str,
) -> Result<(), DataError> {
    if range.is_empty() {
        return Err(DataError::invalid_value(label, "window must be non-empty"));
    }
    if range.end > series_len {
        return Err(DataError::invalid_value(label, "window exceeds series_len"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use antecedent_core::DeclaredSamplingDesign;

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

    #[test]
    fn embargo_split_drops_the_gap() {
        let embargo = 4u32;
        let split = DiscoveryEstimationSplit::dropping_embargo(
            DeclaredSamplingDesign::DependentTimeSeries,
            20,
            8,
            embargo,
            8,
        )
        .expect("a series split drops the embargo");
        assert_eq!(split.gap, 4);
        assert_eq!(split.discovery, TimeRange { start: 0, end: 8 });
        assert_eq!(split.estimation, TimeRange { start: 12, end: 20 });
        let gap: Vec<usize> = (split.discovery.end..split.estimation.start).collect();
        assert_eq!(gap, vec![8, 9, 10, 11]);
        for index in gap {
            let in_discovery = index >= split.discovery.start && index < split.discovery.end;
            let in_estimation = index >= split.estimation.start && index < split.estimation.end;
            assert!(!in_discovery && !in_estimation, "gap row {index} was kept");
        }
        let leaked = DiscoveryEstimationSplit::from_sizes(20, 10, 2, 8).unwrap();
        let error = leaked
            .bind_embargo(DeclaredSamplingDesign::DependentTimeSeries, embargo)
            .expect_err("a split inside the embargo leaks");
        assert!(error.to_string().contains("embargo"), "{error}");
    }

    #[test]
    fn env_holdout_prefix() {
        let s = EnvHoldoutSplit::try_prefix(4, 2).unwrap();
        assert_eq!(s.discovery_envs.as_ref(), &[0, 1]);
        assert_eq!(s.estimation_envs.as_ref(), &[2, 3]);
    }
}
