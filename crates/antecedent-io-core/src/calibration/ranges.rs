//! Resolve measured coverage ranges from existing records.

use super::{
    BASIS_MISSING, CalibrationSlotWire, CoverageGridPoint, CoverageRecord, calibration_slots,
};

/// Row-count range a record measured: the span of its sample-size grid points
/// (`n_min` of the smallest point to `n_max` of the largest), or its own
/// `n_min..n_max` for a record measured at a single sample size.
#[must_use]
pub fn measured_range(record: &CoverageRecord) -> (u64, u64) {
    let points: &[CoverageGridPoint] = record.grid;
    match (points.iter().map(|p| p.n_min).min(), points.iter().map(|p| p.n_max).max()) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (record.n_min, record.n_max),
    }
}

/// Whether a record is a boundary measurement: flagged itself, or under-covering
/// (outside its nominal band, or a named boundary) at any grid point. A failing
/// point is never averaged into a pass over the range.
#[must_use]
pub fn is_boundary(record: &CoverageRecord) -> bool {
    record.boundary || record.grid.iter().any(|point| point.boundary)
}

/// The independently confirmed passing region of a boundary record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConfirmedRegion {
    /// Smallest row count of the region.
    pub n_min: u64,
    /// Largest row count of the region.
    pub n_max: u64,
    /// Lowest coverage over the region's points, in the measurement and in the
    /// confirmation.
    pub observed: f64,
}

/// The confirmed passing region of a boundary `record` that covers `rows`.
///
/// `Some` only when the record states a `confirmed_passing_range` holding
/// `rows` and the evidence agrees with it: at least two grid points lie inside
/// the range, none of them failed, and each has a confirmation point of the
/// same index and sample sizes that did not fail either. The failing points
/// stay on the record, so a row count outside the range (in the failing
/// region, or between it and the range) is still governed by the boundary. A
/// record that is not a boundary has nothing to recover and returns `None`.
#[must_use]
pub fn confirmed_passing_region(record: &CoverageRecord, rows: u64) -> Option<ConfirmedRegion> {
    let (lo, hi) = record.confirmed_passing_range?;
    if !is_boundary(record) || rows < lo || hi < rows {
        return None;
    }
    let inside = |point: &&CoverageGridPoint| lo <= point.n_min && point.n_max <= hi;
    let measured: Vec<&CoverageGridPoint> = record.grid.iter().filter(inside).collect();
    let confirmed = |point: &CoverageGridPoint| {
        record.confirmation.iter().find(|other| {
            other.point == point.point
                && (other.n_min, other.n_max) == (point.n_min, point.n_max)
                && !other.boundary
        })
    };
    let all_pass = measured.iter().all(|point| !point.boundary && confirmed(point).is_some());
    if measured.len() < 2 || !all_pass {
        return None;
    }
    let observed = measured
        .iter()
        .flat_map(|point| [Some(point.observed), confirmed(point).map(|other| other.observed)])
        .flatten()
        .fold(f64::INFINITY, f64::min);
    Some(ConfirmedRegion { n_min: lo, n_max: hi, observed })
}

/// Why a row count inside a boundary record is, or is not, recovered by a
/// passing subrange: the typed explanation behind the slot's
/// [`BOUNDARY_RECORD`] refusal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RangeResolution {
    /// The record is not a boundary: there is nothing to recover.
    NotABoundary,
    /// The row count lies outside every sample size the record measured.
    OutsideMeasuredRange,
    /// The row count lies in an independently confirmed passing region.
    Recovered(ConfirmedRegion),
    /// The row count lies at, or nearest to, a grid point that failed: the
    /// boundary governs with that point's coverage.
    FailedPoint {
        /// Grid point index.
        point: u8,
        /// Smallest row count the point measured.
        n_min: u64,
        /// Largest row count the point measured.
        n_max: u64,
        /// Coverage the point observed.
        observed: f64,
    },
    /// The row count lies among grid points that passed the measurement and
    /// reach an end of the grid, but no independent confirmation passed at
    /// every one of them: the missing confirmation keeps the record a boundary.
    Unconfirmed {
        /// Smallest row count of the passing points.
        n_min: u64,
        /// Largest row count of the passing points.
        n_max: u64,
    },
    /// The row count lies between passing grid points that a failed interior
    /// point separates: the passing region is not contiguous and is not merged.
    NotContiguous {
        /// Grid point index of the failed interior point.
        failed_point: u8,
        /// Smallest row count the failed point measured.
        n_min: u64,
        /// Largest row count the failed point measured.
        n_max: u64,
    },
    /// The row count lies among passing grid points with failed points on both
    /// sides: a subrange must reach the top or the bottom of the grid.
    InteriorRun {
        /// Smallest row count of the passing points.
        n_min: u64,
        /// Largest row count of the passing points.
        n_max: u64,
    },
    /// Only one grid point passed: two points are the least a subrange rests on.
    LonePassingPoint {
        /// Smallest row count of the point.
        n_min: u64,
        /// Largest row count of the point.
        n_max: u64,
    },
}

/// Resolve a row count against the grid of a boundary `record`: recovered by
/// a confirmed passing subrange, or the reason it is not.
///
/// The slot of such a row count is [`calibration_slot_with`]'s: `calibrated`
/// exactly when this is [`RangeResolution::Recovered`], and otherwise a
/// [`BOUNDARY_RECORD`] refusal that this names.
#[must_use]
pub fn resolve_passing_range(record: &CoverageRecord, rows: u64) -> RangeResolution {
    if !is_boundary(record) {
        return RangeResolution::NotABoundary;
    }
    let (lo, hi) = measured_range(record);
    if rows < lo || hi < rows {
        return RangeResolution::OutsideMeasuredRange;
    }
    if let Some(region) = confirmed_passing_region(record, rows) {
        return RangeResolution::Recovered(region);
    }
    let passing: Vec<&CoverageGridPoint> = record.grid.iter().filter(|p| !p.boundary).collect();
    if let (Some(first), Some(last)) = (passing.first(), passing.last())
        && first.n_min <= rows
        && rows <= last.n_max
    {
        let interior = record
            .grid
            .iter()
            .find(|p| p.boundary && first.n_max < p.n_min && p.n_max < last.n_min);
        if let Some(failed) = interior {
            return RangeResolution::NotContiguous {
                failed_point: failed.point,
                n_min: failed.n_min,
                n_max: failed.n_max,
            };
        }
        let (n_min, n_max) = (first.n_min, last.n_max);
        let below = record.grid.iter().any(|p| p.boundary && p.n_max < first.n_min);
        let above = record.grid.iter().any(|p| p.boundary && last.n_max < p.n_min);
        return if passing.len() < 2 {
            RangeResolution::LonePassingPoint { n_min, n_max }
        } else if below && above {
            RangeResolution::InteriorRun { n_min, n_max }
        } else {
            RangeResolution::Unconfirmed { n_min, n_max }
        };
    }
    let distance = |p: &CoverageGridPoint| {
        if rows < p.n_min { p.n_min - rows } else { rows.saturating_sub(p.n_max) }
    };
    record.grid.iter().filter(|p| p.boundary).min_by_key(|&p| distance(p)).map_or(
        RangeResolution::OutsideMeasuredRange,
        |p| RangeResolution::FailedPoint {
            point: p.point,
            n_min: p.n_min,
            n_max: p.n_max,
            observed: p.observed,
        },
    )
}

/// Coverage that governs `record` at `rows`: the worst confirmed point inside
/// a confirmed passing region, otherwise the record's own (worst) coverage.
fn governing_observed(record: &CoverageRecord, rows: u64) -> f64 {
    confirmed_passing_region(record, rows).map_or(record.observed, |region| region.observed)
}

/// Lowest governing coverage among `records` at `rows` (ties broken by id): the
/// record that governs a slot when several measurements cover one execution.
pub(super) fn worst<'a>(records: &[&'a CoverageRecord], rows: u64) -> Option<&'a CoverageRecord> {
    records.iter().copied().min_by(|a, b| {
        governing_observed(a, rows).total_cmp(&governing_observed(b, rows)).then(a.id.cmp(b.id))
    })
}

/// Re-derive a claim's slot (primary and secondary) from the bases it carries.
///
/// A slot without a basis re-derives to `not_measured` / `calibration_basis_missing`.
/// Caller-attested evidence that records `attested_not_reverifiable` is
/// uncalibrated by construction, not a missing match key.
#[must_use]
pub fn rederive_calibration(slot: &CalibrationSlotWire) -> CalibrationSlotWire {
    let Some(primary) = slot.basis.as_ref() else {
        if slot.status == "not_measured"
            && slot.reason.as_deref()
                == Some(antecedent_core::reason_code!("attested_not_reverifiable"))
        {
            return CalibrationSlotWire::not_measured("attested_not_reverifiable");
        }
        return CalibrationSlotWire::not_measured(BASIS_MISSING);
    };
    let mut bases = vec![primary.clone()];
    for secondary in &slot.secondary {
        let Some(reported) = secondary.basis.as_ref() else {
            let mut out = calibration_slots(&bases);
            out.secondary.push(CalibrationSlotWire::not_measured(BASIS_MISSING));
            return out;
        };
        bases.push(reported.clone());
    }
    calibration_slots(&bases)
}
