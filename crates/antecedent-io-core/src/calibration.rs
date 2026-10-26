//! Calibration match: one reported interval against the coverage records.
//!
//! The only implementation of the rule. The runtime claim
//! (`antecedent::StudyResult::claim`) and the independent consumer
//! ([`crate::verify_contract_against_body`]) both call [`calibration_slot`]
//! on the same [`CalibrationBasisWire`], which the claim carries and
//! `claim_id` covers. Records come from the generated
//! [`crate::coverage_records_data::RECORDS`] table
//! (`parity/coverage_records.toml`, emitted by the coverage tests through
//! `CoverageTally::for_record` and collected by
//! `scripts/collect_coverage_records.py`).
//!
//! Vocabulary:
//!
//! - `calibrated`: a non-boundary record that still attests the current code
//!   (its facets have not drifted since `calibration_sha`) measured this
//!   construction (every key field equal, including the nominal level and the
//!   identification label) and the execution lies inside what it measured: its
//!   row count inside the record's measured row-count range (the span of its
//!   sample-size grid points, never extrapolated beyond them), at least as many
//!   successful resampling replicates and posterior draws, and no more
//!   non-identified mass. A record is a boundary when any of its grid points
//!   is: a construction that failed at one measured sample size is not
//!   calibrated anywhere in the range, except inside a
//!   [`confirmed_passing_region`]: a contiguous run of grid points that passed
//!   in the measurement and again in an independent re-measurement with fresh
//!   seeds. There the slot is `calibrated`, governed by the worst of those
//!   points; a row count in the failing region still refuses. A matching but
//!   stale (non-attesting) record is never `calibrated`.
//! - `scope_not_assessed`: a record measured the construction but the
//!   execution is outside its scope (the reason names which bound), the
//!   covering record is a named boundary / under-coverage measurement
//!   (`boundary_record`, with its observed coverage), or the covering record
//!   no longer attests the current code (`coverage_record_not_attesting`).
//! - `unavailable`: no record measured this construction, level or
//!   identification; or no interval was reported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::contract_section::{CalibrationSlotWire, CalibrationStatus};
use crate::coverage_records_data::{
    ATTESTING_RECORD_IDS, CoverageGridPoint, CoverageRecord, FUNCTIONAL_IDENTITY_ALIASES,
    FunctionalIdentityAlias, RECORDS,
};

/// No interval was reported.
pub const NO_INTERVAL_REPORTED: &str = "no_interval_reported";
/// No coverage record measured this construction.
pub const CONSTRUCTION_NOT_MEASURED: &str = "estimator_grid_not_measured";
/// Records measured the construction, but at another nominal level.
pub const LEVEL_NOT_MEASURED: &str = "interval_level_not_measured";
/// Records measured the construction, but under another identification label.
pub const IDENTIFICATION_NOT_MEASURED: &str = "identification_not_measured";
/// The covering record is a named boundary (under-coverage) measurement.
pub const BOUNDARY_RECORD: &str = "boundary_record";
/// The execution's row count lies outside every measured range.
pub const SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE: &str = "sample_size_outside_measured_range";
/// Fewer resampling replicates succeeded than the record measured.
pub const REPLICATES_BELOW_MEASURED: &str = "resampling_replicates_below_measured";
/// Fewer posterior draws than the record measured.
pub const POSTERIOR_DRAWS_BELOW_MEASURED: &str = "posterior_draws_below_measured";
/// More non-identified structural mass than the record measured.
pub const UNIDENTIFIED_MASS_ABOVE_MEASURED: &str = "unidentified_mass_above_measured";
/// The claim carries no match basis to re-derive the slot from.
pub const BASIS_MISSING: &str = "calibration_basis_missing";
/// A covering record matches, but no longer attests the current statistical
/// surface (facets drifted since its `calibration_sha`, or it was never
/// attested for this tree).
pub const RECORD_NOT_ATTESTING: &str = "coverage_record_not_attesting";

/// Every reason code the matcher can emit (all listed in `parity/reason_codes.toml`,
/// except [`RECORD_NOT_ATTESTING`], which is runtime-only).
pub const REASON_CODES: [&str; 11] = [
    NO_INTERVAL_REPORTED,
    CONSTRUCTION_NOT_MEASURED,
    LEVEL_NOT_MEASURED,
    IDENTIFICATION_NOT_MEASURED,
    BOUNDARY_RECORD,
    SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE,
    REPLICATES_BELOW_MEASURED,
    POSTERIOR_DRAWS_BELOW_MEASURED,
    UNIDENTIFIED_MASS_ABOVE_MEASURED,
    BASIS_MISSING,
    RECORD_NOT_ATTESTING,
];

/// Whether `record` attests the code this binary was built from.
///
/// A `calibrated` slot requires this to be true for the governing record. The
/// answer is generated with the registry ([`ATTESTING_RECORD_IDS`], from
/// `scripts/calibration_facets.py`), so it changes when a record is re-measured
/// or the statistical surface drifts, and never by hand. Stale or otherwise
/// non-attesting records may still be reported as `scope_not_assessed` with
/// [`RECORD_NOT_ATTESTING`].
#[must_use]
pub fn record_attests_current_code(record: &CoverageRecord) -> bool {
    attesting_ids().binary_search(&record.id).is_ok()
}

fn attesting_ids() -> &'static [&'static str] {
    static SORTED: OnceLock<Vec<&'static str>> = OnceLock::new();
    SORTED
        .get_or_init(|| {
            let mut ids: Vec<&str> = ATTESTING_RECORD_IDS.iter().copied().collect();
            ids.sort_unstable();
            ids
        })
        .as_slice()
}

/// Tolerance for comparing nominal levels and structural masses.
const TOLERANCE: f64 = 1e-9;

/// Construction of one reported interval: the fields a coverage record must
/// equal to describe it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationKeyWire {
    /// Support-matrix query axis name.
    pub query: String,
    /// Support-matrix graph axis, including `CoDetermined` / `Unknown`.
    pub graph_class: String,
    /// `fixed` (explicit or accepted structure) or `graph_posterior`.
    pub structure: String,
    /// Data modality the execution ran on (`tabular`, `series`, `panel`,
    /// `event`, `multi_env`): a record binds only to the modality it measured.
    pub modality: String,
    /// `Frequentist` or `Bayesian`.
    pub inference: String,
    /// Resolved plan estimator (`logical_plan.estimator`); empty when none.
    pub estimator: String,
    /// `IntervalMethod::as_str()`.
    pub interval_method: String,
    /// Analytic SE kind recorded by the estimator; empty when not analytic.
    pub se_kind: String,
    /// `iid`, `panel_cluster`, `circular_block:<family>`, or, for a frequentist standard error on
    /// time-ordered rows that names no block family, `serial_unmodelled` / `serial_hac`.
    pub dependence: String,
    /// Posterior construction (`<backend>.<likelihood>.<prior>`); empty for
    /// Frequentist executions.
    pub posterior: String,
    /// Target population, outcome functional, contrast, horizon and policy
    /// the support axis query name leaves open.
    pub functional: String,
    /// Nominal level of the reported interval.
    pub level: f64,
    /// `point` (all structural mass identified under an identified status) or
    /// `partial`, followed by `+<rule>` for every derivation rule in
    /// [`CONSTRUCTION_RULES`] the identification used ([`identification_key`]):
    /// an interval over a different identified adjustment is a different
    /// construction and binds only to records measured for it.
    pub identification: String,
}

/// Derivation rules that make an identification a distinct interval
/// construction for coverage matching. A parent-adjusted temporal pulse
/// (`temporal.parent_adjustment`) fits a different adjustment set than the
/// unfolding-derived one on the same query, so its coverage is separate
/// evidence.
pub const CONSTRUCTION_RULES: &[&str] = &["temporal.parent_adjustment"];

/// The calibration key's `identification` field: the `point` / `partial`
/// label plus `+<rule>` for each [`CONSTRUCTION_RULES`] entry among
/// `derivation_rules`, in [`CONSTRUCTION_RULES`] order.
#[must_use]
pub fn identification_key<'a>(
    label: &str,
    derivation_rules: impl IntoIterator<Item = &'a str>,
) -> String {
    let used: Vec<&str> = derivation_rules.into_iter().collect();
    let mut key = label.to_string();
    for rule in CONSTRUCTION_RULES {
        if used.contains(rule) {
            key.push('+');
            key.push_str(rule);
        }
    }
    key
}

/// Execution facts a record's scope is checked against.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationScopeWire {
    /// Analysis-sample rows that informed the interval, not the raw snapshot.
    pub row_count: u64,
    /// Resampling replicates that succeeded, when resampling-based.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicates_ok: Option<u32>,
    /// Posterior draws, when posterior-based.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub posterior_draws: Option<u32>,
    /// Structural mass that is not identified (unidentified + unevaluable +
    /// incomplete search).
    pub unidentified_mass: f64,
}

/// Match key plus scope facts of one reported interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationBasisWire {
    /// Construction.
    pub key: CalibrationKeyWire,
    /// Execution facts.
    pub scope: CalibrationScopeWire,
}

impl From<&antecedent_core::CalibrationBasis> for CalibrationBasisWire {
    fn from(basis: &antecedent_core::CalibrationBasis) -> Self {
        Self {
            key: CalibrationKeyWire {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                level: basis.level,
                identification: basis.identification.to_string(),
            },
            scope: CalibrationScopeWire {
                row_count: basis.row_count,
                replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws,
                unidentified_mass: basis.unidentified_mass,
            },
        }
    }
}

/// Whether the structure behind a calibrated interval was fixed or learned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationStructure {
    /// Explicit or accepted structure.
    Fixed,
    /// Averaged over a graph posterior.
    GraphPosterior,
}

impl CalibrationStructure {
    /// The key spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::GraphPosterior => "graph_posterior",
        }
    }

    /// Classify a support-matrix structure axis name.
    #[must_use]
    pub fn from_axis(name: &str) -> Self {
        if name == Self::GraphPosterior.as_str() { Self::GraphPosterior } else { Self::Fixed }
    }
}

/// Inference family recorded in a calibration key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationInference {
    /// Frequentist inference.
    Frequentist,
    /// Bayesian inference.
    Bayesian,
}

impl CalibrationInference {
    /// The key spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Frequentist => "Frequentist",
            Self::Bayesian => "Bayesian",
        }
    }

    /// Classify an inference label (any ASCII case of `bayesian` is Bayesian).
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        if label.eq_ignore_ascii_case("bayesian") { Self::Bayesian } else { Self::Frequentist }
    }
}

impl CalibrationKeyWire {
    fn same_construction(&self, record: &CoverageRecord) -> bool {
        self.same_construction_with_aliases(record, &FUNCTIONAL_IDENTITY_ALIASES)
    }

    fn same_construction_with_aliases(
        &self,
        record: &CoverageRecord,
        aliases: &[FunctionalIdentityAlias],
    ) -> bool {
        record.query == self.query
            && record.graph_class == self.graph_class
            && record.structure == self.structure
            && record.modality == self.modality
            && record.inference == self.inference
            && record.estimator == self.estimator
            && record.interval_method == self.interval_method
            && record.se_kind == self.se_kind
            && record.dependence == self.dependence
            && record.posterior == self.posterior
            && functional_matches(record, &self.functional, aliases)
    }
}

fn functional_matches(
    record: &CoverageRecord,
    functional: &str,
    aliases: &[FunctionalIdentityAlias],
) -> bool {
    record.functional == functional
        || aliases.iter().any(|alias| {
            alias.record_id == record.id
                && alias.measurement_sha == record.calibration_sha
                && alias.original_functional == record.functional
                && alias.canonical_functional == functional
        })
}

/// Calibration slot of one reported interval (secondary list empty).
///
/// Uses the shipped [`RECORDS`] table and requires the governing record to
/// [`record_attests_current_code`].
#[must_use]
pub fn calibration_slot(basis: &CalibrationBasisWire) -> CalibrationSlotWire {
    let candidates = shipped_candidates(&basis.key);
    calibration_slot_with(basis, &candidates, record_attests_current_code)
}

/// Records in the shipped table that share every construction axis except the
/// nominal level and the identification label. Those two are filtered by
/// [`calibration_slot_with`], so the candidate set is exactly the set that
/// function used to scan the whole table for.
fn shipped_candidates(key: &CalibrationKeyWire) -> Vec<CoverageRecord> {
    construction_index()
        .get(&construction_axis(key))
        .map(|indexes| {
            indexes
                .iter()
                .map(|&index| RECORDS[index])
                .filter(|record| {
                    functional_matches(record, &key.functional, &FUNCTIONAL_IDENTITY_ALIASES)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn construction_axis(key: &CalibrationKeyWire) -> String {
    [
        key.query.as_str(),
        key.graph_class.as_str(),
        key.structure.as_str(),
        key.modality.as_str(),
        key.inference.as_str(),
        key.estimator.as_str(),
        key.interval_method.as_str(),
        key.se_kind.as_str(),
        key.dependence.as_str(),
        key.posterior.as_str(),
    ]
    .join("\u{1f}")
}

fn construction_index() -> &'static HashMap<String, Vec<usize>> {
    static INDEX: OnceLock<HashMap<String, Vec<usize>>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut index: HashMap<String, Vec<usize>> = HashMap::new();
        for (offset, record) in RECORDS.iter().enumerate() {
            let axis = [
                record.query,
                record.graph_class,
                record.structure,
                record.modality,
                record.inference,
                record.estimator,
                record.interval_method,
                record.se_kind,
                record.dependence,
                record.posterior,
            ]
            .join("\u{1f}");
            index.entry(axis).or_default().push(offset);
        }
        index
    })
}

/// Calibration slot of the primary interval, with one secondary slot per
/// further reported interval, in order.
#[must_use]
pub fn calibration_slots(bases: &[CalibrationBasisWire]) -> CalibrationSlotWire {
    let Some((primary, rest)) = bases.split_first() else {
        return CalibrationSlotWire::unavailable(BASIS_MISSING);
    };
    let mut slot = calibration_slot(primary);
    slot.secondary = rest.iter().map(calibration_slot).collect();
    slot
}

/// [`calibration_slot`] against an explicit record table.
///
/// Caller-supplied tables are treated as attesting so match-logic unit tests
/// can pin construction and scope without restamping attestation. Production
/// claim binding goes through [`calibration_slot`], which applies
/// [`record_attests_current_code`]. Use [`calibration_slot_with`] to pin
/// attestation itself.
#[must_use]
pub fn calibration_slot_in(
    basis: &CalibrationBasisWire,
    records: &[CoverageRecord],
) -> CalibrationSlotWire {
    calibration_slot_with(basis, records, |_| true)
}

/// [`calibration_slot_in`] with an explicit attestation predicate.
///
/// `attests` decides whether a covering record may yield `calibrated`. A
/// matching record that fails `attests` is `scope_not_assessed` with
/// [`RECORD_NOT_ATTESTING`].
#[must_use]
pub fn calibration_slot_with(
    basis: &CalibrationBasisWire,
    records: &[CoverageRecord],
    attests: impl Fn(&CoverageRecord) -> bool,
) -> CalibrationSlotWire {
    let key = &basis.key;
    let scope = &basis.scope;
    let unavailable = |reason: &str| CalibrationSlotWire::unavailable(reason).with_basis(basis);
    if key.interval_method == "none" {
        return unavailable(NO_INTERVAL_REPORTED);
    }
    let construction: Vec<&CoverageRecord> =
        records.iter().filter(|record| key.same_construction(record)).collect();
    if construction.is_empty() {
        return unavailable(CONSTRUCTION_NOT_MEASURED);
    }
    let at_level: Vec<&CoverageRecord> = construction
        .into_iter()
        .filter(|record| (record.nominal - key.level).abs() <= TOLERANCE)
        .collect();
    if at_level.is_empty() {
        return unavailable(LEVEL_NOT_MEASURED);
    }
    let measured: Vec<&CoverageRecord> =
        at_level.into_iter().filter(|record| record.identification == key.identification).collect();
    if measured.is_empty() {
        return unavailable(IDENTIFICATION_NOT_MEASURED);
    }
    let covering: Vec<&CoverageRecord> = measured
        .iter()
        .copied()
        .filter(|record| {
            let (lo, hi) = measured_range(record);
            lo <= scope.row_count && scope.row_count <= hi
        })
        .collect();
    let rows = scope.row_count;
    let Some(governing) = worst(&covering, rows) else {
        let nearest = measured
            .iter()
            .copied()
            .min_by_key(|record| {
                let (lo, hi) = measured_range(record);
                (lo.saturating_sub(scope.row_count)).max(scope.row_count.saturating_sub(hi))
            })
            .expect("measured is non-empty");
        return CalibrationSlotWire::from_record(nearest, CalibrationStatus::ScopeNotAssessed)
            .with_reason(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE)
            .with_basis(basis);
    };
    let boundaries: Vec<&CoverageRecord> = covering
        .iter()
        .copied()
        .filter(|record| is_boundary(record) && confirmed_passing_region(record, rows).is_none())
        .collect();
    if let Some(boundary) = worst(&boundaries, rows) {
        let mut slot =
            CalibrationSlotWire::from_record(boundary, CalibrationStatus::ScopeNotAssessed)
                .with_reason(BOUNDARY_RECORD)
                .with_basis(basis);
        // The coverage that makes it a boundary: the worst failing grid point.
        if let Some(point) = boundary
            .grid
            .iter()
            .filter(|point| point.boundary)
            .min_by(|a, b| a.observed.total_cmp(&b.observed))
        {
            slot.observed = Some(point.observed);
        }
        return slot;
    }
    let outside = if covering.iter().any(|record| {
        record.replicates_min > 0 && scope.replicates_ok.unwrap_or(0) < record.replicates_min
    }) {
        Some(REPLICATES_BELOW_MEASURED)
    } else if covering.iter().any(|record| {
        record.posterior_draws_min > 0
            && scope.posterior_draws.unwrap_or(0) < record.posterior_draws_min
    }) {
        Some(POSTERIOR_DRAWS_BELOW_MEASURED)
    } else if covering
        .iter()
        .any(|record| scope.unidentified_mass > record.unidentified_mass_max + TOLERANCE)
    {
        Some(UNIDENTIFIED_MASS_ABOVE_MEASURED)
    } else if !attests(governing) {
        Some(RECORD_NOT_ATTESTING)
    } else {
        None
    };
    if let Some(reason) = outside {
        return CalibrationSlotWire::from_record(governing, CalibrationStatus::ScopeNotAssessed)
            .with_reason(reason)
            .with_basis(basis);
    }
    let mut slot = CalibrationSlotWire::from_record(governing, CalibrationStatus::Calibrated)
        .with_basis(basis);
    if let Some(region) = confirmed_passing_region(governing, rows) {
        slot.scope_n = Some(region.n_min);
        slot.scope_n_max = Some(region.n_max);
        slot.observed = Some(region.observed);
    }
    slot
}

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
    if let (Some(first), Some(last)) = (passing.first(), passing.last()) {
        if first.n_min <= rows && rows <= last.n_max {
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
fn worst<'a>(records: &[&'a CoverageRecord], rows: u64) -> Option<&'a CoverageRecord> {
    records.iter().copied().min_by(|a, b| {
        governing_observed(a, rows).total_cmp(&governing_observed(b, rows)).then(a.id.cmp(b.id))
    })
}

/// Re-derive a claim's slot (primary and secondary) from the bases it carries.
///
/// A slot without a basis re-derives to `unavailable` / `calibration_basis_missing`.
/// Caller-attested evidence that records `attested_not_reverifiable` is
/// uncalibrated by construction, not a missing match key.
#[must_use]
pub fn rederive_calibration(slot: &CalibrationSlotWire) -> CalibrationSlotWire {
    let Some(primary) = slot.basis.as_ref() else {
        if slot.status == "unavailable"
            && slot.reason.as_deref()
                == Some(antecedent_core::reason_code!("attested_not_reverifiable"))
        {
            return CalibrationSlotWire::unavailable("attested_not_reverifiable");
        }
        return CalibrationSlotWire::unavailable(BASIS_MISSING);
    };
    let mut bases = vec![primary.clone()];
    for secondary in &slot.secondary {
        let Some(reported) = secondary.basis.as_ref() else {
            let mut out = calibration_slots(&bases);
            out.secondary.push(CalibrationSlotWire::unavailable(BASIS_MISSING));
            return out;
        };
        bases.push(reported.clone());
    }
    calibration_slots(&bases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_status_keeps_historical_cbor_text() {
        for status in [
            CalibrationStatus::Calibrated,
            CalibrationStatus::ScopeNotAssessed,
            CalibrationStatus::Unavailable,
        ] {
            crate::convert::assert_cbor_text_code(status.as_str(), &status);
        }
    }

    #[test]
    fn basis_wire_carries_every_field_of_the_basis() {
        let names = ["q", "g", "fixed", "tabular", "Bayesian", "e", "m", "s", "d", "p", "f"];
        let basis = antecedent_core::CalibrationBasis::new(
            names.map(std::sync::Arc::<str>::from),
            0.95,
            std::sync::Arc::from("point"),
            120,
            Some(7),
            Some(9),
            0.25,
        );
        let wire = CalibrationBasisWire::from(&basis);
        let key = &wire.key;
        assert_eq!(
            [
                key.query.as_str(),
                key.graph_class.as_str(),
                key.structure.as_str(),
                key.modality.as_str(),
                key.inference.as_str(),
                key.estimator.as_str(),
                key.interval_method.as_str(),
                key.se_kind.as_str(),
                key.dependence.as_str(),
                key.posterior.as_str(),
                key.functional.as_str(),
            ],
            names
        );
        assert_eq!((key.level, key.identification.as_str()), (0.95, "point"));
        assert_eq!(wire.scope.row_count, 120);
        assert_eq!((wire.scope.replicates_ok, wire.scope.posterior_draws), (Some(7), Some(9)));
        assert_eq!(wire.scope.unidentified_mass, 0.25);
    }

    #[test]
    fn structure_and_inference_spell_their_keys() {
        assert_eq!(CalibrationStructure::from_axis("graph_posterior").as_str(), "graph_posterior");
        assert_eq!(CalibrationStructure::from_axis("anything else").as_str(), "fixed");
        assert_eq!(CalibrationInference::from_label("BAYESIAN"), CalibrationInference::Bayesian);
        assert_eq!(CalibrationInference::from_label("Bayesian").as_str(), "Bayesian");
        assert_eq!(CalibrationInference::from_label("frequentist").as_str(), "Frequentist");
    }

    fn key_from(record: &CoverageRecord) -> CalibrationKeyWire {
        CalibrationKeyWire {
            query: record.query.to_owned(),
            graph_class: record.graph_class.to_owned(),
            structure: record.structure.to_owned(),
            modality: record.modality.to_owned(),
            inference: record.inference.to_owned(),
            estimator: record.estimator.to_owned(),
            interval_method: record.interval_method.to_owned(),
            se_kind: record.se_kind.to_owned(),
            dependence: record.dependence.to_owned(),
            posterior: record.posterior.to_owned(),
            functional: record.functional.to_owned(),
            level: record.nominal,
            identification: record.identification.to_owned(),
        }
    }

    fn matching_ids(key: &CalibrationKeyWire) -> Vec<&'static str> {
        RECORDS
            .iter()
            .filter(|record| key.same_construction(record))
            .map(|record| record.id)
            .collect()
    }

    #[test]
    fn construction_index_matches_a_linear_scan() {
        for record in &RECORDS {
            let key = key_from(record);
            let indexed: Vec<_> = shipped_candidates(&key)
                .iter()
                .filter(|candidate| key.same_construction(candidate))
                .map(|candidate| candidate.id)
                .collect();
            assert_eq!(indexed, matching_ids(&key), "{}", record.id);
        }
        for alias in &FUNCTIONAL_IDENTITY_ALIASES {
            let Some(record) = RECORDS.iter().find(|record| record.id == alias.record_id) else {
                continue;
            };
            let mut key = key_from(record);
            key.functional = alias.canonical_functional.to_owned();
            let indexed: Vec<_> = shipped_candidates(&key)
                .iter()
                .filter(|candidate| key.same_construction(candidate))
                .map(|candidate| candidate.id)
                .collect();
            assert_eq!(indexed, matching_ids(&key), "{}", alias.record_id);
        }
    }

    #[test]
    fn attesting_index_matches_linear_membership() {
        for record in &RECORDS {
            assert_eq!(
                record_attests_current_code(record),
                ATTESTING_RECORD_IDS.contains(&record.id),
                "{}",
                record.id
            );
        }
        assert!(attesting_ids().binary_search(&"not-a-coverage-record").is_err());
    }

    fn record(id: &'static str) -> CoverageRecord {
        CoverageRecord {
            id,
            query: "AverageEffect",
            graph_class: "Dag",
            structure: "fixed",
            modality: "tabular",
            inference: "Frequentist",
            estimator: "aipw",
            interval_method: "bootstrap_se",
            se_kind: "",
            dependence: "iid",
            posterior: "",
            functional: "all_observed.mean",
            identification: "point",
            nominal: 0.95,
            n_min: 500,
            n_max: 500,
            replicates_min: 200,
            posterior_draws_min: 0,
            unidentified_mass_max: 0.0,
            observed: 0.948,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
            grid: &[],
            confirmation: &[],
            confirmed_passing_range: None,
            dgp: "crates/antecedent/tests/x.rs::dgp",
            test: "crates/antecedent/tests/x.rs::test",
            calibration_sha: "0123456789abcdef0123456789abcdef01234567",
        }
    }

    fn basis() -> CalibrationBasisWire {
        CalibrationBasisWire {
            key: CalibrationKeyWire {
                query: "AverageEffect".into(),
                graph_class: "Dag".into(),
                structure: "fixed".into(),
                modality: "tabular".into(),
                inference: "Frequentist".into(),
                estimator: "aipw".into(),
                interval_method: "bootstrap_se".into(),
                se_kind: String::new(),
                dependence: "iid".into(),
                posterior: String::new(),
                functional: "all_observed.mean".into(),
                level: 0.95,
                identification: "point".into(),
            },
            scope: CalibrationScopeWire {
                row_count: 500,
                replicates_ok: Some(200),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        }
    }

    #[test]
    fn functional_alias_is_bound_to_exact_original_record_identity() {
        let measured = record("cov.a");
        let alias = FunctionalIdentityAlias {
            record_id: measured.id,
            measurement_sha: measured.calibration_sha,
            original_functional: measured.functional,
            canonical_functional: "canonical_recovery_functional",
        };
        assert!(functional_matches(&measured, alias.canonical_functional, &[alias]));
        assert!(functional_matches(&measured, measured.functional, &[]));
        assert!(!functional_matches(&measured, "foreign_functional", &[alias]));
        assert!(!functional_matches(&record("cov.foreign"), alias.canonical_functional, &[alias]));
        let mut changed = measured;
        changed.calibration_sha = "foreign_measurement";
        assert!(!functional_matches(&changed, alias.canonical_functional, &[alias]));
        changed = measured;
        changed.functional = "changed_original_functional";
        assert!(!functional_matches(&changed, alias.canonical_functional, &[alias]));
        let mut actual = basis();
        actual.key.functional = alias.canonical_functional.into();
        assert!(actual.key.same_construction_with_aliases(&measured, &[alias]));
        assert!(!actual.key.same_construction_with_aliases(&measured, &[]));
        for axis in 0..10 {
            let mut incompatible = actual.clone();
            match axis {
                0 => incompatible.key.query = "foreign".into(),
                1 => incompatible.key.graph_class = "foreign".into(),
                2 => incompatible.key.structure = "foreign".into(),
                3 => incompatible.key.modality = "foreign".into(),
                4 => incompatible.key.inference = "foreign".into(),
                5 => incompatible.key.estimator = "foreign".into(),
                6 => incompatible.key.interval_method = "foreign".into(),
                7 => incompatible.key.se_kind = "foreign".into(),
                8 => incompatible.key.dependence = "foreign".into(),
                _ => incompatible.key.posterior = "foreign".into(),
            }
            assert!(!incompatible.key.same_construction_with_aliases(&measured, &[alias]));
        }
    }

    #[test]
    fn inside_scope_is_calibrated_against_the_worst_covering_record() {
        let mut low = record("cov.low");
        low.observed = 0.94;
        let slot = calibration_slot_in(&basis(), &[record("cov.high"), low]);
        assert_eq!(slot.status, "calibrated");
        assert_eq!(slot.record_id.as_deref(), Some("cov.low"));
        assert_eq!(slot.observed, Some(0.94));
        assert_eq!(slot.reason, None);
    }

    #[test]
    fn too_few_successful_replicates_are_not_calibrated() {
        let mut b = basis();
        b.scope.replicates_ok = Some(2);
        let slot = calibration_slot_in(&b, &[record("cov.a")]);
        assert_eq!(slot.status, "scope_not_assessed");
        assert_eq!(slot.reason.as_deref(), Some(REPLICATES_BELOW_MEASURED));
    }

    #[test]
    fn level_and_identification_mismatches_are_unavailable() {
        let mut b = basis();
        b.key.level = 0.9;
        assert_eq!(
            calibration_slot_in(&b, &[record("cov.a")]).reason.as_deref(),
            Some(LEVEL_NOT_MEASURED)
        );
        let mut b = basis();
        b.key.identification = "partial".into();
        let slot = calibration_slot_in(&b, &[record("cov.a")]);
        assert_eq!(slot.status, "unavailable");
        assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
    }

    #[test]
    fn parent_adjustment_binds_only_to_records_measured_for_it() {
        assert_eq!(identification_key("point", ["backdoor.criterion", "temporal.unfold"]), "point");
        let parent = identification_key(
            "point",
            ["temporal.parent_adjustment", "backdoor.adjustment_set", "temporal.unfold"],
        );
        assert_eq!(parent, "point+temporal.parent_adjustment");

        let unfolding = record("cov.unfolding");
        let mut measured = record("cov.parent");
        measured.identification = "point+temporal.parent_adjustment";

        let mut b = basis();
        b.key.identification = parent;
        let slot = calibration_slot_in(&b, &[unfolding]);
        assert_eq!(slot.status, "unavailable");
        assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
        let slot = calibration_slot_in(&b, &[unfolding, measured]);
        assert_eq!(slot.status, "calibrated");
        assert_eq!(slot.record_id.as_deref(), Some("cov.parent"));

        // And an unfolding-identified result never binds to the parent record.
        let slot = calibration_slot_in(&basis(), &[measured]);
        assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
    }

    #[test]
    fn row_count_outside_the_measured_range_is_not_assessed() {
        let mut b = basis();
        b.scope.row_count = 1200;
        let slot = calibration_slot_in(&b, &[record("cov.a")]);
        assert_eq!(slot.status, "scope_not_assessed");
        assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
        assert_eq!(slot.scope_n, Some(500));
    }

    const GRID: [CoverageGridPoint; 3] = [
        CoverageGridPoint {
            point: 0,
            n_min: 250,
            n_max: 250,
            observed: 0.945,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
        },
        CoverageGridPoint {
            point: 1,
            n_min: 500,
            n_max: 500,
            observed: 0.948,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
        },
        CoverageGridPoint {
            point: 2,
            n_min: 1000,
            n_max: 1000,
            observed: 0.951,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
        },
    ];
    static FAILING_SMALL: [CoverageGridPoint; 3] = {
        let mut grid = GRID;
        grid[0].observed = 0.90;
        grid[0].boundary = true;
        grid
    };

    fn gridded(id: &'static str, grid: &'static [CoverageGridPoint]) -> CoverageRecord {
        let mut out = record(id);
        out.n_min = 250;
        out.n_max = 1000;
        out.grid = grid;
        out
    }

    #[test]
    fn a_passing_grid_is_calibrated_across_its_whole_measured_range() {
        for n in [250, 251, 500, 777, 1000] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[gridded("cov.grid", &GRID)]);
            assert_eq!(slot.status, "calibrated", "n = {n}: {slot:?}");
            assert_eq!((slot.scope_n, slot.scope_n_max), (Some(250), Some(1000)));
        }
        for n in [249, 1001, 5000] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[gridded("cov.grid", &GRID)]);
            assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
        }
    }

    #[test]
    fn the_range_is_the_grid_span_not_a_contradicting_summary() {
        let mut stale = gridded("cov.grid", &GRID);
        stale.n_min = 500;
        stale.n_max = 500;
        let mut b = basis();
        b.scope.row_count = 300;
        assert_eq!(calibration_slot_in(&b, &[stale]).status, "calibrated");
    }

    #[test]
    fn a_grid_that_fails_at_any_point_is_a_boundary_over_the_whole_range() {
        // Even at the passing base point: a failure at 250 rows is not averaged
        // into a pass at 500.
        for n in [250, 500, 1000] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[gridded("cov.grid", &FAILING_SMALL)]);
            assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
            assert_eq!(slot.observed, Some(0.90), "the failing point's coverage is reported");
        }
    }

    const CONFIRMED_UPPER: [CoverageGridPoint; 2] = {
        let mut second = GRID[1];
        second.observed = 0.944;
        let mut third = GRID[2];
        third.observed = 0.957;
        [second, third]
    };

    fn recovered(confirmation: &'static [CoverageGridPoint]) -> CoverageRecord {
        let mut out = gridded("cov.recovered", &FAILING_SMALL);
        out.observed = 0.90;
        out.boundary = true;
        out.confirmation = confirmation;
        out.confirmed_passing_range = Some((500, 1000));
        out
    }

    #[test]
    fn a_confirmed_upper_range_is_calibrated_by_its_worst_passing_point() {
        for n in [500, 777, 1000] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[recovered(&CONFIRMED_UPPER)]);
            assert_eq!(slot.status, "calibrated", "n = {n}: {slot:?}");
            assert_eq!(slot.reason, None);
            assert_eq!((slot.scope_n, slot.scope_n_max), (Some(500), Some(1000)));
            // The measurement's 0.948 and 0.951 and the confirmation's 0.944 and
            // 0.957: the lowest of them, never the failing point's 0.90.
            assert_eq!(slot.observed, Some(0.944));
        }
    }

    #[test]
    fn a_confirmed_range_leaves_the_failing_region_refused() {
        for n in [250, 251, 499] {
            let mut b = basis();
            b.scope.row_count = n;
            let slot = calibration_slot_in(&b, &[recovered(&CONFIRMED_UPPER)]);
            assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
            assert_eq!(slot.observed, Some(0.90), "the failing point stays on the record");
        }
        let mut b = basis();
        b.scope.row_count = 1001;
        assert_eq!(
            calibration_slot_in(&b, &[recovered(&CONFIRMED_UPPER)]).reason.as_deref(),
            Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE)
        );
    }

    #[test]
    fn a_range_without_a_passing_confirmation_of_every_point_is_not_recovered() {
        static ONE_POINT: [CoverageGridPoint; 1] = [CONFIRMED_UPPER[0]];
        static FAILED_CONFIRMATION: [CoverageGridPoint; 2] = {
            let mut points = CONFIRMED_UPPER;
            points[1].boundary = true;
            points[1].observed = 0.91;
            points
        };
        static SHIFTED_CONFIRMATION: [CoverageGridPoint; 2] = {
            let mut points = CONFIRMED_UPPER;
            points[1].n_min = 900;
            points[1].n_max = 900;
            points
        };
        let cases: [&'static [CoverageGridPoint]; 4] =
            [&[], &ONE_POINT, &FAILED_CONFIRMATION, &SHIFTED_CONFIRMATION];
        for confirmation in cases {
            let mut b = basis();
            b.scope.row_count = 800;
            let slot = calibration_slot_in(&b, &[recovered(confirmation)]);
            assert_eq!(slot.status, "scope_not_assessed", "{confirmation:?}");
            assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
        }
    }

    #[test]
    fn a_confirmed_range_never_covers_a_failing_grid_point() {
        // A range that reaches back over the failing point is not a passing
        // region, whatever the record states.
        let mut forged = recovered(&CONFIRMED_UPPER);
        forged.confirmed_passing_range = Some((250, 1000));
        let mut b = basis();
        b.scope.row_count = 800;
        assert_eq!(calibration_slot_in(&b, &[forged]).reason.as_deref(), Some(BOUNDARY_RECORD));
    }

    #[test]
    fn a_confirmed_range_still_requires_attestation_and_every_other_axis() {
        let record = recovered(&CONFIRMED_UPPER);
        let mut b = basis();
        b.scope.row_count = 800;
        let stale = calibration_slot_with(&b, &[record], |_| false);
        assert_eq!(stale.reason.as_deref(), Some(RECORD_NOT_ATTESTING));
        let mut other_level = b.clone();
        other_level.key.level = 0.9;
        assert_eq!(
            calibration_slot_in(&other_level, &[record]).reason.as_deref(),
            Some(LEVEL_NOT_MEASURED)
        );
        let mut other_dependence = b.clone();
        other_dependence.key.dependence = "panel_cluster".into();
        assert_eq!(
            calibration_slot_in(&other_dependence, &[record]).reason.as_deref(),
            Some(CONSTRUCTION_NOT_MEASURED)
        );
        let mut other_identification = b;
        other_identification.key.identification = "partial".into();
        assert_eq!(
            calibration_slot_in(&other_identification, &[record]).reason.as_deref(),
            Some(IDENTIFICATION_NOT_MEASURED)
        );
    }

    #[test]
    fn a_covering_boundary_record_is_reported_with_its_coverage() {
        let mut boundary = record("cov.boundary");
        boundary.boundary = true;
        boundary.observed = 0.885;
        let slot = calibration_slot_in(&basis(), &[record("cov.a"), boundary]);
        assert_eq!(slot.status, "scope_not_assessed");
        assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
        assert_eq!(slot.record_id.as_deref(), Some("cov.boundary"));
        assert_eq!(slot.observed, Some(0.885));
    }

    #[test]
    fn no_interval_and_no_record_are_unavailable_with_codes() {
        let mut b = basis();
        b.key.interval_method = "none".into();
        assert_eq!(
            calibration_slot_in(&b, &[record("cov.a")]).reason.as_deref(),
            Some(NO_INTERVAL_REPORTED)
        );
        assert_eq!(
            calibration_slot_in(&basis(), &[]).reason.as_deref(),
            Some(CONSTRUCTION_NOT_MEASURED)
        );
    }

    #[test]
    fn rederive_reproduces_primary_and_secondary_slots() {
        let mut secondary = basis();
        secondary.key.interval_method = "identified_set".into();
        let slot = calibration_slots(&[basis(), secondary]);
        assert_eq!(slot.secondary.len(), 1);
        assert_eq!(rederive_calibration(&slot), slot);
        let mut forged = slot.clone();
        forged.status = "calibrated".into();
        forged.secondary[0].status = "calibrated".into();
        assert_ne!(rederive_calibration(&forged), forged);
    }

    #[test]
    fn calibrated_requires_a_fresh_attesting_record() {
        let fresh = record("cov.fresh");
        let mut stale = record("cov.stale");
        stale.calibration_sha = "stale00000000000000000000000000000000000";
        let attests = |record: &CoverageRecord| record.id == "cov.fresh";

        let fresh_slot = calibration_slot_with(&basis(), &[fresh], attests);
        assert_eq!(fresh_slot.status, "calibrated");
        assert_eq!(fresh_slot.record_id.as_deref(), Some("cov.fresh"));
        assert_eq!(fresh_slot.reason, None);

        let stale_slot = calibration_slot_with(&basis(), &[stale], attests);
        assert_eq!(stale_slot.status, "scope_not_assessed");
        assert_eq!(stale_slot.record_id.as_deref(), Some("cov.stale"));
        assert_eq!(stale_slot.reason.as_deref(), Some(RECORD_NOT_ATTESTING));

        // A measured-looking record the registry does not list as attesting is not
        // calibrated.
        let drifted = record("cov.drifted");
        assert!(!record_attests_current_code(&drifted));
        let drifted_slot = calibration_slot_with(&basis(), &[drifted], record_attests_current_code);
        assert_ne!(drifted_slot.status, "calibrated");
        assert_eq!(drifted_slot.reason.as_deref(), Some(RECORD_NOT_ATTESTING));
    }

    #[test]
    fn attested_external_estimate_stays_uncalibrated_without_a_basis() {
        let slot = CalibrationSlotWire::unavailable("attested_not_reverifiable");
        assert_eq!(rederive_calibration(&slot), slot);
        assert_eq!(
            rederive_calibration(&CalibrationSlotWire::unavailable("forged")).reason.as_deref(),
            Some(BASIS_MISSING)
        );
    }

    // -- recovery of passing subranges on a four-point grid ------------------------------------

    const fn grid_point(point: u8, n: u64, observed: f64, boundary: bool) -> CoverageGridPoint {
        CoverageGridPoint {
            point,
            n_min: n,
            n_max: n,
            observed,
            mcse: 0.011,
            replicates: 2000,
            boundary,
        }
    }

    /// Four sample sizes, the smallest failing.
    const LOWER_FAILS: [CoverageGridPoint; 4] = [
        grid_point(0, 400, 0.902, true),
        grid_point(1, 800, 0.951, false),
        grid_point(2, 1600, 0.949, false),
        grid_point(3, 3200, 0.955, false),
    ];
    /// The same upper three sizes measured again with fresh seeds, all passing.
    const UPPER_CONFIRMED: [CoverageGridPoint; 3] = [
        grid_point(1, 800, 0.944, false),
        grid_point(2, 1600, 0.957, false),
        grid_point(3, 3200, 0.953, false),
    ];

    fn boundary_over(
        grid: &'static [CoverageGridPoint],
        confirmation: &'static [CoverageGridPoint],
        range: Option<(u64, u64)>,
    ) -> CoverageRecord {
        let mut out = record("cov.subrange");
        out.n_min = grid[0].n_min;
        out.n_max = grid[grid.len() - 1].n_max;
        out.observed = grid.iter().map(|p| p.observed).fold(f64::INFINITY, f64::min);
        out.boundary = true;
        out.grid = grid;
        out.confirmation = confirmation;
        out.confirmed_passing_range = range;
        out
    }

    fn at(rows: u64) -> CalibrationBasisWire {
        let mut b = basis();
        b.scope.row_count = rows;
        b
    }

    #[test]
    fn independently_confirmed_upper_subrange_recovered() {
        let subrange = boundary_over(&LOWER_FAILS, &UPPER_CONFIRMED, Some((800, 3200)));
        for n in [800, 1200, 1600, 2400, 3200] {
            let slot = calibration_slot_in(&at(n), &[subrange]);
            assert_eq!(slot.status, "calibrated", "n = {n}: {slot:?}");
            assert_eq!(slot.reason, None);
            assert_eq!((slot.scope_n, slot.scope_n_max), (Some(800), Some(3200)));
            // The lowest of the measurement's 0.951, 0.949, 0.955 and the confirmation's 0.944,
            // 0.957, 0.953; never the failed point's 0.902.
            assert_eq!(slot.observed, Some(0.944));
            assert!(matches!(
                resolve_passing_range(&subrange, n),
                RangeResolution::Recovered(region) if (region.n_min, region.n_max) == (800, 3200)
            ));
        }
        // The failed lower point and its evidence stay on the record, and govern its own region.
        assert!(subrange.boundary && subrange.grid[0].boundary);
        assert!((subrange.grid[0].observed - 0.902).abs() < 1e-12);
        assert!(
            subrange.confirmation.iter().all(|p| p.point != 0),
            "the failed point was not confirmed"
        );
        for n in [400, 500, 799] {
            let slot = calibration_slot_in(&at(n), &[subrange]);
            assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
            assert_eq!(slot.observed, Some(0.902), "the failing point's coverage is reported");
            assert_eq!(
                resolve_passing_range(&subrange, n),
                RangeResolution::FailedPoint { point: 0, n_min: 400, n_max: 400, observed: 0.902 }
            );
        }
        // Outside every measured size nothing is recovered.
        for n in [399, 3201] {
            let slot = calibration_slot_in(&at(n), &[subrange]);
            assert_eq!(slot.reason.as_deref(), Some(SAMPLE_SIZE_OUTSIDE_MEASURED_RANGE));
            assert_eq!(resolve_passing_range(&subrange, n), RangeResolution::OutsideMeasuredRange);
        }
        assert_eq!(resolve_passing_range(&record("cov.plain"), 500), RangeResolution::NotABoundary);
    }

    #[test]
    fn unconfirmed_subrange_not_recovered() {
        static PARTIAL: [CoverageGridPoint; 2] = [UPPER_CONFIRMED[0], UPPER_CONFIRMED[1]];
        static FAILED_CONFIRMATION: [CoverageGridPoint; 3] = {
            let mut points = UPPER_CONFIRMED;
            points[1].boundary = true;
            points[1].observed = 0.912;
            points
        };
        static OTHER_SIZES: [CoverageGridPoint; 3] = {
            let mut points = UPPER_CONFIRMED;
            points[0].n_min = 801;
            points[0].n_max = 801;
            points
        };
        // The upper three points passed the measurement, but nothing independent confirms them:
        // the record is still a boundary, and the reason is the missing confirmation.
        let unconfirmed: [CoverageRecord; 5] = [
            boundary_over(&LOWER_FAILS, &[], None),
            boundary_over(&LOWER_FAILS, &PARTIAL, None),
            boundary_over(&LOWER_FAILS, &FAILED_CONFIRMATION, None),
            boundary_over(&LOWER_FAILS, &OTHER_SIZES, None),
            // A range stated without evidence for every point in it is not honoured.
            boundary_over(&LOWER_FAILS, &PARTIAL, Some((800, 3200))),
        ];
        for (index, record) in unconfirmed.iter().enumerate() {
            for n in [800, 1600, 3200] {
                let slot = calibration_slot_in(&at(n), &[*record]);
                assert_eq!(slot.status, "scope_not_assessed", "record {index} n = {n}: {slot:?}");
                assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
                assert_eq!(
                    resolve_passing_range(record, n),
                    RangeResolution::Unconfirmed { n_min: 800, n_max: 3200 },
                    "record {index} n = {n}"
                );
            }
        }
    }

    #[test]
    fn non_contiguous_passing_points_not_merged() {
        // 400 and 1600 passed, 800 between them failed. Even with both passing points
        // re-measured and a range stated across them, the failed interior point is not merged.
        static MIDDLE_FAILS: [CoverageGridPoint; 3] = [
            grid_point(0, 400, 0.951, false),
            grid_point(1, 800, 0.905, true),
            grid_point(2, 1600, 0.949, false),
        ];
        static ENDS_CONFIRMED: [CoverageGridPoint; 2] =
            [grid_point(0, 400, 0.953, false), grid_point(2, 1600, 0.955, false)];
        for range in [None, Some((400, 1600))] {
            let record = boundary_over(&MIDDLE_FAILS, &ENDS_CONFIRMED, range);
            for n in [400, 600, 800, 1000, 1600] {
                let slot = calibration_slot_in(&at(n), &[record]);
                assert_eq!(slot.status, "scope_not_assessed", "n = {n}: {slot:?}");
                assert_eq!(slot.reason.as_deref(), Some(BOUNDARY_RECORD));
                assert_eq!(slot.observed, Some(0.905), "the failed interior point is reported");
                assert_eq!(
                    resolve_passing_range(&record, n),
                    RangeResolution::NotContiguous { failed_point: 1, n_min: 800, n_max: 800 },
                    "n = {n}"
                );
            }
        }
    }

    #[test]
    #[allow(
        clippy::items_after_statements,
        reason = "each fixture grid is declared next to its case"
    )]
    fn a_passing_run_must_reach_an_end_of_the_grid_and_hold_two_points() {
        static ONE_PASSES: [CoverageGridPoint; 3] = [
            grid_point(0, 400, 0.902, true),
            grid_point(1, 800, 0.951, false),
            grid_point(2, 1600, 0.903, true),
        ];
        static MIDDLE_RUN: [CoverageGridPoint; 4] = [
            grid_point(0, 400, 0.902, true),
            grid_point(1, 800, 0.951, false),
            grid_point(2, 1600, 0.949, false),
            grid_point(3, 3200, 0.903, true),
        ];
        let lone = boundary_over(&ONE_PASSES, &[], None);
        assert_eq!(
            resolve_passing_range(&lone, 800),
            RangeResolution::LonePassingPoint { n_min: 800, n_max: 800 }
        );
        let middle = boundary_over(&MIDDLE_RUN, &[], None);
        assert_eq!(
            resolve_passing_range(&middle, 1200),
            RangeResolution::InteriorRun { n_min: 800, n_max: 1600 }
        );
        // The lower run of the mirror image is recovered like the upper one.
        static UPPER_FAILS: [CoverageGridPoint; 4] = [
            grid_point(0, 400, 0.951, false),
            grid_point(1, 800, 0.949, false),
            grid_point(2, 1600, 0.955, false),
            grid_point(3, 3200, 0.902, true),
        ];
        static LOWER_CONFIRMED: [CoverageGridPoint; 3] = [
            grid_point(0, 400, 0.952, false),
            grid_point(1, 800, 0.946, false),
            grid_point(2, 1600, 0.956, false),
        ];
        let lower = boundary_over(&UPPER_FAILS, &LOWER_CONFIRMED, Some((400, 1600)));
        let slot = calibration_slot_in(&at(1000), &[lower]);
        assert_eq!(slot.status, "calibrated", "{slot:?}");
        assert_eq!(
            (slot.scope_n, slot.scope_n_max, slot.observed),
            (Some(400), Some(1600), Some(0.946))
        );
        assert_eq!(
            calibration_slot_in(&at(3200), &[lower]).reason.as_deref(),
            Some(BOUNDARY_RECORD)
        );
    }

    /// A named change to one field of a calibration key.
    type KeyMutation = fn(&mut CalibrationKeyWire);

    #[test]
    fn a_confirmed_range_binds_only_to_the_exact_construction_estimand_level_dependence_identification_and_attestation()
     {
        let record = boundary_over(&LOWER_FAILS, &UPPER_CONFIRMED, Some((800, 3200)));
        let positive = calibration_slot_in(&at(1600), &[record]);
        assert_eq!(positive.status, "calibrated");
        // Construction (estimator, interval method), estimand (functional) and dependence differ:
        // no record measured this construction, so nothing is combined.
        let variants: [(&str, KeyMutation); 5] = [
            ("estimator", |k| k.estimator = "foreign".into()),
            ("interval method", |k| k.interval_method = "foreign".into()),
            ("estimand", |k| k.functional = "foreign.contrast".into()),
            ("dependence", |k| k.dependence = "panel_cluster".into()),
            ("graph class", |k| k.graph_class = "foreign".into()),
        ];
        for (what, edit) in variants {
            let mut b = at(1600);
            edit(&mut b.key);
            let slot = calibration_slot_in(&b, &[record]);
            assert_eq!(slot.status, "unavailable", "{what}: {slot:?}");
            assert_eq!(slot.reason.as_deref(), Some(CONSTRUCTION_NOT_MEASURED), "{what}");
        }
        // The nominal level and the identification are exact too.
        let mut level = at(1600);
        level.key.level = 0.9;
        assert_eq!(
            calibration_slot_in(&level, &[record]).reason.as_deref(),
            Some(LEVEL_NOT_MEASURED)
        );
        let mut identification = at(1600);
        identification.key.identification = "partial".into();
        assert_eq!(
            calibration_slot_in(&identification, &[record]).reason.as_deref(),
            Some(IDENTIFICATION_NOT_MEASURED)
        );
        // And the record must still attest the current code.
        let stale = calibration_slot_with(&at(1600), &[record], |_| false);
        assert_eq!(stale.status, "scope_not_assessed");
        assert_eq!(stale.reason.as_deref(), Some(RECORD_NOT_ATTESTING));
    }
}
