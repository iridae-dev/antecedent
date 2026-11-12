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
//! Vocabulary (an annotation on every reported interval, never a gate):
//!
//! - `calibrated`: a non-boundary record that still attests the current code
//!   (its facets have not drifted since `calibration_sha`) measured this
//!   construction (every key field equal, including the nominal level and the
//!   identification label) and the execution lies inside what it measured: its
//!   row count inside the record's measured row-count range (the span of its
//!   sample-size grid points, never extrapolated beyond them), at least as many
//!   successful resampling replicates and posterior draws, and no more
//!   non-identified mass. A record is a boundary when any of its grid points
//!   is: a construction that measured below nominal at one sample size is not
//!   calibrated anywhere in the range, except inside a
//!   [`confirmed_passing_region`]: a contiguous run of grid points that passed
//!   in the measurement and again in an independent re-measurement with fresh
//!   seeds. There the slot is `calibrated`, governed by the worst of those
//!   points. A matching but stale (non-attesting) record is never `calibrated`.
//! - `outside_measured_scope`: a record measured the construction but the
//!   execution is outside its scope (the reason names which bound), or the
//!   covering record no longer attests the current code
//!   (`coverage_record_not_attesting`).
//! - `measured_below_nominal`: the covering record is a named boundary /
//!   under-coverage measurement at this sample size (`boundary_record`, with
//!   its observed coverage in `observed`).
//! - `not_measured`: no record measured this construction, level or
//!   identification; or no interval was reported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::contract_section::{CalibrationSlotWire, CalibrationStatus};
use crate::coverage_records_data::{
    ATTESTING_RECORD_IDS, CoverageGridPoint, CoverageRecord, FUNCTIONAL_IDENTITY_ALIASES,
    FunctionalIdentityAlias, HISTORICAL_VALIDATION_BINDINGS, HistoricalValidationBinding, RECORDS,
};

mod ranges;
use ranges::worst;
pub use ranges::{
    ConfirmedRegion, RangeResolution, confirmed_passing_region, is_boundary, measured_range,
    rederive_calibration, resolve_passing_range,
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
/// `scripts/calibration_status.py`), so it changes when a record is re-measured,
/// probed or its dependency fingerprint changes, and never by hand. Stale or otherwise
/// non-attesting records may still be reported as `outside_measured_scope` with
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
    /// Executed support validation mode, or empty when it is not established.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub validation: String,
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
                validation: basis.validation.to_string(),
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
    /// Explicit structure.
    Fixed,
    /// Accepted structure, retained separately for provenance checks.
    Accepted,
    /// Averaged over a graph posterior.
    GraphPosterior,
}

impl CalibrationStructure {
    /// The key spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Accepted => "accepted",
            Self::GraphPosterior => "graph_posterior",
        }
    }

    /// Classify a support-matrix structure axis name.
    #[must_use]
    pub fn from_axis(name: &str) -> Self {
        match name {
            "graph_posterior" => Self::GraphPosterior,
            "accepted" => Self::Accepted,
            _ => Self::Fixed,
        }
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

fn validation_matches_with_bindings(
    key: &CalibrationKeyWire,
    record: &CoverageRecord,
    bindings: &[HistoricalValidationBinding],
) -> bool {
    if !matches!(key.validation.as_str(), "none" | "cheap" | "full" | "not_applicable") {
        return false;
    }
    if record.validation_schema == 2 {
        let mut markers =
            record.id.split('.').filter_map(|part| part.strip_prefix("v2_validation_"));
        let mut structures =
            record.id.split('.').filter_map(|part| part.strip_prefix("v2_structure_"));
        // A structure that shares a test with the fixed graph is named in the id.
        let named =
            matches!(record.structure, "accepted" | "graph_posterior").then_some(record.structure);
        let source_ok = (record.structure == "accepted") == (key.structure == "accepted")
            && structures.next() == named
            && structures.next().is_none();
        return source_ok
            && record.validation == key.validation
            && markers.next() == Some(record.validation)
            && markers.next().is_none();
    }
    if record.validation_schema != 0 || !record.validation.is_empty() {
        return false;
    }
    bindings.iter().any(|binding| {
        binding.record_id == record.id
            && binding.calibration_sha == record.calibration_sha
            && binding.test == record.test
            && binding.validation == key.validation
            && (key.validation == "not_applicable"
                || binding.source_structure
                    == match key.structure.as_str() {
                        "fixed" => "explicit",
                        "accepted" => "accepted",
                        "graph_posterior" => "graph_posterior",
                        _ => "",
                    })
    })
}

fn validation_matches(key: &CalibrationKeyWire, record: &CoverageRecord) -> bool {
    validation_matches_with_bindings(key, record, &HISTORICAL_VALIDATION_BINDINGS)
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
        validation_matches(self, record)
            && record.query == self.query
            && record.graph_class == self.graph_class
            && (record.structure == self.structure
                || (record.structure == "fixed" && self.structure == "accepted"))
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
                    validation_matches(key, record)
                        && functional_matches(record, &key.functional, &FUNCTIONAL_IDENTITY_ALIASES)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn construction_axis(key: &CalibrationKeyWire) -> String {
    [
        key.query.as_str(),
        key.graph_class.as_str(),
        if key.structure == "accepted" { "fixed" } else { key.structure.as_str() },
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
                if record.structure == "accepted" { "fixed" } else { record.structure },
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
        return CalibrationSlotWire::not_measured(BASIS_MISSING);
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
/// matching record that fails `attests` is `outside_measured_scope` with
/// [`RECORD_NOT_ATTESTING`].
#[must_use]
pub fn calibration_slot_with(
    basis: &CalibrationBasisWire,
    records: &[CoverageRecord],
    attests: impl Fn(&CoverageRecord) -> bool,
) -> CalibrationSlotWire {
    let key = &basis.key;
    let scope = &basis.scope;
    let not_measured = |reason: &str| CalibrationSlotWire::not_measured(reason).with_basis(basis);
    if key.interval_method == "none" {
        return not_measured(NO_INTERVAL_REPORTED);
    }
    let construction: Vec<&CoverageRecord> =
        records.iter().filter(|record| key.same_construction(record)).collect();
    if construction.is_empty() {
        return not_measured(CONSTRUCTION_NOT_MEASURED);
    }
    let at_level: Vec<&CoverageRecord> = construction
        .into_iter()
        .filter(|record| (record.nominal - key.level).abs() <= TOLERANCE)
        .collect();
    if at_level.is_empty() {
        return not_measured(LEVEL_NOT_MEASURED);
    }
    let measured: Vec<&CoverageRecord> =
        at_level.into_iter().filter(|record| record.identification == key.identification).collect();
    if measured.is_empty() {
        return not_measured(IDENTIFICATION_NOT_MEASURED);
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
        return CalibrationSlotWire::from_record(nearest, CalibrationStatus::OutsideMeasuredScope)
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
            CalibrationSlotWire::from_record(boundary, CalibrationStatus::MeasuredBelowNominal)
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
        return CalibrationSlotWire::from_record(
            governing,
            CalibrationStatus::OutsideMeasuredScope,
        )
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

#[cfg(test)]
mod tests;
